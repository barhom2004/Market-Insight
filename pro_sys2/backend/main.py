from fastapi import FastAPI, Query
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import threading
import time
import ccxt
import yfinance as yf
from typing import Optional

from stream import stream_market_data
from news_model import NewsSentimentPredictor, train_news_model, MODEL_DIR
from news_service import (
    news_cache, fetch_all_news, enrich_news_with_sentiment,
    start_news_background_fetcher, SYMBOL_TO_CATEGORY,
)
from hybrid_signal_model import HybridSignalGenerator
import paper_trading


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

states = {}
states_lock = threading.Lock()
stop_flags = {}
stop_flags_lock = threading.Lock()

# ── Hybrid model retraining state ──────────────────────────────────
_hybrid_retrain_state: dict = {"status": "idle", "horizon": None, "started_at": None, "error": None}

CRYPTO_SYMBOLS = {
    "BTC/USDT",
    "ETH/USDT",
    "BNB/USDT",
}

# Yahoo Finance ticker map for metals, FX, and stocks
YF_SYMBOL_MAP = {
    # Metals
    "XAUUSD": "GC=F",
    "XAGUSD": "SI=F",
    "XPTUSD": "PL=F",
    # FX
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "EURGBP": "EURGBP=X",
    # Stocks
    "AAPL": "AAPL",
    "AMZN": "AMZN",
    "TSLA": "TSLA",
}

YF_SYMBOLS = set(YF_SYMBOL_MAP.keys())


class StartStreamRequest(BaseModel):
    symbol: str


def normalize_crypto_symbol(symbol: str) -> str:
    normalized = symbol.strip().upper()
    if "/" not in normalized and normalized.endswith("USD") and len(normalized) > 3:
        return f"{normalized[:-3]}/USDT"
    if "/" not in normalized and normalized.endswith("USDT") and len(normalized) > 4:
        return f"{normalized[:-4]}/USDT"
    return normalized


def normalize_symbol(symbol: str) -> str:
    """Normalize to either a crypto symbol (BTC/USDT) or YF key (XAUUSD)."""
    upper = symbol.strip().upper()
    if upper in YF_SYMBOLS:
        return upper
    return normalize_crypto_symbol(upper)


def sanitize_value(value, default=0.0):
    """Ensure value is finite and valid, otherwise return default."""
    try:
        v = float(value)
        if not (v == v):  # NaN check
            return default
        if v == float('inf') or v == float('-inf'):
            return default
        return v
    except (ValueError, TypeError):
        return default


def ensure_symbol_state(symbol: str):
    with states_lock:
        if symbol not in states:
            states[symbol] = {
                "candles": [],
                "price": None,
                "signal": None,
                "thread": None,
            }
        return states[symbol]


# ─────────────────────────────────────────────
# Binance helpers (crypto)
# ─────────────────────────────────────────────

def load_initial_candles(symbol: str, timeframe: str = "1m", total: int = 5000):
    exchange = ccxt.binance({"timeout": 30000, "enableRateLimit": True})
    exchange.load_markets()

    timeframe_map = {
        "1m": 60, "3m": 180, "5m": 300, "15m": 900,
        "30m": 1800, "1h": 3600, "4h": 14400,
        "1d": 86400, "1w": 604800, "1M": 2592000,
    }
    tf_ms = timeframe_map[timeframe] * 1000

    all_data = exchange.fetch_ohlcv(symbol, timeframe, limit=1000)
    while len(all_data) < total:
        since = all_data[0][0] - (1000 * tf_ms)
        batch = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
        if not batch or batch[0][0] == all_data[0][0]:
            break
        all_data = batch + all_data
        all_data = sorted(
            list({x[0]: x for x in all_data}.values()), key=lambda x: x[0]
        )

    all_data = all_data[-total:]
    return [
        {
            "timestamp": int(row[0]),
            "Open": sanitize_value(row[1]),
            "High": sanitize_value(row[2]),
            "Low": sanitize_value(row[3]),
            "Close": sanitize_value(row[4]),
            "Volume": sanitize_value(row[5]),
        }
        for row in all_data
        if sanitize_value(row[1]) > 0 and sanitize_value(row[4]) > 0
    ]


def run_crypto_stream_with_timeframe(symbol: str, timeframe: str = "1h", horizon: Optional[int] = None):
    """Stream live data with specific timeframe."""
    symbol_state = ensure_symbol_state(symbol)
    
    with stop_flags_lock:
        stop_flags[symbol] = False
    
    # Load historical data with the specified timeframe
    try:
        print(f"📊 Loading historical candles for {symbol} with {timeframe} timeframe...")
        initial_candles = load_initial_candles(symbol, timeframe, total=5000)
        with states_lock:
            symbol_state["candles"] = initial_candles
        print(f"✅ Loaded {len(initial_candles)} candles for {symbol} ({timeframe})")
    except Exception as e:
        print(f"Failed to load initial candles for {symbol}: {e}")

    stream = stream_market_data(symbol, timeframe, total=5000, horizon=horizon)
    for data in stream:
        with stop_flags_lock:
            if stop_flags.get(symbol, False):
                print(f"🛑 [{symbol}] Stream stopped by request")
                break
        
        if data["type"] == "live_price":
            price = sanitize_value(data.get('price', 0))
            if price > 0:
                with states_lock:
                    symbol_state["price"] = {
                        "type": "live_price",
                        "price": price,
                        "time": data.get('time', '')
                    }
                print(f"📈 [{symbol}] Live: {price}")

        elif data["type"] == "signal":
            with states_lock:
                symbol_state["signal"] = data
            print(
                f"🔥 [{symbol}] Signal: {data['signal']} | "
                f"Buy: {data['buy_prob']:.2f} | Sell: {data['sell_prob']:.2f}"
            )
            try:
                exchange = ccxt.binance({"timeout": 30000, "enableRateLimit": True})
                latest = exchange.fetch_ohlcv(symbol, timeframe, limit=2)[-1]
                candle = {
                    "timestamp": int(latest[0]),
                    "Open": sanitize_value(latest[1]),
                    "High": sanitize_value(latest[2]),
                    "Low": sanitize_value(latest[3]),
                    "Close": sanitize_value(latest[4]),
                    "Volume": sanitize_value(latest[5]),
                }
                if candle["Open"] > 0 and candle["Close"] > 0:
                    with states_lock:
                        candles = symbol_state["candles"]
                        if candles and candles[-1]["timestamp"] == candle["timestamp"]:
                            candles[-1] = candle
                        else:
                            candles.append(candle)
                            if len(candles) > 5000:
                                del candles[:-5000]
            except Exception as e:
                print(f"Failed to update candles for {symbol}: {e}")
    
    with stop_flags_lock:
        stop_flags[symbol] = False
    print(f"✅ [{symbol}] Stream ended")


def run_crypto_stream(symbol: str, load_history: bool = False):
    """Stream live data. Only load 5000 candles if load_history=True."""
    symbol_state = ensure_symbol_state(symbol)
    
    with stop_flags_lock:
        stop_flags[symbol] = False
    
    # Only load historical data if explicitly requested (when user opens symbol page)
    if load_history:
        try:
            print(f"📊 Loading 5000 historical candles for {symbol}...")
            initial_candles = load_initial_candles(symbol, "1m", total=5000)
            with states_lock:
                symbol_state["candles"] = initial_candles
            print(f"✅ Loaded {len(initial_candles)} candles for {symbol}")
        except Exception as e:
            print(f"Failed to load initial candles for {symbol}: {e}")

        stream = stream_market_data(symbol, "1m", total=5000)
        for data in stream:
            with stop_flags_lock:
                if stop_flags.get(symbol, False):
                    print(f"🛑 [{symbol}] Stream stopped by request")
                    break
            
            if data["type"] == "live_price":
                price = sanitize_value(data.get('price', 0))
                if price > 0:
                    with states_lock:
                        symbol_state["price"] = {
                            "type": "live_price",
                            "price": price,
                            "time": data.get('time', '')
                        }
                    print(f"📈 [{symbol}] Live: {price}")

            elif data["type"] == "signal":
                with states_lock:
                    symbol_state["signal"] = data
                print(
                    f"🔥 [{symbol}] Signal: {data['signal']} | "
                    f"Buy: {data['buy_prob']:.2f} | Sell: {data['sell_prob']:.2f}"
                )
                try:
                    exchange = ccxt.binance({"timeout": 30000, "enableRateLimit": True})
                    latest = exchange.fetch_ohlcv(symbol, "1m", limit=2)[-1]
                    candle = {
                        "timestamp": int(latest[0]),
                        "Open": sanitize_value(latest[1]),
                        "High": sanitize_value(latest[2]),
                        "Low": sanitize_value(latest[3]),
                        "Close": sanitize_value(latest[4]),
                        "Volume": sanitize_value(latest[5]),
                    }
                    if candle["Open"] > 0 and candle["Close"] > 0:
                        with states_lock:
                            candles = symbol_state["candles"]
                            if candles and candles[-1]["timestamp"] == candle["timestamp"]:
                                candles[-1] = candle
                            else:
                                candles.append(candle)
                                if len(candles) > 5000:
                                    del candles[:-5000]
                except Exception as e:
                    print(f"Failed to update candles for {symbol}: {e}")
    else:
        # Live-only mode: just stream prices without ML model
        import time
        exchange = ccxt.binance({"timeout": 30000, "enableRateLimit": True})
        exchange.load_markets()
        
        print(f"📡 Starting live-only stream for {symbol} (no historical data)")
        
        while True:
            with stop_flags_lock:
                if stop_flags.get(symbol, False):
                    print(f"🛑 [{symbol}] Live stream stopped")
                    break
            
            try:
                ticker = exchange.fetch_ticker(symbol)
                price = sanitize_value(ticker['last'])
                if price > 0:
                    with states_lock:
                        symbol_state["price"] = {
                            "type": "live_price",
                            "price": price,
                            "time": ""
                        }
                        # Set default HOLD signal for live-only mode
                        if not symbol_state.get("signal"):
                            symbol_state["signal"] = {
                                "type": "signal",
                                "signal": "HOLD",
                                "buy_prob": 0.0,
                                "sell_prob": 0.0
                            }
                    print(f"📈 [{symbol}] Live: {price}")
                time.sleep(2)
            except Exception as e:
                print(f"Live price error for {symbol}: {e}")
                time.sleep(5)
    
    with stop_flags_lock:
        stop_flags[symbol] = False
    print(f"✅ [{symbol}] Stream ended")


# ─────────────────────────────────────────────
# Yahoo Finance helpers (metals & FX)
# ─────────────────────────────────────────────

# Map app timeframe strings to yfinance interval strings
YF_INTERVAL_MAP = {
    "1m": ("1m", "7d"),
    "5m": ("5m", "60d"),
    "15m": ("15m", "60d"),
    "30m": ("30m", "60d"),
    "1h": ("1h", "730d"),
    "4h": ("1h", "730d"),
    "1d": ("1d", "730d"),
    "1w": ("1wk", "730d"),
    "1M": ("1mo", "730d"),
}

# Horizon mapping: how many candles to look ahead for each timeframe
# Optimized for realistic trading scenarios
HORIZON_MAP = {
    "1m": 5,    # 5 minutes ahead
    "5m": 6,    # 30 minutes ahead (6 * 5m)
    "15m": 4,   # 1 hour ahead (4 * 15m)
    "30m": 4,   # 2 hours ahead (4 * 30m)
    "1h": 3,    # 3 hours ahead
    "4h": 3,    # 12 hours ahead (3 * 4h)
    "1d": 3,    # 3 days ahead
    "1w": 2,    # 2 weeks ahead
    "1M": 2,    # 2 months ahead
}

# Threshold mapping: minimum price change % to consider as signal
# Different per asset class because volatility varies significantly
THRESHOLD_CRYPTO = {
    "1m": 0.0005,   "5m": 0.0008,  "15m": 0.001,
    "30m": 0.0015,  "1h": 0.002,   "4h": 0.003,
    "1d": 0.005,    "1w": 0.01,    "1M": 0.02,
}
THRESHOLD_FOREX = {
    "1m": 0.0001,   "5m": 0.0002,  "15m": 0.0003,
    "30m": 0.0005,  "1h": 0.0007,  "4h": 0.001,
    "1d": 0.0015,   "1w": 0.003,   "1M": 0.006,
}
THRESHOLD_METALS = {
    "1m": 0.0003,   "5m": 0.0005,  "15m": 0.0007,
    "30m": 0.001,   "1h": 0.0015,  "4h": 0.002,
    "1d": 0.0035,   "1w": 0.007,   "1M": 0.015,
}
THRESHOLD_STOCKS = {
    "1m": 0.0004,   "5m": 0.0006,  "15m": 0.0008,
    "30m": 0.0012,  "1h": 0.0015,  "4h": 0.0025,
    "1d": 0.004,    "1w": 0.008,   "1M": 0.015,
}

FX_SYMBOLS = {"EURUSD", "GBPUSD", "EURGBP"}
METAL_SYMBOLS = {"XAUUSD", "XAGUSD", "XPTUSD"}
STOCK_SYMBOLS = {"AAPL", "AMZN", "TSLA"}

def get_threshold_for_symbol(symbol: str, timeframe: str) -> float:
    """Get the correct threshold based on asset class and timeframe."""
    if symbol in FX_SYMBOLS:
        return THRESHOLD_FOREX.get(timeframe, 0.0007)
    elif symbol in METAL_SYMBOLS:
        return THRESHOLD_METALS.get(timeframe, 0.0015)
    elif symbol in STOCK_SYMBOLS:
        return THRESHOLD_STOCKS.get(timeframe, 0.0015)
    else:
        return THRESHOLD_CRYPTO.get(timeframe, 0.002)

def run_yf_stream(symbol: str, load_history: bool = False, timeframe: str = "1h", custom_horizon: Optional[int] = None):
    """Stream Yahoo Finance data. Load history and ML model if load_history=True."""
    symbol_state = ensure_symbol_state(symbol)
    yf_ticker = YF_SYMBOL_MAP[symbol]
    
    with stop_flags_lock:
        stop_flags[symbol] = False

    model = None
    scaler = None
    df = None

    # Resolve yf interval/period for this timeframe (used in both history load and streaming)
    yf_interval, yf_period = YF_INTERVAL_MAP.get(timeframe, ("1h", "730d"))
    # Shorter periods for the live streaming loop (we only need the latest candle)
    YF_STREAM_PERIOD = {
        "1m": "1d", "5m": "5d", "15m": "5d", "30m": "5d",
        "1h": "5d", "1d": "1mo", "1wk": "3mo", "1mo": "1y",
    }
    yf_stream_period = YF_STREAM_PERIOD.get(yf_interval, "5d")

    # Load historical data and train model if requested
    if load_history:
        try:
            print(f"📊 Loading historical data for {symbol} from Yahoo Finance ({timeframe} -> {yf_interval}, period={yf_period})...")
            ticker_obj = yf.Ticker(yf_ticker)
            hist = ticker_obj.history(period=yf_period, interval=yf_interval)
            hist = hist.dropna()
            
            print(f"📈 Loaded {len(hist)} candles for {symbol}")
            
            if len(hist) < 10:
                print(f"⚠️ Not enough data for {symbol} ({len(hist)} candles), falling back to live-only mode")
                load_history = False
            else:
                candles = []
                for ts, row in hist.iterrows():
                    o = sanitize_value(row["Open"])
                    h = sanitize_value(row["High"])
                    l = sanitize_value(row["Low"])
                    c = sanitize_value(row["Close"])
                    v = sanitize_value(row.get("Volume", 0))
                    
                    if o > 0 and c > 0:
                        candles.append({
                            "timestamp": int(ts.timestamp() * 1000),
                            "Open": o,
                            "High": h,
                            "Low": l,
                            "Close": c,
                            "Volume": v,
                        })
                
                with states_lock:
                    symbol_state["candles"] = candles[-5000:]  # Keep last 5000
                
                print(f"✅ [{symbol}] Loaded {len(candles)} candles from Yahoo Finance")
                
                # Train ML model with timeframe-specific parameters
                horizon = custom_horizon if custom_horizon is not None else HORIZON_MAP.get(timeframe, 3)
                threshold = get_threshold_for_symbol(symbol, timeframe)
                print(f"🚀 Training ML model for {symbol} (timeframe={timeframe}, horizon={horizon} candles, threshold={threshold*100:.2f}%)...")
                from train_model import train_xgb_model
                from feature_engineering2 import compute_all_features
                
                df = hist.copy()
                df = df.rename(columns={"Open": "Open", "High": "High", "Low": "Low", "Close": "Close", "Volume": "Volume"})
                df = compute_all_features(df, is_new_row=False, threshold=threshold, horizon=horizon)
                df_for_training = df.reset_index()
                # yfinance uses 'Date' for daily, 'Datetime' for intraday
                if 'Date' in df_for_training.columns:
                    df_for_training = df_for_training.rename(columns={"Date": "timestamp"})
                elif 'Datetime' in df_for_training.columns:
                    df_for_training = df_for_training.rename(columns={"Datetime": "timestamp"})
                else:
                    df_for_training.columns.values[0] = 'timestamp'
                
                preds, model, scaler = train_xgb_model(df_for_training, threshold=threshold, horizon=horizon)
                print(f"🔥 ML model ready for {symbol} - predicting {horizon} candles ahead with {threshold*100:.2f}% threshold")
        except Exception as e:
            print(f"Failed to load/train for {symbol}: {e}")
            load_history = False

    # Streaming loop
    import time
    while True:
        with stop_flags_lock:
            if stop_flags.get(symbol, False):
                print(f"🛑 [{symbol}] YF Stream stopped by request")
                break
        
        try:
            ticker_obj = yf.Ticker(yf_ticker)
            
            # Get real-time price from latest 1-minute candle
            price = None
            try:
                latest_data = ticker_obj.history(period="1d", interval="1m")
                if not latest_data.empty:
                    price = sanitize_value(latest_data['Close'].iloc[-1])
            except:
                # Fallback to fast_info if history fails
                try:
                    info = ticker_obj.fast_info
                    price = getattr(info, "last_price", None)
                    if price:
                        price = sanitize_value(price)
                except:
                    pass

            if price and price > 0:
                import datetime
                now = datetime.datetime.utcnow()
                with states_lock:
                    symbol_state["price"] = {
                        "type": "live_price",
                        "price": price,
                        "time": now.isoformat(),
                    }
                print(f"📈 [{symbol}] YF Live: {price}")

                # Append latest candle using the selected timeframe interval
                try:
                    hist = ticker_obj.history(period=yf_stream_period, interval=yf_interval)
                    hist = hist.dropna()
                    if not hist.empty:
                        row = hist.iloc[-1]
                        ts = hist.index[-1]
                        o = sanitize_value(row["Open"])
                        h = sanitize_value(row["High"])
                        l = sanitize_value(row["Low"])
                        c = sanitize_value(row["Close"])
                        v = sanitize_value(row.get("Volume", 0))
                        
                        if o > 0 and c > 0:
                            candle = {
                                "timestamp": int(ts.timestamp() * 1000),
                                "Open": o,
                                "High": h,
                                "Low": l,
                                "Close": c,
                                "Volume": v,
                            }
                            with states_lock:
                                candles = symbol_state["candles"]
                                if candles and candles[-1]["timestamp"] == candle["timestamp"]:
                                    candles[-1] = candle
                                else:
                                    candles.append(candle)
                                    if len(candles) > 5000:
                                        del candles[:-5000]
                            
                            # Generate ML signal if model is loaded
                            if load_history and model and scaler and df is not None:
                                try:
                                    from feature_engineering2 import compute_all_features
                                    import pandas as pd
                                    
                                    new_row = pd.DataFrame([{
                                        'Open': o,
                                        'High': h,
                                        'Low': l,
                                        'Close': c,
                                        'Volume': v
                                    }], index=[ts])
                                    
                                    df = pd.concat([df, new_row])
                                    df = compute_all_features(df, is_new_row=True)
                                    df = df.tail(5000)
                                    
                                    X_live = df.drop(columns=['Target'], errors='ignore')
                                    X_scaled = scaler.transform(X_live)
                                    last_row = X_scaled[-1].reshape(1, -1)
                                    
                                    probs = model.predict_proba(last_row)[0]
                                    sell_prob = probs[0]
                                    buy_prob = probs[1]
                                    
                                    if buy_prob > 0.65:
                                        signal = "BUY"
                                    elif sell_prob > 0.65:
                                        signal = "SELL"
                                    else:
                                        signal = "HOLD"
                                    
                                    with states_lock:
                                        symbol_state["signal"] = {
                                            "type": "signal",
                                            "signal": signal,
                                            "buy_prob": float(buy_prob),
                                            "sell_prob": float(sell_prob)
                                        }
                                    
                                    print(f"🔥 [{symbol}] Signal: {signal} | Buy: {buy_prob:.2f} | Sell: {sell_prob:.2f}")
                                except Exception as e:
                                    print(f"ML signal error for {symbol}: {e}")
                except Exception:
                    pass
                
                # Set default HOLD signal if no ML model
                if not load_history:
                    with states_lock:
                        if not symbol_state.get("signal"):
                            symbol_state["signal"] = {
                                "type": "signal",
                                "signal": "HOLD",
                                "buy_prob": 0.0,
                                "sell_prob": 0.0
                            }
            else:
                print(f"⚠️ [{symbol}] No valid price data from Yahoo Finance")

            time.sleep(5)
        except Exception as e:
            print(f"[YF Stream Error] {symbol}: {e}")
            time.sleep(10)
    
    with stop_flags_lock:
        stop_flags[symbol] = False
    print(f"✅ [{symbol}] YF Stream ended")


# ─────────────────────────────────────────────
# API Endpoints
# ─────────────────────────────────────────────

@app.get("/")
def home():
    return {"message": "Trading API Running 🚀"}


@app.post("/start-stream")
def start_stream(payload: StartStreamRequest):
    """Start live streaming. Use /load-history to load 5000 candles."""
    symbol = normalize_symbol(payload.symbol)
    symbol_state = ensure_symbol_state(symbol)

    with states_lock:
        existing_thread = symbol_state["thread"]
        if existing_thread and existing_thread.is_alive():
            return {"message": "stream already active", "symbol": symbol}

    # Start live-only streaming (no historical data)
    if symbol in CRYPTO_SYMBOLS:
        thread = threading.Thread(target=run_crypto_stream, args=(symbol, False), daemon=True)
    elif symbol in YF_SYMBOLS:
        thread = threading.Thread(target=run_yf_stream, args=(symbol, False), daemon=True)
    else:
        return {"message": "symbol not supported", "symbol": symbol}

    with states_lock:
        symbol_state["thread"] = thread
    thread.start()
    return {"message": "live stream started", "symbol": symbol}


@app.post("/load-history")
def load_history(payload: StartStreamRequest):
    """Load 5000 historical candles and start ML model streaming."""
    symbol = normalize_symbol(payload.symbol)
    symbol_state = ensure_symbol_state(symbol)

    # Stop existing stream if any
    with stop_flags_lock:
        stop_flags[symbol] = True
    
    # Wait for old thread to finish
    with states_lock:
        old_thread = symbol_state.get("thread")
    if old_thread and old_thread.is_alive():
        old_thread.join(timeout=2)

    # Start new stream with historical data
    if symbol in CRYPTO_SYMBOLS:
        thread = threading.Thread(target=run_crypto_stream, args=(symbol, True), daemon=True)
    elif symbol in YF_SYMBOLS:
        thread = threading.Thread(target=run_yf_stream, args=(symbol, True), daemon=True)
    else:
        return {"message": "symbol not supported", "symbol": symbol}

    with states_lock:
        symbol_state["thread"] = thread
    thread.start()
    return {"message": "loading 5000 candles and starting ML stream", "symbol": symbol}


class LoadHistoryTimeframeRequest(BaseModel):
    symbol: str
    timeframe: str
    horizon: Optional[int] = None


@app.post("/load-history-timeframe")
def load_history_timeframe(payload: LoadHistoryTimeframeRequest):
    """Load historical candles with specific timeframe and start streaming."""
    symbol = normalize_symbol(payload.symbol)
    timeframe = payload.timeframe
    horizon = payload.horizon
    symbol_state = ensure_symbol_state(symbol)

    # Stop existing stream if any
    with stop_flags_lock:
        stop_flags[symbol] = True
    
    # Wait for old thread to finish
    with states_lock:
        old_thread = symbol_state.get("thread")
    if old_thread and old_thread.is_alive():
        old_thread.join(timeout=2)

    # Start new stream with historical data for the specified timeframe
    if symbol in CRYPTO_SYMBOLS:
        thread = threading.Thread(target=run_crypto_stream_with_timeframe, args=(symbol, timeframe, horizon), daemon=True)
    elif symbol in YF_SYMBOLS:
        thread = threading.Thread(target=run_yf_stream, args=(symbol, True, timeframe, horizon), daemon=True)
    else:
        return {"message": "symbol not supported", "symbol": symbol}

    with states_lock:
        symbol_state["thread"] = thread
    thread.start()
    return {"message": f"loading historical data with {timeframe} timeframe", "symbol": symbol, "timeframe": timeframe, "horizon": horizon if horizon is not None else HORIZON_MAP.get(timeframe, 3)}


@app.post("/stop-stream")
def stop_stream(payload: StartStreamRequest):
    symbol = normalize_symbol(payload.symbol)
    with stop_flags_lock:
        stop_flags[symbol] = True
    return {"message": "stop signal sent", "symbol": symbol}


@app.get("/price/{symbol:path}")
def get_price(symbol: str):
    normalized = normalize_symbol(symbol)
    symbol_state = ensure_symbol_state(normalized)
    with states_lock:
        if symbol_state["price"]:
            return symbol_state["price"]
    return {"error": "No data yet", "symbol": normalized}


@app.get("/signal/{symbol:path}")
def get_signal(symbol: str):
    normalized = normalize_symbol(symbol)
    symbol_state = ensure_symbol_state(normalized)
    with states_lock:
        if symbol_state["signal"]:
            return symbol_state["signal"]
    return {"error": "No signal yet", "symbol": normalized}


@app.get("/candles/{symbol:path}")
def get_candles(symbol: str, limit: int = 200, timeframe: str = "1h"):
    """
    Get candles for a symbol.
    timeframe: 1m, 5m, 15m, 30m, 1h, 4h, 1d, 1w, 1M
    The candles are already stored in the correct timeframe from the stream.
    """
    normalized = normalize_symbol(symbol)
    symbol_state = ensure_symbol_state(normalized)
    
    with states_lock:
        all_candles = symbol_state["candles"]
    
    # Apply limit - return the most recent candles
    safe_limit = max(1, min(limit, 5000))
    raw_candles = all_candles[-safe_limit:] if all_candles else []
    
    # Filter out any candles with NaN/Infinity values
    valid_candles = []
    for candle in raw_candles:
        try:
            ts = candle.get("timestamp", 0)
            o = candle.get("Open", 0)
            h = candle.get("High", 0)
            l = candle.get("Low", 0)
            c = candle.get("Close", 0)
            v = candle.get("Volume", 0)
            
            # Check all values are valid numbers
            if (isinstance(ts, (int, float)) and 
                isinstance(o, (int, float)) and 
                isinstance(h, (int, float)) and 
                isinstance(l, (int, float)) and 
                isinstance(c, (int, float)) and 
                isinstance(v, (int, float))):
                
                # Check no NaN or Infinity
                if (ts == ts and o == o and h == h and l == l and c == c and v == v and
                    abs(ts) != float('inf') and abs(o) != float('inf') and 
                    abs(h) != float('inf') and abs(l) != float('inf') and 
                    abs(c) != float('inf') and abs(v) != float('inf') and
                    o > 0 and c > 0 and h > 0 and l > 0 and ts > 0):
                    
                    valid_candles.append({
                        "timestamp": int(ts),
                        "Open": float(o),
                        "High": float(h),
                        "Low": float(l),
                        "Close": float(c),
                        "Volume": float(v)
                    })
        except (ValueError, TypeError, OverflowError):
            continue
    
    return {"symbol": normalized, "candles": valid_candles, "timeframe": timeframe}


# ─────────────────────────────────────────────
# News Sentiment Model & Endpoints
# ─────────────────────────────────────────────

news_predictor = NewsSentimentPredictor()
hybrid_signal_gen = HybridSignalGenerator()


@app.on_event("startup")
def startup_event():
    """Load news model and start background news fetcher on startup."""
    import os
    if os.path.exists(os.path.join(MODEL_DIR, "model.pkl")):
        news_predictor.load()
    else:
        print("⚠️ News model not found. Training now (this may take a few minutes)...")
        try:
            train_news_model()
            news_predictor.load()
        except Exception as e:
            print(f"❌ Failed to train news model: {e}")
            print("   News endpoints will return neutral sentiment.")

    # Load hybrid signal model — prefer real-data model, retrain if missing or synthetic
    hybrid_model_path = os.path.join(os.path.dirname(__file__), "hybrid_model_artifacts", "hybrid_model.pkl")
    flag_path = os.path.join(os.path.dirname(__file__), "hybrid_model_artifacts", "data_version.txt")
    is_real_data = os.path.exists(flag_path) and open(flag_path).read().strip() == "real_data_v1"

    if os.path.exists(hybrid_model_path) and is_real_data:
        hybrid_signal_gen.load()
    else:
        if os.path.exists(hybrid_model_path) and not is_real_data:
            print("🔄 Hybrid model was trained on synthetic data — retraining on real market data...")
        else:
            print("⚠️ Hybrid signal model not found — training on real market data...")
        try:
            from hybrid_signal_model import train_hybrid_from_real_data
            import threading
            def _train():
                try:
                    train_hybrid_from_real_data(hybrid_signal_gen)
                    print("✅ Hybrid model trained on real data and ready.")
                except Exception as e:
                    print(f"❌ Real data training failed: {e} — falling back to synthetic data")
                    try:
                        from hybrid_signal_model import create_synthetic_training_data
                        df_train = create_synthetic_training_data()
                        hybrid_signal_gen.train(df_train)
                        hybrid_signal_gen.save()
                    except Exception as e2:
                        print(f"❌ Synthetic fallback also failed: {e2}")
            threading.Thread(target=_train, daemon=True).start()
        except Exception as e:
            print(f"❌ Failed to start hybrid model training: {e}")

    start_news_background_fetcher(news_predictor)


@app.get("/news")
def get_news(
    category: Optional[str] = Query(None, description="Filter by category: stocks, fx, crypto, metals"),
    symbol: Optional[str] = Query(None, description="Filter by symbol: AAPL, BTCUSD, etc."),
    limit: int = Query(50, ge=1, le=200),
):
    """Get latest news articles with sentiment analysis."""
    if symbol:
        articles = news_cache.get_by_symbol(symbol.upper())
    elif category:
        articles = news_cache.get_by_category(category.lower())
    else:
        articles = news_cache.get_all()

    # Apply limit
    articles = articles[:limit]

    return {
        "count": len(articles),
        "articles": articles,
    }


@app.get("/news/symbol/{symbol}")
def get_news_for_symbol(symbol: str, limit: int = Query(20, ge=1, le=100)):
    """Get news for a specific symbol."""
    normalized = symbol.strip().upper()
    # Also check normalized forms
    from news_service import NEWS_TICKER_MAP
    if normalized not in NEWS_TICKER_MAP:
        # Try to normalize crypto symbols
        if "/" in normalized:
            base = normalized.split("/")[0]
            normalized = f"{base}USD"

    articles = news_cache.get_by_symbol(normalized)
    return {
        "symbol": normalized,
        "count": len(articles[:limit]),
        "articles": articles[:limit],
    }


@app.get("/news/categories")
def get_news_categories():
    """Get available news categories with article counts."""
    all_articles = news_cache.get_all()
    categories = {}
    for article in all_articles:
        cat = article.get("category", "unknown")
        if cat not in categories:
            categories[cat] = {"count": 0, "symbols": set()}
        categories[cat]["count"] += 1
        categories[cat]["symbols"].add(article.get("symbol", ""))

    result = {}
    for cat, info in categories.items():
        result[cat] = {
            "count": info["count"],
            "symbols": list(info["symbols"]),
        }

    return {"categories": result}


class AnalyzeNewsRequest(BaseModel):
    title: str
    summary: str = ""
    symbol: str = ""
    asset_type: str = ""


@app.post("/news/analyze")
def analyze_news(payload: AnalyzeNewsRequest):
    """Analyze a single news article's sentiment and market impact."""
    result = news_predictor.predict(
        title=payload.title,
        summary=payload.summary,
        symbol=payload.symbol,
        asset_type=payload.asset_type,
    )
    return result


@app.get("/news/impact/{symbol}")
def get_news_impact(symbol: str):
    """Get aggregated news sentiment impact for a symbol."""
    normalized = symbol.strip().upper()
    articles = news_cache.get_by_symbol(normalized)

    if not articles:
        return {
            "symbol": normalized,
            "article_count": 0,
            "avg_impact": 0.5,
            "sentiment_summary": "Neutral",
            "bullish_count": 0,
            "bearish_count": 0,
            "neutral_count": 0,
        }

    bullish = sum(1 for a in articles if a.get("sentiment") == "Bullish")
    bearish = sum(1 for a in articles if a.get("sentiment") == "Bearish")
    neutral = sum(1 for a in articles if a.get("sentiment") == "Neutral")

    avg_impact = sum(a.get("impact_score", 0.5) for a in articles) / len(articles)

    if bullish > bearish and bullish > neutral:
        summary = "Bullish"
    elif bearish > bullish and bearish > neutral:
        summary = "Bearish"
    else:
        summary = "Neutral"

    return {
        "symbol": normalized,
        "article_count": len(articles),
        "avg_impact": round(avg_impact, 3),
        "sentiment_summary": summary,
        "bullish_count": bullish,
        "bearish_count": bearish,
        "neutral_count": neutral,
    }


@app.get("/signal-hybrid")
def get_hybrid_signal(
    symbol: str = Query(..., description="Symbol to get hybrid signal for"),
    timeframe: str = Query("1h", description="Timeframe for analysis (1m, 5m, 15m, 30m, 1h, 4h, 1d, 1w, 1M)"),
    horizon: Optional[int] = Query(None, description="Number of future candles to predict ahead (overrides default)")
):
    """
    Get hybrid trading signal combining technical indicators + news sentiment.
    Returns BUY/SELL/HOLD with detailed explanation.
    IMPORTANT: Signal is based on the specified timeframe.
    """
    normalized = normalize_symbol(symbol)
    
    # Get current state (technical indicators)
    # NOTE: The state contains data from the LAST loaded timeframe
    # Make sure to call /load-history-timeframe with the correct timeframe BEFORE getting signal
    with states_lock:
        state = states.get(normalized)
    
    if not state:
        return {
            "symbol": normalized,
            "signal": "HOLD",
            "confidence": 0.0,
            "explanation": f"No data available for {timeframe}. Load history with /load-history-timeframe first.",
            "probabilities": {"SELL": 0.33, "HOLD": 0.34, "BUY": 0.33},
            "timeframe": timeframe
        }
    
    # Compute real technical indicators from stored candles
    candles = list(state.get("candles", []))
    price_data = state.get("price", {})
    if isinstance(price_data, dict):
        price = price_data.get("price", 0)
    else:
        price = price_data if price_data else 0
        
    technical_features = None
    if len(candles) >= 30:
        import pandas as pd
        df = pd.DataFrame(candles)
        # Ensure correct types
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        
        close = df["Close"]
        vol = df["Volume"]
        
        try:
            # RSI 14
            delta = close.diff()
            gain = delta.clip(lower=0)
            loss = (-delta).clip(lower=0)
            avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
            avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
            rs = avg_gain / (avg_loss + 1e-10)
            rsi = 100 - (100 / (1 + rs))
            
            # MACD
            ema12 = close.ewm(span=12, adjust=False).mean()
            ema26 = close.ewm(span=26, adjust=False).mean()
            macd = ema12 - ema26
            macd_signal = macd.ewm(span=9, adjust=False).mean()
            macd_hist = macd - macd_signal
            
            # SMA / EMA
            sma_20 = close.rolling(20).mean()
            sma_50 = close.rolling(50).mean()
            
            # Bollinger Bands
            bb_mid = close.rolling(20).mean()
            bb_std = close.rolling(20).std()
            bb_upper = bb_mid + 2 * bb_std
            bb_lower = bb_mid - 2 * bb_std
            
            # Volume ratio
            vol_sma = vol.rolling(20).mean()
            vol_ratio = vol / (vol_sma + 1e-10)
            
            # Momentum / Price Change
            price_change_pct = close.pct_change(1) * 100
            momentum_5 = close.pct_change(5) * 100
            momentum_10 = close.pct_change(10) * 100
            
            # Average True Range (ATR) for volatility-based TP/SL
            high = df["High"]
            low = df["Low"]
            prev_close = close.shift(1)
            tr1 = high - low
            tr2 = (high - prev_close).abs()
            tr3 = (low - prev_close).abs()
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(14).mean()
            
            tech_dict = {
                'current_price': close.iloc[-1],
                'atr': atr.iloc[-1],
                'rsi': rsi.iloc[-1],
                'macd': macd.iloc[-1],
                'macd_signal': macd_signal.iloc[-1],
                'macd_hist': macd_hist.iloc[-1],
                'sma_20': sma_20.iloc[-1],
                'sma_50': sma_50.iloc[-1],
                'ema_12': ema12.iloc[-1],
                'ema_26': ema26.iloc[-1],
                'bb_upper': bb_upper.iloc[-1],
                'bb_middle': bb_mid.iloc[-1],
                'bb_lower': bb_lower.iloc[-1],
                'volume_sma_ratio': vol_ratio.iloc[-1],
                'price_change_pct': price_change_pct.iloc[-1],
                'momentum_5': momentum_5.iloc[-1],
                'momentum_10': momentum_10.iloc[-1],
            }
            
            # Handle NaNs
            technical_features = {}
            for k, v in tech_dict.items():
                if pd.isna(v) or v == float('inf') or v == float('-inf'):
                    technical_features[k] = 0.0
                else:
                    technical_features[k] = float(v)
                    
            close_price = float(close.iloc[-1])
            if technical_features['sma_20'] == 0: technical_features['sma_20'] = close_price
            if technical_features['sma_50'] == 0: technical_features['sma_50'] = close_price
            if technical_features['rsi'] == 0.0 and close.iloc[-1] > 0: technical_features['rsi'] = 50.0

        except Exception as e:
            print(f"Error computing technicals manually: {e}")
            technical_features = None

    if technical_features is None:
        # Fallback to neutral values if not enough data
        close_price = price if price > 0 else 100
        technical_features = {
            'rsi': 50,
            'macd': 0,
            'macd_signal': 0,
            'macd_hist': 0,
            'sma_20': close_price,
            'sma_50': close_price,
            'ema_12': close_price,
            'ema_26': close_price,
            'bb_upper': close_price * 1.02,
            'bb_middle': close_price,
            'bb_lower': close_price * 0.98,
            'volume_sma_ratio': 1.0,
            'price_change_pct': 0,
            'momentum_5': 0,
            'momentum_10': 0,
            'atr': close_price * 0.005, # Default 0.5% ATR fallback
            'current_price': close_price,
        }

    # Get news sentiment for this symbol
    # Normalize symbol for news lookup
    news_symbol = normalized
    if "/" in news_symbol:
        base = news_symbol.split("/")[0]
        news_symbol = f"{base}USD"
    
    articles = news_cache.get_by_symbol(news_symbol)
    
    if articles:
        # Calculate aggregated sentiment
        bullish_probs = [a.get('probabilities', {}).get('Bullish', 0.5) for a in articles]
        bearish_probs = [a.get('probabilities', {}).get('Bearish', 0.5) for a in articles]
        impacts = [a.get('impact_score', 0.5) for a in articles]
        
        sentiment_features = {
            'news_bullish_prob': sum(bullish_probs) / len(bullish_probs),
            'news_bearish_prob': sum(bearish_probs) / len(bearish_probs),
            'news_impact_score': sum(impacts) / len(impacts),
            'news_article_count': len(articles),
        }
    else:
        sentiment_features = None
    
    # Generate hybrid signal
    try:
        result = hybrid_signal_gen.predict(technical_features, sentiment_features)
        result['symbol'] = normalized
        result['timeframe'] = timeframe
        _h = horizon if horizon is not None else HORIZON_MAP.get(timeframe, 3)
        _tf_unit = {
            '1m': 'minute', '5m': 'minutes', '15m': 'minutes', '30m': 'minutes',
            '1h': 'hour', '4h': 'hours', '1d': 'day', '1w': 'week', '1M': 'month'
        }
        _tf_mult = {
            '1m': 1, '5m': 5, '15m': 15, '30m': 30,
            '1h': 1, '4h': 4, '1d': 1, '1w': 1, '1M': 1
        }
        _mult = _tf_mult.get(timeframe, 1)
        _unit = _tf_unit.get(timeframe, 'candle')
        _total = _h * _mult
        result['horizon'] = _h
        result['horizon_label'] = f"next {_h} candle{'s' if _h > 1 else ''} ({_total} {_unit}{'s' if _total > 1 else ''} ahead)"
        result['note'] = f"Signal based on {timeframe} timeframe — predicting {_total} {_unit}{'s' if _total > 1 else ''} ahead."
        return result
    except Exception as e:
        return {
            "symbol": normalized,
            "signal": "HOLD",
            "confidence": 0.0,
            "explanation": f"Error generating signal: {str(e)}",
            "probabilities": {"SELL": 0.33, "HOLD": 0.34, "BUY": 0.33},
            "timeframe": timeframe
        }


# ─────────────────────────────────────────────
# Hybrid Model Retrain Endpoints
# ─────────────────────────────────────────────

@app.post("/retrain-hybrid")
def retrain_hybrid(
    horizon: int = Query(..., ge=1, le=100, description="New horizon for retraining"),
):
    """
    Retrain the Hybrid Signal Model on real data with a new horizon.
    Runs in the background — poll /retrain-hybrid/status to check progress.
    """
    global _hybrid_retrain_state

    if _hybrid_retrain_state["status"] == "training":
        return {
            "status": "already_training",
            "horizon": _hybrid_retrain_state["horizon"],
            "message": f"Already retraining for horizon={_hybrid_retrain_state['horizon']}. Please wait.",
        }

    _hybrid_retrain_state = {
        "status": "training",
        "horizon": horizon,
        "started_at": time.time(),
        "error": None,
    }

    def _do_retrain():
        global _hybrid_retrain_state
        try:
            from hybrid_signal_model import train_hybrid_from_real_data
            print(f"🔄 Retraining Hybrid Model with horizon={horizon}...")
            train_hybrid_from_real_data(hybrid_signal_gen, horizon=horizon)
            hybrid_signal_gen.load()
            elapsed = round(time.time() - _hybrid_retrain_state["started_at"], 1)
            _hybrid_retrain_state["status"] = "ready"
            _hybrid_retrain_state["elapsed"] = elapsed
            print(f"✅ Hybrid Model retrained (horizon={horizon}) in {elapsed}s")
        except Exception as e:
            _hybrid_retrain_state["status"] = "error"
            _hybrid_retrain_state["error"] = str(e)
            print(f"❌ Hybrid retrain failed: {e}")

    threading.Thread(target=_do_retrain, daemon=True).start()
    return {"status": "training", "horizon": horizon, "message": "Retraining started in background."}


@app.get("/retrain-hybrid/status")
def get_retrain_hybrid_status():
    """Check the current retraining status of the Hybrid Model."""
    return _hybrid_retrain_state


# ─────────────────────────────────────────────
# Paper Trading Endpoints (التداول الوهمي)
# ─────────────────────────────────────────────

class OpenTradeRequest(BaseModel):
    user_id: str
    symbol: str
    type: str  # "BUY" or "SELL"
    quantity: float
    signal_confidence: Optional[float] = None
    notes: Optional[str] = None


class CloseTradeRequest(BaseModel):
    user_id: str
    trade_id: str


class ResetPortfolioRequest(BaseModel):
    user_id: str
    initial_balance: float = 10000.0


@app.post("/paper-trading/open")
def open_paper_trade(payload: OpenTradeRequest):
    """فتح صفقة تداول وهمية"""
    normalized = normalize_symbol(payload.symbol)
    
    # Get current price
    with states_lock:
        state = states.get(normalized)
    
    if not state or not state.get("price"):
        return {
            "success": False,
            "error": "السعر الحالي غير متوفر. يرجى بدء البث أولاً."
        }
    
    current_price = state["price"].get("price", 0)
    if current_price <= 0:
        return {
            "success": False,
            "error": "السعر غير صالح"
        }
    
    result = paper_trading.open_trade(
        user_id=payload.user_id,
        symbol=normalized,
        trade_type=payload.type,
        quantity=payload.quantity,
        entry_price=current_price,
        signal_confidence=payload.signal_confidence,
        notes=payload.notes
    )
    
    return result


@app.post("/paper-trading/close")
def close_paper_trade(payload: CloseTradeRequest):
    """إغلاق صفقة تداول وهمية"""
    
    # Get current price from the trade's symbol
    portfolio = paper_trading.load_portfolio(payload.user_id)
    if not portfolio:
        return {"success": False, "error": "المحفظة غير موجودة"}
    
    # Find the trade to get its symbol
    trade = None
    for t in portfolio.open_positions:
        if t.id == payload.trade_id:
            trade = t
            break
    
    if not trade:
        return {"success": False, "error": "الصفقة غير موجودة"}
    
    # Get current price for this symbol
    with states_lock:
        state = states.get(trade.symbol)
    
    if not state or not state.get("price"):
        return {
            "success": False,
            "error": "السعر الحالي غير متوفر"
        }
    
    current_price = state["price"].get("price", 0)
    if current_price <= 0:
        return {"success": False, "error": "السعر غير صالح"}
    
    result = paper_trading.close_trade(
        user_id=payload.user_id,
        trade_id=payload.trade_id,
        exit_price=current_price
    )
    
    return result


@app.get("/paper-trading/portfolio/{user_id}")
def get_paper_portfolio(user_id: str):
    """الحصول على ملخص المحفظة"""
    return paper_trading.get_portfolio_summary(user_id)


@app.get("/paper-trading/history/{user_id}")
def get_paper_history(user_id: str, limit: int = Query(50, ge=1, le=200)):
    """الحصول على سجل الصفقات"""
    return paper_trading.get_trade_history(user_id, limit)


@app.get("/paper-trading/stats/{user_id}")
def get_paper_stats(user_id: str):
    """إحصائيات الأداء التفصيلية"""
    return paper_trading.get_performance_stats(user_id)


@app.post("/paper-trading/reset")
def reset_paper_portfolio(payload: ResetPortfolioRequest):
    """إعادة تعيين المحفظة"""
    return paper_trading.reset_portfolio(payload.user_id, payload.initial_balance)
