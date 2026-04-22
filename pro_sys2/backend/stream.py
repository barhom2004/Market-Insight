import ccxt
import pandas as pd
import time
from datetime import datetime, timedelta

from feature_engineering2 import compute_all_features
from train_model import train_xgb_model


def safe_fetch(fetch_function, retries=5, delay=2):
    for i in range(retries):
        try:
            return fetch_function()
        except Exception as e:
            print(f"[Retry {i+1}] بسبب: {e}")
            time.sleep(delay)
    raise Exception("فشل بعد عدة محاولات")


def stream_market_data(symbol: str, timeframe: str, total=5000, horizon: int = None):

    exchange = ccxt.binance({
        'timeout': 30000,
        'enableRateLimit': True
    })

    safe_fetch(lambda: exchange.load_markets())

    timeframe_map = {
        '1m': 60,
        '3m': 180,
        '5m': 300,
        '15m': 900,
        '30m': 1800,
        '1h': 3600,
        '4h': 14400,
        '1d': 86400,
        '1w': 604800,
        '1M': 2592000,
    }

    tf_seconds = timeframe_map[timeframe]
    tf_ms = tf_seconds * 1000

    print("⏳ تحميل البيانات...")

    # أول 1000 شمعة
    all_data = safe_fetch(
        lambda: exchange.fetch_ohlcv(symbol, timeframe, limit=1000)
    )

    # تحميل باقي البيانات
    while len(all_data) < total:
        since = all_data[0][0] - (1000 * tf_ms)

        print(f"⏳ تحميل دفعة إضافية... (الحالي: {len(all_data)})")

        batch = safe_fetch(
            lambda: exchange.fetch_ohlcv(
                symbol, timeframe, since=since, limit=1000
            )
        )

        # حماية من التكرار أو التوقف
        if not batch or batch[0][0] == all_data[0][0]:
            print("⚠️ توقف التحميل - لا توجد بيانات جديدة")
            break

        all_data = batch + all_data

        # إزالة التكرار
        all_data = sorted(
            list({x[0]: x for x in all_data}.values()),
            key=lambda x: x[0]
        )

        time.sleep(1)

    print(f"✅ تم تحميل {len(all_data)} شمعة")

    # قص البيانات حسب المطلوب
    all_data = all_data[-total:]

    df = pd.DataFrame(all_data, columns=[
        'timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'
    ])

    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)

    print("✅ Data جاهزة - نحسب المؤشرات...")

    # Horizon and threshold based on timeframe
    horizon_map = {
        "1m": 5, "5m": 6, "15m": 4, "30m": 4,
        "1h": 3, "4h": 3, "1d": 3, "1w": 2, "1M": 2
    }
    threshold_map = {
        "1m": 0.0005, "5m": 0.0008, "15m": 0.001, "30m": 0.0015,
        "1h": 0.002, "4h": 0.003, "1d": 0.005, "1w": 0.01, "1M": 0.02
    }
    
    if horizon is None:
        horizon = horizon_map.get(timeframe, 3)
    threshold = threshold_map.get(timeframe, 0.001)
    
    print(f"⚙️ Using horizon={horizon} candles, threshold={threshold*100:.2f}% for {timeframe}")

    df = compute_all_features(df, is_new_row=False, threshold=threshold, horizon=horizon)

    print("🚀 تدريب المودل...")

    df_for_training = df.reset_index()
    preds, model, scaler = train_xgb_model(df_for_training, threshold=threshold, horizon=horizon)

    print("🔥 المودل جاهز للبث")

    last_candle_time = df.index[-1]
    next_candle_close = last_candle_time + timedelta(seconds=tf_seconds)

    # =========================
    # 🔴 STREAM LOOP
    # =========================
    while True:
        try:
            ticker = safe_fetch(lambda: exchange.fetch_ticker(symbol))
            current_price = ticker['last']

            now = datetime.utcnow()

            yield {
                "type": "live_price",
                "price": current_price,
                "time": now.isoformat()
            }

            # عند إغلاق شمعة جديدة
            if now >= next_candle_close:
                print("🕯️ شمعة جديدة")

                latest = safe_fetch(
                    lambda: exchange.fetch_ohlcv(symbol, timeframe, limit=2)
                )[-1]

                new_row = {
                    'timestamp': pd.to_datetime(latest[0], unit='ms'),
                    'Open': latest[1],
                    'High': latest[2],
                    'Low': latest[3],
                    'Close': latest[4],
                    'Volume': latest[5]
                }

                new_df = pd.DataFrame([new_row])
                new_df.set_index('timestamp', inplace=True)

                df = pd.concat([df, new_df])

                df = compute_all_features(df, is_new_row=True)
                df = df.tail(total)

                next_candle_close = new_row['timestamp'] + timedelta(seconds=tf_seconds)

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

                print(f"🔥 Signal: {signal} | Buy: {buy_prob:.2f} | Sell: {sell_prob:.2f}")

                yield {
                    "type": "signal",
                    "signal": signal,
                    "buy_prob": float(buy_prob),
                    "sell_prob": float(sell_prob)
                }

            time.sleep(1)

        except Exception as e:
            print("Error:", e)
            time.sleep(3)