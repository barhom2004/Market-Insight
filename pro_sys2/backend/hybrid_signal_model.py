"""
Hybrid Trading Signal Model
============================
Combines technical indicators from price data + news sentiment
to generate high-confidence trading signals (BUY/SELL/HOLD) with explanations.

Features:
- Technical: RSI, MACD, Moving Averages, Volume, Momentum
- Sentiment: News sentiment scores (Bullish/Bearish probability)
- Output: BUY/SELL/HOLD + detailed explanation of why
"""

import os
import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "hybrid_model_artifacts")
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, "hybrid_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "hybrid_scaler.pkl")


class HybridSignalGenerator:
    """
    Generates trading signals by combining technical + sentiment features.
    """
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self._loaded = False
        
        # Feature names for reference
        self.technical_features = [
            'rsi', 'macd', 'macd_signal', 'macd_hist',
            'sma_20', 'sma_50', 'ema_12', 'ema_26',
            'bb_upper', 'bb_middle', 'bb_lower',
            'volume_sma_ratio', 'price_change_pct',
            'momentum_5', 'momentum_10'
        ]
        
        self.sentiment_features = [
            'news_bullish_prob',
            'news_bearish_prob',
            'news_impact_score',
            'news_article_count'
        ]
        
    def train(self, df_technical, df_sentiment=None):
        """
        Train the hybrid model.
        
        Args:
            df_technical: DataFrame with technical indicators + 'target' column
                         target: 0=SELL, 1=HOLD, 2=BUY
            df_sentiment: Optional DataFrame with news sentiment features
        """
        print("🚀 Training Hybrid Signal Model...")
        
        # Merge technical + sentiment if provided
        if df_sentiment is not None and len(df_sentiment) > 0:
            df = df_technical.merge(df_sentiment, left_index=True, right_index=True, how='left')
            # Fill missing sentiment with neutral values
            df['news_bullish_prob'] = df.get('news_bullish_prob', 0.5).fillna(0.5)
            df['news_bearish_prob'] = df.get('news_bearish_prob', 0.5).fillna(0.5)
            df['news_impact_score'] = df.get('news_impact_score', 0.5).fillna(0.5)
            df['news_article_count'] = df.get('news_article_count', 0).fillna(0)
            feature_cols = self.technical_features + self.sentiment_features
        else:
            df = df_technical.copy()
            # Add neutral sentiment features
            df['news_bullish_prob'] = 0.5
            df['news_bearish_prob'] = 0.5
            df['news_impact_score'] = 0.5
            df['news_article_count'] = 0
            feature_cols = self.technical_features + self.sentiment_features
        
        # Drop rows with NaN in features or target
        df = df.dropna(subset=feature_cols + ['target'])
        
        if len(df) < 100:
            raise ValueError(f"Not enough training data: {len(df)} rows")
        
        X = df[feature_cols].values
        y = df['target'].values
        
        print(f"   Training samples: {len(X)}")
        print(f"   Features: {len(feature_cols)} ({len(self.technical_features)} technical + {len(self.sentiment_features)} sentiment)")
        print(f"   Target distribution: {np.bincount(y.astype(int))}")
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train XGBoost classifier
        self.model = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='mlogloss'
        )
        
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        print(f"\n📊 Model Performance:")
        print(f"   Accuracy: {acc:.2%}")
        print("\n   Classification Report:")
        target_names = ['SELL', 'HOLD', 'BUY']
        print(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\n   Top 10 Most Important Features:")
        for idx, row in feature_importance.head(10).iterrows():
            print(f"      {row['feature']}: {row['importance']:.4f}")
        
        self._loaded = True
        return acc
    
    def save(self):
        """Save model to disk."""
        if not self._loaded:
            raise ValueError("Model not trained yet")
        
        with open(MODEL_PATH, 'wb') as f:
            pickle.dump(self.model, f)
        
        print(f"✅ Hybrid model saved to {MODEL_DIR}")
    
    def load(self):
        """Load model from disk."""
        if not os.path.exists(MODEL_PATH):
            print("⚠️ Hybrid model not found. Train it first.")
            return False
        
        with open(MODEL_PATH, 'rb') as f:
            self.model = pickle.load(f)
        
        self._loaded = True
        print("✅ Hybrid model loaded")
        return True
    
    def predict(self, technical_features, sentiment_features=None):
        """
        Generate trading signal with explanation.
        
        Args:
            technical_features: dict with technical indicator values
            sentiment_features: dict with news sentiment (optional)
                - news_bullish_prob
                - news_bearish_prob
                - news_impact_score
                - news_article_count
        
        Returns:
            {
                'signal': 'BUY' | 'SELL' | 'HOLD',
                'confidence': 0.0-1.0,
                'explanation': 'Detailed reason',
                'probabilities': {'SELL': 0.1, 'HOLD': 0.2, 'BUY': 0.7}
            }
        """
        if not self._loaded:
            raise ValueError("Model not loaded. Call load() first.")
        
        # Build feature vector
        feature_vector = []
        for feat in self.technical_features:
            feature_vector.append(technical_features.get(feat, 0))
        
        # Add sentiment features (default to neutral if not provided)
        if sentiment_features:
            feature_vector.append(sentiment_features.get('news_bullish_prob', 0.5))
            feature_vector.append(sentiment_features.get('news_bearish_prob', 0.5))
            feature_vector.append(sentiment_features.get('news_impact_score', 0.5))
            feature_vector.append(sentiment_features.get('news_article_count', 0))
        else:
            feature_vector.extend([0.5, 0.5, 0.5, 0])
        
        X = np.array(feature_vector).reshape(1, -1)
        
        # Predict
        probs = self.model.predict_proba(X)[0]
        signal_idx = np.argmax(probs)
        signals = ['SELL', 'HOLD', 'BUY']
        signal = signals[signal_idx]
        confidence = probs[signal_idx]
        
        # Confidence threshold for 3-class model:
        # Random baseline = 33%, so 42% is a meaningful edge.
        # 60% was too aggressive — caused near-constant HOLD.
        dominant_signal = signal
        dominant_confidence = confidence
        if signal in ['BUY', 'SELL'] and confidence < 0.42:
            signal = 'HOLD'
            confidence = probs[1]  # HOLD confidence

        # Reliability: how far above the random baseline (33% for 3 classes)
        max_prob = float(max(probs))
        reliability_score = max((max_prob - 1/3) / (1 - 1/3), 0.0)
        if reliability_score > 0.50:
            signal_strength = 'Strong'
        elif reliability_score > 0.25:
            signal_strength = 'Moderate'
        elif reliability_score > 0.10:
            signal_strength = 'Weak'
        else:
            signal_strength = 'Uncertain'

        # Flag when BUY ≈ SELL (market is undecided)
        is_market_undecided = abs(float(probs[0]) - float(probs[2])) < 0.05

        # Calculate TP / SL
        current_price = technical_features.get('current_price', 0)
        atr = technical_features.get('atr', 0)
        take_profit = None
        stop_loss = None

        if current_price > 0 and atr > 0:
            if signal == 'BUY':
                take_profit = current_price + (atr * 2.0)
                stop_loss = current_price - (atr * 1.5)
            elif signal == 'SELL':
                take_profit = current_price - (atr * 2.0)
                stop_loss = current_price + (atr * 1.5)

        # Build precise data-backed explanation
        explanation = self._build_explanation(
            signal, confidence, technical_features, sentiment_features
        )
        
        return {
            'signal': signal,
            'confidence': float(confidence),
            'dominant_signal': dominant_signal,
            'dominant_confidence': float(dominant_confidence),
            'reliability_score': round(reliability_score, 3),
            'signal_strength': signal_strength,
            'is_market_undecided': bool(is_market_undecided),
            'explanation': explanation,
            'take_profit': float(take_profit) if take_profit else None,
            'stop_loss': float(stop_loss) if stop_loss else None,
            'technical_features': technical_features,
            'sentiment_features': sentiment_features if sentiment_features else {},
            'probabilities': {
                'SELL': float(probs[0]),
                'HOLD': float(probs[1]),
                'BUY': float(probs[2])
            }
        }
    
    def _build_explanation(self, signal, confidence, tech, sent):
        """Generate human-readable explanation for the signal."""
        reasons = []
        
        # Technical reasons
        rsi = tech.get('rsi', 50)
        macd_hist = tech.get('macd_hist', 0)
        price_change = tech.get('price_change_pct', 0)
        sma_20 = tech.get('sma_20', 0)
        sma_50 = tech.get('sma_50', 0)
        
        if signal == 'BUY':
            if rsi < 30:
                reasons.append(f"RSI strongly oversold at {rsi:.1f}")
            elif rsi < 45:
                reasons.append(f"RSI favorable at {rsi:.1f}")
            
            if macd_hist > 0:
                reasons.append(f"MACD bullish histogram ({macd_hist:.4f})")
            
            if sma_20 > sma_50:
                reasons.append("Golden cross / uptrend confirmed")
            
            if price_change > 0:
                reasons.append(f"Positive momentum (+{price_change:.2f}%)")
        
        elif signal == 'SELL':
            if rsi > 70:
                reasons.append(f"RSI strongly overbought at {rsi:.1f}")
            elif rsi > 55:
                reasons.append(f"RSI unfavorable at {rsi:.1f}")
            
            if macd_hist < 0:
                reasons.append(f"MACD bearish histogram ({macd_hist:.4f})")
            
            if sma_20 < sma_50:
                reasons.append("Death cross / downtrend confirmed")
            
            if price_change < 0:
                reasons.append(f"Negative momentum ({price_change:.2f}%)")
        
        else:  # HOLD
            if rsi < 30:
                reasons.append(f"RSI oversold at {rsi:.1f} — watch for reversal")
            elif rsi < 40:
                reasons.append(f"RSI in bearish zone at {rsi:.1f} — conflicting signals")
            elif rsi > 70:
                reasons.append(f"RSI overbought at {rsi:.1f} — watch for reversal")
            elif rsi > 60:
                reasons.append(f"RSI in bullish zone at {rsi:.1f} — conflicting signals")
            else:
                reasons.append(f"RSI neutral at {rsi:.1f}")
            if abs(macd_hist) < 0.1:
                reasons.append(f"MACD indecisive ({macd_hist:.4f})")
        
        # Sentiment reasons
        if sent:
            bullish_prob = sent.get('news_bullish_prob', 0.5)
            bearish_prob = sent.get('news_bearish_prob', 0.5)
            impact = sent.get('news_impact_score', 0.5)
            count = sent.get('news_article_count', 0)
            
            if count > 0:
                if bullish_prob > 0.6:
                    reasons.append(f"News strongly bullish ({bullish_prob*100:.1f}%, {int(count)} articles)")
                elif bearish_prob > 0.6:
                    reasons.append(f"News strongly bearish ({bearish_prob*100:.1f}%, {int(count)} articles)")
                elif impact > 0.6:
                    reasons.append(f"High market instability detected ({impact*100:.1f}%)")
        
        if not reasons:
            reasons.append(f"Mixed technical signals, awaiting divergence")
        
        explanation = " • ".join(reasons)
        return f"{signal} ({confidence*100:.1f}% confidence): {explanation}"


def train_hybrid_from_real_data(
    generator=None,
    symbols=None,
    period="1y",
    timeframe="1h",
    horizon=3,
    threshold=0.002,
):
    """
    Train the hybrid model on REAL historical market data via Yahoo Finance.
    Replaces the synthetic data approach entirely.
    """
    import yfinance as yf

    if symbols is None:
        symbols = ["BTC-USD", "ETH-USD", "GC=F", "EURUSD=X", "AAPL", "TSLA"]

    print(f"📊 Downloading real data for Hybrid Model ({len(symbols)} symbols, {period}, {timeframe})...")

    all_dfs = []
    for sym in symbols:
        try:
            print(f"   ⬇️  {sym} …")
            raw = yf.download(sym, period=period, interval=timeframe, progress=False, auto_adjust=True)
            if len(raw) < 100:
                print(f"   ⚠️  {sym}: only {len(raw)} rows — skipping")
                continue

            # Flatten MultiIndex columns if present (yfinance ≥ 0.2)
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)

            close = raw["Close"].squeeze()
            high  = raw["High"].squeeze()
            low   = raw["Low"].squeeze()
            vol   = raw["Volume"].squeeze() if "Volume" in raw.columns else pd.Series(1.0, index=raw.index)

            # ── Technical indicators ──────────────────────────────────
            delta     = close.diff()
            avg_gain  = delta.clip(lower=0).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
            avg_loss  = (-delta).clip(lower=0).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
            rsi       = 100 - (100 / (1 + avg_gain / (avg_loss + 1e-10)))

            ema12      = close.ewm(span=12, adjust=False).mean()
            ema26      = close.ewm(span=26, adjust=False).mean()
            macd_line  = ema12 - ema26
            macd_sig   = macd_line.ewm(span=9, adjust=False).mean()
            macd_hist  = macd_line - macd_sig

            sma20 = close.rolling(20).mean()
            sma50 = close.rolling(50).mean()

            bb_mid   = close.rolling(20).mean()
            bb_std   = close.rolling(20).std()
            bb_upper = bb_mid + 2 * bb_std
            bb_lower = bb_mid - 2 * bb_std

            vol_sma   = vol.rolling(20).mean()
            vol_ratio = vol / (vol_sma + 1e-10)

            pct_change  = close.pct_change(1) * 100
            momentum_5  = close.pct_change(5) * 100
            momentum_10 = close.pct_change(10) * 100

            # ── Future return → target label ──────────────────────────
            future_ret = (close.shift(-horizon) - close) / close

            feat = pd.DataFrame({
                "rsi":               rsi,
                "macd":              macd_line,
                "macd_signal":       macd_sig,
                "macd_hist":         macd_hist,
                "sma_20":            sma20,
                "sma_50":            sma50,
                "ema_12":            ema12,
                "ema_26":            ema26,
                "bb_upper":          bb_upper,
                "bb_middle":         bb_mid,
                "bb_lower":          bb_lower,
                "volume_sma_ratio":  vol_ratio,
                "price_change_pct":  pct_change,
                "momentum_5":        momentum_5,
                "momentum_10":       momentum_10,
                "news_bullish_prob":  0.5,
                "news_bearish_prob":  0.5,
                "news_impact_score":  0.5,
                "news_article_count": 0,
                "future_return":     future_ret,
            })

            feat["target"] = np.where(
                feat["future_return"] > threshold, 2,
                np.where(feat["future_return"] < -threshold, 0, 1)
            )
            feat = feat.drop(columns=["future_return"])
            feat = feat.dropna()
            feat = feat.iloc[:-horizon]   # Remove rows with no future label

            all_dfs.append(feat)
            print(f"   ✅ {sym}: {len(feat)} samples  |  dist={feat['target'].value_counts().to_dict()}")

        except Exception as e:
            print(f"   ❌ {sym}: {e}")

    if not all_dfs:
        raise ValueError("No real data downloaded — check internet connection.")

    df_combined = pd.concat(all_dfs, ignore_index=True)
    print(f"\n📦 Total training samples: {len(df_combined)}")
    print(f"   Target distribution: {df_combined['target'].value_counts().to_dict()}")

    if generator is None:
        generator = HybridSignalGenerator()

    generator.train(df_combined)
    generator.save()

    # Write version flag so startup knows this is a real-data model
    flag_path = os.path.join(MODEL_DIR, "data_version.txt")
    with open(flag_path, "w") as f:
        f.write("real_data_v1")

    print("✅ Hybrid model trained on REAL data and saved.")
    return generator


def create_synthetic_training_data():
    """
    Create synthetic training data for demonstration.
    In production, use real historical data with labeled outcomes.
    """
    print("📦 Creating synthetic training data...")
    
    np.random.seed(42)
    n_samples = 5000
    
    data = {
        # Technical indicators
        'rsi': np.random.uniform(20, 80, n_samples),
        'macd': np.random.uniform(-2, 2, n_samples),
        'macd_signal': np.random.uniform(-2, 2, n_samples),
        'macd_hist': np.random.uniform(-1, 1, n_samples),
        'sma_20': np.random.uniform(90, 110, n_samples),
        'sma_50': np.random.uniform(90, 110, n_samples),
        'ema_12': np.random.uniform(90, 110, n_samples),
        'ema_26': np.random.uniform(90, 110, n_samples),
        'bb_upper': np.random.uniform(105, 115, n_samples),
        'bb_middle': np.random.uniform(95, 105, n_samples),
        'bb_lower': np.random.uniform(85, 95, n_samples),
        'volume_sma_ratio': np.random.uniform(0.5, 2.0, n_samples),
        'price_change_pct': np.random.uniform(-5, 5, n_samples),
        'momentum_5': np.random.uniform(-3, 3, n_samples),
        'momentum_10': np.random.uniform(-5, 5, n_samples),
        
        # Sentiment features
        'news_bullish_prob': np.random.uniform(0.2, 0.8, n_samples),
        'news_bearish_prob': np.random.uniform(0.2, 0.8, n_samples),
        'news_impact_score': np.random.uniform(0.3, 0.9, n_samples),
        'news_article_count': np.random.randint(0, 20, n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # Create target based on rules (simplified)
    def generate_target(row):
        score = 0
        
        # RSI
        if row['rsi'] < 30:
            score += 2  # Oversold -> BUY
        elif row['rsi'] > 70:
            score -= 2  # Overbought -> SELL
        
        # MACD
        if row['macd_hist'] > 0.5:
            score += 1
        elif row['macd_hist'] < -0.5:
            score -= 1
        
        # Trend
        if row['sma_20'] > row['sma_50']:
            score += 1
        else:
            score -= 1
        
        # Momentum
        if row['price_change_pct'] > 2:
            score += 1
        elif row['price_change_pct'] < -2:
            score -= 1
        
        # Sentiment
        if row['news_bullish_prob'] > 0.65:
            score += 1
        elif row['news_bearish_prob'] > 0.65:
            score -= 1
        
        # Map to signal
        if score >= 3:
            return 2  # BUY
        elif score <= -3:
            return 0  # SELL
        else:
            return 1  # HOLD
    
    df['target'] = df.apply(generate_target, axis=1)
    
    print(f"   Generated {len(df)} samples")
    print(f"   Target distribution: {df['target'].value_counts().to_dict()}")
    
    return df


if __name__ == "__main__":
    # Train the hybrid model
    generator = HybridSignalGenerator()
    
    # Create synthetic data (replace with real data in production)
    df_train = create_synthetic_training_data()
    
    # Train
    accuracy = generator.train(df_train)
    
    # Save
    generator.save()
    
    # Test prediction
    print("\n🧪 Test Prediction:")
    test_tech = {
        'rsi': 25,  # Oversold
        'macd': 0.5,
        'macd_signal': 0.3,
        'macd_hist': 0.2,  # Bullish
        'sma_20': 105,
        'sma_50': 100,  # Uptrend
        'ema_12': 106,
        'ema_26': 102,
        'bb_upper': 110,
        'bb_middle': 100,
        'bb_lower': 90,
        'volume_sma_ratio': 1.5,
        'price_change_pct': 2.5,  # Positive momentum
        'momentum_5': 1.2,
        'momentum_10': 2.0,
    }
    
    test_sent = {
        'news_bullish_prob': 0.75,
        'news_bearish_prob': 0.25,
        'news_impact_score': 0.8,
        'news_article_count': 5
    }
    
    result = generator.predict(test_tech, test_sent)
    print(f"\n   Signal: {result['signal']}")
    print(f"   Confidence: {result['confidence']:.2%}")
    print(f"   Explanation: {result['explanation']}")
    print(f"   Probabilities: {result['probabilities']}")
