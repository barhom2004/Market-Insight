"""
News Sentiment Model
====================
Trains a TF-IDF + XGBoost model on historical news data to predict
how news sentiment affects market prices.

The model:
1. Loads the CSV with columns: symbol, asset_type, published_date, source_url,
   image_url, title, content, summary, language, sentiment
2. Builds TF-IDF features from title + summary
3. Trains XGBoost to predict sentiment (Bullish / Bearish / Neutral)
4. Provides a predict function for live news
"""

import os
import pickle
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
from scipy.sparse import hstack, csr_matrix

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(BASE_DIR, "..", "..")
CSV_PATH = os.path.join(PROJECT_ROOT, "filtered_markets_news.csv")
MODEL_DIR = os.path.join(BASE_DIR, "news_model_artifacts")

# Symbol → category mapping
SYMBOL_CATEGORY = {
    "AAPL": "stocks", "AMZN": "stocks", "TSLA": "stocks",
    "EURUSD": "fx", "GBPUSD": "fx", "EURGBP": "fx",
    "BTCUSD": "crypto", "ETHUSD": "crypto", "BNBUSD": "crypto",
    "BTC/USDT": "crypto", "ETH/USDT": "crypto", "BNB/USDT": "crypto",
    "XAUUSD": "metals", "XAGUSD": "metals", "XPTUSD": "metals",
}

# Normalize symbol names from CSV to our standard codes
SYMBOL_NORMALIZE = {
    "XAUUSD": "XAUUSD", "XAGUSD": "XAGUSD", "XPTUSD": "XPTUSD",
    "EURUSD": "EURUSD", "GBPUSD": "GBPUSD", "EURGBP": "EURGBP",
    "AAPL": "AAPL", "AMZN": "AMZN", "TSLA": "TSLA",
    "BTCUSD": "BTCUSD", "ETHUSD": "ETHUSD", "BNBUSD": "BNBUSD",
    "BTC/USDT": "BTCUSD", "ETH/USDT": "ETHUSD", "BNB/USDT": "BNBUSD",
}

# Normalize CSV symbols to standard codes
SYMBOL_CSV_NORMALIZE = {
    "BTC": "BTCUSD",
    "ETH": "ETHUSD",
    "BNB": "BNBUSD",
    "XAUUSD": "XAUUSD",
    "XAGUSD": "XAGUSD",
    "XPTUSD": "XPTUSD",
    "EURUSD": "EURUSD",
    "GBPUSD": "GBPUSD",
    "EURGBP": "EURGBP",
    "AAPL": "AAPL",
    "AMZN": "AMZN",
    "TSLA": "TSLA",
    "MSFT": "MSFT",
}

# Sentiment label mapping (CSV uses positive/negative)
SENTIMENT_MAP = {
    "positive": 1,
    "negative": 0,
    # Also support other formats just in case
    "bullish": 1,
    "somewhat-bullish": 1,
    "neutral": 0,  # treat neutral as 0 for binary
    "somewhat-bearish": 0,
    "bearish": 0,
}

SENTIMENT_LABELS = {0: "Bearish", 1: "Bullish"}

# Impact score: how strongly this sentiment affects price
IMPACT_SCORES = {
    "Bullish": 0.85,
    "Neutral": 0.50,
    "Bearish": 0.15,
}


def clean_text(text):
    """Clean and prepare text for TF-IDF."""
    if pd.isna(text) or not isinstance(text, str):
        return ""
    text = text.strip()
    # Remove very short texts
    if len(text) < 10:
        return ""
    return text.lower()


def load_and_prepare_data(csv_path=CSV_PATH, max_rows=100000):
    """Load CSV and prepare for training."""
    print(f"📰 Loading news data from {csv_path}...")

    # Read with chunking for large files
    chunks = []
    for chunk in pd.read_csv(csv_path, chunksize=50000, low_memory=False):
        chunks.append(chunk)
        if len(chunks) * 50000 >= max_rows:
            break

    df = pd.concat(chunks, ignore_index=True)
    df = df.head(max_rows)
    print(f"   Loaded {len(df)} rows")

    # Clean columns
    df.columns = df.columns.str.strip().str.lower()

    # Ensure required columns exist
    required = ['symbol', 'title', 'sentiment']
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Clean text fields
    df['title_clean'] = df['title'].apply(clean_text)
    # Summary is NaN in most rows, so use title as fallback
    df['summary_clean'] = df['summary'].fillna('').apply(clean_text)
    df['combined_text'] = df['title_clean']

    # Filter out empty texts (only need title)
    df = df[df['title_clean'].str.len() > 10].copy()

    # Map sentiment to numeric labels (CSV uses lowercase: positive/negative)
    df['sentiment_clean'] = df['sentiment'].astype(str).str.strip().str.lower()
    df['label'] = df['sentiment_clean'].map(SENTIMENT_MAP)
    df = df.dropna(subset=['label'])
    df['label'] = df['label'].astype(int)

    # Normalize symbols (CSV has BTC, ETH etc. -> BTCUSD, ETHUSD)
    df['symbol_clean'] = df['symbol'].str.strip().str.upper().map(SYMBOL_CSV_NORMALIZE)
    df = df.dropna(subset=['symbol_clean'])

    # Parse dates
    df['published_date'] = pd.to_datetime(df.get('published_date', None), errors='coerce')

    print(f"   After cleaning: {len(df)} rows")
    print(f"   Sentiment distribution:\n{df['label'].value_counts()}")
    print(f"   Symbols: {df['symbol_clean'].nunique()} unique")

    return df


def train_news_model(df=None, csv_path=CSV_PATH):
    """Train the news sentiment model."""

    if df is None:
        df = load_and_prepare_data(csv_path)

    print("\n🚀 Training News Sentiment Model...")

    # ── TF-IDF on title (primary text feature) ──
    tfidf_title = TfidfVectorizer(
        max_features=8000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
    )
    X_title = tfidf_title.fit_transform(df['title_clean'])

    # ── TF-IDF on summary (many rows are empty, use min_df=1) ──
    tfidf_summary = TfidfVectorizer(
        max_features=3000,
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95,
        sublinear_tf=True,
    )
    X_summary = tfidf_summary.fit_transform(df['summary_clean'])

    # ── Symbol one-hot encoding ──
    le_symbol = LabelEncoder()
    symbol_encoded = le_symbol.fit_transform(df['symbol_clean'])
    X_symbol = csr_matrix(pd.get_dummies(symbol_encoded).values)

    # ── Asset type encoding ──
    asset_types = df.get('asset_type', pd.Series(['unknown'] * len(df), dtype=str))
    le_asset = LabelEncoder()
    asset_encoded = le_asset.fit_transform(asset_types.fillna('unknown').str.strip().str.lower())
    X_asset = csr_matrix(pd.get_dummies(asset_encoded).values)

    # ── Combine all features ──
    X = hstack([X_title, X_summary, X_symbol, X_asset])
    y = df['label'].values

    # ── Train/Test split (stratified to ensure both classes in test) ──
    train_idx, test_idx = train_test_split(
        range(len(df)), test_size=0.2, random_state=42, stratify=y
    )

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    print(f"   Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

    # ── XGBoost Model (binary: Bearish=0 vs Bullish=1) ──
    model = XGBClassifier(
        n_estimators=600,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=3,
        gamma=0.1,
        random_state=42,
        eval_metric='logloss',
        objective='binary:logistic',
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    # ── Evaluate ──
    preds = model.predict(X_test)
    print("\n📊 Classification Report:")
    print(classification_report(
        y_test, preds,
        target_names=["Bearish", "Bullish"]
    ))

    # ── Save artifacts ──
    os.makedirs(MODEL_DIR, exist_ok=True)

    with open(os.path.join(MODEL_DIR, "model.pkl"), "wb") as f:
        pickle.dump(model, f)
    with open(os.path.join(MODEL_DIR, "tfidf_title.pkl"), "wb") as f:
        pickle.dump(tfidf_title, f)
    with open(os.path.join(MODEL_DIR, "tfidf_summary.pkl"), "wb") as f:
        pickle.dump(tfidf_summary, f)
    with open(os.path.join(MODEL_DIR, "le_symbol.pkl"), "wb") as f:
        pickle.dump(le_symbol, f)
    with open(os.path.join(MODEL_DIR, "le_asset.pkl"), "wb") as f:
        pickle.dump(le_asset, f)

    print(f"\n✅ Model saved to {MODEL_DIR}")

    return model, tfidf_title, tfidf_summary, le_symbol, le_asset


class NewsSentimentPredictor:
    """Loads trained model and predicts sentiment for new news articles."""

    def __init__(self, model_dir=MODEL_DIR):
        self.model_dir = model_dir
        self.model = None
        self.tfidf_title = None
        self.tfidf_summary = None
        self.le_symbol = None
        self.le_asset = None
        self._loaded = False

    def load(self):
        """Load model artifacts from disk."""
        if self._loaded:
            return True

        try:
            with open(os.path.join(self.model_dir, "model.pkl"), "rb") as f:
                self.model = pickle.load(f)
            with open(os.path.join(self.model_dir, "tfidf_title.pkl"), "rb") as f:
                self.tfidf_title = pickle.load(f)
            with open(os.path.join(self.model_dir, "tfidf_summary.pkl"), "rb") as f:
                self.tfidf_summary = pickle.load(f)
            with open(os.path.join(self.model_dir, "le_symbol.pkl"), "rb") as f:
                self.le_symbol = pickle.load(f)
            with open(os.path.join(self.model_dir, "le_asset.pkl"), "rb") as f:
                self.le_asset = pickle.load(f)
            self._loaded = True
            print("✅ News sentiment model loaded")
            return True
        except FileNotFoundError:
            print("⚠️ News model not found. Run train_news_model() first.")
            return False

    def predict(self, title: str, summary: str = "", symbol: str = "", asset_type: str = ""):
        """
        Predict sentiment for a single news article.
        
        Returns:
            dict with keys: sentiment, confidence, impact_score, probabilities
        """
        if not self._loaded:
            if not self.load():
                return {
                    "sentiment": "Neutral",
                    "confidence": 0.0,
                    "impact_score": 0.5,
                    "probabilities": {"Bearish": 0.33, "Neutral": 0.34, "Bullish": 0.33},
                }

        title_clean = clean_text(title)
        summary_clean = clean_text(summary)

        X_title = self.tfidf_title.transform([title_clean])
        X_summary = self.tfidf_summary.transform([summary_clean])

        # Handle unknown symbols
        symbol_upper = symbol.strip().upper()
        try:
            sym_enc = self.le_symbol.transform([symbol_upper])
        except ValueError:
            sym_enc = np.array([0])
        n_symbols = len(self.le_symbol.classes_)
        X_symbol = csr_matrix(np.zeros((1, n_symbols)))
        if sym_enc[0] < n_symbols:
            X_symbol[0, sym_enc[0]] = 1

        # Handle unknown asset types
        asset_lower = asset_type.strip().lower() if asset_type else "unknown"
        try:
            asset_enc = self.le_asset.transform([asset_lower])
        except ValueError:
            asset_enc = np.array([0])
        n_assets = len(self.le_asset.classes_)
        X_asset = csr_matrix(np.zeros((1, n_assets)))
        if asset_enc[0] < n_assets:
            X_asset[0, asset_enc[0]] = 1

        X = hstack([X_title, X_summary, X_symbol, X_asset])

        probs = self.model.predict_proba(X)[0]
        # Binary model: probs[0]=Bearish, probs[1]=Bullish
        bearish_prob = float(probs[0])
        bullish_prob = float(probs[1])

        pred_label = int(np.argmax(probs))
        sentiment = SENTIMENT_LABELS[pred_label]
        confidence = float(probs[pred_label])
        impact = IMPACT_SCORES[sentiment]

        # Derive neutral zone: if both probs are close (within 0.15 of 0.5)
        if abs(bullish_prob - 0.5) < 0.15:
            sentiment = "Neutral"
            confidence = 1.0 - abs(bullish_prob - 0.5) * 2
            impact = IMPACT_SCORES["Neutral"]

        return {
            "sentiment": sentiment,
            "confidence": round(confidence, 3),
            "impact_score": round(impact, 3),
            "probabilities": {
                "Bearish": round(bearish_prob, 3),
                "Neutral": round(1.0 - abs(bullish_prob - bearish_prob), 3),
                "Bullish": round(bullish_prob, 3),
            },
        }

    def predict_batch(self, articles: list):
        """Predict sentiment for multiple articles."""
        return [
            self.predict(
                title=a.get("title", ""),
                summary=a.get("summary", ""),
                symbol=a.get("symbol", ""),
                asset_type=a.get("asset_type", ""),
            )
            for a in articles
        ]


# ─────────────────────────────────────────────
# CLI: Train the model
# ─────────────────────────────────────────────
if __name__ == "__main__":
    model, tfidf_title, tfidf_summary, le_symbol, le_asset = train_news_model()

    # Test prediction
    predictor = NewsSentimentPredictor()
    predictor.load()

    test_result = predictor.predict(
        title="Gold surges to new all-time high amid inflation fears",
        summary="Gold prices rose sharply as investors seek safe haven assets",
        symbol="XAUUSD",
        asset_type="metal",
    )
    print(f"\n🧪 Test prediction: {test_result}")
