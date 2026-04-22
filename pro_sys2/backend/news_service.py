"""
News Service
============
Fetches live news from Yahoo Finance for all supported symbols,
runs them through the news sentiment model, and caches results.
"""

import time
import threading
import datetime
import yfinance as yf
from typing import List, Dict, Optional

# Symbol → Yahoo Finance ticker for news
NEWS_TICKER_MAP = {
    # Stocks
    "AAPL": "AAPL",
    "AMZN": "AMZN",
    "TSLA": "TSLA",
    "MSFT": "MSFT",
    # FX
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "EURGBP": "EURGBP=X",
    # Metals
    "XAUUSD": "GC=F",
    "XAGUSD": "SI=F",
    "XPTUSD": "PL=F",
    # Crypto
    "BTCUSD": "BTC-USD",
    "ETHUSD": "ETH-USD",
    "BNBUSD": "BNB-USD",
}

SYMBOL_TO_CATEGORY = {
    "AAPL": "stocks", "AMZN": "stocks", "TSLA": "stocks", "MSFT": "stocks",
    "EURUSD": "fx", "GBPUSD": "fx", "EURGBP": "fx",
    "XAUUSD": "metals", "XAGUSD": "metals", "XPTUSD": "metals",
    "BTCUSD": "crypto", "ETHUSD": "crypto", "BNBUSD": "crypto",
}

SYMBOL_DISPLAY_NAMES = {
    "AAPL": "Apple Inc.",
    "AMZN": "Amazon.com",
    "TSLA": "Tesla Inc.",
    "MSFT": "Microsoft Corp.",
    "EURUSD": "EUR/USD",
    "GBPUSD": "GBP/USD",
    "EURGBP": "EUR/GBP",
    "XAUUSD": "Gold",
    "XAGUSD": "Silver",
    "XPTUSD": "Platinum",
    "BTCUSD": "Bitcoin",
    "ETHUSD": "Ethereum",
    "BNBUSD": "BNB",
}


class NewsCache:
    """Thread-safe news cache."""

    def __init__(self):
        self._lock = threading.Lock()
        self._news: List[Dict] = []
        self._last_fetch: Optional[datetime.datetime] = None
        self._fetch_interval = 300  # 5 minutes

    def get_all(self) -> List[Dict]:
        with self._lock:
            return list(self._news)

    def get_by_symbol(self, symbol: str) -> List[Dict]:
        with self._lock:
            return [n for n in self._news if n.get("symbol", "").upper() == symbol.upper()]

    def get_by_category(self, category: str) -> List[Dict]:
        with self._lock:
            return [n for n in self._news if n.get("category", "").lower() == category.lower()]

    def update(self, news_list: List[Dict]):
        with self._lock:
            self._news = news_list
            self._last_fetch = datetime.datetime.utcnow()

    def needs_refresh(self) -> bool:
        with self._lock:
            if self._last_fetch is None:
                return True
            elapsed = (datetime.datetime.utcnow() - self._last_fetch).total_seconds()
            return elapsed >= self._fetch_interval


# Global cache
news_cache = NewsCache()


def fetch_yahoo_news_for_symbol(symbol: str, yf_ticker: str) -> List[Dict]:
    """Fetch news articles for a single symbol from Yahoo Finance."""
    articles = []
    try:
        ticker = yf.Ticker(yf_ticker)
        news = ticker.news

        if not news:
            return []

        for item in news:
            # Yahoo Finance v2 nests data under "content"
            content = item.get("content", item)

            # Title & summary
            title = content.get("title", "") or item.get("title", "")
            summary = content.get("summary", "") or content.get("description", "") or item.get("summary", "")
            if not summary:
                summary = title

            # Publisher
            provider = content.get("provider", {})
            if isinstance(provider, dict):
                publisher = provider.get("displayName", "")
            else:
                publisher = item.get("publisher", "")

            # URL
            canonical = content.get("canonicalUrl", {})
            if isinstance(canonical, dict):
                link = canonical.get("url", "")
            else:
                link = item.get("link", "") or item.get("url", "")
            # Fallback: clickThroughUrl
            if not link:
                click_through = content.get("clickThroughUrl", {})
                if isinstance(click_through, dict):
                    link = click_through.get("url", "")

            # Thumbnail
            thumbnail = ""
            thumbs = content.get("thumbnail", item.get("thumbnail", {}))
            if isinstance(thumbs, dict):
                resolutions = thumbs.get("resolutions", [])
                if resolutions and len(resolutions) > 0:
                    thumbnail = resolutions[-1].get("url", "")

            # Published date (v2 uses ISO string, v1 uses unix timestamp)
            pub_date_str = content.get("pubDate", "") or content.get("displayTime", "")
            pub_time = item.get("providerPublishTime", 0)
            if pub_date_str:
                pub_date = pub_date_str
            elif pub_time:
                pub_date = datetime.datetime.utcfromtimestamp(pub_time).isoformat() + "Z"
            else:
                pub_date = datetime.datetime.utcnow().isoformat() + "Z"

            # Skip articles with no title
            if not title or not title.strip():
                continue

            articles.append({
                "title": title,
                "summary": summary if summary else title,
                "source_url": link,
                "image_url": thumbnail,
                "publisher": publisher,
                "published_date": pub_date,
                "symbol": symbol,
                "category": SYMBOL_TO_CATEGORY.get(symbol, "unknown"),
                "symbol_name": SYMBOL_DISPLAY_NAMES.get(symbol, symbol),
                "language": "en",
            })

    except Exception as e:
        print(f"⚠️ Error fetching news for {symbol}: {e}")

    return articles


def fetch_all_news() -> List[Dict]:
    """Fetch news for all supported symbols."""
    all_news = []
    seen_titles = set()

    for symbol, yf_ticker in NEWS_TICKER_MAP.items():
        try:
            articles = fetch_yahoo_news_for_symbol(symbol, yf_ticker)
            for article in articles:
                # Deduplicate by title
                title_key = article["title"].strip().lower()
                if title_key not in seen_titles:
                    seen_titles.add(title_key)
                    all_news.append(article)
        except Exception as e:
            print(f"⚠️ Error processing {symbol}: {e}")

    # Sort by date descending
    all_news.sort(key=lambda x: x.get("published_date", ""), reverse=True)
    return all_news


def enrich_news_with_sentiment(news_list: List[Dict], predictor) -> List[Dict]:
    """Add sentiment predictions to news articles."""
    enriched = []
    for article in news_list:
        try:
            prediction = predictor.predict(
                title=article.get("title", ""),
                summary=article.get("summary", ""),
                symbol=article.get("symbol", ""),
                asset_type=article.get("category", ""),
            )
            article["sentiment"] = prediction["sentiment"]
            article["confidence"] = prediction["confidence"]
            article["impact_score"] = prediction["impact_score"]
            article["probabilities"] = prediction["probabilities"]
        except Exception as e:
            print(f"⚠️ Sentiment error: {e}")
            article["sentiment"] = "Neutral"
            article["confidence"] = 0.0
            article["impact_score"] = 0.5
            article["probabilities"] = {"Bearish": 0.33, "Neutral": 0.34, "Bullish": 0.33}

        enriched.append(article)
    return enriched


def start_news_background_fetcher(predictor):
    """Start a background thread that fetches news periodically."""

    def _fetch_loop():
        while True:
            try:
                if news_cache.needs_refresh():
                    print("📰 Fetching latest news from Yahoo Finance...")
                    raw_news = fetch_all_news()
                    print(f"   Fetched {len(raw_news)} articles")

                    if predictor and predictor._loaded:
                        enriched = enrich_news_with_sentiment(raw_news, predictor)
                        print(f"   Enriched {len(enriched)} articles with sentiment")
                    else:
                        enriched = raw_news
                        for a in enriched:
                            a.setdefault("sentiment", "Neutral")
                            a.setdefault("confidence", 0.0)
                            a.setdefault("impact_score", 0.5)

                    news_cache.update(enriched)
                    print(f"✅ News cache updated: {len(enriched)} articles")
            except Exception as e:
                print(f"❌ News fetch error: {e}")

            time.sleep(60)  # Check every minute

    thread = threading.Thread(target=_fetch_loop, daemon=True)
    thread.start()
    print("🔄 News background fetcher started")
    return thread
