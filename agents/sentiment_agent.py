"""
Sentiment Agent
───────────────
Aggregates sentiment from two FREE sources:
  1. Yahoo Finance & Google News RSS feeds (via feedparser)
  2. Reddit PRAW — optional, only used when credentials are set in settings.yaml

Uses NLTK VADER for scoring.

Output schema:
{
  "ticker":        str,
  "agent":         "sentiment",
  "score":         float,      # -1.0 → +1.0  (compound VADER score)
  "buzz_level":    str,        # Low | Medium | High
  "article_count": int,
  "top_headlines": list[str],
  "risk_flags":    list[str],  # e.g. "Overhyped (WSB)"
  "breakdown":     dict,       # news_score, reddit_score
  "status":        str,
  "error":         str | None
}
"""

import logging
import re
import time
from typing import Dict, List, Optional, Tuple

from utils.helpers import load_config, cache_get, cache_set

logger = logging.getLogger(__name__)

# ─── VADER setup (download silently if missing) ───────────────────────────────
_VADER_READY = False

def _init_vader():
    global _VADER_READY
    if _VADER_READY:
        return True
    try:
        import nltk
        try:
            from nltk.sentiment import SentimentIntensityAnalyzer
            sia = SentimentIntensityAnalyzer()
            sia.polarity_scores("test")
        except LookupError:
            nltk.download("vader_lexicon", quiet=True)
        _VADER_READY = True
        return True
    except Exception as e:
        logger.warning(f"VADER unavailable: {e}")
        return False


def _vader_score(text: str) -> float:
    """Return compound VADER score (-1 → +1)."""
    try:
        from nltk.sentiment import SentimentIntensityAnalyzer
        sia = SentimentIntensityAnalyzer()
        return sia.polarity_scores(str(text))["compound"]
    except Exception:
        return 0.0


# ─── News RSS helpers ─────────────────────────────────────────────────────────

def _yahoo_rss_url(ticker: str) -> str:
    return (
        f"https://feeds.finance.yahoo.com/rss/2.0/headline"
        f"?s={ticker}&region=US&lang=en-US"
    )


def _google_news_url(ticker: str) -> str:
    return (
        f"https://news.google.com/rss/search"
        f"?q={ticker}+stock+market&hl=en-US&gl=US&ceid=US:en"
    )


def _fetch_rss(url: str, max_items: int = 10) -> List[str]:
    """Return list of headline + summary strings from an RSS feed."""
    try:
        import feedparser
        feed = feedparser.parse(url)
        texts = []
        for entry in feed.entries[:max_items]:
            parts = []
            if hasattr(entry, "title"):
                parts.append(entry.title)
            if hasattr(entry, "summary"):
                # strip HTML tags
                clean = re.sub(r"<[^>]+>", "", entry.summary)
                parts.append(clean[:200])
            texts.append(" ".join(parts))
        return texts
    except Exception as e:
        logger.debug(f"RSS fetch failed for {url}: {e}")
        return []


# ─── Reddit helper ────────────────────────────────────────────────────────────

def _fetch_reddit(ticker: str, cfg: dict) -> Tuple[List[str], bool]:
    """
    Returns (list_of_texts, wsb_heavy_flag).
    Skips gracefully if PRAW credentials are missing.
    """
    if not cfg.get("enabled", False):
        return [], False
    try:
        import praw
        reddit = praw.Reddit(
            client_id     = cfg.get("client_id", ""),
            client_secret = cfg.get("client_secret", ""),
            user_agent    = cfg.get("user_agent", "StockBot/1.0"),
        )
        subreddits = cfg.get("subreddits", ["stocks", "investing"])
        max_posts  = cfg.get("max_posts", 100)
        texts     = []
        wsb_count = 0
        total     = 0

        for sub in subreddits:
            try:
                results = reddit.subreddit(sub).search(
                    f'"{ticker}"', limit=max_posts // len(subreddits), time_filter="week"
                )
                for post in results:
                    texts.append(f"{post.title} {post.selftext[:300]}")
                    if sub == "wallstreetbets":
                        wsb_count += 1
                    total += 1
            except Exception as e:
                logger.debug(f"Reddit r/{sub} fetch failed: {e}")

        wsb_heavy = wsb_count > max(total * 0.5, 3) if cfg.get("wsb_hype_penalty", True) else False
        return texts, wsb_heavy

    except Exception as e:
        logger.debug(f"Reddit fetch failed for {ticker}: {e}")
        return [], False


# ─── Main Agent ──────────────────────────────────────────────────────────────

class SentimentAgent:
    def __init__(self, config: Optional[dict] = None):
        self.config = config or load_config()
        _init_vader()

    # ─────────────────────────── public API ──────────────────────────

    def analyze(self, ticker: str) -> Dict:
        cache_key = f"sentiment_{ticker}"
        ttl = self.config.get("data", {}).get("cache_ttl_minutes", 60)

        if self.config.get("data", {}).get("cache_enabled", True):
            cached = cache_get(cache_key, ttl)
            if cached:
                return cached

        result = self._run_analysis(ticker)

        if self.config.get("data", {}).get("cache_enabled", True):
            cache_set(cache_key, result, ttl)

        return result

    # ──────────────────────── internal logic ─────────────────────────

    def _run_analysis(self, ticker: str) -> Dict:
        news_cfg   = self.config.get("news", {})
        reddit_cfg = self.config.get("reddit", {})
        max_art    = news_cfg.get("max_articles_per_ticker", 10)
        sources    = news_cfg.get("sources", {})

        all_headlines: List[str] = []
        news_texts:    List[str] = []
        risk_flags:    List[str] = []

        # ── News ──────────────────────────────────────────────────────
        try:
            if news_cfg.get("enabled", True):
                if sources.get("yahoo_finance", True):
                    texts = _fetch_rss(_yahoo_rss_url(ticker), max_art)
                    news_texts.extend(texts)
                    all_headlines.extend([t.split(" ")[0:12] and " ".join(t.split()[:12]) for t in texts])

                if sources.get("google_news", True):
                    texts = _fetch_rss(_google_news_url(ticker), max_art)
                    news_texts.extend(texts)
        except Exception as e:
            logger.warning(f"[sentiment] News fetch error for {ticker}: {e}")

        news_score = 0.0
        if news_texts:
            scores = [_vader_score(t) for t in news_texts]
            news_score = sum(scores) / len(scores)

        # ── Reddit ────────────────────────────────────────────────────
        reddit_texts, wsb_heavy = _fetch_reddit(ticker, reddit_cfg)
        reddit_score = 0.0
        if reddit_texts:
            scores = [_vader_score(t) for t in reddit_texts]
            reddit_score = sum(scores) / len(scores)
            if wsb_heavy:
                risk_flags.append("Overhyped (WSB)")

        # ── Combined ──────────────────────────────────────────────────
        total_count = len(news_texts) + len(reddit_texts)

        if news_texts and reddit_texts:
            combined_score = 0.6 * news_score + 0.4 * reddit_score
        elif news_texts:
            combined_score = news_score
        elif reddit_texts:
            combined_score = reddit_score
        else:
            combined_score = 0.0

        # Clamp
        combined_score = max(-1.0, min(1.0, combined_score))

        buzz = self._buzz_level(total_count)
        if buzz == "High" and combined_score > 0.5:
            risk_flags.append("High hype detected – use caution")

        # Headline deduplication & top-5
        seen, headlines = set(), []
        for h in all_headlines:
            key = h[:40].lower()
            if key not in seen:
                seen.add(key)
                headlines.append(h)
        top_headlines = headlines[:5]

        return {
            "ticker":        ticker,
            "agent":         "sentiment",
            "score":         round(combined_score, 4),
            "buzz_level":    buzz,
            "article_count": total_count,
            "top_headlines": top_headlines,
            "risk_flags":    risk_flags,
            "breakdown": {
                "news_score":   round(news_score, 4),
                "news_count":   len(news_texts),
                "reddit_score": round(reddit_score, 4),
                "reddit_count": len(reddit_texts),
            },
            "status": "success",
            "error":  None,
        }

    # ──────────────────────── helpers ────────────────────────────────

    @staticmethod
    def _buzz_level(article_count: int) -> str:
        if article_count >= 15:
            return "High"
        if article_count >= 6:
            return "Medium"
        return "Low"
