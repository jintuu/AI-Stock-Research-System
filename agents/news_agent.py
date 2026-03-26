"""
News Intelligence Agent
────────────────────────
Dedicated news layer that replaces the news portion of the old sentiment agent.

Sources (all free):
  1. yfinance .news property (Yahoo Finance article metadata)
  2. Yahoo Finance RSS feed
  3. Google News RSS feed

NLP:
  • VADER compound score for overall sentiment
  • Keyword-based catalyst detection with signed weights

Output schema:
{
  "ticker":          str,
  "agent":           "news",
  "sentiment":       float,       # VADER compound −1 → +1
  "buzz":            str,         # "Low" | "Medium" | "High"
  "catalyst":        str,         # human-readable label e.g. "Earnings Beat (Positive)"
  "catalyst_score":  float,       # 0–100  (50 = neutral, >50 = positive, <50 = negative)
  "impact_score":    float,       # 0–100  combined sentiment + catalyst
  "article_count":   int,
  "catalysts_found": list[str],   # e.g. ["Upgrade", "Product Launch"]
  "top_headlines":   list[str],   # up to 5 deduplicated headlines
  "status":          str,
  "error":           str | None
}
"""

import logging
import re
import time
from typing import Dict, List, Optional, Tuple

from utils.helpers import load_config, safe_float, cache_get, cache_set

logger = logging.getLogger(__name__)

# ─── Catalyst keyword table ───────────────────────────────────────────────────
# weight > 0 → bullish catalyst; weight < 0 → bearish catalyst

_CATALYSTS = {
    "earnings_beat":    (["earnings beat", "eps beat", "revenue beat",
                          "topped estimates", "exceeded expectations",
                          "beat estimates", "blew past"],                     +15),
    "earnings_miss":    (["earnings miss", "eps miss", "revenue miss",
                          "missed estimates", "below expectations",
                          "missed forecasts"],                                 -15),
    "upgrade":          (["upgrade", "buy rating", "outperform", "overweight",
                          "price target raised", "raised target",
                          "raised price target", "initiates buy"],            +12),
    "downgrade":        (["downgrade", "sell rating", "underperform",
                          "underweight", "price target cut", "lowered target",
                          "lowered price target", "initiates sell"],          -12),
    "guidance_raised":  (["raised guidance", "raised outlook",
                          "increased forecast", "raised full-year",
                          "positive guidance", "raised annual guidance"],     +10),
    "guidance_lowered": (["lowered guidance", "cut guidance",
                          "reduced forecast", "below guidance",
                          "cautious outlook"],                                 -10),
    "merger_acq":       (["acquisition", "merger", "buyout", "takeover",
                          "acquired by", "to acquire", "deal valued"],        +15),
    "product_launch":   (["new product", "product launch", "launches",
                          "unveiled", "partnership", "new contract",
                          "ai demand", "new model", "new service"],            +8),
    "regulation_neg":   (["investigation", "fine", "lawsuit", "sec probe",
                          "antitrust", "ban", "penalty", "probe"],            -10),
    "insider_buy":      (["insider buy", "director buy", "ceo buys",
                          "executive buy", "insider purchase"],                +7),
    "insider_sell":     (["insider sell", "ceo sells", "director sells",
                          "executive sold", "insider selling"],                -5),
    "layoffs":          (["layoff", "job cuts", "downsizing",
                          "workforce reduction", "headcount reduction"],       -5),
    "buyback":          (["buyback", "share repurchase", "stock repurchase",
                          "authorizes repurchase"],                            +6),
    "dividend_raise":   (["dividend increase", "special dividend",
                          "dividend raise", "increased dividend"],             +5),
    "short_squeeze":    (["short squeeze", "heavily shorted",
                          "high short interest"],                              +4),
    "data_breach":      (["data breach", "hack", "cyberattack",
                          "security breach"],                                  -8),
}

# ─── VADER helper ─────────────────────────────────────────────────────────────

_VADER_OK = False

def _ensure_vader():
    global _VADER_OK
    if _VADER_OK:
        return
    try:
        import nltk
        try:
            from nltk.sentiment import SentimentIntensityAnalyzer
            SentimentIntensityAnalyzer().polarity_scores("test")
        except LookupError:
            nltk.download("vader_lexicon", quiet=True)
        _VADER_OK = True
    except Exception as e:
        logger.debug(f"VADER init failed: {e}")


def _vader(text: str) -> float:
    try:
        from nltk.sentiment import SentimentIntensityAnalyzer
        return SentimentIntensityAnalyzer().polarity_scores(str(text))["compound"]
    except Exception:
        return 0.0


# ─── RSS helper ───────────────────────────────────────────────────────────────

def _rss(url: str, max_items: int = 10) -> List[dict]:
    try:
        import feedparser
        feed = feedparser.parse(url)
        out  = []
        for entry in feed.entries[:max_items]:
            title   = getattr(entry, "title",   "") or ""
            summary = re.sub(r"<[^>]+>", "", getattr(entry, "summary", "") or "")[:300]
            if title:
                out.append({"headline": title, "text": f"{title} {summary}"})
        return out
    except Exception:
        return []


# ─── Main Agent ───────────────────────────────────────────────────────────────

class NewsAgent:
    def __init__(self, config: Optional[dict] = None):
        self.config   = config or load_config()
        self._ttl     = self.config.get("data", {}).get("cache_ttl_minutes", 60)
        self._cache   = self.config.get("data", {}).get("cache_enabled", True)
        self._max_art = self.config.get("news", {}).get("max_articles_per_ticker", 10)
        _ensure_vader()

    # ─────────────────────────── public API ──────────────────────────

    def analyze(self, ticker: str) -> Dict:
        key = f"news2_{ticker}"
        if self._cache:
            hit = cache_get(key, self._ttl)
            if hit:
                return hit

        result = self._run(ticker)

        if self._cache:
            cache_set(key, result, self._ttl)
        return result

    # ──────────────────────── internal ───────────────────────────────

    def _run(self, ticker: str) -> Dict:
        articles: List[dict] = []

        # 1. yfinance news ─────────────────────────────────────────────
        try:
            import yfinance as yf
            yf_items = yf.Ticker(ticker).news or []
            for item in yf_items[:self._max_art]:
                title   = item.get("title",   "") or ""
                summary = item.get("summary", "") or ""
                if title:
                    articles.append({
                        "headline": title,
                        "text":     f"{title} {summary[:300]}",
                    })
        except Exception as e:
            logger.debug(f"yfinance news {ticker}: {e}")

        # 2. Yahoo Finance RSS ─────────────────────────────────────────
        yf_rss = _rss(
            f"https://feeds.finance.yahoo.com/rss/2.0/headline"
            f"?s={ticker}&region=US&lang=en-US",
            self._max_art,
        )
        articles.extend(yf_rss)

        # 3. Google News RSS ───────────────────────────────────────────
        gn_rss = _rss(
            f"https://news.google.com/rss/search"
            f"?q={ticker}+stock&hl=en-US&gl=US&ceid=US:en",
            self._max_art // 2,
        )
        articles.extend(gn_rss)

        if not articles:
            return self._empty(ticker)

        # Deduplicate by headline prefix
        seen, unique = set(), []
        for a in articles:
            key2 = a["headline"][:40].lower()
            if key2 not in seen:
                seen.add(key2)
                unique.append(a)
        articles = unique

        # Sentiment
        scores    = [_vader(a["text"]) for a in articles]
        sentiment = sum(scores) / len(scores)

        # Catalysts
        cat_score, cat_list, cat_label = self._catalysts(articles)

        # Buzz
        buzz = "High" if len(articles) >= 15 else ("Medium" if len(articles) >= 6 else "Low")

        # Impact: blend sentiment (±1 → ±30 pts from 50 midpoint) + catalyst
        impact = min(100.0, max(0.0, 50.0 + sentiment * 30.0 + (cat_score - 50.0) * 0.4))

        return {
            "ticker":          ticker,
            "agent":           "news",
            "sentiment":       round(sentiment,  4),
            "buzz":            buzz,
            "catalyst":        cat_label,
            "catalyst_score":  round(cat_score,  1),
            "impact_score":    round(impact,     1),
            "article_count":   len(articles),
            "catalysts_found": cat_list,
            "top_headlines":   [a["headline"][:120] for a in articles[:5]],
            "status":          "success",
            "error":           None,
        }

    # ──────────────────────── catalyst detection ─────────────────────

    def _catalysts(self, articles: List[dict]) -> Tuple[float, List[str], str]:
        combined = " ".join(a["text"].lower() for a in articles)
        found    = {}
        raw      = 0

        for name, (keywords, weight) in _CATALYSTS.items():
            for kw in keywords:
                if kw in combined:
                    if name not in found:
                        found[name] = weight
                        raw        += weight
                    break

        # Normalise raw (roughly −100 … +100) → 0–100
        norm  = min(100.0, max(0.0, 50.0 + raw))
        names = [k.replace("_", " ").title() for k in found]

        if not names:
            label = "No major catalyst"
        elif raw > 15:
            label = f"{', '.join(names[:2])} (Positive)"
        elif raw < -8:
            label = f"{', '.join(names[:2])} (Negative)"
        else:
            label = ", ".join(names[:2]) or "Mixed signals"

        return norm, names, label

    # ──────────────────────── empty result ───────────────────────────

    @staticmethod
    def _empty(ticker: str) -> dict:
        return {
            "ticker":          ticker,
            "agent":           "news",
            "sentiment":       0.0,
            "buzz":            "Low",
            "catalyst":        "No data",
            "catalyst_score":  50.0,
            "impact_score":    50.0,
            "article_count":   0,
            "catalysts_found": [],
            "top_headlines":   [],
            "status":          "no_data",
            "error":           None,
        }
