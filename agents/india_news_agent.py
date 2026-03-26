"""
India News Agent
────────────────
Extracts and scores news for Indian (NSE) stocks using free sources:

  Primary (general market news):
    • Economic Times Markets RSS
    • Moneycontrol Top News RSS
    • Business Standard Markets RSS
    • LiveMint Markets RSS
    • NDTV Profit RSS

  Per-ticker:
    • yfinance .news (works for .NS tickers, limited)
    • Google News RSS with NSE/India context

  NLP:
    • VADER sentiment
    • Indian market-specific catalyst keywords
      (Results, Q-results, QIP, FPO, SEBI, RBI, FII/DII flows, etc.)
"""

import logging
import re
import time
from typing import Dict, List, Optional, Tuple

import feedparser

from utils.helpers import safe_float, load_config, cache_get, cache_set

logger = logging.getLogger(__name__)

# ── India-specific catalyst keywords ─────────────────────────────────────────

INDIA_CATALYSTS = {
    # Earnings
    "quarterly_beat":    {"keywords": ["quarterly results beat", "profit rises", "net profit up",
                                        "PAT up", "revenue beat", "earnings beat",
                                        "net profit jumps", "Q1 results beat", "Q2 results beat",
                                        "Q3 results beat", "Q4 results beat"], "weight": 15},
    "quarterly_miss":    {"keywords": ["quarterly results miss", "profit falls", "net profit down",
                                        "PAT down", "revenue miss", "net profit drops",
                                        "disappoints on earnings", "below estimates"], "weight": -15},
    # Analyst actions
    "upgrade":           {"keywords": ["upgrade", "buy rating", "target raised", "overweight",
                                        "outperform", "strong buy", "price target increased"],   "weight": 12},
    "downgrade":         {"keywords": ["downgrade", "sell rating", "target cut", "underweight",
                                        "underperform", "reduce rating", "price target reduced"], "weight": -12},
    # Guidance
    "guidance_raised":   {"keywords": ["guidance raised", "outlook positive", "revenue guidance up",
                                        "raises FY guidance", "upgrades outlook"],               "weight": 10},
    "guidance_cut":      {"keywords": ["guidance cut", "outlook cautious", "reduces guidance",
                                        "lowers FY target", "downgrades outlook"],               "weight": -10},
    # Corporate actions
    "qip_fpo":           {"keywords": ["QIP", "FPO", "follow-on public offer", "rights issue",
                                        "block deal", "bulk deal", "institutional placement"],    "weight": 5},
    "buyback":           {"keywords": ["buyback", "share repurchase", "buy-back"],                "weight": 8},
    "dividend":          {"keywords": ["dividend declared", "interim dividend", "special dividend",
                                        "dividend per share"],                                    "weight": 6},
    "merger_acq":        {"keywords": ["acquisition", "merger", "takeover", "amalgamation",
                                        "stake acquisition", "strategic investment"],             "weight": 12},
    "demerger":          {"keywords": ["demerger", "spin-off", "hive off", "restructuring"],     "weight": 7},
    # Regulatory / macro
    "sebi_action":       {"keywords": ["SEBI action", "SEBI notice", "regulatory action",
                                        "NSE ban", "BSE suspension", "SEBI fine"],               "weight": -12},
    "rbi_impact":        {"keywords": ["RBI rate cut", "rate cut", "repo rate cut",
                                        "rate hike negative", "monetary policy positive"],        "weight": 8},
    "fii_buying":        {"keywords": ["FII buying", "FPI inflow", "foreign buying",
                                        "DII buying", "institutional buying"],                    "weight": 10},
    "fii_selling":       {"keywords": ["FII selling", "FPI outflow", "foreign selling",
                                        "FII exits"],                                             "weight": -10},
    # Promoter activity
    "promoter_buy":      {"keywords": ["promoter buying", "promoter increases stake",
                                        "insider buying"],                                        "weight": 8},
    "promoter_sell":     {"keywords": ["promoter selling", "promoter reduces stake",
                                        "promoter offloads"],                                     "weight": -8},
    # Order / contract wins
    "order_win":         {"keywords": ["order win", "order received", "contract awarded",
                                        "bagged order", "major order", "wins contract"],          "weight": 10},
    "order_cancel":      {"keywords": ["order cancellation", "contract cancelled",
                                        "order terminated"],                                      "weight": -8},
    # Debt
    "debt_reduction":    {"keywords": ["debt reduced", "debt free", "repays debt",
                                        "deleveraging"],                                          "weight": 7},
    "debt_concern":      {"keywords": ["debt rises", "debt burden", "overleveraged",
                                        "NPA", "non-performing asset", "NCLT"],                  "weight": -10},
}

# General market news RSS feeds (not ticker-specific but good for macro + sector context)
INDIA_GENERAL_RSS = [
    "https://economictimes.indiatimes.com/markets/stocks/rssfeeds/2146842.cms",
    "https://www.moneycontrol.com/rss/MCtopnews.xml",
    "https://www.business-standard.com/rss/markets-106.rss",
    "https://feeds.feedburner.com/ndtvprofit-latest",
    "https://www.livemint.com/rss/markets",
]


def _vader_score(text: str) -> float:
    try:
        from nltk.sentiment import SentimentIntensityAnalyzer
        return SentimentIntensityAnalyzer().polarity_scores(str(text))["compound"]
    except Exception:
        return 0.0


class IndiaNewsAgent:
    def __init__(self, config: Optional[dict] = None):
        self.config = config or load_config()
        self._init_vader()
        self._max_art = self.config.get("news", {}).get("max_articles_per_ticker", 10)

    def _init_vader(self):
        try:
            import nltk
            try:
                from nltk.sentiment import SentimentIntensityAnalyzer
                SentimentIntensityAnalyzer().polarity_scores("test")
            except LookupError:
                nltk.download("vader_lexicon", quiet=True)
        except Exception:
            pass

    def analyze(self, ticker: str) -> Dict:
        cache_key = f"india_news_{ticker}"
        ttl = self.config.get("data", {}).get("cache_ttl_minutes", 60)

        if self.config.get("data", {}).get("cache_enabled", True):
            cached = cache_get(cache_key, ttl)
            if cached:
                return cached

        result = self._run_analysis(ticker)

        if self.config.get("data", {}).get("cache_enabled", True):
            cache_set(cache_key, result, ttl)

        return result

    def _run_analysis(self, ticker: str) -> Dict:
        # Clean ticker: remove .NS/.BO for search queries
        clean = ticker.replace(".NS", "").replace(".BO", "")
        articles = []

        # 1. yfinance .news (works for Indian stocks too)
        try:
            import yfinance as yf
            stock    = yf.Ticker(ticker)
            yf_news  = stock.news or []
            for item in yf_news[:8]:
                title   = item.get("title", "")
                summary = item.get("summary", "") or ""
                if title:
                    articles.append({
                        "text":     f"{title} {summary}",
                        "headline": title,
                        "source":   item.get("publisher", "Yahoo Finance"),
                    })
        except Exception as e:
            logger.debug(f"yfinance news failed for {ticker}: {e}")

        # 2. Google News RSS (India context)
        google_urls = [
            f"https://news.google.com/rss/search?q={clean}+NSE+stock&hl=en-IN&gl=IN&ceid=IN:en",
            f"https://news.google.com/rss/search?q={clean}+share+price+India&hl=en-IN&gl=IN&ceid=IN:en",
        ]
        for url in google_urls:
            try:
                feed = feedparser.parse(url)
                for entry in feed.entries[:self._max_art // 2]:
                    title   = getattr(entry, "title", "")
                    summary = re.sub(r"<[^>]+>", "", getattr(entry, "summary", ""))[:300]
                    if title:
                        articles.append({
                            "text":     f"{title} {summary}",
                            "headline": title,
                            "source":   "Google News",
                        })
            except Exception:
                pass

        # 3. Economic Times — general market news, filter for ticker mentions
        try:
            feed = feedparser.parse(
                "https://economictimes.indiatimes.com/markets/stocks/rssfeeds/2146842.cms"
            )
            for entry in feed.entries[:20]:
                title = getattr(entry, "title", "")
                if clean.lower() in title.lower():
                    articles.append({
                        "text":     title,
                        "headline": title,
                        "source":   "Economic Times",
                    })
        except Exception:
            pass

        if not articles:
            return self._empty_result(ticker)

        # Deduplicate
        seen, unique = set(), []
        for a in articles:
            key = a["headline"][:50].lower()
            if key not in seen:
                seen.add(key)
                unique.append(a)
        articles = unique

        # Sentiment
        scores        = [_vader_score(a["text"]) for a in articles]
        sentiment_avg = sum(scores) / len(scores) if scores else 0.0

        # Catalyst detection
        cat_score, cats_found, cat_label = self._detect_catalysts(articles)

        buzz = "High" if len(articles) >= 15 else ("Medium" if len(articles) >= 5 else "Low")

        impact_score = min(100, max(0, 50 + sentiment_avg * 30 + (cat_score - 50) * 0.2))

        return {
            "ticker":          ticker,
            "agent":           "india_news",
            "sentiment":       round(sentiment_avg, 4),
            "buzz":            buzz,
            "catalyst":        cat_label,
            "catalyst_score":  round(cat_score, 1),
            "impact_score":    round(impact_score, 1),
            "article_count":   len(articles),
            "catalysts_found": cats_found,
            "top_headlines":   [a["headline"][:120] for a in articles[:5]],
            "status":          "success",
            "error":           None,
        }

    def _detect_catalysts(self, articles: List[dict]) -> Tuple[float, List[str], str]:
        combined = " ".join(a["text"].lower() for a in articles)
        found    = {}
        raw      = 0

        for name, cfg in INDIA_CATALYSTS.items():
            for kw in cfg["keywords"]:
                if kw.lower() in combined:
                    if name not in found:
                        found[name] = cfg["weight"]
                        raw        += cfg["weight"]
                    break

        norm       = min(100, max(0, 50 + raw))
        found_list = [k.replace("_", " ").title() for k in found]

        if not found_list:
            label = "No major catalyst"
        elif raw > 20:
            label = ", ".join(found_list[:2]) + " (Positive)"
        elif raw < -10:
            label = ", ".join(found_list[:2]) + " (Negative)"
        else:
            label = ", ".join(found_list[:2])

        return norm, found_list, label

    def _empty_result(self, ticker: str) -> dict:
        return {
            "ticker":          ticker,
            "agent":           "india_news",
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
