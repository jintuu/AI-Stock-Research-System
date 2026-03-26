"""
Sector Agent
────────────
Tracks sector strength using the 11 SPDR Select Sector ETFs + SPY.
Provides:
  • Per-sector: 1D / 5D / 1M returns, relative strength vs SPY, trend, classification
  • SPY market regime (bullish / sideways / bearish via 200-SMA)
  • Sector → multiplier mapping for the scoring engine
  • Rotation insight (top gainers vs laggards)

Output schema for analyze_all():
{
  "Technology": {
    "sector":            str,
    "etf":               str,
    "strength":          "Strong" | "Neutral" | "Weak",
    "return_1d":         float,   # %
    "return_5d":         float,   # %
    "return_1m":         float,   # %
    "relative_strength": float,   # % vs SPY 1M
    "momentum":          float,   # -1 → +1
    "volume_change":     float,   # ratio vs 20-day MA
    "trend":             "up" | "down",
    "multiplier":        float,   # 1.15 | 1.0 | 0.85
  },
  ...
}

SPY regime schema:
{
  "regime":        "bullish" | "sideways" | "bearish",
  "spy_price":     float,
  "sma200":        float,
  "pct_vs_sma200": float,   # %
  "return_3m":     float,   # %
  "return_1m":     float,   # %
}
"""

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd
import yfinance as yf

from utils.helpers import safe_float, load_config, cache_get, cache_set

logger = logging.getLogger(__name__)

# ─── Sector → ETF ────────────────────────────────────────────────────────────

SECTOR_ETFS: Dict[str, str] = {
    "Technology":             "XLK",
    "Financial Services":     "XLF",
    "Energy":                 "XLE",
    "Healthcare":             "XLV",
    "Consumer Cyclical":      "XLY",
    "Industrials":            "XLI",
    "Utilities":              "XLU",
    "Basic Materials":        "XLB",
    "Real Estate":            "XLRE",
    "Communication Services": "XLC",
    "Consumer Defensive":     "XLP",
}

# ─── India: NSE Sectoral Indices ─────────────────────────────────────────────
# Available on yfinance as ^CNX* / ^NSE* symbols

INDIA_SECTOR_INDICES: Dict[str, str] = {
    "IT":              "^CNXIT",
    "Banking":         "^NSEBANK",
    "Auto":            "^CNXAUTO",
    "FMCG":            "^CNXFMCG",
    "Metal":           "^CNXMETAL",
    "Pharma":          "^CNXPHARMA",
    "Realty":          "^CNXREALTY",
    "Energy":          "^CNXENERGY",
    "Finance":         "^CNXFINANCE",
    "Infrastructure":  "^CNXINFRA",
    "Media":           "^CNXMEDIA",
    "PSU Bank":        "^CNXPSUBANK",
}

# yfinance sector names for Indian stocks → India sector index key
_INDIA_ALIAS: Dict[str, str] = {
    "Technology":             "IT",
    "Information Technology": "IT",
    "Financial Services":     "Banking",
    "Finance":                "Banking",
    "Consumer Defensive":     "FMCG",
    "Consumer Staples":       "FMCG",
    "Consumer Cyclical":      "Auto",
    "Consumer Discretionary": "Auto",
    "Basic Materials":        "Metal",
    "Materials":              "Metal",
    "Healthcare":             "Pharma",
    "Health Care":            "Pharma",
    "Energy":                 "Energy",
    "Real Estate":            "Realty",
    "Industrials":            "Infrastructure",
    "Communication Services": "Media",
    "Utilities":              "Energy",
}

# yfinance sector name aliases → our canonical names
_ALIAS: Dict[str, str] = {
    "Technology":                "Technology",
    "Information Technology":    "Technology",
    "Financial Services":        "Financial Services",
    "Finance":                   "Financial Services",
    "Energy":                    "Energy",
    "Healthcare":                "Healthcare",
    "Health Care":               "Healthcare",
    "Consumer Cyclical":         "Consumer Cyclical",
    "Consumer Discretionary":    "Consumer Cyclical",
    "Industrials":               "Industrials",
    "Utilities":                 "Utilities",
    "Basic Materials":           "Basic Materials",
    "Materials":                 "Basic Materials",
    "Real Estate":               "Real Estate",
    "Communication Services":    "Communication Services",
    "Consumer Defensive":        "Consumer Defensive",
    "Consumer Staples":          "Consumer Defensive",
}


class SectorAgent:
    def __init__(self, config: Optional[dict] = None):
        self.config = config or load_config()
        self._ttl   = self.config.get("data", {}).get("cache_ttl_minutes", 60)
        self._cache = self.config.get("data", {}).get("cache_enabled", True)

    # ─────────────────────────── public API ──────────────────────────

    def analyze_all(self) -> Dict[str, dict]:
        """Return strength analysis for every sector.  Cached."""
        if self._cache:
            hit = cache_get("sector_all", self._ttl)
            if hit:
                return hit

        spy_raw = self._fetch_raw("SPY", period="90d")
        spy_1m  = self._pct_return(spy_raw, 22) if spy_raw is not None else 0.0

        results: Dict[str, dict] = {}
        for sector, etf in SECTOR_ETFS.items():
            try:
                raw = self._fetch_raw(etf, period="90d")
                if raw is None:
                    continue

                r1d = self._pct_return(raw,  1)
                r5d = self._pct_return(raw,  5)
                r1m = self._pct_return(raw, 22)
                rel = r1m - spy_1m
                trend, momentum, vol_change = self._trend_stats(raw)
                strength = self._classify(rel, trend, r1m)

                results[sector] = {
                    "sector":             sector,
                    "etf":                etf,
                    "strength":           strength,
                    "return_1d":          round(r1d * 100, 2),
                    "return_5d":          round(r5d * 100, 2),
                    "return_1m":          round(r1m * 100, 2),
                    "relative_strength":  round(rel * 100, 2),
                    "momentum":           round(momentum, 3),
                    "volume_change":      round(vol_change, 2),
                    "trend":              trend,
                    "multiplier":         {"Strong": 1.15, "Neutral": 1.0, "Weak": 0.85}[strength],
                }
            except Exception as e:
                logger.debug(f"Sector {sector} ({etf}) failed: {e}")

        if self._cache:
            cache_set("sector_all", results, self._ttl)
        return results

    def get_spy_regime(self) -> dict:
        """Return SPY market-regime dict.  Cached."""
        if self._cache:
            hit = cache_get("spy_regime", self._ttl)
            if hit:
                return hit

        result = self._compute_spy_regime()

        if self._cache:
            cache_set("spy_regime", result, self._ttl)
        return result

    def normalise_sector(self, yf_sector: str, market: str = "us") -> str:
        """Map a raw yfinance sector string to our canonical sector name."""
        if market == "india":
            return _INDIA_ALIAS.get(yf_sector, yf_sector)
        return _ALIAS.get(yf_sector, yf_sector)

    # ─────────────────────────── India API ───────────────────────────────────

    def analyze_india_sectors(self) -> Dict[str, dict]:
        """Return strength analysis for every NSE sector.  Cached."""
        if self._cache:
            hit = cache_get("india_sector_all", self._ttl)
            if hit:
                return hit

        nifty_raw = self._fetch_raw("^NIFTY", period="90d")
        nifty_1m  = self._pct_return(nifty_raw, 22) if nifty_raw is not None else 0.0

        results: Dict[str, dict] = {}
        for sector, idx in INDIA_SECTOR_INDICES.items():
            try:
                raw = self._fetch_raw(idx, period="90d")
                if raw is None:
                    continue

                r1d      = self._pct_return(raw,  1)
                r5d      = self._pct_return(raw,  5)
                r1m      = self._pct_return(raw, 22)
                rel      = r1m - nifty_1m
                trend, momentum, vol_change = self._trend_stats(raw)
                strength = self._classify(rel, trend, r1m)

                results[sector] = {
                    "sector":             sector,
                    "etf":                idx,
                    "strength":           strength,
                    "return_1d":          round(r1d * 100, 2),
                    "return_5d":          round(r5d * 100, 2),
                    "return_1m":          round(r1m * 100, 2),
                    "relative_strength":  round(rel * 100, 2),
                    "momentum":           round(momentum, 3),
                    "volume_change":      round(vol_change, 2),
                    "trend":              trend,
                    "multiplier":         {"Strong": 1.15, "Neutral": 1.0, "Weak": 0.85}[strength],
                }
            except Exception as e:
                logger.debug(f"India sector {sector} ({idx}) failed: {e}")

        if self._cache:
            cache_set("india_sector_all", results, self._ttl)
        return results

    def get_nifty_regime(self) -> dict:
        """Return NIFTY 50 market-regime dict (mirrors get_spy_regime for India).  Cached."""
        if self._cache:
            hit = cache_get("nifty_regime", self._ttl)
            if hit:
                return hit

        result = self._compute_benchmark_regime("^NIFTY", label="nifty_price")

        if self._cache:
            cache_set("nifty_regime", result, self._ttl)
        return result

    def rotation_insight(self, sector_data: Dict[str, dict]) -> str:
        """Generate a one-line rotation insight string."""
        if not sector_data:
            return "Insufficient sector data."
        ranked = sorted(sector_data.values(),
                        key=lambda s: s.get("relative_strength", 0), reverse=True)
        top    = [s["sector"] for s in ranked if s.get("strength") == "Strong"][:2]
        weak   = [s["sector"] for s in ranked if s.get("strength") == "Weak"][:2]
        if top and weak:
            return f"Money rotating into {', '.join(top)}; away from {', '.join(weak)}."
        if top:
            return f"Strength concentrated in {', '.join(top)}."
        if weak:
            return f"Broad weakness in {', '.join(weak)}."
        return "Markets broadly neutral — no clear rotation signal."

    # ──────────────────────── internals ──────────────────────────────

    def _fetch_raw(self, ticker: str, period: str) -> Optional[pd.DataFrame]:
        try:
            df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
            if df is None or df.empty or len(df) < 10:
                return None
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            return df
        except Exception as e:
            logger.debug(f"_fetch_raw {ticker}: {e}")
            return None

    @staticmethod
    def _pct_return(df: pd.DataFrame, days: int) -> float:
        close = df["Close"]
        if len(close) <= days:
            days = len(close) - 1
        if days <= 0:
            return 0.0
        return float((close.iloc[-1] - close.iloc[-days - 1]) / close.iloc[-days - 1])

    @staticmethod
    def _trend_stats(df: pd.DataFrame):
        close = df["Close"]
        vol   = df["Volume"]

        sma20 = close.rolling(20).mean()
        trend = "up" if float(close.iloc[-1]) > float(sma20.iloc[-1]) else "down"

        # Momentum: compare last-5d gain vs prior-5d gain
        if len(close) >= 11:
            r_rec  = float(close.iloc[-1]  - close.iloc[-6])  / float(close.iloc[-6])
            r_prev = float(close.iloc[-6]  - close.iloc[-11]) / float(close.iloc[-11])
            denom  = max(abs(r_prev), 0.001)
            momentum = max(-1.0, min(1.0, (r_rec - r_prev) / denom))
        else:
            momentum = 0.0

        vol_ma     = vol.rolling(20).mean()
        vol_change = float(vol.iloc[-1] / vol_ma.iloc[-1]) if float(vol_ma.iloc[-1]) > 0 else 1.0

        return trend, momentum, vol_change

    @staticmethod
    def _classify(rel: float, trend: str, r1m: float) -> str:
        if rel > 0.02 and trend == "up":
            return "Strong"
        if rel < -0.02 and trend == "down":
            return "Weak"
        # Borderline cases
        if rel > 0.01 or (trend == "up" and r1m > 0):
            return "Neutral"
        return "Neutral"

    def _compute_spy_regime(self) -> dict:
        return self._compute_benchmark_regime("SPY", label="spy_price")

    def _compute_benchmark_regime(self, ticker: str, label: str = "spy_price") -> dict:
        empty = {"regime": "sideways", label: 0.0, "sma200": 0.0,
                 "pct_vs_sma200": 0.0, "return_3m": 0.0, "return_1m": 0.0}
        try:
            df = self._fetch_raw(ticker, period="400d")
            if df is None or len(df) < 60:
                return empty

            close   = df["Close"]
            sma200  = close.rolling(200).mean()
            price   = float(close.iloc[-1])
            s200    = float(sma200.iloc[-1]) if not pd.isna(sma200.iloc[-1]) else price
            pct     = (price - s200) / s200 if s200 else 0.0

            regime  = "bullish" if pct > 0.02 else ("bearish" if pct < -0.02 else "sideways")
            r3m     = self._pct_return(df, 63)
            r1m     = self._pct_return(df, 22)

            return {
                "regime":         regime,
                label:            round(price, 2),
                "sma200":         round(s200,  2),
                "pct_vs_sma200":  round(pct * 100, 2),
                "return_3m":      round(r3m * 100, 2),
                "return_1m":      round(r1m * 100, 2),
            }
        except Exception as e:
            logger.warning(f"Benchmark regime ({ticker}) failed: {e}")
            return empty
