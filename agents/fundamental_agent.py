"""
Fundamental Agent
─────────────────
Pulls financial data via yfinance and scores each stock on:
  Revenue growth, EPS growth, ROE, Debt/Equity, FCF,
  Profit margins, P/E, PEG, Price-to-Book.

Output schema:
{
  "ticker":       str,
  "agent":        "fundamental",
  "score":        float,          # 0–100
  "summary":      str,            # Strong | Moderate | Neutral | Weak
  "valuation":    str,            # Undervalued | Fair Value | Overvalued
  "metrics":      dict,
  "status":       str,            # success | error | partial
  "error":        str | None
}
"""

import logging
import time
from typing import Dict, Optional

import yfinance as yf
import numpy as np

from utils.helpers import safe_float, score_to_label, load_config, cache_get, cache_set

logger = logging.getLogger(__name__)


class FundamentalAgent:
    # ──────────────────────── scoring weights ────────────────────────
    _WEIGHTS = {
        "revenue_growth":   20,
        "eps_growth":       15,
        "roe":              15,
        "debt_equity":      10,
        "fcf_positive":     10,
        "profit_margin":    10,
        "pe_ratio":         10,
        "peg_ratio":        10,
    }  # total = 100

    def __init__(self, config: Optional[dict] = None):
        self.config = config or load_config()

    # ─────────────────────────── public API ──────────────────────────

    def analyze(self, ticker: str) -> Dict:
        cache_key = f"fundamental_{ticker}"
        ttl = self.config.get("data", {}).get("cache_ttl_minutes", 60)

        if self.config.get("data", {}).get("cache_enabled", True):
            cached = cache_get(cache_key, ttl)
            if cached:
                logger.debug(f"[fundamental] Cache hit: {ticker}")
                return cached

        result = self._run_analysis(ticker)

        if self.config.get("data", {}).get("cache_enabled", True):
            cache_set(cache_key, result, ttl)

        return result

    # ──────────────────────── internal logic ─────────────────────────

    def _run_analysis(self, ticker: str) -> Dict:
        try:
            stock = yf.Ticker(ticker)
            info = stock.info or {}

            if not info or info.get("regularMarketPrice") is None and info.get("currentPrice") is None:
                # Try fast_info as fallback
                fi = stock.fast_info
                if not fi:
                    raise ValueError("No data returned by yfinance")

            score, breakdown = self._calculate_score(stock, info)
            valuation = self._assess_valuation(info)
            metrics = self._extract_metrics(stock, info)

            return {
                "ticker":    ticker,
                "agent":     "fundamental",
                "score":     round(score, 1),
                "summary":   score_to_label(score),
                "valuation": valuation,
                "metrics":   metrics,
                "breakdown": breakdown,
                "status":    "success",
                "error":     None,
            }

        except Exception as e:
            logger.warning(f"[fundamental] {ticker}: {e}")
            return {
                "ticker":    ticker,
                "agent":     "fundamental",
                "score":     0.0,
                "summary":   "Weak",
                "valuation": "Unknown",
                "metrics":   {},
                "breakdown": {},
                "status":    "error",
                "error":     str(e),
            }

    # ──────────────────────── scoring ────────────────────────────────

    def _calculate_score(self, stock, info: dict):
        score = 0.0
        breakdown = {}

        # 1. Revenue growth (20 pts) ─────────────────────────────────
        pts = 0
        try:
            fin = stock.financials
            if fin is not None and not fin.empty and len(fin.columns) >= 2:
                rev_key = None
                for k in ["Total Revenue", "Revenue"]:
                    if k in fin.index:
                        rev_key = k
                        break
                if rev_key:
                    r0 = safe_float(fin.loc[rev_key, fin.columns[0]])
                    r1 = safe_float(fin.loc[rev_key, fin.columns[1]])
                    if r0 and r1 and r1 > 0:
                        growth = (r0 - r1) / abs(r1)
                        if growth > 0.25:   pts = 20
                        elif growth > 0.15: pts = 16
                        elif growth > 0.08: pts = 11
                        elif growth > 0.02: pts = 6
                        elif growth > 0:    pts = 2
        except Exception:
            pass
        # Supplement with yfinance info field
        if pts == 0:
            rg = safe_float(info.get("revenueGrowth"))
            if rg is not None:
                if rg > 0.25:   pts = 18
                elif rg > 0.15: pts = 14
                elif rg > 0.08: pts = 10
                elif rg > 0.02: pts = 5
                elif rg > 0:    pts = 2
        score += pts
        breakdown["revenue_growth"] = pts

        # 2. EPS growth (15 pts) ──────────────────────────────────────
        pts = 0
        eg = safe_float(info.get("earningsGrowth"))
        if eg is not None:
            if eg > 0.25:   pts = 15
            elif eg > 0.15: pts = 12
            elif eg > 0.08: pts = 8
            elif eg > 0.02: pts = 4
            elif eg > 0:    pts = 2
        score += pts
        breakdown["eps_growth"] = pts

        # 3. ROE (15 pts) ─────────────────────────────────────────────
        pts = 0
        roe = safe_float(info.get("returnOnEquity"))
        if roe is not None:
            if roe > 0.25:   pts = 15
            elif roe > 0.18: pts = 12
            elif roe > 0.12: pts = 8
            elif roe > 0.06: pts = 4
            elif roe > 0:    pts = 1
        score += pts
        breakdown["roe"] = pts

        # 4. Debt/Equity (10 pts) ─────────────────────────────────────
        pts = 0
        de = safe_float(info.get("debtToEquity"))
        if de is not None:
            if de < 20:    pts = 10   # very low leverage
            elif de < 50:  pts = 8
            elif de < 100: pts = 5
            elif de < 200: pts = 2
        else:
            pts = 5  # unknown → neutral
        score += pts
        breakdown["debt_equity"] = pts

        # 5. Free Cash Flow (10 pts) ──────────────────────────────────
        pts = 0
        try:
            cf = stock.cashflow
            if cf is not None and not cf.empty:
                for k in ["Free Cash Flow", "FreeCashFlow"]:
                    if k in cf.index:
                        fcf = safe_float(cf.loc[k, cf.columns[0]])
                        if fcf is not None:
                            pts = 10 if fcf > 0 else 0
                        break
        except Exception:
            pass
        if pts == 0:
            fcf_val = safe_float(info.get("freeCashflow"))
            if fcf_val is not None:
                pts = 10 if fcf_val > 0 else 0
        score += pts
        breakdown["fcf_positive"] = pts

        # 6. Profit Margin (10 pts) ───────────────────────────────────
        pts = 0
        pm = safe_float(info.get("profitMargins"))
        if pm is not None:
            if pm > 0.25:   pts = 10
            elif pm > 0.15: pts = 8
            elif pm > 0.08: pts = 5
            elif pm > 0.03: pts = 2
            elif pm > 0:    pts = 1
        score += pts
        breakdown["profit_margin"] = pts

        # 7. P/E Ratio (10 pts) ───────────────────────────────────────
        pts = 0
        pe = safe_float(info.get("trailingPE"))
        if pe and pe > 0:
            if pe < 12:     pts = 10
            elif pe < 18:   pts = 8
            elif pe < 25:   pts = 6
            elif pe < 35:   pts = 3
            elif pe < 50:   pts = 1
        elif pe is None:
            pts = 3  # no PE (may be pre-earnings growth stock) → slight credit
        score += pts
        breakdown["pe_ratio"] = pts

        # 8. PEG Ratio (10 pts) ───────────────────────────────────────
        pts = 0
        peg = safe_float(info.get("pegRatio"))
        if peg is not None and peg > 0:
            if peg < 0.8:   pts = 10
            elif peg < 1.2: pts = 8
            elif peg < 1.8: pts = 5
            elif peg < 2.5: pts = 2
        score += pts
        breakdown["peg_ratio"] = pts

        return min(score, 100.0), breakdown

    # ──────────────────────── valuation ──────────────────────────────

    def _assess_valuation(self, info: dict) -> str:
        signals = []
        pe = safe_float(info.get("trailingPE"))
        pb = safe_float(info.get("priceToBook"))
        peg = safe_float(info.get("pegRatio"))
        ps = safe_float(info.get("priceToSalesTrailing12Months"))

        if pe and pe > 0:
            signals.append(-1 if pe < 15 else (1 if pe > 30 else 0))
        if pb and pb > 0:
            signals.append(-1 if pb < 1.5 else (1 if pb > 5 else 0))
        if peg and peg > 0:
            signals.append(-1 if peg < 1.0 else (1 if peg > 2.5 else 0))
        if ps and ps > 0:
            signals.append(-1 if ps < 2 else (1 if ps > 10 else 0))

        if not signals:
            return "Unknown"
        avg = sum(signals) / len(signals)
        if avg <= -0.4:
            return "Undervalued"
        if avg >= 0.4:
            return "Overvalued"
        return "Fair Value"

    # ──────────────────────── metrics dict ───────────────────────────

    def _extract_metrics(self, stock, info: dict) -> dict:
        price = safe_float(info.get("currentPrice") or info.get("regularMarketPrice"))
        try:
            fi = stock.fast_info
            price = price or safe_float(getattr(fi, "last_price", None))
        except Exception:
            pass

        return {
            "price":             price,
            "market_cap":        safe_float(info.get("marketCap")),
            "pe_ratio":          safe_float(info.get("trailingPE")),
            "forward_pe":        safe_float(info.get("forwardPE")),
            "peg_ratio":         safe_float(info.get("pegRatio")),
            "pb_ratio":          safe_float(info.get("priceToBook")),
            "ps_ratio":          safe_float(info.get("priceToSalesTrailing12Months")),
            "roe":               safe_float(info.get("returnOnEquity")),
            "roa":               safe_float(info.get("returnOnAssets")),
            "profit_margin":     safe_float(info.get("profitMargins")),
            "operating_margin":  safe_float(info.get("operatingMargins")),
            "debt_to_equity":    safe_float(info.get("debtToEquity")),
            "current_ratio":     safe_float(info.get("currentRatio")),
            "revenue_growth":    safe_float(info.get("revenueGrowth")),
            "earnings_growth":   safe_float(info.get("earningsGrowth")),
            "free_cashflow":     safe_float(info.get("freeCashflow")),
            "dividend_yield":    safe_float(info.get("dividendYield")),
            "sector":            info.get("sector", ""),
            "industry":          info.get("industry", ""),
            "company_name":      info.get("longName", ""),
            "52w_high":          safe_float(info.get("fiftyTwoWeekHigh")),
            "52w_low":           safe_float(info.get("fiftyTwoWeekLow")),
        }
