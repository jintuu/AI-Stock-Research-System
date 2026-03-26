"""
Master Agent  —  Decision Engine  (v2)
───────────────────────────────────────
Orchestrates five sub-agents and runs the advanced scoring engine.

Pipeline per scan:
  1. SectorAgent.analyze_all()   — run ONCE, shared across all tickers
  2. SectorAgent.get_spy_regime() — run ONCE
  3. Per ticker (all four agents in parallel threads):
       FundamentalAgent · TechnicalAgent · SentimentAgent · NewsAgent
  4. ScoringEngine.compute() with sector / regime context
  5. Categorise → Short-Term / Medium-Term / Long-Term / Watchlist

Output schema per ticker (extends v1 with new fields):
  All v1 fields preserved for backward compatibility, plus:
    base_score, context_multiplier, opportunity_boost, risk_penalty,
    boost_breakdown, penalty_breakdown, confidence_num,
    sector, sector_strength, sector_multiplier,
    spy_regime, spy_price, rel_strength_vs_spy,
    news_catalyst, news_impact_score, news_sentiment,
    catalysts_found, news_headlines
"""

import logging
import time
import concurrent.futures
from pathlib import Path
from typing import Dict, List, Optional, Callable

from tracking.performance_tracker import PerformanceTracker
from agents.fundamental_agent  import FundamentalAgent
from agents.technical_agent    import TechnicalAgent
from agents.sentiment_agent    import SentimentAgent
from agents.news_agent         import NewsAgent
from agents.india_news_agent   import IndiaNewsAgent
from agents.sector_agent       import SectorAgent
from utils.scoring_engine      import ScoringEngine
from utils.helpers              import load_config, safe_float, save_results

logger = logging.getLogger(__name__)


class MasterAgent:
    def __init__(self, config: Optional[dict] = None):
        self.config   = config or load_config()
        self.weights  = self.config.get("weights", {
            "fundamentals": 0.40, "technicals": 0.35, "sentiment": 0.25,
        })
        self.thresholds = self.config.get("thresholds", {})
        self.delay      = self.config.get("data", {}).get("request_delay_seconds", 0.3)

        # Market: "us" or "india"
        self.market   = self.config.get("market", "us")
        self.currency = "₹" if self.market == "india" else "$"

        # Sub-agents
        self.fundamental = FundamentalAgent(self.config)
        self.technical   = TechnicalAgent(self.config)
        self.sentiment   = SentimentAgent(self.config)
        self.news        = IndiaNewsAgent(self.config) if self.market == "india" else NewsAgent(self.config)
        self.sector_ag   = SectorAgent(self.config)
        self.scorer      = ScoringEngine(self.weights)

        # Shared context (populated once per scan)
        self._sector_data: Dict[str, dict] = {}
        self._spy_data:    dict             = {}  # also used for NIFTY regime

        # Performance tracker
        results_dir = self.config.get("output", {}).get("results_dir", "data/results")
        tracking_dir = str(Path(results_dir).parent / "tracking")
        self.tracker = PerformanceTracker(storage_dir=tracking_dir)

    # ─────────────────────────── public API ──────────────────────────

    def scan(
        self,
        tickers:     List[str],
        progress_cb: Optional[Callable[[int, int, str], None]] = None,
    ) -> List[Dict]:
        total = len(tickers)

        # ── 1. Market context (once per scan) ────────────────────────
        benchmark = "NIFTY" if self.market == "india" else "SPY"
        logger.info(f"Fetching sector & {benchmark} context …")
        if progress_cb:
            progress_cb(0, total, f"{benchmark} context …")

        if self.market == "india":
            self._sector_data = self.sector_ag.analyze_india_sectors()
            self._spy_data    = self.sector_ag.get_nifty_regime()
        else:
            self._sector_data = self.sector_ag.analyze_all()
            self._spy_data    = self.sector_ag.get_spy_regime()

        logger.info(f"{benchmark} regime: {self._spy_data.get('regime')} | "
                    f"{len(self._sector_data)} sectors loaded")

        # ── 2. Per-ticker analysis ────────────────────────────────────
        results = []
        for i, ticker in enumerate(tickers):
            logger.info(f"[{i+1}/{total}] {ticker}")
            try:
                result = self.analyse_ticker(ticker)
                if result.get("status") != "error":
                    results.append(result)
            except Exception as e:
                logger.error(f"Master scan failed for {ticker}: {e}")

            if progress_cb:
                progress_cb(i + 1, total, ticker)

            time.sleep(self.delay)

        results.sort(key=lambda r: r.get("combined_score", 0), reverse=True)

        # ── 3. Persist results ────────────────────────────────────────
        if self.config.get("output", {}).get("save_results", True):
            try:
                path = save_results(results, self.config)
                logger.info(f"Results saved → {path}")
            except Exception as e:
                logger.warning(f"Could not save results: {e}")

        # ── 4. Store predictions for performance tracking ─────────────
        try:
            n_stored = self.tracker.store_predictions(results)
            logger.info(f"Performance tracker: stored {n_stored} predictions")
        except Exception as e:
            logger.warning(f"Performance tracker storage failed: {e}")

        return results

    def analyse_ticker(self, ticker: str) -> Dict:
        # ── Run 4 sub-agents concurrently ────────────────────────────
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
            f_fut = ex.submit(self.fundamental.analyze, ticker)
            t_fut = ex.submit(self.technical.analyze,   ticker)
            s_fut = ex.submit(self.sentiment.analyze,   ticker)
            n_fut = ex.submit(self.news.analyze,        ticker)

        f_result = f_fut.result()
        t_result = t_fut.result()
        s_result = s_fut.result()
        n_result = n_fut.result()

        # ── Sector context ────────────────────────────────────────────
        raw_sector       = f_result.get("metrics", {}).get("sector", "")
        norm_sector      = self.sector_ag.normalise_sector(raw_sector, self.market)
        sector_info      = self._sector_data.get(norm_sector, {})
        sector_strength  = sector_info.get("strength", "Neutral")
        sector_mult      = sector_info.get("multiplier", 1.0)

        # ── Relative strength vs SPY ──────────────────────────────────
        spy_3m_pct       = self._spy_data.get("return_3m", 0.0)   # already in %
        ticker_r3m       = safe_float(t_result.get("return_3m"), 0.0) * 100.0  # convert to %
        rel_strength     = ticker_r3m - spy_3m_pct

        # ── Advanced scoring ──────────────────────────────────────────
        spy_regime = self._spy_data.get("regime", "sideways")
        scoring    = self.scorer.compute(
            f_result, t_result, s_result, n_result,
            sector_strength, spy_regime, rel_strength / 100.0,
        )

        final_score = scoring["final_score"]
        category    = self._categorise(
            safe_float(f_result.get("score"), 0),
            safe_float(t_result.get("score"), 0),
            safe_float(t_result.get("momentum_score"), 0),
            safe_float(s_result.get("score"), 0),
            final_score,
        )

        # ── Trade levels ──────────────────────────────────────────────
        entry_l  = t_result.get("entry_low",   0.0)
        entry_h  = t_result.get("entry_high",  0.0)
        exit_t   = t_result.get("exit_target", 0.0)
        stop_l   = t_result.get("stop_loss",   0.0)

        c = self.currency
        entry_str = f"{c}{entry_l:.2f} – {c}{entry_h:.2f}" if entry_l else "N/A"
        exit_str  = f"{c}{exit_t:.2f}"                      if exit_t  else "N/A"
        stop_str  = f"{c}{stop_l:.2f}"                      if stop_l  else "N/A"

        # ── Risk flags ────────────────────────────────────────────────
        risk_flags = list(s_result.get("risk_flags", []))
        if t_result.get("signal") == "Bearish":
            risk_flags.append("Bearish technicals")
        if f_result.get("valuation") == "Overvalued":
            risk_flags.append("Overvalued by fundamentals")
        if safe_float(s_result.get("score"), 0) < -0.3:
            risk_flags.append("Negative market sentiment")
        benchmark_label = "NIFTY" if self.market == "india" else "SPY"
        if spy_regime == "bearish":
            risk_flags.append(f"Bearish market regime ({benchmark_label} < 200 SMA)")
        if sector_strength == "Weak":
            risk_flags.append(f"Weak sector: {norm_sector}")

        # ── Reason string ─────────────────────────────────────────────
        reason = self._build_reason(f_result, t_result, s_result, n_result,
                                    sector_strength, spy_regime, scoring)

        metrics     = f_result.get("metrics", {})
        company     = metrics.get("company_name", ticker)

        return {
            # ── Identity ───────────────────────────────────────────────
            "ticker":              ticker,
            "company_name":        company,
            "category":            category or "Watchlist",
            "market":              self.market,
            "currency":            self.currency,
            "status":              "success",

            # ── Scores (v2) ────────────────────────────────────────────
            "combined_score":      final_score,           # backward-compat alias
            "base_score":          scoring["base_score"],
            "context_multiplier":  scoring["context_multiplier"],
            "opportunity_boost":   scoring["opportunity_boost"],
            "risk_penalty":        scoring["risk_penalty"],
            "final_score":         final_score,
            "boost_breakdown":     scoring["boost_breakdown"],
            "penalty_breakdown":   scoring["penalty_breakdown"],

            # ── Sub-agent scores ───────────────────────────────────────
            "fundamental_score":   round(safe_float(f_result.get("score"), 0), 1),
            "technical_score":     round(safe_float(t_result.get("score"), 0), 1),
            "momentum_score":      round(safe_float(t_result.get("momentum_score"), 0), 1),
            "sentiment_score":     round(safe_float(s_result.get("score"), 0), 4),

            # ── Confidence ─────────────────────────────────────────────
            "confidence":          scoring["confidence"],
            "confidence_num":      scoring["confidence_num"],

            # ── Sector / market context ───────────────────────────────
            "sector":              norm_sector,
            "sector_strength":     sector_strength,
            "sector_multiplier":   sector_mult,
            "spy_regime":          spy_regime,
            "spy_price":           self._spy_data.get("spy_price", 0.0),
            "rel_strength_vs_spy": round(rel_strength, 2),

            # ── News ───────────────────────────────────────────────────
            "news_catalyst":       n_result.get("catalyst", ""),
            "news_impact_score":   n_result.get("impact_score", 50.0),
            "news_sentiment":      n_result.get("sentiment",   0.0),
            "catalysts_found":     n_result.get("catalysts_found", []),
            "news_headlines":      n_result.get("top_headlines", []),
            "news_buzz":           n_result.get("buzz", "Low"),

            # ── Trade levels ───────────────────────────────────────────
            "entry_price_low":     round(entry_l, 2),
            "entry_price_high":    round(entry_h, 2),
            "exit_target_price":   round(exit_t, 2),
            "stop_loss_price":     round(stop_l, 2),
            "signal":              t_result.get("signal", "Neutral"),
            "valuation":           f_result.get("valuation", "Unknown"),
            "entry":               entry_str,
            "exit":                exit_str,
            "stop_loss":           stop_str,
            "current_price":       t_result.get("current_price", 0.0),

            # ── Narrative ─────────────────────────────────────────────
            "reason":              reason,
            "risk_flags":          risk_flags,
            "technical_signals":   t_result.get("signals", []),

            # ── Raw data passthrough ───────────────────────────────────
            "metrics":             metrics,
            "indicators":          t_result.get("indicators", {}),
            "price_history":       t_result.get("price_history", []),

            # ── Legacy / compat ────────────────────────────────────────
            "buzz_level":          s_result.get("buzz_level", "Low"),
            "top_headlines":       s_result.get("top_headlines", []),    # sentiment headlines
            "fundamental_summary": f_result.get("summary", ""),
            "fundamental_breakdown": f_result.get("breakdown", {}),
        }

    # ──────────────────────── categorisation ─────────────────────────

    def _categorise(
        self,
        f_score: float,
        t_score: float,
        m_score: float,
        raw_sent: float,
        combined: float,
    ) -> Optional[str]:
        th        = self.thresholds
        min_score = th.get("min_combined_score", 40)

        if combined < min_score:
            return None

        st  = th.get("short_term",  {})
        mdt = th.get("medium_term", {})
        lt  = th.get("long_term",   {})

        if (t_score  >= st.get("technical_min", 62) and
                raw_sent >= st.get("sentiment_min", 0.05) and
                combined >= st.get("combined_min", 55)):
            return "Short-Term"

        if (f_score  >= lt.get("fundamental_min", 62) and
                combined >= lt.get("combined_min", 50)):
            return "Long-Term"

        if (t_score  >= mdt.get("technical_min", 45) and
                f_score  >= mdt.get("fundamental_min", 42) and
                combined >= mdt.get("combined_min", 52)):
            return "Medium-Term"

        return None

    # ──────────────────────── reason builder ─────────────────────────

    def _build_reason(
        self,
        f_result:        dict,
        t_result:        dict,
        s_result:        dict,
        n_result:        dict,
        sector_strength: str,
        spy_regime:      str,
        scoring:         dict,
    ) -> str:
        parts = []

        # Fundamentals
        f_sum = f_result.get("summary", "")
        val   = f_result.get("valuation", "")
        if f_sum in ("Strong", "Moderate"):
            parts.append(f"{f_sum.lower()} fundamentals")
        if val == "Undervalued":
            parts.append("undervalued")

        # Technicals
        signal = t_result.get("signal", "")
        inds   = t_result.get("indicators", {})
        if signal == "Bullish":
            parts.append("bullish technical setup")
        rsi = inds.get("rsi")
        if rsi and rsi < 35:
            parts.append(f"RSI oversold ({rsi:.0f})")
        if safe_float(inds.get("macd_diff"), 0) > 0:
            parts.append("positive MACD")

        # Sector & regime
        if sector_strength == "Strong":
            parts.append(f"strong sector momentum")
        if spy_regime == "bullish":
            parts.append("bullish market regime")

        # News catalyst
        catalyst = n_result.get("catalyst", "")
        if catalyst and "No" not in catalyst and "No major" not in catalyst:
            parts.append(catalyst.lower())

        # Sentiment
        sent = safe_float(s_result.get("score"), 0)
        if sent > 0.2:
            parts.append("positive sentiment")
        elif sent < -0.2:
            parts.append("negative sentiment")

        if not parts:
            parts.append(f"combined score {scoring['final_score']:.0f}/100")

        return " + ".join(parts).capitalize() + "."

    # ──────────────────────── filter helpers ─────────────────────────

    @staticmethod
    def filter_by_category(results: List[Dict], category: str) -> List[Dict]:
        return [r for r in results if r.get("category") == category]

    @staticmethod
    def top_picks(results: List[Dict], n: int = 10) -> List[Dict]:
        return sorted(results, key=lambda r: r.get("combined_score", 0), reverse=True)[:n]

    # ── Expose last scan context for the dashboard ────────────────────
    @property
    def sector_data(self) -> Dict[str, dict]:
        return self._sector_data

    @property
    def spy_data(self) -> dict:
        return self._spy_data
