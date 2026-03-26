"""
Advanced Scoring Engine
────────────────────────
Replaces the old linear model with a context-aware formula:

  Final = (Base × Context Multiplier) + Opportunity Boost − Risk Penalty

  Base              = 0.4·F + 0.35·T + 0.25·S  (weighted sub-scores)
  Context Multiplier = sector_mult × regime_mult × rs_mult
  Opportunity Boost  = breakout + volume + news catalyst + sentiment
  Risk Penalty       = RSI overbought + volatility + weak fundamentals + overvaluation
"""

from typing import Dict, Tuple
from utils.helpers import safe_float


class ScoringEngine:
    def __init__(self, weights: dict):
        self.w_f = weights.get("fundamentals", 0.40)
        self.w_t = weights.get("technicals",   0.35)
        self.w_s = weights.get("sentiment",    0.25)

    # ─────────────────────────── public API ──────────────────────────

    def compute(
        self,
        f_result:             dict,
        t_result:             dict,
        s_result:             dict,
        news_result:          dict,
        sector_strength:      str,   # "Strong" | "Neutral" | "Weak"
        spy_regime:           str,   # "bullish" | "sideways" | "bearish"
        rel_strength_vs_spy:  float, # positive = outperforming
    ) -> Dict:
        # 1. Base score ────────────────────────────────────────────────
        f_score  = safe_float(f_result.get("score"), 0.0)
        t_score  = safe_float(t_result.get("score"), 0.0)
        raw_sent = safe_float(s_result.get("score"), 0.0)   # -1 → +1
        s_norm   = (raw_sent + 1.0) / 2.0 * 100.0           # 0 → 100
        base     = f_score * self.w_f + t_score * self.w_t + s_norm * self.w_s

        # 2. Context multiplier ────────────────────────────────────────
        sect_m   = {"Strong": 1.15, "Neutral": 1.0, "Weak": 0.85}.get(sector_strength, 1.0)
        regime_m = {"bullish": 1.1, "sideways": 1.0, "bearish": 0.8}.get(spy_regime, 1.0)
        rs_m     = 1.05 if rel_strength_vs_spy > 0.01 else (0.95 if rel_strength_vs_spy < -0.01 else 1.0)
        context  = sect_m * regime_m * rs_m

        # 3. Opportunity boost ─────────────────────────────────────────
        boost, boost_bd = self._opportunity_boost(t_result, news_result, s_result)

        # 4. Risk penalty ──────────────────────────────────────────────
        penalty, penalty_bd = self._risk_penalty(t_result, f_result)

        # 5. Final score ───────────────────────────────────────────────
        final = min(100.0, max(0.0, base * context + boost - penalty))

        # 6. Confidence ────────────────────────────────────────────────
        conf_num, conf_label = self._confidence(
            f_result, t_result, s_result, news_result, sector_strength, spy_regime
        )

        return {
            "base_score":          round(base,    1),
            "context_multiplier":  round(context, 3),
            "opportunity_boost":   round(boost,   1),
            "risk_penalty":        round(penalty, 1),
            "final_score":         round(final,   1),
            "confidence_num":      round(conf_num, 2),
            "confidence":          conf_label,
            # sub-multipliers (for UI display)
            "sector_multiplier":   sect_m,
            "regime_multiplier":   regime_m,
            "rs_multiplier":       rs_m,
            # detailed breakdowns
            "boost_breakdown":     boost_bd,
            "penalty_breakdown":   penalty_bd,
        }

    # ──────────────────────── opportunity boost ───────────────────────

    def _opportunity_boost(self, t_result: dict, news_result: dict, s_result: dict) -> Tuple[float, dict]:
        boost = 0.0
        bd    = {}

        inds      = t_result.get("indicators", {})
        vol_ratio = safe_float(inds.get("vol_ratio"), 0.0)
        signals   = t_result.get("signals", [])
        raw_sent  = safe_float(s_result.get("score"), 0.0)

        # Breakout / golden cross
        brk = 0
        for sig in signals:
            sl = sig.lower()
            if "golden cross" in sl:   brk = 10; break
            if "above sma-200"  in sl: brk = max(brk, 5)
            if "above sma-50"   in sl: brk = max(brk, 3)
        boost += brk
        bd["breakout"] = brk

        # Volume spike
        vol_b = 7 if vol_ratio > 2.0 else (3 if vol_ratio > 1.5 else 0)
        boost += vol_b
        bd["volume_spike"] = vol_b

        # News catalyst (0-100 → 0-15)
        cat_score = safe_float(news_result.get("catalyst_score"), 50.0)
        news_b    = round(max(0.0, (cat_score - 50.0) / 50.0 * 15.0), 1)
        boost    += news_b
        bd["news_catalyst"] = news_b

        # Sentiment
        sent_b = 5 if raw_sent > 0.5 else (3 if raw_sent > 0.2 else 0)
        boost += sent_b
        bd["sentiment"] = sent_b

        return boost, bd

    # ──────────────────────── risk penalty ───────────────────────────

    def _risk_penalty(self, t_result: dict, f_result: dict) -> Tuple[float, dict]:
        penalty = 0.0
        bd      = {}

        inds  = t_result.get("indicators", {})
        rsi   = safe_float(inds.get("rsi"),  50.0)
        atr   = safe_float(inds.get("atr"),  0.0)
        price = safe_float(t_result.get("current_price"), 1.0) or 1.0

        # RSI overbought
        rsi_p = 10 if rsi > 80 else (5 if rsi > 75 else 0)
        penalty += rsi_p
        bd["rsi_overbought"] = rsi_p

        # High ATR-based volatility
        atr_pct = atr / price * 100.0 if price else 0.0
        vol_p   = 5 if atr_pct > 5 else (3 if atr_pct > 3 else 0)
        penalty += vol_p
        bd["high_volatility"] = vol_p

        # Weak fundamentals
        f_score  = safe_float(f_result.get("score"), 50.0)
        fund_p   = 10 if f_score < 30 else (5 if f_score < 40 else 0)
        penalty += fund_p
        bd["weak_fundamentals"] = fund_p

        # Overvalued per P/E
        val_p = 0
        if f_result.get("valuation") == "Overvalued":
            pe = safe_float(f_result.get("metrics", {}).get("pe_ratio"), 0.0)
            val_p = 10 if (pe or 0) > 50 else (5 if (pe or 0) > 35 else 2)
        penalty += val_p
        bd["overvalued"] = val_p

        return penalty, bd

    # ──────────────────────── confidence ─────────────────────────────

    def _confidence(
        self,
        f_result:        dict,
        t_result:        dict,
        s_result:        dict,
        news_result:     dict,
        sector_strength: str,
        spy_regime:      str,
    ) -> Tuple[float, str]:
        votes = []

        inds  = t_result.get("indicators", {})
        price = safe_float(t_result.get("current_price"), 0.0)

        # Technical alignment
        sma50  = safe_float(inds.get("sma_50"),   0.0)
        sma200 = safe_float(inds.get("sma_200"),  0.0)
        rsi    = safe_float(inds.get("rsi"),      50.0)
        mdiff  = safe_float(inds.get("macd_diff"), 0.0)

        votes.append(1 if (price > sma50  > 0) else 0)
        votes.append(1 if (price > sma200 > 0) else 0)
        votes.append(1 if mdiff > 0             else 0)
        votes.append(1 if 35 <= rsi <= 70        else 0)

        # Fundamental strength
        votes.append(1 if safe_float(f_result.get("score"), 0) >= 55 else 0)

        # Sentiment
        votes.append(1 if safe_float(s_result.get("score"), 0) > 0 else 0)

        # Sector / market context
        votes.append(1 if sector_strength == "Strong" else 0)
        votes.append(1 if spy_regime      == "bullish" else 0)

        # News catalyst
        votes.append(1 if safe_float(news_result.get("catalyst_score"), 50) > 55 else 0)

        num   = sum(votes) / len(votes) if votes else 0.5
        label = "High" if num > 0.75 else ("Medium" if num >= 0.50 else "Low")
        return num, label
