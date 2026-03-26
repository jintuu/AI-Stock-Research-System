"""
Performance Tracker
═══════════════════
Stores every scan's predictions and evaluates their real-world outcomes
over time.  Feeds back into learning insights and weight optimisation.

Storage layout
──────────────
  data/tracking/
    predictions.json   ← append-only log of all predictions + outcomes
    scan_log.json      ← one entry per scan (id, timestamp, ticker count)

Prediction record schema
────────────────────────
{
  "id":               str,          # ticker_YYYYMMDD_HHMMSS
  "scan_id":          str,          # scan_YYYYMMDD_HHMMSS
  "timestamp":        str,          # ISO-8601 UTC when the scan ran
  "ticker":           str,
  "company_name":     str,
  "category":         str,          # Short-Term | Medium-Term | Long-Term | Watchlist
  "market":           str,          # us | india
  "currency":         str,          # $ | ₹

  # Trade levels (raw floats — stored at scan time)
  "entry_price":      float,        # midpoint of entry range
  "exit_target":      float,
  "stop_loss":        float,

  # Scores at scan time
  "final_score":      float,
  "confidence_num":   float,        # 0-1
  "confidence":       str,          # High | Medium | Low

  # Signal snapshot
  "signals": {
    "fundamentals":      float,     # 0-100
    "technicals":        float,     # 0-100
    "momentum":          float,     # 0-100
    "sentiment":         float,     # -1 to 1
    "news_impact_score": float,     # 0-100
    "smart_money_score": float,     # 0-100
    "sector_strength":   str,       # Strong | Neutral | Weak
    "sector":            str,
    "spy_regime":        str,       # bullish | sideways | bearish
    "accumulation":      str        # Accumulating | Distributing | Neutral
  },

  # Outcome (filled when evaluated)
  "outcome":          str,          # IN_PROGRESS | SUCCESS | FAILURE | EXPIRED
  "current_price":    float | None,
  "return_pct":       float | None,
  "max_gain_pct":     float | None,
  "max_loss_pct":     float | None,
  "days_elapsed":     int   | None,
  "evaluated_at":     str   | None  # ISO-8601 UTC
}
"""

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# ── Evaluation windows (calendar days) ───────────────────────────────────────
EVAL_WINDOWS: Dict[str, int] = {
    "Short-Term":  7,
    "Medium-Term": 28,
    "Long-Term":   90,
    "Watchlist":   14,
}


class PerformanceTracker:
    def __init__(self, storage_dir: str = "data/tracking"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._pred_file = self.storage_dir / "predictions.json"
        self._scan_file = self.storage_dir / "scan_log.json"
        self._predictions: List[dict] = self._load_predictions()

    # ═════════════════════════════════════════════════════════════════
    # PUBLIC API
    # ═════════════════════════════════════════════════════════════════

    def store_predictions(self, results: List[dict], scan_id: Optional[str] = None) -> int:
        """
        Convert scan results → prediction records and append to storage.
        Returns the number of new records stored.
        """
        if not scan_id:
            scan_id = "scan_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

        new_records = []
        for r in results:
            rec = self._build_record(r, scan_id)
            if rec:
                new_records.append(rec)

        if new_records:
            self._predictions.extend(new_records)
            self._save_predictions()
            self._append_scan_log(scan_id, len(new_records))
            logger.info(f"Stored {len(new_records)} predictions (scan={scan_id})")

        return len(new_records)

    def evaluate_pending(self, max_eval: int = 100) -> int:
        """
        Fetch updated prices for IN_PROGRESS predictions and determine outcomes.
        Returns the count of predictions whose status changed.
        """
        pending = [
            p for p in self._predictions
            if p.get("outcome") == "IN_PROGRESS"
        ][:max_eval]

        changed = 0
        for pred in pending:
            try:
                updated = self._evaluate_single(pred)
                if updated.get("outcome") != "IN_PROGRESS":
                    changed += 1
                # Update in-place
                idx = next(
                    (i for i, p in enumerate(self._predictions) if p["id"] == pred["id"]),
                    None,
                )
                if idx is not None:
                    self._predictions[idx] = updated
            except Exception as e:
                logger.warning(f"Evaluation failed for {pred.get('ticker')}: {e}")

        if changed > 0:
            self._save_predictions()
            logger.info(f"Evaluated {len(pending)} predictions, {changed} outcomes resolved.")

        return changed

    # ─── Analytics ────────────────────────────────────────────────────────────

    def compute_metrics(self) -> dict:
        """Overall performance metrics across all evaluated predictions."""
        all_p     = self._predictions
        completed = [p for p in all_p if p.get("outcome") in ("SUCCESS", "FAILURE", "EXPIRED")]
        pending   = [p for p in all_p if p.get("outcome") == "IN_PROGRESS"]

        if not all_p:
            return {"total": 0, "completed": 0, "in_progress": 0}

        if not completed:
            return {
                "total":       len(all_p),
                "completed":   0,
                "in_progress": len(pending),
                "message":     "No completed trades yet — run 'Evaluate Pending'.",
            }

        returns    = [p["return_pct"] for p in completed if p.get("return_pct") is not None]
        wins       = [p for p in completed if p.get("outcome") == "SUCCESS"]
        losses     = [p for p in completed if p.get("outcome") == "FAILURE"]
        expired    = [p for p in completed if p.get("outcome") == "EXPIRED"]

        win_rate   = len(wins) / len(completed)
        avg_return = float(np.mean(returns)) if returns else 0.0

        win_rets   = [p["return_pct"] for p in wins    if p.get("return_pct") is not None]
        loss_rets  = [abs(p["return_pct"]) for p in losses if p.get("return_pct") is not None]
        avg_win    = float(np.mean(win_rets))  if win_rets  else 0.0
        avg_loss   = float(np.mean(loss_rets)) if loss_rets else 0.0
        rr_ratio   = avg_win / avg_loss if avg_loss > 0 else 0.0

        by_cat = {}
        for cat in ("Short-Term", "Medium-Term", "Long-Term", "Watchlist"):
            cp = [p for p in completed if p.get("category") == cat]
            if cp:
                by_cat[cat] = self._cat_stats(cp)

        best  = max(completed, key=lambda p: p.get("return_pct", -999), default=None)
        worst = min(completed, key=lambda p: p.get("return_pct",  999), default=None)

        # Streak
        recent = sorted(completed, key=lambda p: p.get("evaluated_at", ""))[-20:]
        streak, streak_type = self._compute_streak(recent)

        return {
            "total":         len(all_p),
            "completed":     len(completed),
            "in_progress":   len(pending),
            "wins":          len(wins),
            "losses":        len(losses),
            "expired":       len(expired),
            "win_rate":      round(win_rate,   3),
            "avg_return":    round(avg_return, 2),
            "avg_win":       round(avg_win,    2),
            "avg_loss":      round(avg_loss,   2),
            "risk_reward":   round(rr_ratio,   2),
            "total_return":  round(sum(returns), 2),
            "by_category":   by_cat,
            "best_pick":     best,
            "worst_pick":    worst,
            "streak":        streak,
            "streak_type":   streak_type,
        }

    def signal_effectiveness(self) -> dict:
        """
        Break down win rates by sentiment, sector strength, score bucket, and regime.
        Requires at least 5 completed trades to be meaningful.
        """
        completed = [
            p for p in self._predictions
            if p.get("outcome") in ("SUCCESS", "FAILURE", "EXPIRED")
            and p.get("return_pct") is not None
        ]

        if len(completed) < 5:
            return {"message": f"Need ≥5 completed trades (have {len(completed)})."}

        def _bucket(preds, key, buckets):
            result = {}
            for label, lo, hi in buckets:
                subset = [
                    p for p in preds
                    if lo <= (p.get("signals", {}).get(key) or 0) < hi
                ]
                result[label] = self._cat_stats(subset)
            return result

        out = {}

        # Sentiment buckets
        out["sentiment"] = _bucket(
            completed, "sentiment",
            [("Negative (<-0.1)", -1, -0.1),
             ("Neutral (-0.1–0.3)", -0.1, 0.3),
             ("Positive (>0.3)", 0.3, 1.1)],
        )

        # Sector strength
        ss_result = {}
        for s in ("Strong", "Neutral", "Weak"):
            subset = [p for p in completed if p.get("signals", {}).get("sector_strength") == s]
            ss_result[s] = self._cat_stats(subset)
        out["sector_strength"] = ss_result

        # Final score bucket
        out["score_bucket"] = _bucket(
            completed, "final_score",
            [("Low (<55)", 0, 55),
             ("Mid (55–70)", 55, 70),
             ("High (≥70)", 70, 101)],
        )

        # Smart money score
        out["smart_money"] = _bucket(
            completed, "smart_money_score",
            [("Weak (<40)", 0, 40),
             ("Neutral (40–60)", 40, 60),
             ("Strong (≥60)", 60, 101)],
        )

        # Market regime
        regime_result = {}
        for reg in ("bullish", "sideways", "bearish"):
            subset = [p for p in completed if p.get("signals", {}).get("spy_regime") == reg]
            regime_result[reg] = self._cat_stats(subset)
        out["regime"] = regime_result

        return out

    def generate_insights(self) -> List[str]:
        """Generate plain-English insights from historical performance data."""
        metrics = self.compute_metrics()
        se      = self.signal_effectiveness()
        insights = []

        if metrics.get("completed", 0) < 5:
            return ["Not enough completed trades yet — evaluate pending picks first."]

        wr  = metrics.get("win_rate", 0)
        rr  = metrics.get("risk_reward", 0)
        avg = metrics.get("avg_return", 0)

        # Overall summary
        insights.append(
            f"Overall win rate: **{wr*100:.0f}%** across {metrics['completed']} completed trades "
            f"| Avg return: **{avg:+.1f}%** | Risk-reward: **{rr:.2f}×**"
        )

        # Category insights
        for cat, data in metrics.get("by_category", {}).items():
            if data.get("count", 0) >= 3:
                cwr = data.get("win_rate", 0)
                car = data.get("avg_return", 0)
                tag = "✅" if cwr >= 0.6 else ("⚠️" if cwr >= 0.4 else "❌")
                insights.append(
                    f"{tag} **{cat}**: {cwr*100:.0f}% win rate, avg {car:+.1f}% return "
                    f"({data['count']} trades)"
                )

        # Sentiment effectiveness
        sent = se.get("sentiment", {})
        high_s = sent.get("Positive (>0.3)", {})
        neg_s  = sent.get("Negative (<-0.1)", {})
        if high_s.get("count", 0) >= 3 and neg_s.get("count", 0) >= 3:
            diff = high_s.get("win_rate", 0) - neg_s.get("win_rate", 0)
            if abs(diff) > 0.1:
                direction = "outperform" if diff > 0 else "underperform"
                insights.append(
                    f"📰 Positive-sentiment stocks {direction} negative-sentiment ones "
                    f"by **{abs(diff)*100:.0f}%** win rate"
                )

        # Sector strength
        ss = se.get("sector_strength", {})
        strong_s = ss.get("Strong", {})
        weak_s   = ss.get("Weak", {})
        if strong_s.get("count", 0) >= 3 and weak_s.get("count", 0) >= 3:
            diff = strong_s.get("win_rate", 0) - weak_s.get("win_rate", 0)
            if abs(diff) > 0.1:
                insights.append(
                    f"🏭 Strong-sector picks have **{abs(diff)*100:.0f}%** "
                    f"{'higher' if diff > 0 else 'lower'} win rate than weak-sector picks"
                )

        # Score bucket
        sc = se.get("score_bucket", {})
        high_sc = sc.get("High (≥70)", {})
        low_sc  = sc.get("Low (<55)", {})
        if high_sc.get("count", 0) >= 3:
            insights.append(
                f"🎯 High-score picks (≥70) achieve **{high_sc.get('win_rate',0)*100:.0f}%** win rate"
            )
        if low_sc.get("count", 0) >= 3 and low_sc.get("win_rate", 1) < 0.4:
            insights.append(
                "⚠️ Low-score picks (<55) are underperforming — consider raising the minimum score threshold"
            )

        # Smart money
        sm = se.get("smart_money", {})
        strong_sm = sm.get("Strong (≥60)", {})
        if strong_sm.get("count", 0) >= 3:
            insights.append(
                f"🏦 Stocks with strong smart-money signals (≥60) win at "
                f"**{strong_sm.get('win_rate',0)*100:.0f}%**"
            )

        # Regime
        reg = se.get("regime", {})
        bull_r = reg.get("bullish", {})
        bear_r = reg.get("bearish", {})
        if bull_r.get("count", 0) >= 3 and bear_r.get("count", 0) >= 3:
            diff = bull_r.get("win_rate", 0) - bear_r.get("win_rate", 0)
            if diff > 0.15:
                insights.append(
                    f"📈 Scans during bullish regimes win **{diff*100:.0f}%** more often than bearish regimes"
                )

        # Streak
        streak = metrics.get("streak", 0)
        stype  = metrics.get("streak_type", "")
        if streak >= 3:
            emoji = "🔥" if stype == "WIN" else "❄️"
            insights.append(f"{emoji} Current {streak}-trade {stype.lower()} streak")

        # Best/worst
        best  = metrics.get("best_pick")
        worst = metrics.get("worst_pick")
        if best:
            insights.append(
                f"🏆 Best pick: **{best['ticker']}** (+{best['return_pct']:.1f}%, {best['category']})"
            )
        if worst:
            insights.append(
                f"📉 Worst pick: **{worst['ticker']}** ({worst['return_pct']:.1f}%, {worst['category']})"
            )

        return insights

    def get_equity_curve(self) -> pd.DataFrame:
        """
        Cumulative return curve — compound $100 starting capital over completed trades
        sorted by evaluation date.
        """
        completed = sorted(
            [p for p in self._predictions
             if p.get("outcome") in ("SUCCESS", "FAILURE", "EXPIRED")
             and p.get("return_pct") is not None
             and p.get("evaluated_at")],
            key=lambda p: p.get("evaluated_at", ""),
        )
        if not completed:
            return pd.DataFrame()

        equity = 100.0
        rows = []
        for p in completed:
            r      = p["return_pct"] / 100.0
            equity = equity * (1 + r)
            rows.append({
                "date":              p["evaluated_at"][:10],
                "ticker":            p["ticker"],
                "category":          p["category"],
                "return_pct":        p["return_pct"],
                "equity":            round(equity, 2),
                "outcome":           p["outcome"],
            })

        return pd.DataFrame(rows)

    def suggest_weight_optimization(self) -> dict:
        """
        Compare average signal values for winning vs losing trades.
        Suggests weights proportional to each signal's discriminating power.
        """
        completed = [
            p for p in self._predictions
            if p.get("outcome") in ("SUCCESS", "FAILURE", "EXPIRED")
        ]

        if len(completed) < 10:
            return {"message": f"Need ≥10 completed trades (have {len(completed)})."}

        wins   = [p for p in completed if p.get("outcome") == "SUCCESS"]
        losses = [p for p in completed if p.get("outcome") in ("FAILURE", "EXPIRED")]

        def _avg(preds, key):
            vals = [p.get("signals", {}).get(key) for p in preds]
            vals = [v for v in vals if v is not None]
            return float(np.mean(vals)) if vals else 0.0

        signal_keys = [
            ("fundamentals",      "Fundamentals"),
            ("technicals",        "Technicals"),
            ("momentum",          "Momentum"),
            ("sentiment",         "Sentiment"),
            ("news_impact_score", "News"),
            ("smart_money_score", "Smart Money"),
        ]

        effectiveness = {}
        for key, label in signal_keys:
            w_avg = _avg(wins,   key)
            l_avg = _avg(losses, key)
            disc  = w_avg - l_avg          # higher = more discriminating
            effectiveness[label] = {
                "win_avg":            round(w_avg, 1),
                "loss_avg":           round(l_avg, 1),
                "discriminating_power": round(disc, 1),
            }

        # Map discriminating power to the three core weights
        core_keys = ["fundamentals", "technicals", "sentiment"]
        core_labels = ["Fundamentals", "Technicals", "Sentiment"]
        powers = {
            lbl: max(0.01, effectiveness.get(lbl, {}).get("discriminating_power", 0))
            for lbl in core_labels
        }
        total_p = sum(powers.values())
        suggested = {
            k.lower(): round(v / total_p, 2)
            for k, v in powers.items()
        }
        # Ensure they sum to 1.0
        keys = list(suggested.keys())
        diff = round(1.0 - sum(suggested.values()), 2)
        suggested[keys[0]] = round(suggested[keys[0]] + diff, 2)

        return {
            "signal_effectiveness":  effectiveness,
            "suggested_weights":     suggested,
            "trade_count":           len(completed),
            "message": (
                f"Based on {len(completed)} completed trades — "
                f"{'confident' if len(completed) >= 30 else 'tentative (need more data)'} suggestion"
            ),
        }

    def get_predictions_df(self) -> pd.DataFrame:
        """Return all predictions as a DataFrame for easy inspection."""
        if not self._predictions:
            return pd.DataFrame()
        rows = []
        for p in self._predictions:
            rows.append({
                "timestamp":    p.get("timestamp", "")[:10],
                "ticker":       p.get("ticker"),
                "category":     p.get("category"),
                "market":       p.get("market", "us"),
                "entry_price":  p.get("entry_price"),
                "exit_target":  p.get("exit_target"),
                "stop_loss":    p.get("stop_loss"),
                "final_score":  p.get("final_score"),
                "confidence":   p.get("confidence"),
                "outcome":      p.get("outcome"),
                "return_pct":   p.get("return_pct"),
                "days_elapsed": p.get("days_elapsed"),
                "sector":       p.get("signals", {}).get("sector"),
                "sector_strength": p.get("signals", {}).get("sector_strength"),
            })
        return pd.DataFrame(rows)

    # ═════════════════════════════════════════════════════════════════
    # INTERNALS
    # ═════════════════════════════════════════════════════════════════

    def _evaluate_single(self, pred: dict) -> dict:
        pred = dict(pred)   # shallow copy — don't mutate caller's dict
        ticker       = pred["ticker"]
        entry_price  = pred.get("entry_price", 0.0)
        exit_target  = pred.get("exit_target",  0.0)
        stop_loss    = pred.get("stop_loss",    0.0)
        category     = pred.get("category", "Medium-Term")
        max_days     = EVAL_WINDOWS.get(category, 28)

        if not entry_price or entry_price <= 0:
            return pred

        try:
            entry_dt = pd.Timestamp(pred["timestamp"]).date()
        except Exception:
            return pred

        try:
            df = yf.download(
                ticker,
                start=str(entry_dt),
                progress=False,
                auto_adjust=True,
            )
        except Exception as e:
            logger.debug(f"yfinance download failed for {ticker}: {e}")
            return pred

        if df is None or df.empty:
            return pred

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        outcome    = "IN_PROGRESS"
        exit_price = None
        exit_date  = None
        max_gain   = 0.0
        max_loss   = 0.0

        for date, row in df.iterrows():
            hi    = float(row["High"])
            lo    = float(row["Low"])
            close = float(row["Close"])
            days  = (date.date() - entry_dt).days

            # Track max gain / drawdown
            max_gain = max(max_gain, (hi   - entry_price) / entry_price * 100)
            max_loss = min(max_loss, (lo   - entry_price) / entry_price * 100)

            # Stop hit
            if stop_loss > 0 and lo <= stop_loss:
                outcome    = "FAILURE"
                exit_price = stop_loss
                exit_date  = date
                break

            # Target hit
            if exit_target > 0 and hi >= exit_target:
                outcome    = "SUCCESS"
                exit_price = exit_target
                exit_date  = date
                break

            # Window expired
            if days >= max_days:
                outcome    = "EXPIRED"
                exit_price = close
                exit_date  = date
                break

        if outcome != "IN_PROGRESS" and exit_price is not None:
            return_pct = (exit_price - entry_price) / entry_price * 100
            pred.update({
                "outcome":      outcome,
                "current_price": round(exit_price, 2),
                "return_pct":   round(return_pct, 2),
                "max_gain_pct": round(max_gain, 2),
                "max_loss_pct": round(max_loss, 2),
                "days_elapsed": (exit_date.date() - entry_dt).days if exit_date else 0,
                "evaluated_at": datetime.now(timezone.utc).isoformat(),
            })
        else:
            # Still in progress — update current price snapshot
            if len(df) > 0:
                curr = float(df["Close"].iloc[-1])
                pred.update({
                    "current_price": round(curr, 2),
                    "return_pct":   round((curr - entry_price) / entry_price * 100, 2),
                    "max_gain_pct": round(max_gain, 2),
                    "max_loss_pct": round(max_loss, 2),
                    "days_elapsed": (df.index[-1].date() - entry_dt).days,
                })

        return pred

    # ─── Record builder ───────────────────────────────────────────────────────

    @staticmethod
    def _build_record(result: dict, scan_id: str) -> Optional[dict]:
        """Convert one analyse_ticker() result dict into a prediction record."""
        ticker = result.get("ticker", "")
        if not ticker:
            return None

        ts = datetime.now(timezone.utc).isoformat()
        rid = f"{ticker}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

        # Numeric trade levels (prefer direct numeric fields added in this PR)
        entry_l = result.get("entry_price_low",  0.0) or 0.0
        entry_h = result.get("entry_price_high", 0.0) or 0.0
        entry_p = (entry_l + entry_h) / 2 if entry_l and entry_h else result.get("current_price", 0.0)
        exit_t  = result.get("exit_target_price", 0.0) or 0.0
        stop_l  = result.get("stop_loss_price",   0.0) or 0.0

        # If raw numeric fields not present, fall back to current_price for entry
        if not entry_p:
            entry_p = result.get("current_price", 0.0)

        inds    = result.get("indicators", {})
        signals = {
            "fundamentals":      result.get("fundamental_score",   0),
            "technicals":        result.get("technical_score",      0),
            "momentum":          result.get("momentum_score",       0),
            "sentiment":         result.get("sentiment_score",      0),
            "news_impact_score": result.get("news_impact_score",   50),
            "smart_money_score": inds.get("smart_money_score",     50),
            "sector_strength":   result.get("sector_strength",  "Neutral"),
            "sector":            result.get("sector",              ""),
            "spy_regime":        result.get("spy_regime",     "sideways"),
            "accumulation":      result.get("accumulation_status", "Neutral"),
            "final_score":       result.get("final_score",          0),
        }

        return {
            "id":             rid,
            "scan_id":        scan_id,
            "timestamp":      ts,
            "ticker":         ticker,
            "company_name":   result.get("company_name", ticker),
            "category":       result.get("category", "Watchlist"),
            "market":         result.get("market", "us"),
            "currency":       result.get("currency", "$"),
            "entry_price":    round(float(entry_p), 4),
            "exit_target":    round(float(exit_t),  4),
            "stop_loss":      round(float(stop_l),  4),
            "final_score":    result.get("final_score",    0),
            "confidence_num": result.get("confidence_num", 0.5),
            "confidence":     result.get("confidence",     "Medium"),
            "signals":        signals,
            # Outcome fields (all null until evaluated)
            "outcome":        "IN_PROGRESS",
            "current_price":  result.get("current_price"),
            "return_pct":     None,
            "max_gain_pct":   None,
            "max_loss_pct":   None,
            "days_elapsed":   None,
            "evaluated_at":   None,
        }

    # ─── Stats helper ─────────────────────────────────────────────────────────

    @staticmethod
    def _cat_stats(preds: List[dict]) -> dict:
        if not preds:
            return {"count": 0, "win_rate": 0.0, "avg_return": 0.0}
        wins    = [p for p in preds if p.get("outcome") == "SUCCESS"]
        returns = [p["return_pct"] for p in preds if p.get("return_pct") is not None]
        return {
            "count":      len(preds),
            "win_rate":   round(len(wins) / len(preds), 3),
            "avg_return": round(float(np.mean(returns)) if returns else 0.0, 2),
        }

    @staticmethod
    def _compute_streak(recent: List[dict]) -> Tuple[int, str]:
        if not recent:
            return 0, ""
        streak, streak_type = 1, "WIN" if recent[-1].get("outcome") == "SUCCESS" else "LOSS"
        for p in reversed(recent[:-1]):
            cur_type = "WIN" if p.get("outcome") == "SUCCESS" else "LOSS"
            if cur_type == streak_type:
                streak += 1
            else:
                break
        return streak, streak_type

    # ─── Persistence ──────────────────────────────────────────────────────────

    def _load_predictions(self) -> List[dict]:
        if self._pred_file.exists():
            try:
                with open(self._pred_file, "r") as f:
                    data = json.load(f)
                    return data if isinstance(data, list) else []
            except Exception as e:
                logger.warning(f"Could not load predictions: {e}")
        return []

    def _save_predictions(self):
        try:
            with open(self._pred_file, "w") as f:
                json.dump(self._predictions, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Could not save predictions: {e}")

    def _append_scan_log(self, scan_id: str, count: int):
        log = []
        if self._scan_file.exists():
            try:
                with open(self._scan_file, "r") as f:
                    log = json.load(f)
            except Exception:
                pass
        log.append({
            "scan_id":   scan_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "count":     count,
        })
        try:
            with open(self._scan_file, "w") as f:
                json.dump(log, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save scan log: {e}")

    # ─── Public helpers ───────────────────────────────────────────────────────

    @property
    def prediction_count(self) -> int:
        return len(self._predictions)

    @property
    def pending_count(self) -> int:
        return sum(1 for p in self._predictions if p.get("outcome") == "IN_PROGRESS")
