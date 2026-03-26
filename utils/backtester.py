"""
Simple Backtester
──────────────────
Given a ticker and a hypothetical entry/stop/target,
simulates the trade over historical data and reports:
  - Win/Loss
  - Actual return %
  - Max drawdown while in trade
  - Days held

Also supports loading past saved results and checking
how the recommendations performed over time.
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import json

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


class Backtester:
    def __init__(self, results_dir: str = "data/results"):
        self.results_dir = Path(results_dir)

    # ─────────────────────────── single trade ────────────────────────

    def backtest_trade(
        self,
        ticker: str,
        entry_price: float,
        stop_loss: float,
        exit_target: float,
        entry_date: Optional[str] = None,
        lookahead_days: int = 90,
    ) -> Dict:
        """
        Simulate a trade entered at entry_price.
        Evaluates over lookahead_days of actual price data.
        """
        try:
            if entry_date is None:
                entry_date = datetime.utcnow().strftime("%Y-%m-%d")

            end_dt = (datetime.strptime(entry_date, "%Y-%m-%d") +
                      timedelta(days=lookahead_days)).strftime("%Y-%m-%d")

            df = yf.download(ticker, start=entry_date, end=end_dt,
                             progress=False, auto_adjust=True)

            if df is None or df.empty:
                return self._error_result(ticker, "No price data available")

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            # Walk through each bar
            outcome   = "Open"
            exit_price = None
            exit_day   = None
            max_high   = entry_price
            min_price  = entry_price

            for i, (date, row) in enumerate(df.iterrows()):
                low   = float(row["Low"])
                high  = float(row["High"])
                close = float(row["Close"])

                max_high  = max(max_high, high)
                min_price = min(min_price, low)

                # Check stop-loss hit
                if low <= stop_loss:
                    outcome    = "Loss"
                    exit_price = stop_loss
                    exit_day   = date
                    break

                # Check target hit
                if high >= exit_target:
                    outcome    = "Win"
                    exit_price = exit_target
                    exit_day   = date
                    break

                # Last bar reached without either
                if i == len(df) - 1:
                    outcome    = "Expired"
                    exit_price = close
                    exit_day   = date

            if exit_price is None:
                return self._error_result(ticker, "Could not simulate trade")

            pnl_pct      = (exit_price - entry_price) / entry_price * 100
            max_drawdown = (min_price - entry_price) / entry_price * 100
            max_runup    = (max_high - entry_price) / entry_price * 100
            days_held    = (pd.Timestamp(exit_day) -
                            pd.Timestamp(entry_date)).days if exit_day else 0
            risk_reward  = abs(exit_target - entry_price) / abs(entry_price - stop_loss) \
                           if abs(entry_price - stop_loss) > 0 else 0

            return {
                "ticker":       ticker,
                "outcome":      outcome,
                "entry_price":  round(entry_price, 2),
                "exit_price":   round(exit_price, 2),
                "stop_loss":    round(stop_loss, 2),
                "exit_target":  round(exit_target, 2),
                "pnl_pct":      round(pnl_pct, 2),
                "max_drawdown": round(max_drawdown, 2),
                "max_runup":    round(max_runup, 2),
                "days_held":    days_held,
                "risk_reward":  round(risk_reward, 2),
                "entry_date":   entry_date,
                "exit_date":    str(exit_day)[:10] if exit_day else None,
                "status":       "success",
            }

        except Exception as e:
            logger.warning(f"Backtest failed for {ticker}: {e}")
            return self._error_result(ticker, str(e))

    # ─────────────────────────── batch backtest ───────────────────────

    def backtest_saved_results(self, json_file: Optional[str] = None) -> pd.DataFrame:
        """
        Load a saved scan JSON and backtest every recommendation.
        Picks the most recent file if none specified.
        """
        if json_file is None:
            files = sorted(self.results_dir.glob("scan_*.json"), reverse=True)
            if not files:
                logger.warning("No saved scan results found.")
                return pd.DataFrame()
            json_file = str(files[0])

        # Parse the scan date from filename e.g. scan_20260324_143000.json
        entry_date = None
        try:
            stem = Path(json_file).stem          # "scan_20260324_143000"
            date_part = stem.split("_")[1]       # "20260324"
            entry_date = datetime.strptime(date_part, "%Y%m%d").strftime("%Y-%m-%d")
        except Exception:
            pass

        # If the scan was run today there's no forward data yet — warn and use
        # 30 days ago so the simulation at least has some bars to work with.
        today = datetime.utcnow().date()
        if entry_date:
            scan_date = datetime.strptime(entry_date, "%Y-%m-%d").date()
            days_since = (today - scan_date).days
            if days_since < 2:
                logger.warning(
                    "Scan was run today — no forward price data exists yet. "
                    "Backtest will simulate using prices from the scan date forward; "
                    "results will show 'Expired' until enough days have passed."
                )
        else:
            entry_date = today.strftime("%Y-%m-%d")

        with open(json_file) as f:
            results = json.load(f)

        rows = []
        for r in results:
            ticker = r.get("ticker")
            if not ticker:
                continue

            # Parse entry price from string e.g. "$182.50 – $184.10" (en-dash)
            entry_raw = r.get("entry", "")
            try:
                # Handle both en-dash (–) and regular hyphen (-)
                clean = entry_raw.replace("$", "").replace("\u2013", "-")
                entry_price = float(clean.split("-")[0].strip())
            except Exception:
                continue

            stop_raw = r.get("stop_loss", "")
            try:
                stop_price = float(stop_raw.replace("$", "").strip())
            except Exception:
                continue

            exit_raw = r.get("exit", "")
            try:
                exit_price = float(exit_raw.replace("$", "").strip())
            except Exception:
                continue

            # Skip degenerate levels
            if stop_price <= 0 or exit_price <= 0 or exit_price <= entry_price:
                continue

            bt = self.backtest_trade(ticker, entry_price, stop_price, exit_price,
                                     entry_date=entry_date)
            bt["category"]      = r.get("category")
            bt["combined_score"] = r.get("combined_score")
            rows.append(bt)

        return pd.DataFrame(rows)

    # ─────────────────────────── summary stats ────────────────────────

    @staticmethod
    def summarise(df: pd.DataFrame) -> Dict:
        if df.empty:
            return {}
        wins   = (df["outcome"] == "Win").sum()
        losses = (df["outcome"] == "Loss").sum()
        total  = len(df)
        win_rate = wins / total * 100 if total else 0
        avg_pnl  = df["pnl_pct"].mean() if "pnl_pct" in df else 0
        avg_win  = df.loc[df["outcome"] == "Win",  "pnl_pct"].mean() if wins else 0
        avg_loss = df.loc[df["outcome"] == "Loss", "pnl_pct"].mean() if losses else 0

        return {
            "total_trades": total,
            "wins":         int(wins),
            "losses":       int(losses),
            "win_rate_pct": round(win_rate, 1),
            "avg_pnl_pct":  round(avg_pnl, 2),
            "avg_win_pct":  round(avg_win, 2),
            "avg_loss_pct": round(avg_loss, 2),
            "expectancy":   round(avg_win * win_rate/100 + avg_loss * (1-win_rate/100), 2),
        }

    # ──────────────────────── internal ───────────────────────────────

    @staticmethod
    def _error_result(ticker: str, msg: str) -> Dict:
        return {
            "ticker":  ticker,
            "outcome": "Error",
            "status":  "error",
            "error":   msg,
        }
