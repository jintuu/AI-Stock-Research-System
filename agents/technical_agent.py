"""
Technical Agent
───────────────
Uses yfinance for OHLCV data and the `ta` library for indicators.

Standard indicators: SMA-20/50/200, EMA-12/26, RSI-14, MACD, Bollinger Bands, ATR-14

Smart Money indicators (NEW):
  - OBV + OBV divergence vs price (accumulation/distribution detection)
  - Chaikin Money Flow (CMF-20)  — buying/selling pressure
  - Money Flow Index (MFI-14)    — volume-weighted RSI
  - Rolling VWAP-20              — institutional reference price
  - Accumulation/Distribution Line
  - Volume Profile / Point of Control (POC)
  - Climax volume detection
  - Breakout volume confirmation

Smart money events detected:
  Institutional Accumulation, Smart Money Distribution, Volume Breakout Confirmed,
  Climax Volume (exhaustion), VWAP Reclaim, VWAP Rejection,
  OBV Bullish Divergence, OBV Bearish Divergence, MFI Oversold Bounce,
  High-Volume Support Zone, CMF Positive Flow

Output schema:
{
  "ticker":              str,
  "agent":               "technical",
  "score":               float,         # 0–100
  "momentum_score":      float,         # 0–100
  "signal":              str,           # Bullish | Bearish | Neutral
  "entry_low":           float,
  "entry_high":          float,
  "exit_target":         float,
  "stop_loss":           float,
  "indicators":          dict,          # includes smart money indicators
  "signals":             list[str],
  "smart_money_signals": list[str],     # NEW — smart money events
  "smart_money_score":   float,         # NEW — 0–100 composite
  "accumulation_status": str,           # NEW — Accumulating | Distributing | Neutral
  "poc_price":           float,         # NEW — Point of Control (highest volume price)
  "current_price":       float,
  "return_3m":           float,
  "price_history":       list[dict],
  "status":              str,
  "error":               str | None
}
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

from utils.helpers import safe_float, load_config, cache_get, cache_set

logger = logging.getLogger(__name__)


class TechnicalAgent:
    def __init__(self, config: Optional[dict] = None):
        self.config    = config or load_config()
        self._tech_cfg = self.config.get("technical", {})

    # ─────────────────────────── public API ──────────────────────────────────

    def analyze(self, ticker: str) -> Dict:
        cache_key = f"technical_{ticker}"
        ttl = self.config.get("data", {}).get("cache_ttl_minutes", 60)

        if self.config.get("data", {}).get("cache_enabled", True):
            cached = cache_get(cache_key, ttl)
            if cached:
                return cached

        result = self._run_analysis(ticker)

        if self.config.get("data", {}).get("cache_enabled", True):
            cache_set(cache_key, result, ttl)

        return result

    # ──────────────────────────── core flow ──────────────────────────────────

    def _run_analysis(self, ticker: str) -> Dict:
        try:
            days   = self.config.get("data", {}).get("price_history_days", 400)
            df     = yf.download(ticker, period=f"{days}d", progress=False, auto_adjust=True)

            if df is None or df.empty or len(df) < 50:
                raise ValueError(f"Insufficient price data ({len(df) if df is not None else 0} rows)")

            df = df.copy()
            df = self._flatten_columns(df)
            df = self._add_indicators(df)
            df = self._add_smart_money_indicators(df)

            score, momentum_score, breakdown  = self._calculate_score(df)
            sm_score, sm_signals, accum_status = self._detect_smart_money(df)
            signal                             = self._determine_signal(score, breakdown)
            entry_l, entry_h, exit_t, stop_l  = self._calculate_levels(df)
            signals_list                       = self._build_signal_list(df, breakdown)
            indicators                         = self._extract_indicators(df, sm_score)
            poc                                = self._volume_profile_poc(df)

            current_price = float(df["Close"].iloc[-1])
            close = df["Close"]
            r3m = float(
                (close.iloc[-1] - close.iloc[-63]) / close.iloc[-63]
                if len(close) >= 63
                else (close.iloc[-1] - close.iloc[0])  / close.iloc[0]
            )

            return {
                "ticker":              ticker,
                "agent":               "technical",
                "score":               round(score, 1),
                "momentum_score":      round(momentum_score, 1),
                "signal":              signal,
                "entry_low":           round(entry_l, 2),
                "entry_high":          round(entry_h, 2),
                "exit_target":         round(exit_t, 2),
                "stop_loss":           round(stop_l, 2),
                "indicators":          indicators,
                "signals":             signals_list,
                "smart_money_signals": sm_signals,
                "smart_money_score":   round(sm_score, 1),
                "accumulation_status": accum_status,
                "poc_price":           round(poc, 2),
                "current_price":       round(current_price, 2),
                "return_3m":           round(r3m, 4),
                "price_history":       self._df_to_records(df),
                "status":              "success",
                "error":               None,
            }

        except Exception as e:
            logger.warning(f"[technical] {ticker}: {e}")
            return {
                "ticker":              ticker,
                "agent":               "technical",
                "score":               0.0,
                "momentum_score":      0.0,
                "signal":              "Neutral",
                "entry_low":           0.0,
                "entry_high":          0.0,
                "exit_target":         0.0,
                "stop_loss":           0.0,
                "indicators":          {},
                "signals":             [],
                "smart_money_signals": [],
                "smart_money_score":   0.0,
                "accumulation_status": "Neutral",
                "poc_price":           0.0,
                "current_price":       0.0,
                "return_3m":           0.0,
                "price_history":       [],
                "status":              "error",
                "error":               str(e),
            }

    # ─────────────────────────── helpers ─────────────────────────────────────

    def _flatten_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df

    # ──────────────────── standard indicator calculation ─────────────────────

    def _add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["Close"]
        high  = df["High"]
        low   = df["Low"]
        vol   = df["Volume"]

        try:
            from ta.trend    import SMAIndicator, EMAIndicator, MACD
            from ta.momentum import RSIIndicator
            from ta.volatility import BollingerBands, AverageTrueRange
            from ta.volume   import OnBalanceVolumeIndicator

            sma_s  = self._tech_cfg.get("sma_short",   20)
            sma_m  = self._tech_cfg.get("sma_mid",     50)
            sma_l  = self._tech_cfg.get("sma_long",   200)
            ema_f  = self._tech_cfg.get("ema_fast",    12)
            ema_sl = self._tech_cfg.get("ema_slow",    26)

            df["sma_20"]  = SMAIndicator(close, window=sma_s).sma_indicator()
            df["sma_50"]  = SMAIndicator(close, window=sma_m).sma_indicator()
            df["sma_200"] = SMAIndicator(close, window=sma_l).sma_indicator()
            df["ema_12"]  = EMAIndicator(close, window=ema_f).ema_indicator()
            df["ema_26"]  = EMAIndicator(close, window=ema_sl).ema_indicator()

            rsi_p     = self._tech_cfg.get("rsi_period", 14)
            df["rsi"] = RSIIndicator(close, window=rsi_p).rsi()

            mf, msl, msi = (self._tech_cfg.get(k, v)
                            for k, v in [("macd_fast", 12), ("macd_slow", 26), ("macd_signal", 9)])
            _macd = MACD(close, window_slow=msl, window_fast=mf, window_sign=msi)
            df["macd"]        = _macd.macd()
            df["macd_signal"] = _macd.macd_signal()
            df["macd_diff"]   = _macd.macd_diff()

            bb_p, bb_sd = self._tech_cfg.get("bb_period", 20), self._tech_cfg.get("bb_std", 2)
            bb = BollingerBands(close, window=bb_p, window_dev=bb_sd)
            df["bb_upper"]  = bb.bollinger_hband()
            df["bb_middle"] = bb.bollinger_mavg()
            df["bb_lower"]  = bb.bollinger_lband()
            df["bb_width"]  = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"]

            atr_p     = self._tech_cfg.get("atr_period", 14)
            df["atr"] = AverageTrueRange(high, low, close, window=atr_p).average_true_range()

            vol_p         = self._tech_cfg.get("volume_ma_period", 20)
            df["vol_ma"]  = vol.rolling(vol_p).mean()
            df["vol_ratio"] = vol / df["vol_ma"]

            df["obv"] = OnBalanceVolumeIndicator(close, vol).on_balance_volume()

        except ImportError:
            logger.error("'ta' library not installed — using manual fallback.")
            df = self._manual_indicators(df)

        return df

    # ───────────────────── smart money indicators (NEW) ──────────────────────

    def _add_smart_money_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["Close"]
        high  = df["High"]
        low   = df["Low"]
        vol   = df["Volume"]

        # ── Chaikin Money Flow (CMF-20) ─────────────────────────────────────
        # Positive = buying pressure, Negative = selling pressure
        hl_range = (high - low).replace(0, np.nan)
        clv      = (2 * close - high - low) / hl_range   # close location value
        df["cmf"] = (clv * vol).rolling(20).sum() / vol.rolling(20).sum()

        # ── Money Flow Index (MFI-14) ── volume-weighted RSI ─────────────────
        tp  = (high + low + close) / 3
        rmf = tp * vol
        pos_mf = rmf.where(tp > tp.shift(1), 0.0).rolling(14).sum()
        neg_mf = rmf.where(tp < tp.shift(1), 0.0).rolling(14).sum()
        mfi_ratio = pos_mf / neg_mf.replace(0, np.nan)
        df["mfi"] = 100 - (100 / (1 + mfi_ratio))

        # ── Rolling VWAP-20 (daily bars approximation) ───────────────────────
        # Institutions use VWAP as reference; above = bullish, below = bearish
        df["vwap_20"] = (tp * vol).rolling(20).sum() / vol.rolling(20).sum()

        # ── Accumulation / Distribution (A/D) Line ──────────────────────────
        # Rising = smart money buying, Falling = smart money selling
        df["ad_line"] = (clv * vol).cumsum()

        # ── OBV Slope (10-bar linear regression normalised) ──────────────────
        # Used for divergence detection: compare OBV slope vs price slope
        def _slope_norm(series: pd.Series, window: int = 10) -> pd.Series:
            slopes = []
            for i in range(len(series)):
                if i < window - 1:
                    slopes.append(np.nan)
                else:
                    seg = series.iloc[i - window + 1 : i + 1].values.astype(float)
                    if seg[0] == 0 or np.isnan(seg[0]):
                        slopes.append(np.nan)
                    else:
                        norm = seg / seg[0]               # normalize to start = 1
                        slope = np.polyfit(np.arange(window), norm, 1)[0]
                        slopes.append(slope)
            return pd.Series(slopes, index=series.index)

        if "obv" in df.columns:
            df["obv_slope"]   = _slope_norm(df["obv"], window=14)
        df["price_slope"] = _slope_norm(close, window=14)

        # ── Relative Volume (vs 50-day avg) ─────────────────────────────────
        df["vol_ma_50"] = vol.rolling(50).mean()
        df["rvol_50"]   = vol / df["vol_ma_50"]   # > 2 = significant institutional activity

        return df

    # ───────────────────── smart money detection (NEW) ───────────────────────

    def _detect_smart_money(self, df: pd.DataFrame) -> Tuple[float, List[str], str]:
        """
        Returns (smart_money_score 0–100, list_of_sm_signals, accumulation_status).
        """
        signals: List[str] = []
        score = 0.0
        last  = df.iloc[-1]
        price = float(last["Close"])

        # Helper
        def val(col):
            return safe_float(last.get(col))

        cmf       = val("cmf")
        mfi       = val("mfi")
        vwap_20   = val("vwap_20")
        obv_slope = val("obv_slope")
        px_slope  = val("price_slope")
        vol_ratio = val("vol_ratio")
        rvol_50   = val("rvol_50")
        ad_line   = val("ad_line")

        # ── 1. OBV divergence ───────────────────────────────────────────────
        obv_divergence = "none"
        if obv_slope is not None and px_slope is not None:
            if obv_slope > 0.001 and px_slope > 0.001:
                signals.append("OBV confirming price (smart money buying)")
                score += 20
                obv_divergence = "confirming"
            elif obv_slope > 0.001 and px_slope < -0.001:
                signals.append("OBV Bullish Divergence — accumulation while price drops")
                score += 25
                obv_divergence = "bullish_div"
            elif obv_slope < -0.001 and px_slope > 0.001:
                signals.append("OBV Bearish Divergence — distribution while price rises")
                score -= 20
                obv_divergence = "bearish_div"
            elif obv_slope < -0.001 and px_slope < -0.001:
                signals.append("OBV confirming downtrend (smart money selling)")
                score -= 15
                obv_divergence = "confirming_down"

        # ── 2. Chaikin Money Flow ────────────────────────────────────────────
        if cmf is not None:
            if cmf > 0.15:
                signals.append(f"Strong CMF buying pressure (CMF={cmf:.2f})")
                score += 18
            elif cmf > 0.05:
                signals.append(f"Moderate CMF buying (CMF={cmf:.2f})")
                score += 10
            elif cmf < -0.15:
                signals.append(f"Strong CMF selling pressure (CMF={cmf:.2f})")
                score -= 18
            elif cmf < -0.05:
                signals.append(f"Moderate CMF selling (CMF={cmf:.2f})")
                score -= 10

        # ── 3. Money Flow Index ──────────────────────────────────────────────
        if mfi is not None:
            if mfi < 20:
                signals.append(f"MFI Oversold ({mfi:.0f}) — institutional accumulation zone")
                score += 15
            elif mfi > 80:
                signals.append(f"MFI Overbought ({mfi:.0f}) — watch for distribution")
                score -= 10
            elif 40 <= mfi <= 60:
                score += 8   # healthy neutral

        # ── 4. VWAP relationship ─────────────────────────────────────────────
        if vwap_20 and vwap_20 > 0:
            vwap_pct = (price - vwap_20) / vwap_20 * 100
            if vwap_pct > 2:
                signals.append(f"Price {vwap_pct:.1f}% above VWAP — institutional tailwind")
                score += 12
            elif vwap_pct > 0:
                signals.append("Price above VWAP — bullish bias")
                score += 6
            elif vwap_pct < -2:
                signals.append(f"Price {abs(vwap_pct):.1f}% below VWAP — institutional headwind")
                score -= 12
            else:
                signals.append("Price below VWAP — bearish bias")
                score -= 6

            # VWAP reclaim / rejection (price crossed VWAP in last 3 bars)
            if len(df) >= 4:
                recent_close  = df["Close"].iloc[-4:-1].values
                recent_vwap   = df["vwap_20"].iloc[-4:-1].values
                was_below_vwap = any(c < v for c, v in zip(recent_close, recent_vwap) if v > 0)
                if was_below_vwap and price > vwap_20:
                    signals.append("VWAP Reclaim — bullish institutional signal")
                    score += 10

        # ── 5. Relative volume (vs 50-day) ───────────────────────────────────
        if rvol_50 is not None:
            if rvol_50 > 3.0:
                signals.append(f"Climax volume ({rvol_50:.1f}× 50-day avg) — potential exhaustion or breakout")
                # Climax is ambiguous — check price action to classify
                if len(df) >= 2:
                    prev_close = float(df["Close"].iloc[-2])
                    if price > prev_close * 1.02:
                        signals.append("Breakout volume confirmed by price surge")
                        score += 15
                    elif price < prev_close * 0.98:
                        signals.append("High-volume reversal — selling climax detected")
                        score -= 10
            elif rvol_50 > 1.5:
                signals.append(f"Elevated relative volume ({rvol_50:.1f}×) — institutional activity")
                score += 8

        # ── 6. Volume breakout confirmation ──────────────────────────────────
        if vol_ratio is not None and len(df) >= 20:
            recent_high_20 = float(df["High"].iloc[-21:-1].max())
            if price > recent_high_20 * 0.99 and vol_ratio > 1.5:
                signals.append("Volume-confirmed breakout above 20-day resistance")
                score += 15
            elif price > recent_high_20 * 0.99 and vol_ratio < 0.8:
                signals.append("Low-volume breakout — potential false breakout, watch closely")
                score -= 5

        # ── 7. A/D Line trend ────────────────────────────────────────────────
        if "ad_line" in df.columns and len(df) >= 10:
            ad_recent = df["ad_line"].iloc[-10:].values
            ad_slope  = np.polyfit(np.arange(10), ad_recent, 1)[0]
            if ad_slope > 0:
                score += 8
            else:
                score -= 5

        # ── Clamp score to 0–100 ─────────────────────────────────────────────
        sm_score = min(100, max(0, 50 + score))   # center at 50

        # ── Accumulation status ──────────────────────────────────────────────
        bullish_flags = (
            (obv_divergence in ("confirming", "bullish_div")) +
            (cmf is not None and cmf > 0.05) +
            (mfi is not None and mfi < 40) +
            (vwap_20 is not None and vwap_20 > 0 and price > vwap_20)
        )
        bearish_flags = (
            (obv_divergence in ("confirming_down", "bearish_div")) +
            (cmf is not None and cmf < -0.05) +
            (mfi is not None and mfi > 70) +
            (vwap_20 is not None and vwap_20 > 0 and price < vwap_20)
        )

        if bullish_flags >= 3:
            accum_status = "Accumulating"
        elif bearish_flags >= 3:
            accum_status = "Distributing"
        elif bullish_flags > bearish_flags:
            accum_status = "Mild Accumulation"
        elif bearish_flags > bullish_flags:
            accum_status = "Mild Distribution"
        else:
            accum_status = "Neutral"

        return sm_score, signals, accum_status

    # ────────────────────── volume profile / POC ─────────────────────────────

    def _volume_profile_poc(self, df: pd.DataFrame, lookback: int = 60, n_bins: int = 20) -> float:
        """
        Point of Control — the price level with the highest cumulative volume
        over the last `lookback` bars. Acts as a key support/resistance reference.
        """
        try:
            recent = df.tail(lookback).copy()
            price_min = float(recent["Low"].min())
            price_max = float(recent["High"].max())
            if price_max <= price_min:
                return float(recent["Close"].iloc[-1])

            bin_edges  = np.linspace(price_min, price_max, n_bins + 1)
            vol_by_bin = np.zeros(n_bins)

            for _, row in recent.iterrows():
                bar_low  = float(row["Low"])
                bar_high = float(row["High"])
                bar_vol  = float(row["Volume"])
                for b in range(n_bins):
                    overlap_lo = max(bin_edges[b],     bar_low)
                    overlap_hi = min(bin_edges[b + 1], bar_high)
                    if overlap_hi > overlap_lo:
                        frac = (overlap_hi - overlap_lo) / max(bar_high - bar_low, 1e-6)
                        vol_by_bin[b] += bar_vol * frac

            poc_idx = int(np.argmax(vol_by_bin))
            poc     = (bin_edges[poc_idx] + bin_edges[poc_idx + 1]) / 2
            return float(poc)
        except Exception:
            return float(df["Close"].iloc[-1])

    # ──────────────────────────── scoring ────────────────────────────────────

    def _calculate_score(self, df: pd.DataFrame) -> Tuple[float, float, dict]:
        """
        Score breakdown (100 pts total):
          Trend:      20 pts
          Cross:      10 pts  (was 15 — reduced to make room for smart money)
          RSI:        20 pts
          MACD:       20 pts
          Volume:     15 pts  (raw volume)
          BB:          5 pts  (was 10)
          Smart Money: 10 pts  (NEW — CMF + OBV divergence contribution)
        """
        score    = 0.0
        momentum = 0.0
        breakdown = {}
        last  = df.iloc[-1]
        price = float(last["Close"])

        # 1. Trend (20 pts) ───────────────────────────────────────────────────
        pts       = 0
        above_50  = price > safe_float(last.get("sma_50"),  0)
        above_200 = price > safe_float(last.get("sma_200"), 0)
        if above_200 and above_50:   pts = 20
        elif above_200:              pts = 13
        elif above_50:               pts = 8
        score += pts
        breakdown["trend"] = pts

        # 2. Golden / Death Cross (10 pts) ────────────────────────────────────
        pts = 0
        sma50_val  = safe_float(last.get("sma_50"))
        sma200_val = safe_float(last.get("sma_200"))
        if sma50_val and sma200_val and len(df) > 5:
            prev50  = safe_float(df["sma_50"].iloc[-5])
            prev200 = safe_float(df["sma_200"].iloc[-5])
            if prev50 and prev200:
                was_below = prev50  < prev200
                is_above  = sma50_val > sma200_val
                if was_below and is_above:    pts = 10   # fresh golden cross
                elif not was_below and not is_above: pts = 0   # death cross
                elif is_above:                pts = 7    # already bullish alignment
                else:                         pts = 2
        score += pts
        breakdown["cross"] = pts

        # 3. RSI (20 pts) ─────────────────────────────────────────────────────
        pts = 0
        rsi = safe_float(last.get("rsi"))
        ob  = self._tech_cfg.get("rsi_overbought", 70)
        os_ = self._tech_cfg.get("rsi_oversold",   30)
        if rsi is not None:
            if 40 <= rsi <= 60:   pts = 20
            elif 30 <= rsi < 40:  pts = 18
            elif 60 < rsi <= 70:  pts = 14
            elif rsi < 30:        pts = 10
            elif rsi > 70:        pts = 5
            if 50 <= rsi <= 65:   momentum += 25
            elif 65 < rsi <= 75:  momentum += 15
            elif rsi > 75:        momentum += 5
        score += pts
        breakdown["rsi"] = pts

        # 4. MACD (20 pts) ────────────────────────────────────────────────────
        pts       = 0
        macd_val  = safe_float(last.get("macd"))
        macd_sig  = safe_float(last.get("macd_signal"))
        macd_diff = safe_float(last.get("macd_diff"))
        if macd_val is not None and macd_sig is not None:
            if macd_val > macd_sig:
                pts = 20 if macd_val > 0 else 14
                momentum += 30
            else:
                pts = 4 if macd_val < 0 else 8
        if macd_diff is not None and len(df) > 2:
            prev_diff = safe_float(df["macd_diff"].iloc[-2])
            if prev_diff is not None and macd_diff > prev_diff and macd_diff > 0:
                pts = min(pts + 3, 20)
                momentum += 10
        score += pts
        breakdown["macd"] = pts

        # 5. Volume (15 pts) — raw vol_ratio ──────────────────────────────────
        pts       = 0
        vol_ratio = safe_float(last.get("vol_ratio"))
        if vol_ratio is not None:
            if vol_ratio > 2.0:    pts = 15
            elif vol_ratio > 1.5:  pts = 12
            elif vol_ratio > 1.1:  pts = 8
            elif vol_ratio > 0.8:  pts = 5
            elif vol_ratio > 0:    pts = 2
            if vol_ratio > 1.5:    momentum += 20
        score += pts
        breakdown["volume"] = pts

        # 6. Bollinger Band position (5 pts) ──────────────────────────────────
        pts      = 0
        bb_upper = safe_float(last.get("bb_upper"))
        bb_lower = safe_float(last.get("bb_lower"))
        bb_mid   = safe_float(last.get("bb_middle"))
        if bb_upper and bb_lower and bb_mid and (bb_upper - bb_lower) > 0:
            bb_pct = (price - bb_lower) / (bb_upper - bb_lower)
            if 0.4 <= bb_pct <= 0.7:   pts = 5
            elif bb_pct > 0.7:         pts = 3
            elif 0.2 <= bb_pct < 0.4:  pts = 4
            elif bb_pct < 0.2:         pts = 2
        score += pts
        breakdown["bollinger"] = pts

        # 7. Smart Money contribution (10 pts) ────────────────────────────────
        pts = 0
        cmf = safe_float(last.get("cmf"))
        mfi = safe_float(last.get("mfi"))
        obv_slope = safe_float(last.get("obv_slope"))
        px_slope  = safe_float(last.get("price_slope"))
        vwap_20   = safe_float(last.get("vwap_20"))

        sm_contrib = 0
        if cmf is not None:
            if cmf > 0.1:   sm_contrib += 3
            elif cmf > 0:   sm_contrib += 1
            elif cmf < -0.1: sm_contrib -= 3
        if mfi is not None:
            if mfi < 25:    sm_contrib += 3   # oversold with smart money loading
            elif mfi > 80:  sm_contrib -= 2
        if obv_slope is not None and px_slope is not None:
            if obv_slope > 0 and px_slope > 0:   sm_contrib += 2   # confirmed
            elif obv_slope > 0 and px_slope <= 0: sm_contrib += 3   # bullish div
            elif obv_slope < 0 and px_slope > 0:  sm_contrib -= 3   # bearish div
        if vwap_20 and vwap_20 > 0 and price > vwap_20:
            sm_contrib += 2

        pts = max(0, min(10, 5 + sm_contrib))
        score += pts
        breakdown["smart_money"] = pts

        score    = min(score, 100.0)
        momentum = min(momentum, 100.0)
        return score, momentum, breakdown

    # ─────────────────────────── signal ──────────────────────────────────────

    def _determine_signal(self, score: float, breakdown: dict) -> str:
        if score >= 65:  return "Bullish"
        if score <= 35:  return "Bearish"
        return "Neutral"

    # ─────────────────────────── levels ──────────────────────────────────────

    def _calculate_levels(self, df: pd.DataFrame) -> Tuple[float, float, float, float]:
        last  = df.iloc[-1]
        price = float(last["Close"])
        atr   = safe_float(last.get("atr")) or (price * 0.02)

        entry_low  = price - 0.3 * atr
        entry_high = price + 0.3 * atr

        # Exit: nearest resistance above price (use recent 30-day high or ATR target)
        recent_high = float(df["High"].tail(30).max())
        exit_target = recent_high if recent_high > price * 1.03 else price + 2.5 * atr

        # Stop: below recent 30-day low or POC, whichever is closer
        recent_low    = float(df["Low"].tail(30).min())
        support_level = max(recent_low, price - 2.0 * atr)
        stop_loss     = support_level - 0.5 * atr

        return entry_low, entry_high, exit_target, stop_loss

    # ─────────────────────────── signal list ─────────────────────────────────

    def _build_signal_list(self, df: pd.DataFrame, breakdown: dict) -> List[str]:
        signals = []
        last  = df.iloc[-1]
        price = float(last["Close"])

        sma50  = safe_float(last.get("sma_50"))
        sma200 = safe_float(last.get("sma_200"))
        rsi    = safe_float(last.get("rsi"))
        macd   = safe_float(last.get("macd"))
        macd_s = safe_float(last.get("macd_signal"))
        vr     = safe_float(last.get("vol_ratio"))
        cmf    = safe_float(last.get("cmf"))
        mfi    = safe_float(last.get("mfi"))
        vwap   = safe_float(last.get("vwap_20"))

        if sma50  and price > sma50:   signals.append("Price above SMA-50")
        if sma200 and price > sma200:  signals.append("Price above SMA-200")
        if sma50 and sma200:
            signals.append(
                "Golden Cross (SMA50 > SMA200)" if sma50 > sma200
                else "Death Cross (SMA50 < SMA200)"
            )
        if rsi:
            if rsi < 30:    signals.append(f"RSI oversold ({rsi:.1f})")
            elif rsi > 70:  signals.append(f"RSI overbought ({rsi:.1f})")
            else:           signals.append(f"RSI neutral ({rsi:.1f})")
        if macd and macd_s:
            signals.append(
                "MACD bullish crossover" if macd > macd_s else "MACD bearish crossover"
            )
        if vr and vr > 1.5:
            signals.append(f"Above-average volume ({vr:.1f}×)")
        if cmf is not None:
            if cmf > 0.1:    signals.append(f"CMF positive (buying pressure {cmf:.2f})")
            elif cmf < -0.1: signals.append(f"CMF negative (selling pressure {cmf:.2f})")
        if mfi is not None:
            if mfi < 25:   signals.append(f"MFI oversold ({mfi:.0f}) — accumulation zone")
            elif mfi > 75: signals.append(f"MFI overbought ({mfi:.0f})")
        if vwap and vwap > 0:
            signals.append(
                f"Above VWAP (${vwap:.2f})" if price > vwap
                else f"Below VWAP (${vwap:.2f})"
            )

        return signals

    # ──────────────────────── indicators dict ────────────────────────────────

    def _extract_indicators(self, df: pd.DataFrame, sm_score: float = 50.0) -> dict:
        last = df.iloc[-1]

        def r(col, dp=2):
            return round(safe_float(last.get(col)) or 0, dp)

        return {
            # Standard
            "sma_20":        r("sma_20"),
            "sma_50":        r("sma_50"),
            "sma_200":       r("sma_200"),
            "ema_12":        r("ema_12"),
            "ema_26":        r("ema_26"),
            "rsi":           r("rsi", 1),
            "macd":          r("macd", 4),
            "macd_signal":   r("macd_signal", 4),
            "macd_diff":     r("macd_diff", 4),
            "bb_upper":      r("bb_upper"),
            "bb_lower":      r("bb_lower"),
            "atr":           r("atr"),
            "vol_ratio":     r("vol_ratio"),
            # Smart money
            "cmf":           r("cmf", 3),
            "mfi":           r("mfi", 1),
            "vwap_20":       r("vwap_20"),
            "obv_slope":     r("obv_slope", 5),
            "rvol_50":       r("rvol_50"),
            "smart_money_score": round(sm_score, 1),
        }

    # ─────────────────────── manual fallback ─────────────────────────────────

    def _manual_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["Close"]
        df["sma_20"]  = close.rolling(20).mean()
        df["sma_50"]  = close.rolling(50).mean()
        df["sma_200"] = close.rolling(200).mean()
        df["ema_12"]  = close.ewm(span=12, adjust=False).mean()
        df["ema_26"]  = close.ewm(span=26, adjust=False).mean()
        delta = close.diff()
        gain  = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta.clip(upper=0)).rolling(14).mean()
        df["rsi"]         = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))
        df["macd"]        = df["ema_12"] - df["ema_26"]
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
        df["macd_diff"]   = df["macd"] - df["macd_signal"]
        hl  = df["High"] - df["Low"]
        hc  = (df["High"] - close.shift()).abs()
        lc  = (df["Low"]  - close.shift()).abs()
        df["atr"]       = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean()
        df["bb_middle"] = close.rolling(20).mean()
        std             = close.rolling(20).std()
        df["bb_upper"]  = df["bb_middle"] + 2 * std
        df["bb_lower"]  = df["bb_middle"] - 2 * std
        df["bb_width"]  = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"]
        df["vol_ma"]    = df["Volume"].rolling(20).mean()
        df["vol_ratio"] = df["Volume"] / df["vol_ma"]
        # OBV manual
        obv = [0]
        for i in range(1, len(df)):
            c, p = float(close.iloc[i]), float(close.iloc[i - 1])
            v    = float(df["Volume"].iloc[i])
            obv.append(obv[-1] + (v if c > p else (-v if c < p else 0)))
        df["obv"] = obv
        return df

    # ─────────────────────── df serialisation ────────────────────────────────

    def _df_to_records(self, df: pd.DataFrame, tail: int = 180) -> list:
        cols = [
            "Open", "High", "Low", "Close", "Volume",
            "sma_20", "sma_50", "sma_200",
            "rsi", "macd", "macd_signal",
            "bb_upper", "bb_lower", "bb_middle",
            "obv", "cmf", "mfi", "vwap_20",   # smart money cols for charting
        ]
        available = [c for c in cols if c in df.columns]
        sub = df[available].tail(tail).copy()
        sub.index.name = "date"
        sub.index = sub.index.astype(str)
        return sub.where(pd.notna(sub), other=None).reset_index().to_dict("records")
