"""
Shared utilities: config loading, caching, logging, safe getters.
"""

import os
import json
import hashlib
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import yaml

logger = logging.getLogger(__name__)

# ─── Config ───────────────────────────────────────────────────────────────────

def load_config(path: Optional[str] = None) -> dict:
    """Load YAML config, falling back to defaults if file is missing."""
    if path is None:
        path = Path(__file__).parent.parent / "config" / "settings.yaml"
    try:
        with open(path, "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logger.warning("settings.yaml not found – using built-in defaults.")
        return _default_config()


def _default_config() -> dict:
    return {
        "weights": {"fundamentals": 0.40, "technicals": 0.35, "sentiment": 0.25},
        "thresholds": {
            "min_combined_score": 40,
            "short_term": {"technical_min": 62, "sentiment_min": 0.05, "combined_min": 55},
            "medium_term": {"technical_min": 45, "fundamental_min": 42, "combined_min": 52},
            "long_term": {"fundamental_min": 62, "combined_min": 50},
        },
        "stock_universe": {"default": "sp500_top100", "custom_tickers": [], "max_stocks": 50},
        "data": {
            "price_history_days": 400,
            "cache_enabled": True,
            "cache_ttl_minutes": 60,
            "request_delay_seconds": 0.3,
        },
        "reddit": {"enabled": False},
        "news": {"enabled": True, "max_articles_per_ticker": 10,
                 "sources": {"yahoo_finance": True, "google_news": True}},
        "output": {"save_results": True, "results_dir": "data/results", "format": "both"},
        "technical": {
            "sma_short": 20, "sma_mid": 50, "sma_long": 200,
            "ema_fast": 12, "ema_slow": 26,
            "rsi_period": 14, "rsi_overbought": 70, "rsi_oversold": 30,
            "macd_fast": 12, "macd_slow": 26, "macd_signal": 9,
            "bb_period": 20, "bb_std": 2, "atr_period": 14, "volume_ma_period": 20,
        },
    }


# ─── File-Based Cache ─────────────────────────────────────────────────────────

CACHE_DIR = Path(__file__).parent.parent / "data" / "cache"


def _cache_path(key: str) -> Path:
    hashed = hashlib.md5(key.encode()).hexdigest()
    return CACHE_DIR / f"{hashed}.json"


def cache_get(key: str, ttl_minutes: int = 60) -> Optional[Any]:
    path = _cache_path(key)
    if not path.exists():
        return None
    try:
        with open(path) as f:
            payload = json.load(f)
        expires = datetime.fromisoformat(payload["expires"])
        if datetime.utcnow() > expires:
            path.unlink(missing_ok=True)
            return None
        return payload["data"]
    except Exception:
        return None


def cache_set(key: str, data: Any, ttl_minutes: int = 60) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _cache_path(key)
    payload = {
        "expires": (datetime.utcnow() + timedelta(minutes=ttl_minutes)).isoformat(),
        "data": data,
    }
    try:
        with open(path, "w") as f:
            json.dump(payload, f, default=str)
    except Exception as e:
        logger.debug(f"Cache write failed: {e}")


def cache_clear():
    if CACHE_DIR.exists():
        for f in CACHE_DIR.glob("*.json"):
            f.unlink(missing_ok=True)


# ─── Results Persistence ──────────────────────────────────────────────────────

def save_results(results: list, config: dict) -> str:
    """Save scan results to disk; returns the file path used."""
    out_dir = Path(config.get("output", {}).get("results_dir", "data/results"))
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    fmt = config.get("output", {}).get("format", "both")

    saved = []
    if fmt in ("json", "both"):
        p = out_dir / f"scan_{ts}.json"
        with open(p, "w") as f:
            json.dump(results, f, indent=2, default=str)
        saved.append(str(p))

    if fmt in ("csv", "both"):
        import pandas as pd
        flat = []
        for r in results:
            flat.append({
                "ticker": r.get("ticker"),
                "category": r.get("category"),
                "combined_score": r.get("combined_score"),
                "confidence": r.get("confidence"),
                "entry": r.get("entry"),
                "exit": r.get("exit"),
                "stop_loss": r.get("stop_loss"),
                "reason": r.get("reason"),
            })
        p = out_dir / f"scan_{ts}.csv"
        pd.DataFrame(flat).to_csv(p, index=False)
        saved.append(str(p))

    return ", ".join(saved)


# ─── Safe Value Helpers ───────────────────────────────────────────────────────

def safe_float(value, default=None) -> Optional[float]:
    try:
        v = float(value)
        return v if v == v else default   # NaN check
    except (TypeError, ValueError):
        return default


def safe_pct(value, default=None) -> Optional[float]:
    v = safe_float(value, default)
    return v


def score_to_label(score: float) -> str:
    if score >= 75:
        return "Strong"
    if score >= 55:
        return "Moderate"
    if score >= 35:
        return "Neutral"
    return "Weak"


def confidence_label(score: float) -> str:
    if score >= 75:
        return "High"
    if score >= 55:
        return "Medium"
    return "Low"


def format_currency(value: Optional[float]) -> str:
    if value is None:
        return "N/A"
    if abs(value) >= 1e12:
        return f"${value / 1e12:.2f}T"
    if abs(value) >= 1e9:
        return f"${value / 1e9:.2f}B"
    if abs(value) >= 1e6:
        return f"${value / 1e6:.2f}M"
    return f"${value:,.2f}"


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
