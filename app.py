"""
app.py  —  Streamlit Dashboard
────────────────────────────────
Run with:   streamlit run app.py
"""

import sys
import os
# Ensure local packages are importable when launched from any directory
sys.path.insert(0, os.path.dirname(__file__))

import json
import time
from pathlib import Path
from typing import List, Optional

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from utils.helpers import load_config, format_currency, cache_clear
from utils.stock_universe import get_universe
from agents.master_agent import MasterAgent

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Stock Research System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .metric-card {
    background: #1e1e2e;
    border-radius: 10px;
    padding: 16px 20px;
    border-left: 4px solid #7c3aed;
  }
  .score-badge {
    font-size: 2rem;
    font-weight: 700;
  }
  .tag-short  { background:#16a34a; color:#fff; border-radius:4px; padding:2px 8px; font-size:.75rem; }
  .tag-medium { background:#d97706; color:#fff; border-radius:4px; padding:2px 8px; font-size:.75rem; }
  .tag-long   { background:#2563eb; color:#fff; border-radius:4px; padding:2px 8px; font-size:.75rem; }
  .flag-box   { background:#7f1d1d; color:#fca5a5; border-radius:4px; padding:3px 8px; font-size:.75rem; }
  h1 { font-size: 1.8rem !important; }
  .stProgress > div > div { background: linear-gradient(90deg, #7c3aed, #2563eb); }
</style>
""", unsafe_allow_html=True)


# ─── Session state defaults ───────────────────────────────────────────────────
if "results"      not in st.session_state: st.session_state.results      = []
if "scan_running" not in st.session_state: st.session_state.scan_running = False
if "last_scan"    not in st.session_state: st.session_state.last_scan    = None
if "config"       not in st.session_state: st.session_state.config       = load_config()
if "sector_data"  not in st.session_state: st.session_state.sector_data  = {}
if "spy_data"     not in st.session_state: st.session_state.spy_data     = {}


# ─── Sidebar ──────────────────────────────────────────────────────────────────

def render_sidebar() -> dict:
    st.sidebar.title("⚙️ Configuration")

    cfg = st.session_state.config

    # ── Market selector ────────────────────────────────────────────────────
    market_choice = st.sidebar.radio(
        "Market",
        ["🇺🇸 US Market", "🇮🇳 India (NSE)"],
        horizontal=True,
    )
    market = "india" if "India" in market_choice else "us"

    # ── Universe ───────────────────────────────────────────────────────────
    st.sidebar.subheader("Stock Universe")

    custom_input = ""
    if market == "india":
        universe = st.sidebar.selectbox(
            "Preset",
            ["nifty50", "nifty100", "nifty200", "nifty500", "custom"],
            index=0,
        )
        if universe == "custom":
            custom_input = st.sidebar.text_area(
                "Custom tickers (NSE symbols, comma-separated)",
                value="RELIANCE, TCS, INFY, HDFCBANK, ICICIBANK",
            )
        max_stocks = st.sidebar.number_input(
            "Max stocks to scan", min_value=5, max_value=500, value=50, step=5
        )
    else:
        universe = st.sidebar.selectbox(
            "Preset",
            ["sp500_top100", "sp500", "nasdaq100", "dow30", "custom"],
            index=0,
        )
        if universe == "custom":
            custom_input = st.sidebar.text_area(
                "Custom tickers (comma-separated)", value="AAPL, MSFT, NVDA, TSLA, AMZN"
            )
        max_stocks = st.sidebar.number_input(
            "Max stocks to scan", min_value=5, max_value=503, value=50, step=5
        )

    # Weights
    st.sidebar.subheader("Agent Weights")
    w_f = st.sidebar.slider("Fundamentals weight", 0.1, 0.8,
                             cfg["weights"].get("fundamentals", 0.40), 0.05)
    w_t = st.sidebar.slider("Technicals weight",   0.1, 0.8,
                             cfg["weights"].get("technicals", 0.35),   0.05)
    w_s = round(max(0.05, 1.0 - w_f - w_t), 2)
    st.sidebar.caption(f"Sentiment weight (auto): **{w_s:.2f}**")

    # Cache
    st.sidebar.subheader("Options")
    use_cache = st.sidebar.checkbox("Use cache", value=True)
    if st.sidebar.button("Clear Cache"):
        cache_clear()
        st.sidebar.success("Cache cleared!")

    # Reddit
    with st.sidebar.expander("Reddit (optional)"):
        reddit_id  = st.text_input("Client ID",     value="")
        reddit_sec = st.text_input("Client Secret", value="", type="password")
        reddit_on  = st.checkbox("Enable Reddit sentiment", value=False)

    # Build updated config
    updated = load_config()
    updated["market"]                        = market
    updated["weights"]["fundamentals"]       = w_f
    updated["weights"]["technicals"]         = w_t
    updated["weights"]["sentiment"]          = w_s
    updated["data"]["cache_enabled"]         = use_cache

    if market == "india":
        if "india" not in updated:
            updated["india"] = {"universe": {}}
        updated["india"]["universe"]["default"]    = universe
        updated["india"]["universe"]["max_stocks"] = int(max_stocks)
        if universe == "custom" and custom_input:
            tickers = [t.strip().upper() for t in custom_input.split(",") if t.strip()]
            updated["india"]["universe"]["custom_tickers"] = tickers
            updated["india"]["universe"]["max_stocks"]     = len(tickers)
    else:
        updated["stock_universe"]["default"]    = universe
        updated["stock_universe"]["max_stocks"] = int(max_stocks)
        if universe == "custom" and custom_input:
            tickers = [t.strip().upper() for t in custom_input.split(",") if t.strip()]
            updated["stock_universe"]["custom_tickers"] = tickers
            updated["stock_universe"]["max_stocks"]     = len(tickers)

    if reddit_on and reddit_id and reddit_sec:
        updated["reddit"]["enabled"]       = True
        updated["reddit"]["client_id"]     = reddit_id
        updated["reddit"]["client_secret"] = reddit_sec

    st.session_state.config = updated
    return updated


# ─── Scan runner ──────────────────────────────────────────────────────────────

def run_scan(config: dict):
    from utils.india_stock_universe import get_india_universe
    market  = config.get("market", "us")
    tickers = get_india_universe(config) if market == "india" else get_universe(config)
    total   = len(tickers)

    st.info(f"Scanning **{total}** stocks. This may take 1–3 minutes …")
    progress_bar  = st.progress(0.0)
    status_text   = st.empty()
    results_store = []

    def progress_cb(i, tot, ticker):
        pct = i / tot
        progress_bar.progress(pct)
        status_text.text(f"[{i}/{tot}] Analysing {ticker} …")

    agent   = MasterAgent(config)
    results = agent.scan(tickers, progress_cb=progress_cb)

    progress_bar.progress(1.0)
    status_text.text("✓ Scan complete!")

    st.session_state.results     = results
    st.session_state.sector_data = agent.sector_data
    st.session_state.spy_data    = agent.spy_data
    st.session_state.last_scan   = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())
    time.sleep(0.5)
    st.rerun()


# ─── Chart builders ───────────────────────────────────────────────────────────

def price_chart(records: list, ticker: str) -> go.Figure:
    if not records:
        return go.Figure()

    df = pd.DataFrame(records)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")

    fig = make_subplots(
        rows=3, cols=1,
        row_heights=[0.55, 0.25, 0.20],
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=(f"{ticker} Price", "RSI", "MACD"),
    )

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df["date"],
        open=df.get("Open", df.get("Close")),
        high=df.get("High", df.get("Close")),
        low=df.get("Low", df.get("Close")),
        close=df["Close"],
        name="Price",
        increasing_line_color="#22c55e",
        decreasing_line_color="#ef4444",
    ), row=1, col=1)

    # SMA lines
    for col, color, name in [
        ("sma_20",  "#facc15", "SMA-20"),
        ("sma_50",  "#60a5fa", "SMA-50"),
        ("sma_200", "#f97316", "SMA-200"),
    ]:
        if col in df.columns:
            fig.add_trace(go.Scatter(
                x=df["date"], y=df[col],
                name=name, line=dict(color=color, width=1.2),
            ), row=1, col=1)

    # Bollinger Bands
    for col, color in [("bb_upper", "rgba(156,163,175,0.4)"), ("bb_lower", "rgba(156,163,175,0.4)")]:
        if col in df.columns:
            fig.add_trace(go.Scatter(
                x=df["date"], y=df[col],
                name=col.replace("_", " ").title(),
                line=dict(color=color, width=0.8, dash="dot"),
            ), row=1, col=1)

    # RSI
    if "rsi" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["date"], y=df["rsi"],
            name="RSI", line=dict(color="#a78bfa", width=1.5),
        ), row=2, col=1)
        fig.add_hline(y=70, line=dict(color="#ef4444", dash="dash", width=0.8), row=2, col=1)
        fig.add_hline(y=30, line=dict(color="#22c55e", dash="dash", width=0.8), row=2, col=1)
        fig.add_hline(y=50, line=dict(color="gray",   dash="dot",  width=0.6), row=2, col=1)

    # MACD
    if "macd" in df.columns and "macd_signal" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["date"], y=df["macd"],
            name="MACD", line=dict(color="#60a5fa", width=1.5),
        ), row=3, col=1)
        fig.add_trace(go.Scatter(
            x=df["date"], y=df["macd_signal"],
            name="Signal", line=dict(color="#f97316", width=1.2),
        ), row=3, col=1)
        if "macd_diff" in df.columns:
            colors = ["#22c55e" if v >= 0 else "#ef4444" for v in df["macd_diff"].fillna(0)]
            fig.add_trace(go.Bar(
                x=df["date"], y=df["macd_diff"],
                name="Histogram", marker_color=colors, opacity=0.6,
            ), row=3, col=1)

    fig.update_layout(
        template="plotly_dark",
        height=550,
        showlegend=True,
        legend=dict(orientation="h", y=1.02, x=0),
        xaxis_rangeslider_visible=False,
        margin=dict(l=0, r=0, t=30, b=0),
    )
    return fig


def sentiment_gauge(score: float) -> go.Figure:
    clamped = max(-1, min(1, score))
    pct     = (clamped + 1) / 2   # 0–1

    color = "#22c55e" if clamped > 0.1 else ("#ef4444" if clamped < -0.1 else "#facc15")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(clamped, 2),
        number={"suffix": "", "font": {"size": 28}},
        gauge={
            "axis": {"range": [-1, 1], "tickwidth": 1, "tickcolor": "#9ca3af"},
            "bar":  {"color": color, "thickness": 0.25},
            "bgcolor": "rgba(0,0,0,0)",
            "steps": [
                {"range": [-1, -0.2], "color": "rgba(239,68,68,0.15)"},
                {"range": [-0.2, 0.2],"color": "rgba(250,204,21,0.15)"},
                {"range": [0.2, 1],   "color": "rgba(34,197,94,0.15)"},
            ],
            "threshold": {"line": {"color": color, "width": 3}, "value": clamped},
        },
        title={"text": "Sentiment Score", "font": {"size": 14}},
    ))
    fig.update_layout(
        template="plotly_dark",
        height=200,
        margin=dict(l=20, r=20, t=30, b=10),
    )
    return fig


def score_radar(r: dict) -> go.Figure:
    categories = ["Fundamentals", "Technicals", "Momentum", "Sentiment"]
    sent_norm  = (r.get("sentiment_score", 0) + 1) / 2 * 100
    values = [
        r.get("fundamental_score", 0),
        r.get("technical_score",   0),
        r.get("momentum_score",    0),
        sent_norm,
    ]
    fig = go.Figure(go.Scatterpolar(
        r=values + [values[0]],
        theta=categories + [categories[0]],
        fill="toself",
        fillcolor="rgba(124,58,237,0.3)",
        line=dict(color="#7c3aed", width=2),
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        template="plotly_dark",
        height=280,
        margin=dict(l=20, r=20, t=20, b=20),
        showlegend=False,
    )
    return fig


# ─── Results rendering ────────────────────────────────────────────────────────

def category_color(cat: str) -> str:
    return {"Short-Term": "green", "Medium-Term": "orange", "Long-Term": "blue"}.get(cat, "gray")


def render_stock_card(r: dict, expanded: bool = False, key_prefix: str = ""):
    ticker   = r.get("ticker", "")
    name     = r.get("company_name", ticker)
    score    = r.get("combined_score", 0)
    cat      = r.get("category", "")
    conf     = r.get("confidence", "")
    signal   = r.get("signal", "Neutral")
    entry    = r.get("entry", "N/A")
    exit_t   = r.get("exit", "N/A")
    stop     = r.get("stop_loss", "N/A")
    reason   = r.get("reason", "")
    val      = r.get("valuation", "")
    price    = r.get("current_price", 0)
    flags    = r.get("risk_flags", [])

    signal_icon = {"Bullish": "🟢", "Bearish": "🔴", "Neutral": "🟡"}.get(signal, "⚪")
    cat_icon    = {"Short-Term": "⚡", "Medium-Term": "📈", "Long-Term": "🏛️"}.get(cat, "")

    with st.expander(
        f"{cat_icon} **{ticker}** — {name[:30]}  |  Score: {score:.0f}/100  |  {signal_icon} {signal}",
        expanded=expanded,
    ):
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Combined Score", f"{score:.1f}", delta=None)
        col2.metric("Confidence",     conf)
        col3.metric("Signal",         signal)
        col4.metric("Current Price",  f"${price:.2f}" if price else "N/A")

        st.markdown(f"**Category:** {cat}  |  **Valuation:** {val}")
        st.caption(f"_{reason}_")

        if flags:
            for flag in flags:
                st.warning(f"⚠ {flag}", icon=None)

        # Trade levels
        lc1, lc2, lc3 = st.columns(3)
        lc1.metric("Entry Range",  entry)
        lc2.metric("Exit Target",  exit_t)
        lc3.metric("Stop Loss",    stop)

        # Tabs: Chart | Scores | Fundamentals | Sentiment | Smart Money | News
        tab_chart, tab_scores, tab_fund, tab_sent, tab_sm, tab_news = st.tabs(
            ["📊 Chart", "🎯 Scores", "📋 Fundamentals", "💬 Sentiment", "🏦 Smart Money", "📰 News"]
        )

        with tab_chart:
            history = r.get("price_history", [])
            if history:
                fig = price_chart(history, ticker)
                st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}chart_{ticker}")
            else:
                st.info("No price history available.")

        with tab_scores:
            sc1, sc2 = st.columns(2)
            with sc1:
                fig = score_radar(r)
                st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}radar_{ticker}")
            with sc2:
                st.markdown("**Sub-Agent Scores**")
                score_data = {
                    "Agent":  ["Fundamentals", "Technicals", "Momentum", "Sentiment (norm)"],
                    "Score":  [
                        r.get("fundamental_score", 0),
                        r.get("technical_score",   0),
                        r.get("momentum_score",    0),
                        round((r.get("sentiment_score", 0) + 1) / 2 * 100, 1),
                    ],
                }
                bar_fig = px.bar(
                    pd.DataFrame(score_data), x="Agent", y="Score",
                    color="Score",
                    color_continuous_scale=["#ef4444", "#facc15", "#22c55e"],
                    range_color=[0, 100], template="plotly_dark", height=220,
                )
                bar_fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), showlegend=False)
                st.plotly_chart(bar_fig, use_container_width=True, key=f"{key_prefix}bar_{ticker}")

            # ── Advanced scoring waterfall ─────────────────────────────
            st.markdown("**Advanced Score Breakdown**")
            base   = r.get("base_score", r.get("combined_score", 0))
            ctx    = r.get("context_multiplier", 1.0)
            boost  = r.get("opportunity_boost", 0)
            pen    = r.get("risk_penalty", 0)
            final  = r.get("final_score", r.get("combined_score", 0))
            adj    = round(base * ctx, 1)

            wc1, wc2, wc3, wc4, wc5 = st.columns(5)
            wc1.metric("Base Score",       f"{base:.1f}")
            wc2.metric("× Context",        f"{ctx:.3f}",
                       delta=f"{'↑' if ctx > 1 else '↓'} sect/regime/RS")
            wc3.metric("Adjusted",         f"{adj:.1f}")
            wc4.metric("+ Boost / − Risk", f"+{boost:.1f} / −{pen:.1f}")
            wc5.metric("Final Score",      f"{final:.1f}",
                       delta=f"{final - base:+.1f} vs base")

            # Boost / penalty detail
            bd_col1, bd_col2 = st.columns(2)
            with bd_col1:
                boost_bd = r.get("boost_breakdown", {})
                if boost_bd:
                    st.caption("**Opportunity Boost detail**")
                    for k, v in boost_bd.items():
                        if v:
                            st.caption(f"  +{v:.1f}  {k.replace('_',' ').title()}")
            with bd_col2:
                pen_bd = r.get("penalty_breakdown", {})
                if pen_bd:
                    st.caption("**Risk Penalty detail**")
                    for k, v in pen_bd.items():
                        if v:
                            st.caption(f"  −{v:.1f}  {k.replace('_',' ').title()}")

            # Context detail
            regime_icon = {"bullish": "🟢", "sideways": "🟡", "bearish": "🔴"}.get(
                r.get("spy_regime", "sideways"), "⚪")
            ss_icon = {"Strong": "🟢", "Neutral": "🟡", "Weak": "🔴"}.get(
                r.get("sector_strength", "Neutral"), "⚪")
            st.caption(
                f"{regime_icon} Market: **{r.get('spy_regime','?')}**  |  "
                f"{ss_icon} Sector ({r.get('sector','?')}): **{r.get('sector_strength','?')}**  |  "
                f"RS vs SPY: **{r.get('rel_strength_vs_spy', 0):+.1f}%**  |  "
                f"Confidence: **{r.get('confidence_num', 0):.0%}** ({r.get('confidence','?')})"
            )

            st.markdown("**Technical Signals**")
            sigs = r.get("technical_signals", [])
            if sigs:
                for s in sigs:
                    st.markdown(f"- {s}")
            else:
                st.caption("No signals available.")

        with tab_fund:
            metrics = r.get("metrics", {})
            if metrics:
                cols = st.columns(3)
                items = [
                    ("P/E Ratio",      metrics.get("pe_ratio")),
                    ("Forward P/E",    metrics.get("forward_pe")),
                    ("PEG Ratio",      metrics.get("peg_ratio")),
                    ("P/B Ratio",      metrics.get("pb_ratio")),
                    ("P/S Ratio",      metrics.get("ps_ratio")),
                    ("ROE",            f"{metrics.get('roe', 0)*100:.1f}%" if metrics.get("roe") else "N/A"),
                    ("ROA",            f"{metrics.get('roa', 0)*100:.1f}%" if metrics.get("roa") else "N/A"),
                    ("Profit Margin",  f"{metrics.get('profit_margin', 0)*100:.1f}%" if metrics.get("profit_margin") else "N/A"),
                    ("Revenue Growth", f"{metrics.get('revenue_growth', 0)*100:.1f}%" if metrics.get("revenue_growth") else "N/A"),
                    ("EPS Growth",     f"{metrics.get('earnings_growth', 0)*100:.1f}%" if metrics.get("earnings_growth") else "N/A"),
                    ("Debt/Equity",    metrics.get("debt_to_equity")),
                    ("Market Cap",     format_currency(metrics.get("market_cap"))),
                    ("Sector",         metrics.get("sector", "N/A")),
                    ("Industry",       metrics.get("industry", "N/A")),
                    ("52W High",       f"${metrics.get('52w_high', 0):.2f}" if metrics.get("52w_high") else "N/A"),
                    ("52W Low",        f"${metrics.get('52w_low', 0):.2f}" if metrics.get("52w_low") else "N/A"),
                ]
                for idx, (label, val) in enumerate(items):
                    cols[idx % 3].metric(label, val if val is not None else "N/A")
            else:
                st.info("Fundamental data unavailable.")

        with tab_sent:
            sc1, sc2 = st.columns([1, 2])
            with sc1:
                sent_score = r.get("sentiment_score", 0)
                fig = sentiment_gauge(sent_score)
                st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}gauge_{ticker}")
                buzz = r.get("buzz_level", "Low")
                st.metric("Buzz Level", buzz)
            with sc2:
                headlines = r.get("top_headlines", [])
                if headlines:
                    st.markdown("**Sentiment Headlines**")
                    for h in headlines:
                        st.markdown(f"- {h}")
                else:
                    st.caption("No sentiment headlines available.")

        with tab_sm:
            inds = r.get("indicators", {})

            # ── Top metrics row ───────────────────────────────────────────
            sm1, sm2, sm3, sm4, sm5 = st.columns(5)
            sm_score_val  = inds.get("smart_money_score", r.get("smart_money_score", 50))
            accum_status  = r.get("accumulation_status", "Neutral")
            cmf_val       = inds.get("cmf", 0)
            mfi_val       = inds.get("mfi", 50)
            vwap_val      = inds.get("vwap_20", 0)
            curr_price    = r.get("current_price", 0)
            poc_val       = r.get("poc_price", 0)

            accum_color = {"Accumulating": "normal", "Mild Accumulation": "normal",
                           "Distributing": "inverse", "Mild Distribution": "inverse"}.get(accum_status, "off")
            sm1.metric("Smart Money Score", f"{sm_score_val:.0f}/100")
            sm2.metric("Status",            accum_status, delta_color=accum_color)
            sm3.metric("CMF",               f"{cmf_val:.3f}",
                       delta="Buying" if cmf_val > 0.05 else ("Selling" if cmf_val < -0.05 else "Neutral"),
                       delta_color="normal" if cmf_val > 0.05 else ("inverse" if cmf_val < -0.05 else "off"))
            sm4.metric("MFI",               f"{mfi_val:.0f}",
                       delta="Oversold" if mfi_val < 25 else ("Overbought" if mfi_val > 75 else "Normal"),
                       delta_color="normal" if mfi_val < 25 else ("inverse" if mfi_val > 75 else "off"))
            sm5.metric("VWAP (20d)",        f"${vwap_val:.2f}" if vwap_val else "N/A",
                       delta="Above" if curr_price > vwap_val > 0 else "Below",
                       delta_color="normal" if curr_price > vwap_val > 0 else "inverse")

            # ── POC ───────────────────────────────────────────────────────
            if poc_val:
                poc_diff_pct = (curr_price - poc_val) / poc_val * 100 if poc_val else 0
                st.caption(
                    f"**Point of Control (POC):** ${poc_val:.2f}  —  "
                    f"price is **{poc_diff_pct:+.1f}%** vs highest-volume price level"
                )

            # ── Smart money signals ───────────────────────────────────────
            sm_signals = r.get("smart_money_signals", [])
            if sm_signals:
                st.markdown("**Detected Smart Money Events**")
                for sig in sm_signals:
                    icon = "🟢" if any(w in sig.lower() for w in
                        ["accumulation", "buying", "bullish", "confirmed", "reclaim", "oversold"]) \
                           else ("🔴" if any(w in sig.lower() for w in
                        ["distribution", "selling", "bearish", "rejection", "overbought", "climax"]) \
                           else "🟡")
                    st.markdown(f"{icon} {sig}")
            else:
                st.caption("No smart money events detected for this scan.")

            # ── Indicator reference table ─────────────────────────────────
            st.markdown("**Smart Money Indicator Reference**")
            rvol = inds.get("rvol_50", 0)
            obv_slope = inds.get("obv_slope", 0)
            ref_data = {
                "Indicator": ["CMF (20)", "MFI (14)", "VWAP (20d)", "Rel. Volume vs 50d", "OBV Slope"],
                "Value":     [f"{cmf_val:.3f}", f"{mfi_val:.1f}",
                              f"${vwap_val:.2f}" if vwap_val else "N/A",
                              f"{rvol:.2f}×",  f"{obv_slope:.5f}"],
                "Signal":    [
                    "Buying" if cmf_val > 0.1 else ("Neutral" if cmf_val > -0.1 else "Selling"),
                    "Oversold" if mfi_val < 25 else ("Overbought" if mfi_val > 75 else "Normal"),
                    "Above" if curr_price > vwap_val > 0 else "Below",
                    "High" if rvol > 2 else ("Normal" if rvol > 0.8 else "Low"),
                    "Rising" if obv_slope > 0 else "Falling",
                ],
                "Interpretation": [
                    ">0.1 = institutions buying, <-0.1 = selling",
                    "<25 = accumulation zone, >75 = distribution risk",
                    "Above = institutional tailwind, Below = headwind",
                    ">2× = unusual institutional activity",
                    "Rising + price up = confirmed. Rising + price down = bullish divergence",
                ],
            }
            st.dataframe(pd.DataFrame(ref_data), use_container_width=True, hide_index=True)

        with tab_news:
            nc1, nc2, nc3 = st.columns(3)
            nc1.metric("News Impact Score", f"{r.get('news_impact_score', 50):.0f}/100")
            nc2.metric("Buzz",              r.get("news_buzz", "Low"))
            nc3.metric("Articles Found",    r.get("article_count", 0) if "article_count" in r
                       else len(r.get("news_headlines", [])))

            catalyst = r.get("news_catalyst", "")
            if catalyst and "No" not in catalyst:
                st.success(f"**Catalyst detected:** {catalyst}")
            else:
                st.info("No major catalyst detected in recent news.")

            cats = r.get("catalysts_found", [])
            if cats:
                st.markdown("**Catalyst types:** " + " · ".join(cats))

            st.markdown("**News Sentiment**")
            ns = r.get("news_sentiment", 0)
            fig2 = sentiment_gauge(ns)
            st.plotly_chart(fig2, use_container_width=True, key=f"{key_prefix}newsgauge_{ticker}")

            headlines2 = r.get("news_headlines", [])
            if headlines2:
                st.markdown("**Top News Headlines**")
                for h in headlines2:
                    st.markdown(f"- {h}")
            else:
                st.caption("No news headlines available.")


def render_results_tab(results: list, category: Optional[str] = None):
    tab_key = (category or "all").lower().replace("-", "") + "_"
    if category:
        picks = [r for r in results if r.get("category") == category]
    else:
        picks = results

    if not picks:
        st.info(f"No picks found for this category in the last scan.")
        return

    # Summary row
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Picks",   len(picks))
    avg_score = sum(r.get("combined_score", 0) for r in picks) / len(picks)
    m2.metric("Avg Score",     f"{avg_score:.1f}")
    high_conf = sum(1 for r in picks if r.get("confidence") == "High")
    m3.metric("High Confidence", high_conf)
    bullish = sum(1 for r in picks if r.get("signal") == "Bullish")
    m4.metric("Bullish Signals", bullish)

    st.markdown("---")

    # Quick summary table
    table_data = []
    for r in picks:
        table_data.append({
            "Ticker":    r.get("ticker"),
            "Score":     r.get("combined_score"),
            "Category":  r.get("category"),
            "Signal":    r.get("signal"),
            "Confidence":r.get("confidence"),
            "Entry":     r.get("entry"),
            "Exit":      r.get("exit"),
            "Stop Loss": r.get("stop_loss"),
            "Valuation": r.get("valuation"),
        })

    df_table = pd.DataFrame(table_data)
    st.dataframe(
        df_table,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Score":     st.column_config.ProgressColumn("Score", min_value=0, max_value=100, format="%.1f"),
            "Ticker":    st.column_config.TextColumn("Ticker",  width="small"),
            "Category":  st.column_config.TextColumn("Category", width="small"),
            "Signal":    st.column_config.TextColumn("Signal",   width="small"),
        },
    )

    st.markdown("### Detailed Picks")
    for r in picks:
        render_stock_card(r, expanded=False, key_prefix=tab_key)


# ─── Main layout ──────────────────────────────────────────────────────────────

def main():
    config = render_sidebar()

    # Header — shows active market
    _mkt = config.get("market", "us")
    _flag = "🇮🇳" if _mkt == "india" else "🇺🇸"
    _mkt_label = "India (NSE)" if _mkt == "india" else "US Markets"
    st.title(f"📈 AI Multi-Agent Stock Research — {_flag} {_mkt_label}")
    st.caption("Powered by yfinance · VADER · ta · Free data only")

    # Last scan info
    if st.session_state.last_scan:
        st.caption(f"Last scan: {st.session_state.last_scan}  |  "
                   f"{len(st.session_state.results)} stocks analysed")

    # Run Scan button (prominent)
    col_btn, col_info = st.columns([1, 4])
    with col_btn:
        if st.button("🚀 Run Scan", type="primary", use_container_width=True,
                     disabled=st.session_state.scan_running):
            st.session_state.scan_running = True
            with st.spinner("Running analysis …"):
                run_scan(config)
            st.session_state.scan_running = False

    with col_info:
        from utils.india_stock_universe import get_india_universe
        _market = config.get("market", "us")
        if _market == "india":
            tickers_preview  = get_india_universe(config)
            _univ_label      = config.get("india", {}).get("universe", {}).get("default", "nifty50")
            _benchmark_label = "🇮🇳 NIFTY 50"
        else:
            tickers_preview  = get_universe(config)
            _univ_label      = config["stock_universe"]["default"]
            _benchmark_label = "🇺🇸 S&P 500 / SPY"
        st.info(
            f"Market: **{_benchmark_label}**  |  "
            f"Universe: **{_univ_label}**  |  "
            f"Stocks: **{len(tickers_preview)}**  |  "
            f"Weights: F={config['weights']['fundamentals']:.0%} "
            f"T={config['weights']['technicals']:.0%} "
            f"S={config['weights']['sentiment']:.0%}"
        )

    results = st.session_state.results

    if not results:
        st.markdown("""
        ---
        ### How it works
        1. Click **Run Scan** to start the multi-agent analysis
        2. Five specialised agents run in parallel:
           - **Fundamental Agent** — revenue growth, margins, P/E, PEG, ROE
           - **Technical Agent** — SMA/EMA, RSI, MACD, Bollinger Bands, ATR, volume
           - **Sentiment Agent** — Reddit + News RSS with VADER NLP
           - **News Agent** — catalyst detection (earnings, upgrades, M&A, guidance)
           - **Sector Agent** — 11 sector ETFs vs SPY, market regime via SPY 200-SMA
        3. A **Master Agent** runs the advanced scoring engine:
           `Final = (Base × Context Multiplier) + Opportunity Boost − Risk Penalty`
        4. Results are categorised into **Short-Term**, **Medium-Term**, and **Long-Term** picks
        5. Explore the **Sector Flow** and **News & Catalysts** tabs for market context
        ---
        """)
        return

    # Category tabs
    short  = [r for r in results if r.get("category") == "Short-Term"]
    medium = [r for r in results if r.get("category") == "Medium-Term"]
    long_  = [r for r in results if r.get("category") == "Long-Term"]
    all_   = results

    sector_data = st.session_state.sector_data
    spy_data    = st.session_state.spy_data

    tab_st, tab_mt, tab_lt, tab_all, tab_sec, tab_news, tab_bt, tab_perf = st.tabs([
        f"⚡ Short-Term ({len(short)})",
        f"📈 Medium-Term ({len(medium)})",
        f"🏛️ Long-Term ({len(long_)})",
        f"🔍 All Results ({len(all_)})",
        "🏭 Sector Flow",
        "📰 News & Catalysts",
        "🧪 Backtest",
        "📊 Performance",
    ])

    with tab_st:
        render_results_tab(results, "Short-Term")
    with tab_mt:
        render_results_tab(results, "Medium-Term")
    with tab_lt:
        render_results_tab(results, "Long-Term")
    with tab_all:
        render_results_tab(results, None)
    with tab_sec:
        render_sector_tab(sector_data, spy_data)
    with tab_news:
        render_news_tab(results)
    with tab_bt:
        render_backtest_tab()
    with tab_perf:
        render_performance_tab()


def render_sector_tab(sector_data: dict, spy_data: dict):
    if not sector_data:
        st.info("No sector data yet — run a scan first.")
        return

    # ── SPY regime banner ─────────────────────────────────────────────
    regime = spy_data.get("regime", "unknown")
    price  = spy_data.get("spy_price", 0)
    s200   = spy_data.get("sma200", 0)
    pct    = spy_data.get("pct_vs_sma200", 0)
    r3m    = spy_data.get("return_3m", 0)
    r1m    = spy_data.get("return_1m", 0)
    color  = {"bullish": "🟢", "sideways": "🟡", "bearish": "🔴"}.get(regime, "⚪")

    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("SPY Price",      f"${price:.2f}")
    mc2.metric("vs SMA-200",     f"{pct:+.1f}%",
               delta=regime.capitalize(), delta_color="normal")
    mc3.metric("SPY 1M Return",  f"{r1m:+.1f}%")
    mc4.metric("SPY 3M Return",  f"{r3m:+.1f}%")

    st.markdown(f"**Market Regime:** {color} **{regime.upper()}**")
    st.markdown("---")

    # ── Rotation insight ──────────────────────────────────────────────
    from agents.sector_agent import SectorAgent as _SA
    insight = _SA().rotation_insight(sector_data)
    st.info(f"💡 {insight}")

    # ── Sector heatmap (horizontal bar sorted by relative strength) ───
    rows = list(sector_data.values())
    rows.sort(key=lambda x: x.get("relative_strength", 0), reverse=True)

    df_s = pd.DataFrame(rows)
    color_map = {"Strong": "#22c55e", "Neutral": "#facc15", "Weak": "#ef4444"}
    bar_colors = [color_map.get(s, "#9ca3af") for s in df_s["strength"]]

    fig = go.Figure(go.Bar(
        x=df_s["relative_strength"],
        y=df_s["sector"],
        orientation="h",
        marker_color=bar_colors,
        text=[f"{v:+.1f}%" for v in df_s["relative_strength"]],
        textposition="outside",
    ))
    fig.update_layout(
        title="Sector Relative Strength vs SPY (1M)",
        template="plotly_dark",
        height=420,
        xaxis_title="Relative Return vs SPY (%)",
        yaxis=dict(autorange="reversed"),
        margin=dict(l=0, r=60, t=40, b=0),
    )
    st.plotly_chart(fig, use_container_width=True, key="sector_heatmap")

    # ── Sector return comparison (1D / 5D / 1M) ───────────────────────
    st.markdown("### Sector Return Comparison")
    df_ret = df_s[["sector", "return_1d", "return_5d", "return_1m",
                   "relative_strength", "strength", "volume_change"]].copy()
    df_ret.columns = ["Sector", "1D %", "5D %", "1M %",
                      "Rel Strength %", "Strength", "Vol Change"]

    st.dataframe(
        df_ret,
        use_container_width=True,
        hide_index=True,
        column_config={
            "1D %":            st.column_config.NumberColumn(format="%.2f"),
            "5D %":            st.column_config.NumberColumn(format="%.2f"),
            "1M %":            st.column_config.NumberColumn(format="%.2f"),
            "Rel Strength %":  st.column_config.NumberColumn(format="%+.2f"),
            "Vol Change":      st.column_config.NumberColumn(format="%.2f×"),
        },
    )

    # ── Top / Weak sector summary ─────────────────────────────────────
    col_top, col_weak = st.columns(2)
    with col_top:
        st.markdown("#### 🟢 Leading Sectors")
        top = [r for r in rows if r["strength"] == "Strong"]
        if top:
            for s in top:
                st.success(f"**{s['sector']}** — RS: {s['relative_strength']:+.1f}%  |  "
                           f"1M: {s['return_1m']:+.1f}%")
        else:
            st.caption("No strong sectors right now.")
    with col_weak:
        st.markdown("#### 🔴 Lagging Sectors")
        weak = [r for r in rows if r["strength"] == "Weak"]
        if weak:
            for s in weak:
                st.error(f"**{s['sector']}** — RS: {s['relative_strength']:+.1f}%  |  "
                         f"1M: {s['return_1m']:+.1f}%")
        else:
            st.caption("No weak sectors right now.")


def render_news_tab(results: list):
    if not results:
        st.info("No scan results yet.")
        return

    st.subheader("News Intelligence — All Scanned Stocks")

    # ── Stocks to Watch Tomorrow ───────────────────────────────────────
    st.markdown("### 👀 Stocks to Watch Tomorrow")
    watch = sorted(
        [r for r in results if r.get("news_impact_score", 50) > 60
         and r.get("news_catalyst", "") and "No" not in r.get("news_catalyst", "")],
        key=lambda r: r.get("news_impact_score", 0),
        reverse=True,
    )[:8]

    if watch:
        for w in watch:
            cat_icon = {"Short-Term": "⚡", "Medium-Term": "📈",
                        "Long-Term": "🏛️"}.get(w.get("category", ""), "")
            impact   = w.get("news_impact_score", 50)
            catalyst = w.get("news_catalyst", "")
            col_a, col_b, col_c, col_d = st.columns([1, 2, 3, 1])
            col_a.markdown(f"**{w['ticker']}** {cat_icon}")
            col_b.markdown(f"Impact: **{impact:.0f}/100**")
            col_c.markdown(f"_{catalyst}_")
            col_d.markdown(f"${w.get('current_price', 0):.2f}")
    else:
        st.caption("No high-impact news catalysts detected in this scan.")

    st.markdown("---")

    # ── Full news table ────────────────────────────────────────────────
    st.markdown("### News Summary Table")
    news_rows = []
    for r in results:
        news_rows.append({
            "Ticker":       r.get("ticker"),
            "Category":     r.get("category"),
            "Impact Score": r.get("news_impact_score", 50),
            "News Buzz":    r.get("news_buzz", "Low"),
            "Catalyst":     r.get("news_catalyst", ""),
            "Sentiment":    r.get("news_sentiment", 0),
        })
    df_news = pd.DataFrame(news_rows).sort_values("Impact Score", ascending=False)
    st.dataframe(
        df_news,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Impact Score": st.column_config.ProgressColumn(
                "Impact Score", min_value=0, max_value=100, format="%.0f"),
            "Sentiment":    st.column_config.NumberColumn(format="%.3f"),
        },
    )

    # ── Per-ticker news breakdown ──────────────────────────────────────
    st.markdown("### Per-Ticker Headlines")
    for r in sorted(results, key=lambda x: x.get("news_impact_score", 0), reverse=True)[:15]:
        ticker   = r.get("ticker", "")
        impact   = r.get("news_impact_score", 50)
        catalyst = r.get("news_catalyst", "No data")
        buzz     = r.get("news_buzz", "Low")
        headlines = r.get("news_headlines", r.get("top_headlines", []))

        with st.expander(f"**{ticker}** — Impact: {impact:.0f}  |  {catalyst}  |  Buzz: {buzz}"):
            if headlines:
                for h in headlines:
                    st.markdown(f"- {h}")
            else:
                st.caption("No headlines.")


def render_performance_tab():
    st.subheader("📊 Performance Tracking & Evaluation")
    st.caption("Track and evaluate past scan predictions against real market outcomes.")

    from tracking.performance_tracker import PerformanceTracker
    from utils.helpers import load_config as _lc

    cfg         = st.session_state.config
    results_dir = cfg.get("output", {}).get("results_dir", "data/results")
    tracking_dir = str(Path(results_dir).parent / "tracking")
    tracker     = PerformanceTracker(storage_dir=tracking_dir)

    # ── Evaluate pending button ────────────────────────────────────────
    col_ev, col_info = st.columns([1, 3])
    with col_ev:
        if st.button("🔄 Evaluate Pending", type="primary"):
            with st.spinner("Fetching prices and evaluating outcomes …"):
                changed = tracker.evaluate_pending(max_eval=200)
            st.success(f"Evaluated: **{changed}** outcome(s) resolved.")
    with col_info:
        st.info(
            "Click **Evaluate Pending** to fetch current prices for all IN_PROGRESS "
            "predictions and determine Win / Loss / Expired outcomes."
        )

    st.markdown("---")

    # ── Overview metrics ───────────────────────────────────────────────
    metrics = tracker.compute_metrics()

    if metrics.get("total", 0) == 0:
        st.info("No predictions stored yet. Run a scan first — predictions are auto-saved after every scan.")
        return

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Total Tracked",   metrics.get("total", 0))
    m2.metric("Completed",       metrics.get("completed", 0))
    m3.metric("In Progress",     metrics.get("in_progress", 0))
    m4.metric("Win Rate",        f"{metrics.get('win_rate', 0):.1%}")
    m5.metric("Avg Return",      f"{metrics.get('avg_return', 0):+.2f}%")
    m6.metric("Risk/Reward",     f"{metrics.get('risk_reward', 0):.2f}×")

    if metrics.get("message"):
        st.info(metrics["message"])
        return

    # Streak
    streak      = metrics.get("streak", 0)
    streak_type = metrics.get("streak_type", "")
    if streak and streak_type:
        streak_color = "🟢" if streak_type == "win" else "🔴"
        st.caption(f"{streak_color} Current streak: **{streak} {streak_type}(s)**")

    st.markdown("---")

    # ── Win rate by category bar chart ────────────────────────────────
    by_cat = metrics.get("by_category", {})
    if by_cat:
        st.markdown("### Win Rate by Category")
        cat_rows = []
        for cat, stats in by_cat.items():
            cat_rows.append({
                "Category":  cat,
                "Win Rate":  round(stats.get("win_rate", 0) * 100, 1),
                "Avg Return": round(stats.get("avg_return", 0), 2),
                "Count":     stats.get("count", 0),
            })
        df_cat = pd.DataFrame(cat_rows)

        col_bar, col_tbl = st.columns([2, 1])
        with col_bar:
            fig_cat = px.bar(
                df_cat, x="Category", y="Win Rate",
                color="Win Rate",
                color_continuous_scale=["#ef4444", "#facc15", "#22c55e"],
                range_color=[0, 100],
                text="Win Rate",
                template="plotly_dark",
                height=280,
            )
            fig_cat.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig_cat.update_layout(margin=dict(l=0, r=0, t=10, b=0), showlegend=False,
                                  yaxis_range=[0, 110])
            st.plotly_chart(fig_cat, use_container_width=True, key="perf_cat_bar")
        with col_tbl:
            st.dataframe(df_cat, use_container_width=True, hide_index=True,
                         column_config={
                             "Win Rate":   st.column_config.NumberColumn("Win Rate %", format="%.1f"),
                             "Avg Return": st.column_config.NumberColumn("Avg Ret %",  format="%+.2f"),
                         })

    st.markdown("---")

    # ── Returns distribution histogram ────────────────────────────────
    df_preds = tracker.get_predictions_df()
    completed_df = df_preds[df_preds["outcome"].isin(["SUCCESS", "FAILURE", "EXPIRED"])].copy()
    if not completed_df.empty and "return_pct" in completed_df.columns:
        st.markdown("### Returns Distribution")
        returns_clean = completed_df["return_pct"].dropna()
        if not returns_clean.empty:
            fig_hist = px.histogram(
                returns_clean,
                nbins=30,
                color_discrete_sequence=["#7c3aed"],
                template="plotly_dark",
                height=260,
                labels={"value": "Return (%)"},
            )
            fig_hist.add_vline(x=0, line_color="#ef4444", line_dash="dash", line_width=1.5)
            fig_hist.add_vline(x=returns_clean.mean(), line_color="#22c55e",
                               line_dash="dot", line_width=1.5,
                               annotation_text=f"Mean: {returns_clean.mean():.1f}%",
                               annotation_position="top right")
            fig_hist.update_layout(margin=dict(l=0, r=0, t=10, b=0), showlegend=False)
            st.plotly_chart(fig_hist, use_container_width=True, key="perf_hist")

    # ── Equity curve ──────────────────────────────────────────────────
    equity_df = tracker.get_equity_curve()
    if not equity_df.empty:
        st.markdown("### Equity Curve (compounded $100 base)")
        fig_eq = px.line(
            equity_df, x="date", y="equity",
            color_discrete_sequence=["#22c55e"],
            template="plotly_dark",
            height=260,
            labels={"equity": "Portfolio Value ($)", "date": "Date"},
        )
        fig_eq.add_hline(y=100, line_color="#9ca3af", line_dash="dash", line_width=1)
        fig_eq.update_layout(margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig_eq, use_container_width=True, key="perf_equity")

    st.markdown("---")

    # ── Signal effectiveness ───────────────────────────────────────────
    st.markdown("### Signal Effectiveness")
    se = tracker.signal_effectiveness()
    if "message" in se:
        st.info(se["message"])
    else:
        se_tabs = st.tabs(["Sentiment", "Sector Strength", "Score Buckets", "Smart Money", "Market Regime"])
        _se_keys = ["sentiment", "sector_strength", "score_bucket", "smart_money", "regime"]
        for se_tab, se_key in zip(se_tabs, _se_keys):
            with se_tab:
                bucket_data = se.get(se_key, {})
                if not bucket_data:
                    st.caption("Insufficient data.")
                    continue
                rows = []
                for label, stats in bucket_data.items():
                    if stats.get("count", 0) > 0:
                        rows.append({
                            "Bucket":     label,
                            "Count":      stats.get("count", 0),
                            "Win Rate":   round(stats.get("win_rate", 0) * 100, 1),
                            "Avg Return": round(stats.get("avg_return", 0), 2),
                        })
                if rows:
                    st.dataframe(
                        pd.DataFrame(rows),
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "Win Rate":   st.column_config.NumberColumn("Win Rate %", format="%.1f"),
                            "Avg Return": st.column_config.NumberColumn("Avg Ret %",  format="%+.2f"),
                        },
                    )
                else:
                    st.caption("No data for this dimension yet.")

    st.markdown("---")

    # ── Learning insights ─────────────────────────────────────────────
    st.markdown("### 💡 Learning Insights")
    insights = tracker.generate_insights()
    if insights:
        for ins in insights:
            st.markdown(f"- {ins}")
    else:
        st.caption("Not enough data for insights yet.")

    # ── Weight optimization suggestions ───────────────────────────────
    st.markdown("### ⚖️ Weight Optimization Suggestions")
    wopt = tracker.suggest_weight_optimization()
    if "message" in wopt:
        st.info(wopt["message"])
    else:
        current = wopt.get("current", {})
        suggested = wopt.get("suggested", {})
        if suggested:
            wo_cols = st.columns(len(suggested))
            for col, (k, v) in zip(wo_cols, suggested.items()):
                curr_v = current.get(k, 0)
                col.metric(
                    k.replace("_", " ").title(),
                    f"{v:.2f}",
                    delta=f"{v - curr_v:+.2f}",
                    delta_color="normal" if v > curr_v else "inverse",
                )
            st.caption("Suggested weights are derived from discriminating power of each signal across completed trades.")

    st.markdown("---")

    # ── Best / Worst picks ────────────────────────────────────────────
    best  = metrics.get("best_pick")
    worst = metrics.get("worst_pick")
    if best or worst:
        st.markdown("### Best & Worst Completed Trades")
        bc1, bc2 = st.columns(2)
        if best:
            with bc1:
                st.success(
                    f"🏆 **{best.get('ticker')}** — {best.get('category')}  \n"
                    f"Return: **{best.get('return_pct', 0):+.2f}%**  |  "
                    f"Score: {best.get('final_score', 0):.0f}  |  "
                    f"Confidence: {best.get('confidence', '')}"
                )
        if worst:
            with bc2:
                st.error(
                    f"📉 **{worst.get('ticker')}** — {worst.get('category')}  \n"
                    f"Return: **{worst.get('return_pct', 0):+.2f}%**  |  "
                    f"Score: {worst.get('final_score', 0):.0f}  |  "
                    f"Confidence: {worst.get('confidence', '')}"
                )

    # ── Raw predictions table ─────────────────────────────────────────
    with st.expander("📋 All Predictions (raw table)"):
        if not df_preds.empty:
            show_cols = [c for c in
                ["ticker", "category", "timestamp", "final_score", "confidence",
                 "entry_price", "exit_target", "stop_loss",
                 "outcome", "return_pct", "days_elapsed"]
                if c in df_preds.columns]
            st.dataframe(
                df_preds[show_cols].sort_values("timestamp", ascending=False),
                use_container_width=True, hide_index=True,
                column_config={
                    "final_score": st.column_config.ProgressColumn(
                        "Score", min_value=0, max_value=100, format="%.1f"),
                    "return_pct":  st.column_config.NumberColumn("Return %", format="%+.2f"),
                },
            )
        else:
            st.caption("No predictions recorded yet.")


def render_backtest_tab():
    st.subheader("Backtest Saved Recommendations")
    st.caption("Simulate how past recommendations would have performed.")

    from pathlib import Path as _Path
    results_dir = _Path("data/results")
    json_files  = sorted(results_dir.glob("scan_*.json"), reverse=True) if results_dir.exists() else []

    if not json_files:
        st.info("No saved scan results yet. Run a scan first.")
        return

    file_options = [f.name for f in json_files]
    selected     = st.selectbox("Select scan file to backtest", file_options)
    lookahead    = st.slider("Lookahead days", 10, 180, 60, 10)

    # Warn if the selected file is from today
    try:
        from datetime import date as _date
        scan_date_str = selected.split("_")[1]          # "20260324"
        scan_date = _date(int(scan_date_str[:4]), int(scan_date_str[4:6]), int(scan_date_str[6:8]))
        days_old  = (_date.today() - scan_date).days
        if days_old < 2:
            st.warning(
                "This scan was run today — there is no forward price data yet. "
                "Backtest results will be meaningful only after a few trading days have passed. "
                "Try again with a scan file that is at least a week old."
            )
    except Exception:
        pass

    if st.button("Run Backtest"):
        from utils.backtester import Backtester
        bt  = Backtester(str(results_dir))
        with st.spinner("Backtesting …"):
            df  = bt.backtest_saved_results(str(results_dir / selected))

        if df.empty:
            st.warning("No valid trades to backtest.")
            return

        summary = Backtester.summarise(df)
        cols = st.columns(len(summary))
        for col, (k, v) in zip(cols, summary.items()):
            col.metric(k.replace("_", " ").title(), v)

        st.dataframe(df[["ticker", "category", "outcome", "pnl_pct",
                          "days_held", "max_drawdown"]].sort_values("pnl_pct", ascending=False),
                     use_container_width=True, hide_index=True)

        # Win/Loss pie
        outcome_counts = df["outcome"].value_counts().reset_index()
        outcome_counts.columns = ["Outcome", "Count"]
        fig = px.pie(outcome_counts, names="Outcome", values="Count",
                     color="Outcome",
                     color_discrete_map={"Win": "#22c55e", "Loss": "#ef4444", "Expired": "#facc15"},
                     template="plotly_dark", height=300)
        st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
