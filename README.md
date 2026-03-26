# AI Multi-Agent Stock Research System

A production-ready, modular stock research system powered entirely by **free data sources**.

## Architecture

```
Master Agent (Decision Engine)
├── Fundamental Agent  — yfinance financial data, valuation scoring
├── Technical Agent    — price data + ta library indicators
└── Sentiment Agent    — Yahoo/Google News RSS + optional Reddit (PRAW)
```

## Quick Start

### 1. Install dependencies

```bash
cd stock_agent
pip install -r requirements.txt
```

### 2. Download NLTK VADER lexicon (one-time)

```bash
python -c "import nltk; nltk.download('vader_lexicon')"
```

### 3. Run the Streamlit dashboard

```bash
streamlit run app.py
```

### 4. Or use the CLI

```bash
# Scan default universe (S&P 500 top 100)
python main.py

# Scan specific tickers
python main.py --tickers AAPL MSFT NVDA TSLA

# Scan NASDAQ-100 top 30
python main.py --universe nasdaq100 --max 30

# Backtest most recent saved scan
python main.py --backtest

# Verbose output
python main.py --verbose
```

## Project Structure

```
stock_agent/
├── agents/
│   ├── fundamental_agent.py   # Revenue, EPS, ROE, FCF, P/E, PEG scoring
│   ├── technical_agent.py     # SMA/EMA, RSI, MACD, BB, ATR, Volume
│   ├── sentiment_agent.py     # News RSS + Reddit + VADER NLP
│   └── master_agent.py        # Orchestration, weighted scoring, categorisation
├── config/
│   └── settings.yaml          # All configuration (weights, thresholds, etc.)
├── data/
│   ├── cache/                 # TTL-based JSON cache
│   └── results/               # Saved scan results (JSON + CSV)
├── utils/
│   ├── helpers.py             # Config loader, cache, formatters
│   ├── stock_universe.py      # S&P500/NASDAQ100/Dow30 ticker lists
│   └── backtester.py          # Simple trade simulator
├── app.py                     # Streamlit dashboard
├── main.py                    # CLI entry point
└── requirements.txt
```

## Configuration (`config/settings.yaml`)

| Setting | Default | Description |
|---------|---------|-------------|
| `weights.fundamentals` | 0.40 | Weight of fundamental score |
| `weights.technicals`   | 0.35 | Weight of technical score |
| `weights.sentiment`    | 0.25 | Weight of sentiment score |
| `stock_universe.default` | `sp500_top100` | Universe preset |
| `stock_universe.max_stocks` | 50 | Hard cap for scan speed |
| `data.cache_ttl_minutes` | 60 | How long to cache API responses |

## Optional: Enable Reddit Sentiment

1. Register a free Reddit app at https://www.reddit.com/prefs/apps
2. Add credentials to `config/settings.yaml`:

```yaml
reddit:
  enabled: true
  client_id: "YOUR_CLIENT_ID"
  client_secret: "YOUR_CLIENT_SECRET"
  user_agent: "StockResearch/1.0 (by u/yourname)"
```

## Output Format

Each recommendation includes:

```json
{
  "ticker": "AAPL",
  "company_name": "Apple Inc.",
  "category": "Medium-Term",
  "combined_score": 74.3,
  "fundamental_score": 68.0,
  "technical_score": 82.0,
  "momentum_score": 65.0,
  "sentiment_score": 0.21,
  "confidence": "High",
  "signal": "Bullish",
  "valuation": "Fair Value",
  "entry": "$182.50 – $184.10",
  "exit": "$198.40",
  "stop_loss": "$174.20",
  "reason": "Strong fundamentals + bullish technical setup + positive market sentiment.",
  "risk_flags": []
}
```

## Scoring Methodology

### Fundamental Score (0–100)
| Metric | Max Points |
|--------|-----------|
| Revenue Growth (YoY) | 20 |
| EPS Growth | 15 |
| Return on Equity | 15 |
| Debt/Equity Ratio | 10 |
| Free Cash Flow (positive) | 10 |
| Profit Margin | 10 |
| P/E Ratio | 10 |
| PEG Ratio | 10 |

### Technical Score (0–100)
| Signal | Max Points |
|--------|-----------|
| Trend (SMA-50/200) | 20 |
| Golden/Death Cross | 15 |
| RSI Position | 20 |
| MACD Signal | 20 |
| Volume Analysis | 15 |
| Bollinger Band Position | 10 |

### Sentiment Score (−1 → +1)
- Yahoo Finance news RSS: 60% weight
- Google News RSS: 40% weight
- Reddit posts (if enabled): blended in at 40%
- VADER compound score aggregated across headlines

### Categorisation Logic
| Category | Criteria |
|----------|----------|
| **Short-Term** | Tech ≥ 62, Sentiment > 0.05, Combined ≥ 55 |
| **Long-Term** | Fundamentals ≥ 62, Combined ≥ 50 |
| **Medium-Term** | Tech ≥ 45, Fund ≥ 42, Combined ≥ 52 |

## Data Sources (100% Free)
- **Price data**: Yahoo Finance via `yfinance`
- **Financials**: Yahoo Finance via `yfinance`
- **News**: Yahoo Finance RSS + Google News RSS
- **Social sentiment**: Reddit via PRAW (optional, free account)
- **NLP**: NLTK VADER (offline, no API needed)

## Disclaimer

This tool is for educational and research purposes only. It is not financial advice. Always do your own due diligence before making investment decisions.
