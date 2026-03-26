"""
main.py  —  CLI runner for the AI Multi-Agent Stock Research System
────────────────────────────────────────────────────────────────────

Usage:
    python main.py                          # scan default universe
    python main.py --tickers AAPL MSFT TSLA
    python main.py --universe nasdaq100 --max 30
    python main.py --backtest              # backtest most recent saved scan
"""

import argparse
import sys
import time

from utils.helpers import setup_logging, load_config
from utils.stock_universe import get_universe
from agents.master_agent import MasterAgent


def print_banner():
    print("\n" + "═" * 60)
    print("  🤖  AI Multi-Agent Stock Research System")
    print("═" * 60 + "\n")


def print_results(results: list):
    categories = ["Short-Term", "Medium-Term", "Long-Term"]
    for cat in categories:
        picks = [r for r in results if r.get("category") == cat]
        if not picks:
            continue
        print(f"\n{'─'*50}")
        print(f"  {cat.upper()} PICKS  ({len(picks)} found)")
        print(f"{'─'*50}")
        for r in picks[:5]:
            ticker  = r.get("ticker", "")
            score   = r.get("combined_score", 0)
            conf    = r.get("confidence", "")
            signal  = r.get("signal", "")
            entry   = r.get("entry", "N/A")
            exit_t  = r.get("exit", "N/A")
            stop    = r.get("stop_loss", "N/A")
            reason  = r.get("reason", "")
            val     = r.get("valuation", "")
            flags   = r.get("risk_flags", [])

            print(f"\n  [{ticker}]  Score: {score:.1f}/100  |  Confidence: {conf}  |  Signal: {signal}")
            print(f"  Valuation : {val}")
            print(f"  Entry     : {entry}")
            print(f"  Target    : {exit_t}    Stop: {stop}")
            print(f"  Reason    : {reason}")
            if flags:
                print(f"  ⚠ Flags   : {', '.join(flags)}")


def run_backtest(config: dict):
    from utils.backtester import Backtester
    bt = Backtester(config.get("output", {}).get("results_dir", "data/results"))
    print("\nRunning backtest on most recent saved scan …")
    df = bt.backtest_saved_results()
    if df.empty:
        print("No saved scan results to backtest.")
        return
    summary = Backtester.summarise(df)
    print(f"\n{'─'*40}")
    print("  BACKTEST SUMMARY")
    print(f"{'─'*40}")
    for k, v in summary.items():
        print(f"  {k:<20}: {v}")
    print()
    print(df[["ticker", "category", "outcome", "pnl_pct", "days_held"]].to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description="AI Multi-Agent Stock Research System")
    parser.add_argument("--tickers",   nargs="+", help="Custom ticker list")
    parser.add_argument("--universe",  default=None,
                        choices=["sp500", "sp500_top100", "nasdaq100", "dow30"],
                        help="Stock universe preset")
    parser.add_argument("--max",       type=int, default=None, help="Max stocks to scan")
    parser.add_argument("--backtest",  action="store_true",    help="Backtest saved results")
    parser.add_argument("--no-cache",  action="store_true",    help="Disable cache")
    parser.add_argument("--verbose",   action="store_true",    help="Debug logging")
    args = parser.parse_args()

    setup_logging("DEBUG" if args.verbose else "INFO")
    print_banner()

    config = load_config()

    # Override config from CLI args
    if args.universe:
        config["stock_universe"]["default"] = args.universe
    if args.max:
        config["stock_universe"]["max_stocks"] = args.max
    if args.no_cache:
        config["data"]["cache_enabled"] = False

    if args.backtest:
        run_backtest(config)
        return

    # Resolve tickers
    if args.tickers:
        tickers = [t.upper() for t in args.tickers]
        config["stock_universe"]["custom_tickers"] = tickers
        config["stock_universe"]["max_stocks"]     = len(tickers)
    else:
        tickers = get_universe(config)

    print(f"Scanning {len(tickers)} stocks …\n")

    # Progress callback
    start = time.time()
    def progress(i, total, ticker):
        pct = int(i / total * 40)
        bar = "█" * pct + "░" * (40 - pct)
        sys.stdout.write(f"\r  [{bar}] {i}/{total}  {ticker:<8}")
        sys.stdout.flush()

    agent   = MasterAgent(config)
    results = agent.scan(tickers, progress_cb=progress)

    elapsed = time.time() - start
    print(f"\n\n  ✓ Scan complete in {elapsed:.1f}s  |  {len(results)} valid results\n")

    print_results(results)

    # Summary table
    cats = {"Short-Term": 0, "Medium-Term": 0, "Long-Term": 0}
    for r in results:
        c = r.get("category", "")
        if c in cats:
            cats[c] += 1
    print(f"\n{'─'*40}")
    print("  SCAN SUMMARY")
    print(f"{'─'*40}")
    for cat, cnt in cats.items():
        print(f"  {cat:<15}: {cnt} picks")
    print()


if __name__ == "__main__":
    main()
