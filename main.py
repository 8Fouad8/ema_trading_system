"""
Main orchestration script for the EMA Trading System.
Runs backtests across all available EGX stocks and strategies, calculates metrics,
renders charts, exports trade logs, and compiles executive performance reports.

Usage:
    python main.py
    python main.py --tickers EGAL ORHD TMGH
"""
import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import BacktestConfig, CHARTS_DIR, REPORTS_DIR, TRADES_DIR
from src.data_loader import DataLoader
from src.strategies import StrategyType
from src.backtester import BacktestEngine, BacktestResult
from src.metrics import PerformanceMetrics
from src.visualizer import Visualizer
from src.reporting import ReportGenerator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("EMASystem")


def run_pipeline(tickers: List[str] = None) -> None:
    """Executes the full algorithmic trading pipeline."""
    print("=" * 80)
    print("  EGX EMA CROSSOVER & DISTANCE-OPTIMIZED TRADING SYSTEM")
    print("=" * 80)

    loader = DataLoader()
    if tickers:
        stocks_data = loader.load_multiple_tickers(tickers)
    else:
        stocks_data = loader.load_all_available_stocks()

    if not stocks_data:
        logger.error("No valid stock datasets found. Please check data/raw/ directory.")
        return

    logger.info(f"Loaded {len(stocks_data)} stock(s): {list(stocks_data.keys())}")

    engine = BacktestEngine(BacktestConfig())
    visualizer = Visualizer()
    reporter = ReportGenerator()

    # Strategies to evaluate
    strategies = [
        StrategyType.STRATEGY_A,
        StrategyType.STRATEGY_B,
        StrategyType.STRATEGY_C,
        StrategyType.STRATEGY_D,
        StrategyType.BUY_AND_HOLD,
    ]

    all_results: List[BacktestResult] = []
    all_metrics: List[Dict] = []
    stock_results_map: Dict[str, Dict[str, BacktestResult]] = {}

    for ticker, df in stocks_data.items():
        logger.info(f"Processing ticker: {ticker} ({len(df)} bars)...")
        stock_results_map[ticker] = {}

        for strat in strategies:
            res = engine.run(df=df, ticker=ticker, strategy_type=strat)
            metrics = PerformanceMetrics.calculate_metrics(res)

            all_results.append(res)
            all_metrics.append(metrics)
            stock_results_map[ticker][strat.value] = res

            # Render individual strategy detail chart (skip benchmark detail chart to keep gallery clean)
            if strat != StrategyType.BUY_AND_HOLD:
                chart_path = visualizer.plot_strategy_detail(res, metrics)
                logger.info(f"  Chart generated: {chart_path.name}")

        # Render combined strategy comparison chart for this stock
        comp_chart = visualizer.plot_stock_strategies_comparison(ticker, stock_results_map[ticker])
        logger.info(f"  Comparative chart generated: {comp_chart.name}")

    # Generate Cross-Stock Summary Visualization
    import pandas as pd
    summary_df = pd.DataFrame(all_metrics)
    # Filter for non-benchmark in cross summary plot for clean comparison
    active_summary = summary_df[summary_df["strategy"] != StrategyType.BUY_AND_HOLD.value]
    if not active_summary.empty:
        summary_chart = visualizer.plot_cross_stock_summary(active_summary)
        logger.info(f"Cross-stock summary chart generated: {summary_chart.name}")

    # Export Trade Logs, CSVs, Markdown Report, and HTML Dashboard
    logger.info("Exporting trade journals and reports...")
    master_trade_path = reporter.export_trade_logs(all_results)
    csv_report_path = reporter.generate_summary_csv(all_metrics)
    md_report_path = reporter.generate_markdown_report(all_metrics)
    html_report_path = reporter.generate_html_dashboard(all_metrics)

    print("\n" + "=" * 95)
    print("                      PERFORMANCE SUMMARY TABLE (ALL STOCKS & STRATEGIES)")
    print("=" * 95)
    print(f"{'Ticker':<7} {'Strategy':<28} {'Return %':<10} {'B&H %':<10} {'CAGR %':<10} {'Sharpe':<8} {'MaxDD %':<10} {'Win %':<8} {'Trades':<6}")
    print("-" * 95)

    for m in all_metrics:
        strat_display = m['strategy'].replace('Strategy_', '').replace('Benchmark_', '')[:26]
        print(
            f"{m['ticker']:<7} {strat_display:<28} "
            f"{m['total_return_pct']:>8.2f}% "
            f"{m['benchmark_return_pct']:>8.2f}% "
            f"{m['cagr_pct']:>8.2f}% "
            f"{m['sharpe_ratio']:>7.2f} "
            f"{m['max_drawdown_pct']:>8.2f}% "
            f"{m['win_rate_pct']:>6.1f}% "
            f"{m['total_trades']:>6}"
        )
    print("=" * 95)
    print(f"\n[Artifacts Successfully Generated]:")
    print(f"  - Trade Logs:    {TRADES_DIR}")
    print(f"  - Charts:        {CHARTS_DIR}")
    print(f"  - Markdown Rep:  {md_report_path}")
    print(f"  - HTML Dash:     {html_report_path}")
    print(f"  - Summary CSV:   {csv_report_path}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EMA Trading System Backtester")
    parser.add_argument(
        "--tickers",
        nargs="+",
        help="List of tickers to backtest (e.g. --tickers EGAL ORHD TMGH). Default: all found in data/raw/",
        default=None
    )
    args = parser.parse_args()
    run_pipeline(tickers=args.tickers)
