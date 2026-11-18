"""
Reporting engine for the EMA Trading System.
Exports trade logs (CSV), performance summary metrics (CSV/Markdown),
and generates a self-contained HTML executive dashboard.
"""
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd

from config.settings import REPORTS_DIR, TRADES_DIR, CHARTS_DIR
from src.backtester import BacktestResult


class ReportGenerator:
    """Creates formatted reports, trade journals, and dashboards."""

    def __init__(self, reports_dir: Path = REPORTS_DIR, trades_dir: Path = TRADES_DIR):
        self.reports_dir = Path(reports_dir)
        self.trades_dir = Path(trades_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.trades_dir.mkdir(parents=True, exist_ok=True)

    def export_trade_logs(self, results: List[BacktestResult]) -> Path:
        """Saves individual strategy trade logs and a master combined trade book."""
        all_trades = []
        for r in results:
            if not r.trades_df.empty:
                safe_strat = r.strategy.replace(" ", "_")
                stock_trade_file = self.trades_dir / f"{r.ticker}_{safe_strat}_trades.csv"
                r.trades_df.to_csv(stock_trade_file, index=False)
                all_trades.append(r.trades_df)

        if all_trades:
            master_df = pd.concat(all_trades, ignore_index=True)
        else:
            master_df = pd.DataFrame()

        master_path = self.trades_dir / "master_trade_book.csv"
        master_df.to_csv(master_path, index=False)
        return master_path

    def generate_summary_csv(self, metrics_list: List[Dict[str, Any]]) -> Path:
        """Saves a summary metrics table as CSV."""
        df = pd.DataFrame(metrics_list)
        out_path = self.reports_dir / "strategy_performance_summary.csv"
        df.to_csv(out_path, index=False)
        return out_path

    def generate_markdown_report(self, metrics_list: List[Dict[str, Any]]) -> Path:
        """Generates a structured markdown report."""
        df = pd.DataFrame(metrics_list)
        out_path = self.reports_dir / "performance_report.md"

        # Highlight top performers
        lines = [
            "# 📈 EMA Trading System — Multi-Stock Performance Report",
            "",
            "**Market:** Egyptian Exchange (EGX equities: EGAL, ORHD, TMGH)  ",
            "**Execution Model:** Zero-lookahead (Signal at Bar $t$, Execution at Bar $t+1$ Open)  ",
            "**Frictions:** 0.15% EGX Commission + 0.10% Slippage per leg  ",
            "",
            "## 1. Executive Summary Table",
            "",
        ]

        # Selected summary columns
        cols = [
            "ticker", "strategy", "total_return_pct", "benchmark_return_pct",
            "alpha_vs_benchmark_pct", "cagr_pct", "sharpe_ratio", "max_drawdown_pct",
            "win_rate_pct", "profit_factor", "total_trades"
        ]
        sub_df = df[cols].copy()
        sub_df.columns = [
            "Ticker", "Strategy", "Return %", "B&H %", "Alpha %",
            "CAGR %", "Sharpe", "Max DD %", "Win Rate %", "Profit Factor", "Trades"
        ]

        header = "| " + " | ".join(sub_df.columns) + " |"
        separator = "| " + " | ".join(["---"] * len(sub_df.columns)) + " |"
        rows = [
            "| " + " | ".join(str(val) for val in row) + " |"
            for row in sub_df.itertuples(index=False)
        ]
        lines.append(header)
        lines.append(separator)
        lines.extend(rows)
        lines.append("")
        lines.append("## 2. Strategy Architecture Descriptions")
        lines.append(
            "- **Strategy A (Standard 10/20 Crossover):** Buys when EMA 10 crosses above EMA 20; exits when EMA 10 crosses below EMA 20.\n"
            "- **Strategy B (Convergence Threshold):** Tracks peak distance between EMAs; exits early if distance contracts below 10% of peak, protecting unrealized profits.\n"
            "- **Strategy C (Confirmation Sessions):** Exits early if fast EMA registers 2 consecutive declining bars, signaling loss of short-term upward momentum.\n"
            "- **Strategy D (Normalized Gap):** Exits early if the absolute normalized distance between EMAs falls below 0.10%.\n"
            "- **Benchmark (Buy & Hold):** Buys at the first available bar and holds through the entire backtest window."
        )
        lines.append("")
        lines.append("## 3. Generated Visualizations")
        lines.append("Detailed visual charts and equity curves have been exported to `outputs/charts/`:")
        lines.append("- Individual stock detail charts: `outputs/charts/{ticker}_{strategy}.png`")
        lines.append("- Comparative stock overlays: `outputs/charts/{ticker}_comparison.png`")
        lines.append("- Cross-stock performance summary: `outputs/charts/cross_stock_summary.png`")

        out_path.write_text("\n".join(lines), encoding="utf-8")
        return out_path

    def generate_html_dashboard(self, metrics_list: List[Dict[str, Any]]) -> Path:
        """Generates an executive, styled HTML dashboard."""
        df = pd.DataFrame(metrics_list)
        out_path = self.reports_dir / "performance_dashboard.html"

        html_table = df[[
            "ticker", "strategy", "total_return_pct", "benchmark_return_pct",
            "alpha_vs_benchmark_pct", "cagr_pct", "sharpe_ratio", "sortino_ratio",
            "max_drawdown_pct", "win_rate_pct", "profit_factor", "total_trades",
            "avg_trade_return_pct", "avg_holding_days", "total_net_pnl_egp"
        ]].to_html(index=False, classes="styled-table", border=0)

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>EGX EMA Trading System — Executive Performance Dashboard</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
            background-color: #f8fafc;
            color: #1e293b;
            margin: 0;
            padding: 30px;
        }}
        .header {{
            background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
            color: white;
            padding: 24px;
            border-radius: 12px;
            margin-bottom: 25px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }}
        .header h1 {{ margin: 0 0 8px 0; font-size: 24px; }}
        .header p {{ margin: 0; opacity: 0.9; font-size: 14px; }}
        .card {{
            background: white;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 25px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        .card h2 {{
            margin-top: 0;
            font-size: 18px;
            color: #0f172a;
            border-bottom: 2px solid #e2e8f0;
            padding-bottom: 10px;
        }}
        .styled-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
            text-align: left;
        }}
        .styled-table th {{
            background-color: #f1f5f9;
            color: #334155;
            font-weight: 600;
            padding: 10px 12px;
            border-bottom: 2px solid #cbd5e1;
        }}
        .styled-table td {{
            padding: 10px 12px;
            border-bottom: 1px solid #e2e8f0;
        }}
        .styled-table tr:hover {{
            background-color: #f8fafc;
        }}
        .badge {{
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: 600;
        }}
        .gallery {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
        }}
        .gallery-item {{
            background: white;
            border-radius: 8px;
            padding: 12px;
            border: 1px solid #e2e8f0;
        }}
        .gallery-item img {{
            width: 100%;
            border-radius: 6px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>EGX EMA Trading System — Executive Backtest Dashboard</h1>
        <p>Zero-Lookahead Execution | Egyptian Exchange (EGX) Friction Model (0.15% Brokerage + 0.10% Slippage)</p>
    </div>

    <div class="card">
        <h2>Quantitative Performance Metrics Across Stocks & Strategies</h2>
        <div style="overflow-x: auto;">
            {html_table}
        </div>
    </div>

    <div class="card">
        <h2>Comparative Stock Performance Charts</h2>
        <div class="gallery">
            <div class="gallery-item">
                <h3>Cross-Stock Performance Summary</h3>
                <img src="../charts/cross_stock_summary.png" alt="Cross Stock Summary">
            </div>
            <div class="gallery-item">
                <h3>EGAL — Strategies Comparison</h3>
                <img src="../charts/EGAL_comparison.png" alt="EGAL Comparison">
            </div>
            <div class="gallery-item">
                <h3>ORHD — Strategies Comparison</h3>
                <img src="../charts/ORHD_comparison.png" alt="ORHD Comparison">
            </div>
            <div class="gallery-item">
                <h3>TMGH — Strategies Comparison</h3>
                <img src="../charts/TMGH_comparison.png" alt="TMGH Comparison">
            </div>
        </div>
    </div>
</body>
</html>
"""
        out_path.write_text(html_content, encoding="utf-8")
        return out_path
