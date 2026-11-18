"""
Visualization engine for the EMA Trading System.
Generates multi-panel strategy execution charts, equity curve comparisons,
drawdown profiles, and cross-stock performance summaries.
"""
from pathlib import Path
from typing import Dict, List, Optional
import matplotlib
matplotlib.use("Agg")  # Headless backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

from config.settings import CHARTS_DIR
from src.backtester import BacktestResult


class Visualizer:
    """Renders and saves high-resolution publication-quality performance graphs."""

    def __init__(self, output_dir: Path = CHARTS_DIR):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Setup clean plotting aesthetics
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    def plot_strategy_detail(self, result: BacktestResult, metrics: Dict) -> Path:
        """
        Creates a comprehensive 3-panel figure for a single stock and strategy:
        1. Price, EMA 10, EMA 20, and Buy/Sell execution points.
        2. EMA Distance / Gap %.
        3. Portfolio Equity Curve vs Buy & Hold Benchmark and Drawdown.
        """
        eq = result.equity_curve
        trades = result.trades_df
        ticker = result.ticker
        strategy = result.strategy

        fig, (ax1, ax2, ax3) = plt.subplots(
            nrows=3, ncols=1, figsize=(14, 10), sharex=True,
            gridspec_kw={"height_ratios": [2.2, 1.0, 1.8]}
        )

        title_str = (
            f"{ticker} — {strategy.replace('_', ' ')}\n"
            f"Return: {metrics.get('total_return_pct', 0.0):+.1f}% | "
            f"Sharpe: {metrics.get('sharpe_ratio', 0.0):.2f} | "
            f"MaxDD: {metrics.get('max_drawdown_pct', 0.0):.1f}% | "
            f"Win Rate: {metrics.get('win_rate_pct', 0.0):.1f}%"
        )
        fig.suptitle(title_str, fontsize=13, fontweight="bold", y=0.98)

        # ---------------- Panel 1: Price and EMAs ----------------
        ax1.plot(eq.index, eq["close"], label="Close Price", color="#1f2937", linewidth=1.2, alpha=0.85)
        ax1.plot(eq.index, eq["ema_fast"], label="EMA Fast (10)", color="#2563eb", linewidth=1.4)
        ax1.plot(eq.index, eq["ema_slow"], label="EMA Slow (20)", color="#ea580c", linewidth=1.4)

        # Plot Buy & Sell markers from trades
        if not trades.empty:
            for _, tr in trades.iterrows():
                e_date = pd.to_datetime(tr["entry_date"])
                e_price = tr["entry_price"]
                ax1.scatter(e_date, e_price, marker="^", color="#16a34a", s=90, zorder=5, label="Buy Entry" if _ == 0 else "")
                
                if pd.notnull(tr["exit_date"]):
                    x_date = pd.to_datetime(tr["exit_date"])
                    x_price = tr["exit_price"]
                    pnl_color = "#16a34a" if tr["net_pnl"] > 0 else "#dc2626"
                    ax1.scatter(x_date, x_price, marker="v", color=pnl_color, s=90, zorder=5, label="Sell Exit" if _ == 0 else "")

        ax1.set_ylabel("Price (EGP)", fontsize=10, fontweight="bold")
        ax1.legend(loc="upper left", framealpha=0.9, fontsize=8)
        ax1.grid(True, linestyle="--", alpha=0.5)

        # ---------------- Panel 2: EMA Distance Spread ----------------
        ax2.plot(eq.index, eq["ema_distance"], label="EMA Distance (10 - 20)", color="#7c3aed", linewidth=1.1)
        ax2.axhline(0, color="#6b7280", linestyle=":", linewidth=1.0)
        ax2.fill_between(eq.index, 0, eq["ema_distance"], where=(eq["ema_distance"] >= 0), color="#10b981", alpha=0.25)
        ax2.fill_between(eq.index, 0, eq["ema_distance"], where=(eq["ema_distance"] < 0), color="#ef4444", alpha=0.25)
        ax2.set_ylabel("Spread (EGP)", fontsize=10, fontweight="bold")
        ax2.legend(loc="upper left", framealpha=0.9, fontsize=8)
        ax2.grid(True, linestyle="--", alpha=0.5)

        # ---------------- Panel 3: Equity Curve & Drawdown ----------------
        ax3.plot(eq.index, eq["equity"], label=f"Strategy Equity ({strategy.split('_')[1]})", color="#059669", linewidth=1.6)
        ax3.plot(eq.index, eq["benchmark_equity"], label="Buy & Hold Benchmark", color="#9ca3af", linestyle="--", linewidth=1.3)
        ax3.set_ylabel("Portfolio Value (EGP)", fontsize=10, fontweight="bold")
        ax3.legend(loc="upper left", framealpha=0.9, fontsize=8)
        ax3.grid(True, linestyle="--", alpha=0.5)

        # Format X axis dates
        ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        fig.autofmt_xdate()

        plt.tight_layout()
        safe_strat = strategy.replace(" ", "_")
        filename = self.output_dir / f"{ticker}_{safe_strat}.png"
        fig.savefig(filename, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return filename

    def plot_stock_strategies_comparison(self, ticker: str, results_dict: Dict[str, BacktestResult]) -> Path:
        """
        Plots an equity curve comparison of all tested strategies for a single stock.
        """
        fig, (ax1, ax2) = plt.subplots(
            nrows=2, ncols=1, figsize=(14, 8), sharex=True,
            gridspec_kw={"height_ratios": [2.5, 1.2]}
        )

        colors = {
            "Strategy_A_Standard_Crossover": "#2563eb",
            "Strategy_B_Convergence_Threshold": "#10b981",
            "Strategy_C_Confirmation_Sessions": "#f59e0b",
            "Strategy_D_Normalized_Gap": "#8b5cf6",
            "Benchmark_Buy_and_Hold": "#6b7280"
        }

        # First add benchmark
        first_res = list(results_dict.values())[0]
        eq_first = first_res.equity_curve
        ax1.plot(eq_first.index, eq_first["benchmark_equity"], label="Buy & Hold Benchmark",
                 color="#9ca3af", linestyle="--", linewidth=1.4)
        ax2.plot(eq_first.index, eq_first["benchmark_drawdown"], label="Buy & Hold DD",
                 color="#9ca3af", linestyle="--", linewidth=1.0)

        for strat_name, res in results_dict.items():
            eq = res.equity_curve
            clr = colors.get(strat_name, "#3b82f6")
            short_name = strat_name.replace("Strategy_", "Strat ").replace("_", " ")
            final_ret = ((res.final_equity - res.initial_capital) / res.initial_capital) * 100.0
            label_str = f"{short_name} ({final_ret:+.1f}%)"

            ax1.plot(eq.index, eq["equity"], label=label_str, color=clr, linewidth=1.5)
            ax2.plot(eq.index, eq["drawdown"], label=short_name, color=clr, linewidth=1.1)

        ax1.set_title(f"{ticker} — Strategy Performance Comparison (Equity & Drawdowns)", fontsize=13, fontweight="bold")
        ax1.set_ylabel("Portfolio Value (EGP)", fontsize=10, fontweight="bold")
        ax1.legend(loc="upper left", framealpha=0.9, fontsize=8)
        ax1.grid(True, linestyle="--", alpha=0.5)

        ax2.set_ylabel("Drawdown %", fontsize=10, fontweight="bold")
        ax2.set_xlabel("Date", fontsize=10, fontweight="bold")
        ax2.axhline(0, color="black", linestyle=":", linewidth=0.8)
        ax2.grid(True, linestyle="--", alpha=0.5)

        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        fig.autofmt_xdate()

        plt.tight_layout()
        filename = self.output_dir / f"{ticker}_comparison.png"
        fig.savefig(filename, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return filename

    def plot_cross_stock_summary(self, summary_df: pd.DataFrame) -> Path:
        """
        Generates bar charts comparing Returns and Max Drawdowns across all stocks and strategies.
        """
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))

        # Filter out benchmark if in table for cleaner comparison
        df_plot = summary_df.copy()
        
        # Shorten strategy names for plotting
        df_plot["short_strat"] = df_plot["strategy"].str.replace("Strategy_", "").str.replace("_", " ")
        
        # 1. Total Return by Stock & Strategy
        pivot_ret = df_plot.pivot(index="ticker", columns="short_strat", values="total_return_pct")
        pivot_ret.plot(kind="bar", ax=ax1, colormap="tab10", width=0.8)
        ax1.set_title("Total Return (%) by Stock & Strategy", fontsize=11, fontweight="bold")
        ax1.set_ylabel("Return %", fontsize=10)
        ax1.axhline(0, color="black", linestyle="-", linewidth=0.8)
        ax1.grid(True, linestyle="--", alpha=0.5)
        ax1.legend(title="Strategy", fontsize=8, title_fontsize=9)

        # 2. Max Drawdown by Stock & Strategy
        pivot_dd = df_plot.pivot(index="ticker", columns="short_strat", values="max_drawdown_pct")
        pivot_dd.plot(kind="bar", ax=ax2, colormap="tab10", width=0.8)
        ax2.set_title("Maximum Drawdown (%) by Stock & Strategy", fontsize=11, fontweight="bold")
        ax2.set_ylabel("Max Drawdown %", fontsize=10)
        ax2.grid(True, linestyle="--", alpha=0.5)
        ax2.legend(title="Strategy", fontsize=8, title_fontsize=9)

        plt.tight_layout()
        filename = self.output_dir / "cross_stock_summary.png"
        fig.savefig(filename, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return filename
