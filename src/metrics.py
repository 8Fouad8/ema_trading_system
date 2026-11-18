"""
Performance metrics calculation module for algorithmic trading strategies.
Computes annualized returns, risk-adjusted ratios (Sharpe, Sortino, Calmar),
drawdown statistics, trade-level metrics, and benchmark comparisons.
"""
from typing import Dict, Any, List
import numpy as np
import pandas as pd

from src.backtester import BacktestResult


class PerformanceMetrics:
    """Calculates risk and return performance metrics from backtest results."""

    TRADING_DAYS_PER_YEAR: int = 245  # Standard for EGX (Sunday-Thursday, minus holidays)

    @classmethod
    def calculate_metrics(cls, result: BacktestResult, risk_free_rate: float = 0.0) -> Dict[str, Any]:
        """
        Computes all standard portfolio and trade statistics.
        """
        equity = result.equity_curve["equity"]
        benchmark = result.equity_curve["benchmark_equity"]
        trades_df = result.trades_df
        
        n_bars = len(equity)
        if n_bars < 2:
            return {"error": "Insufficient history for metric computation"}

        # 1. Returns and CAGR
        initial_val = equity.iloc[0]
        final_val = equity.iloc[-1]
        total_return_pct = ((final_val - initial_val) / initial_val) * 100.0

        b_initial_val = benchmark.iloc[0]
        b_final_val = benchmark.iloc[-1]
        b_total_return_pct = ((b_final_val - b_initial_val) / b_initial_val) * 100.0

        years = max(n_bars / cls.TRADING_DAYS_PER_YEAR, 0.01)
        cagr_pct = ((final_val / initial_val) ** (1.0 / years) - 1.0) * 100.0 if final_val > 0 else -100.0
        b_cagr_pct = ((b_final_val / b_initial_val) ** (1.0 / years) - 1.0) * 100.0 if b_final_val > 0 else -100.0

        # 2. Daily returns and Risk Ratios
        daily_returns = equity.pct_change().dropna()
        daily_rf = (1.0 + risk_free_rate) ** (1.0 / cls.TRADING_DAYS_PER_YEAR) - 1.0
        excess_returns = daily_returns - daily_rf

        ann_volatility_pct = float(daily_returns.std() * np.sqrt(cls.TRADING_DAYS_PER_YEAR) * 100.0)

        # Sharpe Ratio
        if daily_returns.std() > 0:
            sharpe_ratio = float((excess_returns.mean() / daily_returns.std()) * np.sqrt(cls.TRADING_DAYS_PER_YEAR))
        else:
            sharpe_ratio = 0.0

        # Sortino Ratio (downside deviation only)
        downside_returns = daily_returns[daily_returns < daily_rf]
        downside_std = downside_returns.std()
        if downside_std > 0 and not np.isnan(downside_std):
            sortino_ratio = float((excess_returns.mean() / downside_std) * np.sqrt(cls.TRADING_DAYS_PER_YEAR))
        else:
            sortino_ratio = 0.0

        # 3. Drawdown Analysis
        max_drawdown_pct = float(result.equity_curve["drawdown"].min())
        b_max_drawdown_pct = float(result.equity_curve["benchmark_drawdown"].min())

        # Calmar Ratio: CAGR / abs(Max Drawdown)
        if abs(max_drawdown_pct) > 0:
            calmar_ratio = float(cagr_pct / abs(max_drawdown_pct))
        else:
            calmar_ratio = 0.0

        # 4. Market Exposure (% of time with active position)
        active_bars = (result.equity_curve["shares"] > 0).sum()
        market_exposure_pct = float((active_bars / n_bars) * 100.0)

        # 5. Trade Analysis
        total_trades = len(trades_df)
        if total_trades > 0:
            closed_trades = trades_df[~trades_df["is_open"]]
            n_closed = len(closed_trades)
            
            wins = closed_trades[closed_trades["net_pnl"] > 0]
            losses = closed_trades[closed_trades["net_pnl"] <= 0]
            
            n_wins = len(wins)
            n_losses = len(losses)
            win_rate_pct = (n_wins / n_closed * 100.0) if n_closed > 0 else 0.0

            gross_profits = wins["net_pnl"].sum()
            gross_losses = abs(losses["net_pnl"].sum())
            if gross_losses > 0:
                profit_factor = float(gross_profits / gross_losses)
            elif gross_profits > 0:
                profit_factor = 999.0
            else:
                profit_factor = 0.0

            avg_trade_return_pct = float(closed_trades["return_pct"].mean()) if n_closed > 0 else 0.0
            avg_win_return_pct = float(wins["return_pct"].mean()) if n_wins > 0 else 0.0
            avg_loss_return_pct = float(losses["return_pct"].mean()) if n_losses > 0 else 0.0
            avg_holding_days = float(closed_trades["duration_days"].mean()) if n_closed > 0 else 0.0
            total_commission = float(closed_trades["total_commission"].sum())
            total_net_pnl = float(closed_trades["net_pnl"].sum())
        else:
            n_closed = 0
            n_wins = 0
            n_losses = 0
            win_rate_pct = 0.0
            profit_factor = 0.0
            avg_trade_return_pct = 0.0
            avg_win_return_pct = 0.0
            avg_loss_return_pct = 0.0
            avg_holding_days = 0.0
            total_commission = 0.0
            total_net_pnl = 0.0

        return {
            "ticker": result.ticker,
            "strategy": result.strategy,
            "initial_capital": round(initial_val, 2),
            "final_equity": round(final_val, 2),
            "total_return_pct": round(total_return_pct, 2),
            "benchmark_return_pct": round(b_total_return_pct, 2),
            "alpha_vs_benchmark_pct": round(total_return_pct - b_total_return_pct, 2),
            "cagr_pct": round(cagr_pct, 2),
            "benchmark_cagr_pct": round(b_cagr_pct, 2),
            "annualized_volatility_pct": round(ann_volatility_pct, 2),
            "sharpe_ratio": round(sharpe_ratio, 3),
            "sortino_ratio": round(sortino_ratio, 3),
            "max_drawdown_pct": round(max_drawdown_pct, 2),
            "benchmark_max_drawdown_pct": round(b_max_drawdown_pct, 2),
            "calmar_ratio": round(calmar_ratio, 3),
            "market_exposure_pct": round(market_exposure_pct, 2),
            "total_trades": total_trades,
            "closed_trades": n_closed,
            "winning_trades": n_wins,
            "losing_trades": n_losses,
            "win_rate_pct": round(win_rate_pct, 2),
            "profit_factor": round(profit_factor, 2),
            "avg_trade_return_pct": round(avg_trade_return_pct, 2),
            "avg_win_return_pct": round(avg_win_return_pct, 2),
            "avg_loss_return_pct": round(avg_loss_return_pct, 2),
            "avg_holding_days": round(avg_holding_days, 1),
            "total_net_pnl_egp": round(total_net_pnl, 2),
            "total_commission_egp": round(total_commission, 2),
        }
