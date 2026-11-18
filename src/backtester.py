"""
Backtesting engine for executing trading strategies with realistic EGX market frictions.
Implements zero-lookahead execution (signal at bar t Close, execution at bar t+1 Open),
commission, slippage, and comprehensive trade logging.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import math
import numpy as np
import pandas as pd

from config.settings import BacktestConfig, StrategyConfig, EGXMarketConfig
from src.strategies import StrategyType, SignalGenerator
from src.indicators import TechnicalIndicators


@dataclass
class Trade:
    trade_id: int
    ticker: str
    strategy: str
    entry_date: pd.Timestamp
    entry_price: float
    entry_reason: str
    exit_date: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    shares: int = 0
    entry_cost: float = 0.0
    exit_proceeds: float = 0.0
    total_commission: float = 0.0
    net_pnl: float = 0.0
    return_pct: float = 0.0
    duration_days: int = 0
    is_open: bool = True

    def to_dict(self) -> Dict:
        return {
            "trade_id": self.trade_id,
            "ticker": self.ticker,
            "strategy": self.strategy,
            "entry_date": self.entry_date.strftime("%Y-%m-%d") if self.entry_date else None,
            "entry_price": round(self.entry_price, 4),
            "entry_reason": self.entry_reason,
            "exit_date": self.exit_date.strftime("%Y-%m-%d") if self.exit_date else None,
            "exit_price": round(self.exit_price, 4) if self.exit_price else None,
            "exit_reason": self.exit_reason,
            "shares": self.shares,
            "entry_cost": round(self.entry_cost, 2),
            "exit_proceeds": round(self.exit_proceeds, 2),
            "total_commission": round(self.total_commission, 2),
            "net_pnl": round(self.net_pnl, 2),
            "return_pct": round(self.return_pct, 4),
            "duration_days": self.duration_days,
            "is_open": self.is_open,
        }


@dataclass
class BacktestResult:
    ticker: str
    strategy: str
    initial_capital: float
    final_equity: float
    trades: List[Trade]
    trades_df: pd.DataFrame
    equity_curve: pd.DataFrame
    config: BacktestConfig


class BacktestEngine:
    """Simulates trading strategy execution with EGX market rules."""

    def __init__(self, config: Optional[BacktestConfig] = None):
        self.config = config or BacktestConfig()

    def run(
        self,
        df: pd.DataFrame,
        ticker: str,
        strategy_type: StrategyType = StrategyType.STRATEGY_A,
    ) -> BacktestResult:
        """
        Executes backtest for a specific stock and strategy.
        """
        cfg = self.config.strategy
        mkt = self.config.market

        # Compute technical indicators
        df_ind = TechnicalIndicators.compute_all_indicators(
            df,
            fast_period=cfg.fast_period,
            slow_period=cfg.slow_period,
            price_col="close"
        )

        # Generate signals
        df_sig = SignalGenerator.generate_signals(
            df_ind,
            strategy_type=strategy_type,
            config=cfg
        )

        n = len(df_sig)
        if n == 0:
            raise ValueError(f"Empty DataFrame passed for {ticker}")

        initial_capital = cfg.initial_capital
        cash = initial_capital
        shares = 0
        trades: List[Trade] = []
        trade_counter = 0
        current_trade: Optional[Trade] = None

        # Tracking arrays for daily equity curve
        dates = df_sig.index
        daily_equity = np.zeros(n)
        daily_cash = np.zeros(n)
        daily_position_val = np.zeros(n)
        daily_shares = np.zeros(n, dtype=int)
        benchmark_equity = np.zeros(n)

        # Benchmark initial setup (Buy & hold equivalent)
        b_init_price = df_sig["open"].iloc[0] * (1.0 + mkt.DEFAULT_SLIPPAGE_PCT)
        b_shares = math.floor(initial_capital / (b_init_price * (1.0 + mkt.DEFAULT_COMMISSION_PCT)))
        b_rem_cash = initial_capital - b_shares * b_init_price * (1.0 + mkt.DEFAULT_COMMISSION_PCT)

        for i in range(n):
            current_bar = df_sig.iloc[i]
            cur_date = dates[i]
            cur_open = float(current_bar["open"])
            cur_close = float(current_bar["close"])

            # Check if there is an execution to perform today from yesterday's signal
            if i > 0:
                prev_bar = df_sig.iloc[i - 1]
                prev_signal = prev_bar.get("signal", 0)
                prev_reason = prev_bar.get("signal_reason", "")

                # 1. Execution of BUY signal (from previous bar)
                if prev_signal == 1 and shares == 0:
                    exec_price = cur_open * (1.0 + mkt.DEFAULT_SLIPPAGE_PCT)
                    cost_per_share = exec_price * (1.0 + mkt.DEFAULT_COMMISSION_PCT)
                    
                    # Position sizing
                    alloc_cash = cash
                    if cfg.position_sizing == "fixed_capital":
                        alloc_cash = min(cash, cfg.fixed_capital_amount)
                    elif cfg.position_sizing == "pct_capital":
                        alloc_cash = cash * cfg.capital_allocation_pct

                    num_shares = math.floor(alloc_cash / cost_per_share)
                    if num_shares > 0:
                        total_entry_val = num_shares * exec_price
                        comm = total_entry_val * mkt.DEFAULT_COMMISSION_PCT
                        entry_cost = total_entry_val + comm

                        cash -= entry_cost
                        shares = num_shares
                        trade_counter += 1

                        current_trade = Trade(
                            trade_id=trade_counter,
                            ticker=ticker,
                            strategy=strategy_type.value,
                            entry_date=cur_date,
                            entry_price=exec_price,
                            entry_reason=prev_reason,
                            shares=num_shares,
                            entry_cost=entry_cost,
                            total_commission=comm,
                            is_open=True
                        )

                # 2. Execution of SELL signal (from previous bar)
                elif prev_signal == -1 and shares > 0 and current_trade is not None:
                    exec_price = cur_open * (1.0 - mkt.DEFAULT_SLIPPAGE_PCT)
                    gross_proceeds = shares * exec_price
                    comm = gross_proceeds * mkt.DEFAULT_COMMISSION_PCT
                    net_proceeds = gross_proceeds - comm

                    cash += net_proceeds
                    current_trade.exit_date = cur_date
                    current_trade.exit_price = exec_price
                    current_trade.exit_reason = prev_reason
                    current_trade.exit_proceeds = net_proceeds
                    current_trade.total_commission += comm
                    current_trade.net_pnl = net_proceeds - current_trade.entry_cost
                    current_trade.return_pct = (current_trade.net_pnl / current_trade.entry_cost) * 100.0
                    current_trade.duration_days = (cur_date - current_trade.entry_date).days
                    current_trade.is_open = False

                    trades.append(current_trade)
                    current_trade = None
                    shares = 0

            # End-of-day portfolio valuation (using Close price)
            position_val = shares * cur_close
            tot_equity = cash + position_val

            daily_cash[i] = cash
            daily_shares[i] = shares
            daily_position_val[i] = position_val
            daily_equity[i] = tot_equity
            benchmark_equity[i] = b_rem_cash + (b_shares * cur_close)

        # Force close open trade at the end if configured
        if self.config.force_close_at_end and shares > 0 and current_trade is not None:
            last_date = dates[-1]
            last_close = float(df_sig["close"].iloc[-1])
            exec_price = last_close * (1.0 - mkt.DEFAULT_SLIPPAGE_PCT)
            gross_proceeds = shares * exec_price
            comm = gross_proceeds * mkt.DEFAULT_COMMISSION_PCT
            net_proceeds = gross_proceeds - comm

            cash += net_proceeds
            current_trade.exit_date = last_date
            current_trade.exit_price = exec_price
            current_trade.exit_reason = "End of Backtest Horizon Liquidation"
            current_trade.exit_proceeds = net_proceeds
            current_trade.total_commission += comm
            current_trade.net_pnl = net_proceeds - current_trade.entry_cost
            current_trade.return_pct = (current_trade.net_pnl / current_trade.entry_cost) * 100.0
            current_trade.duration_days = (last_date - current_trade.entry_date).days
            current_trade.is_open = False

            trades.append(current_trade)
            daily_equity[-1] = cash
            daily_cash[-1] = cash
            daily_shares[-1] = 0
            daily_position_val[-1] = 0

        # Construct equity curve DataFrame
        equity_df = pd.DataFrame({
            "cash": daily_cash,
            "shares": daily_shares,
            "position_val": daily_position_val,
            "equity": daily_equity,
            "benchmark_equity": benchmark_equity,
            "close": df_sig["close"].values,
            "ema_fast": df_sig["ema_fast"].values,
            "ema_slow": df_sig["ema_slow"].values,
            "ema_distance": df_sig["ema_distance"].values,
            "ema_gap_pct": df_sig["ema_gap_pct"].values,
            "signal": df_sig["signal"].values,
        }, index=dates)

        # Calculate drawdowns
        peak = equity_df["equity"].cummax()
        equity_df["drawdown"] = (equity_df["equity"] - peak) / peak * 100.0

        b_peak = equity_df["benchmark_equity"].cummax()
        equity_df["benchmark_drawdown"] = (equity_df["benchmark_equity"] - b_peak) / b_peak * 100.0

        # Trades DataFrame
        trades_records = [t.to_dict() for t in trades]
        trades_df = pd.DataFrame(trades_records) if trades_records else pd.DataFrame(
            columns=["trade_id", "ticker", "strategy", "entry_date", "entry_price", "exit_date", "exit_price",
                     "shares", "entry_cost", "exit_proceeds", "net_pnl", "return_pct", "duration_days", "exit_reason"]
        )

        return BacktestResult(
            ticker=ticker,
            strategy=strategy_type.value,
            initial_capital=initial_capital,
            final_equity=daily_equity[-1],
            trades=trades,
            trades_df=trades_df,
            equity_curve=equity_df,
            config=self.config
        )
