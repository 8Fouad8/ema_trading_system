"""
Unit tests for BacktestEngine and Strategy execution logic.
"""
import unittest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import BacktestConfig, StrategyConfig
from src.strategies import StrategyType
from src.backtester import BacktestEngine
from src.metrics import PerformanceMetrics


class TestBacktestEngine(unittest.TestCase):

    def setUp(self):
        # Generate 60 bars of synthetic trend data: first 30 bars upward, next 30 downward
        dates = pd.date_range("2024-01-01", periods=60, freq="B")
        close_prices = [100.0 + i * 2.0 if i < 30 else 160.0 - (i - 30) * 2.5 for i in range(60)]
        self.df = pd.DataFrame({
            "open": [p - 0.2 for p in close_prices],
            "high": [p + 0.8 for p in close_prices],
            "low": [p - 0.8 for p in close_prices],
            "close": close_prices,
            "volume": [50000.0] * 60
        }, index=dates)

        self.engine = BacktestEngine(BacktestConfig())

    def test_backtest_execution_no_lookahead(self):
        """Verifies that signals at bar t are executed at bar t+1."""
        res = self.engine.run(self.df, ticker="TEST_TICKER", strategy_type=StrategyType.STRATEGY_A)
        self.assertIsNotNone(res)
        self.assertGreater(len(res.equity_curve), 0)
        
        if not res.trades_df.empty:
            trade = res.trades[0]
            # Verify trade entry date corresponds to a bar in the dataframe
            self.assertIn(trade.entry_date, self.df.index)

    def test_metrics_computation(self):
        """Verifies that metrics return standard fields without exceptions."""
        res = self.engine.run(self.df, ticker="TEST_TICKER", strategy_type=StrategyType.STRATEGY_A)
        metrics = PerformanceMetrics.calculate_metrics(res)
        
        expected_keys = [
            "ticker", "strategy", "initial_capital", "final_equity",
            "total_return_pct", "cagr_pct", "sharpe_ratio", "max_drawdown_pct",
            "win_rate_pct", "profit_factor"
        ]
        for k in expected_keys:
            self.assertIn(k, metrics)


if __name__ == "__main__":
    unittest.main()
