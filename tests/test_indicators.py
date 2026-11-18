"""
Unit tests for TechnicalIndicators module.
"""
import unittest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.indicators import TechnicalIndicators


class TestTechnicalIndicators(unittest.TestCase):

    def setUp(self):
        # Create deterministic synthetic price series
        dates = pd.date_range("2024-01-01", periods=50, freq="B")
        # Prices start at 100, rise to 120, then fall to 90
        prices = [100.0 + i * 1.5 if i < 25 else 137.5 - (i - 25) * 2.0 for i in range(50)]
        self.df = pd.DataFrame({
            "open": [p - 0.5 for p in prices],
            "high": [p + 1.0 for p in prices],
            "low": [p - 1.0 for p in prices],
            "close": prices,
            "volume": [100000.0] * 50
        }, index=dates)

    def test_ema_formula_correctness(self):
        """Verify EMA matches exact recursive formula: EMA_t = alpha*P_t + (1-alpha)*EMA_{t-1}."""
        prices = pd.Series([10.0, 11.0, 12.0, 13.0, 14.0])
        period = 3
        alpha = 2.0 / (period + 1.0)  # alpha = 0.5
        
        calculated_ema = TechnicalIndicators.calculate_ema(prices, period=period, adjust=False)
        
        # Manual expectation:
        # t=0: 10.0
        # t=1: 0.5*11 + 0.5*10 = 10.5
        # t=2: 0.5*12 + 0.5*10.5 = 11.25
        # t=3: 0.5*13 + 0.5*11.25 = 12.125
        # t=4: 0.5*14 + 0.5*12.125 = 13.0625
        expected = [10.0, 10.5, 11.25, 12.125, 13.0625]
        for c, e in zip(calculated_ema, expected):
            self.assertAlmostEqual(c, e, places=5)

    def test_indicator_computation_columns(self):
        res = TechnicalIndicators.compute_all_indicators(self.df, fast_period=10, slow_period=20)
        expected_cols = [
            "ema_fast", "ema_slow", "ema_distance", "ema_gap_pct",
            "ema_signed_gap_pct", "ema_distance_change", "ema_gap_expanding",
            "ema_fast_change", "ema_fast_decline_streak",
            "bullish_crossover", "bearish_crossover"
        ]
        for col in expected_cols:
            self.assertIn(col, res.columns)

    def test_crossover_mutual_exclusion(self):
        """A single bar cannot be both a bullish and a bearish crossover."""
        res = TechnicalIndicators.compute_all_indicators(self.df, fast_period=10, slow_period=20)
        both = res["bullish_crossover"] & res["bearish_crossover"]
        self.assertFalse(both.any(), "A bar was flagged as both bullish and bearish crossover!")

    def test_invalid_period_ordering(self):
        """Fast period must be strictly smaller than slow period."""
        with self.assertRaises(ValueError):
            TechnicalIndicators.compute_all_indicators(self.df, fast_period=20, slow_period=10)
        with self.assertRaises(ValueError):
            TechnicalIndicators.compute_all_indicators(self.df, fast_period=15, slow_period=15)

    def test_distance_and_gap_math(self):
        res = TechnicalIndicators.compute_all_indicators(self.df, fast_period=5, slow_period=10)
        for idx in range(len(res)):
            row = res.iloc[idx]
            expected_dist = row["ema_fast"] - row["ema_slow"]
            self.assertAlmostEqual(row["ema_distance"], expected_dist, places=5)
            
            expected_gap = abs(expected_dist) / row["ema_slow"] * 100.0
            self.assertAlmostEqual(row["ema_gap_pct"], expected_gap, places=5)


if __name__ == "__main__":
    unittest.main()
