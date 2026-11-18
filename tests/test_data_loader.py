"""
Unit tests for DataLoader and OHLCVValidator modules.
"""
import unittest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_validator import OHLCVValidator, DataValidationError
from src.data_loader import DataLoader


class TestDataLoaderAndValidator(unittest.TestCase):

    def test_standardize_and_clean_valid_df(self):
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        raw = pd.DataFrame({
            "Date": dates,
            "Open": [10.0] * 30,
            "High": [11.0] * 30,
            "Low": [9.0] * 30,
            "Close": [10.5] * 30,
            "Volume": [1000] * 30
        })
        clean, report = OHLCVValidator.validate_and_clean(raw, ticker="TEST")
        self.assertEqual(len(clean), 30)
        self.assertTrue(report["is_valid"])
        self.assertIn("open", clean.columns)
        self.assertIn("close", clean.columns)

    def test_reject_empty_df(self):
        with self.assertRaises(DataValidationError):
            OHLCVValidator.validate_and_clean(pd.DataFrame(), ticker="EMPTY")

    def test_reject_short_series(self):
        # Under 25 bars should raise DataValidationError
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        raw = pd.DataFrame({
            "Date": dates, "Open": [10]*10, "High": [11]*10, "Low": [9]*10, "Close": [10]*10, "Volume": [100]*10
        })
        with self.assertRaises(DataValidationError):
            OHLCVValidator.validate_and_clean(raw, ticker="SHORT")

    def test_repair_high_low_inversion(self):
        # High < Close or Low > Open should be expanded and reported
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        raw = pd.DataFrame({
            "Date": dates,
            "Open": [10.0] * 30,
            "High": [9.0] * 30,  # Invalid: high < open
            "Low": [11.0] * 30,  # Invalid: low > open
            "Close": [10.5] * 30,
            "Volume": [1000] * 30
        })
        clean, report = OHLCVValidator.validate_and_clean(raw, ticker="INV")
        self.assertGreater(report["ohlc_violations_fixed"], 0)
        # Verify repaired envelope
        self.assertTrue((clean["high"] >= clean["open"]).all())
        self.assertTrue((clean["high"] >= clean["close"]).all())
        self.assertTrue((clean["low"] <= clean["open"]).all())
        self.assertTrue((clean["low"] <= clean["close"]).all())

    def test_load_real_egx_data(self):
        loader = DataLoader()
        stocks = loader.load_all_available_stocks()
        self.assertGreater(len(stocks), 0, "No real stocks loaded from data/raw!")
        for ticker, df in stocks.items():
            self.assertGreaterEqual(len(df), 25)
            self.assertIn("close", df.columns)
            self.assertTrue(df.index.is_monotonic_increasing)


if __name__ == "__main__":
    unittest.main()
