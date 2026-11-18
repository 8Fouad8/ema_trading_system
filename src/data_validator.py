"""
Data validation module for OHLCV financial market data.
Ensures data integrity, chronological sorting, absence of anomalies, and zero look-ahead contamination.
"""
import logging
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DataValidationError(Exception):
    """Raised when critical OHLCV data validation fails."""
    pass


class OHLCVValidator:
    """Validates and sanitizes raw historical equity data according to quantitative standards."""
    
    REQUIRED_COLUMNS = ["open", "high", "low", "close", "volume"]

    @classmethod
    def standardize_columns(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Standardizes column names to lowercase and resolves common timestamp aliases."""
        df = df.copy()
        # Strip whitespace and convert to lowercase
        df.columns = [str(c).strip().lower() for c in df.columns]
        
        # Timestamp alias mapping
        rename_map = {}
        for c in df.columns:
            if c in ["date", "datetime", "time", "trade_date"]:
                rename_map[c] = "timestamp"
        if rename_map:
            df.rename(columns=rename_map, inplace=True)
            
        return df

    @classmethod
    def validate_and_clean(cls, df: pd.DataFrame, ticker: str = "UNKNOWN") -> Tuple[pd.DataFrame, Dict[str, any]]:
        """
        Validates OHLCV structure and cleans minor issues while rejecting corrupt data.
        
        Returns:
            Tuple of (cleaned_df, validation_report_dict)
        """
        report: Dict[str, any] = {
            "ticker": ticker,
            "raw_rows": len(df),
            "duplicates_removed": 0,
            "missing_values_dropped": 0,
            "ohlc_violations_fixed": 0,
            "zero_volume_bars": 0,
            "is_valid": True,
            "warnings": [],
            "start_date": None,
            "end_date": None
        }

        if df.empty:
            raise DataValidationError(f"[{ticker}] Dataset is empty.")

        # 1. Standardize column names
        df = cls.standardize_columns(df)

        # 2. Check required timestamp
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            missing_dates = df["timestamp"].isna().sum()
            if missing_dates > 0:
                report["warnings"].append(f"Dropped {missing_dates} rows with unparseable timestamp.")
                df = df.dropna(subset=["timestamp"])
            df = df.sort_values("timestamp").reset_index(drop=True)
        elif isinstance(df.index, pd.DatetimeIndex):
            df["timestamp"] = df.index
            df = df.sort_values("timestamp").reset_index(drop=True)
        else:
            raise DataValidationError(f"[{ticker}] No valid timestamp/date column or index found.")

        # 3. Check required OHLCV columns
        missing_cols = [col for col in cls.REQUIRED_COLUMNS if col not in df.columns]
        if missing_cols:
            raise DataValidationError(f"[{ticker}] Missing required columns: {missing_cols}")

        # 4. Remove duplicate timestamps
        initial_len = len(df)
        df = df.drop_duplicates(subset=["timestamp"], keep="last").reset_index(drop=True)
        duplicates_removed = initial_len - len(df)
        report["duplicates_removed"] = duplicates_removed
        if duplicates_removed > 0:
            report["warnings"].append(f"Removed {duplicates_removed} duplicate timestamp rows.")

        # 5. Type casting to float64
        for col in cls.REQUIRED_COLUMNS:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # 6. Drop missing OHLC values
        missing_ohlc = df[["open", "high", "low", "close"]].isna().any(axis=1).sum()
        if missing_ohlc > 0:
            report["missing_values_dropped"] = int(missing_ohlc)
            report["warnings"].append(f"Dropped {missing_ohlc} rows with missing OHLC values.")
            df = df.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)

        if len(df) < 25:
            raise DataValidationError(
                f"[{ticker}] Insufficient observations ({len(df)} rows). "
                "Minimum required for stable EMA(20) is 25 rows."
            )

        # 7. Volume treatment (fill NaN volume with 0, non-negative check)
        df["volume"] = df["volume"].fillna(0.0)
        df["volume"] = df["volume"].clip(lower=0.0)
        report["zero_volume_bars"] = int((df["volume"] == 0).sum())

        # 8. Sanity check: Price positivity
        negative_or_zero = (df[["open", "high", "low", "close"]] <= 0).any(axis=1)
        if negative_or_zero.any():
            invalid_count = int(negative_or_zero.sum())
            report["warnings"].append(f"Dropped {invalid_count} rows with non-positive prices.")
            df = df[~negative_or_zero].reset_index(drop=True)

        # 9. OHLC Logical Consistency Checks:
        # High must be >= max(Open, Close, Low)
        # Low must be <= min(Open, Close, High)
        violating_high = (df["high"] < df["open"]) | (df["high"] < df["close"])
        violating_low = (df["low"] > df["open"]) | (df["low"] > df["close"])
        ohlc_violations = violating_high | violating_low
        violation_count = int(ohlc_violations.sum())
        
        if violation_count > 0:
            report["ohlc_violations_fixed"] = violation_count
            report["warnings"].append(
                f"Repaired {violation_count} bars where High < max(Open,Close) or Low > min(Open,Close) "
                "by expanding High/Low envelope."
            )
            df["high"] = df[["open", "high", "close"]].max(axis=1)
            df["low"] = df[["open", "low", "close"]].min(axis=1)

        # Set clean timestamp as index
        df.set_index("timestamp", inplace=True)
        df.sort_index(inplace=True)

        report["clean_rows"] = len(df)
        report["start_date"] = str(df.index[0].date())
        report["end_date"] = str(df.index[-1].date())

        return df, report
