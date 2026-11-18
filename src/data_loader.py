"""
Data loader module for loading, sanitizing, and caching EGX historical stock data.
"""
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
import pandas as pd

from config.settings import RAW_DATA_DIR, PROCESSED_DATA_DIR
from src.data_validator import OHLCVValidator, DataValidationError

logger = logging.getLogger(__name__)


class DataLoader:
    """Manages loading of historical price datasets from CSV or raw directories."""

    def __init__(self, raw_dir: Path = RAW_DATA_DIR, processed_dir: Path = PROCESSED_DATA_DIR):
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def load_csv(self, file_path: Union[str, Path], ticker: Optional[str] = None) -> pd.DataFrame:
        """
        Loads a single CSV file, validates OHLCV invariants, and returns a clean DataFrame.
        """
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"Data file not found at: {path}")

        sym = ticker or path.stem.upper()
        raw_df = pd.read_csv(path)
        clean_df, report = OHLCVValidator.validate_and_clean(raw_df, ticker=sym)
        
        # Save processed data cache
        processed_path = self.processed_dir / f"{sym}_clean.parquet"
        try:
            clean_df.to_parquet(processed_path)
        except Exception:
            # Fallback to CSV if parquet engine not available
            clean_df.to_csv(self.processed_dir / f"{sym}_clean.csv")

        logger.info(
            f"Successfully loaded and cleaned {sym}: {len(clean_df)} bars "
            f"from {report['start_date']} to {report['end_date']}."
        )
        return clean_df

    def load_ticker(self, ticker: str) -> pd.DataFrame:
        """Finds and loads a ticker from the raw data directory."""
        ticker_clean = ticker.upper().replace(".CA", "")
        # Search for ticker in raw_dir
        candidates = list(self.raw_dir.glob(f"*{ticker_clean}*.csv")) + list(self.raw_dir.glob(f"*{ticker_clean.lower()}*.csv"))
        if not candidates:
            raise FileNotFoundError(
                f"No CSV file found for ticker '{ticker}' in directory '{self.raw_dir}'"
            )
        return self.load_csv(candidates[0], ticker=ticker_clean)

    def load_multiple_tickers(self, tickers: List[str]) -> Dict[str, pd.DataFrame]:
        """Loads and cleans multiple specified tickers."""
        results = {}
        for t in tickers:
            try:
                results[t.upper()] = self.load_ticker(t)
            except Exception as e:
                logger.error(f"Error loading {t}: {e}")
        return results

    def load_all_available_stocks(self) -> Dict[str, pd.DataFrame]:
        """Scans the raw data folder and loads all valid CSV datasets."""
        csv_files = list(self.raw_dir.glob("*.csv"))
        if not csv_files:
            logger.warning(f"No CSV files found in {self.raw_dir}")
            return {}

        results = {}
        for f in csv_files:
            ticker = f.stem.upper().replace(".CA", "")
            try:
                df = self.load_csv(f, ticker=ticker)
                results[ticker] = df
            except Exception as e:
                logger.error(f"Failed to load file {f.name}: {e}")
        return results
