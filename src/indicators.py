"""
Technical indicator computation engine.
Computes configurable EMAs, mathematical distances, normalized percentage gaps,
momentum acceleration metrics, and crossover triggers.
"""
from typing import Optional, Tuple
import numpy as np
import pandas as pd


class TechnicalIndicators:
    """Computes technical indicators and distance metrics for moving average strategies."""

    @staticmethod
    def calculate_ema(series: pd.Series, period: int, adjust: bool = False) -> pd.Series:
        """
        Calculates the Exponential Moving Average (EMA).
        
        Args:
            series: Price series (typically close price).
            period: Smoothing window span (N).
            adjust: If False, uses the recursive formulation EMA_t = alpha*P_t + (1-alpha)*EMA_{t-1}
                    with alpha = 2 / (N + 1).
        """
        if period <= 0:
            raise ValueError(f"EMA period must be positive, got {period}")
        return series.ewm(span=period, adjust=adjust).mean()

    @classmethod
    def compute_all_indicators(
        cls,
        df: pd.DataFrame,
        fast_period: int = 10,
        slow_period: int = 20,
        price_col: str = "close"
    ) -> pd.DataFrame:
        """
        Calculates all indicator columns and distance derivatives on a copy of the DataFrame.
        
        Added Columns:
            - ema_fast: EMA of fast_period
            - ema_slow: EMA of slow_period
            - ema_distance: (ema_fast - ema_slow)
            - ema_gap_pct: (|ema_fast - ema_slow| / ema_slow) * 100
            - ema_signed_gap_pct: ((ema_fast - ema_slow) / ema_slow) * 100
            - ema_distance_change: distance_t - distance_{t-1}
            - ema_gap_expanding: True if distance is expanding positively or negatively
            - ema_fast_change: ema_fast_t - ema_fast_{t-1}
            - ema_fast_decline_streak: Consecutive bars ema_fast has declined
            - bullish_crossover: True on bars where fast crosses above slow
            - bearish_crossover: True on bars where fast crosses below slow
        """
        if fast_period >= slow_period:
            raise ValueError(
                f"fast_period ({fast_period}) must be strictly less than slow_period ({slow_period})"
            )
        if price_col not in df.columns:
            raise KeyError(f"Price column '{price_col}' not found in dataframe columns: {df.columns.tolist()}")

        out = df.copy()
        prices = out[price_col]

        # 1. EMA Calculations
        out["ema_fast"] = cls.calculate_ema(prices, period=fast_period, adjust=False)
        out["ema_slow"] = cls.calculate_ema(prices, period=slow_period, adjust=False)

        # 2. EMA Distance (Spread): EMA_fast - EMA_slow
        out["ema_distance"] = out["ema_fast"] - out["ema_slow"]

        # 3. Normalized EMA Gap %: |EMA_fast - EMA_slow| / EMA_slow * 100
        out["ema_gap_pct"] = (out["ema_distance"].abs() / out["ema_slow"]) * 100.0

        # Signed Gap %: (EMA_fast - EMA_slow) / EMA_slow * 100
        out["ema_signed_gap_pct"] = (out["ema_distance"] / out["ema_slow"]) * 100.0

        # 4. EMA Distance Change: Delta_t = Distance_t - Distance_{t-1}
        out["ema_distance_change"] = out["ema_distance"].diff()

        # Expanding vs Contracting gap
        # When distance is positive and diff > 0 -> expanding bullish momentum
        # When distance is positive and diff < 0 -> contracting bullish momentum (pre-convergence)
        out["ema_gap_expanding"] = out["ema_distance_change"] > 0

        # 5. EMA Fast Momentum & Consecutive Declines (for Strategy C)
        out["ema_fast_change"] = out["ema_fast"].diff()
        
        # Calculate consecutive declining streak for EMA fast
        is_declining = (out["ema_fast_change"] < 0).astype(int)
        # Vectorized consecutive streak counting
        streaks = []
        cur_streak = 0
        for dec in is_declining:
            if dec == 1:
                cur_streak += 1
            else:
                cur_streak = 0
            streaks.append(cur_streak)
        out["ema_fast_decline_streak"] = streaks

        # 6. Crossover Detection
        # Bullish: Fast_t > Slow_t AND Fast_{t-1} <= Slow_{t-1}
        fast = out["ema_fast"]
        slow = out["ema_slow"]
        fast_prev = fast.shift(1)
        slow_prev = slow.shift(1)

        out["bullish_crossover"] = (fast > slow) & (fast_prev <= slow_prev)
        out["bearish_crossover"] = (fast < slow) & (fast_prev >= slow_prev)

        # First row cannot be a crossover because t-1 is NaN
        out.iloc[0, out.columns.get_loc("bullish_crossover")] = False
        out.iloc[0, out.columns.get_loc("bearish_crossover")] = False

        return out
