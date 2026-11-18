"""
Strategy definitions and signal generators for EMA trading system.
Supports:
- Strategy A: Standard EMA 10/20 Crossover
- Strategy B: Distance Shrink / Convergence Threshold Exit
- Strategy C: Fast EMA Consecutive Decline Streak Exit
- Strategy D: Normalized Gap % Exit
- Benchmark: Buy & Hold
"""
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

from config.settings import StrategyConfig
from src.indicators import TechnicalIndicators


class StrategyType(str, Enum):
    STRATEGY_A = "Strategy_A_Standard_Crossover"
    STRATEGY_B = "Strategy_B_Convergence_Threshold"
    STRATEGY_C = "Strategy_C_Confirmation_Sessions"
    STRATEGY_D = "Strategy_D_Normalized_Gap"
    BUY_AND_HOLD = "Benchmark_Buy_and_Hold"


@dataclass
class Signal:
    timestamp: pd.Timestamp
    action: str  # 'BUY', 'SELL', 'HOLD'
    reason: str  # Description of rule triggered
    price: float


class SignalGenerator:
    """Generates entry and exit signals based on indicator series and strategy rules."""

    @classmethod
    def generate_signals(
        cls,
        df_indicators: pd.DataFrame,
        strategy_type: StrategyType = StrategyType.STRATEGY_A,
        config: Optional[StrategyConfig] = None
    ) -> pd.DataFrame:
        """
        Generates signal columns ['signal', 'signal_action', 'signal_reason'] on df.
        signal: 1 for Buy trigger, -1 for Sell trigger, 0 for Neutral/Hold.
        """
        cfg = config or StrategyConfig()
        df = df_indicators.copy()
        
        n = len(df)
        signals = np.zeros(n, dtype=int)
        reasons = [""] * n
        
        if strategy_type == StrategyType.BUY_AND_HOLD:
            # Buy on bar 1 (after initial bar to have open next day), hold until end
            if n > 0:
                signals[0] = 1
                reasons[0] = "Buy & Hold Initial Entry"
            df["signal"] = signals
            df["signal_reason"] = reasons
            return df

        # State tracking for position-dependent exits (Strategies B, C, D)
        in_position = False
        peak_distance = 0.0

        for i in range(1, n):
            row = df.iloc[i]
            prev_row = df.iloc[i - 1]
            date = df.index[i]
            
            # --- Entry logic (Common to A, B, C, D): Bullish Crossover ---
            bullish_cross = row.get("bullish_crossover", False)
            bearish_cross = row.get("bearish_crossover", False)

            if not in_position:
                if bullish_cross:
                    signals[i] = 1
                    reasons[i] = "Bullish EMA Crossover (Fast > Slow)"
                    in_position = True
                    peak_distance = max(0.0, float(row["ema_distance"]))
            else:
                # Update peak distance while in trade
                current_distance = float(row["ema_distance"])
                if current_distance > peak_distance:
                    peak_distance = current_distance

                # --- Exit logic ---
                exit_triggered = False
                exit_reason = ""

                # 1. Bearish Crossover is an exit for all strategies
                if bearish_cross:
                    exit_triggered = True
                    exit_reason = "Bearish EMA Crossover (Fast < Slow)"

                # Strategy-specific early exit rules
                elif strategy_type == StrategyType.STRATEGY_B:
                    # Exit if current distance drops below convergence_threshold * peak_distance
                    threshold_val = cfg.convergence_threshold * peak_distance
                    if peak_distance > 0 and current_distance <= threshold_val:
                        exit_triggered = True
                        exit_reason = (
                            f"Distance Convergence Exit: spread {current_distance:.2f} <= "
                            f"{cfg.convergence_threshold * 100:.0f}% of peak {peak_distance:.2f}"
                        )

                elif strategy_type == StrategyType.STRATEGY_C:
                    # Exit if fast EMA has declined for >= confirmation_sessions consecutively
                    streak = row.get("ema_fast_decline_streak", 0)
                    if streak >= cfg.confirmation_sessions:
                        exit_triggered = True
                        exit_reason = (
                            f"Momentum Decline Exit: EMA fast declined for {streak} consecutive sessions"
                        )

                elif strategy_type == StrategyType.STRATEGY_D:
                    # Exit if normalized gap percentage is below normalized_gap_threshold
                    gap_pct = float(row.get("ema_gap_pct", 0.0))
                    # Only exit if fast is still above slow but gap is dangerously thin
                    if current_distance > 0 and gap_pct <= cfg.normalized_gap_threshold:
                        exit_triggered = True
                        exit_reason = (
                            f"Normalized Gap Exit: gap {gap_pct:.2f}% <= {cfg.normalized_gap_threshold:.2f}%"
                        )

                if exit_triggered:
                    signals[i] = -1
                    reasons[i] = exit_reason
                    in_position = False
                    peak_distance = 0.0

        df["signal"] = signals
        df["signal_reason"] = reasons
        return df
