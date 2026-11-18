"""
Configuration settings for the EMA Crossover Backtesting & Optimization System.
Designed specifically for Egyptian Exchange (EGX) equities.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

# Project root path
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data directory paths
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RESULTS_DATA_DIR = DATA_DIR / "results"

# Outputs directory paths
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
CHARTS_DIR = OUTPUTS_DIR / "charts"
REPORTS_DIR = OUTPUTS_DIR / "reports"
TRADES_DIR = OUTPUTS_DIR / "trades"

# Ensure runtime directories exist
for p in [RAW_DATA_DIR, PROCESSED_DATA_DIR, RESULTS_DATA_DIR, CHARTS_DIR, REPORTS_DIR, TRADES_DIR]:
    p.mkdir(parents=True, exist_ok=True)


@dataclass
class EGXMarketConfig:
    """Egyptian Exchange (EGX) trading calendar and execution rules."""
    # EGX standard trading week: Sunday (6) through Thursday (3)
    # Python weekday(): Monday=0, Tuesday=1, Wednesday=2, Thursday=3, Friday=4, Saturday=5, Sunday=6
    TRADING_DAYS: List[int] = field(default_factory=lambda: [6, 0, 1, 2, 3])
    
    # Typical EGX transaction costs (approx 0.15% - 0.20% all-in one way: broker + FRA + Misr Clearing + stamp)
    # Brokerage commission + Financial Regulatory Authority (FRA) + EGX fees + MCDR + Stamp Duty
    DEFAULT_COMMISSION_PCT: float = 0.0015  # 0.15% per leg (buy and sell)
    DEFAULT_SLIPPAGE_PCT: float = 0.0010    # 0.10% estimated execution slippage
    
    # Currency
    CURRENCY: str = "EGP"


@dataclass
class StrategyConfig:
    """Strategy parameters and technical indicator settings."""
    fast_period: int = 20
    slow_period: int = 50
    
    # Convergence threshold (for Strategy B & C) as fraction of peak distance (e.g., 0.10 for 10%)
    convergence_threshold: float = 0.10
    
    # Normalized gap threshold (for Strategy D) as absolute percentage (e.g., 0.10 for 0.10%)
    normalized_gap_threshold: float = 0.10
    
    # Confirmation sessions for Strategy C (consecutive EMA fast declines)
    confirmation_sessions: int = 2
    
    # Initial portfolio capital in EGP
    initial_capital: float = 100_000.0
    
    # Execution price model: 'next_open' (zero lookahead), 'same_close'
    execution_price_model: str = "next_open"
    
    # Position sizing: 'all_in' (100% available capital), 'fixed_capital', 'pct_capital'
    position_sizing: str = "all_in"
    fixed_capital_amount: float = 50_000.0
    capital_allocation_pct: float = 1.0


@dataclass
class BacktestConfig:
    """Backtesting runner configuration."""
    market: EGXMarketConfig = field(default_factory=EGXMarketConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    
    # Force close open trade at backtest end for final accounting
    force_close_at_end: bool = True
