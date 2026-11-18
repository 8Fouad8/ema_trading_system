# 📈 EMA Trading System — Multi-Stock Performance Report

**Market:** Egyptian Exchange (EGX equities: EGAL, ORHD, TMGH)  
**Execution Model:** Zero-lookahead (Signal at Bar $t$, Execution at Bar $t+1$ Open)  
**Frictions:** 0.15% EGX Commission + 0.10% Slippage per leg  

## 1. Executive Summary Table

| Ticker | Strategy | Return % | B&H % | Alpha % | CAGR % | Sharpe | Max DD % | Win Rate % | Profit Factor | Trades |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EGAL | Strategy_A_Standard_Crossover | 60.14 | 185.92 | -125.78 | 26.98 | 0.824 | -31.86 | 40.0 | 3.11 | 5 |
| EGAL | Strategy_B_Convergence_Threshold | 35.42 | 185.92 | -150.51 | 16.62 | 0.651 | -29.87 | 40.0 | 2.63 | 5 |
| EGAL | Strategy_C_Confirmation_Sessions | 8.25 | 185.92 | -177.67 | 4.1 | 0.334 | -12.32 | 40.0 | 1.83 | 5 |
| EGAL | Strategy_D_Normalized_Gap | 64.88 | 185.92 | -121.04 | 28.87 | 0.864 | -31.01 | 40.0 | 3.36 | 5 |
| EGAL | Benchmark_Buy_and_Hold | 184.65 | 185.92 | -1.28 | 70.0 | 1.498 | -26.01 | 100.0 | 999.0 | 1 |
| ORHD | Strategy_A_Standard_Crossover | -46.77 | -31.46 | -15.31 | -27.37 | -0.028 | -72.49 | 40.0 | 0.36 | 5 |
| ORHD | Strategy_B_Convergence_Threshold | -40.61 | -31.46 | -9.14 | -23.22 | 0.06 | -72.49 | 40.0 | 0.52 | 5 |
| ORHD | Strategy_C_Confirmation_Sessions | 73.61 | -31.46 | 105.07 | 32.29 | 1.33 | -11.96 | 80.0 | 11.83 | 5 |
| ORHD | Strategy_D_Normalized_Gap | -48.06 | -31.46 | -16.6 | -28.27 | -0.049 | -72.5 | 40.0 | 0.36 | 5 |
| ORHD | Benchmark_Buy_and_Hold | -31.81 | -31.46 | -0.34 | -17.65 | 0.204 | -78.9 | 0.0 | 0.0 | 1 |
| TMGH | Strategy_A_Standard_Crossover | 25.12 | 38.28 | -13.17 | 12.04 | 0.613 | -23.6 | 40.0 | 2.93 | 5 |
| TMGH | Strategy_B_Convergence_Threshold | 35.85 | 38.28 | -2.44 | 16.81 | 0.807 | -20.62 | 40.0 | 3.75 | 5 |
| TMGH | Strategy_C_Confirmation_Sessions | -0.09 | 38.28 | -38.37 | -0.05 | 0.047 | -13.19 | 40.0 | 0.99 | 5 |
| TMGH | Strategy_D_Normalized_Gap | 31.49 | 38.28 | -6.79 | 14.9 | 0.728 | -20.61 | 40.0 | 3.42 | 5 |
| TMGH | Benchmark_Buy_and_Hold | 37.61 | 38.28 | -0.68 | 17.58 | 0.683 | -23.43 | 100.0 | 999.0 | 1 |

## 2. Strategy Architecture Descriptions
- **Strategy A (Standard 10/20 Crossover):** Buys when EMA 10 crosses above EMA 20; exits when EMA 10 crosses below EMA 20.
- **Strategy B (Convergence Threshold):** Tracks peak distance between EMAs; exits early if distance contracts below 10% of peak, protecting unrealized profits.
- **Strategy C (Confirmation Sessions):** Exits early if fast EMA registers 2 consecutive declining bars, signaling loss of short-term upward momentum.
- **Strategy D (Normalized Gap):** Exits early if the absolute normalized distance between EMAs falls below 0.10%.
- **Benchmark (Buy & Hold):** Buys at the first available bar and holds through the entire backtest window.

## 3. Generated Visualizations
Detailed visual charts and equity curves have been exported to `outputs/charts/`:
- Individual stock detail charts: `outputs/charts/{ticker}_{strategy}.png`
- Comparative stock overlays: `outputs/charts/{ticker}_comparison.png`
- Cross-stock performance summary: `outputs/charts/cross_stock_summary.png`