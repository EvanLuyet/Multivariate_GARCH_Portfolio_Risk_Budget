# GARCH Portfolio Risk Monitor 

A quantitative portfolio management tool that uses GARCH-family models to forecast 
volatility and generate quarterly Buy/Hold/Sell signals for a multi-asset equity portfolio.

## Overview
This project implements a full portfolio risk budgeting pipeline on a 3-asset portfolio
(S&P 500 · Nasdaq 100 · MSCI Europe) with the following architecture:

- **Volatility Forecasting** — GJR-GARCH(1,1) with Student-t innovations fitted 
  independently on each asset via a rolling 252-day window
- **Correlation Layer** — Rolling 63-day realized correlation matrix built from 
  standardized GARCH residuals (DCC-GARCH approximation)
- **Portfolio Risk Engine** — Conditional covariance matrix used to compute forecasted 
  portfolio volatility and per-asset Risk Contributions (RC)
- **Signal Generation** — Quarterly Buy/Hold/Sell signals based on portfolio vol regime 
  vs. a 12% annualized target, with automatic weight trimming when any single asset 
  exceeds 60% of total portfolio risk
- **Backtesting** — Full historical backtest vs. static 70/15/15 benchmark with Sharpe, 
  drawdown, and vol-targeting accuracy metrics

## Portfolio
| Asset         | Ticker | Base Weight |
|---------------|--------|-------------|
| S&P 500       | ^GSPC  | 70%         |
| Nasdaq 100    | QQQ    | 15%         |
| MSCI Europe   | EZU    | 15%         |

## Signal Logic
| Forecasted Portfolio Vol | Signal | Action                              |
|--------------------------|--------|-------------------------------------|
| < 10%                    | BUY    | Scale up weights (max 1.3x lever)   |
| 10% – 14%                | HOLD   | Maintain base weights               |
| > 14%                    | SELL   | Scale down, remainder goes to cash  |

## Output
- Quarterly signal table (vol forecast · risk contributions · weights · action)
- 4-panel plot: cumulative returns · vol forecast · risk contributions · dynamic weights

## Stack
Python · yfinance · arch · pandas · numpy · matplotlib · scipy · tqdm

## Academic Context
Built alongside coursework in financial econometrics — covers ARCH/GARCH theory, 
volatility forecasting (MSE loss, h-step ahead forecasts), and portfolio risk budgeting.
Reference: Engle (2003 Nobel) ARCH model family.
