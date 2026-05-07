# Portfolio Intelligence Engine

A quantitative portfolio management system for a 3-ETF portfolio (S&P 500, Nasdaq 100,
MSCI Europe) with a Swiss investor perspective (functional currency: CHF). Runs quarterly
and produces a full intelligence report: regime detection, FX analysis, stock screening,
optimised weights, IBKR cost breakdown, forecasts, backtest, and risk metrics.

## Architecture

```
Layer 1 — Volatility   GJR-GARCH(1,1) Student-t, rolling 252-day window per asset
Layer 2 — Regime       Gaussian HMM (3-state: Bull/Transition/Crisis) on US daily data
                       Separate lightweight HMMs for US / EU / Swiss market display
Layer 3 — Optimizer    Black-Litterman + SLSQP Sharpe maximisation + regime blend
Layer 4 — Forecaster   Walk-forward XGBoost (mean) + LightGBM quantile (Q+1 and Q+2)
Layer 5 — Costs        IBKR Tiered commission model; Scenario A (rebalance) + B (new capital)
Layer FX — Currency    USDCHF / EURCHF / EURUSD spot, YTD/1Y, directional forecast, CHF drag
Layer Stocks — Screener Regime-filtered factor model (momentum + Sharpe + drawdown)
Report — Dashboard     8-section terminal report + 6-panel PNG dashboard
```

## Portfolio

| Asset       | Ticker | Base Weight |
|-------------|--------|-------------|
| S&P 500     | ^GSPC  | 65 %        |
| Nasdaq 100  | QQQ    | 20 %        |
| MSCI Europe | EZU    | 15 %        |

Weight bounds enforced by the BL optimizer: SP500 40–80 %, NDX 10–40 %, EUROPE 10–30 %.

## Methodology Notes

### Covariance (DCC-style approximation)
`Σ = D R̂ D` where D = diag(σ₁, σ₂, σ₃) from GJR-GARCH and R̂ is the 63-day realized
correlation of standardised GARCH residuals. This is **not** a full DCC-GARCH model —
it is a computationally cheaper approximation that still captures crisis correlation spikes.
Eigenvalue flooring (λ_min ≥ 1e-6) guarantees positive-definiteness.

### Black-Litterman
`μ_BL = [(τΣ)⁻¹ + Ω⁻¹]⁻¹ [(τΣ)⁻¹Π + Ω⁻¹Q]`  
Views Q = XGBoost conditional mean × 4 (annualised from quarterly forecast).  
Ω is scaled by the LightGBM IQR (wider interval = less confident view = more reversion
toward equilibrium). Regime blend: `α·w_BL + (1−α)·w_base` where
`α = p_Bull×1.0 + p_Trans×0.5 + p_Crisis×0.25`.

### ETF Forecasts
Point estimate = LGB q50 (median), guaranteed within [q25, q75] by construction.
XGB conditional mean (`bl_point`) is used only inside the BL optimizer — never displayed.

### Stock Screener
Factor z-scores: mom_1m / mom_3m / mom_6m / Sharpe_6m / low_drawdown_6m.
Weights tilt by regime (Bull → momentum heavy; Crisis → drawdown heavy).
`mom_3m_z` in the output is the **normalised z-score**, not a raw percentage return.

## Report sections

| # | Section | What it shows |
|---|---------|---------------|
| ① | Market Regimes | US / EU / Swiss Bull/Trans/Crisis probs and trend |
| ② | Currencies | USDCHF, EURCHF, EURUSD spot + YTD/1Y + CHF drag on portfolio |
| ③ | Top 5 Stocks | Regime-filtered picks with factor scores |
| ④ | Weight Recommendations | Scenario A (rebalance) and B (new capital) |
| ⑤ | Transaction Costs | IBKR commission + FX + spread, breakeven, rebalance decision |
| ⑥ | ETF Forecasts | Q+1 and Q+2 with 25th/75th percentile intervals |
| ⑦ | Backtest | 3-way: Base 65/20/15 vs Current weights vs Proposed BL weights |
| ⑧ | Risk Assessment | RC per ETF, Effective N, daily VaR/CVaR at 95 % and 99 % |

## Installation

```bash
pip install -r requirements.txt

# Optional: FRED macro data (VIX, yield curve, credit spread, USD index)
export FRED_API_KEY=your_key_here
```

## Usage

```bash
# First run (or after code changes) — clears model cache
python main.py --refresh

# Normal quarterly run
python main.py

# Deploy new capital (Scenario B)
python main.py --capital 10000

# Specify portfolio value for IBKR cost calculation
python main.py --capital 10000 --portfolio-value 150000

# Non-interactive mode (no "Did you rebalance?" prompt, for scripts/CI)
python main.py --non-interactive
```

After each run the engine asks _"Did you execute this rebalance?"_ and saves the actual
executed weights to `portfolio_state.json` for accurate next-run comparison.
Use `--non-interactive` to skip this prompt and auto-save the proposed weights.

## Stack

Python 3.11 · yfinance · arch · hmmlearn · xgboost · lightgbm · scikit-learn ·
pandas · numpy · scipy · matplotlib · joblib · fredapi · pyarrow · tqdm

## Cache invalidation

Model caches are keyed by `{config_hash}_{date}`. If you change `BASE_WEIGHTS`,
`GARCH_WINDOW`, `HMM_STATES`, `XGB_PARAMS`, or any other key parameter in `config.py`,
the stale cache is automatically bypassed on the next run — no `--refresh` needed.
To force a full re-fit regardless, use `--refresh`.

## Academic Context

Built for coursework in financial econometrics covering ARCH/GARCH theory, volatility
forecasting (MSE/QLIKE evaluation), portfolio risk budgeting, and Hidden Markov Models.
Key references: Engle (2003), Black & Litterman (1992), Glosten-Jagannathan-Runkle (1993).
