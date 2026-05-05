"""
layer3_optimizer/black_litterman.py — Black-Litterman + Mean-Variance Optimizer.

Black-Litterman (1990) solves the classical MVO problem of extreme, unstable
weights by anchoring the expected return vector to market equilibrium (implied
by current market-cap weights) and then tilting it towards investor views.

Pipeline:
  1. Equilibrium returns Π = δ Σ w  (reverse-engineer returns implied by weights)
  2. View matrix Q = ML quarterly forecasts annualized, P = identity (absolute views)
  3. View uncertainty Ω = τ P Σ P'  (proportional to prior uncertainty)
  4. BL posterior μ_BL = [(τΣ)⁻¹ + P'Ω⁻¹P]⁻¹ [(τΣ)⁻¹Π + P'Ω⁻¹Q]
  5. Mean-variance optimization: maximize Sharpe(μ_BL, Σ) subject to WEIGHT_BOUNDS
     + transaction-cost penalty λ Σ|w_new − w_old|
  6. Regime overlay: blend BL weights with BASE_WEIGHTS based on HMM regime

When ML forecasts are unavailable (early quarters) step 2–4 collapse and the
optimizer uses Π directly as the expected return vector.
"""

import sys
import warnings
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime
from scipy.optimize import minimize

warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (ASSETS, BASE_WEIGHTS, WEIGHT_BOUNDS, TRADING_DAYS,
                    RISK_FREE_RATE, DELTA, TAU, LAMBDA_TC, MODEL_CACHE_DIR)

W_BASE_ARR = np.array([BASE_WEIGHTS[a] for a in ASSETS])
BOUNDS     = [WEIGHT_BOUNDS[a] for a in ASSETS]


# ── Black-Litterman formulas ───────────────────────────────────────────────────

def _equilibrium_returns(Sigma_annual: np.ndarray, w: np.ndarray) -> np.ndarray:
    """
    Reverse-optimize the returns implied by current weights w.
    Π = δ Σ w  — if the market is in equilibrium, these are the expected returns
    that would make a risk-averse investor choose exactly w.
    """
    return DELTA * Sigma_annual @ w


def _bl_posterior(Pi: np.ndarray, Sigma_annual: np.ndarray,
                  Q: np.ndarray) -> np.ndarray:
    """
    Compute the BL posterior expected return vector.

    With P = I (absolute views on every asset):
        μ_BL = [(τΣ)⁻¹ + Ω⁻¹]⁻¹ [(τΣ)⁻¹Π + Ω⁻¹Q]
        Ω    = τ Σ  (uniform view uncertainty proportional to prior)
    This simplifies to a precision-weighted average of Π and Q with equal
    weight on each, which is conservative and stable.
    """
    tau_Sigma = TAU * Sigma_annual
    Omega     = TAU * Sigma_annual                 # P=I so PΣP' = Σ
    inv_tauS  = np.linalg.inv(tau_Sigma)
    inv_Omega = np.linalg.inv(Omega)

    M        = np.linalg.inv(inv_tauS + inv_Omega)
    mu_BL    = M @ (inv_tauS @ Pi + inv_Omega @ Q)
    return mu_BL


# ── MVO with transaction-cost penalty ─────────────────────────────────────────

def _optimize_weights(mu: np.ndarray, Sigma_annual: np.ndarray,
                      w_current: np.ndarray) -> tuple:
    """
    Maximize Sharpe ratio subject to weight bounds + turnover penalty.

    Objective: -(μ'w - rf) / sqrt(w'Σw) + λ Σ|w_new - w_old|
    The penalty discourages unnecessary rebalancing when the gain in Sharpe is
    small relative to the transaction cost.
    """
    n = len(ASSETS)

    def neg_sharpe(w):
        port_ret = float(w @ mu)
        port_var = float(w @ Sigma_annual @ w)
        port_vol = np.sqrt(max(port_var, 1e-12))
        tc       = LAMBDA_TC * np.sum(np.abs(w - w_current))
        return -(port_ret - RISK_FREE_RATE) / port_vol + tc

    constraints = [{'type': 'eq', 'fun': lambda w: w.sum() - 1.0}]

    best_res, best_sharpe = None, -np.inf
    # Try multiple starting points to avoid local optima
    starts = [w_current, W_BASE_ARR,
              np.array([b[0] for b in BOUNDS]),       # lower bounds
              np.array([b[1] for b in BOUNDS]) / sum(b[1] for b in BOUNDS)]

    for w0 in starts:
        w0 = np.clip(w0, [b[0] for b in BOUNDS], [b[1] for b in BOUNDS])
        w0 /= w0.sum()
        try:
            res = minimize(neg_sharpe, w0, method='SLSQP',
                           bounds=BOUNDS, constraints=constraints,
                           options={'ftol': 1e-9, 'maxiter': 500})
            if res.success and -res.fun > best_sharpe:
                best_sharpe = -res.fun
                best_res    = res
        except Exception:
            pass

    if best_res is None or not best_res.success:
        return w_current.copy(), -neg_sharpe(w_current)

    w_opt = np.clip(best_res.x, 0.0, 1.0)
    w_opt /= w_opt.sum()
    return w_opt, best_sharpe


def _regime_blend(w_bl: np.ndarray, regime: int,
                  regime_probs: list) -> np.ndarray:
    """
    Blend BL optimal weights toward base weights based on the HMM regime.

    🟢 Bull      (regime 0): use BL weights fully — trust the model in calm markets
    🟡 Transition (regime 1): blend 50/50 — partial conviction
    🔴 Crisis    (regime 2): 75% base weights — revert to safety in stressed markets

    Crisis blending is the key risk-management feature: it ensures the portfolio
    de-risks automatically when the HMM detects high-stress conditions.
    """
    p0, p1, p2 = regime_probs
    # Soft blending weighted by regime probabilities
    alpha_bl = p0 * 1.0 + p1 * 0.50 + p2 * 0.25
    alpha_bl = float(np.clip(alpha_bl, 0.0, 1.0))
    w_blend  = alpha_bl * w_bl + (1 - alpha_bl) * W_BASE_ARR
    w_blend  = np.clip(w_blend, [b[0] for b in BOUNDS], [b[1] for b in BOUNDS])
    w_blend /= w_blend.sum()
    return w_blend


# ── Main entry point ───────────────────────────────────────────────────────────

def run_black_litterman(garch_history: dict, hmm_history: dict,
                        ml_history: dict, current_weights: dict) -> dict:
    """
    Run BL optimizer at every quarter-end with available GARCH + HMM + ML data.

    Returns:
        history   — {qe: {optimal_weights, bl_returns, equilibrium_returns, sharpe}}
        current   — the most recent optimization result (used in the live report)
    """
    cache_dir  = Path(MODEL_CACHE_DIR)
    cache_dir.mkdir(exist_ok=True)
    cache_file = cache_dir / f'bl_{datetime.today().strftime("%Y%m%d")}.joblib'

    if cache_file.exists():
        print('Layer 3 — Loading BL history from cache …')
        return joblib.load(cache_file)

    print('Layer 3 — Running Black-Litterman optimizer …')

    q_ends     = sorted(set(garch_history.keys()) & set(hmm_history.keys()))
    w_prev     = np.array([current_weights[a] for a in ASSETS])
    history    = {}

    for qe in q_ends:
        g   = garch_history[qe]
        hmm = hmm_history[qe]

        # Annualized covariance
        Sigma_a = g['cov_matrix'] * TRADING_DAYS
        Pi      = _equilibrium_returns(Sigma_a, w_prev)

        # BL views from ML forecasts (quarterly → annualized × 4)
        ml = ml_history.get(qe, {}).get('return_forecasts')
        if ml is not None:
            Q    = np.array([ml[a]['point'] * 4.0 for a in ASSETS])
            mu   = _bl_posterior(Pi, Sigma_a, Q)
        else:
            mu = Pi.copy()
            Q  = Pi.copy()

        # Optimize
        w_opt, sharpe = _optimize_weights(mu, Sigma_a, w_prev)

        # Regime overlay
        w_final = _regime_blend(w_opt, hmm['regime'], hmm['regime_probs'])

        history[qe] = dict(
            optimal_weights    = {a: float(w_final[i]) for i, a in enumerate(ASSETS)},
            raw_bl_weights     = {a: float(w_opt[i])   for i, a in enumerate(ASSETS)},
            bl_returns         = {a: float(mu[i])       for i, a in enumerate(ASSETS)},
            equilibrium_returns= {a: float(Pi[i])       for i, a in enumerate(ASSETS)},
            sharpe_ratio       = float(sharpe),
            regime             = hmm['regime'],
            regime_label       = hmm['regime_label'],
        )
        w_prev = w_final.copy()

    current_qe = max(history.keys())
    result = dict(history=history, current=history[current_qe])

    joblib.dump(result, cache_file)
    return result


if __name__ == '__main__':
    from data.fetcher import fetch_data
    from layer1_garch.garch_model import run_garch
    from layer2_hmm.regime_model import run_hmm
    from layer4_ml.return_forecaster import run_forecaster
    from config import BASE_WEIGHTS

    df   = fetch_data()
    g1   = run_garch(df, BASE_WEIGHTS)
    hmm  = run_hmm(g1, df)
    ml   = run_forecaster(g1, hmm['history'], df)
    res  = run_black_litterman(g1, hmm['history'], ml['history'], BASE_WEIGHTS)

    c = res['current']
    print('\nOptimal weights    :', {a: f"{v:.1%}" for a, v in c['optimal_weights'].items()})
    print('BL expected returns:', {a: f"{v:.2%}" for a, v in c['bl_returns'].items()})
    print('Sharpe ratio       :', f"{c['sharpe_ratio']:.2f}")
    print('Regime             :', c['regime_label'])
