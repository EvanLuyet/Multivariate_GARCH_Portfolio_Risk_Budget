"""
layer3_optimizer/black_litterman.py — Black-Litterman + Mean-Variance Optimizer.

Pipeline:
  1. Equilibrium returns Π = δ Σ w
  2. View matrix Q = ML forecasts annualised (×4), P = identity
  3. BL posterior μ_BL = [(τΣ)⁻¹ + Ω⁻¹]⁻¹[(τΣ)⁻¹Π + Ω⁻¹Q]
     Ω is diagonal, scaled by the LGB IQR per asset:
       Ω_ii = ((q75_i - q25_i) × 4 / 1.349)²   (IQR → annualised std → variance)
     Wider forecast interval = less confident view = more reversion toward Π.
     Falls back to Ω = τΣ when ML forecasts are unavailable.
  4. SLSQP maximise Sharpe(μ_BL, Σ) subject to WEIGHT_BOUNDS + TC penalty
  5. Regime overlay: blend toward BASE_WEIGHTS using HMM posterior probs

Returns two weight sets:
  optimal_weights  — regime-blended final recommendation (Scenario A: rebalance)
  raw_bl_weights   — pure optimizer output before regime blend
"""

import sys
import logging
import warnings
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime
from scipy.optimize import minimize

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    ASSETS,
    BASE_WEIGHTS,
    WEIGHT_BOUNDS,
    TRADING_DAYS,
    RISK_FREE_RATE,
    DELTA,
    TAU,
    LAMBDA_TC,
    MODEL_CACHE_DIR,
    CONFIG_HASH,
)

W_BASE_ARR = np.array([BASE_WEIGHTS[a] for a in ASSETS])
BOUNDS = [WEIGHT_BOUNDS[a] for a in ASSETS]


def _equilibrium_returns(Sigma_annual: np.ndarray, w: np.ndarray) -> np.ndarray:
    return DELTA * Sigma_annual @ w


def _bl_posterior(
    Pi: np.ndarray, Sigma_annual: np.ndarray, Q: np.ndarray, view_iqr: np.ndarray | None = None
) -> np.ndarray:
    """
    BL posterior with P=I and Omega scaled by ML forecast uncertainty.

    view_iqr: annualised (q75 - q25) per asset from LightGBM quantile models.
      Wider IQR → larger Omega_ii → posterior reverts more toward equilibrium Π.
      Falls back to Ω = τΣ when view_iqr is None (no ML forecasts available).
    """
    tau_S = TAU * Sigma_annual
    inv_tauS = np.linalg.inv(tau_S)

    if view_iqr is not None and len(view_iqr) == len(Q):
        # IQR → std approximation: std ≈ IQR / 1.349 (normal distribution)
        view_var = (view_iqr / 1.349) ** 2
        # Floor at a small fraction of τΣ_ii to prevent infinite confidence
        floor = TAU * np.diag(Sigma_annual) * 0.01
        view_var = np.maximum(view_var, floor)
        Omega = np.diag(view_var)
    else:
        Omega = tau_S  # fallback: equal prior/view uncertainty

    inv_Omega = np.linalg.inv(Omega)
    M = np.linalg.inv(inv_tauS + inv_Omega)
    return M @ (inv_tauS @ Pi + inv_Omega @ Q)


def _optimize_weights(mu: np.ndarray, Sigma_annual: np.ndarray, w_current: np.ndarray) -> tuple:
    """SLSQP Sharpe maximisation with turnover penalty, multi-start."""

    def neg_sharpe(w):
        ret = float(w @ mu)
        vol = np.sqrt(max(float(w @ Sigma_annual @ w), 1e-12))
        tc = LAMBDA_TC * np.sum(np.abs(w - w_current))
        return -(ret - RISK_FREE_RATE) / vol + tc

    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    starts = [
        w_current,
        W_BASE_ARR,
        np.array([b[0] for b in BOUNDS]),
        np.array([b[1] for b in BOUNDS]) / sum(b[1] for b in BOUNDS),
    ]

    best_res, best_sharpe = None, -np.inf
    for w0 in starts:
        w0 = np.clip(w0, [b[0] for b in BOUNDS], [b[1] for b in BOUNDS])
        w0 = w0 / w0.sum()
        try:
            res = minimize(
                neg_sharpe,
                w0,
                method="SLSQP",
                bounds=BOUNDS,
                constraints=constraints,
                options={"ftol": 1e-9, "maxiter": 500},
            )
            if res.success and -res.fun > best_sharpe:
                best_sharpe = -res.fun
                best_res = res
        except Exception:
            pass

    if best_res is None:
        return w_current.copy(), float(-neg_sharpe(w_current))
    w_opt = np.clip(best_res.x, 0.0, 1.0)
    w_opt /= w_opt.sum()
    return w_opt, best_sharpe


def _optimize_min_variance(Sigma_annual: np.ndarray) -> np.ndarray:
    """Minimum-variance portfolio (no return input needed)."""

    def port_var(w):
        return float(w @ Sigma_annual @ w)

    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    w0 = np.array([b[0] for b in BOUNDS])
    w0 = np.clip(w0 / w0.sum(), [b[0] for b in BOUNDS], [b[1] for b in BOUNDS])
    w0 /= w0.sum()
    res = minimize(
        port_var,
        w0,
        method="SLSQP",
        bounds=BOUNDS,
        constraints=constraints,
        options={"ftol": 1e-9, "maxiter": 500},
    )
    w_opt = np.clip(res.x if res.success else w0, 0.0, 1.0)
    return w_opt / w_opt.sum()


def _optimize_risk_parity(Sigma_annual: np.ndarray) -> np.ndarray:
    """
    Risk-parity (equal risk contribution) via SLSQP.
    Objective: sum_i (RC_i - 1/n)² — minimise deviation from equal risk.
    """
    n = len(ASSETS)
    target = 1.0 / n

    def rp_objective(w):
        port_var = float(w @ Sigma_annual @ w)
        port_vol = np.sqrt(max(port_var, 1e-12))
        mrc = Sigma_annual @ w / port_vol
        rc = w * mrc / port_vol
        return float(np.sum((rc - target) ** 2))

    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    w0 = W_BASE_ARR.copy()
    res = minimize(
        rp_objective,
        w0,
        method="SLSQP",
        bounds=BOUNDS,
        constraints=constraints,
        options={"ftol": 1e-9, "maxiter": 1000},
    )
    w_opt = np.clip(res.x if res.success else w0, 0.0, 1.0)
    return w_opt / w_opt.sum()


def _optimize_max_return(
    mu: np.ndarray, Sigma_annual: np.ndarray, vol_target: float = 0.12
) -> np.ndarray:
    """
    Maximum expected return subject to annualised vol ≤ vol_target.
    Falls back to max-Sharpe if the vol constraint is infeasible.
    """
    constraints = [
        {"type": "eq", "fun": lambda w: w.sum() - 1.0},
        {
            "type": "ineq",
            "fun": lambda w: vol_target - np.sqrt(max(float(w @ Sigma_annual @ w), 1e-12)),
        },
    ]
    w0 = W_BASE_ARR.copy()
    res = minimize(
        lambda w: -float(w @ mu),
        w0,
        method="SLSQP",
        bounds=BOUNDS,
        constraints=constraints,
        options={"ftol": 1e-9, "maxiter": 500},
    )
    if res.success:
        w_opt = np.clip(res.x, 0.0, 1.0)
        return w_opt / w_opt.sum()
    # Fallback to max-Sharpe
    w_opt, _ = _optimize_weights(mu, Sigma_annual, W_BASE_ARR)
    return w_opt


def _regime_blend(w_bl: np.ndarray, regime_probs: list) -> np.ndarray:
    """
    Soft blend: α = p_Bull×1.0 + p_Trans×0.5 + p_Crisis×0.25
    Crisis → revert 75% to base weights; Bull → trust optimizer fully.
    """
    p0, p1, p2 = regime_probs
    alpha = float(np.clip(p0 * 1.0 + p1 * 0.50 + p2 * 0.25, 0.0, 1.0))
    w = alpha * w_bl + (1 - alpha) * W_BASE_ARR
    w = np.clip(w, [b[0] for b in BOUNDS], [b[1] for b in BOUNDS])
    return w / w.sum()


def run_black_litterman(
    garch_history: dict, hmm_history: dict, ml_history: dict, current_weights: dict
) -> dict:
    """
    BL optimizer at every quarter-end.

    Returns:
        history  — {qe: {optimal_weights, raw_bl_weights, bl_returns,
                          equilibrium_returns, sharpe_ratio, regime_label}}
        current  — most recent quarter's result
    """
    cache_dir = Path(MODEL_CACHE_DIR)
    cache_dir.mkdir(exist_ok=True)
    cache_file = cache_dir / f'bl_{CONFIG_HASH}_{datetime.today().strftime("%Y%m%d")}.joblib'

    if cache_file.exists():
        logger.info("Layer 3 — Loading BL history from cache …")
        return joblib.load(cache_file)

    logger.info("Layer 3 — Running Black-Litterman optimizer …")

    q_ends = sorted(set(garch_history.keys()) & set(hmm_history.keys()))
    w_prev = np.array([current_weights[a] for a in ASSETS])
    history = {}

    for qe in q_ends:
        g = garch_history[qe]
        hmm = hmm_history[qe]

        Sigma_a = g["cov_matrix"] * TRADING_DAYS
        Pi = _equilibrium_returns(Sigma_a, w_prev)

        ml = ml_history.get(qe, {}).get("return_forecasts")
        if ml is not None:
            # Use bl_point (XGB conditional mean) for BL views; fall back to q50
            Q = np.array([ml[a].get("bl_point", ml[a]["point"]) * 4.0 for a in ASSETS])
            # Annualised IQR per asset: (q75 - q25) × 4 for quarterly → annual scaling
            view_iqr = np.array(
                [(ml[a].get("high", 0.05) - ml[a].get("low", -0.05)) * 4.0 for a in ASSETS]
            )
            mu = _bl_posterior(Pi, Sigma_a, Q, view_iqr=view_iqr)
        else:
            mu = Pi.copy()
            Q = Pi.copy()

        w_opt, sharpe = _optimize_weights(mu, Sigma_a, w_prev)
        w_final = _regime_blend(w_opt, hmm["regime_probs"])

        # Alternative objectives (stored for comparison; not used by default)
        w_minvar = _optimize_min_variance(Sigma_a)
        w_rp = _optimize_risk_parity(Sigma_a)
        from config import TARGET_VOL

        w_maxret = _optimize_max_return(mu, Sigma_a, vol_target=TARGET_VOL)

        history[qe] = dict(
            optimal_weights={a: float(w_final[i]) for i, a in enumerate(ASSETS)},
            raw_bl_weights={a: float(w_opt[i]) for i, a in enumerate(ASSETS)},
            bl_returns={a: float(mu[i]) for i, a in enumerate(ASSETS)},
            equilibrium_returns={a: float(Pi[i]) for i, a in enumerate(ASSETS)},
            sharpe_ratio=float(sharpe),
            regime=hmm["regime"],
            regime_label=hmm["regime_label"],
            min_variance_weights={a: float(w_minvar[i]) for i, a in enumerate(ASSETS)},
            risk_parity_weights={a: float(w_rp[i]) for i, a in enumerate(ASSETS)},
            max_return_weights={a: float(w_maxret[i]) for i, a in enumerate(ASSETS)},
        )
        w_prev = w_final.copy()

    current_qe = max(history.keys())
    result = dict(history=history, current=history[current_qe])
    joblib.dump(result, cache_file)
    return result


if __name__ == "__main__":
    from data.fetcher import fetch_data
    from layer1_garch.garch_model import run_garch
    from layer2_hmm.regime_model import run_hmm
    from layer4_ml.return_forecaster import run_forecaster
    from config import BASE_WEIGHTS

    df = fetch_data()
    g1 = run_garch(df, BASE_WEIGHTS)
    hmm = run_hmm(g1, df)
    ml = run_forecaster(g1, hmm["history"], df)
    res = run_black_litterman(g1, hmm["history"], ml["history"], BASE_WEIGHTS)

    c = res["current"]
    print("\nOptimal weights    :", {a: f"{v:.1%}" for a, v in c["optimal_weights"].items()})
    print("BL expected returns:", {a: f"{v:.2%}" for a, v in c["bl_returns"].items()})
    print("Sharpe ratio       :", f"{c['sharpe_ratio']:.2f}")
    print("Regime             :", c["regime_label"])
