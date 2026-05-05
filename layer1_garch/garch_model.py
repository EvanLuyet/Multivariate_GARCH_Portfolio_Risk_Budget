"""
layer1_garch/garch_model.py — GJR-GARCH volatility forecasts + DCC covariance.

We fit a GJR-GARCH(1,1) with Student-t innovations on each asset independently.
The GJR extension (Glosten-Jagannathan-Runkle) adds an asymmetric gamma term:
when returns are negative, the variance shock is amplified relative to an
equivalent positive shock — the well-documented 'leverage effect' in equities.

The conditional covariance matrix is built via the DCC approximation:
    Σ = D R̂ D,   D = diag(σ₁, σ₂, σ₃)
where R̂ is the realized correlation of the standardized GARCH residuals over
the trailing CORR_WINDOW days. This captures crisis correlation spikes cheaply
without fitting a full multivariate GARCH system.

Results are joblib-cached by date so the 120+ model fits run only once per day.
"""

import sys
import warnings
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime
from arch import arch_model
from tqdm import tqdm

warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (ASSETS, GARCH_WINDOW, CORR_WINDOW, TRADING_DAYS,
                    MODEL_CACHE_DIR, BASE_WEIGHTS)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _quarter_ends(index: pd.DatetimeIndex) -> np.ndarray:
    """Last trading day of every calendar quarter present in index."""
    s = pd.Series(index, index=index)
    return s.groupby([s.index.year, s.index.quarter]).last().values


def _fit_gjr_garch(series: np.ndarray):
    """
    Fit GJR-GARCH(1,1) / Student-t on a scaled return series.

    Returns (fitted_result, h_next_daily) where h_next_daily is the
    1-step-ahead conditional variance in un-scaled daily units.
    The Student-t distribution captures fat tails common in equity returns.
    """
    am  = arch_model(series * 100, vol='GARCH', p=1, o=1, q=1,
                     dist='t', mean='Zero')
    res = am.fit(disp='off', show_warning=False)
    fc  = res.forecast(horizon=1, reindex=False)
    h_next = fc.variance.values[-1, 0] / 1e4   # back to daily unscaled variance
    return res, h_next


def _build_cov(daily_sigmas: np.ndarray, resid_df: pd.DataFrame):
    """
    Build the conditional covariance matrix Σ = D R̂ D.

    Eigenvalue flooring (λ_min ≥ 1e-6) guarantees positive definiteness,
    which is required for the BL optimizer and risk-contribution formulas.
    """
    R = resid_df.corr().values
    eigvals, eigvecs = np.linalg.eigh(R)
    eigvals = np.maximum(eigvals, 1e-6)
    R = eigvecs @ np.diag(eigvals) @ eigvecs.T
    d = np.sqrt(np.diag(R))
    R /= np.outer(d, d)

    D     = np.diag(daily_sigmas)
    Sigma = D @ R @ D   # daily covariance
    return Sigma, R


def compute_risk_contributions(w: np.ndarray, Sigma_daily: np.ndarray):
    """
    Per-asset risk contributions: RC_i = w_i·(Σw)_i / σ_port.

    Each RC_i measures what fraction of annualized portfolio volatility is
    attributable to asset i. By construction RC_i sum to 1.
    Returns (rc_fractions, annualized_port_vol).
    """
    port_var   = float(w @ Sigma_daily @ w)
    port_vol_d = np.sqrt(max(port_var, 1e-14))
    port_vol_a = port_vol_d * np.sqrt(TRADING_DAYS)
    mrc    = (Sigma_daily @ w) / port_vol_d
    rc     = w * mrc
    rc_pct = rc / (rc.sum() + 1e-14)
    return rc_pct, port_vol_a


# ── Main entry point ───────────────────────────────────────────────────────────

def run_garch(data: pd.DataFrame, weights: dict) -> dict:
    """
    Rolling GJR-GARCH + DCC covariance at every quarter-end in the data.

    At each quarter-end date t we:
      1. Fit GJR-GARCH on the trailing GARCH_WINDOW days for each asset.
      2. Extract in-sample standardized residuals z_t = r_t / σ_t.
      3. Build the DCC covariance from the CORR_WINDOW trailing realized
         correlation of z_t.
      4. Compute portfolio vol and per-asset risk contributions.

    Results are cached to MODEL_CACHE_DIR/garch_YYYYMMDD.joblib.
    """
    cache_dir  = Path(MODEL_CACHE_DIR)
    cache_dir.mkdir(exist_ok=True)
    cache_file = cache_dir / f'garch_{datetime.today().strftime("%Y%m%d")}.joblib'

    if cache_file.exists():
        print('Layer 1 — Loading GARCH history from cache …')
        return joblib.load(cache_file)

    ret_cols = [f'ret_{a}' for a in ASSETS]
    dates    = data.index
    q_ends   = _quarter_ends(dates)
    w_arr    = np.array([weights[a] for a in ASSETS])

    history = {}
    print('Layer 1 — Running rolling GJR-GARCH …')
    for qe in tqdm(q_ends, desc='GARCH quarter-ends'):
        loc = dates.get_loc(qe)
        if loc < GARCH_WINDOW:
            continue

        window    = data[ret_cols].iloc[loc - GARCH_WINDOW + 1 : loc + 1]
        sigmas_d  = np.zeros(len(ASSETS))   # daily sigmas
        sigmas_a  = np.zeros(len(ASSETS))   # annualized
        residuals = pd.DataFrame(index=window.index, columns=ASSETS, dtype=float)

        for i, asset in enumerate(ASSETS):
            col = f'ret_{asset}'
            res, h_next   = _fit_gjr_garch(window[col].values)
            sigmas_d[i]   = np.sqrt(h_next)
            sigmas_a[i]   = np.sqrt(h_next * TRADING_DAYS)

            cond_vol          = np.sqrt(res.conditional_volatility) / 100
            cond_vol          = np.where(cond_vol > 1e-8, cond_vol, 1e-8)
            residuals[asset]  = window[col].values / cond_vol

        Sigma, R = _build_cov(sigmas_d, residuals.iloc[-CORR_WINDOW:])
        rc_pct, port_vol_a = compute_risk_contributions(w_arr, Sigma)

        history[qe] = dict(
            vol_forecasts      = {a: float(sigmas_a[i]) for i, a in enumerate(ASSETS)},
            daily_sigmas       = {a: float(sigmas_d[i]) for i, a in enumerate(ASSETS)},
            cov_matrix         = Sigma,          # daily covariance (3×3)
            corr_matrix        = R,
            port_vol           = float(port_vol_a),
            risk_contributions = {a: float(rc_pct[i]) for i, a in enumerate(ASSETS)},
            residuals          = residuals,
        )

    joblib.dump(history, cache_file)
    return history


if __name__ == '__main__':
    from data.fetcher import fetch_data

    df  = fetch_data()
    res = run_garch(df, BASE_WEIGHTS)
    qe  = max(res.keys())
    r   = res[qe]
    print(f'\nLatest quarter-end : {qe.date()}')
    print(f'Vol forecasts (ann): {r["vol_forecasts"]}')
    print(f'Portfolio vol      : {r["port_vol"]:.2%}')
    print(f'Risk contributions : {r["risk_contributions"]}')
