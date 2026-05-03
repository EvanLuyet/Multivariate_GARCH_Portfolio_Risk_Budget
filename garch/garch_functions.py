import numpy as np
import pandas as pd
from arch import arch_model


def get_quarter_ends(index):
    """Return the last trading day of each calendar quarter in index."""
    s = pd.Series(index, index=index)
    return s.groupby([s.index.year, s.index.quarter]).last().values


def fit_gjr_garch(series):
    """
    Fit GJR-GARCH(1,1) with Student-t innovations.
    Returns 1-step-ahead daily sigma and the fitted result object.
    Raises RuntimeError if the optimizer did not converge.
    """
    am  = arch_model(series * 100, vol="GARCH", p=1, o=1, q=1, dist="t", mean="Zero")
    res = am.fit(disp="off", show_warning=False)
    if res.convergence_flag != 0:
        raise RuntimeError(
            f"GJR-GARCH did not converge (convergence_flag={res.convergence_flag})"
        )
    fc  = res.forecast(horizon=1, reindex=False)
    sigma_daily = np.sqrt(fc.variance.values[-1, 0]) / 100
    return sigma_daily, res


def build_covariance(sigmas, corr_window_df):
    """
    Build conditional covariance matrix via the DCC approximation: Σ = D R̂ D.
    corr_window_df: DataFrame of standardized GARCH residuals (columns = assets).
    sigmas: array of daily sigma forecasts per asset.
    """
    R = corr_window_df.corr().values

    # Ensure positive definiteness by flooring eigenvalues
    eigvals, eigvecs = np.linalg.eigh(R)
    eigvals = np.maximum(eigvals, 1e-6)
    R = eigvecs @ np.diag(eigvals) @ eigvecs.T

    # Re-normalise to a proper correlation matrix
    d = np.sqrt(np.diag(R))
    R = R / np.outer(d, d)

    D     = np.diag(sigmas)
    Sigma = D @ R @ D
    return Sigma, R
