import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from tqdm import tqdm

from garch.garch_functions import get_quarter_ends, fit_gjr_garch, build_covariance
from portfolio_config import ROLL_GARCH, ROLL_CORR


def rolling_garch_and_corr(log_ret, labels):
    """
    At every quarter-end date:
      1. Fit GJR-GARCH on trailing ROLL_GARCH days → per-asset daily sigma
      2. Compute in-sample standardized residuals z = r / σ
      3. Build 63-day realized correlation of z → conditional covariance Σ = D R̂ D

    Returns a dict  {date: {"sigmas": ..., "R": ..., "Sigma": ...}}.
    """
    dates  = log_ret.index
    q_ends = get_quarter_ends(dates)

    results = {}
    print("Running rolling GJR-GARCH estimation …")
    for qe in tqdm(q_ends, desc="Quarter-ends"):
        loc = dates.get_loc(qe)
        if loc < ROLL_GARCH:
            continue

        window = log_ret.iloc[loc - ROLL_GARCH + 1 : loc + 1]
        sigmas    = np.zeros(len(labels))
        residuals = pd.DataFrame(index=window.index, columns=labels, dtype=float)

        for i, lbl in enumerate(labels):
            sigma_daily, res = fit_gjr_garch(window[lbl].values)
            sigmas[i] = sigma_daily

            cond_vol     = np.sqrt(res.conditional_volatility) / 100
            cond_vol     = np.where(cond_vol > 1e-8, cond_vol, 1e-8)
            residuals[lbl] = window[lbl].values / cond_vol

        corr_window      = residuals.iloc[-ROLL_CORR:]
        Sigma, R         = build_covariance(sigmas, corr_window)
        results[qe]      = dict(sigmas=sigmas, R=R, Sigma=Sigma)

    return results
