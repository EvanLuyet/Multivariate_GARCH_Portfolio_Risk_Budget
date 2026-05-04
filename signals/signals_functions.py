import numpy as np

from portfolio_config import TRADING_DAYS, VOL_TARGET, VOL_LOW, VOL_HIGH, SCALE_CAP, RC_TRIM_THR, RC_TRIM_AMT


def compute_risk_contributions(w, Sigma):
    """
    Return annualized risk-contribution fractions (sum to 1).
    rc_pct[i] = fraction of portfolio vol attributed to asset i.
    """
    port_var   = w @ Sigma @ w
    port_vol_d = np.sqrt(port_var)
    port_vol_a = port_vol_d * np.sqrt(TRADING_DAYS)

    mrc    = (Sigma @ w) / (port_vol_d + 1e-10)
    rc_a   = w * mrc * np.sqrt(TRADING_DAYS)
    rc_pct = rc_a / (port_vol_a + 1e-10)
    return rc_pct, port_vol_a


def apply_rc_trim(w, Sigma, max_iter=10):
    """
    Trim any asset whose risk contribution exceeds RC_TRIM_THR of total portfolio
    risk by RC_TRIM_AMT, redistributing the trimmed weight to the other assets.
    Iterates until no asset exceeds the threshold or max_iter is reached.
    """
    w = w.copy()
    for _ in range(max_iter):
        rc_pct, _ = compute_risk_contributions(w, Sigma)
        trimmed = False
        for i in range(len(w)):
            if rc_pct[i] > RC_TRIM_THR:
                trim   = w[i] * RC_TRIM_AMT
                w[i]  -= trim
                others = [j for j in range(len(w)) if j != i]
                total_others = w[others].sum()
                if total_others > 1e-8:
                    w[others] += trim * (w[others] / total_others)
                trimmed = True
                break  # recompute RCs before checking next asset
        if not trimmed:
            break
    return w


def apply_vol_signal(w, Sigma):
    """
    Scale risky weights based on the forecasted portfolio vol vs the 12% target.
    Returns (w_risky, w_cash, action).  Weights sum to 1; w_cash is negative
    when leveraged (BUY with scale > 1 means borrowing cash to hold more risk).
    """
    _, port_vol_a = compute_risk_contributions(w, Sigma)

    if port_vol_a < VOL_LOW:
        scale  = min(VOL_TARGET / port_vol_a, SCALE_CAP)
        action = "BUY"
    elif port_vol_a > VOL_HIGH:
        scale  = VOL_TARGET / port_vol_a
        action = "SELL"
    else:
        scale  = 1.0
        action = "HOLD"

    # No normalization — w_cash absorbs the difference and is negative when
    # scale > 1 (leveraged), positive when scale < 1 (defensive cash buffer).
    w_risky = w * scale
    w_cash  = 1.0 - w_risky.sum()
    return w_risky, w_cash, action
