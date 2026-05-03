import numpy as np

TRADING_DAYS = 252
VOL_TARGET   = 0.12
VOL_LOW      = 0.10
VOL_HIGH     = 0.14
SCALE_CAP    = 1.30
RC_TRIM_THR  = 0.60
RC_TRIM_AMT  = 0.20


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


def apply_rc_trim(w, Sigma):
    """
    Trim any asset whose risk contribution exceeds RC_TRIM_THR of total portfolio
    risk by RC_TRIM_AMT, redistributing the trimmed weight to the other assets.
    """
    w = w.copy()
    rc_pct, _ = compute_risk_contributions(w, Sigma)
    for i in range(len(w)):
        if rc_pct[i] > RC_TRIM_THR:
            trim   = w[i] * RC_TRIM_AMT
            w[i]  -= trim
            others = [j for j in range(len(w)) if j != i]
            total_others = w[others].sum()
            if total_others > 1e-8:
                w[others] += trim * (w[others] / total_others)
    return w


def apply_vol_signal(w, Sigma):
    """
    Scale risky weights based on the forecasted portfolio vol vs the 12% target.
    Returns (w_risky, w_cash, action).  All weights sum to 1.
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

    w_risky = np.clip(w * scale, 0.0, 1.0)
    w_cash  = max(0.0, 1.0 - w_risky.sum())

    total   = w_risky.sum() + w_cash
    w_risky /= total
    w_cash  /= total
    return w_risky, w_cash, action
