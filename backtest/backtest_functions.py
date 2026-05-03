import numpy as np

TRADING_DAYS = 252
RF           = 0.02
VOL_TARGET   = 0.12


def perf(r):
    """Return (ann_return, ann_vol, sharpe, max_drawdown) for a daily return series."""
    ann_ret  = r.mean() * TRADING_DAYS
    ann_vol  = r.std()  * np.sqrt(TRADING_DAYS)
    sharpe   = (ann_ret - RF) / ann_vol if ann_vol > 0 else float("nan")
    cum      = (1 + r).cumprod()
    roll_max = cum.cummax()
    mdd      = ((cum - roll_max) / roll_max).min()
    return ann_ret, ann_vol, sharpe, mdd


def vol_target_hit_rate(q_realized_vols, target=VOL_TARGET, band=0.04):
    """Fraction of quarters where realized vol stayed within ±band of target."""
    q_realized_vols = np.asarray(q_realized_vols)
    return float(np.mean(np.abs(q_realized_vols - target) <= band))
