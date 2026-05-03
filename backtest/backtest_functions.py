import numpy as np

from config import TRADING_DAYS, RF, VOL_TARGET


def perf(r):
    """Return (ann_return, ann_vol, sharpe, max_drawdown) for a daily log-return series."""
    ann_ret  = np.exp(r.mean() * TRADING_DAYS) - 1   # geometric annualized return
    ann_vol  = r.std() * np.sqrt(TRADING_DAYS)
    sharpe   = (ann_ret - RF) / ann_vol if ann_vol > 0 else float("nan")
    cum      = np.exp(r).cumprod()                    # exact for log returns
    roll_max = cum.cummax()
    mdd      = ((cum - roll_max) / roll_max).min()
    return ann_ret, ann_vol, sharpe, mdd


def vol_target_hit_rate(q_realized_vols, target=VOL_TARGET, band=0.04):
    """Fraction of quarters where realized vol stayed within ±band of target."""
    q_realized_vols = np.asarray(q_realized_vols)
    return float(np.mean(np.abs(q_realized_vols - target) <= band))
