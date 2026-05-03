import numpy as np
import pandas as pd

from backtest.backtest_functions import perf, vol_target_hit_rate
from config import W_BASE, TRADING_DAYS


def run_backtest(sig_df, log_ret, labels):
    """
    Apply quarterly dynamic weights to daily returns.
    Returns (strat_daily_returns, bench_daily_returns).
    """
    sig_dates  = sorted(sig_df.index)
    col_w      = [f"w_{l.lower()}" for l in labels]

    strat_ret, bench_ret = [], []

    for k, qe in enumerate(sig_dates):
        w_vec = sig_df.loc[qe, col_w].values.astype(float)
        start = qe
        end   = sig_dates[k + 1] if k + 1 < len(sig_dates) else log_ret.index[-1]
        mask  = (log_ret.index > start) & (log_ret.index <= end)
        period = log_ret.loc[mask, labels]
        if period.empty:
            continue

        simple = np.exp(period.values) - 1
        strat_ret.append(pd.Series(np.log1p(simple @ w_vec),   index=period.index))
        bench_ret.append(pd.Series(np.log1p(simple @ W_BASE),  index=period.index))

    strat_ret = pd.concat(strat_ret).sort_index()
    bench_ret = pd.concat(bench_ret).sort_index()
    return strat_ret, bench_ret


def print_performance(sig_df, log_ret, strat_ret, bench_ret, labels):
    """Compute and print the full performance comparison table."""
    s_ret, s_vol, s_sr, s_mdd = perf(strat_ret)
    b_ret, b_vol, b_sr, b_mdd = perf(bench_ret)

    # Realized vol per quarter (strategy)
    col_w     = [f"w_{l.lower()}" for l in labels]
    sig_dates = sorted(sig_df.index)
    q_vols    = []
    for k, qe in enumerate(sig_dates):
        w_vec  = sig_df.loc[qe, col_w].values.astype(float)
        start  = qe
        end    = sig_dates[k + 1] if k + 1 < len(sig_dates) else log_ret.index[-1]
        mask   = (log_ret.index > start) & (log_ret.index <= end)
        period = log_ret.loc[mask, labels]
        if len(period) < 5:
            continue
        simple = np.exp(period.values) - 1
        r_vol  = (simple @ w_vec).std() * np.sqrt(TRADING_DAYS)
        q_vols.append(r_vol)

    hit_rate = vol_target_hit_rate(q_vols)

    print("=" * 60)
    print(f"{'Metric':<35} {'Strategy':>10} {'Benchmark':>10}")
    print("-" * 60)
    print(f"{'Annualized Return':<35} {s_ret:>10.2%} {b_ret:>10.2%}")
    print(f"{'Annualized Volatility':<35} {s_vol:>10.2%} {b_vol:>10.2%}")
    print(f"{'Sharpe Ratio (rf=2%)':<35} {s_sr:>10.2f} {b_sr:>10.2f}")
    print(f"{'Maximum Drawdown':<35} {s_mdd:>10.2%} {b_mdd:>10.2%}")
    print(f"{'Qtrs vol within ±4% of 12% target':<35} {hit_rate:>10.1%}")
    print("=" * 60 + "\n")
