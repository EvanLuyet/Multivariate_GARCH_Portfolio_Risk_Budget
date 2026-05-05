"""
report/dashboard.py — Terminal intelligence report + 6-panel matplotlib dashboard.

Prints a concise, human-readable summary of every layer's output to the terminal
and saves a 6-panel PNG capturing the full portfolio history.
"""

import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import to_rgba

warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import ASSETS, TRADING_DAYS, RISK_FREE_RATE, TARGET_VOL, VOL_LOW, VOL_HIGH

REGIME_COLORS = {0: '#2ecc71', 1: '#f39c12', 2: '#e74c3c'}
ASSET_COLORS  = ['#2c3e50', '#3498db', '#e67e22']
CASH_COLOR    = '#bdc3c7'


# ── Risk metrics ──────────────────────────────────────────────────────────────

def _compute_var_es(daily_returns: pd.Series, confidence: float = 0.95):
    """
    Historical simulation VaR and Expected Shortfall.

    VaR(95%) is the daily loss exceeded only 5% of the time in the historical
    sample — a regulatory standard measure of tail risk.
    ES (CVaR) is the average loss given that VaR has been breached, capturing
    the severity of tail events beyond the VaR threshold.
    """
    cutoff = np.percentile(daily_returns, (1 - confidence) * 100)
    var    = -cutoff
    es     = -daily_returns[daily_returns <= cutoff].mean()
    return float(var), float(es)


def _build_strategy_returns(garch_history: dict, bl_history: dict,
                             data: pd.DataFrame) -> tuple:
    """
    Apply historical BL weights quarter-by-quarter to daily returns.
    Returns (strategy_daily_returns, benchmark_daily_returns) as pd.Series.
    """
    from config import BASE_WEIGHTS
    W_BENCH = np.array([BASE_WEIGHTS[a] for a in ASSETS])

    q_ends  = sorted(bl_history.keys())
    strat, bench = [], []

    for k, qe in enumerate(q_ends):
        w_vec = np.array([bl_history[qe]['optimal_weights'][a] for a in ASSETS])
        end   = q_ends[k + 1] if k + 1 < len(q_ends) else data.index[-1]
        mask  = (data.index > qe) & (data.index <= end)
        ret_cols = [f'ret_{a}' for a in ASSETS]
        period   = data.loc[mask, ret_cols]
        if period.empty:
            continue
        strat.append(pd.Series(period.values @ w_vec,   index=period.index))
        bench.append(pd.Series(period.values @ W_BENCH, index=period.index))

    return pd.concat(strat).sort_index(), pd.concat(bench).sort_index()


def _perf_metrics(r: pd.Series) -> dict:
    ann_ret = r.mean() * TRADING_DAYS
    ann_vol = r.std()  * np.sqrt(TRADING_DAYS)
    sharpe  = (ann_ret - RISK_FREE_RATE) / ann_vol if ann_vol > 1e-8 else np.nan
    cum     = (1 + r).cumprod()
    mdd     = ((cum - cum.cummax()) / cum.cummax()).min()
    return dict(ann_ret=ann_ret, ann_vol=ann_vol, sharpe=sharpe, mdd=mdd)


# ── Terminal report ────────────────────────────────────────────────────────────

def print_report(garch_current: dict, hmm_current: dict,
                 ml_current: dict, bl_current: dict,
                 cost_result: dict, final_weights: dict,
                 strat_ret: pd.Series, bench_ret: pd.Series) -> None:
    """Print the full intelligence report to stdout."""
    date_str   = datetime.today().strftime('%Y-%m-%d')
    regime_lbl = hmm_current['regime_label']
    dom_prob   = max(hmm_current['regime_probs'])
    port_vol   = garch_current['port_vol']

    if port_vol < VOL_LOW:
        vol_signal = 'BUY'
    elif port_vol > VOL_HIGH:
        vol_signal = 'SELL'
    else:
        vol_signal = 'HOLD'

    w    = final_weights
    w_bl = bl_current['optimal_weights']
    rc   = garch_current['risk_contributions']
    fc   = ml_current if ml_current else {a: dict(point=0, low=0, high=0) for a in ASSETS}

    strat_r = _perf_metrics(strat_ret)

    # VaR / ES from last year of strategy returns
    last_year = strat_ret.iloc[-TRADING_DAYS:]
    var95, es95 = _compute_var_es(last_year)

    rebal_str = 'YES' if cost_result['rebalance_recommended'] else 'NO'

    border = '═' * 56
    print(f'\n{border}')
    print(f'  PORTFOLIO INTELLIGENCE REPORT — {date_str}')
    print(border)
    print(f'  Regime:         {regime_lbl}  (p={dom_prob:.0%})')
    print(f'  Portfolio Vol:  {port_vol:.1%}  →  {vol_signal}')
    print()
    print(f'  Weights:        SP {w["SP500"]:.0%}  |  NDX {w["NDX"]:.0%}  |  EUR {w["EUROPE"]:.0%}')
    print(f'  Optimal:        SP {w_bl["SP500"]:.0%}  |  NDX {w_bl["NDX"]:.0%}  |  EUR {w_bl["EUROPE"]:.0%}')
    print(f'  Rebalance?      {rebal_str}  (turnover cost: {cost_result["cost_bps"]:.2f} bps)')
    print()
    print('  Return Forecasts (next quarter):')
    for asset, lbl in zip(ASSETS, ['S&P 500   ', 'Nasdaq 100', 'MSCI Eur. ']):
        f = fc.get(asset, dict(point=0, low=0, high=0))
        print(f'    {lbl}  {f["point"]:+.1%}  [{f["low"]:+.1%} / {f["high"]:+.1%}]')
    print()
    print('  Risk:')
    print(f'    VaR (95%):          {var95:.2%} daily')
    print(f'    Expected Shortfall: {es95:.2%} daily')
    print(f'    Risk Contributions: SP {rc["SP500"]:.0%} | NDX {rc["NDX"]:.0%} | EUR {rc["EUROPE"]:.0%}')
    print()
    print('  Backtest vs Static 70/15/15:')
    bench_r = _perf_metrics(bench_ret)
    print(f'    {"":20}  {"Strategy":>10}  {"Benchmark":>10}')
    print(f'    {"Ann. Return":20}  {strat_r["ann_ret"]:>10.2%}  {bench_r["ann_ret"]:>10.2%}')
    print(f'    {"Ann. Vol":20}  {strat_r["ann_vol"]:>10.2%}  {bench_r["ann_vol"]:>10.2%}')
    print(f'    {"Sharpe":20}  {strat_r["sharpe"]:>10.2f}  {bench_r["sharpe"]:>10.2f}')
    print(f'    {"Max Drawdown":20}  {strat_r["mdd"]:>10.2%}  {bench_r["mdd"]:>10.2%}')
    print(border + '\n')


# ── 6-panel dashboard ─────────────────────────────────────────────────────────

def save_dashboard(garch_history: dict, hmm_result: dict,
                   bl_history: dict, ml_result: dict,
                   strat_ret: pd.Series, bench_ret: pd.Series) -> str:
    """
    Build and save a 6-panel portfolio intelligence dashboard.

    Panel 1 — Cumulative returns: strategy vs static benchmark (log scale)
    Panel 2 — Portfolio vol forecast per quarter, colored by HMM regime
    Panel 3 — Dynamic BL weights over time (stacked area)
    Panel 4 — Risk contributions over time (stacked area)
    Panel 5 — HMM regime history (colored background bands)
    Panel 6 — XGBoost feature importance (top 10 features, horizontal bars)
    """
    q_ends  = sorted(garch_history.keys())
    dates_q = [pd.Timestamp(d) for d in q_ends]

    hmm_hist  = hmm_result['history']
    regime_s  = hmm_result['regime_series']
    probs_df  = hmm_result['probs_df']
    feat_imp  = ml_result.get('feature_importance', pd.Series(dtype=float))

    fig, axes = plt.subplots(3, 2, figsize=(18, 18))
    fig.suptitle('Portfolio Intelligence Engine — Full Stack Report',
                 fontsize=16, fontweight='bold', y=1.002)
    axes = axes.flatten()

    # ── Panel 1: Cumulative returns ───────────────────────────────────────────
    ax = axes[0]
    cum_s = (1 + strat_ret).cumprod()
    cum_b = (1 + bench_ret).cumprod()
    ax.semilogy(cum_s.index, cum_s.values, label='BL Strategy', color='#2c3e50', lw=2)
    ax.semilogy(cum_b.index, cum_b.values, label='Static 70/15/15',
                color='#95a5a6', lw=1.5, ls='--')
    ax.set_title('Cumulative Returns (log scale)', fontweight='bold')
    ax.set_ylabel('Growth of $1')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)

    # ── Panel 2: Quarterly vol forecast colored by regime ────────────────────
    ax = axes[1]
    vols   = [garch_history[qe]['port_vol'] for qe in q_ends]
    colors = [REGIME_COLORS.get(hmm_hist[qe]['regime'], '#3498db')
              if qe in hmm_hist else '#3498db' for qe in q_ends]
    ax.bar(dates_q, np.array(vols) * 100, color=colors, width=70, alpha=0.85, edgecolor='white')
    ax.axhline(VOL_LOW    * 100, color='#2ecc71', ls='--', lw=1.2, label='10% BUY')
    ax.axhline(VOL_HIGH   * 100, color='#e74c3c', ls='--', lw=1.2, label='14% SELL')
    ax.axhline(TARGET_VOL * 100, color='#3498db', ls=':',  lw=1.2, label='12% target')
    patches = [mpatches.Patch(color=c, label=l)
               for l, c in [('Bull', '#2ecc71'), ('Transition', '#f39c12'), ('Crisis', '#e74c3c')]]
    ax.legend(handles=patches, fontsize=8, loc='upper left')
    ax.set_title('Quarterly Portfolio Vol Forecast (regime-colored)', fontweight='bold')
    ax.set_ylabel('Annualized Vol (%)')
    ax.grid(True, alpha=0.25, axis='y')

    # ── Panel 3: Dynamic BL weights ──────────────────────────────────────────
    ax   = axes[2]
    bl_q = [qe for qe in q_ends if qe in bl_history]
    bl_dates = [pd.Timestamp(d) for d in bl_q]
    w_data = [[bl_history[qe]['optimal_weights'][a] * 100 for qe in bl_q] for a in ASSETS]
    ax.stackplot(bl_dates, *w_data, labels=ASSETS,
                 colors=ASSET_COLORS, alpha=0.85)
    ax.set_title('Dynamic BL Weights over Time', fontweight='bold')
    ax.set_ylabel('Weight (%)')
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.25)
    ax.set_ylim(0, 105)

    # ── Panel 4: Risk contributions ───────────────────────────────────────────
    ax = axes[3]
    rc_data = [[garch_history[qe]['risk_contributions'][a] * 100 for qe in q_ends]
               for a in ASSETS]
    ax.stackplot(dates_q, *rc_data, labels=ASSETS, colors=ASSET_COLORS, alpha=0.85)
    ax.set_title('Risk Contributions over Time', fontweight='bold')
    ax.set_ylabel('Risk Contribution (%)')
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.25)

    # ── Panel 5: HMM regime history ───────────────────────────────────────────
    ax = axes[4]
    if not regime_s.empty:
        rs = regime_s.reindex(sorted(regime_s.index))
        for i in range(len(rs) - 1):
            t0, t1 = pd.Timestamp(rs.index[i]), pd.Timestamp(rs.index[i + 1])
            ax.axvspan(t0, t1, alpha=0.4,
                       color=REGIME_COLORS.get(int(rs.iloc[i]), '#999999'), lw=0)
        if len(rs) > 0:
            t_last = pd.Timestamp(rs.index[-1])
            ax.axvspan(t_last, pd.Timestamp(strat_ret.index[-1]),
                       alpha=0.4, color=REGIME_COLORS.get(int(rs.iloc[-1]), '#999999'), lw=0)
    # Overlay probabilities as lines
    if not probs_df.empty:
        for i, (lbl, col) in enumerate(
                zip(['Bull p', 'Transition p', 'Crisis p'],
                    ['#2ecc71', '#f39c12', '#e74c3c'])):
            col_name = f'p{i}'
            if col_name in probs_df.columns:
                ax.plot([pd.Timestamp(d) for d in probs_df.index],
                        probs_df[col_name].values, color=col, lw=1.4, label=lbl)
    patches = [mpatches.Patch(color=c, label=l, alpha=0.5)
               for l, c in [('🟢 Bull', '#2ecc71'), ('🟡 Transition', '#f39c12'),
                             ('🔴 Crisis', '#e74c3c')]]
    ax.legend(handles=patches, fontsize=8, loc='upper left')
    ax.set_title('HMM Regime History', fontweight='bold')
    ax.set_ylabel('Regime Probability')
    ax.set_ylim(-0.05, 1.1)
    ax.grid(True, alpha=0.25)

    # ── Panel 6: Feature importance ───────────────────────────────────────────
    ax = axes[5]
    if not feat_imp.empty:
        top10 = feat_imp.nlargest(10).sort_values()
        bars  = ax.barh(range(len(top10)), top10.values * 100,
                        color='#3498db', alpha=0.8, edgecolor='white')
        ax.set_yticks(range(len(top10)))
        ax.set_yticklabels(top10.index, fontsize=8)
        ax.set_title('XGBoost Feature Importance (top 10, mean across assets)',
                     fontweight='bold')
        ax.set_xlabel('Mean Importance Score (%)')
        ax.grid(True, alpha=0.25, axis='x')
    else:
        ax.text(0.5, 0.5, 'Feature importance unavailable\n(need ≥ MIN_TRAIN_QTRS quarters)',
                ha='center', va='center', transform=ax.transAxes, fontsize=10, color='#777')
        ax.set_title('XGBoost Feature Importance', fontweight='bold')

    fig.tight_layout(rect=[0, 0, 1, 1.0])
    date_tag  = datetime.today().strftime('%Y%m%d')
    out_path  = f'results/portfolio_report_{date_tag}.png'
    Path('results').mkdir(exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'Dashboard saved → {out_path}')
    return out_path


if __name__ == '__main__':
    print('dashboard.py: run main.py to generate the full report.')
