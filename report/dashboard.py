"""
report/dashboard.py — Terminal intelligence report + 6-panel matplotlib dashboard.

Terminal report sections:
  1. Market Regimes      — US / EU / Swiss separately
  2. Currency (FX)       — USDCHF, EURCHF, EURUSD + CHF drag
  3. Top 5 Stocks        — Regime-filtered stock screener picks
  4. Weight Recommendation — Scenario A (rebalance) + Scenario B (new capital)
  5. Transaction Costs   — IBKR breakdown
  6. ETF Forecasts       — Current quarter (Q+1) and next quarter (Q+2)
  7. Backtest            — 3-way: Base / Current / Proposed weights
  8. Risk Assessment     — VaR, ES, risk contributions, effective N
"""

import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import ASSETS, TRADING_DAYS, RISK_FREE_RATE, TARGET_VOL, VOL_LOW, VOL_HIGH

REGIME_COLORS = {0: "#2ecc71", 1: "#f39c12", 2: "#e74c3c"}
ASSET_COLORS = ["#2c3e50", "#3498db", "#e67e22"]


# ── Risk / performance helpers ────────────────────────────────────────────────


def _compute_var_es(daily_returns: pd.Series, confidence: float = 0.95):
    cutoff = np.percentile(daily_returns, (1 - confidence) * 100)
    var = -cutoff
    es = -daily_returns[daily_returns <= cutoff].mean()
    return float(var), float(es)


def _perf_metrics(r: pd.Series) -> dict:
    ann_ret = r.mean() * TRADING_DAYS
    ann_vol = r.std() * np.sqrt(TRADING_DAYS)
    sharpe = (ann_ret - RISK_FREE_RATE) / ann_vol if ann_vol > 1e-8 else np.nan
    sortino_denom = r[r < 0].std() * np.sqrt(TRADING_DAYS) if (r < 0).any() else 1e-8
    sortino = (ann_ret - RISK_FREE_RATE) / sortino_denom
    cum = (1 + r).cumprod()
    mdd = float(((cum - cum.cummax()) / cum.cummax()).min())
    calmar = ann_ret / abs(mdd) if mdd != 0 else np.nan
    return dict(
        ann_ret=ann_ret, ann_vol=ann_vol, sharpe=sharpe, sortino=sortino, mdd=mdd, calmar=calmar
    )


def _build_strategy_returns(
    garch_history: dict,
    bl_history: dict,
    data: pd.DataFrame,
    weights_override: dict = None,
    include_tc: bool = True,
) -> pd.Series:
    """
    Build daily return series for a given weight history or static weights_override.

    weights_override = {asset: weight} applies fixed weights for the full period.
    include_tc       = True deducts a flat TRANSACTION_COST on each quarterly
                       rebalance day (applies to BL strategy only, not overrides).
    """
    from config import TRANSACTION_COST

    q_ends = sorted(bl_history.keys())
    series = []
    w_prev = None

    for k, qe in enumerate(q_ends):
        if weights_override:
            w_vec = np.array([weights_override[a] for a in ASSETS])
        else:
            w_vec = np.array([bl_history[qe]["optimal_weights"][a] for a in ASSETS])

        end = q_ends[k + 1] if k + 1 < len(q_ends) else data.index[-1]
        mask = (data.index > qe) & (data.index <= end)
        period = data.loc[mask, [f"ret_{a}" for a in ASSETS]]
        if period.empty:
            w_prev = w_vec
            continue

        daily = pd.Series(period.values @ w_vec, index=period.index)

        # Deduct transaction cost on the first day of each new period
        if include_tc and not weights_override and w_prev is not None:
            turnover = float(np.abs(w_vec - w_prev).sum())
            tc_drag = turnover * TRANSACTION_COST
            daily.iloc[0] -= tc_drag

        series.append(daily)
        w_prev = w_vec

    return pd.concat(series).sort_index() if series else pd.Series(dtype=float)


def compute_garch_vol_evaluation(garch_history: dict, data: pd.DataFrame) -> dict:
    """
    Compare GARCH one-step-ahead vol forecasts against realised volatility.

    Realised vol for quarter Q_k → Q_{k+1} is the annualised std of daily
    log-returns in that period.

    Returns dict with per-asset MSE and QLIKE:
      MSE   = mean((σ_forecast² - σ_realised²)²)
      QLIKE = mean(σ_forecast²/σ_realised² - log(σ_forecast²/σ_realised²) - 1)
    """
    q_ends = sorted(garch_history.keys())
    results = {a: {"mse": [], "qlike": []} for a in ASSETS}

    for k in range(len(q_ends) - 1):
        qe = q_ends[k]
        next_qe = q_ends[k + 1]
        mask = (data.index > qe) & (data.index <= next_qe)
        period = data[mask]
        if len(period) < 5:
            continue

        for asset in ASSETS:
            forecast_var = garch_history[qe]["vol_forecasts"][asset] ** 2
            realised_var = (float(period[f"ret_{asset}"].std()) * np.sqrt(TRADING_DAYS)) ** 2
            if realised_var < 1e-10:
                continue
            results[asset]["mse"].append((forecast_var - realised_var) ** 2)
            ratio = forecast_var / realised_var
            results[asset]["qlike"].append(ratio - np.log(ratio) - 1)

    summary = {}
    for asset in ASSETS:
        mse_list = results[asset]["mse"]
        qlike_list = results[asset]["qlike"]
        summary[asset] = dict(
            mse=float(np.mean(mse_list)) if mse_list else np.nan,
            qlike=float(np.mean(qlike_list)) if qlike_list else np.nan,
            n_obs=len(mse_list),
        )
    return summary


# ── Terminal report ───────────────────────────────────────────────────────────


def print_report(
    garch_current: dict,
    hmm_current: dict,
    market_regimes: dict,
    fx_result: dict,
    stock_picks: list,
    ml_current: dict,
    ml_next_q: dict,
    bl_current: dict,
    cost_result: dict,
    current_weights: dict,
    final_weights: dict,
    strat_ret: pd.Series,
    bench_ret: pd.Series,
    current_w_ret: pd.Series,
    vol_eval: dict | None = None,
) -> None:

    border = "═" * 60
    thin = "─" * 60
    date_str = datetime.today().strftime("%Y-%m-%d")

    print(f"\n{border}")
    print(f"  PORTFOLIO INTELLIGENCE REPORT — {date_str}")
    print(border)

    # ── 1. Market Regimes ─────────────────────────────────────────────────────
    print("\n  ① MARKET REGIMES")
    print(thin)
    for market, info in market_regimes.items():
        probs = info["regime_probs"]
        trend_sym = {"improving": "↗", "deteriorating": "↘", "stable": "→"}.get(info["trend"], "→")
        print(
            f'  {market:6s}: {info["regime_label"]:18s} '
            f"Bull {probs[0]:.0%} / Trans {probs[1]:.0%} / Crisis {probs[2]:.0%}  "
            f'{trend_sym} {info["trend"]}'
        )

    port_vol = garch_current["port_vol"]
    vol_sig = (
        "BUY ↑ add equity"
        if port_vol < VOL_LOW
        else "SELL ↓ reduce equity" if port_vol > VOL_HIGH else "HOLD — within target"
    )
    print(f"\n  Portfolio Vol: {port_vol:.1%}  (target {TARGET_VOL:.0%})  →  {vol_sig}")

    # ── 2. Currency ───────────────────────────────────────────────────────────
    print("\n  ② CURRENCIES (CHF perspective)")
    print(thin)
    for line in fx_result.get("summary_lines", ["  FX data unavailable"]):
        print(line)

    # ── 3. Top 5 Stocks ───────────────────────────────────────────────────────
    print("\n  ③ TOP 5 STOCK OPPORTUNITIES (next quarter)")
    print(thin)
    regime_name = hmm_current.get("regime_label", "?")
    print(f"  Filtered for {regime_name} regime:")
    for p in stock_picks:
        if p["ticker"] == "N/A":
            print(f'    {p["note"]}')
            break
        print(
            f'    {p["rank"]}. {p["ticker"]:8s} {p["sector"]:22s} '
            f'3m_z {p["mom_3m_z"]:+.3f}  Sharpe_z {p["sharpe_z"]:+.2f}  '
            f'— {p["note"]}'
        )

    # ── 4. Weight Recommendations ─────────────────────────────────────────────
    print("\n  ④ WEIGHT RECOMMENDATIONS")
    print(thin)
    w_opt = bl_current["optimal_weights"]
    print(
        "  Current weights:   " + "  ".join(f"{a} {current_weights.get(a, 0):.0%}" for a in ASSETS)
    )
    print(
        "  Base weights:      "
        + "  ".join(f"{a} {v:.0%}" for a, v in __import__("config").BASE_WEIGHTS.items())
    )
    print("  Scenario A (Rebalance):")
    print("    Proposed:        " + "  ".join(f"{a} {w_opt[a]:.0%}" for a in ASSETS))
    print(
        "    Change:          "
        + "  ".join(f"{a} {w_opt[a] - current_weights.get(a, 0):+.0%}" for a in ASSETS)
    )

    sb = cost_result.get("scenario_b")
    if sb and sb["new_capital_usd"] > 0:
        rw = sb["resulting_weights"]
        print(f'  Scenario B (Deploy ${sb["new_capital_usd"]:,.0f}):')
        print("    Result weights:  " + "  ".join(f"{a} {rw.get(a, 0):.0%}" for a in ASSETS))
        print("    Buys:")
        for a, info in sb["buys"].items():
            if info["amount"] > 0:
                print(
                    f'      {a}: ${info["amount"]:,.0f} ({info["shares"]:.2f} shares)  '
                    f'commission ${info["commission"]:.2f}'
                )

    # ── 5. Transaction Costs ──────────────────────────────────────────────────
    print("\n  ⑤ TRANSACTION COSTS (IBKR)")
    print(thin)
    sa = cost_result["scenario_a"]
    rebal_str = "YES — execute" if cost_result["rebalance_recommended"] else "NO — defer"
    print(f'  Turnover:          {cost_result["turnover"]:.1%}')
    print(f'  Commission:        ${sa["commission_usd"]:.2f}')
    print(f'  FX conversion:     ${sa["fx_cost_usd"]:.2f}')
    print(f'  Spread:            ${sa["spread_usd"]:.2f}')
    print(f'  Total cost:        ${sa["total_cost_usd"]:.2f}  ({cost_result["cost_bps"]:.2f} bps)')
    print(f'  Expected gain:     {cost_result["expected_gain"]:.3%}  (quarterly)')
    beq = sa.get("breakeven_qtrs")
    print(f"  Breakeven:         {beq} quarters" if beq else "  Breakeven:         N/A")
    bd = cost_result.get("band_breach", {})
    drifted = [a for a, v in bd.items() if v]
    print(
        f'  Band breaches:     {", ".join(drifted) if drifted else "None (no asset drifted >5%)"}'
    )
    print(f"  Rebalance?         {rebal_str}")

    # ── 6. ETF Forecasts ──────────────────────────────────────────────────────
    print("\n  ⑥ ETF FORECASTS")
    print(thin)
    labels = {"SP500": "S&P 500   ", "NDX": "Nasdaq 100", "EUROPE": "MSCI Eur. "}
    print(f'  {"":12}  {"Current Qtr (Q+1)":>20}  {"Next Qtr (Q+2)":>20}')
    for a in ASSETS:
        fc1 = (ml_current or {}).get(a, dict(point=0, low=0, high=0))
        fc2 = (ml_next_q or {}).get(a, dict(point=0, low=0, high=0))
        print(
            f"  {labels[a]}  "
            f'{fc1["point"]:+.1%} [{fc1["low"]:+.1%}/{fc1["high"]:+.1%}]     '
            f'{fc2["point"]:+.1%} [{fc2["low"]:+.1%}/{fc2["high"]:+.1%}]'
        )

    # ── 7. Backtest ───────────────────────────────────────────────────────────
    print("\n  ⑦ BACKTEST  (3-way comparison)")
    print(thin)
    cur_w_m = _perf_metrics(current_w_ret) if not current_w_ret.empty else None
    strat_m = _perf_metrics(strat_ret) if not strat_ret.empty else None
    bench_m = _perf_metrics(bench_ret) if not bench_ret.empty else None

    col_w = 12
    print(
        f'  {"":22}  {"Base (65/20/15)":>{col_w}}  {"Current Wts":>{col_w}}  {"Proposed":>{col_w}}'
    )
    for metric, label, fmt in [
        ("ann_ret", "Ann. Return", "{:.2%}"),
        ("ann_vol", "Ann. Vol", "{:.2%}"),
        ("sharpe", "Sharpe", "{:.2f}"),
        ("sortino", "Sortino", "{:.2f}"),
        ("mdd", "Max Drawdown", "{:.2%}"),
        ("calmar", "Calmar", "{:.2f}"),
    ]:

        def _fmtv(m):
            if m is None:
                return "N/A".rjust(col_w)
            v = m.get(metric, np.nan)
            if np.isnan(v):
                return "N/A".rjust(col_w)
            return fmt.format(v).rjust(col_w)

        print(f"  {label:22}  {_fmtv(bench_m)}  {_fmtv(cur_w_m)}  {_fmtv(strat_m)}")

    # ── 8. Risk Assessment ────────────────────────────────────────────────────
    print("\n  ⑧ RISK ASSESSMENT")
    print(thin)
    rc = garch_current["risk_contributions"]
    last_year = strat_ret.iloc[-TRADING_DAYS:] if len(strat_ret) >= TRADING_DAYS else strat_ret
    var95, es95 = _compute_var_es(last_year)
    var99, es99 = _compute_var_es(last_year, confidence=0.99)

    rc_arr = np.array([rc[a] for a in ASSETS])
    eff_n = 1.0 / (rc_arr**2).sum() if (rc_arr**2).sum() > 0 else 1.0

    print("  Risk Contributions: " + "  ".join(f"{a} {rc[a]:.0%}" for a in ASSETS))
    print(f"  Effective N (diversification): {eff_n:.2f} / {len(ASSETS)}")
    print(f"  VaR  95% (daily): {var95:.2%}   |   VaR  99%: {var99:.2%}")
    print(f"  CVaR 95% (daily): {es95:.2%}   |   CVaR 99%: {es99:.2%}")

    if vol_eval:
        print("\n  GARCH Vol Forecast Evaluation (out-of-sample, quarterly):")
        for a in ASSETS:
            ev = vol_eval.get(a, {})
            mse = ev.get("mse", float("nan"))
            qlike = ev.get("qlike", float("nan"))
            n = ev.get("n_obs", 0)
            print(f"    {a:8s}  MSE={mse:.2e}  QLIKE={qlike:.4f}  (n={n})")

    print(
        "\n  Final weights applied: "
        + "  ".join(f"{a} {final_weights.get(a, 0):.0%}" for a in ASSETS)
    )
    print(border + "\n")


# ── 6-panel Dashboard PNG ─────────────────────────────────────────────────────


def save_dashboard(
    garch_history: dict,
    hmm_result: dict,
    bl_history: dict,
    ml_result: dict,
    market_regimes: dict,
    fx_result: dict,
    strat_ret: pd.Series,
    bench_ret: pd.Series,
    current_w_ret: pd.Series,
) -> str:

    q_ends = sorted(garch_history.keys())
    dates_q = [pd.Timestamp(d) for d in q_ends]
    regime_s = hmm_result["regime_series"]
    probs_df = hmm_result["probs_df"]
    feat_imp = ml_result.get("feature_importance", pd.Series(dtype=float))

    fig, axes = plt.subplots(3, 2, figsize=(18, 18))
    fig.suptitle(
        "Portfolio Intelligence Engine — Full Stack Report", fontsize=16, fontweight="bold"
    )
    axes = axes.flatten()

    # ── Panel 1: 3-way cumulative returns ─────────────────────────────────────
    ax = axes[0]
    for ret, label, color, lw, ls in [
        (strat_ret, "BL Proposed", "#2c3e50", 2.0, "-"),
        (current_w_ret, "Current Weights", "#e67e22", 1.8, "--"),
        (bench_ret, "Base 65/20/15", "#95a5a6", 1.5, ":"),
    ]:
        if not ret.empty:
            cum = (1 + ret).cumprod()
            ax.semilogy(cum.index, cum.values, label=label, color=color, lw=lw, ls=ls)
    ax.set_title("Cumulative Returns — 3-way (log scale)", fontweight="bold")
    ax.set_ylabel("Growth of $1")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)

    # ── Panel 2: Per-market regime bars ───────────────────────────────────────
    ax = axes[1]
    markets = list(market_regimes.keys())
    y_pos = np.arange(len(markets))
    for i, market in enumerate(markets):
        info = market_regimes[market]
        probs = info["regime_probs"]
        colors = ["#2ecc71", "#f39c12", "#e74c3c"]
        left = 0.0
        for j, (p, c) in enumerate(zip(probs, colors)):
            ax.barh(i, p, left=left, color=c, alpha=0.85, edgecolor="white")
            if p > 0.1:
                ax.text(
                    left + p / 2,
                    i,
                    f"{p:.0%}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    fontweight="bold",
                    color="white",
                )
            left += p
        ax.text(
            1.02,
            i,
            info["regime_label"],
            va="center",
            fontsize=9,
            transform=ax.get_yaxis_transform(),
        )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(markets)
    ax.set_xlim(0, 1)
    ax.set_title("Market Regime Probabilities (US / EU / Swiss)", fontweight="bold")
    ax.set_xlabel("Probability")
    patches = [
        mpatches.Patch(color=c, label=lbl)
        for c, lbl in [("#2ecc71", "Bull"), ("#f39c12", "Transition"), ("#e74c3c", "Crisis")]
    ]
    ax.legend(handles=patches, fontsize=8, loc="lower right")

    # ── Panel 3: Dynamic BL weights over time ─────────────────────────────────
    ax = axes[2]
    bl_q = [qe for qe in q_ends if qe in bl_history]
    bl_dates = [pd.Timestamp(d) for d in bl_q]
    w_data = [[bl_history[qe]["optimal_weights"][a] * 100 for qe in bl_q] for a in ASSETS]
    ax.stackplot(bl_dates, *w_data, labels=ASSETS, colors=ASSET_COLORS, alpha=0.85)
    ax.set_title("Dynamic BL Weights over Time", fontweight="bold")
    ax.set_ylabel("Weight (%)")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.25)
    ax.set_ylim(0, 105)

    # ── Panel 4: Risk contributions ───────────────────────────────────────────
    ax = axes[3]
    rc_data = [[garch_history[qe]["risk_contributions"][a] * 100 for qe in q_ends] for a in ASSETS]
    ax.stackplot(dates_q, *rc_data, labels=ASSETS, colors=ASSET_COLORS, alpha=0.85)
    ax.set_title("Risk Contributions over Time", fontweight="bold")
    ax.set_ylabel("Risk Contribution (%)")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.25)

    # ── Panel 5: HMM US regime history with probability lines ─────────────────
    ax = axes[4]
    if not regime_s.empty:
        rs = regime_s.sort_index()
        for i in range(len(rs) - 1):
            t0 = pd.Timestamp(rs.index[i])
            t1 = pd.Timestamp(rs.index[i + 1])
            ax.axvspan(t0, t1, alpha=0.3, color=REGIME_COLORS.get(int(rs.iloc[i]), "#999"), lw=0)
    if not probs_df.empty:
        for i, (lbl, col) in enumerate(
            zip(["Bull p", "Transition p", "Crisis p"], ["#2ecc71", "#f39c12", "#e74c3c"])
        ):
            if f"p{i}" in probs_df.columns:
                ax.plot(
                    [pd.Timestamp(d) for d in probs_df.index],
                    probs_df[f"p{i}"].values,
                    color=col,
                    lw=1.4,
                    label=lbl,
                )
    ax.set_title("US HMM Regime History (daily posteriors)", fontweight="bold")
    ax.set_ylabel("Probability")
    ax.set_ylim(-0.05, 1.1)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.25)

    # ── Panel 6: XGBoost feature importance ───────────────────────────────────
    ax = axes[5]
    if not feat_imp.empty:
        top10 = feat_imp.nlargest(10).sort_values()
        ax.barh(
            range(len(top10)), top10.values * 100, color="#3498db", alpha=0.8, edgecolor="white"
        )
        ax.set_yticks(range(len(top10)))
        ax.set_yticklabels(top10.index, fontsize=8)
        ax.set_title("XGBoost Feature Importance (top 10)", fontweight="bold")
        ax.set_xlabel("Mean Importance Score (%)")
        ax.grid(True, alpha=0.25, axis="x")
    else:
        ax.text(
            0.5,
            0.5,
            "Feature importance unavailable\n(need ≥ MIN_TRAIN_QTRS quarters)",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=10,
            color="#777",
        )
        ax.set_title("XGBoost Feature Importance", fontweight="bold")

    fig.tight_layout()
    date_tag = datetime.today().strftime("%Y%m%d")
    out_path = f"results/portfolio_report_{date_tag}.png"
    Path("results").mkdir(exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Dashboard saved → {out_path}")
    return out_path


if __name__ == "__main__":
    print("dashboard.py: run main.py to generate the full report.")
