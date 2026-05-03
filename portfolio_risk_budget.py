"""
Multivariate GARCH Portfolio Risk Budget — Buy/Sell Signals
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import yfinance as yf
from arch import arch_model
from tqdm import tqdm
from scipy import stats

warnings.filterwarnings("ignore")

# ── Constants ──────────────────────────────────────────────────────────────────
TICKERS      = ["^GSPC", "QQQ",  "EZU"]
LABELS       = ["S&P",   "NDX",  "EUR"]
W_BASE       = np.array([0.70, 0.15, 0.15])
VOL_TARGET   = 0.12
VOL_LOW      = 0.10
VOL_HIGH     = 0.14
SCALE_CAP    = 1.30
RC_TRIM_THR  = 0.60
RC_TRIM_AMT  = 0.20
ROLL_GARCH   = 252
ROLL_CORR    = 63
TRADING_DAYS = 252
RF           = 0.02


# ── 1. Data ────────────────────────────────────────────────────────────────────
def fetch_data():
    print("Fetching 10 years of daily price data …")
    raw = yf.download(TICKERS, period="10y", auto_adjust=True, progress=False)["Close"]
    raw.columns = LABELS
    raw = raw.dropna()
    log_ret = np.log(raw / raw.shift(1)).dropna()
    return raw, log_ret


# ── 2 & 3. Rolling GJR-GARCH + Correlation ────────────────────────────────────
def get_quarter_ends(index):
    """Return the last trading day of each calendar quarter present in index."""
    s = pd.Series(index, index=index)
    return s.groupby([s.index.year, s.index.quarter]).last().values


def fit_gjr_garch(series):
    """Fit GJR-GARCH(1,1) with Student-t, return 1-step-ahead sigma (daily)."""
    am = arch_model(series * 100, vol="GARCH", p=1, o=1, q=1, dist="t", mean="Zero")
    res = am.fit(disp="off", show_warning=False)
    fc  = res.forecast(horizon=1, reindex=False)
    sigma_daily = np.sqrt(fc.variance.values[-1, 0]) / 100
    return sigma_daily, res


def rolling_garch_and_corr(log_ret):
    """
    At each quarter-end:
      - Fit GJR-GARCH on trailing ROLL_GARCH days for each asset → σ_i (daily)
      - Compute GARCH residuals z = r / σ_i over trailing ROLL_CORR days
      - Build conditional covariance Σ = D R D
    Returns a dict keyed by quarter-end date.
    """
    dates = log_ret.index
    q_ends = get_quarter_ends(dates)

    results = {}
    print("Running rolling GJR-GARCH estimation …")
    for qe in tqdm(q_ends, desc="Quarter-ends"):
        loc = dates.get_loc(qe)
        if loc < ROLL_GARCH:
            continue

        window = log_ret.iloc[loc - ROLL_GARCH + 1: loc + 1]
        sigmas = np.zeros(len(LABELS))
        residuals = pd.DataFrame(index=window.index, columns=LABELS, dtype=float)

        for i, lbl in enumerate(LABELS):
            sigma_daily, res = fit_gjr_garch(window[lbl].values)
            sigmas[i] = sigma_daily
            # In-sample conditional vols (daily)
            cond_vol = np.sqrt(res.conditional_volatility) / 100
            resid    = window[lbl].values / np.where(cond_vol > 1e-8, cond_vol, 1e-8)
            residuals[lbl] = resid

        # Rolling 63-day realized correlation of standardized residuals
        corr_window = residuals.iloc[-ROLL_CORR:]
        R = corr_window.corr().values
        # Ensure positive definiteness (clip eigenvalues)
        eigvals, eigvecs = np.linalg.eigh(R)
        eigvals = np.maximum(eigvals, 1e-6)
        R = eigvecs @ np.diag(eigvals) @ eigvecs.T
        # Re-normalize to correlation matrix
        d = np.sqrt(np.diag(R))
        R = R / np.outer(d, d)

        D = np.diag(sigmas)
        Sigma = D @ R @ D  # conditional covariance (daily)

        results[qe] = dict(sigmas=sigmas, R=R, Sigma=Sigma)

    return results


# ── 4. Portfolio Risk Budget Signal ───────────────────────────────────────────
def compute_signals(garch_results):
    """For each quarter-end, compute port vol, RCs, action, and final weights."""
    records = []

    for qe, info in sorted(garch_results.items()):
        Sigma = info["Sigma"]
        w = W_BASE.copy()

        # Annualized portfolio variance / vol
        port_var    = w @ Sigma @ w
        port_vol_d  = np.sqrt(port_var)
        port_vol_a  = port_vol_d * np.sqrt(TRADING_DAYS)

        # Risk contributions (annualized, fractions of portfolio vol)
        mrc = (Sigma @ w) / (port_vol_d + 1e-10)
        rc  = w * mrc                             # absolute RC (daily)
        rc_a = rc * np.sqrt(TRADING_DAYS)         # annualized
        rc_pct = rc_a / (port_vol_a + 1e-10)      # fraction of port vol

        # ── Trim any asset with RC > 60 % ──
        trimmed = np.zeros(len(LABELS), dtype=bool)
        for i in range(len(LABELS)):
            if rc_pct[i] > RC_TRIM_THR:
                trim = w[i] * RC_TRIM_AMT
                w[i] -= trim
                others = [j for j in range(len(LABELS)) if j != i]
                total_others = w[others].sum()
                if total_others > 1e-8:
                    w[others] += trim * (w[others] / total_others)
                trimmed[i] = True

        # Recompute port vol after trim
        port_var   = w @ Sigma @ w
        port_vol_d = np.sqrt(port_var)
        port_vol_a = port_vol_d * np.sqrt(TRADING_DAYS)

        # ── Determine action & scale ──
        w_cash = 0.0
        if port_vol_a < VOL_LOW:
            scale  = min(VOL_TARGET / port_vol_a, SCALE_CAP)
            action = "BUY"
        elif port_vol_a > VOL_HIGH:
            scale  = VOL_TARGET / port_vol_a
            action = "SELL"
        else:
            scale  = 1.0
            action = "HOLD"

        w_risky = w * scale
        # Clamp individual weights to [0, 1]
        w_risky = np.clip(w_risky, 0.0, 1.0)
        w_cash  = max(0.0, 1.0 - w_risky.sum())
        # Normalise so all sum to 1
        total = w_risky.sum() + w_cash
        w_risky /= total
        w_cash  /= total

        # Final RC under scaled weights (for reporting)
        port_var_f   = w_risky @ Sigma @ w_risky
        port_vol_f   = np.sqrt(port_var_f) * np.sqrt(TRADING_DAYS)
        mrc_f        = (Sigma @ w_risky) / (np.sqrt(port_var_f) + 1e-10)
        rc_f         = w_risky * mrc_f * np.sqrt(TRADING_DAYS)
        rc_pct_f     = rc_f / (port_vol_f + 1e-10)

        records.append(dict(
            date     = qe,
            port_vol = port_vol_a,
            rc_sp    = rc_pct_f[0],
            rc_ndx   = rc_pct_f[1],
            rc_eur   = rc_pct_f[2],
            action   = action,
            w_sp     = w_risky[0],
            w_ndx    = w_risky[1],
            w_eur    = w_risky[2],
            w_cash   = w_cash,
        ))

    return pd.DataFrame(records).set_index("date")


# ── 5. Print Signal Table ──────────────────────────────────────────────────────
def print_table(sig_df):
    header = (
        f"{'Date':<12}| {'Port.Vol':>8} | {'S&P RC':>6} | {'NDX RC':>6} | "
        f"{'EUR RC':>6} | {'Action':>6} | {'w_SP':>6} | {'w_NDX':>6} | "
        f"{'w_EUR':>6} | {'w_Cash':>6}"
    )
    sep = "-" * len(header)
    print("\n" + sep)
    print(header)
    print(sep)
    for dt, row in sig_df.iterrows():
        yr  = pd.Timestamp(dt).year
        qtr = pd.Timestamp(dt).quarter
        print(
            f"{yr}-Q{qtr:<8}| {row.port_vol:>7.1%} | {row.rc_sp:>6.0%} | "
            f"{row.rc_ndx:>6.0%} | {row.rc_eur:>6.0%} | {row.action:>6} | "
            f"{row.w_sp:>6.1%} | {row.w_ndx:>6.1%} | {row.w_eur:>6.1%} | "
            f"{row.w_cash:>6.1%}"
        )
    print(sep + "\n")


# ── 6. Backtesting ─────────────────────────────────────────────────────────────
def run_backtest(sig_df, log_ret):
    """Apply quarterly weights to daily returns; compare to static benchmark."""
    sig_dates = sorted(sig_df.index)
    strat_ret  = []
    bench_ret  = []

    for k, qe in enumerate(sig_dates):
        row   = sig_df.loc[qe]
        w_vec = np.array([row.w_sp, row.w_ndx, row.w_eur])

        # Quarter interval: (current qe, next qe]
        start = qe
        end   = sig_dates[k + 1] if k + 1 < len(sig_dates) else log_ret.index[-1]
        mask  = (log_ret.index > start) & (log_ret.index <= end)
        period_ret = log_ret.loc[mask]
        if period_ret.empty:
            continue

        daily_strat = period_ret[LABELS].values @ w_vec  # cash earns 0 log-ret
        daily_bench = period_ret[LABELS].values @ W_BASE

        strat_ret.append(pd.Series(daily_strat, index=period_ret.index))
        bench_ret.append(pd.Series(daily_bench, index=period_ret.index))

    strat_ret = pd.concat(strat_ret).sort_index()
    bench_ret = pd.concat(bench_ret).sort_index()

    def perf(r):
        ann_ret  = r.mean() * TRADING_DAYS
        ann_vol  = r.std()  * np.sqrt(TRADING_DAYS)
        sharpe   = (ann_ret - RF) / ann_vol if ann_vol > 0 else np.nan
        cum      = (1 + r).cumprod()
        roll_max = cum.cummax()
        mdd      = ((cum - roll_max) / roll_max).min()
        return ann_ret, ann_vol, sharpe, mdd

    s_ret, s_vol, s_sr, s_mdd = perf(strat_ret)
    b_ret, b_vol, b_sr, b_mdd = perf(bench_ret)

    # % quarters where realized vol within ±4% of 12% target
    q_vols = []
    for k, qe in enumerate(sig_dates):
        start = qe
        end   = sig_dates[k + 1] if k + 1 < len(sig_dates) else log_ret.index[-1]
        mask  = (log_ret.index > start) & (log_ret.index <= end)
        period_ret = log_ret.loc[mask]
        if len(period_ret) < 5:
            continue
        w_vec   = np.array([sig_df.loc[qe, "w_sp"], sig_df.loc[qe, "w_ndx"],
                             sig_df.loc[qe, "w_eur"]])
        r_port  = period_ret[LABELS].values @ w_vec
        r_vol   = r_port.std() * np.sqrt(TRADING_DAYS)
        q_vols.append(r_vol)

    q_vols    = np.array(q_vols)
    vol_in_band = np.mean(np.abs(q_vols - VOL_TARGET) <= 0.04)

    print("=" * 60)
    print(f"{'Metric':<35} {'Strategy':>10} {'Benchmark':>10}")
    print("-" * 60)
    print(f"{'Annualized Return':<35} {s_ret:>10.2%} {b_ret:>10.2%}")
    print(f"{'Annualized Volatility':<35} {s_vol:>10.2%} {b_vol:>10.2%}")
    print(f"{'Sharpe Ratio (rf=2%)':<35} {s_sr:>10.2f} {b_sr:>10.2f}")
    print(f"{'Maximum Drawdown':<35} {s_mdd:>10.2%} {b_mdd:>10.2%}")
    print(f"{'Qtrs vol within ±4% of 12% target':<35} {vol_in_band:>10.1%}")
    print("=" * 60 + "\n")

    return strat_ret, bench_ret


# ── 7. Plots ───────────────────────────────────────────────────────────────────
def make_plots(sig_df, strat_ret, bench_ret):
    fig, axes = plt.subplots(4, 1, figsize=(14, 22))
    fig.suptitle("Multivariate GARCH Portfolio Risk Budget", fontsize=15, fontweight="bold", y=0.995)

    colors = {"BUY": "#2ecc71", "HOLD": "#3498db", "SELL": "#e74c3c"}

    # ── Plot 1: Cumulative Returns ──────────────────────────────────────────────
    ax1 = axes[0]
    cum_strat = (1 + strat_ret).cumprod()
    cum_bench = (1 + bench_ret).cumprod()
    ax1.semilogy(cum_strat.index, cum_strat.values, label="Dynamic Strategy", color="#2c3e50", lw=1.8)
    ax1.semilogy(cum_bench.index, cum_bench.values, label="Static 70/15/15", color="#95a5a6", lw=1.4, ls="--")
    ax1.set_title("Cumulative Returns (log scale)", fontweight="bold")
    ax1.set_ylabel("Growth of \$1")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # ── Plot 2: Quarterly Portfolio Vol Forecast ────────────────────────────────
    ax2  = axes[1]
    dates_q = [pd.Timestamp(d) for d in sig_df.index]
    vols_q  = sig_df["port_vol"].values
    bar_colors = [colors[a] for a in sig_df["action"]]
    ax2.bar(dates_q, vols_q * 100, color=bar_colors, width=70, alpha=0.85, edgecolor="white")
    ax2.axhline(VOL_LOW  * 100, color="#2ecc71", ls="--", lw=1.2, label="10% (BUY threshold)")
    ax2.axhline(VOL_HIGH * 100, color="#e74c3c", ls="--", lw=1.2, label="14% (SELL threshold)")
    ax2.axhline(VOL_TARGET * 100, color="#3498db", ls=":",  lw=1.2, label="12% target")
    legend_patches = [mpatches.Patch(color=c, label=l) for l, c in colors.items()]
    ax2.legend(handles=legend_patches + [
        plt.Line2D([0], [0], color="#2ecc71", ls="--", label="10% BUY"),
        plt.Line2D([0], [0], color="#e74c3c", ls="--", label="14% SELL"),
        plt.Line2D([0], [0], color="#3498db", ls=":",  label="12% target"),
    ], fontsize=8, ncol=3)
    ax2.set_title("Quarterly Portfolio Volatility Forecast", fontweight="bold")
    ax2.set_ylabel("Annualized Vol (%)")
    ax2.grid(True, alpha=0.3, axis="y")

    # ── Plot 3: Risk Contributions ──────────────────────────────────────────────
    ax3 = axes[2]
    rc_sp  = sig_df["rc_sp"].values  * 100
    rc_ndx = sig_df["rc_ndx"].values * 100
    rc_eur = sig_df["rc_eur"].values * 100
    ax3.stackplot(dates_q, rc_sp, rc_ndx, rc_eur,
                  labels=["S&P 500", "Nasdaq 100", "MSCI Europe"],
                  colors=["#2c3e50", "#3498db", "#e67e22"], alpha=0.85)
    ax3.set_title("Risk Contributions over Time", fontweight="bold")
    ax3.set_ylabel("Risk Contribution (%)")
    ax3.legend(loc="upper left", fontsize=8)
    ax3.grid(True, alpha=0.3)

    # ── Plot 4: Dynamic Weights ──────────────────────────────────────────────
    ax4 = axes[3]
    w_sp   = sig_df["w_sp"].values   * 100
    w_ndx  = sig_df["w_ndx"].values  * 100
    w_eur  = sig_df["w_eur"].values  * 100
    w_cash = sig_df["w_cash"].values * 100
    ax4.stackplot(dates_q, w_sp, w_ndx, w_eur, w_cash,
                  labels=["S&P 500", "Nasdaq 100", "MSCI Europe", "Cash"],
                  colors=["#2c3e50", "#3498db", "#e67e22", "#bdc3c7"], alpha=0.85)
    ax4.set_title("Dynamic Weights over Time", fontweight="bold")
    ax4.set_ylabel("Weight (%)")
    ax4.legend(loc="upper left", fontsize=8)
    ax4.grid(True, alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.995])
    out = "portfolio_risk_budget.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved plot → {out}")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    _, log_ret = fetch_data()
    garch_res  = rolling_garch_and_corr(log_ret)
    sig_df     = compute_signals(garch_res)
    print_table(sig_df)
    strat_ret, bench_ret = run_backtest(sig_df, log_ret)
    make_plots(sig_df, strat_ret, bench_ret)


if __name__ == "__main__":
    main()
