import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from portfolio_config import VOL_LOW, VOL_HIGH, VOL_TARGET

OUTPUT = "results/portfolio_risk_budget.png"

_COLORS = {"BUY": "#2ecc71", "HOLD": "#3498db", "SELL": "#e74c3c"}
_ASSET_COLORS = ["#2c3e50", "#3498db", "#e67e22"]


def make_plots(sig_df, strat_ret, bench_ret, labels):
    fig, axes = plt.subplots(4, 1, figsize=(14, 22))
    fig.suptitle(
        "Multivariate GARCH Portfolio Risk Budget",
        fontsize=15, fontweight="bold", y=0.995,
    )

    dates_q   = [pd.Timestamp(d) for d in sig_df.index]
    col_rc    = [f"rc_{l.lower()}" for l in labels]
    col_w     = [f"w_{l.lower()}"  for l in labels]

    os.makedirs("results", exist_ok=True)

    # ── Plot 1: Cumulative Returns ──────────────────────────────────────────────
    ax = axes[0]
    cum_strat = np.exp(strat_ret).cumprod()
    cum_bench = np.exp(bench_ret).cumprod()
    ax.semilogy(cum_strat.index, cum_strat.values, label="Dynamic Strategy",
                color="#2c3e50", lw=1.8)
    ax.semilogy(cum_bench.index, cum_bench.values, label="Static 70/15/15",
                color="#95a5a6", lw=1.4, ls="--")
    ax.set_title("Cumulative Returns (log scale)", fontweight="bold")
    ax.set_ylabel("Growth of $1")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── Plot 2: Quarterly Portfolio Vol Forecast ────────────────────────────────
    ax = axes[1]
    vols_q     = sig_df["port_vol"].values
    bar_colors = [_COLORS[a] for a in sig_df["action"]]
    ax.bar(dates_q, vols_q * 100, color=bar_colors, width=70, alpha=0.85, edgecolor="white")
    ax.axhline(VOL_LOW    * 100, color="#2ecc71", ls="--", lw=1.2)
    ax.axhline(VOL_HIGH   * 100, color="#e74c3c", ls="--", lw=1.2)
    ax.axhline(VOL_TARGET * 100, color="#3498db", ls=":",  lw=1.2)
    legend_handles = [mpatches.Patch(color=c, label=l) for l, c in _COLORS.items()] + [
        plt.Line2D([0], [0], color="#2ecc71", ls="--", label="10% BUY threshold"),
        plt.Line2D([0], [0], color="#e74c3c", ls="--", label="14% SELL threshold"),
        plt.Line2D([0], [0], color="#3498db", ls=":",  label="12% target"),
    ]
    ax.legend(handles=legend_handles, fontsize=8, ncol=3)
    ax.set_title("Quarterly Portfolio Volatility Forecast", fontweight="bold")
    ax.set_ylabel("Annualized Vol (%)")
    ax.grid(True, alpha=0.3, axis="y")

    # ── Plot 3: Risk Contributions ──────────────────────────────────────────────
    ax = axes[2]
    rc_data = [sig_df[c].values * 100 for c in col_rc]
    ax.stackplot(dates_q, *rc_data, labels=labels, colors=_ASSET_COLORS, alpha=0.85)
    ax.set_title("Risk Contributions over Time", fontweight="bold")
    ax.set_ylabel("Risk Contribution (%)")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Plot 4: Dynamic Weights ─────────────────────────────────────────────────
    ax = axes[3]
    w_data = [sig_df[c].values * 100 for c in col_w] + [sig_df["w_cash"].values * 100]
    ax.stackplot(dates_q, *w_data, labels=labels + ["Cash"],
                 colors=_ASSET_COLORS + ["#bdc3c7"], alpha=0.85)
    ax.set_title("Dynamic Weights over Time", fontweight="bold")
    ax.set_ylabel("Weight (%)")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.995])
    fig.savefig(OUTPUT, dpi=150, bbox_inches="tight")
    print(f"Saved plot → {OUTPUT}")
