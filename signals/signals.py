import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from signals.signals_functions import (
    compute_risk_contributions,
    apply_rc_trim,
    apply_vol_signal,
)

from portfolio_config import W_BASE


def compute_signals(garch_results, labels):
    """
    Iterate over every quarter-end in garch_results and produce:
      - Forecasted portfolio volatility
      - Per-asset risk contributions
      - BUY / HOLD / SELL action
      - Final risky + cash weights

    Returns a DataFrame indexed by quarter-end date.
    """
    records = []

    for qe, info in sorted(garch_results.items()):
        Sigma = info["Sigma"]
        w     = W_BASE.copy()

        # Pre-signal portfolio vol (under base weights)
        _, port_vol_base = compute_risk_contributions(w, Sigma)

        # Step 1: trim any overweight-risk asset
        w = apply_rc_trim(w, Sigma)

        # Step 2: vol-targeting signal → final weights
        w_risky, w_cash, action = apply_vol_signal(w, Sigma)

        # Compute final RCs for reporting
        rc_pct, port_vol_final = compute_risk_contributions(w_risky, Sigma)

        row = dict(date=qe, port_vol=port_vol_base, action=action, w_cash=w_cash)
        for i, lbl in enumerate(labels):
            row[f"rc_{lbl.lower()}"] = rc_pct[i]
            row[f"w_{lbl.lower()}"]  = w_risky[i]

        records.append(row)

    return pd.DataFrame(records).set_index("date")


def print_signal_table(sig_df, labels):
    """Print the quarterly signal table to stdout."""
    col_rc = [f"rc_{l.lower()}" for l in labels]
    col_w  = [f"w_{l.lower()}"  for l in labels]

    lbl_rc = [f"{l} RC" for l in labels]
    lbl_w  = [f"w_{l}"  for l in labels]

    header = (
        f"{'Date':<12}| {'Port.Vol':>8} | "
        + " | ".join(f"{h:>6}" for h in lbl_rc)
        + f" | {'Action':>6} | "
        + " | ".join(f"{h:>6}" for h in lbl_w)
        + f" | {'w_Cash':>6}"
    )
    sep = "-" * len(header)

    print("\n" + sep)
    print(header)
    print(sep)

    for dt, row in sig_df.iterrows():
        ts  = pd.Timestamp(dt)
        rc_vals = " | ".join(f"{row[c]:>6.0%}" for c in col_rc)
        w_vals  = " | ".join(f"{row[c]:>6.1%}" for c in col_w)
        print(
            f"{ts.year}-Q{ts.quarter:<8}| {row.port_vol:>7.1%} | "
            f"{rc_vals} | {row.action:>6} | {w_vals} | {row.w_cash:>6.1%}"
        )

    print(sep + "\n")
