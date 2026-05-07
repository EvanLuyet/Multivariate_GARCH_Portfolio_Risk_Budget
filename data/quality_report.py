"""
data/quality_report.py — Data quality diagnostics.

Checks:
  - Date range and number of trading days per column
  - Missing value count and percentage per column
  - Proxy indicators: EU/Swiss benchmarks falling back to SP500
  - FRED macro series availability
  - Stale data warning if last row is >5 trading days old

Call print_data_quality_report(data) after fetch_data() to get a summary.
"""

import sys
import logging
import warnings
import pandas as pd
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import FRED_SERIES


def compute_data_quality(data: pd.DataFrame) -> dict:
    """
    Analyse a fetched data DataFrame and return quality metadata.

    Returns
    -------
    dict with:
        date_range      — (first_date, last_date)
        n_rows          — number of rows
        stale_warning   — bool: last date >5 trading days ago
        columns         — {col: {n_missing, pct_missing, first_valid, last_valid}}
        proxy_flags     — {market: bool} — True if market uses SP500 proxy data
        fred_available  — {series: bool}
    """
    first_date = data.index[0]
    last_date = data.index[-1]
    n_rows = len(data)
    today = pd.Timestamp(datetime.today().date())
    stale_warning = bool((today - last_date).days > 7)  # >7 calendar days

    # Per-column quality
    col_quality = {}
    for col in data.columns:
        series = data[col]
        n_missing = int(series.isna().sum())
        pct_missing = round(n_missing / n_rows * 100, 2)
        valid = series.dropna()
        first_valid = valid.index[0] if not valid.empty else None
        last_valid = valid.index[-1] if not valid.empty else None
        col_quality[col] = dict(
            n_missing=n_missing,
            pct_missing=pct_missing,
            first_valid=first_valid,
            last_valid=last_valid,
        )

    # Proxy detection: EU/Swiss benchmark == SP500?
    us_ret = data.get("ret_SP500", data.get("ret_mkt_US", pd.Series(dtype=float)))
    proxy_flags = {}
    for market in ["EU", "Swiss"]:
        col = f"ret_mkt_{market}"
        if col in data.columns:
            proxy_flags[market] = bool(data[col].equals(us_ret))
        else:
            proxy_flags[market] = True  # column missing → definitely using proxy

    # FRED availability
    fred_available = {}
    for name in FRED_SERIES:
        fred_available[name] = name in data.columns and not data[name].isna().all()

    return dict(
        date_range=(first_date, last_date),
        n_rows=n_rows,
        stale_warning=stale_warning,
        columns=col_quality,
        proxy_flags=proxy_flags,
        fred_available=fred_available,
    )


def print_data_quality_report(data: pd.DataFrame) -> dict:
    """Print a human-readable data quality summary. Returns the quality dict."""
    q = compute_data_quality(data)
    border = "═" * 60

    print(f"\n{border}")
    print("  DATA QUALITY REPORT")
    print(border)

    first, last = q["date_range"]
    print(f"  Date range : {first.date()} → {last.date()}  ({q['n_rows']} rows)")
    if q["stale_warning"]:
        print("  ⚠️  Last data point is >7 calendar days old — consider --refresh")

    # ── FRED availability ─────────────────────────────────────────────────────
    print("\n  FRED macro series:")
    for name, available in q["fred_available"].items():
        status = "✓" if available else "✗ (not available — FRED_API_KEY needed?)"
        print(f"    {name:18}  {status}")

    # ── Proxy flags ────────────────────────────────────────────────────────────
    print("\n  Market benchmark proxies (True = falling back to SP500):")
    for market, is_proxy in q["proxy_flags"].items():
        flag = "⚠️  PROXY" if is_proxy else "OK"
        print(f"    {market:8}  {flag}")

    # ── Missing values (only non-zero) ────────────────────────────────────────
    missing_cols = {c: m for c, m in q["columns"].items() if m["n_missing"] > 0}
    if missing_cols:
        print(f"\n  Columns with missing values ({len(missing_cols)} of {len(q['columns'])}):")
        print(f'  {"Column":30}  {"Missing":>8}  {"Pct":>6}')
        for col, m in sorted(missing_cols.items(), key=lambda x: -x[1]["n_missing"]):
            print(f"  {col:30}  {m['n_missing']:>8}  {m['pct_missing']:>5.1f}%")
    else:
        print("\n  No missing values detected.")

    print(border + "\n")
    return q


if __name__ == "__main__":
    from data.fetcher import fetch_data

    df = fetch_data()
    print_data_quality_report(df)
