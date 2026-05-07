"""
layer_fx/fx_analyzer.py — FX analysis for a CHF-based investor.

Computes:
  - Current spot levels for USDCHF, EURCHF, EURUSD
  - YTD and 1Y performance (has CHF strengthened or weakened?)
  - 1-quarter directional forecast (momentum + mean-reversion signal)
  - Portfolio FX impact: how much of YTD return was FX vs underlying ETF performance
  - CHF safe-haven alert: CHF tends to strengthen sharply in crises

Key insight for a Swiss investor:
  - SP500 and NDX are USD-priced ETFs → 100% USD/CHF exposure
  - EZU is a USD-priced ETF with EUR underlying → ~EUR/CHF exposure
  - When CHF strengthens (USDCHF falls), your USD assets lose CHF value
"""

import sys
import logging
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import FX_TICKERS


def analyze_fx(data: pd.DataFrame, current_weights: dict) -> dict:
    """
    Full FX analysis for a CHF-based investor.

    Parameters
    ----------
    data : DataFrame with fx_USDCHF, fx_EURCHF, fx_EURUSD columns
    current_weights : {asset: weight} dict

    Returns
    -------
    dict with keys:
        levels       — {pair: current spot rate}
        ytd_pct      — {pair: YTD % change in CHF terms}
        y1_pct       — {pair: 1Y % change}
        forecast     — {pair: 'CHF strengthening' / 'CHF weakening' / 'Neutral'}
        portfolio_fx_drag — estimated FX drag on portfolio YTD (decimal, negative = drag)
        chf_safe_haven    — bool: is CHF in safe-haven strengthening mode?
        summary      — human-readable string summary
    """
    levels, ytd_pct, y1_pct, forecast = {}, {}, {}, {}

    today = data.index[-1]
    try:
        ytd_start = data.index[data.index >= pd.Timestamp(today.year, 1, 1)][0]
    except IndexError:
        ytd_start = data.index[max(0, len(data) - 252)]
    for pair in FX_TICKERS:
        col = f"fx_{pair}"
        if col not in data.columns or data[col].isna().all():
            levels[pair] = np.nan
            ytd_pct[pair] = np.nan
            y1_pct[pair] = np.nan
            forecast[pair] = "Data unavailable"
            continue

        series = data[col].dropna()
        if series.empty:
            continue

        spot = float(series.iloc[-1])
        ytd_val = (
            float(series.reindex([ytd_start], method="ffill").iloc[0]) if len(series) > 0 else spot
        )
        y1_val = float(series.iloc[max(0, len(series) - 252)])

        levels[pair] = round(spot, 4)
        ytd_pct[pair] = round((spot / ytd_val - 1) * 100, 2) if ytd_val > 0 else 0.0
        y1_pct[pair] = round((spot / y1_val - 1) * 100, 2) if y1_val > 0 else 0.0

        # Directional forecast: momentum (3m) + mean reversion (1Y deviation)
        mom_3m = (
            (spot / float(series.iloc[max(0, len(series) - 63)]) - 1) if len(series) > 63 else 0
        )
        mean_1y = series.iloc[-252:].mean() if len(series) >= 252 else series.mean()
        dev_1y = (spot - mean_1y) / (mean_1y + 1e-8)

        # For USDCHF / EURCHF: rising = CHF weakening (bad for Swiss investor)
        # Momentum says continues; mean-reversion says reverses
        signal = 0.6 * mom_3m - 0.4 * dev_1y  # blend momentum + mean-reversion

        if pair in ("USDCHF", "EURCHF"):
            # Rising pair = CHF weakening
            if signal > 0.02:
                forecast[pair] = "CHF weakening vs " + pair.replace("CHF", "")
            elif signal < -0.02:
                forecast[pair] = "CHF strengthening vs " + pair.replace("CHF", "")
            else:
                forecast[pair] = "Neutral"
        else:
            forecast[pair] = (
                "Rising EUR/USD"
                if signal > 0.02
                else "Falling EUR/USD" if signal < -0.02 else "Neutral"
            )

    # ── Portfolio FX drag (YTD estimate) ──────────────────────────────────────
    # SP500 + NDX are pure USD assets; EUROPE (EZU) is EUR-exposed
    w_usd = current_weights.get("SP500", 0) + current_weights.get("NDX", 0)
    w_eur = current_weights.get("EUROPE", 0)

    # USDCHF change YTD: if CHF strengthened (USDCHF fell), USD assets lost value in CHF
    # For CHF investor: CHF return of USD asset = USD asset return + (USDCHF_end/USDCHF_start - 1)
    usdchf_ytd = ytd_pct.get("USDCHF", 0) or 0
    eurchf_ytd = ytd_pct.get("EURCHF", 0) or 0

    portfolio_fx_drag = round((w_usd * usdchf_ytd / 100 + w_eur * eurchf_ytd / 100), 4)

    # ── CHF safe-haven alert ───────────────────────────────────────────────────
    # CHF is in safe-haven mode if it has strengthened >3% vs USD in last 63 days
    usdchf_col = "fx_USDCHF"
    chf_safe_haven = False
    if usdchf_col in data.columns and not data[usdchf_col].isna().all():
        s63 = data[usdchf_col].dropna()
        if len(s63) > 63:
            chg_63d = s63.iloc[-1] / s63.iloc[-63] - 1
            chf_safe_haven = bool(chg_63d < -0.03)  # USDCHF falling = CHF strengthening

    # ── Summary string ─────────────────────────────────────────────────────────
    usdchf_lvl = levels.get("USDCHF", "N/A")
    eurchf_lvl = levels.get("EURCHF", "N/A")
    eurusd_lvl = levels.get("EURUSD", "N/A")

    def _fmt(val, pair):
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return "N/A"
        direction = "↑" if val > 0 else "↓"
        return f"{direction}{abs(val):.1f}%"

    summary_lines = [
        f'  USD/CHF: {usdchf_lvl}   YTD {_fmt(ytd_pct.get("USDCHF"), "USDCHF")}  '
        f'1Y {_fmt(y1_pct.get("USDCHF"), "USDCHF")}   → {forecast.get("USDCHF", "N/A")}',
        f'  EUR/CHF: {eurchf_lvl}   YTD {_fmt(ytd_pct.get("EURCHF"), "EURCHF")}  '
        f'1Y {_fmt(y1_pct.get("EURCHF"), "EURCHF")}   → {forecast.get("EURCHF", "N/A")}',
        f'  EUR/USD: {eurusd_lvl}   YTD {_fmt(ytd_pct.get("EURUSD"), "EURUSD")}  '
        f'1Y {_fmt(y1_pct.get("EURUSD"), "EURUSD")}',
        f"  Portfolio FX drag YTD: {portfolio_fx_drag:+.2%}  "
        f"(SP500+NDX={w_usd:.0%} USD, EUROPE={w_eur:.0%} EUR)",
    ]
    if chf_safe_haven:
        summary_lines.append("  ⚠️  CHF safe-haven alert: CHF has strengthened >3% vs USD (63d)")

    return dict(
        levels=levels,
        ytd_pct=ytd_pct,
        y1_pct=y1_pct,
        forecast=forecast,
        portfolio_fx_drag=portfolio_fx_drag,
        chf_safe_haven=chf_safe_haven,
        summary_lines=summary_lines,
    )


def compute_chf_adjusted_returns(data: pd.DataFrame, weights: dict) -> dict:
    """
    Compute CHF-adjusted return series (hedged vs unhedged) for a CHF-based investor.

    For each USD asset:
      CHF_return = USD_return + USDCHF_daily_return  (unhedged)
    For the EUR asset (EUROPE = EZU):
      CHF_return = EUR_return + EURCHF_daily_return   (unhedged)
    Hedged: strips out the FX return (just the USD/EUR underlying return).

    Returns
    -------
    dict with keys:
        unhedged_daily   — pd.Series of portfolio daily returns in CHF (unhedged)
        hedged_daily     — pd.Series of portfolio daily returns in CHF (fully hedged)
        half_hedged_daily — 50% hedge ratio
        unhedged_ann_ret — annualised return unhedged
        hedged_ann_ret   — annualised return hedged
        fx_contribution  — difference = annual FX impact
    """
    from config import TRADING_DAYS

    usdchf_col = "fx_USDCHF"
    eurchf_col = "fx_EURCHF"

    has_usdchf = usdchf_col in data.columns and not data[usdchf_col].isna().all()
    has_eurchf = eurchf_col in data.columns and not data[eurchf_col].isna().all()

    # Daily FX returns
    fx_usdchf = (
        data[usdchf_col].pct_change().fillna(0.0)
        if has_usdchf
        else pd.Series(0.0, index=data.index)
    )
    fx_eurchf = (
        data[eurchf_col].pct_change().fillna(0.0)
        if has_eurchf
        else pd.Series(0.0, index=data.index)
    )

    # SP500 and NDX: USD exposure → USDCHF FX
    # EUROPE (EZU): EUR exposure → EURCHF FX
    # Weights
    w_sp500 = weights.get("SP500", 0.0)
    w_ndx = weights.get("NDX", 0.0)
    w_europe = weights.get("EUROPE", 0.0)

    # Underlying asset returns
    port_usd = pd.Series(0.0, index=data.index)
    for asset in ["SP500", "NDX"]:
        col = f"ret_{asset}"
        if col in data.columns:
            port_usd += data[col].fillna(0.0) * weights.get(asset, 0.0)
    port_eur = (
        data["ret_EUROPE"].fillna(0.0) * w_europe
        if "ret_EUROPE" in data.columns
        else pd.Series(0.0, index=data.index)
    )

    # FX contribution per day
    fx_contrib = (w_sp500 + w_ndx) * fx_usdchf + w_europe * fx_eurchf

    hedged_daily = port_usd + port_eur  # underlying only
    unhedged_daily = hedged_daily + fx_contrib
    half_hedged_daily = hedged_daily + 0.5 * fx_contrib

    def _ann(s):
        return float(s.mean() * TRADING_DAYS) if not s.empty else np.nan

    return dict(
        unhedged_daily=unhedged_daily,
        hedged_daily=hedged_daily,
        half_hedged_daily=half_hedged_daily,
        unhedged_ann_ret=_ann(unhedged_daily),
        hedged_ann_ret=_ann(hedged_daily),
        fx_contribution=_ann(unhedged_daily) - _ann(hedged_daily),
    )


if __name__ == "__main__":
    from data.fetcher import fetch_data
    from config import BASE_WEIGHTS

    df = fetch_data()
    res = analyze_fx(df, BASE_WEIGHTS)

    print("\n=== FX Analysis ===")
    for line in res["summary_lines"]:
        print(line)

    chf = compute_chf_adjusted_returns(df, BASE_WEIGHTS)
    print(f"\n  Unhedged ann. return : {chf['unhedged_ann_ret']:.2%}")
    print(f"  Hedged ann. return   : {chf['hedged_ann_ret']:.2%}")
    print(f"  FX contribution      : {chf['fx_contribution']:+.2%}")
