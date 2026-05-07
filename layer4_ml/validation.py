"""
layer4_ml/validation.py — Out-of-sample ML forecast evaluation.

Computes per-asset and aggregate metrics by comparing walk-forward point forecasts
(from the XGBoost model) and interval forecasts (LightGBM q25/q75) against
realised quarterly returns.

Metrics:
  hit_rate       — fraction of quarters where forecast direction matches realised
  mae            — mean absolute error (forecast vs realised, annualised %)
  rmse           — root mean squared error (annualised %)
  spearman_corr  — Spearman rank correlation between forecast and realised
  coverage       — empirical coverage of the [q25, q75] interval (should be ≈ 50%)
  interval_width — mean width of the [q25, q75] interval (annualised %)
"""

import sys
import logging
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import ASSETS


def compute_ml_validation(ml_history: dict, data: pd.DataFrame) -> dict:
    """
    Compare walk-forward ML forecasts to realised returns.

    Parameters
    ----------
    ml_history : output of run_forecaster()["history"]
                 keyed by quarter-end Timestamp; each value has "return_forecasts"
                 → {asset: {point, low, high, bl_point}}
    data       : main DataFrame with ret_{asset} columns

    Returns
    -------
    dict with keys: per_asset, aggregate
      per_asset[asset] = {hit_rate, mae, rmse, spearman_corr, coverage, interval_width, n_obs}
      aggregate         = same metrics averaged over assets
    """
    q_ends = sorted(ml_history.keys())
    records = {a: [] for a in ASSETS}

    for k, qe in enumerate(q_ends):
        ml_q = ml_history[qe].get("return_forecasts")
        if ml_q is None:
            continue

        # Realised return: from qe to the next quarter-end (or end of data)
        next_qe = q_ends[k + 1] if k + 1 < len(q_ends) else data.index[-1]
        mask = (data.index > qe) & (data.index <= next_qe)
        period = data[mask]
        if len(period) < 5:
            continue

        for asset in ASSETS:
            if asset not in ml_q:
                continue
            fc = ml_q[asset]
            point = fc.get("bl_point", fc.get("point", 0.0))
            low = fc.get("low", 0.0)
            high = fc.get("high", 0.0)

            # Realised quarterly return (annualise to match forecast scale)
            col = f"ret_{asset}"
            if col not in period.columns:
                continue
            realised_q = float(period[col].sum())  # approx log return, quarterly
            realised_ann = realised_q * 4.0  # annualise to match forecast convention

            records[asset].append(
                dict(
                    qe=qe,
                    point=point,
                    low=low,
                    high=high,
                    realised=realised_ann,
                )
            )

    per_asset = {}
    for asset in ASSETS:
        recs = records[asset]
        if len(recs) < 4:
            per_asset[asset] = dict(
                hit_rate=np.nan,
                mae=np.nan,
                rmse=np.nan,
                spearman_corr=np.nan,
                coverage=np.nan,
                interval_width=np.nan,
                n_obs=len(recs),
            )
            continue

        points = np.array([r["point"] for r in recs])
        lows = np.array([r["low"] for r in recs])
        highs = np.array([r["high"] for r in recs])
        reals = np.array([r["realised"] for r in recs])

        errors = points - reals
        hit_rate = float(np.mean(np.sign(points) == np.sign(reals)))
        mae = float(np.mean(np.abs(errors)))
        rmse = float(np.sqrt(np.mean(errors**2)))

        corr, _ = stats.spearmanr(points, reals)
        spearman_corr = float(corr) if not np.isnan(corr) else np.nan

        # Interval calibration: realised inside [low, high]?
        in_interval = (reals >= lows) & (reals <= highs)
        coverage = float(np.mean(in_interval))
        interval_width = float(np.mean(highs - lows))

        per_asset[asset] = dict(
            hit_rate=hit_rate,
            mae=mae,
            rmse=rmse,
            spearman_corr=spearman_corr,
            coverage=coverage,
            interval_width=interval_width,
            n_obs=len(recs),
        )

    # Aggregate: simple mean over assets (exclude NaN)
    agg = {}
    for metric in ["hit_rate", "mae", "rmse", "spearman_corr", "coverage", "interval_width"]:
        vals = [per_asset[a][metric] for a in ASSETS if not np.isnan(per_asset[a][metric])]
        agg[metric] = float(np.mean(vals)) if vals else np.nan
    agg["n_obs"] = int(np.mean([per_asset[a]["n_obs"] for a in ASSETS]) if ASSETS else 0)

    return dict(per_asset=per_asset, aggregate=agg)


def print_ml_validation(validation: dict) -> None:
    """Print a formatted ML validation summary."""
    print("\n  ML FORECAST VALIDATION (out-of-sample)")
    print("  " + "─" * 56)
    per_asset = validation.get("per_asset", {})
    agg = validation.get("aggregate", {})

    header = (
        f'  {"Asset":8}  {"Hit%":>6}  {"MAE":>7}  {"RMSE":>7}'
        f'  {"Spear":>6}  {"Cov%":>6}  {"IQR%":>6}  {"N":>4}'
    )
    print(header)
    for asset in ASSETS:
        m = per_asset.get(asset, {})

        def _f(v, fmt):
            return "N/A" if np.isnan(v) else fmt.format(v)

        print(
            f"  {asset:8}  "
            f'{_f(m.get("hit_rate", np.nan), "{:.0%}"):>6}  '
            f'{_f(m.get("mae", np.nan), "{:.2%}"):>7}  '
            f'{_f(m.get("rmse", np.nan), "{:.2%}"):>7}  '
            f'{_f(m.get("spearman_corr", np.nan), "{:+.2f}"):>6}  '
            f'{_f(m.get("coverage", np.nan), "{:.0%}"):>6}  '
            f'{_f(m.get("interval_width", np.nan), "{:.1%}"):>6}  '
            f'{m.get("n_obs", 0):>4}'
        )

    print(
        f'  {"Aggregate":8}  '
        f'{_f(agg.get("hit_rate", np.nan), "{:.0%}"):>6}  '
        f'{_f(agg.get("mae", np.nan), "{:.2%}"):>7}  '
        f'{_f(agg.get("rmse", np.nan), "{:.2%}"):>7}  '
        f'{_f(agg.get("spearman_corr", np.nan), "{:+.2f}"):>6}  '
        f'{_f(agg.get("coverage", np.nan), "{:.0%}"):>6}  '
        f'{_f(agg.get("interval_width", np.nan), "{:.1%}"):>6}  '
        f'{agg.get("n_obs", 0):>4}'
    )
    print("  (Coverage target: 50% for q25/q75 interval)")
