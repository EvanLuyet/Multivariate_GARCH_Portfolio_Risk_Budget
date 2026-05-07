"""
layer_risk/stress_tester.py — Historical stress tests for the portfolio.

Tests three named crisis windows plus custom date ranges:
  - 2008 GFC      : 2008-09-01 → 2009-03-31
  - COVID crash   : 2020-02-19 → 2020-03-23
  - 2022 drawdown : 2022-01-01 → 2022-10-31 (rate/inflation shock)

For each scenario, computes:
  - cumulative return of each weight set
  - max drawdown during the window
  - daily VaR 95% during the window
  - annualised vol during the window

Results are printed in a table and optionally written to results/stress_tests.csv.
"""

import sys
import csv
import logging
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import ASSETS, TRADING_DAYS

SCENARIOS = {
    "2008 GFC": ("2008-09-01", "2009-03-31"),
    "COVID crash": ("2020-02-19", "2020-03-23"),
    "2022 rate shock": ("2022-01-01", "2022-10-31"),
}


def _scenario_metrics(returns: pd.Series) -> dict:
    """Compute stress metrics for a single return series."""
    if returns.empty:
        return dict(cum_ret=np.nan, max_dd=np.nan, var95=np.nan, ann_vol=np.nan, n_days=0)

    cum = (1 + returns).cumprod()
    cum_ret = float(cum.iloc[-1] - 1)
    roll_max = cum.cummax()
    dd = (cum - roll_max) / roll_max
    max_dd = float(dd.min())
    var95 = float(np.percentile(returns, 5)) * -1  # positive = loss
    ann_vol = float(returns.std() * np.sqrt(TRADING_DAYS))
    return dict(cum_ret=cum_ret, max_dd=max_dd, var95=var95, ann_vol=ann_vol, n_days=len(returns))


def run_stress_tests(
    data: pd.DataFrame,
    weight_sets: dict,
    extra_scenarios: Optional[dict] = None,
    save_csv: bool = True,
) -> dict:
    """
    Run stress tests across named historical scenarios.

    Parameters
    ----------
    data        : main DataFrame with ret_{asset} columns
    weight_sets : {label: {asset: weight}} — e.g. {"BL": {...}, "Base": {...}}
    extra_scenarios : optional additional windows {name: (start_str, end_str)}
    save_csv    : write results to results/stress_tests.csv

    Returns
    -------
    dict keyed by scenario name → {weight_label: metrics_dict}
    """
    all_scenarios = dict(SCENARIOS)
    if extra_scenarios:
        all_scenarios.update(extra_scenarios)

    results = {}

    border = "═" * 60
    thin = "─" * 60
    print(f"\n{border}")
    print("  STRESS TEST RESULTS")
    print(border)

    for scenario_name, (start_str, end_str) in all_scenarios.items():
        mask = (data.index >= pd.Timestamp(start_str)) & (data.index <= pd.Timestamp(end_str))
        period = data[mask]

        if period.empty:
            logger.warning(f"Stress test '{scenario_name}': no data in range {start_str}→{end_str}")
            results[scenario_name] = {}
            continue

        print(f"\n  {scenario_name}  ({start_str} → {end_str}, {mask.sum()} trading days)")
        print(thin)
        print(f'  {"Strategy":22}  {"CumRet":>8}  {"MaxDD":>8}  {"VaR95":>8}  {"AnnVol":>8}')

        scenario_results = {}
        for label, weights in weight_sets.items():
            w_vec = np.array([weights.get(a, 0.0) for a in ASSETS])
            w_vec /= w_vec.sum()
            ret_cols = [f"ret_{a}" for a in ASSETS if f"ret_{a}" in period.columns]
            if not ret_cols:
                continue
            port_rets = pd.Series(
                period[ret_cols].values @ w_vec[: len(ret_cols)], index=period.index
            )
            m = _scenario_metrics(port_rets)
            scenario_results[label] = m
            print(
                f"  {label:22}  "
                f'{m["cum_ret"]:>8.2%}  '
                f'{m["max_dd"]:>8.2%}  '
                f'{m["var95"]:>8.3%}  '
                f'{m["ann_vol"]:>8.2%}'
            )

        results[scenario_name] = scenario_results

    print(f"\n{border}\n")

    if save_csv:
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        csv_path = results_dir / "stress_tests.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["scenario", "strategy", "cum_ret", "max_dd", "var95", "ann_vol", "n_days"]
            )
            for scenario_name, scenario_results in results.items():
                for label, m in scenario_results.items():
                    writer.writerow(
                        [
                            scenario_name,
                            label,
                            f"{m.get('cum_ret', np.nan):.4f}",
                            f"{m.get('max_dd', np.nan):.4f}",
                            f"{m.get('var95', np.nan):.4f}",
                            f"{m.get('ann_vol', np.nan):.4f}",
                            str(m.get("n_days", 0)),
                        ]
                    )
        print(f"Stress test CSV saved → {csv_path}")

    return results


if __name__ == "__main__":
    from data.fetcher import fetch_data
    from config import BASE_WEIGHTS

    df = fetch_data()
    run_stress_tests(df, {"Base": BASE_WEIGHTS})
