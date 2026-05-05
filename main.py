"""
main.py — Portfolio Intelligence Engine master runner.

Orchestrates all five analytical layers in the correct dependency order,
then produces a terminal intelligence report and a 6-panel PNG dashboard.
Saves portfolio state to JSON after each run so the next run can compare
current weights against recommended weights for accurate turnover costing.

Run:
    python main.py
    python main.py --refresh    # force re-fetch data + re-fit models
"""

import sys
import json
import argparse
import warnings
from pathlib import Path
from datetime import datetime

warnings.filterwarnings('ignore')

from config import ASSETS, BASE_WEIGHTS, STATE_FILE

from data.fetcher                   import fetch_data
from layer1_garch.garch_model       import run_garch
from layer2_hmm.regime_model        import run_hmm
from layer4_ml.return_forecaster    import run_forecaster
from layer3_optimizer.black_litterman import run_black_litterman
from layer5_costs.transaction       import compute_costs
from report.dashboard               import (print_report, save_dashboard,
                                            _build_strategy_returns)


def load_state() -> dict:
    """Load saved portfolio weights from the previous run, defaulting to BASE_WEIGHTS."""
    p = Path(STATE_FILE)
    if p.exists():
        with p.open() as f:
            state = json.load(f)
        weights = state.get('weights', BASE_WEIGHTS)
        print(f'Loaded previous weights from {STATE_FILE}: '
              f'{", ".join(f"{a}={v:.1%}" for a, v in weights.items())}')
        return weights
    return BASE_WEIGHTS.copy()


def save_state(weights: dict) -> None:
    """Persist current weights and timestamp for the next run."""
    state = dict(
        weights   = weights,
        timestamp = datetime.today().isoformat(),
    )
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2)
    print(f'State saved → {STATE_FILE}')


def run(force_refresh: bool = False) -> None:
    print('\n' + '=' * 56)
    print('  PORTFOLIO INTELLIGENCE ENGINE')
    print('=' * 56 + '\n')

    # ── Load previous state ────────────────────────────────────────────────────
    current_weights = load_state()

    # ── 1. Data ────────────────────────────────────────────────────────────────
    data = fetch_data(force_refresh=force_refresh)

    # ── 2. Layer 1 — GARCH + DCC covariance ───────────────────────────────────
    garch_history = run_garch(data, current_weights)
    latest_qe     = max(garch_history.keys())
    garch_current = garch_history[latest_qe]

    # ── 3. Layer 2 — HMM regime detection ─────────────────────────────────────
    hmm_result  = run_hmm(garch_history, data)
    hmm_current = hmm_result['history'][latest_qe]

    # ── 4. Layer 4 — ML return forecasts (feeds views into BL) ────────────────
    ml_result       = run_forecaster(garch_history, hmm_result['history'], data)
    ml_history      = ml_result['history']
    ml_current      = ml_result['current_forecast']

    # ── 5. Layer 3 — Black-Litterman + MVO ────────────────────────────────────
    bl_result    = run_black_litterman(
        garch_history, hmm_result['history'], ml_history, current_weights,
    )
    bl_current   = bl_result['current']
    bl_history   = bl_result['history']

    # ── 6. Layer 5 — Transaction costs ────────────────────────────────────────
    optimal_weights = bl_current['optimal_weights']
    cost_result     = compute_costs(current_weights, optimal_weights,
                                    bl_current['bl_returns'])

    # Honour the rebalance decision: hold if cost exceeds expected gain
    final_weights = optimal_weights if cost_result['rebalance_recommended'] else current_weights

    # ── 7. Build strategy return series for backtest stats ────────────────────
    strat_ret, bench_ret = _build_strategy_returns(garch_history, bl_history, data)

    # ── 8. Print terminal report + save dashboard ──────────────────────────────
    print_report(
        garch_current, hmm_current, ml_current, bl_current,
        cost_result, final_weights, strat_ret, bench_ret,
    )
    save_dashboard(
        garch_history, hmm_result, bl_history, ml_result, strat_ret, bench_ret,
    )

    # ── 9. Persist state for next run ─────────────────────────────────────────
    save_state(final_weights)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Portfolio Intelligence Engine')
    parser.add_argument('--refresh', action='store_true',
                        help='Force re-fetch data and re-fit all models')
    args = parser.parse_args()
    run(force_refresh=args.refresh)
