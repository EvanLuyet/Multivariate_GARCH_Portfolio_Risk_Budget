"""
main.py — Portfolio Intelligence Engine master runner.

Usage:
    python main.py                        # normal run, loads cached data
    python main.py --refresh              # force re-fetch + re-fit all models
    python main.py --capital 10000        # deploy CHF 10 000 new capital
    python main.py --portfolio-value 150000  # override portfolio value (USD)

After printing the report the bot prompts:
    "Did you execute this rebalance? [y/n]"
    → If yes, enter your actual executed weights for accurate next-run comparison.
"""

import sys
import json
import argparse
import warnings
from pathlib import Path
from datetime import datetime

warnings.filterwarnings('ignore')

from config import ASSETS, BASE_WEIGHTS, STATE_FILE, MODEL_CACHE_DIR

from data.fetcher                     import fetch_data
from layer1_garch.garch_model         import run_garch
from layer2_hmm.regime_model          import run_hmm, run_market_regimes
from layer4_ml.return_forecaster      import run_forecaster
from layer3_optimizer.black_litterman import run_black_litterman
from layer5_costs.transaction         import compute_costs
from layer_fx.fx_analyzer             import analyze_fx
from layer_stocks.screener            import run_screener
from report.dashboard                 import (print_report, save_dashboard,
                                              _build_strategy_returns)


# ── State management ──────────────────────────────────────────────────────────

def load_state() -> dict:
    p = Path(STATE_FILE)
    if p.exists():
        with p.open() as f:
            state = json.load(f)
        weights = state.get('weights', BASE_WEIGHTS)
        print(f'Loaded previous weights from {STATE_FILE}: '
              f'{", ".join(f"{a}={v:.1%}" for a, v in weights.items())}')
        return state
    return {'weights': BASE_WEIGHTS.copy()}


def save_state(weights: dict, portfolio_value: float = None) -> None:
    state = dict(
        weights          = weights,
        timestamp        = datetime.today().isoformat(),
        portfolio_value  = portfolio_value,
    )
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2)
    print(f'State saved → {STATE_FILE}')


def _clear_model_cache() -> None:
    cache_dir = Path(MODEL_CACHE_DIR)
    if cache_dir.exists():
        deleted = list(cache_dir.glob('*.joblib'))
        for f in deleted:
            f.unlink()
        print(f'Model cache cleared ({len(deleted)} file(s) removed).')


# ── Post-run interactive state prompt ─────────────────────────────────────────

def _prompt_state_update(optimal_weights: dict) -> dict | None:
    """
    Ask the user if they executed the rebalance and what weights they applied.
    Returns the weights to save, or None to save optimal_weights unchanged.
    """
    try:
        ans = input('\n  Did you execute this rebalance? [y/n]: ').strip().lower()
    except (EOFError, KeyboardInterrupt):
        return None

    if ans != 'y':
        print('  OK — keeping current weights unchanged for next run.')
        return None

    print('  Enter your actual executed weights (press Enter to use proposed):')
    executed = {}
    for asset in ASSETS:
        proposed = optimal_weights.get(asset, 0)
        try:
            raw = input(f'    {asset} (proposed {proposed:.0%}): ').strip()
            if raw == '':
                executed[asset] = proposed
            else:
                val = float(raw.replace('%', '')) / 100
                executed[asset] = val
        except (ValueError, EOFError):
            executed[asset] = proposed

    total = sum(executed.values())
    if abs(total - 1.0) > 0.02:
        print(f'  ⚠️  Weights sum to {total:.1%} — normalising to 100%.')
        executed = {a: v / total for a, v in executed.items()}

    print('  Saved:  ' + '  '.join(f'{a} {v:.1%}' for a, v in executed.items()))
    return executed


# ── Main run ──────────────────────────────────────────────────────────────────

def run(force_refresh: bool = False,
        new_capital_usd: float = 0.0,
        portfolio_value_usd: float = 100_000.0) -> None:

    print('\n' + '=' * 60)
    print('  PORTFOLIO INTELLIGENCE ENGINE')
    print('=' * 60 + '\n')

    if force_refresh:
        _clear_model_cache()

    # ── Load state ────────────────────────────────────────────────────────────
    state           = load_state()
    current_weights = state.get('weights', BASE_WEIGHTS)
    portfolio_value_usd = state.get('portfolio_value', portfolio_value_usd) or portfolio_value_usd

    # ── 1. Data ───────────────────────────────────────────────────────────────
    data = fetch_data(force_refresh=force_refresh)

    # ── 2. Layer 1 — GARCH + DCC ──────────────────────────────────────────────
    garch_history = run_garch(data, current_weights)
    latest_qe     = max(garch_history.keys())
    garch_current = garch_history[latest_qe]

    # ── 3. Layer 2 — HMM regime (US, for BL optimizer) ────────────────────────
    hmm_result    = run_hmm(garch_history, data)
    hmm_current   = hmm_result['history'][latest_qe]

    # ── 4. Layer 2b — Per-market regime (US / EU / Swiss) ────────────────────
    market_regimes = run_market_regimes(data)

    # ── 5. Layer 4 — ML return forecasts ─────────────────────────────────────
    ml_result    = run_forecaster(garch_history, hmm_result['history'], data)
    ml_history   = ml_result['history']
    ml_current   = ml_result['current_forecast']
    ml_next_q    = ml_result['next_quarter_forecast']

    # ── 6. Layer 3 — Black-Litterman optimizer ────────────────────────────────
    bl_result  = run_black_litterman(
        garch_history, hmm_result['history'], ml_history, current_weights,
    )
    bl_current = bl_result['current']
    bl_history = bl_result['history']

    # ── 7. FX analysis ────────────────────────────────────────────────────────
    fx_result = analyze_fx(data, current_weights)

    # ── 8. Stock screener ─────────────────────────────────────────────────────
    current_regime = hmm_current.get('regime', 1)
    stock_picks    = run_screener(regime=current_regime, data=data)

    # ── 9. Transaction costs ──────────────────────────────────────────────────
    optimal_weights = bl_current['optimal_weights']

    # Last known ETF prices for IBKR commission calculation
    etf_prices = {a: float(data[f'price_{a}'].iloc[-1]) for a in ASSETS
                  if f'price_{a}' in data.columns}

    cost_result = compute_costs(
        current_weights, optimal_weights, bl_current['bl_returns'],
        portfolio_value_usd = portfolio_value_usd,
        etf_prices          = etf_prices,
        new_capital_usd     = new_capital_usd,
    )

    final_weights = optimal_weights if cost_result['rebalance_recommended'] else current_weights

    # ── 10. Build return series for backtest (3-way) ──────────────────────────
    strat_ret    = _build_strategy_returns(garch_history, bl_history, data)
    bench_ret    = _build_strategy_returns(garch_history, bl_history, data,
                                           weights_override=BASE_WEIGHTS)
    current_w_ret = _build_strategy_returns(garch_history, bl_history, data,
                                            weights_override=current_weights)

    # ── 11. Print report + save dashboard ────────────────────────────────────
    print_report(
        garch_current, hmm_current, market_regimes, fx_result,
        stock_picks, ml_current, ml_next_q, bl_current, cost_result,
        current_weights, final_weights, strat_ret, bench_ret, current_w_ret,
    )
    save_dashboard(
        garch_history, hmm_result, bl_history, ml_result,
        market_regimes, fx_result, strat_ret, bench_ret, current_w_ret,
    )

    # ── 12. Interactive state update ──────────────────────────────────────────
    executed = _prompt_state_update(final_weights)
    weights_to_save = executed if executed is not None else final_weights
    save_state(weights_to_save, portfolio_value_usd)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Portfolio Intelligence Engine')
    parser.add_argument('--refresh', action='store_true',
                        help='Force re-fetch data and re-fit all models')
    parser.add_argument('--capital', type=float, default=0.0,
                        help='New capital to deploy in USD (Scenario B)')
    parser.add_argument('--portfolio-value', type=float, default=100_000.0,
                        help='Total portfolio value in USD (for cost calculation)')
    args = parser.parse_args()

    run(force_refresh      = args.refresh,
        new_capital_usd    = args.capital,
        portfolio_value_usd= args.portfolio_value)
