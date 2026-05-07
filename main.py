"""
main.py — Portfolio Intelligence Engine master runner.

Subcommands:
    python main.py run      [--refresh] [--capital N] [--portfolio-value N] [--non-interactive]
    python main.py backtest [--refresh]
    python main.py report
    python main.py screen   [--regime {0,1,2}]

Backward-compatible: bare flags without a subcommand still invoke `run`.
    python main.py --refresh          → same as: python main.py run --refresh
    python main.py --non-interactive  → same as: python main.py run --non-interactive
"""

import json
import logging
import argparse
import sys
import warnings
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
)

from config import ASSETS, BASE_WEIGHTS, STATE_FILE, MODEL_CACHE_DIR

from data.fetcher import fetch_data
from layer1_garch.garch_model import run_garch
from layer2_hmm.regime_model import run_hmm, run_market_regimes
from layer4_ml.return_forecaster import run_forecaster
from layer3_optimizer.black_litterman import run_black_litterman
from layer5_costs.transaction import compute_costs
from layer_fx.fx_analyzer import analyze_fx
from layer_stocks.screener import run_screener
from report.dashboard import (
    print_report,
    save_dashboard,
    _build_strategy_returns,
    compute_garch_vol_evaluation,
)

# ── State management ──────────────────────────────────────────────────────────


def load_state() -> dict:
    p = Path(STATE_FILE)
    if p.exists():
        with p.open() as f:
            state = json.load(f)
        weights = state.get("weights", BASE_WEIGHTS)
        print(
            f"Loaded previous weights from {STATE_FILE}: "
            f'{", ".join(f"{a}={v:.1%}" for a, v in weights.items())}'
        )
        return state
    return {"weights": BASE_WEIGHTS.copy()}


def save_state(weights: dict, portfolio_value: float = None) -> None:
    state = dict(
        weights=weights,
        timestamp=datetime.today().isoformat(),
        portfolio_value=portfolio_value,
    )
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)
    print(f"State saved → {STATE_FILE}")


def _clear_model_cache() -> None:
    cache_dir = Path(MODEL_CACHE_DIR)
    if cache_dir.exists():
        deleted = list(cache_dir.glob("*.joblib"))
        for f in deleted:
            f.unlink()
        print(f"Model cache cleared ({len(deleted)} file(s) removed).")


# ── Post-run interactive state prompt ─────────────────────────────────────────


def _prompt_state_update(optimal_weights: dict) -> dict | None:
    """
    Ask the user if they executed the rebalance and what weights they applied.
    Returns the weights to save, or None to save optimal_weights unchanged.
    """
    try:
        ans = input("\n  Did you execute this rebalance? [y/n]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return None

    if ans != "y":
        print("  OK — keeping current weights unchanged for next run.")
        return None

    print("  Enter your actual executed weights (press Enter to use proposed):")
    executed = {}
    for asset in ASSETS:
        proposed = optimal_weights.get(asset, 0)
        try:
            raw = input(f"    {asset} (proposed {proposed:.0%}): ").strip()
            if raw == "":
                executed[asset] = proposed
            else:
                val = float(raw.replace("%", "")) / 100
                executed[asset] = val
        except (ValueError, EOFError):
            executed[asset] = proposed

    total = sum(executed.values())
    if abs(total - 1.0) > 0.02:
        print(f"  ⚠️  Weights sum to {total:.1%} — normalising to 100%.")
        executed = {a: v / total for a, v in executed.items()}

    print("  Saved:  " + "  ".join(f"{a} {v:.1%}" for a, v in executed.items()))
    return executed


# ── Subcommand: run ───────────────────────────────────────────────────────────


def run(
    force_refresh: bool = False,
    new_capital_usd: float = 0.0,
    portfolio_value_usd: float = 100_000.0,
    non_interactive: bool = False,
) -> None:

    print("\n" + "=" * 60)
    print("  PORTFOLIO INTELLIGENCE ENGINE")
    print("=" * 60 + "\n")

    if force_refresh:
        _clear_model_cache()

    # ── Load state ────────────────────────────────────────────────────────────
    state = load_state()
    current_weights = state.get("weights", BASE_WEIGHTS)
    portfolio_value_usd = state.get("portfolio_value", portfolio_value_usd) or portfolio_value_usd

    # ── 1. Data ───────────────────────────────────────────────────────────────
    data = fetch_data(force_refresh=force_refresh)

    # ── 2. Layer 1 — GARCH + DCC-style covariance ─────────────────────────────
    garch_history = run_garch(data, current_weights)
    latest_qe = max(garch_history.keys())
    garch_current = garch_history[latest_qe]

    # ── 3. Layer 2 — HMM regime (US, for BL optimizer) ────────────────────────
    hmm_result = run_hmm(garch_history, data)
    hmm_current = hmm_result["history"][latest_qe]

    # ── 4. Layer 2b — Per-market regime (US / EU / Swiss) ────────────────────
    market_regimes = run_market_regimes(data)

    # ── 5. Layer 4 — ML return forecasts ─────────────────────────────────────
    ml_result = run_forecaster(garch_history, hmm_result["history"], data)
    ml_history = ml_result["history"]
    ml_current = ml_result["current_forecast"]
    ml_next_q = ml_result["next_quarter_forecast"]

    # ── 6. Layer 3 — Black-Litterman optimizer ────────────────────────────────
    bl_result = run_black_litterman(
        garch_history,
        hmm_result["history"],
        ml_history,
        current_weights,
    )
    bl_current = bl_result["current"]
    bl_history = bl_result["history"]

    # ── 7. FX analysis ────────────────────────────────────────────────────────
    fx_result = analyze_fx(data, current_weights)

    # ── 8. Stock screener ─────────────────────────────────────────────────────
    current_regime = hmm_current.get("regime", 1)
    stock_picks = run_screener(regime=current_regime, data=data)

    # ── 9. Transaction costs ──────────────────────────────────────────────────
    optimal_weights = bl_current["optimal_weights"]
    etf_prices = {
        a: float(data[f"price_{a}"].iloc[-1]) for a in ASSETS if f"price_{a}" in data.columns
    }
    cost_result = compute_costs(
        current_weights,
        optimal_weights,
        bl_current["bl_returns"],
        portfolio_value_usd=portfolio_value_usd,
        etf_prices=etf_prices,
        new_capital_usd=new_capital_usd,
    )
    final_weights = optimal_weights if cost_result["rebalance_recommended"] else current_weights

    # ── 10. Build return series for backtest ──────────────────────────────────
    strat_ret = _build_strategy_returns(garch_history, bl_history, data, include_tc=True)
    bench_ret = _build_strategy_returns(
        garch_history, bl_history, data, weights_override=BASE_WEIGHTS
    )
    current_w_ret = _build_strategy_returns(
        garch_history, bl_history, data, weights_override=current_weights
    )

    # ── 10b. GARCH vol evaluation ─────────────────────────────────────────────
    vol_eval = compute_garch_vol_evaluation(garch_history, data)

    # ── 11. Print report + save dashboard ────────────────────────────────────
    print_report(
        garch_current,
        hmm_current,
        market_regimes,
        fx_result,
        stock_picks,
        ml_current,
        ml_next_q,
        bl_current,
        cost_result,
        current_weights,
        final_weights,
        strat_ret,
        bench_ret,
        current_w_ret,
        vol_eval=vol_eval,
    )
    save_dashboard(
        garch_history,
        hmm_result,
        bl_history,
        ml_result,
        market_regimes,
        fx_result,
        strat_ret,
        bench_ret,
        current_w_ret,
    )

    # ── 12. Interactive state update ──────────────────────────────────────────
    if non_interactive:
        save_state(final_weights, portfolio_value_usd)
    else:
        executed = _prompt_state_update(final_weights)
        weights_to_save = executed if executed is not None else final_weights
        save_state(weights_to_save, portfolio_value_usd)


# ── Subcommand: backtest ──────────────────────────────────────────────────────


def cmd_backtest(force_refresh: bool = False) -> None:
    """Run a focused backtest comparison and write CSV artifacts."""
    from report.dashboard import build_backtest_report

    print("\n" + "=" * 60)
    print("  BACKTEST MODE")
    print("=" * 60 + "\n")

    if force_refresh:
        _clear_model_cache()

    state = load_state()
    current_weights = state.get("weights", BASE_WEIGHTS)

    data = fetch_data(force_refresh=force_refresh)
    garch_history = run_garch(data, current_weights)
    hmm_result = run_hmm(garch_history, data)
    ml_result = run_forecaster(garch_history, hmm_result["history"], data)
    bl_result = run_black_litterman(
        garch_history, hmm_result["history"], ml_result["history"], current_weights
    )

    build_backtest_report(garch_history, bl_result["history"], data, current_weights)


# ── Subcommand: report ────────────────────────────────────────────────────────


def cmd_report() -> None:
    """Re-generate the dashboard PNG + terminal report from cached models (no re-fit)."""
    print("\n" + "=" * 60)
    print("  REPORT MODE  (cached models)")
    print("=" * 60 + "\n")

    state = load_state()
    current_weights = state.get("weights", BASE_WEIGHTS)

    data = fetch_data(force_refresh=False)
    garch_history = run_garch(data, current_weights)
    hmm_result = run_hmm(garch_history, data)
    market_regimes = run_market_regimes(data)
    ml_result = run_forecaster(garch_history, hmm_result["history"], data)
    bl_result = run_black_litterman(
        garch_history, hmm_result["history"], ml_result["history"], current_weights
    )

    latest_qe = max(garch_history.keys())
    garch_current = garch_history[latest_qe]
    hmm_current = hmm_result["history"][latest_qe]
    bl_current = bl_result["current"]
    bl_history = bl_result["history"]

    fx_result = analyze_fx(data, current_weights)
    stock_picks = run_screener(regime=hmm_current.get("regime", 1), data=data)

    portfolio_value_usd = state.get("portfolio_value", 100_000.0) or 100_000.0
    etf_prices = {
        a: float(data[f"price_{a}"].iloc[-1]) for a in ASSETS if f"price_{a}" in data.columns
    }
    cost_result = compute_costs(
        current_weights,
        bl_current["optimal_weights"],
        bl_current["bl_returns"],
        portfolio_value_usd=portfolio_value_usd,
        etf_prices=etf_prices,
    )

    strat_ret = _build_strategy_returns(garch_history, bl_history, data, include_tc=True)
    bench_ret = _build_strategy_returns(
        garch_history, bl_history, data, weights_override=BASE_WEIGHTS
    )
    current_w_ret = _build_strategy_returns(
        garch_history, bl_history, data, weights_override=current_weights
    )
    vol_eval = compute_garch_vol_evaluation(garch_history, data)

    print_report(
        garch_current,
        hmm_current,
        market_regimes,
        fx_result,
        stock_picks,
        ml_result["current_forecast"],
        ml_result["next_quarter_forecast"],
        bl_current,
        cost_result,
        current_weights,
        bl_current["optimal_weights"],
        strat_ret,
        bench_ret,
        current_w_ret,
        vol_eval=vol_eval,
    )
    save_dashboard(
        garch_history,
        hmm_result,
        bl_history,
        ml_result,
        market_regimes,
        fx_result,
        strat_ret,
        bench_ret,
        current_w_ret,
    )


# ── Subcommand: screen ────────────────────────────────────────────────────────


def cmd_screen(regime: int = 1) -> None:
    """Run the stock screener for a given regime and print + save picks."""
    import csv

    print("\n" + "=" * 60)
    print(f"  STOCK SCREENER  (regime={regime}: 0=Bull / 1=Transition / 2=Crisis)")
    print("=" * 60 + "\n")

    data = fetch_data(force_refresh=False)
    picks = run_screener(regime=regime, data=data)

    regime_names = {0: "Bull", 1: "Transition", 2: "Crisis"}
    print(f"  Top picks for {regime_names.get(regime, 'Unknown')} regime:\n")
    for p in picks:
        if p["ticker"] == "N/A":
            print(f'    {p["note"]}')
            break
        ret_str = f'  ret_3m {p.get("ret_3m_pct", 0):+.1%}' if "ret_3m_pct" in p else ""
        print(
            f'  {p["rank"]}. {p["ticker"]:8s} {p["sector"]:22s} '
            f'score {p["score"]:+.3f}  3m_z {p["mom_3m_z"]:+.3f}'
            f'{ret_str}  — {p["note"]}'
        )

    # CSV output
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    csv_path = results_dir / "stock_picks.csv"
    if picks and picks[0]["ticker"] != "N/A":
        keys = list(picks[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(picks)
        print(f"\n  Saved → {csv_path}")


# ── Argument parsing ──────────────────────────────────────────────────────────

LEGACY_FLAGS = {"--refresh", "--capital", "--portfolio-value", "--non-interactive"}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Portfolio Intelligence Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py run --refresh          Full pipeline, force re-fetch + re-fit
  python main.py run --non-interactive  Full pipeline, no interactive prompt
  python main.py backtest               Focused backtest report + CSV artifacts
  python main.py report                 Regenerate report from cached models
  python main.py screen --regime 0      Stock picks for Bull regime
        """,
    )
    sub = parser.add_subparsers(dest="cmd")

    # run
    p_run = sub.add_parser("run", help="Full pipeline (default)")
    p_run.add_argument("--refresh", action="store_true", help="Force re-fetch + re-fit all models")
    p_run.add_argument("--capital", type=float, default=0.0, help="New capital to deploy (USD)")
    p_run.add_argument(
        "--portfolio-value",
        type=float,
        default=100_000.0,
        help="Total portfolio value (USD)",
    )
    p_run.add_argument(
        "--non-interactive",
        action="store_true",
        help="Skip rebalance prompt, auto-save proposed weights",
    )

    # backtest
    p_bt = sub.add_parser("backtest", help="Focused backtest comparison + CSV output")
    p_bt.add_argument("--refresh", action="store_true", help="Force re-fetch + re-fit")

    # report
    sub.add_parser("report", help="Regenerate dashboard/report from cached models")

    # screen
    p_sc = sub.add_parser("screen", help="Run stock screener")
    p_sc.add_argument(
        "--regime",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="Regime override: 0=Bull, 1=Transition, 2=Crisis (default: 1)",
    )

    return parser


if __name__ == "__main__":
    # ── Backward-compat: if first arg looks like a flag (not a subcommand), prepend "run" ──
    known_cmds = {"run", "backtest", "report", "screen"}
    if len(sys.argv) > 1 and sys.argv[1].startswith("-"):
        sys.argv.insert(1, "run")

    parser = _build_parser()
    args = parser.parse_args()

    if args.cmd == "run" or args.cmd is None:
        run(
            force_refresh=getattr(args, "refresh", False),
            new_capital_usd=getattr(args, "capital", 0.0),
            portfolio_value_usd=getattr(args, "portfolio_value", 100_000.0),
            non_interactive=getattr(args, "non_interactive", False),
        )
    elif args.cmd == "backtest":
        cmd_backtest(force_refresh=args.refresh)
    elif args.cmd == "report":
        cmd_report()
    elif args.cmd == "screen":
        cmd_screen(regime=args.regime)
    else:
        parser.print_help()
