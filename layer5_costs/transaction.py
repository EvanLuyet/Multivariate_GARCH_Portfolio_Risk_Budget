"""
layer5_costs/transaction.py — Rebalancing cost and decision engine.

Transaction costs are often ignored in academic backtests but can erode
a significant fraction of alpha — especially for strategies that rebalance
frequently. We compute the full-round-trip turnover cost and compare it
to the expected gain from switching to the new BL weights.

Only recommend rebalancing when: cost < expected_gain
where expected_gain is the return differential (new weights − current weights)
applied to the BL expected returns over the next quarter.
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import ASSETS, TRANSACTION_COST


def compute_costs(current_weights: dict, optimal_weights: dict,
                  bl_returns: dict) -> dict:
    """
    Compute rebalancing turnover, cost in basis points, and the rebalance decision.

    Cost model: Cost = Σ_i |w_new_i − w_old_i| × c_i
    where c_i = TRANSACTION_COST (5 bps) per ETF leg (one-way).

    Rebalance is recommended only when the expected return gain from the new
    weights exceeds the round-trip transaction cost, preventing unnecessary
    churning in stable regimes.

    Returns a dict with:
        turnover              — total absolute weight change (0–2)
        cost_bps              — cost in basis points (100 bps = 1%)
        cost_pct              — cost as a decimal fraction
        rebalance_recommended — bool
        expected_gain         — expected quarterly return differential
        cost_breakdown        — per-asset cost breakdown dict
    """
    w_old = np.array([current_weights.get(a, 0.0) for a in ASSETS])
    w_new = np.array([optimal_weights.get(a, 0.0) for a in ASSETS])
    mu    = np.array([bl_returns.get(a, 0.0) for a in ASSETS])

    delta       = np.abs(w_new - w_old)
    turnover    = float(delta.sum())
    cost_pct    = float((delta * TRANSACTION_COST).sum())
    cost_bps    = cost_pct * 1e4

    # Expected quarterly return gain from rebalancing
    # (annualized BL returns / 4 = one-quarter estimate)
    expected_gain = float((w_new - w_old) @ (mu / 4.0))
    rebalance     = expected_gain > cost_pct

    breakdown = {a: float(delta[i] * TRANSACTION_COST) for i, a in enumerate(ASSETS)}

    return dict(
        turnover              = turnover,
        cost_bps              = cost_bps,
        cost_pct              = cost_pct,
        rebalance_recommended = rebalance,
        expected_gain         = expected_gain,
        cost_breakdown        = breakdown,
    )


if __name__ == '__main__':
    from config import BASE_WEIGHTS

    current = BASE_WEIGHTS
    optimal = {'SP500': 0.55, 'NDX': 0.30, 'EUROPE': 0.15}
    bl_ret  = {'SP500': 0.08, 'NDX': 0.11, 'EUROPE': 0.06}

    res = compute_costs(current, optimal, bl_ret)
    print(f"Turnover   : {res['turnover']:.1%}")
    print(f"Cost       : {res['cost_bps']:.2f} bps")
    print(f"Exp. gain  : {res['expected_gain']:.3%}")
    print(f"Rebalance? : {'YES' if res['rebalance_recommended'] else 'NO'}")
    print(f"Breakdown  : {res['cost_breakdown']}")
