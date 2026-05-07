"""
layer5_costs/transaction.py — Rebalancing cost engine (IBKR Tiered model).

Cost model (Interactive Brokers Tiered plan, US-listed ETFs):
  Commission  = max(IBKR_MIN_COMMISSION, IBKR_RATE_PER_SHARE × shares_traded)
  FX cost     = IBKR_FX_COST × trade_value_USD  (only if buying from CHF account)
  Spread cost = IBKR_SPREAD_BPS / 10000 × trade_value

Two scenarios are computed:
  A — Rebalancing: sell overweight positions, buy underweight ones
  B — New capital: invest new money to move toward target weights (no selling)

Rebalance is only recommended if expected_gain > total_cost.
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (ASSETS, IBKR_MIN_COMMISSION, IBKR_RATE_PER_SHARE,
                    IBKR_FX_COST, IBKR_SPREAD_BPS, REBALANCE_BAND)


def _ibkr_commission(trade_value_usd: float, price_per_share: float) -> float:
    """IBKR Tiered commission for a single ETF order."""
    if price_per_share <= 0:
        return IBKR_MIN_COMMISSION
    shares = trade_value_usd / price_per_share
    return max(IBKR_MIN_COMMISSION, IBKR_RATE_PER_SHARE * shares)


def compute_costs(current_weights: dict, optimal_weights: dict,
                  bl_returns: dict,
                  portfolio_value_usd: float = 100_000.0,
                  etf_prices: dict = None,
                  new_capital_usd: float = 0.0) -> dict:
    """
    Compute full rebalancing costs and the rebalance/invest decision.

    Parameters
    ----------
    current_weights    : {asset: weight} — today's actual weights
    optimal_weights    : {asset: weight} — BL recommended weights
    bl_returns         : {asset: annualized expected return} from BL
    portfolio_value_usd: total portfolio value in USD (default 100k)
    etf_prices         : {asset: last_price} for IBKR commission calc
    new_capital_usd    : new money to deploy (Scenario B)

    Returns
    -------
    dict with:
        scenario_a  — rebalancing cost breakdown
        scenario_b  — new-capital allocation (if new_capital_usd > 0)
        rebalance_recommended — bool
        cost_bps    — total cost in basis points
        cost_pct    — total cost as decimal
        expected_gain
        turnover
        band_breach — which assets have drifted beyond REBALANCE_BAND
    """
    if etf_prices is None:
        etf_prices = {a: 100.0 for a in ASSETS}   # fallback price for commission calc

    w_old = np.array([current_weights.get(a, 0.0) for a in ASSETS])
    w_new = np.array([optimal_weights.get(a, 0.0) for a in ASSETS])
    mu    = np.array([bl_returns.get(a, 0.0)      for a in ASSETS])

    delta    = w_new - w_old
    abs_d    = np.abs(delta)
    turnover = float(abs_d.sum())

    # ── Scenario A: rebalancing ────────────────────────────────────────────────
    commission_a = 0.0
    fx_cost_a    = 0.0
    spread_a     = 0.0
    breakdown_a  = {}

    for i, asset in enumerate(ASSETS):
        if abs_d[i] < 1e-4:
            breakdown_a[asset] = 0.0
            continue
        trade_val = abs_d[i] * portfolio_value_usd
        price     = etf_prices.get(asset, 100.0)
        comm      = _ibkr_commission(trade_val, price)
        fx        = IBKR_FX_COST * trade_val    # FX conversion if CHF account
        spread    = (IBKR_SPREAD_BPS / 10_000) * trade_val

        total_leg          = comm + fx + spread
        breakdown_a[asset] = round(total_leg, 2)
        commission_a      += comm
        fx_cost_a         += fx
        spread_a          += spread

    total_cost_a = commission_a + fx_cost_a + spread_a
    cost_pct_a   = total_cost_a / max(portfolio_value_usd, 1.0)
    cost_bps_a   = cost_pct_a * 1e4

    # Expected quarterly return gain from rebalancing
    expected_gain = float((w_new - w_old) @ (mu / 4.0))

    # Rebalance band check
    band_breach = {a: abs(delta[i]) > REBALANCE_BAND for i, a in enumerate(ASSETS)}
    any_breach  = any(band_breach.values())

    rebalance_recommended = any_breach and (expected_gain > cost_pct_a)

    # Breakeven: how many quarters to recover cost
    breakeven_qtrs = (cost_pct_a / expected_gain) if expected_gain > 0 else float('inf')

    scenario_a = dict(
        trade_amounts   = {a: round(delta[i] * portfolio_value_usd, 2) for i, a in enumerate(ASSETS)},
        commission_usd  = round(commission_a, 2),
        fx_cost_usd     = round(fx_cost_a, 2),
        spread_usd      = round(spread_a, 2),
        total_cost_usd  = round(total_cost_a, 2),
        cost_bps        = round(cost_bps_a, 2),
        breakdown       = breakdown_a,
        breakeven_qtrs  = round(breakeven_qtrs, 1) if breakeven_qtrs != float('inf') else None,
    )

    # ── Scenario B: new capital deployment ────────────────────────────────────
    scenario_b = None
    if new_capital_usd > 0:
        # Target weights after adding new capital
        total_after = portfolio_value_usd + new_capital_usd
        # Allocate new money to move actual holdings toward target weights
        # without selling: buy only
        target_values  = w_new * total_after
        current_values = w_old * portfolio_value_usd
        buys           = np.maximum(target_values - current_values, 0)

        # Scale buys to exactly = new_capital
        if buys.sum() > 0:
            buys = buys * (new_capital_usd / buys.sum())

        buy_commission = 0.0
        buy_breakdown  = {}
        for i, asset in enumerate(ASSETS):
            if buys[i] < 1.0:
                buy_breakdown[asset] = dict(amount=0, shares=0, commission=0)
                continue
            price  = etf_prices.get(asset, 100.0)
            shares = buys[i] / price
            comm   = _ibkr_commission(buys[i], price)
            buy_commission += comm
            buy_breakdown[asset] = dict(
                amount     = round(buys[i], 2),
                shares     = round(shares, 4),
                commission = round(comm, 2),
            )

        scenario_b = dict(
            new_capital_usd = new_capital_usd,
            buys            = buy_breakdown,
            total_commission= round(buy_commission, 2),
            resulting_weights = {a: round(float(
                (w_old[i] * portfolio_value_usd + buys[i]) / total_after
            ), 4) for i, a in enumerate(ASSETS)},
        )

    return dict(
        scenario_a            = scenario_a,
        scenario_b            = scenario_b,
        rebalance_recommended = rebalance_recommended,
        turnover              = round(turnover, 4),
        cost_bps              = round(cost_bps_a, 2),
        cost_pct              = round(cost_pct_a, 6),
        expected_gain         = round(expected_gain, 6),
        band_breach           = band_breach,
    )


if __name__ == '__main__':
    from config import BASE_WEIGHTS

    current = {'SP500': 0.80, 'NDX': 0.10, 'EUROPE': 0.10}
    optimal = {'SP500': 0.65, 'NDX': 0.22, 'EUROPE': 0.13}
    bl_ret  = {'SP500': 0.08, 'NDX': 0.11, 'EUROPE': 0.06}
    prices  = {'SP500': 5300.0, 'NDX': 480.0, 'EUROPE': 42.0}

    res = compute_costs(current, optimal, bl_ret,
                        portfolio_value_usd=150_000,
                        etf_prices=prices,
                        new_capital_usd=10_000)

    print(f"\nScenario A — Rebalancing")
    print(f"  Turnover      : {res['turnover']:.1%}")
    print(f"  Total cost    : ${res['scenario_a']['total_cost_usd']:.2f}  ({res['cost_bps']:.2f} bps)")
    print(f"  Expected gain : {res['expected_gain']:.3%}")
    print(f"  Rebalance?    : {'YES' if res['rebalance_recommended'] else 'NO'}")
    print(f"  Band breach   : {res['band_breach']}")

    if res['scenario_b']:
        print(f"\nScenario B — Deploy ${res['scenario_b']['new_capital_usd']:,.0f}")
        for asset, info in res['scenario_b']['buys'].items():
            if info['amount'] > 0:
                print(f"  Buy {asset}: ${info['amount']:,.2f} ({info['shares']:.2f} shares)  "
                      f"commission ${info['commission']:.2f}")
