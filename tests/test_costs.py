"""
tests/test_costs.py — Unit tests for layer5_costs/transaction.py

Tests:
  - IBKR commission minimum is respected
  - Per-share commission calculation
  - Scenario A trade amounts and signs
  - Scenario B buy-only (no negative buys)
  - Resulting weights after Scenario B sum to ≈ 1
  - Band-breach detection
  - Weight bounds post-rebalance
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from layer5_costs.transaction import compute_costs, _ibkr_commission
from config import IBKR_MIN_COMMISSION, IBKR_RATE_PER_SHARE, ASSETS

# ── IBKR commission ────────────────────────────────────────────────────────────


class TestIBKRCommission:
    def test_minimum_enforced_for_cheap_trade(self):
        """A $10 trade at $500/share → 0.02 shares → $0.0001 < $1 min."""
        comm = _ibkr_commission(trade_value_usd=10.0, price_per_share=500.0)
        assert comm == IBKR_MIN_COMMISSION

    def test_per_share_rate_dominates_large_trade(self):
        """$10 000 trade at $10/share → 1 000 shares × $0.005 = $5 > $1."""
        comm = _ibkr_commission(trade_value_usd=10_000.0, price_per_share=10.0)
        assert comm == pytest.approx(IBKR_RATE_PER_SHARE * 1_000, rel=1e-6)

    def test_zero_price_returns_minimum(self):
        comm = _ibkr_commission(trade_value_usd=1_000.0, price_per_share=0.0)
        assert comm == IBKR_MIN_COMMISSION

    def test_negative_price_returns_minimum(self):
        comm = _ibkr_commission(trade_value_usd=1_000.0, price_per_share=-5.0)
        assert comm == IBKR_MIN_COMMISSION


# ── compute_costs — Scenario A ─────────────────────────────────────────────────


@pytest.fixture
def base_inputs():
    return dict(
        current_weights={"SP500": 0.80, "NDX": 0.10, "EUROPE": 0.10},
        optimal_weights={"SP500": 0.65, "NDX": 0.22, "EUROPE": 0.13},
        bl_returns={"SP500": 0.08, "NDX": 0.11, "EUROPE": 0.06},
        portfolio_value_usd=150_000.0,
        etf_prices={"SP500": 530.0, "NDX": 480.0, "EUROPE": 42.0},
        new_capital_usd=0.0,
    )


class TestScenarioA:
    def test_turnover_correct(self, base_inputs):
        res = compute_costs(**base_inputs)
        expected = abs(0.65 - 0.80) + abs(0.22 - 0.10) + abs(0.13 - 0.10)
        assert res["turnover"] == pytest.approx(expected, rel=1e-6)

    def test_trade_directions(self, base_inputs):
        """Over-weight SP500 should be sold (negative amount)."""
        res = compute_costs(**base_inputs)
        trades = res["scenario_a"]["trade_amounts"]
        assert trades["SP500"] < 0, "SP500 overweight — should sell"
        assert trades["NDX"] > 0, "NDX underweight — should buy"

    def test_cost_positive(self, base_inputs):
        res = compute_costs(**base_inputs)
        assert res["scenario_a"]["total_cost_usd"] > 0

    def test_band_breach_detected(self, base_inputs):
        res = compute_costs(**base_inputs)
        # SP500 drift = |0.65 - 0.80| = 15pp > 5pp band
        assert bool(res["band_breach"]["SP500"]) is True

    def test_no_band_breach_when_drift_small(self):
        res = compute_costs(
            current_weights={"SP500": 0.65, "NDX": 0.20, "EUROPE": 0.15},
            optimal_weights={"SP500": 0.66, "NDX": 0.20, "EUROPE": 0.14},
            bl_returns={"SP500": 0.08, "NDX": 0.11, "EUROPE": 0.06},
            portfolio_value_usd=100_000.0,
        )
        assert not any(res["band_breach"].values())

    def test_scenario_b_none_when_no_capital(self, base_inputs):
        res = compute_costs(**base_inputs)
        assert res["scenario_b"] is None


# ── compute_costs — Scenario B ─────────────────────────────────────────────────


class TestScenarioB:
    def test_no_negative_buys(self, base_inputs):
        base_inputs["new_capital_usd"] = 10_000.0
        res = compute_costs(**base_inputs)
        sb = res["scenario_b"]
        assert sb is not None
        for asset, info in sb["buys"].items():
            assert info["amount"] >= 0, f"Negative buy for {asset}"

    def test_buy_amounts_sum_to_new_capital(self, base_inputs):
        base_inputs["new_capital_usd"] = 10_000.0
        res = compute_costs(**base_inputs)
        sb = res["scenario_b"]
        total_buys = sum(info["amount"] for info in sb["buys"].values())
        assert total_buys == pytest.approx(10_000.0, rel=1e-4)

    def test_resulting_weights_sum_to_one(self, base_inputs):
        base_inputs["new_capital_usd"] = 10_000.0
        res = compute_costs(**base_inputs)
        sb = res["scenario_b"]
        weight_sum = sum(sb["resulting_weights"].values())
        assert weight_sum == pytest.approx(1.0, abs=1e-4)

    def test_resulting_weights_all_assets_present(self, base_inputs):
        base_inputs["new_capital_usd"] = 10_000.0
        res = compute_costs(**base_inputs)
        sb = res["scenario_b"]
        assert set(sb["resulting_weights"].keys()) == set(ASSETS)


# ── Weight-bounds sanity ───────────────────────────────────────────────────────


class TestWeightBounds:
    def test_config_weights_sum_to_one(self):
        from config import BASE_WEIGHTS

        assert sum(BASE_WEIGHTS.values()) == pytest.approx(1.0, abs=1e-10)

    def test_config_bounds_positive(self):
        from config import WEIGHT_BOUNDS

        for asset, (lo, hi) in WEIGHT_BOUNDS.items():
            assert lo >= 0, f"{asset} lower bound < 0"
            assert hi <= 1, f"{asset} upper bound > 1"
            assert lo < hi, f"{asset} lower bound >= upper bound"

    def test_base_weights_within_bounds(self):
        from config import BASE_WEIGHTS, WEIGHT_BOUNDS

        for asset in ASSETS:
            lo, hi = WEIGHT_BOUNDS[asset]
            w = BASE_WEIGHTS[asset]
            assert lo <= w <= hi, f"{asset} base weight {w} outside [{lo}, {hi}]"
