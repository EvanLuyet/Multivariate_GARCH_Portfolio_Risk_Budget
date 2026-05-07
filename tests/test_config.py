"""
tests/test_config.py — Sanity checks on config.py constants

Tests:
  - BASE_WEIGHTS sum to 1
  - WEIGHT_BOUNDS are valid (lo < hi, 0 ≤ lo, hi ≤ 1)
  - BASE_WEIGHTS sit within WEIGHT_BOUNDS
  - ASSETS matches TICKERS keys
  - CONFIG_HASH is an 8-char hex string
  - Positive windows and model parameters
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config


class TestWeights:
    def test_base_weights_sum_to_one(self):
        total = sum(config.BASE_WEIGHTS.values())
        assert abs(total - 1.0) < 1e-10, f"BASE_WEIGHTS sum = {total}"

    def test_base_weights_within_bounds(self):
        for asset, weight in config.BASE_WEIGHTS.items():
            lo, hi = config.WEIGHT_BOUNDS[asset]
            assert lo <= weight <= hi, f"{asset}: weight {weight:.3f} outside [{lo}, {hi}]"

    def test_weight_bounds_valid(self):
        for asset, (lo, hi) in config.WEIGHT_BOUNDS.items():
            assert 0.0 <= lo < hi <= 1.0, f"{asset}: invalid bounds [{lo}, {hi}]"

    def test_all_assets_have_bounds(self):
        for asset in config.ASSETS:
            assert asset in config.WEIGHT_BOUNDS, f"{asset} missing from WEIGHT_BOUNDS"


class TestAssets:
    def test_assets_match_tickers(self):
        assert set(config.ASSETS) == set(config.TICKERS.keys())

    def test_assets_nonempty(self):
        assert len(config.ASSETS) > 0


class TestModelParams:
    def test_positive_windows(self):
        assert config.GARCH_WINDOW > 0
        assert config.CORR_WINDOW > 0
        assert config.TRADING_DAYS > 0
        assert config.LOOKBACK_YEARS > 0

    def test_hmm_states_at_least_two(self):
        assert config.HMM_STATES >= 2

    def test_prob_floor_valid(self):
        assert 0.0 <= config.HMM_PROB_FLOOR < 1.0 / config.HMM_STATES

    def test_bl_params_positive(self):
        assert config.DELTA > 0
        assert config.TAU > 0

    def test_min_train_qtrs_positive(self):
        assert config.MIN_TRAIN_QTRS > 0

    def test_risk_free_rate_reasonable(self):
        assert 0.0 <= config.RISK_FREE_RATE <= 0.10


class TestConfigHash:
    def test_hash_is_8_chars(self):
        assert len(config.CONFIG_HASH) == 8

    def test_hash_is_hex(self):
        int(config.CONFIG_HASH, 16)  # raises ValueError if not hex

    def test_hash_deterministic(self):
        from config import _compute_config_hash

        h1 = _compute_config_hash()
        h2 = _compute_config_hash()
        assert h1 == h2

    def test_hash_changes_with_param(self, monkeypatch):
        """Mutating a key parameter must change the hash."""
        import config as cfg

        original_hash = cfg.CONFIG_HASH
        monkeypatch.setattr(cfg, "GARCH_WINDOW", cfg.GARCH_WINDOW + 1)
        new_hash = cfg._compute_config_hash()
        assert new_hash != original_hash
