"""
tests/test_forecaster.py — Unit tests for layer4_ml/return_forecaster.py

Tests:
  - _point_and_bounds: q50 (display point) is always within [q25, q75]
  - _point_and_bounds: returns float values
  - _compute_target: returns dict keyed by ASSETS
  - Feature column list contains no duplicates
  - Fallback bounds are valid when LGB fails
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from layer4_ml.return_forecaster import _point_and_bounds, _compute_target, FEATURE_COLS
from config import ASSETS

# ── _point_and_bounds ─────────────────────────────────────────────────────────


def _make_training_data(n=50, seed=0):
    rng = np.random.default_rng(seed)
    n_feat = len(FEATURE_COLS)
    X = rng.standard_normal((n, n_feat))
    y = rng.standard_normal(n) * 0.05
    X_pred = rng.standard_normal((1, n_feat))
    valid = np.ones(n, dtype=bool)
    return X, y, X_pred, valid


class TestPointAndBounds:
    def test_q50_within_q25_q75(self):
        """The display point (q50) must always be between q25 and q75."""
        X, y, X_pred, valid = _make_training_data(n=60)
        xgb_mean, q25, q50, q75 = _point_and_bounds(X, y, X_pred, valid)
        assert q25 <= q50 <= q75, f"Interval violated: q25={q25:.4f} q50={q50:.4f} q75={q75:.4f}"

    def test_returns_floats(self):
        X, y, X_pred, valid = _make_training_data(n=60)
        results = _point_and_bounds(X, y, X_pred, valid)
        for val in results:
            assert isinstance(val, float), f"Expected float, got {type(val)}"

    def test_q25_le_q75(self):
        X, y, X_pred, valid = _make_training_data(n=60)
        _, q25, _, q75 = _point_and_bounds(X, y, X_pred, valid)
        assert q25 <= q75, f"q25={q25} > q75={q75}"

    def test_with_partial_valid_mask(self):
        """Should work when only half the training rows are valid."""
        X, y, X_pred, _ = _make_training_data(n=80)
        valid = np.zeros(80, dtype=bool)
        valid[:40] = True
        xgb_mean, q25, q50, q75 = _point_and_bounds(X, y, X_pred, valid)
        assert q25 <= q50 <= q75

    def test_fallback_interval_valid_when_lgb_might_fail(self):
        """Even with minimal data the fallback symmetric ±3% interval is valid."""
        X, y, X_pred, valid = _make_training_data(n=30, seed=99)
        _, q25, q50, q75 = _point_and_bounds(X, y, X_pred, valid)
        # Fallback is xgb_mean ± 0.03; q50 set to xgb_mean → still within interval
        assert q25 <= q50 <= q75


# ── _compute_target ───────────────────────────────────────────────────────────


class TestComputeTarget:
    def _make_data(self):
        idx = pd.bdate_range("2020-01-01", periods=100)
        data = pd.DataFrame(
            {f"ret_{a}": np.random.default_rng(42).standard_normal(100) * 0.01 for a in ASSETS},
            index=idx,
        )
        return data

    def test_returns_all_assets(self):
        data = self._make_data()
        qe = data.index[10]
        next_qe = data.index[30]
        result = _compute_target(qe, next_qe, data)
        assert set(result.keys()) == set(ASSETS)

    def test_empty_period_returns_nan(self):
        data = self._make_data()
        qe = data.index[10]
        # qe == next_qe → empty mask
        result = _compute_target(qe, qe, data)
        for a in ASSETS:
            assert np.isnan(result[a]), f"{a} should be nan for empty period"


# ── Feature columns ───────────────────────────────────────────────────────────


class TestFeatureCols:
    def test_no_duplicates(self):
        assert len(FEATURE_COLS) == len(set(FEATURE_COLS)), "Duplicate feature columns"

    def test_nonempty(self):
        assert len(FEATURE_COLS) > 0
