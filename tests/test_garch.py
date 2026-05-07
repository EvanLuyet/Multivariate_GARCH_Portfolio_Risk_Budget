"""
tests/test_garch.py — Unit tests for layer1_garch/garch_model.py

Tests:
  - risk contributions sum to ≈ 1
  - covariance matrix is positive-definite (all eigenvalues > 0)
  - covariance matrix is symmetric
  - correlation matrix diagonal == 1
  - eigenvalue floor prevents negative eigenvalues
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from layer1_garch.garch_model import compute_risk_contributions, _build_cov

# ── Fixtures ───────────────────────────────────────────────────────────────────


def _simple_cov(corr=0.5, vols=(0.15, 0.20, 0.18)):
    """Build a simple 3×3 daily covariance from annual vols + constant correlation."""
    daily_sigmas = np.array(vols) / np.sqrt(252)
    D = np.diag(daily_sigmas)
    R = np.full((3, 3), corr)
    np.fill_diagonal(R, 1.0)
    return D @ R @ D, daily_sigmas


# ── Risk-contribution tests ────────────────────────────────────────────────────


class TestRiskContributions:
    def test_sums_to_one(self):
        Sigma, daily_sigmas = _simple_cov()
        w = np.array([0.65, 0.20, 0.15])
        rc, _ = compute_risk_contributions(w, Sigma)
        assert abs(rc.sum() - 1.0) < 1e-10, f"RC sum = {rc.sum()}"

    def test_all_nonnegative(self):
        Sigma, _ = _simple_cov()
        w = np.array([0.65, 0.20, 0.15])
        rc, _ = compute_risk_contributions(w, Sigma)
        assert (rc >= 0).all(), f"Negative RC: {rc}"

    def test_equal_weights_equal_vols_equal_corr(self):
        """Equal weights + equal vols + equal corr → equal risk contributions."""
        Sigma, _ = _simple_cov(corr=0.5, vols=(0.15, 0.15, 0.15))
        w = np.array([1 / 3, 1 / 3, 1 / 3])
        rc, _ = compute_risk_contributions(w, Sigma)
        np.testing.assert_allclose(rc, [1 / 3, 1 / 3, 1 / 3], atol=1e-8)

    def test_portfolio_vol_positive(self):
        Sigma, _ = _simple_cov()
        w = np.array([0.65, 0.20, 0.15])
        _, port_vol = compute_risk_contributions(w, Sigma)
        assert port_vol > 0

    def test_single_asset_all_risk(self):
        """A portfolio 100% in one asset should have RC = [1, 0, 0]."""
        Sigma, _ = _simple_cov()
        w = np.array([1.0, 0.0, 0.0])
        rc, _ = compute_risk_contributions(w, Sigma)
        assert abs(rc[0] - 1.0) < 1e-8

    def test_higher_vol_asset_gets_more_risk(self):
        """Asset with higher vol should generally have higher RC given equal weights."""
        Sigma, _ = _simple_cov(corr=0.0, vols=(0.10, 0.30, 0.10))
        w = np.array([1 / 3, 1 / 3, 1 / 3])
        rc, _ = compute_risk_contributions(w, Sigma)
        assert rc[1] > rc[0]


# ── Covariance-matrix tests ────────────────────────────────────────────────────


class TestBuildCov:
    def _make_residuals(self, n=63, seed=42):
        rng = np.random.default_rng(seed)
        assets = ["SP500", "NDX", "EUROPE"]
        data = rng.standard_normal((n, 3))
        return pd.DataFrame(data, columns=assets)

    def test_positive_definite(self):
        resid = self._make_residuals()
        daily_sigmas = np.array([0.01, 0.012, 0.009])
        Sigma, R = _build_cov(daily_sigmas, resid)
        eigvals = np.linalg.eigvalsh(Sigma)
        assert (eigvals > 0).all(), f"Non-positive eigenvalues: {eigvals}"

    def test_symmetric(self):
        resid = self._make_residuals()
        daily_sigmas = np.array([0.01, 0.012, 0.009])
        Sigma, R = _build_cov(daily_sigmas, resid)
        np.testing.assert_allclose(Sigma, Sigma.T, atol=1e-12)

    def test_correlation_diagonal_ones(self):
        resid = self._make_residuals()
        daily_sigmas = np.array([0.01, 0.012, 0.009])
        _, R = _build_cov(daily_sigmas, resid)
        np.testing.assert_allclose(np.diag(R), np.ones(3), atol=1e-10)

    def test_eigenvalue_floor_prevents_negative(self):
        """Near-singular residuals (high correlation) must still produce a PD matrix."""
        rng = np.random.default_rng(7)
        n = 63
        # Two highly correlated columns + tiny independent noise → near-singular R
        base = rng.standard_normal(n)
        noise = rng.standard_normal((n, 3)) * 1e-4
        resid = pd.DataFrame(
            np.column_stack([base, base, base]) + noise,
            columns=["SP500", "NDX", "EUROPE"],
        )
        daily_sigmas = np.array([0.01, 0.01, 0.01])
        Sigma, _ = _build_cov(daily_sigmas, resid)
        eigvals = np.linalg.eigvalsh(Sigma)
        assert (eigvals > 0).all(), f"Non-positive eigenvalues after floor: {eigvals}"

    def test_shape(self):
        resid = self._make_residuals()
        daily_sigmas = np.array([0.01, 0.012, 0.009])
        Sigma, R = _build_cov(daily_sigmas, resid)
        assert Sigma.shape == (3, 3)
        assert R.shape == (3, 3)
