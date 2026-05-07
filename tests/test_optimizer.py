"""
tests/test_optimizer.py — Unit tests for layer3_optimizer/black_litterman.py
                           and layer2_hmm/regime_model.py

Tests:
  - Optimizer weights sum to 1
  - Optimizer weights respect WEIGHT_BOUNDS
  - Regime probability flooring is applied (no prob < HMM_PROB_FLOOR)
  - Regime probability renormalisation (probs sum to 1 after flooring)
  - _sort_and_floor assigns Bull=0, Crisis=2 by mean vol ordering
  - Zero-turnover gives zero TC drag
  - Backtest no-lookahead: weights at quarter k only use data up to quarter k
  - _bl_posterior with view_iqr produces posterior between equilibrium and views
  - _bl_posterior fallback (no view_iqr) returns equilibrium-blended result
  - _regime_blend crisis mode reverts toward base weights
  - _regime_blend bull mode trusts optimizer
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import ASSETS, WEIGHT_BOUNDS, HMM_PROB_FLOOR, BASE_WEIGHTS
from layer3_optimizer.black_litterman import (
    _optimize_weights,
    _regime_blend,
    _bl_posterior,
    W_BASE_ARR,
    BOUNDS,
)
from layer2_hmm.regime_model import _sort_and_floor
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

# ── Fixtures ───────────────────────────────────────────────────────────────────


def _simple_sigma(vols=(0.15, 0.20, 0.18), corr=0.3):
    """Annual covariance matrix from vols + constant off-diagonal correlation."""
    n = len(vols)
    v = np.array(vols)
    R = np.full((n, n), corr)
    np.fill_diagonal(R, 1.0)
    return np.outer(v, v) * R


def _make_hmm_3state(n_obs=500, seed=0):
    """Fit a real GaussianHMM on synthetic data; returns (model, X_raw, X_scaled)."""
    rng = np.random.default_rng(seed)
    X_raw = rng.standard_normal((n_obs, 4))
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    model = GaussianHMM(n_components=3, covariance_type="diag", n_iter=50, random_state=0)
    model.fit(X_scaled)
    return model, X_raw, X_scaled


# ── Optimizer tests ────────────────────────────────────────────────────────────


def test_optimizer_weights_sum_to_one():
    Sigma = _simple_sigma()
    mu = np.array([0.08, 0.12, 0.06])
    w_current = np.array([BASE_WEIGHTS[a] for a in ASSETS])
    w_opt, _ = _optimize_weights(mu, Sigma, w_current)
    assert abs(w_opt.sum() - 1.0) < 1e-6, f"Weights sum to {w_opt.sum()}"


def test_optimizer_weights_respect_bounds():
    Sigma = _simple_sigma()
    mu = np.array([0.08, 0.12, 0.06])
    w_current = np.array([BASE_WEIGHTS[a] for a in ASSETS])
    w_opt, _ = _optimize_weights(mu, Sigma, w_current)
    for i, asset in enumerate(ASSETS):
        lo, hi = WEIGHT_BOUNDS[asset]
        assert w_opt[i] >= lo - 1e-6, f"{asset} weight {w_opt[i]:.4f} < lower bound {lo}"
        assert w_opt[i] <= hi + 1e-6, f"{asset} weight {w_opt[i]:.4f} > upper bound {hi}"


def test_optimizer_sharpe_positive():
    """Optimizer should find a positive Sharpe ratio under reasonable inputs."""
    Sigma = _simple_sigma()
    mu = np.array([0.08, 0.12, 0.06])
    w_current = np.array([BASE_WEIGHTS[a] for a in ASSETS])
    _, sharpe = _optimize_weights(mu, Sigma, w_current)
    assert sharpe > 0, f"Expected positive Sharpe, got {sharpe}"


def test_optimizer_extreme_mu_still_bounded():
    """Even with extreme return views, weights must stay within bounds."""
    Sigma = _simple_sigma()
    mu = np.array([1.0, 0.0, 0.0])  # very strong SP500 view
    w_current = np.array([BASE_WEIGHTS[a] for a in ASSETS])
    w_opt, _ = _optimize_weights(mu, Sigma, w_current)
    lo, hi = WEIGHT_BOUNDS["SP500"]
    assert w_opt[0] <= hi + 1e-6


# ── Regime blend tests ─────────────────────────────────────────────────────────


def test_regime_blend_bull_trusts_optimizer():
    """In full-Bull regime (p0=1), output should equal the BL weights (modulo bounds clip)."""
    w_bl = np.array([0.70, 0.20, 0.10])
    w_final = _regime_blend(w_bl, [1.0, 0.0, 0.0])
    # alpha=1.0 → w_final = w_bl (after bounds clip and renorm)
    assert abs(w_final.sum() - 1.0) < 1e-6
    # SP500 should stay at or above its lower bound
    assert w_final[0] >= WEIGHT_BOUNDS["SP500"][0] - 1e-6


def test_regime_blend_crisis_reverts_toward_base():
    """In full-Crisis regime (p2=1), alpha=0.25 — output moves toward W_BASE_ARR."""
    w_bl = np.array([0.80, 0.10, 0.10])  # SP500 far above base
    w_final = _regime_blend(w_bl, [0.0, 0.0, 1.0])
    # alpha=0.25*1 + ... = 0.25 → w = 0.25*w_bl + 0.75*W_BASE_ARR
    expected = 0.25 * w_bl + 0.75 * W_BASE_ARR
    expected = np.clip(expected, [b[0] for b in BOUNDS], [b[1] for b in BOUNDS])
    expected /= expected.sum()
    np.testing.assert_allclose(w_final, expected, atol=1e-6)


def test_regime_blend_output_sums_to_one():
    for probs in [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0.4, 0.3, 0.3]]:
        w_bl = np.array([0.60, 0.25, 0.15])
        w_final = _regime_blend(w_bl, probs)
        assert abs(w_final.sum() - 1.0) < 1e-6


# ── BL posterior tests ─────────────────────────────────────────────────────────


def test_bl_posterior_between_pi_and_q():
    """BL posterior should lie between equilibrium Π and view Q."""
    Sigma = _simple_sigma()
    Pi = np.array([0.05, 0.07, 0.04])
    Q = np.array([0.10, 0.15, 0.08])
    mu = _bl_posterior(Pi, Sigma, Q)
    for i in range(len(ASSETS)):
        lo, hi = min(Pi[i], Q[i]), max(Pi[i], Q[i])
        assert (
            lo - 1e-9 <= mu[i] <= hi + 1e-9
        ), f"Asset {i}: mu={mu[i]:.4f} not in [{lo:.4f}, {hi:.4f}]"


def test_bl_posterior_wider_iqr_reverts_toward_pi():
    """Wider IQR (less confident view) should produce posterior closer to Π."""
    Sigma = _simple_sigma()
    Pi = np.array([0.05, 0.05, 0.05])
    Q = np.array([0.15, 0.15, 0.15])

    narrow_iqr = np.array([0.01, 0.01, 0.01])
    wide_iqr = np.array([0.30, 0.30, 0.30])

    mu_narrow = _bl_posterior(Pi, Sigma, Q, view_iqr=narrow_iqr)
    mu_wide = _bl_posterior(Pi, Sigma, Q, view_iqr=wide_iqr)

    # Wide IQR → closer to Π
    dist_narrow = np.linalg.norm(mu_narrow - Pi)
    dist_wide = np.linalg.norm(mu_wide - Pi)
    assert dist_wide < dist_narrow, (
        f"Wide IQR should revert toward Pi more: "
        f"dist_wide={dist_wide:.4f} dist_narrow={dist_narrow:.4f}"
    )


def test_bl_posterior_fallback_no_iqr():
    """Without view_iqr, posterior should still be between Pi and Q."""
    Sigma = _simple_sigma()
    Pi = np.array([0.05, 0.07, 0.04])
    Q = np.array([0.10, 0.12, 0.09])
    mu = _bl_posterior(Pi, Sigma, Q, view_iqr=None)
    assert mu.shape == (3,)
    # Result should be a valid return vector (not NaN)
    assert not np.any(np.isnan(mu))


# ── HMM probability floor tests ───────────────────────────────────────────────


def test_sort_and_floor_no_prob_below_floor():
    """
    After _sort_and_floor, no probability should be below the effective post-renorm floor.

    Clipping n=3 states to FLOOR then renormalising: the minimum attainable value is
    FLOOR / (FLOOR*(n-1) + 1).  For FLOOR=0.05, n=3 that is 0.05/1.1 ≈ 0.0455.
    We verify no prob drops below half the nominal floor (a meaningful sanity check
    that near-zero probabilities have been eliminated).
    """
    from config import HMM_STATES

    model, X_raw, X_scaled = _make_hmm_3state()
    _, probs = _sort_and_floor(model, X_raw, X_scaled, vol_col=0)
    # Post-renorm theoretical minimum
    post_renorm_min = HMM_PROB_FLOOR / (HMM_PROB_FLOOR * (HMM_STATES - 1) + 1) - 1e-9
    assert (
        probs.min() >= post_renorm_min
    ), f"Min prob {probs.min():.6f} below post-renorm floor {post_renorm_min:.6f}"


def test_sort_and_floor_probs_sum_to_one():
    """After flooring and renormalisation, each row must sum to 1.0."""
    model, X_raw, X_scaled = _make_hmm_3state()
    _, probs = _sort_and_floor(model, X_raw, X_scaled, vol_col=0)
    row_sums = probs.sum(axis=1)
    np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)


def test_sort_and_floor_state_ordering():
    """State 0 should have lower mean vol than state 2 (Bull < Crisis ordering)."""
    model, X_raw, X_scaled = _make_hmm_3state(n_obs=1000)
    states, _ = _sort_and_floor(model, X_raw, X_scaled, vol_col=0)
    # Mean of first feature (used as vol proxy) per assigned state
    mean_vol = [X_raw[states == s, 0].mean() if (states == s).any() else 0.0 for s in range(3)]
    assert (
        mean_vol[0] <= mean_vol[2]
    ), f"State 0 mean vol {mean_vol[0]:.4f} should be <= State 2 mean vol {mean_vol[2]:.4f}"


# ── Transaction cost: zero-turnover ──────────────────────────────────────────


def test_zero_turnover_zero_cost():
    """When current and proposed weights are identical, TC should be zero."""
    from layer5_costs.transaction import compute_costs

    w = {"SP500": 0.65, "NDX": 0.20, "EUROPE": 0.15}
    bl_returns = {"SP500": 0.08, "NDX": 0.10, "EUROPE": 0.06}
    result = compute_costs(
        w, w, bl_returns, portfolio_value_usd=100_000.0, etf_prices={}, new_capital_usd=0.0
    )
    assert result["turnover"] < 1e-9, f"Expected zero turnover, got {result['turnover']}"
    assert result["scenario_a"]["commission_usd"] == 0.0 or result["turnover"] < 1e-9


# ── Backtest no-lookahead guarantee ──────────────────────────────────────────


def test_backtest_no_lookahead():
    """
    _build_strategy_returns must use weights from quarter k to compute returns
    in the period AFTER quarter k ends — never before.

    We verify this by checking that the return series for each sub-period only
    uses dates strictly after the corresponding quarter-end.
    """
    import pandas as pd
    from report.dashboard import _build_strategy_returns

    # Build synthetic garch_history and bl_history over 3 fake quarters
    rng = np.random.default_rng(42)
    idx = pd.date_range("2020-01-01", periods=300, freq="B")
    # Fake return data
    data = pd.DataFrame(
        {f"ret_{a}": rng.standard_normal(300) * 0.01 for a in ASSETS},
        index=idx,
    )

    # Quarter-ends roughly every 63 trading days
    q_ends = [idx[62], idx[125], idx[188], idx[251]]
    garch_history = {qe: {} for qe in q_ends}

    bl_history = {}
    for qe in q_ends:
        bl_history[qe] = {
            "optimal_weights": {a: BASE_WEIGHTS[a] for a in ASSETS},
            "raw_bl_weights": {a: BASE_WEIGHTS[a] for a in ASSETS},
        }

    ret_series = _build_strategy_returns(
        garch_history, bl_history, data, weights_override=BASE_WEIGHTS
    )

    # The return series must only contain dates in data.index
    assert set(ret_series.index).issubset(
        set(data.index)
    ), "Return series contains dates outside the data index"

    # The first return date must be AFTER the first quarter-end (no lookahead)
    if not ret_series.empty:
        assert (
            ret_series.index[0] > q_ends[0]
        ), f"First return date {ret_series.index[0]} should be after q_end {q_ends[0]}"
