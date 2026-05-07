"""
layer2_hmm/regime_model.py — Hidden Markov Model regime detection.

Financial markets cycle through identifiable but unobservable states (regimes).
A Gaussian HMM learns to identify these states from a multivariate feature
vector combining GARCH vol, realized vol, macro conditions (VIX, yield curve,
credit spreads) and price momentum.

KEY: The HMM is trained on DAILY observations (~2 500 rows over 10 years),
not on quarterly snapshots. This gives the model enough data to reliably
separate three regimes and produce well-calibrated posteriors. Regime labels
are then looked up at each quarter-end date for use in the BL optimizer.

We use HMM_STATES=3 states and sort them by mean annualized vol so that
the labelling is always: 0 = 🟢 Bull (low vol), 1 = 🟡 Transition, 2 = 🔴 Crisis.
The posterior state probabilities are used in Layer 3 (BL optimizer) to blend
between aggressive (Bull) and conservative (Crisis) weight allocations.

The fitted model and daily feature matrix are cached to avoid re-training on
repeated runs the same day.
"""

import sys
import warnings
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (ASSETS, HMM_STATES, HMM_PROB_FLOOR,
                    RANDOM_SEED, MODEL_CACHE_DIR, TRADING_DAYS)

REGIME_LABELS = {0: '🟢 Bull', 1: '🟡 Transition', 2: '🔴 Crisis'}
REGIME_COLORS = {0: '#2ecc71', 1: '#f39c12', 2: '#e74c3c'}


def _build_daily_features(daily_garch_vols: pd.DataFrame,
                           data: pd.DataFrame) -> pd.DataFrame:
    """
    Build the HMM input feature matrix at daily frequency.

    Feature design principle: every feature must have meaningfully different
    distributions across Bull / Transition / Crisis regimes. Smooth signals
    (e.g. full-sample GARCH conditional vol) are poor discriminators because
    they barely change between calm and stressed days.

    Core features (work even without FRED data):
      - Realized vol at 5d / 21d / 63d — spikes sharply in crises
      - Vol acceleration (5d/63d ratio) — detects regime transitions early
      - Rolling max-drawdown (63d) — distinguishes bear markets
      - Absolute daily return of S&P — captures tail shock days
      - 21d and 63d momentum — trend direction signal
    Optional macro features (if FRED_API_KEY is set):
      - VIX, yield curve, credit spread, USD index
    """
    sp_ret = 'ret_SP500'
    r      = data[sp_ret].reindex(daily_garch_vols.index)
    feat   = pd.DataFrame(index=daily_garch_vols.index)

    # ── Multi-horizon realized vol — the primary regime discriminator ─────────
    for window, label in [(5, '5d'), (21, '21d'), (63, '63d')]:
        feat[f'realized_vol_{label}'] = r.rolling(window).std() * np.sqrt(TRADING_DAYS)

    # ── Vol acceleration: short/long ratio spikes at regime transitions ───────
    feat['vol_accel'] = feat['realized_vol_5d'] / (feat['realized_vol_63d'] + 1e-8)

    # ── Rolling 63-day max drawdown — negative in bear markets ───────────────
    prices = data[f'price_SP500'].reindex(daily_garch_vols.index)
    roll_max = prices.rolling(63, min_periods=1).max()
    feat['drawdown_63d'] = (prices - roll_max) / (roll_max + 1e-8)

    # ── Absolute daily return — captures tail shock days ──────────────────────
    feat['abs_ret'] = r.abs()

    # ── Momentum at 21d and 63d — trend signal ───────────────────────────────
    feat['momentum_21d'] = r.rolling(21).sum()
    feat['momentum_63d'] = r.rolling(63).sum()

    # ── Macro features (optional — zero-filled if FRED key absent) ───────────
    for col in ['vix', 'yield_curve', 'credit_spread', 'usd_index']:
        feat[col] = data[col].reindex(feat.index, method='ffill') \
                   if col in data.columns else 0.0

    feat = feat.ffill().fillna(feat.median(numeric_only=True)).fillna(0.0)
    feat = feat.dropna()
    return feat


def run_hmm(garch_history: dict, data: pd.DataFrame) -> dict:
    """
    Fit a 3-state Gaussian HMM on daily features and label each day with a regime.

    Training on daily data gives ~2 500 observations (vs ~40 quarterly) which:
      1. Produces stable, well-separated Gaussian clusters per regime
      2. Gives meaningful posterior probabilities (not degenerate [1, 0, 0])
      3. Enables a rich daily regime-history plot in the dashboard

    Regime labels at each quarter-end are looked up from the daily series and
    stored in `history` for use by the BL optimizer.

    States are sorted ascending by mean GARCH vol so Bull=0, Transition=1,
    Crisis=2 — labels remain consistent across re-fits.

    Returns a dict with:
        history        — {qe: {regime, regime_probs, regime_label, regime_color}}
        regime_series  — pd.Series of integer regime, DAILY index
        probs_df       — pd.DataFrame of posterior probabilities, DAILY index (n×3)
        feature_df     — the daily feature matrix (for diagnostics)
        model / scaler — fitted objects
    """
    from layer1_garch.garch_model import get_daily_garch_vols

    cache_dir  = Path(MODEL_CACHE_DIR)
    cache_dir.mkdir(exist_ok=True)
    cache_file = cache_dir / f'hmm_{datetime.today().strftime("%Y%m%d")}.joblib'

    daily_garch_vols = get_daily_garch_vols(data)
    feat_df          = _build_daily_features(daily_garch_vols, data)

    X_raw    = feat_df.values.astype(float)
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    n_obs = len(feat_df)
    print(f'Layer 2 — HMM training on {n_obs} daily observations '
          f'({feat_df.index[0].date()} → {feat_df.index[-1].date()}) …')

    # ── Fit or load — validate feature count before trusting cache ────────────
    n_features = X_scaled.shape[1]
    cache_valid = False
    if cache_file.exists():
        try:
            model, scaler = joblib.load(cache_file)
            # A model trained on a different number of features will crash on
            # predict(); check means_ shape before accepting the cache.
            if model.means_.shape[1] == n_features:
                print('Layer 2 — Loading HMM from cache …')
                cache_valid = True
            else:
                print(f'Layer 2 — Cache feature mismatch '
                      f'({model.means_.shape[1]} vs {n_features}), re-fitting …')
                cache_file.unlink()
        except Exception:
            cache_file.unlink()

    if not cache_valid:
        # ── Multi-seed fit: pick the model with the highest log-likelihood ────
        # 'diag' covariance (n_states × n_features × 2 params) is far more
        # stable than 'full' (n_states × n_features² params) with ~2 500 rows.
        # Multiple random seeds guard against local optima.
        best_model, best_score = None, -np.inf
        for seed in [RANDOM_SEED, RANDOM_SEED + 1, RANDOM_SEED + 2]:
            candidate = GaussianHMM(
                n_components=HMM_STATES,
                covariance_type='diag',
                n_iter=300,
                random_state=seed,
                tol=1e-4,
            )
            candidate.fit(X_scaled)
            score = candidate.score(X_scaled)
            if score > best_score:
                best_score = score
                best_model = candidate
        model = best_model
        joblib.dump((model, scaler), cache_file)

    raw_states  = model.predict(X_scaled)
    state_probs = model.predict_proba(X_scaled)

    # ── Sort states ascending by mean realized vol (state 0 = lowest = Bull) ──
    vol_col_idx = feat_df.columns.get_loc('realized_vol_21d')
    state_means = np.array([
        X_raw[raw_states == s, vol_col_idx].mean()
        if (raw_states == s).any() else 0.0
        for s in range(HMM_STATES)
    ])
    order   = np.argsort(state_means)
    remap   = {old: new for new, old in enumerate(order)}
    states  = np.array([remap[s] for s in raw_states])
    probs   = state_probs[:, order]

    # ── Probability floor — prevents degenerate [1, 0, 0] posteriors ──────────
    # Even with 2 500 daily observations the HMM can still produce near-certain
    # posteriors on very calm or very extreme days. The floor ensures each regime
    # always retains at least ~HMM_PROB_FLOOR weight in the BL blend.
    probs = np.clip(probs, HMM_PROB_FLOOR, 1.0)
    probs = probs / probs.sum(axis=1, keepdims=True)

    regime_series = pd.Series(states, index=feat_df.index, name='regime')
    probs_df      = pd.DataFrame(
        probs, index=feat_df.index,
        columns=[f'p{i}' for i in range(HMM_STATES)],
    )

    # ── Build quarterly lookup for downstream layers ───────────────────────────
    q_ends  = sorted(garch_history.keys())
    history = {}
    for qe in q_ends:
        # Find the closest available daily date at or before qe
        available = regime_series.index[regime_series.index <= qe]
        if available.empty:
            continue
        ref = available[-1]
        r   = int(regime_series.loc[ref])
        p   = probs_df.loc[ref].values.tolist()
        history[qe] = dict(
            regime       = r,
            regime_probs = p,
            regime_label = REGIME_LABELS[r],
            regime_color = REGIME_COLORS[r],
        )

    counts = pd.Series(states).value_counts().sort_index()
    print(f'Layer 2 — Regime distribution: '
          + '  '.join(f'{REGIME_LABELS[i]}: {counts.get(i, 0)} days '
                      f'({counts.get(i, 0)/n_obs:.0%})'
                      for i in range(HMM_STATES)))

    return dict(
        history       = history,
        regime_series = regime_series,   # daily
        probs_df      = probs_df,        # daily
        feature_df    = feat_df,         # daily
        model         = model,
        scaler        = scaler,
    )


if __name__ == '__main__':
    from data.fetcher import fetch_data
    from layer1_garch.garch_model import run_garch
    from config import BASE_WEIGHTS

    df  = fetch_data()
    g1  = run_garch(df, BASE_WEIGHTS)
    res = run_hmm(g1, df)

    print(f'\nDaily regime series shape : {res["regime_series"].shape}')
    print(f'Feature matrix shape      : {res["feature_df"].shape}')

    latest = max(res['history'].keys())
    r      = res['history'][latest]
    print(f'\nLatest quarter-end : {latest.date()}')
    print(f'Regime             : {r["regime_label"]}')
    print(f'Probabilities      : {[f"{p:.1%}" for p in r["regime_probs"]]}')
