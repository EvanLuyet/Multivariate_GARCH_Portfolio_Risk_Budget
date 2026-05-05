"""
layer2_hmm/regime_model.py — Hidden Markov Model regime detection.

Financial markets cycle through identifiable but unobservable states (regimes).
A Gaussian HMM learns to identify these states from a multivariate feature
vector combining GARCH vol, realized vol, macro conditions (VIX, yield curve,
credit spreads) and price momentum.

We use HMM_STATES=3 states and sort them by mean portfolio volatility so that
the labelling is always: 0 = 🟢 Bull (low vol), 1 = 🟡 Transition, 2 = 🔴 Crisis.
The posterior state probabilities are used in Layer 3 (BL optimizer) to blend
between aggressive (Bull) and conservative (Crisis) weight allocations.

The fitted model is cached to avoid re-training on repeated runs the same day.
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
from config import ASSETS, HMM_STATES, RANDOM_SEED, MODEL_CACHE_DIR, TRADING_DAYS

REGIME_LABELS = {0: '🟢 Bull', 1: '🟡 Transition', 2: '🔴 Crisis'}
REGIME_COLORS = {0: '#2ecc71', 1: '#f39c12', 2: '#e74c3c'}


def _build_feature_matrix(garch_history: dict, data: pd.DataFrame) -> pd.DataFrame:
    """
    Construct the HMM input feature matrix using only information available
    at each quarter-end (no look-ahead).

    Features capture volatility level, macro stress, and momentum — the three
    dimensions that historically discriminate market regimes most reliably.
    """
    sp_ret = f'ret_SP500'
    macro_cols = ['vix', 'yield_curve', 'credit_spread', 'usd_index']
    records = []

    for qe, g in sorted(garch_history.items()):
        row = {'date': qe}

        # ── Vol layer features ────────────────────────────────────────────────
        row['garch_port_vol'] = g['port_vol']
        for asset in ASSETS:
            row[f'garch_vol_{asset}'] = g['vol_forecasts'][asset]
            row[f'rc_{asset}']        = g['risk_contributions'][asset]

        # ── 21-day realized vol of S&P (captures short-term stress) ──────────
        if qe in data.index:
            loc = data.index.get_loc(qe)
            sp_window = data[sp_ret].iloc[max(0, loc - 20): loc + 1]
            row['realized_vol_21d'] = float(sp_window.std() * np.sqrt(TRADING_DAYS))
        else:
            row['realized_vol_21d'] = g['port_vol']

        # ── Macro features (forward-filled, may be NaN if no FRED key) ───────
        for col in macro_cols:
            if col in data.columns and qe in data.index:
                row[col] = float(data.loc[qe, col]) if not pd.isna(data.loc[qe, col]) else np.nan
            else:
                row[col] = np.nan

        # ── S&P momentum at 1m / 3m / 6m (trend signal) ─────────────────────
        if qe in data.index:
            loc = data.index.get_loc(qe)
            for days, lbl in [(21, '1m'), (63, '3m'), (126, '6m')]:
                mom = data[sp_ret].iloc[max(0, loc - days): loc].sum() if loc >= days else 0.0
                row[f'momentum_{lbl}'] = float(mom)

        records.append(row)

    feat_df = pd.DataFrame(records).set_index('date')
    # Fill NaNs with column median so the HMM still trains when macro is absent
    feat_df = feat_df.fillna(feat_df.median(numeric_only=True)).fillna(0.0)
    return feat_df


def run_hmm(garch_history: dict, data: pd.DataFrame) -> dict:
    """
    Fit a 3-state Gaussian HMM and label each quarter-end with a regime.

    States are sorted ascending by mean portfolio vol so Bull=0, Transition=1,
    Crisis=2 — labels remain consistent across re-fits regardless of the HMM's
    internal state numbering.

    Returns a dict with:
        history       — {qe: {regime, regime_probs, regime_label, regime_color}}
        regime_series — pd.Series of integer regime per quarter-end
        probs_df      — pd.DataFrame of posterior probabilities (n_quarters × 3)
        feature_df    — the raw feature matrix (for diagnostics)
        model / scaler — fitted objects
    """
    cache_dir  = Path(MODEL_CACHE_DIR)
    cache_dir.mkdir(exist_ok=True)
    cache_file = cache_dir / f'hmm_{datetime.today().strftime("%Y%m%d")}.joblib'

    feat_df = _build_feature_matrix(garch_history, data)
    X_raw   = feat_df.values.astype(float)

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    # ── Fit or load ────────────────────────────────────────────────────────────
    if cache_file.exists():
        model, scaler = joblib.load(cache_file)
        print('Layer 2 — Loading HMM from cache …')
    else:
        print('Layer 2 — Fitting Gaussian HMM …')
        model = GaussianHMM(
            n_components=HMM_STATES,
            covariance_type='full',
            n_iter=300,
            random_state=RANDOM_SEED,
            tol=1e-5,
        )
        model.fit(X_scaled)
        joblib.dump((model, scaler), cache_file)

    raw_states  = model.predict(X_scaled)
    state_probs = model.predict_proba(X_scaled)

    # ── Sort states by mean portfolio-vol feature (index 0) ───────────────────
    port_vol_idx = feat_df.columns.get_loc('garch_port_vol')
    state_means  = np.array([
        X_raw[raw_states == s, port_vol_idx].mean()
        if (raw_states == s).any() else 0.0
        for s in range(HMM_STATES)
    ])
    order    = np.argsort(state_means)          # ascending vol → 0=Bull
    remap    = {old: new for new, old in enumerate(order)}
    states   = np.array([remap[s] for s in raw_states])
    probs    = state_probs[:, order]            # reorder columns to match

    regime_series = pd.Series(states, index=feat_df.index, name='regime')
    probs_df      = pd.DataFrame(
        probs, index=feat_df.index,
        columns=[f'p{i}' for i in range(HMM_STATES)],
    )

    history = {}
    for i, qe in enumerate(feat_df.index):
        r = int(states[i])
        p = probs[i].tolist()
        history[qe] = dict(
            regime       = r,
            regime_probs = p,
            regime_label = REGIME_LABELS[r],
            regime_color = REGIME_COLORS[r],
        )

    return dict(
        history       = history,
        regime_series = regime_series,
        probs_df      = probs_df,
        feature_df    = feat_df,
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

    latest = max(res['history'].keys())
    r      = res['history'][latest]
    print(f'\nLatest quarter-end : {latest.date()}')
    print(f'Regime             : {r["regime_label"]}')
    print(f'Probabilities      : {[f"{p:.0%}" for p in r["regime_probs"]]}')
    print(f'\nRegime counts:\n{res["regime_series"].value_counts().sort_index()}')
