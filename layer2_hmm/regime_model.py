"""
layer2_hmm/regime_model.py — Hidden Markov Model regime detection.

Two functions:
  run_hmm()            — Full 12-feature HMM on US daily data (~2 500 rows).
                         Output drives the Black-Litterman optimizer.
  run_market_regimes() — Lightweight 4-feature HMM run independently for each of
                         US / EU / Swiss markets. Output drives the terminal report.

State ordering is always: 0 = Bull (low vol), 1 = Transition, 2 = Crisis.
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
                    RANDOM_SEED, MODEL_CACHE_DIR, TRADING_DAYS, MARKET_TICKERS)

REGIME_LABELS = {0: '🟢 Bull', 1: '🟡 Transition', 2: '🔴 Crisis'}
REGIME_COLORS = {0: '#2ecc71', 1: '#f39c12', 2: '#e74c3c'}


# ── Feature builders ──────────────────────────────────────────────────────────

def _build_daily_features(daily_garch_vols: pd.DataFrame,
                           data: pd.DataFrame) -> pd.DataFrame:
    """
    12-feature matrix for the main US HMM used by the BL optimizer.

    Core (no FRED key needed):
      realized_vol_5d/21d/63d, vol_accel, drawdown_63d, abs_ret, momentum_21d/63d
    Optional macro (FRED key):
      vix, yield_curve, credit_spread, usd_index
    """
    sp_ret = 'ret_SP500'
    r      = data[sp_ret].reindex(daily_garch_vols.index)
    feat   = pd.DataFrame(index=daily_garch_vols.index)

    for window, label in [(5, '5d'), (21, '21d'), (63, '63d')]:
        feat[f'realized_vol_{label}'] = r.rolling(window).std() * np.sqrt(TRADING_DAYS)

    feat['vol_accel'] = feat['realized_vol_5d'] / (feat['realized_vol_63d'] + 1e-8)

    prices   = data['price_SP500'].reindex(daily_garch_vols.index)
    roll_max = prices.rolling(63, min_periods=1).max()
    feat['drawdown_63d'] = (prices - roll_max) / (roll_max + 1e-8)

    feat['abs_ret']      = r.abs()
    feat['momentum_21d'] = r.rolling(21).sum()
    feat['momentum_63d'] = r.rolling(63).sum()

    for col in ['vix', 'yield_curve', 'credit_spread', 'usd_index']:
        feat[col] = data[col].reindex(feat.index, method='ffill') \
                   if col in data.columns else 0.0

    feat = feat.ffill().fillna(feat.median(numeric_only=True)).fillna(0.0)
    return feat.dropna()


def _build_market_features(ret: pd.Series, price: pd.Series,
                            us_ret: pd.Series | None = None) -> pd.DataFrame:
    """
    8-feature matrix for per-market HMMs.

    Absolute features capture the market's own vol/trend regime.
    Relative features (vs US) are the key differentiators: even during
    global risk-off, EU and Swiss can diverge meaningfully from the US.
    For the US market itself, rel_* features are zero by construction,
    which still produces a distinct feature distribution vs EU/Swiss.
    """
    feat = pd.DataFrame(index=ret.index)

    # ── Absolute ──────────────────────────────────────────────────────────────
    feat['vol_21d']      = ret.rolling(21).std() * np.sqrt(TRADING_DAYS)
    feat['vol_63d']      = ret.rolling(63).std() * np.sqrt(TRADING_DAYS)
    feat['vol_accel']    = feat['vol_21d'] / (feat['vol_63d'] + 1e-8)
    roll_max             = price.rolling(63, min_periods=1).max()
    feat['drawdown_63d'] = (price - roll_max) / (roll_max + 1e-8)
    feat['momentum_21d'] = ret.rolling(21).sum()
    feat['momentum_63d'] = ret.rolling(63).sum()

    # ── Relative vs US (zero for US itself, non-zero for EU/Swiss) ───────────
    if us_ret is not None:
        us_vol = us_ret.rolling(21).std() * np.sqrt(TRADING_DAYS)
        feat['rel_vol']     = feat['vol_21d'] / (us_vol.reindex(feat.index) + 1e-8)
        feat['rel_mom_21d'] = (ret.rolling(21).sum()
                               - us_ret.reindex(ret.index).rolling(21).sum())
    else:
        feat['rel_vol']     = 1.0
        feat['rel_mom_21d'] = 0.0

    return feat.ffill().fillna(0.0).dropna()


# ── HMM fitting helper ────────────────────────────────────────────────────────

def _fit_hmm(X_scaled: np.ndarray, n_seeds: int = 3) -> GaussianHMM:
    """Multi-seed GaussianHMM fit; returns best model by log-likelihood."""
    best_model, best_score = None, -np.inf
    for i in range(n_seeds):
        m = GaussianHMM(
            n_components=HMM_STATES,
            covariance_type='diag',
            n_iter=300,
            random_state=RANDOM_SEED + i,
            tol=1e-4,
        )
        m.fit(X_scaled)
        s = m.score(X_scaled)
        if s > best_score:
            best_score = s
            best_model = m
    return best_model


def _sort_and_floor(model: GaussianHMM, X_raw: np.ndarray,
                    X_scaled: np.ndarray, vol_col: int):
    """
    Predict states + probs, sort ascending by mean vol (Bull=0, Crisis=2),
    then apply HMM_PROB_FLOOR and renormalise.
    """
    raw_states  = model.predict(X_scaled)
    state_probs = model.predict_proba(X_scaled)

    state_means = np.array([
        X_raw[raw_states == s, vol_col].mean()
        if (raw_states == s).any() else 0.0
        for s in range(HMM_STATES)
    ])
    order  = np.argsort(state_means)
    remap  = {old: new for new, old in enumerate(order)}
    states = np.array([remap[s] for s in raw_states])
    probs  = state_probs[:, order]

    probs = np.clip(probs, HMM_PROB_FLOOR, 1.0)
    probs = probs / probs.sum(axis=1, keepdims=True)
    return states, probs


# ── Main HMM (used by BL optimizer) ──────────────────────────────────────────

def run_hmm(garch_history: dict, data: pd.DataFrame) -> dict:
    """
    Full 12-feature Gaussian HMM on US daily data.

    Returns:
        history        — {qe: {regime, regime_probs, regime_label, regime_color}}
        regime_series  — pd.Series of integer regime, DAILY index
        probs_df       — pd.DataFrame of posteriors (n×3), DAILY index
        feature_df     — daily feature matrix (diagnostics)
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
    n_obs    = len(feat_df)
    n_feat   = X_scaled.shape[1]

    print(f'Layer 2 — HMM training on {n_obs} daily observations '
          f'({feat_df.index[0].date()} → {feat_df.index[-1].date()}) …')

    cache_valid = False
    if cache_file.exists():
        try:
            model, scaler = joblib.load(cache_file)
            if model.means_.shape[1] == n_feat:
                print('Layer 2 — Loading HMM from cache …')
                cache_valid = True
            else:
                print(f'Layer 2 — Cache feature mismatch '
                      f'({model.means_.shape[1]} vs {n_feat}), re-fitting …')
                cache_file.unlink()
        except Exception:
            cache_file.unlink()

    if not cache_valid:
        model = _fit_hmm(X_scaled)
        joblib.dump((model, scaler), cache_file)

    vol_col = list(feat_df.columns).index('realized_vol_21d')
    states, probs = _sort_and_floor(model, X_raw, X_scaled, vol_col)

    regime_series = pd.Series(states, index=feat_df.index, name='regime')
    probs_df      = pd.DataFrame(probs, index=feat_df.index,
                                 columns=[f'p{i}' for i in range(HMM_STATES)])

    q_ends  = sorted(garch_history.keys())
    history = {}
    for qe in q_ends:
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
    print('Layer 2 — Regime distribution: '
          + '  '.join(f'{REGIME_LABELS[i]}: {counts.get(i, 0)} days '
                      f'({counts.get(i, 0)/n_obs:.0%})'
                      for i in range(HMM_STATES)))

    return dict(
        history       = history,
        regime_series = regime_series,
        probs_df      = probs_df,
        feature_df    = feat_df,
        model         = model,
        scaler        = scaler,
    )


# ── Per-market lightweight HMMs (US / EU / Swiss) ────────────────────────────

def run_market_regimes(data: pd.DataFrame) -> dict:
    """
    Run a lightweight 8-feature HMM independently for each regional market.
    Relative-to-US features ensure EU and Swiss produce distinct outputs even
    when global correlations are high.

    Returns a dict keyed by market name ('US', 'EU', 'Swiss'), each with:
        regime, regime_label, regime_probs, trend, regime_series, using_proxy
    """
    cache_dir  = Path(MODEL_CACHE_DIR)
    cache_dir.mkdir(exist_ok=True)
    cache_file = cache_dir / f'mkt_regimes_{datetime.today().strftime("%Y%m%d")}.joblib'

    if cache_file.exists():
        print('Layer 2 — Loading market regimes from cache …')
        return joblib.load(cache_file)

    print('Layer 2 — Fitting per-market regime HMMs (US / EU / Swiss) …')

    us_ret = data['ret_SP500']

    results = {}
    for market in MARKET_TICKERS:
        ret_col   = f'ret_mkt_{market}'
        price_col = f'price_mkt_{market}'

        ret   = data[ret_col]   if ret_col   in data.columns else us_ret
        price = data[price_col] if price_col in data.columns else data['price_SP500']

        # Detect if EU/Swiss silently fell back to SP500 data (identical series)
        using_proxy = False
        if market != 'US' and ret.equals(us_ret):
            using_proxy = True
            print(f'  ⚠  {market}: benchmark download failed — regime will mirror US. '
                  f'Check that ret_mkt_{market} is populated in the data cache.')

        # Pass us_ret so relative features are computed; US gets rel_* = 0/1
        mkt_us_ret = us_ret if market != 'US' else None
        feat = _build_market_features(ret, price, us_ret=mkt_us_ret)
        if len(feat) < 100:
            results[market] = _fallback_regime(market)
            continue

        X_raw    = feat.values.astype(float)
        scaler   = StandardScaler()
        X_scaled = scaler.fit_transform(X_raw)

        model = _fit_hmm(X_scaled, n_seeds=5)   # more seeds for stability

        vol_col_idx = list(feat.columns).index('vol_21d')
        states, probs = _sort_and_floor(model, X_raw, X_scaled, vol_col_idx)

        regime_series = pd.Series(states, index=feat.index, name='regime')
        probs_series  = pd.DataFrame(probs, index=feat.index,
                                     columns=['p0', 'p1', 'p2'])

        current_regime = int(regime_series.iloc[-1])
        current_probs  = probs_series.iloc[-1].values.tolist()

        # Trend: compare Bull probability vs 63 trading days ago
        trend = 'stable'
        if len(probs_series) > 63:
            p_bull_now  = probs_series['p0'].iloc[-1]
            p_bull_past = probs_series['p0'].iloc[-63]
            if p_bull_now - p_bull_past > 0.10:
                trend = 'improving'
            elif p_bull_past - p_bull_now > 0.10:
                trend = 'deteriorating'

        results[market] = dict(
            regime        = current_regime,
            regime_label  = REGIME_LABELS[current_regime],
            regime_probs  = current_probs,
            trend         = trend,
            regime_series = regime_series,
            using_proxy   = using_proxy,
        )
        proxy_note = ' [SP500 proxy]' if using_proxy else ''
        print(f'  {market:6s}: {REGIME_LABELS[current_regime]}{proxy_note}  '
              f'(Bull {current_probs[0]:.0%} / Trans {current_probs[1]:.0%} / '
              f'Crisis {current_probs[2]:.0%})  trend: {trend}')

    joblib.dump(results, cache_file)
    return results


def _fallback_regime(market: str) -> dict:
    """Return a neutral placeholder when data is insufficient."""
    return dict(
        regime        = 1,
        regime_label  = REGIME_LABELS[1],
        regime_probs  = [0.33, 0.34, 0.33],
        trend         = 'stable',
        regime_series = pd.Series(dtype=int),
    )


if __name__ == '__main__':
    from data.fetcher import fetch_data
    from layer1_garch.garch_model import run_garch
    from config import BASE_WEIGHTS

    df  = fetch_data()
    g1  = run_garch(df, BASE_WEIGHTS)
    res = run_hmm(g1, df)
    mkt = run_market_regimes(df)

    print('\n=== Per-market regimes ===')
    for market, info in mkt.items():
        print(f'  {market}: {info["regime_label"]}  probs={[f"{p:.0%}" for p in info["regime_probs"]]}')
