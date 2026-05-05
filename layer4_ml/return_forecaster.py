"""
layer4_ml/return_forecaster.py — Walk-forward XGBoost return forecaster.

For each asset we train an XGBoost regressor to predict the next quarter's
log return using features derived from GARCH vol, HMM regime probabilities,
price momentum and macro conditions. Walk-forward validation ensures no
look-ahead: at quarter-end Q_k we train only on history before Q_k.

Uncertainty bounds (25th / 75th percentile) are produced by separate LightGBM
quantile regression models, giving a calibrated forecast interval alongside
the XGBoost point estimate.

Minimum training window: MIN_TRAIN_QTRS quarters (≈6 years). Quarter-ends
with insufficient history receive None forecasts; Layer 3 falls back to
equilibrium returns in that case.

Results are cached daily with joblib.
"""

import sys
import warnings
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime

import xgboost as xgb
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (ASSETS, TRADING_DAYS, MODEL_CACHE_DIR,
                    XGB_PARAMS, LGB_QUANTILES, MIN_TRAIN_QTRS, RANDOM_SEED)

FEATURE_COLS = [
    # Layer 1 — volatility and risk structure
    'garch_vol_SP500', 'garch_vol_NDX', 'garch_vol_EUROPE',
    'port_vol', 'rc_SP500', 'rc_NDX', 'rc_EUROPE',
    # Layer 2 — regime probabilities
    'p0', 'p1', 'p2',
    # Price momentum (sum of log-returns over window)
    'mom_1m_SP500', 'mom_3m_SP500', 'mom_6m_SP500',
    'mom_1m_NDX',   'mom_3m_NDX',
    'mom_1m_EUROPE','mom_3m_EUROPE',
    # Macro (may be NaN → filled with 0)
    'vix', 'yield_curve', 'credit_spread', 'usd_index',
    # Seasonality
    'quarter', 'month',
]


def _build_feature_row(qe, garch_info: dict, hmm_info: dict,
                       data: pd.DataFrame) -> dict:
    """
    Assemble the feature vector for a single quarter-end.
    Only data available at qe (no look-ahead).
    """
    row = {}

    # GARCH features
    row['garch_vol_SP500']  = garch_info['vol_forecasts']['SP500']
    row['garch_vol_NDX']    = garch_info['vol_forecasts']['NDX']
    row['garch_vol_EUROPE'] = garch_info['vol_forecasts']['EUROPE']
    row['port_vol']         = garch_info['port_vol']
    row['rc_SP500']         = garch_info['risk_contributions']['SP500']
    row['rc_NDX']           = garch_info['risk_contributions']['NDX']
    row['rc_EUROPE']        = garch_info['risk_contributions']['EUROPE']

    # HMM regime probabilities
    probs = hmm_info['regime_probs']
    row['p0'], row['p1'], row['p2'] = probs[0], probs[1], probs[2]

    # Momentum from daily returns
    if qe in data.index:
        loc = data.index.get_loc(qe)
        for asset in ASSETS:
            col = f'ret_{asset}'
            for days, lbl in [(21, '1m'), (63, '3m'), (126, '6m')]:
                key = f'mom_{lbl}_{asset}'
                if key in FEATURE_COLS:
                    row[key] = float(data[col].iloc[max(0, loc-days): loc].sum()
                                     if loc >= days else 0.0)

    # Macro at qe
    for macro_col in ['vix', 'yield_curve', 'credit_spread', 'usd_index']:
        val = np.nan
        if macro_col in data.columns and qe in data.index:
            val = data.loc[qe, macro_col]
        row[macro_col] = float(val) if not pd.isna(val) else 0.0

    # Seasonality
    ts = pd.Timestamp(qe)
    row['quarter'] = ts.quarter
    row['month']   = ts.month

    return row


def _compute_target(qe, next_qe, data: pd.DataFrame) -> dict:
    """
    Compute realized log return for each asset between qe and next_qe.
    This is the walk-forward target: known only after next_qe has passed.
    """
    mask   = (data.index > qe) & (data.index <= next_qe)
    period = data[mask]
    targets = {}
    for asset in ASSETS:
        col = f'ret_{asset}'
        targets[asset] = float(period[col].sum()) if not period.empty else np.nan
    return targets


def run_forecaster(garch_history: dict, hmm_history: dict,
                   data: pd.DataFrame) -> dict:
    """
    Walk-forward XGBoost + LightGBM quantile forecaster.

    Training expands by one quarter at each step. At quarter-end Q_k we:
      1. Build feature vector X_k (GARCH + HMM + momentum + macro at Q_k).
      2. Set target y_k = realized return from Q_k to Q_{k+1} (known in hindsight).
      3. Train XGB on X[0:k], y[0:k] and predict y_k (out-of-sample point estimate).
      4. Train LGB quantile models for 25th / 75th percentile bounds.

    The final model (trained on all history) produces the current forecast for
    the upcoming quarter — used as views in the Black-Litterman optimizer.

    Returns:
        history          — {qe: {return_forecasts: {asset: {point, low, high}}}}
        current_forecast — forecasts for the most recent quarter-end
        feature_importance — pd.DataFrame of mean XGBoost importances per asset
    """
    cache_dir  = Path(MODEL_CACHE_DIR)
    cache_dir.mkdir(exist_ok=True)
    cache_file = cache_dir / f'ml_{datetime.today().strftime("%Y%m%d")}.joblib'

    if cache_file.exists():
        print('Layer 4 — Loading ML forecasts from cache …')
        return joblib.load(cache_file)

    print('Layer 4 — Running walk-forward ML forecaster …')

    q_ends = sorted(set(garch_history.keys()) & set(hmm_history.keys()))

    # ── Build full feature + target arrays ────────────────────────────────────
    rows, targets_list = [], []
    for k, qe in enumerate(q_ends):
        if qe not in garch_history or qe not in hmm_history:
            continue
        feat = _build_feature_row(qe, garch_history[qe], hmm_history[qe], data)
        feat['_date'] = qe
        rows.append(feat)

        if k + 1 < len(q_ends):
            tgt = _compute_target(qe, q_ends[k + 1], data)
        else:
            tgt = {a: np.nan for a in ASSETS}
        targets_list.append(tgt)

    feat_df  = pd.DataFrame(rows).set_index('_date')
    tgt_df   = pd.DataFrame(targets_list, index=feat_df.index)

    # Align columns to FEATURE_COLS (fill any missing with 0)
    for col in FEATURE_COLS:
        if col not in feat_df.columns:
            feat_df[col] = 0.0
    feat_df = feat_df[FEATURE_COLS].fillna(0.0)

    n = len(feat_df)
    history       = {}
    importances   = {asset: [] for asset in ASSETS}

    # ── Walk-forward loop ─────────────────────────────────────────────────────
    for k in range(MIN_TRAIN_QTRS, n):
        qe          = feat_df.index[k]
        X_train     = feat_df.values[:k]
        X_pred      = feat_df.values[k:k+1]

        forecasts = {}
        for asset in ASSETS:
            y_train = tgt_df[asset].values[:k]
            valid   = ~np.isnan(y_train)
            if valid.sum() < MIN_TRAIN_QTRS // 2:
                forecasts[asset] = dict(point=0.0, low=-0.05, high=0.05)
                continue

            # ── XGBoost point forecast ──────────────────────────────────────
            xg = xgb.XGBRegressor(**XGB_PARAMS)
            xg.fit(X_train[valid], y_train[valid])
            point = float(xg.predict(X_pred)[0])
            importances[asset].append(xg.feature_importances_)

            # ── LightGBM quantile bounds ────────────────────────────────────
            low_val, high_val = point - 0.03, point + 0.03  # default fallback
            try:
                lg_lo = lgb.LGBMRegressor(
                    objective='quantile', alpha=LGB_QUANTILES[0],
                    n_estimators=200, random_state=RANDOM_SEED, verbose=-1,
                )
                lg_hi = lgb.LGBMRegressor(
                    objective='quantile', alpha=LGB_QUANTILES[1],
                    n_estimators=200, random_state=RANDOM_SEED, verbose=-1,
                )
                lg_lo.fit(X_train[valid], y_train[valid])
                lg_hi.fit(X_train[valid], y_train[valid])
                low_val  = float(lg_lo.predict(X_pred)[0])
                high_val = float(lg_hi.predict(X_pred)[0])
            except Exception:
                pass

            forecasts[asset] = dict(point=point, low=low_val, high=high_val)

        history[qe] = dict(return_forecasts=forecasts)

    # ── Current forecast (train on full history, predict next quarter) ────────
    current_qe = feat_df.index[-1]
    X_all      = feat_df.values
    current_fc = {}
    fi_records = {}

    for asset in ASSETS:
        y_all = tgt_df[asset].values
        valid = ~np.isnan(y_all)
        if valid.sum() < MIN_TRAIN_QTRS // 2:
            current_fc[asset] = dict(point=0.0, low=-0.05, high=0.05)
            continue

        xg = xgb.XGBRegressor(**XGB_PARAMS)
        xg.fit(X_all[valid], y_all[valid])
        point = float(xg.predict(X_all[-1:] )[0])
        fi_records[asset] = xg.feature_importances_

        low_val, high_val = point - 0.03, point + 0.03
        try:
            lg_lo = lgb.LGBMRegressor(
                objective='quantile', alpha=LGB_QUANTILES[0],
                n_estimators=200, random_state=RANDOM_SEED, verbose=-1,
            )
            lg_hi = lgb.LGBMRegressor(
                objective='quantile', alpha=LGB_QUANTILES[1],
                n_estimators=200, random_state=RANDOM_SEED, verbose=-1,
            )
            lg_lo.fit(X_all[valid], y_all[valid])
            lg_hi.fit(X_all[valid], y_all[valid])
            low_val  = float(lg_lo.predict(X_all[-1:])[0])
            high_val = float(lg_hi.predict(X_all[-1:])[0])
        except Exception:
            pass

        current_fc[asset] = dict(point=point, low=low_val, high=high_val)

    history[current_qe] = dict(return_forecasts=current_fc)

    # ── Feature importance (mean across walk-forward steps) ──────────────────
    fi_df = pd.DataFrame(
        {asset: pd.Series(np.mean(importances[asset], axis=0)
                          if importances[asset] else np.zeros(len(FEATURE_COLS)),
                          index=FEATURE_COLS)
         for asset in ASSETS}
    ).mean(axis=1).sort_values(ascending=False)

    result = dict(
        history              = history,
        current_forecast     = current_fc,
        feature_importance   = fi_df,
        feat_df              = feat_df,
    )
    joblib.dump(result, cache_file)
    return result


if __name__ == '__main__':
    from data.fetcher import fetch_data
    from layer1_garch.garch_model import run_garch
    from layer2_hmm.regime_model import run_hmm
    from config import BASE_WEIGHTS

    df   = fetch_data()
    g1   = run_garch(df, BASE_WEIGHTS)
    hmm  = run_hmm(g1, df)
    res  = run_forecaster(g1, hmm['history'], df)

    print('\nCurrent quarter forecast:')
    for asset, fc in res['current_forecast'].items():
        print(f"  {asset}: {fc['point']:+.2%}  [{fc['low']:+.2%} / {fc['high']:+.2%}]")
    print('\nTop 5 features:')
    print(res['feature_importance'].head(5).to_string())
