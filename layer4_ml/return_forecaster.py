"""
layer4_ml/return_forecaster.py — Walk-forward XGBoost return forecaster.

Produces forecasts for TWO horizons:
  current_quarter_forecast  — return from the most recent quarter-end to the next
                               (this is the 1-quarter-ahead model used by BL)
  next_quarter_forecast     — return 2 quarters ahead (lower confidence, for display)

Walk-forward protocol: at quarter-end Q_k, train only on Q_0 … Q_{k-1}.
Minimum training window: MIN_TRAIN_QTRS (24 quarters = 6 years).

LightGBM quantile models give 25th/75th percentile bounds alongside XGB point forecast.
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
    'garch_vol_SP500', 'garch_vol_NDX', 'garch_vol_EUROPE',
    'port_vol', 'rc_SP500', 'rc_NDX', 'rc_EUROPE',
    'p0', 'p1', 'p2',
    'mom_1m_SP500', 'mom_3m_SP500', 'mom_6m_SP500',
    'mom_1m_NDX',   'mom_3m_NDX',
    'mom_1m_EUROPE','mom_3m_EUROPE',
    'vix', 'yield_curve', 'credit_spread', 'usd_index',
    'quarter', 'month',
]


def _build_feature_row(qe, garch_info: dict, hmm_info: dict,
                       data: pd.DataFrame) -> dict:
    row = {}
    row['garch_vol_SP500']  = garch_info['vol_forecasts']['SP500']
    row['garch_vol_NDX']    = garch_info['vol_forecasts']['NDX']
    row['garch_vol_EUROPE'] = garch_info['vol_forecasts']['EUROPE']
    row['port_vol']         = garch_info['port_vol']
    row['rc_SP500']         = garch_info['risk_contributions']['SP500']
    row['rc_NDX']           = garch_info['risk_contributions']['NDX']
    row['rc_EUROPE']        = garch_info['risk_contributions']['EUROPE']

    probs = hmm_info['regime_probs']
    row['p0'], row['p1'], row['p2'] = probs[0], probs[1], probs[2]

    if qe in data.index:
        loc = data.index.get_loc(qe)
        for asset in ASSETS:
            col = f'ret_{asset}'
            for days, lbl in [(21, '1m'), (63, '3m'), (126, '6m')]:
                key = f'mom_{lbl}_{asset}'
                if key in FEATURE_COLS:
                    row[key] = float(data[col].iloc[max(0, loc - days): loc].sum()
                                     if loc >= days else 0.0)

    for macro_col in ['vix', 'yield_curve', 'credit_spread', 'usd_index']:
        val = np.nan
        if macro_col in data.columns and qe in data.index:
            val = data.loc[qe, macro_col]
        row[macro_col] = float(val) if not pd.isna(val) else 0.0

    ts = pd.Timestamp(qe)
    row['quarter'] = ts.quarter
    row['month']   = ts.month
    return row


def _compute_target(qe, next_qe, data: pd.DataFrame) -> dict:
    mask   = (data.index > qe) & (data.index <= next_qe)
    period = data[mask]
    return {a: float(period[f'ret_{a}'].sum()) if not period.empty else np.nan
            for a in ASSETS}


def _point_and_bounds(X_train, y_train, X_pred, valid_mask):
    """
    Fit XGB point forecast + LGB 25th/75th quantile bounds.

    The point and quantile models are independent, so the point can fall
    outside [low, high]. We enforce containment after fitting:
        low  = min(lgb_q25, point)
        high = max(lgb_q75, point)
    This guarantees the interval always brackets the point estimate.
    """
    xg = xgb.XGBRegressor(**XGB_PARAMS)
    xg.fit(X_train[valid_mask], y_train[valid_mask])
    point = float(xg.predict(X_pred)[0])

    low_val, high_val = point - 0.03, point + 0.03
    try:
        lg_lo = lgb.LGBMRegressor(objective='quantile', alpha=LGB_QUANTILES[0],
                                   n_estimators=200, random_state=RANDOM_SEED,
                                   verbose=-1)
        lg_hi = lgb.LGBMRegressor(objective='quantile', alpha=LGB_QUANTILES[1],
                                   n_estimators=200, random_state=RANDOM_SEED,
                                   verbose=-1)
        lg_lo.fit(X_train[valid_mask], y_train[valid_mask])
        lg_hi.fit(X_train[valid_mask], y_train[valid_mask])
        low_val  = float(lg_lo.predict(X_pred)[0])
        high_val = float(lg_hi.predict(X_pred)[0])
    except Exception:
        pass

    # Enforce: point must be within [low, high]
    low_val  = min(low_val,  point)
    high_val = max(high_val, point)

    return point, low_val, high_val


def run_forecaster(garch_history: dict, hmm_history: dict,
                   data: pd.DataFrame) -> dict:
    """
    Walk-forward forecaster returning 1Q and 2Q ahead predictions.

    Returns:
        history              — {qe: {return_forecasts: {asset: {point, low, high}}}}
        current_forecast     — 1Q-ahead forecasts at most recent quarter-end
        next_quarter_forecast— 2Q-ahead forecasts (lower accuracy, for display)
        feature_importance   — pd.Series of mean XGB importances
    """
    cache_dir  = Path(MODEL_CACHE_DIR)
    cache_dir.mkdir(exist_ok=True)
    cache_file = cache_dir / f'ml_{datetime.today().strftime("%Y%m%d")}.joblib'

    if cache_file.exists():
        print('Layer 4 — Loading ML forecasts from cache …')
        return joblib.load(cache_file)

    print('Layer 4 — Running walk-forward ML forecaster …')

    q_ends = sorted(set(garch_history.keys()) & set(hmm_history.keys()))

    rows, targets_1q, targets_2q = [], [], []
    for k, qe in enumerate(q_ends):
        feat = _build_feature_row(qe, garch_history[qe], hmm_history[qe], data)
        feat['_date'] = qe
        rows.append(feat)

        # 1Q target: Q_k → Q_{k+1}
        if k + 1 < len(q_ends):
            targets_1q.append(_compute_target(qe, q_ends[k + 1], data))
        else:
            targets_1q.append({a: np.nan for a in ASSETS})

        # 2Q target: Q_k → Q_{k+2}
        if k + 2 < len(q_ends):
            targets_2q.append(_compute_target(qe, q_ends[k + 2], data))
        else:
            targets_2q.append({a: np.nan for a in ASSETS})

    feat_df  = pd.DataFrame(rows).set_index('_date')
    tgt_1q   = pd.DataFrame(targets_1q, index=feat_df.index)
    tgt_2q   = pd.DataFrame(targets_2q, index=feat_df.index)

    for col in FEATURE_COLS:
        if col not in feat_df.columns:
            feat_df[col] = 0.0
    feat_df = feat_df[FEATURE_COLS].fillna(0.0)

    n = len(feat_df)
    history     = {}
    importances = {a: [] for a in ASSETS}

    for k in range(MIN_TRAIN_QTRS, n):
        qe      = feat_df.index[k]
        X_train = feat_df.values[:k]
        X_pred  = feat_df.values[k:k+1]

        forecasts_1q = {}
        for asset in ASSETS:
            y = tgt_1q[asset].values[:k]
            valid = ~np.isnan(y)
            if valid.sum() < MIN_TRAIN_QTRS // 2:
                forecasts_1q[asset] = dict(point=0.0, low=-0.05, high=0.05)
                continue
            pt, lo, hi = _point_and_bounds(X_train, y, X_pred, valid)
            importances[asset].append(
                xgb.XGBRegressor(**XGB_PARAMS).fit(X_train[valid], y[valid]).feature_importances_
            )
            forecasts_1q[asset] = dict(point=pt, low=lo, high=hi)

        history[qe] = dict(return_forecasts=forecasts_1q)

    # ── Current (full-sample) 1Q and 2Q forecasts ─────────────────────────────
    X_all = feat_df.values
    current_1q, current_2q = {}, {}

    for asset in ASSETS:
        y1 = tgt_1q[asset].values
        y2 = tgt_2q[asset].values
        v1, v2 = ~np.isnan(y1), ~np.isnan(y2)

        if v1.sum() >= MIN_TRAIN_QTRS // 2:
            pt, lo, hi = _point_and_bounds(X_all, y1, X_all[-1:], v1)
            current_1q[asset] = dict(point=pt, low=lo, high=hi)
        else:
            current_1q[asset] = dict(point=0.0, low=-0.05, high=0.05)

        if v2.sum() >= MIN_TRAIN_QTRS // 2:
            pt2, lo2, hi2 = _point_and_bounds(X_all, y2, X_all[-1:], v2)
            current_2q[asset] = dict(point=pt2, low=lo2, high=hi2)
        else:
            # Fallback: attenuate 1Q forecast toward zero (lower conviction at 2Q)
            pt1 = current_1q[asset]['point']
            current_2q[asset] = dict(point=pt1 * 0.7, low=pt1 * 0.7 - 0.04,
                                     high=pt1 * 0.7 + 0.04)

    current_qe = feat_df.index[-1]
    history[current_qe] = dict(return_forecasts=current_1q)

    fi_df = pd.DataFrame(
        {a: pd.Series(np.mean(importances[a], axis=0) if importances[a]
                      else np.zeros(len(FEATURE_COLS)), index=FEATURE_COLS)
         for a in ASSETS}
    ).mean(axis=1).sort_values(ascending=False)

    result = dict(
        history               = history,
        current_forecast      = current_1q,
        next_quarter_forecast = current_2q,
        feature_importance    = fi_df,
        feat_df               = feat_df,
    )
    joblib.dump(result, cache_file)
    return result


if __name__ == '__main__':
    from data.fetcher import fetch_data
    from layer1_garch.garch_model import run_garch
    from layer2_hmm.regime_model import run_hmm
    from config import BASE_WEIGHTS

    df  = fetch_data()
    g1  = run_garch(df, BASE_WEIGHTS)
    hmm = run_hmm(g1, df)
    res = run_forecaster(g1, hmm['history'], df)

    print('\nCurrent quarter forecast (Q+1):')
    for a, fc in res['current_forecast'].items():
        print(f"  {a}: {fc['point']:+.2%}  [{fc['low']:+.2%} / {fc['high']:+.2%}]")

    print('\nNext quarter forecast (Q+2):')
    for a, fc in res['next_quarter_forecast'].items():
        print(f"  {a}: {fc['point']:+.2%}  [{fc['low']:+.2%} / {fc['high']:+.2%}]")
