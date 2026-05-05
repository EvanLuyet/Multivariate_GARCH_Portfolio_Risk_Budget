"""
config.py — Single source of truth for all model parameters.
Edit this file to change universe, targets, or model hyper-parameters.
"""
import os

# ── Asset universe ─────────────────────────────────────────────────────────────
TICKERS = {
    'SP500':  '^GSPC',
    'NDX':    'QQQ',
    'EUROPE': 'EZU',
}
ASSETS = list(TICKERS.keys())

BASE_WEIGHTS  = {'SP500': 0.65, 'NDX': 0.20, 'EUROPE': 0.15}
WEIGHT_BOUNDS = {'SP500': (0.40, 0.80), 'NDX': (0.10, 0.40), 'EUROPE': (0.10, 0.30)}

# ── Vol targeting ──────────────────────────────────────────────────────────────
TARGET_VOL  = 0.12
VOL_LOW     = 0.10
VOL_HIGH    = 0.14

# ── Risk / cost ────────────────────────────────────────────────────────────────
RISK_FREE_RATE   = 0.02
TRANSACTION_COST = 0.0005   # 5 bps per ETF leg
LAMBDA_TC        = 0.001    # TC penalty weight inside BL optimizer

# ── Model windows ─────────────────────────────────────────────────────────────
GARCH_WINDOW   = 252
CORR_WINDOW    = 63
HMM_STATES     = 3
LOOKBACK_YEARS = 10
REBALANCE_FREQ = 'Q'
TRADING_DAYS   = 252

# ── Black-Litterman ────────────────────────────────────────────────────────────
DELTA = 2.5   # market risk-aversion coefficient
TAU   = 0.05  # uncertainty scaling of the prior

# ── XGBoost / LightGBM ────────────────────────────────────────────────────────
XGB_PARAMS = dict(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbosity=0,
)
LGB_QUANTILES   = [0.25, 0.75]
MIN_TRAIN_QTRS  = 24   # 6 years of quarter-ends before first ML forecast

# ── Paths ──────────────────────────────────────────────────────────────────────
CACHE_PATH      = 'data/cache.parquet'
MODEL_CACHE_DIR = '.model_cache'
STATE_FILE      = 'portfolio_state.json'

# ── Randomness ─────────────────────────────────────────────────────────────────
RANDOM_SEED = 42

# ── FRED macro series ──────────────────────────────────────────────────────────
FRED_SERIES = {
    'vix':           'VIXCLS',
    'yield_curve':   'T10Y2Y',
    'credit_spread': 'BAMLH0A0HYM2',
    'usd_index':     'DTWEXBGS',
}
FRED_API_KEY = os.getenv('FRED_API_KEY', '')
