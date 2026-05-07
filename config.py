"""
config.py — Single source of truth for all model parameters.
Edit this file to change universe, targets, or model hyper-parameters.
"""

import hashlib
import json
import os

# ── Asset universe ─────────────────────────────────────────────────────────────
TICKERS = {
    "SP500": "^GSPC",
    "NDX": "QQQ",
    "EUROPE": "EZU",
}
ASSETS = list(TICKERS.keys())

BASE_WEIGHTS = {"SP500": 0.65, "NDX": 0.20, "EUROPE": 0.15}
WEIGHT_BOUNDS = {"SP500": (0.40, 0.80), "NDX": (0.10, 0.40), "EUROPE": (0.10, 0.30)}

# ── Regional market benchmarks (for per-market regime detection) ───────────────
MARKET_TICKERS = {
    "US": "^GSPC",
    "EU": "^STOXX50E",
    "Swiss": "^SSMI",
}

# ── FX pairs — expressed as CHF per foreign unit ──────────────────────────────
# USDCHF=X : CHF per 1 USD  (high = CHF weak vs USD, hurts CHF investor in USD assets)
# EURCHF=X : CHF per 1 EUR  (high = CHF weak vs EUR)
# EURUSD=X : USD per 1 EUR
FX_TICKERS = {
    "USDCHF": "USDCHF=X",
    "EURCHF": "EURCHF=X",
    "EURUSD": "EURUSD=X",
}

# ── Stock screener universe (US + European large-caps) ────────────────────────
STOCK_UNIVERSE = [
    # US mega-cap
    "AAPL",
    "MSFT",
    "NVDA",
    "GOOGL",
    "AMZN",
    "META",
    "BRK-B",
    "JPM",
    "JNJ",
    "XOM",
    "UNH",
    "V",
    "MA",
    "AVGO",
    "PG",
    "HD",
    "COST",
    "LLY",
    # European / Swiss (US-listed or ADRs)
    "ASML",
    "NVO",
    "SAP",
    "AZN",
    "SHEL",
    "TTE",
    "UBS",
    "ABB",
]

# ── IBKR Tiered commission model (USD) ────────────────────────────────────────
IBKR_MIN_COMMISSION = 1.00  # minimum per order
IBKR_RATE_PER_SHARE = 0.005  # per share
IBKR_FX_COST = 0.00002  # 0.002% of converted FX amount
IBKR_SPREAD_BPS = 0.5  # estimated bid-ask spread for liquid ETFs

# ── Rebalancing band — only recommend if weight has drifted beyond this ────────
REBALANCE_BAND = 0.05  # 5 percentage points

# ── Vol targeting ──────────────────────────────────────────────────────────────
TARGET_VOL = 0.12
VOL_LOW = 0.10
VOL_HIGH = 0.14

# ── Risk / cost ────────────────────────────────────────────────────────────────
RISK_FREE_RATE = 0.02
TRANSACTION_COST = 0.0005  # fallback 5 bps per leg when share price unknown
LAMBDA_TC = 0.001  # turnover penalty inside BL optimizer

# ── Model windows ─────────────────────────────────────────────────────────────
GARCH_WINDOW = 252
CORR_WINDOW = 63
HMM_STATES = 3
HMM_PROB_FLOOR = 0.05
LOOKBACK_YEARS = 10
REBALANCE_FREQ = "Q"
TRADING_DAYS = 252

# ── Black-Litterman ────────────────────────────────────────────────────────────
DELTA = 2.5
TAU = 0.05

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
LGB_QUANTILES = [0.25, 0.75]
MIN_TRAIN_QTRS = 24

# ── Paths ──────────────────────────────────────────────────────────────────────
CACHE_PATH = "data/cache.parquet"
MODEL_CACHE_DIR = ".model_cache"
STATE_FILE = "portfolio_state.json"

# ── Randomness ─────────────────────────────────────────────────────────────────
RANDOM_SEED = 42

# ── FRED macro series ──────────────────────────────────────────────────────────
FRED_SERIES = {
    "vix": "VIXCLS",
    "yield_curve": "T10Y2Y",
    "credit_spread": "BAMLH0A0HYM2",
    "usd_index": "DTWEXBGS",
}
FRED_API_KEY = os.getenv("FRED_API_KEY", "")


def _compute_config_hash() -> str:
    """
    Short hash of the parameters that affect cached model outputs.
    If any of these change, old cache files are bypassed automatically —
    no --refresh needed.
    """
    key_params = {
        "ASSETS": ASSETS,
        "BASE_WEIGHTS": BASE_WEIGHTS,
        "GARCH_WINDOW": GARCH_WINDOW,
        "CORR_WINDOW": CORR_WINDOW,
        "HMM_STATES": HMM_STATES,
        "HMM_PROB_FLOOR": HMM_PROB_FLOOR,
        "MIN_TRAIN_QTRS": MIN_TRAIN_QTRS,
        "XGB_PARAMS": XGB_PARAMS,
        "DELTA": DELTA,
        "TAU": TAU,
        "LOOKBACK_YEARS": LOOKBACK_YEARS,
        "RANDOM_SEED": RANDOM_SEED,
        "STOCK_UNIVERSE": STOCK_UNIVERSE,
    }
    blob = json.dumps(key_params, sort_keys=True, default=str).encode()
    return hashlib.sha1(blob).hexdigest()[:8]


CONFIG_HASH = _compute_config_hash()
