"""
data/fetcher.py — Market data and macro data ingestion.

Fetches 10 years of daily OHLCV from yfinance and four macro time series
from FRED (VIX, yield curve spread, credit spread, USD index). Results are
cached locally as a Parquet file for 1 trading day to avoid redundant API
calls. If the FRED API key is absent the macro columns are left as NaN —
all downstream layers detect and handle this gracefully.
"""

import os
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings('ignore')

# Support both `python main.py` (root) and `python data/fetcher.py` (direct)
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (TICKERS, ASSETS, FRED_SERIES, FRED_API_KEY,
                    CACHE_PATH, LOOKBACK_YEARS)


def _cache_is_fresh(path: str) -> bool:
    """Return True if the cache file exists and was written within the last calendar day."""
    p = Path(path)
    if not p.exists():
        return False
    age = datetime.now() - datetime.fromtimestamp(p.stat().st_mtime)
    return age < timedelta(days=1)


def _fetch_fred_series(series_id: str, start: str, end: str) -> pd.Series:
    """
    Download a single FRED series.
    Returns an empty Series on any failure so callers always get a Series back.
    """
    try:
        from fredapi import Fred
        if not FRED_API_KEY:
            raise ValueError('FRED_API_KEY env var not set')
        fred = Fred(api_key=FRED_API_KEY)
        s = fred.get_series(series_id, observation_start=start, observation_end=end)
        s.name = series_id
        return s
    except Exception as exc:
        warnings.warn(
            f'FRED fetch failed for {series_id}: {exc}. '
            'Macro column will be NaN — set FRED_API_KEY to enable it.'
        )
        return pd.Series(dtype=float, name=series_id)


def fetch_data(force_refresh: bool = False) -> pd.DataFrame:
    """
    Return a single DataFrame aligned on equity trading days with columns:

        ret_{ASSET}    — daily log return for each asset
        price_{ASSET}  — adjusted closing price
        vix            — CBOE VIX index (fear gauge)
        yield_curve    — 10Y-2Y Treasury spread (recession indicator)
        credit_spread  — US HY credit spread (risk appetite proxy)
        usd_index      — USD broad index (global dollar demand)

    Macro series are forward-filled from their last observed value so there
    are no gaps on equity trading days. The DataFrame is cached locally for
    1 trading day; pass force_refresh=True to bypass the cache.
    """
    if not force_refresh and _cache_is_fresh(CACHE_PATH):
        return pd.read_parquet(CACHE_PATH)

    today      = datetime.today()
    end_str    = today.strftime('%Y-%m-%d')
    start_str  = (today - timedelta(days=int(LOOKBACK_YEARS * 365.25 + 60))).strftime('%Y-%m-%d')

    # ── Equity prices via yfinance ─────────────────────────────────────────────
    ticker_list = list(TICKERS.values())
    raw = yf.download(ticker_list, start=start_str, end=end_str,
                      auto_adjust=True, progress=False)['Close']
    raw.columns = list(TICKERS.keys())
    raw = raw.dropna()

    log_ret = np.log(raw / raw.shift(1)).dropna()
    prices  = raw.loc[log_ret.index]

    df = pd.DataFrame(index=log_ret.index)
    for asset in ASSETS:
        df[f'ret_{asset}']   = log_ret[asset]
        df[f'price_{asset}'] = prices[asset]

    # ── Macro data via FRED ────────────────────────────────────────────────────
    macro_ok = bool(FRED_API_KEY)
    if not macro_ok:
        warnings.warn(
            'FRED_API_KEY is not set — macro features will be NaN. '
            'Downstream layers will skip macro inputs or substitute medians.'
        )

    for col_name, series_id in FRED_SERIES.items():
        if macro_ok:
            s = _fetch_fred_series(series_id, start_str, end_str)
        else:
            s = pd.Series(dtype=float, name=series_id)
        # Reindex to equity trading calendar, forward-fill gaps.
        # Skip reindex on empty series (e.g. no FRED_API_KEY) — index dtypes
        # would be incompatible (int64 vs datetime64) and raise TypeError.
        df[col_name] = s.reindex(df.index, method='ffill') if not s.empty else np.nan

    # ── Cache ──────────────────────────────────────────────────────────────────
    Path(CACHE_PATH).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CACHE_PATH)
    print(f'Data fetched: {df.shape[0]} trading days, {df.shape[1]} columns.')
    return df


if __name__ == '__main__':
    df = fetch_data(force_refresh=True)
    print(df.tail(3).to_string())
    print(f'\nMissing:\n{df.isnull().sum()[df.isnull().sum() > 0]}')
