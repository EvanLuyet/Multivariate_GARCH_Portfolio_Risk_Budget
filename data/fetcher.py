"""
data/fetcher.py — Market data ingestion: ETFs, regional benchmarks, FX, macro.

Fetches 10 years of daily data from yfinance:
  - ETF prices + returns (SP500/NDX/EUROPE)
  - Regional benchmark prices + returns (US/EU/Swiss) for per-market regime detection
  - FX rates (USDCHF, EURCHF, EURUSD) for CHF-adjusted portfolio analysis
  - Macro series from FRED (VIX, yield curve, credit spread, USD index)

All data is aligned on the equity trading calendar and cached as Parquet for 1 day.
"""

import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (TICKERS, ASSETS, MARKET_TICKERS, FX_TICKERS,
                    FRED_SERIES, FRED_API_KEY, CACHE_PATH, LOOKBACK_YEARS)


def _cache_is_fresh(path: str) -> bool:
    p = Path(path)
    if not p.exists():
        return False
    age = datetime.now() - datetime.fromtimestamp(p.stat().st_mtime)
    return age < timedelta(days=1)


def _fetch_fred_series(series_id: str, start: str, end: str) -> pd.Series:
    try:
        from fredapi import Fred
        if not FRED_API_KEY:
            raise ValueError('FRED_API_KEY not set')
        fred = Fred(api_key=FRED_API_KEY)
        s = fred.get_series(series_id, observation_start=start, observation_end=end)
        s.name = series_id
        return s
    except Exception as exc:
        warnings.warn(f'FRED fetch failed for {series_id}: {exc}')
        return pd.Series(dtype=float, name=series_id)


def _safe_download(tickers: list, start: str, end: str) -> pd.DataFrame:
    """Download closing prices; always returns a DataFrame (empty on failure)."""
    try:
        raw = yf.download(tickers, start=start, end=end,
                          auto_adjust=True, progress=False)
        # Handle both multi-ticker (MultiIndex cols) and single-ticker outputs
        if isinstance(raw.columns, pd.MultiIndex):
            raw = raw['Close']
        elif 'Close' in raw.columns:
            raw = raw[['Close']]
        if len(tickers) == 1 and raw.shape[1] == 1:
            raw.columns = tickers
        return raw.dropna(how='all')
    except Exception as exc:
        warnings.warn(f'Download failed for {tickers}: {exc}')
        return pd.DataFrame()


def fetch_data(force_refresh: bool = False) -> pd.DataFrame:
    """
    Return a single DataFrame aligned on equity trading days with columns:

        ret_{ASSET}, price_{ASSET}           — ETF log-returns and closing prices
        ret_mkt_{market}, price_mkt_{market}  — US / EU / Swiss benchmark series
        fx_{pair}                             — USDCHF, EURCHF, EURUSD spot rates
        vix, yield_curve, credit_spread, usd_index — FRED macro (NaN if no key)
    """
    if not force_refresh and _cache_is_fresh(CACHE_PATH):
        return pd.read_parquet(CACHE_PATH)

    today     = datetime.today()
    end_str   = today.strftime('%Y-%m-%d')
    start_str = (today - timedelta(days=int(LOOKBACK_YEARS * 365.25 + 90))).strftime('%Y-%m-%d')

    # ── ETF prices ─────────────────────────────────────────────────────────────
    etf_tickers = list(TICKERS.values())
    raw = _safe_download(etf_tickers, start_str, end_str)
    if raw.empty:
        raise RuntimeError('Failed to download ETF data — check internet connection.')
    raw.columns = list(TICKERS.keys())
    raw = raw.dropna()

    log_ret = np.log(raw / raw.shift(1)).dropna()
    prices  = raw.loc[log_ret.index]

    df = pd.DataFrame(index=log_ret.index)
    for asset in ASSETS:
        df[f'ret_{asset}']   = log_ret[asset]
        df[f'price_{asset}'] = prices[asset]

    # ── US benchmark reuses SP500 ──────────────────────────────────────────────
    df['ret_mkt_US']   = df['ret_SP500']
    df['price_mkt_US'] = df['price_SP500']

    # ── EU and Swiss benchmarks ────────────────────────────────────────────────
    for market, ticker in {k: v for k, v in MARKET_TICKERS.items() if k != 'US'}.items():
        try:
            mkt_raw = _safe_download([ticker], start_str, end_str)
            if mkt_raw.empty:
                raise ValueError('empty response')
            mkt_raw.columns = [market]
            mkt_ret = np.log(mkt_raw / mkt_raw.shift(1)).dropna()
            df[f'ret_mkt_{market}']   = mkt_ret[market].reindex(df.index, method='ffill').fillna(0.0)
            df[f'price_mkt_{market}'] = mkt_raw[market].reindex(df.index, method='ffill')
        except Exception as exc:
            warnings.warn(f'Benchmark {market} ({ticker}) unavailable: {exc}. Using SP500 proxy.')
            df[f'ret_mkt_{market}']   = df['ret_SP500']
            df[f'price_mkt_{market}'] = df['price_SP500']

    # ── FX rates ───────────────────────────────────────────────────────────────
    fx_tickers = list(FX_TICKERS.values())
    try:
        fx_raw = _safe_download(fx_tickers, start_str, end_str)
        if not fx_raw.empty:
            ticker_to_name = {v: k for k, v in FX_TICKERS.items()}
            fx_raw = fx_raw.rename(columns=ticker_to_name)
            for pair in FX_TICKERS:
                df[f'fx_{pair}'] = fx_raw[pair].reindex(df.index, method='ffill') \
                                   if pair in fx_raw.columns else np.nan
        else:
            for pair in FX_TICKERS:
                df[f'fx_{pair}'] = np.nan
    except Exception as exc:
        warnings.warn(f'FX data unavailable: {exc}')
        for pair in FX_TICKERS:
            df[f'fx_{pair}'] = np.nan

    # ── FRED macro ─────────────────────────────────────────────────────────────
    if not FRED_API_KEY:
        warnings.warn('FRED_API_KEY not set — macro features will be NaN.')
    for col_name, series_id in FRED_SERIES.items():
        s = _fetch_fred_series(series_id, start_str, end_str) if FRED_API_KEY \
            else pd.Series(dtype=float, name=series_id)
        df[col_name] = s.reindex(df.index, method='ffill') if not s.empty else np.nan

    # ── Cache ──────────────────────────────────────────────────────────────────
    Path(CACHE_PATH).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CACHE_PATH)
    print(f'Data fetched: {df.shape[0]} trading days, {df.shape[1]} columns.')
    return df


if __name__ == '__main__':
    df = fetch_data(force_refresh=True)
    print(df.tail(3).to_string())
    print(f'\nColumns: {list(df.columns)}')
    print(f'Missing:\n{df.isnull().sum()[df.isnull().sum() > 0]}')
