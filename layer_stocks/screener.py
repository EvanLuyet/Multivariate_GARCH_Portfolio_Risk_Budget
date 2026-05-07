"""
layer_stocks/screener.py — Quantitative top-5 stock screener.

Scores a universe of large-cap stocks (US + European ADRs) using a composite
factor model and filters by current market regime:

  Bull regime   → favour momentum + growth (high recent returns, low vol)
  Transition    → favour quality (low vol, positive momentum, diversified sector)
  Crisis regime → favour defensives (low drawdown, high 1Y return vs market)

Scoring factors (all normalised to z-scores before weighting):
  mom_1m   (20%)  — 1-month log return
  mom_3m   (30%)  — 3-month log return  ← primary signal
  mom_6m   (20%)  — 6-month log return
  sharpe   (20%)  — 6-month Sharpe ratio (excess return / rolling vol)
  low_dd   (10%)  — negative of max drawdown over 6 months (low drawdown = good)

Factor weights are tilted by regime:
  Bull     → overweight momentum
  Crisis   → overweight low_dd (capital preservation)
  Transition → balanced

Data is cached for 1 day alongside the main data cache.
"""

import sys
import time
import warnings
import numpy as np
import pandas as pd
import joblib
import yfinance as yf
from pathlib import Path
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import STOCK_UNIVERSE, MODEL_CACHE_DIR, TRADING_DAYS, RISK_FREE_RATE

# Per-regime factor weights: [mom_1m, mom_3m, mom_6m, sharpe, low_dd]
REGIME_WEIGHTS = {
    0: np.array([0.25, 0.35, 0.20, 0.15, 0.05]),   # Bull — momentum heavy
    1: np.array([0.15, 0.30, 0.20, 0.25, 0.10]),   # Transition — balanced
    2: np.array([0.05, 0.15, 0.15, 0.25, 0.40]),   # Crisis — defensives
}

# Sector tags for display (best-effort, not comprehensive)
SECTOR_MAP = {
    'AAPL': 'Technology', 'MSFT': 'Technology', 'NVDA': 'Technology',
    'GOOGL': 'Technology', 'AMZN': 'Consumer', 'META': 'Technology',
    'BRK-B': 'Financials', 'JPM': 'Financials', 'JNJ': 'Healthcare',
    'XOM': 'Energy', 'UNH': 'Healthcare', 'V': 'Financials',
    'MA': 'Financials', 'AVGO': 'Technology', 'PG': 'Consumer Staples',
    'HD': 'Consumer', 'COST': 'Consumer Staples', 'LLY': 'Healthcare',
    'ASML': 'Technology (EU)', 'NVO': 'Healthcare (EU)', 'SAP': 'Technology (EU)',
    'AZN': 'Healthcare (EU)', 'SHEL': 'Energy (EU)', 'TTE': 'Energy (EU)',
    'UBS': 'Financials (CH)', 'ABB': 'Industrials (CH)',
}


def _extract_close(raw: pd.DataFrame, tickers: list) -> pd.DataFrame:
    """Robust close-price extractor — handles all yfinance MultiIndex variants."""
    if isinstance(raw.columns, pd.MultiIndex):
        lvl0 = raw.columns.get_level_values(0).unique().tolist()
        lvl1 = raw.columns.get_level_values(1).unique().tolist()
        for label in ('Close', 'Price', 'Adj Close'):
            if label in lvl0:
                return raw[label]
            if label in lvl1:
                return raw.xs(label, axis=1, level=1)
        return raw[lvl0[0]]
    if 'Close' in raw.columns:
        out = raw[['Close']]
        if len(tickers) == 1:
            out.columns = tickers
        return out
    if isinstance(raw, pd.Series):
        return raw.to_frame(name=tickers[0])
    return raw


def _fetch_stock_data(tickers: list, lookback_days: int = 180) -> pd.DataFrame:
    """
    Download closing prices in batches of 5 with a 1 s pause between batches
    to stay within Yahoo Finance rate limits.
    Falls back to per-ticker download for any batch that fails.
    """
    end   = datetime.today().strftime('%Y-%m-%d')
    start = (datetime.today() - timedelta(days=lookback_days + 30)).strftime('%Y-%m-%d')

    frames = []
    n_batches = (len(tickers) + 4) // 5
    for i in range(0, len(tickers), 5):
        batch = tickers[i:i + 5]
        batch_num = i // 5 + 1
        try:
            raw = yf.download(batch, start=start, end=end,
                              auto_adjust=True, progress=False, threads=False)
            if raw.empty:
                print(f'  Screener batch {batch_num}/{n_batches} empty — skipping {batch}')
            else:
                chunk = _extract_close(raw, batch)
                if isinstance(chunk, pd.Series):
                    chunk = chunk.to_frame(name=batch[0])
                elif len(batch) == 1 and chunk.shape[1] == 1:
                    chunk.columns = batch
                frames.append(chunk)
                print(f'  Screener batch {batch_num}/{n_batches} OK '
                      f'({chunk.shape[1]} tickers, {chunk.shape[0]} rows)')
        except Exception as exc:
            print(f'  Screener batch {batch_num}/{n_batches} failed: {exc}')
            # Per-ticker fallback for the failed batch
            for ticker in batch:
                try:
                    t_raw = yf.download(ticker, start=start, end=end,
                                        auto_adjust=True, progress=False)
                    if not t_raw.empty and 'Close' in t_raw.columns:
                        frames.append(t_raw[['Close']].rename(columns={'Close': ticker}))
                except Exception:
                    pass
        time.sleep(1.0)

    if not frames:
        return pd.DataFrame()
    result = pd.concat(frames, axis=1)
    # Drop columns that are entirely NaN (tickers that never downloaded)
    result = result.dropna(axis=1, how='all')
    return result


def _compute_factors(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute factor scores for each stock in the universe."""
    log_ret = np.log(prices / prices.shift(1)).dropna()

    rows = []
    for ticker in prices.columns:
        r = log_ret[ticker].dropna()
        if len(r) < 60:
            continue

        mom_1m = float(r.iloc[-21:].sum())
        mom_3m = float(r.iloc[-63:].sum())
        mom_6m = float(r.iloc[-126:].sum()) if len(r) >= 126 else mom_3m

        vol_6m = float(r.iloc[-126:].std() * np.sqrt(TRADING_DAYS)) if len(r) >= 126 else \
                 float(r.std() * np.sqrt(TRADING_DAYS))
        ann_ret_6m = mom_6m * (TRADING_DAYS / 126)
        sharpe = (ann_ret_6m - RISK_FREE_RATE) / (vol_6m + 1e-8)

        cum = (1 + r.iloc[-126:]).cumprod()
        mdd = float(((cum - cum.cummax()) / cum.cummax()).min()) if len(cum) > 0 else 0.0

        rows.append(dict(
            ticker=ticker, mom_1m=mom_1m, mom_3m=mom_3m,
            mom_6m=mom_6m, sharpe=sharpe, low_dd=-mdd,
        ))

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).set_index('ticker')

    # Z-score normalise each factor across the universe
    for col in ['mom_1m', 'mom_3m', 'mom_6m', 'sharpe', 'low_dd']:
        mu  = df[col].mean()
        std = df[col].std()
        df[col] = (df[col] - mu) / (std + 1e-8)

    return df


def run_screener(regime: int, data: pd.DataFrame) -> list:
    """
    Screen the stock universe and return the top 5 picks for the current regime.

    Parameters
    ----------
    regime : current US regime integer (0=Bull, 1=Transition, 2=Crisis)
    data   : main DataFrame (used only for the date anchor)

    Returns
    -------
    List of dicts, each with:
        rank, ticker, sector, score, mom_3m_pct, sharpe, regime_fit, note
    """
    cache_dir  = Path(MODEL_CACHE_DIR)
    cache_dir.mkdir(exist_ok=True)
    cache_file = cache_dir / f'screener_{datetime.today().strftime("%Y%m%d")}.joblib'

    if cache_file.exists():
        cached = joblib.load(cache_file)
        # Invalidate if regime has changed since last cache
        if cached.get('regime') == regime:
            return cached['picks']

    print('Layer Stocks — Running stock screener …')

    prices = _fetch_stock_data(STOCK_UNIVERSE)
    if prices.empty:
        return _fallback_picks()

    factors = _compute_factors(prices)
    if factors.empty:
        return _fallback_picks()

    w      = REGIME_WEIGHTS.get(regime, REGIME_WEIGHTS[1])
    cols   = ['mom_1m', 'mom_3m', 'mom_6m', 'sharpe', 'low_dd']
    scores = factors[cols].values @ w
    factors['score'] = scores

    top5  = factors.nlargest(5, 'score')
    picks = []
    for rank, (ticker, row) in enumerate(top5.iterrows(), 1):
        # Regime fit description
        if regime == 0:
            note = 'Strong momentum in Bull market'
        elif regime == 1:
            note = 'Quality/balanced profile for Transition'
        else:
            note = 'Defensive characteristics for Crisis regime'

        picks.append(dict(
            rank       = rank,
            ticker     = ticker,
            sector     = SECTOR_MAP.get(ticker, 'Large Cap'),
            score      = round(float(row['score']), 3),
            mom_3m_pct = round(float(row['mom_3m']) * 100, 1),  # normalised z-score × 100
            sharpe_z   = round(float(row['sharpe']), 2),
            note       = note,
        ))

    joblib.dump({'regime': regime, 'picks': picks}, cache_file)
    return picks


def _fallback_picks() -> list:
    """Return placeholder when data download fails."""
    return [dict(rank=i, ticker='N/A', sector='N/A', score=0,
                 mom_3m_pct=0, sharpe_z=0,
                 note='Stock data unavailable — check internet connection')
            for i in range(1, 6)]


if __name__ == '__main__':
    from data.fetcher import fetch_data

    df    = fetch_data()
    picks = run_screener(regime=0, data=df)
    print('\nTop 5 stocks (Bull regime):')
    for p in picks:
        print(f"  {p['rank']}. {p['ticker']:8s} {p['sector']:20s} "
              f"score={p['score']:+.2f}  3m={p['mom_3m_pct']:+.1f}z  {p['note']}")
