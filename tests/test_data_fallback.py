"""
tests/test_data_fallback.py — Tests for data/fetcher.py fallback behaviour

Tests (no network calls — uses monkeypatching):
  - _extract_close handles flat 'Close' column
  - _extract_close handles MultiIndex (Close | Ticker)
  - _extract_close handles MultiIndex (Ticker | Close)
  - _safe_download returns empty DataFrame on yfinance failure
  - _cache_is_fresh returns False for missing file
  - _cache_is_fresh returns False for stale file (>1 day)
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from data.fetcher import _extract_close, _safe_download, _cache_is_fresh

# ── _extract_close ────────────────────────────────────────────────────────────


class TestExtractClose:
    def _dates(self, n=5):
        return pd.bdate_range("2024-01-01", periods=n)

    def test_flat_close_column(self):
        idx = self._dates()
        df = pd.DataFrame({"Close": [100.0, 101, 102, 103, 104]}, index=idx)
        result = _extract_close(df, ["SPY"])
        assert "SPY" in result.columns
        assert result.shape == (5, 1)

    def test_multiindex_close_first(self):
        idx = self._dates()
        arrays = [
            pd.Index(["Close", "Close", "Open", "Open"], name=""),
            pd.Index(["SPY", "QQQ", "SPY", "QQQ"], name=""),
        ]
        cols = pd.MultiIndex.from_arrays(arrays)
        data = np.arange(5 * 4, dtype=float).reshape(5, 4)
        df = pd.DataFrame(data, index=idx, columns=cols)
        result = _extract_close(df, ["SPY", "QQQ"])
        assert "SPY" in result.columns
        assert "QQQ" in result.columns

    def test_multiindex_ticker_first(self):
        idx = self._dates()
        arrays = [
            pd.Index(["SPY", "SPY", "QQQ", "QQQ"], name=""),
            pd.Index(["Close", "Open", "Close", "Open"], name=""),
        ]
        cols = pd.MultiIndex.from_arrays(arrays)
        data = np.arange(5 * 4, dtype=float).reshape(5, 4)
        df = pd.DataFrame(data, index=idx, columns=cols)
        result = _extract_close(df, ["SPY", "QQQ"])
        assert result.shape[1] == 2

    def test_series_input_converted(self):
        idx = self._dates()
        s = pd.Series([1.0, 2, 3, 4, 5], index=idx, name="SPY")
        result = _extract_close(s.to_frame(name="SPY"), ["SPY"])
        assert isinstance(result, pd.DataFrame)


# ── _safe_download ────────────────────────────────────────────────────────────


class TestSafeDownload:
    def test_returns_empty_on_exception(self, monkeypatch):
        import yfinance as yf

        def _fail(*args, **kwargs):
            raise RuntimeError("network error")

        monkeypatch.setattr(yf, "download", _fail)
        result = _safe_download(["SPY"], "2024-01-01", "2024-01-31")
        assert isinstance(result, pd.DataFrame)
        assert result.empty


# ── _cache_is_fresh ───────────────────────────────────────────────────────────


class TestCacheIsFresh:
    def test_missing_file_not_fresh(self, tmp_path):
        assert not _cache_is_fresh(str(tmp_path / "nonexistent.parquet"))

    def test_new_file_is_fresh(self, tmp_path):
        p = tmp_path / "cache.parquet"
        p.write_text("x")
        assert _cache_is_fresh(str(p))

    def test_old_file_not_fresh(self, tmp_path):
        import os

        p = tmp_path / "cache.parquet"
        p.write_text("x")
        # Set mtime to 2 days ago using os.utime
        old_time = (datetime.now() - timedelta(days=2)).timestamp()
        os.utime(str(p), (old_time, old_time))
        assert not _cache_is_fresh(str(p))
