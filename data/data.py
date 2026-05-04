import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import yfinance as yf

from portfolio_config import TICKERS, LABELS


def fetch_data():
    print("Fetching 10 years of daily price data …")
    raw = yf.download(TICKERS, period="10y", auto_adjust=True, progress=False)["Close"]
    raw.columns = LABELS
    raw = raw.dropna()
    log_ret = np.log(raw / raw.shift(1)).dropna()
    return raw, log_ret
