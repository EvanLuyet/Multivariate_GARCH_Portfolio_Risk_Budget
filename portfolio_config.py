import numpy as np

TICKERS      = ["^GSPC", "QQQ", "EZU"]
LABELS       = ["S&P",   "NDX", "EUR"]
W_BASE       = np.array([0.70, 0.15, 0.15])

TRADING_DAYS = 252
RF           = 0.02

VOL_TARGET   = 0.12
VOL_LOW      = 0.10
VOL_HIGH     = 0.14
SCALE_CAP    = 1.30

RC_TRIM_THR  = 0.60
RC_TRIM_AMT  = 0.20

ROLL_GARCH   = 252
ROLL_CORR    = 63
