# coding:utf-8
"""
K-line data fetching stubs (to be populated).

Functions referenced by services/persistence/ for fetching price data.
"""
import pandas as pd


class KlineResult:
    def __init__(self, data=None):
        self.data = data if data is not None else pd.DataFrame()


def get_kline_price_min(symbol, start=None, end=None, verbose=False, realtime=True):
    """Fetch minute kline data (stub)."""
    return KlineResult(), 'unknown'


def get_kline_price_v3(symbol, start=None, end=None, verbose=False, realtime=True):
    """Fetch daily kline data v3 (stub)."""
    return KlineResult(), 'unknown'
