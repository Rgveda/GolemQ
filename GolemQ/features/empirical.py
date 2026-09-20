# coding:utf-8
"""
Feature empirical data stubs (to be populated).

Functions referenced by services/persistence/ for data loading.
"""
import pandas as pd


def load_massive_reviews(symbol, start=None, end=None, compact=True, collections=None):
    """Load massive review features (stub)."""
    return pd.DataFrame()


def load_massive_model(revision='compact', eval_range='fast', start=None, end=None, collections=None):
    """Load massive reality model (stub)."""
    return pd.DataFrame()


def save_stock_metadata(data, collections=None):
    """Save stock metadata (stub)."""
    pass


def GQ_fetch_stock_metadata_major(code, verbose=False, start=None, end=None, collections=None):
    """Fetch major stock metadata (stub)."""
    return pd.DataFrame()
