# coding:utf-8
"""
Risk model constants (stub — to be populated).

Used by services/persistence/ for CVaR risk feature column references.
"""


class _StubMeta(type):
    def __getattr__(cls, name):
        return f'rsk_{name.lower()}'


class RSK(metaclass=_StubMeta):
    """Risk model feature constants (stub)."""
    _instance = None

    CVaR_PEAK_PRICE = 'cvar_peak_price'
    CVaR_PEAK_LOW = 'cvar_peak_low'
    CVaR_PEAK_LOW_PRICE = 'cvar_peak_low_price'
    CVaR_PEAK_LOW_BEFORE = 'cvar_peak_low_before'

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(RSK, cls).__new__(cls)
        return cls._instance
