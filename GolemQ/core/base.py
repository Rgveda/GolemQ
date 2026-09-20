# coding:utf-8
"""
Core base utilities (stub — to be populated).

Provides fundamental utility functions used across the codebase.
"""
from datetime import datetime as dt, timedelta


def GQ_util_get_last_day(ts: dt = None, n: int = 0) -> dt:
    """Get the last trading day (stub)."""
    today = ts if ts is not None else dt.now()
    if today.weekday() >= 5:  # Saturday or Sunday
        today = today - timedelta(days=today.weekday() - 4)
    return today.replace(hour=0, minute=0, second=0, microsecond=0)


def set_cpu_affinity_even():
    """Set CPU affinity to even-numbered cores (stub)."""
    pass
