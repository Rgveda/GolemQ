# coding: utf-8
"""`ZenPivotLag` / `ZenPivotLagMaj` / `ZenPivotLagMin` 的**因果/滚动**守卫。

用户 2026-11-20 定的 checkpoint：三条都必须是滚动计算、没有未来函数。
这些用例用**截断对拍**锁住性质：

* 用全量数据算某日 T 的值；
* 用只到 T 的数据再算一次；
* 两者必须逐值相同。

⚠️ 历史教训：日线「全帧一次 CZSC」在 000711 2026-01-23 会给出 -1（截断是 239），
就是最终 `bi_list` 被后续 bar 改写 —— 本文件第一条用例专门防这个回归。
"""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from GolemQ.models.alias import ZEN
from GolemQ.pipeline.review.contract import MINOR_15MIN_COLUMNS
from GolemQ.pipeline.review.steps.hour_extra import (pivot_lag_by_day,
                                                     pivot_lag_minor_window)


def _ohlc(n: int, freq: str) -> pd.DataFrame:
    idx = pd.date_range('2018-01-01', periods=n, freq=freq)
    x = np.arange(n, dtype=float)
    close = 10.0 + 3.0 * np.sin(x / 17.0) + 0.002 * x
    open_ = close + 0.1 * np.sin(x / 3.0)
    high = np.maximum(open_, close) + 0.2
    low = np.minimum(open_, close) - 0.2
    return pd.DataFrame({
        'open': open_, 'high': high, 'low': low, 'close': close,
        'volume': np.full(n, 1000.0), 'amount': np.full(n, 10000.0),
    }, index=pd.MultiIndex.from_arrays(
        [idx, ['000711'] * n], names=['ts', 'code']))


class TestPivotLagRolling(unittest.TestCase):
    def test_minor_columns_include_pivot_lag(self):
        self.assertIn(ZEN.PIVOT_TIMING_LAG_MINOR, MINOR_15MIN_COLUMNS)

    def test_daily_lag_is_causal(self):
        day = _ohlc(2600, 'B')
        cut = pd.DatetimeIndex(day.index.get_level_values(0))[2000]
        full = pivot_lag_by_day(day, pd.DatetimeIndex([cut]), '000711')
        sub = day[pd.DatetimeIndex(day.index.get_level_values(0)) <= cut]
        trunc = pivot_lag_by_day(sub, pd.DatetimeIndex([cut]), '000711')
        a = full.get(cut, np.nan)
        b = trunc.get(cut, np.nan)
        self.assertTrue((pd.isna(a) and pd.isna(b)) or a == b,
                        '日线 lag 有未来函数：full={} trunc={}'.format(a, b))

    def test_15min_lag_is_causal_at_tail(self):
        m15 = _ohlc(2200, '15min')
        cut = pd.DatetimeIndex(m15.index.get_level_values(0))[2100]
        full = pivot_lag_minor_window(m15, '000711')
        sub = m15[pd.DatetimeIndex(m15.index.get_level_values(0)) <= cut]
        trunc = pivot_lag_minor_window(sub, '000711')
        a, b = full.get(cut, np.nan), trunc.get(cut, np.nan)
        self.assertTrue((pd.isna(a) and pd.isna(b)) or a == b,
                        '15min lag 有未来函数：full={} trunc={}'.format(a, b))


if __name__ == '__main__':
    unittest.main()
