# coding: utf-8
"""`analysis/timeseries.py` 的重采样器 —— 钉住 pandas 3 的口径变化。

2026-10-20 实跑 `--save tdx` 时出现：

    timeseries.py:106: RuntimeWarning: The 'offset' keyword does not take effect
    when resampling with a 'freq' that is not Tick-like (h, m, s, ms, us, ns)

`GQ_data_min_to_day` 的目标频率是 `1D`（非 Tick-like），`offset="0min"` 本来
就是零偏移；pandas 3 会忽略它并告警。这里既钉「不再告警」，也钉日线聚合值没变。
"""
import os
import sys

try:
    import GolemQ  # noqa: F401
except ImportError:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

import unittest
import warnings

import pandas as pd

from GolemQ.analysis import timeseries as TS


def _min_frame(n=8, freq='15min', code='600519'):
    ts = pd.date_range('2026-10-19 09:30', periods=n, freq=freq)
    idx = pd.MultiIndex.from_arrays([ts, [code] * n], names=['datetime', 'code'])
    return pd.DataFrame({
        'open': range(n),
        'high': [x + 1 for x in range(n)],
        'low': [x - 1 for x in range(n)],
        'close': range(n),
        'volume': [100] * n,
        'amount': [1000] * n,
    }, index=idx)


class TestMinToDay(unittest.TestCase):
    def test_no_offset_warning(self):
        df = _min_frame()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            out = TS.GQ_data_min_to_day(df)

        offset_warnings = [w for w in caught if 'offset' in str(w.message)]
        self.assertEqual(offset_warnings, [],
                         'pandas 3 对 1D 频率会忽略 offset 并告警；不要再传它')

    def test_daily_aggregation_is_unchanged(self):
        df = _min_frame()
        out = TS.GQ_data_min_to_day(df)
        self.assertEqual(len(out), 1)
        self.assertEqual(out['open'].iloc[0], 0)
        self.assertEqual(out['close'].iloc[0], 7)
        self.assertEqual(out['high'].iloc[0], 8)
        self.assertEqual(out['low'].iloc[0], -1)
        self.assertEqual(out['volume'].iloc[0], 800)
        self.assertEqual(out['amount'].iloc[0], 8000)
        self.assertEqual(out['code'].iloc[0], '600519')


if __name__ == '__main__':
    unittest.main(verbosity=2)
