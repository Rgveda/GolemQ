#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""多周期买点：基础信号 `QuadMsUprBf*` + 权重 `QuadMsCredCpd`（2026-11-24 用户算子）。

口径：15min 基础 1 / 60min 2 / day 3；该周期命中且
`Mainstream_leadin_Before < MACD_CROSS_SX_BEFORE` 再 +1；三周期按自然日相加，
≥4 视为共振。日权重按「盘中用上一交易日、15:00 用当日」因果映射到 60min。

本文件钉住：权重算术、按日汇总、因果映射，以及常量名（LTT/FTR 风格）。
"""
import os
import sys

try:
    import GolemQ  # noqa: F401
except ImportError:
    sys.path.insert(0, os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', '..', '..')))

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from GolemQ.analysis import mtf_buy_weight as M
from GolemQ.models.alias import LTT


def _frame(idx, freq):
    f = pd.DataFrame({'open': 1.0, 'high': 1.0, 'low': 1.0,
                      'close': 1.0, 'volume': 1.0}, index=idx)
    f.attrs['freq'] = freq
    return f


class TestNaming(unittest.TestCase):
    """常量名：全称 `QUADRANT_MAINSTREAM_CREDIT_COMPOUNDED`，缩写 `QuadMsCredCpd`。"""

    def test_constant_abbreviation(self):
        self.assertEqual(LTT.QUADRANT_MAINSTREAM_CREDIT_COMPOUNDED, 'QuadMsCredCpd')

    def test_base_weights_and_threshold(self):
        self.assertEqual(M.BASE_WEIGHTS, {'15min': 1, '60min': 2, 'day': 3})
        self.assertEqual(M.RESONANCE_THRESHOLD, 4)

    def test_base_signal_constants(self):
        self.assertEqual(LTT.QUADRANT_MAINSTREAM_UPRISING_BEFORE_MINOR,
                         'QuadMsUprBfMin')
        self.assertEqual(LTT.QUADRANT_MAINSTREAM_UPRISING_BEFORE,
                         'QuadMsUprBf')
        self.assertEqual(LTT.QUADRANT_MAINSTREAM_UPRISING_BEFORE_MAJOR,
                         'QuadMsUprBfMaj')


class TestCompose(unittest.TestCase):
    def test_weight_arithmetic(self):
        self.assertEqual(M._compose_day_weight(True, True, True, True,
                                               True, True), 1 + 2 + 3 + 3)
        self.assertEqual(M._compose_day_weight(False, False, False, False,
                                               False, False), 0)
        self.assertEqual(M._compose_day_weight(True, False, True, True,
                                               False, False), 1 + 2 + 1)


class TestDailyWeights(unittest.TestCase):
    """`_daily_weights`：按自然日把三周期权重相加。"""

    def _patched_state(self, ohlc, annual):
        n = len(ohlc)
        has, q = {'15min': (True, True), '60min': (True, False),
                  'day': (True, True)}[ohlc.attrs['freq']]
        return pd.DataFrame({'base': [has] * n, 'qual': [q] * n},
                            index=ohlc.index)

    def test_sums_per_day(self):
        d1 = pd.Timestamp('2024-01-02')
        f60 = _frame(pd.DatetimeIndex([d1 + pd.Timedelta(hours=10, minutes=30),
                                       d1 + pd.Timedelta(hours=15)]), '60min')
        f15 = _frame(pd.DatetimeIndex([d1 + pd.Timedelta(hours=10, minutes=30)]),
                     '15min')
        fd = _frame(pd.DatetimeIndex([d1]), 'day')
        with patch.object(M, '_freq_state', self._patched_state):
            w = M._daily_weights(f60, f15, fd)
        # 15min: 1+1 = 2；60min: 2；day: 3+1 = 4
        self.assertEqual(float(w.loc[d1]), 8.0)

    def test_missing_qual_only_base(self):
        d1 = pd.Timestamp('2024-01-02')

        def state(ohlc, annual):
            n = len(ohlc)
            return pd.DataFrame({'base': [True] * n, 'qual': [False] * n},
                                index=ohlc.index)

        fd = _frame(pd.DatetimeIndex([d1]), 'day')
        with patch.object(M, '_freq_state', state):
            w = M._daily_weights(fd, fd, fd)
        self.assertEqual(float(w.loc[d1]), 1 + 2 + 3)


class TestBaseSignals(unittest.TestCase):
    """`calc_mtf_buy_signals`：三列基础信号，day 列按因果映射。"""

    def test_columns_and_causal_day(self):
        d1, d2 = pd.Timestamp('2024-01-02'), pd.Timestamp('2024-01-03')
        idx60 = pd.DatetimeIndex([
            d1 + pd.Timedelta(hours=10, minutes=30),
            d1 + pd.Timedelta(hours=15),
            d2 + pd.Timedelta(hours=10, minutes=30),
            d2 + pd.Timedelta(hours=15),
        ])
        f60, f15 = _frame(idx60, '60min'), _frame(idx60, '15min')
        fd = _frame(pd.DatetimeIndex([d1, d2]), 'day')

        def state(ohlc, annual):
            n = len(ohlc)
            base = [True] * n
            if ohlc.attrs['freq'] == 'day':
                base = [t >= d2 for t in ohlc.index]   # d1 无、d2 有
            return pd.DataFrame({'base': base, 'qual': [False] * n},
                                index=ohlc.index)

        with patch.object(M, '_freq_state', state):
            out = M.calc_mtf_buy_signals(f60, f15, fd)
        self.assertEqual(list(out.columns), [
            LTT.QUADRANT_MAINSTREAM_UPRISING_BEFORE_MINOR,
            LTT.QUADRANT_MAINSTREAM_UPRISING_BEFORE,
            LTT.QUADRANT_MAINSTREAM_UPRISING_BEFORE_MAJOR])
        day = out[LTT.QUADRANT_MAINSTREAM_UPRISING_BEFORE_MAJOR]
        self.assertEqual(list(day), [False, False, False, True],
                         '盘中用上一交易日；15:00 用当日')


class TestCausalMapping(unittest.TestCase):
    """日权重 → 60min：盘中用上一交易日，15:00 用当日。"""

    def test_intraday_uses_previous_day(self):
        d1, d2 = pd.Timestamp('2024-01-02'), pd.Timestamp('2024-01-03')
        idx = pd.DatetimeIndex([
            d1 + pd.Timedelta(hours=10, minutes=30),
            d1 + pd.Timedelta(hours=15),
            d2 + pd.Timedelta(hours=10, minutes=30),
            d2 + pd.Timedelta(hours=15),
        ])
        f60 = _frame(idx, '60min')
        daily = pd.Series([1.0, 2.0], index=pd.DatetimeIndex([d1, d2]))
        with patch.object(M, '_daily_weights_from_states', return_value=daily):
            w = M.calc_mtf_buy_weight(f60, f60, f60)
        self.assertTrue(np.isnan(w.iloc[0]), '第一天的盘中没有上一交易日')
        self.assertEqual(w.iloc[1], 1.0)
        self.assertEqual(w.iloc[2], 1.0, '第二天盘中用 1-02 的 1')
        self.assertEqual(w.iloc[3], 2.0, '15:00 用当日 1-03 的 2')
        self.assertEqual(w.name, LTT.QUADRANT_MAINSTREAM_CREDIT_COMPOUNDED)


if __name__ == '__main__':
    unittest.main()
