#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""`markets/StockCN/metadata_align.py` —— 元数据对 K 线的对齐。

为什么这份用例值得存在
======================
1. **「按日折叠」是这次改造的核心**：分钟线一个交易日有 240 根 bar，而元数据
   一天只有一行 ⇒ 同一天的 bar 必须拿到**同一个值**。旧实现叫
   `stock_min_aligned`（「股票分钟对齐」）却**只对日线做** —— 这个行为
   在这里第一次被钉住。
2. **不许前向填充**：对不上的日子必须是 NaN。用 `ffill` 会**凭空造出当天并不存在**
   的元数据（那天的换手率/资金流被算成前一天的），且**不报错** ——
   这是最像"对"的错法。
3. 索引与长度必须与输入**逐行一致**，否则下游按索引拼接会静默错位。
"""

import os
import sys

try:
    import GolemQ  # noqa: F401
except ImportError:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import unittest

import numpy as np
import pandas as pd

from GolemQ.markets.StockCN.metadata_align import align_metadata_frame


def _kline(rows):
    idx = pd.MultiIndex.from_tuples(rows, names=['ts', 'code'])
    return pd.DataFrame({'close': np.arange(len(rows), dtype=float) + 100.0}, index=idx)


def _meta(rows):
    idx = pd.MultiIndex.from_tuples(rows, names=['date', 'code'])
    return pd.DataFrame({'TurnoverRate': [r[2] for r in rows]}, index=idx).drop(
        columns=[])


class TestDayFolding(unittest.TestCase):
    def test_same_day_bars_get_the_same_value(self):
        """⚠️ **核心行为**：同一天的所有 bar 拿到同一行元数据。"""
        k = _kline([(pd.Timestamp('2026-10-09 10:30'), '600519'),
                    (pd.Timestamp('2026-10-09 11:30'), '600519'),
                    (pd.Timestamp('2026-10-09 14:00'), '600519')])
        m = pd.DataFrame({'TurnoverRate': [0.0028]},
                         index=pd.MultiIndex.from_tuples([('2026-10-09', '600519')],
                                                         names=['date', 'code']))
        out = align_metadata_frame(k, m)
        self.assertEqual(out['TurnoverRate'].tolist(), [0.0028] * 3)

    def test_two_days_two_values(self):
        k = _kline([(pd.Timestamp('2026-10-09 10:30'), '600519'),
                    (pd.Timestamp('2026-10-09 15:00'), '600519'),
                    (pd.Timestamp('2026-10-12 10:30'), '600519')])
        m = pd.DataFrame({'TurnoverRate': [0.0028, 0.0031]},
                         index=pd.MultiIndex.from_tuples(
                             [('2026-10-09', '600519'), ('2026-10-12', '600519')],
                             names=['date', 'code']))
        self.assertEqual(align_metadata_frame(k, m)['TurnoverRate'].tolist(),
                         [0.0028, 0.0028, 0.0031])


class TestNoForwardFill(unittest.TestCase):
    def test_missing_day_is_nan_not_previous_value(self):
        """⚠️ **不许前向填充** —— 缺的那天是 NaN，不是前一天的值。"""
        k = _kline([(pd.Timestamp('2026-10-09 10:30'), '600519'),
                    (pd.Timestamp('2026-10-12 10:30'), '600519'),
                    (pd.Timestamp('2026-10-13 10:30'), '600519')])
        m = pd.DataFrame({'TurnoverRate': [0.0028, 0.0099]},
                         index=pd.MultiIndex.from_tuples(
                             [('2026-10-09', '600519'), ('2026-10-13', '600519')],
                             names=['date', 'code']))
        got = align_metadata_frame(k, m)['TurnoverRate'].tolist()
        self.assertEqual(got[0], 0.0028)
        self.assertTrue(np.isnan(got[1]), '10-12 缺数据 —— 应为 NaN，实测 {}'.format(got[1]))
        self.assertEqual(got[2], 0.0099)


class TestShapeContract(unittest.TestCase):
    def test_index_and_length_preserved(self):
        rows = [(pd.Timestamp('2026-10-09 10:30'), '600519'),
                (pd.Timestamp('2026-10-09 10:30'), '000001'),
                (pd.Timestamp('2026-10-12 10:30'), '600519')]
        k = _kline(rows)
        m = pd.DataFrame({'TurnoverRate': [0.0028]},
                         index=pd.MultiIndex.from_tuples([('2026-10-09', '600519')],
                                                         names=['date', 'code']))
        out = align_metadata_frame(k, m)
        self.assertEqual(len(out), len(k))
        self.assertEqual(list(out.index), list(k.index))

    def test_multi_code_does_not_cross_contaminate(self):
        """同一天两只票 —— 值不能串。"""
        k = _kline([(pd.Timestamp('2026-10-09 10:30'), '600519'),
                    (pd.Timestamp('2026-10-09 10:30'), '000001')])
        m = pd.DataFrame({'TurnoverRate': [0.0028, 0.0056]},
                         index=pd.MultiIndex.from_tuples(
                             [('2026-10-09', '600519'), ('2026-10-09', '000001')],
                             names=['date', 'code']))
        self.assertEqual(align_metadata_frame(k, m)['TurnoverRate'].tolist(),
                         [0.0028, 0.0056])

    def test_empty_meta_gives_all_nan(self):
        k = _kline([(pd.Timestamp('2026-10-09 10:30'), '600519')])
        m = pd.DataFrame({'TurnoverRate': []},
                         index=pd.MultiIndex.from_arrays([[], []], names=['date', 'code']))
        out = align_metadata_frame(k, m)
        self.assertEqual(len(out), 1)
        self.assertTrue(out['TurnoverRate'].isna().all())

    def test_columns_subset(self):
        k = _kline([(pd.Timestamp('2026-10-09 10:30'), '600519')])
        m = pd.DataFrame({'TurnoverRate': [0.0028], 'PERation': [17.73]},
                         index=pd.MultiIndex.from_tuples([('2026-10-09', '600519')],
                                                         names=['date', 'code']))
        out = align_metadata_frame(k, m, columns=['PERation'])
        self.assertEqual(list(out.columns), ['PERation'])
        self.assertEqual(out['PERation'].tolist(), [17.73])


if __name__ == '__main__':
    unittest.main(verbosity=2)
