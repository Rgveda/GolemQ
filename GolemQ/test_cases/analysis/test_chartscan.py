#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ChartScan 上升主干时序 `ChrtScanUprTukLag`（用户 2026-11-25 算子）。

口径：``Timeline_duration(where((60min Buy < Leadout Maj & < 日线 Sell)
| (日线 Buy < Leadout Maj & < 日线 Sell), 1, -1))``
—— `0` = 该 bar 命中；`>0` = 距最近一次命中的 bar 数。

本文件钉住：事件距离（`cs_class_before`）、两支 Buy 的**或**、`Timeline_duration`
的清零语义，以及「输入未就绪 ⇒ NaN」（不造数）。
"""
import os
import sys

try:
    import GolemQ  # noqa: F401
except ImportError:
    sys.path.insert(0, os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', '..', '..')))

import unittest

import numpy as np
import pandas as pd

from GolemQ.analysis import chartscan as CS
from GolemQ.models.alias import LTT


def _ms(ts):
    return pd.Timestamp(ts).value // 10 ** 6


TS = pd.DatetimeIndex(['2024-01-02 10:30', '2024-01-02 11:30',
                       '2024-01-02 14:00', '2024-01-02 15:00',
                       '2024-01-03 10:30', '2024-01-03 15:00'])


def _feats(cls60=None, coord60=None, clsday=None, coordday=None,
           index=None):
    """ChartScan 列（`Cls*` 取值 ''/'Buy'/'Sell'，`Coord*` 为毫秒）。

    索引 = **扫描行时间**（`cs_class_before` 拿它当事件时间）。
    """
    idx = TS if index is None else index
    n = len(idx)
    return pd.DataFrame({
        'Cls60': cls60 if cls60 is not None else [''] * n,
        'Coord60': coord60 if coord60 is not None else [None] * n,
        'ClsDay': clsday if clsday is not None else [''] * n,
        'CoordDay': coordday if coordday is not None else [None] * n,
    }, index=idx)


class TestNaming(unittest.TestCase):
    def test_constant(self):
        self.assertEqual(LTT.CHARTSCAN_UPRISING_TRUNK_TIMING_LAG,
                         'ChrtScanUprTukLag')

    def test_families(self):
        self.assertEqual(CS.CHARTSCAN_BUY_FAMILIES, ('60', 'Day'))


class TestCsClassBefore(unittest.TestCase):
    def test_distance_to_last_buy(self):
        f = _feats(cls60=['', 'Buy', '', '', 'Buy', ''],
                   coord60=[None, _ms(TS[1]), None, None, _ms(TS[4]), None])
        out = CS.cs_class_before(f, TS, '60')
        np.testing.assert_allclose(out, [np.nan, 0, 1, 2, 0, 1], equal_nan=True)

    def test_none_without_event_or_column(self):
        f = _feats()
        self.assertIsNone(CS.cs_class_before(f, TS, '60'))
        f2 = _feats(cls60=['Buy'] * len(TS))          # 有 Class 没 Coordinates
        self.assertIsNone(CS.cs_class_before(f2, TS, '60'))

    def test_sell_class(self):
        f = _feats(clsday=['Sell', '', '', '', '', ''],
                   coordday=[_ms(TS[0])] + [None] * 5)
        self.assertEqual(CS.cs_class_before(f, TS, 'Day', 'Sell').tolist(),
                         [0.0, 1.0, 2.0, 3.0, 4.0, 5.0])


class TestMask(unittest.TestCase):
    def _frame(self):
        # 60min Buy 在 bar1；日线 Sell/Buy 都在 bar0
        return _feats(cls60=['', 'Buy', '', '', '', ''],
                      coord60=[None, _ms(TS[1]), None, None, None, None],
                      clsday=['Sell', 'Buy', '', '', '', ''],
                      coordday=[_ms(TS[0]), _ms(TS[0]), None, None, None, None])

    def test_union_of_two_families(self):
        lead = np.array([1.0, 2.0, 2.0, 2.0, 2.0, 2.0])
        m = CS.chartscan_buy_mask(self._frame(), TS, lead)
        # 日线族：`buy == day_sell`（同日）永不成立 ⇒ 全部来自 60min 族
        self.assertEqual(m.tolist(),
                         [False, True, True, False, False, False])

    def test_requires_both_comparisons(self):
        f = self._frame()
        # Leadout 变很小 ⇒ Buy 不再早于它 ⇒ 无命中
        m = CS.chartscan_buy_mask(f, TS, np.zeros(len(TS)))
        self.assertFalse(m.any())
        # 缺 Leadout ⇒ None（不猜）
        self.assertIsNone(CS.chartscan_buy_mask(f, TS, None))

    def test_none_without_chartscan(self):
        self.assertIsNone(CS.chartscan_buy_mask(_feats(), TS,
                                                np.ones(len(TS))))


class TestNoLookAhead(unittest.TestCase):
    """事件时间 = 框**首次被扫到**的时刻，不是框坐标（用户 2026-11-26）。"""

    def test_first_seen_not_coord(self):
        # 同一个框（coord = TS[0]）在 TS[1..3] 三次扫描里都被报成 Buy
        idx = TS[:4]
        f = _feats(cls60=['', 'Buy', 'Buy', 'Buy'],
                   coord60=[None, _ms(TS[0]), _ms(TS[0]), _ms(TS[0])],
                   index=idx)
        out = CS.cs_class_before(f, idx, '60')
        # 事件时间 = TS[1]（首次扫到）⇒ TS[0] 那根还**不知道**，必须 NaN
        np.testing.assert_allclose(out, [np.nan, 0, 1, 2], equal_nan=True)

    def test_repeated_scans_do_not_reset(self):
        # 同一个框反复被扫到，也只算一个事件（不重置计数）
        idx = TS[:4]
        f = _feats(cls60=['', 'Buy', 'Buy', 'Buy'],
                   coord60=[None, _ms(TS[0]), _ms(TS[0]), _ms(TS[0])],
                   index=idx)
        out = CS.cs_class_before(f, idx, '60')
        self.assertEqual(out[3], 2.0, '同框重复报到不应把计数清零')

    def test_distinct_boxes_are_distinct_events(self):
        idx = TS[:4]
        f = _feats(cls60=['', 'Buy', '', 'Buy'],
                   coord60=[None, _ms(TS[0]), None, _ms(TS[1])],
                   index=idx)
        out = CS.cs_class_before(f, idx, '60')
        np.testing.assert_allclose(out, [np.nan, 0, 1, 0], equal_nan=True)


class TestLeadoutMajorDistance(unittest.TestCase):
    """日线 `MsLeadoutBfMaj` → **60min bar 数**（不是日线根数）。"""

    def test_maps_to_60min_bars(self):
        idx = pd.DatetimeIndex(
            [f'2024-01-{d:02d} {t}' for d in (2, 3, 4)
             for t in ('10:30', '11:30', '14:00', '15:00')])
        day_idx = pd.DatetimeIndex(['2024-01-02', '2024-01-03', '2024-01-04'])
        # 事件在 2024-01-03（值 <1e-12），前一天是 5（日线单位）
        ms = pd.DataFrame({LTT.MAINSTREAM_LEADOUT_BEFORE_MAJOR: [5.0, 0.0, 1.0]},
                          index=day_idx)
        out = CS.leadout_major_distance(ms, idx)
        # 事件日期落到「前一天 15:00」那根（as-of 口径），之后按 **60min bar** 数
        expected = [np.nan, np.nan, np.nan, 0, 1, 2, 3, 4, 5, 6, 7, 8]
        np.testing.assert_allclose(out, expected, equal_nan=True)

    def test_missing_is_none(self):
        idx = pd.DatetimeIndex(['2024-01-02 15:00'])
        self.assertIsNone(CS.leadout_major_distance(None, idx))
        empty = pd.DataFrame({LTT.MAINSTREAM_LEADOUT_BEFORE_MAJOR: [5.0]},
                             index=pd.DatetimeIndex(['2024-01-02']))
        self.assertIsNone(CS.leadout_major_distance(empty, idx), '没有事件 ⇒ None')


class TestTrunkLag(unittest.TestCase):
    def test_counts_since_last_hit(self):
        f = TestMask()._frame()
        lead = np.array([1.0, 2.0, 2.0, 2.0, 2.0, 2.0])
        s = CS.calc_chartscan_uprising_trunk_lag(f, TS, lead)
        self.assertEqual(s.name, LTT.CHARTSCAN_UPRISING_TRUNK_TIMING_LAG)
        # `calc_event_timing_lag`：命中段 (1,2) ⇒ +1,+2；未命中段 ⇒ -1,-2,-3
        np.testing.assert_allclose(s, [np.nan, 1, 2, -1, -2, -3],
                                   equal_nan=True)

    def test_not_ready_is_nan(self):
        # 60min 无 Buy、日线 Buy 从 bar2 起 ⇒ bar0/1 输入未就绪（NaN）
        f = _feats(clsday=['Sell', '', 'Buy', '', '', ''],
                   coordday=[_ms(TS[0]), None, _ms(TS[2]), None, None, None])
        lead = np.full(len(TS), 5.0)
        s = CS.calc_chartscan_uprising_trunk_lag(f, TS, lead)
        np.testing.assert_allclose(s, [np.nan, np.nan, 1, 2, 3, 4],
                                   equal_nan=True)

    def test_missing_day_sell_is_all_nan(self):
        # 日线 Buy 有、日线 Sell 缺 ⇒ 掩码整体不可用
        f = _feats(clsday=['', '', 'Buy', '', '', ''],
                   coordday=[None, None, _ms(TS[2]), None, None, None])
        s = CS.calc_chartscan_uprising_trunk_lag(f, TS, np.full(len(TS), 5.0))
        self.assertTrue(s.isna().all())

    def test_all_nan_without_columns(self):
        s = CS.calc_chartscan_uprising_trunk_lag(_feats(), TS, np.ones(len(TS)))
        self.assertEqual(len(s), len(TS))
        self.assertTrue(s.isna().all())


if __name__ == '__main__':
    unittest.main()
