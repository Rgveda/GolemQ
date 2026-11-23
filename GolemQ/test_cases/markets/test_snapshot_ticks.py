#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""快照型 L1 tick → bar 的量纲（2026-11-23 实测后新增）。

腾讯 L1 落库的是**快照**，不是逐笔成交：

* `vol` / `amount` 是**当日累计**；`close` 是**昨收**、现价在 `price`；
* 累计 vol 单位是**股**，历史 60min 表的 `volume` 是**手**（差 100）。

老路径（`tick_resample_1min` 取 last → `min_resample` 取 sum）会把累计量重复
累加 —— 实测 000711 当天 14:00 那根虚高 **3500 倍**（9.77e9 vs 日线 2.75e6）。
本用例钉住新口径：**每根 bar = 区间末累计 − 区间首之前那根**，vol 换成手。
"""

import os
import sys
import unittest

import numpy as np
import pandas as pd

try:
    import GolemQ  # noqa: F401
except ImportError:
    sys.path.insert(0, os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from GolemQ.markets.StockCN.realtime import _snapshot_ticks_to_bars


def _ticks(n=180, start='2026-11-23 13:00:00', step='10s',
           px=10.0, per_tick_shares=1000):
    ts = pd.date_range(start, periods=n, freq=step)
    cum = np.arange(1, n + 1) * per_tick_shares          # 累计股数
    return pd.DataFrame({
        'code': '000711',
        'price': px,
        'close': px - 1,                                  # 昨收（不应被当现价用）
        'vol': cum,
        'amount': cum * px,
    }, index=ts)


class TestSnapshotTicksToBars(unittest.TestCase):

    def test_volume_is_delta_in_hands(self):
        t = _ticks(n=180, step='10s', per_tick_shares=1000)   # 30 分钟，180 条
        out = _snapshot_ticks_to_bars(t, '60min')
        self.assertGreater(len(out), 0)
        # 每根 bar 的量 = 该区间内的增量 / 100（股 → 手）。
        # ⚠️ 第一根没有「上一根」可减 —— 约定用**首条 tick 的累计**做基准
        # （等价于「首条之前的量不可见」，与实测 000711 13:21 才开始订阅的情形一致）。
        total = 180 * 1000 / 100.0
        self.assertLess(float(out['volume'].sum()), total * 1.1,
                        '累计量被当逐笔 sum 了（量纲又错了）')
        self.assertAlmostEqual(float(out['volume'].sum()),
                               (180 - 1) * 1000 / 100.0, delta=1.0)

    def test_amount_is_delta_not_cumulative(self):
        t = _ticks(n=60, step='10s', px=10.0, per_tick_shares=1000)
        out = _snapshot_ticks_to_bars(t, '60min')
        # 同上：首条 tick 的 1000 股作为基准被跳过
        self.assertAlmostEqual(float(out['amount'].sum()), (60 - 1) * 1000 * 10.0,
                               delta=1.0)

    def test_price_comes_from_price_not_close(self):
        """"close" 是昨收 —— OHLC 必须取 `price`。"""
        t = _ticks(n=60, px=10.0)
        out = _snapshot_ticks_to_bars(t, '60min')
        self.assertTrue((out['close'] == 10.0).all())
        self.assertTrue((out['close'] != 9.0).all())

    def test_60min_buckets(self):
        """下午段 13:00–15:00 按 60min 分桶（offset=0min、closed=right）。"""
        t = _ticks(n=721, start='2026-11-23 13:00:00', step='10s')   # 13:00→15:00
        out = _snapshot_ticks_to_bars(t, '60min')
        labels = [str(x) for x in out.index.get_level_values(0)]
        # ⚠️ 恰好落在 13:00:00 的那条 tick 会自成 (12:00,13:00] 桶 ⇒ 多出一根 13:00。
        # 真实订阅的 tick 不会正好压在整点上，所以实测 000711 只有 14:00/15:00 两根。
        self.assertIn('2026-11-23 14:00:00', labels)
        self.assertIn('2026-11-23 15:00:00', labels)
        self.assertLessEqual(len(labels), 3)


class TestSnapshotTicksToDay(unittest.TestCase):
    """日线：半场订阅也要得到**当日**的 open/high/low 与总量（D90 补）。"""

    def _day_ticks(self, with_day_fields=True):
        # 13:21 起才订阅（000711 当天实况），累计量已含上午
        ts = pd.date_range('2026-11-23 13:21:30', periods=60, freq='60s')
        cum = 270_433_600 + np.arange(60) * 1000
        t = pd.DataFrame({
            'code': '000711', 'price': 4.25, 'close': 3.86,
            'vol': cum, 'amount': cum * 4.02,
        }, index=ts)
        if with_day_fields:
            t['open'] = 3.80
            t['high'] = 4.25
            t['low'] = 3.68
        return t

    def test_day_uses_snapshot_day_fields(self):
        """open/high/low 取快照的**日级**字段，不是「订阅起点的现价」。"""
        out = _snapshot_ticks_to_bars(self._day_ticks(), 'day')
        self.assertEqual(len(out), 1)
        row = out.iloc[0]
        self.assertAlmostEqual(float(row['open']), 3.80)
        self.assertAlmostEqual(float(row['high']), 4.25)
        self.assertAlmostEqual(float(row['low']), 3.68)
        self.assertAlmostEqual(float(row['close']), 4.25)

    def test_day_volume_is_last_cumulative_in_hands(self):
        """日线量 = 最后一条**累计**值 ÷ 100 —— 不是增量、也不是逐笔 sum。"""
        t = self._day_ticks()
        out = _snapshot_ticks_to_bars(t, 'day')
        self.assertAlmostEqual(float(out['volume'].iloc[0]),
                               float(t['vol'].iloc[-1]) / 100.0, delta=1.0)
        self.assertAlmostEqual(float(out['amount'].iloc[0]),
                               float(t['amount'].iloc[-1]), delta=1.0)

    def test_day_falls_back_when_day_fields_absent(self):
        """没有日级字段的来源 ⇒ 用快照 price 的 首/最高/最低/末。"""
        out = _snapshot_ticks_to_bars(self._day_ticks(with_day_fields=False), 'day')
        row = out.iloc[0]
        self.assertAlmostEqual(float(row['open']), 4.25)
        self.assertAlmostEqual(float(row['high']), 4.25)
        self.assertAlmostEqual(float(row['low']), 4.25)

    def test_day_zero_open_is_treated_missing(self):
        """集合竞价未定价时快照 open/low 为 0 ⇒ 当缺失，回落 price。"""
        t = self._day_ticks()
        t['open'] = 0.0
        t['low'] = 0.0
        row = _snapshot_ticks_to_bars(t, 'day').iloc[0]
        self.assertAlmostEqual(float(row['open']), 4.25)
        self.assertAlmostEqual(float(row['low']), 4.25)
        self.assertAlmostEqual(float(row['high']), 4.25)


if __name__ == '__main__':
    unittest.main()
