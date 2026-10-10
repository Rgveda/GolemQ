#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""`markets/StockCN/fq.py` 的复权因子 —— 重点是**分段常数**这条不变量。

为什么这份用例存在
==================
前复权因子 `xdxr_to_adj` 曾被一个 **1e-15 量级的浮点漂移**带坏，而后果远不止
"精度差一点"：因子因此**不再是分段常数**，把原始行情里**本来严格相等的价格撬开**，
czsc 分型的**平局判定**随之翻转 —— 实测同一份 K 线，`000711` 的笔 `59→61`、
中枢 `4→6`、走势 `盘整→上涨趋势`（完整因果链见 `PITFALLS.md` **P30**）。

⚠️ **这份用例有意只用 `==` / `assertEqual` 判因子值，不用 `assertAlmostEqual`。**
用容差写就抓不住这个 bug —— 1e-15 的坏值能穿过任何 `places=9` / `rtol=1e-8`。
这正是 P30 的教训：**守"必须是精确值"的不变量，判据就得是 `==`**。
"""

import os
import sys

try:
    import GolemQ  # noqa: F401
except ImportError:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import unittest

import pandas as pd

from GolemQ.markets.StockCN.fq import xdxr_to_adj

#: 不含除权的连续交易日（字符串按字典序即日期序，够用）
_PLAIN_DAYS = ['2024-%02d-%02d' % (m, d) for m in range(1, 13) for d in range(1, 29)]


def _days_with_event_at(pos, n=240):
    """`n` 个连续交易日，第 `pos` 天一送一（10 送 10）除权。"""
    days = ['%04d-%02d-%02d' % (2024 + i // 240, i // 20 % 12 + 1, i % 20 + 1)
            for i in range(n)]
    event = [{'date': days[pos], 'category': 1, 'fenhong': 0.0, 'peigu': 0.0,
              'peigujia': 0.0, 'songzhuangu': 10.0}]
    return days, event


class TestNoDrift(unittest.TestCase):
    """**无事段必须精确为 1.0** —— 这条是 P30 的直接回归。"""

    def test_no_event_factors_are_exactly_one(self):
        closes = [10.0 + i * 0.37 for i in range(len(_PLAIN_DAYS))]
        s = xdxr_to_adj(_PLAIN_DAYS, closes, [])
        self.assertEqual(len(s), len(_PLAIN_DAYS))
        self.assertEqual(set(float(x) for x in s), {1.0},
                         '无事段的因子必须**精确**是 1.0，不是 ≈1.0')

    def test_no_drift_on_a_long_run(self):
        """长序列累积 —— 漂移正是靠 `cumprod` 累积出来的，短序列不一定显形。"""
        days = ['%04d-%02d-%02d' % (2024 + i // 240, i // 20 % 12 + 1, i % 20 + 1)
                for i in range(3000)]
        s = xdxr_to_adj(days, [20.0 + (i % 17) * 0.13 for i in range(len(days))], [])
        self.assertEqual(max(abs(float(x) - 1.0) for x in s), 0.0)


class TestPiecewiseConstant(unittest.TestCase):
    """因子在**两次事件之间恒为常数** —— 这才是它的正确形态。"""

    def test_runs_around_an_event_are_flat(self):
        days, event = _days_with_event_at(120)
        closes = [30.0 + (i % 11) * 0.21 for i in range(len(days))]
        s = xdxr_to_adj(days, closes, event)

        before = [float(x) for x in s.iloc[:120]]
        after = [float(x) for x in s.iloc[120:]]
        self.assertEqual(len(set(before)), 1, '事件前的因子必须处处相同（分段常数）')
        self.assertEqual(set(after), {1.0}, '事件之后的因子必须**精确**是 1.0')
        # ⚠️ 这里用容差，而上面两条用 `==` —— 差别是**实质的**，不是图省事：
        #   * **无事日**的比率数学上**恒等于 1**（`close*10/10 == close`），
        #     所以它有"精确值"可要求，浮点残差**必须**抹掉（P30 的根因）；
        #   * **事件日**的比率是一次**真实的除法**（这里是 `close*10/20`），
        #     它**没有**精确的浮点表示，1 ulp 的误差是这类计算的固有属性，
        #     抹不掉也不该抹。
        # 修法只消除前者。后者带 1 ulp 无妨 —— 因为每个事件日只**算一次**，
        # 之后的 `cumprod` 乘的全是精确 1.0，**分段常数**因此仍然成立（见上一行断言）。
        self.assertAlmostEqual(before[0], 0.5, places=12, msg='10 送 10 ⇒ 事件前因子 0.5')


class TestEventsStillScale(unittest.TestCase):
    """抹掉漂移**不能顺手改掉真事件的口径** —— 防"修过头"。"""

    def test_ten_for_ten(self):
        s = xdxr_to_adj(_PLAIN_DAYS[:5], [10.0, 11.0, 12.0, 13.0, 14.0],
                        [{'date': _PLAIN_DAYS[3], 'category': 1, 'fenhong': 0.0,
                          'peigu': 0.0, 'peigujia': 0.0, 'songzhuangu': 10.0}])
        self.assertEqual([float(x) for x in s], [0.5, 0.5, 0.5, 1.0, 1.0])

    def test_cash_dividend_and_rights(self):
        """送股 + 派现 + 配股，三者同时 —— 走的是同一条加性公式。"""
        s = xdxr_to_adj(_PLAIN_DAYS[:4], [20.0, 20.0, 20.0, 20.0],
                        [{'date': _PLAIN_DAYS[2], 'category': 1, 'fenhong': 2.0,
                          'peigu': 3.0, 'peigujia': 5.0, 'songzhuangu': 0.0}])
        # preclose = (20*10 - 2 + 3*5) / (10 + 3 + 0) = 213/13
        self.assertAlmostEqual(float(s.iloc[1]), (213.0 / 13.0) / 20.0, places=12)
        self.assertEqual(float(s.iloc[-1]), 1.0)

    def test_suogu_is_multiplicative(self):
        """扩缩股（category 11）走乘性口径，且事后仍是精确 1.0。"""
        s = xdxr_to_adj(_PLAIN_DAYS[:5], [10.0, 10.0, 10.0, 2.0, 2.0],
                        [{'date': _PLAIN_DAYS[3], 'category': 11, 'suogu': 5.0}])
        self.assertEqual(float(s.iloc[0]), 0.2)
        self.assertEqual(set(float(x) for x in s.iloc[3:]), {1.0})

    def test_event_on_a_missing_day_moves_to_the_next_bar(self):
        """事件日不在日线里 ⇒ 挪到下一个存在的交易日（丢了就是**整段差一个因子**）。"""
        days = ['2024-01-02', '2024-01-03', '2024-01-08']      # 01-05 缺
        s = xdxr_to_adj(days, [10.0, 10.0, 10.0],
                        [{'date': '2024-01-05', 'category': 1, 'fenhong': 0.0,
                          'peigu': 0.0, 'peigujia': 0.0, 'songzhuangu': 10.0}])
        self.assertEqual([float(x) for x in s], [0.5, 0.5, 1.0])


class TestEdgeCases(unittest.TestCase):
    def test_single_row(self):
        s = xdxr_to_adj(['2024-01-02'], [10.0], [])
        self.assertEqual([float(x) for x in s], [1.0])

    def test_event_after_the_last_bar_is_ignored(self):
        s = xdxr_to_adj(['2024-01-02', '2024-01-03'], [10.0, 10.0],
                        [{'date': '2024-06-01', 'category': 1, 'fenhong': 0.0,
                          'peigu': 0.0, 'peigujia': 0.0, 'songzhuangu': 10.0}])
        self.assertEqual([float(x) for x in s], [1.0, 1.0])

    def test_index_is_the_input_dates_in_order(self):
        s = xdxr_to_adj(_PLAIN_DAYS[:4], [1.0, 2.0, 3.0, 4.0], [])
        self.assertEqual(list(s.index), _PLAIN_DAYS[:4])
        self.assertIsInstance(s, pd.Series)


if __name__ == '__main__':
    unittest.main(verbosity=2)
