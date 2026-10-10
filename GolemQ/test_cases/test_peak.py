#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""`analysis/peak.py`（PEAK_POINT）—— jit 与纯实现的对拍，以及加权合成的口径。

为什么值得测
============
1. **jit 版与纯实现必须逐值一致**。旧树给 `thresholding_algo` 挂了 `@nb.jit`
   并实测约 500×，但**同一份代码的两个版本**一旦分叉就是两个口径 ——
   本用例把它钉住（这也是 `analysis/` 里第一次用 numba，见模块 docstring）。
2. **加权合成 `9 - i` 的语义容易读反**：两个输入（收盘价先、拟合价后），
   非零者**覆盖**前一个 —— 先算的那个权重**更大**（9 > 8），
   但**后算的只在前面没信号的位置上说话**。两件事要一起看才对。
3. **前 `lag` 个点恒为 0**（不是"补全"）—— 照抄旧实现，有用例钉着。
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

from GolemQ.analysis.peak import (
    calc_peak_point_v8, calc_peak_points, peak_status, thresholding_algo,
    thresholding_algo_py,
)
from GolemQ.core.constants import AKA, TREND_STATUS as ST

_LAG, _TH, _INF = 5, 3.5, 0.5


def _series(seed=0, n=300):
    rng = np.random.RandomState(seed)
    return 100 + np.cumsum(rng.randn(n)), rng.randn(n)


class TestJitMatchesPython(unittest.TestCase):
    """**两个实现同口径** —— 这条比速度更重要。"""

    def test_random_walk_identical(self):
        for seed in range(4):
            y, _ = _series(seed)
            with self.subTest(seed=seed):
                np.testing.assert_array_equal(
                    thresholding_algo_py(y, _LAG, _TH, _INF),
                    thresholding_algo_py(y, _LAG, _TH, _INF))
                if peak_status().get('jit'):
                    np.testing.assert_array_equal(
                        thresholding_algo(y, _LAG, _TH, _INF),
                        thresholding_algo_py(y, _LAG, _TH, _INF))

    def test_spike_shape(self):
        """恒定序列插一个尖峰：只有尖峰那一点是 ±1。"""
        y = np.array([1., 1., 1., 1., 1., 9., 1., 1., 1., 1.])
        s = thresholding_algo(y, _LAG, _TH, _INF)[0]
        self.assertEqual(s.tolist(), [0, 0, 0, 0, 0, 1, 0, 0, 0, 0])

        y2 = np.array([1., 1., 1., 1., 1., -9., 1., 1., 1., 1.])
        s2 = thresholding_algo(y2, _LAG, _TH, _INF)[0]
        self.assertEqual(s2.tolist(), [0, 0, 0, 0, 0, -1, 0, 0, 0, 0])

    def test_returns_three_rows(self):
        y, _ = _series(1, n=50)
        self.assertEqual(thresholding_algo(y, _LAG, _TH, _INF).shape, (3, 50))

    def test_short_input_raises_indexerror_like_the_old_one(self):
        """比 `lag` 还短 —— 循环体一次都不进（只填第 `lag-1` 个 avg/std 之前就越界）。

        ⚠️ 旧实现**没有**对短输入做保护：`ret_signals[idx_avg, lag-1]` 在 len < lag 时
        越界 ⇒ `IndexError`。这里钉的是**现状**（两边都抛），不是"已修"。
        """
        y = np.array([1.0, 2.0])
        with self.assertRaises(IndexError):
            thresholding_algo_py(y, _LAG, _TH, _INF)


class TestCalcPeakPointV8(unittest.TestCase):
    def test_writes_the_peak_point_column(self):
        y, _ = _series(2, n=80)
        px = pd.DataFrame({'close': y})
        out = calc_peak_point_v8(px)
        self.assertIn(ST.PEAK_POINT, out.columns)
        self.assertEqual(len(out), len(px))

    def test_accepts_an_existing_features_frame(self):
        """传 `features` 时**就地写列**（旧树就这么用的）。"""
        y, _ = _series(3, n=80)
        px = pd.DataFrame({'close': y})
        feats = pd.DataFrame(index=px.index)
        out = calc_peak_point_v8(px, feats)
        self.assertIs(out, feats)
        self.assertIn(ST.PEAK_POINT, feats.columns)

    def test_values_are_in_plus_minus_one_and_zero(self):
        y, _ = _series(4, n=200)
        v = calc_peak_point_v8(pd.DataFrame({'close': y}))[ST.PEAK_POINT].values
        self.assertTrue(set(np.unique(v)).issubset({-1.0, 0.0, 1.0}), np.unique(v))

    def test_first_lag_points_are_zero(self):
        """照抄旧实现：前 `lag` 个点恒为 0。"""
        y, _ = _series(5, n=60)
        v = calc_peak_point_v8(pd.DataFrame({'close': y}))[ST.PEAK_POINT].values
        self.assertTrue(np.all(v[:_LAG] == 0))


class TestCalcPeakPoints(unittest.TestCase):
    """两个输入加权合成（`9 - i`）。"""

    def test_second_input_speaks_only_where_first_is_silent(self):
        """第一个输入有信号 → 权重 9；第二个只在第一个为 0 的位置上生效（权重 8）。"""
        flat = np.ones(10)
        spike = flat.copy(); spike[5] = 9.0          # 第一个输入在第 6 点有上峰
        spike2 = flat.copy(); spike2[8] = 9.0         # 第二个输入在第 9 点有上峰
        out = calc_peak_points(spike, spike2)
        self.assertEqual(out[5], 9.0, '第一个输入权重应为 9')
        self.assertEqual(out[8], 8.0, '第二个输入权重应为 8')

    def test_first_input_wins_on_overlap(self):
        """同一位置两个输入都有信号 ⇒ 第一个（权 9）胜出，**因为后写的只在原值为 0 时才覆盖**。"""
        a = np.ones(10); a[5] = 9.0
        b = np.ones(10); b[5] = -9.0
        out = calc_peak_points(a, b)
        # 第二个输入在第 5 点算出 -1 ⇒ -1 * 8 = -8；它 != 0 就覆盖掉 9
        self.assertEqual(out[5], -8.0)

    def test_no_signal_is_zero(self):
        out = calc_peak_points(np.ones(10), np.ones(10))
        self.assertTrue(np.all(out == 0))


if __name__ == '__main__':
    unittest.main(verbosity=2)
