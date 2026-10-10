#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""`analysis/renko_jit.py` —— jit 版与 `renko.py` 的**逐值对拍**。

为什么这份用例是这次加速的**唯一防线**
======================================
`renko_jit.py` 里有一份**无法消除的重复**：`renko_in_cluster_group_jit` 与
`renko_trend_cross_func_jit` 是 `renko.py` 同名函数的近似拷贝（只把 `fminbound`
的目标函数换成 jit 版）。之所以不共用，是因为那两个函数把 `evaluate_renko` /
`renko_in_cluster_group` 当**模块全局**取用，没有注入口。

**平行实现不会报错，只会分叉** —— 所以这里一条容差都不给：**全部 `array_equal`**。

⚠️ **判据必须逐值，不许用容差**：`fminbound` 是**迭代收敛**的，目标函数差哪怕
1 ulp，搜索路径就可能落到另一个 `optimal_brick_sfo`，于是 L 族整套砖界**全变**。
用 `assert_allclose` 会把"搜到了另一个极值点"当成通过。

本用例钉住的三件事
==================
1. **砖序列逐值相同**（含两个**边界**：跳空 ⇒ 砖数远超 bar 数；反向 `|gap|==1`
   ⇒ 一块都不推）。⚠️ 第一条是**回归**：见下面 `test_capacity_...`。
2. **目标函数逐值相同** —— 那是实测的热点（654×）。
3. **端到端 14 列逐值相同**，且**不是全空**（防"两边都静默早退"冒充通过）。

numba 缺失时整份**跳过**（与 `test_regtree_jit.py` 同）。
"""

import os
import sys

try:
    import GolemQ  # noqa: F401
except ImportError:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import time
import unittest

import numpy as np
import pandas as pd

from GolemQ.analysis import renko as R
from GolemQ.analysis import renko_jit as J
from GolemQ.core.constants import FIELD as FLD

_HAS_NUMBA = J.available()


def _ohlc(n=420, seed=0):
    """合成行情（列名必须是 `high`/`low`/`open`/`close` —— 纯版走属性访问）。"""
    rng = np.random.RandomState(seed)
    close = 20 + np.cumsum(rng.randn(n)) * 0.3
    idx = pd.MultiIndex.from_arrays(
        [pd.date_range('2024-01-01', periods=n, freq='D'), ['600519'] * n],
        names=['date', 'code'])
    return pd.DataFrame({'open': close + rng.randn(n) * 0.1,
                         'high': close + np.abs(rng.randn(n)) * 0.2,
                         'low': close - np.abs(rng.randn(n)) * 0.2,
                         'close': close,
                         'volume': np.full(n, 1e5)}, index=idx)


def _ref_dirs(prices, brick_size):
    """纯 Python 基准：走 `class renko` 的 `build_history`。"""
    obj = R.renko()
    obj.set_brick_size(auto=False, brick_size=brick_size)
    obj.build_history(prices=np.asarray(prices, dtype=np.float64))
    return np.asarray(obj.get_renko_directions(), dtype=np.float64)


@unittest.skipUnless(_HAS_NUMBA, 'numba 不可用 —— jit 对拍跳过')
class TestBricksDirectionsMatches(unittest.TestCase):
    """jit 核 vs `class renko.__renko_rule` —— **逐字复刻**，逐值相同。"""

    def test_capacity_estimation_is_not_the_bar_count(self):
        """⚠️ **回归用例**：砖数**不是** O(bar 数)。

        一次 `1 → 100` 的跳空配 `0.01` 的砖高 ⇒ **9901 块砖**，而 bar 数只有 **2**。
        第一版原型按 `n + 16` 开缓冲区并 `break` —— **静默截断成 18 块**，
        而且随机游走的对拍**恰好没触发它**。这条把那个形状钉死。

        真实触发场景不是这种极端跳空，而是**低波动 + 小砖高**（可转债/ETF 常见）。
        """
        px = np.array([1.0, 100.0])
        got = J.bricks_directions(px, 0.01)
        ref = _ref_dirs(px, 0.01)
        self.assertEqual(len(ref), 9901)
        self.assertEqual(len(got), 9901, '砖序列被截断了 —— 容量估算错了')
        np.testing.assert_array_equal(got, ref)

    def test_reversal_with_gap_one_pushes_nothing(self):
        """反向且 `|gap| == 1` ⇒ **一块都不推**（旧实现 `is_new_brick` 留 False 的分支）。"""
        px = np.array([10.0, 10.4, 10.0, 10.4])
        got = J.bricks_directions(px, 0.5)
        np.testing.assert_array_equal(got, _ref_dirs(px, 0.5))
        self.assertEqual(len(got), 1, '这个形状应当只剩起始那颗')

    def test_matches_on_handmade_and_random_cases(self):
        cases = [('七点', np.array([100., 100., 95., 95., 99., 99., 103.]), 2.0),
                 ('单调涨', np.linspace(10, 40, 50), 0.05),
                 ('来回震荡', np.array([10., 12., 10., 12., 10., 12.]), 0.5)]
        rng = np.random.RandomState(0)
        for i in range(6):
            n = rng.randint(50, 1200)
            cases.append(('随机{}'.format(i), 100 + np.cumsum(rng.randn(n)) * 0.5,
                          float(rng.uniform(0.005, 1.0))))
        for name, px, bs in cases:
            with self.subTest(case=name):
                np.testing.assert_array_equal(J.bricks_directions(np.asarray(px, dtype=float), bs),
                                              _ref_dirs(px, bs))

    def test_zero_brick_raises_like_python(self):
        """`brick_size == 0` 显式抛 `ZeroDivisionError` —— 与纯 Python 的 `float/0.0` 对齐。"""
        with self.assertRaises(ZeroDivisionError):
            J.bricks_directions(np.array([1.0, 2.0]), 0.0)


@unittest.skipUnless(_HAS_NUMBA, 'numba 不可用 —— jit 对拍跳过')
class TestEvaluateMatches(unittest.TestCase):
    """目标函数逐值相同 —— 它差一点，`fminbound` 就会搜到另一个砖高。"""

    def test_bit_identical(self):
        rng = np.random.RandomState(1)
        for i in range(8):
            n = rng.randint(100, 1200)
            bs = float(rng.uniform(0.005, 1.0))
            px = np.ascontiguousarray(100 + np.cumsum(rng.randn(n)) * 0.5)
            with self.subTest(i=i, n=n, brick=bs):
                self.assertEqual(R.evaluate_renko(brick=bs, history=px, column_name='score'),
                                 J.evaluate_renko_jit(brick=bs, history=px, column_name='score'))

    def test_all_columns_of_the_metric_dict(self):
        """四个键都要能取（含那个**带冒号**的 `'sign_changes:'`）。"""
        px = np.ascontiguousarray(np.arange(10., 20., 0.6))
        for key in ('balance', 'sign_changes:', 'price_ratio', 'score'):
            with self.subTest(key=key):
                self.assertEqual(R.evaluate_renko(brick=1.0, history=px, column_name=key),
                                 J.evaluate_renko_jit(brick=1.0, history=px, column_name=key))

    def test_price_ratio_below_one_gives_minus_one(self):
        """砖数多于 bar 数 ⇒ `price_ratio < 1` ⇒ 恒 `-1.0`（两种实现都如此）。"""
        px = np.array([100., 95., 99., 103.])
        self.assertEqual(J.evaluate_renko_jit(2.0, px, 'score'), -1.0)
        self.assertEqual(R.evaluate_renko(brick=2.0, history=px, column_name='score'), -1.0)


@unittest.skipUnless(_HAS_NUMBA, 'numba 不可用 —— jit 对拍跳过')
class TestEndToEndMatches(unittest.TestCase):
    """端到端 14 列逐值相同 —— **这才是那份重复拷贝的唯一验收**。"""

    def _both(self, df):
        return (R.renko_trend_cross_func(df.copy()),
                J.renko_trend_cross_func_jit(df.copy()))

    def test_all_columns_bit_identical(self):
        for n in (200, 420, 1300):
            with self.subTest(n=n):
                a, b = self._both(_ohlc(n))
                self.assertEqual(list(a.columns), list(b.columns))
                self.assertGreater(len(a.columns), 10, '列数不对 —— 夹具没覆盖到')
                for col in a.columns:
                    # float16 是**无损放大**，不是容差（S 族末尾被 cast 过）
                    np.testing.assert_array_equal(
                        a[col].astype(np.float64).values,
                        b[col].astype(np.float64).values,
                        err_msg='列 {} 不一致'.format(col))

    def test_output_is_not_all_empty(self):
        """⚠️ 防"两边都静默早退"造成的**假通过** —— 结果必须真有内容。"""
        _, b = self._both(_ohlc(420))
        for col in (FLD.RENKO_TREND_S, FLD.RENKO_TREND_L, FLD.RENKO_TREND,
                    FLD.RENKO_BOOST_L_TIMING_LAG):
            with self.subTest(col=col):
                self.assertGreater(int((b[col].astype(np.float64) != 0).sum()), 0,
                                   '{} 全零 —— 很可能两边都没真算'.format(col))

    def test_multiwindow_case_is_covered(self):
        """⚠️ >1200 bar 才走**多窗口**分支（`fminbound` 被调两次）—— 别只测单窗口。"""
        a, b = self._both(_ohlc(1300, seed=3))
        self.assertEqual(len(a), 1300)
        opt = b[FLD.RENKO_OPTIMAL].astype(np.float64)
        self.assertGreater(int((opt != 0).sum()), 1,
                           'RENKO_OPTIMAL 只有一处非零 —— 多窗口分支没走到')

    def test_short_input_contract_unchanged(self):
        short = _ohlc(20)
        a, b = self._both(short)
        self.assertEqual(list(a.columns), list(b.columns))
        for col in b.columns:
            with self.subTest(col=col):
                self.assertTrue(b[col].isna().all())


@unittest.skipUnless(_HAS_NUMBA, 'numba 不可用 —— jit 对拍跳过')
class TestSpeedup(unittest.TestCase):
    """**加速要有数字**（用户口径）。这里只做**松**断言：CI 机器抖动不该让用例红。

    实测 **5.0×（1200 bar）/ 6.8×（2059 bar）**，这里只要求 **≥ 2×** ——
    低于它说明 jit 没真的生效（`DISABLE_JIT=1` 时 `renko_jit_status()` 也看得出来）。

    ⚠️ 上限就是 ~6×：剩下的是 pandas 特征装配 + `renko_chart`（本就是 jit，占 0.3%）。
    """

    def test_jit_is_at_least_two_times_faster(self):
        status = J.renko_jit_status()
        if status.get('disable_jit'):
            self.skipTest('DISABLE_JIT 生效 —— 比速度没有意义：{}'.format(status))
        df = _ohlc(1200)
        J.renko_trend_cross_func_jit(df.copy())          # 预热编译

        def bench(fn, k=3):
            best = 1e9
            for _ in range(k):
                t = time.perf_counter()
                fn()
                best = min(best, time.perf_counter() - t)
            return best
        t_py = bench(lambda: R.renko_trend_cross_func(df.copy()))
        t_jit = bench(lambda: J.renko_trend_cross_func_jit(df.copy()))
        self.assertGreater(t_py / max(t_jit, 1e-9), 2.0,
                           '只快 {:.2f}×（py {:.1f}ms / jit {:.1f}ms）'.format(
                               t_py / max(t_jit, 1e-9), t_py * 1000, t_jit * 1000))


class TestStatusWithoutNumba(unittest.TestCase):
    """`renko_jit_status` / `available` 不管 numba 在不在都要能答。"""

    def test_status_shape(self):
        st = J.renko_jit_status()
        self.assertIn('available', st)
        if st['available']:
            self.assertIn('numba', st)
            self.assertIn('disable_jit', st)
        else:
            self.assertIn('reason', st)


if __name__ == '__main__':
    unittest.main(verbosity=2)
