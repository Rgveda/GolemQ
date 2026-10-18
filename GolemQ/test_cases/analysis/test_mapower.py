# coding:utf-8
"""`analysis/quadrant.py` 的 `mapower` 一族 —— 与老树**逐值对拍** + jit 一致性。

判据 `np.array_equal`，**不用容差**（`DECISIONS.md` D35/D36）。
老树 `GolemQ_old/` 不在场时对拍整份跳过（harness 见 `old_tree.py`）。

⚠️ 这一族**没有随机性**（不像 `calc_garident_price` 的 KMeans），不需要对齐种子。

⚠️ 老树下**有多个孪生**：`inv_num`（int32/f8 版，`base.py:1808`）、
`mapower_np_vXII`（f8 版，`:1878`）。按「多副本只搬规范版」**只搬 `_vXIIs`**
（float32，`reviews.py:1606` 的主路径在用的那份）—— 所以对拍也只对它。
"""
from __future__ import annotations

import os
import sys

try:
    import GolemQ  # noqa: F401
except ImportError:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

import unittest

import numpy as np

from GolemQ.analysis.quadrant import inv_nums, mapower_np_vXIIs
from GolemQ.analysis.quadrant_jit import (
    available as jit_available,
    inv_nums_jit,
    mapower_np_vXIIs_jit,
    quadrant_jit_status,
)
from GolemQ.test_cases.old_tree import load_old

#: ⚠️ 老 `features/base.py` 从 `GolemQ.analysis.timeseries` 拿 7 个名字，
#: 新树那个模块只有 `rolling_poly9_v2` ⇒ 必须顶替。
#: 被测的两个函数**只用 numpy**，顶替安全（`test_magic_nine` 同款）。
_OLD = load_old('features/base.py', override=('GolemQ.analysis.timeseries',))


def _px(n, seed=3):
    rng = np.random.default_rng(seed)
    return (np.cumsum(rng.normal(0, 1.0, n)) + 50.0).astype(np.float32)


@unittest.skipIf(_OLD is None, '老树 GolemQ_old/ 不在场，对拍整份跳过')
class TestMatchesOld(unittest.TestCase):
    """与老树 `inv_nums` / `mapower_np_vXIIs` 逐值相同。"""

    def test_inv_nums_small_cases(self):
        cases = [
            np.array([3, 1, 2], dtype=np.float32),
            np.array([1, 2, 3], dtype=np.float32),
            np.array([3, 2, 1], dtype=np.float32),
            np.array([1.0] * 10, dtype=np.float32),
            np.array([], dtype=np.float32),
            np.array([2.5], dtype=np.float32),
        ]
        for i, arr in enumerate(cases):
            with self.subTest(case=i):
                self.assertEqual(inv_nums(arr), int(_OLD.inv_nums(arr)))

    def test_mapower_random_walks(self):
        for seed in range(5):
            for n in (40, 300, 900):
                with self.subTest(seed=seed, n=n):
                    px = _px(n, seed)
                    new = mapower_np_vXIIs(px)
                    old = _OLD.mapower_np_vXIIs(
                        px, np.array(range(5, 30)).astype(np.int16))
                    self.assertEqual(len(new), len(old))
                    self.assertTrue(np.array_equal(np.asarray(new, dtype=np.float32),
                                                   np.asarray(old, dtype=np.float32)))

    def test_mapower_custom_range_list(self):
        """非默认窗口列表也要一致（老树这个形参是可传的）。"""
        px = _px(200, 9)
        rl = np.array([5, 10, 20], dtype=np.int16)
        self.assertTrue(np.array_equal(mapower_np_vXIIs(px, rl),
                                       _OLD.mapower_np_vXIIs(px, rl)))

    def test_mapower_monotone_series(self):
        """严格下行 ⇒ 短均线恒高于长均线 ⇒ 逆序数为 0 ⇒ 末值 0。"""
        px = (200.0 - np.arange(120, dtype=np.float32))
        self.assertTrue(np.array_equal(mapower_np_vXIIs(px, None),
                                       _OLD.mapower_np_vXIIs(
                                           px, np.array(range(5, 30)).astype(np.int16))))


class TestJitMatchesPure(unittest.TestCase):
    """jit 版必须与纯 Python 版**逐值一致**（`DECISIONS.md` D36）。"""

    def test_status_shape(self):
        self.assertIn('available', quadrant_jit_status())

    @unittest.skipUnless(jit_available(), 'numba 不在场')
    def test_mapower_jit_equals_pure(self):
        for seed in range(6):
            for n in (40, 300, 900, 2000):
                with self.subTest(seed=seed, n=n):
                    px = _px(n, seed)
                    self.assertTrue(np.array_equal(
                        np.asarray(mapower_np_vXIIs(px), dtype=np.float32),
                        np.asarray(mapower_np_vXIIs_jit(px), dtype=np.float32)))

    @unittest.skipUnless(jit_available(), 'numba 不在场')
    def test_inv_nums_jit_equals_pure(self):
        rng = np.random.default_rng(11)
        for n in (1, 5, 25):
            arr = rng.normal(0, 1, n).astype(np.float32)
            with self.subTest(n=n):
                self.assertEqual(int(inv_nums(arr)), int(inv_nums_jit(arr)))

    @unittest.skipUnless(jit_available(), 'numba 不在场')
    def test_jit_accepts_float64_input(self):
        """入参不是 float32 时两版都要先转 —— 否则会在最低位上分叉。"""
        px = np.cumsum(np.random.default_rng(2).normal(0, 1, 200)) + 50.0  # float64
        self.assertTrue(np.array_equal(
            np.asarray(mapower_np_vXIIs(px), dtype=np.float32),
            np.asarray(mapower_np_vXIIs_jit(px), dtype=np.float32)))


class TestContract(unittest.TestCase):
    """不依赖老树的结构性钉子。"""

    def test_output_shape_and_range(self):
        y = mapower_np_vXIIs(_px(400, 5))
        self.assertEqual(len(y), 400)
        self.assertEqual(y.dtype, np.dtype('float32'))
        self.assertGreaterEqual(float(y.min()), 0.0)
        self.assertLessEqual(float(y.max()), 1.0)

    def test_warmup_prefix_is_zero_not_nan(self):
        """前 `max(range_list)-1` 个是 **0 不是 NaN** —— 下游别当"无数据"去 drop。"""
        y = mapower_np_vXIIs(_px(400, 5))
        self.assertTrue((y[:28] == 0).all())
        self.assertFalse(np.isnan(y).any())

    def test_perfect_bullish_alignment_is_one(self):
        """严格上行 ⇒ 完美多头排列 ⇒ 1.0。"""
        y = mapower_np_vXIIs(np.arange(60, dtype=np.float32) + 1.0)
        self.assertAlmostEqual(float(y[-1]), 1.0, places=6)

    def test_perfect_bearish_alignment_is_zero(self):
        y = mapower_np_vXIIs((60.0 - np.arange(60, dtype=np.float32)))
        self.assertAlmostEqual(float(y[-1]), 0.0, places=6)


if __name__ == '__main__':
    unittest.main()
