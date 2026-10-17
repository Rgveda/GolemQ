# coding:utf-8
"""`analysis/magic_nine.py` 的**逐值对拍** + 它的 jit 版一致性。

判据一律 `np.array_equal(..., equal_nan=True)` —— **不用容差**（`DECISIONS.md` D35/D36）。
老树 `GolemQ_old/` 不在场时，对拍用例**整份跳过**（同 `test_timing.py` 惯例）。

harness 见 `old_tree.py`
====================
被测的 `calc_magic_nine_turns_np_vXs` **只依赖 numpy**（不碰 talib / QUANTAXIS /
任何 `GolemQ.*`），所以老 `features/base.py` 那一大串顶层 import **只需解析成功** ——
交给 `load_old` 的替身查找器即可，不必喂真 `LTT`/`TRD`（那是 `test_quadrant` 的负担）。
"""
from __future__ import annotations

import os
import sys

try:
    import GolemQ  # noqa: F401
except ImportError:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import unittest

import numpy as np

from GolemQ.analysis import magic_nine as NEW
from GolemQ.test_cases.old_tree import load_old
from GolemQ.analysis.magic_nine_jit import (
    available as jit_available,
    calc_magic_nine_turns_np_vXs_jit,
    magic_nine_jit_status,
)

#: harness 抽在 `old_tree.py`（三处对拍共用）。
#: ⚠️ `GolemQ.analysis.timeseries` 要**顶替**：老 `features/base.py` 从它拿 7 个名字
#: （`Timeline_Integral(s)` / `thresholding_algo` / `lineareg_intercept` …），
#: 新树那个模块只有 `rolling_poly9_v2`。
#: 被测的 `calc_magic_nine_turns_np_vXs` **只用 numpy**，一个都不用 —— 所以顶替安全。
_OLD = load_old('features/base.py', override=('GolemQ.analysis.timeseries',))


def _prices(n, seed, drift=0.0):
    rng = np.random.default_rng(seed)
    return (np.cumsum(rng.normal(drift, 1.0, n)) + 50.0).astype(np.float32)


@unittest.skipIf(_OLD is None, '老树 GolemQ_old/ 不在场，对拍整份跳过')
class TestMatchesOld(unittest.TestCase):
    """与老树 `calc_magic_nine_turns_np_vXs` 逐值相同。"""

    def _cmp(self, px):
        old = _OLD.calc_magic_nine_turns_np_vXs(px)
        new = NEW.calc_magic_nine_turns_np_vXs(px)
        self.assertEqual(len(new), len(old))
        self.assertTrue(np.array_equal(np.asarray(new, dtype=np.float32),
                                       np.asarray(old, dtype=np.float32),
                                       equal_nan=True))

    def test_random_walks(self):
        for seed in range(6):
            for n in (5, 40, 300, 1500):
                with self.subTest(seed=seed, n=n):
                    self._cmp(_prices(n, seed))

    def test_strong_uptrend_hits_nine(self):
        """单调上涨 ⇒ 必然走满 9 天 ⇒ 输出里必须有 `1`。"""
        px = np.arange(200, dtype=np.float32) + 10.0
        got = NEW.calc_magic_nine_turns_np_vXs(px)
        self.assertIn(1.0, got.tolist())

    def test_strong_downtrend_hits_nine(self):
        px = 200.0 - np.arange(200, dtype=np.float32)
        got = NEW.calc_magic_nine_turns_np_vXs(px)
        self.assertIn(-1.0, got.tolist())

    def test_flat_series(self):
        """全平 ⇒ 走 `i == _list_t` 分支。"""
        px = np.full(60, 3.5, dtype=np.float32)
        self._cmp(px)

    def test_sample_paths(self):
        for px in (_prices(300, 99),
                   np.array([1, 2, 3, 4, 5, 4, 3, 2, 1, 5], dtype=np.float32)):
            with self.subTest(px=px[:3]):
                self._cmp(px)


class TestJitMatchesPure(unittest.TestCase):
    """jit 版必须与纯 Python 版**逐值一致**（`DECISIONS.md` D36）。"""

    def test_status_shape(self):
        st = magic_nine_jit_status()
        self.assertIn('available', st)

    @unittest.skipUnless(jit_available(), 'numba 不在场')
    def test_jit_equals_pure(self):
        for seed in range(8):
            for n in (5, 40, 300, 1500):
                with self.subTest(seed=seed, n=n):
                    px = _prices(n, seed)
                    a = NEW.calc_magic_nine_turns_np_vXs(px)
                    b = calc_magic_nine_turns_np_vXs_jit(px)
                    self.assertTrue(np.array_equal(np.asarray(a, dtype=np.float32),
                                                   np.asarray(b, dtype=np.float32),
                                                   equal_nan=True))

    @unittest.skipUnless(jit_available(), 'numba 不在场')
    def test_jit_accepts_non_float32_input(self):
        """入参不是 float32 时两版都要先转 —— 否则会在最低位上分叉。"""
        px = np.cumsum(np.random.default_rng(3).normal(0, 1, 200)) + 50.0  # float64
        a = NEW.calc_magic_nine_turns_np_vXs(px)
        b = calc_magic_nine_turns_np_vXs_jit(px)
        self.assertTrue(np.array_equal(np.asarray(a, dtype=np.float32),
                                       np.asarray(b, dtype=np.float32),
                                       equal_nan=True))


class TestShape(unittest.TestCase):
    """结构性钉子（不依赖老树）。"""

    def test_length_matches_input(self):
        for n in (1, 2, 3, 4, 5, 9, 10, 100, 999):
            with self.subTest(n=n):
                self.assertEqual(len(NEW.calc_magic_nine_turns_np_vXs(_prices(n, 1))), n)

    def test_dtype_is_float32(self):
        self.assertEqual(NEW.calc_magic_nine_turns_np_vXs(_prices(50, 1)).dtype,
                         np.dtype('float32'))

    def test_values_are_in_known_set(self):
        """取值只可能是 {1, -1, 2, -2, NaN} —— 出现别的就是算法走岔了。"""
        got = NEW.calc_magic_nine_turns_np_vXs(_prices(400, 7))
        uniq = {v for v in got.tolist() if v == v}
        self.assertTrue(uniq <= {1.0, -1.0, 2.0, -2.0}, uniq)


if __name__ == '__main__':
    unittest.main()
