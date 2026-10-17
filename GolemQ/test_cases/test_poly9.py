# coding:utf-8
"""`analysis/timeseries.py` 的 `rolling_poly9_v2` —— 与老树的**逐值对拍**。

判据 `np.array_equal` —— **不用容差**（`DECISIONS.md` D35/D36）。
老树 `GolemQ_old/` 不在场时整份跳过（harness 见 `old_tree.py`）。

这一族**没有随机性**（不像 `calc_garident_price` 的 KMeans），所以不需要对齐种子 ——
同一输入两次调用必然同结果。`w=252` 时它会**递归折半**（252 → 126 → 63），
三段拼起来恰好等于输入长度，这条边界要覆盖到。
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

from GolemQ.analysis.timeseries import rolling_poly9_v2
# ⚠️ **绝对导入**（不是 `from .old_tree`）：`run_tests.py` 按**顶层模块**
# 发现用例，相对导入会 `Attempted relative import with no known parent package`。
from GolemQ.test_cases.old_tree import load_old, old_tree_available

_OLD = load_old('analysis/timeseries.py')


def _px(n, seed=3):
    """一段**确定**的伪价格（用固定种子的 Generator ⇒ 每次一样）。"""
    rng = np.random.default_rng(seed)
    return np.cumsum(rng.normal(0, 0.8, n)) + 100.0


@unittest.skipIf(_OLD is None, '老树 GolemQ_old/ 不在场，对拍整份跳过')
class TestMatchesOld(unittest.TestCase):
    """与老树 `rolling_poly9_v2` 逐值相同。"""

    def _cmp(self, px, w):
        old = _OLD.rolling_poly9_v2(px, w)
        new = rolling_poly9_v2(px, w)
        self.assertEqual(len(new), len(old), '长度必须一致')
        self.assertTrue(np.array_equal(np.asarray(new, dtype=float),
                                       np.asarray(old, dtype=float),
                                       equal_nan=True))

    def test_window_sweep(self):
        px = _px(900)
        for w in (252, 126, 63, 20, 400):
            with self.subTest(w=w):
                self._cmp(px, w)

    def test_length_sweep(self):
        """长度跨过 `w`、`w/2` 两条边界（递归折半的两个分支）。"""
        for n in (10, 30, 63, 64, 65, 125, 126, 127, 251, 252, 253, 600, 1500):
            with self.subTest(n=n):
                self._cmp(_px(n, seed=n % 7), 252)

    def test_short_and_flat(self):
        for px in (np.arange(30, dtype=float), np.full(50, 7.5)):
            with self.subTest(shape=px.shape):
                self._cmp(px, 252)

    def test_empty_input_raises_on_both_sides(self):
        """⚠️ **空序列两边都不支持**（`Chebyshev.fit` 不吃空数组）。

        这不是"本树的分歧" —— 老树同一句代码抛同一个错。钉在这里是为了说明
        **别指望它返回空数组**：上游给空序列就是上游的问题。
        """
        with self.assertRaises(ValueError):
            _OLD.rolling_poly9_v2(np.array([], dtype=float), 252)
        with self.assertRaises(ValueError):
            rolling_poly9_v2(np.array([], dtype=float), 252)


class TestContract(unittest.TestCase):
    """不依赖老树的结构性钉子。"""

    def test_output_length_equals_input(self):
        for n in (5, 30, 200, 900, 1500):
            with self.subTest(n=n):
                self.assertEqual(len(rolling_poly9_v2(_px(n), 252)), n)

    def test_linear_series_is_reproduced_exactly(self):
        """线性序列上九次拟合是精确的 ⇒ 端点值就是原值（`w` 大于序列长时）。"""
        y = rolling_poly9_v2(np.arange(30, dtype=float), 20)
        self.assertAlmostEqual(float(y[-1]), 29.0, places=12)

    def test_no_nan_when_window_halves(self):
        """`w=252` 作用在 900 长序列上要走递归折半 —— **不许留 NaN**。"""
        y = np.asarray(rolling_poly9_v2(_px(900), 252), dtype=float)
        self.assertEqual(len(y), 900)
        self.assertFalse(np.isnan(y).any(), '折半补前缀那段必须也填满')


if __name__ == '__main__':
    unittest.main()
