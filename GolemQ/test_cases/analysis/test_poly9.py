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
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

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

    def test_rank_warning_is_silenced(self):
        """平盘窗口的 `RankWarning` 是预期的，不许刷屏（`_cheb_fit` 压掉）。"""
        import warnings
        from GolemQ.analysis.timeseries import _RankWarning
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            out = rolling_poly9_v2(np.ones(600), 252)
        self.assertFalse([w for w in caught
                          if issubclass(w.category, _RankWarning)])
        self.assertEqual(len(out), 600)

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


class TestWeeklyPoly9(unittest.TestCase):
    """`calc_polynomial9_weekly_vXIs` —— 周线 `POLYNOMIAL9`（`poly9Wek`）。

    它是 `quadrant_coef.calc_wavlet_coef_of_fractal` 的**首选参照序列**
    （老树 `pool_coef.py:299` 的优先分支）—— 选错会让上升系数与老树**几乎不相关**。
    """

    @staticmethod
    def _ohlc(n=800, code='600519'):
        import pandas as pd
        idx = pd.MultiIndex.from_arrays(
            [pd.bdate_range('2020-01-01', periods=n), [code] * n],
            names=['ts', 'code'])
        close = 10 + np.cumsum(np.random.default_rng(0).normal(0, 0.1, n)) \
            + np.arange(n) * 0.01
        return pd.DataFrame({'open': close, 'high': close * 1.01,
                             'low': close * 0.99, 'close': close,
                             'volume': np.full(n, 1e4)}, index=idx)

    def test_aligned_and_non_null(self):
        from GolemQ.analysis.poly9 import calc_polynomial9_weekly_vXIs
        ohlc = self._ohlc()
        got = calc_polynomial9_weekly_vXIs(ohlc)
        self.assertEqual(len(got), len(ohlc), '必须与输入等长（ffill 回填）')
        self.assertTrue(got.index.equals(ohlc.index))
        self.assertGreater(int(got.notna().sum()), 0)

    def test_matches_poly9_of_the_weekly_frame(self):
        """口径 = **周线帧上的 POLYNOMIAL9**（老树 `features_week[POLYNOMIAL9_WEEKLY]`）。"""
        import pandas as pd
        from GolemQ.analysis.poly9 import (calc_polynomial9_vXIs,
                                           calc_polynomial9_weekly_vXIs)
        from GolemQ.core.constants import AKA, FIELD as FLD
        ohlc = self._ohlc()
        got = calc_polynomial9_weekly_vXIs(ohlc)

        d = ohlc.copy()
        d.index = pd.to_datetime(d.index.get_level_values(0))
        w = d.resample('W-FRI').agg({'open': 'first', 'high': 'max', 'low': 'min',
                                     'close': 'last', 'volume': 'sum'}).dropna()
        w.index = pd.MultiIndex.from_arrays([w.index, ['600519'] * len(w)],
                                            names=['ts', 'code'])
        from GolemQ.analysis.indicators import TA_HMA
        wf = pd.DataFrame(index=w.index)
        wf[FLD.HMA10] = TA_HMA(pd.to_numeric(w[AKA.CLOSE], errors='coerce'), 10)
        direct = calc_polynomial9_vXIs(w, features=wf, annual=52)[FLD.POLYNOMIAL9]

        # 末根对齐：周线最后一根 ffill 后应等于主帧末值
        self.assertAlmostEqual(float(got.iloc[-1]), float(direct.iloc[-1]), places=6)
