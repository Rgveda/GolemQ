# coding:utf-8
"""`analysis/quadrant_coef.py` —— 上升系数（`QUprICoefD`）的**解耦编排**。

老树实际消费的链是 `models/mainstream.py`（CANDIDATE）+ `analysis/zen_pivot.py`
（DUMMY = CANDIDATE × log10(有效 lag)）；`models/fourquad.py:717` 那条
（DUMMY = 参照序列稳健回归）是**另一列**，不是用户看到的 `OmniPathUpR1CoefDmy`
（2026-10-19 实查更正）。

用例钉四件事：

1. **与老树 `analysis/pool_coef.py` 逐值对拍**（随机输入，rtol 1e-12）；
2. 逐 bar 的**窗口切片**口径（长度取不到 ⇒ `NaN`，不臆造）；
3. `DUMMY = CANDIDATE × log10(有效 lag)`（含夹到 10 的两条分支）；
4. 两列都写进 `features`（`QUprICoefCan` / `QUprICoefD`）。
"""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from GolemQ.analysis import quadrant_coef as QC
from GolemQ.core.constants import AKA, FIELD
from GolemQ.models.alias import LTT
from GolemQ.test_cases.old_tree import load_old, old_tree_available


def _old():
    """载入老树 `analysis/pool_coef.py`（不在场返回 `None`）。"""
    if not old_tree_available():
        return None
    try:
        return load_old('analysis/pool_coef.py')
    except Exception:                   # noqa: BLE001 老树不可载入时整份跳过
        return None


class TestPoolRateMatchesOld(unittest.TestCase):
    """`get_pool_rate` / `get_pool_rate_vX` —— **与老树逐值对拍**。"""

    @classmethod
    def setUpClass(cls):
        cls.old = _old()
        if cls.old is None:
            raise unittest.SkipTest('老树 GolemQ_old/analysis/pool_coef.py 载入不了')

    def test_pool_rate(self):
        rng = np.random.default_rng(7)
        for _ in range(30):
            x = np.sort(rng.normal(size=40)) + 10
            a, b = float(self.old.getPoolRate(x)), QC.get_pool_rate(x)
            self.assertTrue(np.isclose(a, b, rtol=1e-12, atol=1e-12),
                            msg=f'{a} vs {b}')

    def test_pool_rate_vX(self):
        rng = np.random.default_rng(11)
        for _ in range(30):
            x = np.sort(rng.normal(size=40)) + 10
            r = np.sort(rng.normal(size=40)) + 10
            a, b = float(self.old.getPoolRate_vX(x, r)), QC.get_pool_rate_vX(x, r)
            self.assertTrue(np.isclose(a, b, rtol=1e-12, atol=1e-12),
                            msg=f'{a} vs {b}')

    def test_peak_high(self):
        rng = np.random.default_rng(3)
        for _ in range(10):
            x = rng.normal(size=25)
            np.testing.assert_array_equal(self.old.get_peak_high_price(x),
                                          QC.get_peak_high_price(x))


class TestWavletCoefWindow(unittest.TestCase):
    """逐 bar 的窗口切片口径（不依赖老树）。"""

    @staticmethod
    def _frame(lens, close):
        idx = pd.MultiIndex.from_arrays(
            [pd.bdate_range('2026-01-05', periods=len(close)), ['600519'] * len(close)],
            names=['ts', 'code'])
        return pd.DataFrame({'LEN': lens, AKA.CLOSE: close}, index=idx)

    def test_length_nan_gives_nan(self):
        f = self._frame([np.nan] * 5, np.arange(5, dtype=float) + 10)
        cand, dummy = QC.calc_wavlet_coef_of_fractal(f, 'LEN')
        self.assertTrue(np.isnan(cand).all())
        self.assertTrue(np.isnan(dummy).all())

    def test_length_zero_is_skipped(self):
        """长度 0 ⇒ `start == i` ⇒ 不算（老树同口径），不是"除零"。"""
        f = self._frame([0] * 5, np.arange(5, dtype=float) + 10)
        cand, _ = QC.calc_wavlet_coef_of_fractal(f, 'LEN')
        self.assertTrue(np.isnan(cand).all())

    def test_window_matches_direct_call(self):
        """第 i 根的候选值 == 直接对 `[i-L, i]` 这段价格调 `get_pool_rate`。"""
        close = np.array([10.0, 11.0, 10.5, 12.0, 13.0, 12.5, 14.0])
        lens = [np.nan, np.nan, 1, 2, 3, 2, 4]
        f = self._frame(lens, close)
        cand, _ = QC.calc_wavlet_coef_of_fractal(f, 'LEN')
        for i, L in enumerate(lens):
            if np.isnan(L):
                self.assertTrue(np.isnan(cand[i]))
                continue
            L = int(L)
            want = round(QC.get_pool_rate(close[i - L:i + 1]), 4)
            self.assertAlmostEqual(cand[i], want, places=12, msg=f'bar {i}')


class TestWritesBothColumns(unittest.TestCase):
    def test_calc_uprising_i_coef_writes_candidate_and_dummy(self):
        n = 60
        idx = pd.MultiIndex.from_arrays(
            [pd.bdate_range('2026-01-05', periods=n), ['600519'] * n],
            names=['ts', 'code'])
        close = np.linspace(10.0, 15.0, n)
        f = pd.DataFrame({
            # 分形长度 = **MACD lag**（老树 `mainstream.py:252-261`）
            LTT.QUADRANT_LEVERAGE_MACD_TIMING_LAG_DUMMY: [np.nan] * 30 + [3.0] * (n - 30),
            AKA.CLOSE: close,
        }, index=idx)
        out = QC.calc_uprising_i_coef(f)
        self.assertIn(LTT.QUADRANT_UPRISING_I_COEF_CANDIDATE, out.columns)
        self.assertIn(LTT.QUADRANT_UPRISING_I_COEF_DUMMY, out.columns)
        self.assertGreater(int(out[LTT.QUADRANT_UPRISING_I_COEF_DUMMY].notna().sum()), 0)


class TestDummyFormula(unittest.TestCase):
    """`DUMMY = CANDIDATE × log10(有效 lag)` —— 老树 `zen_pivot.py:276-284` 逐字。

    2026-10-19 实测（301308 现跑老树）：1904/1904 行逐位相等，最大差 `8.9e-16`。
    上一版把 DUMMY 接成 `get_pool_rate_vX`（`fourquad.py:717` 的返回值）——
    那是**另一列**（`OMNIPATH_UPRISING_I_COEF`），不是用户看到的 `OmniPathUpR1CoefDmy`。
    """

    @staticmethod
    def _frame(n=90, lag=89.0):
        idx = pd.MultiIndex.from_arrays(
            [pd.bdate_range('2025-01-06', periods=n), ['600519'] * n],
            names=['ts', 'code'])
        close = 10 + np.cumsum(np.random.default_rng(5).normal(0, 0.2, n))
        return pd.DataFrame({
            LTT.QUADRANT_LEVERAGE_MACD_TIMING_LAG_DUMMY: [np.nan] * 10 + [lag] * (n - 10),
            AKA.CLOSE: close,
        }, index=idx)

    def test_dummy_is_candidate_times_log10_lag(self):
        f = self._frame()
        out = QC.calc_uprising_i_coef(f.copy())
        cand = out[LTT.QUADRANT_UPRISING_I_COEF_CANDIDATE].to_numpy()
        got = out[LTT.QUADRANT_UPRISING_I_COEF_DUMMY].to_numpy()
        want = cand * np.log10(89.0)
        np.testing.assert_allclose(got, want, rtol=0, atol=1e-12, equal_nan=True)

    def test_lag_at_or_below_10_is_clamped_to_10(self):
        for lag in (10.0, 4.0, 1.0):
            with self.subTest(lag=lag):
                out = QC.calc_uprising_i_coef(self._frame(lag=lag))
                cand = out[LTT.QUADRANT_UPRISING_I_COEF_CANDIDATE].to_numpy()
                got = out[LTT.QUADRANT_UPRISING_I_COEF_DUMMY].to_numpy()
                # log10(10) == 1 ⇒ DUMMY 就等于 CANDIDATE
                np.testing.assert_allclose(got, cand, rtol=0, atol=1e-12, equal_nan=True)

    def test_clamp_columns_override_when_present(self):
        """条件列在场时按老树原式：条件不成立 ⇒ 夹到 10（即使 lag > 10）。"""
        f = self._frame(lag=89.0)
        f[LTT.QUADRANT_ZEN_DASH_COUNT] = 0.0
        f[LTT.QUADRANT_LEVERAGE_POWERLINE_UPRISING_TIMING_LAG] = 0.0
        out = QC.calc_uprising_i_coef(f)
        cand = out[LTT.QUADRANT_UPRISING_I_COEF_CANDIDATE].to_numpy()
        np.testing.assert_allclose(out[LTT.QUADRANT_UPRISING_I_COEF_DUMMY].to_numpy(),
                                   cand * 1.0, rtol=0, atol=1e-12, equal_nan=True)
        # 条件成立 ⇒ 用 lag
        f[LTT.QUADRANT_LEVERAGE_POWERLINE_UPRISING_TIMING_LAG] = 3.0
        out = QC.calc_uprising_i_coef(f)
        cand = out[LTT.QUADRANT_UPRISING_I_COEF_CANDIDATE].to_numpy()
        np.testing.assert_allclose(out[LTT.QUADRANT_UPRISING_I_COEF_DUMMY].to_numpy(),
                                   cand * np.log10(89.0), rtol=0, atol=1e-12, equal_nan=True)


class TestReferenceSeriesPreference(unittest.TestCase):
    """⚠️ `calc_wavlet_coef_of_fractal` 的第二返回值（老树 `COEF`）**优先
    `POLYNOMIAL9_WEEKLY`**，没有才退到 `LINEAREG_PRICE`。

    老树 `analysis/pool_coef.py:299` 的 `if FLD.POLYNOMIAL9_WEEKLY in target_features`。
    ⚠️ **这只影响 `COEF`，不影响 `QUprICoefD`** —— DUMMY 是乘积式，与参照序列无关。
    """

    @staticmethod
    def _frame(n=80):
        idx = pd.MultiIndex.from_arrays(
            [pd.bdate_range('2025-01-06', periods=n), ['600519'] * n],
            names=['ts', 'code'])
        close = np.linspace(10.0, 20.0, n)
        return pd.DataFrame({
            LTT.QUADRANT_LEVERAGE_MACD_TIMING_LAG_DUMMY: [np.nan] * 40 + [3.0] * (n - 40),
            AKA.CLOSE: close,
            FIELD.POLYNOMIAL9_WEEKLY: np.linspace(9.0, 19.0, n),   # 明显的另一条序列
        }, index=idx)

    def _coef(self, f, regression=None):
        return QC.calc_wavlet_coef_of_fractal(
            f, column_name=LTT.QUADRANT_LEVERAGE_MACD_TIMING_LAG_DUMMY,
            regression=regression)[1]

    def test_weekly_poly9_is_used_when_present(self):
        f = self._frame()
        implicit = self._coef(f.copy())
        explicit = self._coef(f.copy(), regression=f[FIELD.POLYNOMIAL9_WEEKLY].to_numpy())
        np.testing.assert_array_equal(implicit, explicit)

    def test_falls_back_to_lineareg_without_it(self):
        f = self._frame().drop(columns=[FIELD.POLYNOMIAL9_WEEKLY])
        got = self._coef(f.copy())
        want = self._coef(f.copy(), regression=QC.calc_lineareg_price(f[AKA.CLOSE].to_numpy()))
        np.testing.assert_array_equal(got, want)


if __name__ == '__main__':
    unittest.main()


if __name__ == '__main__':
    unittest.main()
