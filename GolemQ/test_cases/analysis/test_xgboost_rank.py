# coding:utf-8
"""`analysis/xgboost/rank.py` 的用例（markup 的 -9..9 评分 rank）。

老树 `portfolio/base.py` 在新环境下**不能原样跑**（pandas 3 的 `.values` 只读 +
`prd` 默认值是 int64，numba 显式签名都不接受）。这里把三处环境差异补上后做
**逐列对拍** —— 补的是「让老代码能跑」，不是改它的算法。
"""
from __future__ import annotations

import os, sys
try:
    import GolemQ
except ImportError:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

import types
import unittest
from unittest import mock

import numpy as np
import pandas as pd

from GolemQ.analysis.indicators import cross_above
from GolemQ.analysis.quadrant_jit import (
    clearup_compact_dataframe_float32 as _cf,
    clearup_compact_dataframe_int16 as _ci)
from GolemQ.analysis.timing import (Timeline_Integral, Timeline_duration,
                                    calc_energy, calc_event_timing_lag)
from GolemQ.analysis.xgboost import markup as M
from GolemQ.analysis.xgboost import rank as R
from GolemQ.core.constants import AKA, FIELD
from GolemQ.models.alias import PFL
from GolemQ.test_cases.old_tree import load_old, old_tree_available

_OLD_NAMES = {'MACD_SX_BF': 'MACD_SX_BF', 'MACD_SX_BF_MAJ': 'MACD_SX_BF_MAJ'}


def _load_old():
    """载入老 `portfolio/base.py`，补掉 pandas3/numba 的三处不兼容。"""
    if not old_tree_available():
        return None
    fake = types.SimpleNamespace(stock_portfolio_stats=object())
    old = load_old('portfolio/base.py', patch_attrs={
        'GolemQ.utils.settings': {'DATABASE': fake},
        'GolemQ.features.base': {
            'clearup_compact_dataframe_float32': _cf,
            'clearup_compact_dataframe_int16': _ci},
        'GolemQ.portfolio.utils': {'calc_fractal_stats': M.calc_fractal_stats},
        'GolemQ.analysis.timeseries': {
            'Timeline_duration': Timeline_duration,
            'Timeline_Integral': Timeline_Integral,
            'calc_event_timing_lag': calc_event_timing_lag,
            'calc_energy': calc_energy},
    })
    # 老模块从 shim 拿到的 TRD 缺这条常量；在模块全局补上。
    old.TRD = types.SimpleNamespace(
        POLYNOMIAL9_CAST_BEFORE_COMPOUNDED='rPoly9CstBfCpd')
    # pandas 3 的 `.values` 只读、`prd=36` 是 int64 —— 老 numba 签名都拒绝。
    opt, neg = old.getWaitAllGreedyOPT, old.getWaitAllGreedyNEG
    old.getWaitAllGreedyOPT = lambda ys, prd=5: opt(
        np.array(ys, copy=True), np.int16(prd))
    old.getWaitAllGreedyNEG = lambda ys, prd=5: neg(
        np.array(ys, copy=True), np.int16(prd))
    return old


try:
    _OLD = _load_old()
    _OLD_ERROR = ''
except Exception as exc:                      # noqa: BLE001 老树加载失败 → 跳过对拍
    _OLD, _OLD_ERROR = None, repr(exc)

_SKIP_OLD = '老树不在场/加载失败，对拍跳过：' + _OLD_ERROR


def _frames(n=600, seed=0):
    idx = pd.MultiIndex.from_arrays(
        [pd.date_range('2026-01-01', periods=n, freq='30min'),
         ['600519'] * n], names=['datetime', 'code'])
    close = 10 + np.cumsum(np.random.default_rng(seed).normal(0, 0.05, n))
    vals = np.random.default_rng(seed + 1).integers(-5, 5, n).astype(np.int16)
    valsm = np.random.default_rng(seed + 2).integers(-5, 5, n).astype(np.int16)
    old = pd.DataFrame({AKA.CLOSE: close, 'MACD_SX_BF': vals,
                        'MACD_SX_BF_MAJ': valsm}, index=idx)
    new = pd.DataFrame({AKA.CLOSE: close,
                        FIELD.MACD_CROSS_SX_BEFORE: vals,
                        FIELD.MACD_CROSS_SX_BEFORE_MAJOR: valsm}, index=idx)
    return old, new


class TestGreedy(unittest.TestCase):

    def test_monotonic_up_is_buy(self):
        out = R.get_wait_all_greedy_opt(np.arange(1.0, 9.0), prd=3)
        self.assertTrue((out[:5] == 1).all())

    def test_neg_is_opt_sign_flipped(self):
        ys = np.array([1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 4.0])
        np.testing.assert_array_equal(
            R.get_wait_all_greedy_neg(ys, prd=2),
            -R.get_wait_all_greedy_opt(ys, prd=2))

    @unittest.skipIf(_OLD is None, _SKIP_OLD)
    def test_matches_old_tree(self):
        ys = np.linspace(1, 100, 200)
        for old_name, new_fn in (('getWaitAllGreedyOPT', R.get_wait_all_greedy_opt),
                                 ('getWaitAllGreedyNEG', R.get_wait_all_greedy_neg)):
            want = getattr(_OLD, old_name)(np.array(ys, copy=True), np.int16(36))
            got = new_fn(ys, 36)
            np.testing.assert_array_equal(got, want, err_msg=old_name)


class TestMacdCross(unittest.TestCase):

    def test_cross_above_fixed_points(self):
        a = pd.Series([1.0, 3.0, 2.0, 4.0])
        b = pd.Series([2.0, 2.0, 3.0, 3.0])
        self.assertEqual(cross_above(a, b).tolist(), [0, 1, 0, 1])

    def test_macd_cross_features_columns_and_major_reset(self):
        n = 400
        idx = pd.MultiIndex.from_arrays(
            [pd.date_range('2026-01-01', periods=n, freq='30min'),
             ['600519'] * n], names=['datetime', 'code'])
        close = 10 + np.sin(np.arange(n) / 7.0) * 2 + np.arange(n) * 0.002
        frame = pd.DataFrame({AKA.CLOSE: close}, index=idx)
        out = R.macd_cross_features(frame)
        self.assertIn(FIELD.MACD_CROSS_SX_BEFORE, out.columns)
        self.assertIn(FIELD.MACD_CROSS_SX_BEFORE_MAJOR, out.columns)
        self.assertTrue(np.issubdtype(out[FIELD.MACD_CROSS_SX_BEFORE].dtype,
                                      np.integer))
        before = out[FIELD.MACD_CROSS_SX_BEFORE]
        major = out[FIELD.MACD_CROSS_SX_BEFORE_MAJOR]
        resets = before.diff(1) < 0
        self.assertTrue(resets.any(), '正弦序列里应当有死叉')
        # `Timeline_duration` 在重置点写 0，之后逐 bar +1。
        self.assertTrue((major[resets] == 0).all())

    def test_missing_close_raises(self):
        with self.assertRaises(ValueError):
            R.macd_cross_features(pd.DataFrame({'open': [1.0, 2.0]}))


class TestRankFrom83(unittest.TestCase):

    def _frame(self, n=600):
        idx = pd.MultiIndex.from_arrays(
            [pd.date_range('2026-01-01', periods=n, freq='30min'),
             ['600519'] * n], names=['datetime', 'code'])
        close = 10 + np.cumsum(np.random.default_rng(0).normal(0, 0.05, n))
        return pd.DataFrame({AKA.CLOSE: close}, index=idx)

    def test_reads_83_and_computes_rank(self):
        frame = self._frame()
        fake = mock.Mock()
        fake.get_kline_price_min.return_value = (
            types.SimpleNamespace(data=frame), '贵州茅台')
        with mock.patch('GolemQ.core.market_registry.get_active_market',
                        return_value=fake):
            out = R.rank_from_83('600519')
        self.assertIsNotNone(out)
        self.assertIn(PFL.STOCK_PORTFOLIO_RANK, out.columns)
        self.assertEqual(len(out), len(frame))
        kwargs = fake.get_kline_price_min.call_args.kwargs
        self.assertEqual(kwargs['frequency'], '30min')
        self.assertFalse(kwargs['realtime'])

    def test_empty_returns_none(self):
        fake = mock.Mock()
        fake.get_kline_price_min.return_value = (
            types.SimpleNamespace(data=pd.DataFrame()), '600519')
        with mock.patch('GolemQ.core.market_registry.get_active_market',
                        return_value=fake):
            self.assertIsNone(R.rank_from_83('600519'))


class TestPortfolioRank(unittest.TestCase):

    @unittest.skipIf(_OLD is None, _SKIP_OLD)
    def test_matches_old_tree(self):
        old_frame, new_frame = _frames()
        want = _OLD.calc_portfolio_returns(old_frame)
        got = R.calc_portfolio_rank(new_frame)
        for col in (PFL.STOCK_PORTFOLIO_RANK, PFL.STOCK_PORTFOLIO_RETURNS,
                    PFL.STOCK_PORTFOLIO_RANK_NEGATIVE,
                    PFL.STOCK_PORTFOLIO_RANK_BEFORE,
                    PFL.STOCK_PORTFOLIO_RANK_TRACING,
                    PFL.STOCK_PORTFOLIO_RANK_METAPHASE,
                    PFL.STOCK_PORTFOLIO_RANK_METAPHASE_MAJOR):
            pd.testing.assert_series_equal(got[col], want[col], check_names=True)

    def test_does_not_mutate_input(self):
        _, new_frame = _frames()
        before = list(new_frame.columns)
        R.calc_portfolio_rank(new_frame)
        self.assertEqual(list(new_frame.columns), before)

    def test_missing_close_raises(self):
        _, new_frame = _frames()
        new_frame = new_frame.drop(columns=[AKA.CLOSE])
        with self.assertRaises(ValueError):
            R.calc_portfolio_rank(new_frame)

    def test_missing_macd_raises(self):
        _, new_frame = _frames()
        new_frame = new_frame.drop(columns=[FIELD.MACD_CROSS_SX_BEFORE_MAJOR])
        with self.assertRaises(ValueError):
            R.calc_portfolio_rank(new_frame)

    @unittest.skipIf(_OLD is None, _SKIP_OLD)
    def test_metaphase_matches_old_tree(self):
        rng = np.random.default_rng(9)
        n = 200
        rank_lag = rng.integers(0, 12, n).astype(np.int32)
        sx = rng.integers(-5, 6, n).astype(np.int16)
        sxm = rng.integers(-5, 6, n).astype(np.int16)
        np.testing.assert_array_equal(
            R.calc_portfolio_metaphase(rank_lag, sx, sxm),
            _OLD.calc_portfolio_metaphase(rank_lag, sx, sxm))


if __name__ == '__main__':
    unittest.main()
