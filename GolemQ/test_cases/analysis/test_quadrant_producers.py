# coding:utf-8
"""上升系数夹取链 / 第二 OR 分支的**最小生产者**用例（2026-10-19）。

老树那条链要 `4QuadZenDCnt` / `QLevPwrlnUprLag` / `MsLeadinx5BfMin`；
本文件钉住我们按「只搬表达式」解耦后的实现：

* `calc_zen_dash_timing_lag` —— **只由 CZSC 笔方向**决定（不碰 ZEN 小波族）；
* `calc_quadrant_zen_dash_count` —— 一条 `calc_energy`；
* `calc_bs_star_down` —— 只搬星线死叉段（不搬整条 BS 链）；
* `calc_quadrant_stardown_count` —— 一条 `calc_energy`；
* `calc_quadrant_powerline_uprisal_lag` —— 老树的**两条**连续赋值；
* `calc_quadrant_leverage_macd_timing_lag` 的第二 OR 分支 —— 三列在场才启用；
* `calc_macd_qa` —— QUANTAXIS 口径（`adjust=True`、柱 ×2），不是 talib。
"""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from GolemQ.analysis import quadrant_jit as Q
from GolemQ.analysis.indicators import calc_macd_qa
from GolemQ.analysis.pivot import calc_zen_dash_timing_lag
from GolemQ.core.constants import AKA, FIELD as FLD
from GolemQ.models.alias import LTT, TRD


def _idx(n, start='2025-01-06'):
    return pd.MultiIndex.from_arrays(
        [pd.bdate_range(start, periods=n), ['600519'] * n], names=['ts', 'code'])


class _Dir:
    def __init__(self, name):
        self.name = name


class _FX:
    def __init__(self, dt):
        self.dt = dt


class _BI:
    def __init__(self, a, b, up):
        self.fx_a = _FX(a)
        self.fx_b = _FX(b)
        self.direction = _Dir('Up' if up else 'Down')


class TestZenDash(unittest.TestCase):
    def test_lag_from_bi_direction(self):
        idx = _idx(20)
        dts = idx.get_level_values(0)
        bis = [_BI(dts[2], dts[7], True),
               _BI(dts[8], dts[13], False),
               _BI(dts[14], dts[18], True)]
        got = calc_zen_dash_timing_lag(bis, idx).to_numpy()
        # 首笔之前的 NaN 走 `>0` 的 false 分支 ⇒ 也从 -1 开始连计
        want = np.array(
            [-1, -2, 1, 2, 3, 4, 5, 6,
             -1, -2, -3, -4, -5, -6,
             1, 2, 3, 4, -1, -2])
        np.testing.assert_array_equal(got, want)

    def test_empty_bi_list(self):
        idx = _idx(3)
        got = calc_zen_dash_timing_lag([], idx).to_numpy()
        np.testing.assert_array_equal(got, [-1, -2, -3])


class TestZenDashCount(unittest.TestCase):
    def test_one_calc_energy(self):
        lag = [0, 0, 5, 5, 5, 5, 5]
        zd = [-1, 1, 1, 1, 2, 1, 1]
        f = pd.DataFrame({
            LTT.QUADRANT_LEVERAGE_MACD_TIMING_LAG_DUMMY: lag,
            FLD.ZEN_DASH_TIMING_LAG_DUMMY: zd,
        }, index=_idx(len(lag)))
        out = Q.calc_quadrant_zen_dash_count(f)
        # signals: 0,0,1,1,1e-12,1,1 → 累加后 astype(int16)
        np.testing.assert_array_equal(
            out[LTT.QUADRANT_ZEN_DASH_COUNT].to_numpy(), [0, 0, 1, 2, 2, 3, 4])

    def test_missing_column_raises(self):
        f = pd.DataFrame(index=_idx(3))
        with self.assertRaises(KeyError):
            Q.calc_quadrant_zen_dash_count(f)


class TestStarDown(unittest.TestCase):
    def test_structural(self):
        n = 180
        close = np.r_[np.linspace(10, 30, 60), np.linspace(30, 5, 60),
                      np.linspace(5, 40, 60)]
        o = pd.DataFrame({'open': close, 'high': close, 'low': close,
                          'close': close}, index=_idx(n))
        got = Q.calc_bs_star_down(o)
        self.assertEqual(len(got), n)
        self.assertTrue(set(np.unique(got.to_numpy())) <= {0, 1})
        self.assertGreater(int(got.sum()), 0, '趋势反转段应至少出现一次星线死叉')

    def test_stardown_count_is_one_calc_energy(self):
        before = [3, 1, 2, 1]
        eng = [0, 1, 1, 0]
        f = pd.DataFrame({
            LTT.BS_STAR_DOWN_BEFORE_MAJOR: before,
            LTT.QUADRANT_LEVERAGE_ENERGY_TIMING_LAG_DUMMY: eng,
        }, index=_idx(4))
        out = Q.calc_quadrant_stardown_count(f)
        np.testing.assert_array_equal(
            out[LTT.QUADRANT_STARDOWN_COUNT].to_numpy(), [0, 1, 1, 0])


class TestPowerlineUprisalLag(unittest.TestCase):
    def _frame(self, n=8):
        return pd.DataFrame({
            LTT.QUADRANT_LEVERAGE_ENERGY_TIMING_LAG_DUMMY: [1.0] * n,
            LTT.QUADRANT_STARDOWN_COUNT: [1] * n,
            LTT.QUADRANT_POWERLINE_MAJOR: [2.0] * n,
            FLD.DEA_MAJOR: [1.0] * n,
        }, index=_idx(n))

    def test_cond1_run(self):
        out = Q.calc_quadrant_powerline_uprisal_lag(self._frame())
        np.testing.assert_array_equal(
            out[LTT.QUADRANT_LEVERAGE_POWERLINE_UPRISING_TIMING_LAG].to_numpy(),
            [1, 2, 3, 4, 5, 6, 7, 8])

    def test_missing_column_raises(self):
        f = self._frame().drop(columns=[FLD.DEA_MAJOR])
        with self.assertRaises(KeyError):
            Q.calc_quadrant_powerline_uprisal_lag(f)


class TestSecondOrBranch(unittest.TestCase):
    """`QLevMACDLagD` 的第二分支：**三列都在场**才启用（老树 `mainstream.py:228`）。"""

    @staticmethod
    def _frame(n=6, with_branch=False):
        f = pd.DataFrame({
            # powerline 恒正 ⇒ energy 累积、EngLag = 1..n
            LTT.QUADRANT_POWERLINE_MINOR: [1.2] * n,
            LTT.QUADRANT_POWERLINE: [1.2] * n,
            LTT.QUADRANT_POWERLINE_MAJOR: [1.2] * n,
            # 第一分支恒假（MACD 柱差分 <= 0）
            FLD.MACD_DELTA_MAJOR: [-1.0] * n,
            AKA.CLOSE: np.linspace(10, 11, n),
        }, index=_idx(n))
        if with_branch:
            f[FLD.MACD_DELTA] = [1.0] * n
            f[FLD.MACD_CROSS_SX_BEFORE] = [100.0] * n
            f[LTT.MAINSTREAM_LEADINx5_BEFORE_MINOR] = [1.0] * n
        return f

    def test_branch_off_without_inputs(self):
        out = Q.calc_quadrant_leverage_macd_timing_lag(self._frame())
        # 第一分支恒假 ⇒ 全是负 lag
        self.assertTrue((out[LTT.QUADRANT_LEVERAGE_MACD_TIMING_LAG_DUMMY] < 0).all())

    def test_branch_on_with_inputs(self):
        out = Q.calc_quadrant_leverage_macd_timing_lag(self._frame(with_branch=True))
        # 第二分支要求 `EngLag > 1` ⇒ 第一根（EngLag=1）仍假，从第二根起成立
        np.testing.assert_array_equal(
            out[LTT.QUADRANT_LEVERAGE_MACD_TIMING_LAG_DUMMY].to_numpy(),
            [-1, 1, 2, 3, 4, 5])


class TestNegativeLeverageLag(unittest.TestCase):
    """`QLevMACDLagD` 的结构反函数 —— 负能量/负 MACD 链（新树自造）。"""

    @staticmethod
    def _frame(n=6, down=True):
        v = -5.0 if down else 5.0
        return pd.DataFrame({
            LTT.QUADRANT_POWERLINE_MINOR: [v] * n,
            LTT.QUADRANT_POWERLINE: [v] * n,
            LTT.QUADRANT_POWERLINE_MAJOR: [v] * n,
            FLD.MACD_DELTA_MAJOR: [(-1.0 if down else 1.0)] * n,
        }, index=_idx(n))

    def test_down_trend_is_negative(self):
        out = Q.calc_quadrant_leverage_negative_timing_lag(
            self._frame(down=True))
        for col in (LTT.QUADRANT_LEVERAGE_ENERGY_NEGATIVE_MINOR,
                    LTT.QUADRANT_LEVERAGE_ENERGY_NEGATIVE,
                    LTT.QUADRANT_LEVERAGE_ENERGY_NEGATIVE_MAJOR,
                    LTT.QUADRANT_LEVERAGE_ENERGY_NEGATIVE_DUMMY,
                    LTT.QUADRANT_LEVERAGE_ENERGY_TIMING_LAG_NEGATIVE_DUMMY,
                    LTT.QUADRANT_LEVERAGE_NEGATIVE_TIMING_LAG):
            self.assertIn(col, out.columns)
        np.testing.assert_array_equal(
            out[LTT.QUADRANT_LEVERAGE_NEGATIVE_TIMING_LAG].to_numpy(),
            [-1, -2, -3, -4, -5, -6])

    def test_up_trend_defaults_positive(self):
        out = Q.calc_quadrant_leverage_negative_timing_lag(
            self._frame(down=False))
        # 上行帧负能量不活跃 ⇒ 原始信号本来就是 +1；门控不改变结果
        np.testing.assert_array_equal(
            out[LTT.QUADRANT_LEVERAGE_NEGATIVE_TIMING_LAG].to_numpy(),
            [1, 2, 3, 4, 5, 6])

    def test_gate_resets_negative_when_positive_lag_gt4(self):
        """门控：正列 > 4 时，负能量原始信号被强制回 +1。"""
        f = pd.DataFrame({
            LTT.QUADRANT_POWERLINE_MINOR: [0.0] * 6,
            LTT.QUADRANT_POWERLINE: [0.0] * 6,
            LTT.QUADRANT_POWERLINE_MAJOR: [0.0] * 6,
            FLD.MACD_DELTA_MAJOR: [1.0] * 6,
        }, index=_idx(6))
        out = Q.calc_quadrant_leverage_negative_timing_lag(f)
        np.testing.assert_array_equal(
            out[LTT.QUADRANT_LEVERAGE_MACD_TIMING_LAG_DUMMY].to_numpy(),
            [1, 2, 3, 4, 5, 6])
        np.testing.assert_array_equal(
            out[LTT.QUADRANT_LEVERAGE_NEGATIVE_TIMING_LAG].to_numpy(),
            [-1, -2, -3, -4, 1, 2])

    def test_is_structural_not_a_negation(self):
        """不是 `-QLevMACDLagD`：负段可以多于正列负段，且包含正列负段的一部分。"""
        f = pd.DataFrame({
            LTT.QUADRANT_POWERLINE_MINOR: [0.0] * 6,
            LTT.QUADRANT_POWERLINE: [0.0] * 6,
            LTT.QUADRANT_POWERLINE_MAJOR: [0.0] * 6,
            FLD.MACD_DELTA_MAJOR: [-1.0, -1.0, -1.0, 1.0, 1.0, 1.0],
        }, index=_idx(6))
        out = Q.calc_quadrant_leverage_negative_timing_lag(f.copy())
        pos = Q.calc_quadrant_leverage_macd_timing_lag(
            f.copy())[LTT.QUADRANT_LEVERAGE_MACD_TIMING_LAG_DUMMY]
        neg = out[LTT.QUADRANT_LEVERAGE_NEGATIVE_TIMING_LAG]
        np.testing.assert_array_equal(pos.to_numpy(), [-1, -2, -3, 1, 2, 3])
        np.testing.assert_array_equal(neg.to_numpy(), [-1, -2, -3, -4, -5, -6])
        self.assertFalse((neg == -pos).all())
        self.assertGreater(int((neg < 0).sum()), int((pos < 0).sum()))
        self.assertGreater(int(((neg < 0) & (pos < 0)).sum()), 0)

    def test_missing_powerline_raises(self):
        with self.assertRaises(KeyError):
            Q.calc_quadrant_leverage_negative_timing_lag(
                pd.DataFrame({FLD.MACD_DELTA_MAJOR: [0.0]}))


class TestOmniPathBaseline(unittest.TestCase):
    """`OmniPathLeverBslLag` —— 旧树 `base.py:1008` 的 baseline lag。"""

    @staticmethod
    def _frame(n=4):
        return pd.DataFrame({
            FLD.DIF_NORM_TIMING_LAG_MAJOR: [1] * n,
            FLD.DEA_NORM_TIMING_LAG_MAJOR: [0] * n,
            FLD.MTM_NORM_TIMING_LAG_MAJOR: [0] * n,
            LTT.THREE_QUADRANT_UPRISING_TIMING_LAG: [1] * n,
        }, index=_idx(n))

    def test_dif_branch_runs(self):
        out = Q.calc_omnipath_leverage_baseline_timing_lag(self._frame())
        np.testing.assert_array_equal(
            out[TRD.OMNIPATH_LEVERAGE_BASELINE_TIMING_LAG].to_numpy(),
            [1, 2, 3, 4])

    def test_missing_raises(self):
        with self.assertRaises(KeyError):
            Q.calc_omnipath_leverage_baseline_timing_lag(pd.DataFrame())


class TestFbProphetLag(unittest.TestCase):
    """`fbpLUpRsgLagDmy` 的降级 producer。"""

    def test_up_flag_event_lag(self):
        f = pd.DataFrame({
            FLD.FBPROPHET_LB_MAJOR: [2, 2, 1],
            FLD.FBPROPHET_LB: [1, 1, 2],
        }, index=_idx(3))
        out = Q.calc_fbprophet_lb_uprising_timing_lag(f)
        np.testing.assert_array_equal(
            out[FLD.FBPROPHET_LB_UPRISING_TIMING_LAG_DUMMY].to_numpy(),
            [1, 2, -1])

    def test_missing_raises(self):
        with self.assertRaises(KeyError):
            Q.calc_fbprophet_lb_uprising_timing_lag(pd.DataFrame())


class TestLinearegWaveletLag(unittest.TestCase):
    """`LnarWavUprLagCand` 的降级 producer。"""

    def test_atr_lb_below_close(self):
        f = pd.DataFrame({FLD.ATR_LB_MAJOR: [1, 1, 5],
                          AKA.CLOSE: [2, 2, 2]}, index=_idx(3))
        out = Q.calc_lineareg_wavelet_uprising_timing_lag(f)
        np.testing.assert_array_equal(
            out[LTT.LINEAREG_WAVELET_UPRISING_TIMING_LAG_CANDIDATE].to_numpy(),
            [1, 2, -1])

    def test_fallback_to_atr_lb(self):
        f = pd.DataFrame({FLD.ATR_LB: [1, 1, 5],
                          AKA.CLOSE: [2, 2, 2]}, index=_idx(3))
        out = Q.calc_lineareg_wavelet_uprising_timing_lag(f)
        np.testing.assert_array_equal(
            out[LTT.LINEAREG_WAVELET_UPRISING_TIMING_LAG_CANDIDATE].to_numpy(),
            [1, 2, -1])

    def test_missing_raises(self):
        with self.assertRaises(KeyError):
            Q.calc_lineareg_wavelet_uprising_timing_lag(pd.DataFrame())


class TestMainstreamMajor(unittest.TestCase):
    """日线 MAJOR 版：`MsLeadinBfMaj` / `MsLeadoutBfMaj` 与基础列同值。"""

    def test_major_columns_equal_base(self):
        idx = _idx(160)
        close = pd.Series(np.linspace(10, 12, len(idx)), index=idx)
        ohlc = pd.DataFrame({
            'open': close, 'high': close + 0.1, 'low': close - 0.1,
            'close': close, 'volume': 1000.0,
        }, index=idx)
        out = Q.calc_mainstream_major_feats(ohlc)
        for base, major in (
                (LTT.MAINSTREAM_LEADIN_BEFORE, LTT.MAINSTREAM_LEADIN_BEFORE_MAJOR),
                (LTT.MAINSTREAM_LEADOUT_BEFORE, LTT.MAINSTREAM_LEADOUT_BEFORE_MAJOR)):
            self.assertIn(major, out.columns)
            np.testing.assert_array_equal(out[major].to_numpy(),
                                          out[base].to_numpy())


class TestMacdQA(unittest.TestCase):
    def test_matches_quantaxis_formula(self):
        close = pd.Series(np.linspace(10, 20, 80) + np.sin(np.arange(80)))
        dif, dea, macd = calc_macd_qa(close)

        def ema(x, n):
            return x.ewm(span=n, min_periods=n - 1, adjust=True).mean()

        want_dif = ema(close, 12) - ema(close, 26)
        want_dea = ema(want_dif, 9)
        np.testing.assert_allclose(dif.to_numpy(), want_dif.to_numpy(), rtol=0, atol=1e-12)
        np.testing.assert_allclose(dea.to_numpy(), want_dea.to_numpy(), rtol=0, atol=1e-12)
        np.testing.assert_allclose(macd.to_numpy(), (want_dif - want_dea).to_numpy() * 2,
                                   rtol=0, atol=1e-12)

    def test_hist_is_double_the_talib_convention(self):
        close = pd.Series(np.linspace(10, 20, 80))
        dif, dea, macd = calc_macd_qa(close)
        np.testing.assert_allclose(macd.to_numpy(), ((dif - dea) * 2).to_numpy(),
                                   rtol=0, atol=1e-12)


if __name__ == '__main__':
    unittest.main()
