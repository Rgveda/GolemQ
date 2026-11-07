# coding: utf-8
"""`analysis/quadrant_gear.py` —— QuadrantGear O 档的公式与 fold 默认。"""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from GolemQ.analysis.quadrant_gear import calc_quadrant_gear_o
from GolemQ.models.alias import LTT


def _idx(n, start='2025-01-06'):
    return pd.MultiIndex.from_arrays(
        [pd.bdate_range(start, periods=n), ['600519'] * n], names=['ts', 'code'])


def _ohlc(idx):
    close = pd.Series(np.linspace(10, 12, len(idx)), index=idx)
    return pd.DataFrame({'open': close, 'high': close + 0.1,
                         'low': close - 0.1, 'close': close,
                         'volume': 1000.0}, index=idx)


def _gear_inputs(idx):
    return pd.DataFrame({
        LTT.QUADRANT_LEVERAGE_MACD_TIMING_LAG_DUMMY: [-1.0] * len(idx),
        LTT.QUADRANT_DOWNRISK_TIMING_LAG_MAJOR: [-1.0] * len(idx),
        LTT.BS_STAR_UP_BEFORE_MAJOR: [2.0] * len(idx),
        LTT.BS_STAR_DOWN_BEFORE_MAJOR: [1.0] * len(idx),
        LTT.BS_BUY_BEFORE_MAJOR: [2.0] * len(idx),
        LTT.BS_SELL_BEFORE_MAJOR: [1.0] * len(idx),
        LTT.MAINSTREAM_LEADIN_BEFORE_MAJOR: [1.0] * len(idx),
        LTT.MAINSTREAM_LEADOUT_BEFORE_MAJOR: [2.0] * len(idx),
        LTT.QUADRANT_LEVERAGE_CHARM_O_BEFORE: [100.0] * len(idx),
        LTT.QUADRANT_LEVERAGE_CHARM_I_BEFORE: [100.0] * len(idx),
    }, index=idx)


class TestQuadrantGearO(unittest.TestCase):

    def test_formula_and_before(self):
        f = _gear_inputs(_idx(3))
        f[LTT.BS_STAR_UP_BEFORE_MAJOR] = [2.0, 2.0, 1.0]
        f[LTT.BS_STAR_DOWN_BEFORE_MAJOR] = [1.0, 1.0, 2.0]
        out = calc_quadrant_gear_o(f)
        np.testing.assert_array_equal(
            out[LTT.QUADRANT_GEAR_O].to_numpy(), [1.0, 1.0, np.nan])
        np.testing.assert_array_equal(
            out[LTT.QUADRANT_GEAR_O_BEFORE].to_numpy(), [0, 0, 1])

    def test_extra_branch_positive_qlev(self):
        """第二分支：QLev 为正，但 `dwn + q < 0`、`BSBuBfMaj < QLev`、
        且日线 `MsLeadinBfMaj < MsLeadoutBfMaj`。"""
        f = _gear_inputs(_idx(4))
        f[LTT.QUADRANT_LEVERAGE_MACD_TIMING_LAG_DUMMY] = [1.0, 2.0, 3.0, 2.0]
        f[LTT.QUADRANT_DOWNRISK_TIMING_LAG_MAJOR] = [-5.0, -5.0, 1.0, -5.0]
        f[LTT.BS_BUY_BEFORE_MAJOR] = [1.0, 1.0, 12.0, 1.0]
        f[LTT.BS_SELL_BEFORE_MAJOR] = [0.0, 0.0, 11.0, 0.0]
        f[LTT.BS_STAR_UP_BEFORE_MAJOR] = [2.0, 2.0, 2.0, 2.0]
        f[LTT.BS_STAR_DOWN_BEFORE_MAJOR] = [1.0, 1.0, 1.0, 1.0]
        f[LTT.MAINSTREAM_LEADIN_BEFORE_MAJOR] = [1.0, 1.0, 1.0, 2.0]
        f[LTT.MAINSTREAM_LEADOUT_BEFORE_MAJOR] = [2.0, 2.0, 2.0, 1.0]
        out = calc_quadrant_gear_o(f)
        np.testing.assert_array_equal(
            out[LTT.QUADRANT_GEAR_O].to_numpy(), [np.nan, 1.0, np.nan, np.nan])
        np.testing.assert_array_equal(
            out[LTT.QUADRANT_GEAR_O_BEFORE].to_numpy(), [1, 0, 1, 2])

    def test_charm_or_branch(self):
        """第五行：`Bu>Sel` 为假时，**Charm_O** `< QLev` 才兜底；Charm_I 不再参与。"""
        f = _gear_inputs(_idx(3))
        f[LTT.QUADRANT_LEVERAGE_MACD_TIMING_LAG_DUMMY] = [2.0, 2.0, 2.0]
        f[LTT.QUADRANT_DOWNRISK_TIMING_LAG_MAJOR] = [-5.0, -5.0, -5.0]
        f[LTT.BS_BUY_BEFORE_MAJOR] = [1.0, 1.0, 1.0]
        f[LTT.BS_SELL_BEFORE_MAJOR] = [2.0, 2.0, 2.0]
        f[LTT.BS_STAR_UP_BEFORE_MAJOR] = [2.0, 2.0, 2.0]
        f[LTT.BS_STAR_DOWN_BEFORE_MAJOR] = [1.0, 1.0, 1.0]
        f[LTT.MAINSTREAM_LEADIN_BEFORE_MAJOR] = [1.0, 1.0, 1.0]
        f[LTT.MAINSTREAM_LEADOUT_BEFORE_MAJOR] = [2.0, 2.0, 2.0]
        f[LTT.QUADRANT_LEVERAGE_CHARM_O_BEFORE] = [0.0, 100.0, 100.0]
        f[LTT.QUADRANT_LEVERAGE_CHARM_I_BEFORE] = [100.0, 0.0, 100.0]
        out = calc_quadrant_gear_o(f)
        np.testing.assert_array_equal(
            out[LTT.QUADRANT_GEAR_O].to_numpy(), [1.0, np.nan, np.nan])
        np.testing.assert_array_equal(
            out[LTT.QUADRANT_GEAR_O_BEFORE].to_numpy(), [0, 1, 2])

    def test_missing_raises(self):
        with self.assertRaises(KeyError):
            calc_quadrant_gear_o(pd.DataFrame())

    def test_auto_major_fold(self):
        idx = _idx(160)
        ohlc = _ohlc(idx)
        f = _gear_inputs(idx).drop(columns=[
            LTT.MAINSTREAM_LEADIN_BEFORE_MAJOR,
            LTT.MAINSTREAM_LEADOUT_BEFORE_MAJOR])
        out = calc_quadrant_gear_o(f, ohlc_day=ohlc, fold=True)
        self.assertIn(LTT.MAINSTREAM_LEADIN_BEFORE_MAJOR, out.columns)
        self.assertIn(LTT.MAINSTREAM_LEADOUT_BEFORE_MAJOR, out.columns)
        self.assertEqual(out[LTT.MAINSTREAM_LEADIN_BEFORE_MAJOR].notna().sum(),
                         len(idx))


if __name__ == '__main__':
    unittest.main()
