# coding: utf-8
"""review step 与 chip backfill 共用的 hour_extra 映射层回归。"""
from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from GolemQ.core.constants import AKA
from GolemQ.pipeline.review.contract import (DAY_SAMPLE_COLUMNS,
                                             HOUR_NATIVE_COLUMNS,
                                             RENKO_DAY_COLUMNS,
                                             RENKO_PRICE_DUMMIES)
from GolemQ.pipeline.review.steps import hour_extra as he

CODE = '600519'
D1 = pd.Timestamp('2024-01-02')
D2 = pd.Timestamp('2024-01-03')
D3 = pd.Timestamp('2024-01-04')


def _ohlc():
    ts = pd.DatetimeIndex([
        pd.Timestamp('2024-01-02 10:00'),
        pd.Timestamp('2024-01-02 15:00'),
        pd.Timestamp('2024-01-03 10:00'),
    ])
    idx = pd.MultiIndex.from_arrays([ts, [CODE] * len(ts)],
                                    names=['ts', 'code'])
    close = np.array([10.0, 20.0, 30.0])
    return pd.DataFrame({
        AKA.OPEN: close, AKA.HIGH: close, AKA.LOW: close,
        AKA.CLOSE: close, 'volume': 1.0,
    }, index=idx)


class TestSharedHourExtraFrame(unittest.TestCase):

    def test_maps_last_snapshot_and_price_ratios(self):
        targets = pd.DatetimeIndex([
            pd.Timestamp('2024-01-02 10:00'),
            pd.Timestamp('2024-01-02 15:00'),
            pd.Timestamp('2024-01-03 10:00'),
        ])
        # 快照只到 2024-01-02 15:00；所有 target 都应取它（as-of）。
        se_days = pd.DatetimeIndex([pd.Timestamp('2024-01-02 15:00')])
        se_vals = np.array([0.5])
        piv_days = pd.DatetimeIndex([pd.Timestamp('2024-01-02 15:00')])
        piv_df = pd.DataFrame([[5.0, 6.0]], index=piv_days,
                              columns=['ZenPivotCnt', 'ZenPivotLag'])
        rk_days = pd.DatetimeIndex([pd.Timestamp('2024-01-02 15:00')])
        rk_row = {c: 7.0 for c in RENKO_DAY_COLUMNS}
        for _base, dummy in RENKO_PRICE_DUMMIES:
            rk_row[dummy] = 100.0
        rk_df = pd.DataFrame([rk_row], index=rk_days,
                             columns=list(RENKO_DAY_COLUMNS))
        nat_days = pd.DatetimeIndex([pd.Timestamp('2024-01-02 15:00')])
        nat_df = pd.DataFrame([[3.0] * len(HOUR_NATIVE_COLUMNS)],
                              index=nat_days,
                              columns=list(HOUR_NATIVE_COLUMNS))

        with patch.object(he, 'sample_entropy_by_day',
                          return_value=(se_days, se_vals)), \
                patch.object(he, 'pivot_stats',
                             return_value=(piv_days, piv_df)), \
                patch.object(he, 'renko_by_day',
                             return_value=(rk_days, rk_df)), \
                patch.object(he, 'native_by_day',
                             return_value=(nat_days, nat_df)):
            out = he.build_hour_extra_frame(_ohlc(), targets, CODE)

        self.assertEqual(len(out), len(targets))
        # 第一根 10:00 在 15:00 快照之前 → 三个按日指标都是 NaN。
        self.assertTrue(pd.isna(out['se'].iloc[0]))
        self.assertEqual(out['se'].iloc[1:].tolist(), [0.5, 0.5])
        self.assertTrue(pd.isna(out['ZenPivotCnt'].iloc[0]))
        self.assertEqual(out['ZenPivotCnt'].iloc[1:].tolist(), [5.0, 5.0])
        self.assertTrue(pd.isna(out['ZenPivotLag'].iloc[0]))
        self.assertEqual(out['ZenPivotLag'].iloc[1:].tolist(), [6.0, 6.0])
        # 价格 dummy 按目标行 close 转比值；非价格列原样保留。
        dummy = RENKO_PRICE_DUMMIES[0][1]
        self.assertTrue(pd.isna(out[dummy].iloc[0]))
        self.assertAlmostEqual(float(out[dummy].iloc[1]), 5.0)
        self.assertAlmostEqual(float(out[dummy].iloc[2]), 100.0 / 30.0)
        non_price = [c for c in RENKO_DAY_COLUMNS
                     if c not in {d for _b, d in RENKO_PRICE_DUMMIES}][0]
        self.assertTrue(pd.isna(out[non_price].iloc[0]))
        self.assertTrue(np.allclose(out[non_price].iloc[1:].to_numpy(dtype=float), 7.0))
        nat = out[list(HOUR_NATIVE_COLUMNS)].to_numpy(dtype=float)
        self.assertTrue(np.isnan(nat[0]).all())
        self.assertTrue(np.allclose(nat[1:], 3.0))


class TestSharedChipMapping(unittest.TestCase):

    def test_ratios_use_last_available_chip_day(self):
        chip_abs = {
            D1: {'cost5Prz': 10.0, 'cost15Prz': 10.0,
                 'cost85Prz': 10.0, 'cost95Prz': 10.0},
            D3: {'cost5Prz': 20.0, 'cost15Prz': 20.0,
                 'cost85Prz': 20.0, 'cost95Prz': 20.0},
        }
        targets = pd.DatetimeIndex([
            pd.Timestamp('2024-01-03 10:00'),
            pd.Timestamp('2024-01-03 15:00'),
            pd.Timestamp('2024-01-05 10:00'),
        ])
        close = pd.Series([5.0, 10.0, 20.0], index=targets)
        out = he.chip_ratios_from_abs(chip_abs, targets, close, code=CODE)
        # D2 没有 chip → 用 D1；D4 没有 → 用 D3。
        self.assertEqual(out['cost15PrzD'].tolist(), [2.0, 1.0, 1.0])

    def test_missing_close_leaves_nan(self):
        chip_abs = {D1: {'cost15Prz': 10.0}}
        targets = pd.DatetimeIndex([pd.Timestamp('2024-01-02 15:00')])
        out = he.chip_ratios_from_abs(chip_abs, targets, None, code=CODE)
        self.assertTrue(pd.isna(out['cost15PrzD'].iloc[0]))


class TestSharedDaySample(unittest.TestCase):

    def test_15_requires_current_day_but_intraday_uses_previous(self):
        cache = {
            D1: {c: 1.0 for c in DAY_SAMPLE_COLUMNS},
            D2: {c: 2.0 for c in DAY_SAMPLE_COLUMNS},
        }
        targets = pd.DatetimeIndex([
            pd.Timestamp('2024-01-02 15:00'),   # D1
            pd.Timestamp('2024-01-03 10:00'),   # D2 盘中 → D1
            pd.Timestamp('2024-01-03 15:00'),   # D2
            pd.Timestamp('2024-01-04 15:00'),   # D3 缺行 → NaN
            pd.Timestamp('2024-01-04 10:00'),   # D3 盘中 → D2
        ])
        out = he.sample_day_frame(cache, targets)
        col = DAY_SAMPLE_COLUMNS[0]
        self.assertEqual(out[col].iloc[0], 1.0)
        self.assertEqual(out[col].iloc[1], 1.0)
        self.assertEqual(out[col].iloc[2], 2.0)
        self.assertTrue(pd.isna(out[col].iloc[3]))
        self.assertEqual(out[col].iloc[4], 2.0)


if __name__ == '__main__':
    unittest.main()
