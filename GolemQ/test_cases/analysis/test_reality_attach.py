# coding: utf-8
"""`analysis/reality_attach.py` 的附加口径测试（mock DB，不连库）。

钉住三件事：

1. 60min coldstart 按日最后值 + **上一可用交易日**映射（周末/节后首日不掉 NaN）；
2. ChartScan 是稀疏事件，**精确时间戳 join**（不 ffill）；
3. Prophet 连续量在 attach 末尾 ffill（15:00 真值之后），曲线不再断线。
"""
from __future__ import annotations

import unittest
from unittest import mock

import numpy as np
import pandas as pd

from GolemQ.analysis.reality_attach import attach_reality_feats
from GolemQ.core.constants import FEATURES as FTR, FIELD as FLD


def _features():
    idx = pd.MultiIndex.from_tuples([
        (pd.Timestamp('2026-01-08 15:00'), '600519'),   # 周四
        (pd.Timestamp('2026-01-09 10:30'), '600519'),   # 周五盘中
        (pd.Timestamp('2026-01-09 15:00'), '600519'),
        (pd.Timestamp('2026-01-12 10:30'), '600519'),   # 周一盘中
        (pd.Timestamp('2026-01-12 15:00'), '600519'),
    ], names=['ts', 'code'])
    return pd.DataFrame({'close': [1.0, 2.0, 3.0, 4.0, 5.0]}, index=idx)


def _r60():
    idx = pd.MultiIndex.from_tuples([
        (pd.Timestamp('2026-01-08 15:00'), '600519'),
        (pd.Timestamp('2026-01-09 10:30'), '600519'),
        (pd.Timestamp('2026-01-09 15:00'), '600519'),
        (pd.Timestamp('2026-01-12 10:30'), '600519'),
        (pd.Timestamp('2026-01-12 15:00'), '600519'),
    ], names=['ts', 'code'])
    return pd.DataFrame({
        'stage': [10.0, 20.0, 30.0, 40.0, 50.0],
        # 周一整天都没有 Prophet 值：验证 attach 末尾的 ffill
        'prophetLbD': [1.0, 2.0, 3.0, np.nan, np.nan],
        'ChartScaned': [np.nan, np.nan, 5.0, np.nan, np.nan],
        'garPD': [[(1.1, 1.0)]] * 5,
        'garPAnchor': ['2026-01-08'] * 5,
        'se': [0.1, 0.2, 0.3, 0.4, 0.5],
        'ZenPivotCnt': [3.0, 4.0, 5.0, 6.0, 7.0],
        'renkoSbar': [-1.0, -1.0, 1.0, 1.0, 1.0],
        'garPCount': [10.0, 20.0, 30.0, 40.0, 50.0],
        'renkoSpriceD': [-0.1, -0.1, 0.1, 0.1, 0.1],
        # 60min 与日表**同时**有 ChipDist 列（实测 510300 / 000711）：
        # 60min 组必须先贴上，日表那份要被跳过（否则 join 直接抛重叠）
        'cost5PrzD': [0.5] * 5,
    }, index=idx)


def _rday():
    """日族：`se` / `ZenPivotCnt` **不在**日表（D78 撤销 D74），只返回空帧。"""
    idx = pd.MultiIndex.from_tuples([
        (pd.Timestamp('2026-01-08'), '600519'),
        (pd.Timestamp('2026-01-09'), '600519'),
        (pd.Timestamp('2026-01-12'), '600519'),
    ], names=['ts', 'code'])
    return pd.DataFrame({'poly9MajD': [1.0, 2.0, 3.0],
                         'cost5PrzD': [0.9, 0.9, 0.9]}, index=idx)


class TestAttachRealityFeats(unittest.TestCase):

    def _call(self):
        def _fetch(codes, start, end, family=None, market_type=None, db=None,
                   **kwargs):
            if family == '60min':
                return _r60()
            if family == 'day':
                return _rday()
            return None

        with mock.patch('GolemQ.analysis.reality_attach.fetch_reality_feat',
                        side_effect=_fetch):
            return attach_reality_feats(
                _features(), '600519', market_type='stock_cn', db=object(),
                day_ohlc=pd.DataFrame())

    def test_coldstart_uses_previous_available_day(self):
        out, loaded = self._call()
        s = out['garPCount']
        # 周四 15:00 = 当日；周五盘中 = 周四；周五 15:00 = 当日
        self.assertEqual(s.loc[(pd.Timestamp('2026-01-08 15:00'), '600519')], 10.0)
        self.assertEqual(s.loc[(pd.Timestamp('2026-01-09 10:30'), '600519')], 10.0)
        self.assertEqual(s.loc[(pd.Timestamp('2026-01-09 15:00'), '600519')], 30.0)
        # 周一盘中 = 周五（跳过周末）；周一 15:00 = 当日
        self.assertEqual(s.loc[(pd.Timestamp('2026-01-12 10:30'), '600519')], 30.0)
        self.assertEqual(s.loc[(pd.Timestamp('2026-01-12 15:00'), '600519')], 50.0)
        self.assertIn('garPCount', loaded['60min_cold'])

    def test_review_m9t_lag_columns_are_not_in_contract(self):
        from GolemQ.pipeline.review.contract import COLDSTART_COLUMNS
        # 老树 REAL 常量保留定义（m9tLagR），但 review 契约/attach 都不再包含它
        self.assertEqual(FTR.MAGIC_NINE_TURNS_TIMING_LAG_REAL, 'm9tLagR')
        for col in ('m9tLag', 'm9tLagD', 'm9tLagR'):
            self.assertNotIn(col, COLDSTART_COLUMNS)

    def test_hour_extra_is_attached_from_60min(self):
        """D78：`se` / `ZenPivotCnt` 来自 60min 表，按时间戳精确 join（不 ffill）。"""
        out, loaded = self._call()
        self.assertIn('se', loaded.get('60min_extra', []))
        self.assertIn('ZenPivotCnt', loaded.get('60min_extra', []))
        self.assertNotIn('se', loaded.get('day', []))
        self.assertEqual(out.loc[(pd.Timestamp('2026-01-09 15:00'), '600519'),
                                 'ZenPivotCnt'], 5.0)
        self.assertEqual(out.loc[(pd.Timestamp('2026-01-09 10:30'), '600519'),
                                 'se'], 0.2)
        self.assertEqual(out.loc[(pd.Timestamp('2026-01-12 15:00'), '600519'),
                                 'se'], 0.5)
        # Renko 14：同样精确 join（D78）
        # `loaded` 记录的是**还原后**的列名（`*D` 已乘回 close，D49）
        self.assertIn('renkoSbar', loaded.get('60min_renko', []))
        self.assertIn('renkoSprice', loaded.get('60min_renko', []))
        i = (pd.Timestamp('2026-01-09 15:00'), '600519')
        self.assertEqual(out.loc[i, 'renkoSbar'], 1.0)
        self.assertAlmostEqual(out.loc[i, 'renkoSprice'], 0.1 * out.loc[i, 'close'])

    def test_60min_chip_wins_over_day_chip(self):
        """两张表都有 ChipDist 列时，60min 的 ÷本 bar close 口径优先。

        实测触发：`510300` / `000711` 的日表与 60min 表都写了 `cost5PrzD`；
        旧实现直接 `features.join(addd)` 抛 `columns overlap`。
        """
        out, loaded = self._call()
        i = (pd.Timestamp('2026-01-09 15:00'), '600519')
        self.assertIn('cost5Prz', loaded.get('60min_chip', []))
        self.assertAlmostEqual(out.loc[i, 'cost5Prz'], 0.5 * out.loc[i, 'close'])
        self.assertNotIn('cost5PrzD', out.columns)

    def test_garident_bands_restored_by_anchor_close(self):
        out, loaded = self._call()
        v = out.loc[(pd.Timestamp('2026-01-09 15:00'), '600519'), 'garP']
        self.assertEqual(tuple(v[0]), (1.1, 1.0))
        # D76：还原成绝对价后取 BOTTOM_P / ROOF_P（不落库、不归一化）
        i = (pd.Timestamp('2026-01-09 15:00'), '600519')
        self.assertEqual(out.loc[i, FLD.BOTTOM_PRICE], 1.0)
        self.assertEqual(out.loc[i, FLD.ROOF_PRICE], 1.1)
        self.assertIn(FLD.BOTTOM_PRICE, loaded['garident'])

    def test_chartscan_is_exact_join_not_ffill(self):
        out, _ = self._call()
        c = out['ChartScaned60']
        self.assertEqual(c.loc[(pd.Timestamp('2026-01-09 15:00'), '600519')], 5.0)
        self.assertTrue(pd.isna(c.loc[(pd.Timestamp('2026-01-09 10:30'), '600519')]))
        self.assertTrue(pd.isna(c.loc[(pd.Timestamp('2026-01-12 15:00'), '600519')]))

    def test_prophet_gap_is_ffilled(self):
        out, loaded = self._call()
        self.assertIn(FLD.FBPROPHET_LB, loaded['ffill_prophet'])
        # 周一 15:00 原为 NaN；ffill 用周五 15:00 的 3.0，再 * close(5)
        i = (pd.Timestamp('2026-01-12 15:00'), '600519')
        self.assertAlmostEqual(out.loc[i, FLD.FBPROPHET_LB], 3.0 * 5.0)
        self.assertFalse(out[FLD.FBPROPHET_LB].isna().any())
        # D 后缀列不再进入 attach 帧
        self.assertNotIn(FLD.FBPROPHET_LB_DUMMY, out.columns)


if __name__ == '__main__':
    unittest.main()
