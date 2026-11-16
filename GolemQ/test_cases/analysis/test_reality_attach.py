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

    def test_unnamed_first_level_does_not_cartesian_expand(self):
        """市场取数返回的索引首层常是 None；join 前必须显式命名 `(ts, code)`。

        否则 pandas 会按不同名字的 MultiIndex 做笛卡尔积（000711 实测
        2,061 × 320 = 659,520 行），不报错但整张表错位。
        """
        def _fetch(codes, start, end, family=None, market_type=None, db=None,
                   **kwargs):
            if family == '60min':
                return _r60()
            if family == 'day':
                return _rday()
            return None

        feats = _features()
        feats.index = feats.index.set_names([None, 'code'])
        with mock.patch('GolemQ.analysis.reality_attach.fetch_reality_feat',
                        side_effect=_fetch):
            out, _ = attach_reality_feats(
                feats, '600519', market_type='stock_cn', db=object(),
                day_ohlc=pd.DataFrame())
        self.assertEqual(len(out), len(feats))
        self.assertEqual(out.index.nlevels, 2)
        self.assertEqual(out.index.names, ['ts', 'code'])

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

    def test_sub60min_features_use_causal_ffill_for_60min_columns(self):
        """15min/1min 帧上接 60min review：按 ts 因果 ffill，不能只 join 整点。

        否则 `se` / `ZenPivotCnt` 只在 60min 时刻有值，曲线出现锯齿和断点。
        """
        idx = pd.MultiIndex.from_tuples([
            (pd.Timestamp('2026-01-09 10:15'), '600519'),
            (pd.Timestamp('2026-01-09 10:30'), '600519'),
            (pd.Timestamp('2026-01-09 10:45'), '600519'),
            (pd.Timestamp('2026-01-09 15:00'), '600519'),
            (pd.Timestamp('2026-01-09 15:15'), '600519'),
        ], names=['ts', 'code'])
        feats = pd.DataFrame({'close': [1.0, 2.0, 3.0, 4.0, 5.0]}, index=idx)

        def _fetch(codes, start, end, family=None, market_type=None, db=None,
                   **kwargs):
            if family == '60min':
                return _r60()
            if family == 'day':
                return _rday()
            return None

        with mock.patch('GolemQ.analysis.reality_attach.fetch_reality_feat',
                        side_effect=_fetch):
            out, loaded = attach_reality_feats(
                feats, '600519', market_type='stock_cn', db=object(),
                day_ohlc=pd.DataFrame())

        self.assertIn('se', loaded.get('60min_extra', []))
        se = out['se']
        self.assertEqual(se.loc[(pd.Timestamp('2026-01-09 10:15'), '600519')], 0.1)
        self.assertEqual(se.loc[(pd.Timestamp('2026-01-09 10:30'), '600519')], 0.2)
        self.assertEqual(se.loc[(pd.Timestamp('2026-01-09 10:45'), '600519')], 0.2)
        self.assertEqual(se.loc[(pd.Timestamp('2026-01-09 15:00'), '600519')], 0.3)
        self.assertEqual(se.loc[(pd.Timestamp('2026-01-09 15:15'), '600519')], 0.3)
        # ZenPivotLag 走同一路径（本 fixture 里没有该列，用 ZenPivotCnt 验证）
        self.assertEqual(out.loc[(pd.Timestamp('2026-01-09 10:45'), '600519'),
                                 'ZenPivotCnt'], 4.0)

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


class TestStageBeforeFill(unittest.TestCase):
    """`bstStgBf / epStgBf` 由 attach **现算**（用户 2026-11-16）。

    `stage` 是 4.4 搬运值（`tools/migrate_stage_from44.py`），attach **不覆盖**；
    bst 由 `stage < -0.00168` 判，ep 由 `garPClr < 0.927 & DEA 金叉时序` 判。
    """

    @staticmethod
    def _frame(stage, clearance=None, with_stage=True, close=None):
        n = len(stage)
        idx = pd.MultiIndex.from_arrays(
            [pd.date_range('2026-10-01 15:00', periods=n, freq='D'),
             ['600000'] * n], names=['ts', 'code'])
        data = {'close': close if close is not None else [10.0] * n,
                FLD.GARIDENT_PRICE_CLEARANCE:
                    clearance if clearance is not None else [0.5] * n}
        if with_stage:
            data['stage'] = stage
        return pd.DataFrame(data, index=idx)

    def _fill(self, frame):
        from GolemQ.analysis import reality_attach as ra
        loaded = {}
        return ra._stage_before_fill(frame, None, loaded), loaded

    def test_bst_is_timeline_duration_of_stage_below_threshold(self):
        out, loaded = self._fill(self._frame([1.0, -0.5, 1.0, -0.5, 0.0]))
        self.assertEqual(out[FLD.BOOTSTRAP_STAGE_BEFORE].tolist(), [1, 0, 1, 0, 1])
        self.assertIn(FLD.BOOTSTRAP_STAGE_BEFORE, loaded['stage_before'])

    def test_stage_is_never_overwritten(self):
        """`stage` 一律保持原值（4.4 搬运值优先）。"""
        frame = self._frame([1.0, -0.5, 2.0])
        out, _ = self._fill(frame)
        self.assertEqual(out['stage'].tolist(), [1.0, -0.5, 2.0])

    def test_clearance_above_threshold_never_triggers_ep(self):
        out, _ = self._fill(self._frame([1.0] * 6, clearance=[1.0] * 6))
        self.assertEqual(int((out[FLD.ENDPOINT_STAGE_BEFORE] == 0).sum()), 0)

    def test_no_stage_column_is_a_noop(self):
        out, loaded = self._fill(self._frame([None] * 3, with_stage=False))
        self.assertNotIn(FLD.BOOTSTRAP_STAGE_BEFORE, out.columns)
        self.assertEqual(loaded, {})




if __name__ == '__main__':
    unittest.main()
