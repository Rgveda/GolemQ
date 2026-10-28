# coding:utf-8
"""`analysis/compact.py` + `pipeline/compact` 的用例（合成数据，不读库）。

钉住：
* 37 列 + rank 7 列的契约完整性；
* 60min-only 与 60min+day 两种模式；
* day → 60min 的**因果**映射（盘中用上一交易日、15:00 用当日）；
* `--strategy compact` 的 pipeline 逐码统计。
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

from GolemQ.analysis import compact as C
from GolemQ.analysis.indicators import atr_rsi_stops, atr_supertrend
from GolemQ.analysis.timing import (Timeline_Integral, Timeline_duration,
                                    calc_event_timing_lag)
from GolemQ.core.constants import AKA, FIELD
from GolemQ.models.alias import PFL
from GolemQ.test_cases.old_tree import load_old, old_tree_available


def _load_old_atr():
    if not old_tree_available():
        return None
    try:
        return load_old('signal/base.py', patch_attrs={
            'GolemQ.analysis.timeseries': {
                'Timeline_duration': Timeline_duration,
                'Timeline_Integral': Timeline_Integral,
                'calc_event_timing_lag': calc_event_timing_lag}})
    except Exception:                        # noqa: BLE001
        return None


_OLD_ATR = _load_old_atr()


def _ohlc(n, start, freq, seed=0, base=10.0):
    rng = np.random.default_rng(seed)
    idx = pd.MultiIndex.from_arrays(
        [pd.date_range(start, periods=n, freq=freq), ['600519'] * n],
        names=['ts', 'code'])
    close = base + np.cumsum(rng.normal(0, 0.05, n))
    return pd.DataFrame({AKA.OPEN: close, AKA.HIGH: close + 0.1,
                         AKA.LOW: close - 0.1, AKA.CLOSE: close,
                         AKA.VOLUME: 1000.0, AKA.AMOUNT: close * 1000},
                        index=idx)


def _day(n=700):
    return _ohlc(n, '2023-01-02', 'B', seed=1)


def _min60(n=2000):
    # 30min 网格上的 60min bar（10:30/11:30/14:00/15:00），与 8.3 的 60min 标签一致
    idx = pd.MultiIndex.from_arrays(
        [pd.date_range('2025-01-02', periods=n, freq='h'), ['600519'] * n],
        names=['ts', 'code'])
    rng = np.random.default_rng(2)
    close = 10 + np.cumsum(rng.normal(0, 0.02, n))
    return pd.DataFrame({AKA.OPEN: close, AKA.HIGH: close + 0.1,
                         AKA.LOW: close - 0.1, AKA.CLOSE: close,
                         AKA.VOLUME: 500.0, AKA.AMOUNT: close * 500},
                        index=idx)


class TestCausalMap(unittest.TestCase):

    def test_intraday_uses_previous_day_1500_uses_today(self):
        day = pd.Series(
            [1.0, 2.0, 3.0],
            index=pd.MultiIndex.from_arrays(
                [pd.to_datetime(['2026-01-05', '2026-01-06', '2026-01-07']),
                 ['600519'] * 3], names=['ts', 'code']))
        target = pd.MultiIndex.from_arrays(
            [pd.to_datetime(['2026-01-06 10:00', '2026-01-06 15:00',
                             '2026-01-07 10:00']),
             ['600519'] * 3], names=['ts', 'code'])
        got = C._map_causal(day, target).tolist()
        self.assertEqual(got, [1.0, 2.0, 2.0])

    def test_missing_day_stays_nan(self):
        """某天没有值就是 NaN —— 不做 ffill（用户 2026-10-24 明确）。"""
        day = pd.Series(
            [1.0, np.nan, np.nan],
            index=pd.MultiIndex.from_arrays(
                [pd.to_datetime(['2026-01-05', '2026-01-06', '2026-01-07']),
                 ['600519'] * 3], names=['ts', 'code']))
        target = pd.MultiIndex.from_arrays(
            [pd.to_datetime(['2026-01-06 10:00', '2026-01-07 10:00',
                             '2026-01-07 15:00']),
             ['600519'] * 3], names=['ts', 'code'])
        got = C._map_causal(day, target)
        self.assertEqual(got.iloc[0], 1.0)      # 盘中用上一交易日 10-05
        self.assertTrue(np.isnan(got.iloc[1]))  # 10-07 盘中 -> 10-06（NaN）
        self.assertTrue(np.isnan(got.iloc[2]))  # 10-07 15:00 -> 10-07（NaN）

    def test_weekend_intraday_uses_last_available_day(self):
        """周一/节后首日盘中用上一**可用**交易日，而不是「前一个自然日」。

        2026-10-26 实测：按自然日回退会落到周日（day 表没有行），于是
        `CCI_NORM` / `RSI` / `MAPOWER30_MAJOR` 在周一盘中全变 NaN。
        """
        day = pd.Series(
            [1.0, 2.0],
            index=pd.MultiIndex.from_arrays(
                [pd.to_datetime(['2026-01-09', '2026-01-12']),   # 周五 / 周一
                 ['600519'] * 2], names=['ts', 'code']))
        target = pd.MultiIndex.from_arrays(
            [pd.to_datetime(['2026-01-12 10:00', '2026-01-12 15:00']),
             ['600519'] * 2], names=['ts', 'code'])
        got = C._map_causal(day, target).tolist()
        self.assertEqual(got, [1.0, 2.0])       # 盘中用周五；15:00 用周一

    def test_1500_missing_today_stays_nan(self):
        """15:00 只认当日行：当日缺行仍是 NaN，不拿上一交易日顶替。"""
        day = pd.Series(
            [1.0],
            index=pd.MultiIndex.from_arrays(
                [pd.to_datetime(['2026-01-09']), ['600519']],
                names=['ts', 'code']))
        target = pd.MultiIndex.from_arrays(
            [pd.to_datetime(['2026-01-12 10:00', '2026-01-12 15:00']),
             ['600519'] * 2], names=['ts', 'code'])
        got = C._map_causal(day, target)
        self.assertEqual(got.iloc[0], 1.0)      # 盘中用周五
        self.assertTrue(np.isnan(got.iloc[1]))  # 15:00 当日缺行 → NaN


class TestCalcCompactFeatures(unittest.TestCase):

    def test_single_frame_has_full_contract(self):
        out = C.calc_compact_features(_min60(), include_rank=False)
        self.assertEqual(len(out), 2000)
        for col in C.COMPACT_COLUMNS:
            self.assertIn(col, out.columns)
        self.assertTrue(out[FIELD.RSI].notna().any())
        self.assertTrue(out[FIELD.MAPOWER30].notna().any())
        # 单帧没有 weekly 派生；deferred 列也必须是 NaN。
        self.assertTrue(out[FIELD.MTM_NORM_WEEKLY].isna().all())
        for col in C.DEFERRED_COLUMNS:
            self.assertTrue(out[col].isna().all(), col)

    def test_day_mode_maps_day_values_causally(self):
        m60, day = _min60(), _day()
        out = C.calc_compact_features(m60, ohlc_day=day, include_rank=False)
        day_feats = C._base_frame(day, annual=252)
        want = C._map_causal(day_feats[FIELD.RSI], m60.index)
        pd.testing.assert_series_equal(
            out[FIELD.RSI].astype(np.float32),
            want.astype(np.float32), check_names=False)
        # 日线派生列存在且不是全 NaN
        for col in (FIELD.MAGIC_NINE_TURNS_BEFORE_MAJOR,
                    FIELD.MAPOWER30_MAJOR, FIELD.MTM_NORM_MAJOR,
                    FIELD.POLYNOMIAL9_NORM_MAJOR):
            self.assertIn(col, out.columns)
            self.assertTrue(out[col].notna().any(), col)

    def test_weekly_atr_price_does_not_overflow_float16(self):
        """⚠️ 老树 `astype(float16).round(3)` 对价格 >65.5 会溢出成 inf。

        numpy 的 `round(3)` 内部要 ×1000（float16 上限 65504），所以必须
        **先在 float64 上 round、再降精度**。这里用 ~200 元的日线复现。
        """
        day = _day()
        for col in (AKA.OPEN, AKA.HIGH, AKA.LOW, AKA.CLOSE):
            day[col] = day[col] * 20
        weekly = C._weekly_derived(day)
        col = FIELD.ATR_Stopline_PRICE_WEEKLY
        vals = weekly[col].dropna()
        self.assertTrue(len(vals) > 0, '周线 ATR stopline 没有产出')
        self.assertTrue((vals > 65.5).any(), '用例价格没有覆盖溢出区间')
        self.assertTrue(np.isfinite(vals).all(),
                        'float16 round 溢出成了 inf：{}'.format(vals[~np.isfinite(vals)]))

    def test_rank_columns_are_emitted(self):
        out = C.calc_compact_features(_min60(), include_rank=True)
        for col in (PFL.STOCK_PORTFOLIO_RANK, PFL.STOCK_PORTFOLIO_RETURNS,
                    PFL.STOCK_PORTFOLIO_RANK_BEFORE,
                    PFL.STOCK_PORTFOLIO_RANK_TRACING):
            self.assertIn(col, out.columns)
        self.assertTrue(out[PFL.STOCK_PORTFOLIO_RANK].notna().any())

    def test_does_not_mutate_input(self):
        frame = _min60()
        before = list(frame.columns)
        C.calc_compact_features(frame, include_rank=False)
        self.assertEqual(list(frame.columns), before)

    def test_missing_ohlc_raises(self):
        with self.assertRaises(ValueError):
            C.calc_compact_features(_min60().drop(columns=[AKA.CLOSE]))


class TestAtr(unittest.TestCase):

    def test_columns_are_produced(self):
        out = C.calc_compact_features(_min60(), include_rank=False)
        for col in (FIELD.ATR_Stopline, FIELD.ATR_Stopline_PRICE,
                    FIELD.ATR_Stopline_TIMING_LAG, FIELD.ATR_SuperTrend,
                    FIELD.ATR_SuperTrend_TIMING_LAG):
            self.assertIn(col, out.columns)
            self.assertTrue(out[col].notna().any(), col)
        self.assertTrue(set(out[FIELD.ATR_Stopline].dropna().unique())
                        <= {-1.0, 0.0, 1.0})
        self.assertTrue(set(out[FIELD.ATR_SuperTrend].dropna().unique())
                        <= {-1.0, 1.0})

    @unittest.skipIf(_OLD_ATR is None, '老树不在场，ATR 对拍跳过')
    def test_matches_old_tree(self):
        frame = _min60(n=300)
        # 旧函数返回 (rsi_ma, stop_line, direction) —— 别把 rsi_ma 当 stop_line。
        _, old_stop_line, old_stop_dir = _OLD_ATR.ATR_RSI_Stops_v2(frame, 27)
        new_stop_line, new_stop_dir = atr_rsi_stops(frame)
        np.testing.assert_allclose(new_stop_line, old_stop_line, equal_nan=True)
        np.testing.assert_array_equal(new_stop_dir, old_stop_dir)

        old_tsl, old_trend = _OLD_ATR.ATR_SuperTrend_cross_v2(frame)
        new_tsl, new_trend = atr_supertrend(frame)
        np.testing.assert_allclose(new_tsl, old_tsl, equal_nan=True)
        np.testing.assert_array_equal(new_trend[1:], np.asarray(old_trend)[1:])


class TestPeakPoint(unittest.TestCase):

    def test_peak_columns_are_produced(self):
        out = C.calc_compact_features(_min60(), include_rank=False)
        for col in (FIELD.PEAK_LOW_TIMING_LAG, FIELD.PEAK_HIGH_TIMING_LAG,
                    FIELD.PEAK_TOWARDS_TIMING_LAG, FIELD.PEAK_TOWARDS_BEFORE,
                    FIELD.LINEAREG_PRICE, FIELD.LINEAREG_BAND_TIMING_LAG,
                    FIELD.ATR_UB, FIELD.ATR_LB,
                    'CLT_GRP_TOWARDS', 'PEAK_TOWARDS'):
            self.assertIn(col, out.columns)
            self.assertTrue(out[col].notna().any(), col)
        self.assertNotIn(FIELD.PEAK_LOW_TIMING_LAG, C.DEFERRED_COLUMNS)


class TestCompactPipeline(unittest.TestCase):

    def test_run_compact_with_codes(self):
        from GolemQ.pipeline.compact import run_compact
        m60, day = _min60(), _day()
        fake = mock.Mock()
        fake.get_kline_price_min.return_value = (
            types.SimpleNamespace(data=m60), '600519')
        fake.get_kline_price_v3.return_value = (
            types.SimpleNamespace(data=day), '600519')
        with mock.patch('GolemQ.get_active_market', return_value=fake):
            results, stats = run_compact(
                eval_range='fast', codes=['600519'], pct_of_cores=0)
        self.assertEqual(stats['success_count'], 1)
        self.assertEqual(results[0].status, 'ok')
        self.assertEqual(results[0].result['rows'], len(m60))
        self.assertEqual(results[0].result['missing'], [])

    def test_calc_features_study_entry(self):
        """Jupyter 研究入口：静态方法直接返回特征 DataFrame（或三元组）。"""
        from GolemQ.pipeline.compact import CompactPipeline
        m60, day = _min60(), _day()
        fake = mock.Mock()
        fake.get_kline_price_min.return_value = (
            types.SimpleNamespace(data=m60), '贵州茅台')
        fake.get_kline_price_v3.return_value = (
            types.SimpleNamespace(data=day), '贵州茅台')
        with mock.patch('GolemQ.get_active_market', return_value=fake):
            feat = CompactPipeline.calc_features(
                '600519', offset='2026-10-21')
            ohlc, feat_meta, name = CompactPipeline.calc_features(
                '600519', offset='2026-10-21', ret_meta=True)

        self.assertIsInstance(feat, pd.DataFrame)
        self.assertEqual(len(feat), len(m60))
        self.assertTrue(feat.index.equals(m60.index))
        for col in (FIELD.CCI_NORM, FIELD.MAPOWER30_MAJOR,
                    PFL.STOCK_PORTFOLIO_RANK):
            self.assertIn(col, feat.columns)
        # ret_meta 三元组：OHLC / 特征 / 名称，与老树 cookbook 同序
        self.assertTrue(ohlc.index.equals(m60.index))
        self.assertTrue(feat_meta.index.equals(feat.index))
        self.assertEqual(name, '贵州茅台')

    def test_calc_features_no_data(self):
        from GolemQ.pipeline.compact import CompactPipeline
        fake = mock.Mock()
        fake.get_kline_price_min.return_value = (None, '600519')
        with mock.patch('GolemQ.get_active_market', return_value=fake):
            self.assertIsNone(CompactPipeline.calc_features(
                '600519', offset='2026-10-21'))
            self.assertEqual(
                CompactPipeline.calc_features(
                    '600519', offset='2026-10-21', ret_meta=True),
                (None, None, None))


class TestCompactCli(unittest.TestCase):

    def test_registered(self):
        from GolemQ.cli.commands import COMMANDS
        self.assertIn('compact', [c.name for c in COMMANDS])

    def test_defaults_to_fast_without_eval(self):
        """用户 2026-10-21：只需要 `--strategy compact`，不给 --eval 默认 fast。"""
        from GolemQ.cli.commands import compact
        args = types.SimpleNamespace(eval_scope='', verbose=False,
                                     pct_of_cores=0, offset='')
        fake = mock.Mock(return_value=([], {
            'total_count': 0, 'success_count': 0, 'error_count': 0,
            'status_counts': {}}))
        with mock.patch('GolemQ.pipeline.compact.run_compact', fake):
            self.assertEqual(compact.run_compact_cli(args), 0)
        self.assertEqual(fake.call_args.kwargs['eval_range'], 'fast')


if __name__ == '__main__':
    unittest.main()
