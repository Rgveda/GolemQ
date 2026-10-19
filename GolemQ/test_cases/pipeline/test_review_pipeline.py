# coding: utf-8
"""统一 review 管线的核心不变量（不连 DB / 不跑全市场）。

钉住四件事：

1. `slice_asof` 只看 `ts <= as_of`（无未来数据）；
2. `StepRunner` 会拉齐依赖（请求 60min 时 15min 的 `minor_signals` 也要算）；
3. **前缀稳定性**：同一个 T，用截止 T 的数据算 vs 全量算，行必须逐值相同；
4. runtime 调用**零写**。
"""
from __future__ import annotations

import unittest
import unittest.mock

import numpy as np
import pandas as pd

from GolemQ.analysis.zen import calc_zen_timing_lag
from GolemQ.core.constants import AKA, MARKET_TYPE
from GolemQ.pipeline.review.pipeline import ReviewPipeline, ReviewOptions
from GolemQ.pipeline.review.runtime import compute_runtime
from GolemQ.pipeline.review.contract import DAY_COLUMNS, MINOR_COLUMNS
from GolemQ.pipeline.review.steps.daily import daily_snapshot_step
from GolemQ.pipeline.review.steps.minor import (hour_minor_sample_step,
                                                minor_signals_step)
from GolemQ.pipeline.steps import (FeatureStep, ReviewContext, StepRunner,
                                   TIER_RUNTIME, slice_asof, target_times)


def _ohlc(n, freq='1D', seed=0, start='2024-01-01', code='600519'):
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(0, 0.8, n))
    ts = pd.date_range(start, periods=n, freq=freq)
    idx = pd.MultiIndex.from_arrays([ts, [code] * n], names=['ts', 'code'])
    return pd.DataFrame({
        'open': close, 'high': close + 0.5, 'low': close - 0.5,
        'close': close, 'volume': np.full(n, 1e4),
    }, index=idx)


class TestSliceAsof(unittest.TestCase):
    def test_no_future(self):
        df = _ohlc(10)
        out = slice_asof(df, df.index.get_level_values(0)[4])
        self.assertEqual(len(out), 5)
        self.assertLessEqual(out.index.get_level_values(0).max(),
                             df.index.get_level_values(0)[4])

    def test_tail_n(self):
        df = _ohlc(10)
        out = slice_asof(df, df.index.get_level_values(0)[-1], n_bars=3)
        self.assertEqual(len(out), 3)
        self.assertEqual(out.index.get_level_values(0)[-1],
                         df.index.get_level_values(0)[-1])


class TestStepRunnerDeps(unittest.TestCase):
    def test_dependency_included(self):
        df = _ohlc(40, freq='15min')
        ctx = ReviewContext(code='600519', market_type=MARKET_TYPE.STOCK_CN,
                            frames={'15min': df, '60min': df},
                            as_of=df.index.get_level_values(0)[-1])
        ctx.targets['15min'] = target_times(df, ctx.as_of)
        ctx.targets['60min'] = ctx.targets['15min']

        calls = []

        def a(ctx, targets):
            calls.append('a')
            return pd.DataFrame({'x': [1.0] * len(targets)},
                                index=pd.MultiIndex.from_tuples(
                                    [(t, ctx.code) for t in targets],
                                    names=['ts', 'code']))

        def b(ctx, targets):
            calls.append('b')
            return pd.DataFrame({'y': [2.0] * len(targets)},
                                index=pd.MultiIndex.from_tuples(
                                    [(t, ctx.code) for t in targets],
                                    names=['ts', 'code']))

        steps = [
            FeatureStep('a', TIER_RUNTIME, '15min', '15min', ('x',), a),
            FeatureStep('b', TIER_RUNTIME, '60min', '60min', ('y',), b, deps=('a',)),
        ]
        StepRunner().run(ctx, steps, families=('60min',))
        self.assertEqual(calls, ['a', 'b'])


class TestDailyStep(unittest.TestCase):
    def test_columns_and_prefix_stability(self):
        df = _ohlc(180, freq='B', seed=3)
        as_of = df.index.get_level_values(0)[-1]
        pipe = ReviewPipeline(options=ReviewOptions(families=('day',)))
        ctx = ReviewContext(code='600519', market_type=MARKET_TYPE.STOCK_CN,
                            frames={'day': df}, as_of=as_of)
        targets = target_times(df, as_of)[-3:]
        full = daily_snapshot_step(ctx, targets)
        self.assertEqual(list(full.columns), list(DAY_COLUMNS))
        self.assertEqual(len(full), len(targets))

        ctx2 = ReviewContext(code='600519', market_type=MARKET_TYPE.STOCK_CN,
                             frames={'day': df}, as_of=targets[-1])
        one = daily_snapshot_step(ctx2, targets[-1:])
        pd.testing.assert_frame_equal(
            one, full.iloc[[-1]], check_names=False)


class TestRatioColumns(unittest.TestCase):
    """价格导出的列必须是 **close 相对量**，不能是绝对价位（复权断层）。"""

    def test_day_price_columns_are_ratios(self):
        # >450 根才会算 RegTree（否则 rTrePrcMajR 是 NaN）
        day = _ohlc(600, freq='B', seed=31)
        as_of = day.index.get_level_values(0)[-1]
        ctx = ReviewContext(code='600519', market_type=MARKET_TYPE.STOCK_CN,
                            frames={'day': day}, as_of=as_of)
        out = daily_snapshot_step(ctx, [as_of])
        close = float(day[AKA.CLOSE].iloc[-1])
        for col in ('poly9MajR', 'poly9WekR', 'rTrePrcMajR'):
            v = float(out.iloc[0][col])
            self.assertTrue(np.isfinite(v), col)
            self.assertLess(abs(v), 10.0,
                            f'{col}={v} 看起来是绝对价（close={close}），不是比值')
        self.assertLess(abs(float(out.iloc[0]['rTreSlopMajR'])), 10.0)


class TestMinorStep(unittest.TestCase):
    def test_columns_and_prefix_stability(self):
        df = _ohlc(320, freq='15min', seed=5)
        as_of = df.index.get_level_values(0)[-1]
        ctx = ReviewContext(code='600519', market_type=MARKET_TYPE.STOCK_CN,
                            frames={'15min': df}, as_of=as_of)
        out = minor_signals_step(ctx, target_times(df, as_of))
        self.assertEqual(list(out.columns), list(MINOR_COLUMNS))

        # 前缀稳定性：只到 T 的帧算出的第 T 行 == 全量帧的第 T 行
        T = df.index.get_level_values(0)[-2]
        ctx_t = ReviewContext(code='600519', market_type=MARKET_TYPE.STOCK_CN,
                              frames={'15min': df}, as_of=T)
        out_t = minor_signals_step(ctx_t, target_times(df, T))
        for c in MINOR_COLUMNS:
            a = float(out.loc[(T, '600519'), c])
            b = float(out_t.loc[(T, '600519'), c])
            self.assertTrue((np.isnan(a) and np.isnan(b)) or a == b,
                            msg=f'{c}: {a} vs {b}')


class TestZen(unittest.TestCase):
    def test_prefix_stability(self):
        df = _ohlc(300, freq='60min', seed=7)
        T = df.index.get_level_values(0)[-3]
        full = calc_zen_timing_lag(df)
        cut = calc_zen_timing_lag(slice_asof(df, T))
        for c in (FLD_ZEN_DASH := 'ZEN_D_LAG', 'ZEN_P_LAG'):
            self.assertEqual(float(full.loc[(T, '600519'), c]),
                             float(cut.loc[(T, '600519'), c]))


class TestColdstart(unittest.TestCase):
    def test_columns_and_prefix_stability(self):
        from GolemQ.pipeline.review.coldstart import coldstart_step
        from GolemQ.pipeline.review.contract import COLDSTART_COLUMNS
        day = _ohlc(300, freq='B', seed=21)
        targets = pd.DatetimeIndex(pd.date_range('2025-01-02 10:30', periods=10,
                                                 freq='1D'))
        ctx = ReviewContext(code='600519', market_type=MARKET_TYPE.STOCK_CN,
                            frames={'day': day}, as_of=day.index.get_level_values(0)[-1])
        full = coldstart_step(ctx, targets)
        self.assertEqual(list(full.columns), list(COLDSTART_COLUMNS))
        self.assertEqual(len(full), len(targets))

        T = targets[3]
        ctx_t = ReviewContext(code='600519', market_type=MARKET_TYPE.STOCK_CN,
                              frames={'day': day}, as_of=T)
        one = coldstart_step(ctx_t, targets[:4])
        pd.testing.assert_frame_equal(one.iloc[[-1]], full.iloc[[3]],
                                      check_dtype=False)


class TestRouting(unittest.TestCase):
    def test_reality_feat_name(self):
        from GolemQ.markets.StockCN.reality_save import reality_feat_name
        self.assertEqual(reality_feat_name('day', MARKET_TYPE.STOCK_CN),
                         'stock_reality_feat_day')
        self.assertEqual(reality_feat_name('60min', MARKET_TYPE.ETF_CN),
                         'etf_reality_feat_60min')
        self.assertEqual(reality_feat_name('15min', MARKET_TYPE.INDEX_CN),
                         'index_reality_feat_15min')
        self.assertEqual(reality_feat_name('day', MARKET_TYPE.FUND_CN),
                         'index_reality_feat_day')


class TestWriteNulls(unittest.TestCase):
    def test_review_can_clear_stale_values(self):
        from GolemQ.markets.StockCN.reality_save import reality_feat_rows
        idx = pd.MultiIndex.from_tuples(
            [(pd.Timestamp('2026-01-05'), '600519')], names=['ts', 'code'])
        df = pd.DataFrame({'x': [np.nan], 'y': [1.0]}, index=idx)
        default = reality_feat_rows(df, '60min', created_at=0)
        self.assertNotIn('x', default[0])
        with_nulls = reality_feat_rows(df, '60min', created_at=0, write_nulls=True)
        self.assertIn('x', with_nulls[0])
        self.assertIsNone(with_nulls[0]['x'])


class TestPersistence(unittest.TestCase):
    """`persistence_features`：按**目标表已有哪些 review 行**决定还算哪些 bar。"""

    def _fake(self, values):
        from GolemQ.pipeline.review.contract import DAY_COLUMNS
        idx = pd.MultiIndex.from_tuples(
            [(pd.Timestamp(d), '600519') for d in ('2026-01-05', '2026-01-06')],
            names=['date', 'code'])
        data = {c: [np.nan, np.nan] for c in DAY_COLUMNS}
        data[DAY_COLUMNS[0]] = values
        return pd.DataFrame(data, index=idx)

    def test_skips_persisted_and_cv_only_rows(self):
        from GolemQ.pipeline.review.persistence import missing_targets
        src = pd.DatetimeIndex(['2026-01-05', '2026-01-06', '2026-01-07'])
        fake = self._fake([1.0, np.nan])       # 01-05 有 review 列；01-06 只有 CV/空
        with unittest.mock.patch(
                'GolemQ.markets.StockCN.reality_save.fetch_reality_feat',
                return_value=fake):
            got = missing_targets(src, '600519', 'day', MARKET_TYPE.STOCK_CN,
                                  db=object())
        self.assertEqual([str(t.date()) for t in got],
                         ['2026-01-06', '2026-01-07'])

    def test_min_columns_is_parameterized(self):
        from GolemQ.pipeline.review.persistence import missing_targets
        src = pd.DatetimeIndex(['2026-01-05', '2026-01-06'])
        fake = self._fake([1.0, 2.0])
        with unittest.mock.patch(
                'GolemQ.markets.StockCN.reality_save.fetch_reality_feat',
                return_value=fake):
            got = missing_targets(src, '600519', 'day', MARKET_TYPE.STOCK_CN,
                                  db=object(), min_columns=2)
        self.assertEqual(len(got), 2, '只有 1 列非空 ⇒ 不满足 min_columns=2')


class TestDbInjection(unittest.TestCase):
    """落库/水位必须**接收调用方传入的 DB 实例**，不能自己取库名句柄。"""

    def test_save_uses_passed_db(self):
        from GolemQ.markets.StockCN import reality_save as rs
        idx = pd.MultiIndex.from_tuples(
            [(pd.Timestamp('2026-01-05'), '600519')], names=['ts', 'code'])
        df = pd.DataFrame({'x': [1.0]}, index=idx)
        fake_db = {'stock_reality_feat_day': object()}
        with unittest.mock.patch('GolemQ.datasource.writer.upsert_fields',
                                 return_value={'upserted': 0, 'modified': 0}) as up, \
                unittest.mock.patch.object(rs, '_db',
                                           side_effect=AssertionError('must not call _db')):
            rs.save_reality_feat(df, 'day', MARKET_TYPE.STOCK_CN,
                                 verbose=False, db=fake_db)
        self.assertTrue(up.called)

    def test_watermark_uses_passed_db(self):
        from GolemQ.markets.StockCN import reality_save as rs
        fake_coll = unittest.mock.MagicMock()
        fake_coll.find_one.return_value = None
        fake_db = unittest.mock.MagicMock()
        fake_db.__getitem__.return_value = fake_coll
        with unittest.mock.patch.object(rs, '_db',
                                        side_effect=AssertionError('must not call _db')):
            self.assertIsNone(rs.get_watermark('review', 't', '600519', db=fake_db))


class TestRestorePrice(unittest.TestCase):
    def test_ratio_columns_multiply_back_by_close(self):
        from GolemQ.markets.StockCN.reality_save import restore_price_columns
        idx = pd.MultiIndex.from_tuples(
            [(pd.Timestamp('2026-01-05'), '600519')], names=['ts', 'code'])
        df = pd.DataFrame({'poly9MajR': [1.01], 'rTreSlopMajR': [0.002],
                           'poly9NorMajR': [0.3]}, index=idx)
        close = pd.Series([100.0], index=idx)
        out = restore_price_columns(df, close)
        self.assertAlmostEqual(out['poly9MajR'].iloc[0], 101.0)
        self.assertAlmostEqual(out['rTreSlopMajR'].iloc[0], 0.2)
        self.assertAlmostEqual(out['poly9NorMajR'].iloc[0], 0.3)   # 归一化列不动


class TestRuntimeNoWrites(unittest.TestCase):
    def test_zero_writes(self):
        day = _ohlc(200, freq='B', seed=11)
        min60 = _ohlc(200, freq='60min', seed=12)
        min15 = _ohlc(400, freq='15min', seed=13)
        frames = {'day': day, '60min': min60, '15min': min15}
        as_of = pd.Timestamp('2025-01-01')

        with unittest.mock.patch.object(ReviewPipeline, '_fetch_frames',
                                        return_value=frames), \
                unittest.mock.patch('GolemQ.markets.StockCN.reality_save.save_reality_feat') as save, \
                unittest.mock.patch('GolemQ.datasource.writer.upsert_fields') as ups, \
                unittest.mock.patch('GolemQ.markets.StockCN.reality_save.fetch_reality_feat',
                                    return_value=None):
            out = compute_runtime('600519', as_of=as_of, families=('60min',))
        self.assertIn('60min', out)
        self.assertFalse(save.called)
        self.assertFalse(ups.called)


class TestCLI(unittest.TestCase):
    def test_parse_review(self):
        from GolemQ.cli.__main__ import build_parser
        from GolemQ.cli.commands import pick
        p = build_parser()
        a = p.parse_args(['--review', '--daily', '--pct-of-cores', '32',
                          '--eval', 'fast'])
        self.assertEqual(pick(a).name, 'review-daily')
        self.assertAlmostEqual(a.pct_of_cores, 32.0)
        b = p.parse_args(['--review-runtime', '--runtime-code', '600519'])
        self.assertEqual(pick(b).name, 'review-runtime')


if __name__ == '__main__':
    unittest.main()
