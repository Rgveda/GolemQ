# coding: utf-8
"""统一 review 管线的核心不变量（不连 DB / 不跑全市场）。

钉住四件事：

1. `slice_asof` 只看 `ts <= as_of`（无未来数据）；
2. `StepRunner` 会拉齐依赖（请求 60min 时 15min 的 `minor_signals` 也要算）；
3. **前缀稳定性**：同一个 T，用截止 T 的数据算 vs 全量算，行必须逐值相同；
4. runtime 调用**零写**。
"""
from __future__ import annotations

import contextlib
import io
import unittest
import unittest.mock

import numpy as np
import pandas as pd

from GolemQ.analysis.pivot import causal_pivot_count
from GolemQ.analysis.zen import calc_zen_timing_lag
from GolemQ.core.constants import AKA, MARKET_TYPE
from GolemQ.pipeline.review.pipeline import ReviewPipeline, ReviewOptions
from GolemQ.pipeline.review.runtime import compute_runtime
from GolemQ.pipeline.review.contract import (DAY_COLUMNS, HOUR_EXTRA_COLUMNS,
                                             MINOR_COLUMNS, MINOR_15MIN_COLUMNS)
from GolemQ.pipeline.review.steps.daily import daily_snapshot_step
from GolemQ.pipeline.review.steps.hour_extra import hour_extra_step
from GolemQ.pipeline.review.steps.minor import minor_signals_step
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
        self.assertEqual(list(out.columns), list(MINOR_15MIN_COLUMNS))
        # 老树 `calc_zen_wavelet_reality_minor` 的 zen 两列必须在，且真的算出了值
        self.assertTrue(out[['zDLagMinR', 'zPLagMinR']].notna().any().all())

        # 前缀稳定性：只到 T 的帧算出的第 T 行 == 全量帧的第 T 行
        T = df.index.get_level_values(0)[-2]
        ctx_t = ReviewContext(code='600519', market_type=MARKET_TYPE.STOCK_CN,
                              frames={'15min': df}, as_of=T)
        out_t = minor_signals_step(ctx_t, target_times(df, T))
        for c in MINOR_15MIN_COLUMNS:
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


def _hour_frame(days=100, seed=41):
    """A 股 60min 交易日形状：每天 10:30 / 11:30 / 14:00 / 15:00。"""
    rng = np.random.default_rng(seed)
    rows, idx = [], []
    d = pd.Timestamp('2025-01-02')
    while len(idx) < days * 4:
        if d.weekday() < 5:
            for hm in ('10:30', '11:30', '14:00', '15:00'):
                idx.append(pd.Timestamp(f'{d.date()} {hm}'))
                rows.append(rng.normal(0, 1))
        d += pd.Timedelta(days=1)
    close = 100 + np.cumsum(rows)
    idx = pd.MultiIndex.from_arrays([idx, ['600519'] * len(idx)],
                                    names=['ts', 'code'])
    return pd.DataFrame({'open': close, 'high': close + 0.5,
                         'low': close - 0.5, 'close': close,
                         'volume': np.full(len(idx), 1e4),
                         'amount': np.full(len(idx), 1e6)}, index=idx)


class TestHourExtra(unittest.TestCase):
    def test_columns_and_causality(self):
        df = _hour_frame()
        as_of = df.index.get_level_values(0)[-1]
        ctx = ReviewContext(code='600519', market_type=MARKET_TYPE.STOCK_CN,
                            frames={'60min': df}, as_of=as_of)
        targets = target_times(df, as_of)
        out = hour_extra_step(ctx, targets)
        self.assertEqual(list(out.columns), list(HOUR_EXTRA_COLUMNS))
        self.assertEqual(len(out), len(targets))

        cnt = out['ZenPivotCnt'].to_numpy()
        self.assertTrue((cnt >= 0).all())
        self.assertTrue((np.diff(cnt) >= 0).all(), '中枢计数必须单调不减')

        # 15:00 的 SE 有值；盘中 bar 沿用上一个 15:00 的值
        times = pd.DatetimeIndex(targets).time
        se = out['SE'].to_numpy()
        self.assertTrue(np.isfinite(se[times == pd.Timestamp('15:00').time()]).any())
        prev = np.nan
        for t, v in zip(times, se):
            if t == pd.Timestamp('15:00').time():
                prev = v
            else:
                if np.isfinite(prev):
                    self.assertEqual(v, prev)

    def test_prefix_stability(self):
        df = _hour_frame()
        T = df.index.get_level_values(0)[-1]
        full = hour_extra_step(
            ReviewContext(code='600519', market_type=MARKET_TYPE.STOCK_CN,
                          frames={'60min': df}, as_of=T),
            target_times(df, T))
        cut_df = slice_asof(df, T)
        cut = hour_extra_step(
            ReviewContext(code='600519', market_type=MARKET_TYPE.STOCK_CN,
                          frames={'60min': cut_df}, as_of=T),
            target_times(cut_df, T))
        pd.testing.assert_frame_equal(full.tail(1), cut.tail(1),
                                      check_dtype=False)


class TestCausalPivotCount(unittest.TestCase):
    def test_counts_only_after_confirmation(self):
        dts = pd.date_range('2026-01-01', periods=5, freq='h')
        pivots = [{
            'start_dt': dts[1],
            'points': [{'dt': dts[1], 'known_dt': dts[2], 'xd': 10.0,
                        'fx_mark': 'g'}] * 4,
        }]
        self.assertEqual(
            causal_pivot_count(pivots, dts).tolist(), [0, 0, 1, 1, 1])


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
        # `garP` 若算出来，必须是**比值带**（~1），不能是绝对价位（D49）
        _gar = full.iloc[-1].get('garP')
        if isinstance(_gar, (list, tuple)):
            _vals = [v for pair in _gar for v in pair]
            self.assertTrue(all(abs(v) < 20 for v in _vals),
                            f'garP={_gar} 看起来是绝对价位，不是比值')

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

    def test_force_recomputes_persisted_rows(self):
        from GolemQ.pipeline.review.persistence import missing_targets
        src = pd.DatetimeIndex(['2026-01-05', '2026-01-06'])
        fake = self._fake([1.0, 2.0])          # 两天都已持久化
        with unittest.mock.patch(
                'GolemQ.markets.StockCN.reality_save.fetch_reality_feat',
                return_value=fake):
            got = missing_targets(src, '600519', 'day', MARKET_TYPE.STOCK_CN,
                                  db=object(), force=True)
        self.assertEqual(len(got), 2, 'force 必须忽略已持久化集合')

    def test_force_respects_from_date(self):
        """`force` 只忽略已持久化，**不绕过** `from_date`。"""
        from GolemQ.pipeline.review.persistence import missing_targets
        src = pd.DatetimeIndex(['2026-01-05', '2026-01-06', '2026-01-07'])
        with unittest.mock.patch(
                'GolemQ.markets.StockCN.reality_save.fetch_reality_feat',
                return_value=None):
            got = missing_targets(src, '600519', 'day', MARKET_TYPE.STOCK_CN,
                                  db=object(), force=True,
                                  from_date='2026-01-05')
        self.assertEqual([str(t.date()) for t in got],
                         ['2026-01-06', '2026-01-07'])

    def test_from_date_is_exclusive_lower_bound(self):
        from GolemQ.pipeline.review.persistence import missing_targets
        src = pd.DatetimeIndex(['2026-01-05', '2026-01-06', '2026-01-07'])
        with unittest.mock.patch(
                'GolemQ.markets.StockCN.reality_save.fetch_reality_feat',
                return_value=None):             # 表里一行都没有
            got = missing_targets(src, '600519', 'day', MARKET_TYPE.STOCK_CN,
                                  db=object(), from_date='2026-01-05')
        self.assertEqual([str(t.date()) for t in got],
                         ['2026-01-06', '2026-01-07'],
                         'from_date 当天不算，只算其后的目标')

    def test_recheck_days_reopens_recent_window(self):
        from GolemQ.pipeline.review.persistence import missing_targets
        src = pd.DatetimeIndex(pd.date_range('2026-01-01', periods=10))
        idx = pd.MultiIndex.from_tuples(
            [(d, '600519') for d in src], names=['date', 'code'])
        data = {c: [np.nan] * len(src) for c in DAY_COLUMNS}
        data[DAY_COLUMNS[0]] = [np.nan] * 10
        fake = pd.DataFrame(data, index=idx)
        # 01-01..01-05 与 01-09 已持久化；01-09 落在 recheck 窗口内，必须重算
        for d in pd.date_range('2026-01-01', '2026-01-05'):
            fake.loc[(d, '600519'), DAY_COLUMNS[0]] = 1.0
        fake.loc[(pd.Timestamp('2026-01-09'), '600519'), DAY_COLUMNS[0]] = 1.0
        with unittest.mock.patch(
                'GolemQ.markets.StockCN.reality_save.fetch_reality_feat',
                return_value=fake):
            got = missing_targets(src, '600519', 'day', MARKET_TYPE.STOCK_CN,
                                  db=object(), recheck_days=2)
        got = [str(t.date()) for t in got]
        self.assertIn('2026-01-09', got, '最近 2 天内的已持久化行应被重算')
        self.assertNotIn('2026-01-05', got, '窗口外的已持久化行不应重算')
        self.assertEqual(got[:4], ['2026-01-06', '2026-01-07',
                                   '2026-01-08', '2026-01-09'])

    def test_empty_source_returns_empty(self):
        from GolemQ.pipeline.review.persistence import missing_targets
        got = missing_targets(pd.DatetimeIndex([]), '600519', 'day',
                              MARKET_TYPE.STOCK_CN, db=object())
        self.assertEqual(len(got), 0)


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
        # 60min 的 minor/zen 现在属于合并后的 daily 档；runtime 默认只跑
        # TIER_RUNTIME，因此没有现算输出（需要时用 --runtime-heavy / 读落库行）。
        self.assertEqual(out, {})
        self.assertFalse(save.called)
        self.assertFalse(ups.called)


class TestCLI(unittest.TestCase):
    def test_parse_review(self):
        from GolemQ.cli.__main__ import build_parser
        from GolemQ.cli.commands import pick
        p = build_parser()
        a = p.parse_args(['--review', '--pct-of-cores', '32',
                          '--eval', 'fast'])
        self.assertEqual(pick(a).name, 'review-daily')
        self.assertAlmostEqual(a.pct_of_cores, 32.0)
        b = p.parse_args(['--review-runtime', '--runtime-code', '600519'])
        self.assertEqual(pick(b).name, 'review-runtime')


class TestReviewCliDefaults(unittest.TestCase):
    """`--strategy review` 的命令语义、scope 必填、pct 默认明示。

    用户 2026-10-20 明确：`--review` 等价于 `--strategy review`；`--eval` 是
    正交的 scope；`--daily` 语义不明，已删除。
    """

    STATS = {'total_count': 1, 'success_count': 1, 'error_count': 0,
             'status_counts': {}}

    def _args(self, argv):
        from GolemQ.cli.__main__ import build_parser
        return build_parser().parse_args(argv)

    def _run(self, argv):
        from GolemQ.cli.commands.review import run_review_daily
        out, err = io.StringIO(), io.StringIO()
        with unittest.mock.patch('GolemQ.pipeline.review.run_review',
                                 return_value=({}, self.STATS)) as run, \
                contextlib.redirect_stdout(out), \
                contextlib.redirect_stderr(err):
            rc = run_review_daily(self._args(argv))
        return rc, out.getvalue(), err.getvalue(), run

    def test_daily_option_was_removed(self):
        """`--daily` 语义不明，已删除；argparse 应直接拒绝。"""
        from GolemQ.cli.__main__ import build_parser
        err = io.StringIO()
        with contextlib.redirect_stderr(err), \
                self.assertRaises(SystemExit) as cm:
            build_parser().parse_args(['--daily'])
        self.assertEqual(cm.exception.code, 2)
        self.assertIn('unrecognized arguments', err.getvalue())

    def test_scope_is_required(self):
        rc, _, err, run = self._run(['--strategy', 'review'])
        self.assertEqual(rc, 2)
        self.assertIn('--eval', err)
        self.assertFalse(run.called, '缺 scope 时不许进入管线')

    def test_review_is_alias_for_strategy_review(self):
        """`--review` 与 `--strategy review` 必须分发到同一条命令。"""
        from GolemQ.cli.commands import pick
        alias = self._args(['--review', '--eval', 'fast'])
        canonical = self._args(['--strategy', 'review', '--eval', 'fast'])
        self.assertEqual(pick(alias).name, 'review-daily')
        self.assertEqual(pick(canonical).name, 'review-daily')

    def test_default_pct_is_announced(self):
        rc, _, err, run = self._run(
            ['--strategy', 'review', '--eval', 'fast'])
        self.assertEqual(rc, 0)
        self.assertIn('默认 0.32', err)
        self.assertAlmostEqual(run.call_args.kwargs['pct_of_cores'], 0.32)

    def test_explicit_pct_is_not_announced(self):
        rc, _, err, run = self._run(
            ['--strategy', 'review', '--eval', 'fast',
             '--pct-of-cores', '50'])
        self.assertEqual(rc, 0)
        self.assertNotIn('默认 0.32', err)
        self.assertAlmostEqual(run.call_args.kwargs['pct_of_cores'], 50.0)

    def test_eval_is_one_global_option(self):
        """`--eval` 是全局唯一选项：`--review` 与 `--scan` 读同一个 `eval_scope`。"""
        review = self._args(['--review', '--eval', 'fast'])
        scan = self._args(['--scan', 'ChartScan', '--eval', 'fast'])
        self.assertEqual(review.eval_scope, 'fast')
        self.assertEqual(scan.eval_scope, 'fast')
        self.assertFalse(hasattr(review, 'review_eval'))
        self.assertFalse(hasattr(review, 'scan_eval'))
        self.assertFalse(hasattr(review, 'review_codes'))

    def test_force_and_from_are_global(self):
        """`--force` / `--from` 是通用修饰符，不跟 review 绑定。"""
        review = self._args(['--strategy', 'review', '--eval', 'fast',
                             '--force', '--from', '2026-09-01'])
        self.assertTrue(review.force)
        self.assertEqual(review.from_date, '2026-09-01')
        self.assertFalse(hasattr(review, 'review_force'))
        self.assertFalse(hasattr(review, 'review_from'))
        # 同一组修饰符也能与 tdx-hosts 组合
        hosts = self._args(['--update-tdx-hosts', '--force'])
        self.assertTrue(hosts.force)

    def test_force_and_from_reach_run_review(self):
        rc, _, _, run = self._run(
            ['--strategy', 'review', '--eval', 'fast',
             '--force', '--from', '2026-09-01'])
        self.assertEqual(rc, 0)
        self.assertTrue(run.call_args.kwargs['force'])
        self.assertEqual(run.call_args.kwargs['from_date'], '2026-09-01')

    def test_eval_accepts_codes_as_second_argument(self):
        """`--eval` 的第二参数可以是逗号分隔代码；没有独立的 `--review-codes`。"""
        rc, _, err, run = self._run(
            ['--strategy', 'review', '--eval', '600519,000001'])
        self.assertEqual(rc, 0)
        self.assertNotIn('用法错误', err)
        self.assertEqual(run.call_args.kwargs['codes'], ['600519', '000001'])
        self.assertEqual(run.call_args.kwargs['eval_range'], 'fast')

    def test_eval_parses_code_name_lines(self):
        """PowerShell here-string：每行 `代码<TAB>名称`，名称里的数字不能误取。"""
        text = ('300641\t正丹股份\n'
                '605179\t一鸣食品\n'
                '001270\tST铖昌\n'
                '600519.SH\t贵州茅台\n'
                'sh000001\t上证指数')
        rc, _, err, run = self._run(
            ['--strategy', 'review', '--eval', text])
        self.assertEqual(rc, 0)
        self.assertNotIn('用法错误', err)
        self.assertEqual(run.call_args.kwargs['codes'],
                         ['300641', '605179', '001270', '600519', '000001'])

    def test_eval_unknown_text_is_usage_error(self):
        rc, _, err, run = self._run(
            ['--strategy', 'review', '--eval', 'not-a-scope'])
        self.assertEqual(rc, 2)
        self.assertIn('6 位代码', err)
        self.assertFalse(run.called)

    def test_keyboard_interrupt_returns_130(self):
        """Ctrl+C 必须干净退出（130），不能被当成运行期失败 1。"""
        from GolemQ.cli.commands.review import run_review_daily
        out, err = io.StringIO(), io.StringIO()
        with unittest.mock.patch('GolemQ.pipeline.review.run_review',
                                 side_effect=KeyboardInterrupt), \
                contextlib.redirect_stdout(out), \
                contextlib.redirect_stderr(err):
            rc = run_review_daily(
                self._args(['--strategy', 'review', '--eval', 'fast']))
        self.assertEqual(rc, 130)
        self.assertIn('已中断', err.getvalue())


class TestPipelineInterrupt(unittest.TestCase):
    """并行段必须用 `with Parallel(...)`，否则 Ctrl+C 后 worker 可能不退出。"""

    def test_parallel_is_terminated_on_keyboard_interrupt(self):
        from GolemQ.pipeline.base import BasePipeline, CodeResult

        class _Pipe(BasePipeline):
            def run_code(self, code, **kwargs):
                return CodeResult(code=code)

        exited = []

        class _FakeParallel:
            def __init__(self, n_jobs):
                self.n_jobs = n_jobs

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                exited.append(True)
                return False

            def __call__(self, tasks):
                raise KeyboardInterrupt

        with unittest.mock.patch('GolemQ.pipeline.base.Parallel', _FakeParallel), \
                unittest.mock.patch('GolemQ.pipeline.base.JOBLIB_AVAILABLE', True), \
                unittest.mock.patch.object(BasePipeline, '_setup_cpu_affinity',
                                           lambda self: None), \
                contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(KeyboardInterrupt):
                _Pipe(name='t').run_codes(['600519', '000001'], pool_size=2)

        self.assertEqual(exited, [True],
                         'KeyboardInterrupt 后必须走 Parallel.__exit__ 终止 worker')


if __name__ == '__main__':
    unittest.main()
