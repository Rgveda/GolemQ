# coding:utf-8
"""`analysis/cv/predict.py` 的**写库路由**用例（`DECISIONS.md` D51）。

为什么单独钉这个
================
那几条写入分支是逐字搬自旧树 `GolemAI/GolemQ/CV/charts.py`，而 **D42 的对拍只覆盖了
`freq='day'`**。15min / 60min 两条支路在合并时被改写过，症状**完全静默**：

* 日族永远收不到 `CoordinatesMin` / `Coordinates` ⇒ `calc_ChartScaned_persistence`
  的 `all_nat_min` 恒真 ⇒ **每天全量重扫 618 交易日 × 3 频率**；
* 整个写块还包在 `try/except: traceback.print_exc()` 里 ⇒ 写错了也不报错。

本用例**不跑真 YOLO**（要 GolemAI env 的 torch / ultralytics），只钉
「**写哪张表、用哪个键**」。模型、出图、取数三处全部替身。
"""
from __future__ import annotations

import os, sys
try:
    import GolemQ
except ImportError:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

import io
import types
import unittest
from unittest import mock

import numpy as np
import pandas as pd

from GolemQ.core.constants import AKA, MARKET_TYPE

_PREDICT = 'GolemQ.analysis.cv.predict'
_REALITY = 'GolemQ.markets.StockCN.reality_save'

def _x_for_bar(i):
    """像素 x，使 `calculate_i(x)` 落到 `tail(chunk)` 的第 `i` 根（`i` 可负 = 从尾部倒数）。

    `calculate_i(x) = int((x - 386) / 6.5) - 1`（照旧树 `charts.py:121`）；
    取 `x = 386 + 6.5*(i+1) + 3` 时 `(x-386)/6.5 = i + 1.46` ⇒ 下取整再 −1 恰好是 `i`。
    """
    return 386 + 6.5 * (i + 1) + 3.0


#: 最后一根 bar（`i=179`；`calculate_i` 自带 clamp，恰好落在 `chunk_size-1`）
_X_LAST_BAR = _x_for_bar(179)
#: 往回第 137 根 —— 分钟族按 **6 小时/根** 造，137×6h ≈ 34 天 > 31 天 ⇒ 触发「太远」支路
_X_FAR_BAR = _x_for_bar(-137)


class _Box:
    """YOLO 框的最小替身：`predict.py` 只读 `.xyxy[0]` / `.cls[0]` / `.conf[0].item()`。"""

    def __init__(self, x, cls=0, conf=0.87):
        self.xyxy = [[x, 0.0, x + 6.5, 10.0]]
        self.cls = [cls]
        self.conf = [np.float32(conf)]


class _Result:
    names = {0: 'Buy'}

    def __init__(self, boxes):
        self.boxes = boxes


class _Model:
    def __init__(self, boxes):
        self._boxes = boxes

    def predict(self, image, conf=None):        # noqa: ARG002 签名照 ultralytics
        return [_Result(self._boxes)]


def _kline(freq, last_ts, n=200):
    """造一份够长（≥ `CHUNK_SIZE`）的行情帧。

    分钟族用 **6 小时**间隔 —— 真实 60min 窗口是 126 天 / 180 根（≈0.7 天/根），
    这里需要「往回 137 根就超过 31 天」，6 小时/根正好够（137×6h ≈ 34 天）。
    """
    if freq == 'day':
        ts = pd.date_range(end=pd.Timestamp(last_ts).normalize(), periods=n, freq='D')
    else:
        ts = pd.date_range(end=pd.Timestamp(last_ts), periods=n, freq='6h')
    idx = pd.MultiIndex.from_arrays([ts, ['600519'] * n], names=['ts', 'code'])
    close = np.linspace(100.0, 200.0, n)
    return pd.DataFrame({
        AKA.OPEN: close, 'high': close + 1, 'low': close - 1,
        AKA.CLOSE: close, 'volume': np.full(n, 1000.0),
    }, index=idx)


class _Recorder:
    """替掉 `reality_save.save_reality_feat`，把「写哪族 / 带什么列 / 什么索引」记下来。"""

    def __init__(self):
        self.calls = []

    def __call__(self, features, family, market_type, verbose=True,   # noqa: ARG002
                 echo=None, created_at=None, db=None):
        from GolemQ.markets.StockCN.reality_save import (reality_feat_name,
                                                         reality_feat_rows)
        # 走**真**组行函数：这样断言的是「**写进库的字段集**」，而不是调用方传进来的列。
        # `time_stamp` / `date_stamp` 都是 `reality_feat_rows` 里现算的，不在这里看不出。
        docs = reality_feat_rows(features, family, created_at=0)
        self.calls.append({
            'name': reality_feat_name(family, market_type),
            'family': family,
            'docs': docs,
            'fields': sorted(docs[0]) if docs else [],
            'index': list(features.index.names),
        })
        return {'name': self.calls[-1]['name'], 'rows': len(features),
                'written': len(features), 'upserted': 1, 'modified': 0}

    @property
    def names(self):
        return [c['name'] for c in self.calls]


class _ChartScanWriteTest(unittest.TestCase):
    """跑 `GQ_predict_ChartScanAI` 的完整写路径，全程替身。"""

    def setUp(self):
        from GolemQ.analysis.cv import predict
        self.predict = predict
        self.rec = _Recorder()

        png = io.BytesIO()
        from PIL import Image
        Image.new('RGB', (4, 4)).save(png, format='PNG')
        png.seek(0)

        patchers = [
            mock.patch.object(predict, 'load_model', lambda m=None: _Model([_Box(_X_LAST_BAR)])),
            mock.patch.object(predict, 'GQ_generate_chart_stock', lambda **kw: png),
            mock.patch.object(predict, 'traceback'),
            mock.patch(_REALITY + '.save_reality_feat', self.rec),
        ]
        for p in patchers:
            p.start()
            self.addCleanup(p.stop)

    def _run(self, freq, last_ts, market_type=MARKET_TYPE.STOCK_CN, end_date=None,
             boxes=None):
        kline = _kline(freq, last_ts)
        model = _Model(boxes if boxes is not None else [_Box(_X_LAST_BAR)])
        with mock.patch.object(self.predict, '_fetch', lambda *a, **kw: (kline, '贵州茅台')), \
                mock.patch.object(self.predict, 'load_model', lambda m=None: model):
            self.predict.GQ_predict_ChartScanAI(
                code='600519', market_type=market_type, freq=freq,
                end_date=pd.Timestamp(end_date or last_ts), verbose=False)

    # ---- 「探到且够近」：三频率各写哪几张表 --------------------------------

    def test_day_writes_day_family_only(self):
        self._run('day', '2026-10-16')
        self.assertEqual(self.rec.names, ['stock_reality_feat_day'])

    def test_60min_writes_day_and_60min(self):
        """⚠️ 旧树 `:406`/`:412` 两笔 —— 漏掉日族那笔会让重扫判定恒真。"""
        self._run('60min', '2026-10-16 15:00')
        self.assertEqual(self.rec.names,
                         ['stock_reality_feat_day', 'stock_reality_feat_60min'])

    def test_60min_writes_60min_even_off_close_time(self):
        """60min 那笔**没有**收盘时刻闸（旧树 `:411-414` 无条件）。"""
        self._run('60min', '2026-10-16 14:30')
        self.assertEqual(self.rec.names,
                         ['stock_reality_feat_day', 'stock_reality_feat_60min'])

    def test_15min_on_close_time_writes_three(self):
        """15min 在收盘时刻：日族 + 60min + 15min（旧树 `:363`/`:370`/`:374`）。"""
        self._run('15min', '2026-10-16 15:00')
        self.assertEqual(self.rec.names,
                         ['stock_reality_feat_day', 'stock_reality_feat_60min',
                          'stock_reality_feat_15min'])

    def test_15min_off_close_time_writes_two(self):
        """不在收盘时刻：**不写** 60min（旧树那个 `if` 闸）。"""
        self._run('15min', '2026-10-16 14:30')
        self.assertEqual(self.rec.names,
                         ['stock_reality_feat_day', 'stock_reality_feat_15min'])

    def test_minute_payload_carries_datetime_and_stamp(self):
        """日族那笔**只带 date_stamp**；分钟族那笔带 `datetime` + `time_stamp`。

        ⚠️ 这是**键的来处**：分钟族的唯一键是 `(code, time_stamp)`，而 `time_stamp`
        只从索引 level 0 现算 ⇒ 索引传错（date 而不是 datetime）会**静默**把 4 根
        60min bar 全压到同一天的零点上，并另插一行垃圾。
        """
        self._run('60min', '2026-10-16 15:00')
        day, minute = self.rec.calls
        self.assertNotIn('time_stamp', day['fields'])
        self.assertIn('date_stamp', day['fields'])
        self.assertEqual(day['index'], ['date', 'code'])
        self.assertIn('time_stamp', minute['fields'])
        self.assertIn('datetime', minute['fields'])
        self.assertEqual(minute['index'], ['datetime', 'code'])
        # 分钟族的 `time_stamp` 必须是 **bar 的时刻**（15:00 北京 = 07:00Z），不是零点
        want = int(pd.Timestamp('2026-10-16 15:00', tz='Asia/Shanghai').timestamp())
        self.assertEqual(minute['docs'][0]['time_stamp'], want)
        self.assertEqual(day['docs'][0]['date_stamp'],
                         int(pd.Timestamp('2026-10-16', tz='Asia/Shanghai').timestamp()))

    def test_minute_closest_uses_close_time_anchor(self):
        """分钟频率的「最近框」以 **15:00** 为锚 ⇒ 选当天**最晚**的检出。

        锚点若是 `end_date`（00:00），`09:00` 比 `15:00` 更近 ⇒ 永远选最早的框。
        `_kline` 的分钟帧按 6 小时一根造，最后两根正好是 09:00 / 15:00。
        """
        self._run('15min', '2026-10-16 15:00',
                  boxes=[_Box(_x_for_bar(-2)), _Box(_x_for_bar(-1))])
        doc = self.rec.calls[0]['docs'][0]        # 日族那笔带 `CoordinatesMin`
        want = int(pd.Timestamp('2026-10-16 15:00', tz='Asia/Shanghai').timestamp()) * 1000
        self.assertEqual(doc['CoordinatesMin'], want)

    # ---- 逐码路由（D51 的 `etf_` 前缀）-----------------------------------

    def test_market_type_routes_prefix(self):
        """逐码路由：`stock_cn → stock_*` / `etf_cn → etf_*` / `index_cn → index_*`。

        2026-10-18 端到端实测过 ETF（510300）：三族全部落在 `etf_*`，
        日族文档带齐 9 列 CV（`ClsMaj`/`ClsMin`/`Class` 三套 + 各自 Coordinates）。
        """
        for mt, prefix in ((MARKET_TYPE.STOCK_CN, 'stock'),
                           (MARKET_TYPE.INDEX_CN, 'index'),
                           (MARKET_TYPE.ETF_CN, 'etf')):
            with self.subTest(market_type=mt):
                self.rec.calls.clear()
                self._run('60min', '2026-10-16 15:00', market_type=mt)
                self.assertEqual(self.rec.names,
                                 [f'{prefix}_reality_feat_day',
                                  f'{prefix}_reality_feat_60min'])

    def test_market_type_routes_prefix_all_families(self):
        """三个频率都要按 market_type 路由（不只 day）。"""
        for freq, last in (('day', '2026-10-16'), ('60min', '2026-10-16 15:00'),
                           ('15min', '2026-10-16 15:00')):
            with self.subTest(freq=freq):
                self.rec.calls.clear()
                self._run(freq, last, market_type=MARKET_TYPE.ETF_CN)
                self.assertTrue(self.rec.names, msg=freq)
                for name in self.rec.names:
                    self.assertTrue(name.startswith('etf_'), name)

    # ---- 「探到但太远」：标记路径 -----------------------------------------

    def test_far_day_writes_nothing(self):
        """日族「探到但太远」**一笔都不写** —— 旧树 `:353` 那句 print 之后直接返回。

        （不像分钟族会留 `ChartScaned` 标记：日族的「没探到」只在**一个框都没有**时标记。）
        """
        self._run('day', '2026-10-16', boxes=[_Box(_X_FAR_BAR)])
        self.assertEqual(self.rec.names, [])

    def test_far_60min_writes_60min_and_daily_markup(self):
        self._run('60min', '2026-10-16 15:00', boxes=[_Box(_X_FAR_BAR)])
        self.assertEqual(self.rec.names,
                         ['stock_reality_feat_60min', 'stock_reality_feat_day'])

    def test_far_15min_writes_both_minute_markups_and_daily(self):
        """⚠️ 旧树 `:380`+`:385`：15min 这条**两张分钟表都写**，第二张无闸。"""
        self._run('15min', '2026-10-16 15:00', boxes=[_Box(_X_FAR_BAR)])
        self.assertEqual(self.rec.names,
                         ['stock_reality_feat_60min', 'stock_reality_feat_15min',
                          'stock_reality_feat_day'])

    def test_far_15min_off_close_time_skips_60min(self):
        self._run('15min', '2026-10-16 14:30', boxes=[_Box(_X_FAR_BAR)])
        self.assertEqual(self.rec.names,
                         ['stock_reality_feat_15min', 'stock_reality_feat_day'])

    # ---- 护栏：写块外的 `except` 不能把断言变成「碰巧过」------------------

    def test_real_traceback_never_fires(self):
        """写入路径不许靠 `except` 兜底 —— 兜住了就说明前面某处已经抛了。"""
        from GolemQ.analysis.cv import predict
        with mock.patch.object(predict, 'traceback') as tb:
            self._run('15min', '2026-10-16 15:00')
            tb.print_exc.assert_not_called()


class TestPersistenceReadsAcrossFamilies(unittest.TestCase):
    """**读**的一侧：重扫判定读**日族**、按 `market_type` 取表，并把三套坐标一起转时间戳。

    `calc_ChartScaned_persistence` 依赖日族里的 `Coordinates`（60min）/
    `CoordinatesMaj`（day）/ `CoordinatesMin`（15min）—— 三套缺一套就会误判成全扫
    （2026-10-18 实测：迁移丢列时 36/40 变成全扫，补齐后 0/40）。
    """

    @staticmethod
    def _frame():
        idx = pd.MultiIndex.from_tuples(
            [(pd.Timestamp('2026-10-16'), '510300')], names=['date', 'code'])
        return pd.DataFrame({
            'Coordinates': [1792134000000.0],
            'CoordinatesMaj': [1790697600000.0],
            'CoordinatesMin': [1792134000000.0],
            'ChartScaned': [2],
        }, index=idx)

    def test_reads_day_family_with_market_type(self):
        from GolemQ.analysis.cv import predict
        with mock.patch('GolemQ.markets.StockCN.reality_save.fetch_reality_feat',
                        return_value=self._frame()) as fetch:
            predict.calc_ChartScaned_persistence('510300', market_type='etf_cn')
        args = fetch.call_args[0]
        self.assertEqual(args[3], 'day', '重扫判定必须读日族（三族结果都在那儿）')
        self.assertEqual(args[4], 'etf_cn', '必须按市场类型取表，否则静默读空')

    def test_coords_are_converted_to_datetime(self):
        from GolemQ.analysis.cv.predict import _coords_to_datetime
        out = _coords_to_datetime(self._frame())
        for col in ('Coordinates', 'CoordinatesMaj', 'CoordinatesMin'):
            self.assertTrue(pd.api.types.is_datetime64_any_dtype(out[col]), col)

    def test_missing_coord_columns_do_not_raise(self):
        """缺列的旧文档不能炸 —— 缺的那套按 None 处理（否则整只标的读不动）。"""
        from GolemQ.analysis.cv.predict import _coords_to_datetime
        frame = self._frame().drop(columns=['Coordinates', 'CoordinatesMin'])
        out = _coords_to_datetime(frame)
        self.assertIn('Coordinates', out.columns)
        self.assertIn('CoordinatesMin', out.columns)


class TestHeadlessRendering(unittest.TestCase):
    """⚠️ 多线程服务端 + matplotlib **GUI 后端** = 每个标的都静默渲染失败。

    2026-10-18 真机踩到：`server.main` 每个端口一个线程，TkAgg 在非主线程里
    `plt.figure()` 先告警、`plt.close()` 抛
    `RuntimeError: main thread is not in main loop`；而 `predict` 的渲染段
    被 `except Exception: print_exc()` 兜着 ⇒ **一条结果都写不进库，还不中断**。
    """

    def test_mpbackend_is_forced_to_agg(self):
        import os
        from GolemQ.analysis.cv import render  # noqa: F401 导入即设
        self.assertEqual(os.environ.get('MPLBACKEND'), 'Agg')

    def test_render_module_sets_it_at_import(self):
        """不是靠调用方传 —— 模块**导入**那一刻就得钉住（matplotlib 可能已被别人先导入）。"""
        from pathlib import Path
        src = Path(__file__).resolve().parents[2] / 'analysis/cv/render.py'
        text = src.read_text(encoding='utf-8')
        self.assertIn("os.environ['MPLBACKEND'] = 'Agg'", text)
        self.assertIn("matplotlib.use('Agg', force=True)", text)


class TestClosestAnchor(unittest.TestCase):
    """分钟频率挑「最近框」的锚点（`_closest_anchor`）。"""

    def test_minute_on_trading_day_uses_close(self):
        from GolemQ.analysis.cv import predict
        for freq in ('15min', '60min'):
            with self.subTest(freq=freq):
                self.assertEqual(
                    predict._closest_anchor(pd.Timestamp('2026-10-16'), freq),
                    pd.Timestamp('2026-10-16 15:00'))

    def test_day_keeps_date(self):
        from GolemQ.analysis.cv import predict
        self.assertEqual(
            predict._closest_anchor(pd.Timestamp('2026-10-16'), 'day'),
            pd.Timestamp('2026-10-16'))

    def test_non_trading_day_keeps_date(self):
        """2026-10-17 是周六 —— 拿不到收盘锚点时保持原行为。"""
        from GolemQ.analysis.cv import predict
        self.assertEqual(
            predict._closest_anchor(pd.Timestamp('2026-10-17'), '60min'),
            pd.Timestamp('2026-10-17'))


class _MarkupFamilyTest(unittest.TestCase):
    """`GQ_markup_ChartScaned` 的 `family=` 参数（旧树的 `collections=` 对应物）。"""

    def test_family_param_routes(self):
        from GolemQ.analysis.cv import predict
        from GolemQ.markets.StockCN.reality_save import reality_feat_name
        kline = _kline('60min', '2026-10-16 15:00')
        rec = _Recorder()
        with mock.patch(_REALITY + '.save_reality_feat', rec):
            for fam in ('60min', '15min'):
                predict.GQ_markup_ChartScaned('600519', kline, MARKET_TYPE.STOCK_CN,
                                              family=fam)
                self.assertEqual(rec.names[-1], reality_feat_name(fam, MARKET_TYPE.STOCK_CN))

    def test_default_family_is_60min(self):
        """默认值必须**保持旧默认行为**（旧签名默认就是 60min 那张表）。"""
        from GolemQ.analysis.cv import predict
        kline = _kline('60min', '2026-10-16 15:00')
        rec = _Recorder()
        with mock.patch(_REALITY + '.save_reality_feat', rec):
            predict.GQ_markup_ChartScaned('600519', kline, MARKET_TYPE.STOCK_CN)
        self.assertEqual(rec.names, ['stock_reality_feat_60min'])


def _day_frame(dates, code='600519'):
    """`(ts, code)` 两层索引的日线帧（`_fetch` 的返回形状）。"""
    idx = pd.MultiIndex.from_arrays(
        [pd.to_datetime(dates), [code] * len(dates)], names=['ts', 'code'])
    return pd.DataFrame({AKA.CLOSE: np.arange(len(dates), dtype=float)}, index=idx)


class _FakeMarket:
    """只实现 `_fetch` 用到的日线读口，按顺序弹出预置响应。"""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def get_kline_price_v3(self, code, start=None, end=None, realtime=True,
                           if_fq='qfq', verbose=False):   # noqa: ARG002
        self.calls.append({'start': start, 'end': end, 'realtime': realtime})
        return self.responses.pop(0)


class TestDayBaselineTodayFallback(unittest.TestCase):
    """单日窗口 `[今天, 今天]` 的日线回退（2026-10-21 盘中 no-op 的回归）。

    背景：`stock_day` 里今天那根要等收盘 `--save` 才有，而 `get_kline_price_v3`
    先判 `df.empty` 再合并实时 ⇒ 单日窗口恒 `None` ⇒ 三频率全部跳过。
    `_fetch` 必须回看一段再裁回原窗口。
    """

    TODAY = pd.Timestamp('2026-10-21')

    def _run(self, responses, start=None, end=None):
        from GolemQ.analysis.cv import predict
        fake = _FakeMarket(responses)
        # ⚠️ `_merge_realtime` 会读**真实**实时库（`realtime_YYYY-MM-DD`）——
        # 盘中一旦那天的集合存在，用例的 `len(data)` 就随库里的数据变化
        # （2026-10-22 实测：单日用例拿到 2 行）。这里是取数回退的用例，
        # 不测实时合并，钉成 no-op 才与库状态无关。
        with mock.patch('GolemQ.core.market_registry.get_active_market',
                        return_value=fake), \
                mock.patch('GolemQ.markets.StockCN.kline83._merge_realtime'), \
                mock.patch.object(predict, '_last_day', return_value=self.TODAY):
            data, name = predict._fetch('600519', 'day',
                                        start or self.TODAY, end or self.TODAY, 'qfq')
        return data, name, fake

    def test_today_only_retries_with_lookback_and_clips(self):
        frame = _day_frame(['2026-10-20', '2026-10-21'])
        data, _, fake = self._run([
            (None, '600519'),                                # 单日历史：空
            (types.SimpleNamespace(data=frame), '600519'),   # 回看 + realtime：有今天
        ])
        self.assertEqual(len(fake.calls), 2)
        self.assertTrue(fake.calls[1]['realtime'])
        self.assertEqual(fake.calls[1]['start'],
                         (self.TODAY - pd.Timedelta(days=512)).strftime('%Y-%m-%d'))
        self.assertEqual(list(data.index.get_level_values(0)), [self.TODAY])

    def test_lookback_rows_outside_window_are_dropped(self):
        """回看只用来「点火」—— 没有落在请求窗口内的行 ⇒ 仍然不扫。"""
        frame = _day_frame(['2026-10-18', '2026-10-19', '2026-10-20'])
        data, _, _ = self._run([
            (None, '600519'),
            (types.SimpleNamespace(data=frame), '600519'),
        ])
        self.assertIsNone(data)

    def test_old_empty_window_does_not_retry(self):
        """历史窗口为空（如上市前）不该白跑一次 512 天回看。"""
        old = pd.Timestamp('2026-01-05')
        data, _, fake = self._run([(None, '600519')], start=old, end=old)
        self.assertIsNone(data)
        self.assertEqual(len(fake.calls), 1)

    def test_non_empty_single_day_does_not_retry(self):
        frame = _day_frame(['2026-10-21'])
        data, _, fake = self._run([(types.SimpleNamespace(data=frame), '600519')])
        self.assertEqual(len(data), 1)
        self.assertEqual(len(fake.calls), 1)

    def test_identical_512d_window_does_not_retry(self):
        """日频预测本身就用 512 天窗口 —— 为空时回看同一段没有意义。"""
        start = self.TODAY - pd.Timedelta(days=512)
        data, _, fake = self._run([(None, '600519')], start=start, end=self.TODAY)
        self.assertIsNone(data)
        self.assertEqual(len(fake.calls), 1)


class TestTodayOnlyJobActuallyRuns(unittest.TestCase):
    """baseline 有今天那根时，三频率必须真的被调用（不再 `without_processing`）。"""

    def test_today_only_runs_all_three_frequencies(self):
        from GolemQ.analysis.cv import predict
        today = pd.Timestamp('2026-10-21')
        baseline = _day_frame([today])
        with mock.patch.object(predict, 'calc_ChartScaned_persistence',
                               return_value=(True, True, True, [today])), \
                mock.patch.object(predict, 'load_model', return_value=object()), \
                mock.patch.object(predict, '_fetch', return_value=(baseline, '600519')), \
                mock.patch.object(predict, 'GQ_predict_ChartScanAI') as pred:
            without_processing = predict.calc_ChartScanAI_reality(
                ticker='600519', start_date=today, end_date=today)
        self.assertFalse(without_processing)
        self.assertEqual([c.kwargs['freq'] for c in pred.call_args_list],
                         ['day', '15min', '60min'])


if __name__ == '__main__':
    unittest.main()
