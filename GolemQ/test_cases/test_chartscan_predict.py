# coding:utf-8
"""`chartscan/predict.py` 的**写库路由**用例（`DECISIONS.md` D51）。

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

import io
import unittest
from unittest import mock

import numpy as np
import pandas as pd

from GolemQ.core.constants import AKA, MARKET_TYPE

_PREDICT = 'GolemQ.markets.StockCN.chartscan.predict'
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
                 echo=None, created_at=None):
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
        from GolemQ.markets.StockCN.chartscan import predict
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
        from GolemQ.markets.StockCN.chartscan import predict
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
        from GolemQ.markets.StockCN.chartscan import predict
        with mock.patch('GolemQ.markets.StockCN.reality_save.fetch_reality_feat',
                        return_value=self._frame()) as fetch:
            predict.calc_ChartScaned_persistence('510300', market_type='etf_cn')
        args = fetch.call_args[0]
        self.assertEqual(args[3], 'day', '重扫判定必须读日族（三族结果都在那儿）')
        self.assertEqual(args[4], 'etf_cn', '必须按市场类型取表，否则静默读空')

    def test_coords_are_converted_to_datetime(self):
        from GolemQ.markets.StockCN.chartscan.predict import _coords_to_datetime
        out = _coords_to_datetime(self._frame())
        for col in ('Coordinates', 'CoordinatesMaj', 'CoordinatesMin'):
            self.assertTrue(pd.api.types.is_datetime64_any_dtype(out[col]), col)

    def test_missing_coord_columns_do_not_raise(self):
        """缺列的旧文档不能炸 —— 缺的那套按 None 处理（否则整只标的读不动）。"""
        from GolemQ.markets.StockCN.chartscan.predict import _coords_to_datetime
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
        from GolemQ.markets.StockCN.chartscan import render  # noqa: F401 导入即设
        self.assertEqual(os.environ.get('MPLBACKEND'), 'Agg')

    def test_render_module_sets_it_at_import(self):
        """不是靠调用方传 —— 模块**导入**那一刻就得钉住（matplotlib 可能已被别人先导入）。"""
        from pathlib import Path
        src = Path(__file__).resolve().parents[1] / 'markets/StockCN/chartscan/render.py'
        text = src.read_text(encoding='utf-8')
        self.assertIn("os.environ['MPLBACKEND'] = 'Agg'", text)
        self.assertIn("matplotlib.use('Agg', force=True)", text)


class _MarkupFamilyTest(unittest.TestCase):
    """`GQ_markup_ChartScaned` 的 `family=` 参数（旧树的 `collections=` 对应物）。"""

    def test_family_param_routes(self):
        from GolemQ.markets.StockCN.chartscan import predict
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
        from GolemQ.markets.StockCN.chartscan import predict
        kline = _kline('60min', '2026-10-16 15:00')
        rec = _Recorder()
        with mock.patch(_REALITY + '.save_reality_feat', rec):
            predict.GQ_markup_ChartScaned('600519', kline, MARKET_TYPE.STOCK_CN)
        self.assertEqual(rec.names, ['stock_reality_feat_60min'])


if __name__ == '__main__':
    unittest.main()
