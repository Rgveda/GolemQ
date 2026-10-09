import contextlib
import io
import os, sys
try:
    import GolemQ
except ImportError:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import unittest
import unittest.mock
from datetime import datetime, timezone
from unittest.mock import MagicMock

from GolemQ.datasource.writer import (
    BAR_BATCH,
    replace_code_rows,
    save_bar_chunk,
)
from GolemQ.markets.StockCN.datasource.pytdx_kline import (
    MAX_RETRY,
    bars_page,
    bars_paged,
)
from GolemQ.markets.StockCN.kline_doc import (
    BARS_PER_DAY,
    TDX_CATEGORY,
    bar_doc,
    bars_needed,
    dedup_docs,
    page_offsets,
    xdxr_doc,
)

UTC = timezone.utc


def _bar(**kw):
    """一根最小可用的 pytdx bar（日线字段齐全）。"""
    base = {'year': 2024, 'month': 1, 'day': 2, 'hour': 15, 'minute': 0,
            'open': 1.0, 'high': 2.0, 'low': 0.5, 'close': 1.5,
            'vol': 100, 'amount': 150.0}
    base.update(kw)
    return base


class TestSaveBarChunk(unittest.TestCase):
    """时序集合的写口：**先删后插**（不能 upsert，见 PITFALLS P14）。"""

    def setUp(self):
        self.coll = MagicMock()
        self.coll.name = 'stock_1min'
        self.docs = [{'code': '600519', 'ts': datetime(2026, 10, 8, 1, 31, tzinfo=UTC),
                      'vol': 1}]

    def test_delete_filter_covers_exactly_the_dedup_keys(self):
        """删除条件 = 本次要写的那些 `(code, ts)` —— **不多删一行**（冻结日不碰）。"""
        stats = save_bar_chunk(self.coll, self.docs, code='600519')

        self.assertEqual(self.coll.delete_many.call_args.args[0],
                         {'code': '600519', 'ts': {'$in': [self.docs[0]['ts']]}})
        self.assertEqual(stats['inserted'], 1)
        # 顺序：先删后插 —— 反了就不幂等
        names = [c[0] for c in self.coll.mock_calls]
        self.assertLess(names.index('delete_many'), names.index('insert_many'))

    def test_replace_code_rows_deletes_whole_code(self):
        """`replace_code_rows` 的语义：整 code 替换（xdxr / adj 用）。"""
        replace_code_rows(self.coll, self.docs, code='600519')
        self.assertEqual(self.coll.delete_many.call_args.args[0], {'code': '600519'})

    def test_empty_docs_never_deletes(self):
        """承重守卫：取到空**绝不删** —— 否则一次上游抽风就清空集合。"""
        stats = save_bar_chunk(self.coll, [], code='600519')
        self.assertTrue(stats['skipped'])
        self.assertEqual(stats['deleted'], 0)
        self.coll.delete_many.assert_not_called()
        self.coll.insert_many.assert_not_called()

    def test_batches_at_bar_batch(self):
        docs = [{'code': '600519', 'ts': datetime(2026, 10, 8, 1, 31, tzinfo=UTC)}
                for _ in range(BAR_BATCH + 1)]
        stats = save_bar_chunk(self.coll, docs, code='600519')
        self.assertEqual(stats['inserted'], BAR_BATCH + 1)
        self.assertEqual(self.coll.insert_many.call_count, 2)


class TestBarDoc(unittest.TestCase):
    """文档字段集必须与 8.3 存量**逐字段一致** —— 多一个键就多一列读出。"""

    DAY_KEYS = {'code', 'date', 'date_stamp', 'time_stamp', 'ts',
                'open', 'high', 'low', 'close', 'vol', 'amount'}

    def test_day_doc_has_exactly_the_stored_key_set(self):
        doc = bar_doc(_bar(), '600519', market='stock', frequency='day')
        self.assertEqual(set(doc), self.DAY_KEYS)
        self.assertNotIn('datetime', doc)          # 日线存量没有
        self.assertNotIn('type', doc)              # 迁移时被剥掉
        self.assertNotIn('up_count', doc)          # 只有 index 才有
        self.assertIsInstance(doc['vol'], int)
        self.assertIsInstance(doc['date_stamp'], int)
        self.assertIsInstance(doc['time_stamp'], int)

    def test_day_time_stamp_is_midnight(self):
        """存量实测 `time_stamp == date_stamp`（日线归零）；不归零会与存量差 15 小时。"""
        doc = bar_doc(_bar(hour=15, minute=0), '600519', market='stock', frequency='day')
        self.assertEqual(doc['time_stamp'], doc['date_stamp'])

    def test_minute_doc_adds_datetime_with_same_label_as_source(self):
        """分钟标签**不做偏移** —— 实测 pytdx 与存量同标签（09:31 … 15:00）。"""
        doc = bar_doc(_bar(hour=9, minute=31), '600519', market='stock', frequency='1min')
        self.assertEqual(set(doc), self.DAY_KEYS | {'datetime'})
        self.assertEqual(doc['datetime'], '2024-01-02 09:31:00')

    def test_index_only_gets_updown_counts_when_given(self):
        plain = bar_doc(_bar(), '000001', market='index', frequency='day')
        self.assertNotIn('up_count', plain)        # 存量异质：没有就不补 0
        with_cnt = bar_doc(_bar(), '000001', market='index', frequency='day',
                           up_count=3, down_count=4)
        self.assertEqual((with_cnt['up_count'], with_cnt['down_count']), (3, 4))
        stock = bar_doc(_bar(), '600519', market='stock', frequency='day',
                        up_count=3, down_count=4)
        self.assertNotIn('up_count', stock)        # 股票不写这两个键

    def test_code_is_truncated_to_six_digits(self):
        doc = bar_doc(_bar(), '600519.XSHG', market='stock', frequency='day')
        self.assertEqual(doc['code'], '600519')

    def test_vol_unit_conversion_and_amount_passthrough(self):
        """`vol` 按市场/频率换单位（实测）；`amount` 直传，哨兵值原样。"""
        min_doc = bar_doc(_bar(vol=25500.0, amount=3.1580928e7), '600519',
                          market='stock', frequency='1min')
        self.assertEqual((min_doc['vol'], min_doc['amount']), (255, 3.1580928e7))  # 股→手
        day_doc_ = bar_doc(_bar(vol=31239.0), '600519', market='stock', frequency='day')
        self.assertEqual(day_doc_['vol'], 31239)                 # 日线不换算
        idx = bar_doc(_bar(vol=4385300.0), '000001', market='index', frequency='day')
        self.assertEqual(idx['vol'], 438530000)                  # 指数日线 ×100
        sentinel = bar_doc(_bar(vol=5.877471754e-39, amount=5.877471754e-39), '600519',
                           market='stock', frequency='1min')
        # 哨兵（源端零成交标记）→ 0：vol 靠取整、amount 靠 normalize_amount，
        # 与 8.3 近期行的约定一致（对拍实测 618 行差异全是「哨兵 vs 0.0」）
        self.assertEqual(sentinel['vol'], 0)
        self.assertEqual(sentinel['amount'], 0.0)


class TestBarDocsMatchesRowWise(unittest.TestCase):
    """列式构造（`bar_docs`，polars）必须与逐行 `bar_doc` **逐字段相等**。

    这是 polars 化的护栏：`bar_doc` 是参照实现（docstring 里有 doctest、口径只有
    一处定义），`bar_docs` 只是为了快 12×（51.5 µs/根 → 4.3 µs/根）。
    任何口径漂移都必须先让这条红。
    """

    BARS = [
        {'year': 2026, 'month': 10, 'day': 8, 'hour': 9, 'minute': 31,
         'open': 10.0, 'high': 10.1, 'low': 9.9, 'close': 10.05,
         'vol': 99300.0, 'amount': 123223312.0},
        {'year': 2026, 'month': 10, 'day': 8, 'hour': 9, 'minute': 32,
         'open': 10.1, 'high': 10.2, 'low': 10.0, 'close': 10.15,
         'vol': 25500.0, 'amount': 31580928.0},
        # 哨兵行（零成交）：vol 与 amount 都要归一
        {'year': 2026, 'month': 10, 'day': 8, 'hour': 14, 'minute': 59,
         'open': 10.0, 'high': 10.0, 'low': 10.0, 'close': 10.0,
         'vol': 5.877471754e-39, 'amount': 5.877471754e-39},
    ]

    def _row_wise(self, market, frequency, extra=None):
        extra = extra or {}
        return [bar_doc(b, '600519', market=market, frequency=frequency,
                        up_count=extra.get('up_count'), down_count=extra.get('down_count'))
                for b in self.BARS]

    def test_day_and_minute_and_index(self):
        from GolemQ.markets.StockCN.kline_doc import bar_docs
        cases = [
            ('stock', 'day', None),
            ('stock', '1min', None),
            ('etf', '5min', None),
        ]
        for market, frequency, extra in cases:
            with self.subTest(market=market, frequency=frequency):
                self.assertEqual(
                    [d for d in self._row_wise(market, frequency, extra)],
                    bar_docs(self.BARS, '600519', market=market, frequency=frequency))

    def test_index_carries_updown_and_x100_vol(self):
        from GolemQ.markets.StockCN.kline_doc import bar_docs
        bars = [dict(self.BARS[0], up_count=3, down_count=4)]
        row = [bar_doc(bars[0], '000001', market='index', frequency='day',
                       up_count=3, down_count=4)]
        col = bar_docs(bars, '000001', market='index', frequency='day')
        self.assertEqual(row, col)
        self.assertEqual(col[0]['vol'], 99300 * 100)    # 指数日线：pytdx 给手，存量存股 → ×100

    def test_empty_bars(self):
        from GolemQ.markets.StockCN.kline_doc import bar_docs
        self.assertEqual(bar_docs([], '600519', market='stock', frequency='day'), [])


class TestXdxrDoc(unittest.TestCase):

    def test_field_mapping(self):
        row = {'year': 2024, 'month': 1, 'day': 2, 'category': 1, 'name': '除权除息',
               'fenhong': 10.0, 'songzhuangu': 0.0, 'peigu': 0.0, 'peigujia': 3.5599999428,
               'panqianliutong': 100, 'panhouliutong': 200,
               'qianzongguben': 300, 'houzongguben': 400}
        doc = xdxr_doc(row, '600519')
        self.assertEqual(doc['liquidity_before'], 100)
        self.assertEqual(doc['liquidity_after'], 200)
        self.assertEqual(doc['shares_before'], 300)
        self.assertEqual(doc['shares_after'], 400)
        self.assertEqual(doc['category_meaning'], '除权除息')
        self.assertEqual(doc['peigujia'], 3.5599999428)   # 不做 float 洗白
        self.assertNotIn('type', doc)
        self.assertNotIn('datetime', doc)


class TestPagingMath(unittest.TestCase):
    """翻页与根数：估多估少都不该丢数据（短页即止才是停的条件）。"""

    def test_page_offsets_ascending(self):
        self.assertEqual(page_offsets(0), [])
        self.assertEqual(page_offsets(800), [0])
        self.assertEqual(page_offsets(2400), [0, 800, 1600])

    def test_bars_needed_is_an_upper_bound(self):
        # 2026-10-08 是交易日（10-01..10-07 国庆休市）
        self.assertEqual(bars_needed('2026-10-08', '2026-10-08', 240), 480)
        # 起点晚于今天 → 只留兜底那几天，不返回负数
        self.assertEqual(bars_needed('2026-10-09', '2026-10-08', 240), 240)

    def test_every_frequency_has_a_tdx_category_and_daily_bar_count(self):
        for freq in ('1min', '5min', '15min', '30min', '60min'):
            with self.subTest(freq=freq):
                self.assertIn(freq, TDX_CATEGORY)
                self.assertIn(freq, BARS_PER_DAY)
                self.assertEqual(BARS_PER_DAY[freq], 240 // int(freq.rstrip('min')))
        self.assertEqual(TDX_CATEGORY['day'], 9)
        self.assertEqual(TDX_CATEGORY['1min'], 8)

    def test_dedup_keeps_first_of_same_code_and_ts(self):
        docs = [{'code': 'a', 'ts': 1, 'v': 'first'}, {'code': 'a', 'ts': 1, 'v': 'second'},
                {'code': 'a', 'ts': 2, 'v': 'third'}]
        self.assertEqual([d['v'] for d in dedup_docs(docs)], ['first', 'third'])


class TestNothingPrintsWhileTheBarIsAlive(unittest.TestCase):
    """⚠️ **tqdm 进度条活着的时候，谁都不许往 stdout 写。**

    这是 banner 与进度条能共存的前提。banner 的表头**压在进度条上面**，靠
    「光标上移 N 行」原地重画（`core/presentation.Banner`）；期间多出任何一行，
    N 就算错 —— **进度条和表头会一起花**。

    排查时发现过两处漏网：`pytdx_kline._retry_page` 的重连诊断是**裸 `print`**，
    而它跑在**每只 code 的内层**（进度条正活着）。现在改走 `note` 回调、由调用方
    缓冲到 `bar.close()` 之后再吐（见 `save_kline_tdx` 的 `_note`）。
    """

    def test_retry_page_reports_through_note_not_stdout(self):
        from GolemQ.markets.StockCN.datasource import pytdx_kline as pk

        notes = []

        def boom():
            raise OSError('模拟连不上')

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            api, bars = pk._retry_page(object(), 0, '600519', 9, 0, 800, False,
                                       1, boom, True, note=notes.append)

        self.assertIsNone(bars)
        self.assertEqual(buf.getvalue(), '', '重连诊断漏到了 stdout —— 会打花进度条和 banner')
        self.assertTrue(notes, 'note 回调反而一条都没收到')

    def test_retry_page_is_silent_without_note(self):
        """没给 `note` 时**必须一个字都不打**（默认是静默，不是退回 print）。"""
        from GolemQ.markets.StockCN.datasource import pytdx_kline as pk

        def boom():
            raise OSError('模拟连不上')

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            pk._retry_page(object(), 0, '600519', 9, 0, 800, False, 1, boom, True)
        self.assertEqual(buf.getvalue(), '')

    def test_per_code_write_is_silent(self):
        """每 code 的落库不打印 —— `_process_code` 调 `save_bar_chunk` **不传 verbose**。"""
        from GolemQ.datasource import writer

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            writer.save_bar_chunk(MagicMock(),
                                  [{'code': '600519', 'ts': 1}], code='600519')
        self.assertEqual(buf.getvalue(), '', '逐 code 落库打了字 —— 会打花条和 banner')


if __name__ == '__main__':
    unittest.main()


class FakeApi:
    """假 pytdx 连接：按 `pages` 依次吐页面；元素为 None 表示「连接已废」（P3b）。"""

    def __init__(self, pages, name='fake'):
        self.pages = list(pages)
        self.name = name
        self.calls = []
        self.disconnected = False

    def _next(self):
        return self.pages.pop(0) if self.pages else []

    def get_security_bars(self, category, market, code, start, count):
        self.calls.append(start)
        return self._next()

    def get_index_bars(self, category, market, code, start, count):
        self.calls.append(start)
        return self._next()

    def disconnect(self):
        self.disconnected = True


class TestBarsPaged(unittest.TestCase):
    """翻页三条路径：正常到头 / 短页即止 / 连接废掉后重连（P3b）。"""

    def test_stops_on_short_page(self):
        api = FakeApi([[{'i': i} for i in range(800)], [{'i': i} for i in range(300)]])
        bars, status, out = bars_paged(api, 1, '600519', 9, [0, 800], reconnect=None)

        self.assertEqual(status, 'ok')
        self.assertEqual(len(bars), 1100)
        self.assertIs(out, api)
        self.assertEqual(api.calls, [0, 800])       # 不再多翻一页

    def test_empty_first_page_is_empty_not_aborted(self):
        api = FakeApi([[]])
        bars, status, _ = bars_paged(api, 1, '600519', 9, [0])
        self.assertEqual((bars, status), ([], 'empty'))

    def test_no_offsets_means_no_request(self):
        api = FakeApi([[]])
        bars, status, _ = bars_paged(api, 1, '600519', 9, [])
        self.assertEqual((bars, status, api.calls), ([], 'empty', []))

    def test_none_means_dead_connection_and_reconnects(self):
        """`None` 不是「没有数据」，是「连接已废」—— 必须换新连接重试同一页。"""
        dead = FakeApi([None])
        fresh = FakeApi([[{'i': 1}], []], name='fresh')
        bars, status, cur = bars_paged(dead, 1, '600519', 9, [0, 800],
                                       reconnect=lambda: fresh)

        self.assertEqual(status, 'ok')
        self.assertEqual(len(bars), 1)
        self.assertIs(cur, fresh)                    # 调用方要拿到**新**连接
        self.assertTrue(dead.disconnected)           # 旧的必须被关掉

    def test_retry_exhausted_aborts_whole_code(self):
        """重试耗尽 → `aborted`：调用方必须**整只跳过**，不能拿半截数据写库。"""
        dead = FakeApi([None] + [None] * MAX_RETRY)
        bars, status, _ = bars_paged(dead, 1, '600519', 9, [0],
                                     reconnect=lambda: FakeApi([None]))
        self.assertEqual((bars, status), ([], 'aborted'))

    def test_index_flag_uses_get_index_bars(self):
        api = FakeApi([[{'i': 1}]])
        bars_paged(api, 1, '000001', 9, [0], is_index=True)
        self.assertEqual(api.calls, [0])
        self.assertEqual(len(api.pages), 0)

    def test_page_returns_none_only_on_dead_connection(self):
        self.assertIsNone(bars_page(FakeApi([None]), 1, '600519', 9, 0))
        self.assertEqual(bars_page(FakeApi([[]]), 1, '600519', 9, 0), [])


class TestTargetRoutedCollections(unittest.TestCase):
    """`save_adj` / `save_xdxr_tdx` 的 `target` 决定**读写哪三个集合**。

    这类参数最危险的失败方式是「静默读错集合」：`target='etf'` 却去写 `stock_adj`，
    不报错、只写错地方。所以这里用假 db 把三条集合名钉死。
    """

    def _fake_db(self, target):
        db = MagicMock()
        db.__getitem__.side_effect = lambda name: MagicMock(name=name)
        return db

    def test_save_adj_routes_to_target_collections(self):
        from GolemQ.markets.StockCN import kline_save as ks

        seen = []
        db = MagicMock()
        db.__getitem__.side_effect = lambda name: seen.append(name) or MagicMock()
        # 没有日线数据 → 走 skipped 出口，不会去碰 xdxr / 写库
        db['etf_day'].find.return_value = []
        with unittest.mock.patch.object(ks, '_db', return_value=db):
            out = ks.save_adj(['510300'], target='etf', verbose=False)

        self.assertEqual(out['skipped'], ['510300'])
        self.assertIn('etf_adj', seen)
        self.assertIn('etf_day', seen)
        self.assertNotIn('stock_adj', seen)
        self.assertNotIn('stock_day', seen)

    def test_save_adj_default_target_is_stock(self):
        from GolemQ.markets.StockCN import kline_save as ks

        seen = []
        db = MagicMock()
        db.__getitem__.side_effect = lambda name: seen.append(name) or MagicMock()
        db['stock_day'].find.return_value = []
        with unittest.mock.patch.object(ks, '_db', return_value=db):
            ks.save_adj(['600519'], verbose=False)

        self.assertIn('stock_adj', seen)
        self.assertNotIn('etf_adj', seen)

    def test_save_xdxr_routes_to_target_collection(self):
        from GolemQ.markets.StockCN import kline_save as ks

        seen = []
        db = MagicMock()
        db.__getitem__.side_effect = lambda name: seen.append(name) or MagicMock()
        with unittest.mock.patch.object(ks, '_db', return_value=db), \
                unittest.mock.patch.object(ks, 'universe', return_value=[]):
            ks.save_xdxr_tdx(target='etf', verbose=False)

        self.assertIn('etf_xdxr', seen)
        self.assertNotIn('stock_xdxr', seen)
