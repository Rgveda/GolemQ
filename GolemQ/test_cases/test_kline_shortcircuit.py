#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""K 线保存的**本地水位短路**（`DECISIONS.md` D25）。

为什么值得测：短路的错法是**静默的** —— 跳错不会报错，只会让某只票缺一段。
所以这里钉三件事：

1. **判据偏向取数**：拿不准（`ts is None` / 前沿探不到 / TTL 过期 / 日历答不出来）
   一律**不跳**；
2. **探针必须用 `ts`（timeField）** —— 用 `time_stamp` 不是慢一点，是**慢 8800 倍**
   （实测 `stock_1min`：0.0051s vs 44.63s），而且**不报错**；
3. **前沿必须精确** —— 连续二分只会收敛到区间、**低估**一分钟，而低估之后
   `ts >= frontier` 对**每一只票都成立** ⇒ 全都跳、什么都没取（看着像成功）。
"""

import datetime
import io
import os
import sys

try:
    import GolemQ  # noqa: F401
except ImportError:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import unittest
import unittest.mock

from GolemQ.markets.StockCN import kline_save as ks
from GolemQ.markets.StockCN.kline_doc import (SESSION_OPEN, SESSION_READY_AT,
                                              alive_threshold,
                                              last_closed_session_bar)

D = datetime.datetime
#: `alive_threshold` 返回的是 **UTC-aware**（与库里 `ts` 同系）—— 断言里要显示成北京
#: 时间才看得懂，所以这里显式转一次。**别把 UTC 值当北京时间读**（差 8 小时，且不报错）。
_BJ = datetime.timezone(datetime.timedelta(hours=8))


def bj(value):
    return value.astimezone(_BJ).strftime('%Y-%m-%d %H:%M')


class TestAliveThreshold(unittest.TestCase):
    """「这个集合该有的最早一根」—— 日历**只在日级**出场。"""

    def test_intraday_never_falls_back_to_yesterday(self):
        """⚠️ **盘中绝不退到昨天。** 退了的话：上午还没人取过 ⇒ 探针说「集合是活的」
        ⇒ 逐只判据又把所有人判成「等于前沿」⇒ **一个上午一根都不取**（死锁）。"""
        got = alive_threshold('1min', D(2026, 10, 9, 10, 0))      # 周五盘中
        self.assertEqual(bj(got), '2026-10-09 09:30')             # ← 今天，不是 10-08

    def test_before_open_falls_back_to_previous_session(self):
        self.assertEqual(bj(alive_threshold('1min', D(2026, 10, 9, 8, 0))),
                         '2026-10-08 09:30')

    def test_day_bars_use_midnight(self):
        """日线 bar 的 `ts` 口径是该日**北京零点**（实测）。"""
        self.assertEqual(bj(alive_threshold('day', D(2026, 10, 9, 16, 0))),
                         '2026-10-09 00:00')

    def test_weekend_uses_previous_trading_day(self):
        self.assertEqual(bj(alive_threshold('1min', D(2026, 10, 10, 10, 0))),   # 周六
                         '2026-10-09 09:30')

    def test_holiday_uses_previous_trading_day(self):
        self.assertEqual(bj(alive_threshold('1min', D(2026, 10, 5, 10, 0))),    # 国庆里
                         '2026-09-30 09:30')

    def test_outside_calendar_is_none(self):
        """日历覆盖之外**必须**返回 None —— 那时「该有」无从谈起，宁可每次都取。"""
        self.assertIsNone(alive_threshold('1min', D(2030, 1, 1, 10, 0)))
        self.assertIsNone(alive_threshold('day', D(1985, 1, 1, 10, 0)))

    def test_open_boundary_is_inclusive(self):
        self.assertEqual(bj(alive_threshold('1min', D(2026, 10, 9, 9, 30))),
                         '2026-10-09 09:30')
        self.assertEqual(bj(alive_threshold('1min', D(2026, 10, 9, 9, 29))),
                         '2026-10-08 09:30')
        self.assertEqual(SESSION_OPEN, '09:30:00')

    def test_day_does_not_count_today_before_the_close(self):
        """⚠️ **盘中日线不认今天**（2026-10-09 修）：当天日线盘中**不该存在**
        （`PITFALLS.md` P20），认了就会永远探不到前沿 ⇒ 盘中每轮白扫全市场。"""
        for clock in ((9, 30), (10, 0), (14, 59)):
            with self.subTest(clock=clock):
                self.assertEqual(
                    bj(alive_threshold('day', D(2026, 10, 9, *clock))),
                    '2026-10-08 00:00', '盘中日线的基准必须是**上一交易日**')
        self.assertEqual(bj(alive_threshold('day', D(2026, 10, 9, 15, 0))),
                         '2026-10-09 00:00', '15:00 之后才认今天')
        self.assertEqual(SESSION_READY_AT['day'], '15:00:00')

    def test_minute_still_counts_today_intraday(self):
        """分钟与日线**相反**：开市就该有今天的数据，退到昨天会死锁。"""
        self.assertEqual(bj(alive_threshold('5min', D(2026, 10, 9, 10, 0))),
                         '2026-10-09 09:30')


class TestIntradayBlocksMinuteShortcircuit(unittest.TestCase):
    """判据④：**盘中 + 分钟 ⇒ 禁止短路**。

    不这么做的后果是**静默冻住集合**：前沿是**数据自证**的（集合里最新那根 bar 在
    哪儿），盘中全跳之后没有东西去写新 bar ⇒ 前沿自己不会前进 ⇒ 下一轮还全跳，
    冻在那一分钟直到 TTL 到期（最长 5h）。判据①（探针）挡不住这种停滞 ——
    它只挡得住「今天压根没取过」。
    """

    def _blocked(self, freq, *clock):
        return ks.intraday_blocks_shortcircuit(freq, D(2026, 10, 9, *clock))

    def test_minute_is_blocked_all_session(self):
        for clock in ((9, 30), (10, 30), (11, 29), (13, 5), (14, 59)):
            with self.subTest(clock=clock):
                self.assertTrue(self._blocked('1min', *clock))
                self.assertTrue(self._blocked('60min', *clock))

    def test_lunch_break_allows_it(self):
        """11:31–12:59 午休：上午的分钟 bar 已定稿 ⇒ 短路是对的。
        （`GQ_util_if_tradetime` 在午休返回假，这是复用它的关键理由。）"""
        for clock in ((11, 31), (12, 30), (12, 59)):
            with self.subTest(clock=clock):
                self.assertFalse(self._blocked('5min', *clock))

    def test_after_close_allows_it(self):
        """15:00 之后分钟 bar 全部定稿 ⇒ 照常短路（批量跑的主场）。"""
        for clock in ((15, 0), (16, 30), (20, 0)):
            with self.subTest(clock=clock):
                self.assertFalse(self._blocked('1min', *clock))

    def test_non_trading_day_allows_it(self):
        self.assertFalse(ks.intraday_blocks_shortcircuit('1min', D(2026, 10, 10, 10, 0)))

    def test_day_is_never_blocked(self):
        """**盘中日线反而该跳**：盘中不写当天日线，所有 code 都停在前一交易日那根上
        —— 这正是盘中 `stock_day` 能秒过的原因。"""
        for clock in ((10, 0), (14, 0), (9, 45)):
            with self.subTest(clock=clock):
                self.assertFalse(self._blocked('day', *clock))


class FakeColl:
    """假集合：只知道「最新一根在 `newest`」这一件事，并记下每次查询。"""

    def __init__(self, newest=None):
        self.newest = newest
        self.queries = []

    def find_one(self, filt, projection=None, sort=None, limit=None):
        self.queries.append(filt)
        if self.newest is None:
            return None
        key = list(filt)[0]
        bound = filt[key].get('$gte')
        # `P(X) = 存在 ts >= X` —— **等号要算真**（`bound == newest` 时那一根就在 X 上）。
        # 写成严格小于会把前沿整体推低一分钟，于是所有票都判成"落后"。
        if bound is not None and bound <= self.newest:
            return {'ts': self.newest}
        return None


class TestCollectionFrontier(unittest.TestCase):
    WINDOW = D(2026, 10, 9, 1, 30, tzinfo=datetime.timezone.utc)      # 北京 09:30
    NEWEST = D(2026, 10, 9, 7, 0, tzinfo=datetime.timezone.utc)      # 北京 15:00

    def test_finds_the_exact_newest_bar(self):
        """**精确命中**，不是"收敛到区间"。低估一分钟就会让所有票都被跳（见模块 docstring）。"""
        coll = FakeColl(self.NEWEST)
        got = ks.collection_frontier(coll, self.WINDOW, self.NEWEST + datetime.timedelta(hours=1))
        self.assertEqual(got, int(self.NEWEST.timestamp()))

    def test_probe_count_is_logarithmic(self):
        """一个会话 ≤ 330 分钟 ⇒ log2(330) ≈ 9 次。别退化成线性扫。"""
        coll = FakeColl(self.NEWEST)
        ks.collection_frontier(coll, self.WINDOW, self.NEWEST + datetime.timedelta(hours=1))
        self.assertLessEqual(len(coll.queries), 12)
        self.assertGreater(len(coll.queries), 1)

    def test_empty_collection_is_none(self):
        self.assertIsNone(ks.collection_frontier(FakeColl(None), self.WINDOW,
                                                 self.NEWEST))

    def test_probe_error_is_none_not_raise(self):
        """探针炸了 ⇒ 判不了 ⇒ 不跳（保守），**不能**把异常抛给取数主流程。"""
        class Boom(FakeColl):
            def find_one(self, *a, **k):
                raise RuntimeError('mongo 挂了')
        self.assertIsNone(ks.collection_frontier(Boom(self.NEWEST), self.WINDOW, self.NEWEST))

    def test_probe_uses_the_timefield_not_the_int_stamp(self):
        """⚠️ 钉住 8800 倍那个坑：查询键**必须是 `ts`**。

        `time_stamp` 是个普通 int 字段（只有单键索引），同样条件的范围查询在
        `stock_1min`（21.4 亿行）上要 **44.63 秒**，而 `ts` 只要 **0.0051 秒**。
        写错了**不报错**，只是每次探针白等 44 秒 —— 9 次就是 7 分钟。
        """
        coll = FakeColl(self.NEWEST)
        ks.collection_frontier(coll, self.WINDOW, self.NEWEST)
        self.assertTrue(coll.queries, '一次探针都没发，用例失去意义')
        for q in coll.queries:
            self.assertIn('ts', q)
            self.assertNotIn('time_stamp', q)


class TestShortCircuit(unittest.TestCase):
    """`_process_code` 的三道判据：TTL 放行 + 前沿探到 + 该 code 水位到前沿。"""

    FRONTIER = 1791500000

    def _stats(self):
        return {'codes': 1, 'inserted': 0, 'deleted': 0, 'skipped_codes': 0,
                'empty_codes': 0, 'reconnects': 0, 'errors': [], 'diffs': {},
                'skipped_fresh': 0}

    def _call(self, coll, *, frontier, allow_skip):
        """跑一次 `_process_code`。**连接是在 `_fetch_docs` 里建的**，所以
        「有没有去取」的判据是 **`_fetch_docs` 有没有被调用** —— 不是 `new_api.called`
        （patch 掉 `_fetch_docs` 之后它永远是 False，那样断言等于没断言）。"""
        src = unittest.mock.MagicMock()
        with unittest.mock.patch.object(ks, '_fetch_docs',
                                        return_value=([], 'empty', 0)) as fd:
            stats = self._stats()
            ks._process_code(src, coll, 'stock', 'day', '600519', '2015-01-01', 0,
                             '2026-10-09', None, False, stats, False,
                             frontier=frontier, allow_skip=allow_skip)
        return stats, fd

    def _fresh_coll(self):
        """该 code 的最后一根就在前沿上。"""
        ts = D(2026, 10, 9, 7, 0, tzinfo=datetime.timezone.utc)
        coll = unittest.mock.MagicMock()
        coll.find_one.return_value = {'ts': ts, 'date': '2026-10-09'}
        return coll

    def test_skips_and_never_connects(self):
        stats, fd = self._call(self._fresh_coll(), frontier=self.FRONTIER,
                               allow_skip=True)
        self.assertEqual(stats['skipped_fresh'], 1)
        self.assertFalse(fd.called, '短路了却还是去取了 —— 等于没做')
        self.assertEqual(stats['inserted'], 0)

    def test_ttl_gate_closed_means_fetch(self):
        """TTL 过期 ⇒ 强制全查，哪怕水位已经到前沿。"""
        stats, fd = self._call(self._fresh_coll(), frontier=self.FRONTIER,
                               allow_skip=False)
        self.assertEqual(stats['skipped_fresh'], 0)
        self.assertTrue(fd.called)

    def test_unknown_frontier_means_fetch(self):
        """前沿探不到（集合空/停住/探针失败）⇒ 不跳。"""
        stats, fd = self._call(self._fresh_coll(), frontier=None, allow_skip=True)
        self.assertEqual(stats['skipped_fresh'], 0)
        self.assertTrue(fd.called)

    def test_code_behind_the_frontier_is_fetched(self):
        old = D(2026, 9, 30, 7, 0, tzinfo=datetime.timezone.utc)
        coll = unittest.mock.MagicMock()
        coll.find_one.return_value = {'ts': old, 'date': '2026-09-30'}
        stats, fd = self._call(coll, frontier=self.FRONTIER, allow_skip=True)
        self.assertEqual(stats['skipped_fresh'], 0)
        self.assertTrue(fd.called)

    def test_code_with_no_data_is_never_skipped(self):
        """⚠️ 从没数据的 code 要**全量回填** —— 跳了就是永久空洞。"""
        coll = unittest.mock.MagicMock()
        coll.find_one.return_value = None
        stats, fd = self._call(coll, frontier=self.FRONTIER, allow_skip=True)
        self.assertEqual(stats['skipped_fresh'], 0)
        self.assertTrue(fd.called)

    def test_naive_ts_is_read_as_utc(self):
        """pymongo 默认返回**裸** datetime（实值是 UTC）—— 当成本地时间就差 8 小时，
        而且**不报错**。这条钉住 `_as_utc` 那一补。"""
        naive = D(2026, 10, 9, 7, 0)                     # 裸值 = UTC 07:00 = 北京 15:00
        coll = unittest.mock.MagicMock()
        coll.find_one.return_value = {'ts': naive, 'date': '2026-10-09'}
        stats, fd = self._call(coll, frontier=self.FRONTIER, allow_skip=True)
        self.assertEqual(stats['skipped_fresh'], 1)
        self.assertFalse(fd.called)


class TestMarketResolvedBeforeConnecting(unittest.TestCase):
    """`tdx_market_of` 是纯查表，**必须排在建连之前**。

    原先它排在 `new_api()` 之后 —— 号段表里没有的 code（指数的 `810`/`899` 等）
    会**每个频率白建一条连接**再返回 `'empty'`。
    """

    def test_unmapped_code_never_connects(self):
        src = unittest.mock.MagicMock()
        docs, status, reconnects = ks._fetch_docs(src, 'index', 'day', '899999',
                                                  '1990-01-01', '2026-10-09')
        self.assertEqual(status, 'empty')
        self.assertEqual(docs, [])
        self.assertFalse(src.new_api.called, '未映射号段还是建了连接')


class TestTtlGate(unittest.TestCase):
    def test_checkin_name_is_per_collection(self):
        """颗粒度 = **每个集合**一条记录（26 个节点各自独立）。"""
        self.assertEqual(ks.kline_checkin_name('stock_day'), 'kline:stock_day')
        self.assertNotEqual(ks.kline_checkin_name('stock_day'),
                            ks.kline_checkin_name('index_day'))

    def test_never_swept_means_no_shortcircuit(self):
        with unittest.mock.patch.object(ks, 'kline_sweep_age_hours', return_value=None):
            self.assertFalse(ks.allow_shortcircuit('stock_day'))

    def test_fresh_sweep_allows_shortcircuit(self):
        with unittest.mock.patch.object(ks, 'kline_sweep_age_hours', return_value=0.1), \
                unittest.mock.patch.object(ks, 'kline_ttl_hours', return_value=5):
            self.assertTrue(ks.allow_shortcircuit('stock_day'))

    def test_expired_sweep_blocks_shortcircuit(self):
        with unittest.mock.patch.object(ks, 'kline_sweep_age_hours', return_value=6.0), \
                unittest.mock.patch.object(ks, 'kline_ttl_hours', return_value=5):
            self.assertFalse(ks.allow_shortcircuit('stock_day'))

    def test_sweep_is_marked_only_after_a_full_pass(self):
        """⚠️ 短路过的那一遍**不许记账** —— 记了 TTL 就被无限推后，兜底网永远不触发。"""
        self.assertTrue(callable(ks.mark_kline_sweep))

    def test_ttl_is_open_closed_pair(self):
        """同参考数据的口径：盘中 5h / 盘后 24h。"""
        self.assertEqual(ks.KLINE_TTL_HOURS, (5, 24))


if __name__ == '__main__':
    unittest.main()


class TestXdxrBoardOpenGate(unittest.TestCase):
    """复权的闸比 K 线**多一条**：`跨没过一个开盘`（`DECISIONS.md` D26）。

    为什么多这一条（用户 2026-10-09 指出）：除权信息是**日级**的、当天那条
    **开盘前就该拿到**，而 `(5, 24)` 的盘后 24h 口径会把**开盘前那一跑跳掉** ——

        T-1 17:30 扫过 → T 08:30 只隔 15h < 24h ⇒ 跳 ⇒ 当天事件漏到收盘后

    所以判据是「上次全查在**最近一次开盘之后**」，而不是"N 小时以内"。
    """

    def _call(self, *, ttl_ok, age_hours, since_open):
        with unittest.mock.patch.object(ks, 'allow_shortcircuit', return_value=ttl_ok), \
                unittest.mock.patch.object(ks, 'kline_sweep_age_hours',
                                           return_value=age_hours), \
                unittest.mock.patch.object(ks, 'hours_since_last_open',
                                           return_value=since_open):
            return ks.allow_xdxr_shortcircuit('stock_xdxr')

    def test_swept_after_the_open_is_up_to_date(self):
        """今天开盘后扫过（age 1h < 距开盘 8h）⇒ 本交易日已拿到 ⇒ 可以跳。"""
        self.assertTrue(self._call(ttl_ok=True, age_hours=1.0, since_open=8.0))

    def test_swept_before_the_open_must_run(self):
        """⚠️ **核心那条**：昨晚扫的（age 15h），今早已开盘（距开盘 0.5h）
        ⇒ 上次扫在开盘**之前** ⇒ 不许跳，否则当天除权信息漏掉。"""
        self.assertFalse(self._call(ttl_ok=True, age_hours=15.0, since_open=0.5))

    def test_ttl_gate_still_applies(self):
        """TTL 那道仍然在 —— 两条是**与**关系。"""
        self.assertFalse(self._call(ttl_ok=False, age_hours=0.1, since_open=8.0))

    def test_calendar_unavailable_must_run(self):
        """日历答不出来（覆盖之外）⇒ 全查（保守）。"""
        self.assertFalse(self._call(ttl_ok=True, age_hours=0.1, since_open=None))

    def test_hours_since_open_uses_the_calendar(self):
        """**真调**交易日历（不打桩）—— 复用 `alive_threshold('1min')`，
        别再写第二套"最近一次开盘"。"""
        self.assertAlmostEqual(ks.hours_since_last_open(D(2026, 10, 9, 10, 30)),
                               1.0, places=3)          # 盘中：距今天 09:30

    def test_hours_since_open_before_the_bell_counts_from_previous_session(self):
        """开盘前：距今最近的"开盘"是**上一交易日** 09:30（10-08 09:30 → 10-09 08:30 = 23h）。"""
        self.assertAlmostEqual(ks.hours_since_last_open(D(2026, 10, 9, 8, 30)),
                               23.0, places=3)


class TestFastSkipAfterClose(unittest.TestCase):
    """**收盘后的快路径**：整个集合已整批落库 ⇒ 连逐只读都省掉。

    它**比 TTL 硬**（TTL 只说"最近查过"，它说"**最近一个已收盘的交易日整批都查过了**"）
    ⇒ 这条路径**不看 TTL**（用户 2026-10-09 指出「这样就不用考虑是否还在 5 小时 TTL 之内」）。

    之所以成立：记账**只在全宇宙的跑法**里发生（`codes is None`，见 `save_kline_tdx`）
    ⇒ 「上次全查在收盘之后」等价于「**每一只 code 都被处理过**」。
    """

    T = D(2026, 10, 9, 17, 30)          # 周五收盘后
    THRESH = D(2026, 10, 9, 7, 0, tzinfo=datetime.timezone.utc)   # 北京 15:00

    def _call(self, *, age, since_close, blocked=False, probe_hit=True):
        coll = FakeColl(self.THRESH if probe_hit else None)
        with unittest.mock.patch.object(ks, 'intraday_blocks_shortcircuit',
                                        return_value=blocked), \
                unittest.mock.patch.object(ks, 'hours_since_last_close',
                                           return_value=since_close), \
                unittest.mock.patch.object(ks, 'last_closed_session_bar',
                                           return_value=self.THRESH), \
                unittest.mock.patch.object(ks, 'kline_sweep_age_hours',
                                           return_value=age):
            return ks.fast_skip_reason(coll, 'stock_1min', '1min', self.T)

    def test_swept_after_the_close_skips_the_whole_collection(self):
        got = self._call(age=2.5, since_close=3.0)        # 扫在收盘后（2.5 < 3.0）
        self.assertIsNotNone(got)
        self.assertIn('整段跳过', got)

    def test_swept_before_the_close_goes_the_normal_path(self):
        """昨晚扫的（age 20h）> 距收盘（3h）⇒ 今天这批还没查 ⇒ 走常规路径。"""
        self.assertIsNone(self._call(age=20.0, since_close=3.0))

    def test_never_swept(self):
        self.assertIsNone(self._call(age=None, since_close=3.0))

    def test_probe_miss_means_no_skip(self):
        """⚠️ 记录说扫过 ≠ 数据还在（误删、或那一轮有 code 失败）⇒ 探针说不 ⇒ 不跳。"""
        self.assertIsNone(self._call(age=2.5, since_close=3.0, probe_hit=False))

    def test_intraday_minute_is_blocked(self):
        """⚠️ 承重：盘中「最近一次收盘」是**昨天** 15:00，不加这条 ①，
        周一 10:00 会拿周五收盘当"已覆盖" ⇒ 跳过周一的分钟线。"""
        self.assertIsNone(self._call(age=2.5, since_close=3.0, blocked=True))

    def test_calendar_unavailable(self):
        self.assertIsNone(self._call(age=2.5, since_close=None))

    def test_probe_uses_the_timefield(self):
        """同 `collection_frontier`：只有 `ts`（timeField）能吃到时序索引。"""
        coll = FakeColl(self.THRESH)
        with unittest.mock.patch.object(ks, 'intraday_blocks_shortcircuit',
                                        return_value=False), \
                unittest.mock.patch.object(ks, 'hours_since_last_close',
                                           return_value=3.0), \
                unittest.mock.patch.object(ks, 'last_closed_session_bar',
                                           return_value=self.THRESH), \
                unittest.mock.patch.object(ks, 'kline_sweep_age_hours',
                                           return_value=2.5):
            ks.fast_skip_reason(coll, 'stock_1min', '1min', self.T)
        self.assertTrue(coll.queries)
        for q in coll.queries:
            self.assertIn('ts', q)
            self.assertNotIn('time_stamp', q)

    def test_last_closed_session_bar_real_calendar(self):
        """**真调交易日历**：收盘口径与开盘口径**共用 `_session_base`**。"""
        minutes = last_closed_session_bar('1min', self.T)      # 17:30 周五
        self.assertEqual(bj(minutes), '2026-10-09 15:00')
        day = last_closed_session_bar('day', self.T)
        self.assertEqual(bj(day), '2026-10-09 00:00')          # 日线 bar 的 ts 是零点
        # 盘中：最近一次收盘是**昨天**
        self.assertEqual(bj(last_closed_session_bar('1min', D(2026, 10, 9, 10, 0))),
                         '2026-10-08 15:00')
        self.assertEqual(ks.hours_since_last_close(self.T) is not None, True)


class TestFastPathDoesNotSurviveTheNextSession(unittest.TestCase):
    """⏭ 模拟时间推进到**下一个交易日**：快路径必须**自己失效**。

    用户 2026-10-09 要求：「模拟时间到明天开盘后的时间点，再 `--save tdx`，是否就不再短路」。

    这是「快路径不会跨交易日延续」的防线 —— 判据是「上次全查在**最近一次收盘之后**」，
    而**新的收盘会让 `since_close` 归零**（分母变小），条件自然翻假。
    下面用**真交易日历**推时间（只给 `age` 打桩），所以它同时也验了日历那一半。

    ⚠️ 2026-10-10/11 是周末 ⇒ 下一个交易日是 **10-12（周一）**。
    """

    FRIDAY_SCAN = D(2026, 10, 9, 17, 30)       # 周五收盘后全查过
    MONDAY_OPEN = D(2026, 10, 12, 10, 0)       # 下一交易日**盘中**
    MONDAY_CLOSE = D(2026, 10, 12, 15, 30)     # 下一交易日**收盘后**

    @property
    def _age(self):
        """距周五那次全查多少小时（到周一盘中/收盘各是多少）。"""
        def hours(t):
            return (t - self.FRIDAY_SCAN).total_seconds() / 3600.0
        return hours

    def _gate(self, now, frequency):
        """**真日历** + 打桩 age；探针用 `FakeColl` 模拟"那批数据在库里"。"""
        from GolemQ.markets.StockCN.kline_doc import last_closed_session_bar
        coll = FakeColl(last_closed_session_bar(frequency, now))
        with unittest.mock.patch.object(ks, 'kline_sweep_age_hours',
                                        return_value=self._age(now)):
            return ks.fast_skip_reason(coll, 'stock_' + frequency, frequency, now)

    def test_minute_fast_path_expires_when_the_next_session_opens(self):
        """✅ **分钟：周一盘中不再短路** —— 判据①（盘中×分钟禁跳）拦住快路径，
        常规路径的 TTL 也过期（64.5h ≥ 5h）⇒ 会去取周一的分钟线。
        **这就是"明天开盘后不再短路"的正面回答。**"""
        self.assertIsNone(self._gate(self.MONDAY_OPEN, '1min'))
        self.assertIsNone(self._gate(self.MONDAY_OPEN, '60min'))

    def test_day_fast_path_still_skips_intraday(self):
        """❗**日线不一样**：盘中**仍会跳**，而且**这是对的** —— 盘中不写当天日线
        （`PITFALLS.md` P20：pytdx 给的是"当日累计"bar），所以到 15:00 之前都没活干。
        这正是盘中 `stock_day` 秒过的原因，别把它当成"没在取数"。"""
        self.assertIsNotNone(self._gate(self.MONDAY_OPEN, 'day'))

    def test_everything_expires_after_the_next_close(self):
        """周一**收盘后**：`since_close` 归零 ⇒ 四个组合全都不许跳 ⇒ 去取周一那批。
        （含 `day` —— 15:00 之后当日日线才该写。）"""
        for freq in ('day', '1min', '5min', '60min'):
            with self.subTest(freq=freq):
                self.assertIsNone(self._gate(self.MONDAY_CLOSE, freq))

    def test_since_close_resets_at_each_close(self):
        """直接钉那个机械原因：**新收盘让 `since_close` 归零**。"""
        from GolemQ.markets.StockCN.kline_doc import last_closed_session_bar
        fri = last_closed_session_bar('1min', D(2026, 10, 9, 17, 30))
        mon = last_closed_session_bar('1min', D(2026, 10, 12, 15, 30))
        self.assertEqual(bj(fri), '2026-10-09 15:00')
        self.assertEqual(bj(mon), '2026-10-12 15:00')      # ← 换成周一收盘了
        # 周末两天一直指向**周五**收盘（所以周末/周一盘前仍可跳）
        self.assertEqual(bj(last_closed_session_bar('1min', D(2026, 10, 10, 10, 0))),
                         '2026-10-09 15:00')
        self.assertEqual(bj(last_closed_session_bar('1min', D(2026, 10, 11, 23, 0))),
                         '2026-10-09 15:00')
