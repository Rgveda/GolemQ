#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""读路径的「**带 REALTIME**」—— `kline83._merge_realtime` 的两道门。

为什么值得测：这条路上有个**容易看不见的写副作用** —— 它依赖的
`realtime.realtime_ts_collection` 是 `create_collection` 包了 `try`，
**直接拿它去读会把不存在的集合建出来**。所以「集合不存在就先返回」那道门
既是省时间（用户 2026-10-09 要求「不比 `_v3` 慢」），也是**防写**。

另外三条不变量：合并失败**不能把已读到手的历史丢掉**；两族（分钟/日线）
各自调对合并器；`realtime=False` 时**一次都不碰实时库**。
"""

import os
import sys

try:
    import GolemQ  # noqa: F401
except ImportError:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import unittest
import unittest.mock

from GolemQ.markets.StockCN import kline83 as k

COLL = 'realtime_2026-10-09'


class _Result:
    """最小可用结果：`_merge_realtime` 只把它原样传给合并器。"""

    def __init__(self):
        self.data = 'HISTORY'


class TestRealtimeReadGuard(unittest.TestCase):
    def _merge(self, *, collections, frequency='1min', merger=None, raises=False):
        rt_db = unittest.mock.MagicMock()
        rt_db.list_collection_names.return_value = collections
        result = _Result()
        kw = {}
        if merger is not None:
            kw = {'side_effect': merger}
        if raises:
            kw = {'side_effect': RuntimeError('tick 库炸了')}
        with unittest.mock.patch.object(k, '_merge_realtime', wraps=k._merge_realtime), \
                unittest.mock.patch('GolemQ.markets.StockCN.GOLEMQ_STOCK_CN_REALTIME',
                                    rt_db), \
                unittest.mock.patch('GolemQ.markets.StockCN.realtime.'
                                    'realtime_collection_name', return_value=COLL), \
                unittest.mock.patch('GolemQ.markets.StockCN.realtime.'
                                    'GQ_fetch_stock_min_realtime_adv', **kw) as min_m, \
                unittest.mock.patch('GolemQ.markets.StockCN.realtime.'
                                    'GQ_fetch_stock_day_realtime_adv', **kw) as day_m:
            out = k._merge_realtime(result, ['600519'], frequency, verbose=False)
        return out, result, min_m, day_m, rt_db

    def test_missing_collection_skips_and_never_touches_it(self):
        """⚠️ **集合不存在 ⇒ 直接返回**：既省一次 tick 读，也**不建集合**。

        `realtime_ts_collection` 是 `create_collection` 包 try —— 读它会把集合
        **建出来**。这条门就是防那个写副作用（实测：读一次 `_v3` 之后实时库集合数不变）。
        """
        out, result, min_m, day_m, rt_db = self._merge(collections=[])   # 空库
        self.assertIs(out, result)                  # 原样返回
        self.assertFalse(min_m.called)
        self.assertFalse(day_m.called)
        # 只做了「列集合名」这一件事，**没有任何 create / 写**
        self.assertTrue(rt_db.list_collection_names.called)
        for forbidden in ('create_collection', 'insert_one', 'insert_many'):
            self.assertFalse(getattr(rt_db, forbidden).called, forbidden)

    def test_existing_collection_calls_the_minute_merger(self):
        _, _, min_m, day_m, _ = self._merge(collections=[COLL], frequency='1min')
        self.assertTrue(min_m.called)
        self.assertFalse(day_m.called, '分钟频率不该走日线合并器')

    def test_existing_collection_calls_the_day_merger(self):
        _, _, min_m, day_m, _ = self._merge(collections=[COLL], frequency='day')
        self.assertTrue(day_m.called)
        self.assertFalse(min_m.called, '日线频率不该走分钟合并器')

    def test_merge_failure_returns_history_not_none(self):
        """合并炸了 ⇒ **返回纯历史** —— 不能把已经读到手的数据也丢掉。"""
        out, result, _, _, _ = self._merge(collections=[COLL], raises=True)
        self.assertIs(out, result)
        self.assertEqual(out.data, 'HISTORY')

    def test_realtime_false_never_touches_the_realtime_db(self):
        """`realtime=False` ⇒ 一次都不碰实时库（这是「不带 REALTIME」该有的代价：0）。"""
        rt_db = unittest.mock.MagicMock()
        with unittest.mock.patch('GolemQ.markets.StockCN.GOLEMQ_STOCK_CN_REALTIME', rt_db), \
                unittest.mock.patch.object(k, '_read_timeseries',
                                           return_value=k.pd.DataFrame()), \
                unittest.mock.patch.object(k, '_apply_adjustments'):
            k.get_kline_price_v3(['600519'], realtime=False, verbose=False)
        self.assertFalse(rt_db.list_collection_names.called)

    def test_both_readers_default_to_realtime_true(self):
        """⚠️ **两族的默认都必须是 `True`**，且必须与 `base_market` 的抽象声明一致。

        `kline83.get_kline_price_v3` 原先写 `None`，而 `base_market.py` 与两个门面
        **一直写的是 `True`** —— 以前那个形参是空转所以看不出来，接通后必须对齐。
        """
        import inspect
        from GolemQ.markets import base_market
        for fn in (k.get_kline_price_min, k.get_kline_price_v3,
                   base_market.BaseMarket.get_kline_price_min,
                   base_market.BaseMarket.get_kline_price_v3):
            with self.subTest(fn=fn.__qualname__):
                self.assertIs(inspect.signature(fn).parameters['realtime'].default,
                              True)

    def test_default_call_does_merge(self):
        """默认（不给 `realtime`）就走合并 —— 用户口径是「带 REALTIME」。

        ⚠️ 桩要**非空**：`_v3` 在 `df.empty` 时**提前返回 `(None, codename)`**
        （没数据就没得合）—— 给空表就永远走不到合并那一步，用例会假绿。
        """
        rt_db = unittest.mock.MagicMock()
        rt_db.list_collection_names.return_value = []      # 空库 ⇒ 门挡掉
        df = k.pd.DataFrame({'ts': [k.dt.datetime(2026, 10, 9)],
                             'code': ['600519'], 'open': [1.0]})
        with unittest.mock.patch('GolemQ.markets.StockCN.GOLEMQ_STOCK_CN_REALTIME', rt_db), \
                unittest.mock.patch.object(k, '_read_timeseries', return_value=df), \
                unittest.mock.patch.object(k, '_apply_adjustments'):
            k.get_kline_price_v3(['600519'], verbose=False)
        self.assertTrue(rt_db.list_collection_names.called)


if __name__ == '__main__':
    unittest.main()
