# coding:utf-8
"""ETF 独立成 `ETF_CN` / `etf_*` 之后的**路由**测试。

为什么单列一个文件
==================
本次改动最危险的地方**全是静默失败** —— 判错不抛异常，只是读错表或选错容器：

1. `market_prefix` 少了 `etf` 分支 → ETF 落回 `index_*`（读到真指数或空）
2. `_min_container` 少了 `etf` 分支 → ETF 拿到 **Stock 容器** →
   多出一个 `to_qfq()`，而它按 `stock_adj` 乘因子 → **价格全错且不报错**
3. `frame_to_datastruct` 少了 `('etf', …)` 键 → `KeyError`（这个是响亮的）
4. ETF 容器少/多了 `to_qfq` → 复权缺失或二次复权

这里把 1/3/4 钉成断言（2 在 `fetch.py` 里，需 DB 才能端到端跑，
但 `_min_container` 本身是纯函数，见下）。

⚠️ 本文件**不需要 MongoDB**：`market_prefix` / `frame_to_datastruct` 都是纯函数。
需要因子表的 `to_qfq()` **不在这里断言数值**（那要 `etf_adj`），只断言
「哪一类容器提供 `to_qfq`」这个层级契约。
"""
import os
import sys
import unittest

try:
    import GolemQ  # noqa: F401
except ImportError:                     # pragma: no cover
    sys.path.insert(0, os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', '..')))

import pandas as pd                                            # noqa: E402
from GolemQ.core.constants import MARKET_TYPE                  # noqa: E402
from GolemQ.markets.StockCN.datastruct import (                # noqa: E402
    GQ_DataStruct_ETF_day,
    GQ_DataStruct_ETF_min,
    GQ_DataStruct_Index_day,
    GQ_DataStruct_Index_min,
    GQ_DataStruct_Stock_day,
    GQ_DataStruct_Stock_min,
    frame_to_datastruct,
)
from GolemQ.markets.StockCN.etf_fq import GQ_is_etf            # noqa: E402
from GolemQ.markets.StockCN.kline83 import market_prefix       # noqa: E402
from GolemQ.markets.StockCN.symbol import is_stock_cn          # noqa: E402


class TestMarketPrefix(unittest.TestCase):
    """`market_prefix` 的三值契约 —— 它决定**读哪张表**。"""

    def test_three_way(self):
        cases = [
            ('600519', 'stock'), ('000001', 'stock'), ('300750', 'stock'),
            ('200037', 'stock'),                       # B股 —— 曾被判成 ETF
            ('510300', 'etf'), ('159915', 'etf'), ('158001', 'etf'),
            ('513050', 'etf'), ('588000', 'etf'),
            ('399001', 'index'), ('000300', 'index'),
            ('150001', 'index'),                       # 分级基金走 index（有意）
            ('160105', 'index'),                       # LOF 同上
        ]
        for code, expected in cases:
            with self.subTest(code=code):
                self.assertEqual(market_prefix(code), expected)

    def test_explicit_market_type_wins(self):
        self.assertEqual(
            market_prefix('600519', market_type=MARKET_TYPE.ETF_CN), 'etf')
        self.assertEqual(
            market_prefix('510300', market_type=MARKET_TYPE.STOCK_CN), 'stock')

    def test_list_uses_first(self):
        self.assertEqual(market_prefix(['510300', '600519']), 'etf')


class TestFrameToDatastruct(unittest.TestCase):
    """`frame_to_datastruct` 的分派表 —— 少了 `('etf', …)` 会 `KeyError`。"""

    @staticmethod
    def _frame():
        return pd.DataFrame({
            'date': ['2026-09-01', '2026-09-02'],
            'code': ['510300', '510300'],
            'open': [1.0, 1.1], 'high': [1.2, 1.3],
            'low': [0.9, 1.0], 'close': [1.1, 1.2],
            'volume': [100.0, 200.0], 'amount': [10.0, 20.0],
        })

    def test_dispatch(self):
        cases = [
            (('stock', 'day'), GQ_DataStruct_Stock_day),
            (('stock', 'min'), GQ_DataStruct_Stock_min),
            (('etf', 'day'), GQ_DataStruct_ETF_day),
            (('etf', 'min'), GQ_DataStruct_ETF_min),
            (('index', 'day'), GQ_DataStruct_Index_day),
            (('index', 'min'), GQ_DataStruct_Index_min),
        ]
        for (market, frequency), cls in cases:
            with self.subTest(market=market, frequency=frequency):
                out = frame_to_datastruct(self._frame(), market, frequency)
                self.assertIsInstance(out, cls)

    def test_empty_returns_none(self):
        self.assertIsNone(frame_to_datastruct(pd.DataFrame(), 'etf', 'day'))


class TestQfqCapability(unittest.TestCase):
    """`to_qfq` 的**层级契约**：股票与 ETF 有，真指数没有。

    这条不能靠"顺手补全" —— 真指数没有除权概念，`etf_adj` 里也没有它们的行；
    而 ETF 若拿不到 `to_qfq`，消费方就得绕开类型系统单独调 `etf_fq`，
    正是这次要消掉的那种分裂。
    """

    def test_capability(self):
        self.assertTrue(hasattr(GQ_DataStruct_Stock_day, 'to_qfq'))
        self.assertTrue(hasattr(GQ_DataStruct_Stock_min, 'to_qfq'))
        self.assertTrue(hasattr(GQ_DataStruct_ETF_day, 'to_qfq'))
        self.assertTrue(hasattr(GQ_DataStruct_ETF_min, 'to_qfq'))
        self.assertFalse(hasattr(GQ_DataStruct_Index_day, 'to_qfq'))
        self.assertFalse(hasattr(GQ_DataStruct_Index_min, 'to_qfq'))

    def test_etf_and_stock_use_different_factor_tables(self):
        """ETF 与股票**必须是两条路** —— 走同一张表会静默算错价。

        这里断言两者绑的不是同一个函数对象（股票的 `apply_qfq` vs ETF 的
        `GQ_apply_etf_qfq`），从而不会有人在重构时"统一"掉。
        """
        self.assertIsNot(GQ_DataStruct_Stock_day.to_qfq,
                         GQ_DataStruct_ETF_day.to_qfq)

    def test_type_attribute(self):
        self.assertEqual(GQ_DataStruct_ETF_day.type, 'etf_day')
        self.assertEqual(GQ_DataStruct_ETF_min.type, 'etf_min')


class TestGQIsEtfAgreesWithClassifier(unittest.TestCase):
    """`GQ_is_etf` 必须与 `is_stock_cn` 的 `ETF_CN` **完全一致**。

    它原先嗅探描述串（`endswith('ETF基金')`），措辞一改就静默失效。
    现在两者共用同一个判据，这条测试防止再次分叉。
    """

    def test_agreement(self):
        codes = ['510300', '513050', '588000', '159915', '158001', '560050',
                 '600519', '000001', '399001', '000300', '150001', '160105',
                 '180101', '184001', '200037', '205001', '280001', '430489',
                 '820001', '920819']
        for code in codes:
            with self.subTest(code=code):
                self.assertEqual(
                    GQ_is_etf(code),
                    is_stock_cn(code)[1] == MARKET_TYPE.ETF_CN,
                    f'{code} 上两个判据不一致')

    def test_tolerates_tagged_forms(self):
        """带交易所标记的写法也要能判 —— `is_stock_cn` 负责归一。"""
        self.assertTrue(GQ_is_etf('510300.XSHG'))
        self.assertTrue(GQ_is_etf('sh.510300'))
        self.assertFalse(GQ_is_etf('600519.XSHG'))


class TestMinContainer(unittest.TestCase):
    """`fetch._min_container` 的三分支 —— 少了 `etf` 会静默拿错容器。

    ⚠️ 这是本次最容易漏的一处：只改 `market_prefix` 而忘了这里，ETF 会落到
    `Stock_min`，于是拿到 `to_qfq()`（按 `stock_adj` 错乘 ETF 价格）
    且被当股票取名，**全程不报错**。
    """

    def test_container_choice(self):
        from GolemQ.markets.StockCN.fetch import _min_container
        df = pd.DataFrame()
        self.assertIsInstance(_min_container(df, '510300'),
                              GQ_DataStruct_ETF_min)
        self.assertIsInstance(_min_container(df, '600519'),
                              GQ_DataStruct_Stock_min)
        self.assertIsInstance(_min_container(df, '399001'),
                              GQ_DataStruct_Index_min)
        self.assertIsInstance(_min_container(df, ['510300', '600519']),
                              GQ_DataStruct_ETF_min)


if __name__ == '__main__':
    unittest.main()
