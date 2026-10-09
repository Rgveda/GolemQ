#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试 CLI 工具功能
"""

import os, sys
try:
    import GolemQ
except ImportError:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import inspect
import types
import unittest
from unittest.mock import patch, MagicMock
from GolemQ import GQMARKETS
from GolemQ.cli.tools import auto_register_markets, purge_mongodb_database
from GolemQ.markets.base_market import BaseMarket
from GolemQ.markets.StockCN import StockCN


def _dummy_market_module(name, abstract):
    """造一个**假**市场模块 —— 正好满足 `auto_register_markets` 的发现判据本身：
    类在**本模块内**定义（`obj.__module__ == module.__name__`）、
    有 `purge_historical_collections`、可选地是抽象类。

    ⚠️ **不碰真实 `GolemQ/markets/` 目录** —— 那正是原来那几个用例的病根：
    它们拿磁盘上有什么市场包当成事实，于是 `StockHK` 一变成可注册的（那是
    **有意**的，见 `StockHK/__init__.py` 的 docstring），用例就红。
    用假模块之后，用例只测**发现逻辑**，与磁盘内容无关。
    """
    if abstract:
        body = {}                                   # 一个抽象成员都不实现
    else:
        body = {n: (lambda self, *a, **k: None)     # 全部实现 → 具体类
                for n in BaseMarket.__abstractmethods__}
    cls = type('DummyMarket', (BaseMarket,), body)
    module = types.ModuleType('GolemQ.markets.' + name)
    cls.__module__ = module.__name__                # 判据要求「类定义在本模块内」
    module.DummyMarket = cls
    return module


def _fake_importer(*pairs):
    """`importlib.import_module` 的替身：只有**点名**的市场返回假模块；
    其余（磁盘上将来冒出来的 StockUS / 期货 …）返回**空模块** →
    `getmembers` 找不到市场类 → 静默忽略。用例因此与磁盘内容解耦。"""
    by_name = {'GolemQ.markets.' + n: m for n, m in pairs}

    def _import(name, *a, **k):
        return by_name.get(name, types.ModuleType(name))

    return _import


class TestCLITools(unittest.TestCase):
    """测试 CLI 工具功能"""

    def setUp(self):
        """测试前备份 GQMARKETS 状态"""
        self.original_markets = GQMARKETS.copy()
        GQMARKETS.clear()

    def tearDown(self):
        """测试后恢复 GQMARKETS 状态"""
        GQMARKETS.clear()
        GQMARKETS.update(self.original_markets)

    def test_auto_register_markets_skips_registered(self):
        """已注册的市场**不被顶掉** —— 钉的是实例同一性，不是总数。

        ⚠️ 原来还断言 `len(GQMARKETS) == 1`：那等于把「磁盘上只有 StockCN 一个
        市场包」当成事实。任何新市场包（StockHK 现在就合法在册）都会让它变红，
        而它其实什么缺陷也没发现。

        用**普通 object**（不是 StockCN 单例）当哨兵：若跳过失效，真实例会把它顶掉，
        `assertIs` 才会真的失败 —— 拿 StockCN 当哨兵的话，单例语义会让断言永远通过。
        """
        sentinel = object()
        GQMARKETS['StockCN'] = sentinel

        auto_register_markets()

        self.assertIs(GQMARKETS['StockCN'], sentinel)

    def test_auto_register_markets_skips_abstract_classes(self):
        """抽象类**被跳过**、具体类**被注册** —— 一对正反断言。

        ⚠️ 原用例靠「StockHK 是抽象类」这个**当时的事实**。而 StockHK 现在**是具体
        类**（`StockHK/__init__.py` 的 docstring：那是**有意**修的一条真缺陷 ——
        名字对不上抽象声明会让它静默不被注册）。所以改成**造**抽象/具体两类当夹具。
        """
        abstract = _dummy_market_module('StockHK', abstract=True)
        concrete = _dummy_market_module('StockCN', abstract=False)

        # 夹具自检：确认造出来的确实是抽象/具体两类
        self.assertTrue(inspect.isabstract(abstract.DummyMarket))
        self.assertFalse(inspect.isabstract(concrete.DummyMarket))

        with patch('GolemQ.cli.tools.importlib.import_module',
                   side_effect=_fake_importer(('StockHK', abstract),
                                              ('StockCN', concrete))):
            auto_register_markets()

        self.assertNotIn('StockHK', GQMARKETS)      # 抽象 → 跳过
        self.assertIn('StockCN', GQMARKETS)         # 具体 → 注册
        # ↑ 这一条是承重的：没有它，「抽象被跳过」会被「整个循环压根没注册任何东西」
        #   冒充通过（原来就是这种只能单向证明的断言）

    @patch('GolemQ.cli.tools.print')
    def test_purge_mongodb_database_delegates_to_market(self, mock_print):
        """`purge_mongodb_database` 只**委托**给市场实例，自己不碰库。

        ⚠️ **这里绝不能调真库。** 原先这两个用例是真跑的，后果有两层：
        1. **会删真实数据**：`--purge-l1` 走的就是这条链 ——
           `cli/tools.py` → `StockCN.purge_historical_collections()`
           （`markets/StockCN/__init__.py`）→ `purge_historical_collections(
           self.GOLEMQ_STOCK_CN_REALTIME)` → `client[name].drop()`。
           测试一跑就在 `GOLEMQ_STOCK_CN_REALTIME` 上真 drop。
        2. **断言必然是空的**：只 `assertTrue(mock_print.called)`，跑的却是真库。
           而且逐日打印来自 `markets/StockCN/tools.py` 的 `print`，与这里
           patch 的 `GolemQ.cli.tools.print` **不是同一个对象** → 每跑一次泄 14 行
           `⏩ 未找到集合: realtime_...`。

        REALTIME 是**按日滚动清理**的、**只有最后一个交易日那个集合有意义**
        （`DECISIONS.md` D10）—— 任何「依赖某个具体 `realtime_YYYY-MM-DD` 存在」
        的断言隔天必炸。故把市场方法整个换成假的，只钉「有没有委托成功」。
        """
        fake = MagicMock()
        fake.purge_historical_collections.return_value = ['realtime_2026-09-01']

        for verbose in (True, False):
            with self.subTest(verbose=verbose):
                fake.reset_mock()
                with patch.dict(GQMARKETS, {'StockCN': fake}, clear=True), \
                        patch('GolemQ.cli.tools.auto_register_markets'):
                    purge_mongodb_database(verbose=verbose)
                fake.purge_historical_collections.assert_called_once_with()

        self.assertTrue(mock_print.called)

    def test_auto_register_markets_with_mock_module(self):
        """测试自动注册处理异常情况"""
        with patch('GolemQ.cli.tools.importlib.import_module') as mock_import:
            mock_import.side_effect = ImportError("模拟导入错误")
            
            # 运行自动注册
            auto_register_markets()
            
            # 应该处理错误而不崩溃
            self.assertEqual(len(GQMARKETS), 0)

    def test_auto_register_markets_with_invalid_module(self):
        """测试自动注册处理无效模块"""
        with patch('GolemQ.cli.tools.importlib.import_module') as mock_import:
            mock_module = MagicMock()
            mock_module.__name__ = 'GolemQ.markets.InvalidMarket'
            # 模拟没有符合条件的类
            mock_import.return_value = mock_module
            
            # 运行自动注册
            auto_register_markets()
            
            # 应该没有注册任何市场
            self.assertEqual(len(GQMARKETS), 0)


if __name__ == '__main__':
    unittest.main()