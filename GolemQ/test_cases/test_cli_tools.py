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

import unittest
from unittest.mock import patch, MagicMock
from GolemQ import GQMARKETS
from GolemQ.cli.tools import auto_register_markets, purge_mongodb_database
from GolemQ.markets.StockCN import StockCN


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
        """测试自动注册跳过已注册的市场"""
        # 先手动注册一个市场
        if 'StockCN' not in GQMARKETS:
            GQMARKETS['StockCN'] = StockCN()
        original_instance = GQMARKETS['StockCN']
        
        # 运行自动注册
        auto_register_markets()
        
        # 应该仍然是同一个实例
        self.assertIs(GQMARKETS['StockCN'], original_instance)
        self.assertEqual(len(GQMARKETS), 1)

    def test_auto_register_markets_skips_abstract_classes(self):
        """测试自动注册跳过抽象类"""
        # 运行自动注册
        auto_register_markets()
        
        # StockHK 是抽象类，应该被跳过
        self.assertNotIn('StockHK', GQMARKETS)

    @patch('GolemQ.cli.tools.print')
    def test_purge_mongodb_database_verbose(self, mock_print):
        """测试数据库清理功能（详细模式）"""
        # 确保有市场实例
        
        # 运行数据库清理
        purge_mongodb_database(verbose=True)
        
        # 检查是否有输出
        self.assertTrue(mock_print.called)

    @patch('GolemQ.cli.tools.print')
    def test_purge_mongodb_database_silent(self, mock_print):
        """测试数据库清理功能（静默模式）"""
        # 确保有市场实例
        
        # 运行数据库清理
        purge_mongodb_database(verbose=False)
        
        # 检查是否有输出
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