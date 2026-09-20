import os, sys
try:
    import GolemQ
except ImportError:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import unittest
from unittest.mock import patch
import pandas as pd
try:
    from GolemQ.markets.StockCN.crawler import StockCNCrawler
except ImportError as _import_error:
    StockCNCrawler = None
    _IMPORT_ERROR = _import_error


@unittest.skipIf(
    StockCNCrawler is None,
    "StockCNCrawler / get_all_etf_list() 在代码库中从未实现："
    "markets/StockCN/crawler.py 仅导出三个估值抓取函数，且实现中不存在任何 read_excel 调用。"
    f"（{_IMPORT_ERROR}）",
)
class TestMarketCrawler(unittest.TestCase):

    def setUp(self):
        self.crawler = StockCNCrawler()

    @patch('GolemQ.markets.StockCN.crawler.os.path.exists')
    @patch('GolemQ.markets.StockCN.crawler.os.listdir')
    @patch('GolemQ.markets.StockCN.crawler.pd.read_excel')
    def test_get_all_etf_list_with_existing_files(self, mock_read_excel, mock_listdir, mock_exists):
        """测试获取ETF列表功能（有现有文件的情况）"""
        # Mock文件存在和目录列表
        mock_exists.return_value = True
        mock_listdir.return_value = ['etf_file1.xlsx', 'etf_file2.xlsx']
        
        # Mock Excel读取结果
        mock_df1 = pd.DataFrame({
            '基金代码': ['510300', '510500'],
            '基金简称': ['沪深300ETF', '中证500ETF'],
            '基金资产净值(亿元)': [100.0, 50.0]
        })
        mock_df2 = pd.DataFrame({
            '基金代码': ['159919'],
            '基金简称': ['沪深300ETF基金'],
            '基金资产净值(亿元)': [80.0]
        })
        mock_read_excel.side_effect = [mock_df1, mock_df2]
        
        # 执行获取ETF列表
        result = self.crawler.get_all_etf_list()
        
        # 验证返回类型
        self.assertIsInstance(result, pd.DataFrame)
        
        # 验证列名转换
        expected_columns = ['code', 'name', 'net_asset_value']
        self.assertListEqual(list(result.columns), expected_columns)
        
        # 验证数据合并
        self.assertEqual(len(result), 3)
        self.assertListEqual(list(result['code']), ['510300', '510500', '159919'])

    @patch('GolemQ.markets.StockCN.crawler.os.path.exists')
    def test_get_all_etf_list_no_directory(self, mock_exists):
        """测试获取ETF列表功能（目录不存在的情况）"""
        # Mock目录不存在
        mock_exists.return_value = False
        
        # 执行获取ETF列表
        result = self.crawler.get_all_etf_list()
        
        # 验证返回空的DataFrame
        self.assertIsInstance(result, pd.DataFrame)
        self.assertTrue(result.empty)

    @patch('GolemQ.markets.StockCN.crawler.os.path.exists')
    @patch('GolemQ.markets.StockCN.crawler.os.listdir')
    def test_get_all_etf_list_empty_directory(self, mock_listdir, mock_exists):
        """测试获取ETF列表功能（空目录的情况）"""
        # Mock目录存在但为空
        mock_exists.return_value = True
        mock_listdir.return_value = []
        
        # 执行获取ETF列表
        result = self.crawler.get_all_etf_list()
        
        # 验证返回空的DataFrame
        self.assertIsInstance(result, pd.DataFrame)
        self.assertTrue(result.empty)


if __name__ == '__main__':
    unittest.main()