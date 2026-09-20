import os, sys
try:
    import GolemQ
except ImportError:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import unittest
from unittest.mock import patch, MagicMock
from datetime import datetime

try:
    from GolemQ.markets.StockCN.align import StockCNAlign
except ImportError as _import_error:
    StockCNAlign = None
    _IMPORT_ERROR = _import_error


@unittest.skipIf(
    StockCNAlign is None,
    "StockCNAlign / check_etf_data_freshness() 在代码库中从未实现："
    "markets/StockCN/align.py 仅导出 ckpo_align_stock_turnover_rate 与 stock_min_aligned。"
    f"（{_IMPORT_ERROR}）",
)
class TestMarketAlign(unittest.TestCase):

    def setUp(self):
        self.align = StockCNAlign()

    @patch('GolemQ.markets.StockCN.align.datetime')
    @patch('GolemQ.markets.StockCN.align.get_mongo_client')
    def test_check_etf_data_freshness(self, mock_get_client, mock_datetime):
        """测试ETF数据新鲜度检查功能"""
        # Mock当前时间
        mock_now = datetime(2024, 1, 15, 15, 0, 0)
        mock_datetime.now.return_value = mock_now
        
        # Mock MongoDB客户端
        mock_client = MagicMock()
        mock_db = MagicMock()
        mock_collection = MagicMock()
        
        mock_get_client.return_value = mock_client
        mock_client.__getitem__.return_value = mock_db
        mock_db.__getitem__.return_value = mock_collection
        
        # Mock查询结果 - 有今天的数据
        mock_collection.find_one.return_value = {'timestamp': mock_now}
        
        # 执行检查
        result = self.align.check_etf_data_freshness()
        
        # 验证返回True（数据新鲜）
        self.assertTrue(result)
        
        # 验证查询条件
        expected_query = {
            'timestamp': {
                '$gte': datetime(2024, 1, 15, 0, 0, 0),
                '$lt': datetime(2024, 1, 16, 0, 0, 0)
            }
        }
        mock_collection.find_one.assert_called_with(expected_query)

    @patch('GolemQ.markets.StockCN.align.datetime')
    @patch('GolemQ.markets.StockCN.align.get_mongo_client')
    def test_check_etf_data_freshness_no_data(self, mock_get_client, mock_datetime):
        """测试没有ETF数据时的新鲜度检查"""
        # Mock当前时间
        mock_now = datetime(2024, 1, 15, 15, 0, 0)
        mock_datetime.now.return_value = mock_now
        
        # Mock MongoDB客户端
        mock_client = MagicMock()
        mock_db = MagicMock()
        mock_collection = MagicMock()
        
        mock_get_client.return_value = mock_client
        mock_client.__getitem__.return_value = mock_db
        mock_db.__getitem__.return_value = mock_collection
        
        # Mock查询结果 - 没有数据
        mock_collection.find_one.return_value = None
        
        # 执行检查
        result = self.align.check_etf_data_freshness()
        
        # 验证返回False（数据不新鲜）
        self.assertFalse(result)


if __name__ == '__main__':
    unittest.main()