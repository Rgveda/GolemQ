import os, sys
try:
    import GolemQ
except ImportError:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import unittest
from unittest.mock import patch, MagicMock
from GolemQ.markets.StockCN.tools import purge_historical_collections


class TestMarketTools(unittest.TestCase):
    
    @patch('GolemQ.markets.StockCN.tools.get_mongo_client')
    def test_purge_historical_collections(self, mock_get_client):
        """测试清理历史数据集合功能"""
        # Mock MongoDB客户端和集合
        mock_client = MagicMock()
        mock_db = MagicMock()
        mock_collection = MagicMock()
        
        mock_get_client.return_value = mock_client
        mock_client.__getitem__.return_value = mock_db
        mock_db.list_collection_names.return_value = [
            'stock_quotes_20240101',
            'stock_quotes_20240102', 
            'realtime_20240101',
            'other_collection'
        ]
        mock_db.__getitem__.return_value = mock_collection
        
        # 执行清理操作
        purge_historical_collections()
        
        # 验证调用了drop方法
        self.assertEqual(mock_collection.drop.call_count, 3)  # 3个历史集合应该被删除
        
    @patch('GolemQ.markets.StockCN.tools.get_mongo_client')
    def test_purge_historical_collections_no_collections(self, mock_get_client):
        """测试没有历史数据集合时的清理功能"""
        mock_client = MagicMock()
        mock_db = MagicMock()
        
        mock_get_client.return_value = mock_client
        mock_client.__getitem__.return_value = mock_db
        mock_db.list_collection_names.return_value = ['other_collection']  # 没有匹配的历史集合
        
        # 执行清理操作
        purge_historical_collections()
        
        # 验证没有调用drop方法
        self.assertEqual(mock_db.__getitem__.call_count, 0)


if __name__ == '__main__':
    unittest.main()