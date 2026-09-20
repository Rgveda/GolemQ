import os, sys
try:
    import GolemQ
except ImportError:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import unittest
import pandas as pd
from GolemQ.markets.StockCN import StockCN


class TestStockCN(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """测试类初始化"""
        cls.stock_cn = StockCN()
        
    @unittest.skip(
        "StockCN 上不存在 get_all_etf_list()。项目真实的 ETF 列表函数是 "
        "markets/StockCN/scribe.py::GQ_get_etf_list()（已从包内导出，返回同样的 "
        "code/name/net_asset_value 三列），但未挂到 StockCN 实例方法上。"
        "且该函数会真实调用 akshare 与 MongoDB，不适合作为单元测试。",
    )
    def test_get_all_etf_list(self):
        """测试get_all_etf_list方法"""
        # 获取ETF列表
        etf_df = self.stock_cn.get_all_etf_list()
        
        # 验证返回类型
        self.assertIsInstance(
            etf_df, pd.DataFrame,
            "返回类型应为DataFrame"
        )
        
        # 验证列名
        expected_columns = ['code', 'name', 'net_asset_value']
        self.assertTrue(
            all(col in etf_df.columns for col in expected_columns),
            f"DataFrame应包含列: {expected_columns}"
        )
        
        # 验证数据完整性(如果DataFrame不为空)
        if not etf_df.empty:
            self.assertFalse(
                etf_df['code'].isnull().any(), 
                "基金代码不应有空值"
            )
            self.assertFalse(
                etf_df['name'].isnull().any(),
                "基金简称不应有空值"
            )
            self.assertFalse(
                etf_df['net_asset_value'].isnull().any(),
                "基金资产净值不应有空值"
            )

    def test_realtime_data_fetch(self):
        """测试实时数据获取功能"""
        # 这里需要mock实时数据获取，因为实际调用会访问外部API
        # 暂时注释掉，需要创建mock测试
        # test_symbol = '600000'
        # data = self.stock_cn.fetch_and_store_realtime(test_symbol)
        # self.assertIsNotNone(data)
        # self.assertIn('open', data)
        # self.assertIn('high', data)
        # self.assertIn('low', data)
        # self.assertIn('close', data)
        pass


if __name__ == '__main__':
    unittest.main()