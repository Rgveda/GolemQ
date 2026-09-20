# coding:utf-8
#
# The MIT License (MIT)
#
# Copyright (c) 2020-2025 azai/GolemQ(uant)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#

from datetime import datetime
from typing import Optional, List
from GolemQ.markets.base_market import BaseMarket
from GolemQ.core.settings import GQSETTING
from GolemQ.core.mongo import GQ_util_mongodb_client
from GolemQ import (
    GQMARKETS,
    GQSUBSCRIBER,
)
from .realtime import (
    sub_l1_from_tencent,
    formater_l1_ticks,
    collections_of_today,
)
from .symbol import (
    is_stock_cn,
    normalize_code,
    is_furture_cn,
    GQ_fetch_stock_info,
    GQ_fetch_etf_name, 
    GQ_fetch_stock_name,
)
import atexit
from .scribe import (
    GQ_get_etf_list,
    GQ_etf_a_spot_em,
    GQ_stock_a_spot_em,
)
from .quotes import StockCNQuotes


mongo_uri = GQSETTING.get_config('MONGODB', 'uri')
DATABASE = GQ_util_mongodb_client(mongo_uri)


def close_mongo_client():
    global DATABASE
    if DATABASE is not None:
        DATABASE.close()
        DATABASE = None


atexit.register(close_mongo_client)


class StockCN(BaseMarket):
    """
    A股市场单例实现
    中国A股市场特定实现逻辑
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._initialized = False
        return cls._instance

    def __init__(self):
        if not self._initialized:
            global DATABASE
            self._initialized = True
            self._name = "中国A股市场"
            self._exchange_codes = ['SH', 'SZ', 'BJ']
            
            self.DATABASE = DATABASE.GolemQ_StockCN
            self.GQREALTIME = DATABASE.GolemQ_StockCN_REALTIME
            self.quotes = StockCNQuotes()

            # 注册到全局市场注册表（确保只注册一次）
            if 'StockCN' not in GQMARKETS:
                GQMARKETS['StockCN'] = self
            if 'l1_tencent' not in GQSUBSCRIBER:
                GQSUBSCRIBER['l1_tencent'] = sub_l1_from_tencent

    @property
    def name(self) -> str:
        return self._name
        
    def get_stock_codes(self) -> List[str]:
        """获取A股全部股票代码，格式如['600000.SH', '000001.SZ']"""
        return [f"{code}.{ex}" for ex in self._exchange_codes
                for code in self._mock_stock_codes(ex)]
                
    def _mock_stock_codes(self, exchange: str) -> list:
        """模拟生成股票代码(示例用)"""
        if exchange == 'SH':
            return [f"600{str(i).zfill(3)}" for i in range(1, 100)]
        else:
            return [f"000{str(i).zfill(3)}" for i in range(1, 100)]
            
    def get_kline_quotes(
        self,
        code: str,
        start: str = '2025-01-01',
        end: str = '2026-01-01',
        fq: int = 1,
    ):
        """获取单只股票日线历史行情"""
        return self.quotes.get_kline_quotes(code, start, end, fq)
        
    def get_kline_quotes_min(
        self,
        code: str,
        start: str = '2025-01-01',
        end: str = '2026-01-01',
        frequency: str = '60min',
        fq: int = 1,
    ):
        """获取单只股票分钟线历史行情"""
        return self.quotes.get_kline_quotes_min(code, start, end, frequency, fq)
            
    # Trading calendar methods
    def is_trading_day(self, date: str) -> bool:
        """Check if a date (YYYY-MM-DD format) is a trading day"""
        from .constants import TRADE_DATE_SSE
        return date in TRADE_DATE_SSE
        
    def get_next_trading_day(self, date: str) -> Optional[str]:
        """Get next trading day after given date (YYYY-MM-DD format)"""
        from .constants import TRADE_DATE_SSE
        try:
            idx = TRADE_DATE_SSE.index(date)
            return TRADE_DATE_SSE[idx + 1] if idx + 1 < len(TRADE_DATE_SSE) else None
        except ValueError:
            return None
            
    def get_previous_trading_day(self, date: str) -> Optional[str]:
        """Get previous trading day before given date (YYYY-MM-DD format)"""
        from .constants import TRADE_DATE_SSE
        try:
            idx = TRADE_DATE_SSE.index(date)
            return TRADE_DATE_SSE[idx - 1] if idx > 0 else None
        except ValueError:
            return None
            
    def get_trading_days(self, start: str, end: str) -> List[str]:
        """Get all trading days between start and end dates (inclusive)"""
        from .constants import TRADE_DATE_SSE
        start_date = datetime.strptime(start, '%Y-%m-%d')
        end_date = datetime.strptime(end, '%Y-%m-%d')
        return [d for d in TRADE_DATE_SSE
                if start_date <= datetime.strptime(d, '%Y-%m-%d') <= end_date]
                
    def purge_historical_collections(self):
        """清理历史数据集合"""
        from .tools import purge_historical_collections
        return purge_historical_collections(self.GQREALTIME)


# 在模块导入时自动创建并注册 StockCN 单例实例
# 这样确保 StockCN 模块被导入时就会自动注册到 GQMARKETS
_stockcn_instance = StockCN()


# 导出公共接口
__all__ = [
    'StockCN',
    'normalize_code',
    'is_stock_cn',
    'is_furture_cn',
    'GQ_etf_a_spot_em',
    'GQ_stock_a_spot_em',
    'GQ_fetch_etf_name',
    'GQ_fetch_stock_name',
    'sub_l1_from_tencent',
    'formater_l1_ticks',
    'collections_of_today',
    'GQ_get_etf_list',
    'GQ_fetch_stock_info',
]

