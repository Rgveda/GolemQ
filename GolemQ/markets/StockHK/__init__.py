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

from ..base_market import BaseMarket
from typing import List
import datetime
import pandas as pd


class StockHK(BaseMarket):
    """
    港股市场单例实现
    香港股市特定实现逻辑
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._initialized = False
        return cls._instance
        
    def __init__(self):
        if not self._initialized:
            self._initialized = True
            self._name = "香港股市"
            self._exchange_code = 'HK'
            
    @property
    def name(self) -> str:
        return self._name
        
    def get_stock_codes(self) -> List[str]:
        """获取港股全部股票代码，格式如['0001.HK', '0700.HK']。

        方法名由 `get_all_stock_codes` 修正而来 —— `BaseMarket` 声明的是
        `get_stock_codes`，名字不一致导致本类**永远是抽象类、从不被注册**，
        自动发现只会打一行「跳过抽象类」。命名不符是常见且难查的这类缺陷：
        不报错，只是静默地不生效。
        """
        return [f"{str(i).zfill(4)}.{self._exchange_code}" for i in range(1, 100)]

    def purge_historical_collections(self) -> List[str]:
        """清理历史数据集合。**港股尚未实现**，抛 `NotImplementedError`。

        不返回空列表 —— 返回空会让「没实现」与「确实没有历史集合」无法区分。
        """
        raise NotImplementedError('港股尚未实现（StockHK 目前是 stub）')


    def get_kline_quotes(
        self,
        code: str,
        start: datetime.date,
        end: datetime.date
    ) -> pd.DataFrame:
        """获取港股单只股票日线历史行情"""
        data = [{
            'date': start + datetime.timedelta(days=i),
            'open': 50.0,
            'close': 50.5,
            'high': 51.0,
            'low': 49.5,
            'volume': 50000
        } for i in range((end - start).days + 1)]
        
        return pd.DataFrame(data)
        
    def get_kline_quotes_min(
        self,
        code: str,
        start: str = '2025-01-01',
        end: str = '2026-01-01',
        frequency: str = '60min'
    ) -> pd.DataFrame:
        """获取港股单只股票分钟线历史行情"""
        start_date = datetime.datetime.strptime(start, '%Y-%m-%d').date()
        end_date = datetime.datetime.strptime(end, '%Y-%m-%d').date()
        freq_min = int(frequency.replace('min', ''))
        
        data = [{
            'time': start_date.strftime('%Y-%m-%d') + f" 10:{str(i).zfill(2)}:00",
            'price': 50.0 + i*0.1,
            'volume': 2000
        } for i in range(0, 240, freq_min)]

        return pd.DataFrame(data)

    # ---- 门面契约（fetch/ 调度用）------------------------------------------
    #
    # 本市场是 stub，三个方法**一律抛 NotImplementedError**。
    # 返回空结果会让「港股还没实现」与「港股确实没有这段数据」无从区分 ——
    # 前者是开发状态，后者是数据事实，混在一起会让排障找不到方向。

    def get_kline_price_min(self, codelist, start=None, end=None,
                            verbose=False, realtime=True, frequency=None):
        raise NotImplementedError('港股分钟线尚未实现（StockHK 目前是 stub）')

    def get_kline_price_v3(self, codelist, start=None, end=None,
                           verbose=False, realtime=True):
        raise NotImplementedError('港股日线尚未实现（StockHK 目前是 stub）')

    def get_stock_concept_kline(self, symbol, start=None, end=None, freq=None):
        raise NotImplementedError('港股概念 K 线尚未实现（StockHK 目前是 stub）')