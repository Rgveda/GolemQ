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

from abc import ABC, abstractmethod
from typing import List
import pandas as pd
import pymongo


class BaseMarket(ABC):
    """
    金融市场抽象基类
    定义所有市场共有的接口规范

    Attributes:
        DATABASE: pymongo.MongoClient实例，用于存储市场元数据
    """

    DATABASE: pymongo.MongoClient = None

    @classmethod
    def init_db(cls, uri: str = "mongodb://localhost:27017/", db_name: str = "market_metadata"):
        """初始化MongoDB连接"""
        cls.DATABASE = pymongo.MongoClient(uri)[db_name]

    @abstractmethod
    def purge_historical_collections(self) -> List[str]:
        """清理历史数据集合"""
        pass

    @abstractmethod
    def get_stock_codes(self) -> List[str]:
        """获取该市场全部股票代码"""
        pass

    @abstractmethod
    def get_kline_quotes(
        self,
        code: str,
        start: str = '2025-01-01',
        end: str = '2026-01-01',
        fq: int = 1,
    ) -> pd.DataFrame:
        """获取单只股票日线历史行情
        Args:
            code: 股票代码
            start: 开始日期
            end: 结束日期
        Returns:
            历史行情数据列表，每个元素为包含日期和行情数据的字典
        """
        pass
        
    @abstractmethod
    def get_kline_quotes_min(
        self,
        code: str,
        start: str = '2025-01-01',
        end: str = '2026-01-01',
        frequency: str = '60min',
        fq: int = 1,
    ) -> pd.DataFrame:
        """获取单只股票分钟线历史行情
        Args:
            stock_code: 股票代码
            trade_date: 交易日期
            frequency: 分钟频率(1/5/15分钟等)
        Returns:
            分钟行情数据列表
        """
        pass
        
    @property
    @abstractmethod
    def name(self) -> str:
        """返回市场名称"""
        pass
