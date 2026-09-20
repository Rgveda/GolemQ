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
        
    # ---- 门面契约（fetch/ 调度用）------------------------------------------
    #
    # 与上面的 get_kline_quotes / get_kline_quotes_min 的区别：
    #   上面两个返回**裸 DataFrame**，是「查一段行情」的简接口；
    #   下面三个返回 **`(结果对象, codename)` 二元组**，是 services 层实际消费的
    #   契约 —— 它们要 `.data`（两层 (时间, code) MultiIndex），还要 codename。
    #
    # 之所以不退化成一种：前者已被 `StockCN.get_kline_quotes*` 实现，改动会波及
    # Quotes 层；后者的二元组形态是 `services/persistence/*` 逐个调用点依赖的。
    # 两套并存是既成事实，此处**显式声明**以免后来者以为可以随手合并。

    @abstractmethod
    def get_kline_price_min(self, codelist, start=None, end=None,
                            verbose=False, realtime=True):
        """分钟线。返回 `(结果对象, codename)`。

        **无数据时返回空结果对象，不是 None** —— `services/persistence/_stock.py:138`
        直接取 `.data` 且未预初始化目标变量，返回 None 会一路变成
        `UnboundLocalError`。详见 `markets/StockCN/MONGODB83.md`。
        """
        pass

    @abstractmethod
    def get_kline_price_v3(self, codelist, start=None, end=None,
                           verbose=False, realtime=True):
        """日线。返回 `(结果对象 | None, codename)`。

        **无数据时返回 None** —— `services/persistence/_daily.py:105` 有
        `if data_baseline is None` 分支依赖它。与上面那个刻意相反，勿「统一」。
        """
        pass

    @abstractmethod
    def get_stock_concept_kline(self, symbol, start=None, end=None, freq=None):
        """概念 K 线。**未实现的市场应抛 NotImplementedError** ——
        返回空会让「未实现」与「真的没有概念数据」无从区分。"""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """返回市场名称"""
        pass
