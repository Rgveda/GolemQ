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
from GolemQ.core.market_registry import (
    register_market,
    register_subscriber,
)
from .realtime import (
    sub_l1_from_tencent,
    sub_l2_from_tencent,
    formater_l1_ticks,
    collections_of_today,
)
from .symbol import (
    is_stock_cn,
    normalize_code,
)
import atexit
from .quotes import StockCNQuotes


mongo_uri = GQSETTING.get_config('MONGODB', 'uri')
DATABASE = GQ_util_mongodb_client(mongo_uri)

# A 股市场专属的 MongoDB 8.3 时序库。
#
# 库名在此硬编码，不写入配置：StockCN 即 A 股市场本身，存储库名是市场定义的
# 一部分，不随部署环境变化。连接地址仍取自 ~/.GolemQ/settings/config.ini
# 的 [MONGODB] uri（见上）。
#
# 内容：由旧 GolemQ 系统的 MongoDB 4.4 (stock_min / index_min) 迁移而来，
# 为 timeField=ts、metaField=code 的时序集合，分 period 存于
# stock_1min|5min|15min|30min|60min 与 index_*。读路径见 kline83.py。
GOLEMQ_STOCK_CN_NAME = 'golemq_stock_cn'
GOLEMQ_STOCK_CN = DATABASE[GOLEMQ_STOCK_CN_NAME]

# 实时行情（L1/L2）的库。与 `golemq_stock_cn`（历史行情）分开：
# 实时是**追加写、不复权、按 ts 时间序列**，与历史库的读多写少性质不同。
# 库名同理由市场硬编码（同上面那条）。
#
# 这解决了原先 `self.GQREALTIME` 的「待定」：它当时指向
# `DATABASE.GolemQ_StockCN_REALTIME`，而那个库在 8.3 服务器上**并不存在**
# （实测 0 集合）。项目所有者 2026-09-21 定为 `golemq_stock_cn_realtime`。
GOLEMQ_STOCK_CN_REALTIME_NAME = 'golemq_stock_cn_realtime'
GOLEMQ_STOCK_CN_REALTIME = DATABASE[GOLEMQ_STOCK_CN_REALTIME_NAME]


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
            
            # 命名规则：**名字即库名**（用户 2026-10-08 定）。`self.DATABASE` /
            # `self.GQREALTIME` 那两个 QUANTAXIS 时代的名字已随 D12 一并去掉。
            self.GOLEMQ_STOCK_CN = GOLEMQ_STOCK_CN
            self.GOLEMQ_STOCK_CN_REALTIME = GOLEMQ_STOCK_CN_REALTIME
            self.quotes = StockCNQuotes()

            # 注册到全局市场注册表。register_market/register_subscriber 默认
            # 不覆盖，重复注册返回 False —— 语义与原先的 `if ... not in` 一致，
            # 但把「重复注册怎么办」收敛到一处，不在每个市场里各写一遍。
            register_market('StockCN', self)
            register_subscriber('l1_tencent', sub_l1_from_tencent)
            # 老树的键名（`GolemQ_old/cli/__main__.py` 里的 `--sub tencent`）
            # —— **同一个函数**，保留旧键是为了让已有的脚本与肌肉记忆继续可用：
            # 老树那条也是「腾讯 L1 写实时库」，只是当时写 4.4 的 `QAREALTIME`，
            # 现在写 8.3 的 `golemq_stock_cn_realtime`（见 `realtime._realtime_db`）。
            register_subscriber('tencent', sub_l1_from_tencent)
            # L2 五档盘口：股票走腾讯（3 秒一轮），ETF 走 MiniQMT。
            # 新浪那条 L2 已加 30s 请求限制，无法连续取，故不在此列。
            register_subscriber('l2_tencent', sub_l2_from_tencent)

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

    # ---- 门面契约的实现（`GolemQ/fetch/*` 调度到这里）----------------------
    #
    # 委托给 `kline83`（MongoDB 8.3 时序读路径）。契约与空值语义见
    # `markets/base_market.py` 的声明，两处**必须一致**。

    def get_kline_price_min(self, codelist, start=None, end=None,
                            verbose=False, realtime=True):
        """分钟线（A 股）。见 `base_market.py` 的契约说明。"""
        from .kline83 import get_kline_price_min as _impl
        return _impl(codelist, start=start, end=end,
                     verbose=verbose, realtime=realtime)

    def get_kline_price_v3(self, codelist, start=None, end=None,
                           verbose=False, realtime=True):
        """日线（A 股）。无数据返回 `None`，见 `base_market.py`。"""
        from .kline83 import get_kline_price_v3 as _impl
        return _impl(codelist, start=start, end=end,
                     verbose=verbose, realtime=realtime)

    def get_stock_concept_kline(self, symbol, start=None, end=None, freq=None):
        """概念 K 线（A 股）。

        **当前未实现** —— 真实实现在老树 `GolemQ_old/fetch/concept.py:865`（读 4.4），
        尚未移植。此处抛 `NotImplementedError` 而非返回空表：
        返回空会让「未实现」与「真的没有概念数据」无从区分。
        """
        raise NotImplementedError(
            'A 股概念 K 线尚未实现；真实实现待从 GolemQ_old/fetch/concept.py:865 移植。')


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
        return purge_historical_collections(self.GOLEMQ_STOCK_CN_REALTIME)


# 在模块导入时自动创建并注册 StockCN 单例实例
# 这样确保 StockCN 模块被导入时就会自动注册到 GQMARKETS
_stockcn_instance = StockCN()


# 导出公共接口
__all__ = [
    'StockCN',
    'GOLEMQ_STOCK_CN',
    'GOLEMQ_STOCK_CN_NAME',
    'normalize_code',
    'is_stock_cn',
    'sub_l1_from_tencent',
    'sub_l2_from_tencent',
    'formater_l1_ticks',
    'collections_of_today',
]

