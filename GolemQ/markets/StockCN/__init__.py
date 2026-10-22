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
            # ⚠️ **`l2_tencent` 键 2026-10-15 已删**（`docs/decisions/D47.md`）：
            # 腾讯 L2 存的字段是 L1 行的**真子集**（L1 存整份快照、本就含五档），
            # 而且它 3 秒一轮、L1 2 秒一轮 —— **既不多个字段、也不快**，
            # 纯粹是同一份 HTTP 响应落两遍。QMT 那条 ETF 占位同批删除。
            # 于是 `GQSUBSCRIBER` 现在只有 `l1_tencent` / `tencent` 两个键。

    def get_database(self):
        """行情库句柄 —— **市场包外部取库的唯一入口**。

        「库名不外泄」：库名 ``golemq_stock_cn`` 与句柄名 ``GOLEMQ_STOCK_CN``
        只允许出现在 ``markets/StockCN/`` 之内。别处（pipeline / CLI / analysis /
        services）一律 ``get_active_market().get_database()``，不要引用句柄常量、
        不要自己连库、不要拼库名 —— 判错**不会报错**，只会静默读到另一个库/空集。
        """
        return self.GOLEMQ_STOCK_CN

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

    # ---- 市场契约的实现（`BaseMarket` 的方法，直接由调用方经市场实例取用）----
    #
    # 取用方式：`GolemQ.get_active_market().get_kline_price_min(...)`
    # （2026-10-10 起 `GolemQ/fetch/` 整包已删，没有中转门面了）
    #
    # 委托给 `kline83`（MongoDB 8.3 时序读路径）。契约与空值语义见
    # `markets/base_market.py` 的声明，两处**必须一致**。

    def get_kline_price_min(self, codelist, start=None, end=None,
                            verbose=False, realtime=True, frequency=None,
                            if_fq='qfq'):
        """分钟线（A 股）。见 `base_market.py` 的契约说明。

        `frequency=None` ⇒ **透传给 kline83 的那个默认**（当前 `'60min'`）。
        刻意用「只在给了才传」的写法：默认值只此一处（`kline83` 的函数签名），
        在这里再写一遍 `or '60min'` 就是第二份定义，会分叉。

        `if_fq`：`'qfq'`（默认，前复权）/ `'hfq'`（后复权，锚**窗口首根**）/ `'bfq'`（不复权）。
        """
        from .kline83 import get_kline_price_min as _impl
        extra = {} if frequency is None else {'frequency': frequency}
        return _impl(codelist, start=start, end=end,
                     verbose=verbose, realtime=realtime, if_fq=if_fq, **extra)

    def get_kline_price_v3(self, codelist, start=None, end=None,
                           verbose=False, realtime=True, if_fq='qfq'):
        """日线（A 股）。无数据返回 `None`，见 `base_market.py`。

        `if_fq`：`'qfq'`（默认，前复权）/ `'hfq'`（后复权，锚**窗口首根**）/ `'bfq'`（不复权）。
        """
        from .kline83 import get_kline_price_v3 as _impl
        return _impl(codelist, start=start, end=end,
                     verbose=verbose, realtime=realtime, if_fq=if_fq)

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
                
    def scan_dataset(self):
        """scan scope 的**数据落点** —— 供 :mod:`GolemQ.analysis.cv.scopes` 用。

        那个模块是**跨市场**的，它不 import 任何市场包，只认这个返回值（契约见
        `ScanDataset`）。这里做的是**薄封**：四个成员一一对应本站**已有的**读口，
        不新造取数逻辑。

        ⚠️ **别拿 `get_stock_codes()` 当这里的 `stock_list`** —— 那是个 mock，
        返回 `600xxx.SH` 之类的**假码**（它名字像真口子）。真名单在
        `refdata.GQ_fetch_stock_list`。
        """
        from GolemQ.analysis.cv.scopes import ScanDataset
        from .refdata import (GQ_fetch_etf_list, GQ_fetch_index_list,
                              GQ_fetch_stock_block, GQ_fetch_stock_list)
        from .symbol import focus_block

        # `etf` scope 用 **8.3 的 `etf_list`**（2026-10-18 起）。
        # 旧树那份 **506 码的硬编码表已删**（原 `symbol.ETF_LEGACY_CODES`）——
        # 实测它比库里的名单**少 1244 只**，还留着 **44 只已退市/换码**的码
        # （其中 13 只近一月一行数据都没有 ⇒ 判 mode=1 ⇒ 拆成 ~34 个月度任务 ⇒
        # 把休市节流的计数从 13 撑到 429，ETF 因此不休）。名单以库为准。
        etf = GQ_fetch_etf_list()
        etf_codes = () if etf is None or len(etf) == 0 else tuple(
            str(c)[:6] for c in etf['code'])
        idx = GQ_fetch_index_list()
        index_codes = () if idx is None or len(idx) == 0 else tuple(
            str(c)[:6] for c in idx['code'])
        return ScanDataset(
            stock_list=GQ_fetch_stock_list,
            stock_block=GQ_fetch_stock_block,
            focus_blocks=focus_block,
            etf_codes=etf_codes,
            index_codes=index_codes,
        )

    def purge_historical_collections(self):
        """清理历史数据集合"""
        from .tools import purge_historical_collections
        return purge_historical_collections(self.GOLEMQ_STOCK_CN_REALTIME)


# 在模块导入时自动创建并注册 StockCN 单例实例
# 这样确保 StockCN 模块被导入时就会自动注册到 GQMARKETS
_stockcn_instance = StockCN()


# 导出公共接口
# ⚠️ 库句柄 `GOLEMQ_STOCK_CN` / 库名 `GOLEMQ_STOCK_CN_NAME` **刻意不在此列**：
# 它们是市场包内部实现，外部取库只能走 `StockCN.get_database()`（「库名不外泄」）。
__all__ = [
    'StockCN',
    'normalize_code',
    'is_stock_cn',
    'sub_l1_from_tencent',
    'formater_l1_ticks',
    'collections_of_today',
]

