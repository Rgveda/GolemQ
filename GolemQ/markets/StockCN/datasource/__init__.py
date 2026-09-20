# coding:utf-8
"""A 股参考数据的可插拔数据源层。

    from GolemQ.markets.StockCN.datasource import get_source, sources_for

    src = get_source('pytdx')
    if src.available():
        rows = src.fetch('stock_list')

设计要点见各子模块；三条最容易踩的：

* **限频**（`throttle.py`）默认 30s，但那是给有频率控制的 HTTP 源准备的。
  pytdx 走 TCP，其适配器显式把间隔覆盖为 0 —— 不要给它套 30s。
* **代理**（`proxy.py`）只做注入点，不造池；且**对 pytdx 无效**（TCP 非 HTTP）。
* **本包不做数据库操作** —— DB 增删改查归 `services/`。本包产出「行」，
  落库由调用方经 `writer.py` 完成。

集合的源优先级（本轮范围）
==========================
========================  ==================  ==============
集合                        主源                 次源
========================  ==================  ==============
`stock_list`               pytdx                MiniQMT
`stock_block`              pytdx                akshare
`stock_info`               MiniQMT（字段齐）      pytdx（降级）
`etf_list`                 akshare               —
`financial`                baostock              tushare
========================  ==================  ==============

尚未接入的源（tushare / tdxaidata / MiniQMT）留同一接口，逐个补即可。
"""
from __future__ import annotations

from .base import (  # noqa: F401
    ALL_COLLECTIONS,
    ETF_LIST,
    FINANCIAL,
    STOCK_BLOCK,
    STOCK_INFO,
    STOCK_LIST,
    DataSource,
    DataSourceNotAvailable,
    UnsupportedCollection,
    register,
    registry,
)

# 导入即注册。新增源在此加一行 import 即可进入注册表。
from . import pytdx_source  # noqa: F401,E402
from . import qmt_source  # noqa: F401,E402
from . import akshare_source  # noqa: F401,E402
from . import tencent_source  # noqa: F401,E402
from . import baostock_source  # noqa: F401,E402
from . import tushare_source  # noqa: F401,E402
from . import eastmoney_source  # noqa: F401,E402
from . import tdxaidata_source  # noqa: F401,E402

#: 每个集合按序尝试的源。**顺序即优先级，且只列已实测可用的源**。
#: 骨架源（baostock/tushare/eastmoney/tdxaidata）不在此表 —— 它们的
#: `available()` 恒为 False，列进来只会让调用方多走一次注定失败的分支。
COLLECTION_SOURCE_PRIORITY = {
    # tdxaidata 列首位是因为它**唯一覆盖北交所**（348 只）；但它只给代码集，
    # 不给 name/pre_close。pytdx 给名称与昨收但缺北交所。两者都填入才能得到
    # 完整的 stock_list —— 这是接口边界，不是谁有 bug。
    STOCK_LIST: ['tdxaidata', 'pytdx', 'qmt', 'tencent'],
    STOCK_BLOCK: ['pytdx', 'qmt', 'tdxaidata'],
    # pytdx 优先：get_finance_info 给的字段名与目标 schema 逐字相同，且**无需
    # QMT 客户端在线**。tdxaidata 的 get_gb_info 同样给股本，作为第二源。
    STOCK_INFO: ['pytdx', 'tdxaidata', 'qmt'],
    ETF_LIST: ['akshare', 'tdxaidata'],
    # 原定 baostock，实测其服务器不可达，改用 akshare（已实测可用）
    FINANCIAL: ['akshare', 'tdxaidata'],
}


def get_source(name: str, **kwargs) -> DataSource:
    """按名构造数据源实例。未注册则 KeyError，并把已注册的名字列出来。"""
    table = registry()
    if name not in table:
        raise KeyError(f'未注册的数据源 {name!r}；已注册: {sorted(table)}')
    return table[name](**kwargs)


def sources_for(collection: str) -> list:
    """返回能提供该集合的源名列表（注册表顺序，非优先级）。"""
    return [n for n, cls in registry().items() if collection in cls.collections]


def throttled_sources():
    """按配置构造一套节流器与代理，供批量取数入口共用。

    节流器是**进程内按源共享**的，所以同一个源被多个集合调用时仍受同一间隔约束 ——
    这正是我们要的：限的是「对上游的请求速率」，不是「每个集合各限各的」。
    """
    from .proxy import ProxyConfig
    from .throttle import from_settings
    throttle = from_settings()
    proxy = ProxyConfig.from_settings()
    return throttle, proxy
