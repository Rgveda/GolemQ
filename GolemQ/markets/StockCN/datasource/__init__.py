# coding:utf-8
"""A 股的**数据源实现层** —— 具体适配器与源优先级。

机制（基类/注册表/限频/代理/落库）在 `GolemQ/datasource/`，与市场无关。
本包只放**A 股特有的东西**：各家 A 股数据源的适配器，以及「哪个集合优先用哪个源」。

导入本包即完成各适配器的自注册
==============================
模块底部的 import 不是摆设 —— 适配器用 `@register` 装饰器自注册，
**导入即生效**。漏掉任何一个 import，那个源就不在注册表里。

A 股源优先级
============
========================  ==========================================================
集合                        顺序与理由
========================  ==========================================================
`stock_list`               tdxaidata → pytdx → qmt → tencent
                           tdxaidata 首位是**唯一覆盖北交所**（348 只）；但它只给
                           代码集，不给 name/pre_close。pytdx 给名称与昨收但缺北交所。
                           **两者都填入才完整 —— 这是接口边界，不是谁的 bug。**
`stock_block`              pytdx → qmt → tdxaidata
`stock_info`               pytdx → tdxaidata → qmt
                           pytdx 的 `get_finance_info` 字段名与目标 schema 逐字相同，
                           且**无需 QMT 客户端在线**。
`etf_list`                 akshare → tdxaidata
                           ⚠️ **tdxaidata 那一位实测是坏的**（2026-10-09）：
                           `get_trackzs_etf_info` 返回 **0 行 + `[错误码 2] 股票代码错误`**。
                           它作为兜底只会**静默返回空**，故 akshare 实际是唯一可用源。
                           （它按设计也只是「跟踪指数的 ETF」而非全量清单，见适配器 docstring。）
`financial`                akshare → tdxaidata
                           原定 baostock，实测其服务器不可达，改用 akshare。
========================  ==========================================================

表里**只列实测可用的源**。骨架源（baostock/tushare/eastmoney）的
`available()` 恒为 False，列进来只会让调用方多走一次注定失败的分支。
⚠️ 上面 `etf_list` 那一行的 tdxaidata **是例外**（表里仍留着，但实测不产出数据）——
留着的坏处是「兜底静默返回空」，好处是不改变既有的回退顺序；改它要连带定回退语义，
故先记账在这里。
"""
from __future__ import annotations

from GolemQ.datasource import (  # noqa: F401  机制层再导出，便于调用方单点导入
    ALL_COLLECTIONS,
    ETF_LIST,
    FINANCIAL,
    INDEX_LIST,
    STOCK_BLOCK,
    STOCK_INFO,
    STOCK_LIST,
    DataSource,
    DataSourceNotAvailable,
    UnsupportedCollection,
    get_source,
    registry,
    sources_for,
)

# 导入即注册 —— 见模块文档，不可省略
from . import akshare_source  # noqa: F401,E402
from . import baostock_source  # noqa: F401,E402
from . import eastmoney_source  # noqa: F401,E402
from . import pytdx_source  # noqa: F401,E402
from . import qmt_source  # noqa: F401,E402
from . import tdxaidata_source  # noqa: F401,E402
from . import tencent_source  # noqa: F401,E402
from . import tushare_source  # noqa: F401,E402

#: A 股的源优先级：集合 → 按序尝试的源名。顺序即优先级。
#: 理由逐条见模块文档。**只列实测可用的源。**
COLLECTION_SOURCE_PRIORITY = {
    # pytdx 主、tdxaidata 补：pytdx 给 code/name/pre_close 真值，但**枚举不到北交所**
    # （`get_security_list(2, 0)` 返回 None 且会毒死连接，见 `pytdx_source` 的模块文档）；
    # tdxaidata 给全 5,579 只但北交所那 351 只**只有代码**（name/pre_close 皆 None）。
    # 两者互补是**接口边界**，不是谁的 bug —— 补行逻辑见 `refdata_save._supplement_stock_list`。
    STOCK_LIST: ['pytdx', 'tdxaidata', 'qmt', 'tencent'],
    STOCK_BLOCK: ['pytdx', 'qmt', 'tdxaidata'],
    STOCK_INFO: ['pytdx', 'tdxaidata', 'qmt'],
    ETF_LIST: ['akshare', 'tdxaidata'],
    FINANCIAL: ['akshare', 'tdxaidata'],
    # ⚠️ **2026-10-12 补**：`INDEX_LIST` 原先**漏在这张表外** —— 它在
    # `ALL_COLLECTIONS`、`REFDATA_BY_SOURCE['pytdx']`、`PytdxSource.collections`
    # **三处都在**，唯独优先级表没有它 ⇒ `_pick_source` 走
    # `COLLECTION_SOURCE_PRIORITY.get('index_list', [])` 拿到**空表**、返回 `None`，
    # `--save tdx` 于是报「index_list 无可用源」（名字都齐了、只有这张表缺一行）。
    # pytdx 走的是**同一份全量证券表**的另一套码段（见 `pytdx_source.INDEX_CODE_PREFIXES`），
    # 所以它与 `STOCK_LIST` 同源互补；tdxaidata 作后备。
    INDEX_LIST: ['pytdx', 'tdxaidata'],
}

__all__ = [
    'COLLECTION_SOURCE_PRIORITY',
    'ALL_COLLECTIONS',
    'get_source',
    'sources_for',
    'registry',
]
