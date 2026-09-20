# coding:utf-8
"""腾讯行情（easyquotation）适配器。

定位：**实时快照源，不是参考数据源**
====================================
腾讯是这 8 个源里唯一天生做**实时 L1 快照**的（`qt.gtimg.cn`）。它对 5 个参考
集合的能力天然受限 —— 实测 `real()` 返回的字段里**没有任何股本数据**：

    有：code name close(昨收) open high low now volume turnover
        总市值 流通市值 市盈(动/静) PB 涨跌 量比 五档委买卖
    无：liutongguben / zongguben / ipo_date / 报告期财务 / 板块成分

故本适配器**只声明 `stock_list`**，且是**降级形态**（见下）。其余四个集合
走 :class:`UnsupportedCollection` 明确报错 —— 不假装能供。

⚠️ `stock_list` 是降级形态，派生字段已标注
==========================================
腾讯只给 `code`/`name`/`close`（即昨收）。目标 schema 还要求
`volunit`/`decimal_point`/`sse`，它们**是派生的、不是取的**：

================  ====================================================
字段               来源
================  ====================================================
`code`            取
`name`            取
`pre_close`       取（腾讯的 `close` 字段）
`sse`             **按代码号段派生**（60/68→sh、00/30→sz、82/92→bj）
`volunit`         **写死 100**（A 股每手 100 股，是全市场约定）
`decimal_point`   **写死 2**（A 股报价小数位）
`sec`             写死 'stock_cn'
================  ====================================================

因此它是 **fallback 源**，不应作为 `stock_list` 的主源 —— 主源是 pytdx
（字段全取自行情端，无派生）。派生字段若与真实情况不符（如某些 B 股/基金的
decimal_point 非 2），本源的输出会与主源不一致。

⚠️ 全量拉取代价高
=================
腾讯按批返回（easyquotation `max_num=800`），5591 只约需 7 次请求。但它是
**HTTP 源，有频率控制**，故沿用 30s 默认间隔 —— 全量一轮约 3 分钟。
"""
from __future__ import annotations

from .base import (
    STOCK_LIST,
    DataSource,
    DataSourceNotAvailable,
    UnsupportedCollection,
    register,
)


def _sse_of(code: str) -> str:
    if code.startswith(('60', '68')):
        return 'sh'
    if code.startswith(('00', '30')):
        return 'sz'
    if code.startswith(('82', '92')):
        return 'bj'
    return 'sh'


@register
class TencentSource(DataSource):
    name = 'tencent'
    collections = (STOCK_LIST,)
    #: HTTP 源，有频率控制
    default_interval = 30.0

    def available(self) -> bool:
        try:
            from GolemQ.markets.StockCN.easyquotation import use  # noqa: F401
        except Exception:  # noqa: BLE001
            return False
        return True

    def fetch(self, collection: str, **kwargs) -> list:
        if not self.supports(collection):
            raise UnsupportedCollection(
                f'{self.name} 不提供 {collection}（可提供 {self.collections}）—— '
                f'腾讯是实时快照源，无股本/财务/板块数据'
            )
        if not self.available():
            raise DataSourceNotAvailable('easyquotation 不可用')
        self.gate()
        return self.fetch_stock_list(**kwargs)

    def fetch_stock_list(self, codelist=None, verbose: bool = False) -> list:
        """降级形态的 stock_list。`codelist` 省略则用 pytdx 的全市场名单做输入。

        之所以要传名单：腾讯的 `market_snapshot()` 会拉全市场，但它返回的是
        **快照**，代码集与「是否有这个标的」无关（停牌也可能返回），
        不如以主源的名单为准、用腾讯补名字与昨收。
        """
        from GolemQ.markets.StockCN.easyquotation import use

        if codelist is None:
            from . import get_source
            codelist = [r['code'] for r in get_source('pytdx').fetch('stock_list')]
        elif isinstance(codelist, str):
            codelist = [codelist]

        q = use('tencent')
        snap = q.real(list(codelist)) or {}
        rows = []
        for code, item in snap.items():
            code = str(code)[-6:]
            rows.append({
                'code': code,
                'volunit': 100,                       # 派生：A 股每手 100 股
                'decimal_point': 2,                   # 派生：A 股报价小数位
                'name': item.get('name'),
                'pre_close': item.get('close'),        # 腾讯的 close 即昨收
                'sse': _sse_of(code),                 # 派生：按号段
                'sec': 'stock_cn',
                'source': self.name,
            })
        if verbose:
            print(f'[tencent:stock_list] 取到 {len(rows)} 条（降级形态）')
        return rows
