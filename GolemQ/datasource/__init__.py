# coding:utf-8
"""数据源适配的**机制层** —— 与市场无关，任何市场都能用。

分层
====
======================  ==========================================
本包（`GolemQ/datasource/`）  **机制**：适配器基类、注册表、限频、代理、落库
`markets/<Market>/datasource/`  **实现**：各市场的具体适配器与源优先级
======================  ==========================================

**为什么分开**：抓数据这件事任何市场都要做 —— pytdx 之外，美股要 IBKR/Alpaca，
港股要富途/港交所。基类、限频、代理注入、upsert 落库这些**没有一处含市场假设**，
把它们留在某个市场目录下是错的。

### 机制层有什么（本包）

* `base.py` —— `DataSource` 抽象基类、`register()` 注册表、集合名常量
* `throttle.py` —— 按源最小请求间隔（默认 30s）
* `proxy.py` —— 代理注入点（不造池）
* `writer.py` —— 通用 upsert + delta 删除

### 实现层有什么（各市场）

具体的源适配器（pytdx / QMT / akshare …）与**该市场的源优先级表**。
适配器用 `@register` 自注册，导入即生效。

### 注册的触发

机制层**不认识任何市场** —— 注册由导入实现层触发：

    from GolemQ.markets.StockCN import datasource   # 触发各适配器自注册
    from GolemQ.datasource import get_source
    get_source('pytdx')

若只 import 机制层而不 import 任何市场，注册表是空的，`get_source` 会明确报错
并列出已注册项（即空表），**不会静默返回 None**。
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

__all__ = [
    'ALL_COLLECTIONS',
    'ETF_LIST',
    'FINANCIAL',
    'STOCK_BLOCK',
    'STOCK_INFO',
    'STOCK_LIST',
    'DataSource',
    'DataSourceNotAvailable',
    'UnsupportedCollection',
    'register',
    'registry',
    'get_source',
    'sources_for',
    'all_sources',
]


def get_source(name: str, **kwargs) -> DataSource:
    """按名构造数据源实例。未注册则 KeyError，并列出已注册的名字。

    ⚠️ 注册由**导入实现层**触发（见模块文档）。若这里报「已注册: []」，
    通常是因为没有 import 任何市场的 `datasource` 包。
    """
    table = registry()
    if name not in table:
        raise KeyError(
            f'未注册的数据源 {name!r}；已注册: {sorted(table)}。'
            f'提示：注册由导入实现层触发，如 '
            f'`from GolemQ.markets.StockCN import datasource`')
    return table[name](**kwargs)


def sources_for(collection: str) -> list:
    """返回能提供该集合的源名（注册表顺序，非优先级）。"""
    return [n for n, cls in registry().items() if collection in cls.collections]


def all_sources():
    """构造所有已注册源的实例，按名字排序。排障用。"""
    return {n: cls() for n, cls in sorted(registry().items())}
