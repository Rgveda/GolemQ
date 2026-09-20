# coding:utf-8
"""数据源适配层 —— 基类与注册机制。

职责边界
========
本包只负责「从外部数据源取回 GolemQ 口径的行」。**不做任何数据库操作** ——
按 CLAUDE.md 约定，DB 增删改查一律归 `services/`，本包产出的行由调用方交给
`writer.py`，而 writer 只是被 services 层调用的工具。

与 `easyquotation/` 的关系
==========================
**无继承、无导入。** `easyquotation/` 是 vendored 的上游 L1 实时快照库
（腾讯/新浪/集思录），由 `realtime.py` 使用，服务的是**实时行情**；本包供应的是
**参考数据集合**（股票列表、板块、财务等），两者需求不同。
若某个 HTTP 源日后需要 `BaseQuotation` 那样的批量与连接池写法，那是**照抄模式**，
不是建立依赖 —— 限频与代理都在本包，不在 `easyquotation/`。

能力声明而非枚举
================
每个源只声明自己**能供哪几个集合**（`collections`），未声明的调用 `fetch()` 会
明确报错，而不是返回空 —— 空与「不支持」在排障时是两回事。
"""
from __future__ import annotations

import abc

# 5 个参考集合的规范名。全项目以此为准，不要用别名。
STOCK_LIST = 'stock_list'
STOCK_INFO = 'stock_info'
ETF_LIST = 'etf_list'
STOCK_BLOCK = 'stock_block'
FINANCIAL = 'financial'

ALL_COLLECTIONS = (STOCK_LIST, STOCK_INFO, ETF_LIST, STOCK_BLOCK, FINANCIAL)

_REGISTRY: dict = {}


def register(cls):
    """类装饰器：把数据源登记进注册表。以 `cls.name` 为键。

    导入即注册 —— 这正是「机制层不认识市场、注册由导入实现层触发」的实现方式。

    >>> @register
    ... class _Demo(DataSource):
    ...     name = '__doctest_demo__'
    ...     collections = (STOCK_LIST,)
    ...     def fetch(self, collection, **kw): return []
    >>> '__doctest_demo__' in registry()
    True

    未声明 `name` 直接报错（不静默登记成空键）：

    >>> @register
    ... class _NoName(DataSource):
    ...     def fetch(self, collection, **kw): return []
    Traceback (most recent call last):
        ...
    ValueError: _NoName 必须有类属性 name

    集合名写错同样报错 —— 否则该源永远匹配不上任何集合且不报错：

    >>> @register
    ... class _BadColl(DataSource):
    ...     name = '__doctest_bad__'
    ...     collections = ('stock_lits',)      # 拼错
    ...     def fetch(self, collection, **kw): return []
    Traceback (most recent call last):
        ...
    ValueError: _BadColl.collections 含未知集合: ['stock_lits']

    >>> _REGISTRY.pop('__doctest_demo__', None) is not None
    True
    """
    if not getattr(cls, 'name', None):
        raise ValueError(f'{cls.__name__} 必须有类属性 name')
    unknown = set(getattr(cls, 'collections', ())) - set(ALL_COLLECTIONS)
    if unknown:
        raise ValueError(f'{cls.__name__}.collections 含未知集合: {sorted(unknown)}')
    _REGISTRY[cls.name] = cls
    return cls


def registry() -> dict:
    return dict(_REGISTRY)


class DataSourceNotAvailable(RuntimeError):
    """源不可用（依赖缺失、无凭证、网络不通）。"""


class UnsupportedCollection(NotImplementedError):
    """该源不提供此集合。与「取回空结果」严格区分。"""


class DataSource(abc.ABC):
    """数据源基类。

    子类需要:
      * `name`        —— 注册表键，也是限频与代理的按源标识
      * `collections` —— 能供的集合名元组
      * `fetch()`     —— 取回 GolemQ 口径的行
      * `available()` —— 依赖/凭证是否就绪；**不得抛异常**
    """

    name: str = ''
    collections: tuple = ()
    #: 该源的默认请求最小间隔（秒）。由 throttle 覆盖，可读配置。
    default_interval: float = 30.0

    def __init__(self, throttle=None, proxy=None):
        self.throttle = throttle
        self.proxy = proxy
        self._session = None

    # ---- 能力 ----------------------------------------------------------

    def supports(self, collection: str) -> bool:
        return collection in self.collections

    @abc.abstractmethod
    def fetch(self, collection: str, **kwargs) -> list:
        """取回该集合的行（list[dict]，GolemQ 口径）。

        不支持的集合必须抛 :class:`UnsupportedCollection`；
        源不可用必须抛 :class:`DataSourceNotAvailable`。
        **两者都不要用「返回空列表」代替** —— 空列表是「取到了，但确实是空的」。
        """

    def available(self) -> bool:
        """依赖与凭证是否就绪。**不得抛异常** —— 排障路径本身不该成为故障点。"""
        return True

    # ---- 供子类使用的取数闸门 -------------------------------------------

    def gate(self):
        """取数前调用：走限频。所有网络请求都应包在这里面。"""
        if self.throttle is not None:
            self.throttle.wait(self.name)

    def session(self):
        """HTTP 源用的 requests 会话（含代理注入）。非 HTTP 源不应调用。"""
        if self._session is None:
            import requests
            self._session = requests.Session()
            if self.proxy is not None:
                self.proxy.apply(self._session)
        return self._session
