# coding:utf-8
"""A 股参考数据的读取层 —— 一律走 MongoDB 8.3 的 `golemq_stock_cn`。

为什么单独成模块
================
`scribe.py` 里已有几个同形态的 reader（`find()` → `drop('_id')` →
`set_index('code')`），但它们读的是 `DATABASE`（QUANTAXIS 驱动的 **4.4** `golemq`）。
参考数据的所有权正在从 4.4 的 `quantaxis` 迁到 8.3 的 `golemq_stock_cn`，
本模块是新家。

**为什么是独立模块而不是塞进 `scribe.py`**：`scribe.py` import 了 `symbol.py`
（`:54`），而 `symbol.py` 若反过来 import `scribe.py` 会成环。本模块只依赖
`core.settings` 与 `pandas`，是个叶子，谁都能 import 它。

读取器一律返回 DataFrame、一律 `set_index('code', drop=False)`
==============================================================
与既有 reader 保持同一形态，调用方不必区分数据来自哪个库：

    df = GQ_fetch_stock_list()
    df.loc['600519', 'name']

`drop=False` 是有意的：`code` 既是索引也是列，两种取法都能用 —— 旧 reader
就是这个约定，改了会打断既有调用点。

缺库/空集合返回**空 DataFrame 而非 None**
=========================================
调用方普遍做 `df.empty` / `len(df)` 判断；返回 None 会把「没数据」变成
`AttributeError`，那是形状错误、会被误读成 bug（kline83 里踩过同一个坑）。
"""
from __future__ import annotations

import pandas as pd

def _stock_cn_db():
    """8.3 库句柄。**函数级导入是刻意的。**

    `markets/StockCN/__init__.py:55` 经 `from .quotes import StockCNQuotes`
    间接导入 `fetch.py`，而 `fetch.py` 要 import 本模块；`GOLEMQ_STOCK_CN`
    到 `:70` 才定义。顶层 `from GolemQ.markets.StockCN import GOLEMQ_STOCK_CN`
    会抛 `ImportError: cannot import name ... from partially initialized module`。
    `kline83.py` 与 `datastruct.py` 出于同一原因也这样写。

    在本模块出现第一个导入者（`fetch.py`）之前，这行一直是模块级的且**恰好**
    能用 —— 它只被独立导入过，而那时父包已初始化完毕。
    """
    from GolemQ.markets.StockCN import GOLEMQ_STOCK_CN
    return GOLEMQ_STOCK_CN

__all__ = [
    'REF_COLLECTIONS',
    'GQ_fetch_stock_list',
    'GQ_fetch_stock_info',
    'GQ_fetch_etf_list',
    'GQ_fetch_stock_block',
    'GQ_fetch_financial',
    'GQ_ref_collection',
]

#: 5 个参考集合的规范名
REF_COLLECTIONS = ('stock_list', 'stock_info', 'etf_list', 'stock_block', 'financial')


def GQ_ref_collection(name: str, database=None):
    """取集合句柄。`name` 必须是 5 个规范名之一。"""
    if name not in REF_COLLECTIONS:
        raise KeyError(f'未知参考集合 {name!r}；可用: {REF_COLLECTIONS}')
    return (database or _stock_cn_db())[name]


def _read(name: str, query: dict = None, projection: dict = None,
          index: str = 'code', database=None) -> pd.DataFrame:
    """通用读取：`find()` → DataFrame → 丢 `_id` → 设索引。

    空集合返回空 DataFrame（不是 None），见模块文档。
    """
    coll = GQ_ref_collection(name, database=database)
    cursor = coll.find(query or {}, projection or {'_id': 0})
    rows = list(cursor)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if '_id' in df.columns:
        df = df.drop('_id', axis=1, inplace=False)
    if index and index in df.columns:
        df = df.set_index(index, drop=False)
    return df


def GQ_fetch_stock_list(codes=None, database=None) -> pd.DataFrame:
    """股票列表。`code` 为 6 位，另有 `name`/`pre_close`/`sse`/`sec`
    /`volunit`/`decimal_point`/`source`。"""
    q = {'code': {'$in': list(codes)}} if codes is not None else None
    return _read('stock_list', q, database=database)


def GQ_fetch_stock_info(codes=None, database=None) -> pd.DataFrame:
    """股本与上市信息。真实被消费的是 `liutongguben` 与 `ipo_date`/`IPODate`。"""
    q = {'code': {'$in': list(codes)}} if codes is not None else None
    return _read('stock_info', q, database=database)


def GQ_fetch_etf_list(codes=None, database=None) -> pd.DataFrame:
    """ETF 列表。`sec` 恒为 `'etf_cn'`。

    ⚠️ 注意与 `symbol.GQ_fetch_etf_list` 的区别 —— 那个读的是**股票**列表
    （历史遗留缺陷，见 `MIGRATION_STATUS.md`）。本函数读的才是 `etf_list`。
    """
    q = {'code': {'$in': list(codes)}} if codes is not None else None
    return _read('etf_list', q, database=database)


def GQ_fetch_stock_block(blocknames=None, codes=None, database=None) -> pd.DataFrame:
    """板块成分。

    **索引是 `(blockname, code)` 两层**，不是 `code` —— 因为 `stock_block` 的
    唯一键就是这两者的组合，一只股票属于多个板块、一个板块含多只股票。
    """
    q: dict = {}
    if blocknames is not None:
        q['blockname'] = {'$in': list(blocknames)}
    if codes is not None:
        q['code'] = {'$in': list(codes)}
    coll = GQ_ref_collection('stock_block', database=database)
    rows = list(coll.find(q, {'_id': 0}))
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    return df.set_index(['blockname', 'code']).sort_index()


def GQ_fetch_financial(codes=None, report_date=None, database=None) -> pd.DataFrame:
    """季频财务。一行 = 一个 `(code, report_date)`。

    列里除 `code`/`report_date`/`year`/`quarter`/`source` 外，其余是**指标**，
    命名取决于来源（akshare 是中文键、tdxaidata 是英文键，见各自适配器的文档）。
    跨源混用时先看清列名，别假设两边一致。
    """
    q: dict = {}
    if codes is not None:
        q['code'] = {'$in': list(codes)}
    if report_date is not None:
        q['report_date'] = report_date
    coll = GQ_ref_collection('financial', database=database)
    rows = list(coll.find(q, {'_id': 0}))
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if 'report_date' in df.columns and 'code' in df.columns:
        df = df.set_index(['code', 'report_date']).sort_index()
    return df
