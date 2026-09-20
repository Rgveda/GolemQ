# coding:utf-8
"""A 股参考集合的取数与落库编排 —— 对应 CLI 的 `--save-x` / `--save-qmt`。

为什么在 `markets/StockCN/` 而不是 `pipeline/`
=============================================
这 5 个集合（`stock_list`/`stock_info`/`etf_list`/`stock_block`/`financial`）
**全是 A 股特有概念**：源是通达信/东财/新浪，schema 是 A 股口径，目标是 A 股库。
它们的获取逻辑属于市场本身，不属于通用数据加工。

`pipeline/` 的实际内容是 `base.py` + 三个 benchmark —— 那是**数据加工流水线**
（特征计算、基准对比），不是数据获取。CLI 里已有的先例也印证这一点：
`--stock-min-aligned` 委托给的是 `markets/StockCN/align.py`。

CLAUDE.md 那条「CLI 只做参数校验再委托」的重点是**别把业务逻辑留在 CLI 层**，
而非「必须放 pipeline」。

为什么不并进 `refdata.py`
=========================
`refdata.py` 是**读取叶子**，只依赖 `core.settings` 与 `pandas`，谁都能轻量
import 它。本模块要用 `datasource/` 整包与 `writer`，并进去会让读路径被迫
拖进整个适配层与网络依赖。读写分开。

设计要点
========
**取数与落库分离。** 适配器（`datasource/`）只产出行、不碰 DB；落库由
`datasource/writer.py` 完成。本模块把两者接起来并决定用哪个源。

**不支持的集合要显式报错，不能静默跳过。** 用户点的是"保存全部"，若某个
集合没有可用源，静默跳过会让人以为存过了 —— 这正是本次重构要消除的那类失败。
故返回结果里逐集合标注 `status`（ok / skipped / failed）。
"""
from __future__ import annotations

import datetime as dt

from . import DATABASE_STOCK_CN
from GolemQ.datasource.base import DataSourceNotAvailable, UnsupportedCollection
from GolemQ.datasource.writer import save_block_collection, save_collection

# 实现层：导入即触发 A 股各适配器自注册，并给出该市场的源优先级
from .datasource import COLLECTION_SOURCE_PRIORITY, get_source

__all__ = [
    'ALL_REF_COLLECTIONS',
    'UNIQUE_KEYS',
    'DELTA_DELETE',
    'save_refdata',
    'refdata_status',
    'format_status',
]

#: 5 个参考集合
ALL_REF_COLLECTIONS = ('stock_list', 'stock_info', 'etf_list', 'stock_block', 'financial')

#: 各集合的 upsert 键。`stock_block` 是复合键 —— 一只股票属于多个板块。
UNIQUE_KEYS = {
    'stock_list': ['code'],
    'stock_info': ['code'],
    'etf_list': ['code'],
    'financial': ['code', 'report_date'],
}

#: 允许 delta 删除的集合 —— **不是所有集合都能删差量**。
#: `stock_list`/`stock_info` 的源是全市场权威名单，本次没出现即为退市，可删。
#: `etf_list` 是快照非权威名单，删差量会误删停牌/新上市未收录的品种。
#: `financial` 是历史累积，本次没拉到不代表没有，**绝不能删**。
DELTA_DELETE = {'stock_list', 'stock_info'}


def _pick_source(collection: str, source: str = None, verbose: bool = False):
    """选源。显式指定则用之；否则按优先级取第一个 `available()` 的。"""
    if source:
        s = get_source(source)
        if not s.supports(collection):
            raise UnsupportedCollection(
                f'{source} 不提供 {collection}（可提供 {s.collections or "无"}）')
        if not s.available():
            raise DataSourceNotAvailable(
                f'{source} 当前不可用：'
                f'{getattr(s, "unavailable_reason", lambda: "原因未知")()}')
        return s
    for name in COLLECTION_SOURCE_PRIORITY.get(collection, []):
        s = get_source(name)
        if s.available():
            return s
    return None


def _supplement_stock_list(rows: list, primary_name: str, verbose: bool) -> list:
    """给缺 `name`/`pre_close` 的行从下一个可用源补上。

    只补空字段，**不覆盖主源给的值** —— 主源是权威，补充源只填空。
    """
    need = [r for r in rows if not r.get('name') or r.get('pre_close') in (None, 0)]
    if not need:
        return rows
    for name in COLLECTION_SOURCE_PRIORITY.get('stock_list', []):
        if name == primary_name:
            continue
        try:
            s = get_source(name)
            if not s.available():
                continue
            fetched = s.fetch('stock_list')
        except Exception as exc:      # noqa: BLE001 补充源失败不该毁掉主结果
            if verbose:
                print(f'[refdata] 补充源 {name} 失败: {exc!r}')
            continue

        # 建索引时**显式处理同码**：裸 6 位代码在跨市场时可能重复
        # （沪市指数 000001 ↔ 深市股票 000001）。静默用后写覆盖先写，
        # 会把指数的名字填进股票记录 —— 实测踩过。
        extra: dict = {}
        dup = 0
        for r in fetched:
            c = r['code']
            if c in extra:
                dup += 1
                continue
            extra[c] = r
        if dup and verbose:
            print(f'[refdata] 补充源 {name} 有 {dup} 个重复 code，已保留首条')

        filled = 0
        for r in rows:
            e = extra.get(r['code'])
            if not e:
                continue
            for k in ('name', 'pre_close'):
                if not r.get(k) and e.get(k):
                    r[k] = e[k]
                    filled += 1
        if verbose:
            print(f'[refdata] 以 {name} 补充 {filled} 个字段')
        break
    return rows


def save_refdata(collections=None, source: str = None, codelist=None,
                 verbose: bool = True) -> dict:
    """把参考集合取回并落库到 8.3 的 `golemq_stock_cn`。

    :param collections: 要保存的集合名列表；None = 全部 5 个
    :param source: 强制使用的源名；None = 按优先级自动选
    :param codelist: 仅供逐只查询的集合（`stock_info`/`financial`）限定范围
    :returns: ``{集合名: {'status':..., 'rows':..., 'source':..., 'detail':...}}``
    """
    targets = list(collections) if collections else list(ALL_REF_COLLECTIONS)
    report: dict = {}

    for coll_name in targets:
        entry = {'status': 'skipped', 'rows': 0, 'source': None, 'detail': ''}
        report[coll_name] = entry

        try:
            src = _pick_source(coll_name, source=source, verbose=verbose)
        except (UnsupportedCollection, DataSourceNotAvailable) as exc:
            entry['detail'] = str(exc)
            if verbose:
                print(f'[refdata] {coll_name}: 跳过 —— {exc}')
            continue

        if src is None:
            entry['detail'] = ('无可用源。已注册的源里，能提供该集合的都不可用；'
                               '见各自适配器的 unavailable_reason()')
            if verbose:
                print(f'[refdata] {coll_name}: 跳过 —— {entry["detail"]}')
            continue

        entry['source'] = src.name
        try:
            kwargs = {}
            if codelist is not None and coll_name in ('stock_info', 'financial'):
                kwargs['codelist'] = codelist
            rows = src.fetch(coll_name, **kwargs)
        except Exception as exc:      # noqa: BLE001 逐集合隔离，一个失败不拖垮其余
            entry['status'] = 'failed'
            entry['detail'] = f'{type(exc).__name__}: {exc}'
            if verbose:
                print(f'[refdata] {coll_name}: 失败 —— {entry["detail"]}')
            continue

        if coll_name == 'stock_list' and rows:
            rows = _supplement_stock_list(rows, src.name, verbose)

        coll = DATABASE_STOCK_CN[coll_name]
        try:
            if coll_name == 'stock_block':
                stats = save_block_collection(coll, rows, verbose=verbose)
            else:
                delta_key = {'stock_list': 'code', 'stock_info': 'code'}.get(coll_name)
                stats = save_collection(
                    coll, rows, UNIQUE_KEYS[coll_name],
                    delete_delta_key=delta_key, verbose=verbose)
        except Exception as exc:      # noqa: BLE001
            entry['status'] = 'failed'
            entry['detail'] = f'落库失败 {type(exc).__name__}: {exc}'
            if verbose:
                print(f'[refdata] {coll_name}: {entry["detail"]}')
            continue

        entry['rows'] = len(rows)
        entry['status'] = 'skipped' if stats.get('skipped') else 'ok'
        entry['detail'] = str(stats)
        if verbose:
            print(f'[refdata] {coll_name}: {entry["status"]} {len(rows)} 行 (源 {src.name})')

    return report


def refdata_status() -> dict:
    """各集合当前的源可用性与库存量 —— 排障用。"""
    out = {}
    for name in ALL_REF_COLLECTIONS:
        try:
            count = DATABASE_STOCK_CN[name].estimated_document_count()
        except Exception:  # noqa: BLE001
            count = None
        avail = []
        for s_name in COLLECTION_SOURCE_PRIORITY.get(name, []):
            s = get_source(s_name)
            avail.append({'source': s_name, 'available': s.available()})
        out[name] = {'docs': count, 'sources': avail}
    return out


def format_status(report: dict = None) -> str:
    """把 `save_refdata` 的结果或 `refdata_status()` 渲染成可读文本。"""
    lines = []
    if report is not None:
        lines.append(f'{"集合":<12}{"状态":<9}{"行数":>8}  源 / 说明')
        for name, e in report.items():
            lines.append(f'{name:<12}{e["status"]:<9}{e["rows"]:>8}  '
                         f'{e.get("source") or "-"} {e.get("detail", "")[:60]}')
    else:
        ts = dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        lines.append(f'参考集合状态 @ {ts}')
        for name, e in refdata_status().items():
            srcs = ', '.join(f'{s["source"]}{"" if s["available"] else "(不可用)"}'
                             for s in e['sources']) or '无'
            lines.append(f'  {name:<12} 库存 {e["docs"] if e["docs"] is not None else "?":>8}  {srcs}')
    return '\n'.join(lines)
