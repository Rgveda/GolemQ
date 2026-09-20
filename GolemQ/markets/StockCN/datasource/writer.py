# coding:utf-8
"""参考集合的通用落库器。

移植自 `GolemQ_old/gateway/xtquant/save_qa.py` 三个 writer 的写语义，抽成一处。
**不引入新的写策略** —— 旧实现的行为在实盘跑过，照搬比重新设计安全。

三条必须保留的语义
==================
1. **空结果绝不删数据。** 旧实现的守卫（`save_qa.py:674-727`）是：取到空就
   直接返回，绝不执行 delta 删除。这条是**承重**的 —— 上游一次抽风返回空，
   若照常做 delta 删除，会把整个集合清空。取数失败与「确实没有标的退市」在
   返回值上无从区分。
2. **upsert 而非 insert。** 重复跑不应产生重复文档。
3. **删 delta 按明确的键。** `stock_list`/`stock_info` 按 `code`；
   `stock_block` 按 `(blockname, code)`，且要先按板块再整体两层删
   （某板块整体消失 vs 板块内某成员消失，是两种不同的删除）。
"""
from __future__ import annotations

from pymongo import ReplaceOne

#: 一次 bulk_write 的批大小。与旧实现一致。
BATCH = 1000


def ensure_indexes(coll, unique_keys, **kwargs):
    """建唯一索引。已存在同键索引时不报错。"""
    try:
        coll.create_index(unique_keys, unique=True, **kwargs)
    except Exception:
        # 索引已存在（键或选项不同）不应中断落库
        pass


def save_collection(coll, rows, unique_keys, delete_delta_key=None,
                    batch: int = BATCH, verbose: bool = False) -> dict:
    """把 `rows` upsert 进 `coll`，可选删 delta。

    :param coll: pymongo Collection
    :param rows: list[dict]，GolemQ 口径
    :param unique_keys: upsert 键，如 ``['code']`` 或 ``['blockname','code']``
    :param delete_delta_key: 若非 None，删除该键不在本次结果里的既有文档。
        **仅当 `rows` 非空时才会执行** —— 见模块文档第 1 条。
    :returns: ``{'upserted': n, 'modified': n, 'deleted': n, 'skipped': bool}``
    """
    stats = {'upserted': 0, 'modified': 0, 'deleted': 0, 'skipped': False}

    if not rows:
        # 承重守卫：取到空绝不删。见模块文档。
        stats['skipped'] = True
        if verbose:
            print(f'[writer] {coll.name}: 取到 0 行，跳过写入与 delta 删除')
        return stats

    ensure_indexes(coll, unique_keys)

    for i in range(0, len(rows), batch):
        chunk = rows[i:i + batch]
        ops = [
            ReplaceOne({k: r[k] for k in unique_keys}, r, upsert=True)
            for r in chunk
            if all(k in r for k in unique_keys)
        ]
        if not ops:
            continue
        res = coll.bulk_write(ops, ordered=False)
        stats['upserted'] += res.upserted_count
        stats['modified'] += res.modified_count

    if delete_delta_key is not None:
        seen = {r[delete_delta_key] for r in rows if delete_delta_key in r}
        res = coll.delete_many({delete_delta_key: {'$nin': sorted(seen)}})
        stats['deleted'] = res.deleted_count

    if verbose:
        print(f'[writer] {coll.name}: {stats}')
    return stats


def save_block_collection(coll, rows, batch: int = BATCH, verbose: bool = False):
    """`stock_block` 专用：键是 (blockname, code)，删除分两层。

    旧实现（`save_qa.py:772-786`）的删除语义分两步，不能合成一步：
      1. 本次出现过的每个板块，删掉该板块内不再出现的成员
      2. 删掉本次完全没出现的板块的全部成员

    只做第 2 步会漏掉「板块还在、但某只成分股被调出」；
    只做第 1 步会漏掉「整个板块消失」。**两种都是真实发生的事件。**
    """
    stats = {'upserted': 0, 'modified': 0, 'deleted': 0, 'skipped': False}

    if not rows:
        stats['skipped'] = True
        if verbose:
            print(f'[writer] {coll.name}: 取到 0 行，跳过写入与 delta 删除')
        return stats

    ensure_indexes(coll, ['blockname', 'code'])

    for i in range(0, len(rows), batch):
        chunk = rows[i:i + batch]
        ops = [
            ReplaceOne({'blockname': r['blockname'], 'code': r['code']}, r, upsert=True)
            for r in chunk
        ]
        res = coll.bulk_write(ops, ordered=False)
        stats['upserted'] += res.upserted_count
        stats['modified'] += res.modified_count

    by_block: dict = {}
    for r in rows:
        by_block.setdefault(r['blockname'], set()).add(r['code'])

    deleted = 0
    for blockname, codes in by_block.items():
        res = coll.delete_many({
            'blockname': blockname,
            'code': {'$nin': sorted(codes)},
        })
        deleted += res.deleted_count
    res = coll.delete_many({'blockname': {'$nin': sorted(by_block)}})
    deleted += res.deleted_count
    stats['deleted'] = deleted

    if verbose:
        print(f'[writer] {coll.name}: {stats}')
    return stats
