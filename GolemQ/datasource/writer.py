# coding:utf-8
"""落库的**机制层**：两种写策略，按集合类型选用。

策略一（**普通集合**）：`save_collection` / `save_block_collection` —— upsert + 删 delta。
移植自 `GolemQ_old/gateway/xtquant/save_qa.py` 三个 writer 的写语义，抽成一处。
旧实现的行为在实盘跑过，照搬比重新设计安全。

策略二（**时间序列集合**）：`save_bar_chunk` —— **先删后插**。
时间序列既**不支持唯一索引**也**不能 upsert**（实测 `PITFALLS.md` P14），
所以策略一那两个函数在这里**一律用不了**；而 8.3 的 K 线/xdxr/adj/实时集合
**全部**是时间序列。两者放在同一个模块里，是为了让「同一层有两种写策略」这件事
一眼可见，而不是散在两个文件里各写一半。

两条**都**必须保留的语义
========================
1. **空结果绝不删数据。** 旧实现的守卫（`save_qa.py:674-727`）是：取到空就
   直接返回，绝不执行 delta 删除。这条是**承重**的 —— 上游一次抽风返回空，
   若照常做 delta 删除，会把整个集合清空。取数失败与「确实没有标的退市」在
   返回值上无从区分。`save_bar_chunk` 同守卫：「取数失败」与「确实没有新 bar」
   在返回值上也分不开。
2. **删除条件必须含全部去重键。** 策略一按 `code` / `(blockname, code)`；
   策略二按 `(code, ts)`——少写一个键就会**误删别的行**（P14 实测过实时库上
   L1/L2 互删）。
3. 策略一 **upsert 而非 insert**（重复跑不产生重复文档）；
   策略二做不到 upsert，靠「删同窗口再插」达到同样的幂等。
"""
from __future__ import annotations

from pymongo import ReplaceOne

#: 策略一：一次 bulk_write 的批大小。与旧实现一致。
BATCH = 1000

#: 策略二：一次 insert_many 的批大小。与旧迁移器 `min_migrate83.migrate_chunk` 一致。
BAR_BATCH = 3000


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


def save_bar_chunk(coll, docs, *, code, batch: int = BAR_BATCH,
                   verbose: bool = False) -> dict:
    """**时间序列集合专用**写口：**只 drop 掉本次要写的那几根，再插**。

    为什么不是「删一个区间再写整批」
    ==============================
    已收盘交易日的 bar 是**冻结数据**（不会再变），所以**没有必要、也不应该**
    去动本次要写范围之外的行。原来的写法是 ``delete_many({'code':…, 'ts': {'$gte':…}})``
    —— 那个区间会越过本次的写入集去删东西：

    * 源端某天**没返回**（临时缺数）、而库里那一行是好的 → 会被删掉且**补不回来**（造洞）；
    * 库里更"新"的行（上次误写的脏行）→ 也被顺手删掉，而那不在本次职责内。

    所以删除条件**只含本次要写的 ``(code, ts)`` 全集**：这正是 `PITFALLS.md` P14
    要求的「删除条件必须含全部去重键」，且**不多删一行**。

    ⚠️ **一次只传一只票**（``code`` 必填）：多只票混在一批时，
    ``{'code': {'$in': […x…]}, 'ts': {'$in': […y…]}}`` 是两个集合的**笛卡尔积**，
    会删掉 (票A, 票B的某个 ts) 这种并不存在、也不该动的组合。

    :param docs: 已构造好的文档（本批要写的全部行）
    :param code: 本批所属的标的（6 位代码）
    :returns: ``{'deleted': n, 'inserted': n, 'skipped': bool}``
    """
    stats = {'deleted': 0, 'inserted': 0, 'skipped': False}

    if not docs:
        # 承重守卫：见模块文档第 1 条。**绝不能**因为「没取到」就去删。
        stats['skipped'] = True
        if verbose:
            print(f'[writer] {coll.name}: 取到 0 行，跳过删除与写入')
        return stats

    # 只删本次要写的那些 (code, ts) —— 冻结日一根都不碰
    stats['deleted'] = coll.delete_many({
        'code': code,
        'ts': {'$in': sorted({d['ts'] for d in docs})},
    }).deleted_count

    for i in range(0, len(docs), batch):
        chunk = docs[i:i + batch]
        coll.insert_many(chunk, ordered=False)
        stats['inserted'] += len(chunk)

    if verbose:
        print(f'[writer] {coll.name}: {stats}')
    return stats


def replace_code_rows(coll, docs, *, code, batch: int = BAR_BATCH,
                      verbose: bool = False) -> dict:
    """按 **code 整体替换**：``delete_many({'code': code})`` → 插。

    给 `stock_xdxr` / `stock_adj` 这类「一只票一份完整历史」的表用 ——
    它们的基准会**整体重标**（前复权因子一旦有新除权事件，整条历史都要重算），
    局部替换必然留下「一半旧基准 + 一半新基准」的静默错价。

    ⚠️ 与 :func:`save_bar_chunk` 的区别（**别再合并成一个**）：那个只 drop
    本次要写的那几根（K 线的冻结日一根都不碰）；这个**故意删掉该 code 的全部**，
    因为这两张表没有"冻结"的概念 —— 它们要么整体更新，要么不写。

    空行守卫同 :func:`save_bar_chunk`：``docs`` 为空则**一行都不删**。

    :returns: 同 :func:`save_bar_chunk`
    """
    stats = {'deleted': 0, 'inserted': 0, 'skipped': False}
    if not docs:
        stats['skipped'] = True
        if verbose:
            print(f'[writer] {coll.name}: 取到 0 行，跳过删除与写入')
        return stats

    stats['deleted'] = coll.delete_many({'code': code}).deleted_count
    for i in range(0, len(docs), batch):
        chunk = docs[i:i + batch]
        coll.insert_many(chunk, ordered=False)
        stats['inserted'] += len(chunk)
    if verbose:
        print(f'[writer] {coll.name}: {stats}')
    return stats
