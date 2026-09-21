# coding:utf-8
"""8.3 库存量数据的**维护操作**（清理类）。

目前只有一件：**把停牌日的 bar 移出主集合**（通达信伪 0 成交，见 ``PITFALLS.md`` P8b）。

**移动而不是删除** —— 落到 ``<集合名>_removed``，与所有者当年在 4.4 的做法一致
（那边的 ``stock_min_removed`` / ``index_min_removed`` 就是同一套约定）。
这样清理**可逆**，且 ``GQ_restore_suspended`` 提供回迁。

为什么是**移动**而不是读时过滤
============================
读时过滤有两条路，都不行：

1. **每次读多查一次日线** —— 直接伤读取速度（项目所有者的硬约束）。
2. **要求每个消费方都记得过滤** —— 必有一个会忘，而忘了的后果是**静默拿着
   伪 0 bar 去算**，正是 P8b 记的那类坑。

移走之后，**所有读取路径天生就是干净的**，零额外开销。这是「读取速度优先」
约束下的唯一解。

为什么用**日线 vol** 而不是外部停牌接口
=====================================
日线集合里**已经有这个信号**，不需要额外依赖：

- 实测在所有者当年人工挑出的真值集 ``stock_min_removed`` 上，均匀抽 200 组
  **200 组全部命中**；反方向的正常交易日不误判
- 外部接口有频率限制、且覆盖历史不如本地数据完整

⚠️ **判据只适用于股票侧**：``index_day`` 里 ``vol<1`` 有 **151 万**行，
那是指数的**结构性**形态（指数没有个股意义上的成交量），不是停牌 ——
对指数/ETF 用这条规则会**大规模误删**。故本模块只碰 ``stock_*``。

⚠️ **两个库的表示不同**：停牌日的日线 ``vol``，4.4 是 float 哨兵
``5.877471754e-39``、8.3 是 **int 0**（哨兵在迁移时被 cast 掉）。
判据统一写成 ``vol < 1`` —— 两边都覆盖；**等值比 0 在 4.4 会漏光**。

归档集合的形状
=============
``<集合名>_removed`` 建成**普通集合**（不是时间序列），并建 ``(code, ts)`` 唯一索引：

- 归档只需要能读能回迁，不需要时序的分桶剪枝
- 普通集合**支持唯一索引与 upsert**（时间序列两者都不支持），所以**重跑幂等** ——
  同一批 bar 移两次不会产生两份
"""

from __future__ import annotations

from pymongo import ReplaceOne

__all__ = [
    'SUSPENDED_TARGETS',
    'GQ_suspension_dates',
    'GQ_purge_suspended',
    'GQ_restore_suspended',
]

#: 要清理的集合（股票侧）。日线也在内 —— 见下。
SUSPENDED_TARGETS = (
    'stock_1min', 'stock_5min', 'stock_15min', 'stock_30min', 'stock_60min',
    'stock_day',
)

#: 归档集合的唯一键。**普通集合才能建唯一索引**，这是选普通集合而非时间序列的原因。
_ARCHIVE_KEYS = ('code', 'ts')


def _stock_cn_db():
    """8.3 库句柄。**函数级导入**（同 `kline83` / `datastruct` 的理由）。"""
    from . import DATABASE_STOCK_CN
    return DATABASE_STOCK_CN


def GQ_suspension_dates(daily_collection=None, verbose: bool = False) -> set:
    """``{(code, date)}`` —— 当日日线 ``vol < 1``，即**停牌日**。

    实测（2026-09-21，8.3）：**17,701 组 / 2,252 只 / 聚合耗时 2.0 秒**，
    且这 17,701 条里 **OHLC 完全持平的有 17,699 条（100%）** —— 与停牌特征一致。

    ⚠️ 这条判据要在**移除日线停牌 bar 之前**跑。移走之后 ``stock_day`` 里就
    没有标记了；但**重新迁移会把日线标记一并带回来**，所以重灌后重跑本函数
    仍然成立（这也是「重灌后必须重跑清理」的原因）。

    **不要**把这个判据用到 ``index_day`` 上（见模块 docstring）。

    :param daily_collection: 覆盖默认的 ``stock_day``（测试用）
    """
    coll = daily_collection if daily_collection is not None else _stock_cn_db()['stock_day']
    cur = coll.aggregate([
        {'$match': {'vol': {'$lt': 1}}},
        {'$group': {'_id': {'c': '$code', 'd': '$date'}}},
    ], allowDiskUse=True)
    out = {(r['_id']['c'], r['_id']['d']) for r in cur}
    if verbose:
        print(f'[maintenance] 停牌日 {len(out)} 组，涉及 {len({c for c, _ in out})} 只标的')
    return out


def _archive_name(name: str) -> str:
    return f'{name}_removed'


def _move(db, name: str, payload: dict, dry_run: bool, verbose: bool) -> dict:
    """把 `payload` 命中的文档从 ``name`` **移到** ``name_removed``。

    顺序是**先写归档、后删源** —— 中途崩了留下的是「归档里有、源里也有」，
    重跑用 upsert 覆盖，不会丢数据。反过来（先删）崩了就真丢了。
    """
    src = db[name]
    dst = db[_archive_name(name)]
    stats = {'found': 0, 'moved': 0, 'deleted': 0}
    if dry_run:
        stats['found'] = src.count_documents(payload)
        return stats

    docs = list(src.find(payload, {'_id': 0}))
    stats['found'] = len(docs)
    if not docs:
        return stats

    # 归档集合：普通集合 + (code, ts) 唯一索引 → 重跑幂等
    try:
        dst.create_index(list(_ARCHIVE_KEYS), unique=True)
    except Exception:                 # 索引已存在（或键不同）不应中断
        pass
    ops = [ReplaceOne({k: d[k] for k in _ARCHIVE_KEYS}, d, upsert=True)
           for d in docs if all(k in d for k in _ARCHIVE_KEYS)]
    if ops:
        dst.bulk_write(ops, ordered=False)
        stats['moved'] = len(ops)
    stats['deleted'] = src.delete_many(payload).deleted_count
    if verbose:
        n = dst.estimated_document_count()
        print(f'[maintenance] {name} → {_archive_name(name)}: '
              f'移 {stats["moved"]} 行（归档现有 {n} 行）')
    return stats


def GQ_purge_suspended(dry_run: bool = True, limit: int = None,
                       targets=None, verbose: bool = True) -> dict:
    """把停牌日的 bar **移出**主集合到 ``<集合名>_removed``。**默认只统计不移动。**

    覆盖日线在内（``stock_day``）—— 移走之后日线在停牌日是**缺失**的，
    与所有者的心智模型一致（他最初的判断就是「停牌当天日线缺失」，
    实际是通达信仍给了一根空日线）。

    按 **每个标的 × 每个集合一次** 操作（用 ``$in`` 覆盖该标的的全部停牌日），
    而不是「每个 (标的, 日期) 一次」—— 后者是前者的数倍往返。

    :param dry_run: True（默认）只统计；False 才真移
    :param limit: 只处理前 N 只标的（试跑用）
    :param targets: 覆盖要处理的集合名（测试用）
    :returns: ``{'suspension_days','targets','dry_run','stats': {集合: {...}}}``
    """
    db = _stock_cn_db()
    names = tuple(targets) if targets else SUSPENDED_TARGETS
    days = GQ_suspension_dates(verbose=verbose)
    if not days:
        if verbose:
            print('[maintenance] 没有停牌日可处理（日线里已无 vol<1 的标记）')
        return {'suspension_days': 0, 'targets': 0, 'dry_run': dry_run, 'stats': {}}

    by_code: dict = {}
    for code, day in sorted(days):
        by_code.setdefault(code, []).append(day)
    codes = sorted(by_code)
    if limit:
        codes = codes[:int(limit)]

    report = {'suspension_days': len(days), 'targets': len(codes),
              'dry_run': dry_run, 'stats': {n: {'found': 0, 'moved': 0, 'deleted': 0}
                                            for n in names}}
    if verbose:
        print(f'[maintenance] {"试跑（不移）" if dry_run else "★ 实际移动"} '
              f'{len(codes)} 只标的 / {len(days)} 个停牌日')
        print(f'[maintenance] 涉及集合: {list(names)}')

    existing = set(db.list_collection_names())
    for i, code in enumerate(codes, 1):
        payload = {'code': code, 'date': {'$in': by_code[code]}}
        for name in names:
            if name not in existing:
                continue
            st = _move(db, name, payload, dry_run, verbose=False)
            for k in st:
                report['stats'][name][k] += st[k]
        if verbose and (i % 200 == 0 or i == len(codes)):
            moved = sum(v['moved'] for v in report['stats'].values())
            found = sum(v['found'] for v in report['stats'].values())
            print(f'[maintenance] {i}/{len(codes)} 只，'
                  f'命中 {found} 行，{"将移" if dry_run else "已移"} {moved} 行…')

    if verbose:
        found = sum(v['found'] for v in report['stats'].values())
        moved = sum(v['moved'] for v in report['stats'].values())
        print(f'[maintenance] 完成：命中 {found} 行，{"将移" if dry_run else "已移"} {moved} 行'
              f'（明细 {report["stats"]}）')
    return report


def GQ_restore_suspended(targets=None, verbose: bool = True) -> dict:
    """**回迁**：把 ``<集合名>_removed`` 的内容搬回主集合。

    这是「移除可逆」的兑现。同样按 ``(code, ts)`` upsert，所以可反复跑。

    ⚠️ 回迁会把停牌日的日线标记**一并搬回**，于是
    :func:`GQ_suspension_dates` 又能重新识别出这批停牌日 —— 这正是
    「重灌/回迁之后可以再清理一次」的闭环。
    """
    db = _stock_cn_db()
    names = tuple(targets) if targets else SUSPENDED_TARGETS
    report = {}
    existing = set(db.list_collection_names())
    for name in names:
        arch = _archive_name(name)
        if arch not in existing:
            report[name] = 0
            continue
        docs = list(db[arch].find({}, {'_id': 0}))
        if not docs:
            report[name] = 0
            continue
        dst = db[name]
        ops = [ReplaceOne({k: d[k] for k in _ARCHIVE_KEYS}, d, upsert=True)
               for d in docs if all(k in d for k in _ARCHIVE_KEYS)]
        if ops:
            dst.bulk_write(ops, ordered=False)
        report[name] = len(ops)
        if verbose:
            print(f'[maintenance] {arch} → {name}: 回迁 {len(ops)} 行')
    return report
