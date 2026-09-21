# coding:utf-8
"""8.3 库存量数据的**维护操作**（清理类）。

目前只有一件：**剔除停牌日的分钟 bar**（通达信伪 0 成交，见 ``PITFALLS.md`` P8b）。

为什么是**物理清理**而不是读时过滤
================================
读时过滤有两条路，都不行：

1. **每次读多查一次日线** —— 直接伤读取速度（项目所有者的硬约束）。
2. **要求每个消费方都记得过滤** —— 必有一个会忘，而忘了的后果是**静默拿着
   伪 0 bar 去算**，正是 P8b 记的那类坑。

清掉之后，**所有读取路径天生就是干净的**，零额外开销。这是「读取速度优先」
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
"""

from __future__ import annotations

__all__ = [
    'SUSPENDED_MINUTE_COLLECTIONS',
    'GQ_suspension_dates',
    'GQ_purge_suspended_minutes',
]

#: 需要清理的分钟集合（股票侧）。
SUSPENDED_MINUTE_COLLECTIONS = (
    'stock_1min', 'stock_5min', 'stock_15min', 'stock_30min', 'stock_60min',
)


def _stock_cn_db():
    """8.3 库句柄。**函数级导入**（同 `kline83` / `datastruct` 的理由）。"""
    from . import DATABASE_STOCK_CN
    return DATABASE_STOCK_CN


def GQ_suspension_dates(daily_collection=None, verbose: bool = False) -> set:
    """``{(code, date)}`` —— 当日日线 ``vol < 1``，即**停牌日**。

    实测（2026-09-21，8.3）：**17,701 组 / 2,252 只 / 聚合耗时 2.0 秒**，
    且这 17,701 条里 **OHLC 完全持平的有 17,699 条（100%）** —— 与停牌特征一致。

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
        codes = {c for c, _ in out}
        print(f'[maintenance] 停牌日 {len(out)} 组，涉及 {len(codes)} 只标的')
    return out


def GQ_purge_suspended_minutes(dry_run: bool = True, limit: int = None,
                               collections=None, verbose: bool = True) -> dict:
    """把停牌日的分钟 bar 从 8.3 删掉。**默认 ``dry_run=True``，只统计不删。**

    删除按 **每个标的 × 每个集合一次** ``delete_many``（用 ``$in`` 覆盖该标的的
    全部停牌日），而不是「每个 (标的, 日期) 一次」—— 后者是前者的数倍往返。
    实测 ``delete_many`` 在**时间序列集合上可用**（唯一索引与 upsert 不行，
    删除可以）。

    ⚠️ **这是不可逆操作**：删掉的 bar 只能靠**重新迁移**恢复。因此判据必须
    可复现 —— 它就在 :func:`GQ_suspension_dates` 里，任何人可随时重算重跑。
    将来的数据重灌之后，**这个清理要再跑一次**。

    :param dry_run: True（默认）只统计；False 才真删
    :param limit: 只处理前 N 只标的（试跑用）
    :param collections: 覆盖要清理的集合名（测试用）
    :returns: ``{'suspension_days': n, 'targets': n, 'deleted': {集合: 行数}}``
    """
    db = _stock_cn_db()
    names = tuple(collections) if collections else SUSPENDED_MINUTE_COLLECTIONS
    days = GQ_suspension_dates(verbose=verbose)

    by_code: dict = {}
    for code, day in sorted(days):
        by_code.setdefault(code, []).append(day)
    codes = sorted(by_code)
    if limit:
        codes = codes[:int(limit)]

    report = {'suspension_days': len(days), 'targets': len(codes),
              'dry_run': dry_run, 'deleted': {n: 0 for n in names}}

    if verbose:
        print(f'[maintenance] {"试跑（不删）" if dry_run else "★ 实际删除"} '
              f'{len(codes)} 只标的 / {len(days)} 个停牌日')
        print(f'[maintenance] 涉及集合: {list(names)}')

    for i, code in enumerate(codes, 1):
        payload = {'code': code, 'date': {'$in': by_code[code]}}
        for name in names:
            if name not in db.list_collection_names():
                continue
            coll = db[name]
            if dry_run:
                report['deleted'][name] += coll.count_documents(payload)
            else:
                report['deleted'][name] += coll.delete_many(payload).deleted_count
        if verbose and (i % 200 == 0 or i == len(codes)):
            total = sum(report['deleted'].values())
            print(f'[maintenance] {i}/{len(codes)} 只，'
                  f'{"将删" if dry_run else "已删"} {total} 行…')

    if verbose:
        total = sum(report['deleted'].values())
        print(f'[maintenance] 完成：{"将删" if dry_run else "已删"} 合计 {total} 行'
              f'（明细 {report["deleted"]}）')
    return report
