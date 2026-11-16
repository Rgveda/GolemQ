# coding: utf-8
"""修复 reality 表的**截断时间戳**并**合并重复行**（用户 2026-11-16）。

背景（`PITFALLS.md` P49）
========================
8.3 的 reality 表里有一批行的时间戳被**除以 1000**：

    datetime 2024-03-21 14:00:00 → 正确 epoch 1711000800，库里 time_stamp = 1711000

`fetch_reality_feat` 按 `time_stamp` 范围过滤 ⇒ 这些行**永远读不出来**
（实测 60min 721,759 行 / day 556,818 行，抽样多数没有正确孪生）。

⚠️ 表上有**唯一索引** `code_1_time_stamp_1`（day 表 `code_1_date_stamp_1`）——
直接把截断值改回正确 epoch 会和「正确孪生行」撞键（`E11000`）。
所以必须**逐码**做「归一 → 合并 → 删除 → 改键」，顺序见 :func:`fix_code`：

1. 按**归一后的键**（截断值用 `datetime`/`date` 重算）给同码所有文档分组；
2. 每组保留**非空字段最多**的那份，把其余各份的非空字段**并进去**；
3. 先删其余副本，再把保留份的键改成归一值（删完才改，避免撞唯一索引）。

用法
====
    python tools/fix_reality_rows.py --apply                 # 两步一起跑
    python tools/fix_reality_rows.py                        # dry-run（只报数）
    python tools/fix_reality_rows.py --codes 000009 --apply
    python tools/fix_reality_rows.py --step2 --apply --workers 24   # 只做去重
"""
from __future__ import annotations

import argparse
import sys
import time

import pandas as pd

sys.path.insert(0, '.')

#: (集合, 时间键, 用于重算的字符串字段)
_TABLES = (
    ('stock_reality_feat_60min', 'time_stamp', 'datetime'),
    ('etf_reality_feat_60min', 'time_stamp', 'datetime'),
    ('index_reality_feat_60min', 'time_stamp', 'datetime'),
    ('stock_reality_feat_day', 'date_stamp', 'date'),
    ('etf_reality_feat_day', 'date_stamp', 'date'),
    ('index_reality_feat_day', 'date_stamp', 'date'),
)

#: 归一化阈值：只拦「被 /1000 的 2016+ 值」（≈1.4e6~1.8e6）。
#: ⚠️ **不能用 1e9** —— day 表里 2001 年以前的**合法** date_stamp 也在 9.7e8，
#: 会被误当损坏（实测 556,818 行里绝大多数是 2001 年的正常行）。
_BAD = 1e8


def _db():
    from GolemQ.core.market_registry import get_active_market
    return get_active_market().get_database()


def _norm_ts(key: str, src: str, s: str):
    """`datetime`/`date` 字符串（**北京时间**）→ 秒级 epoch（与 8.3 口径一致）。

    解析不了返回 None（**跳过该行**，不做改动）—— 表里有极少数畸形 `date`
    （如 `'-28800000 00:00:00'`），不能让一行把整码任务打崩。
    """
    try:
        if key == 'date_stamp':
            from GolemQ.markets.StockCN.metadata_save import day_stamp
            return int(day_stamp(str(s)[:10]))
        return int(pd.Timestamp(str(s), tz='Asia/Shanghai').timestamp())
    except Exception:               # noqa: BLE001 单行畸形日期跳过
        return None


def _nn(v) -> bool:
    return v not in (None, '', [], {})


def _merge(docs):
    """多份同 bar 文档 → (保留 `_id`, 合并字段)；保留非空最多的那份。"""
    docs = sorted(docs, key=lambda d: sum(_nn(v) for k, v in d.items() if k != '_id'),
                  reverse=True)
    keep = docs[0]
    merged = {k: v for k, v in keep.items() if k != '_id'}
    for d in docs[1:]:
        for k, v in d.items():
            if k == '_id':
                continue
            if not _nn(merged.get(k)) and _nn(v):
                merged[k] = v
    stamps = [d.get('created_at') for d in docs if d.get('created_at')]
    if stamps:
        merged['created_at'] = max(stamps)
    return keep['_id'], merged, len(docs)


def fix_code(coll_name: str, key: str, src: str, code: str, apply: bool = False):
    """单码：归一时间戳 + 合并同名 bar 的重复行（撞唯一索引也安全）。"""
    from pymongo import DeleteMany, UpdateOne
    coll = _db()[coll_name]
    docs = list(coll.find({'code': str(code)}))
    if not docs:
        return {'code': code, 'groups': 0, 'deleted': 0, 'rekeyed': 0}

    groups = {}
    for d in docs:
        raw = d.get(key)
        good = isinstance(raw, (int, float)) and raw >= _BAD
        k = int(raw) if good else (_norm_ts(key, src, d.get(src)) if d.get(src)
                                   else None)
        if k is None:
            continue
        groups.setdefault(k, []).append(d)

    ops, n_del, n_rekey = [], 0, 0
    for k, ds in groups.items():
        if len(ds) == 1 and ds[0].get(key) == k:
            continue
        keep_id, merged, _n = _merge(ds)
        fields = {c: v for c, v in merged.items() if c != key}
        if fields:
            ops.append(UpdateOne({'_id': keep_id}, {'$set': fields}))
        others = [d['_id'] for d in ds if d['_id'] != keep_id]
        if others:
            ops.append(DeleteMany({'_id': {'$in': others}}))
            n_del += len(others)
        if merged.get(key) != k:
            # ⚠️ 改键**必须排在删除之后**（同批 ordered=True），否则会撞唯一索引
            ops.append(UpdateOne({'_id': keep_id}, {'$set': {key: k}}))
            n_rekey += 1
    if apply and ops:
        coll.bulk_write(ops, ordered=True)
    return {'code': code, 'groups': sum(1 for k, ds in groups.items()
                                        if len(ds) > 1 or ds[0].get(key) != k),
            'deleted': n_del, 'rekeyed': n_rekey}


def main(argv=None):
    from tqdm import tqdm
    ap = argparse.ArgumentParser(description='修复 reality 时间戳 + 合并重复行')
    ap.add_argument('--apply', action='store_true')
    ap.add_argument('--codes', default='')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--workers', type=int, default=1)
    ap.add_argument('--only', default='', help='只跑这些集合（逗号分隔）')
    a = ap.parse_args(argv)

    only = {x.strip() for x in a.only.split(',') if x.strip()}
    t0 = time.time()
    for coll_name, key, src in _TABLES:
        db = _db()
        if coll_name not in db.list_collection_names():
            continue
        if only and coll_name not in only:
            continue
        codes = ([c.strip() for c in a.codes.split(',') if c.strip()]
                 or sorted({str(c) for c in db[coll_name].distinct('code')}))
        if a.limit:
            codes = codes[:a.limit]
        print(f'[fix] {coll_name}: {len(codes)} 个码 apply={a.apply}', flush=True)
        if a.workers > 1 and len(codes) > 1:
            from joblib import Parallel, delayed
            res = Parallel(n_jobs=a.workers, backend='loky')(
                delayed(fix_code)(coll_name, key, src, c, a.apply) for c in codes)
        else:
            res = [fix_code(coll_name, key, src, c, a.apply)
                   for c in tqdm(codes, unit='code')]
        print(f'[fix] {coll_name}: fixed_groups={sum(r["groups"] for r in res)} '
              f'deleted={sum(r["deleted"] for r in res)} '
              f'rekeyed={sum(r["rekeyed"] for r in res)} '
              f'{time.time()-t0:.0f}s', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
