# coding: utf-8
"""把 60min 的 **10 个比值列**改成 as-of 收盘口径（D82 / P51，用户 2026-11-16）。

背景
====
60min 行的分子取自「≤T 的最后一个 15:00 快照」；比值列的分母必须是**同一个快照日**
的 close（盘中 = 上一交易日），否则同一份快照被 4 根 bar 各除一次。
存量里只有两组列是按 bar close 除的：

* ChipDist 4 列：`cost5PrzD` / `cost15PrzD` / `cost85PrzD` / `cost95PrzD`
* Renko 价格 6 列：`renkoSpriceD` / `renkoLpriceD` / `renkosBarLbD` /
  `renkosBarUbD` / `renkolBarLbD` / `renkolBarUbD`

（其余 28 个比值列本来就已是 as-of，实测盘中 3 根 0/18 天不同 —— 不动。）

为什么**不用重算**
==================
15:00 行存的就是「当日分子 ÷ 当日 close」，正好是要的值；盘中 3 根应该**等于上一
交易日 15:00 行**的同一组值。所以重建 = **把上一交易日 15:00 行的 10 列复制到其后
各盘中行**。幂等（重跑结果不变），不碰 15:00 行，也不需要 ChipDist（0.43s/天）与
Renko 重算 —— 全库几分钟量级。

用法
====
    python tools/fix_asof_ratios.py                    # dry-run（只报数）
    python tools/fix_asof_ratios.py --apply            # 真写
    python tools/fix_asof_ratios.py --codes 000711 --apply
    python tools/fix_asof_ratios.py --workers 8 --shard 1/3
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import time as dtime

import numpy as np
import pandas as pd

sys.path.insert(0, '.')

from GolemQ.pipeline.review.contract import (HOUR_CHIP_COLUMNS,
                                             RENKO_DAY_COLUMNS,
                                             RENKO_PRICE_DUMMIES)

#: 需要重建的列（顺序固定，便于分组比较）
_TARGET_COLUMNS = tuple(HOUR_CHIP_COLUMNS) + tuple(
    d for _b, d in RENKO_PRICE_DUMMIES)
_TABLES = ('stock_reality_feat_60min', 'etf_reality_feat_60min',
           'index_reality_feat_60min')


def _db():
    from GolemQ.core.market_registry import get_active_market
    return get_active_market().get_database()


def _clean(v):
    """NaN → None（与落库口径一致；也用于分组键，NaN 不可比较）。"""
    if v is None:
        return None
    try:
        if isinstance(v, float) and np.isnan(v):
            return None
    except Exception:               # noqa: BLE001
        pass
    return v


def _same(a, b) -> bool:
    if _clean(a) is None and _clean(b) is None:
        return True
    try:
        return float(a) == float(b)
    except (TypeError, ValueError):
        return a == b


def fix_code(coll_name: str, code: str, apply: bool) -> dict:
    """把一个码的盘中行改成「上一交易日 15:00 行」的同一组比值。"""
    coll = _db()[coll_name]
    proj = {'_id': 1, 'datetime': 1, **{c: 1 for c in _TARGET_COLUMNS}}
    docs = list(coll.find({'code': str(code)}, proj))
    if not docs:
        return {'code': code, 'rows': 0, 'changed': 0}
    rows = []
    for d in docs:
        try:
            rows.append((pd.Timestamp(d['datetime']), d))
        except (KeyError, ValueError):
            continue
    rows.sort(key=lambda x: x[0])
    closes = [(t, d) for t, d in rows if t.time() == dtime(15, 0)]
    if not closes:
        return {'code': code, 'rows': len(rows), 'changed': 0}
    close_ts = [t for t, _d in closes]

    # 按 payload 分组 → 一条 update_many 写一批（同值行通常很多）
    groups = {}
    n_changed = 0
    for t, d in rows:
        if t.time() == dtime(15, 0) or t.time() == dtime(0, 0):
            continue                # 15:00 行本来就是对的；日频行不存在
        pos = int(np.searchsorted(np.array(close_ts, dtype='datetime64[ns]'),
                                  np.datetime64(t), side='left')) - 1
        if pos < 0:
            continue                # 没有更早的 15:00（历史首日）⇒ 不动
        src = closes[pos][1]
        payload = {c: _clean(src.get(c)) for c in _TARGET_COLUMNS}
        if not any(v is not None for v in payload.values()):
            continue                # 源行这 10 列全空 ⇒ 无从复制
        if all(_same(d.get(c), payload[c]) for c in _TARGET_COLUMNS):
            continue                # 已是目标值（幂等）
        key = tuple(payload[c] for c in _TARGET_COLUMNS)
        groups.setdefault(key, []).append(d['datetime'])
        n_changed += 1

    if apply:
        for key, dts in groups.items():
            payload = {c: v for c, v in zip(_TARGET_COLUMNS, key)}
            coll.update_many({'code': str(code), 'datetime': {'$in': dts}},
                             {'$set': payload})
    return {'code': code, 'rows': len(rows), 'changed': n_changed}


def main(argv=None) -> int:
    from tqdm import tqdm
    ap = argparse.ArgumentParser(description='60min 比值列改 as-of 收盘（D82）')
    ap.add_argument('--apply', action='store_true', help='真写（默认 dry-run）')
    ap.add_argument('--codes', default='')
    ap.add_argument('--codes-file', default='')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--workers', type=int, default=1)
    ap.add_argument('--shard', default='',
                    help='分布式分片 "i/N"（与 --review-shard 同一实现）')
    a = ap.parse_args(argv)

    db = _db()
    if a.codes_file:
        with open(a.codes_file, encoding='utf-8') as fh:
            codes = [ln.strip() for ln in fh if ln.strip()]
    else:
        codes = [c.strip() for c in a.codes.split(',') if c.strip()]
    if not codes:
        found = set()
        for name in _TABLES:
            if name in db.list_collection_names():
                found.update(str(c) for c in db[name].distinct('code'))
        codes = sorted(found)
    if a.limit:
        codes = codes[:a.limit]
    if a.shard:
        from GolemQ.pipeline.review.pipeline import shard_codes
        try:
            codes = shard_codes(codes, a.shard)
        except ValueError as exc:
            raise SystemExit(str(exc))

    print(f'[asof] {len(codes)} 个码 apply={a.apply} '
          f'columns={" ".join(_TARGET_COLUMNS)}', flush=True)
    t0 = time.time()
    results = []
    if a.workers > 1 and len(codes) > 1:
        from joblib import Parallel, delayed
        results = Parallel(n_jobs=a.workers, backend='loky')(
            delayed(fix_code)(name, c, a.apply)
            for name in _TABLES for c in codes)
    else:
        for name in _TABLES:
            for c in tqdm(codes, desc=name, unit='code'):
                results.append(fix_code(name, c, a.apply))
    rows = sum(r['rows'] for r in results)
    changed = sum(r['changed'] for r in results)
    print(f'[asof] rows={rows} changed={changed} {time.time() - t0:.0f}s')
    return 0


if __name__ == '__main__':
    sys.exit(main())
