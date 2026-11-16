# coding: utf-8
"""从 **4.4 旧库**搬运 `stage / garPD / garPAnchor / garPCount / garPClr`（用户 2026-11-16）。

为什么
======
新树自己重算 stage 的两条链（日线 coldstart / 60min `calc_stage_frame`）**都不对**
（用户 2026-11-16 判定「算法有重大隐患」）。权威数据在 4.4 旧树：

* 源：`golemq.stock_ta_reviews`（4.4，208 列，**每根 60min bar 一行**，到 2026-11-13）；
* 列：`STAGE` / `GAR_P`（**绝对价带**）/ `GAR_P_COUNT`；
* `bstStgBf / epStgBf` **不搬** —— 由 attach 用 `garPD → garP` 现算（用户口径）。

`garPD`（比值带）怎么来
====================
8.3 的 `GAR_P` 一律**存比值**（D49）。4.4 存的是绝对价带，这里换算：

    garPD = GAR_P / 锚日 **15:00 的 close（4.4 自己的价）**
    garPAnchor = 锚日（`YYYY-MM-DD`）

锚日 = `GAR_P` **发生变化**的那根 bar 的日期（带子只在 checkpoint 更新，之后 ffill）。
用 4.4 自己的 close 做分母 ⇒ 比值与复权无关；8.3 attach 用 **8.3 同日的 close**
乘回即可（`_restore_garident_bands` 已是这个口径）。

只更新**库里已存在**的 bar（按 `(code, time_stamp)`），不新增行。

用法
====
    python tools/migrate_stage_from44.py --codes 000711 --verify     # 对拍 stage
    python tools/migrate_stage_from44.py --codes 000711              # dry-run
    python tools/migrate_stage_from44.py --apply --workers 24
    python tools/migrate_stage_from44.py --verify                    # 全量对拍（只读）
"""
from __future__ import annotations

import argparse
import sys
import time

import pandas as pd

sys.path.insert(0, '.')

#: 4.4 源（`core.migrate44.db44` 是唯一入口；本工具只读 4.4）。
SRC_DB = 'golemq'
SRC_COLL = 'stock_ta_reviews'

#: 目标表（8.3）。
DST_COLL = 'stock_reality_feat_60min'
#: 日频表：每个交易日 15:00 的 stage 族写一条（用户 2026-11-16）
DST_DAY = 'stock_reality_feat_day'

_FIELDS = ('stage', 'garPD', 'garPAnchor', 'garPCount', 'garPClr')


def _db():
    from GolemQ.core.market_registry import get_active_market
    return get_active_market().get_database()


def _src():
    from GolemQ.core.migrate44 import db44
    return db44(SRC_DB)[SRC_COLL]


def all_codes(limit: int = 0) -> list:
    codes = {str(c) for c in _src().distinct('code')}
    # 只搬 8.3 里确实存在的码
    have = {str(c) for c in _db()[DST_COLL].distinct('code')}
    out = sorted(codes & have)
    return out[:limit] if limit else out


def _bands_key(v):
    """把 `GAR_P` 归一成可比较的 key（list of [u,l] → tuple of tuples）。"""
    if v is None:
        return None
    try:
        return tuple(tuple(float(x) for x in pair) for pair in v)
    except (TypeError, ValueError):
        return None


def _payload(rec) -> dict:
    """一条 4.4 记录 → 8.3 的 `$set` 载荷（60min 与 day 表**共用**）。"""
    payload = {}
    for c in _FIELDS:
        v = rec.get(c)
        if v is None:
            continue
        if c == 'garPAnchor':
            payload[c] = str(v)
        elif c == 'garPD':
            payload[c] = [[float(u), float(l)] for u, l in v]
        else:
            payload[c] = float(v)
    return payload


def migrate_rows_for(code: str):
    """单码 → `[{code,time_stamp,stage,garPD,garPAnchor,garPCount,garPClr}, ...]`。"""
    src = _src()
    rows = list(src.find({'code': str(code)},
                         {'_id': 0, 'datetime': 1, 'close': 1, 'STAGE': 1,
                          'GAR_P': 1, 'GAR_P_COUNT': 1}).sort('datetime', 1))
    if not rows:
        return []
    # 15:00 close（锚日分母）
    close15 = {}
    for r in rows:
        dt = str(r.get('datetime') or '')
        if dt[11:16] == '15:00' and r.get('close') is not None:
            close15[dt[:10]] = float(r['close'])

    out = []
    prev_key = None
    anchor_date = None
    anchor_close = None
    for r in rows:
        dt = str(r.get('datetime') or '')
        if not dt:
            continue
        key = _bands_key(r.get('GAR_P'))
        if key is not None and key != prev_key:
            prev_key = key
            anchor_date = dt[:10]
            anchor_close = close15.get(anchor_date)
            if anchor_close is None and r.get('close') is not None:
                anchor_close = float(r['close'])
        rec = {'code': str(code), 'datetime': dt,
               'stage': r.get('STAGE'),
               'garPCount': r.get('GAR_P_COUNT'),
               'garPAnchor': anchor_date}
        if key is not None and anchor_close:
            # 复用 `stages.garident_bands_to_ratio`（D49 的比值口径，别手写除法）
            from GolemQ.analysis.stages import garident_bands_to_ratio
            rec['garPD'] = garident_bands_to_ratio(
                [list(p) for p in key], float(anchor_close))
        st, cnt = r.get('STAGE'), r.get('GAR_P_COUNT')
        if st is not None and cnt is not None:
            rec['garPClr'] = float(cnt) - float(st)
        out.append(rec)
    return out


def migrate_code(code: str, apply: bool = False):
    try:
        recs = migrate_rows_for(code)
        if not recs:
            return {'code': code, 'status': 'skipped'}
        dst = _db()[DST_COLL]
        # ⚠️ **按 `(code, datetime)` 更新，不按 `time_stamp`**：8.3 里有一批码的
        # `time_stamp` 被截断（如 000009：1711436 vs 1711436400）且有 1,000+ 重复
        # datetime，用 time_stamp 做键只会命中其中一份，其余旧值留在库里。
        from pymongo import UpdateMany
        have = {str(d.get('datetime')) for d in dst.find(
            {'code': str(code)}, {'_id': 0, 'datetime': 1})}
        # 连续相同 payload 的 bar **合并成一条 `$in`**：stage/garP 只在 checkpoint
        # 变化（单码约 50 次），逐行 UpdateMany 会把 3,000 根写成 3,000 次批量，
        # 单码要 ~1 分钟；合并后降到几十次。
        groups = []          # [(payload_key, payload, [datetime, ...])]
        for rec in recs:
            if rec['datetime'] not in have:
                continue
            payload = _payload(rec)
            if not payload:
                continue
            key = repr(sorted(payload.items()))
            if groups and groups[-1][0] == key:
                groups[-1][2].append(rec['datetime'])
            else:
                groups.append((key, payload, [rec['datetime']]))
        ops, n = [], 0
        for _key, payload, dts in groups:
            ops.append(UpdateMany({'code': str(code),
                                   'datetime': {'$in': dts}}, {'$set': payload}))
            n += len(dts)
        st = {'code': code, 'status': 'ok', 'rows': n}
        if apply and ops:
            res_m = 0
            for i in range(0, len(ops), 500):
                res = dst.bulk_write(ops[i:i + 500], ordered=False)
                res_m += res.modified_count
            st.update(modified=res_m)
        # ---- 日频表：每个交易日取 15:00 那根的 stage 族 ------------------------ #
        # 与 60min 同一份搬运数据（不重算）；按 `(code, date)` 覆盖全部重复行。
        day_coll = _db()[DST_DAY]
        day_have = {str(d.get('date')) for d in day_coll.find(
            {'code': str(code)}, {'_id': 0, 'date': 1})}
        groups_d = []       # 连续相同 payload 的日期合并成 $in（同 60min 的理由）
        for rec in recs:
            if rec['datetime'][11:16] != '15:00':
                continue
            date = rec['datetime'][:10]
            if date not in day_have:
                continue
            payload = _payload(rec)
            if not payload:
                continue
            key = repr(sorted(payload.items()))
            if groups_d and groups_d[-1][0] == key:
                groups_d[-1][2].append(date)
            else:
                groups_d.append((key, payload, [date]))
        day_ops = [UpdateMany({'code': str(code), 'date': {'$in': dts}},
                              {'$set': payload})
                   for _k, payload, dts in groups_d]
        day_n = sum(len(dts) for _k, _p, dts in groups_d)
        st['day_rows'] = day_n
        if apply and day_ops:
            for i in range(0, len(day_ops), 500):
                day_coll.bulk_write(day_ops[i:i + 500], ordered=False)
        return st
    except Exception as exc:            # noqa: BLE001
        return {'code': code, 'status': 'error', 'error': repr(exc)}


def verify_code(code: str):
    """对拍：4.4 的 `STAGE` vs 8.3 的 `stage`（按 datetime 对齐）。"""
    src = _src()
    src_map = {str(r['datetime']): r.get('STAGE')
               for r in src.find({'code': str(code)}, {'_id': 0, 'datetime': 1,
                                                        'STAGE': 1})}
    dst = _db()[DST_COLL]
    ok = bad = missing = 0
    for d in dst.find({'code': str(code)}, {'_id': 0, 'datetime': 1, 'stage': 1}):
        dt = str(d.get('datetime'))
        s = src_map.get(dt, None)
        v = d.get('stage')
        if s is None or v is None or pd.isna(v):
            missing += 1
            continue
        if float(s) == float(v):
            ok += 1
        else:
            bad += 1
    return {'code': code, 'ok': ok, 'mismatch': bad, 'missing': missing}


def main(argv=None):
    from tqdm import tqdm
    ap = argparse.ArgumentParser(description='从 4.4 搬 stage/garP*（只读 4.4）')
    ap.add_argument('--codes', default='')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--shard', default='')
    ap.add_argument('--workers', type=int, default=1)
    ap.add_argument('--apply', action='store_true', help='真写（默认 dry-run）')
    ap.add_argument('--verify', action='store_true', help='只对拍 STAGE（不写）')
    args = ap.parse_args(argv)

    codes = ([c.strip() for c in args.codes.split(',') if c.strip()]
             or all_codes(limit=args.limit))
    if args.shard:
        from GolemQ.pipeline.review.pipeline import shard_codes
        codes = shard_codes(codes, args.shard)
    print(f'[stage44] {len(codes)} 个码 verify={args.verify} apply={args.apply}',
          flush=True)

    t0 = time.time()
    if args.verify:
        if args.workers > 1 and len(codes) > 1:
            from joblib import Parallel, delayed
            results = Parallel(n_jobs=args.workers, backend='loky')(
                delayed(verify_code)(c) for c in codes)
        else:
            results = [verify_code(c) for c in tqdm(codes, unit='code')]
        ok = sum(r['ok'] for r in results)
        bad = sum(r['mismatch'] for r in results)
        mis = sum(r['missing'] for r in results)
        print(f'[stage44] verify: match={ok} mismatch={bad} missing={mis} '
              f'{time.time()-t0:.0f}s', flush=True)
        for r in results:
            if r['mismatch']:
                print('  MISMATCH', r, flush=True)
                break
        return 0 if bad == 0 else 1

    if args.workers > 1 and len(codes) > 1:
        from joblib import Parallel, delayed
        results = Parallel(n_jobs=args.workers, backend='loky')(
            delayed(migrate_code)(c, args.apply) for c in codes)
    else:
        results = [migrate_code(c, args.apply) for c in tqdm(codes, unit='code')]
    ok = [r for r in results if r.get('status') == 'ok']
    err = [r for r in results if r.get('status') == 'error']
    skip = [r for r in results if r.get('status') == 'skipped']
    print(f'[stage44] ok={len(ok)} skipped={len(skip)} error={len(err)} '
          f'rows={sum(r.get("rows", 0) for r in ok)} {time.time()-t0:.0f}s',
          flush=True)
    for r in err[:5]:
        print('  ERROR', r, flush=True)
    return 0 if not err else 1


if __name__ == '__main__':
    sys.exit(main())
