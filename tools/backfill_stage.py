# coding: utf-8
"""把 `stock/etf/index_reality_feat_60min.stage` 用**修正后的 vXII 口径**重算。

背景（D79 修正）
================
旧实现把「先 ffill 带子、再逐 bar 判价」写反了（`analysis/stages.py` 死文件
如此；真正被调用的 `fractal/vXII.py` 是反过来的）。症状：非 checkpoint 行
判价时带子是 NaN ⇒ stage 只剩 checkpoint 的整数值，`-0.5` / 半档永远不出现，
`STAGE_MODE` 因此无法复现 4.4。

本工具：
* 逐码取 60min（2016 起）→ `calc_stage_series`（全帧、`iloc[::21]` checkpoint）；
* 取每个交易日 **15:00** 的 stage，写到当日 4 根 bar 的 `stage` 列；
* 只更新**库里已存在**的 bar（不新增行）。

用法
====
    python tools/backfill_stage.py                 # dry-run（默认）
    python tools/backfill_stage.py --apply --workers 24
    python tools/backfill_stage.py --apply --shard 1/3
"""
from __future__ import annotations

import argparse
import sys
import time

import pandas as pd

sys.path.insert(0, '.')

ANCHOR = '2016-01-01'
_TABLES = ('stock_reality_feat_60min', 'etf_reality_feat_60min',
           'index_reality_feat_60min')


def _db():
    from GolemQ.core.market_registry import get_active_market
    return get_active_market().get_database()


def all_codes(limit: int = 0) -> list:
    from GolemQ.pipeline.review.contract import HOUR_ZEN_COLUMNS  # noqa: F401
    db = _db()
    codes = set()
    for t in _TABLES:
        if t in db.list_collection_names():
            codes |= {str(c) for c in db[t].distinct(
                'code', {'zDLagR': {'$ne': None}})}
    out = sorted(codes)
    return out[:limit] if limit else out


def _market_type(code: str):
    from GolemQ.markets.StockCN.symbol import is_stock_cn
    return is_stock_cn(str(code))[1]


def _table(code: str) -> str:
    from GolemQ.core.constants import MARKET_TYPE
    mt = _market_type(code)
    if mt == MARKET_TYPE.STOCK_CN:
        return _TABLES[0]
    if mt == MARKET_TYPE.ETF_CN:
        return _TABLES[1]
    return _TABLES[2]


def stage_rows_for(code: str):
    """单码 → `[(time_stamp, stage), ...]`（当日 4 根共用 15:00 的 stage）。"""
    from GolemQ import get_active_market
    from GolemQ.markets.StockCN.date_utils import GQ_util_last_trade_as_of
    from GolemQ.pipeline.review.coldstart import calc_stage_series

    m = get_active_market()
    res, _ = m.get_kline_price_min(code, start=ANCHOR,
                                   end=str(GQ_util_last_trade_as_of().date()) + ' 23:59:59',
                                   frequency='60min', realtime=False)
    f60 = None if res is None else getattr(res, 'data', None)
    if f60 is None or len(f60) < 300:
        return []
    st = calc_stage_series(f60)
    ts = pd.DatetimeIndex(pd.to_datetime(f60.index.get_level_values(0)))
    is15 = (ts.time == pd.Timestamp('15:00').time())
    if not is15.any():
        return []
    day_stage = pd.Series(st.to_numpy()[is15], index=ts[is15].normalize())
    day_stage = day_stage[~day_stage.index.duplicated(keep='last')]

    col = _db()[_table(code)]
    docs = list(col.find({'code': str(code)}, {'datetime': 1, 'time_stamp': 1,
                                               '_id': 0}))
    out = []
    for d in docs:
        t = pd.Timestamp(d['datetime'])
        v = day_stage.get(t.normalize())
        if v is None or not pd.notna(v):
            continue
        out.append({'code': str(code), 'time_stamp': d['time_stamp'],
                    'stage': float(v)})
    return out


def backfill_code(code, apply=False):
    try:
        rows = stage_rows_for(code)
        if not rows:
            return {'code': code, 'status': 'skipped'}
        stats = {'code': code, 'status': 'ok', 'rows': len(rows),
                 'halves': sum(1 for r in rows if r['stage'] % 1 != 0)}
        if apply:
            from GolemQ.datasource.writer import upsert_fields
            out = upsert_fields(_db()[_table(code)], rows,
                                ['code', 'time_stamp'])
            stats.update(upserted=out['upserted'], modified=out['modified'])
        return stats
    except Exception as exc:            # noqa: BLE001
        return {'code': code, 'status': 'error', 'error': repr(exc)}


def main(argv=None):
    ap = argparse.ArgumentParser(description='重算 reality_feat_60min.stage')
    ap.add_argument('--codes', default='')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--shard', default='', help='分布式分片 "i/N"')
    ap.add_argument('--workers', type=int, default=1)
    ap.add_argument('--apply', action='store_true', help='真写（默认 dry-run）')
    ap.add_argument('--verbose', action='store_true')
    args = ap.parse_args(argv)

    codes = ([c.strip() for c in args.codes.split(',') if c.strip()]
             or all_codes(limit=args.limit))
    if args.shard:
        from GolemQ.pipeline.review.pipeline import shard_codes
        codes = shard_codes(codes, args.shard)
    print(f'[stage] {len(codes)} 个码，apply={args.apply}', flush=True)

    t0 = time.time()
    if args.workers > 1 and len(codes) > 1:
        from joblib import Parallel, delayed
        results = Parallel(n_jobs=args.workers, backend='loky')(
            delayed(backfill_code)(c, args.apply) for c in codes)
    else:
        results = []
        for i, c in enumerate(codes, 1):
            results.append(backfill_code(c, args.apply))
            if i % 100 == 0:
                print(f'  … {i}/{len(codes)} ({time.time()-t0:.0f}s)', flush=True)

    ok = [r for r in results if r.get('status') == 'ok']
    err = [r for r in results if r.get('status') == 'error']
    skip = [r for r in results if r.get('status') == 'skipped']
    print(f'[stage] ok={len(ok)} skipped={len(skip)} error={len(err)} '
          f'rows={sum(r.get("rows", 0) for r in ok)} '
          f'halves={sum(r.get("halves", 0) for r in ok)} '
          f'{time.time()-t0:.0f}s', flush=True)
    for r in err[:5]:
        print('  ERROR', r, flush=True)
    return 0 if not err else 1


if __name__ == '__main__':
    sys.exit(main())
