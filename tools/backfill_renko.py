#!/usr/bin/env python3
# coding: utf-8
"""给已有 review **日线行**补 Renko 全族（14 列）。

为什么需要
==========
review 的 `DAY_COLUMNS` 2026-10-27 才加入 Renko；在此之前写过的日行没有这些列。
正常 `--review` 按「任一契约列非空即已持久化」跳过旧行，不会补；
本工具只算 Renko 族，不碰 coldstart/Prophet/Stage。

口径
====
**逐目标日 as-of**：每个日行 T 只喂 `≤T` 的**全部**日线（取数锚 2016）跑
`renko_trend_cross_func_jit`，再取 T 那行 —— **不许**全帧跑一次再按 T 取值
（有后视镜扰动，实测 600519 2024-08-16 `renkoLbarLag` 全量 -18 / as-of -50）。
取 `TARGET_LOOKBACK_DAYS['day']` 窗口内的日行；**不重算其它列**。
价格量纲的 6 列（`renkoSprice/renkoLprice/renkosBarLb/Ub/renkolBarLb/Ub`）
按 D70 落库为 **÷close 的比值**（`*D` 后缀），与 review 写入口径一致。
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import pandas as pd

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from GolemQ import get_active_market
from GolemQ.analysis.renko_jit import (available,
                                       renko_trend_cross_func_jit)
from GolemQ.core.constants import AKA
from GolemQ.markets.StockCN.reality_save import (reality_feat_name,
                                                 save_reality_feat)
from GolemQ.markets.StockCN.symbol import is_stock_cn
from GolemQ.pipeline.review.contract import (RENKO_DAY_COLUMNS,
                                             RENKO_PRICE_DUMMIES)
from GolemQ.pipeline.review.pipeline import (REVIEW_ANCHOR_DAY,
                                             TARGET_LOOKBACK_DAYS)

_COLS = RENKO_DAY_COLUMNS
#: 判「这行还没补过 Renko」用的标记列（第一个价格列的比值版）。
_MARKER = RENKO_PRICE_DUMMIES[0][1]


def partial_codes(db):
    coll = db[reality_feat_name('day', 'stock_cn')]
    # 只挑**成块缺失**的码（>=10 行）；短历史导致的零星 NaN（<10 行）不反复补
    rows = coll.aggregate([
        # `$exists` 不够：metadata-only 行也有这个键、值为 None
        {'$match': {'zDLagMajR': {'$exists': True, '$ne': None}}},
        {'$group': {'_id': '$code', 'review': {'$sum': 1},
                    # ⚠️ 字段引用必须带 `$`：漏了就是字面量，恒非空 ⇒ 永远「不缺」
                    'renko': {'$sum': {'$cond': [
                        {'$ne': [{'$ifNull': ['$' + _MARKER, None]}, None]},
                        1, 0]}}}},
        {'$addFields': {'missing': {'$subtract': ['$review', '$renko']}}},
        {'$match': {'missing': {'$gte': 10}}},
        {'$sort': {'_id': 1}},
    ], allowDiskUse=True)
    return [r['_id'] for r in rows]


def calc_one(market, code, as_of):
    res, _ = market.get_kline_price_v3(
        code, start=REVIEW_ANCHOR_DAY, end=str(as_of.date()),
        realtime=False, verbose=False)
    day = None if res is None else getattr(res, 'data', None)
    if day is None or len(day) == 0:
        return None
    ts = pd.DatetimeIndex(pd.to_datetime(day.index.get_level_values(0)))
    close = pd.to_numeric(day[AKA.CLOSE], errors='coerce')
    # review 的日行最早到 2019-10-18，比 `as_of - 2560 天` 的边界略早；
    # 多留 120 天缓冲，避免边界上每码 1~5 行永远补不到。
    floor = as_of - pd.Timedelta(days=TARGET_LOOKBACK_DAYS['day'] + 120)
    targets = ts[(ts >= floor) & (ts <= as_of)]

    rows, idx = {}, []
    for T in targets:
        w = day.loc[ts <= T]                     # as-of：≤T 的全部历史（锚点 2016）
        if len(w) < 30:
            continue
        rk = renko_trend_cross_func_jit(w)
        if rk is None or len(rk) == 0:
            continue
        cw = pd.to_numeric(w[AKA.CLOSE], errors='coerce')
        for _base, _dummy in RENKO_PRICE_DUMMIES:
            if _base in rk.columns:
                rk[_dummy] = rk[_base] / cw
        key = (pd.Timestamp(T).normalize(), str(code))
        if key not in rk.index:
            continue
        row = rk.loc[key]
        rows[key] = {c: row.get(c, float('nan')) for c in _COLS
                     if c in rk.columns}
        idx.append(key)
    if not rows:
        return None
    out = pd.DataFrame([rows[k] for k in idx],
                       index=pd.MultiIndex.from_tuples(idx, names=['ts', 'code']))
    return out.rename_axis(['ts', 'code'])


def all_codes(db):
    """有 review 日行的全部码（Renko 全量重算用）。"""
    codes = set()
    for mt in ('stock_cn', 'etf_cn', 'index_cn'):
        name = reality_feat_name('day', mt)
        if name in db.list_collection_names():
            codes |= set(str(c) for c in db[name].distinct(
                'code', {'zDLagMajR': {'$ne': None}}))
    return sorted(codes)


def run_code(code, as_of, apply):
    """重算一个码的 Renko（只更新已有 day 行）；返回统计。"""
    from GolemQ import get_active_market
    market = get_active_market()
    db = market.get_database()
    try:
        payload = calc_one(market, code, as_of)
        if payload is None or len(payload) == 0:
            return {'code': code, 'rows': 0}
        mt = is_stock_cn(str(code))[1]
        coll = db[reality_feat_name('day', mt)]
        existing = set(coll.distinct('date', {'code': str(code),
                                              'zDLagMajR': {'$exists': True,
                                                            '$ne': None}}))
        if not existing:
            return {'code': code, 'rows': 0}
        keep = pd.Index(payload.index.get_level_values(0)
                        .strftime('%Y-%m-%d')).isin(existing)
        payload = payload[keep]
        if len(payload) == 0:
            return {'code': code, 'rows': 0}
        if apply:
            save_reality_feat(payload, 'day', mt, verbose=False,
                              write_nulls=False, db=db)
        return {'code': code, 'rows': len(payload)}
    except Exception as exc:            # noqa: BLE001
        return {'code': code, 'rows': 0, 'error': repr(exc)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--codes', default='', help='逗号分隔的 6 位代码')
    ap.add_argument('--partial', action='store_true',
                    help='自动选 day 行存在但 Renko 缺的码')
    ap.add_argument('--all-codes', action='store_true',
                    help='全部有 review 日行的码（重算：后视镜修复后必须跑这个）')
    ap.add_argument('--workers', type=int, default=1)
    ap.add_argument('--apply', action='store_true', help='真正写库（默认 dry-run）')
    ap.add_argument('--as-of', default='', help='截止日 YYYY-MM-DD；默认现在')
    args = ap.parse_args()
    if not available():
        print('renko numba 不可用')
        return 2

    market = get_active_market()
    db = market.get_database()
    codes = [c.strip() for c in args.codes.split(',') if c.strip()]
    if args.partial:
        codes = sorted(set(codes) | set(partial_codes(db)))
    if args.all_codes:
        codes = all_codes(db)
    if not codes:
        print('没有代码：用 --codes 或 --partial')
        return 2

    as_of = pd.Timestamp(args.as_of) if args.as_of else pd.Timestamp.now()
    print('{} {} 个代码 as_of={} apply={}'.format(
        'PARTIAL' if args.partial else 'CODES', len(codes), as_of, args.apply))
    t0 = time.time()
    if args.workers > 1 and len(codes) > 1:
        from joblib import Parallel, delayed
        results = Parallel(n_jobs=args.workers, backend='loky')(
            delayed(run_code)(c, as_of, args.apply) for c in codes)
    else:
        results = []
        for i, c in enumerate(codes, 1):
            results.append(run_code(c, as_of, args.apply))
            if i % 50 == 0:
                print('  … {}/{}  ({:.0f}s)'.format(i, len(codes), time.time() - t0))
    total = sum(r.get('rows', 0) for r in results)
    errs = [r for r in results if r.get('error')]
    print('  codes={} err={}  {:.0f}s'.format(len(results), len(errs),
                                              time.time() - t0))
    for r in errs[:5]:
        print('  ERROR', r)
    print('{} rows {}'.format('written' if args.apply else 'would update', total))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
