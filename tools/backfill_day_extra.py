#!/usr/bin/env python3
# coding: utf-8
"""给已有 review **日行**补 `se` / `ZenPivotCnt`（D74）。

口径（与 review 写入口径完全一致）
==================================
* `se`：**日收盘**样本熵，截至该日最后 ≤540 个交易日，`round(3)`
  （`pipeline/review/steps/day_extra.sample_entropy_daily`）；
* `ZenPivotCnt`：优先从 `stock_reality_feat_60min` 每**日最后一根**复制
  （= 15:00 采样）；没有可复制的值时，用 **2016-01-01 起全历史 60min** 现算
  （与 review 同口径）。

用法
====
```bash
python tools/backfill_day_extra.py --partial             # 看多少码/行
python tools/backfill_day_extra.py --partial --apply     # 真正写库
python tools/backfill_day_extra.py --all --apply         # 全部码重算（全历史）
python tools/backfill_day_extra.py --codes 000711 --apply
python tools/backfill_day_extra.py --unset-hour --apply  # 写完后删 60min 两列
```
⚠️ `--unset-hour` 会**删除** 60min 的 `se`/`ZenPivotCnt`（D74：列只留在日族）。
它必须在复制完成之后执行；默认 dry-run 时只计数不删。
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from GolemQ import get_active_market
from GolemQ.core.constants import AKA
from GolemQ.markets.StockCN.reality_save import (reality_feat_name,
                                                 save_reality_feat)
from GolemQ.markets.StockCN.symbol import is_stock_cn
from GolemQ.models.alias import ZEN
from GolemQ.pipeline.review.pipeline import REVIEW_ANCHOR_DAY
from GolemQ.pipeline.review.steps.day_extra import (pivot_counts,
                                                    sample_entropy_daily)

_COLS = (AKA.SAMPLE_ENTROPY, ZEN.PIVOT_COUNT)
_MARKET_TYPES = ('stock_cn', 'etf_cn', 'index_cn')


def review_dates(db, code, mt):
    """该码已有的 review 日行日期（字符串，升序）。"""
    coll = db[reality_feat_name('day', mt)]
    cur = coll.find({'code': str(code),
                     'zDLagMajR': {'$exists': True, '$ne': None}}, {'date': 1})
    return sorted({r['date'] for r in cur if r.get('date')})


def partial_codes(db, threshold=10, warmup=30):
    """review 日行存在、但 `se` 成块缺失（>= threshold 行）的码。

    ⚠️ 跳过每码最早的 `warmup` 行：新股上市初期窗口不足，样本熵本来就算不出
    （实测前 ~16 行为 NaN）—— 不跳过的话这些码会被 `--partial` 永远选中。
    """
    coll = db[reality_feat_name('day', 'stock_cn')]
    rows = coll.aggregate([
        {'$match': {'zDLagMajR': {'$exists': True, '$ne': None}}},
        {'$setWindowFields': {'partitionBy': '$code', 'sortBy': {'date': 1},
                              'output': {'rn': {'$documentNumber': {}}}}},
        {'$match': {'rn': {'$gt': warmup}}},
        {'$group': {'_id': '$code', 'review': {'$sum': 1},
                    'have': {'$sum': {'$cond': [
                        {'$ne': [{'$ifNull': ['$' + AKA.SAMPLE_ENTROPY, None]},
                                 None]}, 1, 0]}}}},
        {'$addFields': {'missing': {'$subtract': ['$review', '$have']}}},
        {'$match': {'missing': {'$gte': threshold}}},
        {'$sort': {'_id': 1}},
    ], allowDiskUse=True)
    return [r['_id'] for r in rows]


def all_codes(db):
    """所有有 review 日行的码（按三种行情族汇总）。"""
    codes = set()
    for mt in _MARKET_TYPES:
        try:
            coll = db[reality_feat_name('day', mt)]
        except Exception:               # noqa: BLE001
            continue
        codes.update(coll.distinct('code', {'zDLagMajR': {'$exists': True,
                                                         '$ne': None}}))
    return sorted(codes)


def hour_counts(db, code, mt):
    """60min 每**日最后一根**的 `ZenPivotCnt`（15:00 采样）→ `{date: value}`。"""
    coll = db[reality_feat_name('60min', mt)]
    rows = coll.aggregate([
        {'$match': {'code': str(code), 'ZenPivotCnt': {'$ne': None}}},
        {'$sort': {'datetime': 1}},
        {'$group': {'_id': '$date',
                    'ZenPivotCnt': {'$last': '$ZenPivotCnt'}}},
    ], allowDiskUse=True)
    return {r['_id']: r['ZenPivotCnt'] for r in rows}


def calc_one(market, db, code, as_of):
    """返回只含 se / ZenPivotCnt 的日族帧（索引 `(date, code)`）；无 review 行返回 None。"""
    mt = is_stock_cn(str(code))[1]
    dates = review_dates(db, code, mt)
    if not dates:
        return None
    dts = pd.DatetimeIndex(pd.to_datetime(dates))
    res, _ = market.get_kline_price_v3(
        code, start=REVIEW_ANCHOR_DAY, end=str(as_of.date()),
        realtime=False, verbose=False)
    day = None if res is None else getattr(res, 'data', None)
    if day is None or len(day) == 0:
        return None
    se = sample_entropy_daily(day, dts)
    piv_map = hour_counts(db, code, mt)
    piv = np.array([piv_map.get(d, np.nan) for d in dates], dtype=float)

    # 60min 列已 `$unset`（D74）后，DB 里没有可复制的值 —— 现算 15:00 采样。
    miss = [i for i, d in enumerate(dates) if d not in piv_map]
    if miss:
        # 全历史 60min（D74）：日行最早到 2019-10，CZSC 还要从 2016 起暖机。
        res60, _ = market.get_kline_price_min(
            code, start=REVIEW_ANCHOR_DAY, end=str(as_of.date()) + ' 23:59:59',
            frequency='60min', realtime=False, verbose=False)
        hour = None if res60 is None else getattr(res60, 'data', None)
        cut = pd.DatetimeIndex([pd.Timestamp(dates[i]) + pd.Timedelta(hours=15)
                                for i in miss])
        piv[miss] = pivot_counts(hour, cut, code)
    idx = pd.MultiIndex.from_arrays(
        [dts, [str(code)] * len(dts)], names=['ts', 'code'])
    return pd.DataFrame({AKA.SAMPLE_ENTROPY: se, ZEN.PIVOT_COUNT: piv},
                        index=idx)


def unset_hour_columns(db, apply=False):
    total = 0
    for mt in _MARKET_TYPES:
        try:
            coll = db[reality_feat_name('60min', mt)]
        except Exception:               # noqa: BLE001 该市场族可能不存在
            continue
        n = coll.count_documents({'$or': [{c: {'$exists': True}} for c in _COLS]})
        total += n
        print('  {}: {} 行含 se/ZenPivotCnt'.format(mt, n))
        if apply and n:
            coll.update_many({}, {'$unset': {c: '' for c in _COLS}})
    print('{} 60min 行{}'.format('已删除' if apply else '将删除', total))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--codes', default='', help='逗号分隔的 6 位代码')
    ap.add_argument('--partial', action='store_true',
                    help='自动选 day 行存在但 se 成块缺失的码')
    ap.add_argument('--all', action='store_true',
                    help='全部有 review 日行的码（重算 ZenPivotCnt 全历史）')
    ap.add_argument('--apply', action='store_true', help='真正写库（默认 dry-run）')
    ap.add_argument('--as-of', default='', help='截止日 YYYY-MM-DD；默认现在')
    ap.add_argument('--unset-hour', action='store_true',
                    help='补完后删除 60min 的 se/ZenPivotCnt（D74）')
    args = ap.parse_args()

    market = get_active_market()
    db = market.get_database()
    if args.unset_hour:
        unset_hour_columns(db, apply=args.apply)
        if not args.codes and not args.partial:
            return 0

    codes = [c.strip() for c in args.codes.split(',') if c.strip()]
    if args.partial:
        codes = sorted(set(codes) | set(partial_codes(db)))
    elif args.all:
        codes = sorted(set(codes) | set(all_codes(db)))
    if not codes:
        print('没有代码：用 --codes 或 --partial')
        return 2

    as_of = pd.Timestamp(args.as_of) if args.as_of else pd.Timestamp.now()
    print('{} {} 个代码 as_of={} apply={}'.format(
        'PARTIAL' if args.partial else 'CODES', len(codes), as_of, args.apply))
    total = 0
    for code in codes:
        payload = calc_one(market, db, code, as_of)
        if payload is None or len(payload) == 0:
            print('  {}  no data'.format(code))
            continue
        total += len(payload)
        print('  {}  rows={}'.format(code, len(payload)))
        if args.apply:
            mt = is_stock_cn(str(code))[1]
            save_reality_feat(payload, 'day', mt, verbose=False,
                              write_nulls=False, db=db)
    print('{} rows {}'.format('written' if args.apply else 'would update', total))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
