# coding: utf-8
"""给已有 review **60min 行**补 `se` / `ZenPivotCnt` / `ChipDist` 四列。

口径与 `pipeline/review/steps/zen.py` 完全一致（直接复用
`steps/hour_extra.py` 的 `sample_entropy_by_day` / `pivot_counts`）：

* `se`：60min 收盘价 `antropy.sample_entropy`，只在 15:00 那根算，
  窗口 ≤2160 根，`round(3)`；盘中沿用上一交易日 15:00（因果 ffill）；
* `ZenPivotCnt`：截至该 bar 已确认的缠论中枢个数（CZSC 因果口径）。

⚠️ **取数一定从 2016 锚点开始**：目标区间头部（如 2024 年以前）也要有
≥2160 根 60min 做基础计算；只取「目标起点往前 900 天」会把头部 SE/中枢
计数算废。这条是用户 2026-10-30 明确要求，别为省 IO 改回去。

用法::

    python tools/backfill_hour_extra.py                  # 报告（dry-run）
    python tools/backfill_hour_extra.py --apply          # 真写
    python tools/backfill_hour_extra.py --codes 600519 --apply
    python tools/backfill_hour_extra.py --unset-day --apply   # 清掉日表两列
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from GolemQ.pipeline.review.steps.hour_extra import (
    build_chip_frame, build_hour_extra_frame,
    sample_day_frame as _sample_day_frame)

#: 目标行标记：有 `zDLagR` 的 60min 行才算 review 写过（CV-only 行不算）。
_MARKER = 'zDLagR'
#: 60min 取数锚点 —— 保证目标头部前有足够暖机（用户 2026-10-30）。
ANCHOR = '2016-01-01'
_DAY_TABLES = ('stock_reality_feat_day', 'etf_reality_feat_day',
               'index_reality_feat_day')


def _db():
    from GolemQ import get_active_market
    return get_active_market().get_database()


def _market_type(code: str):
    from GolemQ.markets.StockCN.symbol import is_stock_cn
    return is_stock_cn(str(code))[1]


def target_datetimes(code: str, since: str = '') -> pd.DatetimeIndex:
    """该码已写入 review 的 60min 时刻（按 `datetime` 字符串解析）。"""
    db = _db()
    from GolemQ.markets.StockCN.reality_save import reality_feat_name
    name = reality_feat_name('60min', _market_type(code))
    q = {'code': str(code), _MARKER: {'$ne': None}}
    cur = db[name].find(q, {'datetime': 1, '_id': 0})
    vals = [pd.Timestamp(d['datetime']) for d in cur if d.get('datetime')]
    if not vals:
        return pd.DatetimeIndex([])
    idx = pd.DatetimeIndex(sorted(set(vals)))
    if since:
        idx = idx[idx >= pd.Timestamp(since)]
    return idx


def day_sample_frame(code: str, targets: pd.DatetimeIndex):
    """从日表采样日族 14 列（15:00=当日、盘中=上一交易日）。

    映射本身走 `hour_extra.sample_day_frame`，与 review 的 `day_sample_step`
    同一份实现；chip 四列由调用方用 `_chip_frame` 独立计算，这里留 NaN。
    """
    from GolemQ.markets.StockCN.reality_save import reality_feat_name
    from GolemQ.pipeline.review.contract import DAY_SAMPLE_COLUMNS
    name = reality_feat_name('day', _market_type(code))
    db = _db()
    if name not in db.list_collection_names():
        return None
    proj = {c: 1 for c in ('date', *DAY_SAMPLE_COLUMNS)}
    docs = list(db[name].find({'code': str(code), 'zDLagMajR': {'$ne': None}},
                              proj))
    if not docs:
        return None
    df = pd.DataFrame(docs).drop_duplicates('date')
    cache = {}
    for _, row in df.iterrows():
        d = pd.Timestamp(row['date']).normalize()
        cache[d] = {c: row.get(c, np.nan) for c in DAY_SAMPLE_COLUMNS}
    out = _sample_day_frame(cache, targets)
    if len(out):
        out.index = pd.MultiIndex.from_arrays(
            [pd.DatetimeIndex(pd.to_datetime(out.index.get_level_values(0))),
             [str(code)] * len(out)], names=['ts', 'code'])
    return out


def _chip_frame(market, code: str, targets: pd.DatetimeIndex,
                close_ts) -> pd.DataFrame:
    """只算 ChipDist 四列（60min 行，÷本 bar close）。

    算法与 review 的 `day_core`/`day_sample_step` 同源：`build_chip_frame`
    是 `hour_extra.py` 里的唯一入口。
    """
    from GolemQ.core.constants import AKA as _AKA
    from GolemQ.markets.StockCN.refdata import (
        GQ_fetch_stock_reality_feat_day)
    from GolemQ.pipeline.review.contract import HOUR_CHIP_COLUMNS

    empty = pd.DataFrame(
        np.full((len(targets), len(HOUR_CHIP_COLUMNS)), np.nan),
        index=pd.MultiIndex.from_arrays(
            [targets, [str(code)] * len(targets)], names=['ts', 'code']),
        columns=list(HOUR_CHIP_COLUMNS))
    try:
        dres, _ = market.get_kline_price_v3(
            code, start=ANCHOR, end=str(targets.max().date()),
            realtime=False, verbose=False)
        day = None if dres is None else getattr(dres, 'data', None)
        if day is None or len(day) < 30:
            return empty
        _tr = GQ_fetch_stock_reality_feat_day(
            str(code), start=str(day.index.get_level_values(0)[0].date()),
            end=str(targets.max().date()), columns=[_AKA.TURNOVER])
        turnover = None if (_tr is None or len(_tr) == 0) else _tr[_AKA.TURNOVER]
        return build_chip_frame(day, targets, str(code),
                                turnover=turnover, close_by_ts=close_ts)
    except Exception:               # noqa: BLE001 单码失败留空
        return empty


def compute_frame(code: str, targets: pd.DatetimeIndex,
                  verbose: bool = False,
                  chip_only: bool = False) -> pd.DataFrame:
    """按 targets 算 60min 一体化行里的**增量列**。

    一次算完：`se` + `ZenPivotCnt` + Renko 14 + M9T 2（60min 原生，2160 as-of）
    + 日族 14（从日表采样，不重算 Poly9/RegTree）。

    ``chip_only=True``：只算 `HOUR_CHIP_COLUMNS` 四列，跳过上面全部列。
    """
    from GolemQ import get_active_market
    from GolemQ.pipeline.review.contract import (DAY_SAMPLE_COLUMNS,
                                                 HOUR_CHIP_COLUMNS,
                                                 RENKO_DAY_COLUMNS)
    market = get_active_market()
    res, _ = market.get_kline_price_min(
        code, start=ANCHOR, end=str(targets.max().date()) + ' 23:59:59',
        frequency='60min', realtime=False, verbose=False)
    ohlc = None if res is None else getattr(res, 'data', None)
    if ohlc is None or len(ohlc) < 30:
        return pd.DataFrame()
    ohlc = ohlc[~ohlc.index.duplicated(keep='last')].sort_index()

    close_ts = pd.Series(
        pd.to_numeric(ohlc['close'], errors='coerce').to_numpy(),
        index=pd.DatetimeIndex(pd.to_datetime(ohlc.index.get_level_values(0))))
    if chip_only:
        return _chip_frame(market, code, targets, close_ts)

    # se/pivot/Renko/M9T：与 review `zen_snapshots_step` **同一构造器**
    frame = build_hour_extra_frame(ohlc, targets, code)
    # ChipDist：同一 `build_chip_frame`（绝对成本 ÷ 本 bar close）
    frame = pd.concat([frame, _chip_frame(market, code, targets, close_ts)],
                      axis=1)

    # 日族采样（不重算）。⚠️ chip 4 列**必须排除**：它们上面已按 60min 行 close
    # 算好，日表里的同名比值（÷当日 close）不能覆盖它。
    ds = day_sample_frame(code, targets)
    if ds is not None:
        _ds_cols = [c for c in DAY_SAMPLE_COLUMNS
                    if c not in HOUR_CHIP_COLUMNS]
        frame[_ds_cols] = ds[_ds_cols]

    if verbose:
        print(f'  {code}: targets={len(targets)} '
              f'se={int(np.isfinite(frame["se"]).sum())} '
              f'piv={int(np.isfinite(frame["ZenPivotCnt"]).sum())} '
              f'renko={int(np.isfinite(frame[RENKO_DAY_COLUMNS[0]]).sum())} '
              f'day14={0 if ds is None else int(ds.notna().any(axis=1).sum())}')
    return frame


def backfill_code(code: str, apply: bool, since: str = '',
                  verbose: bool = False, redo: bool = False,
                  chip_only: bool = False) -> dict:
    """补一个码；返回统计。异常不抛出（单码失败不该拖垮整批）。

    `apply=True` 且 `redo=False` 时**跳过已经有 `se` 的码**（断点续跑）。
    """
    try:
        targets = target_datetimes(code, since=since)
        if len(targets) == 0:
            return {'code': code, 'status': 'no_targets'}
        if apply and not redo:
            from GolemQ.markets.StockCN.reality_save import reality_feat_name
            from GolemQ.pipeline.review.contract import HOUR_CHIP_COLUMNS
            name = reality_feat_name('60min', _market_type(code))
            # 只有**全部新列**都已有值才算完成（旧版只写了 se/piv，不能跳过）。
            # ⚠️ 2026-11-06：补上 ChipDist 四列 —— 此前 probe 漏了它们，
            # 导致「se/piv/renko 都有、cost*PrzD 全空」的码被误判为完成，
            # 全库 898 个码因此没补上（000711 就是其中之一）。
            if chip_only:
                # chip 可能只有部分日期已补（000711 01-22/23 就是手工补的），
                # 不能用 find_one；按「target 行数 vs 已有 cost15PrzD 行数」判。
                q = {'code': str(code), 'cost15PrzD': {'$ne': None}}
                if since:
                    q['datetime'] = {'$gte': str(since)}
                if _db()[name].count_documents(q) >= len(targets):
                    return {'code': code, 'status': 'skipped'}
            else:
                probe = {'code': str(code), 'se': {'$ne': None},
                         'ZenPivotCnt': {'$ne': None},
                         'renkoSbar': {'$ne': None},
                         'poly9D': {'$ne': None},
                         'rTrePrcD': {'$ne': None},
                         'poly9MajD': {'$ne': None}}
                for c in HOUR_CHIP_COLUMNS:
                    probe[c] = {'$ne': None}
                if _db()[name].find_one(probe, {'_id': 1}):
                    return {'code': code, 'status': 'skipped'}
        frame = compute_frame(code, targets, verbose=verbose,
                              chip_only=chip_only)
        if frame.empty:
            return {'code': code, 'status': 'no_kline'}
        stats = {'code': code, 'status': 'ok', 'rows': len(frame)}
        if apply:
            from GolemQ.markets.StockCN.reality_save import save_reality_feat
            out = save_reality_feat(frame, '60min', _market_type(code),
                                    verbose=False, write_nulls=False,
                                    db=_db())
            stats.update(upserted=out['upserted'], modified=out['modified'])
        return stats
    except Exception as exc:            # noqa: BLE001
        return {'code': code, 'status': 'error', 'error': repr(exc)}


def all_codes(limit: int = 0) -> list:
    """有 60min review 行的码（按 code 去重）。"""
    db = _db()
    from GolemQ.markets.StockCN.reality_save import reality_feat_name
    codes = set()
    for mt in ('stock_cn', 'etf_cn', 'index_cn'):
        name = reality_feat_name('60min', mt)
        if name not in db.list_collection_names():
            continue
        for c in db[name].distinct('code', {_MARKER: {'$ne': None}}):
            codes.add(str(c))
    out = sorted(codes)
    return out[:limit] if limit else out


def unset_day(apply: bool) -> None:
    """删除日表里的 `se` / `ZenPivotCnt`（D78：列只留 60min）。"""
    db = _db()
    for name in _DAY_TABLES:
        if name not in db.list_collection_names():
            continue
        n = db[name].count_documents({'$or': [{'se': {'$exists': True}},
                                              {'ZenPivotCnt': {'$exists': True}}]})
        print(f'[unset-day] {name}: {n} 行含这两列', end='')
        if apply and n:
            res = db[name].update_many({}, {'$unset': {'se': '', 'ZenPivotCnt': ''}})
            print(f' → 已清理 {res.modified}')
        else:
            print('（dry-run）')


def unset_day_renko(apply: bool) -> None:
    """删除日表里的 Renko 14 列（D78：Renko 只落 60min）。"""
    from GolemQ.pipeline.review.contract import RENKO_DAY_COLUMNS
    db = _db()
    proj = {'$or': [{c: {'$exists': True}} for c in RENKO_DAY_COLUMNS]}
    for name in _DAY_TABLES:
        if name not in db.list_collection_names():
            continue
        n = db[name].count_documents(proj)
        print(f'[unset-day-renko] {name}: {n} 行含 Renko 列', end='')
        if apply and n:
            res = db[name].update_many(
                {}, {'$unset': {c: '' for c in RENKO_DAY_COLUMNS}})
            print(f' → 已清理 {res.modified}')
        else:
            print('（dry-run）')


def main(argv=None):
    parser = argparse.ArgumentParser(description='补 60min se / ZenPivotCnt（D78）')
    parser.add_argument('--codes', default='', help='逗号分隔；默认取全库有 review 行的码')
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--since', default='', help='只补该日期之后的 60min 行')
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--shard', default='',
                        help='分布式分片 "i/N"（1-based）：按 code 哈希取模，'
                             '多台机器各跑一个分片、写同一个 8.3 库')
    parser.add_argument('--apply', action='store_true', help='真写（默认 dry-run）')
    parser.add_argument('--unset-day', action='store_true',
                        help='同时清理日表的 se / ZenPivotCnt')
    parser.add_argument('--unset-day-renko', action='store_true',
                        help='清理日表的 Renko 14 列（D78：Renko 只落 60min）')
    parser.add_argument('--redo', action='store_true',
                        help='重算已有 se 的码（默认跳过 = 断点续跑）')
    parser.add_argument('--chip-only', action='store_true',
                        help='只补 ChipDist 四列（跳过 se/pivot/renko/native）')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args(argv)

    if args.unset_day:
        unset_day(args.apply)
    if args.unset_day_renko:
        unset_day_renko(args.apply)

    codes = ([c.strip() for c in args.codes.split(',') if c.strip()]
             or all_codes(limit=args.limit))
    if args.shard:
        # 与 `--review-shard` **同一实现**（pipeline.review.pipeline.shard_codes），
        # 两处各写一份迟早会分叉。
        from GolemQ.pipeline.review.pipeline import shard_codes
        try:
            codes = shard_codes(codes, args.shard)
        except ValueError as exc:
            raise SystemExit(str(exc))
    print(f'[backfill] {len(codes)} 个码，apply={args.apply}，since={args.since or "-"}')
    t0 = time.time()
    results = []
    if args.workers > 1 and len(codes) > 1:
        from joblib import Parallel, delayed
        results = Parallel(n_jobs=args.workers, backend='loky')(
            delayed(backfill_code)(c, args.apply, args.since, args.verbose,
                                   args.redo, args.chip_only)
            for c in codes)
    else:
        for i, c in enumerate(codes, 1):
            r = backfill_code(c, args.apply, args.since, args.verbose,
                              args.redo, args.chip_only)
            results.append(r)
            if i % 100 == 0:
                print(f'  … {i}/{len(codes)}  ({time.time() - t0:.0f}s)')

    ok = [r for r in results if r.get('status') == 'ok']
    err = [r for r in results if r.get('status') == 'error']
    skipped = [r for r in results if r.get('status') == 'skipped']
    rows = sum(r.get('rows', 0) for r in ok)
    print(f'[backfill] ok={len(ok)}  skipped={len(skipped)}  error={len(err)}  '
          f'rows={rows}  {time.time() - t0:.0f}s')
    for r in err[:10]:
        print('  ERROR', r)


if __name__ == '__main__':
    sys.exit(main())
