# coding: utf-8
"""审计 review 落库值的 NaN 形态（稠密列用）。

重点抓这个 Bug 形态（用户 2026-10-30 点名）：
**同一天 15:00 有值、盘中 bar 却是 NaN**。对 60min 稠密列逐日检查
「最后一根非空但有更早的 bar 为空」的天数。

另报每列的缺失率，便于区分：
* 稠密列（每 bar 应有值）：缺失应能落到「暖机不足 / 源数据缺失 / 算法失败」；
* 稀疏列（ChartScan 事件）：缺失是正常的。

用法::

    python tools/audit_review_nan.py --codes 600519,000001,301308 --days 60
    python tools/audit_review_nan.py --family day --codes 600519 --days 400
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

from GolemQ.pipeline.review.contract import (COLDSTART_COLUMNS, DAY_COLUMNS,
                                             DAY_SAMPLE_COLUMNS, HOUR_COLUMNS,
                                             HOUR_EXTRA_COLUMNS,
                                             HOUR_NATIVE_COLUMNS,
                                             RENKO_DAY_COLUMNS)
from GolemQ.markets.StockCN.symbol import is_stock_cn

#: 稀疏列：NaN 是正常（事件型 / 拟合型），不按稠密规则要求每 bar 有值。
SPARSE = {'Class', 'Confidence', 'Coordinates', 'ChartScaned'}
DENSE_HOUR = tuple(c for c in HOUR_COLUMNS if c not in SPARSE)
#: A 股 60min 的合法 bar 时刻；00:00 / 13:00 之类是脏行，不计入盘缺判定。
LEGAL_TIMES = ('10:30', '11:30', '14:00', '15:00')
#: 盘中取「上一交易日快照」的列：昨日快照本身为 NaN 时，盘中 NaN 是正确值。
PREV_DAY_COLS = (set(DAY_SAMPLE_COLUMNS) | set(COLDSTART_COLUMNS) |
                 {'se', 'ZenPivotCnt', 'ZenPivotLag'} |
                 set(RENKO_DAY_COLUMNS) | set(HOUR_NATIVE_COLUMNS))
DENSE_DAY = tuple(DAY_COLUMNS)


def _db():
    from GolemQ import get_active_market
    return get_active_market().get_database()


def _coll(code, family):
    from GolemQ import get_active_market
    from GolemQ.markets.StockCN.reality_save import reality_feat_name
    mt = is_stock_cn(str(code))[1]
    return get_active_market().get_database()[reality_feat_name(family, mt)]


def audit_hour(code, days):
    """60min：逐日检查 15:00/盘中 的 NaN 形态。"""
    coll = _coll(code, '60min')
    docs = list(coll.find({'code': str(code)}, {c: 1 for c in
                                                ('datetime', *DENSE_HOUR, *SPARSE)}))
    if not docs:
        return None
    rows = pd.DataFrame(docs)
    rows['dt'] = pd.to_datetime(rows['datetime'])
    rows = rows.sort_values('dt')
    if days:
        rows = rows[rows['dt'] >= rows['dt'].max() - pd.Timedelta(days=int(days))]
    # 脏时刻（非 10:30/11:30/14:00/15:00）单独报告，不参与盘缺判定
    dirty = rows[~rows['dt'].dt.strftime('%H:%M').isin(LEGAL_TIMES)]
    if len(dirty):
        print(f'  ⚠️ 脏 bar 时刻: {sorted(set(dirty["dt"].dt.strftime("%H:%M")))}'
              f'（{len(dirty)} 根，已排除）')
    rows = rows[rows['dt'].dt.strftime('%H:%M').isin(LEGAL_TIMES)]
    if not len(rows):
        return None
    rows['date'] = rows['dt'].dt.normalize()
    rows['is_last'] = rows.groupby('date')['dt'].transform('max') == rows['dt']
    # 冷启动列盘中取**上一交易日**的值：只有上一交易日该列非空时，盘中缺才算 bug
    lastvals = rows[rows['is_last']].set_index('date')
    dates = sorted(lastvals.index)
    prev_of = {d: dates[i - 1] for i, d in enumerate(dates) if i > 0}
    out = []
    for c in DENSE_HOUR:
        if c not in rows.columns:
            continue
        s = rows[c]
        null_rate = float(s.isna().mean())
        bad = 0
        for d, g in rows.groupby('date'):
            if len(g) < 2:
                continue
            last = g.loc[g['is_last'], c]
            if last.isna().any():
                continue
            miss = g.loc[~g['is_last'] & g[c].isna()]
            if len(miss) == 0:
                continue
            if c in PREV_DAY_COLS:
                pd_ = prev_of.get(d)
                if pd_ is None:
                    continue
                prev_vals = lastvals.loc[pd_, c]
                prev_vals = np.asarray(prev_vals).ravel()
                if len(prev_vals) == 0 or pd.isna(prev_vals[0]):
                    continue        # 上一交易日本身就是 NaN —— 不是写入缺口
            bad += 1
        out.append({'col': c, 'rows': len(rows), 'null%': round(null_rate * 100, 2),
                    '15:00有值/盘中缺(天)': bad})
    return pd.DataFrame(out)


def audit_day(code, days):
    coll = _coll(code, 'day')
    docs = list(coll.find({'code': str(code)}, {c: 1 for c in ('date', *DENSE_DAY)}))
    if not docs:
        return None
    rows = pd.DataFrame(docs)
    if days:
        d = pd.to_datetime(rows['date'])
        rows = rows[d >= d.max() - pd.Timedelta(days=int(days))]
    out = [{'col': c, 'rows': len(rows),
            'null%': round(float(rows[c].isna().mean()) * 100, 2)}
           for c in DENSE_DAY if c in rows.columns]
    return pd.DataFrame(out)


def _bad_coldstart_days(code, days):
    """返回 [(date, [缺失的盘中时刻])] —— **任一 coldstart 列** 15:00 有值但盘中缺。"""
    from GolemQ.pipeline.review.contract import COLDSTART_COLUMNS
    coll = _coll(code, '60min')
    docs = list(coll.find({'code': str(code)},
                          {'datetime': 1, **{c: 1 for c in COLDSTART_COLUMNS}}))
    if not docs:
        return []
    rows = pd.DataFrame(docs)
    rows['dt'] = pd.to_datetime(rows['datetime'])
    rows = rows.sort_values('dt')
    if days:
        rows = rows[rows['dt'] >= rows['dt'].max() - pd.Timedelta(days=int(days))]
    # 脏时刻（非 10:30/11:30/14:00/15:00）单独报告，不参与盘缺判定
    dirty = rows[~rows['dt'].dt.strftime('%H:%M').isin(LEGAL_TIMES)]
    if len(dirty):
        print(f'  ⚠️ 脏 bar 时刻: {sorted(set(dirty["dt"].dt.strftime("%H:%M")))}'
              f'（{len(dirty)} 根，已排除）')
    rows = rows[rows['dt'].dt.strftime('%H:%M').isin(LEGAL_TIMES)]
    if not len(rows):
        return None
    rows['date'] = rows['dt'].dt.normalize()
    rows['is_last'] = rows.groupby('date')['dt'].transform('max') == rows['dt']
    cols = [c for c in COLDSTART_COLUMNS if c in rows.columns]
    bad = []
    for d, g in rows.groupby('date'):
        if len(g) < 2:
            continue
        need = set()
        for c in cols:
            if g.loc[g['is_last'], c].isna().any():
                continue
            m = g.loc[~g['is_last'] & g[c].isna(), 'dt']
            need |= set(m)
        if need:
            bad.append((d, sorted(need)))
    return bad


def repair_coldstart(code, days, apply):
    """重算「15:00 有值、盘中缺」的 coldstart 列（只写盘中行）。"""
    from GolemQ import get_active_market
    from GolemQ.pipeline.review.coldstart import _coldstart_row
    from GolemQ.pipeline.steps import slice_asof
    from GolemQ.pipeline.review.contract import COLDSTART_COLUMNS
    from GolemQ.markets.StockCN.reality_save import save_reality_feat

    bad = _bad_coldstart_days(code, days)
    if not bad:
        return 0
    market = get_active_market()
    end = max(t for _d, ts in bad for t in ts)
    res, _ = market.get_kline_price_v3(str(code), start='2016-01-01',
                                       end=str(end.date()),
                                       realtime=False, verbose=False)
    day_all = None if res is None else getattr(res, 'data', None)
    if day_all is None or len(day_all) == 0:
        return 0
    day_all = day_all[~day_all.index.duplicated(keep='last')].sort_index()
    day_dates = pd.DatetimeIndex(pd.to_datetime(
        day_all.index.get_level_values(0))).normalize().unique().sort_values()

    idx, rows = [], []
    for d, miss in bad:
        pos = day_dates.searchsorted(pd.Timestamp(d), side='left') - 1
        if pos < 0:
            continue
        prev = day_dates[pos]
        row = _coldstart_row(slice_asof(day_all, prev, 832))
        for t in miss:
            idx.append((pd.Timestamp(t), str(code)))
            rows.append([row.get(c, float('nan')) for c in COLDSTART_COLUMNS])
        print(f'  repair {code} {d:%Y-%m-%d}: {len(miss)} 根盘中 ← coldstart({prev:%Y-%m-%d})')
    if not idx:
        return 0
    frame = pd.DataFrame(rows,
                         index=pd.MultiIndex.from_tuples(idx, names=['ts', 'code']),
                         columns=list(COLDSTART_COLUMNS))
    if apply:
        save_reality_feat(frame, '60min', is_stock_cn(str(code))[1],
                          verbose=False, write_nulls=False, db=_db())
    return len(frame)


def main(argv=None):
    ap = argparse.ArgumentParser(description='review 落库 NaN 形态审计')
    ap.add_argument('--codes', default='600519,000001,301308,601869')
    ap.add_argument('--family', default='60min', choices=('60min', 'day', 'both'))
    ap.add_argument('--days', type=int, default=60)
    ap.add_argument('--repair', action='store_true',
                    help='重算命中「15:00 有值、盘中缺」的 coldstart 列')
    ap.add_argument('--apply', action='store_true', help='--repair 时真写')
    args = ap.parse_args(argv)

    fams = ('60min', 'day') if args.family == 'both' else (args.family,)
    for code in [c.strip() for c in args.codes.split(',') if c.strip()]:
        for fam in fams:
            df = (audit_hour if fam == '60min' else audit_day)(code, args.days)
            print(f'\n=== {code} {fam}（最近 {args.days} 天）')
            if df is None or df.empty:
                print('  无数据')
                continue
            print(df.to_string(index=False))
            if fam == '60min':
                n_bad = int(df['15:00有值/盘中缺(天)'].max())
                print(f'  → 15:00 有值但盘中缺：{n_bad} 天'
                      + ('  ✅' if n_bad == 0 else '  ⚠️ 需要查写入端是否漏目标'))
                if args.repair and n_bad:
                    n = repair_coldstart(code, args.days, args.apply)
                    print(f'  repair rows={n} apply={args.apply}')


if __name__ == '__main__':
    sys.exit(main())
