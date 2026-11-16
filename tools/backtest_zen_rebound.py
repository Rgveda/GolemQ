# coding: utf-8
"""Research backtest: ZenPivotLag re-arm + MAIN STREAM hold regime.

Rules (user 2026-11):
* 60min `ZenPivotLag` from `stock_reality_feat_60min`.
* arm when ZenPivotLag > 84; enter on the first `ZenPivotLag == -1`
  at the **next 60min open**, only if daily-major
  `MAINSTREAM_LEADIN_BEFORE_MAJOR < MAINSTREAM_LEADOUT_BEFORE_MAJOR`.
* hold while that leadin<leadout condition remains true at bar close;
  exit at next 60min open when it fails.
* equal-weight entries, 100-share lots, principal 200,000,
  fee = max(amount * 0.0001, 5) on each side.

This is a research tool, not production: it reads Mongo directly to avoid
per-code market-object overhead, and uses qfq (raw OHLC * stock_adj).
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, '.')

from GolemQ import get_active_market
from GolemQ.analysis.compact import _map_causal
from GolemQ.analysis.indicators import calc_macd_qa
from GolemQ.analysis.quadrant_jit import calc_mainstream_feats
from GolemQ.core.constants import AKA, FIELD as FLD
from GolemQ.models.alias import LTT

START = '2024-08-12'
END = '2026-11-09 23:59:59'
DAY_START = '2016-01-01'


def _db():
    return get_active_market().get_database()


def _docs(coll, code, start, end, fields):
    return list(_db()[coll].find(
        {'code': code, 'datetime': {'$gte': start, '$lte': end}},
        {'_id': 0, **fields}))


def _day_docs(code):
    return list(_db()['stock_day'].find(
        {'code': code, 'date': {'$gte': DAY_START}},
        {'_id': 0, 'date': 1, 'open': 1, 'high': 1, 'low': 1,
         'close': 1, 'vol': 1}))


def _adj_map(code):
    return {d['date']: float(d['adj']) for d in _db()['stock_adj'].find(
        {'code': code, 'date': {'$gte': DAY_START}},
        {'_id': 0, 'date': 1, 'adj': 1})}


def load_events(code: str):
    """Return (events, last_close, error). events: (ts, price, side)."""
    try:
        rev = _docs('stock_reality_feat_60min', code, START, END,
                    {'datetime': 1, 'ZenPivotLag': 1})
        if len(rev) < 120:
            return [], np.nan, 'short_review'
        k60 = _docs('stock_60min', code, START[:10], END,
                    {'datetime': 1, 'date': 1, 'open': 1, 'close': 1})
        if len(k60) < 120:
            return [], np.nan, 'no_60min'
        day = _day_docs(code)
        if len(day) < 300:
            return [], np.nan, 'short_day'
        adj = _adj_map(code)

        r = pd.DataFrame(rev).drop_duplicates('datetime')
        r['ts'] = pd.to_datetime(r['datetime'])
        r = r.dropna(subset=['ZenPivotLag']).sort_values('ts')
        k = pd.DataFrame(k60).drop_duplicates('datetime')
        k['ts'] = pd.to_datetime(k['datetime'])
        k = k.sort_values('ts')
        k['adj'] = k['date'].map(adj).fillna(1.0)
        for c in ('open', 'close'):
            k[c + '_q'] = pd.to_numeric(k[c], errors='coerce') * k['adj']
        m = r.merge(k[['ts', 'open_q', 'close_q']], on='ts', how='inner')
        m = m.dropna(subset=['ZenPivotLag', 'open_q', 'close_q'])
        if len(m) < 120:
            return [], np.nan, 'short_merged'

        d = pd.DataFrame(day).drop_duplicates('date').sort_values('date')
        d['adj'] = d['date'].map(adj).fillna(1.0)
        for c in ('open', 'high', 'low', 'close'):
            d[c + '_q'] = pd.to_numeric(d[c], errors='coerce') * d['adj']
        d = d.dropna(subset=['open_q', 'high_q', 'low_q', 'close_q'])
        d['volume'] = pd.to_numeric(d.get('vol'), errors='coerce')
        day_q = pd.DataFrame({
            AKA.OPEN: d['open_q'].to_numpy(),
            AKA.HIGH: d['high_q'].to_numpy(),
            AKA.LOW: d['low_q'].to_numpy(),
            AKA.CLOSE: d['close_q'].to_numpy(),
            AKA.VOLUME: d['volume'].to_numpy(),
        }, index=pd.DatetimeIndex(pd.to_datetime(d['date'])).normalize())
        day_q = day_q[~day_q.index.duplicated(keep='last')].sort_index()
        f = pd.DataFrame(index=day_q.index)
        f[FLD.MACD] = calc_macd_qa(day_q[AKA.CLOSE])[2]
        ms = calc_mainstream_feats(day_q, f, fold=True)

        def to_60(series):
            s = pd.Series(series.to_numpy(), index=day_q.index)
            return _map_causal(s, pd.MultiIndex.from_arrays(
                [m['ts'], [code] * len(m)], names=['ts', 'code'])).to_numpy()

        leadin = to_60(ms[LTT.MAINSTREAM_LEADIN_BEFORE])
        leadout = to_60(ms[LTT.MAINSTREAM_LEADOUT_BEFORE])
        cond = leadin < leadout
        zen = pd.to_numeric(m['ZenPivotLag'], errors='coerce').to_numpy()
        op = m['open_q'].to_numpy()
        cl = m['close_q'].to_numpy()
        ts = pd.DatetimeIndex(m['ts'])

        events = []
        armed = False
        in_pos = False
        for i in range(len(m) - 1):
            # 修复后 `ZenPivotLag` 是中枢内 +1 累计，所以“远离中枢 >84”
            # 就是 `> 84`；等它**首次转负**（离开中枢后的第一根）再入场。
            # ⚠️ review 的 ZenPivotLag 按日 15:00 采样，离开中枢若发生在盘中，
            # 15:00 已经是 -2/-3，**不会**恰好等于 -1；因此用 `< 0`。
            if zen[i] > 84:
                armed = True
            if not in_pos:
                if armed and zen[i] < 0:
                    if cond[i]:
                        events.append((ts[i + 1], float(op[i + 1]), 'B', code))
                        in_pos = True
                    armed = False
            else:
                if not cond[i]:
                    events.append((ts[i + 1], float(op[i + 1]), 'S', code))
                    in_pos = False
        if in_pos:
            events.append((ts[-1], float(cl[-1]), 'S', code))
        return events, float(cl[-1]), None
    except Exception as exc:        # noqa: BLE001
        return [], np.nan, '{}: {}'.format(type(exc).__name__, exc)


def simulate(all_events, last_close=None, principal=200_000.0,
             fee_rate=0.0001, min_fee=5.0, max_positions=0):
    by_ts = defaultdict(list)
    for ts, price, side, code in all_events:
        by_ts[pd.Timestamp(ts)].append((side, price, code))
    cash = float(principal)
    positions = {}
    realized = []
    max_pos = 0
    counts = []
    skipped_lot = skipped_cash = 0
    total_fee = 0.0
    for ts in sorted(by_ts):
        bucket = by_ts[ts]
        # exits first
        for side, price, code in bucket:
            if side != 'S' or code not in positions:
                continue
            pos = positions.pop(code)
            amount = pos['qty'] * price
            fee = max(amount * fee_rate, min_fee)
            total_fee += fee
            cash += amount - fee
            realized.append((amount - pos['cost'] - fee - pos['fee'], code))
        entries = [(p, c) for s, p, c in bucket if s == 'B' and c not in positions]
        entries.sort(key=lambda x: x[1])
        if max_positions and len(positions) >= max_positions:
            entries = []
        elif max_positions:
            entries = entries[:max_positions - len(positions)]
        n_new = len(entries)
        if n_new:
            # 等权目标：成本口径权益 / (现有持仓 + 本次新开)
            equity_cost = cash + sum(p['cost'] for p in positions.values())
            target = equity_cost / (len(positions) + n_new)
            for price, code in sorted(entries, key=lambda x: x[1]):
                if not (price > 0 and np.isfinite(price)):
                    continue
                qty = int(target // (price * 100.0)) * 100
                if qty < 100:
                    skipped_lot += 1
                    continue
                amount = qty * price
                fee = max(amount * fee_rate, min_fee)
                if cash < amount + fee:
                    skipped_cash += 1
                    continue
                total_fee += fee
                cash -= amount + fee
                positions[code] = {'qty': qty, 'cost': amount,
                                   'price': price, 'fee': fee}
        max_pos = max(max_pos, len(positions))
        counts.append(len(positions))
    # 未平仓按最后一个可用 close 计市值；没有 close 才退回成本价。
    last_close = last_close or {}
    open_value = sum(
        p['qty'] * float(last_close.get(code, p['price']))
        for code, p in positions.items())
    final = cash + open_value
    wins = sum(1 for pnl, _ in realized if pnl > 0)
    years = (pd.Timestamp(END) - pd.Timestamp(START)).days / 365.25
    cagr = (final / principal) ** (1.0 / years) - 1.0 if years > 0 else np.nan
    return {
        'principal': principal,
        'final_equity': final,
        'return': final / principal - 1.0,
        'years': years,
        'cagr': cagr,
        'open_value': open_value,
        'trades': len(realized),
        'wins': wins,
        'win_rate': wins / len(realized) if realized else 0.0,
        'cap': int(max_positions or 0),
        'max_positions': max_pos,
        'avg_positions': float(np.mean(counts)) if counts else 0.0,
        'open_positions': len(positions),
        'skipped_lot': skipped_lot,
        'skipped_cash': skipped_cash,
        'total_fee': total_fee,
        'realized_pnl': sum(p for p, _ in realized),
    }


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--codes', default='')
    p.add_argument('--codes-file', default='')
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--max-positions', default='',
                   help='逗号分隔的持仓上限（如 20,30,50）；不传则不限仓')
    a = p.parse_args(argv)
    db = _db()
    if a.codes_file:
        with open(a.codes_file, encoding='utf-8') as fh:
            codes = [ln.strip() for ln in fh if ln.strip()]
    elif a.codes:
        codes = [x.strip() for x in a.codes.split(',') if x.strip()]
    else:
        codes = sorted(db['stock_reality_feat_60min'].distinct(
            'code', {'ZenPivotLag': {'$ne': None}}))
    if a.limit:
        codes = codes[:a.limit]
    print('[bt] codes', len(codes), flush=True)
    from joblib import Parallel, delayed
    res = Parallel(n_jobs=a.workers, backend='loky')(
        delayed(load_events)(c) for c in codes)
    all_events = []
    errors = defaultdict(int)
    last_close = {}
    for c, (ev, lc, err) in zip(codes, res):
        all_events.extend(ev)
        last_close[c] = lc
        if err:
            errors[err] += 1
    print('[bt] events', len(all_events), 'errors', dict(errors), flush=True)
    caps = [int(x) for x in str(a.max_positions).split(',') if x.strip()]
    stats = None
    for cap in (caps or [0]):
        stats = simulate(all_events, last_close=last_close,
                         max_positions=cap)
        print('[bt] stats', stats, flush=True)
    return stats


if __name__ == '__main__':
    main()
