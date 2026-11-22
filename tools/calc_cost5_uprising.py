# coding: utf-8
"""算样板码的 `COST5_UPRISING_BEFORE`（D85 L2 化简版），列出 `<1e-12` 时间戳。

口径：
* `RETURN_COST5`：60min ChipDist，**逐日滚动 1,920 根**窗口（换手率按日映射）；
* `MACD` / `macdMaj`：`calc_compact_features`；
* `stage`：8.3 `stock_reality_feat_60min` 落库值；
* `cost5UprBf` = `analysis/cost5_uprising.py::calc_cost5_uprising_before`。

用法::

    python tools/calc_cost5_uprising.py --codes 600519,000711,603366 \
        --start 2023-04-06 --end 2026-11-20
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

from GolemQ.analysis.ChipDistribution_jit import (  # noqa: E402
    calc_stock_chip_distribution_jit)
from GolemQ.analysis.compact import calc_compact_features  # noqa: E402
from GolemQ.analysis.cost5_uprising import (  # noqa: E402
    calc_cost5_uprising_before)
from GolemQ.core.constants import AKA, FIELD as FLD  # noqa: E402
from GolemQ.markets.StockCN.refdata import (  # noqa: E402
    GQ_fetch_stock_reality_feat_day)


def _frames(code: str, start: str, end: str):
    from GolemQ import get_active_market
    m = get_active_market()
    # 1920 根 60min ≈ 480 交易日；多取两年保证窗口头部
    hist = str((pd.Timestamp(start) - pd.Timedelta(days=900)).date())
    r60, _ = m.get_kline_price_min(
        code, start=hist, end=end + ' 23:59:59', frequency='60min',
        realtime=False)
    rd, _ = m.get_kline_price_v3(code, start=hist, end=end, realtime=False)
    d = r60.data
    d = d[~d.index.duplicated(keep='last')].sort_index()
    tr = GQ_fetch_stock_reality_feat_day(
        code, start=hist, end=end, columns=[AKA.TURNOVER])
    turnover = pd.Series(
        tr[AKA.TURNOVER].to_numpy(),
        index=pd.DatetimeIndex(
            pd.to_datetime(tr.index.get_level_values(0))).normalize())
    turnover = turnover[~turnover.index.duplicated(keep='last')]
    return m, d, rd.data, turnover


def _return_cost5(d: pd.DataFrame, turnover: pd.Series, code: str,
                  start: str, end: str, window: int = 1920,
                  verbose: bool = False) -> pd.Series:
    ts = pd.DatetimeIndex(pd.to_datetime(d.index.get_level_values(0)))
    days = [t for t in sorted(set(ts.normalize()))
            if pd.Timestamp(start) <= t <= pd.Timestamp(end)]
    parts = []
    t0 = time.time()
    for i, D in enumerate(days, 1):
        w = d[ts <= D + pd.Timedelta(hours=15)].tail(window).copy()
        wt = pd.DatetimeIndex(pd.to_datetime(w.index.get_level_values(0)))
        w[FLD.TURNOVER_RATE] = turnover.reindex(wt.normalize()).to_numpy()
        w = w[w[FLD.TURNOVER_RATE].notna()]
        if len(w) < 30:
            continue
        try:
            out = calc_stock_chip_distribution_jit(
                features=pd.DataFrame(index=w.index), ohlc_data=w, code=code)
            day_mask = (pd.DatetimeIndex(pd.to_datetime(
                out.index.get_level_values(0))).normalize() == D)
            if day_mask.any():
                parts.append(out.loc[day_mask, AKA.RETURN_COST5])
        except Exception as exc:            # noqa: BLE001
            if verbose:
                print(f'  {code} {D.date()} skip: {exc}')
        if verbose and i % 100 == 0:
            print(f'  {code} {i}/{len(days)}  {time.time() - t0:.0f}s')
    return pd.concat(parts) if parts else pd.Series(dtype=float)


def compute(code: str, start: str, end: str, window: int = 1920,
            verbose: bool = False) -> pd.DataFrame:
    m, d, day, turnover = _frames(code, start, end)
    ret5 = _return_cost5(d, turnover, code, start, end, window, verbose)
    f = calc_compact_features(d, day)
    f[AKA.RETURN_COST5] = ret5.reindex(f.index)
    rows = list(m.get_database()['stock_reality_feat_60min'].find(
        {'code': code, 'datetime': {'$gte': start, '$lte': end + ' 23:59:59'},
         'stage': {'$ne': None}}, {'datetime': 1, 'stage': 1, '_id': 0}))
    st = pd.Series({pd.Timestamp(r['datetime']): r['stage'] for r in rows})
    f[AKA.STAGE] = st.reindex(pd.DatetimeIndex(
        pd.to_datetime(f.index.get_level_values(0)))).to_numpy()
    out = calc_cost5_uprising_before(f)
    return out[out == 0]


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--codes', required=True)
    p.add_argument('--start', default='2023-04-06')
    p.add_argument('--end', default='2026-11-20')
    p.add_argument('--window', type=int, default=1920)
    p.add_argument('--verbose', action='store_true')
    a = p.parse_args(argv)
    for code in [c.strip() for c in a.codes.split(',') if c.strip()]:
        t0 = time.time()
        z = compute(code, a.start, a.end, a.window, a.verbose)
        print(f'== {code}  {a.start} ~ {a.end}  zeros={len(z)}  '
              f'{time.time() - t0:.0f}s')
        for t in z.index.get_level_values(0):
            print('  ', pd.Timestamp(t))
    return 0


if __name__ == '__main__':
    sys.exit(main())
