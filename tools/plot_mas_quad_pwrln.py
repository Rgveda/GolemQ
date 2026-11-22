# coding: utf-8
"""全市场 `4QPwrln`（日线）→ `MasQuadPwrlnMajMed` / `MasQuadPwrlnMajMode` + PNG。

口径：每个码用日线 OHLC 调 `calc_four_quadrant_boost(annual=252)` 得 `4QPwrln`；
按日取横截面中位数 / 众数；再与 massive 表的 `MasPosOnhold` 对比。

用法::

    python tools/plot_mas_quad_pwrln.py --start 2024-06-01 --end 2026-11-20
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

from GolemQ.models.alias import LTT  # noqa: E402


def _one(code: str, start: str, end: str):
    from GolemQ import get_active_market
    from GolemQ.analysis.quadrant_jit import calc_four_quadrant_boost
    m = get_active_market()
    r, _ = m.get_kline_price_v3(code, start='2016-01-01', end=end,
                                realtime=False, verbose=False)
    day = None if r is None else getattr(r, 'data', None)
    if day is None or len(day) < 60:
        return None
    try:
        f = calc_four_quadrant_boost(
            code, features=pd.DataFrame(index=day.index), ohlc_data=day,
            annual=252)
    except Exception:               # noqa: BLE001
        return None
    if LTT.QUADRANT_POWERLINE not in f.columns:
        return None
    s = pd.Series(np.asarray(f[LTT.QUADRANT_POWERLINE], dtype=float),
                  index=pd.DatetimeIndex(
                      pd.to_datetime(day.index.get_level_values(0))).normalize())
    s = s[~s.index.duplicated(keep='last')]
    s = s[(s.index >= pd.Timestamp(start)) & (s.index <= pd.Timestamp(end))]
    if s.empty:
        return None
    return pd.DataFrame({'date': s.index, 'code': str(code),
                         'pwrln': s.to_numpy()})


def compute(start: str, end: str, workers: int = 16):
    from joblib import Parallel, delayed
    from GolemQ import get_active_market
    ds = get_active_market().scan_dataset()
    codes = sorted({str(c) for c in ds.stock_list()['code']})
    out = Parallel(n_jobs=workers, backend='loky')(
        delayed(_one)(c, start, end) for c in codes)
    parts = [x for x in out if x is not None and len(x)]
    return pd.concat(parts, ignore_index=True)


def aggregate(long: pd.DataFrame):
    g = long.groupby('date')['pwrln']
    med = g.median().rename('MasQuadPwrlnMajMed')
    mode = g.agg(lambda s: s.mode().iloc[0] if len(s.mode()) else np.nan
                 ).rename('MasQuadPwrlnMajMode')
    return pd.concat([med, mode], axis=1).sort_index()


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--start', default='2024-06-01')
    p.add_argument('--end', default='2026-11-20')
    p.add_argument('--workers', type=int, default=16)
    p.add_argument('--out', default='output/mas_quad_pwrln')
    p.add_argument('--reuse', action='store_true',
                   help='复用已存在的 CSV，不重算全市场 4QPwrln')
    a = p.parse_args(argv)
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)

    if a.reuse and os.path.exists(a.out + '.csv'):
        agg = pd.read_csv(a.out + '.csv', index_col=0, parse_dates=True)
        print('[quad] reuse', a.out + '.csv', 'rows', len(agg))
    else:
        long = compute(a.start, a.end, a.workers)
        print('[quad] long rows', len(long), 'codes', long['code'].nunique())
        agg = aggregate(long)
        agg.to_csv(a.out + '.csv', encoding='utf-8')
        print(agg.tail(5).to_string())

    from GolemQ import get_active_market
    db = get_active_market().get_database()
    _fields = ('datetime', 'MasPosOnhold', 'PosFull', 'PosFullOnset',
               'PosReClear', 'PosReClearStrict', 'PosBottom',
               'PosDeepBottom', 'STAGE_MODE')
    rows = list(db.stock_massive_reality_60min_models.find(
        {'datetime': {'$regex': ' 15:00:00$',
                      '$gte': a.start, '$lte': a.end}},
        {c: 1 for c in _fields} | {'_id': 0}
    ).sort('datetime', 1))
    ev = pd.DataFrame(rows)
    ev['date'] = pd.to_datetime(ev['datetime']).dt.normalize()
    ev = ev.drop_duplicates('date').set_index('date').sort_index()
    pos = pd.to_numeric(ev['MasPosOnhold'], errors='coerce')
    panel = agg.join(pos.rename('MasPosOnhold'), how='left')

    # 稳定性：平均绝对日变化 / 标准差
    stats = {}
    for c in ('MasQuadPwrlnMajMed', 'MasQuadPwrlnMajMode', 'MasPosOnhold'):
        s = panel[c].dropna()
        stats[c] = (float(s.diff().abs().mean()), float(s.std()),
                    len(s))
    print('[stability] mean|diff| / std / n:')
    for k, v in stats.items():
        print(' ', k, v)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 9), sharex=True,
        gridspec_kw={'height_ratios': [2, 1]})
    ax1.plot(panel.index, panel['MasQuadPwrlnMajMed'],
             label='MasQuadPwrlnMajMed', lw=1.2)
    ax1.plot(panel.index, panel['MasQuadPwrlnMajMode'],
             label='MasQuadPwrlnMajMode', lw=1.0, alpha=0.8)
    ax1.set_title('Market quadrant powerline (daily cross-section), '
                  '{} ~ {}'.format(a.start, a.end))
    ax1.legend(loc='upper left')
    ax1.grid(alpha=0.3)
    ax2.plot(panel.index, panel['MasPosOnhold'], color='tab:red',
             label='MasPosOnhold', lw=1.0)
    _ev = ev.reindex(panel.index)
    _pos = panel['MasPosOnhold']

    def _mark(mask, y, **kw):
        idx = panel.index[mask.to_numpy()]
        if not len(idx):
            return
        yy = _pos.loc[idx] if isinstance(y, pd.Series) else np.full(len(idx), y)
        ax2.scatter(idx, yy, zorder=5, **kw)

    _mark(_ev['PosFull'] == 1, _pos, marker='.', s=14, color='lime',
          alpha=0.5, label='PosFull')
    _mark(_ev['PosFullOnset'] == 1, _pos, marker='^', s=55, color='green',
          label='PosFullOnset')
    _mark(_ev['PosReClear'] == 1, 0.0, marker='v', s=55, color='red',
          label='PosReClear')
    _mark(_ev['PosReClearStrict'] == 1, 0.0, marker='x', s=45,
          color='darkred', label='PosReClearStrict')
    _mark(_ev['PosBottom'] == 1, 0.0, marker='o', s=35, color='orange',
          label='PosBottom')
    _mark(_ev['PosDeepBottom'] == 1, 0.0, marker='*', s=110, color='purple',
          label='PosDeepBottom')
    ax2.legend(loc='upper left', ncol=2, fontsize=8)
    ax2.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    png = a.out + '.png'
    fig.savefig(png, dpi=140)
    print('[quad] wrote', png)
    return 0


if __name__ == '__main__':
    sys.exit(main())
