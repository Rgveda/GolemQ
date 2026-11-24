# coding: utf-8
"""MasQuad `4QPwrln` **日线(Maj) + 周线(Wek)** → Med/Mode + CSV + PNG。

本工具是 **Maj/Wek 统一入口**（2026-11-23 起）：

* **Maj**：逐码日线 OHLC → `calc_four_quadrant_boost(annual=252)` → 按日横截面
  中位数 / 众数（`MasQuadPwrlnMajMed/Mode`）；
* **Wek**：直接调 ``tools/backfill_quad_pwrln_wek.py::compute_cross_section``
  （周线口径的唯一实现，含 `--min-weeks 233` 成熟度门槛）；
* 两者 + massive 的 ``MasPosOnhold`` **输出到同一份 CSV**（默认
  ``output/mas_quad_pwrln_wek_2018_2026.csv``），PNG 走 ``mas_quad_pwrln.png``。

``--apply`` 把两条横截面序列写进 ``stock_massive_reality_60min_models`` 的
**15:00 行**（只写已存在的行，不插稀疏行）—— 与周线工具的 `--apply` 同一口径。
口径一致性：massive 的 ``pwrln_major``（`run_market` 里算的）就是日线
`4QPwrln` 按日映射到当天 4 根 60min（`feature_flow.pwrln_major`），所以 15:00
行的横截面与这里逐值同源；Wek 则直接复用周线工具的实现。

用法::

    python tools/plot_mas_quad_pwrln.py --start 2018-01-01 --end 2026-11-20 \
        --apply --workers 16
    python tools/plot_mas_quad_pwrln.py --reuse    # 复用已有 CSV，只重画/重写
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


def _one(code: str, start: str, end: str, min_days: int = 0):
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
    if min_days and len(s):
        # `4QPwrln` 最长窗口是 **MA233**：该码攒够 233 根日线之前 MA233 恒 NaN
        # ⇒ 该项恒 0、powerline 结构性偏低。与周线的 `--min-weeks 233` 同一理由
        # （D86），不做门槛会让早年 IPO 把截面中位数往 1~2 档下拖。
        seen = np.arange(1, len(s) + 1)
        s = s[np.asarray(seen >= int(min_days))]
    s = s[(s.index >= pd.Timestamp(start)) & (s.index <= pd.Timestamp(end))]
    if s.empty:
        return None
    return pd.DataFrame({'date': s.index, 'code': str(code),
                         'pwrln': s.to_numpy()})


def compute(start: str, end: str, workers: int = 16, min_days: int = 0):
    from joblib import Parallel, delayed
    from GolemQ import get_active_market
    ds = get_active_market().scan_dataset()
    codes = sorted({str(c) for c in ds.stock_list()['code']})
    out = Parallel(n_jobs=workers, backend='loky')(
        delayed(_one)(c, start, end, min_days) for c in codes)
    parts = [x for x in out if x is not None and len(x)]
    return pd.concat(parts, ignore_index=True)


def aggregate(long: pd.DataFrame):
    g = long.groupby('date')['pwrln']
    med = g.median().rename('MasQuadPwrlnMajMed')
    mode = g.agg(lambda s: s.mode().iloc[0] if len(s.mode()) else np.nan
                 ).rename('MasQuadPwrlnMajMode')
    return pd.concat([med, mode], axis=1).sort_index()


#: 横截面列（Wek 来自周线工具，Maj 来自本工具）；`--apply` 只写这些。
MAS_QUAD_COLUMNS = ('MasQuadPwrlnWekMed', 'MasQuadPwrlnWekMode',
                    'MasQuadPwrlnMajMed', 'MasQuadPwrlnMajMode')


def apply_to_massive(agg: pd.DataFrame, start: str, end: str, db=None) -> dict:
    """横截面序列 → massive 的 **15:00 行**（只 upsert 已存在的行）。

    ⚠️ 不插稀疏行：massive 的 15:00 行本身有缺（2022-06-16 之前没有行），
    凭空插入会造出只有两列的「假 bar」（与 ``backfill_quad_pwrln_wek.py``
    的同一约定）。
    """
    from GolemQ.datasource.writer import upsert_fields
    if db is None:
        from GolemQ import get_active_market
        db = get_active_market().get_database()
    cols = [c for c in MAS_QUAD_COLUMNS if c in agg.columns]
    rows = list(db.stock_massive_reality_60min_models.find(
        {'datetime': {'$regex': ' 15:00:00$',
                      '$gte': start + ' 00:00:00',
                      '$lte': end + ' 23:59:59'}},
        {'_id': 0, 'datetime': 1, 'time_stamp': 1}).sort('datetime', 1))
    stamp_by_dt = {pd.Timestamp(r['datetime']): int(r['time_stamp'])
                   for r in rows if r.get('time_stamp') is not None}
    docs = []
    for d, row in agg.iterrows():
        dt = pd.Timestamp(d).normalize() + pd.Timedelta(hours=15)
        if dt not in stamp_by_dt:
            continue
        doc = {'datetime': dt.strftime('%Y-%m-%d %H:%M:%S'),
               'time_stamp': stamp_by_dt[dt]}
        for c in cols:
            v = row.get(c)
            doc[c] = None if v is None or not np.isfinite(v) else float(v)
        docs.append(doc)
    res = upsert_fields(db.stock_massive_reality_60min_models, docs,
                        ['time_stamp'])
    print('[quad] massive upserted', res['upserted'], 'modified',
          res['modified'], 'docs', len(docs), 'cols', cols)
    return res


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--start', default='2018-01-01')
    p.add_argument('--end', default='2026-11-20')
    p.add_argument('--workers', type=int, default=16)
    p.add_argument('--out',
                   default='output/mas_quad_pwrln_wek_2018_2026.csv',
                   help='统一 CSV（Wek + Maj + MasPosOnhold）')
    p.add_argument('--png', default='output/mas_quad_pwrln.png')
    p.add_argument('--min-days', type=int, default=0,
                   help='日线截面成熟度门槛（0=关，与 mainstream 的 Maj 口径一致；'
                        '设 233 会剔除上市不足 233 日的码 —— 但 mainstream 不做'
                        '这层过滤，两边会分叉）')
    p.add_argument('--min-weeks', type=int, default=233,
                   help='周线截面成熟度门槛（D86；233 才是完整 MA233）')
    p.add_argument('--reuse', action='store_true',
                   help='复用已存在的统一 CSV，不重算（仍会重画/重写库）')
    p.add_argument('--apply', action='store_true',
                   help='把 Wek/Maj 四条序列写进 massive 的 15:00 行')
    a = p.parse_args(argv)
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    os.makedirs(os.path.dirname(a.png) or '.', exist_ok=True)

    if a.reuse and os.path.exists(a.out):
        agg = pd.read_csv(a.out, index_col=0, parse_dates=True)
        print('[quad] reuse', a.out, 'rows', len(agg))
    else:
        # Maj：日线 4QPwrln 横截面
        long = compute(a.start, a.end, a.workers, min_days=a.min_days)
        print('[quad] Maj long rows', len(long),
              'codes', long['code'].nunique())
        agg = aggregate(long)
        # Wek：周线 —— **复用**周线工具的实现，不平行重写（D86 的三个口径）
        if os.path.join(_REPO, 'tools') not in sys.path:
            sys.path.insert(0, os.path.join(_REPO, 'tools'))
        from backfill_quad_pwrln_wek import compute_cross_section
        wek = compute_cross_section(a.start, a.end, workers=a.workers,
                                    min_weeks=a.min_weeks)
        agg = agg.join(wek[['MasQuadPwrlnWekMed', 'MasQuadPwrlnWekMode',
                            'n']], how='outer').sort_index()
        agg.index.name = 'date'
        print(agg.tail(5).to_string())

    if a.apply:
        apply_to_massive(agg, a.start, a.end)

    from GolemQ import get_active_market
    db = get_active_market().get_database()
    _fields = ('datetime', 'MasPosOnhold', 'PosFull', 'PosFullOnset',
               'PosReClear', 'PosReClearStrict', 'PosBottom',
               'PosDeepBottom', 'STAGE_MODE')
    rows = list(db.stock_massive_reality_60min_models.find(
        # ⚠️ 右界要带时分秒：`$lte: '2026-11-23'` 比 '2026-11-23 15:00:00' 小，
        # 会把区间**最后一天**的 15:00 行排除掉（MasPosOnhold 因此为空）。
        {'datetime': {'$regex': ' 15:00:00$',
                      '$gte': a.start, '$lte': a.end + ' 23:59:59'}},
        {c: 1 for c in _fields} | {'_id': 0}
    ).sort('datetime', 1))
    ev = pd.DataFrame(rows)
    if ev.empty:                      # 区间里 massive 一行都没有（如 2018~2022-06）
        ev = pd.DataFrame(columns=list(_fields))
    ev['date'] = pd.to_datetime(ev['datetime']).dt.normalize()
    ev = ev.drop_duplicates('date').set_index('date').sort_index()
    pos = pd.to_numeric(ev['MasPosOnhold'], errors='coerce')
    # `--reuse` 读回来的 CSV 里已经有 `MasPosOnhold` —— 先丢掉再按库重挂，
    # 否则 `join` 会报 "columns overlap but no suffix specified"。
    agg = agg.drop(columns=['MasPosOnhold'], errors='ignore')
    panel = agg.join(pos.rename('MasPosOnhold'), how='left')
    panel.index.name = 'date'
    # 统一 CSV：Wek → Maj → n → MasPosOnhold
    order = [c for c in ('MasQuadPwrlnWekMed', 'MasQuadPwrlnWekMode',
                         'MasQuadPwrlnMajMed', 'MasQuadPwrlnMajMode', 'n',
                         'MasPosOnhold') if c in panel.columns]
    panel.loc[:, order].to_csv(a.out, encoding='utf-8')
    print('[quad] wrote', a.out, 'rows', len(panel), 'cols', order)

    # 稳定性：平均绝对日变化 / 标准差
    stats = {}
    for c in order:
        s = pd.to_numeric(panel[c], errors='coerce').dropna()
        if len(s):
            stats[c] = (float(s.diff().abs().mean()), float(s.std()), len(s))
    print('[stability] mean|diff| / std / n:')
    for k, v in stats.items():
        print(' ', k, v)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 9), sharex=True,
        gridspec_kw={'height_ratios': [2, 1]})
    _lines = (('MasQuadPwrlnWekMed', 'tab:blue', '-'),
              ('MasQuadPwrlnWekMode', 'tab:cyan', '--'),
              ('MasQuadPwrlnMajMed', 'tab:purple', '-'),
              ('MasQuadPwrlnMajMode', 'tab:orange', '--'))
    for c, color, ls in _lines:
        if c not in panel.columns:
            continue
        ax1.plot(panel.index, pd.to_numeric(panel[c], errors='coerce'),
                 label=c, lw=1.1, ls=ls, color=color)
    ax1.set_title('MasQuad 4QPwrln cross-section: Wek (weekly) + Maj (daily), '
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
    fig.savefig(a.png, dpi=140)
    print('[quad] wrote', a.png)
    return 0


if __name__ == '__main__':
    sys.exit(main())
