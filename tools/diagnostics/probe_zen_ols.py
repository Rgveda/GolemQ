# coding: utf-8
"""**探针**：`ZenOLSLag` 的峰点口径对比（scipy 相邻极值 vs CZSC 笔端点）。

只读、不落库。用 000711 / 600519 / 603366 三个样板，比较：

* 段数 / 段长（来自 `ZenPivotLag` 符号）
* 交叉标记数（+1 / -1）
* `ZenOLSLag` 的符号翻转次数、正负占比
* **翻转点**与「段边界」的距离（好的信号应该贴着段边界或拐点，而不是乱翻）

用法::

    python tools/diagnostics/probe_zen_ols.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from GolemQ.analysis.zen_ols import (calc_zen_ols_timing_lag, cross_marks,
                                     czsc_peak_markers, find_peaks_scipy,
                                     segment_ids)

CODES = ('000711', '600519', '603366')
START, END = '2024-06-01', '2026-11-20 23:59:59'


def _frame(code: str) -> pd.DataFrame:
    from GolemQ import get_active_market
    m = get_active_market()
    r, _ = m.get_kline_price_min(code, start='2022-01-01', end=END,
                                 frequency='60min', realtime=False, verbose=False)
    df = None if r is None else getattr(r, 'data', None)
    if df is None:
        return pd.DataFrame()
    df = df.copy()
    df.index = pd.MultiIndex.from_arrays(
        [pd.to_datetime(df.index.get_level_values(0)), [code] * len(df)],
        names=['ts', 'code'])
    return df.loc[df.index.get_level_values(0) >= pd.Timestamp(START)]


def _pivot_lag(code: str, idx) -> pd.Series:
    from GolemQ import get_active_market
    db = get_active_market().get_database()
    rows = list(db.stock_reality_feat_60min.find(
        {'code': code, 'datetime': {'$gte': START, '$lte': END}},
        {'_id': 0, 'datetime': 1, 'ZenPivotLag': 1}).sort('datetime', 1))
    if not rows:
        return pd.Series(np.nan, index=idx)
    s = pd.Series([r.get('ZenPivotLag') for r in rows],
                  index=pd.DatetimeIndex(pd.to_datetime([r['datetime'] for r in rows])))
    return s.reindex(pd.DatetimeIndex(idx.get_level_values(0)))


def _summarise(tag, lag: pd.Series, seg: np.ndarray, cross_n, bounds_n) -> str:
    v = pd.to_numeric(lag, errors='coerce')
    sign = np.sign(v.fillna(0).to_numpy())
    flip_idx = np.where(np.diff(sign) != 0)[0] + 1
    # 段边界位置
    bnd = np.where(np.diff(seg) != 0)[0] + 1
    if len(flip_idx) and len(bnd):
        dist = np.min(np.abs(flip_idx[:, None] - bnd[None, :]), axis=1)
        near = float(np.median(dist))
    else:
        near = float('nan')
    return (f'    {tag:<14s} 交叉 {cross_n:4d}  lag 正 {100*(sign>0).mean():5.1f}% '
            f'翻转 {len(flip_idx):4d}  距段边界中位 {near:6.1f} 根  '
            f'|lag|中位 {np.median(np.abs(v.dropna())):6.0f}')


def main() -> int:
    for code in CODES:
        df = _frame(code)
        if len(df) < 200:
            print(f'{code}: 数据不足 {len(df)}')
            continue
        pl = _pivot_lag(code, df.index)
        seg = segment_ids(pl)
        nb = int(len(np.unique(seg)))
        print(f'\n=== {code}: {len(df)} 根 60min（{START}~{END}）段 {nb} '
              f'pivotLag 非空 {100*pl.notna().mean():.1f}% ===')
        for peak in ('scipy', 'czsc'):
            for causal in (True, False):
              for in_piv in (True, False):
                lag = calc_zen_ols_timing_lag(df, pivot_lag=pl, peak=peak,
                                              causal=causal,
                                              in_pivot_marks=in_piv)
                cm = cross_marks(df, pivot_lag=pl, peak=peak, causal=causal,
                                 in_pivot_marks=in_piv)
                n_up, n_dn = int((cm == 1).sum()), int((cm == -1).sum())
                tag = f'{peak}/{"因果" if causal else "事后"}/{"含中枢" if in_piv else "仅中枢外"}'
                print(_summarise(tag, lag, seg, n_up + n_dn, nb))
    return 0


if __name__ == '__main__':
    sys.exit(main())
