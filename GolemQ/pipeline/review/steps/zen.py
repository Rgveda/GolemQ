# coding: utf-8
"""60min 快照：zen60 + 15min minor/zen（**按 60min 目标对齐**）。

用户 2026-10-20 定的计算维度：

* 外层只遍历 **60min 目标**（约 2,160 个），不再按 8,432 个 15min 目标算；
* 每个目标 T：
  - 60min 取最后 **2,160** 根 → `zDLagR/zPLagR`；
  - 15min 取最后 **2,160** 根（截止 T，取与 T 对齐的那根，即每小时第 4 根）
    → 一次算完 17 个 minor + 2 个 15min zen；
* 15min 族由 `minute15_aligned` 从本 step 的输出取样，不重算。

`minor_signals` 原来独立跑一趟；现在合并进这里的 15min 窗口，避免重复。
"""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from GolemQ.analysis.zen import calc_zen_timing_lag
from GolemQ.core.constants import FIELD as FLD
from GolemQ.pipeline.review.contract import (HOUR_EXTRA_COLUMNS,
                                             HOUR_ZEN_COLUMNS,
                                             MINOR_15MIN_COLUMNS,
                                             MINOR_COLUMNS)
from GolemQ.pipeline.review.steps.hour_extra import (_pivot_counts,
                                                     _sample_entropy_by_day)
from GolemQ.pipeline.review.steps.minor import minor_features
from GolemQ.pipeline.steps import (FeatureStep, ReviewContext, TIER_DAILY,
                                   slice_asof)

#: 用户 2026-10-20 定：60min 窗口 2160；15min minor 窗口 **2160**。
WINDOW60 = 2160
WINDOW15 = 2160

#: 本 step 的 60min 输出：4 个 zen + 17 个 minor + 2 个 hour_extra。
ZEN_COLUMNS = HOUR_ZEN_COLUMNS + MINOR_COLUMNS + HOUR_EXTRA_COLUMNS
_NAN_MINOR = np.full(len(MINOR_COLUMNS), np.nan)


def _zen_last(ohlc: pd.DataFrame):
    if ohlc is None or len(ohlc) == 0:
        return np.nan, np.nan
    feats = calc_zen_timing_lag(ohlc)
    return (feats[FLD.ZEN_DASH_TIMING_LAG].iloc[-1],
            feats[FLD.ZEN_PEAK_TIMING_LAG].iloc[-1])


def zen_snapshots_step(ctx: ReviewContext, targets: pd.DatetimeIndex) -> pd.DataFrame:
    """每个 60min 目标：zen60 + 15min 窗口里算出的 minor 17 + zen15 2。"""
    if targets is None or len(targets) == 0:
        return pd.DataFrame(columns=ZEN_COLUMNS)
    rows, idx = [], []
    minute_rows = {}
    src60 = ctx.frame('60min')
    src15 = ctx.frame('15min')
    # `hour_extra` 也并进这一趟：SE 按 15:00 预计算，pivot 计数一次算到 targets。
    ohlc60 = slice_asof(src60, ctx.as_of)
    se_days, se_vals = _sample_entropy_by_day(ohlc60)
    piv_counts = _pivot_counts(ohlc60, targets, ctx.code)
    for _i, T in enumerate(targets):
        T = pd.Timestamp(T)
        d60, p60 = _zen_last(slice_asof(src60, T, WINDOW60))
        w15 = slice_asof(src15, T, WINDOW15)
        m = minor_features(w15, ctx.code, market_type=ctx.market_type)
        d15 = p15 = np.nan
        minor_vals = _NAN_MINOR
        if m is not None and len(m):
            level0 = pd.DatetimeIndex(pd.to_datetime(m.index.get_level_values(0)))
            # 15min 表写 15min：取该 60min 时刻所在小时的 4 根（含 T）
            last_hour = m.loc[(level0 > T - pd.Timedelta(minutes=60)) & (level0 <= T)]
            for row_idx, row in last_hour.iterrows():
                key = row_idx[0] if isinstance(row_idx, tuple) else row_idx
                minute_rows[pd.Timestamp(key)] = row
            sub = m.loc[level0 <= T]
            if len(sub):
                r = sub.iloc[-1]
                d15 = r.get('zDLagMinR', np.nan)
                p15 = r.get('zPLagMinR', np.nan)
                minor_vals = np.array([r.get(c, np.nan) for c in MINOR_COLUMNS])
        se = np.nan
        if len(se_days):
            pos = int(se_days.searchsorted(T, side='right')) - 1
            if pos >= 0:
                se = se_vals[pos]
        rows.append([d60, p60, d15, p15, *minor_vals, se, piv_counts[_i]])
        idx.append((T, ctx.code))

    # 15min 族：把每个 60min 目标所在小时的 4 根 15min 结果缓存，供
    # `minute15_aligned` 直接取样（不重算）。
    if minute_rows:
        m_idx = pd.MultiIndex.from_tuples(
            [(t, ctx.code) for t in sorted(minute_rows)], names=['ts', 'code'])
        ctx.outputs['_minute15_rows'] = pd.DataFrame(
            [minute_rows[t] for t in sorted(minute_rows)], index=m_idx,
            columns=list(MINOR_15MIN_COLUMNS))
    return pd.DataFrame(
        rows,
        index=pd.MultiIndex.from_tuples(idx, names=['ts', 'code']),
        columns=list(ZEN_COLUMNS),
    )


def build_zen_steps() -> List[FeatureStep]:
    return [FeatureStep(
        name='zen_snapshots', tier=TIER_DAILY, input_freq='60min',
        output_family='60min', columns=ZEN_COLUMNS,
        func=zen_snapshots_step, history='bars', warmup=max(WINDOW60, WINDOW15),
    )]
