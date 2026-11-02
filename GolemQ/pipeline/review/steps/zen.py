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
from GolemQ.core.constants import AKA
from GolemQ.core.constants import FIELD as FLD
from GolemQ.pipeline.review.contract import (HOUR_EXTRA_COLUMNS,
                                             HOUR_NATIVE_COLUMNS,
                                             HOUR_ZEN_COLUMNS,
                                             MINOR_15MIN_COLUMNS,
                                             MINOR_COLUMNS,
                                             RENKO_DAY_COLUMNS,
                                             RENKO_PRICE_DUMMIES)
from GolemQ.pipeline.review.steps.hour_extra import (native_by_day,
                                                     pivot_stats,
                                                     renko_by_day,
                                                     sample_entropy_by_day)
from GolemQ.pipeline.review.steps.minor import minor_features
from GolemQ.pipeline.steps import (FeatureStep, ReviewContext, TIER_DAILY,
                                   slice_asof)

#: 用户 2026-10-20 定：60min 窗口 2160；15min minor 窗口 **2160**。
WINDOW60 = 2160
WINDOW15 = 2160

#: 本 step 的 60min 输出：zen 4 + minor 17 + se/ZenPivotCnt 2
#: + **Renko 14（60min 原生）** + **M9T 2（60min 原生）**（D78 一体化）。
ZEN_COLUMNS = (HOUR_ZEN_COLUMNS + MINOR_COLUMNS + HOUR_EXTRA_COLUMNS +
               RENKO_DAY_COLUMNS + HOUR_NATIVE_COLUMNS)
_NAN_MINOR = np.full(len(MINOR_COLUMNS), np.nan)
_RENKO_BASE_OF = {dummy: base for base, dummy in RENKO_PRICE_DUMMIES}
_NAN_RENKO = np.full(len(RENKO_DAY_COLUMNS), np.nan)
_NAN_NATIVE = np.full(len(HOUR_NATIVE_COLUMNS), np.nan)


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
    # `hour_extra` 并进这一趟（D78）：SE 只算 15:00，pivot 计数一次算到 targets。
    ohlc60 = slice_asof(src60, ctx.as_of)
    se_days, se_vals = sample_entropy_by_day(ohlc60, targets)
    piv_days, piv_df = pivot_stats(ohlc60, targets, ctx.code)
    # Renko 14 + M9T 2：60min 原生，2160 根 as-of 窗口（与 se 同口径）
    rk_days, rk_df = renko_by_day(ohlc60, targets)
    nat_days, nat_df = native_by_day(ohlc60, targets)
    close_by_ts = pd.Series(
        pd.to_numeric(ohlc60[AKA.CLOSE], errors='coerce').to_numpy(),
        index=pd.DatetimeIndex(pd.to_datetime(
            ohlc60.index.get_level_values(0)))) if len(ohlc60) else pd.Series(dtype=float)
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
        piv_cnt = piv_lag = np.nan
        if len(piv_days):
            pos = int(piv_days.searchsorted(T, side='right')) - 1
            if pos >= 0:
                piv_cnt = piv_df.iloc[pos, 0]
                piv_lag = piv_df.iloc[pos, 1]
        # Renko：快照里的价格列按**本目标行** close 转 *D（restore 时乘回一致）
        rk_vals = _NAN_RENKO
        if len(rk_days):
            p = int(rk_days.searchsorted(T, side='right')) - 1
            if p >= 0:
                snap = rk_df.iloc[p]
                cT = close_by_ts.get(T, np.nan)
                vals = []
                for col in RENKO_DAY_COLUMNS:
                    v = snap.get(col, np.nan)
                    if col in _RENKO_BASE_OF:
                        # 快照里 `*D` 列放的是绝对价，这里转比值
                        vals.append(v / cT if (cT and np.isfinite(cT)) else np.nan)
                    else:
                        vals.append(v)
                rk_vals = np.array(vals, dtype=float)
        nat_vals = _NAN_NATIVE
        if len(nat_days):
            p = int(nat_days.searchsorted(T, side='right')) - 1
            if p >= 0:
                nat_vals = nat_df.iloc[p].to_numpy(dtype=float)
        rows.append([d60, p60, d15, p15, piv_lag, *minor_vals, se, piv_cnt,
                     *rk_vals, *nat_vals])
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
