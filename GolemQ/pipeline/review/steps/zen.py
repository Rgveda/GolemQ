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
                                             HOUR_NATIVE_COLUMNS,
                                             HOUR_ZEN_COLUMNS,
                                             MINOR_15MIN_COLUMNS,
                                             MINOR_COLUMNS,
                                             RENKO_DAY_COLUMNS)
from GolemQ.pipeline.review.steps.hour_extra import build_hour_extra_frame
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


def _zen_last(ohlc: pd.DataFrame):
    if ohlc is None or len(ohlc) == 0:
        return np.nan, np.nan
    feats = calc_zen_timing_lag(ohlc)
    return (feats[FLD.ZEN_DASH_TIMING_LAG].iloc[-1],
            feats[FLD.ZEN_PEAK_TIMING_LAG].iloc[-1])


def _hour_targets(ctx: ReviewContext, targets) -> pd.DatetimeIndex:
    """60min 目标 + 15min 目标各自**所在整点**的 60min bar（F4）。

    15min 行由本 step 的 60min 目标循环顺带产出（每小时第 4 根）。60min 已全部
    落库、只剩 15min 缺行时，`ctx.targets['60min']` 为空 ⇒ 以前整趟被跳过，
    15min 永远补不上。这里把 15min 目标映射回"第一个 ≥ 它的 60min bar"
    （10:15 → 10:30、13:15 → 14:00），多出来的 60min 行**写库前**会被
    `pipeline._restrict_to_targets` 按 60min 目标裁掉。
    """
    base = (pd.DatetimeIndex(sorted(set(pd.to_datetime(targets))))
            if targets is not None and len(targets) else pd.DatetimeIndex([]))
    t15 = ctx.targets.get('15min')
    src60 = ctx.frame('60min')
    if t15 is None or len(t15) == 0 or src60 is None or len(src60) == 0:
        return base
    bars = pd.DatetimeIndex(
        pd.to_datetime(src60.index.get_level_values(0))).unique().sort_values()
    pos = bars.searchsorted(pd.DatetimeIndex(pd.to_datetime(t15)), side='left')
    pos = pos[pos < len(bars)]
    if not len(pos):
        return base
    return pd.DatetimeIndex(sorted(set(base) | set(bars[pos])))


def zen_snapshots_step(ctx: ReviewContext, targets: pd.DatetimeIndex) -> pd.DataFrame:
    """每个 60min 目标：zen60 + 15min 窗口里算出的 minor 17 + zen15 2。"""
    # 本 step 自己的目标 = 传入的族目标（runner 传的就是 `ctx.targets['60min']`）；
    # `_hour_targets` 之后可能追加**只服务 15min** 的 bar，那不是它的写入目标。
    own = (pd.DatetimeIndex(sorted(set(pd.to_datetime(targets))))
           if targets is not None and len(targets) else pd.DatetimeIndex([]))
    targets = _hour_targets(ctx, targets)
    if len(targets) == 0:
        return pd.DataFrame(columns=ZEN_COLUMNS)
    rows, idx = [], []
    minute_rows = {}
    src60 = ctx.frame('60min')
    src15 = ctx.frame('15min')
    # `hour_extra` 并进这一趟（D78）：SE/pivot/Renko/M9T 的映射统一走共享构造器。
    # ⚠️ 只给**真正的 60min 目标**算：`_hour_targets` 为 15min 依赖补出来的 bar
    # 不写 60min 表（写前会被 `_restrict_to_targets` 裁掉），算 se/中枢/Renko/
    # 原生列纯属浪费（CZSC + Prophet 是这里最贵的两段）。
    ohlc60 = slice_asof(src60, ctx.as_of)
    extra_by_ts = {}
    if len(own):
        _extra = build_hour_extra_frame(ohlc60, own, ctx.code)
        extra_by_ts = {T: _extra.iloc[i] for i, T in enumerate(own)}
    for T in targets:
        T = pd.Timestamp(T)
        ex = extra_by_ts.get(T)
        d60, p60 = (_zen_last(slice_asof(src60, T, WINDOW60))
                    if ex is not None else (np.nan, np.nan))
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
        # 共享构造器已按 target 完成 se/pivot/Renko/M9T 的 as-of 映射；
        # 不允许再写一份 searchsorted。
        if ex is not None:
            se = ex['se']
            piv_cnt = ex['ZenPivotCnt']
            piv_lag = ex['ZenPivotLag']
            rk_vals = ex[list(RENKO_DAY_COLUMNS)].to_numpy(dtype=float)
            # ⚠️ 逐列取（历史上有过 obj/list 列）；数值列在 DataFrame 构造时自然成为 float。
            nat_vals = [ex[c] for c in HOUR_NATIVE_COLUMNS]
        else:
            # 依赖补出来的 bar：只填 15min 需要的两列，其余留 NaN。
            se = piv_cnt = piv_lag = np.nan
            rk_vals = np.full(len(RENKO_DAY_COLUMNS), np.nan)
            nat_vals = [np.nan] * len(HOUR_NATIVE_COLUMNS)
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
