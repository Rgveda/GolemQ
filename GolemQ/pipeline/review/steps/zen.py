# coding: utf-8
"""60min / 15min 的 zen 快照（`zDLagR/zPLagR` 与 `zDLagMinR/zPLagMinR`）。

⚠️ **不搬老树 `KlineAnalyze` 0.53**；笔/线段由新树 czsc 0.7.10 生成
（`analysis/zen.py`）。每个快照取**截至 T 的固定 2016 根**，不跨快照复用，
保证前缀稳定性。

2026-10-19 合并精简：原先 `hour_zen` / `minor_zen` 两个 step 各跑一遍循环，
现在一个 `zen_snapshots` 在同一轮目标里同时产出 4 列。
"""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from GolemQ.analysis.zen import calc_zen_timing_lag
from GolemQ.core.constants import FIELD as FLD
from GolemQ.pipeline.review.contract import HOUR_ZEN_COLUMNS
from GolemQ.pipeline.steps import (FeatureStep, ReviewContext, TIER_DAILY,
                                   slice_asof)

WINDOW = 2016


def _zen_last(ohlc: pd.DataFrame):
    if ohlc is None or len(ohlc) == 0:
        return np.nan, np.nan
    feats = calc_zen_timing_lag(ohlc)
    return (feats[FLD.ZEN_DASH_TIMING_LAG].iloc[-1],
            feats[FLD.ZEN_PEAK_TIMING_LAG].iloc[-1])


def zen_snapshots_step(ctx: ReviewContext, targets: pd.DatetimeIndex) -> pd.DataFrame:
    """对每个 60min 目标同时算 60min 与 15min 两条 zen（`zDLag*` / `zDLagMin*`）。"""
    rows, idx = [], []
    src60 = ctx.frame('60min')
    src15 = ctx.frame('15min')
    for T in targets:
        d60, p60 = _zen_last(slice_asof(src60, T, WINDOW))
        d15, p15 = _zen_last(slice_asof(src15, T, WINDOW))
        rows.append((d60, p60, d15, p15))
        idx.append((pd.Timestamp(T), ctx.code))
    if not idx:
        return pd.DataFrame(columns=HOUR_ZEN_COLUMNS)
    return pd.DataFrame(rows,
                        index=pd.MultiIndex.from_tuples(idx, names=['ts', 'code']),
                        columns=list(HOUR_ZEN_COLUMNS))


def build_zen_steps() -> List[FeatureStep]:
    return [FeatureStep(
        name='zen_snapshots', tier=TIER_DAILY, input_freq='60min',
        output_family='60min', columns=HOUR_ZEN_COLUMNS,
        func=zen_snapshots_step, history='bars', warmup=WINDOW,
    )]
