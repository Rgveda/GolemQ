# coding: utf-8
"""60min 额外两列：样本熵 `SE` + 因果中枢计数 `ZenPivotCnt`。

口径（用户 2026-10-20 指定，参考旧树）：

* `SE`：60min 收盘价的 `antropy.sample_entropy`，只在 **15:00** 那根算，
  窗口为截至该 bar 的最后 **≤2160 根**，`round(..., 3)`；盘中/更早的 bar
  沿用**上一交易日 15:00** 的值（因果 ffill，**不**复刻旧树把当日 15:00 值
  回填到当日更早 bar 的未来函数）。
* `ZenPivotCnt`：截至当前 bar **已确认**的缠论中枢个数 —— CZSC 笔 + 确认时刻
  （`bi_confirm_map` / `calc_pivots` / `causal_pivot_count`）。

只输出到 60min 族。
"""
from __future__ import annotations

from datetime import time as dtime
from typing import List

import numpy as np
import pandas as pd

from GolemQ.core.constants import AKA
from GolemQ.models.alias import ZEN
from GolemQ.pipeline.review.contract import HOUR_EXTRA_COLUMNS
from GolemQ.pipeline.steps import (FeatureStep, ReviewContext, TIER_DAILY,
                                   slice_asof)

SE_WINDOW = 2160
CZSC_MAX_BI = 512
_DAY_END = dtime(15, 0)


def _sample_entropy_by_day(ohlc: pd.DataFrame):
    """返回每个 15:00 bar 的 `(timestamps, SE)`；只算调用方已切好的帧。"""
    try:
        import antropy as ant
    except Exception:               # noqa: BLE001 环境缺 antropy ⇒ 全 NaN
        return pd.DatetimeIndex([]), np.array([], dtype=float)
    ts = pd.DatetimeIndex(pd.to_datetime(ohlc.index.get_level_values(0)))
    close = pd.to_numeric(ohlc[AKA.CLOSE], errors='coerce')
    days, vals = [], []
    for t in ts[ts.time == _DAY_END]:
        win = close.loc[ts <= t].tail(SE_WINDOW).dropna().to_numpy()
        if len(win) < 2:
            continue
        try:
            v = ant.sample_entropy(win)
        except Exception:           # noqa: BLE001 常数序列等
            v = np.nan
        if np.isfinite(v):
            days.append(t)
            vals.append(round(float(v), 3))
    return pd.DatetimeIndex(days), np.asarray(vals, dtype=float)


def _pivot_counts(ohlc: pd.DataFrame, targets, code: str):
    """截至每个目标 bar 已确认的中枢个数（int32）；失败返回 NaN。"""
    if ohlc is None or len(ohlc) < 30:
        return np.full(len(targets), np.nan)
    try:
        from czsc import CZSC
        from czsc.objects import Freq, RawBar

        from GolemQ.analysis.pivot import (bi_confirm_map, calc_pivots,
                                           causal_pivot_count)
        ts = pd.DatetimeIndex(pd.to_datetime(ohlc.index.get_level_values(0)))
        bars = []
        for i, (t, (_, row)) in enumerate(zip(ts, ohlc.iterrows())):
            bars.append(RawBar(
                symbol=code, dt=t, id=i, freq=Freq.F60,
                open=float(row.get('open', np.nan)),
                close=float(row.get('close', np.nan)),
                high=float(row.get('high', np.nan)),
                low=float(row.get('low', np.nan)),
                vol=float(row.get('volume', row.get('vol', 0)) or 0),
                amount=float(row.get('amount', 0) or 0),
            ))
        czsc = CZSC(bars, max_bi_count=CZSC_MAX_BI, verbose=False)
        conf = bi_confirm_map(bars, max_bi_count=CZSC_MAX_BI)
        pivots = calc_pivots(bi_list=czsc.bi_list, conf_map=conf, strict=True)
        return causal_pivot_count(pivots, targets, strict=True)
    except Exception:               # noqa: BLE001 CZSC 失败不该毁掉整条管线
        return np.full(len(targets), np.nan)


def hour_extra_step(ctx: ReviewContext, targets: pd.DatetimeIndex) -> pd.DataFrame:
    if targets is None or len(targets) == 0:
        return pd.DataFrame(columns=HOUR_EXTRA_COLUMNS)
    idx = pd.MultiIndex.from_tuples(
        [(pd.Timestamp(T), ctx.code) for T in targets], names=['ts', 'code'])
    ohlc = slice_asof(ctx.frame('60min'), ctx.as_of)
    if ohlc is None or len(ohlc) < 30:
        return pd.DataFrame(columns=HOUR_EXTRA_COLUMNS, index=idx)

    counts = _pivot_counts(ohlc, targets, ctx.code)
    days, vals = _sample_entropy_by_day(ohlc)
    se = np.full(len(targets), np.nan)
    if len(days):
        # 每个目标取「<= 它的最近一个 15:00」的值（因果 ffill）
        pos = days.searchsorted(pd.DatetimeIndex(targets), side='right') - 1
        ok = pos >= 0
        se[ok] = vals[pos[ok]]
    return pd.DataFrame({AKA.SAMPLE_ENTROPY: se, ZEN.PIVOT_COUNT: counts},
                        index=idx)


def build_hour_extra_steps() -> List[FeatureStep]:
    return [FeatureStep(
        name='hour_extra', tier=TIER_DAILY, input_freq='60min',
        output_family='60min', columns=HOUR_EXTRA_COLUMNS,
        func=hour_extra_step, history='anchor',
    )]
