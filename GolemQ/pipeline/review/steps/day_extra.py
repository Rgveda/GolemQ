# coding: utf-8
"""日族额外两列：日频样本熵 `se` + 15:00 取样的 `ZenPivotCnt`（D74）。

口径（用户 2026-10-28 定：**day 表日频，60min 列移除**）：

* `se`：**日收盘价**的 `antropy.sample_entropy`，窗口为截至该日最后
  **≤540 个交易日**（≈ 旧 60min 版 2160 根的时间跨度，4 根/日），`round(..., 3)`。
* `ZenPivotCnt`：截至该日 **15:00** 已确认的缠论中枢个数 —— 与旧 60min 版
  **同一个 60min CZSC 口径**，只是每行只留 15:00 那个值。

⚠️ `ZenPivotCnt` 需要 60min 帧；`ReviewPipeline._required_freqs` 在请求日族时
一并取 60min（旧 60min 版也只有 ~800 天 review 数据，更早的日行保持 NaN）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from GolemQ.core.constants import AKA

#: 日频样本熵窗口（交易日数）。旧 60min 版是 2160 根 = 约 540 个交易日。
SE_WINDOW_DAYS = 540
CZSC_MAX_BI = 512


def sample_entropy_daily(day: pd.DataFrame, dates) -> np.ndarray:
    """每个目标日的日收盘样本熵（`round(..., 3)`）；不足 2 个点 / antropy 缺失 ⇒ NaN。"""
    try:
        import antropy as ant
    except Exception:               # noqa: BLE001 环境缺 antropy ⇒ 全 NaN
        return np.full(len(dates), np.nan, dtype=float)
    if day is None or len(day) == 0:
        return np.full(len(dates), np.nan, dtype=float)

    ts = pd.DatetimeIndex(pd.to_datetime(day.index.get_level_values(0)))
    close = pd.to_numeric(day[AKA.CLOSE], errors='coerce').to_numpy(dtype=float)
    out = np.full(len(dates), np.nan, dtype=float)
    for i, t in enumerate(dates):
        pos = int(ts.searchsorted(pd.Timestamp(t), side='right'))
        win = close[max(0, pos - SE_WINDOW_DAYS):pos]
        win = win[np.isfinite(win)]
        if len(win) < 2:
            continue
        try:
            v = ant.sample_entropy(win)
        except Exception:           # noqa: BLE001 常数序列等
            v = np.nan
        if np.isfinite(v):
            out[i] = round(float(v), 3)
    return out


def pivot_counts(ohlc: pd.DataFrame, cutoffs, code: str) -> np.ndarray:
    """截至每个 `cutoffs` 时刻已确认的中枢个数（int32）；失败返回 NaN。

    `cutoffs` 应是该日 **15:00**（不是零点）—— 中枢算到 15:00 收盘那一根。
    """
    if ohlc is None or len(ohlc) < 30:
        return np.full(len(cutoffs), np.nan)
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
        counts = causal_pivot_count(pivots, cutoffs, strict=True)
        # 帧未覆盖的时刻要留 NaN，不能拿 "0 个中枢" 冒充「没数据」：
        # `causal_pivot_count` 在首根 bar 之前同样返回 0。
        cut = pd.DatetimeIndex(cutoffs)
        ok = (cut >= ts[0]) & (cut <= ts[-1])
        return np.where(ok, counts, np.nan)
    except Exception:               # noqa: BLE001 CZSC 失败不该毁掉整条管线
        return np.full(len(cutoffs), np.nan)
