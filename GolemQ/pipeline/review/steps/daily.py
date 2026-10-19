# coding: utf-8
"""日族：老树 `calc_daily_snapshot_features` 的 13 列（逐日快照、因果）。"""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from GolemQ.analysis.indicators import TA_HMA
from GolemQ.analysis.magic_nine import calc_magic_nine_turns_np_vXs
from GolemQ.analysis.poly9 import calc_polynomial9_vXIs
from GolemQ.analysis.quadrant import mapower_np_vXIIs
from GolemQ.analysis.regtree import calc_regtree_fractal_vXIs
from GolemQ.analysis.timing import calc_event_timing_lag
from GolemQ.analysis.timeseries import weekly_ohlc
from GolemQ.analysis.zen import calc_zen_timing_lag
from GolemQ.core.constants import AKA, FEATURES as FTR, FIELD as FLD
from GolemQ.pipeline.steps import (FeatureStep, ReviewContext, TIER_DAILY,
                                   slice_asof)
from GolemQ.pipeline.review.contract import DAY_COLUMNS


def _last(series, default=np.nan):
    if series is None or len(series) == 0:
        return default
    try:
        return series.iloc[-1]
    except Exception:            # noqa: BLE001
        return default


def _frame_features(ohlc: pd.DataFrame, annual: int = 252) -> pd.DataFrame:
    """在**因果切片**上算日/周线 zen、Poly9、M9T、RegTree。"""
    feats = pd.DataFrame(index=ohlc.index)
    close = pd.to_numeric(ohlc[AKA.CLOSE], errors='coerce')
    typical = (ohlc[AKA.OPEN] + ohlc[AKA.HIGH] + ohlc[AKA.LOW] + close) / 4
    feats[FLD.HMA10] = TA_HMA(typical.to_numpy(), 10)

    m9t = calc_magic_nine_turns_np_vXs(close.to_numpy().astype(np.float32))
    feats[FLD.MAGIC_NINE_TURNS] = m9t
    feats[FLD.MAGIC_NINE_TURNS_TIMING_LAG] = calc_event_timing_lag(m9t)

    try:
        feats = calc_polynomial9_vXIs(ohlc, feats)
        if FLD.POLYNOMIAL9 in feats.columns:
            # ⚠️ **norm 的输入也必须是 close 相对量**：绝对价位在复权换挡处有
            # 断层，会改变 MA 排列。除以同一行 close 后再做 mapower。
            _p9_ratio = (feats[FLD.POLYNOMIAL9] /
                         pd.to_numeric(ohlc[AKA.CLOSE], errors='coerce'))
            feats[FLD.POLYNOMIAL9_NORM] = mapower_np_vXIIs(
                _p9_ratio.to_numpy(),
                range_list=np.array(range(5, 30)).astype(np.int16))
    except Exception:            # noqa: BLE001 短历史/异常时保留空列
        for c in (FLD.POLYNOMIAL9, FLD.POLYNOMIAL9_NORM):
            if c not in feats.columns:
                feats[c] = np.nan

    feats = calc_zen_timing_lag(ohlc, feats)

    if len(ohlc) > 450:
        try:
            feats = calc_regtree_fractal_vXIs(
                ohlc, features=feats, bar_limit=450, rate=25, dur=10)
        except Exception:        # noqa: BLE001
            pass
    for c in (FLD.REGTREE_PRICE, FLD.REGTREE_TIMING_LAG, FLD.REGTREE_SLOPE):
        if c not in feats.columns:
            feats[c] = np.nan
    return feats


def _weekly_m9t_lag(wk_feats: pd.DataFrame):
    """周线 M9T 的 event-lag（老树 `cli/review.py:2266-2272` 的公式）。

    ⚠️ **刻意与老树不同**：老树把周线 `m9t` 只写到**本次批次新增的日行**上，再对
    这条「大部分 NaN」的日线序列做 event-lag ⇒ 结果依赖**批次窗口**，不满足前缀
    稳定性（实测同一 T 在不同批次可差 900+）。这里改在**周线帧**上算，因果且稳定；
    与老树的差通常 ≤2 根，归为 `design/batch-window`。
    """
    if (wk_feats is None or len(wk_feats) == 0 or
            FLD.MAGIC_NINE_TURNS not in wk_feats.columns):
        return np.nan
    m = wk_feats[FLD.MAGIC_NINE_TURNS].to_numpy(dtype=float)
    if len(m) == 0:
        return np.nan
    if m[-1] > 0.000168:
        return calc_event_timing_lag(np.where(m > 0, 1, -1))[-1]
    if m[-1] < -0.000168:
        return calc_event_timing_lag(np.where(m < -0.000168, -1, 1))[-1]
    return 0


def daily_snapshot_step(ctx: ReviewContext, targets: pd.DatetimeIndex) -> pd.DataFrame:
    """每个日线快照返回一行 13 列。"""
    rows = []
    day_all = ctx.frame('day')
    for T in targets:
        day = slice_asof(day_all, T, 1008)
        if len(day) < 30:
            continue
        feats = _frame_features(day, annual=252)

        wk_raw = slice_asof(day_all, T, 2160)
        wk_feats = pd.DataFrame()
        if len(wk_raw) > 0:
            try:
                wk = weekly_ohlc(wk_raw)
                if len(wk) > 0:
                    wk_feats = _frame_features(wk, annual=52)
            except Exception:    # noqa: BLE001
                wk_feats = pd.DataFrame()

        # ⚠️ **价格导出的列一律存比值**（D49 通则）：Poly9 / RegTree 的拟合线
        # 是绝对价位，直接落库会在复权换挡时断层。除以**同一行**的 close。
        close = pd.to_numeric(day[AKA.CLOSE], errors='coerce')
        poly9_ratio = (feats[FLD.POLYNOMIAL9] / close.to_numpy()
                       if FLD.POLYNOMIAL9 in feats.columns else pd.Series(np.nan, index=feats.index))
        regtree_ratio = (feats[FLD.REGTREE_PRICE] / close.to_numpy()
                         if FLD.REGTREE_PRICE in feats.columns else pd.Series(np.nan, index=feats.index))
        # 斜率也是「价格 / bar」—— 同样除以 close，变成「相对收益 / bar」。
        regtree_slope_ratio = (feats[FLD.REGTREE_SLOPE] / close.to_numpy()
                               if FLD.REGTREE_SLOPE in feats.columns else pd.Series(np.nan, index=feats.index))
        if len(wk_feats):
            wk_close = pd.to_numeric(wk_raw[AKA.CLOSE], errors='coerce').reindex(wk_feats.index)
            poly9_week_ratio = wk_feats[FLD.POLYNOMIAL9] / wk_close
        else:
            poly9_week_ratio = pd.Series(dtype=float)

        row = {
            FTR.ZEN_DASH_TIMING_LAG_MAJOR_REAL: _last(feats[FLD.ZEN_DASH_TIMING_LAG]),
            FTR.ZEN_PEAK_TIMING_LAG_MAJOR_REAL: _last(feats[FLD.ZEN_PEAK_TIMING_LAG]),
            FTR.ZEN_DASH_TIMING_LAG_WEEKLY_REAL: _last(
                wk_feats[FLD.ZEN_DASH_TIMING_LAG]) if len(wk_feats) else np.nan,
            FTR.POLYNOMIAL9_MAJOR_REAL: _last(poly9_ratio),
            FTR.POLYNOMIAL9_NORM_MAJOR_REAL: _last(feats[FLD.POLYNOMIAL9_NORM]),
            FTR.POLYNOMIAL9_WEEKLY_REAL: _last(poly9_week_ratio),
            FTR.POLYNOMIAL9_NORM_WEEKLY_REAL: _last(
                wk_feats[FLD.POLYNOMIAL9_NORM]) if len(wk_feats) else np.nan,
            FTR.MAGIC_NINE_TURNS_MAJOR_REAL: _last(feats[FLD.MAGIC_NINE_TURNS]),
            FTR.MAGIC_NINE_TURNS_TIMING_LAG_MAJOR_REAL: _last(
                feats[FLD.MAGIC_NINE_TURNS_TIMING_LAG]),
            FTR.MAGIC_NINE_TURNS_WEEKLY_REAL: _last(
                wk_feats[FLD.MAGIC_NINE_TURNS]) if len(wk_feats) else np.nan,
            FTR.MAGIC_NINE_TURNS_TIMING_LAG_WEEKLY_REAL: _weekly_m9t_lag(wk_feats),
            FTR.REGTREE_PRICE_MAJOR_REAL: _last(regtree_ratio),
            FTR.REGTREE_TIMING_LAG_MAJOR_REAL: _last(feats[FLD.REGTREE_TIMING_LAG]),
            FTR.REGTREE_SLOPE_MAJOR_REAL: _last(regtree_slope_ratio),
        }
        rows.append((pd.Timestamp(T), ctx.code, row))

    if not rows:
        return pd.DataFrame(columns=DAY_COLUMNS)
    idx = pd.MultiIndex.from_tuples([(t, c) for t, c, _ in rows],
                                    names=['ts', 'code'])
    return pd.DataFrame([r for _, _, r in rows], index=idx)


def build_daily_steps() -> List[FeatureStep]:
    return [FeatureStep(
        name='daily_snapshot', tier=TIER_DAILY, input_freq='day',
        output_family='day', columns=DAY_COLUMNS,
        func=daily_snapshot_step, history='bars', warmup=1008,
    )]
