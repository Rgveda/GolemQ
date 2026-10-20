# coding: utf-8
"""日族：老树 `calc_daily_snapshot_features` 的 13 列（逐日快照、因果）。"""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from GolemQ.analysis.indicators import TA_HMA
from GolemQ.analysis.magic_nine_jit import calc_magic_nine_turns_np_vXs_jit
from GolemQ.analysis.poly9 import calc_polynomial9_vXIs
from GolemQ.analysis.quadrant_jit import mapower_np_vXIIs
from GolemQ.analysis.regtree_jit import calc_regtree_fractal_vXIs
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


def _frame_features(ohlc: pd.DataFrame, annual: int = 252,
                    with_regtree: bool = True) -> pd.DataFrame:
    """在**因果切片**上算日/周线 zen、Poly9、M9T、RegTree。

    ``with_regtree=False`` 用于**周线**：`_day_row` 只用周线的 zen/Poly9/M9T，
    周线 RegTree 是纯浪费（而且是最重的部分之一）。
    """
    feats = pd.DataFrame(index=ohlc.index)
    close = pd.to_numeric(ohlc[AKA.CLOSE], errors='coerce')
    typical = (ohlc[AKA.OPEN] + ohlc[AKA.HIGH] + ohlc[AKA.LOW] + close) / 4
    feats[FLD.HMA10] = TA_HMA(typical.to_numpy(), 10)

    m9t = calc_magic_nine_turns_np_vXs_jit(close.to_numpy().astype(np.float32))
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

    if with_regtree and len(ohlc) > 450:
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


def _week_start(t) -> pd.Timestamp:
    """该 bar 所在周的周一（用于「已完成周」缓存）。"""
    t = pd.Timestamp(t).normalize()
    return t - pd.Timedelta(days=t.weekday())


def _day_row(day: pd.DataFrame, feats: pd.DataFrame,
             wk_raw: pd.DataFrame, wk_feats: pd.DataFrame) -> dict:
    """从已算好的 day / week 特征抽一行（14 列）。"""
    close = pd.to_numeric(day[AKA.CLOSE], errors='coerce')
    poly9_ratio = (feats[FLD.POLYNOMIAL9] / close.to_numpy()
                   if FLD.POLYNOMIAL9 in feats.columns else pd.Series(np.nan, index=feats.index))
    regtree_ratio = (feats[FLD.REGTREE_PRICE] / close.to_numpy()
                     if FLD.REGTREE_PRICE in feats.columns else pd.Series(np.nan, index=feats.index))
    regtree_slope_ratio = (feats[FLD.REGTREE_SLOPE] / close.to_numpy()
                           if FLD.REGTREE_SLOPE in feats.columns else pd.Series(np.nan, index=feats.index))
    if len(wk_feats):
        wk_close = pd.to_numeric(wk_raw[AKA.CLOSE], errors='coerce').reindex(wk_feats.index)
        poly9_week_ratio = wk_feats[FLD.POLYNOMIAL9] / wk_close
    else:
        poly9_week_ratio = pd.Series(dtype=float)
    return {
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


def day_core_step(ctx: ReviewContext, targets: pd.DatetimeIndex) -> None:
    """**day 唯一一趟**：day 特征 + coldstart（M9T/Poly9 共享，Prophet 逐日）。

    周线按**上一个已完成周**缓存：同一周内的日目标共用一份周线特征。
    `daily_snapshot` / `coldstart` 只从缓存取样。
    """
    day_all = ctx.frame('day')
    day_targets = ctx.targets.get('day', pd.DatetimeIndex([]))
    cold_dates = {pd.Timestamp(t).normalize()
                  for t in ctx.targets.get('60min', pd.DatetimeIndex([]))}
    if len(day_targets) == 0:
        return None

    from ..coldstart import _coldstart_row     # 延迟导入避免循环

    day_rows, cold_rows = {}, {}
    wk_cache = {}
    for T in day_targets:
        T = pd.Timestamp(T)
        day = slice_asof(day_all, T, 832)
        if len(day) < 30:
            continue
        feats = _frame_features(day, annual=252)

        # 周线：同一「已完成周」只算一次（用该周之前的 2160 根日线）
        wkey = _week_start(T)
        if wkey not in wk_cache:
            wk_end = wkey - pd.Timedelta(days=1)
            wk_raw = slice_asof(day_all, wk_end, 2160)
            wk_feats = pd.DataFrame()
            if len(wk_raw) > 0:
                try:
                    wk = weekly_ohlc(wk_raw)
                    if len(wk) > 0:
                        wk_feats = _frame_features(wk, annual=52,
                                                   with_regtree=False)
                except Exception:    # noqa: BLE001
                    wk_feats = pd.DataFrame()
            wk_cache[wkey] = (wk_raw, wk_feats)
        wk_raw, wk_feats = wk_cache[wkey]

        day_rows[T] = _day_row(day, feats, wk_raw, wk_feats)
        if T.normalize() in cold_dates:
            cold_rows[T.normalize()] = _coldstart_row(day, feats=feats)

    if day_rows:
        idx = pd.MultiIndex.from_tuples(
            [(t, ctx.code) for t in day_rows], names=['ts', 'code'])
        ctx.outputs['_day_rows'] = pd.DataFrame(
            list(day_rows.values()), index=idx, columns=list(DAY_COLUMNS))
    if cold_rows:
        ctx.outputs['_coldstart_rows'] = cold_rows
    return None


def daily_snapshot_step(ctx: ReviewContext, targets: pd.DatetimeIndex) -> pd.DataFrame:
    """取样：`day_core` 已经算好，这里不再重算。"""
    src = ctx.outputs.get('_day_rows')
    if src is None or len(src) == 0:
        return pd.DataFrame(columns=DAY_COLUMNS)
    return src


def build_daily_steps() -> List[FeatureStep]:
    return [
        FeatureStep(name='day_core', tier=TIER_DAILY, input_freq='day',
                    output_family=None, columns=(), func=day_core_step,
                    history='bars', warmup=832),
        FeatureStep(name='daily_snapshot', tier=TIER_DAILY, input_freq='day',
                    output_family='day', columns=DAY_COLUMNS,
                    func=daily_snapshot_step, deps=('day_core',),
                    history='bars', warmup=832),
    ]
