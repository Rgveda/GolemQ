# coding: utf-8
"""coldstart 最小集：`M9T major / Poly9 / Prophet`（模型层消费）。

用户 2026-10-19 定：只合模型层实际读取、纯 K 线可算的部分，**不接**东财/同花顺/
massive；且**不再单独写 `*_coldstart_reviews`** —— 冷启动列合并进
`*_reality_feat_60min`（同一张 60min 快照表，避免两套真相）。
`m9t`（Maj）由 `_coldstart_row`（日线 `_core_daily`）产出；
**stage 族不在这里算** —— `stage/garPD/garPAnchor/garPCount/garPClr` 从
4.4 旧库搬运（`tools/migrate_stage_from44.py`，用户 2026-11-16），
`bstStgBf/epStgBf` 由 `reality_attach` 现算、不落库。

⚠️ 与老树的两处**刻意差异**：

1. **STAGE 的 checkpoint 网格**：老树用 BOLL-HMA5 checkout 等一串 compact 特征选
   checkpoint；那条链尚未搬。这里用**每 21 个交易日**为 checkpoint（+ 最后一行），
   语义仍是「在稀疏检查点上 KMeans 定带、其余行按最近带分类」。
2. **Prophet 落比值**（D53）：`calc_fbprophET_bands` 产出的 `LB/UB` 是
   `yhat/close`；净空公式相应改成比值（`(UB - HMA5/close) / (UB-LB)`），
   不是老树的绝对价。模型层迁移时按比值消费。
"""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from GolemQ.analysis.fbprophet import calc_fbprophet_bands
from GolemQ.analysis.indicators import TA_HMA
from GolemQ.analysis.magic_nine_jit import calc_magic_nine_turns_np_vXs_jit
from GolemQ.analysis.poly9 import calc_polynomial9_vXIs
from GolemQ.analysis.quadrant_jit import mapower_np_vXIIs
from GolemQ.analysis.timing import calc_event_timing_lag
from GolemQ.core.constants import AKA, FIELD as FLD
from GolemQ.pipeline.review.contract import COLDSTART_COLUMNS
from GolemQ.pipeline.steps import (FeatureStep, ReviewContext, TIER_DAILY,
                                   slice_asof)

CHECKPOINT_STRIDE = 21


def _core_daily(day: pd.DataFrame, feats: pd.DataFrame = None) -> pd.DataFrame:
    """日线核心列：M9T(+lag) / Poly9（**不含 stage 族** —— 那由 60min 链算）。

    ``feats`` 传入时复用 `day_core` 已算好的 M9T/Poly9，**不再重算**。
    """
    feats = features if feats is None else feats
    feats[AKA.CLOSE] = pd.to_numeric(day[AKA.CLOSE], errors='coerce')
    close = pd.to_numeric(day[AKA.CLOSE], errors='coerce')
    typical = (day[AKA.OPEN] + day[AKA.HIGH] + day[AKA.LOW] + close) / 4
    if FLD.HMA10 not in feats.columns:
        feats[FLD.HMA10] = TA_HMA(typical.to_numpy(), 10)

    if FLD.MAGIC_NINE_TURNS not in feats.columns:
        feats[FLD.MAGIC_NINE_TURNS] = calc_magic_nine_turns_np_vXs_jit(
            close.to_numpy().astype(np.float32))
    feats[FLD.MAGIC_NINE_TURNS_MAJOR] = feats[FLD.MAGIC_NINE_TURNS]
    # `day_core` 的 `_frame_features` 已算过同一列；存在就复用，不重复扫一遍。
    if FLD.MAGIC_NINE_TURNS_TIMING_LAG not in feats.columns:
        feats[FLD.MAGIC_NINE_TURNS_TIMING_LAG] = calc_event_timing_lag(
            feats[FLD.MAGIC_NINE_TURNS].to_numpy())
    if FLD.POLYNOMIAL9 in feats.columns:
        feats[FLD.POLYNOMIAL9_MAJOR] = feats[FLD.POLYNOMIAL9]
        if FLD.POLYNOMIAL9_NORM in feats.columns:
            feats[FLD.POLYNOMIAL9_NORM_MAJOR] = feats[FLD.POLYNOMIAL9_NORM]
    else:
        try:
            feats = calc_polynomial9_vXIs(day, feats)
            feats[FLD.POLYNOMIAL9_MAJOR] = feats[FLD.POLYNOMIAL9]
            # 与 day step 同口径：norm 输入用 **Poly9 / close**，避免复权断层。
            _p9_ratio = feats[FLD.POLYNOMIAL9] / close
            feats[FLD.POLYNOMIAL9_NORM_MAJOR] = mapower_np_vXIIs(
                _p9_ratio.to_numpy(),
                range_list=np.array(range(5, 30)).astype(np.int16))
        except Exception:           # noqa: BLE001
            for c in (FLD.POLYNOMIAL9_MAJOR, FLD.POLYNOMIAL9_NORM_MAJOR):
                feats[c] = np.nan

    return feats


def _prophet_columns(day: pd.DataFrame, feats: pd.DataFrame) -> pd.DataFrame:
    """Prophet 比值 + 净空（D53 比值口径）。"""
    close = pd.to_numeric(day[AKA.CLOSE], errors='coerce')
    typical = (day[AKA.OPEN] + day[AKA.HIGH] + day[AKA.LOW] + close) / 4
    if FLD.HMA5 not in feats.columns:
        feats[FLD.HMA5] = TA_HMA(typical.to_numpy(), 5)
    if FLD.HMA10 not in feats.columns:
        feats[FLD.HMA10] = TA_HMA(typical.to_numpy(), 10)
    try:
        feats = calc_fbprophet_bands(day, feats, major_ohlc=None)
    except Exception:               # noqa: BLE001 prophet 未装/拟合失败
        for c in (FLD.FBPROPHET_LB, FLD.FBPROPHET_UB,
                  FLD.FBPROPHET_LB_MAJOR, FLD.FBPROPHET_UB_MAJOR):
            feats[c] = np.nan
    width = feats[FLD.FBPROPHET_UB] - feats[FLD.FBPROPHET_LB]
    feats[FLD.FBPROPHET_CLEARANCE] = (
        (feats[FLD.FBPROPHET_UB] - feats[FLD.HMA5] / close) / width)
    feats[FLD.FBPROPHET_CLEARANCE_NORM] = 1 - mapower_np_vXIIs(
        feats[FLD.FBPROPHET_CLEARANCE].to_numpy().astype(np.float32),
        range_list=np.array(range(5, 30)).astype(np.int16))
    width_maj = feats[FLD.FBPROPHET_UB_MAJOR] - feats[FLD.FBPROPHET_LB_MAJOR]
    feats[FLD.FBPROPHET_CLEARANCE_MAJOR] = (
        (feats[FLD.FBPROPHET_UB_MAJOR] - feats[FLD.HMA10] / close) / width_maj)
    feats[FLD.FBPROPHET_CLEARANCE_NORM_MAJOR] = 1 - mapower_np_vXIIs(
        feats[FLD.FBPROPHET_CLEARANCE_MAJOR].to_numpy().astype(np.float32),
        range_list=np.array(range(5, 30)).astype(np.int16))
    return feats


def _coldstart_row(day: pd.DataFrame, feats: pd.DataFrame = None) -> dict:
    """在**截至某个目标日**的日线帧上算一行 coldstart（因果）。"""
    if day is None or len(day) < 30:
        return {c: np.nan for c in COLDSTART_COLUMNS}

    feats = _core_daily(day, feats=feats)
    feats = _prophet_columns(day, feats)
    last = feats.iloc[-1]
    # ⚠️ Poly9 拟合线是绝对价位 ⇒ **除 close 存比值**（D49 通则，避免复权断层）。
    close = pd.to_numeric(day[AKA.CLOSE], errors='coerce')
    _close = float(close.iloc[-1])
    _poly9 = last.get(FLD.POLYNOMIAL9_MAJOR, np.nan)
    _poly9_ratio = (_poly9 / _close) if (_close and not pd.isna(_poly9)) else np.nan
    return {
        # ⚠️ **stage 族不在这里**（用户 2026-11-16）：`stage/bstStgBf/epStgBf/
        # garP*` 由 60min 链（`native_by_day`）一次算出，日线链不再重复计算。
        'm9t': last.get(FLD.MAGIC_NINE_TURNS, np.nan),
        'm9tMaj': last.get(FLD.MAGIC_NINE_TURNS_MAJOR, np.nan),
        FLD.POLYNOMIAL9_MAJOR: _poly9_ratio,
        FLD.POLYNOMIAL9_NORM_MAJOR: last.get(FLD.POLYNOMIAL9_NORM_MAJOR, np.nan),
        FLD.FBPROPHET_LB_DUMMY: last.get(FLD.FBPROPHET_LB, np.nan),
        FLD.FBPROPHET_UB_DUMMY: last.get(FLD.FBPROPHET_UB, np.nan),
        FLD.FBPROPHET_CLEARANCE_NORM: last.get(FLD.FBPROPHET_CLEARANCE_NORM, np.nan),
        FLD.FBPROPHET_LB_MAJOR_DUMMY: last.get(FLD.FBPROPHET_LB_MAJOR, np.nan),
        FLD.FBPROPHET_UB_MAJOR_DUMMY: last.get(FLD.FBPROPHET_UB_MAJOR, np.nan),
        FLD.FBPROPHET_CLEARANCE_NORM_MAJOR: last.get(FLD.FBPROPHET_CLEARANCE_NORM_MAJOR, np.nan),
    }


def coldstart_step(ctx: ReviewContext, targets: pd.DatetimeIndex) -> pd.DataFrame:
    """取样：`day_core` 已对每个目标日期算好 coldstart，这里只映射到 60min 目标。

    ⚠️ **因果口径**：日线 bar 的时间戳是当日 00:00，close 却是**当日收盘**。
    因此：
    * 60min 目标 **< 15:00**（10:30 / 11:30 / 14:00）：只能用**上一交易日**的
      coldstart 行（那一行的日线已收盘）；
    * 60min 目标 **>= 15:00**：当日已收盘，用当日行。
    否则盘中行会拿到当日收盘价（未来函数）。

    Prophet 是全样本拟合，仍按目标日期逐个拟合（在 `day_core` 内）；本 step 不重算。
    """
    if targets is None or len(targets) == 0:
        return pd.DataFrame(columns=COLDSTART_COLUMNS)
    cache = ctx.outputs.get('_coldstart_rows') or {}
    dates = sorted(cache)
    nan_row = {c: np.nan for c in COLDSTART_COLUMNS}

    rows, idx = [], []
    for t in targets:
        t = pd.Timestamp(t)
        d = t.normalize()
        if t.time() >= pd.Timestamp('15:00').time():
            key = d
        else:
            prev = [x for x in dates if x < d]
            key = prev[-1] if prev else None
        row = cache.get(key, nan_row)
        rows.append([row.get(c, np.nan) for c in COLDSTART_COLUMNS])
        idx.append((t, ctx.code))
    out = pd.DataFrame(
        rows, index=pd.MultiIndex.from_tuples(idx, names=['ts', 'code']),
        columns=list(COLDSTART_COLUMNS))
    return out


def build_coldstart_steps() -> List[FeatureStep]:
    return [FeatureStep(
        name='coldstart', tier=TIER_DAILY, input_freq='day',
        output_family='60min', columns=COLDSTART_COLUMNS,
        func=coldstart_step, deps=('day_core',), history='anchor')]
