# coding: utf-8
"""coldstart 最小集：`M9T major / Poly9 / Prophet`（模型层消费）。

用户 2026-10-19 定：只合模型层实际读取、纯 K 线可算的部分，**不接**东财/同花顺/
massive；且**不再单独写 `*_coldstart_reviews`** —— 冷启动列合并进
`*_reality_feat_60min`（同一张 60min 快照表，避免两套真相）。
`STAGE / BOOTSTRAP / m9t / m9tLag` 由 `hour_legacy` step 提供。

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
from GolemQ.analysis.magic_nine import calc_magic_nine_turns_np_vXs
from GolemQ.analysis.poly9 import calc_polynomial9_vXIs
from GolemQ.analysis.quadrant import mapower_np_vXIIs
from GolemQ.analysis.stages import calc_garident_stages
from GolemQ.analysis.timing import Timeline_duration, calc_event_timing_lag
from GolemQ.core.constants import AKA, FIELD as FLD
from GolemQ.pipeline.review.contract import COLDSTART_COLUMNS
from GolemQ.pipeline.steps import (FeatureStep, ReviewContext, TIER_DAILY,
                                   slice_asof)

CHECKPOINT_STRIDE = 21


def _stage_columns(day: pd.DataFrame) -> pd.DataFrame:
    """日线 STAGE / BOOTSTRAP（稀疏 checkpoint + KMeans）。"""
    feats = pd.DataFrame(index=day.index)
    feats[AKA.CLOSE] = pd.to_numeric(day[AKA.CLOSE], errors='coerce')
    ck = feats.iloc[::CHECKPOINT_STRIDE].copy()
    if len(ck) == 0 or ck.index[-1] != feats.index[-1]:
        ck = pd.concat([ck, feats.tail(1)])
    try:
        feats = calc_garident_stages(ck, feats, random_state=0)
    except Exception:               # noqa: BLE001 缺 sklearn/kneed 或定带失败
        feats[AKA.STAGE] = np.nan
    st = pd.to_numeric(feats.get(AKA.STAGE, pd.Series(np.nan, index=feats.index)),
                       errors='coerce').to_numpy()
    feats[AKA.STAGE] = st
    feats[FLD.BOOTSTRAP_STAGE_BEFORE] = Timeline_duration(
        np.where(st < -0.00168, 1, 0)).astype(np.int16)
    return feats


def _core_daily(day: pd.DataFrame) -> pd.DataFrame:
    """日线核心列：STAGE / BOOTSTRAP / M9T(+lag) / Poly9。**因果**（逐 checkpoint）。"""
    feats = _stage_columns(day)
    close = pd.to_numeric(day[AKA.CLOSE], errors='coerce')
    typical = (day[AKA.OPEN] + day[AKA.HIGH] + day[AKA.LOW] + close) / 4
    feats[FLD.HMA10] = TA_HMA(typical.to_numpy(), 10)
    m9t = calc_magic_nine_turns_np_vXs(close.to_numpy().astype(np.float32))
    feats[FLD.MAGIC_NINE_TURNS] = m9t
    feats[FLD.MAGIC_NINE_TURNS_MAJOR] = m9t
    feats[FLD.MAGIC_NINE_TURNS_TIMING_LAG] = calc_event_timing_lag(m9t)
    try:
        feats = calc_polynomial9_vXIs(day, feats)
        feats[FLD.POLYNOMIAL9_MAJOR] = feats[FLD.POLYNOMIAL9]
        # 与 day step 同口径：norm 输入用 **Poly9 / close**，避免复权断层。
        _p9_ratio = feats[FLD.POLYNOMIAL9] / close
        feats[FLD.POLYNOMIAL9_NORM_MAJOR] = mapower_np_vXIIs(
            _p9_ratio.to_numpy(),
            range_list=np.array(range(5, 30)).astype(np.int16))
    except Exception:               # noqa: BLE001
        for c in (FLD.POLYNOMIAL9_MAJOR, FLD.POLYNOMIAL9_NORM_MAJOR):
            feats[c] = np.nan
    return feats


def _prophet_columns(day: pd.DataFrame, feats: pd.DataFrame) -> pd.DataFrame:
    """Prophet 比值 + 净空（D53 比值口径）。"""
    close = pd.to_numeric(day[AKA.CLOSE], errors='coerce')
    typical = (day[AKA.OPEN] + day[AKA.HIGH] + day[AKA.LOW] + close) / 4
    feats[FLD.HMA5] = TA_HMA(typical.to_numpy(), 5)
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


def _coldstart_row(day: pd.DataFrame) -> dict:
    """在**截至某个目标日**的日线帧上算一行 coldstart（因果）。"""
    if day is None or len(day) < 30:
        return {c: np.nan for c in COLDSTART_COLUMNS}

    feats = _core_daily(day)
    feats = _prophet_columns(day, feats)
    last = feats.iloc[-1]
    # ⚠️ Poly9 拟合线是绝对价位 ⇒ **除 close 存比值**（D49 通则，避免复权断层）。
    _close = float(pd.to_numeric(day[AKA.CLOSE], errors='coerce').iloc[-1])
    _poly9 = last.get(FLD.POLYNOMIAL9_MAJOR, np.nan)
    _poly9_ratio = (_poly9 / _close) if (_close and not pd.isna(_poly9)) else np.nan
    return {
        'STAGE': last.get(AKA.STAGE, np.nan),
        'BST_STG_BF': last.get(FLD.BOOTSTRAP_STAGE_BEFORE, np.nan),
        'm9t': last.get(FLD.MAGIC_NINE_TURNS, np.nan),
        'm9tMaj': last.get(FLD.MAGIC_NINE_TURNS_MAJOR, np.nan),
        'poly9_MAJ': _poly9_ratio,
        'poly9pwr3_MAJ': last.get(FLD.POLYNOMIAL9_NORM_MAJOR, np.nan),
        'PROPHET_LB': last.get(FLD.FBPROPHET_LB, np.nan),
        'PROPHET_UB': last.get(FLD.FBPROPHET_UB, np.nan),
        'PROPHET_CLR_NORM': last.get(FLD.FBPROPHET_CLEARANCE_NORM, np.nan),
        'PROPHET_LB_MAJ': last.get(FLD.FBPROPHET_LB_MAJOR, np.nan),
        'PROPHET_UB_MAJ': last.get(FLD.FBPROPHET_UB_MAJOR, np.nan),
        'PROPHET_CLR_NORM_MAJ': last.get(FLD.FBPROPHET_CLEARANCE_NORM_MAJOR, np.nan),
    }


def coldstart_step(ctx: ReviewContext, targets: pd.DatetimeIndex) -> pd.DataFrame:
    """每个目标**日期**只用截至该日的数据算一行 —— 因果，且前缀稳定。

    ⚠️ Prophet 是全样本拟合：**不能**在整段帧上拟合后再回填历史行（会用到未来）。
    因此按目标日期逐个拟合；增量日更通常只有 1 个日期，代价可接受。
    """
    if targets is None or len(targets) == 0:
        return pd.DataFrame(columns=COLDSTART_COLUMNS)
    day_all = ctx.frame('day')

    dates = sorted({pd.Timestamp(t).normalize() for t in targets})
    cache = {}
    for d in dates:
        cache[d] = _coldstart_row(slice_asof(day_all, d))

    rows, idx = [], []
    for t in targets:
        d = pd.Timestamp(t).normalize()
        row = cache.get(d, {c: np.nan for c in COLDSTART_COLUMNS})
        rows.append([row.get(c, np.nan) for c in COLDSTART_COLUMNS])
        idx.append((pd.Timestamp(t), ctx.code))
    return pd.DataFrame(rows,
                        index=pd.MultiIndex.from_tuples(idx, names=['ts', 'code']),
                        columns=list(COLDSTART_COLUMNS))


def build_coldstart_steps() -> List[FeatureStep]:
    return [FeatureStep(
        name='coldstart', tier=TIER_DAILY, input_freq='day',
        output_family='60min', columns=COLDSTART_COLUMNS,
        func=coldstart_step, history='anchor')]
