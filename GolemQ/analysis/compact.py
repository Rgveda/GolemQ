# coding:utf-8
"""`calc_compact_features` —— XGBoost `xgb_scan` 的 37 个 derivable 特征。

来源：旧树 `features/empirical.py:3410-3790`（`calc_compact_features`）及其
`features/base.py` 里的辅助函数。**只搬能直接计算的部分**：

* 基础指标：CCI / RSI / MAXFACTOR / BOLL / MA90 / HMA / 回撤；
* MAPOWER30 / MTM / MTMMA / Poly9 / Magic Nine / 成交量归一；
* day / weekly / `_MAJ` 派生（**因果映射**：盘中 60min bar 用上一交易日，
  15:00 bar 才用当日）；
* `calc_portfolio_rank` 的 7 个 rank 列。

⚠️ **明确未搬**（返回 NaN，并在 :data:`DEFERRED_COLUMNS` 里列出）：
`MA90_CLR_RET_GAR`（依赖 ZEN peak checkout close）、`PeakLowLag`（依赖
`calc_peak_point_vX` 的 cluster-group 链）。

⚠️ 本模块是**纯函数**：不读库、不写库；输入/输出都是 DataFrame。
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd

from GolemQ.analysis.indicators import (TA_HMA, atr_rsi_stops, atr_supertrend,
                                        calc_macd_qa, cross_above)
from GolemQ.analysis.magic_nine_jit import calc_magic_nine_turns_np_vXs_jit
from GolemQ.analysis.peak import calc_peak_point_vX
from GolemQ.analysis.poly9 import (calc_polynomial9_norm_vXIs,
                                   calc_polynomial9_vXIs)
from GolemQ.analysis.quadrant_jit import mapower_np_vXIIs
from GolemQ.analysis.timeseries import weekly_ohlc
from GolemQ.analysis.timing import (Timeline_duration, calc_event_timing_lag,
                                    calc_feature_event_timing_lag)
from GolemQ.analysis.xgboost.rank import calc_portfolio_rank
from GolemQ.core.constants import AKA, FIELD, TREND_STATUS as ST
from GolemQ.models.alias import PFL

__all__ = ['calc_compact_features', 'DEFERRED_COLUMNS', 'COMPACT_COLUMNS']

#: 旧树 `calc_compact_features` 里存在、但本轮**未搬**的列。
#: 2026-10-21 用户定 `MA90_CLR_RET_GAR` 放弃引入 ⇒ 目前为空。
DEFERRED_COLUMNS = ()

#: 本模块会产出的列（37 个 derivable + rank 7 列）。
COMPACT_COLUMNS = (
    FIELD.CCI_NORM, FIELD.RSI, FIELD.MAXFACTOR,
    FIELD.MA90_CLEARANCE, FIELD.MA90_CLEARANCE_TIMING_LAG,
    FIELD.DRAWDOWN_RATIO,
    FIELD.VOLUME_NORM, FIELD.VOLUME_NORM_TIMING_LAG,
    FIELD.BOLL_DELTA, FIELD.BOLL_CHANNEL,
    FIELD.BOLL_LB_HMA5_TIMING_LAG_MAJOR,
    FIELD.MACD_ZERO_TIMING_LAG_MAJOR,
    FIELD.STOCK_SCORE_M15, FIELD.STOCK_SCORE_M15_NORM,
    FIELD.STOCK_SCORE_M15_NORM_TIMING_LAG,
    FIELD.MAGIC_NINE_TURNS_BEFORE, FIELD.MAGIC_NINE_TURNS_SX_BEFORE,
    FIELD.MAGIC_NINE_TURNS_BEFORE_MAJOR,
    FIELD.MAGIC_NINE_TURNS_SX_BEFORE_MAJOR,
    FIELD.MAPOWER30, FIELD.MAPOWER30_TIMING_LAG, FIELD.MAPOWER30_MAJOR,
    FIELD.MAPOWER30_TIMING_LAG_MAJOR, FIELD.MAPOWER30_MEDIAN,
    FIELD.MAPOWER30_MAJOR_MEDIAN,
    FIELD.MTM_NORM, FIELD.MTM_NORM_MAJOR, FIELD.MTM_NORM_WEEKLY,
    FIELD.MTMMA_NORM, FIELD.MTMMA_NORM_WEEKLY, FIELD.MTMMA_NORM_MAJOR,
    FIELD.POLYNOMIAL9_TIMING_LAG, FIELD.POLYNOMIAL9_TIMING_LAG_MAJOR,
    FIELD.POLYNOMIAL9_NORM, FIELD.POLYNOMIAL9_NORM_MAJOR,
    FIELD.PEAK_LOW_TIMING_LAG,
    FIELD.PEAK_HIGH_TIMING_LAG, FIELD.PEAK_TOWARDS_TIMING_LAG,
    FIELD.PEAK_TOWARDS_BEFORE, ST.CLUSTER_GROUP_TOWARDS, ST.PEAK_TOWARDS,
    FIELD.LINEAREG_PRICE, FIELD.LINEAREG_BAND_TIMING_LAG,
    FIELD.ATR_UB, FIELD.ATR_LB,
    # ATR Stopline / SuperTrend（2026-10-21 补；markup 标签侧输入）
    FIELD.ATR_Stopline, FIELD.ATR_Stopline_PRICE,
    FIELD.ATR_Stopline_TIMING_LAG,
    FIELD.ATR_SuperTrend, FIELD.ATR_SuperTrend_TIMING_LAG,
    FIELD.ATR_Stopline_TIMING_LAG_MAJOR,
    FIELD.ATR_SuperTrend_TIMING_LAG_MAJOR,
    FIELD.ATR_Stopline_PRICE_MAJOR,
    FIELD.ATR_Stopline_TIMING_LAG_WEEKLY,
    FIELD.ATR_SuperTrend_TIMING_LAG_WEEKLY,
    FIELD.ATR_Stopline_PRICE_WEEKLY,
    PFL.STOCK_PORTFOLIO_RANK, PFL.STOCK_PORTFOLIO_RETURNS,
    PFL.STOCK_PORTFOLIO_RANK_NEGATIVE, PFL.STOCK_PORTFOLIO_RANK_BEFORE,
    PFL.STOCK_PORTFOLIO_RANK_TRACING, PFL.STOCK_PORTFOLIO_RANK_METAPHASE,
    PFL.STOCK_PORTFOLIO_RANK_METAPHASE_MAJOR,
)


def _require(frame: pd.DataFrame, cols) -> None:
    missing = [c for c in cols if c not in frame.columns]
    if missing:
        raise ValueError('calc_compact_features 缺列：{}'.format(missing))


def _stock_score_sliding(close: np.ndarray, window: int = 15) -> np.ndarray:
    """15 根窗口的对数线性回归打分（旧 `calc_stock_socre_sliding` 的纯 numpy 版）。

    每年化收益 × R²；前 `window-1` 位 NaN。
    """
    close = np.asarray(close, dtype=np.float64)
    n = close.size
    out = np.full(n, np.nan)
    x = np.arange(window, dtype=np.float64)
    xm = x.mean()
    sxx = ((x - xm) ** 2).sum()
    with np.errstate(invalid='ignore', divide='ignore'):
        logc = np.log(close)
    for i in range(window - 1, n):
        y = logc[i - window + 1:i + 1]
        ym = y.mean()
        slope = ((x - xm) * (y - ym)).sum() / sxx
        intercept = ym - slope * xm
        annual = math.expm1(250.0 * slope)
        ss_res = ((y - (slope * x + intercept)) ** 2).sum()
        ss_tot = ((y - ym) ** 2).sum()
        out[i] = annual * (1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan
    return out


def _base_frame(ohlc: pd.DataFrame, annual: int) -> pd.DataFrame:
    """在一个频率上算全部可直接计算的列（不含 rank / day 映射）。"""
    _require(ohlc, (AKA.OPEN, AKA.HIGH, AKA.LOW, AKA.CLOSE, AKA.VOLUME))
    close = ohlc[AKA.CLOSE].astype(np.float64)
    high = ohlc[AKA.HIGH].astype(np.float64)
    low = ohlc[AKA.LOW].astype(np.float64)
    f = pd.DataFrame(index=ohlc.index)

    # ---- MA / HMA ----
    import talib
    f[FIELD.HMA5] = TA_HMA(close, 5)
    f[FIELD.HMA10] = TA_HMA(close, 10)
    f[FIELD.MA90] = talib.MA(close, 90)
    f[FIELD.MA90_SLOPE] = talib.LINEARREG_SLOPE(f[FIELD.MA90], timeperiod=14)

    # ---- MACD（QUANTAXIS 口径）----
    dif, dea, macd = calc_macd_qa(close)
    f[FIELD.DIF], f[FIELD.DEA], f[FIELD.MACD] = dif, dea, macd
    f[FIELD.MACD_CROSS_SX_BEFORE] = Timeline_duration(
        cross_above(dea, dif).to_numpy(np.int16)).astype(np.int32)
    f[FIELD.MACD_ZERO_TIMING_LAG] = calc_event_timing_lag(
        np.where(macd > 0, 1, np.where(macd < 0, -1, 0))).astype(np.int16)

    # ---- CCI / RSI / MAXFACTOR ----
    rsi = talib.RSI(close.to_numpy(), timeperiod=12)
    cci = talib.CCI(high.to_numpy(), low.to_numpy(), close.to_numpy(),
                    timeperiod=14)
    k, d = talib.STOCH(high.to_numpy(), low.to_numpy(), close.to_numpy(),
                       fastk_period=9, slowk_matype=0,
                       slowk_period=3, slowd_period=3)
    j = 3.0 * k - 2.0 * d
    f[FIELD.RSI] = rsi.astype(np.float32)
    f[FIELD.CCI] = cci.astype(np.float32)
    f[FIELD.CCI_NORM] = (cci / 600.0 + 0.5).astype(np.float32)
    f[FIELD.MAXFACTOR] = (cci + (rsi - 50.0) * 4.0
                          + (j - 50.0) * 4.0).astype(np.float32)

    # ---- BOLL ----
    up, mid, lb = talib.BBANDS(close.to_numpy(), timeperiod=20,
                               nbdevup=2, nbdevdn=2, matype=0)
    width = (up - lb) / mid
    f[FIELD.BOLL_UB] = up.astype(np.float32)
    f[FIELD.BOLL] = mid.astype(np.float32)
    f[FIELD.BOLL_LB] = lb.astype(np.float32)
    f[FIELD.BOLL_CHANNEL] = width.astype(np.float32)
    f[FIELD.BOLL_DELTA] = pd.Series(width, index=f.index).diff().astype(np.float32)

    # ---- MA90 clearance / timing ----
    f[FIELD.MA90_CLEARANCE] = (
        (f[FIELD.BOLL_UB] - f[FIELD.MA90]) / f[FIELD.BOLL]
        / f[FIELD.BOLL_CHANNEL]).astype(np.float32)
    f[FIELD.MA90_CLEARANCE_TIMING_LAG] = calc_event_timing_lag(
        np.where(f[FIELD.MA90_CLEARANCE] > 0, 1, -1)).astype(np.int16)
    # `bollLHma5Lag` 的 base（major 在 day 帧上再派生）
    f[FIELD.BOLL_LB_HMA5_TIMING_LAG] = calc_event_timing_lag(
        np.where(f[FIELD.BOLL_LB] > f[FIELD.HMA5], -1, 1)).astype(np.int16)

    # ---- ATR Stopline / SuperTrend ----
    stop_line, atr_stop = atr_rsi_stops(ohlc)
    f[FIELD.ATR_Stopline] = atr_stop
    f[FIELD.ATR_Stopline_PRICE] = stop_line
    f[FIELD.ATR_Stopline_TIMING_LAG] = calc_event_timing_lag(
        atr_stop).astype(np.int16)
    super_line, atr_super = atr_supertrend(ohlc)
    f[FIELD.ATR_SuperTrend] = atr_super
    f[FIELD.ATR_SuperTrend_TIMING_LAG] = calc_event_timing_lag(
        atr_super).astype(np.int16)

    # ---- MAPOWER30 / DRDN_RATIO ----
    mpwr = mapower_np_vXIIs(close.to_numpy())
    f[FIELD.MAPOWER30] = mpwr.astype(np.float32)
    f[FIELD.MAPOWER30_TIMING_LAG] = calc_event_timing_lag(
        np.where((pd.Series(mpwr).diff() > 0) | (mpwr > 0.999), 1, -1)
    ).astype(np.int16)
    prev = pd.Series(mpwr, index=f.index).shift(1)
    peak_hi = Timeline_duration(np.where(
        (prev > 0.999) & (mpwr < 0.99), 1,
        np.where((prev > mpwr) & (mpwr > 0.982), 1, 0))).astype(np.int16)
    f[FIELD.MAPOWER30_PEAK_HIGH_BEFORE] = peak_hi
    f[FIELD.PEAK_HIGH_BACKWARD] = pd.Series(
        np.where(peak_hi == 0,
                 np.maximum(high.shift(1), high.shift(3)), np.nan),
        index=f.index).ffill().astype(np.float32)
    f[FIELD.DRAWDOWN_RATIO] = (
        np.minimum(f[FIELD.HMA10], f[FIELD.HMA5])
        / f[FIELD.PEAK_HIGH_BACKWARD]).astype(np.float32)
    f[FIELD.MAPOWER30_MEDIAN] = (
        f[FIELD.MAPOWER30].expanding(min_periods=1).median().astype(np.float32))

    # ---- MTM / MTMMA ----
    mtm = talib.MOM(close.to_numpy(), timeperiod=12)
    f[FIELD.MTM] = mtm.astype(np.float32)
    f[FIELD.MTM_NORM] = mapower_np_vXIIs(mtm).astype(np.float32)
    mtmma = talib.MA(mtm, 6)
    f[FIELD.MTMMA] = mtmma.astype(np.float32)
    f[FIELD.MTMMA_NORM] = mapower_np_vXIIs(mtmma).astype(np.float32)

    # ---- Magic Nine Turns ----
    m9t = calc_magic_nine_turns_np_vXs_jit(
        np.array(close, dtype=np.float32, copy=True))
    f[FIELD.MAGIC_NINE_TURNS] = np.asarray(m9t, dtype=np.float32)
    f[FIELD.MAGIC_NINE_TURNS_MAJOR] = f[FIELD.MAGIC_NINE_TURNS]
    f[FIELD.MAGIC_NINE_TURNS_BEFORE] = Timeline_duration(
        np.where(f[FIELD.MAGIC_NINE_TURNS] > 0.00168, 1, 0)).astype(np.int16)
    f[FIELD.MAGIC_NINE_TURNS_SX_BEFORE] = Timeline_duration(
        np.where(f[FIELD.MAGIC_NINE_TURNS] < -0.00168, 1, 0)).astype(np.int16)

    # ---- Poly9 ----
    f = calc_polynomial9_vXIs(ohlc, features=f, annual=annual)
    f = calc_polynomial9_norm_vXIs(f, annual=annual)
    # 单帧的 major 版本：先占位，day 帧会在 `_derive_major` 覆盖。
    f[FIELD.POLYNOMIAL9_TIMING_LAG_MAJOR] = calc_event_timing_lag(
        np.where(f[FIELD.POLYNOMIAL9].diff(4) > 0, 1, -1)).astype(np.int16)
    f[FIELD.POLYNOMIAL9_NORM_MAJOR] = f[FIELD.POLYNOMIAL9_NORM]
    f[FIELD.MAPOWER30_MAJOR] = f[FIELD.MAPOWER30]
    f[FIELD.MAPOWER30_TIMING_LAG_MAJOR] = calc_event_timing_lag(
        np.where(f[FIELD.MAPOWER30_TIMING_LAG] > 0, 1, -1)).astype(np.int16)
    f[FIELD.MAPOWER30_MAJOR_MEDIAN] = f[FIELD.MAPOWER30_MEDIAN]
    f[FIELD.MTM_NORM_MAJOR] = f[FIELD.MTM_NORM]
    f[FIELD.MTMMA_NORM_MAJOR] = f[FIELD.MTMMA_NORM]
    f[FIELD.MAGIC_NINE_TURNS_BEFORE_MAJOR] = f[FIELD.MAGIC_NINE_TURNS_BEFORE]
    f[FIELD.MAGIC_NINE_TURNS_SX_BEFORE_MAJOR] = f[FIELD.MAGIC_NINE_TURNS_SX_BEFORE]
    f[FIELD.BOLL_LB_HMA5_TIMING_LAG_MAJOR] = calc_event_timing_lag(
        np.where(f[FIELD.BOLL_LB_HMA5_TIMING_LAG] > 0, 1, -1)).astype(np.int16)
    f[FIELD.MACD_ZERO_TIMING_LAG_MAJOR] = calc_event_timing_lag(
        np.where(f[FIELD.MACD_ZERO_TIMING_LAG] > 0, 1, -1)).astype(np.int16)

    # ---- Stock score / VOLUME_NORM ----
    f[FIELD.STOCK_SCORE_M15] = _stock_score_sliding(close.to_numpy()).astype(np.float32)
    f[FIELD.STOCK_SCORE_M15_NORM] = mapower_np_vXIIs(
        f[FIELD.STOCK_SCORE_M15].fillna(0).to_numpy()).astype(np.float32)
    f[FIELD.STOCK_SCORE_M15_NORM_TIMING_LAG] = calc_feature_event_timing_lag(
        f, FIELD.STOCK_SCORE_M15_NORM).astype(np.int16)
    f[FIELD.VOLUME_NORM] = mapower_np_vXIIs(
        ohlc[AKA.VOLUME].to_numpy(dtype=np.float32)).astype(np.float32)
    f[FIELD.VOLUME_NORM_TIMING_LAG] = calc_feature_event_timing_lag(
        f, FIELD.VOLUME_NORM).astype(np.int16)

    # ---- Peak point（含 LINEAREG band / PEAK_LOW_TIMING_LAG）----
    f = calc_peak_point_vX(ohlc, features=f)

    # ---- Deferred ----
    for col in DEFERRED_COLUMNS:
        f[col] = np.nan
    return f


def _weekly_derived(ohlc_day: pd.DataFrame) -> pd.DataFrame:
    """周线 MTM/MTMMA 归一（旧 `merge_features_week_*` 的副本语义）。"""
    weekly = weekly_ohlc(ohlc_day)
    if weekly is None or len(weekly) < 13:
        return pd.DataFrame(index=ohlc_day.index)
    import talib
    close = weekly[AKA.CLOSE].astype(np.float64)
    mtm = talib.MOM(close.to_numpy(), timeperiod=12)
    mtmma = talib.MA(mtm, 6)
    out = pd.DataFrame(index=weekly.index)
    out[FIELD.MTM_NORM_WEEKLY] = mapower_np_vXIIs(mtm).astype(np.float32)
    out[FIELD.MTMMA_NORM_WEEKLY] = mapower_np_vXIIs(mtmma).astype(np.float32)

    stop_line, atr_stop = atr_rsi_stops(weekly)
    super_line, atr_super = atr_supertrend(weekly)
    out[FIELD.ATR_Stopline_TIMING_LAG_WEEKLY] = calc_event_timing_lag(
        atr_stop).astype(np.int16)
    out[FIELD.ATR_SuperTrend_TIMING_LAG_WEEKLY] = calc_event_timing_lag(
        atr_super).astype(np.int16)
    out[FIELD.ATR_Stopline_PRICE_WEEKLY] = stop_line.astype(np.float16).round(3)
    return out


def _map_causal(day_values: pd.Series, target_index: pd.MultiIndex) -> pd.Series:
    """把日线值因果映射到 60min 索引。

    盘中 bar（< 15:00）用**上一交易日**的值；15:00 bar 才用当日值。
    """
    if day_values is None or len(day_values) == 0:
        return pd.Series(np.nan, index=target_index)
    day_values = day_values.copy()
    day_values.index = pd.DatetimeIndex(pd.to_datetime(
        day_values.index.get_level_values(0) if isinstance(day_values.index, pd.MultiIndex)
        else day_values.index)).normalize()
    day_values = day_values[~day_values.index.duplicated(keep='last')].sort_index()
    ts = pd.DatetimeIndex(pd.to_datetime(target_index.get_level_values(0)))
    use_today = ts >= ts.normalize() + pd.Timedelta(hours=15)
    lookup_dates = pd.DatetimeIndex(
        np.where(use_today, ts.normalize(), ts.normalize() - pd.Timedelta(days=1)))
    mapped = day_values.reindex(lookup_dates, method='ffill')
    mapped.index = target_index
    return mapped


def _derive_major(ohlc_day: pd.DataFrame, target_index: pd.MultiIndex) -> pd.DataFrame:
    """日线帧上派生 `_Maj` / 周线列，并按因果映射回 60min。"""
    day = _base_frame(ohlc_day, annual=252)
    day[FIELD.MAGIC_NINE_TURNS_MAJOR] = day[FIELD.MAGIC_NINE_TURNS]
    day[FIELD.MAGIC_NINE_TURNS_BEFORE_MAJOR] = Timeline_duration(
        np.where(day[FIELD.MAGIC_NINE_TURNS] > 0.00168, 1, 0)).astype(np.int16)
    day[FIELD.MAGIC_NINE_TURNS_SX_BEFORE_MAJOR] = Timeline_duration(
        np.where(day[FIELD.MAGIC_NINE_TURNS] < -0.00168, 1, 0)).astype(np.int16)
    day[FIELD.MAPOWER30_MAJOR] = day[FIELD.MAPOWER30]
    day[FIELD.MAPOWER30_TIMING_LAG_MAJOR] = calc_event_timing_lag(
        np.where(day[FIELD.MAPOWER30_TIMING_LAG] > 0, 1, -1)).astype(np.int16)
    day[FIELD.MAPOWER30_MAJOR_MEDIAN] = day[FIELD.MAPOWER30_MEDIAN]
    day[FIELD.MTM_NORM_MAJOR] = day[FIELD.MTM_NORM]
    day[FIELD.MTMMA_NORM_MAJOR] = day[FIELD.MTMMA_NORM]
    day[FIELD.POLYNOMIAL9_TIMING_LAG_MAJOR] = calc_event_timing_lag(
        np.where(day[FIELD.POLYNOMIAL9].diff(4) > 0, 1, -1)).astype(np.int16)
    day[FIELD.POLYNOMIAL9_NORM_MAJOR] = day[FIELD.POLYNOMIAL9_NORM]
    day[FIELD.BOLL_LB_HMA5_TIMING_LAG_MAJOR] = calc_event_timing_lag(
        np.where(day[FIELD.BOLL_LB_HMA5_TIMING_LAG] > 0, 1, -1)).astype(np.int16)
    day[FIELD.MACD_ZERO_TIMING_LAG_MAJOR] = calc_event_timing_lag(
        np.where(day[FIELD.MACD_ZERO_TIMING_LAG] > 0, 1, -1)).astype(np.int16)
    day[FIELD.MACD_CROSS_SX_BEFORE_MAJOR] = Timeline_duration(
        np.where(day[FIELD.MACD_CROSS_SX_BEFORE].diff(1) < 0, 1, 0)).astype(np.int32)
    day[FIELD.ATR_Stopline_TIMING_LAG_MAJOR] = calc_event_timing_lag(
        np.where(day[FIELD.ATR_Stopline_TIMING_LAG] > 0, 1, -1)).astype(np.int16)
    day[FIELD.ATR_SuperTrend_TIMING_LAG_MAJOR] = calc_event_timing_lag(
        np.where(day[FIELD.ATR_SuperTrend_TIMING_LAG] > 0, 1, -1)).astype(np.int16)
    day[FIELD.ATR_Stopline_PRICE_MAJOR] = day[FIELD.ATR_Stopline_PRICE]

    cols = [
        FIELD.CCI_NORM, FIELD.RSI, FIELD.MAXFACTOR, FIELD.MA90_CLEARANCE,
        FIELD.MA90_CLEARANCE_TIMING_LAG, FIELD.DRAWDOWN_RATIO,
        FIELD.VOLUME_NORM, FIELD.VOLUME_NORM_TIMING_LAG,
        FIELD.BOLL_DELTA, FIELD.BOLL_CHANNEL,
        FIELD.BOLL_LB_HMA5_TIMING_LAG_MAJOR,
        FIELD.MACD_ZERO_TIMING_LAG_MAJOR,
        FIELD.MACD_CROSS_SX_BEFORE_MAJOR,
        FIELD.STOCK_SCORE_M15, FIELD.STOCK_SCORE_M15_NORM,
        FIELD.STOCK_SCORE_M15_NORM_TIMING_LAG,
        FIELD.MAGIC_NINE_TURNS_MAJOR,
        FIELD.MAGIC_NINE_TURNS_BEFORE_MAJOR,
        FIELD.MAGIC_NINE_TURNS_SX_BEFORE_MAJOR,
        FIELD.MAPOWER30_MAJOR, FIELD.MAPOWER30_TIMING_LAG_MAJOR,
        FIELD.MAPOWER30_MAJOR_MEDIAN,
        FIELD.MTM_NORM_MAJOR, FIELD.MTMMA_NORM_MAJOR,
        FIELD.POLYNOMIAL9_TIMING_LAG_MAJOR, FIELD.POLYNOMIAL9_NORM_MAJOR,
        FIELD.ATR_Stopline_TIMING_LAG_MAJOR,
        FIELD.ATR_SuperTrend_TIMING_LAG_MAJOR,
        FIELD.ATR_Stopline_PRICE_MAJOR,
    ]
    out = pd.DataFrame(index=target_index)
    for col in cols:
        out[col] = _map_causal(day[col], target_index)

    weekly = _weekly_derived(ohlc_day)
    for col in (FIELD.MTM_NORM_WEEKLY, FIELD.MTMMA_NORM_WEEKLY,
                FIELD.ATR_Stopline_TIMING_LAG_WEEKLY,
                FIELD.ATR_SuperTrend_TIMING_LAG_WEEKLY,
                FIELD.ATR_Stopline_PRICE_WEEKLY):
        if col in weekly.columns:
            out[col] = _map_causal(weekly[col], target_index)
        else:
            out[col] = np.nan
    return out


def calc_compact_features(ohlc_data: pd.DataFrame,
                          ohlc_day: Optional[pd.DataFrame] = None,
                          annual: int = None,
                          include_rank: bool = True,
                          verbose: bool = False) -> pd.DataFrame:
    """60min（或任意单帧）的 compact 特征；可选叠加 day 周期 / 周线派生。

    :param ohlc_data: 主帧（默认 60min），MultiIndex `(ts, code)`，含 OHLCV。
    :param ohlc_day: 可选的日线帧；给了就用它派生 day / `_MAJ` / weekly 列
        并按**因果**（盘中用上一交易日、15:00 用当日）映射回主帧。
    :param annual: 传给 Poly9 等闸；``None`` ⇒ 主帧 1008、day 帧 252。
    :param include_rank: 是否调用 :func:`calc_portfolio_rank` 产出 rank 族。
    :returns: 主帧 + 全部 compact 列（含 rank）；入参不被修改。
    :raises ValueError: 缺 OHLCV 列，或 rank 所需的 MACD 死叉列缺失。
    """
    annual = 1008 if annual is None else int(annual)
    out = _base_frame(ohlc_data, annual=annual)
    if ohlc_day is not None and len(ohlc_day) > 0:
        major = _derive_major(ohlc_day, ohlc_data.index)
        for col in major.columns:
            out[col] = major[col]
    else:
        out[FIELD.MACD_CROSS_SX_BEFORE_MAJOR] = Timeline_duration(
            np.where(pd.Series(out[FIELD.MACD_CROSS_SX_BEFORE]).diff(1) < 0,
                     1, 0)).astype(np.int32)

    # 单帧模式没有的周线列 / 延迟列：补 NaN，保证 37 列契约完整。
    for col in COMPACT_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan

    # 保留 OHLCV（rank 需要 close；落库/下游也常用）。
    for col in (AKA.OPEN, AKA.HIGH, AKA.LOW, AKA.CLOSE, AKA.VOLUME, AKA.AMOUNT):
        if col in ohlc_data.columns:
            out[col] = ohlc_data[col]

    # 重新索引回主帧（`_derive_major` 已按 target_index 建索引）
    out = out.reindex(ohlc_data.index)
    if include_rank:
        ranked = calc_portfolio_rank(out, verbose=verbose)
        for col in (PFL.STOCK_PORTFOLIO_RANK, PFL.STOCK_PORTFOLIO_RETURNS,
                    PFL.STOCK_PORTFOLIO_RANK_NEGATIVE,
                    PFL.STOCK_PORTFOLIO_RANK_BEFORE,
                    PFL.STOCK_PORTFOLIO_RANK_TRACING,
                    PFL.STOCK_PORTFOLIO_RANK_METAPHASE,
                    PFL.STOCK_PORTFOLIO_RANK_METAPHASE_MAJOR):
            out[col] = ranked[col]
    if verbose:
        print('[compact] {} bars / {} cols'.format(len(out), len(out.columns)))
    return out
