# coding:utf-8
"""四量图（`calc_four_quadrant_boost`）与四象限体系的特征计算 —— 逐字搬自老树。

从哪来 / 搬了什么
================
老树 `GolemQ_old/analysis/base.py` 的 `calc_four_quadrant_boost`（119-696，~580 行）
是「四量图」的真身 —— ⚠️ **它不是画图，是特征计算**：Q1–Q4（四条均线结构打分）、
`4QPwrln` 电力线（8 条 MA 的钝化计数 −3.8）、一串 quadrant timing-lag，以及
`QuadSt` / `ChoBits` / `QuadCodex` 状态机。「四量图」这个词只活在老树 docstring 里。

本模块另外搬了用户点名的 6 个下游指标（老树在别处算、新树一处都没有）：

======================  ==========================================  ==========================
产出列                    函数                                          老树来源
======================  ==========================================  ==========================
`QLevMACDLagD`          :func:`calc_quadrant_leverage_macd_timing_lag`  `analysis/base.py:2171-2188`
`ZenPivotLag`           :func:`calc_pivot_timing_lag`                 `analysis/zen_pivot.py:859`
`QuadBsTukLag`          :func:`calc_quadrant_bs_trunk_timing_lag`      `analysis/quad_gear.py:648`
`MsLeadinBf`/`x3`/`x5`  :func:`calc_mainstream_feats`                 `analysis/base.py:2638-2676`
6 个 `BS*Bf*`            :func:`calc_bs_before`                        `base.py:2488` + `models/mainstream.py:1417`
`BSStarDwn`             :func:`calc_bs_star_down`                     `base.py:2429`（**只搬星线段**）
`4QuadStDwnCnt`         :func:`calc_quadrant_stardown_count`          `models/mainstream.py:515`
`QLevPwrlnUprLag`       :func:`calc_quadrant_powerline_uprisal_lag`   `models/mainstream.py:530` + `:544`（**两条**）
`4QuadZenDCnt`          :func:`calc_quadrant_zen_dash_count`          `analysis/mainstream.py:180`
======================  ==========================================  ==========================

解耦动作（**改动影响可对拍性，逐条记**）
===================================
1. **丢掉了 `try/except` 数据修复序言**（老树 158-247）。它**不是补数据** —— 是
   `GQ_SU_remove_stock_min_adv(...)` 把疑似**停牌伪 0** 的 bar 从 `stock_min` **移走**
   到 `stock_min_removed`（**破坏性 Mongo 写**），外加 `get_kline_price_min` 读库与一堆 print。
   新树**已经清过**这次（`markets/StockCN/maintenance.py::GQ_purge_suspended` 移走
   3,908,747 行，`GQ_suspension_dates` 现返回 0）⇒ 照搬是「破坏性写 + 常态 no-op」。
   **故本模块把「数据已清理」当既定前置条件，函数只做算术。**
   ⚠️ **行为差异**：老树在 `len(features) != len(ohlc_data)` 时**先修复再算**；
   这里直接算 —— 长度不等会由 numpy 广播/报错暴露，**不静默**。
2. `market_type=QA.MARKET_TYPE.STOCK_CN` → 新树 `core.constants.MARKET_TYPE.STOCK_CN`；
   删掉 `import QUANTAXIS` 与 `from QUANTAXIS.QAUtil import DATABASE`。
3. 删掉 `from GolemQ.fractal.v0 import kdj_cross_func` —— 实测该名在老树 `base.py`
   **只在 import 行出现、函数体从未调用**，新树也无 `fractal/` 包。

⚠️ 老树的平行实现（**只搬了这一条**，见 `DECISIONS.md` D35「只搬一条」）
- 四量图另有 polars 版（`GolemQ_old/models/polars.py:461`）与 notebook 版 —— **未搬**。
- `QuadBsTukLag` 老树有三份副本（`quad_gear.py:648` / `zen_pivot.py:603` / `:1047`）——
  只搬 `quad_gear.py` 的规范版。

⚠️ **本模块不连库、不写库、不被任何管线调用**（用户 2026-10-11：「尽可能解耦，
越少增加引用和管线越好」）。它是**库代码**，消费方自便。
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import talib

from GolemQ.analysis.timing import Timeline_duration, calc_energy, calc_event_timing_lag
from GolemQ.core.constants import AKA
from GolemQ.core.constants import FIELD as FLD
from GolemQ.core.constants import MARKET_TYPE
from GolemQ.core.constants import STATE as STE
from GolemQ.models.alias import LTT, TRD, ZEN

__all__ = [
    'clearup_compact_dataframe_float32',
    'clearup_compact_dataframe_int16',
    'sma',
    'ema',
    'llv',
    'kdj',
    'calc_four_quadrant_boost',
    'calc_quadrant_leverage_macd_timing_lag',
    'calc_bs_signals',
    'calc_bs_star_down',
    'calc_quadrant_stardown_count',
    'calc_quadrant_powerline_uprisal_lag',
    'calc_quadrant_zen_dash_count',
    'calc_quadrant_bs_trunk_timing_lag',
    'calc_mainstream_feats',
    'calc_bs_before',
    'calc_pivot_timing_lag',
]


def clearup_compact_dataframe_float32(features_dummy: pd.DataFrame,
                                      check_columns: list = None) -> pd.DataFrame:
    """把 `check_columns` 里的 float64 列压成 float32。

    逐字搬自老树 `features/base.py:174`，行为不变：**只动 float64**、不动 int、
    跳过 `created_at` / `date_stamp` / `time_stamp` 三个时间戳列。
    """
    check_columns = check_columns or []
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        for column in check_columns:
            if (column in features_dummy.columns) and \
                ((features_dummy[column].dtype == np.int32) or
                 (features_dummy[column].dtype == np.int64) or
                 (features_dummy[column].dtype == np.float64)) and \
                (column not in ['created_at', 'date_stamp', 'time_stamp']):
                if (features_dummy[column].dtype == np.float64):
                    features_dummy[column] = features_dummy[column].astype(np.float32)
    return features_dummy


def clearup_compact_dataframe_int16(features_dummy: pd.DataFrame,
                                    check_columns: list = None) -> pd.DataFrame:
    """把 `check_columns` 里的整型列压成 int16。

    逐字搬自老树 `features/base.py:144`：填充值 `32767`、`暂无 → NaN` 的转换都照搬。
    """
    check_columns = check_columns or []
    for column in check_columns:
        if (column in features_dummy.columns) and \
            ((features_dummy[column].dtype == np.int32) or
             (features_dummy[column].dtype == np.int64) or
             (features_dummy[column].dtype == np.float64)) and \
            (column not in ['created_at', 'date_stamp', 'time_stamp']):
            if ((features_dummy[column].dtype == np.float64) or
                (features_dummy[column].dtype == np.int32) or
                (features_dummy[column].dtype == np.int64)):
                features_dummy[column] = (features_dummy[column]
                    .fillna(32767).replace([np.inf, -np.inf], 32767)
                    .astype(np.int16))
            elif (features_dummy[column].dtype == object):
                features_dummy[column] = (pd.to_numeric(
                    features_dummy[column].replace(u'暂无', np.nan),
                    errors='coerce')
                    .fillna(32767).replace([np.inf, -np.inf], 32767)
                    .astype(np.int16))
    return features_dummy


def sma(series, n, m):
    """SMA 指标 —— 逐字搬自老树 `analysis/base.py:2519`（含其递归定义与 NaN 处理）。

    >>> sma(pd.Series([1.0, 2.0, 3.0]), 2, 1).tolist()
    [1.0, 1.5, 2.25]
    """
    result = np.zeros(len(series))
    if len(series) == 0:
        return result
    result[0] = series.iloc[0] if not pd.isna(series.iloc[0]) else 0
    for i in range(1, len(series)):
        if pd.isna(series.iloc[i]):
            result[i] = result[i - 1]
        else:
            result[i] = (m * series.iloc[i] + (n - m) * result[i - 1]) / n
    return pd.Series(result, index=series.index)


def ema(series, n):
    """EMA（`ewm(span=n, adjust=False)`）—— 逐字搬自老树 `base.py:2540`。

    >>> [round(v, 4) for v in ema(pd.Series([1.0, 2.0, 3.0]), 2)]
    [1.0, 1.6667, 2.5556]
    """
    return series.ewm(span=n, adjust=False).mean()


def llv(series, n):
    """滚动最低值（`min_periods=1`）—— 逐字搬自老树 `base.py:2545`。

    >>> llv(pd.Series([3.0, 1.0, 2.0]), 2).tolist()
    [3.0, 1.0, 1.0]
    """
    return series.rolling(window=n, min_periods=1).min()


def kdj(hight, low, close, fastk_period=9, slowk_matype=0, slowk_period=3, slowd_period=3):
    """KDJ → `np.c_[K, D, J]` —— 逐字搬自老树 `base.py:2550`（走 `talib.STOCH`）。"""
    K, D = talib.STOCH(
        hight, low, close,
        fastk_period=fastk_period, slowk_matype=slowk_matype,
        slowk_period=slowk_period, slowd_period=slowd_period)
    J = 3 * K - 2 * D
    return np.c_[K, D, J]


def calc_four_quadrant_boost(
    code: str,
    features: pd.DataFrame = None,
    ohlc_data: pd.DataFrame = None,
    annual: int = 1008,
    market_type=MARKET_TYPE.STOCK_CN,
):
    """
    最牛四量图，
    """
    fourQuad_columns = [
        LTT.QUADRANT_1,
        LTT.QUADRANT_2,
        LTT.QUADRANT_3,
        LTT.QUADRANT_4,
        LTT.QUADRANT_POWERLINE,
        LTT.ONE_QUADRANT_UPRISING_TIMING_LAG,
        LTT.TWO_QUADRANT_UPRISING_TIMING_LAG,
        LTT.TWO_QUADRANT_DOWNRISK_TIMING_LAG,
        LTT.THREE_QUADRANT_UPRISING_TIMING_LAG,
        LTT.THREE_QUADRANT_DOWNRISK_TIMING_LAG,
        LTT.FOUR_QUADRANT_UPRISING_TIMING_LAG,
        LTT.QUADRANT_POWERLINE_LEADIN,
        LTT.QUADRANT_POWERLINE_LEADOUT,
        LTT.QUADRANT_POWERLINE_LEADIN_BEFORE,
        LTT.QUADRANT_POWERLINE_LEADOUT_BEFORE,]
    features = features.reindex(
        columns=list(set([
            *features.columns,
            *fourQuad_columns])))
    
    mid = (3*ohlc_data[AKA.CLOSE]+ohlc_data[AKA.LOW]+ohlc_data[AKA.OPEN]+ohlc_data[AKA.HIGH])/6
    
    bullish = (20*mid+19*mid.shift(1)+18*mid.shift(2)+17*mid.shift(3)+16*mid.shift(4)+15*mid.shift(5)+\
            14*mid.shift(6)+13*mid.shift(7)+12*mid.shift(8)+11*mid.shift(9)+10*mid.shift(10)+\
            9*mid.shift(11)+8*mid.shift(12)+7*mid.shift(13)+6*mid.shift(14)+5*mid.shift(15)+\
            4*mid.shift(16)+3*mid.shift(17)+2*mid.shift(18)+mid.shift(20))/210
    
    mars_line = talib.MA(bullish, 6)
    features[LTT.QUADRANT_1] = np.where(
        (ohlc_data[AKA.CLOSE] > bullish), 1,
        np.where(
            (ohlc_data[AKA.CLOSE] < mars_line), -1, 
            np.where(
                (bullish >= ohlc_data[AKA.CLOSE]) & \
                (ohlc_data[AKA.CLOSE] >= mars_line), -2, 0)))

    if (FLD.MA60 not in features.columns):
        ma60=talib.MA(ohlc_data[AKA.CLOSE], 60)
        ma120=talib.MA(ohlc_data[AKA.CLOSE], 120)
    else:
        ma60=features[FLD.MA60]
        ma120=features[FLD.MA120]
    a=np.maximum(ma60,
                ma120,)
    b=np.minimum(ma60,
                ma120,)
    features[LTT.QUADRANT_2]=np.where((ohlc_data[AKA.CLOSE]>=a), 1, 
                        np.where((ohlc_data[AKA.CLOSE]<=b), -1, 
                                 np.where((a>=ohlc_data[AKA.CLOSE]) & \
                                          (ohlc_data[AKA.CLOSE]>=b), -2, 0)))

    ma45=talib.MA(ohlc_data[AKA.CLOSE], 45)
    if (FLD.MA90 not in features.columns):    
        ma90=talib.MA(ohlc_data[AKA.CLOSE], 90)
    else:
        ma90=features[FLD.MA90]
    r=0
    e=np.maximum(ma45,
                ma90,)
    f=np.minimum(ma45,
                ma90,)
    features[LTT.QUADRANT_3]=np.where((ohlc_data[AKA.CLOSE]>e), 1, 
                        np.where((ohlc_data[AKA.CLOSE]<f), -1, 
                                 np.where((e>=ohlc_data[AKA.CLOSE]) & \
                                          (ohlc_data[AKA.CLOSE]>=f), -2, 0)))
    
    if (FLD.MA30 not in features.columns):
        ma30=talib.MA(ohlc_data[AKA.CLOSE], 30)
    else:
        ma30=features[FLD.MA30]
    j=np.maximum(ma30,
                ma60,)
    k=np.minimum(ma30,
                ma60,)
    features[LTT.QUADRANT_4]=np.where((ohlc_data[AKA.CLOSE]>j), 1, 
                        np.where((ohlc_data[AKA.CLOSE]<k), -1, 
                                 np.where((j>=ohlc_data[AKA.CLOSE]) & \
                                          (ohlc_data[AKA.CLOSE]>=k), -2, 0)))

    '''/////////////实验////////,nodraw////////
    ///////·强弱钝化线·///////'''
    ma5=talib.MA(ohlc_data[AKA.CLOSE], 5)
    ma13=talib.MA(ohlc_data[AKA.CLOSE], 13)
    ma21=talib.MA(ohlc_data[AKA.CLOSE], 21)
    ma34=talib.MA(ohlc_data[AKA.CLOSE], 34)
    ma55=talib.MA(ohlc_data[AKA.CLOSE], 55)
    ma89=talib.MA(ohlc_data[AKA.CLOSE], 89)
    ma144=talib.MA(ohlc_data[AKA.CLOSE], 144)
    ma233=talib.MA(ohlc_data[AKA.CLOSE], 233)
    tmp1=np.where((ohlc_data[AKA.CLOSE]>ma5),1,0)
    tmp2=np.where((ohlc_data[AKA.CLOSE]>ma13),1,0)
    tmp3=np.where((ohlc_data[AKA.CLOSE]>ma21),1,0)
    tmp4=np.where((ohlc_data[AKA.CLOSE]>ma34),1,0)
    tmp5=np.where((ohlc_data[AKA.CLOSE]>ma55),1,0)
    tmp6=np.where((ohlc_data[AKA.CLOSE]>ma89),1,0)
    tmp7=np.where((ohlc_data[AKA.CLOSE]>ma144),1,0)
    tmp8=np.where((ohlc_data[AKA.CLOSE]>ma233),1,0)
    features[LTT.QUADRANT_POWERLINE]=tmp1+tmp2+tmp3+tmp4+tmp5+tmp6+tmp7+tmp8+-3.8

    features[LTT.ONE_QUADRANT_UPRISING_TIMING_LAG]=calc_event_timing_lag(np.where(((features[LTT.QUADRANT_1]>0.000168) | \
                                                                                  (features[LTT.QUADRANT_2]>0.000168) | \
                                                                                  (features[LTT.QUADRANT_3]>0.000168) | \
                                                                                  (features[LTT.QUADRANT_4]>0.000168)), 1, -1))
    features[LTT.TWO_QUADRANT_UPRISING_TIMING_LAG]=calc_event_timing_lag(np.where(((np.where((features[LTT.QUADRANT_1]>0.000168), 1, 0)+\
                                                                                  np.where((features[LTT.QUADRANT_2]>0.000168), 1, 0)+\
                                                                                  np.where((features[LTT.QUADRANT_3]>0.000168), 1, 0)+\
                                                                                  np.where((features[LTT.QUADRANT_4]>0.000168), 1, 0))>1.68), 1, -1))
    features[LTT.TWO_QUADRANT_DOWNRISK_TIMING_LAG]=calc_event_timing_lag(np.where(((np.where((features[LTT.QUADRANT_1]>0.000168), 0, 1)+\
                                                                                  np.where((features[LTT.QUADRANT_2]>0.000168), 0, 1)+\
                                                                                  np.where((features[LTT.QUADRANT_3]>0.000168), 0, 1)+\
                                                                                  np.where((features[LTT.QUADRANT_4]>0.000168), 0, 1))>1.68), -1, 1))
    features[LTT.THREE_QUADRANT_UPRISING_TIMING_LAG]=calc_event_timing_lag(np.where(((np.where((features[LTT.QUADRANT_1]>0.000168), 1, 0)+\
                                                                                  np.where((features[LTT.QUADRANT_2]>0.000168), 1, 0)+\
                                                                                  np.where((features[LTT.QUADRANT_3]>0.000168), 1, 0)+\
                                                                                  np.where((features[LTT.QUADRANT_4]>0.000168), 1, 0))>2.68), 1, -1))
    features[LTT.THREE_QUADRANT_DOWNRISK_TIMING_LAG]=calc_event_timing_lag(np.where(((np.where((features[LTT.QUADRANT_1]>0.000168), 0, 1)+\
                                                                                  np.where((features[LTT.QUADRANT_2]>0.000168), 0, 1)+\
                                                                                  np.where((features[LTT.QUADRANT_3]>0.000168), 0, 1)+\
                                                                                  np.where((features[LTT.QUADRANT_4]>0.000168), 0, 1))>2.68), -1, 1))
    features[LTT.FOUR_QUADRANT_UPRISING_TIMING_LAG]=calc_event_timing_lag(np.where(((np.where((features[LTT.QUADRANT_1]>0.000168), 1, 0)+\
                                                                                  np.where((features[LTT.QUADRANT_2]>0.000168), 1, 0)+\
                                                                                  np.where((features[LTT.QUADRANT_3]>0.000168), 1, 0)+\
                                                                                  np.where((features[LTT.QUADRANT_4]>0.000168), 1, 0))>3.82), 1, -1))
    
    features = features.reindex(columns=list(set([
        *features.columns,
        *[LTT.QUADRANT_DOWNRISK_TIMING_LAG,]])))
    features[LTT.QUADRANT_DOWNRISK_TIMING_LAG] = calc_event_timing_lag(
        np.where(
            ((features[LTT.QUADRANT_1] < 0.000168) & \
            (features[LTT.QUADRANT_1].shift(1) < 0.000168) & \
            (features[LTT.QUADRANT_POWERLINE].diff(3)<-0.000168)) | \
            ((features[LTT.QUADRANT_1] < 0.000168) & \
            (features[LTT.QUADRANT_1].shift(1) < 0.000168) & \
            (features[LTT.QUADRANT_1].shift(2) < 0.000168) & \
            (features[LTT.QUADRANT_POWERLINE].diff(2)<-0.000168)) | \
            ((features[LTT.QUADRANT_2] < 0.000168) & \
            (features[LTT.QUADRANT_2].shift(1) < 0.000168) & \
            (features[LTT.QUADRANT_2].shift(2) < 0.000168) & \
            (features[LTT.TWO_QUADRANT_DOWNRISK_TIMING_LAG] < -3.82)) | \
            ((features[LTT.QUADRANT_3] < 0.000168) & \
            (features[LTT.QUADRANT_3].shift(1) < 0.000168) & \
            (features[LTT.QUADRANT_3].shift(2) < 0.000168) & \
            (features[LTT.TWO_QUADRANT_DOWNRISK_TIMING_LAG] < -3.82)) | \
            ((features[LTT.QUADRANT_4] < 0.000168) & \
            (features[LTT.QUADRANT_4].shift(1) < 0.000168) & \
            (features[LTT.QUADRANT_4].shift(2) < 0.000168) & \
            (features[LTT.TWO_QUADRANT_DOWNRISK_TIMING_LAG] < -3.82)) | \
            ((features[LTT.QUADRANT_1] < 0.000168) & \
            (features[LTT.QUADRANT_2] < 0.000168) & \
            ((features[LTT.QUADRANT_1].shift(1) < 0.000168) | \
            (features[LTT.QUADRANT_2].shift(1) < 0.000168))) | \
            ((features[LTT.QUADRANT_2] < 0.000168) & \
            (features[LTT.QUADRANT_3] < 0.000168) & \
            ((features[LTT.QUADRANT_2].shift(1) < 0.000168) | \
            (features[LTT.QUADRANT_3].shift(1) < 0.000168))) | \
            ((features[LTT.QUADRANT_3] < 0.000168) & \
            (features[LTT.QUADRANT_4] < 0.000168) & \
            ((features[LTT.QUADRANT_3].shift(1) < 0.000168) | \
            (features[LTT.QUADRANT_4].shift(1) < 0.000168))) | \
            ((features[LTT.QUADRANT_1] < 0.000168) & \
            (features[LTT.QUADRANT_2] < 0.000168) & \
            (features[LTT.QUADRANT_3] < 0.000168) & \
            ((features[LTT.QUADRANT_1].shift(1) < 0.000168) | \
            (features[LTT.QUADRANT_2].shift(1) < 0.000168) | \
            (features[LTT.QUADRANT_3].shift(1) < 0.000168))) | \
            ((features[LTT.QUADRANT_2] < 0.000168) & \
            (features[LTT.QUADRANT_3] < 0.000168) & \
            (features[LTT.QUADRANT_4] < 0.000168) & \
            ((features[LTT.QUADRANT_2].shift(1) < 0.000168) | \
            (features[LTT.QUADRANT_3].shift(1) < 0.000168) | \
            (features[LTT.QUADRANT_4].shift(1) < 0.000168))), -1, 1))
    
    if (annual > 512) and (TRD.AROON_UPRISING_TIMING_LAG in features.columns):
        virtual_pressure_line = np.where(
            (features[LTT.ONE_QUADRANT_UPRISING_TIMING_LAG] < -0.00168), 4.2, 
            np.where(
                (features[LTT.FOUR_QUADRANT_UPRISING_TIMING_LAG] > 0.00168), -3.8, 
                np.where(
                    ((features[LTT.QUADRANT_1]==1) & \
                    (features[LTT.QUADRANT_2]==1) & \
                    (features[LTT.QUADRANT_3]==1)) | \
                    ((features[LTT.QUADRANT_1]==1) & \
                    (features[LTT.QUADRANT_3]==1) & \
                    (features[LTT.QUADRANT_4]==1) & \
                    (features[TRD.AROON_UPRISING_TIMING_LAG]<-0.000168) if (annual==1008) else
                    (features[TRD.AROON_UPRISING_TIMING_LAG_MINOR]<-0.000168)) | \
                    ((features[LTT.QUADRANT_1]==1) & \
                    (features[LTT.QUADRANT_2]==1) & \
                    (features[LTT.QUADRANT_4]==1) & \
                    (features[TRD.AROON_UPRISING_TIMING_LAG]<-0.000168) if (annual==1008) else
                    (features[TRD.AROON_UPRISING_TIMING_LAG_MINOR]<-0.000168)), -1.8,
                    np.where(
                        ((features[LTT.QUADRANT_1]==1) & \
                        (features[LTT.QUADRANT_2]==1)) | \
                        ((features[LTT.QUADRANT_1]==1) & \
                        (features[LTT.QUADRANT_3]==1) & \
                        (features[LTT.QUADRANT_4]==1)) | \
                        ((features[LTT.QUADRANT_1]==1) & \
                        (features[LTT.QUADRANT_2]==1) & \
                        (features[LTT.QUADRANT_4]==1)), 0.2,
                        np.where(
                            (features[LTT.QUADRANT_1]==1) | \
                            ((features[LTT.QUADRANT_2]==1) & \
                            (features[LTT.QUADRANT_3]==1)), 2.2,
                            np.nan)))))

        features = features.reindex(
            columns=list(set([
                *features.columns,
                *['virtual_pressure_line']])))
        features['virtual_pressure_line'] = virtual_pressure_line
        features[LTT.QUADRANT_POWERLINE_LEADIN] = np.where(
            (features[LTT.QUADRANT_POWERLINE].diff(1) > 0.00168) &
            (features['virtual_pressure_line'].shift(1) > features[LTT.QUADRANT_POWERLINE].shift(1)) &
            (features['virtual_pressure_line'] < features[LTT.QUADRANT_POWERLINE]), 1, np.nan)
        features[LTT.QUADRANT_POWERLINE_LEADOUT] = np.where(
            (features[LTT.QUADRANT_POWERLINE].diff(1) < -0.00168) &
            (features['virtual_pressure_line'].shift(1) < features[LTT.QUADRANT_POWERLINE].shift(1)) &
            (features['virtual_pressure_line'] > features[LTT.QUADRANT_POWERLINE]), 1, np.nan)
        features[LTT.QUADRANT_POWERLINE_LEADIN_BEFORE] = Timeline_duration(
            np.where((features[LTT.QUADRANT_POWERLINE_LEADIN] > 0.000168), 1, 0))
        features[LTT.QUADRANT_POWERLINE_LEADOUT_BEFORE] = Timeline_duration(
            np.where((features[LTT.QUADRANT_POWERLINE_LEADOUT] > 0.000168), 1, 0))
        features = features.drop(['virtual_pressure_line'], axis=1)

    # CODEX
    features = features.reindex(columns=list(set([
        *features.columns,
        *[LTT.ONE_QUADRANT_UPRISING_TIMING_LAG,
          LTT.FOUR_QUADRANT_PRE_PRELUDE_BEFORE,
          ]])))
    features[LTT.ONE_QUADRANT_UPRISING_TIMING_LAG] = calc_event_timing_lag(
        np.where(((features[LTT.QUADRANT_1] > 0.000168) | \
                  (features[LTT.QUADRANT_2] > 0.000168) | \
                  (features[LTT.QUADRANT_3] > 0.000168) | \
                  (features[LTT.QUADRANT_4] > 0.000168)), 1, -1))
    
    features[LTT.FOUR_QUADRANT_PRE_PRELUDE_BEFORE] = Timeline_duration(
        np.where(((features[LTT.QUADRANT_1] > 0.000168) | \
                  (features[LTT.QUADRANT_2] > 0.000168) | \
                  (features[LTT.QUADRANT_3] > 0.000168) | \
                  (features[LTT.QUADRANT_4] > 0.000168)) & \
                  ((features[LTT.QUADRANT_1].shift(1) < -0.000168) & \
                  (features[LTT.QUADRANT_2].shift(1) < -0.000168) & \
                  (features[LTT.QUADRANT_3].shift(1) < -0.000168) & \
                  (features[LTT.QUADRANT_4].shift(1) < -0.000168)), 1, 0))
    
    fourQuad_autocodex_candidate=[]
    codex_15 = ((features[LTT.QUADRANT_POWERLINE]+3.8)==8) | \
                (((features[LTT.QUADRANT_POWERLINE].shift(1)+3.8)==7) & \
                ((features[LTT.QUADRANT_POWERLINE]+3.8)==7)) | \
                (((features[LTT.QUADRANT_POWERLINE].shift(1)+3.8)==8) & \
                ((features[LTT.QUADRANT_POWERLINE]+3.8)==7)) | \
                (((features[LTT.QUADRANT_POWERLINE].shift(1)+3.8) < 7) & \
                ((features[LTT.QUADRANT_POWERLINE]+3.8)==7))
    # 1101
    codex_13 = (((features[LTT.QUADRANT_1] > 0.168) & \
               (features[LTT.QUADRANT_2] > 0.168) & \
               (features[LTT.QUADRANT_3] > 0.168) & \
               (features[LTT.QUADRANT_4] > 0.168) & \
               ((features[LTT.QUADRANT_POWERLINE].shift(1)+3.8)==6)) | \
               (((features[LTT.QUADRANT_1] + \
                 features[LTT.QUADRANT_2] + \
                 features[LTT.QUADRANT_3] + \
                 features[LTT.QUADRANT_4]).diff(1) > 0.168) & \
               (features[LTT.QUADRANT_POWERLINE].diff(1) > 0.168)) & \
               ((features[LTT.QUADRANT_POWERLINE]+3.8)==6)) | \
                (((features[LTT.QUADRANT_1] + \
                 features[LTT.QUADRANT_2] + \
                 features[LTT.QUADRANT_3] + \
                 features[LTT.QUADRANT_4]).diff(1) < -1.168) & \
               (features[LTT.QUADRANT_POWERLINE].diff(1) < -1.168) & \
               ((features[LTT.QUADRANT_POWERLINE]+3.8)==7))
    # 1110
    codex_14 = (((features[LTT.QUADRANT_POWERLINE].shift(2)+3.8)==8) & \
               ((features[LTT.QUADRANT_POWERLINE].shift(1)+3.8)==7) & \
               ((features[LTT.QUADRANT_POWERLINE]+3.8)==6)) | \
               ((features[LTT.QUADRANT_1] > 0.168) & \
               (features[LTT.QUADRANT_2] > 0.168) & \
               (features[LTT.QUADRANT_3] > 0.168) & \
               (features[LTT.QUADRANT_4] > 0.168) & \
               ((features[LTT.QUADRANT_POWERLINE].shift(2)+3.8)==8) & \
               ((features[LTT.QUADRANT_POWERLINE].shift(1)+3.8)==8) & \
               ((features[LTT.QUADRANT_POWERLINE]+3.8)==6)) | \
               (~codex_15 & \
               ~codex_13 & \
               ((features[LTT.QUADRANT_POWERLINE]+3.8)==7))
    # 1011
    codex_11 = ((((features[LTT.QUADRANT_1] + \
                 features[LTT.QUADRANT_2] + \
                 features[LTT.QUADRANT_3] + \
                 features[LTT.QUADRANT_4]).diff(1) < -0.168) & \
               ((features[LTT.QUADRANT_POWERLINE].diff(1) < -0.168))) & \
               ((features[LTT.QUADRANT_POWERLINE]+3.8)==6)) | \
               (((features[LTT.QUADRANT_1] + \
                 features[LTT.QUADRANT_2] + \
                 features[LTT.QUADRANT_3] + \
                 features[LTT.QUADRANT_4]).diff(1) > 0.168) & \
               (features[LTT.QUADRANT_POWERLINE].diff(1) > 0.168) & \
               ((features[LTT.QUADRANT_POWERLINE]+3.8)==5))
    # 1100
    codex_12 = ~codex_13 & \
               ~codex_11 & \
               ((features[LTT.QUADRANT_POWERLINE]+3.8)==6)
    # 1001
    codex_09 = (((features[LTT.QUADRANT_1] + \
                 features[LTT.QUADRANT_2] + \
                 features[LTT.QUADRANT_3] + \
                 features[LTT.QUADRANT_4]).diff(1) < -0.168) & \
               (features[LTT.QUADRANT_POWERLINE].diff(1) < -0.168) & \
               ((features[LTT.QUADRANT_POWERLINE]+3.8)==5)) | \
               (((features[LTT.QUADRANT_1] + \
                 features[LTT.QUADRANT_2] + \
                 features[LTT.QUADRANT_3] + \
                 features[LTT.QUADRANT_4]).diff(1) > 0.168) & \
               (features[LTT.QUADRANT_POWERLINE].diff(1) > 0.168) & \
               ((features[LTT.QUADRANT_POWERLINE]+3.8)==4))
    # 1010
    codex_10 = ~codex_09 & \
               ~codex_11 & \
               ((features[LTT.QUADRANT_POWERLINE]+3.8)==5)
    # 0111
    codex_07 = (((features[LTT.QUADRANT_1] + \
                 features[LTT.QUADRANT_2] + \
                 features[LTT.QUADRANT_3] + \
                 features[LTT.QUADRANT_4]).diff(1) > 0.168) & \
               (features[LTT.QUADRANT_POWERLINE].diff(1) > 0.168) & \
               ((features[LTT.QUADRANT_POWERLINE]+3.8)==3)) | \
               (((features[LTT.QUADRANT_1] + \
                 features[LTT.QUADRANT_2] + \
                 features[LTT.QUADRANT_3] + \
                 features[LTT.QUADRANT_4]).diff(1) < -0.168) & \
               (features[LTT.QUADRANT_POWERLINE].diff(1) < -0.168) & \
               ((features[LTT.QUADRANT_POWERLINE]+3.8)==4))
    # 1000
    codex_08 = ~codex_07 & \
               ~codex_09 & \
               ((features[LTT.QUADRANT_POWERLINE]+3.8)==4)    
    # 0101
    codex_05 = (((features[LTT.QUADRANT_1] + \
                 features[LTT.QUADRANT_2] + \
                 features[LTT.QUADRANT_3] + \
                 features[LTT.QUADRANT_4]).diff(1) > 0.168) & \
               (features[LTT.QUADRANT_POWERLINE].diff(1) > 0.168) & \
               ((features[LTT.QUADRANT_POWERLINE]+3.8)==2)) | \
               (((features[LTT.QUADRANT_1] + \
                 features[LTT.QUADRANT_2] + \
                 features[LTT.QUADRANT_3] + \
                 features[LTT.QUADRANT_4]).diff(1) < -0.168) & \
               (features[LTT.QUADRANT_POWERLINE].diff(1) < -0.168) & \
               ((features[LTT.QUADRANT_POWERLINE]+3.8)==3))
    # 0110
    codex_06 = ~codex_07 & \
               ~codex_09 & \
               ((features[LTT.QUADRANT_POWERLINE]+3.8)==3)
    # 0011
    codex_03 = (((features[LTT.QUADRANT_1] + \
                 features[LTT.QUADRANT_2] + \
                 features[LTT.QUADRANT_3] + \
                 features[LTT.QUADRANT_4]).diff(1) > 0.168) & \
               (features[LTT.QUADRANT_POWERLINE].diff(1) > 0.168) & \
               ((features[LTT.QUADRANT_POWERLINE]+3.8)==1)) | \
               (((features[LTT.QUADRANT_1] + \
                 features[LTT.QUADRANT_2] + \
                 features[LTT.QUADRANT_3] + \
                 features[LTT.QUADRANT_4]).diff(1) < -0.168) & \
               (features[LTT.QUADRANT_POWERLINE].diff(1) < -0.168) & \
               ((features[LTT.QUADRANT_POWERLINE]+3.8)==2))
    # 0100
    codex_04 = ~codex_05 & \
               ~codex_03 & \
               ((features[LTT.QUADRANT_POWERLINE]+3.8)==2)
    # 0001
    codex_01 = ((features[LTT.QUADRANT_1] <-0.168) & \
               (features[LTT.QUADRANT_2] <-0.168) & \
               (features[LTT.QUADRANT_3] <-0.168) & \
               (features[LTT.QUADRANT_4] <-0.168) & \
               ((features[LTT.QUADRANT_POWERLINE].shift(1)+3.8)==1)) | \
               ((features[LTT.QUADRANT_1] <-0.168) & \
               (features[LTT.QUADRANT_2] <-0.168) & \
               (features[LTT.QUADRANT_3] <-0.168) & \
               (features[LTT.QUADRANT_4] <-0.168) & \
               (features[LTT.QUADRANT_POWERLINE].diff(1) < -0.168) & \
               ((features[LTT.QUADRANT_POWERLINE]+3.8)==1)) | \
               (((features[LTT.QUADRANT_1] + \
                 features[LTT.QUADRANT_2] + \
                 features[LTT.QUADRANT_3] + \
                 features[LTT.QUADRANT_4]).diff(1) < -0.168) & \
               (features[LTT.QUADRANT_POWERLINE].diff(1) < -0.168) & \
               ((features[LTT.QUADRANT_POWERLINE]+3.8)==2))
    # 0010
    codex_02 = ~codex_01 & \
               ~codex_03 & \
               ((features[LTT.QUADRANT_POWERLINE]+3.8)==1)
    fourQuad_autocodex_candidate.append(codex_15 | \
                                        codex_13 | \
                                        codex_11 | \
                                        codex_09 | \
                                        codex_07 | \
                                        codex_05 | \
                                        codex_03 | \
                                        codex_01)       # 0001
    fourQuad_autocodex_candidate.append(codex_15 | \
                                        codex_14 | \
                                        codex_11 | \
                                        codex_10 | \
                                        codex_07 | \
                                        codex_06 | \
                                        codex_03 | \
                                        codex_02)       # 0010
    fourQuad_autocodex_candidate.append(codex_15 | \
                                        codex_14 | \
                                        codex_13 | \
                                        codex_12 | \
                                        codex_07 | \
                                        codex_06 | \
                                        codex_05 | \
                                        codex_04)       # 0100
    fourQuad_autocodex_candidate.append(codex_15 | \
                                        codex_14 | \
                                        codex_13 | \
                                        codex_12 | \
                                        codex_11 | \
                                        codex_10 | \
                                        codex_09 | \
                                        codex_08)       # 1000
    fourQuad_autocodex_candidate.append(features[LTT.QUADRANT_4]>0.00168)
    fourQuad_autocodex_candidate.append(features[LTT.QUADRANT_3]>0.00168)
    fourQuad_autocodex_candidate.append(features[LTT.QUADRANT_2]>0.00168)
    fourQuad_autocodex_candidate.append(features[LTT.QUADRANT_1]>0.00168)

    #macd_autocodex_candidate.append(features[FLD.MA250_MAJOR]>features[FLD.MA250])

    features = features.reindex(columns=list(set([
        *features.columns,
        *[LTT.QUADRANT_STATUS,
          LTT.CHOBITS,
          LTT.QUADRANT_CODEX]])))
    features[LTT.QUADRANT_STATUS]=0
    for i in range(0, len(fourQuad_autocodex_candidate)):
        fourQuad_autocodex = fourQuad_autocodex_candidate[i]
        if (i==0):
            features[LTT.QUADRANT_STATUS] = np.where(fourQuad_autocodex > 0, 1, 0)
        else:
            features[LTT.QUADRANT_STATUS] = np.where(fourQuad_autocodex > 0, 
                                                     2 ** i + features[LTT.QUADRANT_STATUS], 
                                                     features[LTT.QUADRANT_STATUS])
    features[LTT.CHOBITS] = ['{:02X}'.format(int(i)) if not np.isnan(i) else np.nan for i in features[LTT.QUADRANT_STATUS].values]
    fourQuad_Previous = pd.DataFrame(
        np.where(features[LTT.CHOBITS] != features[LTT.CHOBITS].shift(1),
                 features[LTT.CHOBITS].shift(1), np.nan))
    fourQuad_Previous_2 = pd.DataFrame(
        np.where(fourQuad_Previous.ffill() != fourQuad_Previous.ffill().shift(1),
                 fourQuad_Previous.ffill().shift(1), np.nan))
    fourQuad_Previous_3 = pd.DataFrame(
        np.where(fourQuad_Previous_2.ffill() != fourQuad_Previous_2.ffill().shift(1),
                 fourQuad_Previous_2.ffill().shift(1), np.nan))
    features[LTT.QUADRANT_CODEX] = np.squeeze(fourQuad_Previous_3.ffill().bfill().values) + np.squeeze(fourQuad_Previous_2.ffill().bfill().values) + np.squeeze(fourQuad_Previous.ffill().bfill().values) + features[LTT.CHOBITS].ffill().bfill().values
    
    checklist_float32_columns=[FLD.ATR_UB_WEEKLY,
                                STE.CVaR_BOWL_CANDIDATE,
                                LTT.QUADRANT_1,
                                LTT.QUADRANT_2,
                                LTT.QUADRANT_3,
                                LTT.QUADRANT_4,
                                LTT.QUADRANT_POWERLINE,
                                LTT.QUADRANT_POWERLINE_LEADIN,
                                LTT.QUADRANT_POWERLINE_LEADOUT,]
    features = clearup_compact_dataframe_float32(features,
                                                checklist_float32_columns)

    checklist_int16_columns=[LTT.ONE_QUADRANT_UPRISING_TIMING_LAG,
                            LTT.TWO_QUADRANT_UPRISING_TIMING_LAG,
                            LTT.TWO_QUADRANT_DOWNRISK_TIMING_LAG,
                            LTT.THREE_QUADRANT_UPRISING_TIMING_LAG,
                            LTT.THREE_QUADRANT_DOWNRISK_TIMING_LAG,
                            LTT.FOUR_QUADRANT_UPRISING_TIMING_LAG,
                            LTT.QUADRANT_POWERLINE_LEADIN_BEFORE,
                            LTT.QUADRANT_POWERLINE_LEADOUT_BEFORE,]
    features = clearup_compact_dataframe_int16(features,
                                               checklist_int16_columns)

    return features


def calc_quadrant_leverage_macd_timing_lag(features: pd.DataFrame) -> pd.DataFrame:
    """`QLevMACDLagD` —— 四象限杠杆能量 × MACD 的合成 timing-lag。

    逐字搬自老树 `analysis/base.py:2171-2188`（原嵌在 `calc_deadpool_downrisk` 里）。
    输入 `4QPwrln` / `4QPwrlnMin` / `4QPwrlnMaj`（由 :func:`calc_four_quadrant_boost`
    在**各自频率**上算出 —— `_MINOR`/`_MAJOR` 是 15min / 日线的重命名）与
    `MACD_diff_1T_MAJ`。

    ⚠️ **只搬了这一条链**：老树同一处还算了 `QUADRANT_LEVERAGE_ENERGY_NEGATIVE_*`
    与 `_ENERGY_I..V_BEFORE`（**别的指标**用的），本模块**不搬** —— 用户点名的是
    `QLevMACDLagD` 一个。

    ⚠️ **第二个 `OR` 分支**（老树 `models/mainstream.py:228-238`）：
    60min `MACD_DELTA > 0` 且 `EngLag ∈ (1, MACD_CROSS_SX_BEFORE)` 且
    `EngLag*4 > MAINSTREAM_LEADINx5_BEFORE_MINOR`。**三列都在场才启用** ——
    缺列时保持原来的单分支行为（向后兼容）。实测 301308 2095 根里第二分支
    **一次也没触发**；计入只是为了让换标的不丢条件（见 worklog 2026-10-19）。
    """
    features = features.reindex(
        columns=list(set([
            *features.columns,
            *[LTT.QUADRANT_LEVERAGE_ENERGY_MINOR,
              LTT.QUADRANT_LEVERAGE_ENERGY,
              LTT.QUADRANT_LEVERAGE_ENERGY_MAJOR,
              LTT.QUADRANT_LEVERAGE_ENERGY_DUMMY,
              LTT.QUADRANT_LEVERAGE_ENERGY_TIMING_LAG_DUMMY,
              LTT.QUADRANT_LEVERAGE_MACD_TIMING_LAG_DUMMY]])))
    features[LTT.QUADRANT_LEVERAGE_ENERGY_MINOR] = calc_energy(np.where(
        ((features[LTT.QUADRANT_POWERLINE_MINOR] + 3.78) > 0.00168),
        (features[LTT.QUADRANT_POWERLINE_MINOR] + 3.78), 0)) / 16
    features[LTT.QUADRANT_LEVERAGE_ENERGY] = calc_energy(np.where(
        ((features[LTT.QUADRANT_POWERLINE] + 3.78) > 0.00168),
        (features[LTT.QUADRANT_POWERLINE] + 3.78), 0)) / 4
    features[LTT.QUADRANT_LEVERAGE_ENERGY_MAJOR] = calc_energy(np.where(
        ((features[LTT.QUADRANT_POWERLINE_MAJOR] + 3.78) > 0.00168),
        (features[LTT.QUADRANT_POWERLINE_MAJOR] + 3.78), 0))
    features[LTT.QUADRANT_LEVERAGE_ENERGY_DUMMY] = \
        features[LTT.QUADRANT_LEVERAGE_ENERGY_MINOR] + \
        features[LTT.QUADRANT_LEVERAGE_ENERGY] + \
        features[LTT.QUADRANT_LEVERAGE_ENERGY_MAJOR]
    features[LTT.QUADRANT_LEVERAGE_ENERGY_TIMING_LAG_DUMMY] = calc_event_timing_lag(
        np.where(
            (features[LTT.QUADRANT_LEVERAGE_ENERGY_DUMMY] > 0.0000168), 1, -1))
    eng = features[LTT.QUADRANT_LEVERAGE_ENERGY_TIMING_LAG_DUMMY]
    inner_major = calc_event_timing_lag(
        np.where((features[FLD.MACD_DELTA_MAJOR] > 1e-12), 1, -1))
    cond = (inner_major + eng) > 1e-12

    # ⚠️ 老树 `models/mainstream.py:228-238` 还有**第二个 OR 分支**。它要三个
    # 60min 输入：`MACD_DELTA`（60min 柱差分）、`MACD_CROSS_SX_BEFORE`（死叉计数）、
    # `MAINSTREAM_LEADINx5_BEFORE_MINOR`（15min 主力吸筹 x5）。**三列都在场才启用**
    # —— 保持本函数向后兼容（现有调用方/用例不受影响）。
    _need = (FLD.MACD_DELTA, FLD.MACD_CROSS_SX_BEFORE,
             LTT.MAINSTREAM_LEADINx5_BEFORE_MINOR)
    if all(c in features.columns for c in _need):
        inner_minor = calc_event_timing_lag(np.where(
            (features[FLD.MACD_DELTA] > 1e-12) &
            (eng > 1) &
            (eng < features[FLD.MACD_CROSS_SX_BEFORE]) &
            (eng * 4 > features[LTT.MAINSTREAM_LEADINx5_BEFORE_MINOR]), 1, -1))
        cond = cond | ((inner_minor + eng) > 1e-12)

    features[LTT.QUADRANT_LEVERAGE_MACD_TIMING_LAG_DUMMY] = calc_event_timing_lag(
        np.where(cond, 1, -1))
    return features


def _linear_regression_slope(y) -> float:
    """最小二乘斜率（老树 `calc_bs_signals` 内联的那个）。"""
    x = np.arange(len(y), dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x_mean = x.mean()
    y_mean = y.mean()
    numerator = ((x - x_mean) * (y - y_mean)).sum()
    denominator = ((x - x_mean) ** 2).sum()
    return numerator / denominator if denominator != 0 else 0.0


def calc_bs_star_down(ohlc: pd.DataFrame) -> pd.Series:
    """`BSStarDwn` —— **只搬星线死叉那一段**（老树 `analysis/base.py:2429` 的子集）。

    老树整份 `calc_bs_signals` 还产出买/卖线与四个 `BS_*`；`4QuadStDwnCnt`
    只需要星线，所以这里**只算到星线为止** —— 这是 2026-10-19 的解耦点：
    **不需要整条 BS 链**，更不需要 `calc_bs_signals` 就地改 `ohlc` 的副作用。

    线上定义（逐字）：
    * 买线 = `EMA(close, 2)`
    * 卖线 = `EMA(slope21*20 + close, 42)`，`slope21` 是 close 的 21 日 OLS 斜率
    * 指导线 = `EMA(mean(EMA(close,4), EMA(close,6), EMA(close,12), EMA(close,24)), 2)`
    * 界 = `MA(close, 27)`
    * 星线死叉 = **界上穿指导线**

    >>> import pandas as pd
    >>> o = pd.DataFrame({'open': range(30), 'high': range(30), 'low': range(30),
    ...                   'close': range(30)}, dtype=float)
    >>> int(calc_bs_star_down(o).sum())
    0
    """
    close = pd.to_numeric(ohlc[AKA.CLOSE], errors='coerce')
    slope_21 = close.rolling(21).apply(_linear_regression_slope, raw=True)
    sell_line = (slope_21 * 20 + close).ewm(span=42, adjust=False).mean()
    _emas = [close.ewm(span=s, adjust=False).mean() for s in (4, 6, 12, 24)]
    ref_line = (sum(_emas) / 4).ewm(span=2, adjust=False).mean()
    border_line = close.rolling(27).mean()

    star_down = (border_line > ref_line) & (border_line.shift(1) <= ref_line.shift(1))
    return pd.Series(star_down.to_numpy(), index=ohlc.index).astype(np.int16)


def calc_bs_signals(ohlc: pd.DataFrame,
                    features: pd.DataFrame = None) -> pd.DataFrame:
    """买/卖线 + 星线（老树 `analysis/base.py:2429` 的**纯函数**版）。

    与 `calc_bs_star_down` 的区别：这里产出**完整**的 `BS_BUY/SELL/STAR_UP/STAR_DOWN`
    及其 `*_BEFORE`（`Timeline_duration`），供 15min minor 链使用。

    ⚠️ 老树会**就地修改**传入的 `ohlc`（加中文列名、删中间列）；本实现只返回
    `features`，不碰入参。
    """
    if features is None:
        features = pd.DataFrame(index=ohlc.index)
    close = pd.to_numeric(ohlc[AKA.CLOSE], errors='coerce')

    buy_line = close.ewm(span=2, adjust=False).mean()
    slope_21 = close.rolling(21).apply(_linear_regression_slope, raw=True)
    sell_line = (slope_21 * 20 + close).ewm(span=42, adjust=False).mean()
    _emas = [close.ewm(span=s, adjust=False).mean() for s in (4, 6, 12, 24)]
    ref_line = (sum(_emas) / 4).ewm(span=2, adjust=False).mean()
    border_line = close.rolling(27).mean()

    bs_buy = (buy_line > sell_line) & (buy_line.shift(1) <= sell_line.shift(1))
    bs_sell = (sell_line > buy_line) & (sell_line.shift(1) <= buy_line.shift(1))
    star_up = (ref_line > border_line) & (ref_line.shift(1) <= border_line.shift(1))
    star_down = (border_line > ref_line) & (border_line.shift(1) <= ref_line.shift(1))

    features = features.reindex(columns=list(set([
        *features.columns,
        LTT.BS_REF_LINE, LTT.BS_BORDER_LINE,
        LTT.BS_BUY, LTT.BS_SELL, LTT.BS_STAR_UP, LTT.BS_STAR_DOWN,
        LTT.BS_BUY_BEFORE, LTT.BS_SELL_BEFORE,
        LTT.BS_STAR_UP_BEFORE, LTT.BS_STAR_DOWN_BEFORE])))
    features[LTT.BS_REF_LINE] = ref_line.to_numpy()
    features[LTT.BS_BORDER_LINE] = border_line.to_numpy()
    for src, dst in ((bs_buy, LTT.BS_BUY), (bs_sell, LTT.BS_SELL),
                     (star_up, LTT.BS_STAR_UP), (star_down, LTT.BS_STAR_DOWN)):
        features[dst] = src.fillna(False).astype(np.int16).to_numpy()
    for src, dst in ((LTT.BS_BUY, LTT.BS_BUY_BEFORE),
                     (LTT.BS_SELL, LTT.BS_SELL_BEFORE),
                     (LTT.BS_STAR_UP, LTT.BS_STAR_UP_BEFORE),
                     (LTT.BS_STAR_DOWN, LTT.BS_STAR_DOWN_BEFORE)):
        features[dst] = Timeline_duration(
            np.where(features[src].to_numpy() > 1e-19, 1, 0))
    return features


def calc_quadrant_stardown_count(features: pd.DataFrame) -> pd.DataFrame:
    """`4QuadStDwnCnt` —— **一条 `calc_energy`**（老树 `models/mainstream.py:515-520`）。

    输入 `BS_STAR_DOWN_BEFORE_MAJOR`（日线星线死叉的 `Timeline_duration`，
    展开到主频）与 `QLevEngLagD`。
    """
    for col in (LTT.BS_STAR_DOWN_BEFORE_MAJOR,
                LTT.QUADRANT_LEVERAGE_ENERGY_TIMING_LAG_DUMMY):
        if col not in features.columns:
            raise KeyError('calc_quadrant_stardown_count 缺少 {!r}'.format(col))
    before = features[LTT.BS_STAR_DOWN_BEFORE_MAJOR]
    eng = features[LTT.QUADRANT_LEVERAGE_ENERGY_TIMING_LAG_DUMMY]
    features[LTT.QUADRANT_STARDOWN_COUNT] = calc_energy(np.where(
        (before == 1) & (eng > 1e-12), 1,
        np.where(eng > 1e-12, 1e-12, 0))).astype(np.int16)
    return features


def calc_quadrant_powerline_uprisal_lag(features: pd.DataFrame) -> pd.DataFrame:
    """`QLevPwrlnUprLag` —— 老树 `models/mainstream.py:530` + `:544` **两条**。

    ⚠️ 老树是**两条连续赋值**：第二条覆盖第一条，并在第二个 `OR` 分支里
    **引用第一条的中间值**（`pwr1 / EngLag > 0.618`）。只搬一条会得到不同结果。

    输入：`QLevEngLagD` / `QLevEngD`（由前者隐含） / `4QuadStDwnCnt` /
    `4QPwrlnMaj` / `DEA_MAJ`。
    """
    need = (LTT.QUADRANT_LEVERAGE_ENERGY_TIMING_LAG_DUMMY,
            LTT.QUADRANT_STARDOWN_COUNT,
            LTT.QUADRANT_POWERLINE_MAJOR,
            FLD.DEA_MAJOR)
    for col in need:
        if col not in features.columns:
            raise KeyError('calc_quadrant_powerline_uprisal_lag 缺少 {!r}'.format(col))

    eng = features[LTT.QUADRANT_LEVERAGE_ENERGY_TIMING_LAG_DUMMY].to_numpy(np.float64)
    stardown = features[LTT.QUADRANT_STARDOWN_COUNT].to_numpy(np.float64)
    pwr_major = features[LTT.QUADRANT_POWERLINE_MAJOR].to_numpy(np.float64)
    dea_major = features[FLD.DEA_MAJOR].to_numpy(np.float64)

    cond1 = (
        (eng > 1e-12) &
        ((stardown < 3) |
         ((eng > 84) &
          ((stardown - np.log10(np.where(eng >= 10, eng, 10))) < 3))) &
        (pwr_major > dea_major))
    pwr1 = calc_event_timing_lag(np.where(cond1, 1, -1))

    with np.errstate(invalid='ignore', divide='ignore'):
        ratio = np.where(eng > 0, pwr1 / eng, -np.inf)
    cond2 = (
        (eng > 1e-12) &
        (stardown > 2.75) & (stardown < 4) &
        (ratio > 0.618) &
        (pwr_major > dea_major))

    features[LTT.QUADRANT_LEVERAGE_POWERLINE_UPRISING_TIMING_LAG] = \
        calc_event_timing_lag(np.where(cond1 | cond2, 1, -1))
    return features


def calc_quadrant_zen_dash_count(features: pd.DataFrame) -> pd.DataFrame:
    """`4QuadZenDCnt` —— **一条 `calc_energy`**（老树 `analysis/mainstream.py:180-188`）。

    输入 `zDLagD`（由 :func:`analysis.pivot.calc_zen_dash_timing_lag` 从 CZSC 笔
    方向产出）与 `QLevMACDLagD`。
    """
    for col in (FLD.ZEN_DASH_TIMING_LAG_DUMMY,
                LTT.QUADRANT_LEVERAGE_MACD_TIMING_LAG_DUMMY):
        if col not in features.columns:
            raise KeyError('calc_quadrant_zen_dash_count 缺少 {!r}'.format(col))
    zd = features[FLD.ZEN_DASH_TIMING_LAG_DUMMY].to_numpy(np.float64)
    lag = features[LTT.QUADRANT_LEVERAGE_MACD_TIMING_LAG_DUMMY].to_numpy(np.float64)
    features[LTT.QUADRANT_ZEN_DASH_COUNT] = calc_energy(np.where(
        (lag > 1e-12),
        np.where((zd < lag) & (zd == 1), 1, 1e-12),
        0)).astype(np.int16)
    return features


def calc_quadrant_bs_trunk_timing_lag(features: pd.DataFrame) -> pd.DataFrame:
    """`QuadBsTuk` / `QuadBsTukLag` —— 买卖星线结构与 MACD lag 的合成。

    逐字搬自老树 `analysis/quad_gear.py:640-656`（**规范版**；老树另有
    `zen_pivot.py:603` 与 `:1047` 两份副本，**未搬** —— 见 D35「只搬一条」）。

    ⚠️ 需要 6 个 `BS_*` 输入列（`BSStarUpBf(Maj)` / `BSStarDwnBf(Maj)` /
    `BSBuBfMaj` / `BSSelBfMaj`），它们由 :func:`calc_bs_before` 从更上游的
    `BS*` 检测列产出；而**那套星线检测（`calc_bs_signals`）本模块未搬**
    ⇒ 本函数吃输入列，**不能自称端到端可跑**。
    """
    cond = (
        (features[LTT.QUADRANT_LEVERAGE_MACD_TIMING_LAG_DUMMY] > 1e-12) &
        ((features[LTT.BS_STAR_UP_BEFORE_MAJOR] < features[LTT.BS_STAR_DOWN_BEFORE_MAJOR]) |
         (features[LTT.BS_STAR_UP_BEFORE] < features[LTT.BS_STAR_DOWN_BEFORE]) |
         ((features[LTT.BS_STAR_UP_BEFORE_MAJOR] > features[LTT.BS_STAR_DOWN_BEFORE_MAJOR]) &
          (features[LTT.BS_BUY_BEFORE_MAJOR] < features[LTT.BS_SELL_BEFORE_MAJOR])))
    )
    features = features.reindex(columns=list(set([
        *features.columns,
        *[LTT.QUADRANT_BS_TRUNK, LTT.QUADRANT_BS_TRUNK_TIMING_LAG]])))
    features[LTT.QUADRANT_BS_TRUNK] = np.where(cond, 1, np.nan)
    features[LTT.QUADRANT_BS_TRUNK_TIMING_LAG] = calc_event_timing_lag(
        np.where(cond, 1, -1))
    return features


def calc_mainstream_feats(ohlc: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    """主力资金吸筹 —— 产出 `MsLeadinBf` / `MsLeadinx3Bf` / `MsLeadinx5Bf`。

    逐字搬自老树 `analysis/base.py:2569-2687`（VAR1..VAR5 → Entry/Shakeout 链，
    加 KDJ_J / MACD 两个判据）。

    ⚠️ **一处刻意偏离**：老树在 `if LTT.BS_BUY_BEFORE not in features.columns:` 时
    会调 `calc_bs_signals(...)`（星线检测链）**顺带产出 BS_* 列**。那条链不在本模块
    范围（用户只点名 6 个 BS_* 的 *BEFORE* 包装），**故此处不调** ⇒ 本函数**不产 BS_* 列**。
    `MsLeadinBf`/`x3`/`x5` 的计算**不读 BS_*** ⇒ 这一偏离对它们无影响。

    ⚠️ **要求 `features` 里有 `MACD` 列**（`x3`/`x5` 的分支读它）—— 老树同此。
    `KDJ_K/D/J` 与 `KDJ_TRD_LAG` 若缺则**本函数自己补**（走 `talib.STOCH`）。
    """
    if (FLD.KDJ_TREND_TIMING_LAG not in features.columns):
        features = features.reindex(
            columns=list(set([
                *features.columns,
                *[FLD.KDJ_K, FLD.KDJ_D, FLD.KDJ_J,
                  FLD.KDJ_TREND_TIMING_LAG]])))
        features[[FLD.KDJ_K, FLD.KDJ_D, FLD.KDJ_J]] = kdj(
            ohlc[AKA.HIGH].values,
            ohlc[AKA.LOW].values,
            ohlc[AKA.CLOSE].values, )
        features[FLD.KDJ_TREND_TIMING_LAG] = calc_event_timing_lag(
            np.where(
                (features[FLD.KDJ_J] > features[FLD.KDJ_K]), 1, -1
            )
        )

    """计算主力吸筹指标"""
    # VAR1:=REF((LOW+OPEN+CLOSE+HIGH)/4,1);
    var1 = ((ohlc[AKA.LOW] + ohlc[AKA.OPEN] + ohlc[AKA.CLOSE] + ohlc[AKA.HIGH]) / 4).shift(1)

    # VAR2:=SMA(ABS(LOW-VAR1),13,1)/SMA(MAX(LOW-VAR1,0),10,1);
    abs_diff = np.abs(ohlc[AKA.LOW] - var1)
    max_diff = np.maximum(ohlc[AKA.LOW] - var1, 0)

    sma_abs = sma(abs_diff, 13, 1)
    sma_max = sma(max_diff, 10, 1)

    # 避免除零
    var2 = sma_abs / np.where(sma_max == 0, 1, sma_max)

    # VAR3:=EMA(VAR2,10);
    var3 = ema(var2, 10)

    # VAR4:=LLV(LOW,33);
    var4 = llv(ohlc[AKA.LOW], 33)

    # VAR5:=EMA(IF(LOW<=VAR4,VAR3,0),3);
    condition = ohlc[AKA.LOW] <= var4
    var5_input = np.where(condition, var3, 0)
    var5 = ema(pd.Series(var5_input, index=ohlc.index), 3)

    # 主力进场:IF(VAR5>REF(VAR5,1),VAR5,0)
    var5_prev = var5.shift(1)
    main_force_entry = np.where(var5 > var5_prev, var5, 0)

    # 洗盘:IF(VAR5<REF(VAR5,1),VAR5,0)
    shakeout = np.where(var5 < var5_prev, var5, 0)
    features = features.reindex(
        columns=list(set([
            *features.columns,
            *[LTT.MAINSTREAM_LEADIN_VAR5,
              LTT.MAINSTREAM_LEADIN_ENTRY,
              LTT.MAINSTREAM_LEADIN_SHAKEOUT,
              LTT.MAINSTREAM_LEADIN_BEFORE,
              LTT.MAINSTREAM_LEADINx3_BEFORE,
              LTT.MAINSTREAM_LEADINx5_BEFORE]])))
    features[LTT.MAINSTREAM_LEADIN_VAR5] = var5
    features[LTT.MAINSTREAM_LEADIN_ENTRY] = main_force_entry
    features[LTT.MAINSTREAM_LEADIN_SHAKEOUT] = shakeout
    features[LTT.MAINSTREAM_LEADIN_BEFORE] = Timeline_duration(
        np.where(
            (features[FLD.KDJ_J].diff(1) > 1e-3) & \
            ((features[LTT.MAINSTREAM_LEADIN_ENTRY].shift(2) > ohlc[AKA.CLOSE].shift(2) * 0.618) & \
            (features[LTT.MAINSTREAM_LEADIN_SHAKEOUT].shift(1) > 0.618) & \
            (features[LTT.MAINSTREAM_LEADIN_SHAKEOUT] > 0.168)), 1, 0
        ))
    features[LTT.MAINSTREAM_LEADINx3_BEFORE] = Timeline_duration(
        np.where(
            (features[FLD.KDJ_J].diff(1) > 1e-3) & \
            (((features[LTT.MAINSTREAM_LEADIN_ENTRY].shift(2) > ohlc[AKA.CLOSE].shift(2) * 3) & \
            (features[LTT.MAINSTREAM_LEADIN_SHAKEOUT].shift(1) > 0.618) & \
            (features[LTT.MAINSTREAM_LEADIN_SHAKEOUT] > 0.168)) | \
            ((features[LTT.MAINSTREAM_LEADIN_ENTRY].shift(1) > ohlc[AKA.CLOSE].shift(1) * 3) & \
            (features[LTT.MAINSTREAM_LEADIN_SHAKEOUT] > ohlc[AKA.CLOSE] * 0.382) & \
            (features[LTT.MAINSTREAM_LEADIN_SHAKEOUT] > 0.168) & \
            (features[FLD.MACD].diff(1) > 1e-9)) | \
            ((features[LTT.MAINSTREAM_LEADIN_ENTRY].shift(2) > ohlc[AKA.CLOSE].shift(2) * 0.618) & \
            (features[LTT.MAINSTREAM_LEADIN_ENTRY].shift(1) > ohlc[AKA.CLOSE].shift(1) * 3) & \
            (features[LTT.MAINSTREAM_LEADIN_SHAKEOUT] > ohlc[AKA.CLOSE] * 0.382) & \
            (features[LTT.MAINSTREAM_LEADIN_SHAKEOUT] > 0.168) & \
            (features[FLD.MACD].diff(1) > 1e-9))), 1, 0
        ))
    features[LTT.MAINSTREAM_LEADINx5_BEFORE] = Timeline_duration(
        np.where(
            (features[FLD.KDJ_J].diff(1) > 1e-3) & \
            (((features[LTT.MAINSTREAM_LEADIN_ENTRY].shift(2) > 5 * ohlc[AKA.CLOSE].shift(2)) & \
            (features[LTT.MAINSTREAM_LEADIN_SHAKEOUT].shift(1) > 0.618) & \
            (features[LTT.MAINSTREAM_LEADIN_SHAKEOUT] > 0.168)) | \
            ((features[LTT.MAINSTREAM_LEADIN_ENTRY].shift(1) > 5 * ohlc[AKA.CLOSE].shift(1)) & \
            (features[LTT.MAINSTREAM_LEADIN_SHAKEOUT] > 0.382 * ohlc[AKA.CLOSE]) & \
            (features[LTT.MAINSTREAM_LEADIN_SHAKEOUT] > 0.168) & \
            (features[FLD.MACD].diff(1) > 1e-9)) | \
            ((features[LTT.MAINSTREAM_LEADIN_ENTRY].shift(2) > 0.832 * ohlc[AKA.CLOSE].shift(2)) & \
            (features[LTT.MAINSTREAM_LEADIN_ENTRY].shift(1) > 5 * ohlc[AKA.CLOSE].shift(1)) & \
            (features[LTT.MAINSTREAM_LEADIN_SHAKEOUT] > 0.382 * ohlc[AKA.CLOSE]) & \
            (features[LTT.MAINSTREAM_LEADIN_SHAKEOUT] > 0.168) & \
            (features[FLD.MACD].diff(1) > 1e-9))), 1, 0
        ))
    return features


def calc_bs_before(features: pd.DataFrame) -> pd.DataFrame:
    """8 个 `BS*Bf*` —— 买卖/星线信号的「距上一次」计数（`Timeline_duration`）。

    逐字搬自老树两处：`base.py:2480-2495`（`BS_BUY/SELL/STAR_UP/STAR_DOWN_BEFORE`）
    与 `models/mainstream.py:1417-1426`（四个 `_MAJOR` 版）。

    ⚠️ **输入是 `BS_BUY` / `BS_SELL` / `BS_STAR_UP` / `BS_STAR_DOWN` 及其 `_MAJOR`
    列** —— 那是**星线检测链**（老树 `calc_bs_signals`）的产物，**本模块未搬**。
    故本函数只做那一步 `Timeline_duration` 包装，**不能自称端到端可跑**。
    """
    features = features.reindex(columns=list(set([
        *features.columns,
        *[LTT.BS_BUY_BEFORE, LTT.BS_SELL_BEFORE,
          LTT.BS_STAR_UP_BEFORE, LTT.BS_STAR_DOWN_BEFORE,
          LTT.BS_BUY_BEFORE_MAJOR, LTT.BS_SELL_BEFORE_MAJOR,
          LTT.BS_STAR_UP_BEFORE_MAJOR, LTT.BS_STAR_DOWN_BEFORE_MAJOR]])))
    for src, dst in ((LTT.BS_BUY, LTT.BS_BUY_BEFORE),
                     (LTT.BS_SELL, LTT.BS_SELL_BEFORE),
                     (LTT.BS_STAR_UP, LTT.BS_STAR_UP_BEFORE),
                     (LTT.BS_STAR_DOWN, LTT.BS_STAR_DOWN_BEFORE),
                     (LTT.BS_BUY_MAJOR, LTT.BS_BUY_BEFORE_MAJOR),
                     (LTT.BS_SELL_MAJOR, LTT.BS_SELL_BEFORE_MAJOR),
                     (LTT.BS_STAR_UP_MAJOR, LTT.BS_STAR_UP_BEFORE_MAJOR),
                     (LTT.BS_STAR_DOWN_MAJOR, LTT.BS_STAR_DOWN_BEFORE_MAJOR)):
        features[dst] = Timeline_duration(np.where((features[src] > 1e-19), 1, 0))
    return features


def calc_pivot_timing_lag(features: pd.DataFrame) -> pd.DataFrame:
    """`ZenPivotLag` —— 距**出中枢**（`ZenPivotZD` 变 NaN）以来的 lag。

    逐字搬自老树 `analysis/zen_pivot.py:859`（一行）。

    ⚠️ 输入 `ZenPivotZD` 由本树 `analysis/pivot.py::attach_pivot_features`（喂给
    **CZSC 中枢计算**）产出 —— 那个函数当前**零调用者**，故本函数吃输入列，
    **不能自称端到端可跑**。
    """
    features = features.reindex(columns=list(set([
        *features.columns, *[ZEN.PIVOT_TIMING_LAG]])))
    features[ZEN.PIVOT_TIMING_LAG] = calc_event_timing_lag(
        np.where(~features[ZEN.PIVOT_ZD].isna(), 1, -1))
    return features


def inv_nums(series) -> int:
    """序列的**逆序数**（inversion count）—— 前面比后面大的对数。

    逐字搬自老树 `features/base.py:1822`（原为 `@nb.jit('int16(f4[:])')`；
    本模块**不带 numba** —— 加速版见 `analysis/quadrant_jit.py`，`DECISIONS.md` D36）。

    ⚠️ **调用方是 :func:`mapower_np_vXIIs`，传进去的是「25 个 MA 窗口」那条切片**
    （不是整个价格序列）—— 所以这里的 O(n²) 是 **25² 量级**，不是价格序列长度的平方。
    别因为看到双重循环就以为它是整个算法的瓶颈。

    >>> inv_nums(np.array([3, 1, 2], dtype=np.float32))
    2
    >>> inv_nums(np.array([1, 2, 3], dtype=np.float32))   # 已升序 ⇒ 无逆序对
    0
    >>> inv_nums(np.array([3, 2, 1], dtype=np.float32))   # 完全逆序
    3
    """
    cnt = 0
    for i in range(len(series)):
        val = series[i]
        for j in range(i):
            if val < series[j]:
                cnt += 1
    return cnt


def mapower_np_vXIIs(price, range_list=None):
    """MA 均线**多头排列能量强度**（`POLYNOMIAL9_NORM` 的生产核）—— 输出 `0..1`。

    逐字搬自老树 `features/base.py:1843`（原为 `@nb.jit('f4[:](f4[:], int16[:])')`）。

    它算什么
    ========
    对每个 bar，取该 bar 上**各条均线**（`range_list` 默认 5..29 共 25 条）的值，
    数它们的**逆序数** —— 逆序数越大，说明**越短的均线反而越低**，即均线是
    「短上长下」的多头排列。再把它归一化：

        ratio = 逆序数 / (窗口数 × (窗口数 − 1)) × 2        ∈ [0, 1]

    分母就是完全逆序的对数上界（`C(25,2)=300` → `300/600*2 = 1`），所以 1.0 = 完美多头排列。

    ⚠️ **前 `max(range_list) − 1` 个 bar 恒为 0**：最长的均线还没成型，无从判断。
    默认 `range_list=5..29` ⇒ **前 28 个 bar 是 0**（不是 NaN）。下游若把它当
    "无信号" 用没问题，但**别当"无数据"**去 drop。

    ⚠️ **`range_list` 里不能有 1**：`r=1` 时 `np.convolve(...)[0:0]` 是**空数组**，
    而左端 `ma_np[0:, k]` 有 `len(price)` 个位置 ⇒ 形状不匹配直接抛。默认值从 5 起，安全。

    >>> import numpy as np
    >>> y = mapower_np_vXIIs(np.arange(40, dtype=np.float32) + 1.0)
    >>> len(y)                                     # 与输入等长
    40
    >>> bool((y[:29] == 0).all())                  # 最长均线成型前恒 0
    True
    >>> round(float(y[-1]), 6)                     # 严格上行 ⇒ 完美多头排列
    1.0
    >>> round(float(mapower_np_vXIIs(np.arange(40, dtype=np.float32)[::-1] + 1.0)[-1]), 6)
    0.0

    :param price: 价格序列（会先转 `float32` —— 老树的 jit 签名 `f4[:]` 就强制了这点）
    :param range_list: 均线窗口列表；`None` = 老树默认的 `5..29`
    :returns: `float32` 数组，与 `price` 等长，值域 `[0, 1]`
    """
    price = np.asarray(price, dtype=np.float32)
    if range_list is None:
        range_list = np.array(range(5, 30)).astype(np.int16)
    else:
        range_list = np.asarray(range_list, dtype=np.int16)

    ma_np = np.empty((len(price), len(range_list)), dtype=np.float32)
    ma_count = 0
    for r in range_list:
        # 手动实现 talib 的 MA：等权卷积。`[r-1 : -r+1]` 取「窗口填满」的那段。
        weights = np.ones(r) / r
        ma_np[r - 1:, ma_count] = np.convolve(weights, price)[r - 1:-r + 1]
        ma_count = ma_count + 1

    ma_max = max(range_list)
    len_range_list = len(range_list)
    num = np.zeros(len(price), dtype=np.float32)
    ratio = np.zeros(len(price), dtype=np.float32)
    for i in range(ma_max, len(price)):
        num[i] = inv_nums(ma_np[i, :])
        ratio[i] = num[i] / (len_range_list * (len_range_list - 1)) * 2

    return ratio
