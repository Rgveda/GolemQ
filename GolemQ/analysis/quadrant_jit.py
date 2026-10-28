# coding:utf-8
"""四量图（`calc_four_quadrant_boost`）与四象限体系的特征计算 —— 逐字搬自老树。

从哪来 / 搬了什么
================
老树 `GolemQ_old/analysis/base.py` 的 `calc_four_quadrant_boost`（119-696，~580 行）
是「四量图」的真身 —— ⚠️ **它不是画图，是特征计算**：Q1–Q4（四条均线结构打分）、
`4QPwrln` 电力线（8 条 MA 的钝化计数 −3.8）、一串 quadrant timing-lag。
「四量图」这个词只活在老树 docstring 里。

⚠️ 老树里的 `QuadSt` / `ChoBits` / `QuadCodex` 状态机**未搬**（2026-10-25，见
`DECISIONS.md` D66）—— 新树没有任何消费方，详见下面「解耦动作」第 4 条。

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
4. **丢掉 `QuadSt` / `ChoBits` / `QuadCodex` 状态机**（老树 `base.py:461-671`）。
   新树**零消费方**：三族 review 契约（`pipeline/review/contract.py`）里没有这三列，
   全树也没有任何读取；老树唯一的消费方 `models/fourquad.py`（`Chobyte` /
   `QChoEngMaj`）**未搬**。实测这条链占本函数约 **32%** 耗时（2,160 根 15min 窗口
   约 13 ms/次），而 `--review` 对每个 60min 目标都调一次 ⇒ 纯浪费。
   ⚠️ **对拍口径**：本函数现在比老树**少这 3 列**，`test_quadrant` 显式排除；
   其余列仍逐值一致（已用 3 seeds × 2 annual × virtual_pressure_line 开关验证）。

⚠️ 老树的平行实现（**只搬了这一条**，见 `DECISIONS.md` D35「只搬一条」）
- 四量图另有 polars 版（`GolemQ_old/models/polars.py:461`）与 notebook 版 —— **未搬**。
- `QuadBsTukLag` 老树有三份副本（`quad_gear.py:648` / `zen_pivot.py:603` / `:1047`）——
  只搬 `quad_gear.py` 的规范版。

⚠️ **本模块不连库、不写库、不被任何管线调用**（用户 2026-10-11：「尽可能解耦，
越少增加引用和管线越好」）。它是**库代码**，消费方自便。
"""
from __future__ import annotations

import warnings

import numba
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
    'hhv',
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
    'calc_mainstream_leadout_feats',
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


def hhv(series, n):
    """滚动最高值（`min_periods=1`）—— :func:`llv` 的镜像。

    >>> hhv(pd.Series([3.0, 1.0, 2.0]), 2).tolist()
    [3.0, 3.0, 2.0]
    """
    return series.rolling(window=n, min_periods=1).max()


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


def _ensure_kdj(ohlc: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    """缺 `KDJ_K/D/J` 时补齐（老树在 `calc_mainstream_feats` 内联的同一段）。"""
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
    return features


def _mainstream_pressure(ohlc: pd.DataFrame, side: str = 'low'):
    """VAR1..VAR5 的**同一套算术**；`side='low'` 吸筹、`'high'` 派发（镜像）。

    - 吸筹：`LOW` / `LLV(LOW,33)` / 向**下**偏离（`LOW-VAR1`）；
    - 派发：`HIGH` / `HHV(HIGH,33)` / 向**上**偏离（`VAR1-HIGH`）。

    返回 `(var5, rising, falling)`：后两者按 VAR5 环比切段 ——
    吸筹口径是「进场 / 洗盘」，派发口径是「派发 / 反抽」。
    """
    if side not in ('low', 'high'):
        raise ValueError("side 只能是 'low' / 'high'：%r" % (side,))
    base = ohlc[AKA.LOW] if side == 'low' else ohlc[AKA.HIGH]

    # VAR1:=REF((LOW+OPEN+CLOSE+HIGH)/4,1);
    var1 = ((ohlc[AKA.LOW] + ohlc[AKA.OPEN] + ohlc[AKA.CLOSE] + ohlc[AKA.HIGH]) / 4).shift(1)

    # VAR2:=SMA(ABS(base-VAR1),13,1)/SMA(MAX(偏离方向,0),10,1);
    diff = (base - var1) if side == 'low' else (var1 - base)
    abs_diff = np.abs(base - var1)
    max_diff = np.maximum(diff, 0)
    sma_abs = sma(abs_diff, 13, 1)
    sma_max = sma(max_diff, 10, 1)
    var2 = sma_abs / np.where(sma_max == 0, 1, sma_max)   # 避免除零

    # VAR3:=EMA(VAR2,10);
    var3 = ema(var2, 10)

    # VAR4:=LLV(LOW,33) / HHV(HIGH,33);
    if side == 'low':
        var4 = llv(base, 33)
        condition = base <= var4
    else:
        var4 = hhv(base, 33)
        condition = base >= var4

    # VAR5:=EMA(IF(创新低/新高,VAR3,0),3);
    var5 = ema(pd.Series(np.where(condition, var3, 0), index=ohlc.index), 3)
    prev = var5.shift(1)
    # 进场/洗盘（吸筹）或 派发/反抽（派发）
    return var5, np.where(var5 > prev, var5, 0), np.where(var5 < prev, var5, 0)


def _mainstream_before(ohlc: pd.DataFrame, features: pd.DataFrame,
                       primary, secondary, *, rising: bool,
                       mult=None, first=None):
    """`Timeline_duration` 包装：`BEFORE`（`mult=None`）与 `x3`/`x5`（`mult=3/5`）。

    `rising=True` 吸筹（KDJ/MACD 向上、`primary` 为上升段、`secondary` 为洗盘）；
    `rising=False` 派发（全部镜像）。
    """
    close = ohlc[AKA.CLOSE]
    j = features[FLD.KDJ_J].diff(1)
    macd = features[FLD.MACD].diff(1)
    turn = (j > 1e-3) if rising else (j < -1e-3)
    macd_turn = (macd > 1e-9) if rising else (macd < -1e-9)
    if mult is None:
        cond = turn & (
            (primary.shift(2) > close.shift(2) * 0.618) &
            (secondary.shift(1) > 0.618) &
            (secondary > 0.168))
    else:
        cond = turn & (
            ((primary.shift(2) > close.shift(2) * mult) &
             (secondary.shift(1) > 0.618) &
             (secondary > 0.168)) |
            ((primary.shift(1) > close.shift(1) * mult) &
             (secondary > close * 0.382) &
             (secondary > 0.168) & macd_turn) |
            ((primary.shift(2) > close.shift(2) * first) &
             (primary.shift(1) > close.shift(1) * mult) &
             (secondary > close * 0.382) &
             (secondary > 0.168) & macd_turn))
    return Timeline_duration(np.where(cond, 1, 0))


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
    features = _ensure_kdj(ohlc, features)

    """计算主力吸筹指标"""
    var5, entry, shakeout = _mainstream_pressure(ohlc, 'low')
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
    features[LTT.MAINSTREAM_LEADIN_ENTRY] = entry
    features[LTT.MAINSTREAM_LEADIN_SHAKEOUT] = shakeout
    features[LTT.MAINSTREAM_LEADIN_BEFORE] = _mainstream_before(
        ohlc, features, features[LTT.MAINSTREAM_LEADIN_ENTRY],
        features[LTT.MAINSTREAM_LEADIN_SHAKEOUT], rising=True)
    features[LTT.MAINSTREAM_LEADINx3_BEFORE] = _mainstream_before(
        ohlc, features, features[LTT.MAINSTREAM_LEADIN_ENTRY],
        features[LTT.MAINSTREAM_LEADIN_SHAKEOUT], rising=True,
        mult=3, first=0.618)
    features[LTT.MAINSTREAM_LEADINx5_BEFORE] = _mainstream_before(
        ohlc, features, features[LTT.MAINSTREAM_LEADIN_ENTRY],
        features[LTT.MAINSTREAM_LEADIN_SHAKEOUT], rising=True,
        mult=5, first=0.832)
    return features


def calc_mainstream_leadout_feats(ohlc: pd.DataFrame,
                                  features: pd.DataFrame) -> pd.DataFrame:
    """主力**派发/出货**（下跌指示）—— :func:`calc_mainstream_feats` 的镜像。

    **新树自造**（老树没有这个指标）：把吸筹口径的
    `LOW / LLV(LOW,33) / 向下偏离` 换成 `HIGH / HHV(HIGH,33) / 向上偏离`，
    KDJ 与 MACD 判据同步反向；系数（0.618 / 0.382 / 0.168 / 3 / 5 / 0.832）
    逐字镜像，便于与原指标并排比较。

    产出 6 列（与 `MAINSTREAM_LEADIN_*` 一一对应）：

    ==============================  ==================================
    `MsLeadoutVar5`                 镜像 VAR5（高位抛压强度）
    `MsLeadoutEntry`                派发：VAR5 环比**上升**的一段
    `MsLeadoutSko`                  反抽：VAR5 环比**下降**的一段
    `MsLeadoutBf` / `x3Bf` / `x5Bf` `Timeline_duration`（0 = 刚触发）
    ==============================  ==================================

    ⚠️ **接入 review 前先校准**：2026-10-27 用 16 码 × 60min 快测，
    镜像 `x5` 触发后 5 根平均 **-0.055%**（原指标 +0.091%），方向对但 edge 弱；
    且原公式的 `ENTRY > close*0.618` 是**绝对价位**量纲，高价股几乎不触发
    （600519 全样本 0 次）。本函数只做算术，**不写库、不被管线调用**。
    """
    features = _ensure_kdj(ohlc, features)

    # "计算主力派发指标"（VAR1..VAR5 的高位镜像）
    var5, distrib, rebound = _mainstream_pressure(ohlc, 'high')
    features = features.reindex(
        columns=list(set([
            *features.columns,
            *[LTT.MAINSTREAM_LEADOUT_VAR5,
              LTT.MAINSTREAM_LEADOUT_ENTRY,
              LTT.MAINSTREAM_LEADOUT_SHAKEOUT,
              LTT.MAINSTREAM_LEADOUT_BEFORE,
              LTT.MAINSTREAM_LEADOUTx3_BEFORE,
              LTT.MAINSTREAM_LEADOUTx5_BEFORE]])))
    features[LTT.MAINSTREAM_LEADOUT_VAR5] = var5
    features[LTT.MAINSTREAM_LEADOUT_ENTRY] = distrib
    features[LTT.MAINSTREAM_LEADOUT_SHAKEOUT] = rebound
    features[LTT.MAINSTREAM_LEADOUT_BEFORE] = _mainstream_before(
        ohlc, features, features[LTT.MAINSTREAM_LEADOUT_ENTRY],
        features[LTT.MAINSTREAM_LEADOUT_SHAKEOUT], rising=False)
    features[LTT.MAINSTREAM_LEADOUTx3_BEFORE] = _mainstream_before(
        ohlc, features, features[LTT.MAINSTREAM_LEADOUT_ENTRY],
        features[LTT.MAINSTREAM_LEADOUT_SHAKEOUT], rising=False,
        mult=3, first=0.618)
    features[LTT.MAINSTREAM_LEADOUTx5_BEFORE] = _mainstream_before(
        ohlc, features, features[LTT.MAINSTREAM_LEADOUT_ENTRY],
        features[LTT.MAINSTREAM_LEADOUT_SHAKEOUT], rising=False,
        mult=5, first=0.832)
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


# --------------------------------------------------------------------------- #
# numba 加速核（唯一实现；原 quadrant.py 的纯 Python 版已删除）
# --------------------------------------------------------------------------- #
def _nb():
    """拿到 numba（已顶层 import）。"""
    return numba


def available() -> bool:
    """numba 在不在。"""
    try:
        import numba as _n  # noqa: F401
        return True
    except ImportError:
        return False


def quadrant_jit_status() -> dict:
    """排障用：numba 版本 + jit 是否真的开了（`DISABLE_JIT` 环境变量会关掉它）。"""
    try:
        import numba as _n
    except ImportError as exc:
        return {'available': False, 'reason': '{}: {}'.format(type(exc).__name__, exc)}
    cfg = getattr(_n, 'config', None)
    return {'available': True, 'numba': _n.__version__,
            'disable_jit': int(getattr(cfg, 'DISABLE_JIT', 0)) if cfg else None}


@numba.jit('int16(float32[:])', cache=True)
def _inv_nums(series):
    """`inv_nums` 的 jit 核：25 条均线切片的逆序对数（int16）。"""
    cnt = 0
    for i in range(len(series)):
        val = series[i]
        for j in range(i):
            if val < series[j]:
                cnt += 1
    return cnt


@numba.jit('float32[:](float32[:], int16[:])', cache=True)
def _mapower(price, range_list):
    """`mapower_np_vXIIs` 的 jit 核（逐字复刻老树）。"""
    ma_np = np.empty((len(price), len(range_list)), dtype=np.float32)
    ma_count = 0
    for r in range_list:
        n = np.ones(r)
        weights = n / r
        ma_np[r - 1:, ma_count] = np.convolve(weights, price)[r - 1:-r + 1]
        ma_count = ma_count + 1

    ma_max = max(range_list)
    len_range_list = len(range_list)
    num = np.zeros(len(price), dtype=np.float32)
    ratio = np.zeros(len(price), dtype=np.float32)
    for i in range(ma_max, len(price)):
        num[i] = _inv_nums(ma_np[i, :])
        ratio[i] = num[i] / (len_range_list * (len_range_list - 1)) * 2
    return ratio


def inv_nums(series) -> int:
    """逆序对数 —— **numba.jit 接口**（显式 `int16(float32[:])`）。"""
    # numba 的 `float32[:]` 不接受 **readonly** 数组；copy=True 保证可写。
    return _inv_nums(np.array(series, dtype=np.float32, copy=True))


def mapower_np_vXIIs(price, range_list=None):
    """`MA` 切片的 `mapower` 归一化 —— **numba.jit 接口**（显式 `float32[:](float32[:], int16[:])`）。

    `range_list=None` 时用老树默认 5..29；wrapper 只负责 dtype/默认值转换。
    """
    arr = np.array(price, dtype=np.float32, copy=True)
    rl = (np.array(range(5, 30)).astype(np.int16) if range_list is None
          else np.array(range_list, dtype=np.int16, copy=True))
    return _mapower(arr, rl)


# 兼容旧名（测试/调用方仍可用 `*_jit`）
inv_nums_jit = inv_nums
mapower_np_vXIIs_jit = mapower_np_vXIIs

__all__ = ['available', 'quadrant_jit_status', 'inv_nums_jit',
           'mapower_np_vXIIs_jit', 'inv_nums', 'mapower_np_vXIIs']
