# coding:utf-8
#
# The MIT License (MIT)
#
# Copyright (c) 2018-2020 azai/Rgveda/GolemQuant
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
基于 QUANTAXIS 的 DataStruct.add_func 使用，也可以单独使用处理 Kline数据，基于技术指标，
用常见技术指标处理 Kline 走势。
"""

import numpy as np
import scipy.stats as scs
import pandas as pd
import numba as nb
from sklearn import preprocessing as skp
try:
    import talib
except:
    print('PLEASE run "pip install TALIB" to call these modules')
    pass
try:
    import QUANTAXIS as QA
    from QUANTAXIS.QAIndicator.base import *
    from QUANTAXIS.QAIndicator.indicators import *
    from QUANTAXIS.QAIndicator.talib_numpy import *
    from QUANTAXIS.QAUtil.QADate_Adv import (
        QA_util_timestamp_to_str,
        QA_util_datetime_to_Unix_timestamp,
        QA_util_print_timestamp
    )
except:
    print('PLEASE run "pip install QUANTAXIS" before call GolemQ.indices.indices modules')
    pass

from GolemQ.indices.base import *
from GolemQ.analysis.timeseries import *
from GolemQ.signal.base import (
    ATR_RSI_Stops_v2,
    ATR_SuperTrend_cross_v2,
    )
from GolemQ.utils.parameter import (
    AKA, 
    INDICATOR_FIELD as FLD, 
    TREND_STATUS as ST
    )
from GolemQ.portfolio.utils import (
    calc_onhold_returns_v2,
)

def kline_returns_func(data, format='pd'):
    """
    计算单个标的每 bar 跟前一bar的利润率差值
    多用途函数，可以是 QA_DataStruct.add_func 调用（可以用于多标的计算），
    也可以是函数式调用（切记单独函数式调用不能多个标的混合计算）。
    Calculating a signal Stock/price timeseries kline's returns.
    For each data[i]/data[i-1] at series value of percentage.

    Parameters
    ----------
    data : (N,) array_like or pd.DataFrame or QA_DataStruct
        传入 OHLC Kline 序列。参数类型可以为 numpy，pd.DataFrame 或者 QA_DataStruct
        The OHLC Kline.
        在传入参数中不可混入多标的数据，如果需要处理多标的价格数据，通过
        QA_DataStruct.add_func 调用。
        It can prossessing multi indices/stocks by QA_DataStruct.add_func
        called. With auto splited into single price series 
        by QA_DataStruct.add_func().
        For standalone called, It should be only pass one stock/price series. 
        If not the result will unpredictable.
    format : str, optional
        返回类型 默认值为 'pd' 将返回 pandas.DataFrame 格式的结果
        可选类型，'np' 或 etc 返回 nparray 格式的结果
        第一个 bar 会被填充为 0.
        Return data format, default is 'pd'. 
        It will return a pandas.DataFrame as result.
        If seted as string: 'np' or etc string value, 
        It will return a nparray as result.
        The first bar will be fill with zero.

    Returns
    -------
    kline returns : pandas.DataFrame or nparray
        'returns' 跟前一收盘价格变化率的百分比

    """
    from QUANTAXIS.QAData.base_datastruct import _quotation_base
    if isinstance(data, pd.DataFrame) or \
        (isinstance(data, _quotation_base)):
        data = data.close

    if (format == 'pd'):
        kline_returns = pd.DataFrame(data.pct_change().values,
                                     columns=['returns'], 
                                     index=data.index)
        #kline_returns = pd.DataFrame(np.nan_to_num(np.log(data /
        #data.shift(1)),
        #                                           nan=0),
        return kline_returns
    else:
        return np.nan_to_num(data.pct_change().values, nan=0)


def macd_cross_np(dif:np.ndarray,
                  dea:np.ndarray,
                  macd:np.ndarray, 
                  N=3,
                  M=21,) -> np.ndarray:
    """
    神一样的指标：MACD
    """
    dif_c0 = dif
    dea_c1 = dea
    macd_c2 = macd

    # 为了避免 warning，进行布尔运算之前必须进行零填充
    dea = np.r_[np.zeros(31), dea[31:]]
    dif = np.r_[np.zeros(24), dif[24:]]
    macd = np.r_[np.zeros(31), macd[31:]]

    # 如果交易日数据达不到31周期长度，填零可能会对不齐，需要判断并且截断
    if (len(dif_c0) != len(dea)):
        dea = dea[-len(dif_c0):]
    if (len(dif_c0) != len(dif)):
        dif = dif[-len(dif_c0):]
    if (len(dif_c0) != len(macd)):
        macd = macd[-len(dif_c0):]

    macd_delta_c3 = np.r_[0, np.diff(macd)]
    dif_shift_1 = np.r_[0, dif[:-1]]
    dea_shift_1 = np.r_[0, dea[:-1]]
    macd_shift_1 = np.r_[0, macd[:-1]]
    macd_cross_jx_before_c5 = np.where((macd > 0) & \
                                       (macd_shift_1 < 0), 1, 0)
    macd_cross_sx_before_c6 = np.where((macd < 0) & \
                                       (macd_shift_1 > 0), 1, 0)
    macd_cross_c4 = np.where(macd_cross_jx_before_c5 == 1, 1, 
                             np.where(macd_cross_sx_before_c6 == 1,
                                      -1, 0))
    macd_cross_jx_before_c5 = Timeline_duration(macd_cross_jx_before_c5)
    macd_cross_sx_before_c6 = Timeline_duration(macd_cross_sx_before_c6)

    dea_cross_jx_before_c7 = Timeline_duration(np.where((dea > 0) & \
                                                        (dea_shift_1 < 0), 1, 0))
    dif_cross_jx_before_c8 = Timeline_duration(np.where((dif > 0) & \
                                                        (dif_shift_1 < 0), 1, 0))
    dea_cross_sx_before_c9 = Timeline_duration(np.where((dea < 0) & \
                                                        (dea_shift_1 > 0), 1, 0))
    dea_slope_c10 = talib.LINEARREG_SLOPE(dea, timeperiod=14)
    macd_tide_median = int(min(np.median(macd_cross_jx_before_c5), 
                               np.median(macd_cross_sx_before_c6)))
    dea_slope_ub_c17 = pd.Series(dea_slope_c10).abs().rolling(macd_tide_median).median()
    dea_slope_change_c18 = (dea_slope_c10 - np.r_[np.nan, 
                                                  np.where(dea_slope_c10[:-1] == 0,
                                                           np.nan, dea_slope_c10[:-1])]) / dea_slope_c10

    negative_lower_price_state = (macd < 0) & \
        (dea < 0) & \
        (macd < dea)
    negative_lower_price_state = (negative_lower_price_state == True) | \
        (macd < 0) & \
        (((dea < 0) & \
        ((dea_cross_sx_before_c9 > 6) | \
        (macd_cross_sx_before_c6 > 12))) | \
        ((dif < 0) & \
        (macd_cross_sx_before_c6 > 12))) & \
        (macd < dea) & \
        (abs(macd) > abs(dea))
    negative_lower_price_c11 = negative_lower_price_state.astype(np.int32)
    negative_lower_price_before_c12 = Timeline_duration(negative_lower_price_c11)

    lower_settle_price_state = ~(negative_lower_price_state == True) & \
        (dea < 0) & \
        (macd_delta_c3 > 0)
    lower_settle_price_c13 = lower_settle_price_state.astype(np.int32)
    lower_settle_price_before_c14 = Timeline_duration(lower_settle_price_c13)

    higher_settle_price_state = (dea > 0) & \
        (macd > dea)
    higher_settle_price_c15 = higher_settle_price_state.astype(np.int32)
    higher_settle_price_before_c16 = Timeline_duration(higher_settle_price_c15)

    # 标准分，z-score 计算在（M个采样中所处的位置）
    dea_norm_c19 = skp.scale(np.c_[dea_c1,
                                   dif_c0,
                                   macd_c2,
                                   macd_delta_c3])

    dea_cross = np.where(dea > 0, 1,
                         np.where(dea < 0, -1, 0))
    dea_zero_timing_lag_c24 = calc_event_timing_lag(dea_cross)
    macd_zero_cross = np.where(macd > 0, 1,
                         np.where(macd < 0, -1, 0))
    macd_zero_timing_lag_c25 = calc_event_timing_lag(macd_zero_cross)

    return np.c_[dif_c0,
                 dea_c1,
                 macd_c2,
                 macd_delta_c3,
                 macd_cross_c4,
                 macd_cross_jx_before_c5,
                 macd_cross_sx_before_c6,
                 dea_cross_jx_before_c7,
                 dif_cross_jx_before_c8,
                 dea_cross_sx_before_c9,
                 dea_slope_c10,
                 negative_lower_price_c11,
                 negative_lower_price_before_c12,
                 lower_settle_price_c13,
                 lower_settle_price_before_c14,
                 higher_settle_price_c15,
                 higher_settle_price_before_c16,
                 dea_slope_ub_c17,
                 dea_slope_change_c18,
                 dea_norm_c19,
                 dea_zero_timing_lag_c24,
                 macd_zero_timing_lag_c25,]


def macd_cross_func(data):
    """
    神一样的指标：MACD
    A pd.DataFrame wrapper for function macd_cross_np()
    此函数只做 np.ndarray 到 pd.DataFrame 的封装，实际计算由
    纯 numpy 完成，便于后期改为 Cython 或者 Numba@jit 优化运行速度。
    """
    if (ST.VERBOSE in data.columns):
        print('Phase macd_cross_func', QA_util_timestamp_to_str())

    MACD = QA.QA_indicator_MACD(data)

    ret_macd_cross = macd_cross_np(dif=MACD.DIF.values,
                                   dea=MACD.DEA.values,
                                   macd=MACD.MACD.values,)

    MACD_CROSS = pd.DataFrame(ret_macd_cross,
                              columns=[FLD.DIF,
                                       FLD.DEA,
                                       FLD.MACD,
                                       FLD.MACD_DELTA,
                                       ST.MACD_CROSS, 
                                       FLD.MACD_CROSS_JX_BEFORE, 
                                       FLD.MACD_CROSS_SX_BEFORE,
                                       FLD.DEA_CROSS_JX_BEFORE,
                                       FLD.DIF_CROSS_JX_BEFORE,
                                       FLD.DEA_CROSS_SX_BEFORE,
                                       FLD.DEA_SLOPE,
                                       FLD.NEGATIVE_LOWER_PRICE,
                                       FLD.NEGATIVE_LOWER_PRICE_BEFORE, 
                                       FLD.LOWER_SETTLE_PRICE,
                                       FLD.LOWER_SETTLE_PRICE_BEFORE,
                                       FLD.HIGHER_SETTLE_PRICE,
                                       FLD.HIGHER_SETTLE_PRICE_BEFORE,
                                       FLD.DEA_SLOPE_UB,
                                       FLD.DEA_SLOPE_CHANGE,
                                       FLD.DEA_NORM,
                                       FLD.DIF_NORM,
                                       FLD.MACD_NORM,
                                       FLD.MACD_DELTA_NORM,
                                       FLD.DEA_ZERO_TIMING_LAG,
                                       FLD.MACD_ZERO_TIMING_LAG,], 
                              index=data.index)

    return MACD_CROSS


def macd_cross_v2_np(dif:np.ndarray,
                     dea:np.ndarray,
                     macd:np.ndarray, 
                     N=3,
                     M=21,
                     annual=252,) -> np.ndarray:
    """
    神一样的指标：MACD
    """
    dif_c0 = dif
    dea_c1 = dea
    macd_c2 = macd

    # 为了避免 warning，进行布尔运算之前必须进行零填充
    dea = np.r_[np.zeros(31), dea[31:]]
    dif = np.r_[np.zeros(24), dif[24:]]
    macd = np.r_[np.zeros(31), macd[31:]]

    # 如果交易日数据达不到31周期长度，填零可能会对不齐，需要判断并且截断
    if (len(dif_c0) != len(dea)):
        dea = dea[-len(dif_c0):]
    if (len(dif_c0) != len(dif)):
        dif = dif[-len(dif_c0):]
    if (len(dif_c0) != len(macd)):
        macd = macd[-len(dif_c0):]

    macd_delta_c3 = np.r_[0, np.diff(macd)]
    dif_shift_1 = np.r_[0, dif[:-1]]
    dea_shift_1 = np.r_[0, dea[:-1]]
    macd_shift_1 = np.r_[0, macd[:-1]]

    macd_cross_jx_before_c5 = np.where((macd > 0) & \
                                       (macd_shift_1 < 0), 1, 0)
    macd_cross_sx_before_c6 = np.where((macd < 0) & \
                                       (macd_shift_1 > 0), 1, 0)
    macd_cross_c4 = np.where((macd > 0) & \
                             (macd_delta_c3 > 0), 1, 
                             np.where((macd < 0) & \
                                      (macd_delta_c3 < 0),
                                      -1, 0))
    macd_cross_jx = Timeline_Integral(np.where(macd_cross_c4 > 0, 1, 0))
    macd_cross_sx = np.sign(macd_cross_c4) * Timeline_Integral(np.where(macd_cross_c4 < 0, 1, 0))
    macd_trend_timing_lag_c22 = macd_cross_jx + macd_cross_sx

    macd_cross_jx_before_c5 = Timeline_duration(macd_cross_jx_before_c5)
    macd_cross_sx_before_c6 = Timeline_duration(macd_cross_sx_before_c6)

    dea_cross_jx_before_c7 = Timeline_duration(np.where((dea > 0) & \
                                                        (dea_shift_1 < 0), 1, 0))
    dif_cross_jx_before_c8 = Timeline_duration(np.where((dif > 0) & \
                                                        (dif_shift_1 < 0), 1, 0))
    dea_cross_sx_before_c9 = Timeline_duration(np.where((dea < 0) & \
                                                        (dea_shift_1 > 0), 1, 0))
    dea_slope_c10 = talib.LINEARREG_SLOPE(dea, timeperiod=14)
    macd_tide_median = int(min(np.median(macd_cross_jx_before_c5), 
                               np.median(macd_cross_sx_before_c6)))
    dea_slope_ub_c17 = pd.Series(dea_slope_c10).abs().rolling(macd_tide_median).median()
    dea_slope_change_c18 = (dea_slope_c10 - np.r_[np.nan, 
                                                  np.where(dea_slope_c10[:-1] == 0,
                                                           np.nan, dea_slope_c10[:-1])]) / dea_slope_c10

    negative_lower_price_state = (macd < 0) & (dea < 0) & (macd < dea)
    negative_lower_price_state = (negative_lower_price_state == True) | \
        (macd < 0) & (((dea < 0) & ((dea_cross_sx_before_c9 > 6) | \
        (macd_cross_sx_before_c6 > 12))) | \
        ((dif < 0) & (macd_cross_sx_before_c6 > 12))) & \
        (macd < dea) & (abs(macd) > abs(dea))
    negative_lower_price_c11 = negative_lower_price_state.astype(np.int32)
    negative_lower_price_before_c12 = Timeline_duration(negative_lower_price_c11)

    lower_settle_price_state = ~(negative_lower_price_state == True) & \
        (dea < 0) & \
        (macd_delta_c3 > 0)
    lower_settle_price_c13 = lower_settle_price_state.astype(np.int32)
    lower_settle_price_before_c14 = Timeline_duration(lower_settle_price_c13)

    higher_settle_price_state = (dea > 0) & (macd > dea)
    higher_settle_price_c15 = higher_settle_price_state.astype(np.int32)
    higher_settle_price_before_c16 = Timeline_duration(higher_settle_price_c15)

    # 标准分，z-score 计算在（M个采样中所处的位置）
    dea_norm_c19 = skp.scale(np.c_[dea_c1,
                                   dif_c0,
                                   macd_c2,
                                   macd_delta_c3])

    dea_intercept_c20 = lineareg_intercept(dea_slope_c10, 
                                          dea_c1, 
                                          np.zeros(len(dea_c1)), 
                                          np.zeros(len(dea_c1)))
    with np.errstate(invalid='ignore', divide='ignore'):
        dea_intercept_c20 = np.where(dea < 0, dea_intercept_c20, -dea_intercept_c20)
        dea_intercept_cross = np.where(dea_intercept_c20 > 0, 1,
                                       np.where(dea_intercept_c20 < 0, -1, 0))
    dea_intercept_timing_lag = calc_event_timing_lag(dea_intercept_cross)

    macd_intercept_c21 = (0 - dea_c1) / pd.Series(macd_delta_c3).rolling(4).mean()
    macd_intercept_c21 = np.where(dea < 0, macd_intercept_c21, -macd_intercept_c21)

    dea_cross = np.where(dea > 0, 1,
                         np.where(dea < 0, -1, 0))
    dea_zero_timing_lag_c24 = calc_event_timing_lag(dea_cross)
    dif_cross = np.where(dif > 0, 1,
                         np.where(dif < 0, -1, 0))
    dif_zero_timing_lag_c25 = calc_event_timing_lag(dif_cross)
    macd_zero_cross = np.where(macd > 0, 1,
                         np.where(macd < 0, -1, 0))
    macd_zero_timing_lag_c26 = calc_event_timing_lag(macd_zero_cross)
    macd_max_c27 = np.where((dea > 0) & \
        (macd > 0) & \
        (macd > macd_shift_1), macd, 
                            np.where((dea < 0) & \
        (macd < 0) & \
        (macd < macd_shift_1), macd, np.nan))

    dif_max_c28 = np.where((dea > 0) & \
        (dif > 0) & \
        (dif > dif_shift_1), macd, 
                            np.where((dea < 0) & \
        (dif < 0) & \
        (dif < dif_shift_1), macd, np.nan))

    with np.errstate(invalid='ignore', divide='ignore'):
        dea_zero_turnover = np.where(np.sign(dea) != np.sign(dea_shift_1), 1, 0)
        dea_zero_turnover_ratio_c29 = rolling_sum(dea_zero_turnover, annual)
        dea_zero_turnover_days_c30 = (np.full((len(dea_c1)), 252) / (dea_zero_turnover_ratio_c29 / 2))

    return np.c_[dif_c0,
                 dea_c1,
                 macd_c2,
                 macd_delta_c3,
                 macd_cross_c4,
                 macd_cross_jx_before_c5,
                 macd_cross_sx_before_c6,
                 dea_cross_jx_before_c7,
                 dif_cross_jx_before_c8,
                 dea_cross_sx_before_c9,
                 dea_slope_c10,
                 negative_lower_price_c11,
                 negative_lower_price_before_c12,
                 lower_settle_price_c13,
                 lower_settle_price_before_c14,
                 higher_settle_price_c15,
                 higher_settle_price_before_c16,
                 dea_slope_ub_c17,
                 dea_slope_change_c18,
                 dea_norm_c19,
                 dea_intercept_c20,
                 macd_intercept_c21,
                 macd_trend_timing_lag_c22,
                 dea_intercept_timing_lag,
                 dea_zero_timing_lag_c24,
                 dif_zero_timing_lag_c25,
                 macd_zero_timing_lag_c26,
                 dea_zero_turnover_ratio_c29 / 2,
                 dea_zero_turnover_days_c30,]


def macd_cross_v2_func(data, *args, **kwargs):
    """
    神一样的指标：MACD
    A pd.DataFrame wrapper for function macd_cross_np()
    此函数只做 np.ndarray 到 pd.DataFrame 的封装，实际计算由
    纯 numpy 完成，便于后期改为 Cython 或者 Numba@jit 优化运行速度。
    支持QA add_func，第二个参数 默认为 indices= 为已经计算指标
    理论上这个函数只计算单一标的，不要尝试传递复杂标的，indices会尝试拆分。
    """
    if (ST.VERBOSE in data.columns):
        print('Phase macd_cross_func', QA_util_timestamp_to_str())

    # 针对多标的，拆分 indices 数据再自动合并
    code = data.index.get_level_values(level=1)[0]
    if ('indices' in kwargs.keys()):
        indices = kwargs['indices'].loc[(slice(None), code), :]
    elif (len(args) > 0):
        indices = args[0].loc[(slice(None), code), :]
    else:
        indices = None
    annual = kwargs['annual'] if ('annual' in kwargs.keys()) else 252
    
    MACD = QA.QA_indicator_MACD(data)

    ret_macd_cross = macd_cross_v2_np(dif=MACD.DIF.values,
                                   dea=MACD.DEA.values,
                                   macd=MACD.MACD.values,
                                   annual=annual)

    MACD_CROSS = pd.DataFrame(ret_macd_cross,
                              columns=[FLD.DIF,
                                       FLD.DEA,
                                       FLD.MACD,
                                       FLD.MACD_DELTA,
                                       ST.MACD_CROSS, 
                                       FLD.MACD_CROSS_JX_BEFORE, 
                                       FLD.MACD_CROSS_SX_BEFORE,
                                       FLD.DEA_CROSS_JX_BEFORE,
                                       FLD.DIF_CROSS_JX_BEFORE,
                                       FLD.DEA_CROSS_SX_BEFORE,
                                       FLD.DEA_SLOPE,
                                       FLD.NEGATIVE_LOWER_PRICE,
                                       FLD.NEGATIVE_LOWER_PRICE_BEFORE, 
                                       FLD.LOWER_SETTLE_PRICE,
                                       FLD.LOWER_SETTLE_PRICE_BEFORE,
                                       FLD.HIGHER_SETTLE_PRICE,
                                       FLD.HIGHER_SETTLE_PRICE_BEFORE,
                                       FLD.DEA_SLOPE_UB,
                                       FLD.DEA_SLOPE_CHANGE,
                                       FLD.DEA_NORM,
                                       FLD.DIF_NORM,
                                       FLD.MACD_NORM,
                                       FLD.MACD_DELTA_NORM,
                                       FLD.DEA_INTERCEPT,
                                       FLD.MACD_INTERCEPT,
                                       FLD.MACD_TREND_TIMING_LAG,
                                       FLD.DEA_INTERCEPT_TIMING_LAG,
                                       FLD.DEA_ZERO_TIMING_LAG,
                                       FLD.DIF_ZERO_TIMING_LAG,
                                       FLD.MACD_ZERO_TIMING_LAG,
                                       FLD.DEA_ZERO_TURNOVER_RATIO,
                                       FLD.DEA_ZERO_TURNOVER_DAYS], 
                              index=data.index)

    if (indices is None):
        return MACD_CROSS
    else:
        return pd.concat([indices,
                          MACD_CROSS],
                         axis=1)


def macd_cross_v3_np(dif:np.ndarray,
                     dea:np.ndarray,
                     macd:np.ndarray, 
                     N=3,
                     M=21,
                     annual=252,) -> np.ndarray:
    """
    神一样的指标：MACD
    """
    dif_c0 = dif
    dea_c1 = dea
    macd_c2 = macd

    # 为了避免 warning，进行布尔运算之前必须进行零填充
    dea = np.r_[np.zeros(31), dea[31:]]
    dif = np.r_[np.zeros(24), dif[24:]]
    macd = np.r_[np.zeros(31), macd[31:]]

    # 如果交易日数据达不到31周期长度，填零可能会对不齐，需要判断并且截断
    if (len(dif_c0) != len(dea)):
        dea = dea[-len(dif_c0):]
    if (len(dif_c0) != len(dif)):
        dif = dif[-len(dif_c0):]
    if (len(dif_c0) != len(macd)):
        macd = macd[-len(dif_c0):]

    macd_delta_c3 = np.r_[0, np.diff(macd)]
    dif_shift_1 = np.r_[0, dif[:-1]]
    dea_shift_1 = np.r_[0, dea[:-1]]
    macd_shift_1 = np.r_[0, macd[:-1]]

    macd_cross_jx_before_c5 = np.where((macd > 0) & \
                                       (macd_shift_1 < 0), 1, 0)
    macd_cross_sx_before_c6 = np.where((macd < 0) & \
                                       (macd_shift_1 > 0), 1, 0)
    macd_cross_c4 = np.where((macd > 0) & \
                             (macd_delta_c3 > 0), 1, 
                             np.where((macd < 0) & \
                                      (macd_delta_c3 < 0),
                                      -1, 0))
    macd_cross_jx = Timeline_Integral(np.where(macd_cross_c4 > 0, 1, 0))
    macd_cross_sx = np.sign(macd_cross_c4) * Timeline_Integral(np.where(macd_cross_c4 < 0, 1, 0))
    macd_trend_timing_lag_c22 = macd_cross_jx + macd_cross_sx

    macd_cross_jx_before_c5 = Timeline_duration(macd_cross_jx_before_c5)
    macd_cross_sx_before_c6 = Timeline_duration(macd_cross_sx_before_c6)

    dea_cross_jx_before_c7 = Timeline_duration(np.where((dea > 0) & \
                                                        (dea_shift_1 < 0), 1, 0))
    dif_cross_jx_before_c8 = Timeline_duration(np.where((dif > 0) & \
                                                        (dif_shift_1 < 0), 1, 0))
    dea_cross_sx_before_c9 = Timeline_duration(np.where((dea < 0) & \
                                                        (dea_shift_1 > 0), 1, 0))
    dea_slope_c10 = talib.LINEARREG_SLOPE(dea, timeperiod=14)
    with np.errstate(invalid='ignore', divide='ignore'):
        dea_slope_lag_c17 = calc_event_timing_lag(np.where((dea_slope_c10 > 0), 1,
                                                           np.where((dea_slope_c10 < 0), -1, 0)))
    negative_lower_price_state = (macd < 0) & (dea < 0) & (macd < dea)
    negative_lower_price_state = (negative_lower_price_state == True) | \
        (macd < 0) & (((dea < 0) & ((dea_cross_sx_before_c9 > 6) | \
        (macd_cross_sx_before_c6 > 12))) | \
        ((dif < 0) & (macd_cross_sx_before_c6 > 12))) & \
        (macd < dea) & (abs(macd) > abs(dea))
    negative_lower_price_c11 = negative_lower_price_state.astype(np.int32)
    negative_lower_price_before_c12 = Timeline_duration(negative_lower_price_c11)

    lower_settle_price_state = ~(negative_lower_price_state == True) & \
        (dea < 0) & \
        (macd_delta_c3 > 0)
    lower_settle_price_c13 = lower_settle_price_state.astype(np.int32)
    lower_settle_price_before_c14 = Timeline_duration(lower_settle_price_c13)

    higher_settle_price_state = (dea > 0) & (macd > dea)
    higher_settle_price_c15 = higher_settle_price_state.astype(np.int32)
    higher_settle_price_before_c16 = Timeline_duration(higher_settle_price_c15)

    # 标准分，z-score 计算在（M个采样中所处的位置）
    dea_norm_c18 = skp.scale(np.c_[dea_c1,
                                   dif_c0,
                                   macd_c2,
                                   macd_delta_c3])

    dea_intercept_c20 = lineareg_intercept(dea_slope_c10, 
                                          dea_c1, 
                                          np.zeros(len(dea_c1)), 
                                          np.zeros(len(dea_c1)))
    with np.errstate(invalid='ignore', divide='ignore'):
        dea_intercept_c20 = np.where(dea < 0, dea_intercept_c20, -dea_intercept_c20)
        dea_intercept_cross = np.where(dea_intercept_c20 > 0, 1,
                                       np.where(dea_intercept_c20 < 0, -1, 0))
    dea_intercept_timing_lag = calc_event_timing_lag(dea_intercept_cross)

    macd_intercept_c21 = (0 - dea_c1) / pd.Series(macd_delta_c3).rolling(4).mean()
    macd_intercept_c21 = np.where(dea < 0, macd_intercept_c21, -macd_intercept_c21)

    dea_cross = np.where(dea > 0, 1,
                         np.where(dea < 0, -1, 0))
    dea_zero_timing_lag_c24 = calc_event_timing_lag(dea_cross)
    dif_cross = np.where(dif > 0, 1,
                         np.where(dif < 0, -1, 0))
    dif_zero_timing_lag_c25 = calc_event_timing_lag(dif_cross)
    macd_zero_cross = np.where(macd > 0, 1,
                         np.where(macd < 0, -1, 0))
    macd_zero_timing_lag_c26 = calc_event_timing_lag(macd_zero_cross)
    macd_max_c27 = np.where((dea > 0) & \
        (macd > 0) & \
        (macd > macd_shift_1), macd, 
                            np.where((dea < 0) & \
        (macd < 0) & \
        (macd < macd_shift_1), macd, np.nan))

    dif_max_c28 = np.where((dea > 0) & \
        (dif > 0) & \
        (dif > dif_shift_1), macd, 
                            np.where((dea < 0) & \
        (dif < 0) & \
        (dif < dif_shift_1), macd, np.nan))

    with np.errstate(invalid='ignore', divide='ignore'):
        dea_zero_turnover = np.where(np.sign(dea) != np.sign(dea_shift_1), 1, 0)
        dea_zero_turnover_ratio_c29 = rolling_sum(dea_zero_turnover, annual)
        dea_zero_turnover_days_c30 = (np.full((len(dea_c1)), 252) / (dea_zero_turnover_ratio_c29 / 2))

    return np.c_[dif_c0,
                 dea_c1,
                 macd_c2,
                 macd_delta_c3,
                 macd_cross_c4,
                 macd_cross_jx_before_c5,
                 macd_cross_sx_before_c6,
                 dea_cross_jx_before_c7,
                 dif_cross_jx_before_c8,
                 dea_cross_sx_before_c9,
                 dea_slope_c10,
                 negative_lower_price_c11,
                 negative_lower_price_before_c12,
                 lower_settle_price_c13,
                 lower_settle_price_before_c14,
                 higher_settle_price_c15,
                 higher_settle_price_before_c16,
                 dea_slope_lag_c17,
                 dea_norm_c18,
                 dea_intercept_c20,
                 macd_intercept_c21,
                 macd_trend_timing_lag_c22,
                 dea_intercept_timing_lag,
                 dea_zero_timing_lag_c24,
                 dif_zero_timing_lag_c25,
                 macd_zero_timing_lag_c26,
                 dea_zero_turnover_ratio_c29 / 2,
                 dea_zero_turnover_days_c30,]


def macd_cross_v3_func(data, *args, **kwargs):
    """
    神一样的指标：MACD
    A pd.DataFrame wrapper for function macd_cross_np()
    此函数只做 np.ndarray 到 pd.DataFrame 的封装，实际计算由
    纯 numpy 完成，便于后期改为 Cython 或者 Numba@jit 优化运行速度。
    支持QA add_func，第二个参数 默认为 indices= 为已经计算指标
    理论上这个函数只计算单一标的，不要尝试传递复杂标的，indices会尝试拆分。
    """
    if (ST.VERBOSE in data.columns):
        print('Phase macd_cross_func', QA_util_timestamp_to_str())

    # 针对多标的，拆分 indices 数据再自动合并
    code = data.index.get_level_values(level=1)[0]
    if ('indices' in kwargs.keys()):
        indices = kwargs['indices'].loc[(slice(None), code), :]
    elif (len(args) > 0):
        indices = args[0].loc[(slice(None), code), :]
    else:
        indices = None
    annual = kwargs['annual'] if ('annual' in kwargs.keys()) else 252
    
    MACD = QA.QA_indicator_MACD(data)

    ret_macd_cross = macd_cross_v3_np(dif=MACD.DIF.values,
                                   dea=MACD.DEA.values,
                                   macd=MACD.MACD.values,
                                   annual=annual)

    MACD_CROSS = pd.DataFrame(ret_macd_cross,
                              columns=[FLD.DIF,
                                       FLD.DEA,
                                       FLD.MACD,
                                       FLD.MACD_DELTA,
                                       ST.MACD_CROSS, 
                                       FLD.MACD_CROSS_JX_BEFORE, 
                                       FLD.MACD_CROSS_SX_BEFORE,
                                       FLD.DEA_CROSS_JX_BEFORE,
                                       FLD.DIF_CROSS_JX_BEFORE,
                                       FLD.DEA_CROSS_SX_BEFORE,
                                       FLD.DEA_SLOPE,
                                       FLD.NEGATIVE_LOWER_PRICE,
                                       FLD.NEGATIVE_LOWER_PRICE_BEFORE, 
                                       FLD.LOWER_SETTLE_PRICE,
                                       FLD.LOWER_SETTLE_PRICE_BEFORE,
                                       FLD.HIGHER_SETTLE_PRICE,
                                       FLD.HIGHER_SETTLE_PRICE_BEFORE,
                                       FLD.DEA_SLOPE_TIMING_LAG,
                                       FLD.DEA_NORM,
                                       FLD.DIF_NORM,
                                       FLD.MACD_NORM,
                                       FLD.MACD_DELTA_NORM,
                                       FLD.DEA_INTERCEPT,
                                       FLD.MACD_INTERCEPT,
                                       FLD.MACD_TREND_TIMING_LAG,
                                       FLD.DEA_INTERCEPT_TIMING_LAG,
                                       FLD.DEA_ZERO_TIMING_LAG,
                                       FLD.DIF_ZERO_TIMING_LAG,
                                       FLD.MACD_ZERO_TIMING_LAG,
                                       FLD.DEA_ZERO_TURNOVER_RATIO,
                                       FLD.DEA_ZERO_TURNOVER_DAYS], 
                              index=data.index)

    if (indices is None):
        return MACD_CROSS
    else:
        return pd.concat([indices,
                          MACD_CROSS],
                          axis=1)


def macd_cross_func_pd(data):
    """
    神一样的指标：MACD
    """
    if (ST.VERBOSE in data.columns):
        print('Phase macd_cross_func', QA_util_timestamp_to_str())
    MACD = QA.QA_indicator_MACD(data)
    
    MACD_CROSS = pd.DataFrame(columns=[ST.MACD_CROSS, 
                                       FLD.MACD_CROSS_JX_BEFORE, 
                                       FLD.MACD_CROSS_SX_BEFORE, 
                                       FLD.NEGATIVE_LOWER_PRICE,
                                       FLD.NEGATIVE_LOWER_PRICE_BEFORE, 
                                       FLD.LOWER_SETTLE_PRICE,
                                       FLD.LOWER_SETTLE_PRICE_BEFORE,
                                       FLD.HIGHER_SETTLE_PRICE,
                                       FLD.HIGHER_SETTLE_PRICE_BEFORE], 
                              index=data.index)
    MACD_CROSS = MACD_CROSS.assign(DIF=MACD[FLD.DIF])
    MACD_CROSS = MACD_CROSS.assign(DEA=MACD[FLD.DEA])
    MACD_CROSS = MACD_CROSS.assign(MACD=MACD[FLD.MACD])
    MACD_CROSS = MACD_CROSS.assign(ZERO=0)
    # 新版考虑合并指标，将 DELTA 重命名为 MACD_DELTA
    MACD_CROSS = MACD_CROSS.assign(MACD_DELTA=MACD[FLD.MACD].diff())

    MACD_CROSS[FLD.MACD_CROSS_JX_BEFORE] = CROSS(MACD_CROSS[FLD.DIF], 
                                                 MACD_CROSS[FLD.DEA])
    MACD_CROSS[FLD.MACD_CROSS_SX_BEFORE] = CROSS(MACD_CROSS[FLD.DEA], 
                                                 MACD_CROSS[FLD.DIF])
    MACD_CROSS[ST.MACD_CROSS] = np.where(MACD_CROSS[FLD.MACD_CROSS_JX_BEFORE] == 1, 1, 
                                        np.where(MACD_CROSS[FLD.MACD_CROSS_SX_BEFORE] == 1, 
                                                 -1, 0))

    MACD_CROSS[FLD.DEA_CROSS_JX_BEFORE] = Timeline_duration(CROSS(MACD_CROSS[FLD.DEA], 
                                                                  MACD_CROSS[FLD.ZERO]).values)
    MACD_CROSS[FLD.DIF_CROSS_JX_BEFORE] = Timeline_duration(CROSS(MACD_CROSS[FLD.DIF], 
                                                                  MACD_CROSS[FLD.ZERO]).values)

    MACD_CROSS[FLD.DEA_CROSS_SX_BEFORE] = Timeline_duration(CROSS(MACD_CROSS[FLD.ZERO], 
                                                                  MACD_CROSS[FLD.DEA]).values)

    MACD_CROSS[FLD.MACD_CROSS_JX_BEFORE] = Timeline_duration(MACD_CROSS[FLD.MACD_CROSS_JX_BEFORE].values)
    MACD_CROSS[FLD.MACD_CROSS_SX_BEFORE] = Timeline_duration(MACD_CROSS[FLD.MACD_CROSS_SX_BEFORE].values)
    MACD_CROSS[FLD.DEA_SLOPE] = talib.LINEARREG_SLOPE(MACD[FLD.DEA], timeperiod=14)
    MACD_CROSS['MACD_TIDE_MEDIAN'] = int(min(MACD_CROSS[FLD.MACD_CROSS_JX_BEFORE].median(), 
                                             MACD_CROSS[FLD.MACD_CROSS_SX_BEFORE].median()))
    MACD_CROSS[FLD.DEA_SLOPE_UB] = MACD_CROSS[FLD.DEA_SLOPE].abs().rolling(MACD_CROSS['MACD_TIDE_MEDIAN'].max()).median()

    negative_lower_price_state = (MACD_CROSS[FLD.MACD] < 0) & \
        (MACD_CROSS[FLD.DEA] < 0) & \
        (MACD_CROSS[FLD.MACD] < MACD_CROSS[FLD.DEA])
    negative_lower_price_state = (negative_lower_price_state == True) | \
        (MACD_CROSS[FLD.MACD] < 0) & \
        (((MACD_CROSS[FLD.DEA] < 0) & \
        ((MACD_CROSS[FLD.DEA_CROSS_SX_BEFORE] > 6) | \
        (MACD_CROSS[FLD.MACD_CROSS_SX_BEFORE] > 12))) | \
        ((MACD_CROSS[FLD.DIF] < 0) & \
        (MACD_CROSS[FLD.MACD_CROSS_SX_BEFORE] > 12))) & \
        (MACD_CROSS[FLD.MACD] < MACD_CROSS[FLD.DEA]) & \
        (abs(MACD_CROSS[FLD.MACD]) > abs(MACD_CROSS[FLD.DEA]))
    MACD_CROSS[FLD.NEGATIVE_LOWER_PRICE] = negative_lower_price_state.apply(int)
    MACD_CROSS[FLD.NEGATIVE_LOWER_PRICE_BEFORE] = Timeline_duration(MACD_CROSS[FLD.NEGATIVE_LOWER_PRICE].values)

    lower_settle_price_state = ~(negative_lower_price_state == True) & \
        (MACD_CROSS[FLD.DEA] < 0) & \
        (MACD_CROSS[FLD.MACD_DELTA] > 0)
    MACD_CROSS[FLD.LOWER_SETTLE_PRICE] = lower_settle_price_state.apply(int)
    MACD_CROSS[FLD.LOWER_SETTLE_PRICE_BEFORE] = Timeline_duration(MACD_CROSS[FLD.LOWER_SETTLE_PRICE].values)

    higher_settle_price_state = (MACD_CROSS[FLD.DEA] > 0) & \
        (MACD_CROSS[FLD.MACD] > MACD_CROSS[FLD.DEA])
    MACD_CROSS[FLD.HIGHER_SETTLE_PRICE] = higher_settle_price_state.apply(int)
    MACD_CROSS[FLD.HIGHER_SETTLE_PRICE_BEFORE] = Timeline_duration(MACD_CROSS[FLD.HIGHER_SETTLE_PRICE].values)

    return MACD_CROSS


def kdj_cross_np(closep:np.ndarray,
                 highp:np.ndarray,
                 lowp:np.ndarray,) -> np.ndarray:
    """
    指标：KDJ 金叉
    为了避免 Warning，计算时忽略了前13个 NaN 的，最后 加入 ret 的时候补回来
    """
    KDJ = TA_KDJ(highp, lowp, closep)
    
    with np.errstate(invalid='ignore', divide='ignore'):
        kdj_cross = np.where(KDJ[:,2] > KDJ[:,0], 1, -1)
    kdj_cross_jx = Timeline_Integral(np.where(kdj_cross == 1, 1, 0))
    kdj_cross_sx = np.sign(kdj_cross) * Timeline_Integral(np.where(kdj_cross == -1, 1, 0))
    return np.c_[kdj_cross,
                 kdj_cross_jx + kdj_cross_sx,]


def rsi_cross_np(closep:np.ndarray,) -> np.ndarray:
    """
    指标：RSI 金叉
    为了避免 Warning，计算时忽略了前13个 NaN 的，最后 加入 ret 的时候补回来
    """
    rsi6 = talib.RSI(closep, timeperiod=6)
    rsi12 = talib.RSI(closep, timeperiod=12)
    rsi24 = talib.RSI(closep, timeperiod=24)

    with np.errstate(invalid='ignore', divide='ignore'):
        rsi_cross = np.where((rsi6 > rsi12) & \
                             (rsi12 > rsi24), 1, 
                             np.where((rsi6 < rsi12) & \
                                      (rsi12 < rsi24), -1, 0))
    rsi_cross_jx = Timeline_Integral(np.where(rsi_cross == 1, 1, 0))
    rsi_cross_sx = np.sign(rsi_cross) * Timeline_Integral(np.where(rsi_cross == -1, 1, 0))
    return np.c_[rsi_cross,
                 rsi_cross_jx + rsi_cross_sx,]


def maxfactor_cross_np(closep:np.ndarray,
                       highp:np.ndarray,
                       lowp:np.ndarray,) -> np.ndarray:
    """
    自创指标：MAXFACTOR
    """
    RSI = TA_RSI(closep, timeperiod=12)
    CCI = TA_CCI(highp, lowp, closep)
    KDJ = TA_KDJ(highp, lowp, closep) 
    MAX_FACTOR = CCI[:,0] + (RSI[:,0] - 50) * 4 + (KDJ[:,2] - 50) * 4
    MAX_FACTOR_delta = np.r_[np.nan, np.diff(MAX_FACTOR)]

    rsi_c0 = RSI[:,0]
    cci_c1 = CCI[:,0]
    mft_c2 = MAX_FACTOR
    mft_delta_c3 = MAX_FACTOR_delta
    mft_bselin_c4 = (RSI[:,0] - 50) * 4

    if (len(closep) > 30):
        mft_ema = talib.EMA(MAX_FACTOR, timeperiod=21)
    if (len(closep) > 21):
        mft_ema = talib.EMA(MAX_FACTOR, timeperiod=5)
    else:
        mft_ema = MAX_FACTOR

    mft_flow = np.nan_to_num(MAX_FACTOR - mft_ema, nan=0)
    mft_cross_c5 = np.where(mft_flow > 0, 1,
                            np.where(mft_flow < 0, -1, 0))
    mft_trend_timing_lag_c9 = calc_event_timing_lag(mft_cross_c5)

    mft_cross_dif = np.nan_to_num(np.r_[0, np.diff(mft_cross_c5)], nan=0)
    mft_cross_jx_c6 = Timeline_duration(np.where((mft_flow > 0) & \
                                                 (mft_cross_dif > 0), 1, 0))
    mft_cross_sx_c7 = Timeline_duration(np.where((mft_flow < 0) & \
                                                 (mft_cross_dif < 0), 1, 0))
    
    rsi_delta_c8 = np.r_[0, np.diff(rsi_c0)]
    return np.c_[rsi_c0,
                 cci_c1,
                 mft_c2,
                 mft_delta_c3,
                 mft_bselin_c4,
                 mft_cross_c5,
                 mft_cross_jx_c6,
                 mft_cross_sx_c7,
                 rsi_delta_c8,
                 mft_trend_timing_lag_c9,]


def maxfactor_cross_func(data):
    """
    自创指标：MAXFACTOR
    A pd.DataFrame wrapper for function maxfactor_cross_np()
    此函数只做 np.ndarray 到 pd.DataFrame 的封装，实际计算由
    纯 numpy 完成，便于后期改为 Cython 或者 Numba@jit 优化运行速度。
    """
    if (ST.VERBOSE in data.columns):
        print('Phase maxfactor_cross_func', QA_util_timestamp_to_str())

    ret_mft_cross = maxfactor_cross_np(closep=data.close.values,
                                       highp=data.high.values,
                                       lowp=data.low.values,)

    MFT_CROSS = pd.DataFrame(ret_mft_cross,
                             columns=[FLD.RSI,
                                      FLD.CCI,
                                      FLD.MAXFACTOR,
                                      FLD.MAXFACTOR_DELTA,
                                      FLD.MAXFACTOR_BASELINE,
                                      FLD.MAXFACTOR_CROSS, 
                                      FLD.MAXFACTOR_CROSS_JX, 
                                      FLD.MAXFACTOR_CROSS_SX,
                                      FLD.RSI_DELTA,
                                      FLD.MAXFACTOR_TREND_TIMING_LAG,],
                             index=data.index)

    return MFT_CROSS


def maxfactor_cross_func_pd(data):
    """
    自创指标：MAXFACTOR
    """
    if (ST.VERBOSE in data.columns):
        print('Phase maxfactor_cross_func', QA_util_timestamp_to_str())
    RSI = TA_RSI(data.close, timeperiod=12)
    CCI = TA_CCI(data.high, data.low, data.close)
    KDJ = TA_KDJ(data.high, data.low, data.close)    
    MAX_FACTOR = CCI[:,0] + (RSI[:,0] - 50) * 4 + (KDJ[:,2] - 50) * 4
    MAX_FACTOR_delta = np.r_[np.nan, np.diff(MAX_FACTOR)]

    MFT_CROSS = pd.DataFrame(columns=[FLD.RSI,
                                            FLD.CCI,
                                            FLD.MAXFACTOR,
                                            FLD.MAXFACTOR_DELTA,
                                            FLD.MAXFACTOR_BASELINE,
                                            FLD.MAXFACTOR_CROSS, 
                                            FLD.MAXFACTOR_CROSS_JX, 
                                            FLD.MAXFACTOR_CROSS_SX],
                             index=data.index)
    MFT_CROSS[FLD.RSI] = RSI[:,0]
    MFT_CROSS[FLD.CCI] = CCI[:,0]
    MFT_CROSS[FLD.MAXFACTOR] = MAX_FACTOR
    MFT_CROSS[FLD.MAXFACTOR_DELTA] = MAX_FACTOR_delta
    MFT_CROSS[FLD.MAXFACTOR_BASELINE] = (RSI[:,0] - 50) * 4

    MFT_CROSS_JX1 = CROSS(MAX_FACTOR + MAX_FACTOR_delta,
                          MFT_CROSS[FLD.MAXFACTOR_BASELINE] - 133)
    MFT_CROSS_JX2 = CROSS(MAX_FACTOR + MAX_FACTOR_delta,
                          MFT_CROSS[FLD.MAXFACTOR_BASELINE])
    MFT_CROSS_JX3 = CROSS(MAX_FACTOR + MAX_FACTOR_delta,
                          MFT_CROSS[FLD.MAXFACTOR_BASELINE] + 133)
    MTF_CROSS_JX_JUNC = (MFT_CROSS_JX1 | \
        MFT_CROSS_JX2 | \
        MFT_CROSS_JX3).values
    MFT_CROSS_SX1 = CROSS(MFT_CROSS[FLD.MAXFACTOR_BASELINE] + 133,
                          MAX_FACTOR + MAX_FACTOR_delta)
    MFT_CROSS_SX2 = CROSS(MFT_CROSS[FLD.MAXFACTOR_BASELINE],
                          MAX_FACTOR + MAX_FACTOR_delta)
    MFT_CROSS_SX3 = CROSS(MFT_CROSS[FLD.MAXFACTOR_BASELINE] - 133,
                          MAX_FACTOR + MAX_FACTOR_delta)
    MFT_CROSS_SX_JUNC = (MFT_CROSS_SX1 | \
        MFT_CROSS_SX2 | \
        MFT_CROSS_SX3).values
    MFT_CROSS[FLD.MAXFACTOR_CROSS] = np.where(MTF_CROSS_JX_JUNC == 1,
                                              1, 
                                              np.where(MFT_CROSS_SX_JUNC == 1,
                                                       -1, 0))
    MFT_CROSS[FLD.MAXFACTOR_CROSS_JX] = Timeline_duration(MTF_CROSS_JX_JUNC)
    MFT_CROSS[FLD.MAXFACTOR_CROSS_SX] = Timeline_duration(MFT_CROSS_SX_JUNC)

    return MFT_CROSS


def dual_cross_np(closep:np.ndarray,
                  highp:np.ndarray,
                  lowp:np.ndarray,) -> np.ndarray:
    """
    自创指标：CCI/KDJ 对 偏移后的 RSI 双金叉
    为了避免 Warning，计算时忽略了前13个 NaN 的，最后 加入 ret 的时候补回来
    """
    RSI = TA_RSI(closep, timeperiod=12)
    CCI = TA_CCI(highp, lowp, closep)
    KDJ = TA_KDJ(highp, lowp, closep)
    
    CCI_CROSS_JX = np.where(CCI[13:,0] > (RSI[13:,0] - 50) * 4, 1, 0)
    KDJ_J_JX = np.where(KDJ[13:,2] > RSI[13:,0], 1, 0)
    KDJ_J_JX_P = np.where(KDJ[13:,2] + KDJ[13:,3] > RSI[13:,0], 1, 0)
    dual_jx = CCI_CROSS_JX * (CCI_CROSS_JX + KDJ_J_JX + KDJ_J_JX_P)
    dual_cross_jx = np.r_[np.zeros(13), 
                          np.where(dual_jx > 1, 1, 0)]
    if (len(dual_cross_jx) != len(closep)):
        dual_cross_jx = dual_cross_jx[-len(closep):]

    CCI_CROSS_SX = np.where((RSI[13:,0] - 50) * 4 > CCI[13:,0], 1, 0)
    KDJ_J_CROSS_SX = np.where(RSI[13:,0] > KDJ[13:,2], 1, 0)
    KDJ_J_CROSS_SX_PLUS = np.where(RSI[13:,0] > (KDJ[13:,2] + KDJ[13:,3]), 1, 0)
    dual_sx = CCI_CROSS_SX * (CCI_CROSS_SX + KDJ_J_CROSS_SX + KDJ_J_CROSS_SX_PLUS)
    dual_cross_sx = np.r_[np.zeros(13), 
                          np.where(dual_sx > 1, 1, 0)]
    if (len(dual_cross_sx) != len(closep)):
        dual_cross_sx = dual_cross_sx[-len(closep):]

    dual_cross = np.where(dual_cross_jx == 1, 1, 
                          np.where(dual_cross_sx == 1,
                                   -1, 0))
    dual_cross_jx = Timeline_Integral(dual_cross_jx)
    dual_cross_sx = Timeline_Integral(dual_cross_sx)
    dual_trend_timing_lag = dual_cross_jx + np.sign(dual_cross) * dual_cross_sx
    return np.c_[dual_cross,
                 dual_cross_jx,
                 dual_cross_sx,
                 dual_trend_timing_lag,]


def dual_cross_func(data):
    """
    A pd.DataFrame wrapper for function dual_cross_np()
    自创指标：CCI/KDJ 对 偏移后的 RSI 双金叉
    此函数只做 np.ndarray 到 pd.DataFrame 的封装，实际计算由
    纯 numpy 完成，便于后期改为 Cython 或者 Numba@jit 优化运行速度。
    """
    if (ST.VERBOSE in data.columns):
        print('Phase dual_cross_func', QA_util_timestamp_to_str())

    ret_dual_cross = dual_cross_np(closep=data.close.values,
                                   highp=data.high.values,
                                   lowp=data.low.values,)

    DUAL_CROSS = pd.DataFrame(ret_dual_cross,
                              columns=[FLD.DUAL_CROSS,
                                       FLD.DUAL_CROSS_JX,
                                       FLD.DUAL_CROSS_SX,
                                       FLD.DUAL_TREND_TIMING_LAG,],
                              index=data.index)

    return DUAL_CROSS


def dual_cross_func_pd(data):
    """
    自创指标：CCI/KDJ 对 偏移后的 RSI 双金叉
    为了避免 Warning，计算时忽略了前13个 NaN 的，最后 加入DataFrame 的时候补回来
    """
    if (ST.VERBOSE in data.columns):
        print('Phase dual_cross_func', QA_util_timestamp_to_str())
    RSI = TA_RSI(data.close, timeperiod=12)
    CCI = TA_CCI(data.high, data.low, data.close)
    KDJ = TA_KDJ(data.high, data.low, data.close)
    
    CCI_CROSS_JX = CROSS_STATUS(CCI[13:,0], (RSI[13:,0] - 50) * 4)
    KDJ_J_JX = CROSS_STATUS(KDJ[13:,2], RSI[13:,0])
    KDJ_J_JX_P = CROSS_STATUS(KDJ[13:,2] + KDJ[13:,3], RSI[13:,0])
    dual_jx = CCI_CROSS_JX * (CCI_CROSS_JX + KDJ_J_JX + KDJ_J_JX_P)
    dual_cross_jx = np.r_[np.zeros(13), 
                          CROSS_STATUS(dual_jx, 1)]
    
    CCI_CROSS_SX = CROSS_STATUS((RSI[13:,0] - 50) * 4, CCI[13:,0])
    KDJ_J_CROSS_SX = CROSS_STATUS(RSI[13:,0], KDJ[13:,2])
    KDJ_J_CROSS_SX_PLUS = CROSS_STATUS(RSI[13:,0], KDJ[13:,2] + KDJ[13:,3])
    dual_sx = CCI_CROSS_SX * (CCI_CROSS_SX + KDJ_J_CROSS_SX + KDJ_J_CROSS_SX_PLUS)
    dual_cross_sx = np.r_[np.zeros(13), 
                          CROSS_STATUS(dual_sx, 1)]

    DUAL_CROSS = pd.DataFrame(columns=[FLD.DUAL_CROSS, 
                                       FLD.DUAL_CROSS_JX, 
                                       FLD.DUAL_CROSS_SX], 
                              index=data.index)

    DUAL_CROSS[FLD.DUAL_CROSS] = np.where(dual_cross_jx == 1, 1, 
                                        np.where(dual_cross_sx == 1,
                                                 -1, 0))
    DUAL_CROSS[FLD.DUAL_CROSS_JX] = Timeline_Integral(dual_cross_jx)
    DUAL_CROSS[FLD.DUAL_CROSS_SX] = Timeline_Integral(dual_cross_sx)
    return DUAL_CROSS


def ma30_cross_np(closep:np.ndarray,) -> np.ndarray:
    """
    MA均线金叉指标
    """
    ma5_c0 = talib.MA(closep, 5)
    ma10_c1 = talib.MA(closep, 10)
    ma20_c2 = talib.MA(closep, 20)
    ma30_c3 = talib.MA(closep, 30)
    ma60_c3 = talib.MA(closep, 60)
    ma90_c4 = talib.MA(closep, 90)
    ma120_c5 = talib.MA(closep, 120)

    # 为了避免 warning，进行布尔运算之前必须进行零填充
    ma5 = np.r_[np.zeros(4), ma5_c0[4:]]
    ma10 = np.r_[np.zeros(9), ma10_c1[9:]]
    ma20 = np.r_[np.zeros(19), ma20_c2[19:]]
    ma30 = np.r_[np.zeros(29), ma30_c3[29:]]
    ma60 = np.r_[np.zeros(59), ma60_c3[59:]]
    ma90 = np.r_[np.zeros(89), ma90_c4[89:]]
    ma120 = np.r_[np.zeros(119), ma120_c5[119:]]

    # 如果交易日数据达不到MA周期长度，填零可能会对不齐，需要判断并且截断
    if (len(closep) != len(ma5)):
        ma5 = ma5[-len(closep):]
    if (len(closep) != len(ma10)):
        ma10 = ma10[-len(closep):]
    if (len(closep) != len(ma30)):
        ma30 = ma30[-len(closep):]
    if (len(closep) != len(ma60)):
        ma60 = ma60[-len(closep):]
    if (len(closep) != len(ma90)):
        ma90 = ma90[-len(closep):]
    if (len(closep) != len(ma120)):
        ma120 = ma120[-len(closep):]

    ma5_cross_c16 = np.where(ma5 > ma10, 1, 
                             np.where(ma5 < ma10, -1, 0))
    ma30_cross_c6 = np.where(ma5 > ma30, 1, 
                             np.where(ma5 < ma30, -1, 0))

    ma5_cross_dif = np.nan_to_num(np.r_[0, np.diff(ma5_cross_c16)], nan=0)
    ma5_cross_c16 = np.where((ma5 > ma10) & \
                             (ma5_cross_dif > 0), 1, 
                             np.where((ma5 < ma10) & \
                                      (ma5_cross_dif < 0), -1, 0))
    ma5_cross_jx_c17 = Timeline_duration(np.where((ma5 > ma10) & \
                                                  (ma5_cross_dif > 0), 1, 0))
    ma5_cross_sx_c18 = Timeline_duration(np.where((ma5 < ma10) & \
                                                  (ma5_cross_dif < 0), 1, 0))

    ma30_cross_dif = np.nan_to_num(np.r_[0, np.diff(ma30_cross_c6)], nan=0)
    ma30_cross_c6 = np.where((ma5 > ma30) & \
                             (ma30_cross_dif > 0), 1, 
                             np.where((ma5 < ma30) & \
                                      (ma30_cross_dif < 0), -1, 0))
    ma30_cross_jx_c7 = Timeline_duration(np.where((ma5 > ma30) & \
                                                  (ma30_cross_dif > 0), 1, 0))
    ma30_cross_sx_c8 = Timeline_duration(np.where((ma5 < ma30) & \
                                                  (ma30_cross_dif < 0), 1, 0))
    try:
        ma30_slope_c9 = talib.LINEARREG_SLOPE(ma30_c3, timeperiod=14)
        ma60_slope_c20 = talib.LINEARREG_SLOPE(ma60_c3, timeperiod=14)
    except:
        # 新股的交易数据太少了
        ma30_slope_c9 = np.full((len(closep),), np.median(closep))
        ma60_slope_c20 = np.full((len(closep),), np.median(closep))

    with np.errstate(invalid='ignore', divide='ignore'):
        ma30_slope_change_c10 = (ma30_slope_c9 - np.r_[np.nan, 
                                                       np.where(ma30_slope_c9[:-1] == 0,
                                                                np.nan, ma30_slope_c9[:-1])]) / ma30_slope_c9

    try:
        ma90_slope_c11 = talib.LINEARREG_SLOPE(ma90_c4, timeperiod=14)
        ma120_slope_c12 = talib.LINEARREG_SLOPE(ma120_c5, timeperiod=14)
    except:
        ma90_slope_c11 = np.full((len(closep),), np.median(closep))
        ma120_slope_c12 = np.full((len(closep),), np.median(closep))

    ma90_cross_c13 = np.where(ma90 > ma120, 1, 
                            np.where(ma90 < ma120, -1, 0))
    ma90_cross_dif = np.nan_to_num(np.r_[0, np.diff(ma90_cross_c13)], nan=0)
    ma90_cross_c13 = np.where((ma90 > ma120) & \
                              (ma90_cross_dif > 0), 1, 
                            np.where((ma90 < ma120) & \
                                     (ma90_cross_dif < 0), -1, 0))
    ma90_cross_jx_c14 = Timeline_duration(np.where((ma90 > ma120) & \
                                                   (ma90_cross_dif > 0), 1, 0))
    ma90_cross_sx_c15 = Timeline_duration(np.where((ma90 < ma120) & \
                                                   (ma90_cross_dif < 0), 1, 0))

    # 计算金叉死叉时间 LAG 间隙
    ma30_cross = np.where((ma5 > ma10) & \
                          (ma10 > ma20) & \
                          (ma20 > ma30), 1,
                          np.where((ma5 < ma10) & \
                                   (ma10 < ma20) & \
                                   (ma20 < ma30), -1, 0))
    MA30_TREND_TIMING_LAG_c19 = calc_event_timing_lag(ma30_cross)

    ma90_trend_timing_lag_c21 = calc_event_timing_lag(np.where(ma90_cross_jx_c14 < ma90_cross_sx_c15, 1, -1))

    return np.c_[ma5_c0,
                 ma10_c1,
                 ma20_c2,
                 ma30_c3,
                 ma60_c3,
                 ma90_c4,
                 ma120_c5,
                 ma30_cross_c6,
                 ma30_cross_jx_c7,
                 ma30_cross_sx_c8,
                 ma30_slope_c9,
                 ma30_slope_change_c10,
                 ma90_slope_c11,
                 ma120_slope_c12,
                 ma90_cross_c13,
                 ma90_cross_jx_c14,
                 ma90_cross_sx_c15,
                 ma5_cross_c16,
                 ma5_cross_jx_c17,
                 ma5_cross_sx_c18,
                 MA30_TREND_TIMING_LAG_c19,
                 ma60_slope_c20,
                 ma90_trend_timing_lag_c21,]


def ma30_cross_func(data,):
    """
    A pd.DataFrame wrapper for function ma30_cross_np()
    MA均线金叉指标
    此函数只做 np.ndarray 到 pd.DataFrame 的封装，实际计算由
    纯 numpy 完成，便于后期改为 Cython 或者 Numba@jit 优化运行速度。
    """
    if (ST.VERBOSE in data.columns):
        print('Phase ma30_cross_func', QA_util_timestamp_to_str())

    ret_ma30_cross = ma30_cross_np(closep=data.close.values,)

    MA30_CROSS = pd.DataFrame(ret_ma30_cross,
                              columns=[FLD.MA5,
                                       FLD.MA10,
                                       FLD.MA20,
                                       FLD.MA30,
                                       FLD.MA60,
                                       FLD.MA90,
                                       FLD.MA120,
                                       FLD.MA30_CROSS,
                                       FLD.MA30_CROSS_JX_BEFORE, 
                                       FLD.MA30_CROSS_SX_BEFORE,
                                       FLD.MA30_SLOPE,
                                       FLD.MA30_SLOPE_CHANGE,
                                       FLD.MA90_SLOPE,
                                       FLD.MA120_SLOPE,
                                       FLD.MA90_CROSS,
                                       FLD.MA90_CROSS_JX_BEFORE,
                                       FLD.MA90_CROSS_SX_BEFORE,
                                       FLD.MA5_CROSS,
                                       FLD.MA5_CROSS_JX_BEFORE,
                                       FLD.MA5_CROSS_SX_BEFORE,
                                       FLD.MA30_TREND_TIMING_LAG,
                                       FLD.MA60_SLOPE,
                                       FLD.MA90_TREND_TIMING_LAG,],
                              index=data.index)

    MA30_CROSS[FLD.HMA10] = TA_HMA((data.open.values + data.high.values + data.low.values + data.close.values) / 4, 10)
    MA30_CROSS[FLD.HMA10_DELTA] = MA30_CROSS[FLD.HMA10].diff(1)
    #MA30_CROSS[FLD.HMA10_LOW] = TA_HMA(data.low.values, 10)
    hma10_cross = np.where(MA30_CROSS[FLD.HMA10] > MA30_CROSS[FLD.HMA10].shift(1), 1,
                           np.where(MA30_CROSS[FLD.HMA10] < MA30_CROSS[FLD.HMA10].shift(1), -1, 0))
    MA30_CROSS[FLD.HMA10_TREND_TIMING_LAG] = calc_event_timing_lag(hma10_cross)
    MA30_CROSS[FLD.HMA5] = TA_HMA((data.open.values + data.high.values + data.low.values + data.close.values) / 4, 5)
    #MA30_CROSS[FLD.HMA5_LOW] = TA_HMA(data.low.values, 5)
    hma5_cross = np.where(MA30_CROSS[FLD.HMA5] > MA30_CROSS[FLD.HMA5].shift(1), 1,
                          np.where(MA30_CROSS[FLD.HMA5] < MA30_CROSS[FLD.HMA5].shift(1), -1, 0))
    MA30_CROSS[FLD.HMA5_TREND_TIMING_LAG] = calc_event_timing_lag(hma5_cross)
    ma30_hma5_cross = np.where((MA30_CROSS[FLD.MA30] > MA30_CROSS[FLD.HMA5]), 1,
                               np.where((MA30_CROSS[FLD.MA30] < MA30_CROSS[FLD.HMA5]), -1, 0))
    MA30_CROSS[FLD.MA30_HMA5_TIMING_LAG] = calc_event_timing_lag(ma30_hma5_cross)
    return MA30_CROSS


def ma30_cross_func_pd(data):
    """
    MA均线金叉指标
    """
    if (ST.VERBOSE in data.columns):
        print('Phase ma30_cross_func', QA_util_timestamp_to_str())
    MA5 = talib.MA(data.close, 5)
    MA10 = talib.MA(data.close, 10)
    MA20 = talib.MA(data.close, 20)
    MA30 = talib.MA(data.close, 30)
    MA90 = talib.MA(data.close, 90)
    MA120 = talib.MA(data.close, 120)
        
    MA30_CROSS = pd.DataFrame(columns=[FLD.MA5,
                                       FLD.MA10,
                                       FLD.MA20,
                                       FLD.MA30,
                                       FLD.MA90,
                                       FLD.MA120,
                                       FLD.MA30_CROSS, 
                                       FLD.MA30_CROSS_JX_BEFORE, 
                                       FLD.MA30_CROSS_SX_BEFORE,], 
                              index=data.index)
    MA30_CROSS[FLD.MA5] = MA5
    MA30_CROSS[FLD.MA10] = MA10
    MA30_CROSS[FLD.MA20] = MA20
    MA30_CROSS[FLD.MA30] = MA30
    MA30_CROSS[FLD.MA90] = MA90
    MA30_CROSS[FLD.MA120] = MA120
    MA30_CROSS[FLD.MA30_CROSS] = np.where(CROSS(MA5, MA30).values == 1, 1, 
                                        np.where(CROSS(MA30, MA5).values == 1, 
                                                 -1, 0))
    
    MA30_CROSS[FLD.MA30_SLOPE] = talib.LINEARREG_SLOPE(MA30, timeperiod=14)
    MA30_CROSS[FLD.MA30_SLOPE_CHANGE] = MA30_CROSS[FLD.MA30_SLOPE].pct_change()
    ma30_slope_c9 = MA30_CROSS[FLD.MA30_SLOPE].values
    ma30_slope_change_c10 = (ma30_slope_c9 - np.r_[np.nan, 
                                                   np.where(ma30_slope_c9[:-1] == 0,
                                                            np.nan, ma30_slope_c9[:-1])]) / ma30_slope_c9
    MA30_CROSS['{}_np'.format(FLD.MA30_SLOPE_CHANGE)] = ma30_slope_change_c10
    #print(MA30_CROSS[FLD.MA30_SLOPE_CHANGE].corr(MA30_CROSS['{}_np'.format(FLD.MA30_SLOPE_CHANGE)]))
    #print(MA30_CROSS[[FLD.MA30_SLOPE_CHANGE,
    #'{}_np'.format(FLD.MA30_SLOPE_CHANGE)]])
    MA30_CROSS[FLD.MA90_SLOPE] = talib.LINEARREG_SLOPE(MA90, timeperiod=14)
    MA30_CROSS[FLD.MA120_SLOPE] = talib.LINEARREG_SLOPE(MA120, timeperiod=14)

    MA30_CROSS[FLD.MA30_CROSS_JX_BEFORE] = np.where(MA30_CROSS[FLD.MA30_CROSS] == 1, 
                                                    1, 0)
    MA30_CROSS[FLD.MA30_CROSS_SX_BEFORE] = np.where(MA30_CROSS[FLD.MA30_CROSS] == -1, 
                                                    1, 0)
    MA30_CROSS[FLD.MA30_CROSS_JX_BEFORE] = Timeline_duration(MA30_CROSS[FLD.MA30_CROSS_JX_BEFORE].values)
    MA30_CROSS[FLD.MA30_CROSS_SX_BEFORE] = Timeline_duration(MA30_CROSS[FLD.MA30_CROSS_SX_BEFORE].values)

    MA30_CROSS[FLD.MA90_CROSS] = np.where(CROSS(MA90, MA120).values == 1, 1, 
                                          np.where(CROSS(MA120, MA90).values == 1, 
                                                   -1, 0))
    MA30_CROSS[FLD.MA90_CROSS_JX_BEFORE] = np.where(MA30_CROSS[FLD.MA90_CROSS] == 1, 
                                                    1, 0)
    MA30_CROSS[FLD.MA90_CROSS_SX_BEFORE] = np.where(MA30_CROSS[FLD.MA90_CROSS] == -1, 
                                                    1, 0)
    MA30_CROSS[FLD.MA90_CROSS_JX_BEFORE] = Timeline_duration(MA30_CROSS[FLD.MA90_CROSS_JX_BEFORE].values)
    MA30_CROSS[FLD.MA90_CROSS_SX_BEFORE] = Timeline_duration(MA30_CROSS[FLD.MA90_CROSS_SX_BEFORE].values)
    MA30_CROSS[FLD.MA90_TREND_TIMING_LAG] = calc_event_timing_lag(np.where(MA30_CROSS[FLD.MA90_CROSS_JX_BEFORE] < MA30_CROSS[FLD.MA90_CROSS_SX_BEFORE], 1, -1))
    return MA30_CROSS


def boll_clearance_func(data, compaerfrom):
    """
    布林（BOLL）通道宽度，布林通道宽度决定了买入操作的方式
    """
    return (data[FLD.BOLL_UB] - compaerfrom) / data[FLD.BOLL]


@nb.jit(nopython=True)
def boll_cross_func_jit(data:np.ndarray,) -> np.ndarray:
    """
    布林线和K线金叉死叉 状态分析 Numba JIT优化
    idx: 0 == open
         1 == high
         2 == low
         3 == close
    """
    BBANDS = TA_BBANDS(data[:,3], timeperiod=20, nbdevup=2)

    return ret_boll_cross


def boll_cross_func(data:pd.DataFrame,) -> pd.DataFrame:
    """
    布林线和K线金叉死叉 状态分析
    """
    if (ST.VERBOSE in data.columns):
        print('Phase boll_cross_func', QA_util_timestamp_to_str())

    BBANDS = TA_BBANDS(data.close, timeperiod=20, nbdevup=2)
    BOLL_CROSS = pd.DataFrame(columns=['BOLL_CROSS', 
                                       FLD.BOLL_CROSS_JX_BEFORE, 
                                       FLD.BOLL_CROSS_SX_BEFORE,
                                       FLD.BOLL_UB,
                                       FLD.BOLL,
                                       FLD.BOLL_LB,
                                       FLD.BOLL_CHANNEL,
                                       FLD.BOLL_DELTA,
                                       FLD.BOLL_CHANNEL_MA30], 
                              index = data.index)

    # 防止插针行情突然搞乱故
    data_price = data.copy()
    data_price[FLD.BOLL] = BBANDS[:,1]
    data_price['smooth_low'] = talib.MA(data.low, 2)
    data_price['smooth_high'] = talib.MA(data.high, 2)
    BOLL_TP_CROSS = pd.DataFrame(columns=['min_peak', 
                                          'max_peak', 
                                          'BOLL_TP_CROSS_JX', 
                                          'BOLL_TP_CROSS_SX'], 
                                 index=data.index)
    BOLL_TP_CROSS['min_peak'] = np.minimum(data_price['open'], 
                                           np.minimum(data_price['close'], 
                                                      np.where(data_price['open'] < data_price['BOLL_MA'], 
                                                               data_price['low'], 
                                                               data_price['smooth_low'])))
    BOLL_TP_CROSS['max_peak'] = np.maximum(data_price['open'], 
                                           np.maximum(data_price['close'], 
                                                      np.where(data_price['open'] > data_price['BOLL_MA'], 
                                                               data_price['high'], 
                                                               data_price['smooth_high'])))

    BOLL_TP_CROSS['BOLL_TP_CROSS_JX'] = CROSS(BOLL_TP_CROSS['min_peak'], BBANDS[:,2])
    BOLL_TP_CROSS['BOLL_TP_CROSS_SX'] = CROSS(BBANDS[:,0], BOLL_TP_CROSS['max_peak'])

    BOLL_CROSS['BOLL_CROSS'] = np.where(BOLL_TP_CROSS['BOLL_TP_CROSS_JX'] == 1, 1,
                                        np.where(BOLL_TP_CROSS['BOLL_TP_CROSS_SX'] == 1, 
                                                 -1, 0))

    BOLL_CROSS[FLD.BOLL_UB] = BBANDS[:,0]
    BOLL_CROSS[FLD.BOLL] = BBANDS[:,1]
    BOLL_CROSS[FLD.BOLL_LB] = BBANDS[:,2]
    BOLL_CROSS[FLD.BOLL_CHANNEL] = BBANDS[:,3]
    BOLL_CROSS[FLD.BOLL_DELTA] = BBANDS[:,4]

    if (len(BOLL_CROSS[FLD.BOLL_CHANNEL_MA30]) > 30):
        BOLL_CROSS[FLD.BOLL_CHANNEL_MA30] = talib.MA(BBANDS[:,3], 21)
    if (len(BOLL_CROSS[FLD.BOLL_CHANNEL_MA30]) > 10):
        BOLL_CROSS[FLD.BOLL_CHANNEL_MA30] = talib.MA(BBANDS[:,3], 5)
    else:
        BOLL_CROSS[FLD.BOLL_CHANNEL_MA30] = BBANDS[:,3]

    BOLL_CROSS[FLD.BOLL_CROSS_JX_BEFORE] = Timeline_duration(BOLL_TP_CROSS['BOLL_TP_CROSS_JX'].values)
    BOLL_CROSS[FLD.BOLL_CROSS_SX_BEFORE] = Timeline_duration(BOLL_TP_CROSS['BOLL_TP_CROSS_SX'].values)
    BOLL_CROSS[FLD.BOLL_TREND_TIMING_LAG] = BOLL_CROSS[FLD.BOLL_CROSS_SX_BEFORE] - BOLL_CROSS[FLD.BOLL_CROSS_JX_BEFORE]

    return BOLL_CROSS


def ma_power_func(data, range_list=range(5, 30)):
    '''
    MA均线多头排列能量强度定义
    '''
    if (ST.VERBOSE in data.columns):
        print('Phase ma_power_func', QA_util_timestamp_to_str())

    def inv_num(series):
        '''
        计算逆序数个数
        '''
        series = np.array(series)  # 提升速度
        return np.sum([np.sum(x < series[:i]) for i, x in enumerate(series)])

    price = data.close
    ma_pd = pd.DataFrame(index=price.index)
    for r in range_list:
        ma = talib.MA(price, r)
        if len(ma_pd) == 0:
            ma_pd = ma
        else:
            ma_pd = pd.concat([ma_pd, ma], axis=1)
    ma_pd.columns = range_list
    df_fixed = ma_pd.dropna()  # 前n个数据部分均线为空值，去除
    num = df_fixed.apply(lambda x: inv_num(x), axis=1)  # 每排逆序个数
    ratio = num / (len(range_list) * (len(range_list) - 1)) * 2
    ratio = ratio.sort_index()
    mapower = pd.DataFrame(columns= ['MAPOWER', 
                                     'MAPOWER_RETURNS', 
                                     'MAPOWER_TREND', 
                                     'MAPOWER_TREND_CROSS_JX', 
                                     'MAPOWER_TREND_CROSS_SX'], 
                           index=price.index)
    ratio = ratio.loc[price.index.get_level_values(level=0).intersection(ratio.index.get_level_values(level=0)), :]
    mapower.loc[ratio.index, ['MAPOWER']] = ratio
    mapower = mapower.assign(MAPOWER_DELTA=mapower['MAPOWER'].diff())
    return mapower


def ma_power_np_func(data, range_list=range(5, 30)):
    '''
    MA均线多头排列能量强度定义，纯np优化版
    '''
    if (ST.VERBOSE in data.columns):
        print('Phase ma_power_np_func', QA_util_timestamp_to_str())

    def inv_num(series):
        '''
        计算逆序数个数
        '''
        #series = np.array(series) # 提升速度
        return np.sum([np.sum(x < series[:i]) for i, x in enumerate(series)])

    price = data.close
    ma_np = np.empty((len(price), len(range_list)))
    ma_count = 0

    for r in range_list:
        ma = talib.MA(price, r)
        ma_np[:, ma_count] = ma 
        ma_count = ma_count + 1

    ma_max = max(range_list)
    len_range_list = len(range_list)
    num = np.zeros(len(price))
    ratio = np.zeros(len(price))
    with np.errstate(invalid='ignore', divide='ignore'):
        for i in range(ma_max,len(price)):
            num[i] = inv_num(ma_np[i, :])
            ratio[i] = num[i] / (len_range_list * (len_range_list - 1)) * 2

    return ratio


def hma_power_np_func(data, range_list=range(5, 30)):
    '''
    MA均线多头排列能量强度定义，纯np优化版
    '''
    if (ST.VERBOSE in data.columns):
        print('Phase hma_power_np_func', QA_util_timestamp_to_str())

    def inv_num(series):
        '''
        计算逆序数个数
        '''
        #series = np.array(series) # 提升速度
        return np.sum([np.sum(x < series[:i]) for i, x in enumerate(series)])

    price = data.close
    ma_np = np.empty((len(price), len(range_list)))
    ma_count = 0

    for r in range_list:
        ma = TA_HMA(price, r)
        ma_np[:, ma_count] = ma 
        ma_count = ma_count + 1

    ma_max = max(range_list)
    len_range_list = len(range_list)
    num = np.zeros(len(price))
    ratio = np.zeros(len(price))
    with np.errstate(invalid='ignore', divide='ignore'):
        for i in range(ma_max,len(price)):
            num[i] = inv_num(ma_np[i, :])
            ratio[i] = num[i] / (len_range_list * (len_range_list - 1)) * 2

    return ratio


#def ma_power_func(data, range_list=range(5, 30)):
#@nb.jit('f8(f8[:], i2[:])', nopython=True)
def ma_power_jit_func(closep:np.ndarray, range_list:list=[5, 10, 20, 30, 60, 90, 120]) -> np.ndarray:
    '''
    多头排列能量强度定义
    '''
    @nb.jit('f8(f8[:], f8[:, :], i2[:])', nopython=True)
    def calc_ratio(closep, ma_np, range_list) -> np.float:
        num = np.empty(len(closep), np.float64)
        ratio = np.empty(len(closep), np.float64)
        ma_max = max(range_list)
        len_range_list = len(range_list)
        #ret_mapower = np.zeros((len(price), 2))
        for i in range(ma_max, len(closep)):
            slice_day = ma_np[i, :]
            bigger = np.empty(len_range_list, np.float32)
            for j in range(len_range_list):
                x = slice_day[j]
                bigger[j] = np.sum(slice_day[slice_day < x])
            num[i] = np.sum(bigger)
            ratio[i] = num[i] / (len_range_list * (len_range_list - 1)) * 2
        return ratio

    #def inv_num(series:np.ndarray) -> np.float:
    #    '''
    #    计算逆序数个数
    #    '''
    #    #series = np.array(series) # 提升速度
    #    return np.sum([np.sum(x < series[:i]) for i, x in enumerate(series)])
    if (ST.VERBOSE in data.columns):
        print('Phase ma_power_func', QA_util_timestamp_to_str())

    ma_np = np.empty((len(closep), len(range_list)))
    ma_count = 0
    for r in range_list:
        ma = talib.MA(closep, r)
        ma_np[:, ma_count] = ma 
        ma_count = ma_count + 1

    ma_max = max(range_list)
    len_range_list = len(range_list)
    ratio = calc_ratio(closep, ma_np, range_list)
    return ratio


def ATR_Stopline_func(data, *args, **kwargs):
    """
    ATR Stopline 策略
    支持QA add_func，第二个参数 默认为 indices= 为已经计算指标
    理论上这个函数只计算单一标的，不要尝试传递复杂标的，indices会尝试拆分。
    """
    if (ST.VERBOSE in data.columns):
        print('Phase ATR_Stopline_func', QA_util_timestamp_to_str())

    # 针对多标的，拆分 indices 数据再自动合并
    code = data.index.get_level_values(level=1)[0]
    if ('indices' in kwargs.keys()):
        features = kwargs['indices'].loc[(slice(None), code), :]
    elif (len(args) > 0):
        features = args[0].loc[(slice(None), code), :]
    else:
        features = None

    rsi_ma, stop_line, ATR_StoplineTrend = ATR_RSI_Stops_v2(data, 27)
    ATR_Stopline_CROSS = pd.DataFrame(np.c_[ATR_StoplineTrend, 
                                            np.zeros(len(data)), 
                                            np.zeros(len(data)), 
                                            np.zeros(len(data)),], 
                                      columns=[FLD.ATR_Stopline,
                                               FLD.ATR_Stopline_TIMING_LAG,
                                               FLD.ATR_Stopline_CROSS_JX,
                                               FLD.ATR_Stopline_CROSS_SX], 
                                      index=data.index)

    ATR_Stopline_CROSS[FLD.ATR_Stopline_TIMING_LAG] = calc_event_timing_lag(ATR_Stopline_CROSS[FLD.ATR_Stopline])
    if (features is None):
        closep = data[AKA.CLOSE].values
        ATR_Stopline_CROSS[FLD.ATR_Stopline_RETURNS] = calc_onhold_returns_v2(np.log(closep / np.r_[closep[0], closep[:-1]]), 
                                                                              ATR_Stopline_CROSS[FLD.ATR_Stopline].values,)
    else:
        ATR_Stopline_CROSS[FLD.ATR_Stopline_RETURNS] = calc_onhold_returns_v2(features[FLD.PCT_CHANGE].values, 
                                                                              ATR_Stopline_CROSS[FLD.ATR_Stopline].values,)

    ATR_Stopline_CROSS[FLD.ATR_Stopline_CROSS_JX] = np.where(np.nan_to_num(ATR_Stopline_CROSS['ATR_Stopline'].values, nan=0) > 0, 1, 0)
    ATR_Stopline_CROSS[FLD.ATR_Stopline_CROSS_SX] = np.where(np.nan_to_num(ATR_Stopline_CROSS['ATR_Stopline'].values, nan=0) < 0, 1, 0)
    ATR_Stopline_CROSS[FLD.ATR_Stopline_CROSS_JX] = (ATR_Stopline_CROSS[FLD.ATR_Stopline_CROSS_JX].apply(int).diff() > 0).apply(int)
    ATR_Stopline_CROSS[FLD.ATR_Stopline_CROSS_SX] = (ATR_Stopline_CROSS[FLD.ATR_Stopline_CROSS_SX].apply(int).diff() > 0).apply(int)
    ATR_Stopline_CROSS[FLD.ATR_Stopline_CROSS_JX] = Timeline_duration(ATR_Stopline_CROSS[FLD.ATR_Stopline_CROSS_JX].values)
    ATR_Stopline_CROSS[FLD.ATR_Stopline_CROSS_SX] = Timeline_duration(ATR_Stopline_CROSS[FLD.ATR_Stopline_CROSS_SX].values)
    ATR_Stopline_CROSS[FLD.ATR_Stopline_MEDIAN] = int(min(ATR_Stopline_CROSS[FLD.ATR_Stopline_CROSS_JX].median(), 
                                                          ATR_Stopline_CROSS[FLD.ATR_Stopline_CROSS_SX].median()))

    if (features is None):
        return ATR_Stopline_CROSS
    else:
        return pd.concat([features,
                          ATR_Stopline_CROSS], axis=1)


def ATR_SuperTrend_func(data, *args, **kwargs):
    """
    ATR 超级趋势策略
    支持QA add_func，第二个参数 默认为 indices= 为已经计算指标
    理论上这个函数只计算单一标的，不要尝试传递复杂标的，indices会尝试拆分。
    """
    if (ST.VERBOSE in data.columns):
        print('Phase ATR_SuperTrend_func', QA_util_timestamp_to_str())

    # 针对多标的，拆分 indices 数据再自动合并
    code = data.index.get_level_values(level=1)[0]
    if ('indices' in kwargs.keys()):
        indices = kwargs['indices'].loc[(slice(None), code), :]
    elif (len(args) > 0):
        indices = args[0].loc[(slice(None), code), :]
    else:
        indices = None

    Tsl, ATR_SuperTrend = ATR_SuperTrend_cross_v2(data)
    ATR_SuperTrend_CROSS = pd.DataFrame(np.c_[ATR_SuperTrend, 
                                              np.zeros(len(data)), 
                                              np.zeros(len(data)), 
                                              np.zeros(len(data)),], 
                                        columns=[FLD.ATR_SuperTrend,
                                                 FLD.ATR_SuperTrend_TIMING_LAG,
                                                 FLD.ATR_SuperTrend_CROSS_JX,
                                                 FLD.ATR_SuperTrend_CROSS_SX], 
                                        index=data.index)

    ATR_SuperTrend_CROSS[FLD.ATR_SuperTrend_TIMING_LAG] = calc_event_timing_lag(ATR_SuperTrend_CROSS[FLD.ATR_SuperTrend])

    ATR_SuperTrend_CROSS[FLD.ATR_SuperTrend_CROSS_JX] = np.where(np.nan_to_num(ATR_SuperTrend_CROSS[FLD.ATR_SuperTrend].values, nan=0) > 0, 1, 0)
    ATR_SuperTrend_CROSS[FLD.ATR_SuperTrend_CROSS_SX] = np.where(np.nan_to_num(ATR_SuperTrend_CROSS[FLD.ATR_SuperTrend].values, nan=0) < 0, 1, 0)
    ATR_SuperTrend_CROSS[FLD.ATR_SuperTrend_CROSS_JX] = (ATR_SuperTrend_CROSS[FLD.ATR_SuperTrend_CROSS_JX].apply(int).diff() > 0).apply(int)
    ATR_SuperTrend_CROSS[FLD.ATR_SuperTrend_CROSS_SX] = (ATR_SuperTrend_CROSS[FLD.ATR_SuperTrend_CROSS_SX].apply(int).diff() > 0).apply(int)
    ATR_SuperTrend_CROSS[FLD.ATR_SuperTrend_CROSS_JX] = Timeline_duration(ATR_SuperTrend_CROSS[FLD.ATR_SuperTrend_CROSS_JX].values)
    ATR_SuperTrend_CROSS[FLD.ATR_SuperTrend_CROSS_SX] = Timeline_duration(ATR_SuperTrend_CROSS[FLD.ATR_SuperTrend_CROSS_SX].values)
    ATR_SuperTrend_CROSS[FLD.ATR_SuperTrend_MEDIAN] = int(min(ATR_SuperTrend_CROSS[FLD.ATR_SuperTrend_CROSS_JX].median(), 
                                                              ATR_SuperTrend_CROSS[FLD.ATR_SuperTrend_CROSS_SX].median()))

    if (indices is None):
        return ATR_SuperTrend_CROSS
    else:
        return pd.concat([indices,
                          ATR_SuperTrend_CROSS], 
                         axis=1)


def ADXm_Trend_func(data, *args, **kwargs):
    """
    ADXm_Trend 策略
    支持QA add_func，第二个参数 默认为 indices= 为已经计算指标
    理论上这个函数只计算单一标的，不要尝试传递复杂标的，indices会尝试拆分。
    """
    if (ST.VERBOSE in data.columns):
        print('Phase ADXm_Trend_func', QA_util_timestamp_to_str())

    # 针对多标的，拆分 indices 数据再自动合并
    code = data.index.get_level_values(level=1)[0]
    if ('features' in kwargs.keys()):
        features = kwargs['features'].loc[(slice(None), code), :]
    elif (len(args) > 0):
        features = args[0].loc[(slice(None), code), :]
    else:
        features = None

    adx, ADXm = ADX_MA(data)
    ADXm_Trend_CROSS = pd.DataFrame(np.c_[ADXm, 
                                          np.zeros(len(data)),], 
                                      columns=[FLD.ADXm_Trend, 
                                               FLD.ADXm_Trend_TIMING_LAG,], 
                                      index=data.index)
    ADXm_Trend_CROSS[FLD.ADXm_Trend_TIMING_LAG] = calc_event_timing_lag(ADXm_Trend_CROSS[FLD.ADXm_Trend])
    if (features is None):
        closep = data[AKA.CLOSE].values
        ADXm_Trend_CROSS[FLD.ADXm_Trend_RETURNS] = calc_onhold_returns_v2(np.log(closep / np.r_[closep[0], closep[:-1]]), 
                                                                          ADXm_Trend_CROSS[FLD.ADXm_Trend].values,)
    else:
        ADXm_Trend_CROSS[FLD.ADXm_Trend_RETURNS] = calc_onhold_returns_v2(features[FLD.PCT_CHANGE].values, 
                                                                          ADXm_Trend_CROSS[FLD.ADXm_Trend].values,)

    if (features is None):
        return ADXm_Trend_CROSS
    else:
        return pd.concat([features,
                          ADXm_Trend_CROSS], axis=1)


def VHMA_Trend_func(data, *args, **kwargs):
    """
    Volume_HMA 策略
    支持QA add_func，第二个参数 默认为 indices= 为已经计算指标
    理论上这个函数只计算单一标的，不要尝试传递复杂标的，indices会尝试拆分。
    """
    if (ST.VERBOSE in data.columns):
        print('Phase Volume_HMA_Trend_func', QA_util_timestamp_to_str())

    # 针对多标的，拆分 indices 数据再自动合并
    code = data.index.get_level_values(level=1)[0]
    if ('features' in kwargs.keys()):
        features = kwargs['features'].loc[(slice(None), code), :]
    elif (len(args) > 0):
        features = args[0].loc[(slice(None), code), :]
    else:
        features = None

    def reifine_vhma_directions(vhma_directions):
        vhma_directions = np.where((vhma_directions > 0), vhma_directions,
                               np.where((vhma_directions == 0) & \
                                        (np.r_[0, vhma_directions[:-1]] > 0) & \
                                        ((data[AKA.CLOSE] > data[AKA.CLOSE].shift(1)) | \
                                        (data[AKA.CLOSE] > data[AKA.CLOSE].shift(2))), 1, 
                                        np.where((vhma_directions < 0), vhma_directions,
                                                 np.where((vhma_directions == 0) & \
                                                          (np.r_[0, vhma_directions[:-1]] < 0) & \
                                                          ((data[AKA.CLOSE] < data[AKA.CLOSE].shift(1)) | \
                                                          (data[AKA.CLOSE] < data[AKA.CLOSE].shift(2))), -1, 0))))
        return vhma_directions

    vhma, vhma_directions5 = Volume_HMA(data, 5)
    vhma, vhma_directions10 = Volume_HMA(data, 10)
    vhma_directions5 = reifine_vhma_directions(vhma_directions5)
    vhma_directions10 = reifine_vhma_directions(vhma_directions10)

    VHMA_Trend_CROSS = pd.DataFrame(np.c_[vhma_directions5, 
                                          vhma_directions10, 
                                          np.zeros(len(data)), 
                                          np.zeros(len(data)),], 
                                      columns = [FLD.Volume_HMA5,
                                                 FLD.Volume_HMA10, 
                                               FLD.Volume_HMA5_TIMING_LAG, 
                                               FLD.Volume_HMA10_TIMING_LAG,], 
                                      index = data.index)

    VHMA_Trend_CROSS[FLD.Volume_HMA5_TIMING_LAG] = calc_event_timing_lag(vhma_directions5)
    VHMA_Trend_CROSS[FLD.Volume_HMA10_TIMING_LAG] = calc_event_timing_lag(vhma_directions10)

    if (features is None):
        return VHMA_Trend_CROSS
    else:
        return pd.concat([features,
                          VHMA_Trend_CROSS], axis = 1)


def ma_clearance_prepar_indices_func(indices=None):
    """
    准备计算指标 MA_CLEARANCE
    MA净空 范围（-0.2~+1.2），MA30净空越大（> 0.5），上涨趋势越明显，（> 0.618）为上升浪形态(也可以为见顶迹象)
    """
    if (FLD.MA30_CLEARANCE not in indices.columns):
        indices[FLD.MA5_CLEARANCE] = boll_clearance_func(indices, indices[FLD.MA5]) / indices[FLD.BOLL_CHANNEL]
        indices[FLD.MA10_CLEARANCE] = boll_clearance_func(indices, indices[FLD.MA10]) / indices[FLD.BOLL_CHANNEL]
        indices[FLD.MA20_CLEARANCE] = boll_clearance_func(indices, indices[FLD.MA20]) / indices[FLD.BOLL_CHANNEL]
        indices[FLD.MA30_CLEARANCE] = boll_clearance_func(indices, indices[FLD.MA30]) / indices[FLD.BOLL_CHANNEL]
        indices[FLD.MA90_CLEARANCE] = boll_clearance_func(indices, indices[FLD.MA90]) / indices[FLD.BOLL_CHANNEL]
        indices[FLD.MA120_CLEARANCE] = boll_clearance_func(indices, indices[FLD.MA120]) / indices[FLD.BOLL_CHANNEL]
        indices[FLD.HMA5_CLEARANCE] = boll_clearance_func(indices, indices[FLD.HMA5]) / indices[FLD.BOLL_CHANNEL]
        indices[FLD.MA_VOL] = ((indices[FLD.MA5_CLEARANCE] - indices[FLD.MA10_CLEARANCE]) + \
            (indices[FLD.MA5_CLEARANCE] - indices[FLD.MA20_CLEARANCE]) + \
            (indices[FLD.MA5_CLEARANCE] - indices[FLD.MA30_CLEARANCE]) + \
            (indices[FLD.MA10_CLEARANCE] - indices[FLD.MA20_CLEARANCE]) + \
            (indices[FLD.MA10_CLEARANCE] - indices[FLD.MA30_CLEARANCE]) + \
            (indices[FLD.MA20_CLEARANCE] - indices[FLD.MA30_CLEARANCE]))
        indices[FLD.MA_VOL] = indices[FLD.MA_VOL] / 6

        features = indices
        upper = np.maximum(features[FLD.MA5],
                           np.maximum(features[FLD.MA10],
                                      np.maximum(features[FLD.MA20],
                                                 features[FLD.MA30])))
        lower = np.minimum(features[FLD.MA5],
                           np.minimum(features[FLD.MA10],
                                      np.minimum(features[FLD.MA20],
                                                 features[FLD.MA30])))
        upper120 = np.maximum(features[FLD.MA5],
                           np.maximum(features[FLD.MA10],
                                      np.maximum(features[FLD.MA20],
                                                 np.maximum(features[FLD.MA30],
                                                            np.maximum(features[FLD.MA90],
                                                                       features[FLD.MA120])))))
        lower120 = np.minimum(features[FLD.MA5],
                           np.minimum(features[FLD.MA10],
                                      np.minimum(features[FLD.MA20],
                                                 np.minimum(features[FLD.MA30],
                                                            np.minimum(features[FLD.MA90],
                                                                       features[FLD.MA120])))))
        middle = features[FLD.MA20]
        ch = (upper - lower) / middle
        features[FLD.MA_CHANNEL] = ch

        ch120 = (upper120 - lower120) / middle
        features[FLD.MA120_CHANNEL] = ch120

        ma90_clearance_cross = np.where(indices[FLD.MA90_CLEARANCE] > 0, 1, -1)
        indices[FLD.MA90_CLEARANCE_TIMING_LAG] = calc_event_timing_lag(ma90_clearance_cross)
        ma120_clearance_cross = np.where(indices[FLD.MA120_CLEARANCE] > 0, 1, -1)
        indices[FLD.MA120_CLEARANCE_TIMING_LAG] = calc_event_timing_lag(ma120_clearance_cross)
    return indices


def lineareg_cross_func(data, *args, **kwargs):
    """
    ATR线性回归带金叉死叉策略，第二版
    支持QA add_func，第二个参数 默认为 indices= 为已经计算指标
    理论上这个函数只计算单一标的，不要尝试传递复杂标的，indices会尝试拆分。
    """
    # 针对多标的，拆分 indices 数据再自动合并
    code = data.index.get_level_values(level=1)[0]
    if ('indices' in kwargs.keys()):
        indices = kwargs['indices'].loc[(slice(None), code), :]
    elif (len(args) > 0):
        indices = args[0].loc[(slice(None), code), :]
    else:
        indices = None
    annual = kwargs['annual'] if ('annual' in kwargs.keys()) else 252

    if (indices is None):
        indices = pd.concat([ma30_cross_func(data),
                             boll_cross_func(data),
                             macd_cross_v2_func(data, annual=annual),
                             dual_cross_v2_func(data),
                             maxfactor_cross_v2_func(data),
                             QA_indicator_BIAS(data, 10, 20, 30),],
                            axis=1)
        indices = bias_cross_func(data, indices=indices)
    elif (FLD.MA30 not in indices.columns):
        indices = pd.concat([indices,
                             ma30_cross_func(data),
                             boll_cross_func(data),
                             macd_cross_v2_func(data, annual=annual),
                             dual_cross_v2_func(data),
                             maxfactor_cross_v2_func(data),
                             QA_indicator_BIAS(data, 10, 20, 30),],
                            axis=1)
        indices = bias_cross_func(data, indices=indices)

    if (FLD.BOLL_CROSS_SX_BIAS3 not in indices.columns):
        indices[FLD.BOLL_CROSS_SX_BIAS3] = np.where(indices[FLD.BOLL_CROSS_SX_BEFORE] == 0, 
                                                    indices[FLD.BIAS3], np.nan)
        indices[FLD.BOLL_CROSS_SX_BIAS3] = indices[FLD.BOLL_CROSS_SX_BIAS3].ffill()


    if (FLD.DRAWDOWN not in indices.columns):
        indices[FLD.PCT_CHANGE] = data.close.pct_change()
        indices[FLD.DRAWDOWN] = np.nan_to_num(np.log(data.close / data.high.shift(1)), nan=0)
        indices[FLD.DRAWDOWN_R4] = indices[FLD.DRAWDOWN].rolling(4).sum()
        indices[FLD.DRAWDOWN_R3] = indices[FLD.DRAWDOWN].rolling(3).sum()

    indices = ma_clearance_prepar_indices_func(indices)

    if (ST.VERBOSE in data.columns):
            print('Phase lineareg_cross_func', QA_util_timestamp_to_str())
    lrc, lrc_u, lrc_l, lrc_dir = lineareg_band(data)
    indices[FLD.ATR_CHANNEL] = lrc_channel = (lrc_u - lrc_l) / lrc
    indices[FLD.LINEAREG_PRICE] = lrc
    lrc_ham10_cross = np.where(indices[FLD.LINEAREG_PRICE] < indices[FLD.HMA10], 1,
                               np.where(indices[FLD.LINEAREG_PRICE] > indices[FLD.HMA10], -1, 0))
    indices[FLD.LRC_HMA10_TIMING_LAG] = calc_event_timing_lag(lrc_ham10_cross)
    indices[FLD.REGRESSION_SLOPE] = talib.LINEARREG_SLOPE(data.close, timeperiod=14)
    indices[FLD.REGRESSION_SLOPE_UB] = indices[FLD.REGRESSION_SLOPE].abs().rolling(20).median()

    indices[FLD.BOLL_DELTA] = indices[FLD.BOLL_DELTA] - indices[FLD.ATR_CHANNEL].diff(1)
    indices[FLD.BOLL_DIFF] = indices[FLD.BOLL].diff(1)
    indices[FLD.MA30_DIFF] = indices[FLD.MA30].diff(1)
    indices[FLD.LRC_CLEARANCE] = boll_clearance_func(indices, lrc) / indices[FLD.BOLL_CHANNEL]
    indices[FLD.HMA10_CLEARANCE] = boll_clearance_func(indices, indices[FLD.HMA10]) / indices[FLD.BOLL_CHANNEL]
    indices[FLD.HMA10_CLEARANCE_Q61] = indices[FLD.HMA10_CLEARANCE].rolling(84).quantile(0.618)
    indices[FLD.HMA10_CLEARANCE_Q75] = indices[FLD.HMA10_CLEARANCE].rolling(84).quantile(0.75)
    indices[FLD.HMA10_CLEARANCE_ZSCORE] = rolling_zscore(indices[FLD.HMA10_CLEARANCE].values, 84)

    indices[FLD.ATR_UB] = lrc_u
    indices[FLD.ATR_LB] = lrc_l

    if (ST.VERBOSE in data.columns):
            print('Phase lineareg_trend', QA_util_timestamp_to_str())
    LR_BAND = pd.DataFrame(columns=[FLD.LINEAREG_CROSS, 
                                    FLD.LINEAREG_CROSS_JX_BEFORE, 
                                    FLD.LINEAREG_CROSS_SX_BEFORE], 
                           index=data.index)

    # PASS 1
    if (ST.VERBOSE in data.columns):
            print('Phase lineareg_trend PASS 1', QA_util_timestamp_to_str())
    LR_BAND[FLD.LINEAREG_CROSS] = lrc_dir
    indices[FLD.LINEAREG_CROSS] = np.where((LR_BAND[FLD.LINEAREG_CROSS].diff() > 0), 1, 
                                           np.where((LR_BAND[FLD.LINEAREG_CROSS].diff() < 0), 
                                                    -1, 0))
    lineareg_cross_jx_before = np.where((LR_BAND[FLD.LINEAREG_CROSS].diff() > 0),
                                        1, 0)
    lineareg_cross_sx_before = np.where((LR_BAND[FLD.LINEAREG_CROSS].diff() < 0),
                                        1, 0)
    indices[FLD.LINEAREG_CROSS_JX_BEFORE] = Timeline_duration(lineareg_cross_jx_before)
    indices[FLD.LINEAREG_CROSS_SX_BEFORE] = Timeline_duration(lineareg_cross_sx_before)

    lineareg_cross_jx = Timeline_Integral(np.where(LR_BAND[FLD.LINEAREG_CROSS] > 0, 1, 0))
    lineareg_cross_sx = np.sign(LR_BAND[FLD.LINEAREG_CROSS]) * Timeline_Integral(np.where(LR_BAND[FLD.LINEAREG_CROSS] < 0, 1, 0))
    indices[FLD.LINEAREG_TREND_TIMING_LAG] = lineareg_cross_jx + lineareg_cross_sx

    # 防止插针行情突然搞乱故
    data['smooth_low'] = talib.MA(data.low, 2)
    data['smooth_high'] = talib.MA(data.high, 2)
    data['lrc'] = lrc
    ATR_BAND_TP_CROSS = pd.DataFrame(columns=['min_peak', 
                                              'max_peak'], 
                                     index=data.index)
    ATR_BAND_TP_CROSS['min_peak'] = np.minimum(data['close'], 
                                               np.minimum(data['open'], 
                                                      np.where(data['open'] < data['lrc'], 
                                                               data['low'], 
                                                               data['smooth_low'])))
    ATR_BAND_TP_CROSS['max_peak'] = np.maximum(data['open'], 
                                           np.maximum(data['close'], 
                                                      np.where(data['open'] > data['lrc'], 
                                                               data['high'], 
                                                               data['smooth_high'])))

    atr_tp_cross_jx = CROSS(ATR_BAND_TP_CROSS['min_peak'], 
                            indices[FLD.ATR_LB]).values
    atr_everst_cross_jx = CROSS(data[AKA.CLOSE], indices[FLD.ATR_LB]) & \
                          (indices[FLD.MA5_CLEARANCE] < 0.382) & \
                          (indices[FLD.MA5_CLEARANCE] < indices[FLD.MA30_CLEARANCE])

    atr_tp_cross_sx = CROSS(indices[FLD.ATR_UB], 
                            ATR_BAND_TP_CROSS['max_peak']).values
    indices[FLD.ATR_CROSS] = np.where((lrc_l > lrc_l.shift(1)) & \
                                      (lrc_l < data[AKA.CLOSE]) & \
                                      (lrc > indices[FLD.MA30]), 1,
                                      np.where(lrc_l < lrc_l.shift(1), -1, 0))

    indices[FLD.ATR_CROSS_JX_BEFORE] = Timeline_duration(atr_tp_cross_jx | \
                                                         atr_everst_cross_jx.values)
    indices[FLD.ATR_CROSS_SX_BEFORE] = Timeline_duration(atr_tp_cross_sx)

    # 价值“浅滩”
    indices[FLD.ATR_LB_PADDING] = (data[AKA.CLOSE] - indices[FLD.ATR_LB])
    price_zscore_21 = rolling_pctrank(data[AKA.CLOSE].values, w=21)
    indices[FLD.ZSCORE_21] = (rolling_pctrank(indices[FLD.ATR_LB_PADDING].values, w=21) + price_zscore_21) / 2
    indices[FLD.ZSCORE_21] = np.where(price_zscore_21 < 0.618, 
                                      indices[FLD.ZSCORE_21], 
                                      price_zscore_21)

    price_zscore_84 = rolling_pctrank(data[AKA.CLOSE].values, w=84)
    indices[FLD.ZSCORE_84] = (rolling_pctrank(indices[FLD.ATR_LB_PADDING].values, w=84) + price_zscore_84) / 2
    indices[FLD.ZSCORE_84] = np.where(price_zscore_84 < 0.618, 
                                      indices[FLD.ZSCORE_84], 
                                      price_zscore_84)

    zscore_boost_cross = np.where(indices[FLD.ZSCORE_21] > indices[FLD.ZSCORE_84], 1, 
                                  np.where(indices[FLD.ZSCORE_21] < indices[FLD.ZSCORE_84], -1, 0)) 
    indices[FLD.ZSCORE_BOOST_TIMING_LAG] = calc_event_timing_lag(zscore_boost_cross)

    if (FLD.VOLUME_FLOW not in indices.columns):
        # 量能（价）指标
        indices = volume_flow_cross(data, indices)

    atr_ub_boll_cross = np.where(indices[FLD.ATR_UB] > indices[FLD.BOLL], 1, 
                               np.where(indices[FLD.ATR_UB] < indices[FLD.BOLL], -1, 0))
    indices[FTR.ATR_UB_BOLL_TIMING_LAG] = calc_event_timing_lag(atr_ub_boll_cross)

    return indices


def lineareg_band_cross_func(data, *args, **kwargs):
    """
    ATR线性回归带金叉死叉策略，支持QA add_func，第二个参数 默认为 indices= 为已经计算指标
    理论上这个函数只计算单一标的，不要尝试传递复杂标的，indices会尝试拆分。
    """
    # 针对多标的，拆分 indices 数据再自动合并
    code = data.index.get_level_values(level=1)[0]
    if ('indices' in kwargs.keys()):
        indices = kwargs['indices'].loc[(slice(None), code), :]
    elif (len(args) > 0):
        indices = args[0].loc[(slice(None), code), :]
    else:
        indices = None
    annual = kwargs['annual'] if ('annual' in kwargs.keys()) else 252

    if (ST.VERBOSE in data.columns):
        print('Phase lineareg_band_cross_func', QA_util_timestamp_to_str())

    features_len_count = len(data)
    if (indices is None):
        indices = pd.concat([ma30_cross_func(data),
                             boll_cross_func(data),
                             macd_cross_v2_func(data, annual=annual),
                             dual_cross_func(data),
                             maxfactor_cross_func(data),
                             QA_indicator_BIAS(data, 10, 20, 30),],
                            axis=1)
        indices = bias_cross_func(data, indices=indices)
    elif (FLD.MA30 not in indices.columns):
        indices = pd.concat([indices,
                             ma30_cross_func(data),
                             boll_cross_func(data),
                             macd_cross_v2_func(data, annual=annual),
                             dual_cross_func(data),
                             maxfactor_cross_func(data),
                             QA_indicator_BIAS(data, 10, 20, 30),],
                            axis=1)
        indices = bias_cross_func(data, indices=indices)

    if (FLD.BOLL_CROSS_SX_BIAS3 not in indices.columns):
        indices[FLD.BOLL_CROSS_SX_BIAS3] = np.where(indices[FLD.BOLL_CROSS_SX_BEFORE] == 0, 
                                                    indices[FLD.BIAS3], np.nan)
        indices[FLD.BOLL_CROSS_SX_BIAS3] = indices[FLD.BOLL_CROSS_SX_BIAS3].ffill()

    if (FLD.DRAWDOWN not in indices.columns):
        indices[FLD.PCT_CHANGE] = data.close.pct_change()
        #print(indices[FLD.PCT_CHANGE].skew(),
        #indices[FLD.PCT_CHANGE].rolling(4).skew(),
        #indices[FLD.PCT_CHANGE].rolling(8).skew())
        with np.errstate(invalid='ignore', divide='ignore'):
            indices['drawdown_today'] = np.nan_to_num(np.log(data.close / data.high), nan=0)
            indices[FLD.DRAWDOWN] = np.nan_to_num(np.log(data.close / data.high.shift(1)), nan=0)
        indices[FLD.DRAWDOWN_R4] = indices[FLD.DRAWDOWN].rolling(4).sum()
        indices[FLD.DRAWDOWN_R3] = indices[FLD.DRAWDOWN].rolling(3).sum()

    indices = ma_clearance_prepar_indices_func(indices)

    if (ST.VERBOSE in data.columns):
            print('Phase lineareg_cross', QA_util_timestamp_to_str())
    lrc, lrc_u, lrc_l, lrc_dir = lineareg_band(data)
    indices[FLD.ATR_CHANNEL] = lrc_channel = (lrc_u - lrc_l) / lrc
    indices[FLD.LINEAREG_PRICE] = lrc
    indices[FLD.REGRESSION_SLOPE] = talib.LINEARREG_SLOPE(data.close, timeperiod=14)
    indices[FLD.REGRESSION_SLOPE_UB] = indices[FLD.REGRESSION_SLOPE].abs().rolling(20).median()

    indices[FLD.BOLL_DELTA] = indices[FLD.BOLL_DELTA] - indices[FLD.ATR_CHANNEL].diff(1)
    indices[FLD.BOLL_DIFF] = indices[FLD.BOLL].diff(1)
    indices[FLD.MA30_DIFF] = indices[FLD.MA30].diff(1)
    indices[FLD.LRC_CLEARANCE] = boll_clearance_func(indices, lrc) / indices[FLD.BOLL_CHANNEL]
    indices[FLD.HMA10_CLEARANCE] = boll_clearance_func(indices, indices[FLD.HMA10]) / indices[FLD.BOLL_CHANNEL]
    indices[FLD.HMA10_CLEARANCE_Q61] = indices[FLD.HMA10_CLEARANCE].rolling(84).quantile(0.618)
    indices[FLD.HMA10_CLEARANCE_Q75] = indices[FLD.HMA10_CLEARANCE].rolling(84).quantile(0.75)
    indices[FLD.HMA10_CLEARANCE_ZSCORE] = rolling_zscore(indices[FLD.HMA10_CLEARANCE].values, 84)

    indices[FLD.ATR_UB] = lrc_u
    indices[FLD.ATR_LB] = lrc_l

    if (ST.VERBOSE in data.columns):
            print('Phase lineareg_trend', QA_util_timestamp_to_str())
    LR_BAND = pd.DataFrame(columns=[FLD.LINEAREG_CROSS, 
                                    FLD.LINEAREG_CROSS_JX_BEFORE, 
                                    FLD.LINEAREG_CROSS_SX_BEFORE], 
                           index=data.index)

    # PASS 1
    if (ST.VERBOSE in data.columns):
            print('Phase lineareg_trend PASS 1', QA_util_timestamp_to_str())
    indices[FLD.LINEAREG_TREND] = LR_BAND[FLD.LINEAREG_CROSS] = lrc_dir
    indices[FLD.LINEAREG_CROSS] = np.where((LR_BAND[FLD.LINEAREG_CROSS].diff() > 0), 1, 
                                           np.where((LR_BAND[FLD.LINEAREG_CROSS].diff() < 0), 
                                                    -1, 0))
    lineareg_cross_jx_before = np.where((LR_BAND[FLD.LINEAREG_CROSS].diff() > 0),
                                        1, 0)
    lineareg_cross_sx_before = np.where((LR_BAND[FLD.LINEAREG_CROSS].diff() < 0),
                                        1, 0)
    indices[FLD.LINEAREG_TREND_CROSS_JX_BEFORE] = indices[FLD.LINEAREG_CROSS_JX_BEFORE] = Timeline_duration(lineareg_cross_jx_before)
    indices[FLD.LINEAREG_TREND_CROSS_SX_BEFORE] = indices[FLD.LINEAREG_CROSS_SX_BEFORE] = Timeline_duration(lineareg_cross_sx_before)
    indices[FLD.LINEAREG_TREND_CROSS_JX] = Timeline_Integral(np.where(indices[FLD.LINEAREG_TREND] == 1, 1, 0))
    indices[FLD.LINEAREG_TREND_CROSS_SX] = Timeline_Integral(np.where(indices[FLD.LINEAREG_TREND] == -1, 1, 0))

    LR_BAND[FLD.MA90_CLEARANCE_RETURNS] = indices[FLD.MA90_CLEARANCE].diff()
    LR_BAND[FLD.MA90_CLEARANCE_CROSS] = np.where((LR_BAND[FLD.MA90_CLEARANCE_RETURNS] > 0), 1, 
                                           np.where((LR_BAND[FLD.MA90_CLEARANCE_RETURNS] < 0), 
                                                    -1, 0))
    ma90_clearance_cross_jx = np.where((LR_BAND[FLD.MA90_CLEARANCE_RETURNS] > 0), 
                                                    1, 0)
    ma90_clearance_cross_sx = np.where((LR_BAND[FLD.MA90_CLEARANCE_RETURNS] < 0), 
                                                    1, 0)
    indices[FLD.MA90_CLEARANCE_CROSS_JX] = Timeline_Integral(ma90_clearance_cross_jx)
    indices[FLD.MA90_CLEARANCE_CROSS_SX] = Timeline_Integral(ma90_clearance_cross_sx)

    lineareg_cross_jx = Timeline_Integral(np.where(LR_BAND[FLD.LINEAREG_CROSS] > 0, 1, 0))
    lineareg_cross_sx = np.sign(LR_BAND[FLD.LINEAREG_CROSS]) * Timeline_Integral(np.where(LR_BAND[FLD.LINEAREG_CROSS] < 0, 1, 0))
    indices[FLD.LINEAREG_TREND_TIMING_LAG] = lineareg_cross_jx + lineareg_cross_sx

    # 防止插针行情突然搞乱故
    data['smooth_low'] = talib.MA(data.low, 2)
    data['smooth_high'] = talib.MA(data.high, 2)
    data['lrc'] = lrc
    ATR_BAND_TP_CROSS = pd.DataFrame(columns=['min_peak', 
                                              'max_peak'], 
                                     index=data.index)
    ATR_BAND_TP_CROSS['min_peak'] = np.minimum(data['close'], 
                                               np.minimum(data['open'], 
                                                      np.where(data['open'] < data['lrc'], 
                                                               data['low'], 
                                                               data['smooth_low'])))
    ATR_BAND_TP_CROSS['max_peak'] = np.maximum(data['open'], 
                                           np.maximum(data['close'], 
                                                      np.where(data['open'] > data['lrc'], 
                                                               data['high'], 
                                                               data['smooth_high'])))

    atr_tp_cross_jx = CROSS(ATR_BAND_TP_CROSS['min_peak'], 
                            indices[FLD.ATR_LB]).values
    atr_everst_cross_jx = CROSS(data[AKA.CLOSE], indices[FLD.ATR_LB]) & \
                          (indices[FLD.MA5_CLEARANCE] < 0.382) & \
                          (indices[FLD.MA5_CLEARANCE] < indices[FLD.MA30_CLEARANCE])

    atr_tp_cross_sx = CROSS(indices[FLD.ATR_UB], 
                            ATR_BAND_TP_CROSS['max_peak']).values
    indices[FLD.ATR_CROSS] = np.where((lrc_l > lrc_l.shift(1)) & \
                                      (lrc_l < data[AKA.CLOSE]) & \
                                      (lrc > indices[FLD.MA30]), 1,
                                      np.where(lrc_l < lrc_l.shift(1), -1, 0))

    indices[FLD.ATR_CROSS_JX_BEFORE] = Timeline_duration(atr_tp_cross_jx | \
                                                         atr_everst_cross_jx.values)
    indices[FLD.ATR_CROSS_SX_BEFORE] = Timeline_duration(atr_tp_cross_sx)

    # 价值“浅滩”
    indices[FLD.ATR_LB_PADDING] = (data[AKA.CLOSE] - indices[FLD.ATR_LB])
    indices[FLD.ZSCORE_21] = (rolling_pctrank(indices[FLD.ATR_LB_PADDING].values, w=21) + \
                              rolling_pctrank(data[AKA.CLOSE].values, w=21)) / 2

    indices[FLD.ZSCORE_84] = (rolling_pctrank(indices[FLD.ATR_LB_PADDING].values, w=84) + \
                              rolling_pctrank(data[AKA.CLOSE].values, w=84)) / 2

    if (FLD.VOLUME_FLOW not in indices.columns):
        # 量能（价）指标
        indices = volume_flow_cross(data, indices)

    atr_ub_boll_cross = np.where(indices[FLD.ATR_UB] > indices[FLD.BOLL], 1, 
                               np.where(indices[FLD.ATR_UB] < indices[FLD.BOLL], -1, 0))
    indices[FTR.ATR_UB_BOLL_TIMING_LAG] = calc_event_timing_lag(atr_ub_boll_cross)

    indices[FLD.ATR_RATIO] = indices[FLD.ATR_CHANNEL] / indices[FLD.BOLL_CHANNEL]

    indices[FLD.LINEAREG_TREND_R4] = indices[FLD.LINEAREG_TREND].rolling(4).sum()
    eject_positive = (((indices[FLD.LINEAREG_TREND] > 0) & \
        (indices[FLD.DEA] < 0) & \
        (indices[FLD.LINEAREG_TREND_R4] < 1) & \
        (indices[FLD.ATR_RATIO] > 0.618)) | \
        ((indices[FLD.LINEAREG_TREND] == 1) & \
        (indices[FLD.LINEAREG_TREND].shift(1) == 0) & \
        (indices[FLD.DEA] < 0) & \
        (indices[FLD.LINEAREG_TREND_R4] < 0) & \
        (indices[FLD.ATR_RATIO] > 0.618))) & \
        ~((indices[FLD.ATR_CROSS_JX_BEFORE] == indices[FLD.BOLL_CROSS_JX_BEFORE]) & \
        (indices[FLD.BOLL_DELTA] > -0.0008))

    # PASS 2 去除抖动点
    if (ST.VERBOSE in data.columns):
        print('Phase lineareg_trend PASS 2', QA_util_timestamp_to_str())
    indices[FLD.LINEAREG_TREND] = np.where((lrc_dir > 0) & ~(eject_positive == True), 1, 
                                           np.where(lrc_dir < 0, -1, 0))
    LR_BAND[FLD.LINEAREG_CROSS] = indices[FLD.LINEAREG_TREND]
    indices[FLD.LINEAREG_CROSS] = np.where((LR_BAND[FLD.LINEAREG_CROSS].diff() > 0), 1, 
                                           np.where((LR_BAND[FLD.LINEAREG_CROSS].diff() < 0), 
                                                    -1, 0))
    lineareg_cross_jx_before = np.where((LR_BAND[FLD.LINEAREG_CROSS].diff() > 0),
                                        1, 0)
    lineareg_cross_sx_before = np.where((LR_BAND[FLD.LINEAREG_CROSS].diff() < 0),
                                        1, 0)
    indices[FLD.LINEAREG_CROSS_JX_BEFORE] = Timeline_duration(lineareg_cross_jx_before)
    indices[FLD.LINEAREG_CROSS_SX_BEFORE] = Timeline_duration(lineareg_cross_sx_before)
    indices[FLD.LINEAREG_CROSS_FADE] = eject_positive

    lineareg_cross_sx = (((indices[FLD.LINEAREG_CROSS] == -1) | \
        (indices[FLD.LINEAREG_CROSS_SX_BEFORE] < indices[FLD.LINEAREG_CROSS_JX_BEFORE])) & (\
        (indices[FLD.ATR_RATIO] > 1 / 2.06) & \
        (indices[FLD.MA90_CLEARANCE] > indices[FLD.MA90_CLEARANCE].median()) & \
        (indices[FLD.LINEAREG_CROSS_JX_BEFORE] < indices[FLD.LINEAREG_CROSS_JX_BEFORE].quantile(0.382)) & (\
        (indices[FLD.BOLL_CROSS_SX_BEFORE] < 6) | \
        (indices[FLD.BOLL_CROSS_SX_BEFORE] < indices[FLD.MACD_CROSS_SX_BEFORE]))) & \
        (indices[FLD.PCT_CHANGE] < -0.0002))
    lineareg_cross_sx = (lineareg_cross_sx == True) | \
        (((indices[FLD.LINEAREG_CROSS] == -1) | \
        (indices[FLD.LINEAREG_CROSS_SX_BEFORE] < indices[FLD.LINEAREG_CROSS_JX_BEFORE])) & \
        (indices[FLD.ATR_RATIO] > 1 / 3.09) & \
        (indices[FLD.BOLL_CROSS_JX_BEFORE] < 6) & \
        (indices[FLD.PCT_CHANGE] < -0.0002)) 
    lineareg_cross_sx = (lineareg_cross_sx == True) | \
        ((indices[FLD.LINEAREG_CROSS] == 1) & \
        (indices[FLD.LINEAREG_CROSS_JX_BEFORE] >= 8) & (\
        (indices[FLD.ATR_RATIO] > 1 / 2.06) & \
        (indices[FLD.MA90_CLEARANCE] > indices[FLD.MA90_CLEARANCE].median()) & \
        (indices[FLD.LINEAREG_CROSS_SX_BEFORE] < indices[FLD.LINEAREG_CROSS_SX_BEFORE].quantile(0.382)) & (\
        (indices[FLD.BOLL_CROSS_SX_BEFORE] < 6) | \
        (indices[FLD.BOLL_CROSS_SX_BEFORE] < indices[FLD.MACD_CROSS_SX_BEFORE]))) & \
        (indices[FLD.PCT_CHANGE] < -0.0002)) 
    lineareg_cross_sx = (lineareg_cross_sx == True) | \
        (((indices[FLD.LINEAREG_CROSS] == -1) | \
        (indices[FLD.LINEAREG_CROSS_SX_BEFORE] < indices[FLD.LINEAREG_CROSS_JX_BEFORE])) & \
        (indices[FLD.BOLL_UB] < indices[FLD.ATR_UB].shift(1)) & (\
        (indices[FLD.BOLL_CROSS_SX_BEFORE] < 6) | \
        ((indices[FLD.LINEAREG_CROSS_JX_BEFORE] > indices[FLD.LINEAREG_CROSS_JX_BEFORE].quantile(0.618)) & \
        ((indices[FLD.MA90_CLEARANCE] < indices[FLD.MA120_CLEARANCE]) | \
        (indices[FLD.LINEAREG_CROSS] > -1))) | \
        (indices[FLD.MA30_CROSS_SX_BEFORE] < 6)) & \
        (indices[FLD.PCT_CHANGE] < -0.0002))
    lineareg_cross_sx = (lineareg_cross_sx == True) | \
        ((indices[FLD.MA90] < indices[FLD.MA120]) & \
        (indices[FLD.LINEAREG_CROSS_JX_BEFORE] >= 8) & \
        ((indices[FLD.MACD] < 0) | \
        (indices[FLD.DUAL_CROSS_SX] > 0) | \
        (indices[FLD.MA90_CLEARANCE_CROSS_SX] < indices[FLD.BOLL_CROSS_JX_BEFORE])) & \
        (indices[FLD.DIF] < 0) & \
        (indices[FLD.MA90_CLEARANCE] > -0.382) & \
        (indices[FLD.MA90_CLEARANCE] < 0.191) & \
        (indices[FLD.MA90_CLEARANCE_CROSS_SX] > 0) & \
        (indices[FLD.PCT_CHANGE] < -0.0002))
    lineareg_cross_sx = (lineareg_cross_sx == True) | \
        (eject_positive == True)

    boll_drawdown_x1 = indices.query('{}>0 & {}<=1 & {}>0 & {}>0'.format(FLD.LINEAREG_TREND, 
                                                                         FLD.BOLL_CROSS_SX_BEFORE, 
                                                                         FLD.DEA, 
                                                                         FLD.MACD))
    #print(boll_drawdown_x1[FLD.DRAWDOWN].quantile(0.382),
    #boll_drawdown_x1[FLD.DRAWDOWN].quantile(0.192),
    #boll_drawdown_x1[FLD.DRAWDOWN].median())
    #print(boll_drawdown_x1[FLD.DRAWDOWN_HIGH].quantile(0.382),
    #boll_drawdown_x1[FLD.DRAWDOWN_HIGH].quantile(0.192),
    #boll_drawdown_x1[FLD.DRAWDOWN_HIGH].median())
    lineareg_cross_sx = (lineareg_cross_sx == True) | \
        ((indices[FLD.LINEAREG_TREND] > 0) & \
        (indices[FLD.LINEAREG_CROSS_JX_BEFORE] >= 8) & \
        ((indices[FLD.BOLL_CROSS_SX_BEFORE] <= 1) | \
        (indices[FLD.ATR_CROSS_SX_BEFORE] <= 1)) & \
        (indices[FLD.DEA] > 0) & \
        (indices[FLD.MACD] > 0) & \
        (indices[FLD.MACD_DELTA] < 0) & \
        (indices[FLD.PCT_CHANGE] < -0.0002) & \
        (indices[FLD.DRAWDOWN] < boll_drawdown_x1[FLD.DRAWDOWN].quantile(0.192)))
    lineareg_cross_sx = (lineareg_cross_sx == True) | \
        ((indices[FLD.LINEAREG_TREND] > 0) & \
        (indices[FLD.LINEAREG_CROSS_JX_BEFORE] >= 8) & \
        ((indices[FLD.BOLL_CROSS_SX_BEFORE] <= 1) & \
        (indices[FLD.ATR_CROSS_SX_BEFORE] <= 1)) & \
        (indices[FLD.DEA] > 0) & \
        (indices[FLD.MACD] > 0) & \
        (indices[FLD.MACD_DELTA] < 0) & \
        (indices[FLD.PCT_CHANGE] < -0.0002) & \
        (indices[FLD.DRAWDOWN].rolling(4).sum() < boll_drawdown_x1[FLD.DRAWDOWN].quantile(0.192)))
    lineareg_cross_sx = (lineareg_cross_sx == True) | \
        ((indices[FLD.LINEAREG_TREND] < 0) & \
        (indices[FLD.BOLL_CROSS_SX_BEFORE] >= indices[FLD.LINEAREG_CROSS_SX_BEFORE]) & \
        ((indices[FLD.BOLL_CROSS_SX_BEFORE] <= indices[FLD.BOLL_CROSS_JX_BEFORE]) & \
        (indices[FLD.ATR_CROSS_SX_BEFORE] <= indices[FLD.ATR_CROSS_JX_BEFORE])) & \
        (indices[FLD.DEA] > 0) & \
        (indices[FLD.MACD] < 0) & \
        (indices[FLD.MACD_DELTA] < 0) & \
        (indices[FLD.PCT_CHANGE] < -0.0002) & \
        (indices[FLD.DRAWDOWN].rolling(4).sum() < boll_drawdown_x1[FLD.DRAWDOWN].quantile(0.382)))
    lineareg_cross_sx = (lineareg_cross_sx == True) | \
        ((indices[FLD.LINEAREG_TREND] < 0) & \
        (indices[FLD.BOLL_CROSS_SX_BEFORE] >= indices[FLD.LINEAREG_CROSS_SX_BEFORE]) & \
        (indices[FLD.ATR_RATIO] > 1 / 2.06) & \
        (indices[FLD.DEA] > 0) & \
        (indices[FLD.MACD] < 0) & \
        (indices[FLD.MACD_DELTA] < 0) & \
        (indices[FLD.DRAWDOWN] < -0.0002) & \
        (indices[FLD.PCT_CHANGE] < -0.0002))
    lineareg_cross_sx = (lineareg_cross_sx == True) | \
        ((indices[FLD.LINEAREG_TREND] < 0) & \
        (indices[FLD.BOLL_CROSS_SX_BEFORE] >= indices[FLD.LINEAREG_CROSS_SX_BEFORE]) & \
        (indices[FLD.ATR_RATIO] > 1 / 2.06) & \
        (indices[FLD.DEA] < 0) & \
        (indices[FLD.MACD_DELTA] < 0) & \
        (indices[FLD.DRAWDOWN] < -0.0002) & \
        (indices[FLD.PCT_CHANGE] < -0.0002))

    lineareg_cross_jx = ((indices[FLD.LOWER_SETTLE_PRICE] == 1) & \
        (indices[FLD.MA90_CLEARANCE] < indices[FLD.MA90_CLEARANCE].quantile(0.191)) & \
        (indices[FLD.LINEAREG_CROSS_SX_BEFORE] > indices[FLD.LINEAREG_CROSS_SX_BEFORE].quantile(0.618)) & \
        (indices[FLD.BOLL_CROSS_JX_BEFORE] < indices[FLD.BOLL_CROSS_SX_BEFORE]) & \
        ~((indices[FLD.MA90] < indices[FLD.MA120]) & (indices[FLD.DIF] < 0) & \
        (indices[FLD.MA90_CLEARANCE] > -0.382) & \
        (indices[FLD.MA90_CLEARANCE] < 0.191) & \
        (indices[FLD.MA90_CLEARANCE_CROSS_SX] > 0)) & \
        ((indices[FLD.LINEAREG_CROSS_JX_BEFORE] >= indices[FLD.MACD_CROSS_JX_BEFORE]) | (indices[FLD.MACD] < 0)) & \
        (lrc_dir > 0))
    lineareg_cross_jx = (lineareg_cross_jx == True) | \
        ((indices[FLD.ATR_RATIO] > 1 / 3.09) & \
        ~((indices[FLD.MA90] < indices[FLD.MA120]) & (indices[FLD.DIF] < 0) & \
        (indices[FLD.MA90_CLEARANCE] > -0.382) & (indices[FLD.MA90_CLEARANCE] < 0.191) & \
        (indices[FLD.MA90_CLEARANCE_CROSS_SX] > 0)) & \
        (indices[FLD.LOWER_SETTLE_PRICE] == 1) & (lrc_dir > 0))
    lineareg_cross_jx = (lineareg_cross_jx == True) | \
        ((indices[FLD.LINEAREG_CROSS_JX_BEFORE] <= 4) & \
        (indices[FLD.LINEAREG_TREND] > 0) & \
        (indices[FLD.LINEAREG_CROSS].rolling(4).sum() > 0) & \
        (((indices[FLD.ATR_CROSS_JX_BEFORE] <= indices[FLD.BOLL_CROSS_JX_BEFORE]) & \
        ((indices[FLD.ATR_CROSS_JX_BEFORE] - indices[FLD.LINEAREG_CROSS_JX_BEFORE]) > 6)) | \
        ((indices[FLD.LINEAREG_CROSS_JX_BEFORE] <= indices[FLD.BOLL_CROSS_JX_BEFORE]) & \
        (indices[FLD.MAXFACTOR_CROSS_JX] <= indices[FLD.LINEAREG_CROSS_JX_BEFORE]) & \
        (indices[FLD.ATR_CROSS_JX_BEFORE] <= indices[FLD.BOLL_CROSS_JX_BEFORE])) | \
        ((indices[FLD.LINEAREG_CROSS_JX_BEFORE] <= indices[FLD.BOLL_CROSS_JX_BEFORE]) & \
        (indices[FLD.MAXFACTOR_CROSS_JX] <= indices[FLD.LINEAREG_CROSS_JX_BEFORE]) & \
        ((indices[FLD.ATR_CROSS_JX_BEFORE] - indices[FLD.LINEAREG_CROSS_JX_BEFORE]) > 6)) | \
        (((indices[FLD.ATR_CROSS_JX_BEFORE] - indices[FLD.BOLL_CROSS_JX_BEFORE]) < 4) & \
        ((indices[FLD.BOLL_CROSS_JX_BEFORE] - indices[FLD.LINEAREG_CROSS_JX_BEFORE]) < 4)) | \
        (((indices[FLD.ATR_CROSS_JX_BEFORE] - indices[FLD.BOLL_CROSS_JX_BEFORE]) > 6) & \
        ((indices[FLD.BOLL_CROSS_JX_BEFORE] - indices[FLD.LINEAREG_CROSS_JX_BEFORE]) < 6) & \
        ((indices[FLD.LINEAREG_CROSS_JX_BEFORE] - indices[FLD.MAXFACTOR_CROSS_JX]) <= 2))) & \
        ((indices[FLD.DRAWDOWN] > -0.0001) | \
        (indices[FLD.PCT_CHANGE] > -0.0001)) & \
        (indices[FLD.MA90] > indices[FLD.LINEAREG_PRICE]) & \
        (indices[FLD.MA90] > indices[FLD.ATR_LB]))

    # lineareg_cross_jx 风控检查，过滤高危型
    bandwidth_limit = min(0.0309, indices[FLD.BOLL_CHANNEL].quantile(0.382))
    lineareg_cross_jx = (lineareg_cross_jx == True) & \
        ((indices[FLD.DRAWDOWN] > -0.0001) | \
        (indices[FLD.PCT_CHANGE] > -0.0001)) & \
        (((indices[FLD.ATR_CROSS_JX_BEFORE] <= indices[FLD.BOLL_CROSS_JX_BEFORE]) & \
        ((indices[FLD.ATR_CROSS_JX_BEFORE] - indices[FLD.LINEAREG_CROSS_JX_BEFORE]) > 6)) | \
        ((indices[FLD.LINEAREG_CROSS_JX_BEFORE] <= indices[FLD.BOLL_CROSS_JX_BEFORE]) & \
        (indices[FLD.MAXFACTOR_CROSS_JX] <= indices[FLD.LINEAREG_CROSS_JX_BEFORE]) & \
        (indices[FLD.ATR_CROSS_JX_BEFORE] <= indices[FLD.BOLL_CROSS_JX_BEFORE])) | \
        ((indices[FLD.LINEAREG_CROSS_JX_BEFORE] <= indices[FLD.BOLL_CROSS_JX_BEFORE]) & \
        (indices[FLD.MAXFACTOR_CROSS_JX] <= indices[FLD.LINEAREG_CROSS_JX_BEFORE]) & \
        ((indices[FLD.ATR_CROSS_JX_BEFORE] - indices[FLD.LINEAREG_CROSS_JX_BEFORE]) > 6)) | \
        (((indices[FLD.ATR_CROSS_JX_BEFORE] - indices[FLD.BOLL_CROSS_JX_BEFORE]) < 4) & \
        ((indices[FLD.BOLL_CROSS_JX_BEFORE] - indices[FLD.LINEAREG_CROSS_JX_BEFORE]) < 4)) | \
        (((indices[FLD.ATR_CROSS_JX_BEFORE] - indices[FLD.BOLL_CROSS_JX_BEFORE]) > 6) & \
        ((indices[FLD.BOLL_CROSS_JX_BEFORE] - indices[FLD.LINEAREG_CROSS_JX_BEFORE]) < 6) & \
        ((indices[FLD.LINEAREG_CROSS_JX_BEFORE] - indices[FLD.MAXFACTOR_CROSS_JX]) <= 2))) & \
        ((data[AKA.OPEN] < indices[FLD.MA30]) | \
        ((data[AKA.OPEN] < indices[FLD.MA30]).rolling(6).sum() > 0) & \
        ((indices[FLD.DEA] < 0) & \
        ((indices[FLD.DEA_SLOPE] > 0) | \
        (indices[FLD.DIF] > 0))) | \
        (indices[FLD.DEA_CROSS_JX_BEFORE] <= indices[FLD.MACD_CROSS_JX_BEFORE]) | \
        (indices[FLD.DEA_CROSS_JX_BEFORE] <= indices[FLD.LINEAREG_CROSS_JX_BEFORE])) & \
        ~((indices[FLD.ATR_CROSS_SX_BEFORE] <= indices[FLD.LINEAREG_CROSS_JX_BEFORE])) & \
        ~((indices[FLD.BOLL_CROSS_SX_BEFORE] <= indices[FLD.LINEAREG_CROSS_JX_BEFORE])) & \
        ~((indices[FLD.BOLL_CHANNEL] < bandwidth_limit) & \
        ((np.minimum(indices[FLD.BOLL_CROSS_JX_BEFORE] - indices[FLD.LINEAREG_CROSS_JX_BEFORE],
                     indices[FLD.ATR_CROSS_JX_BEFORE] - indices[FLD.LINEAREG_CROSS_JX_BEFORE]) > 2) | \
        (indices[FLD.LINEAREG_CROSS_JX_BEFORE] > 2))) & \
        ~((indices[FLD.MA30_SLOPE] < 0) & \
        (indices[FLD.DEA_SLOPE] < 0) & \
        (indices[FLD.REGRESSION_SLOPE] < 0) & \
        (indices['BIAS3'] < -9.27) & \
        ((indices[FLD.MA90_CLEARANCE] + indices[FLD.MA120_CLEARANCE]) > 0.0)) & \
        ~((indices[FLD.MA30_SLOPE] < 0) & \
        (indices[FLD.DEA_SLOPE] < 0) & \
        (indices[FLD.REGRESSION_SLOPE] < 0) & \
        (indices[FLD.DEA] < 0) & \
        (indices['BIAS3'] < -6.18) & \
        ((indices[FLD.MA90_CLEARANCE] + indices[FLD.MA120_CLEARANCE]) > 0.0))

    indices[FLD.LINEAREG_BAND_CROSS] = np.where(lineareg_cross_jx == True, 1, 
                                                np.where(lineareg_cross_sx == True, -1, 0))
    lineareg_band_jx_before = np.where((indices[FLD.LINEAREG_BAND_CROSS] == 1), 
                                       1, 0)
    lineareg_band_sx_before = np.where((indices[FLD.LINEAREG_BAND_CROSS] == -1), 
                                       1, 0)
    indices[FLD.LINEAREG_BAND_JX_BEFORE] = Timeline_duration(lineareg_band_jx_before)
    indices[FLD.LINEAREG_BAND_SX_BEFORE] = Timeline_duration(lineareg_band_sx_before)

    #indices[FLD.LINEAREG_TREND_R4] = indices.apply(lambda x:
    #                                               timeline_corr_from_last_phase_func(data,
    #                                                                                  indices,
    #                                                                                  column_name=FLD.LINEAREG_TREND,
    #                                                                                  bar_ind=x),
    #                                               axis=1)
    #print(indices[FLD.LINEAREG_TREND_R4]>indices[FLD.LINEAREG_TREND_CROSS_JX])
    #LINEAREG_TREND = indices_density_func(data, FLD.LINEAREG_TREND, indices)
    #LINEAREG_BAND = indices_density_func(data, FLD.LINEAREG_BAND, indices)

    return indices


def volume_flow_cross_func(data, *args, **kwargs):
    """
    准备计算 量能（价）指标，
    支持QA add_func，第二个参数 默认为 indices= 为已经计算指标
    理论上这个函数只计算单一标的，不要尝试传递复杂标的，indices会尝试拆分。
    """
    # 针对多标的，拆分 indices 数据再自动合并
    code = data.index.get_level_values(level=1)[0]
    if ('indices' in kwargs.keys()):
        indices = kwargs['indices'].loc[(slice(None), code), :]
    elif (len(args) > 0):
        indices = args[0].loc[(slice(None), code), :]
    else:
        print(u'Missing paramters: agrs[0] or kwargs[\'indices\']')
        indices = None

    return volume_flow_cross(data, indices)


def volume_flow_cross(data, indices):
    """
    准备计算 量能（价）指标
    单标的模式，不要尝试传递复杂标的
    """
    if (ST.VERBOSE in data.columns):
        print('Phase volume_flow_cross', QA_util_timestamp_to_str())

    indices[FLD.VOLUME_FLOW] = np.sign(indices[FLD.PCT_CHANGE].rolling(4).sum().fillna(-0.0002)) * data[AKA.VOLUME].rolling(4).mean().fillna(-0.0002)
    indices[FLD.VOLUME_FLOW_RATIO] = indices[FLD.VOLUME_FLOW] / (indices[FLD.VOLUME_FLOW].max() - indices[FLD.VOLUME_FLOW].min())
    indices[FLD.VOLUME_AD] = talib.AD(data.high.values, 
                                        data.low.values, 
                                        data.close.values, 
                                        data.volume.values,)

    indices[FLD.VOLUME_AD_MA] = talib.SMA(indices[FLD.VOLUME_AD].values, 
                                        timeperiod=21)
    indices[FLD.VOLUME_AD_FLOW] = (indices[FLD.VOLUME_AD] - indices[FLD.VOLUME_AD_MA])
    indices[FLD.VOLUME_ADOSC] = talib.ADOSC(data.high.values, 
                                        data.low.values, 
                                        data.close.values, 
                                        data.volume.values, 
                                        fastperiod=3, 
                                        slowperiod=10)
    indices[FLD.VOLUME_ADOSC_MA] = talib.SMA(indices[FLD.VOLUME_ADOSC].values, 
                                        timeperiod=21)
    indices[FLD.VOLUME_ADOSC_FLOW] = (indices[FLD.VOLUME_ADOSC] - indices[FLD.VOLUME_ADOSC_MA])
    indices[FLD.VOLUME_FLOW_TRI_CROSS_JX] = Timeline_Integral(((indices[FLD.VOLUME_FLOW] > 0) & \
                                                        (indices[FLD.VOLUME_AD_FLOW] > 0) & \
                                                        (indices[FLD.VOLUME_ADOSC_FLOW] > 0)).values)
    indices[FLD.VOLUME_FLOW_TRI_CROSS_SX] = Timeline_Integral(((indices[FLD.VOLUME_FLOW] < 0) & \
                                                        (indices[FLD.VOLUME_AD_FLOW] < 0) & \
                                                        (indices[FLD.VOLUME_ADOSC_FLOW] < 0)).values)
    indices[FLD.VOLUME_FLOW_TRI_CROSS] = np.where(indices[FLD.VOLUME_FLOW_TRI_CROSS_JX].values > 0, 1, 
                                                  np.where(indices[FLD.VOLUME_FLOW_TRI_CROSS_SX].values > 0, -1, 0))

    indices[FLD.VOL_MA5] = talib.MA(data[AKA.VOLUME], timeperiod=5)
    indices[FLD.VOL_MA10] = talib.MA(data[AKA.VOLUME], timeperiod=10)

    indices[FLD.VOL_CROSS] = np.where(indices[FLD.VOLUME_FLOW_TRI_CROSS] != 0, 
                                      indices[FLD.VOLUME_FLOW_TRI_CROSS],
                                      np.where(indices[FLD.VOL_MA5] > indices[FLD.VOL_MA10], 
                                               1, -1))
    indices[FLD.VOL_TREND_TIMING_LAG] = calc_event_timing_lag(indices[FLD.VOL_CROSS].values)

    # 新老版本变量交替
    features = indices
    features[FLD.VOLUME_AD_ZS21] = rolling_pctrank(features[FLD.VOLUME_AD].values, w=21)
    features[FLD.VOLUME_AD_ZS84] = rolling_pctrank(features[FLD.VOLUME_AD].values, w=84)
    features[FLD.VOLUME_ZS21] = rolling_pctrank(data[AKA.VOLUME].values, w=21)
    features[FLD.VOLUME_ZS84] = rolling_pctrank(data[AKA.VOLUME].values, w=84)
    #features[FLD.VOLUME_AD_ZS21] =
    #features[FLD.VOLUME_AD].rolling(21).apply(lambda x:
    #                                                                         pd.Series(x).rank(pct=True).iat[-1],
    #                                                        raw=True)
    #features[FLD.VOLUME_AD_ZS84] =
    #features[FLD.VOLUME_AD].rolling(84).apply(lambda x:
    #                                                                         pd.Series(x).rank(pct=True).iat[-1],
    #                                                        raw=True)
    #features[FLD.VOLUME_ZS21] = data[AKA.VOLUME].rolling(21).apply(lambda x:
    #                                                               pd.Series(x).rank(pct=True).iat[-1],
    #                                                        raw=True)
    #features[FLD.VOLUME_ZS84] = data[AKA.VOLUME].rolling(84).apply(lambda x:
    #                                                               pd.Series(x).rank(pct=True).iat[-1],
    #                                                        raw=True)
    features[FLD.VOLUME_FLOW] = features[FLD.VOL_MA5] - features[FLD.VOL_MA10]
    volume_flow_cross = np.where((features[FLD.VOL_MA5] - features[FLD.VOL_MA10]) > 0, 1, 
                                 np.where((features[FLD.VOL_TREND_TIMING_LAG] > 0) & \
                                          (features[FLD.DEA_ZERO_TIMING_LAG] > 0) & \
                                          ((features[FLD.VOLUME_ZS21] + features[FLD.VOLUME_ZS84]) > 0.618) & \
                                          (features[FLD.VOLUME_ZS21] > features[FLD.VOLUME_ZS21].shift(1)) & \
                                          (features[FLD.VOLUME_ZS84] > features[FLD.VOLUME_ZS84].shift(1)) & \
                                          ((features[FLD.VOLUME_AD_ZS21] + features[FLD.VOLUME_AD_ZS84]) > 0.618) & \
                                          (features[FLD.VOLUME_AD_ZS21] > features[FLD.VOLUME_AD_ZS21].shift(1)) & \
                                          (features[FLD.VOLUME_AD_ZS84] > features[FLD.VOLUME_AD_ZS84].shift(1)), 1, 
                                          np.where((features[FLD.VOL_TREND_TIMING_LAG] > 0) & \
                                          ((features[FLD.VOLUME_ZS21] + features[FLD.VOLUME_ZS84]) > 0.618) & \
                                          (features[FLD.VOLUME_ZS21] > features[FLD.VOLUME_ZS21].shift(2)) & \
                                          (features[FLD.VOLUME_ZS84] > features[FLD.VOLUME_ZS84].shift(2)) & \
                                          ((features[FLD.VOLUME_AD_ZS21] + features[FLD.VOLUME_AD_ZS84]) > 0.618) & \
                                          (features[FLD.VOLUME_AD_ZS21] > features[FLD.VOLUME_AD_ZS21].shift(2)) & \
                                          (features[FLD.VOLUME_AD_ZS84] > features[FLD.VOLUME_AD_ZS84].shift(2)), 1, -1)))
    volume_flow_cross = np.where((np.r_[0, volume_flow_cross[:-1]] > 0) & \
        (features[FLD.VOLUME_ZS21] > features[FLD.VOLUME_ZS21].shift(1)) & \
        (features[FLD.VOLUME_ZS84] > features[FLD.VOLUME_ZS84].shift(1)) & \
        (features[FLD.VOLUME_AD_ZS21] > features[FLD.VOLUME_AD_ZS21].shift(1)) & \
        (features[FLD.VOLUME_AD_ZS84] > features[FLD.VOLUME_AD_ZS84].shift(1)), 1, volume_flow_cross)                                     
    features[FLD.VOLUME_FLOW_TIMING_LAG] = calc_event_timing_lag(volume_flow_cross)
    volume_flow_cross = features[FLD.VOLUME_FLOW_TIMING_LAG] + features[FLD.VOL_TREND_TIMING_LAG]
    volume_flow_cross = np.where(volume_flow_cross > 0, 1,
                                np.where(volume_flow_cross < 0, -1, 0))
    features[FLD.VOLUME_FLOW_TIMING_LAG] = calc_event_timing_lag(volume_flow_cross)

    return features


def boll_cross_drawdown_func(data, *args, **kwargs):
    """
    准备计算 布林金叉死叉背离
    支持QA add_func，第二个参数 默认为 indices= 为已经计算指标
    理论上这个函数只计算单一标的，不要尝试传递复杂标的，indices会尝试拆分。
    """
    # 针对多标的，拆分 indices 数据再自动合并
    code = data.index.get_level_values(level=1)[0]
    if ('indices' in kwargs.keys()):
        indices = kwargs['indices'].loc[(slice(None), code), :]
    elif (len(args) > 0):
        indices = args[0].loc[(slice(None), code), :]
    else:
        print(u'Missing paramters: agrs[0] or kwargs[\'indices\']')
        indices = None

    return boll_cross_drawdown(data, indices)


def boll_cross_drawdown(data, indices):
    """
    尝试画出箱体震荡范围
    """
    if (ST.VERBOSE in data.columns):
        print('Phase calc_drawdown', QA_util_timestamp_to_str())
        
    indices[FLD.BOLL_CROSS_SX_DEA] = np.where(indices[FLD.BOLL_CROSS_SX_BEFORE] == 0, 
                                              indices[FLD.DEA], np.nan)
    indices[FLD.BOLL_CROSS_SX_DEA] = indices[FLD.BOLL_CROSS_SX_DEA].ffill()
    indices[FLD.BOLL_SX_DIF] = np.where(indices[FLD.BOLL_CROSS_SX_BEFORE] == 0, 
                                        indices[FLD.DIF], np.nan)
    indices[FLD.BOLL_SX_DIF] = indices[FLD.BOLL_SX_DIF].ffill()
    indices[FLD.BOLL_SX_MACD] = np.where(indices[FLD.BOLL_CROSS_SX_BEFORE] == 0, 
                                         indices[FLD.MACD], np.nan)
    indices[FLD.BOLL_SX_MACD] = indices[FLD.BOLL_SX_MACD].ffill()

    # 尝试画出箱体震荡范围
    indices[FLD.BOLL_CROSS_SX_ATR_UB] = np.where(indices[FLD.BOLL_CROSS_SX_BEFORE] == 0,
                                                    indices[FLD.ATR_UB], np.nan)
    indices[FLD.BOLL_CROSS_SX_ATR_UB] = indices[FLD.BOLL_CROSS_SX_ATR_UB].ffill()
    indices[FLD.BOLL_CROSS_SX_ATR_LB] = np.where(indices[FLD.BOLL_CROSS_SX_BEFORE] == 0, 
                                                indices[FLD.ATR_LB], np.nan)
    indices[FLD.BOLL_CROSS_SX_ATR_LB] = indices[FLD.BOLL_CROSS_SX_ATR_LB].ffill()

    indices[FLD.BOLL_CROSS_JX_ATR_UB] = np.where(indices[FLD.BOLL_CROSS_JX_BEFORE] == 0, 
                                                indices[FLD.ATR_UB], np.nan)
    indices[FLD.BOLL_CROSS_JX_ATR_UB] = indices[FLD.BOLL_CROSS_JX_ATR_UB].ffill()
    indices[FLD.BOLL_CROSS_JX_ATR_LB] = np.where(indices[FLD.BOLL_CROSS_JX_BEFORE] == 0, 
                                                indices[FLD.ATR_LB], np.nan)
    indices[FLD.BOLL_CROSS_JX_ATR_LB] = indices[FLD.BOLL_CROSS_JX_ATR_LB].ffill()

    if (FLD.COMBINE_DENSITY in indices.columns):
        indices[FLD.BOLL_SX_COMBO_DENSITY] = np.where(indices[FLD.BOLL_CROSS_SX_BEFORE] == 0, 
                                                        indices[FLD.COMBINE_DENSITY], np.nan)
        indices[FLD.BOLL_SX_COMBO_DENSITY] = indices[FLD.BOLL_SX_COMBO_DENSITY].ffill()

    indices[FLD.MACD_CROSS_SX_DEA] = np.where(indices[FLD.MACD_CROSS_SX_BEFORE] == 0, 
                                                indices[FLD.DEA], np.nan)
    indices[FLD.MACD_CROSS_SX_DEA] = indices[FLD.MACD_CROSS_SX_DEA].ffill()
    indices[FLD.BOLL_CROSS_SX_WIDTH] = np.where(indices[FLD.BOLL_CROSS_SX_BEFORE] == 0, 
                                                indices[FLD.BOLL_CHANNEL], np.nan)
    indices[FLD.BOLL_CROSS_SX_WIDTH] = indices[FLD.BOLL_CROSS_SX_WIDTH].ffill()
    indices[FLD.BOLL_CROSS_SX_RSI] = np.where(indices[FLD.BOLL_CROSS_SX_BEFORE] == 0, 
                                                np.maximum(indices[FLD.RSI],
                                                            indices[FLD.RSI].shift(1)),
                                                np.nan)
    indices[FLD.BOLL_CROSS_SX_RSI] = indices[FLD.BOLL_CROSS_SX_RSI].ffill()
    indices[FLD.DRAWDOWN_RSI] = indices[FLD.RSI] - indices[FLD.BOLL_CROSS_SX_RSI]
    indices[FLD.BOLL_CROSS_SX_HIGH] = np.where(indices[FLD.BOLL_CROSS_SX_BEFORE] == 0, 
                                                np.maximum(data[AKA.HIGH],
                                                           data[AKA.HIGH].shift(1)), 
                                                np.nan)
    indices[FLD.BOLL_CROSS_SX_HIGH] = indices[FLD.BOLL_CROSS_SX_HIGH].ffill()

    indices[FLD.DRAWDOWN_HIGH] = (data[AKA.CLOSE] - indices[FLD.BOLL_CROSS_SX_HIGH]) / data[AKA.CLOSE]

    indices[FLD.BOLL_SX_DRAWDOWN] = (data[AKA.CLOSE] - indices[FLD.BOLL_CROSS_SX_HIGH]) / data[AKA.CLOSE]

    indices[FLD.BOLL_CROSS_JX_WIDTH] = np.where(indices[FLD.BOLL_CROSS_JX_BEFORE] == 0, 
                                                indices[FLD.BOLL_CHANNEL], np.nan)
    indices[FLD.BOLL_CROSS_JX_WIDTH] = indices[FLD.BOLL_CROSS_JX_WIDTH].ffill()
    indices[FLD.BOLL_JX_RSI] = np.where(indices[FLD.BOLL_CROSS_JX_BEFORE] == 0, 
                                              indices[FLD.RSI], np.nan)
    indices[FLD.BOLL_JX_RSI] = indices[FLD.BOLL_JX_RSI].ffill()
    indices[FLD.BOLL_JX_MAXFACTOR] = np.where(indices[FLD.BOLL_CROSS_JX_BEFORE] == 0, 
                                              indices[FLD.MAXFACTOR], np.nan)
    indices[FLD.BOLL_JX_MAXFACTOR] = indices[FLD.BOLL_JX_MAXFACTOR].ffill()
    indices[FLD.BOLL_CROSS_JX_DEA] = np.where(indices[FLD.BOLL_CROSS_JX_BEFORE] == 0, 
                                              indices[FLD.DEA], np.nan)
    indices[FLD.BOLL_CROSS_JX_DEA] = indices[FLD.BOLL_CROSS_JX_DEA].ffill()
    indices[FLD.BOLL_JX_DRAWDOWN_RATIO] = ((indices[FLD.BOLL_CROSS_JX_ATR_LB] - \
        indices[FLD.BOLL_CROSS_SX_HIGH]) / indices[FLD.BOLL_CROSS_JX_ATR_LB]) / \
        (indices[FLD.BOLL_CROSS_SX_BEFORE] - indices[FLD.BOLL_CROSS_JX_BEFORE])

    indices[FLD.ATR_CROSS_SX_HIGH] = np.where(indices[FLD.ATR_CROSS_SX_BEFORE] == 0, 
                                              np.maximum(data[AKA.HIGH],
                                                         data[AKA.HIGH].shift(1)), 
                                              np.nan)
    indices[FLD.ATR_CROSS_SX_HIGH] = indices[FLD.ATR_CROSS_SX_HIGH].ffill()
    indices[FLD.ATR_JX_DRAWDOWN_RATIO] = ((indices[FLD.BOLL_CROSS_JX_ATR_LB] - \
        indices[FLD.ATR_CROSS_SX_HIGH]) / indices[FLD.BOLL_CROSS_JX_ATR_LB]) / \
        (indices[FLD.ATR_CROSS_SX_BEFORE] - indices[FLD.BOLL_CROSS_JX_BEFORE])

    features = indices
    features[FLD.BOLL_JX_CLOSE] = np.where(features[FLD.BOLL_CROSS_JX_BEFORE] == 0, 
                                                np.maximum(data[AKA.CLOSE],
                                                           data[AKA.CLOSE].shift(1)), 
                                                np.nan)
    features[FLD.BOLL_JX_CLOSE] = features[FLD.BOLL_JX_CLOSE].ffill()
    features[FLD.BOLL_JX_RAISED] = (data[AKA.CLOSE] - features[FLD.BOLL_JX_CLOSE]) / data[AKA.CLOSE]
    features[FLD.BOLL_RAISED_TIMING_LAG] = calc_event_timing_lag(np.where((features[FLD.BOLL_CROSS_JX_BEFORE] < features[FLD.BOLL_CROSS_SX_BEFORE]) & \
                                                                          (features[FLD.BOLL_JX_RAISED] > 0), 1,               
                                                                          np.where(features[FLD.BOLL_JX_RAISED] > 0, 1, -1)))

    # 补充趋势信息，补全布林带不确定趋势判断
    features[FLD.BOLL_CROSS] = np.where(features[FLD.BOLL_CROSS] == 0,
                                        np.where(features[FLD.BOLL_JX_RAISED] > 0, 1,
                                                 np.where(features[FLD.BOLL_JX_RAISED] < 0, -1, 0)),
                                                 features[FLD.BOLL_CROSS])
    boll_cross_jx = Timeline_Integral(np.where(features[FLD.BOLL_CROSS] >= 0, 1, 0))
    boll_cross_sx = np.sign(features[FLD.BOLL_CROSS]) * Timeline_Integral(np.where(features[FLD.BOLL_CROSS] < 0, 1, 0))
    features[FLD.BOLL_TREND_TIMING_LAG] = boll_cross_jx + boll_cross_sx

    return features


def boll_cross_drawdown_v2(data, features):
    """
    尝试画出箱体震荡范围
    """
    if (ST.VERBOSE in data.columns):
        print('Phase boll_cross_drawdown_v2', QA_util_timestamp_to_str())
        
    features[FLD.BOLL_CROSS_SX_DEA] = np.where(features[FLD.BOLL_CROSS_SX_BEFORE] == 0, 
                                               features[FLD.DEA], np.nan)
    features[FLD.BOLL_CROSS_SX_DEA] = features[FLD.BOLL_CROSS_SX_DEA].ffill()
    features[FLD.BOLL_SX_DIF] = np.where(features[FLD.BOLL_CROSS_SX_BEFORE] == 0, 
                                         features[FLD.DIF], np.nan)
    features[FLD.BOLL_SX_DIF] = features[FLD.BOLL_SX_DIF].ffill()
    features[FLD.BOLL_SX_MACD] = np.where(features[FLD.BOLL_CROSS_SX_BEFORE] == 0, 
                                          features[FLD.MACD], np.nan)
    features[FLD.BOLL_SX_MACD] = features[FLD.BOLL_SX_MACD].ffill()

    features[FLD.BOLL_CROSS_SX_RSI] = np.where(features[FLD.BOLL_CROSS_SX_BEFORE] == 0, 
                                                np.maximum(features[FLD.RSI],
                                                           features[FLD.RSI].shift(1)),
                                                np.nan)
    features[FLD.BOLL_CROSS_SX_RSI] = features[FLD.BOLL_CROSS_SX_RSI].ffill()

    features[FLD.DRAWDOWN_RSI] = features[FLD.RSI] - features[FLD.BOLL_CROSS_SX_RSI]

    features[FLD.BOLL_CROSS_SX_HIGH] = np.where(features[FLD.BOLL_CROSS_SX_BEFORE] == 0, 
                                                np.maximum(data[AKA.HIGH],
                                                           data[AKA.HIGH].shift(1)), 
                                                np.nan)
    features[FLD.BOLL_CROSS_SX_HIGH] = features[FLD.BOLL_CROSS_SX_HIGH].ffill()

    features[FLD.DRAWDOWN_HIGH] = (data[AKA.CLOSE] - features[FLD.BOLL_CROSS_SX_HIGH]) / data[AKA.CLOSE]

    features[FLD.BOLL_SX_DRAWDOWN] = (data[AKA.CLOSE] - features[FLD.BOLL_CROSS_SX_HIGH]) / data[AKA.CLOSE]

    features[FLD.BOLL_CROSS_JX_WIDTH] = np.where(features[FLD.BOLL_CROSS_JX_BEFORE] == 0, 
                                                features[FLD.BOLL_CHANNEL], np.nan)
    features[FLD.BOLL_CROSS_JX_WIDTH] = features[FLD.BOLL_CROSS_JX_WIDTH].ffill()
    features[FLD.BOLL_JX_RSI] = np.where(features[FLD.BOLL_CROSS_JX_BEFORE] == 0, 
                                         features[FLD.RSI], np.nan)
    features[FLD.BOLL_JX_RSI] = features[FLD.BOLL_JX_RSI].ffill()
    features[FLD.BOLL_JX_MAXFACTOR] = np.where(features[FLD.BOLL_CROSS_JX_BEFORE] == 0, 
                                               features[FLD.MAXFACTOR], np.nan)
    features[FLD.BOLL_JX_MAXFACTOR] = features[FLD.BOLL_JX_MAXFACTOR].ffill()
    features[FLD.BOLL_JX_MAPOWER30] = np.where(features[FLD.BOLL_CROSS_JX_BEFORE] == 0, 
                                               features[FLD.MAPOWER30], np.nan)
    features[FLD.BOLL_JX_MAPOWER30] = features[FLD.BOLL_JX_MAPOWER30].ffill()
    features[FLD.BOLL_JX_MAPOWER120] = np.where(features[FLD.BOLL_CROSS_JX_BEFORE] == 0, 
                                               features[FLD.MAPOWER120], np.nan)
    features[FLD.BOLL_JX_MAPOWER120] = features[FLD.BOLL_JX_MAPOWER120].ffill()
    features[FLD.BOLL_JX_HMAPOWER120] = np.where(features[FLD.BOLL_CROSS_JX_BEFORE] == 0, 
                                               features[FLD.HMAPOWER120], np.nan)
    features[FLD.BOLL_JX_HMAPOWER120] = features[FLD.BOLL_JX_HMAPOWER120].ffill()
    features[FLD.BOLL_CROSS_JX_DEA] = np.where(features[FLD.BOLL_CROSS_JX_BEFORE] == 0, 
                                              features[FLD.DEA], np.nan)
    features[FLD.BOLL_CROSS_JX_DEA] = features[FLD.BOLL_CROSS_JX_DEA].ffill()

    features[FLD.BOLL_JX_CLOSE] = np.where(features[FLD.BOLL_CROSS_JX_BEFORE] == 0, 
                                                np.maximum(data[AKA.CLOSE],
                                                           data[AKA.CLOSE].shift(1)), 
                                                np.nan)
    features[FLD.BOLL_JX_CLOSE] = features[FLD.BOLL_JX_CLOSE].ffill()
    features[FLD.BOLL_JX_RAISED] = (data[AKA.CLOSE] - features[FLD.BOLL_JX_CLOSE]) / data[AKA.CLOSE]
    features[FLD.BOLL_RAISED_TIMING_LAG] = calc_event_timing_lag(np.where((features[FLD.BOLL_CROSS_JX_BEFORE] < features[FLD.BOLL_CROSS_SX_BEFORE]) & \
                                                                          (features[FLD.BOLL_JX_RAISED] > 0), 1,               
                                                                          np.where(features[FLD.BOLL_JX_RAISED] > 0, 1, -1)))

    # 补充趋势信息，补全布林带不确定趋势判断
    features[FLD.BOLL_CROSS] = np.where(features[FLD.BOLL_CROSS] == 0,
                                        np.where(features[FLD.BOLL_JX_RAISED] > 0, 1,
                                                 np.where(features[FLD.BOLL_JX_RAISED] < 0, -1, 0)),
                                                 features[FLD.BOLL_CROSS])
    features[FLD.BOLL_TREND_TIMING_LAG] = calc_event_timing_lag(features[FLD.BOLL_CROSS])

    return features

def macd_trend_predict_func(data, *args, **kwargs):
    """
    准备计算 Combo Flow 趋势方向
    支持QA add_func，第二个参数 默认为 indices= 为已经计算指标
    理论上这个函数只计算单一标的，不要尝试传递复杂标的，indices会尝试拆分。
    """
    # 针对多标的，拆分 indices 数据再自动合并
    code = data.index.get_level_values(level=1)[0]
    if ('indices' in kwargs.keys()):
        indices = kwargs['indices'].loc[(slice(None), code), :]
    elif (len(args) > 0):
        indices = args[0].loc[(slice(None), code), :]
    else:
        print(u'Missing paramters: agrs[0] or kwargs[\'indices\']')
        indices = None

    return macd_trend_predict(data, indices)


def macd_trend_predict(data, indices):
    """
    MACD 下降趋势（趋势不一定代表价格马上体现）
    """
    macd_trend = np.where(indices[FLD.MACD_CROSS_SX_BEFORE] == 0, -1,
                          np.where(indices[FLD.MACD_CROSS_JX_BEFORE] == 0, 1, 0))

    # 比MACD更先进的指标辅助判断规则
    macd_trend = np.where((indices[FLD.DEA] > 0) & \
                          (indices[FLD.MACD_CROSS_JX_BEFORE] > indices[FLD.BOLL_CROSS_SX_BEFORE]) & \
                          ((indices[FLD.MACD] > 0) | \
                          (indices[FLD.MACD_CROSS_SX_BEFORE] < 4)) & \
                          (indices[FLD.LINEAREG_CROSS_SX_BEFORE] < indices[FLD.LINEAREG_CROSS_JX_BEFORE]) & \
                          (indices[FLD.MAXFACTOR_CROSS] < 0) & \
                          ((indices[FLD.DUAL_CROSS_JX] < 0) | \
                          (indices[FLD.VOLUME_FLOW_TRI_CROSS] < 0)),
                          -1, macd_trend)
    macd_trend = np.where((indices[FLD.ZEN_WAVELET_CROSS_SX_BEFORE] < indices[FLD.ZEN_WAVELET_CROSS_JX_BEFORE]) & \
                          (indices[FLD.LINEAREG_CROSS_SX_BEFORE] < indices[FLD.LINEAREG_CROSS_JX_BEFORE]) & \
                          ((indices[FLD.ZEN_WAVELET_CROSS_SX_BEFORE] <= 1) | \
                          ((indices[FLD.RENKO_TREND_S] + indices[FLD.RENKO_TREND_L]) < 0) | \
                          (indices[FLD.DEA] < 0) | \
                          (np.r_[0, macd_trend[:-1]] < 0)) & \
                          (indices[FLD.MACD_DELTA] < 0),
                          -1, macd_trend)
    macd_trend = np.where((indices[FLD.BOLL_UB].shift(1) > indices[FLD.BOLL_UB]) & \
                          ((indices[FLD.ZEN_WAVELET_CROSS_SX_BEFORE] <= 1) | \
                          ((indices[FLD.RENKO_TREND_S] + indices[FLD.RENKO_TREND_L]) < 0) | \
                          (indices[FLD.DEA] < 0) | \
                          (np.r_[0, macd_trend[:-1]] < 0)) & \
                          (indices[FLD.MACD_DELTA] < 0),
                          -1, macd_trend)
    macd_trend = np.where((indices[FLD.ZEN_WAVELET_CROSS_SX_BEFORE] < indices[FLD.ZEN_WAVELET_CROSS_JX_BEFORE]) & \
                          (indices[FLD.LINEAREG_CROSS_SX_BEFORE] < indices[FLD.LINEAREG_CROSS_JX_BEFORE]) & \
                          (indices[FLD.RENKO_TREND_S] < 0) & \
                          (indices[FLD.RENKO_TREND_L] < 0) & \
                          (indices[FLD.MACD_DELTA] < 0), -1, macd_trend)
    macd_trend = np.where((indices[FLD.ZEN_WAVELET_CROSS_SX_BEFORE] < indices[FLD.ZEN_WAVELET_CROSS_JX_BEFORE]) & \
                          (indices[FLD.LINEAREG_CROSS_SX_BEFORE] < indices[FLD.LINEAREG_CROSS_JX_BEFORE]) & \
                          (indices[FLD.RENKO_TREND_S] < 0) & \
                          (indices[FLD.RENKO_TREND_L] < 0) & \
                          (indices[FLD.DEA] < 0), -1, macd_trend)
    macd_trend = np.where((indices[FLD.DIF] < indices[FLD.DEA]) & \
                          (indices[FLD.DIF] < indices[FLD.DIF].shift(1)) & \
                          (indices[FLD.MACD_DELTA] < 0), -1, macd_trend)
    macd_trend = np.where((indices[FLD.DIF] < indices[FLD.DEA]) & \
                          (indices[FLD.DEA] < 0) & \
                          (indices[FLD.DIF] < indices[FLD.DIF].shift(1)) & \
                          (indices[FLD.MACD_DELTA] < 0), -1, macd_trend)

    macd_trend = np.where((indices[FLD.ZEN_WAVELET_CROSS_JX_BEFORE] < indices[FLD.ZEN_WAVELET_CROSS_SX_BEFORE]) & \
                          (indices[FLD.LINEAREG_CROSS_JX_BEFORE] < indices[FLD.LINEAREG_CROSS_SX_BEFORE]) & \
                          ((indices[FLD.ZEN_WAVELET_CROSS_SX_BEFORE] > 4) | \
                          (np.r_[0, macd_trend[:-1]] > 0)) & \
                          (indices[FLD.DEA] > 0) & \
                          (indices[FLD.MACD_DELTA] > 0), 1, macd_trend)
    macd_trend = np.where((indices[FLD.ZEN_WAVELET_CROSS_JX_BEFORE] < indices[FLD.ZEN_WAVELET_CROSS_SX_BEFORE]) & \
                          (indices[FLD.LINEAREG_CROSS_JX_BEFORE] < indices[FLD.LINEAREG_CROSS_SX_BEFORE]) & \
                          ((indices[FLD.ZEN_WAVELET_CROSS_SX_BEFORE] > 4) | \
                          (np.r_[0, macd_trend[:-1]] > 0)) & \
                          (indices[FLD.RENKO_TREND_S] > 0) & \
                          (indices[FLD.RENKO_TREND_L] > 0) & \
                          (indices[FLD.MACD_DELTA] > 0), 1, macd_trend)
    macd_trend = np.where((indices[FLD.ZEN_WAVELET_CROSS_JX_BEFORE] < indices[FLD.ZEN_WAVELET_CROSS_SX_BEFORE]) & \
                          (indices[FLD.LINEAREG_CROSS_JX_BEFORE] < indices[FLD.LINEAREG_CROSS_SX_BEFORE]) & \
                          ((indices[FLD.ZEN_WAVELET_CROSS_SX_BEFORE] > 4) | \
                          (np.r_[0, macd_trend[:-1]] > 0)) & \
                          (indices[FLD.RENKO_TREND_S] > 0) & \
                          (indices[FLD.RENKO_TREND_L] > 0) & \
                          (indices[FLD.DEA] > 0), 1, macd_trend)
    macd_trend = np.where((indices[FLD.MA90] < indices[FLD.MA120]) & \
                          (indices[FLD.MA30] < indices[FLD.MA90]) & \
                          ((indices[FLD.LINEAREG_PRICE] < indices[FLD.MA30]) | \
                          ((indices[FLD.LINEAREG_PRICE] < indices[FLD.MA90]) & \
                          (indices[FLD.COMBINE_DENSITY_SMA] < 0.512))) & \
                          (indices[FLD.ZEN_WAVELET_CROSS_SX_BEFORE] < indices[FLD.ZEN_WAVELET_CROSS_JX_BEFORE]), 
                          -1, macd_trend)
    macd_trend = np.where((indices[FLD.MA90] < indices[FLD.MA120]) & \
                          (indices[FLD.MA30] < indices[FLD.MA90]) & \
                          ((indices[FLD.DIF] + indices[FLD.MACD] * 2) < 0) & \
                          (indices[FLD.MA90_CLEARANCE] < 0.512) & \
                          ((indices[FLD.LINEAREG_PRICE] < indices[FLD.MA30]) | \
                          ((indices[FLD.LINEAREG_PRICE] < indices[FLD.MA90]) & \
                          (indices[FLD.COMBINE_DENSITY_SMA] < 0.512))) & \
                          (indices[FLD.MA30].shift(1) > indices[FLD.MA30]) & \
                          ((indices[FLD.LINEAREG_PRICE].shift(1) > indices[FLD.LINEAREG_PRICE]) | \
                          (indices[FLD.BOLL_UB].shift(1) > indices[FLD.BOLL_UB])) & \
                          ((indices[FLD.ZSCORE_21].shift(1) > indices[FLD.ZSCORE_21]) | \
                          (indices[FLD.MA90_CLEARANCE] < -0.382) | \
                          (np.r_[0, macd_trend[:-1]] < 0)), 
                          -1, macd_trend)
    macd_trend = np.where((indices[FLD.MA90] < indices[FLD.MA120]) & \
                          (indices[FLD.MA30] < indices[FLD.MA90]) & \
                          (indices[FLD.MA90_CLEARANCE] < 0.512) & \
                          ((indices[FLD.LINEAREG_PRICE] < indices[FLD.MA30]) | \
                          ((indices[FLD.LINEAREG_PRICE] < indices[FLD.MA90]) & \
                          (indices[FLD.COMBINE_DENSITY_SMA] < 0.512))) & \
                          ((indices[FLD.ZSCORE_21].shift(1) > indices[FLD.ZSCORE_21]) | \
                          (indices[FLD.MA90_CLEARANCE] < -0.382) | \
                          (np.r_[0, macd_trend[:-1]] < 0)) & \
                          (indices[FLD.BOLL_CHANNEL] < 0.0927), 
                          -1, macd_trend)
    macd_trend = np.where((indices[FLD.MA90] < indices[FLD.MA120]) & \
                          (indices[FLD.LINEAREG_PRICE] < indices[FLD.MA120]) & \
                          (indices[FLD.MA30] < indices[FLD.MA120]) & \
                          (indices[FLD.MACD] < 0) & \
                          ((indices[FLD.ZSCORE_21].shift(1) > indices[FLD.ZSCORE_21]) | \
                          ((indices[FLD.BOLL_DELTA] < 0) & \
                          (indices[FLD.DEA] > 0)) | \
                          (indices[FLD.MA90_CLEARANCE] < -0.382) | \
                          (np.r_[0, macd_trend[:-1]] < 0)) & \
                          ((indices[FLD.BOLL_CHANNEL] < 0.0927) | \
                          (indices[FLD.BOLL_DELTA] < 0)), 
                          -1, macd_trend)
    macd_trend = np.where((macd_trend == 0) & \
                          (indices[FLD.DIF] > indices[FLD.DEA]) & \
                          (indices[FLD.DEA] > 0) & \
                          (indices[FLD.DIF] > indices[FLD.DIF].shift(1)) & \
                          (indices[FLD.MACD_DELTA] > 0), 
                          1, macd_trend)

    macd_trend = np.where((macd_trend == 0) & \
                          ((indices[FLD.MACD] < 0) | \
                          (indices[FLD.DEA] < 0)) & \
                          (indices[FLD.ZEN_WAVELET_CROSS_SX_BEFORE] < indices[FLD.ZEN_WAVELET_CROSS_JX_BEFORE]), 
                          -1, macd_trend)

    macd_trend = np.where((macd_trend == 0) & \
                          (np.r_[0,macd_trend[:-1]] < 0) & \
                          (indices[FLD.ZEN_WAVELET_CROSS_SX_BEFORE] < indices[FLD.ZEN_WAVELET_CROSS_JX_BEFORE]), 
                          -1, macd_trend)
    
    macd_trend = np.where((macd_trend == 0) & \
                          (indices[FLD.ZEN_WAVELET_CROSS_JX_BEFORE] < indices[FLD.ZEN_WAVELET_CROSS_SX_BEFORE]) & \
                          (indices[FLD.BOLL_CROSS_JX_BEFORE] < indices[FLD.BOLL_CROSS_SX_BEFORE]) & \
                          (indices[FLD.MACD_DELTA] > 0), 1, macd_trend)

    macd_trend = np.where((macd_trend == 0) & \
                          (indices[FLD.ZEN_WAVELET_CROSS_JX_BEFORE] < indices[FLD.ZEN_WAVELET_CROSS_SX_BEFORE]) & \
                          (indices[FLD.DEA] > 0) & \
                          (indices[FLD.MA120] < indices[FLD.MA90]) & \
                          (indices[FLD.BOLL] < indices[FLD.ATR_LB]), 1, macd_trend)

    macd_trend = np.where((macd_trend == 0) & \
                          (indices[FLD.COMBO_FLOW] < 0) & \
                          (indices[FLD.COMBINE_TIDE_CROSS_SX] > 0) & \
                          (indices[FLD.MACD] < 0), -1, macd_trend)

    macd_trend = np.where((macd_trend == 0) & \
                          (indices[FLD.MACD] < 0) & \
                          (indices[FLD.DEA] > 0) & \
                          ((indices[FLD.BOLL_CROSS_SX_BEFORE] > indices[FLD.MACD_CROSS_SX_BEFORE]) | \
                          (indices[FLD.RENKO_TREND_S] < 0)) & \
                          ((indices[FLD.COMBINE_DENSITY_SMA] < 0.512) | \
                          (indices[FLD.COMBO_FLOW] < 0)) & \
                          (indices[FLD.ZEN_WAVELET_CROSS_SX_BEFORE] < indices[FLD.ZEN_WAVELET_CROSS_JX_BEFORE]), 
                          -1, macd_trend)

    macd_super_trend = (indices[FLD.DUAL_CROSS_JX] > 0) & (indices[FLD.MACD] > 0) & \
        (indices[FLD.DEA] > 0) & \
        (indices[FLD.HIGHER_SETTLE_PRICE] == True)

    indices[FLD.MACD_TREND] = np.where(macd_super_trend == True, 1, macd_trend)

    return indices 
        

def combo_flow_cross_func(data, *args, **kwargs):
    """
    准备计算 Combo Flow 趋势方向
    支持QA add_func，第二个参数 默认为 indices= 为已经计算指标
    理论上这个函数只计算单一标的，不要尝试传递复杂标的，indices会尝试拆分。
    """
    # 针对多标的，拆分 indices 数据再自动合并
    code = data.index.get_level_values(level=1)[0]
    if ('indices' in kwargs.keys()):
        indices = kwargs['indices'].loc[(slice(None), code), :]
    elif (len(args) > 0):
        indices = args[0].loc[(slice(None), code), :]
    else:
        print(u'Missing paramters: agrs[0] or kwargs[\'indices\']')
        indices = None

    return combo_flow_cross(data, indices)


def combo_flow_cross(data, indices):
    """
    计算 Combo Flow 趋势方向
    """
    COMBINE_DENSITY = indices[FLD.ZEN_TIDE_DENSITY] + \
            indices[FLD.ATR_SuperTrend_DENSITY] + \
            indices[FLD.ATR_Stopline_DENSITY] + \
            indices[FLD.MA30_CROSS_DENSITY]
    indices[FLD.COMBINE_DENSITY] = COMBINE_DENSITY / 4
    try:
        indices[FLD.COMBINE_DENSITY_SMA] = talib.SMA(indices[FLD.COMBINE_DENSITY], 21)
        indices[FLD.COMBINE_DENSITY_SLOPE] = talib.LINEARREG_SLOPE(COMBINE_DENSITY, timeperiod=14)
        indices[FLD.COMBINE_DENSITY_REGRESSION] = talib.LINEARREG(COMBINE_DENSITY, timeperiod=20)
    except:
        if (ST.VERBOSE in data.columns):
            print(u'{} 交易数据不足({})，新股或者长期停牌，请检查。'.format(indices.index.get_level_values(level=1)[0], len(indices)))
        indices[FLD.COMBINE_DENSITY_SMA] = np.full((len(indices),), np.nan)
        indices[FLD.COMBINE_DENSITY_SLOPE] = np.full((len(indices),), np.nan)
        indices[FLD.COMBINE_DENSITY_REGRESSION] = np.full((len(indices),), np.nan)

    indices[FLD.COMBO_ZSCORE] = rolling_pctrank(indices[FLD.COMBINE_DENSITY].values, 84)
    indices[FLD.COMBO_FLOW] = (indices[FLD.COMBINE_DENSITY] - indices[FLD.COMBINE_DENSITY_SMA])
    combo_flow = np.where(indices[FLD.COMBO_FLOW] > 0, 1, 
                          np.where(indices[FLD.COMBO_FLOW] < 0, -1, 0))
    combo_flow_dif = np.nan_to_num(np.r_[0, np.diff(combo_flow)], nan=0)
    indices[FLD.COMBO_FLOW_JX_BEFORE] = Timeline_duration(np.where((combo_flow_dif > 0), 1, 0))
    indices[FLD.COMBO_FLOW_SX_BEFORE] = Timeline_duration(np.where((combo_flow_dif < 0), 1, 0))
    combo_flow_cross = np.where((indices[FLD.COMBO_FLOW_SX_BEFORE] - indices[FLD.COMBO_FLOW_JX_BEFORE]) > 0, 1,
                                np.where((indices[FLD.COMBO_FLOW_SX_BEFORE] - indices[FLD.COMBO_FLOW_JX_BEFORE]) < 0, -1, 0))
    indices[FLD.COMBO_FLOW_TIMING_LAG] = calc_event_timing_lag(combo_flow_cross)
        
    COMBINE_DENSITY_RETURNS = COMBINE_DENSITY.pct_change()
    COMBINE_TIDE_DENSITY_Uptrend = np.where(COMBINE_DENSITY_RETURNS > 0, 1, 0)
    indices[FLD.COMBINE_TIDE_CROSS_JX] = Timeline_Integral(COMBINE_TIDE_DENSITY_Uptrend)
    COMBINE_TIDE_DENSITY_Downtrend = np.where(COMBINE_DENSITY_RETURNS < 0, 1, 0)
    indices[FLD.COMBINE_TIDE_CROSS_SX] = Timeline_Integral(COMBINE_TIDE_DENSITY_Downtrend)

    COMBINE_TIDE_JX = (indices[FLD.DEA] > 0) & \
        (indices[FLD.COMBINE_TIDE_CROSS_JX] > 0) & \
        (indices[FLD.COMBINE_TIDE_CROSS_JX] > indices[FLD.DEA_CROSS_JX_BEFORE])
    indices[FLD.COMBINE_DUAL_TIDE_JX] = COMBINE_TIDE_JX

    COMBINE_TIDE_SX = (indices[FLD.DEA] < 0) & \
        (indices[FLD.COMBINE_TIDE_CROSS_SX] > 0) & \
        (indices[FLD.COMBINE_TIDE_CROSS_SX] > indices[FLD.DEA_CROSS_SX_BEFORE])
    indices[FLD.COMBINE_DUAL_TIDE_SX] = COMBINE_TIDE_SX

    return indices


def combo_flow_cross_v2(data, indices):
    """
    计算 Combo Flow 趋势方向
    """
    COMBINE_DENSITY = indices[FLD.ZEN_TIDE_DENSITY] + \
            indices[FLD.ATR_SuperTrend_DENSITY] + \
            indices[FLD.ATR_Stopline_DENSITY] + \
            indices[FLD.MA30_CROSS_DENSITY]
    indices[FLD.COMBINE_DENSITY] = COMBINE_DENSITY / 4
    try:
        indices[FLD.COMBINE_DENSITY_SMA] = talib.SMA(indices[FLD.COMBINE_DENSITY], 21)
        indices[FLD.COMBINE_DENSITY_SLOPE] = talib.LINEARREG_SLOPE(COMBINE_DENSITY, timeperiod=14)
        indices[FLD.COMBINE_DENSITY_REGRESSION] = talib.LINEARREG(COMBINE_DENSITY, timeperiod=20)
    except:
        if (ST.VERBOSE in data.columns):
            print(u'{} 交易数据不足({})，新股或者长期停牌，请检查。'.format(indices.index.get_level_values(level=1)[0], len(indices)))
        indices[FLD.COMBINE_DENSITY_SMA] = np.full((len(indices),), np.nan)
        indices[FLD.COMBINE_DENSITY_SLOPE] = np.full((len(indices),), np.nan)
        indices[FLD.COMBINE_DENSITY_REGRESSION] = np.full((len(indices),), np.nan)

    indices[FLD.COMBO_FLOW] = (indices[FLD.COMBINE_DENSITY] - indices[FLD.COMBINE_DENSITY_SMA])
    combo_flow = np.where(indices[FLD.COMBO_FLOW] > 0, 1, 
                          np.where(indices[FLD.COMBO_FLOW] < 0, -1, 0))
    combo_flow_dif = np.nan_to_num(np.r_[0, np.diff(combo_flow)], nan=0)
    indices[FLD.COMBO_FLOW_JX_BEFORE] = Timeline_duration(np.where((combo_flow_dif > 0), 1, 0))
    indices[FLD.COMBO_FLOW_SX_BEFORE] = Timeline_duration(np.where((combo_flow_dif < 0), 1, 0))
    combo_flow_cross = np.where((indices[FLD.COMBO_FLOW_SX_BEFORE] - indices[FLD.COMBO_FLOW_JX_BEFORE]) > 0, 1,
                                np.where((indices[FLD.COMBO_FLOW_SX_BEFORE] - indices[FLD.COMBO_FLOW_JX_BEFORE]) < 0, -1, 0))
    indices[FLD.COMBO_FLOW_TIMING_LAG] = calc_event_timing_lag(combo_flow_cross)
        
    return indices

def risk_free_baseline_func(data, *args, **kwargs):
    """
    生成一条年利率 4% 的模拟货币基金基准线
    支持QA add_func，第二个参数 默认为 indices= 为已经计算指标
    理论上这个函数只计算单一标的，不要尝试传递复杂标的，indices会尝试拆分。
    """
    risk_free, lag, w = 0.04, 365, 63
    indices = None
    random = False    # 模拟随机波动，备用

    # 针对多标的，拆分 indices 数据再自动合并
    code = data.index.get_level_values(level=1)[0]
    if (len(kwargs.keys()) > 0):
        risk_free = kwargs['risk_free'] if ('risk_free' in kwargs.keys()) else risk_free
        frequency = kwargs['frequency'] if ('frequency' in kwargs.keys()) else None
        if (frequency == 'day'):
            lag = 365             # 日线一年
        elif (frequency == 'hour'):
            lag = 8760            # 小时线一年
        else:
            lag = kwargs['lag'] if ('lag' in kwargs.keys()) else lag
        indices = kwargs['indices'].loc[(slice(None), code), :] if ('indices' in kwargs.keys()) else indices

    if (len(args) > 0):
        risk_free = args[0] if (len(args) > 0) else risk_free
        lag = args[1] if (len(args) > 1) else lag
        indices = args[2].loc[(slice(None), code), :] if (len(args) >= 2) else indices

    indices[FLD.BASELINE] = np.linspace(1, pow((1 + risk_free), 
                                               (len(data) / lag)), 
                                        len(data))
    
    indices[FLD.BASELINE_ROC] = talib.ROC(indices[FLD.BASELINE].values, w) / 100
    indices[FLD.BASELINE_SLOPE] = talib.LINEARREG_SLOPE(indices[FLD.BASELINE].values, timeperiod=14)
    indices[FLD.BASELINE_ROC] = indices[FLD.BASELINE_ROC].bfill()
    indices[FLD.BASELINE_SLOPE] = indices[FLD.BASELINE_SLOPE].bfill()

    return indices


def bias_cross_func(data, *args, **kwargs):
    """
    生成一条年利率 4% 的模拟货币基金基准线
    支持QA add_func，第二个参数 默认为 indices= 为已经计算指标
    理论上这个函数只计算单一标的，不要尝试传递复杂标的，indices会尝试拆分。
    """
    if (ST.VERBOSE in data.columns):
        print('Phase bias_cross_func', QA_util_timestamp_to_str())

    # 针对多标的，拆分 indices 数据再自动合并
    code = data.index.get_level_values(level=1)[0]
    if (len(kwargs.keys()) > 0):
        indices = kwargs['indices'].loc[(slice(None), code), :] if ('indices' in kwargs.keys()) else None

    bias_zscore_21 = rolling_pctrank(indices[FLD.BIAS3].values, w=21)
    #bias_zscore_21 = indices[FLD.BIAS3].rolling(21).apply(lambda x:
    #                                                      pd.Series(x).rank(pct=True).iat[-1],
    #                                                      raw=True)

    if (ST.VERBOSE in data.columns):
        print('Phase bias_cross_func Phase 1', QA_util_timestamp_to_str())

    indices[FLD.BIAS3_ZSCORE] = bias_zscore_21
    indices[FLD.BIAS3_CROSS] = CROSS(indices[FLD.BIAS3], 0)
    indices[FLD.BIAS3_CROSS] = np.where(CROSS(0, indices[FLD.BIAS3]) == 1, -1, 
                                        indices[FLD.BIAS3_CROSS])
    indices[FLD.BIAS3_CROSS_JX_BEFORE] = Timeline_duration(np.where(indices[FLD.BIAS3_CROSS] == 1, 1, 0))
    indices[FLD.BIAS3_CROSS_SX_BEFORE] = Timeline_duration(np.where(indices[FLD.BIAS3_CROSS] == -1, 1, 0))

    if (ST.VERBOSE in data.columns):
        print('Phase bias_cross_func Phase 2', QA_util_timestamp_to_str())

    bias = QA.QA_indicator_BIAS(data, 6, 12, 24)
    bias_cross = np.where((bias['BIAS3'] > 0) & \
                          (bias['BIAS2'] > 0) & \
                          (bias['BIAS1'] > 0), 1, 
                          np.where((bias['BIAS3'] < 0) & \
                                   (bias['BIAS2'] < 0) & \
                                   (bias['BIAS1'] < 0), -1, 0))
    indices[FLD.BIAS_TREND_TIMING_LAG] = calc_event_timing_lag(bias_cross)
    indices[FLD.BIAS3_TREND_TIMING_LAG] = indices[FLD.BIAS3_CROSS_SX_BEFORE] - indices[FLD.BIAS3_CROSS_JX_BEFORE] 

    bias3_delta = indices[FLD.BIAS3] - indices[FLD.BIAS3].shift(2)
    bias3_cross = np.where(indices[FLD.BIAS3_TREND_TIMING_LAG] > 0, 1, 
                           np.where(indices[FLD.BIAS3_TREND_TIMING_LAG] < 0, -1, 
                                    np.where(bias3_delta > 0, 1,
                                             np.where(bias3_delta < 0, -1, 0))))
    indices[FLD.BIAS3_TREND_TIMING_LAG] = calc_event_timing_lag(bias3_cross)

    if (ST.VERBOSE in data.columns):
        print('Phase bias_cross_func Done', QA_util_timestamp_to_str())

    return indices


if __name__ == '__main__':

    try:
        import talib
    except:
        print('PLEASE run "pip install TALIB" to call these modules')
        pass
    try:
        import QUANTAXIS as QA
        from QUANTAXIS.QAUtil.QAParameter import ORDER_DIRECTION
        from QUANTAXIS.QAData.QADataStruct import (
            QA_DataStruct_Index_min, 
            QA_DataStruct_Index_day, 
            QA_DataStruct_Stock_day, 
            QA_DataStruct_Stock_min,
            QA_DataStruct_CryptoCurrency_day,
            QA_DataStruct_CryptoCurrency_min,
            )
        from QUANTAXIS.QAIndicator.talib_numpy import *
        from QUANTAXIS.QAUtil.QADate_Adv import (
            QA_util_timestamp_to_str,
            QA_util_datetime_to_Unix_timestamp,
            QA_util_print_timestamp
        )
        from QUANTAXIS.QAUtil.QALogs import (
            QA_util_log_info, 
            QA_util_log_debug,
            QA_util_log_expection)
        from QUANTAXIS.QAFetch.QAhuobi import (
            FIRST_PRIORITY,
        )
    except:
        print('PLEASE run "pip install QUANTAXIS" before call GolemQ.indices.indices modules')
        pass

    from GolemQ.utils.parameter import (
        AKA, 
        INDICATOR_FIELD as FLD, 
        TREND_STATUS as ST
    )

    import pandas as pd
    import datetime

    # ETF/股票代码，如果选股以后：我们假设有这些代码
    codelist = ['159919', '159908', '159902', '510900', 
                '513100', '512980', '515000', '512800', 
                '512170', '510300', '159941', '512690',
                '159928']
    codelist = ['159919']
    codelist = ['HUOBI.btcusdt']
    codelist = ['600276']
    data_day = QA.QA_fetch_stock_day_adv(codelist,
        start='2014-01-01',
        end='{}'.format(datetime.date.today())).to_qfq()

    ## 获取ETF/股票中文名称，只是为了看得方便，交易策略并不需要ETF/股票中文名称
    #stock_names = QA.QA_fetch_etf_name(codelist)
    #codename = [stock_names.at[code, 'name'] for code in codelist]

    ## 读取 ETF基金 日线，存在index_day中
    #data_day = QA.QA_fetch_index_day_adv(codelist,
    #    start='2014-01-01',
    #    end='{}'.format(datetime.date.today()))

    #frequency = '60min'
    #data_day = QA.QA_fetch_cryptocurrency_min_adv(code=codelist,
    #                start='2019-06-20',
    #                end=QA_util_timestamp_to_str(),
    #                frequence=frequency)
    #data_day = QA.QA_DataStruct_CryptoCurrency_min(data_day.resample('4h'))

    indices = data_day.add_func(boll_cross_func)
    indices = data_day.add_func(renko_trend_cross_func, indices)

    ## Get optimal brick size based
    #optimal_brick = renko().set_brick_size(auto = True, HLC_history =
    #data_day.data[["high", "low", "close"]])

    ## Build Renko chart
    #renko_obj_atr = renko()
    #print('Set brick size to optimal: ', renko_obj_atr.set_brick_size(auto =
    #False, brick_size = optimal_brick))
    #renko_obj_atr.build_history(hlc = data_day.data[["high", "low",
    #"close"]].values)
    ##print('Renko bar prices: ', renko_obj_atr.get_renko_prices())
    ##print('Renko bar directions: ', renko_obj_atr.get_renko_directions())
    #print('Renko bar evaluation: ', renko_obj_atr.evaluate())

    #if len(renko_obj_atr.get_renko_prices()) > 1:
    #    indices = pd.concat([indices,
    #                        pd.DataFrame(renko_obj_atr.source_aligned,
    #                                   columns=[FLD.RENKO_TREND_S_LB,
    #                                            FLD.RENKO_TREND_S_UB,
    #                                            FLD.RENKO_TREND_S],
    #                                   index=data_day.data.index)]
    #                        , axis=1)

    #    #renko_obj_atr.plot_renko(data_day.data)

    ## Function for optimization
    #def evaluate_renko(brick, history, column_name):
    #    renko_obj = renko()
    #    renko_obj.set_brick_size(brick_size = brick, auto = False)
    #    renko_obj.build_history(prices = history)
    #    return renko_obj.evaluate()[column_name]

    ## Get ATR values (it needs to get boundaries)
    ## Drop NaNs
    #atr = talib.ATR(high = np.double(data_day.data.high),
    #                low = np.double(data_day.data.low),
    #                close = np.double(data_day.data.close),
    #                timeperiod = 14)
    #atr = atr[np.isnan(atr) == False]

    ## Get optimal brick size as maximum of score function by Brent's (or
    ## similar) method
    ## First and Last ATR values are used as the boundaries
    #optimal_brick_sfo = opt.fminbound(lambda x: -evaluate_renko(brick = x,
    #                                                            history =
    #                                                            data_day.data.close,
    #                                                            column_name =
    #                                                            'score'),
    #                                  np.min(atr), np.max(atr), disp=0)
    ## Build Renko chart
    #renko_obj_sfo = renko()
    #print('Set brick size to optimal: ', renko_obj_sfo.set_brick_size(auto =
    #False, brick_size = optimal_brick_sfo))
    #renko_obj_sfo.build_history(hlc = data_day.data[["high", "low",
    #"close"]].values)
    ##print('Renko bar prices: ', renko_obj_sfo.get_renko_prices())
    ##print('Renko bar gap: ', renko_obj_sfo.get_renko_gaps())
    ##print('Renko bar upper shadow: ', renko_obj_sfo.get_renko_upper_shadow())
    ##print('Renko bar lower shadow: ', renko_obj_sfo.get_renko_lower_shadow())

    ##print('Renko bar directions: ', renko_obj_sfo.get_renko_directions())
    ##print('Renko bar evaluation: ', renko_obj_sfo.evaluate())

    #if len(renko_obj_sfo.get_renko_prices()) > 1:
    #    indices = pd.concat([indices,
    #                         pd.DataFrame(renko_obj_sfo.source_aligned,
    #                           columns=[FLD.RENKO_TREND_L_LB,
    #                                    FLD.RENKO_TREND_L_UB,
    #                                    FLD.RENKO_TREND_L],
    #                           index=data_day.data.index)], axis=1)
    ##    print(len(renko_obj_sfo.get_renko_prices()))
    ##    print(len(data_day.data))
    #    #renko_obj_sfo.plot_renko(data_day.data)
    
    #indices[FLD.RENKO_TREND] = np.where((indices[FLD.RENKO_TREND_L] == 1) & \
    #                                    (indices[FLD.RENKO_TREND_S] == 1), 1,
    #                                    np.where((indices[FLD.RENKO_TREND_L]
    #                                    == -1) & \
    #                                    (indices[FLD.RENKO_TREND_S] == -1),
    #                                    -1, 0))

    plot_renko(data_day.data, indices)

    plt.show()