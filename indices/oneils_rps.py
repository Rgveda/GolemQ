# coding:utf-8
# Author: 阿财（Rgveda@github）（4910163#qq.com）
# Created date: 2020-02-27
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
#
"""
RPS英文全称Relative Price Strength Rating，即股价相对强度，
该指标是欧奈尔CANSLIM选股法则中的趋势分析，具有很强的实战指导意义。
RPS指标是指在一段时间内，个股涨幅在全部股票涨幅排名中的位次值。
"""

import datetime
import numpy as np
import pandas as pd

try:
    import talib
except:
    print('PLEASE run "pip install TALIB" to call these modules')
    pass

try:
    import QUANTAXIS as QA
    from QUANTAXIS.QAUtil.QADate_Adv import (
        QA_util_timestamp_to_str,
        QA_util_datetime_to_Unix_timestamp,
        QA_util_print_timestamp
    )
    from QUANTAXIS.QAUtil.QALogs import (
        QA_util_log_info, 
        QA_util_log_debug, 
        QA_util_log_expection
    )
except:
    print('PLEASE run "pip install QUANTAXIS" before call GolemQ.indices.oneils_rps modules')
    pass

from GolemQ.utils.parameter import (
    AKA, 
    INDICATOR_FIELD as FLD, 
    TREND_STATUS as ST
)
from GolemQ.indices.indices import (
    lineareg_cross_func,
    boll_cross_drawdown,
    combo_flow_cross,
    risk_free_baseline_func,
)
from GolemQ.analysis.timeseries import (
    Timeline_duration,
    Timeline_Integral,
)


def calc_rps_returns_v1_func(data, *args, **kwargs):
    """
    计算收益率 实现算法1
    w:周5;月20;半年：120; 一年250
    """
    w = 63
    indices = None

    # 针对多标的，拆分 indices 数据再自动合并
    code = data.index.get_level_values(level=1)[0]
    if (len(kwargs.keys()) > 0):
        w = kwargs['w'] if ('w' in kwargs.keys()) else w
        indices = kwargs['indices'].loc[(slice(None), code), :] if ('indices' in kwargs.keys()) else None

    if (len(args) > 0):
        w = args[0] if (len(args) > 0) else w
        indices = args[1].loc[(slice(None), code), :] if (len(args) >= 2) else None

    rps_reutrns = data.close / data.close.shift(w) - 1
    rps_reutrns.name = FLD.RPS_RETURNS
    if (indices is None):
        return pd.DataFrame(rps_reutrns, 
                            index=data.index).fillna(0)
    else:
        return pd.concat([indices,
                          rps_reutrns.fillna(0)], 
                         axis=1)


def calc_rps_returns_roc_func(data, *args, **kwargs):
    """
    计算收益率 实现算法2，经测试和算法1无速度差别，结果一致
    w:周5;月20;半年：120; 一年250
    """
    w = 63
    indices = None

    # 针对多标的，拆分 indices 数据再自动合并
    code = data.index.get_level_values(level=1)[0]
    if (len(kwargs.keys()) > 0):
        w = kwargs['w'] if ('w' in kwargs.keys()) else w
        indices = kwargs['indices'].loc[(slice(None), code), :] if ('indices' in kwargs.keys()) else None

    if (len(args) > 0):
        w = args[0] if (len(args) > 0) else w
        indices = args[1].loc[(slice(None), code), :] if (len(args) >= 2) else None

    rps_reutrns = talib.ROC(data.close, w) / 100
    if (indices is None):
        return pd.DataFrame(rps_reutrns,
                            columns = [FLD.RPS_RETURNS],
                            index=data.index).fillna(0)
    else:
        return pd.concat([indices,
                          pd.Series(rps_reutrns,
                                    name=FLD.RPS_RETURNS,
                                    index=data.index).fillna(0)], 
                         axis=1)


def get_RPS_snapshot(ser):
    """
    计算RPS(时间截面)，执行一次时间大约5~6ms
    """
    ret_indices = pd.DataFrame(ser)
    ret_indices[FLD.RPS_RANK] = ser.rank(ascending=False, 
                                         pct=False)
    ret_indices[FLD.RPS_PCT] = ser.rank(pct=True)
    return ret_indices


def all_RPS_rank(data, by_cloumn=FLD.RPS_RETURNS):
    """
    计算每个交易日所有股票滚动w日的RPS
    方法1
    """
    ret_indices = data
    ret_indices[FLD.RPS_RANK] = np.nan
    ret_indices[FLD.RPS_PCT] = np.nan
    ret_indices.loc[:, [by_cloumn,
                        FLD.RPS_RANK, 
                        FLD.RPS_PCT]] = data.groupby(level=[0]).apply(lambda x:
                                                                      get_RPS_snapshot(x[by_cloumn]))

    return ret_indices


def all_RPS_rank_pd(data, by_cloumn=FLD.RPS_RETURNS):
    """
    计算每个交易日所有股票滚动w日的RPS
    方法2，大约比方法1慢3倍
    """
    ret_indices = data
    ret_indices[FLD.RPS_RANK] = np.nan
    ret_indices[FLD.RPS_PCT] = np.nan

    each_day = sorted(data.index.get_level_values(level=0).unique())
    for day in each_day:
        ret_indices.loc[(day, slice(None)), 
                        [FLD.by_cloumn,
                         FLD.RPS_RANK, 
                         FLD.RPS_PCT]] = get_RPS_snapshot(data.loc[(day, slice(None)), 
                                                                   by_cloumn])

    return ret_indices


def rps_punch_position_func(data, *args, **kwargs):
    """
    计算 RPS 排名支撑下的”浅滩“突破买点信号。
    """
    indices = None

    # 针对多标的，拆分 indices 数据再自动合并
    code = data.index.get_level_values(level=1)[0]
    if (len(kwargs.keys()) > 0):
        indices = kwargs['features'].loc[(slice(None), code), :] if ('features' in kwargs.keys()) else None

    if (len(args) > 0):
        indices = args[0].loc[(slice(None), code), :] if (len(args) >= 1) else None

    if (ST.BOOTSTRAP_I not in indices.columns):
        if (ST.VERBOSE in data.columns):
            print('Phase bootstrap@rps_punch', QA_util_timestamp_to_str())
        print(u'Missing bootstrap_fractal func...')

    if (FLD.COMBINE_DENSITY not in indices.columns):
        indices = combo_flow_cross(data, indices)

    if (FLD.BOLL_CROSS_SX_DEA not in indices.columns):
        indices = boll_cross_drawdown(data, indices)

        # 布林/RENKO带交易策略，买卖信号（金叉阶段为安全持有，死叉阶段检查Bootstrap启动）
        boll_renko_cross_jx = Timeline_duration(np.where((indices[FLD.BOLL_SX_DRAWDOWN] > 0) & \
                                                (indices[FLD.RENKO_TREND_S] > 0), 1, 0))
        boll_renko_cross_sx = Timeline_duration(np.where((indices[FLD.BOLL_SX_DRAWDOWN] < 0) & \
                                                (indices[FLD.RENKO_TREND_S] < 0), 1, 0))
        boll_renko_cross = np.where((boll_renko_cross_sx - boll_renko_cross_jx) > 0, 1, 
                                    np.where((boll_renko_cross_jx - boll_renko_cross_sx) > 0, -1, 0))
        indices[FLD.BOLL_RENKO_TIMING_LAG] = calc_event_timing_lag(boll_renko_cross)

    if (FLD.BASELINE_ROC not in indices.columns):
        indices = risk_free_baseline_func(data, 
                                          risk_free=0.04, 
                                          frequency='day', 
                                          indices=indices)
    
    # 逆势反转检查
    reversal_spike_checked = (((indices[FLD.MA5] > data[AKA.OPEN]) & \
        (indices[FLD.MACD] < 0)) | \
        ((indices[FLD.RENKO_TREND_S_UB] < data[AKA.CLOSE]) & \
        (indices[FLD.BIAS3] > indices[FLD.BIAS3].shift(1)) & \
        (indices[FLD.MACD] < 0)) | \
        ((indices[FLD.RENKO_TREND_S] > 0) & \
        (indices[FLD.BIAS3] > indices[FLD.BIAS3].shift(1)) & \
        (indices[FLD.MACD] < 0)) | \
        (indices[FLD.MACD] > 0)) & \
        ~((indices[FLD.DEA_SLOPE] < 0) & \
        (indices[FLD.MACD] < 0) & \
        (indices[FLD.MAXFACTOR_CROSS] < 0) & \
        (indices[FLD.MA5] < data[AKA.OPEN]))

    trigger_rps_01 = ((indices[FLD.MAXFACTOR_CROSS_JX] < 6) | \
        ((indices[FLD.BIAS3].shift(1) < 0) & \
        (indices[FLD.ZSCORE_21].shift(1) < 0.382) & \
        (indices[FLD.BIAS3].shift(1) < indices[FLD.BIAS3]))) & \
        (indices[FLD.ZSCORE_21] < 0.809) & \
        (indices[FLD.BIAS3_ZSCORE] < 0.93) & \
        (reversal_spike_checked == True) & \
        ~(((indices[FLD.ZSCORE_21].shift(1) + indices[FLD.ZSCORE_84].shift(1)) > 1.1) & \
        (indices[FLD.DEA] > 0) & \
        ((indices[FLD.COMBINE_TIDE_CROSS_SX] >= (indices[FLD.MAXFACTOR_CROSS_JX] + 4)) | \
        ((indices[FLD.MAXFACTOR_CROSS] + indices[FLD.DUAL_CROSS]) < 0))) & \
        (indices[FLD.ZSCORE_21].shift(1) <= indices[FLD.ZSCORE_21]) & \
        ((indices[FLD.ZSCORE_21].shift(1) < 0.382) | \
        (indices[FLD.ZSCORE_21] < 0.382) | \
        ((indices[FLD.ZSCORE_21].rolling(8).mean() < 0.512) & \
        (indices[FLD.DEA] < 0) & \
        (indices[FLD.BOLL_CROSS_SX_DEA] > 0) & \
        (indices[FLD.ZSCORE_21].rolling(4).mean() < 0.618))) & \
        ((indices[FLD.DEA] > 0) | \
        ((indices[FLD.NEGATIVE_LOWER_PRICE] == True) & \
        ((indices[FLD.BOLL_CROSS_SX_DEA] > 0) | \
        (indices[FLD.COMBINE_DENSITY].rolling(8).mean() > 0.512))) | \
        ((indices[FLD.DEA] < 0) & \
        (indices[FLD.BOOTSTRAP_I_BEFORE] < indices[FLD.MACD_CROSS_SX_BEFORE]))) & \
        (((indices[FLD.DUAL_CROSS] + indices[FLD.VOLUME_FLOW_TRI_CROSS]) >= 0) | \
        (((indices[FLD.REGRESSION_SLOPE] > indices[FLD.MA30_SLOPE]) & \
        (indices[FLD.BOLL_CROSS_SX_DEA] > 0)) | \
        (indices[FLD.MA30_SLOPE] > indices[FLD.MA90_SLOPE]) | \
        (indices[FLD.MA90_SLOPE] > indices[FLD.MA120_SLOPE]) & \
        (indices[FLD.DEA] > 0)) & \
        ((indices[FLD.NEGATIVE_LOWER_PRICE] == True) & \
        ((indices[FLD.BOLL_CROSS_SX_DEA] > 0) | \
        (indices[FLD.COMBINE_DENSITY].rolling(8).mean() > 0.512)) & \
        (indices[FLD.MACD_DELTA] > 0))) & \
        ~((indices[FLD.MACD] < 0) & \
        (indices[FLD.BIAS3] < 0) & \
        (indices[FLD.MACD_DELTA] < 0))
    trigger_rps = np.where(trigger_rps_01 == True, 1, 0)

    trigger_rps_02 = ((indices[FLD.MAXFACTOR_CROSS_JX] < 6) | \
        (indices[FLD.DUAL_CROSS_JX] >= 4)) & \
        (indices[FLD.ZSCORE_21] < 0.809) & \
        ~(((indices[FLD.ZSCORE_21] + indices[FLD.ZSCORE_84]) > 1) & \
        (indices[FLD.DEA] > 0) & \
        ((indices[FLD.COMBINE_TIDE_CROSS_SX] >= (indices[FLD.MAXFACTOR_CROSS_JX] + 4)) | \
        ((indices[FLD.MAXFACTOR_CROSS] + indices[FLD.DUAL_CROSS]) < 0))) & \
        (indices[FLD.ZSCORE_21].shift(1) <= indices[FLD.ZSCORE_21]) & \
        ((indices[FLD.ZSCORE_21].shift(1) < 0.382) | \
        (indices[FLD.ZSCORE_21] < 0.382)) & \
        (indices[FLD.COMBINE_DENSITY].rolling(8).mean() > 0.618) & \
        (reversal_spike_checked == True) & \
        (indices[FLD.DEA] > 0) & \
        (indices[FLD.DEA_SLOPE] > 0)

    trigger_rps = np.where(trigger_rps_02 == True, 2, trigger_rps)

    trigger_rps_03 = ((indices[FLD.MAXFACTOR_CROSS_JX] < 6) | \
        (indices[FLD.DUAL_CROSS_JX] >= 4)) & \
        (indices[FLD.ZSCORE_21] < 0.809) & \
        ~(((indices[FLD.ZSCORE_21] + indices[FLD.ZSCORE_84]) > 1) & \
        (indices[FLD.DEA] > 0) & \
        ((indices[FLD.COMBINE_TIDE_CROSS_SX] >= (indices[FLD.MAXFACTOR_CROSS_JX] + 4)) | \
        ((indices[FLD.MAXFACTOR_CROSS] + indices[FLD.DUAL_CROSS]) < 0))) & \
        (indices[FLD.ZSCORE_21].shift(1) <= indices[FLD.ZSCORE_21]) & \
        ((indices[FLD.ZSCORE_21].shift(1) < 0.512) | \
        (indices[FLD.ZSCORE_21] < 0.512)) & \
        (indices[FLD.COMBINE_DENSITY].rolling(8).mean() > 0.618) & \
        (indices[FLD.ZSCORE_84].shift(1) <= indices[FLD.ZSCORE_84]) & \
        (((indices[FLD.ZSCORE_84].shift(1) < 0.512) | \
        (indices[FLD.ZSCORE_84] < 0.512)) | \
        (((indices[FLD.DUAL_CROSS] + indices[FLD.VOLUME_FLOW_TRI_CROSS]) >= 0) & \
        (indices[FLD.BOLL_CROSS_SX_DEA] > 0))) & \
        (indices[FLD.DEA] > 0) & \
        (reversal_spike_checked == True) & \
        (((indices[FLD.REGRESSION_SLOPE] > indices[FLD.MA30_SLOPE]) & \
        (indices[FLD.BOLL_CROSS_SX_DEA] > 0)) | \
        (indices[FLD.MA30_SLOPE] > indices[FLD.MA90_SLOPE]) | \
        (indices[FLD.MA90_SLOPE] > indices[FLD.MA120_SLOPE]) & \
        (indices[FLD.DEA] > 0))
    trigger_rps = np.where(trigger_rps_03 == True, 3, trigger_rps)

    trigger_rps_04 = (indices[FLD.MAXFACTOR_CROSS_JX] < 6) & \
        (indices[FLD.ZSCORE_21] < 0.809) & \
        ~(((indices[FLD.ZSCORE_21] + indices[FLD.ZSCORE_84]) > 1) & \
        (indices[FLD.DEA] > 0) & \
        ((indices[FLD.COMBINE_TIDE_CROSS_SX] >= (indices[FLD.MAXFACTOR_CROSS_JX] + 4)) | \
        ((indices[FLD.MAXFACTOR_CROSS] + indices[FLD.DUAL_CROSS]) < 0))) & \
        (indices[FLD.ZSCORE_21].shift(1) <= indices[FLD.ZSCORE_21]) & \
        ((indices[FLD.ZSCORE_21].shift(1) < 0.382) | \
        (indices[FLD.ZSCORE_21] < 0.382)) & \
        (indices[FLD.NEGATIVE_LOWER_PRICE].shift(1) == True) & \
        (indices[FLD.COMBINE_DENSITY].rolling(8).mean() > 0.512)
    trigger_rps = np.where(trigger_rps_04 == True, 4, trigger_rps)

    trigger_rps_05 = ((indices[FLD.MAXFACTOR_CROSS_JX] < 6) | \
        (indices[FLD.DUAL_CROSS_JX] >= 4) | \
        (indices[ST.BOOTSTRAP_I] == True)) & \
        (indices[FLD.ZSCORE_21] < 0.809) & \
        ~(((indices[FLD.ZSCORE_21] + indices[FLD.ZSCORE_84]) > 1) & \
        (indices[FLD.DEA] > 0) & \
        ((indices[FLD.COMBINE_TIDE_CROSS_SX] >= (indices[FLD.MAXFACTOR_CROSS_JX] + 4)) | \
        ((indices[FLD.MAXFACTOR_CROSS] + indices[FLD.DUAL_CROSS]) < 0))) & \
        (indices[FLD.BIAS3_ZSCORE] < 0.93) & \
        (indices[FLD.ZSCORE_21].shift(1) <= indices[FLD.ZSCORE_21]) & \
        ((indices[FLD.ZSCORE_21].shift(1) < 0.382) | \
        (indices[FLD.ZSCORE_21] < 0.382)) & \
        (indices[FLD.ZSCORE_84].shift(1) <= indices[FLD.ZSCORE_84]) & \
        ((indices[FLD.ZSCORE_84].shift(1) < 0.382) | \
        (indices[FLD.ZSCORE_84] < 0.382)) & \
        (((indices[FLD.REGRESSION_SLOPE] > indices[FLD.MA30_SLOPE]) & \
        (indices[FLD.BOLL_CROSS_SX_DEA] > 0)) | \
        (indices[FLD.MA30_SLOPE] > indices[FLD.MA90_SLOPE]) | \
        (indices[FLD.MA90_SLOPE] > indices[FLD.MA120_SLOPE]) & \
        (indices[FLD.DEA] > 0)) & \
        ~((indices[FLD.MACD] < 0) & \
        (indices[FLD.BIAS3] < 0) & \
        (indices[FLD.MACD_DELTA] < 0))
    trigger_rps = np.where(trigger_rps_05 == True, 5, trigger_rps)

    trigger_rps_06 = (((indices[FLD.BOLL_CROSS_SX_BEFORE] <= 8) & \
        (indices[FLD.MA90_CLEARANCE] > -0.618) & \
        (indices[FLD.MA90_CLEARANCE] < 0.382)) | \
        ((indices[FLD.BOLL_CROSS_JX_BEFORE] <= 1) & \
        (indices[FLD.MACD] < 0) & \
        (indices[FLD.DEA] > 0))) & \
        (((indices[FLD.MAXFACTOR_CROSS_JX] < 4) | \
        (indices[FLD.MACD] > 0) | \
        ((indices[FLD.BIAS3].shift(1) < 0)) & \
        (indices[FLD.BIAS3] > 0) & \
        (indices[FLD.BIAS3].shift(1) < indices[FLD.BIAS3]))) & \
        (indices[FLD.ZSCORE_21].shift(1) < 0.382) & \
        (indices[FLD.ZSCORE_84].shift(1) < 0.382) & \
        (indices[FLD.ZSCORE_21].shift(1) <= indices[FLD.ZSCORE_21]) & \
        (indices[FLD.ZSCORE_84].shift(1) <= indices[FLD.ZSCORE_84]) & \
        (((indices[FLD.BOLL_CROSS_JX_BEFORE] > 36) & \
        (indices[FLD.DIF] > 0)) | \
        ((indices[FLD.BOLL_CROSS_JX_BEFORE] <= 2) & \
        (indices[FLD.ZEN_WAVELET_CROSS_SX_BEFORE] > 18) & \
        (indices[FLD.DIF] > 0)) | \
        ((indices[FLD.DEA_CROSS_JX_BEFORE] > 36) & \
        (indices[FLD.MA90_CLEARANCE] > 1.236) & \
        (indices[FLD.DIF] > 0)) | \
        ((indices[FLD.BOLL_CROSS_JX_BEFORE] > indices[FLD.DEA_CROSS_JX_BEFORE]) & \
        (indices[FLD.MA90_CLEARANCE] > 1.236) & \
        (indices[FLD.DIF] > 0)) | \
        ((indices[FLD.BOLL_CROSS_JX_BEFORE] < indices[FLD.BOLL_CROSS_SX_BEFORE]) & \
        (indices[FLD.BOOTSTRAP_I_BEFORE] < indices[FLD.MACD_CROSS_SX_BEFORE])))
    trigger_rps = np.where(trigger_rps_06 == True, 6, trigger_rps)

    trigger_rps_07 = (indices[FLD.BOLL_CROSS_SX_BEFORE] <= 8) & \
        (indices[FLD.MA90_CLEARANCE] > 1.236) & \
        (indices[FLD.MACD_CROSS_SX_BEFORE] <= 6) & \
        (indices[FLD.MACD] < 0) & \
        (indices[FLD.BIAS3].shift(1) < 0) & \
        (indices[FLD.BIAS3] > 0) & \
        (indices[FLD.BIAS3].shift(1) < indices[FLD.BIAS3]) & \
        (indices[FLD.ZSCORE_21].shift(1) < 0.9) & \
        (indices[FLD.ZSCORE_84].shift(1) < 0.9) & \
        (indices[FLD.ZSCORE_21].shift(1) <= indices[FLD.ZSCORE_21]) & \
        (indices[FLD.ZSCORE_84].shift(1) <= indices[FLD.ZSCORE_84]) & \
        ~((indices[FLD.BIAS3].shift(1) < 0) & \
        (indices[FLD.MACD] < 0) & \
        (indices[FLD.MAXFACTOR] < 0) & \
        (indices[FLD.MAXFACTOR_CROSS] < 0)) & \
        (((indices[FLD.BOLL_CROSS_JX_BEFORE] > 36) & \
        (indices[FLD.DIF] > 0)) | \
        ((indices[FLD.BOLL_CROSS_JX_BEFORE] <= 2) & \
        (indices[FLD.ZEN_WAVELET_CROSS_SX_BEFORE] > 18) & \
        (indices[FLD.DIF] > 0)) | \
        ((indices[FLD.DEA_CROSS_JX_BEFORE] > 36) & \
        (indices[FLD.MA90_CLEARANCE] > 1.236) & \
        (indices[FLD.DIF] > 0)) | \
        ((indices[FLD.BOLL_CROSS_JX_BEFORE] > indices[FLD.DEA_CROSS_JX_BEFORE]) & \
        (indices[FLD.MA90_CLEARANCE] > 1.236) & \
        (indices[FLD.DIF] > 0)) | \
        ((indices[FLD.BOLL_CROSS_JX_BEFORE] < indices[FLD.BOLL_CROSS_SX_BEFORE]) & \
        (indices[FLD.BOOTSTRAP_I_BEFORE] < indices[FLD.MACD_CROSS_SX_BEFORE])))
    trigger_rps = np.where(trigger_rps_07 == True, 7, trigger_rps)

    trigger_rps_10 = ((indices[FLD.MA90_CLEARANCE] > 1.236) | \
        ((indices[FLD.MA90_CLEARANCE] > 0.809) & \
        (indices[FLD.BIAS3_ZSCORE] < 0.382))) & \
        (indices[FLD.BIAS3].shift(1) > 0) & \
        (indices[FLD.BIAS3].shift(3) > 0) & \
        (indices[FLD.BIAS3].shift(5) > 0) & \
        (indices[FLD.BIAS3].shift(7) > 0) & \
        (reversal_spike_checked == True) & \
        (((indices[FLD.MACD_CROSS_SX_BEFORE] <= 6) & \
        ((indices[FLD.MA5] > data[AKA.OPEN]) | \
        (indices[FLD.ZEN_WAVELET_CROSS_JX_BEFORE] < indices[FLD.ZEN_WAVELET_CROSS_SX_BEFORE])) & \
        (indices[FLD.MACD] < 0)) | (indices[FLD.MACD] > 0)) & \
        ((indices[FLD.BIAS3_ZSCORE].shift(1) < indices[FLD.BIAS3_ZSCORE]) | \
        (indices[FLD.BIAS3_ZSCORE].shift(2) < indices[FLD.BIAS3_ZSCORE]) | \
        (indices[FLD.BIAS3_ZSCORE].shift(3) < indices[FLD.BIAS3_ZSCORE])) & \
        (indices[FLD.BIAS3] > 0) & \
        (indices[FLD.BIAS3] < 9.27) & \
        (indices[FLD.MAXFACTOR_CROSS] > 0) & \
        ((indices[FLD.MAXFACTOR_CROSS].shift(1) < 0) | \
        (indices[FLD.MAXFACTOR_CROSS].shift(2) < 0)) & \
        (((indices[FLD.BOLL_CROSS_JX_BEFORE] > 36) & \
        (indices[FLD.DIF] > 0)) | \
        ((indices[FLD.BOLL_CROSS_JX_BEFORE] <= 2) & \
        (indices[FLD.ZEN_WAVELET_CROSS_SX_BEFORE] > 18) & \
        (indices[FLD.DIF] > 0)) | \
        ((indices[FLD.DEA_CROSS_JX_BEFORE] > 36) & \
        (indices[FLD.MA90_CLEARANCE] > 1.236) & \
        (indices[FLD.DIF] > 0)) | \
        ((indices[FLD.BOLL_CROSS_JX_BEFORE] > indices[FLD.DEA_CROSS_JX_BEFORE]) & \
        (indices[FLD.MA90_CLEARANCE] > 1.236) & \
        (indices[FLD.DIF] > 0)) | \
        ((indices[FLD.BOLL_CROSS_JX_BEFORE] < indices[FLD.BOLL_CROSS_SX_BEFORE]) & \
        (indices[FLD.BOOTSTRAP_I_BEFORE] < indices[FLD.MACD_CROSS_SX_BEFORE])))
    trigger_rps = np.where(trigger_rps_10 == True, 10, trigger_rps)

    trigger_rps_11 = (indices[FLD.MA90_CLEARANCE] > -0.618) & \
        (indices[FLD.MA90_CLEARANCE] < 0.309) & \
        (indices[FLD.MAXFACTOR_CROSS] > 0) & \
        (indices[FLD.MAXFACTOR_CROSS_JX] <= 6) & \
        (indices[FLD.BIAS3] > -9.27) & \
        (indices[FLD.BIAS3].shift(1) < 0) & \
        (indices[FLD.BIAS3_ZSCORE].shift(1) < 0.382) & \
        (indices[FLD.BIAS3_ZSCORE].shift(1) < indices[FLD.BIAS3_ZSCORE]) & \
        (indices[FLD.ZSCORE_21].shift(1) < 0.382) & \
        (indices[FLD.ZSCORE_21].shift(1) < indices[FLD.ZSCORE_21]) & \
        (indices[FLD.ZSCORE_84].shift(1) < 0.382) & \
        (indices[FLD.ZSCORE_84].shift(1) < indices[FLD.ZSCORE_84]) & \
        (indices[FLD.MACD_DELTA] > 0) & \
        (indices[FLD.BOLL_DELTA] > 0) & \
        ((indices[FLD.LINEAREG_CROSS_JX_BEFORE] < indices[FLD.LINEAREG_CROSS_SX_BEFORE]) | \
        (indices[FLD.MAXFACTOR_CROSS_JX] < 4)) & \
        (indices[FLD.MACD] < 0) & \
        ((indices[FLD.MACD] + indices[FLD.MACD_DELTA] * 8) > 0)
    trigger_rps = np.where(trigger_rps_11 == True, 11, trigger_rps)

    trigger_rps_12 = (indices[FLD.MA90_CLEARANCE] > -0.618) & \
        (indices[FLD.MA90_CLEARANCE] < 0.309) & \
        (indices[FLD.MAXFACTOR_CROSS] > 0) & \
        (indices[FLD.MAXFACTOR_CROSS_JX] <= 8) & \
        ((np.r_[0, trigger_rps[:-1]] > 0) | \
        (np.r_[np.zeros(2), trigger_rps[:-2]] > 0) | \
        (np.r_[np.zeros(3), trigger_rps[:-3]] > 0) | \
        (np.r_[np.zeros(4), trigger_rps[:-4]] > 0)) & \
        (indices[FLD.BIAS3] > -9.27) & \
        (indices[FLD.BIAS3].shift(1) < 0) & \
        (indices[FLD.BIAS3].shift(1) < indices[FLD.BIAS3]) & \
        (indices[FLD.BIAS3_ZSCORE] < 0.93) & \
        (indices[FLD.BIAS3_ZSCORE].shift(1) < 0.809) & \
        (indices[FLD.BIAS3_ZSCORE].shift(1) < indices[FLD.BIAS3_ZSCORE]) & \
        (indices[FLD.ZSCORE_21].shift(1) < 0.809) & \
        (indices[FLD.ZSCORE_21].shift(1) < indices[FLD.ZSCORE_21]) & \
        (indices[FLD.ZSCORE_84].shift(1) < 0.512) & \
        (indices[FLD.ZSCORE_84].shift(1) < indices[FLD.ZSCORE_84]) & \
        (indices[FLD.MACD_DELTA] > 0) & \
        ((indices[FLD.BOLL_DELTA] > 0) | \
        (indices[FLD.MACD_CROSS_JX_BEFORE] < 2)) & \
        ((indices[FLD.LINEAREG_CROSS_JX_BEFORE] < indices[FLD.LINEAREG_CROSS_SX_BEFORE]) | \
        (indices[FLD.MAXFACTOR_CROSS_JX] < 4))
    trigger_rps = np.where(trigger_rps_12 == True, 12, trigger_rps)

    trigger_rps_13 = (indices[FLD.MA90_CLEARANCE] > -0.382) & \
        (indices[FLD.MA90_CLEARANCE] < 0.618) & \
        (indices[FLD.BIAS3].shift(3) > 0) & \
        (indices[FLD.BIAS3].shift(4) > 0) & \
        (indices[FLD.BIAS3_ZSCORE] < 0.382) & \
        (indices[FLD.BIAS3_ZSCORE].shift(1) < 0.382) & \
        (indices[FLD.BIAS3_ZSCORE].shift(2) < 0.382) & \
        (indices[FLD.ZSCORE_21] < 0.382) & \
        (indices[FLD.ZSCORE_21].shift(1) < 0.382) & \
        (indices[FLD.ZSCORE_21].shift(2) < 0.382) & \
        (indices[FLD.ZSCORE_84] < 0.382) & \
        (indices[FLD.ZSCORE_84].shift(1) < 0.382) & \
        (indices[FLD.ZSCORE_84].shift(2) < 0.382) & \
        ((indices[FLD.LINEAREG_CROSS_JX_BEFORE] < indices[FLD.LINEAREG_CROSS_SX_BEFORE]) | \
        (indices[FLD.MAXFACTOR_CROSS_SX] <= 8)) & \
        (indices[FLD.MACD] < 0) & \
        (indices[FLD.MACD_CROSS_SX_BEFORE] <= 8)
    trigger_rps = np.where(trigger_rps_13 == True, 13, trigger_rps)

    trigger_rps_14 = (((indices[FLD.BOOTSTRAP_I_BEFORE] < indices[FLD.MACD_CROSS_SX_BEFORE]) & \
        ((indices[FLD.DEA] < 0) | \
        ((indices[FLD.DEA_CROSS_JX_BEFORE] < indices[FLD.MACD_CROSS_JX_BEFORE]) & \
        (indices[FLD.DEA] > 0) & \
        (indices[FLD.MACD] > 0)))) | \
        ((indices[FLD.DEA] > 0) & \
        (indices[FLD.BOOTSTRAP_II_BEFORE] < indices[FLD.DEA_CROSS_JX_BEFORE]) & \
        ((indices[FLD.MACD_CROSS_SX_BEFORE] <= 6) | \
        ((indices[FLD.RENKO_TREND_L] > 0) & \
        ((indices[FLD.RENKO_TREND_S] > 0) | \
        (indices[FLD.RENKO_TREND_S_BEFORE] <= 8))) | \
        ((indices[FLD.MACD] > 0) & \
        (indices[FLD.MACD_CROSS_JX_BEFORE] > 8))) & \
        (indices[FLD.RENKO_TREND_L] > 0) & \
        (indices[FLD.RENKO_TREND_S] > 0))) & \
        (indices[FLD.BIAS3_ZSCORE] < 0.93) & \
        (indices[FLD.MAXFACTOR_CROSS] > 0) & \
        (indices[FLD.MAXFACTOR_CROSS].shift(1) < 0) & \
        (indices[FLD.BIAS3] > 0) & \
        (((indices[FLD.BIAS3].shift(1) > 0) & \
        (indices[FLD.BIAS3].shift(3) > 0) & \
        (indices[FLD.BIAS3].shift(5) > 0) & \
        (indices[FLD.BIAS3].shift(7) > 0)) | \
        (indices[FLD.BIAS3].shift(1) < 0)) & \
        ((indices[FLD.MACD_DELTA] > 0) | \
        ((indices[FLD.MACD_CROSS_JX_BEFORE] > 8) & \
        (indices[FLD.MACD] > 0))) & \
        ((indices[FLD.BOLL_DELTA] > 0) | \
        ((indices[FLD.BIAS3] > 0) & \
        (indices[FLD.BIAS3].shift(1) < 0)))
    trigger_rps = np.where(trigger_rps_14 == True, 14, trigger_rps)

    trigger_rps_15 = (((indices[FLD.BOOTSTRAP_I_BEFORE] < indices[FLD.MACD_CROSS_SX_BEFORE]) & \
        ((indices[FLD.DEA] < 0) | \
        ((indices[FLD.DEA_CROSS_JX_BEFORE] < indices[FLD.MACD_CROSS_JX_BEFORE]) & \
        (indices[FLD.DEA] > 0) & \
        (indices[FLD.MACD] > 0)))) | \
        ((indices[FLD.DEA] > 0) & \
        (indices[FLD.BOOTSTRAP_II_BEFORE] < indices[FLD.DEA_CROSS_JX_BEFORE]) & \
        ((indices[FLD.MACD_CROSS_SX_BEFORE] <= 6) | \
        ((indices[FLD.RENKO_TREND_L] > 0) & \
        ((indices[FLD.RENKO_TREND_S] > 0) | \
        (indices[FLD.RENKO_TREND_S_BEFORE] <= 8))) | \
        ((indices[FLD.MACD] > 0) & \
        (indices[FLD.MACD_CROSS_JX_BEFORE] > 8))) & \
        (indices[FLD.RENKO_TREND_L] > 0) & \
        (indices[FLD.RENKO_TREND_S] > 0))) & \
        (indices[FLD.MAXFACTOR_CROSS] > 0) & \
        (indices[FLD.MAXFACTOR_CROSS].shift(1) < 0) & \
        (indices[FLD.BIAS3_ZSCORE] < 0.93) & \
        (indices[FLD.BIAS3] > 0) & \
        (((indices[FLD.BIAS3].shift(1) > 0) & \
        (indices[FLD.BIAS3].shift(3) > 0) & \
        (indices[FLD.BIAS3].shift(5) > 0) & \
        (indices[FLD.BIAS3].shift(7) > 0)) | \
        (indices[FLD.BIAS3].shift(1) < 0)) & \
        ((indices[FLD.LOWER_SETTLE_PRICE] == True) | \
        (indices[FLD.DIF] > 0)) & \
        (indices[FLD.ZSCORE_84].shift() < 0.512) & \
        (indices[FLD.BIAS3_ZSCORE].shift(1) < indices[FLD.BIAS3_ZSCORE]) & \
        (indices[FLD.ZSCORE_21].shift(1) < indices[FLD.ZSCORE_21]) & \
        (indices[FLD.ZSCORE_84].shift(1) < indices[FLD.ZSCORE_84]) & \
        (indices[FLD.MA90_CLEARANCE] > -0.618) & \
        (indices[FLD.MA90_CLEARANCE] < 0.382)
    trigger_rps = np.where(trigger_rps_15 == True, 15, trigger_rps)

    # 以下是鸭子策略 如果它看起来像反弹，走势像反弹，形态像反弹，那么它可能就是反弹。
    trigger_rps_16 = (reversal_spike_checked == True) & \
        (indices[FLD.MACD] > 0) & \
        (indices[FLD.MA5] > data[AKA.OPEN].shift(1)) & \
        ((indices[FLD.BIAS3_ZSCORE] < 0.93) | \
        (indices[FLD.RENKO_TREND_S].shift(1) < 0)) & \
        (((indices[FLD.MAXFACTOR].shift(2) > 0) & \
        (indices[FLD.MAXFACTOR].shift(3) > 0)) | \
        ((indices[FLD.MAXFACTOR].shift(3) > 0) & \
        (indices[FLD.MAXFACTOR].shift(4) > 0))) & \
        (indices[FLD.MACD_DELTA].shift(1) < indices[FLD.MACD_DELTA]) & \
        (indices[FLD.MAXFACTOR].shift(1) < indices[FLD.MAXFACTOR]) & \
        (indices[FLD.RENKO_TREND_S_UB].shift(1) < indices[FLD.RENKO_TREND_S_UB]) & \
        (indices[FLD.RENKO_TREND_S_UB].shift(2) < indices[FLD.RENKO_TREND_S_UB]) & \
        (indices[FLD.BIAS3].shift(1) < indices[FLD.BIAS3]) & \
        ((indices[FLD.BIAS3_ZSCORE].shift(1) < indices[FLD.BIAS3_ZSCORE]) | \
        ((indices[FLD.BIAS3_ZSCORE].shift(1) <= indices[FLD.BIAS3_ZSCORE]) & \
        (indices[FLD.MAXFACTOR].shift(1) < 0))) & \
        (indices[FLD.ZSCORE_21].shift(1) < indices[FLD.ZSCORE_21]) & \
        (indices[FLD.ZSCORE_84].shift(1) < indices[FLD.ZSCORE_84]) & \
        (indices[FLD.MA90_CLEARANCE] > 0.618)
    trigger_rps = np.where(trigger_rps_16 == True, 16, trigger_rps)
 
    trigger_rps_17 = (reversal_spike_checked == True) & \
        (indices[FLD.RENKO_TREND_S_UB] < data[AKA.CLOSE]) & \
        ((indices[FLD.MACD] < 0) | \
        ((indices[FLD.MACD_DELTA] > 0) & \
        (indices[FLD.MAXFACTOR] > 0) & \
        (indices[FLD.MAXFACTOR].shift(1) < 0))) & \
        ((indices[FLD.BOLL_CROSS_SX_BEFORE] <= 4) | \
        (indices[FLD.MAXFACTOR_CROSS_JX] <= 1)) & \
        ((indices[FLD.RENKO_TREND_S_BEFORE] > 1) | \
        ((indices[FLD.RENKO_TREND_S_UB] > indices[FLD.RENKO_TREND_S_UB].shift(1)) & \
        ((indices[FLD.MACD] < 0) | \
        (indices[FLD.MAXFACTOR] < 0))))
    trigger_rps = np.where(trigger_rps_17 == True, 17, trigger_rps)
   
    trigger_rps_18 = (reversal_spike_checked == True) & \
        (indices[FLD.MACD] < 0) & \
        (indices[FLD.DEA] > 0) & \
        (indices[FLD.RENKO_TREND_S] > 0) & \
        ((indices[ST.POSITION_RSRS] > 0) | \
        (indices[ST.TRIGGER_RSRS] > 0)) & \
        (indices[FLD.MA5] > data[AKA.OPEN].shift(1)) & \
        (indices[FLD.MAXFACTOR_CROSS_JX] == 0)
    trigger_rps = np.where(trigger_rps_18 == True, 18, trigger_rps)
   
    trigger_rps_19 = (reversal_spike_checked == True) & \
        (indices[FLD.RENKO_TREND_L] > 0) & \
        (indices[FLD.DEA] > 0) & \
        (indices[FLD.MACD] < 0) & \
        (indices[ST.TRIGGER_RSRS] > 0) & \
        (indices[FLD.ZEN_WAVELET_CROSS_JX_BEFORE] == 0) & \
        (indices[FLD.MAXFACTOR_CROSS] > 0) & \
        (indices[FLD.MAXFACTOR] > 0) & \
        ((indices[FLD.BIAS3] > 0) | \
        (indices[FLD.MACD_DELTA] > 0))
    trigger_rps = np.where(trigger_rps_19 == True, 19, trigger_rps)

    if (FLD.MACD_TREND in indices.columns):
        # 超级牛，超级牛，超级牛
        trigger_rps_08 = ((indices[FLD.MAXFACTOR_CROSS_JX] < 6) | \
            ((indices[FLD.MACD_TREND] > 0) & \
            (indices[FLD.MACD_TREND].shift(1) > 0))) & \
            (indices[FLD.LINEAREG_PRICE] > indices[FLD.MA90]) & \
            (indices[FLD.MA90] > indices[FLD.MA120]) & \
            (indices[FLD.MA120_CLEARANCE] < 1.236) & \
            (((indices[FLD.MAXFACTOR_CROSS_JX] < 4) | \
            (indices[FLD.BIAS3].shift(1) < 0)) & \
            (indices[FLD.BIAS3] > 0) & \
            (reversal_spike_checked == True) & \
            (indices[FLD.BIAS3].shift(1) < indices[FLD.BIAS3])) & \
            (indices[FLD.ZSCORE_21].shift(1) < 0.382) & \
            (indices[FLD.ZSCORE_21] < 0.512) & \
            (indices[FLD.ZSCORE_21].shift(1) <= indices[FLD.ZSCORE_21]) & \
            (indices[FLD.ZSCORE_84].shift(1) <= indices[FLD.ZSCORE_84]) & \
            (((indices[FLD.BOLL_CROSS_JX_BEFORE] > 36) & \
            (indices[FLD.DIF] > 0)) | \
            ((indices[FLD.BOLL_CROSS_JX_BEFORE] <= 2) & \
            (indices[FLD.ZEN_WAVELET_CROSS_SX_BEFORE] > 18) & \
            (indices[FLD.DIF] > 0)) | \
            ((indices[FLD.DEA_CROSS_JX_BEFORE] > 36) & \
            (indices[FLD.MA90_CLEARANCE] > 1.236) & \
            (indices[FLD.DIF] > 0)) | \
            ((indices[FLD.BOLL_CROSS_JX_BEFORE] > indices[FLD.DEA_CROSS_JX_BEFORE]) & \
            (indices[FLD.MA90_CLEARANCE] > 1.236) & \
            (indices[FLD.DIF] > 0)) | \
            ((indices[FLD.BOLL_CROSS_JX_BEFORE] < indices[FLD.BOLL_CROSS_SX_BEFORE]) & \
            (indices[FLD.BOOTSTRAP_I_BEFORE] < indices[FLD.MACD_CROSS_SX_BEFORE])))
        trigger_rps = np.where(trigger_rps_08 == True, 8, trigger_rps)

        trigger_rps_09 = ((indices[FLD.MAXFACTOR_CROSS_JX] < 6) | \
            ((indices[FLD.MACD_TREND] > 0) & \
            (indices[FLD.MACD_TREND].shift(1) > 0))) & \
            (indices[FLD.LINEAREG_PRICE] > indices[FLD.MA90]) & \
            ((indices[FLD.MA90] > indices[FLD.MA120]) | \
            ((indices[FLD.LINEAREG_PRICE] > indices[FLD.MA120]) & \
            (indices[FLD.MA30_SLOPE] > indices[FLD.BASELINE_SLOPE]))) & \
            (((indices[FLD.MAXFACTOR_CROSS_JX] < 4) | \
            (indices[FLD.MACD] > 0) | \
            ((indices[FLD.BIAS3].shift(1) < 0)) & \
            (indices[FLD.BIAS3] > 0) & \
            (indices[FLD.BIAS3].shift(1) < indices[FLD.BIAS3]))) & \
            (indices[FLD.ZSCORE_21].shift(1) < 0.382) & \
            (indices[FLD.ZSCORE_84].shift(1) < 0.382) & \
            (indices[FLD.ZSCORE_21].shift(1) <= indices[FLD.ZSCORE_21]) & \
            (indices[FLD.ZSCORE_84].shift(1) <= indices[FLD.ZSCORE_84]) & \
            (((indices[FLD.BOLL_CROSS_JX_BEFORE] > 36) & \
            (indices[FLD.DIF] > 0)) | \
            ((indices[FLD.BOLL_CROSS_JX_BEFORE] <= 2) & \
            (indices[FLD.ZEN_WAVELET_CROSS_SX_BEFORE] > 18) & \
            (indices[FLD.DIF] > 0)) | \
            ((indices[FLD.DEA_CROSS_JX_BEFORE] > 36) & \
            (indices[FLD.MA90_CLEARANCE] > 1.236) & \
            (indices[FLD.DIF] > 0)) | \
            ((indices[FLD.BOLL_CROSS_JX_BEFORE] > indices[FLD.DEA_CROSS_JX_BEFORE]) & \
            (indices[FLD.MA90_CLEARANCE] > 1.236) & \
            (indices[FLD.DIF] > 0)) | \
            ((indices[FLD.BOLL_CROSS_JX_BEFORE] < indices[FLD.BOLL_CROSS_SX_BEFORE]) & \
            (indices[FLD.BOOTSTRAP_I_BEFORE] < indices[FLD.MACD_CROSS_SX_BEFORE])))
        trigger_rps = np.where(trigger_rps_09 == True, 9, trigger_rps)
        
    trigger_rps_ne_01 = ((indices[FLD.ATR_CROSS] <= 0) & \
            ~((indices[FLD.MAXFACTOR_CROSS] > 0) & \
            (indices[FLD.BIAS3] > 0) & \
            (indices[FLD.ZSCORE_21].shift(1) <= indices[FLD.ZSCORE_21]) & \
            (indices[FLD.ZSCORE_84].shift(1) <= indices[FLD.ZSCORE_84])) & \
            ~((indices[FLD.DUAL_CROSS_JX] > 0) & \
            (indices[FLD.VOLUME_FLOW_TRI_CROSS] > 0)) & \
            (indices[FLD.MACD] > 0) & \
            (indices[FLD.MACD_DELTA] < 0) & \
            (indices[FLD.ZEN_WAVELET_CROSS_SX_BEFORE] < indices[FLD.ZEN_WAVELET_CROSS_JX_BEFORE]) & \
            (indices[FLD.BIAS3_ZSCORE].shift(1) < 0.96))
    trigger_rps_ne = np.where(trigger_rps_ne_01 == True, -1, 0)

    trigger_rps_ne_02 = (indices[FLD.DEA] > 0) & \
        (indices[FLD.MA5] > indices[FLD.MA20]) & \
        (indices[FLD.MA10] > indices[FLD.MA20]) & \
        (indices[FLD.MA20] > indices[FLD.MA30]) & \
        (indices[FLD.MA30] > indices[FLD.MA90]) & \
        (indices[FLD.PCT_CHANGE].shift(1) < 0) & \
        ((indices[FLD.PCT_CHANGE].shift(2) < 0) | \
        (indices[FLD.PCT_CHANGE].shift(3) < 0) | \
        (indices[FLD.PCT_CHANGE].shift(4) < 0)) & \
        (indices[FLD.MACD_DELTA] < 0) & \
        ~((indices[FLD.MACD_CROSS_SX_BEFORE] < 4) & \
        (indices[FLD.ZSCORE_21] > 0.809) & \
        (indices[FLD.ZSCORE_84] > 0.809)) & \
        ((indices[FLD.ATR_CROSS_SX_BEFORE] < indices[FLD.ATR_CROSS_JX_BEFORE]) | \
        (indices[FLD.BOLL_DELTA] < 0) | \
        (indices[FLD.MAXFACTOR_CROSS] < 0)) & \
        (indices[FLD.BOLL_CROSS_SX_BEFORE] < indices[FLD.BOLL_CROSS_JX_BEFORE]) & \
        ((indices[FLD.BIAS3_ZSCORE] > 0.168) | \
        (indices[FLD.BIAS3_ZSCORE] < 0.0512))
    trigger_rps_ne = np.where(trigger_rps_ne_02 == True, -2, trigger_rps_ne)

    trigger_rps_ne_03 = (indices[FLD.DEA] > 0) & \
        (indices[FLD.MA5] > indices[FLD.MA20]) & \
        (indices[FLD.MA10] > indices[FLD.MA20]) & \
        (indices[FLD.MA20] > indices[FLD.MA30]) & \
        (indices[FLD.MA30] > indices[FLD.MA90]) & \
        (indices[FLD.PCT_CHANGE].shift(2) < 0) & \
        ((indices[FLD.PCT_CHANGE].shift(3) < 0) | \
        (indices[FLD.PCT_CHANGE].shift(4) < 0) | \
        (indices[FLD.PCT_CHANGE].shift(5) < 0)) & \
        (indices[FLD.MACD_DELTA] < 0) & \
        ((indices[FLD.ATR_CROSS_SX_BEFORE] < indices[FLD.ATR_CROSS_JX_BEFORE]) | \
        (indices[FLD.BOLL_DELTA] < 0) | \
        (indices[FLD.MAXFACTOR_CROSS] < 0)) & \
        (indices[FLD.BOLL_CROSS_SX_BEFORE] < indices[FLD.BOLL_CROSS_JX_BEFORE]) & \
        ((indices[FLD.BIAS3_ZSCORE] > 0.168) | \
        (indices[FLD.BIAS3_ZSCORE] < 0.0512))
    trigger_rps_ne = np.where(trigger_rps_ne_03 == True, -3, trigger_rps_ne)

    trigger_rps_ne_04 = ((indices[FLD.RENKO_TREND_S] < 0) & \
            ~((indices[FLD.DUAL_CROSS_JX] > 0) & \
            (indices[FLD.VOLUME_FLOW_TRI_CROSS] > 0)) & \
            ~((indices[FLD.BIAS3].shift(1) < 0) & \
            (indices[FLD.BIAS3] > 0)) & \
            ~(indices[FLD.PCT_CHANGE] > 0) & \
            ~(trigger_rps == True) & \
            ~((indices[FLD.BIAS3].shift(1) < indices[FLD.BIAS3]) & \
            (indices[FLD.ZSCORE_21].shift(1) < indices[FLD.ZSCORE_21]) & \
            (indices[FLD.ZSCORE_84].shift(1) < indices[FLD.ZSCORE_84])) & \
            (indices[FLD.DUAL_CROSS_JX] < 1) & \
            (indices[FLD.MACD] > 0) & \
            (((indices[FLD.RENKO_S_SX_BEFORE] < 8) & \
            (indices[FLD.ZEN_WAVELET_CROSS_SX_BEFORE] < 24)) | \
            (indices[FLD.ZEN_WAVELET_CROSS_SX_BEFORE] < indices[FLD.ZEN_WAVELET_CROSS_JX_BEFORE])))
    trigger_rps_ne = np.where(trigger_rps_ne_04 == True, -4, trigger_rps_ne)

    trigger_rps_ne_05 = (indices[FLD.DEA] > 0) & \
        (indices[FLD.MA5] > indices[FLD.MA20]) & \
        (indices[FLD.MA10] > indices[FLD.MA20]) & \
        (indices[FLD.MA20] > indices[FLD.MA30]) & \
        (indices[FLD.MA30] > indices[FLD.MA90]) & \
        (indices[FLD.MAXFACTOR_CROSS].shift(1) < 0) & \
        ((indices[FLD.MAXFACTOR_CROSS].shift(2) < 0) | \
        (indices[FLD.MAXFACTOR_CROSS].shift(3) < 0) | \
        (indices[FLD.MAXFACTOR_CROSS].shift(4) < 0)) & \
        (indices[FLD.MAXFACTOR_CROSS] < 0) & \
        ((indices[FLD.ATR_CROSS_SX_BEFORE] < indices[FLD.ATR_CROSS_JX_BEFORE]) | \
        (indices[FLD.BOLL_DELTA] < 0) | \
        (indices[FLD.MAXFACTOR_CROSS] < 0)) & \
        (indices[FLD.BOLL_CROSS_SX_BEFORE] < indices[FLD.BOLL_CROSS_JX_BEFORE]) & \
        ((indices[FLD.BIAS3_ZSCORE] > 0.168) | \
        (indices[FLD.BIAS3_ZSCORE] < 0.0512)) & \
        ~((indices[FLD.BIAS3].shift(1) < indices[FLD.BIAS3]) & \
        (indices[FLD.BIAS3_ZSCORE].shift(1) < indices[FLD.BIAS3_ZSCORE]) & \
        (indices[FLD.ZSCORE_21].shift(1) < indices[FLD.ZSCORE_21]) & \
        (indices[FLD.ZSCORE_84].shift(1) < indices[FLD.ZSCORE_84]))
    trigger_rps_ne = np.where(trigger_rps_ne_05 == True, -5, trigger_rps_ne)

    trigger_rps_ne_06 = (indices[FLD.DEA] > 0) & \
        (indices[FLD.MA5] > indices[FLD.MA20]) & \
        (indices[FLD.MA10] > indices[FLD.MA20]) & \
        (indices[FLD.MA20] > indices[FLD.MA30]) & \
        (indices[FLD.MA30] > indices[FLD.MA90]) & \
        (indices[FLD.MAXFACTOR_CROSS].shift(2) < 0) & \
        ((indices[FLD.MAXFACTOR_CROSS].shift(3) < 0) | \
        (indices[FLD.MAXFACTOR_CROSS].shift(4) < 0) | \
        (indices[FLD.MAXFACTOR_CROSS].shift(5) < 0)) & \
        (indices[FLD.MAXFACTOR_CROSS] < 0) & \
        ((indices[FLD.ATR_CROSS_SX_BEFORE] < indices[FLD.ATR_CROSS_JX_BEFORE]) | \
        (indices[FLD.BOLL_DELTA] < 0) | \
        (indices[FLD.MAXFACTOR_CROSS] < 0)) & \
        (indices[FLD.BOLL_CROSS_SX_BEFORE] < indices[FLD.BOLL_CROSS_JX_BEFORE]) & \
        ((indices[FLD.BIAS3_ZSCORE] > 0.168) | \
        (indices[FLD.BIAS3_ZSCORE] < 0.0512)) & \
        ~((indices[FLD.BIAS3].shift(1) < indices[FLD.BIAS3]) & \
        (indices[FLD.BIAS3_ZSCORE].shift(1) < indices[FLD.BIAS3_ZSCORE]) & \
        (indices[FLD.ZSCORE_21].shift(1) < indices[FLD.ZSCORE_21]) & \
        (indices[FLD.ZSCORE_84].shift(1) < indices[FLD.ZSCORE_84]))
    trigger_rps_ne = np.where(trigger_rps_ne_06 == True, -6, trigger_rps_ne)

    trigger_rps_ne_07 = (indices[FLD.DEA] > 0) & \
        (indices[FLD.BIAS3] < 0) & \
        (indices[FLD.MACD] < 0) & \
        (indices[FLD.MACD_DELTA] < 0) & \
        (indices[FLD.MAXFACTOR_CROSS].shift(1) < 0) & \
        ((indices[FLD.MAXFACTOR_CROSS].shift(2) < 0) | \
        (indices[FLD.MAXFACTOR_CROSS].shift(3) < 0) | \
        (indices[FLD.MAXFACTOR_CROSS].shift(4) < 0)) & \
        (indices[FLD.MAXFACTOR_CROSS] < 0) & \
        (indices[FLD.LINEAREG_CROSS_SX_BEFORE] < indices[FLD.LINEAREG_CROSS_JX_BEFORE]) & \
        ((indices[FLD.BIAS3_ZSCORE] > 0.168) | \
        (indices[FLD.BIAS3_ZSCORE] < 0.0512)) & \
        ~((indices[FLD.BIAS3].shift(1) < indices[FLD.BIAS3]) & \
        (indices[FLD.BIAS3_ZSCORE].shift(1) < indices[FLD.BIAS3_ZSCORE]) & \
        (indices[FLD.ZSCORE_21].shift(1) < indices[FLD.ZSCORE_21]) & \
        (indices[FLD.ZSCORE_84].shift(1) < indices[FLD.ZSCORE_84]))
    trigger_rps_ne = np.where(trigger_rps_ne_07 == True, -7, trigger_rps_ne)

    trigger_rps_ne = np.where((trigger_rps_ne < 0) & \
                        (indices[FLD.MAXFACTOR] > 0) & \
                        (indices[FLD.BIAS3] > 0) & \
                        (indices[FLD.MACD] > 0) & \
                        (indices[FLD.ZEN_WAVELET_CROSS_SX_BEFORE] > 4) & \
                        ((indices[FLD.RENKO_TREND_S_UB].shift(1) < indices[FLD.RENKO_TREND_S_UB]) | \
                        ((indices[FLD.RENKO_TREND_S_UB] < data[AKA.CLOSE]) & \
                        (indices[FLD.RENKO_TREND_S_UB] > data[AKA.CLOSE].rolling(3).min()))), -666,
                        trigger_rps_ne)

    indices[ST.TRIGGER_RPS] = np.where(trigger_rps > 0, trigger_rps, 
                                       np.where(trigger_rps_ne < 0, trigger_rps_ne, 0))

    # 计算 RPS PUNCH TIME LAG
    rps_cross_jx = Timeline_duration(np.where(indices[ST.TRIGGER_RPS] > 0, 1, 0))
    rps_cross_sx = Timeline_duration(np.where(indices[ST.TRIGGER_RPS] < 0, 1, 0))
    position_rps = np.where((rps_cross_sx - rps_cross_jx) > 0, 1, 
                            np.where((rps_cross_jx - rps_cross_sx) > 0, -1, 0))
    rps_cross_jx = Timeline_Integral(np.where(position_rps > 0, 1, 0))
    rps_cross_sx = np.sign(position_rps) * Timeline_Integral(np.where(position_rps < 0, 1, 0))
    indices[FLD.RPS_PUNCH_TIMING_LAG] = rps_cross_jx + rps_cross_sx

    if (ST.VERBOSE in data.columns):
        if (code == '002230'):
            check_point = pd.concat([data[AKA.CLOSE],
                                     (indices[FLD.MA90_CLEARANCE] > -0.618) & \
        (indices[FLD.MA90_CLEARANCE] < 0.309) & \
        (indices[FLD.MAXFACTOR_CROSS] > 0) & \
        (indices[FLD.MAXFACTOR_CROSS_JX] <= 8),
        (indices[FLD.BIAS3] > -9.27) & \
        (indices[FLD.BIAS3].shift(1) < 0) & \
        (indices[FLD.BIAS3].shift(1) < indices[FLD.BIAS3]),
        (indices[FLD.BIAS3_ZSCORE].shift(1) < 0.809) & \
        (indices[FLD.BIAS3_ZSCORE].shift(1) < indices[FLD.BIAS3_ZSCORE]),
        (indices[FLD.ZSCORE_21].shift(1) < 0.809) & \
        (indices[FLD.ZSCORE_21].shift(1) < indices[FLD.ZSCORE_21]),
        (indices[FLD.ZSCORE_84].shift(1) < 0.512) & \
        (indices[FLD.ZSCORE_84].shift(1) < indices[FLD.ZSCORE_84]),
        (indices[FLD.MACD_DELTA] > 0),
        ((indices[FLD.BOLL_DELTA] > 0) | \
        (indices[FLD.MACD_CROSS_JX_BEFORE] < 2)),
        ((indices[FLD.LINEAREG_CROSS_JX_BEFORE] < indices[FLD.LINEAREG_CROSS_SX_BEFORE]) | \
        (indices[FLD.MAXFACTOR_CROSS_JX] < 4)),], 
                                                                axis = 1)
            print('check_point:\n', check_point.tail(120))

    return indices


if __name__ == '__main__':
    pd.set_option('display.float_format',lambda x : '%.3f' % x)
    pd.set_option('display.max_columns', 15)
    pd.set_option("display.max_rows", 160)
    pd.set_option('display.width', 180)  # 设置打印宽度

    etflist = ['159919', '159997', '159805', '159987', 
               '159952', '159920', '518880', '159934', 
               '159985', '515050', '159994', '159941', 
               '512800', '515000', '512170', '512980', 
               '510300', '513100', '510900', '512690', 
               '510050', '159916', '512910', '510310', 
               '512090', '513050', '513030', '513500', 
               '159905', '159949', '510330', '510500', 
               '510180', '159915', '510810', '159901', 
               '512710', '510850', '512500', '512000',]

    from GolemQ.fetch.kline import (
        get_kline_price,
    )

    # '501021' 香港中小LOF(SH:501021)
    data_day, codename = get_kline_price(etflist)

    start_t = datetime.datetime.now()
    print(start_t)

    rps_returns = data_day.add_func(calc_rps_returns_roc_func, w=63)
    rps120 = all_RPS_rank(rps_returns)

    indices = data_day.add_func(lineareg_cross_func, rps120)

    end_t = datetime.datetime.now()
    print(end_t, 'spent:{}'.format((end_t - start_t)))
    print(indices[[FLD.RPS_RETURNS, 
                   FLD.RPS_RANK,  
                   FLD.RPS_PCT,
                   FLD.ZSCORE_21,
                   FLD.MAXFACTOR_CROSS,
                   FLD.DUAL_CROSS,
                   FLD.VOLUME_FLOW_TRI_CROSS,]].tail(120))
