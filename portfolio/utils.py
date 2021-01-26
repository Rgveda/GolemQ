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
import numpy as np
import numba as nb
from numba import vectorize, float64, jit as njit
import scipy.stats as scs
from datetime import datetime as dt, timezone, timedelta

from GolemQ.analysis.timeseries import *
from QUANTAXIS.QAUtil.QADate_Adv import (QA_util_print_timestamp,)
import pandas as pd
import empyrical
from GolemQ.utils.parameter import (
    AKA, 
    INDICATOR_FIELD as FLD, 
    TREND_STATUS as ST,
    FEATURES as FTR,
    )

"""
这里定义的是一些 fractal, strategy, portfolio 的绩效统计相关工具函数
"""

def calc_fractal_stats(symbol, display_name, fractal_uid, fractal_triggers,
                  ref_features=None, rsk_fre=0.04, annual=252, taxfee=0.0003, 
                  long=1, mode='lrc', format='pd'):
    """
    explanation:
        分型触发点(卖点or买点，取决于交易方向)绩效分析

        什么是分型触发点呢？对大众认知的交易里，最通俗的含义就是特定交易策略的金叉/死叉点。
        例如MACD金叉/死叉点，双MA金叉、死叉点，KDJ金叉、死叉点等。
        它们全部都可以纳入这套 fractal_stats 评估体系中衡量。

        它的功能用法有点像 alphalens，但是 alphalens 使用的框架体系是传统经济学量化金融
        的那一套理论，跟我朴素的衡量标准并不一致。比如 alpha_beta 只计算当前 bar。
        在我的交易系统内，并没有发现我认知的“趋势买点”和传统金融量化理论计算的 IC alpha值有
        什么显著的相关性特征。
        所以我只能另起炉灶，用统计学习的方式来量化衡量一个分型触发点的优劣。
        这套分型系统的数学涵义上更类似深度学习中Attention模型的注意力关注点，能否与机器学习
        结合有待后续研究。

    params:
        symbol: str, 交易标的代码
        display_name: str, 交易标的显示名称
        fractal_uid: str, 分型唯一标识编码
        fractal_triggers : np.array, 分型触发信号
        ref_features: np.array, 参考指标特征
        rsk_fre: float32, 无风险利率
        annual: int32, 年化周期
        taxfee: float32, 税费
        long: int32, 交易方向
        mode: str, 'lrc' or 'zen' 趋势判断模式为 zen趋势 或者 lrc回归线趋势，
                   'hmapower'为追踪hmapower120的MA逆序趋势，
                   'raw'不进行趋势判断，完全跟随 fractal_triggers 状态信号
        format : string, 返回格式

    return:
        pd.Series or np.array or string
    """
    # 这里严格定义应该是考虑交易方向，但是暂时先偷懒简化了计算，以后做双向策略出现问题再完善
    if (long > 0):
        # 做多方向
        fractal_cross_before = Timeline_duration(np.where(fractal_triggers > 0, 1, 0))
    else:
        # 做空方向
        fractal_cross_before = Timeline_duration(np.where(fractal_triggers < 0, 1, 0))

    if (annual > 125) and (annual < 366):
        # 推断为日线级别的数据周期
        fractal_forcast_position = np.where(fractal_cross_before < 3, 1, 0)
        fractal_limited = 3
    elif ((annual > 1680) and (annual < 2560)):
        # 推断为数字币 4小时先级别的数据周期
        fractal_limited = 24
        fractal_forcast_position = np.where(fractal_cross_before < 24, 1, 0)
    elif ((annual > 512) and (annual < 1280)):
        # 推断为股票/证券1小时先级别的数据周期
        fractal_limited = 12
        fractal_forcast_position = np.where(fractal_cross_before < 12, 1, 0)
    elif ((annual > 6180) and (annual < 9600)):
        # 推断为股票/证券1小时先级别的数据周期
        fractal_limited = 72
        fractal_forcast_position = np.where(fractal_cross_before < 72, 1, 0)

    # 固定统计3交易日内收益
    fractal_forcast_3d_lag = calc_event_timing_lag(np.where(fractal_forcast_position > 0, 1, -1))
    fractal_forcast_3d_lag = np.where(fractal_forcast_3d_lag <= fractal_limited, fractal_forcast_3d_lag, 0)
    closep = ref_features[AKA.CLOSE].values

    if (mode == 'lrc'):
        # 统计到下一次 lineareg_band / 死叉等对等交易信号结束时的 收益，时间长度不固定
        if (long > 0):
            # 做多方向
            lineareg_endpoint_before = Timeline_duration(np.where(ref_features[FLD.LINEAREG_BAND_TIMING_LAG] == -1, 1, 0))
        else:
            # 做空方向
            lineareg_endpoint_before = Timeline_duration(np.where(ref_features[FLD.LINEAREG_BAND_TIMING_LAG] == 1, 1, 0))
        fractal_lineareg_position = np.where(fractal_cross_before < lineareg_endpoint_before, 1, 0)
        fractal_lineareg_lag = calc_event_timing_lag(np.where(fractal_lineareg_position > 0, 1, -1))

        transcation_stats = calc_transcation_stats(fractal_triggers,
                                                   closep,
                                                   fractal_forcast_3d_lag,
                                                   fractal_lineareg_lag,
                                                   ref_features[FLD.LINEAREG_BAND_TIMING_LAG].values,
                                                   taxfee=taxfee,
                                                   long=long)

        transcation_stats_df = pd.DataFrame(transcation_stats, columns=['trans_stats',
                                                                        'trans_start',
                                                                        'trans_act',
                                                                        'trans_end', 
                                                                        'start_principle', 
                                                                        'ret_3d',
                                                                        'ret_fractal',
                                                                        'pric_settle',
                                                                        'trans_3d',
                                                                        'price_end_3d',
                                                                        'price_end_fractal',
                                                                        'ret_fractal_sim',
                                                                        'long',
                                                                        'duration_time',])
    elif (mode == 'zen') or \
        (mode == 'mapower'):
        if (long > 0):
            # 做多方向
            zen_wavelet_endpoint_before = Timeline_duration(np.where(ref_features[FLD.ZEN_WAVELET_TIMING_LAG] == -1, 1, 0))
        else:
            # 做空方向
            zen_wavelet_endpoint_before = Timeline_duration(np.where(ref_features[FLD.ZEN_WAVELET_TIMING_LAG] == 1, 1, 0))
        fractal_zen_wavelet_position = np.where(fractal_cross_before < zen_wavelet_endpoint_before, 1, 0)
        fractal_zen_wavelet_lag = calc_event_timing_lag(np.where(fractal_zen_wavelet_position > 0, 1, -1))
        transcation_stats = calc_transcation_stats_np(fractal_triggers,
                                                   closep,
                                                   fractal_forcast_3d_lag,
                                                   fractal_zen_wavelet_lag,
                                                   taxfee=taxfee,
                                                   long=long)

        transcation_stats_df = pd.DataFrame(transcation_stats, columns=['trans_stats',
                                                                        'trans_start',
                                                                        'trans_act',
                                                                        'trans_end', 
                                                                        'start_principle', 
                                                                        'ret_3d',
                                                                        'ret_fractal',
                                                                        'pric_settle',
                                                                        'trans_3d',
                                                                        'price_end_3d',
                                                                        'price_end_fractal',
                                                                        'ret_fractal_sim',
                                                                        'long',
                                                                        'duration_time',])
    elif (mode == 'hmapower') or \
        (mode == 'hmapower120') or \
        (mode == 'hmapower30'):
        if (long > 0):
            # 做多方向
            hmapower120_endpoint_before = Timeline_duration(np.where(ref_features[FLD.HMAPOWER120_TIMING_LAG] == -1, 1, 0))
        else:
            # 做空方向
            hmapower120_endpoint_before = Timeline_duration(np.where(ref_features[FLD.HMAPOWER120_TIMING_LAG] == 1, 1, 0))
        fractal_hmapower120_position = np.where(fractal_cross_before < hmapower120_endpoint_before, 1, 0)
        fractal_hmapower120_lag = calc_event_timing_lag(np.where(fractal_hmapower120_position > 0, 1, -1))
        transcation_stats = calc_transcation_stats_np(fractal_triggers,
                                                   closep,
                                                   fractal_forcast_3d_lag,
                                                   fractal_hmapower120_lag,
                                                   taxfee=taxfee,
                                                   long=long)

        transcation_stats_df = pd.DataFrame(transcation_stats, columns=['trans_stats',
                                                                        'trans_start',
                                                                        'trans_act',
                                                                        'trans_end', 
                                                                        'start_principle', 
                                                                        'ret_3d',
                                                                        'ret_fractal',
                                                                        'pric_settle',
                                                                        'trans_3d',
                                                                        'price_end_3d',
                                                                        'price_end_fractal',
                                                                        'ret_fractal_sim',
                                                                        'long',
                                                                        'duration_time',])
    elif (mode == 'raw'):
        fractal_position = np.where(fractal_triggers > 0, 1, 0)
        fractal_timing_lag = calc_event_timing_lag(np.where(fractal_position > 0, 1, -1))
        if (np.max(fractal_timing_lag) < 12):
            #print('A spot Fractal, not a Complete Cycle Fractal')
            pass
        transcation_stats = calc_transcation_stats_np(fractal_triggers,
                                                      closep,
                                                      fractal_forcast_3d_lag,
                                                      fractal_timing_lag,
                                                      taxfee=taxfee,
                                                      long=long)

        transcation_stats_df = pd.DataFrame(transcation_stats, columns=['trans_stats',
                                                                        'trans_start',
                                                                        'trans_act',
                                                                        'trans_end', 
                                                                        'start_principle', 
                                                                        'ret_3d',
                                                                        'ret_fractal',
                                                                        'pric_settle',
                                                                        'trans_3d',
                                                                        'price_end_3d',
                                                                        'price_end_fractal',
                                                                        'ret_fractal_sim',
                                                                        'long',
                                                                        'duration_time',])

    transcation_stats_df[AKA.CODE] = symbol
    transcation_stats_df['fractal_uid'] = fractal_uid

    # bar ID索引 转换成交易时间戳
    selected_trans_start = ref_features.iloc[transcation_stats[:, 1], :]
    transcation_stats_df['trans_start'] = pd.to_datetime(selected_trans_start.index.get_level_values(level=0))
    selected_trans_action = ref_features.iloc[transcation_stats[:, 2], :]
    transcation_stats_df['trans_act'] = pd.to_datetime(selected_trans_action.index.get_level_values(level=0))
    selected_trans_end = ref_features.iloc[transcation_stats[:, 3], :]
    transcation_stats_df['trans_end'] = pd.to_datetime(selected_trans_end.index.get_level_values(level=0))

    transcation_stats_df = transcation_stats_df.assign(datetime=pd.to_datetime(selected_trans_start.index.get_level_values(level=0))).drop_duplicates((['datetime',
                                'code'])).set_index(['datetime',
                                'code'],
                                    drop=True)

    return transcation_stats_df


@nb.jit(nopython=True)
def calc_transcation_stats(fractal_triggers: np.ndarray, 
                           closep: np.ndarray,
                           fractal_forcast_position: np.ndarray,
                           fractal_sim_position: np.ndarray,
                           principle_timing_lag: np.ndarray,
                           taxfee: float=0.0003, 
                           long: int=1):

    """
    explanation:
        在“大方向”（规则）引导下，计算当前交易盈亏状况
        np.ndarray 实现，编码规范支持JIT和Cython加速

    params:
        fractal_triggers : np.array, 分型触发信号
        closep: np.array, 参考指标特征
        fractal_forcast_position: np.ndarray,
        fractal_principle_position: np.ndarray,
        principle_timing_lag:np.ndarray,
        taxfee: float32, 税费
        long: int32, 交易方向

    return:
        np.array
    """
    # 交易状态，状态机规则，低状态可以向高状态迁移
    stats_nop = 0            # 无状态
    stats_onhold = 1         # 执行交易并持有
    stats_suspended = 2      # 挂起，不执行交易，观察走势
    stats_closed = 3         # 结束交易
    stats_teminated = 4      # 趋势走势不对，终止交易

    idx_transcation = -1
    idx_transcation_stats = 0
    idx_transcation_start = 1
    idx_transcation_action = 2
    idx_transcation_endpoint = 3
    idx_start_in_principle = 4
    idx_forcast_returns = 5
    idx_principle_returns = 6
    idx_settle_price = 7
    idx_transcation_3d = 8
    idx_endpoint_price_3d = 9
    idx_endpoint_price_principle = 10
    idx_fractal_sim_returns = 11
    idx_long = 12
    idx_duration_time = 13
    #idx_lineareg_band_lag = 12
 
    ret_transcation_stats = np.zeros((len(closep), 14))
    onhold_price = onhold_returns = 0.0
    onhold_position_3d = onhold_position_lineareg = False
    assert long == 1 or long == -1
    ret_transcation_stats[:, idx_long] = long
    for i in range(0, len(closep)):
        # 开启交易判断
        if (fractal_triggers[i] > 0) and \
            (not onhold_position_3d) and \
            (not onhold_position_lineareg):
            onhold_position_3d = True
            onhold_position_lineareg = True
            idx_transcation = idx_transcation + 1
            ret_transcation_stats[idx_transcation, idx_transcation_start] = i

            if (principle_timing_lag[i] * long > 0):
                ret_transcation_stats[idx_transcation, 
                                      idx_start_in_principle] = principle_timing_lag[i]
                if (ret_transcation_stats[idx_transcation, 
                                          idx_start_in_principle] * long == -1):
                    ret_transcation_stats[idx_transcation, 
                                          idx_transcation_stats] = stats_suspended
                elif (ret_transcation_stats[idx_transcation, 
                                            idx_transcation_stats] < stats_onhold):
                    ret_transcation_stats[idx_transcation, 
                                          idx_transcation_stats] = stats_onhold
                    ret_transcation_stats[idx_transcation, 
                                          idx_transcation_action] = i
            else:
                ret_transcation_stats[idx_transcation, 
                                      idx_start_in_principle] = principle_timing_lag[i]
                if (ret_transcation_stats[idx_transcation, 
                                          idx_transcation_stats] < stats_suspended):
                    ret_transcation_stats[idx_transcation, 
                                          idx_transcation_stats] = stats_suspended

            if (principle_timing_lag[i] * long > 0):
                if (int(ret_transcation_stats[idx_transcation, 
                                              idx_transcation_stats]) == stats_onhold):
                    onhold_price = closep[i]
                    ret_transcation_stats[idx_transcation, 
                                          idx_settle_price] = onhold_price
            elif (i != len(closep)):
                if (int(ret_transcation_stats[idx_transcation, 
                                              idx_transcation_stats]) == stats_onhold):
                    onhold_price = closep[i + 1]
                    ret_transcation_stats[idx_transcation, 
                                          idx_settle_price] = onhold_price

        if (onhold_position_lineareg) and (fractal_forcast_position[i] > 0):
            if (principle_timing_lag[i] * long > 0):
                if (int(ret_transcation_stats[idx_transcation, 
                                              idx_transcation_stats]) == stats_suspended):
                    ret_transcation_stats[idx_transcation, 
                                          idx_transcation_stats] = stats_onhold
                    ret_transcation_stats[idx_transcation, 
                                          idx_transcation_action] = i
                    onhold_price = closep[i]
                    ret_transcation_stats[idx_transcation, 
                                          idx_settle_price] = onhold_price
            else:
                ret_transcation_stats[idx_transcation, 
                                      idx_transcation_stats] = stats_suspended

        # 结束交易判断
        if (onhold_position_lineareg) and (fractal_sim_position[i] <= 0):
            onhold_position_lineareg = False
            onhold_position_3d = False
            ret_transcation_stats[idx_transcation, 
                                  idx_transcation_endpoint] = i
            onhold_sim_price = closep[int(ret_transcation_stats[idx_transcation, idx_transcation_start])]
            ret_transcation_stats[idx_transcation, 
                                  idx_fractal_sim_returns] = (closep[i] - onhold_sim_price) / onhold_sim_price * long
            if (int(ret_transcation_stats[idx_transcation, 
                                          idx_transcation_stats]) == stats_onhold):
                ret_transcation_stats[idx_transcation, 
                                      idx_principle_returns] = (closep[i] - onhold_price) / onhold_price * long
                ret_transcation_stats[idx_transcation, 
                                      idx_endpoint_price_principle] = closep[i]
                ret_transcation_stats[idx_transcation, 
                                      idx_transcation_stats] = stats_closed
                onhold_price = 0.0
            elif (int(ret_transcation_stats[idx_transcation, 
                                            idx_transcation_stats]) == stats_suspended):
                ret_transcation_stats[idx_transcation, 
                                      idx_transcation_stats] = stats_teminated
                onhold_price = 0.0

        if (onhold_position_3d) and (fractal_forcast_position[i] <= 0):
            onhold_position_3d = False
            ret_transcation_stats[idx_transcation, 
                                  idx_transcation_3d] = principle_timing_lag[i]
            if (int(ret_transcation_stats[idx_transcation, 
                                          idx_transcation_stats]) == stats_onhold):
                ret_transcation_stats[idx_transcation, 
                                      idx_forcast_returns] = (closep[i] - onhold_price) / onhold_price * long
                ret_transcation_stats[idx_transcation, 
                                      idx_endpoint_price_3d] = closep[i]
            elif (int(ret_transcation_stats[idx_transcation, 
                                            idx_transcation_stats]) == stats_suspended):
                ret_transcation_stats[idx_transcation, 
                                      idx_transcation_stats] = stats_teminated
                onhold_price = 0.0
            else:
                pass

        if (onhold_position_lineareg) and (i == len(closep)):
            # 交易当前处于未结束状态
            if (int(ret_transcation_stats[idx_transcation, 
                                          idx_transcation_stats]) == stats_onhold):
                ret_transcation_stats[idx_transcation, 
                                      idx_principle_returns] = (closep[i] - onhold_price) / onhold_price * long
            pass
        ret_transcation_stats[idx_transcation, 
                              idx_duration_time] = ret_transcation_stats[idx_transcation, 
                                                                         idx_transcation_endpoint] - ret_transcation_stats[idx_transcation, 
                                                                                                                           idx_transcation_action]

    return ret_transcation_stats[:idx_transcation + 1, :]


@nb.jit(nopython=True)
def calc_transcation_stats_np(fractal_triggers:np.ndarray, 
                              closep:np.ndarray,
                              fractal_forcast_position:np.ndarray,
                              fractal_timing_lag:np.ndarray,
                              taxfee:float=0.0003, 
                              long:int=1):

    """
    计算当前交易盈亏状况
    np.ndarray 实现，编码规范支持JIT和Cython加速
    """
    # 交易状态，状态机规则，低状态可以向高状态迁移
    stats_nop = 0            # 无状态
    stats_onhold = 1         # 执行交易并持有
    stats_suspended = 2      # 挂起，不执行交易，观察走势
    stats_closed = 3         # 结束交易
    stats_teminated = 4      # 趋势走势不对，终止交易

    idx_transcation = -1
    idx_transcation_stats = 0
    idx_transcation_start = 1
    idx_transcation_action = 2
    idx_transcation_endpoint = 3
    idx_start_zen_wavelet = 4
    idx_forcast_returns = 5
    idx_fractal_returns = 6
    idx_settle_price = 7
    idx_transcation_3d = 8
    idx_endpoint_price_3d = 9
    idx_endpoint_price_fractal = 10
    idx_fractal_sim_returns = 11
    idx_long = 12
    idx_duration_time = 13
    #idx_lineareg_band_lag = 12
 
    ret_transcation_stats = np.zeros((len(closep), 14))
    onhold_price = onhold_returns = 0.0
    onhold_position_3d = onhold_position_lineareg = False
    assert long == 1 or long == -1
    ret_transcation_stats[:, idx_long] = long
    for i in range(0, len(closep)):
        # 开启交易判断
        if (fractal_triggers[i] > 0) and \
            (not onhold_position_3d) and \
            (not onhold_position_lineareg):
            onhold_position_3d = True
            onhold_position_lineareg = True
            idx_transcation = idx_transcation + 1
            ret_transcation_stats[idx_transcation, idx_transcation_start] = i
            ret_transcation_stats[idx_transcation, 
                                  idx_transcation_stats] = stats_onhold
            ret_transcation_stats[idx_transcation, idx_transcation_action] = i
            onhold_price = closep[i]
            ret_transcation_stats[idx_transcation, 
                                  idx_settle_price] = onhold_price

        # 结束交易判断
        if (onhold_position_lineareg) and (fractal_timing_lag[i] <= 0):
            onhold_position_lineareg = False
            onhold_position_3d = False
            ret_transcation_stats[idx_transcation, 
                                  idx_transcation_endpoint] = i
            ret_transcation_stats[idx_transcation, 
                                  idx_fractal_sim_returns] = (closep[i] - onhold_price) / onhold_price * long
            ret_transcation_stats[idx_transcation, 
                                  idx_fractal_returns] = (closep[i] - onhold_price) / onhold_price * long
            ret_transcation_stats[idx_transcation, 
                                  idx_endpoint_price_fractal] = closep[i]
            ret_transcation_stats[idx_transcation, 
                                  idx_transcation_stats] = stats_closed
            onhold_price = 0.0

        if (onhold_position_3d) and (fractal_forcast_position[i] <= 0):
            onhold_position_3d = False
            ret_transcation_stats[idx_transcation, 
                                  idx_transcation_3d] = fractal_timing_lag[i]
            if (onhold_position_lineareg):
                ret_transcation_stats[idx_transcation, 
                                      idx_forcast_returns] = (closep[i] - onhold_price) / onhold_price * long
                ret_transcation_stats[idx_transcation, 
                                      idx_endpoint_price_3d] = closep[i]
            else:
                ret_transcation_stats[idx_transcation, 
                                      idx_transcation_stats] = stats_teminated
                onhold_price = 0.0

        if (onhold_position_lineareg) and (i == len(closep)):
            # 交易当前处于未结束状态
            if (int(ret_transcation_stats[idx_transcation, 
                                          idx_transcation_stats]) == stats_onhold):
                ret_transcation_stats[idx_transcation, 
                                      idx_fractal_returns] = (closep[i] - onhold_price) / onhold_price * long
            pass
        ret_transcation_stats[idx_transcation, 
                              idx_duration_time] = ret_transcation_stats[idx_transcation, 
                                                                         idx_transcation_endpoint] - ret_transcation_stats[idx_transcation, 
                                                                                                                         idx_transcation_action]

    return ret_transcation_stats[:idx_transcation + 1, :]


def calc_strategy_stats(codelist, codename, strategy_name, timing_lag,
                   ref_features=None, rsk_fre=0.04, annual=252, taxfee=0.0003, 
                   long=1, format='pd'):
    """
    策略绩效分析，
    包括年化 Sharpe Ratio，换手率, Max Drawdown 等

    """
    portfolio_position = np.where(timing_lag > 0, 1, 0)
    portfolio_returns = ref_features[idx_PCT_CHANGE] * np.r_[0, portfolio_position[:-1]]

    # Turnover Analysis 简化计算，全仓操作 100%换手
    portfolio_turnover = np.where(portfolio_position != np.r_[0, portfolio_position[:-1]], 1, 0)
    portfolio_turnover_ratio = rolling_sum(portfolio_turnover, annual)

    portfolio_annual_return = portfolio_returns.rolling(annual).apply(lambda x: 
                                                                      empyrical.annual_return(x, annualization=annual), 
                                                                      raw=True)
    portfolio_sharpe_ratio = empyrical.roll_sharpe_ratio(portfolio_returns, risk_free=rsk_fre / annual, 
                                                         annualization=annual, window=annual)
    portfolio_max_drawdown = empyrical.roll_max_drawdown(portfolio_returns, annual)

    turnover_ratio_mean = np.mean(portfolio_turnover_ratio[annual:])
    turnover_ratio_mean = turnover_ratio_mean if (turnover_ratio_mean > 2) else 24
    annual_return_mean = np.mean((portfolio_annual_return.values - portfolio_turnover_ratio * taxfee)[annual:])

    if (format != 'pd'):
        ret_portfolio_state_template = 'Code {}, {}, {} sharpe:{:.2f}, annual_return:{:.2%}, max_drawdown:{:.2%}, turnover:{:.0%}'
        ret_strategy_stats = ret_portfolio_state_template.format(codelist, 
                                         codename,
                                         strategy_name,
                                         portfolio_sharpe_ratio[-1], 
                                         (portfolio_annual_return.values - portfolio_turnover_ratio * taxfee)[-1], 
                                         portfolio_max_drawdown[-1],
                                         portfolio_turnover_ratio[-1])
        return ret_strategy_stats
    else:
        #print(ret_portfolio_state)
        return pd.Series({'symbol':codelist,
                          'name':codename,
                          'portfolio':strategy_name,
                          'sharpe':portfolio_sharpe_ratio[-1],
                          'annual_return':(portfolio_annual_return.values - portfolio_turnover_ratio * taxfee)[-1],
                          idx_TRANSACTION_RETURN_MEAN:annual_return_mean / (turnover_ratio_mean / 2),
                          'max_drawdown': portfolio_max_drawdown[-1], 
                          'turnover_ratio': portfolio_turnover_ratio[-1]})


def portfolio_stats(codelist, codename, strategy_name, timing_lag, 
                    ref_features, rsk_fre=0.04, annual=252, taxfee=0.0003):
    """
    策略组合/投资组合绩效分析

    """
    portfolio_position = np.where(timing_lag > 0, 1, 0)
    portfolio_returns = ref_features[idx_PCT_CHANGE] * np.r_[0, portfolio_position[:-1]]

    # Turnover Analysis 简化计算，全仓操作 100%换手
    portfolio_turnover = np.where(portfolio_position != np.r_[0, portfolio_position[:-1]], 1, 0)
    portfolio_turnover_ratio = rolling_sum(portfolio_turnover, annual)

    portfolio_annual_return = portfolio_returns.rolling(annual).apply(lambda x: 
                                                                      empyrical.annual_return(x, annualization=annual), 
                                                                      raw=True)
    portfolio_sharpe_ratio = empyrical.roll_sharpe_ratio(portfolio_returns, risk_free=rsk_fre / annual, 
                                                         annualization=annual, window=annual)
    portfolio_max_drawdown = empyrical.roll_max_drawdown(portfolio_returns, annual)

    turnover_ratio_mean = np.mean(portfolio_turnover_ratio[annual:])
    turnover_ratio_mean = turnover_ratio_mean if (turnover_ratio_mean > 2) else 24
    annual_return_mean = np.mean((portfolio_annual_return.values - portfolio_turnover_ratio * taxfee)[annual:])

    ret_portfolio_state_template = 'Code {}, {}, {} sharpe:{:.2f}, annual_return:{:.2%}, max_drawdown:{:.2%}, turnover:{:.0%}'
    ret_portfolio_state = ret_portfolio_state_template.format(codelist, 
                                     codename,
                                     strategy_name,
                                     portfolio_sharpe_ratio[-1], 
                                     (portfolio_annual_return.values - portfolio_turnover_ratio * taxfee)[-1], 
                                     portfolio_max_drawdown[-1],
                                     portfolio_turnover_ratio[-1])
    #print(ret_portfolio_state)
    return pd.Series({'symbol':codelist,
                      'name':codename,
                      'portfolio':strategy_name,
                      'sharpe':portfolio_sharpe_ratio[-1],
                      'annual_return':(portfolio_annual_return.values - portfolio_turnover_ratio * taxfee)[-1],
                      idx_TRANSACTION_RETURN_MEAN:annual_return_mean / (turnover_ratio_mean / 2),
                      'max_drawdown': portfolio_max_drawdown[-1], 
                      'turnover_ratio': portfolio_turnover_ratio[-1]})


def calc_onhold_positions(data, *args, **kwargs):
    """
    计算复合仓位，支持QA add_func，第二个参数 默认为 indices= 为已经计算指标
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
        
    try:
        indices[idx_ONHOLD_BEFORE] = Timeline_duration(np.where(((indices[ST.TRIGGER_R5] > 0).shift(1) == False) & \
                                                                (indices[ST.TRIGGER_R5] > 0), 
                                                                1, 0))
        indices[idx_OFFHOLD_BEFORE] = Timeline_duration(np.where(((indices[ST.TRIGGER_R5] < 0).shift(1) == False) & \
                                                                 (indices[ST.TRIGGER_R5] < 0), 
                                                                 1, 0))
    except:
        print(code, 
              't1', indices.index.get_level_values(level=0)[0], 
              't2', indices.index.get_level_values(level=0)[-1])
        raise Exception('fooo666:{}'.format(code))

    indices[ST.POSITION_ONHOLD] = np.where(indices[idx_ONHOLD_BEFORE] < indices[idx_OFFHOLD_BEFORE], 
                                           1, 0)

    # 平顺买点波动，平滑买点天数计数
    indices[idx_ONHOLD_BEFORE] = Timeline_duration(np.where(((indices[ST.POSITION_ONHOLD] > 0).shift(1) == False) & \
                                                            (indices[ST.POSITION_ONHOLD] > 0), 
                                                            1, 0))
    indices[idx_LEVERAGE_ONHOLD] = np.where(indices[ST.POSITION_ONHOLD] == 1, 
                                            indices[idx_LEVERAGE_NORM], 0)

    #if (ST.VERBOSE in data.columns):
    #    print(indices[[AKA.CLOSE,
    #                   idx_BOLL_CROSS_SX_BEFORE,
    #                   idx_ZSCORE_21,
    #                   idx_ATR_CROSS,
    #                   idx_MAXFACTOR_CROSS,
    #                   idx_ZEN_TIDE_DENSITY,
    #                   idx_COMBINE_DENSITY,
    #                   ST.CANDIDATE]].tail(50))

    return indices


#@nb.jit(nopython=True)
#def calc_onhold_returns_np(closep:np.ndarray,
#                           features:np.ndarray,
#                           long:int=1,
#                           verbose:bool=True) -> np.ndarray:
#    """
#    计算持仓利润，np.ndarray 实现，编码规范支持JIT和Cython加速
#    """
#    idx_DATETIME = 0 # DATETIME
#    idx_DAILY_RETURNS = 1 # DAILY_RETURNS
#    idx_TRIGGER = 2 # BUY ACTION
#    idx_ONHOLD_RETURNS = 2 # LEVERAGE 仓位 默认为 1
#    idx_POSITION = 3 # ONHOLD STATE after BUY ACTION 1 bar
#    idx_LEVERAGE = 4 # LEVERAGE 仓位 默认为 1

#    ret_onhold_returns = np.zeros((len(closep), 4),)
#    for i in range(1, len(features)):
#        if (features[i, idx_POSITION] == 1) or \
#            ((features[i - 1, idx_TRIGGER] == 1)):
#            if ((features[i - 1, idx_TRIGGER] == 1) and (features[i,
#            idx_POSITION] != 1)):
#                if (verbose):
#                    print(u'买入状态设置，本应设置的状态没有设置。')
#                features[i, idx_POSITION] == 1

#            ret_onhold_returns[i, idx_DATETIME] = features[i, idx_DATETIME]
#            ret_onhold_returns[i, idx_DAILY_RETURNS] = features[i,
#            idx_DAILY_RETURNS]
#            ret_onhold_returns[i, idx_POSITION] = features[i, idx_POSITION]
#            ret_onhold_returns[i, idx_ONHOLD_RETURNS] = ret_onhold_returns[i,
#            idx_ONHOLD_RETURNS] + features[i, idx_DAILY_RETURNS] *
#            ret_onhold_returns[i, idx_POSITION]

#    if (verbose):
#        pass

#    return ret_onhold_returns
@nb.jit(nopython=True)
def calc_onhold_returns_v2(daily_returns:np.ndarray, 
                           daily_position:np.ndarray,
                           long:int=1,) -> np.ndarray:
    """
    计算当前持仓利润，当一次持仓状态结束的时候清零
    np.ndarray 实现，编码规范支持JIT和Cython加速
    """
    ret_onhold_returns = np.zeros(len(daily_returns),)
    onhold_returns = 0.0
    onhold_position = False
    assert long == 1 or long == -1
    for i in range(0, len(daily_position)):
        if (onhold_position):
            onhold_returns = onhold_returns + daily_returns[i] * long
        else:
            onhold_returns = 0.0            
        ret_onhold_returns[i] = onhold_returns

        if (daily_position[i] > 0):
            onhold_position = True

        if (daily_position[i] <= 0):
            onhold_position = False

    return ret_onhold_returns


@nb.jit(nopython=True)
def calc_onhold_returns_np(closep:np.ndarray, 
                           daily_position:np.ndarray,
                           long:int=1,) -> np.ndarray:
    """
    计算当前持仓利润，当一次持仓状态结束的时候清零
    np.ndarray 实现，编码规范支持JIT和Cython加速
    """
    ret_onhold_returns = np.zeros(len(closep),)
    onhold_price = onhold_returns = 0.0
    onhold_position = False
    assert long == 1 or long == -1
    for i in range(0, len(daily_position)):
        if (np.isnan(daily_position[i])):
            continue

        if (onhold_position):
            if (daily_position[i - 1] <= 0):
                onhold_price = closep[i]
        else:
            onhold_price = closep[i]
        if (onhold_price > 0.001):
            ret_onhold_returns[i] = (closep[i] - onhold_price) / onhold_price * long

        if (daily_position[i] > 0):
            onhold_position = True

        if (daily_position[i] <= 0):
            onhold_position = False

    return ret_onhold_returns


def calc_onhold_returns(data, *args, **kwargs):
    """
    计算持仓利润，支持QA add_func，第二个参数 默认为 indices= 为已经计算指标
    理论上这个函数只计算单一标的，不要尝试传递复杂标的，indices会尝试拆分。
    这是针对 pd 参数和 QA.add_func 的 Wrapper，因为不能JIT和Cython加速。
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
       
    return indices


#@nb.jit(nopython=True)
def calc_positions_StockCN(position:np.ndarray,
                           indices:np.ndarray,
                           upper_limit:int=1.0,
                           lower_limit:int=0.0) -> np.ndarray:
    """
    计算配合开仓条件的实际仓位变化，
    """
    leverage_delta = position[:, 0]
    position_norm = position[:, 1]
    position_pre = position[:, 2]
    position_signal = position[:, 3]

    ma90_clearance = indices[:, 0]
    ma120_clearance = indices[:, 1]
    closep = indices[:, 2]

    # 实际仓位变化为
    position_delta = position_pre - position_norm

    order_position = np.zeros((len(leverage_delta), 3))
    for i in range(len(position_signal)):
        if (i == 0):
            order_position[i, 2] = 0.0
        else:
            order_position[i, 2] = np.nan

        if (position_signal[i] > 0):
            # 开仓模式
            if (position_delta[i] > 0):
                if ((ma90_clearance[i] + ma120_clearance[i]) > 1.24):
                    # 高位冒险，尝试逐步加仓
                    order_position[i, 0] = abs(max(position_delta[i], leverage_delta[i]))
                else:
                    # 直接一步到位建仓
                    order_position[i, 0] = abs(position_norm[i])

                order_position[i, 2] = order_position[i, 0]
                continue
        elif (position_signal[i] < 0):
            if (position_delta[i] < 0):
                if ((ma90_clearance[i] + ma120_clearance[i]) > 1.24) or \
                    (position_delta[i] < -0.168):
                    # 高位冒险，直接清仓
                    order_position[i, 0] = -max(position_pre[i], 
                                                order_position[i - 1, 2])
                    order_position[i, 2] = 0.0
                else:
                    # 逐步减仓
                    order_position[i, 0] = position_delta[i]
                    order_position[i, 2] = order_position[i - 1, 2] + position_delta[i]
                continue
        
        if (position_delta[i] < 0):
            if ((order_position[i - 1, 2] + position_delta[i]) > 0):
                # 逐步减仓
                order_position[i, 0] = position_delta[i]
                order_position[i, 2] = order_position[i - 1, 2] + position_delta[i]
            else:
                # 被动清仓
                order_position[i, 0] = -max(position_pre[i],
                                            order_position[i - 1, 2])
                order_position[i, 2] = 0.0
            continue
        elif (position_delta[i] > 0):
            if (order_position[i - 1, 2] > 0):
                # 被动加仓
                order_position[i, 0] = position_delta[i]
                order_position[i, 2] = order_position[i - 1, 2] + position_delta[i]
            else:
                # 疑似踏空
                order_position[i:, 1] = 1
                order_position[i, 2] = order_position[i - 1, 2]
        elif(abs(position_delta[i]) < 0.005):
            # Do nothing 判断条件不足，阈值不足以做任何操作
            order_position[i, 2] = order_position[i - 1, 2]

    return order_position


@nb.jit(nopython=True)
def calc_leverages(leverage_delta:np.ndarray,
                   upper_limit:int=1.0,
                   lower_limit:int=0.0) -> np.ndarray:
    """
    计算连续加减杠杆（仓位）比例变化
    """
    leverage_norm = np.zeros((len(leverage_delta),2))
    for i in range(len(leverage_delta)):
        leverage_norm[i, 1] = leverage_norm[i - 1, 0]
        leverage_norm[i, 0] = leverage_norm[i - 1, 0] + leverage_delta[i]
        if (leverage_norm[i, 0] > upper_limit):
            leverage_norm[i, 0] = upper_limit
        elif (leverage_norm[i, 0] < lower_limit):
            leverage_norm[i, 0] = lower_limit
    
    return leverage_norm


def calc_massive_fractal_trend(features:pd.DataFrame=None,) -> pd.DataFrame:
    """
    A股中某些特殊日子是建仓的黄道吉日，比如2020年8月20日，2020年8月27日，
    2020年9月28~30日，20-12-28 29 30，2021年01月20日。
    本函数用缠论折点的时间共振 + 高斯聚类判断(大雾) 其实只需要判断TIMING_LAG=1~4小时即可
    计算出这些（决定性的建仓日）日子和在决定性的建仓日适合购买的股票。
    """
    #print(u'calc_massive_fractal_trend')
    symbol_list = sorted(features.index.get_level_values(level=1).unique())
    column_list = [FLD.PEAK_LOW_TIMING_LAG,
                   FLD.HMAPOWER120_TIMING_LAG,
                   FLD.HMAPOWER120_QUARTER,
                   FLD.MAPOWER30_TIMING_LAG,
                   FLD.MAPOWER30_QUARTER,
                   FLD.MA90_CLEARANCE,
                   FLD.MA_CHANNEL,
                   FLD.MACD_ZERO_TIMING_LAG,
                   FLD.DIF_ZERO_TIMING_LAG,
                   FLD.ZEN_WAVELET_TIMING_LAG,
                   FLD.BOLL_RAISED_TIMING_LAG,
                   FLD.POLYNOMIAL9_TIMING_LAG,
                   FTR.POLYNOMIAL9_DUAL,
                   FLD.MAPOWER30,
                   FLD.MAPOWER30_MAJOR,
                   FLD.HMAPOWER120_MAJOR,
                   FLD.MAINFEST_UPRISING_COEFFICIENT,
                   FTR.BOOTSTRAP_ENHANCED_TIMING_LAG,
                   FLD.MAINFEST_DOWNRISK_TIMING_LAG,
                   'dt']

    #try:
    if (True):
        features['dt'] = pd.to_datetime(features.index.get_level_values(level=0),).tz_localize('Asia/Shanghai')
        print(len(features.columns), len(set(features.columns.values)), len(features.columns), len(set(features.columns.values)))
        # 对齐数据，DataFrame转为3维array数组，用于cython或者jit加速运行
        features_aligned = features[column_list].unstack().ffill().bfill().stack()
        features_np = features_aligned.drop(['dt'], 
                                            axis=1).values.reshape(*features.index.levshape,
                                                                   -1)
        #features_aligned['dt'] =
        #pd.to_datetime(features_aligned.index.get_level_values(level=0),).tz_localize('Asia/Shanghai')
        symbol_list = features_aligned.index.get_level_values(level=1).unique()
        each_day = features_aligned['dt'].values
        each_day_epoch = sorted(np.unique(each_day.astype(np.int64)) // 10 ** 9)
        #print(type(np.array(each_day_epoch, dtype=np.int64)))
        ret_massive_fractal_trend = calc_massive_uprising_func(np.array(each_day_epoch, dtype=np.int64), 
                                                               features_np)
    #except Exception as e:
    #    print(e)
    #    each_day = features.index.get_level_values(level=0).unique()
    #    symbol_list = features.index.get_level_values(level=1).unique()
    #    each_day_epoch = sorted(each_day.astype(np.int64) // 10 ** 9)
    #    ret_massive_fractal_trend = None

    #print(u'calc_massive_fractal_trend phase #2')

    if (ret_massive_fractal_trend is None):
        features[FLD.MASSIVE_TREND] = np.nan
        features[FLD.MASSIVE_TREND_BEFORE] = np.nan
        features[FLD.MASSIVE_TREND_RETURNS] = np.nan
        features[FLD.MASSIVE_TREND_CHECKOUT_CLOSE] = np.nan
    else:
        ret_massive_trend_pd = pd.DataFrame(ret_massive_fractal_trend, 
                                            index=pd.to_datetime(ret_massive_fractal_trend[:, 0],
                                                                 unit='s').tz_localize('UTC').tz_convert('Asia/Shanghai').tz_localize(None),
                                            columns=[AKA.DATETIME, FLD.MASSIVE_TREND, FLD.MAPOWER_MEDIAN])
        #print(ret_massive_trend_pd.tail(100))
        if (FLD.MASSIVE_TREND not in features.columns):
            features = features.reindex(columns=[*features.columns,
                                                 *[FLD.MASSIVE_TREND,
                                                   FLD.MASSIVE_TREND_BEFORE,]])
        for symbol in symbol_list:
            features_slice = features.loc[(slice(None), symbol), :]
            features.loc[(slice(None), symbol),
                         FLD.MASSIVE_TREND] = np.where((features_slice[FLD.MAINFEST_DOWNRISK_TIMING_LAG] > 0) & \
                                                       (features_slice[FLD.MAINFEST_DOWNRISK_TIMING_LAG] <= 4) & \
                                                       (features_slice[FLD.SEMI_DOWNRISK] < 0.5) & \
                                                       (ret_massive_trend_pd.loc[features_slice.index.get_level_values(level=0), 
                                                                                 FLD.MASSIVE_TREND] > 0), 1, 0)
            features.loc[(slice(None), symbol), 
                         FLD.MASSIVE_TREND_BEFORE] = Timeline_duration(features.loc[(slice(None), symbol),
                                                                                    FLD.MASSIVE_TREND].values)
            features.loc[(slice(None), symbol), 
                         FLD.MASSIVE_TREND_CHECKOUT_CLOSE] = np.where(features.loc[(slice(None), symbol), 
                                                                                   FLD.MASSIVE_TREND_BEFORE] == 0, 
                                                                      features_slice[AKA.CLOSE], np.nan)
            features.loc[(slice(None), symbol), 
                         FLD.MASSIVE_TREND_CHECKOUT_CLOSE] = features.loc[(slice(None), symbol), 
                                                                          FLD.MASSIVE_TREND_CHECKOUT_CLOSE].ffill()
            features.loc[(slice(None), symbol), 
                         FLD.MASSIVE_TREND_RETURNS] = np.log(features_slice[AKA.CLOSE] / features.loc[(slice(None), symbol), 
                                                                                                      FLD.MASSIVE_TREND_CHECKOUT_CLOSE])

    return features, ret_massive_fractal_trend


@nb.jit('f8[:,:](i8[:], f8[:,:,:])', nopython=True)
def calc_massive_fractal_trend_func(each_day_epoch:np.ndarray, 
                                    features_np:np.ndarray,) -> np.ndarray:
    """
    在版块行情特征数据中提取上升浪
    """
    #print(features.shape)
    idx_PEAK_LOW_TIMING_LAG = 0
    idx_HMAPOWER120_TIMING_LAG = 1
    idx_HMAPOWER120_QUARTER = 2
    idx_MAPOWER30_TIMING_LAG = 3
    idx_MAPOWER30_QUARTER = 4
    idx_MA90_CLEARANCE = 5
    idx_MA_CHANNEL = 6
    idx_MACD_ZERO_TIMING_LAG = 7
    idx_DIF_ZERO_TIMING_LAG = 8
    idx_ZEN_WAVELET_TIMING_LAG = 9
    idx_BOLL_RAISED_TIMING_LAG = 10
    idx_POLYNOMIAL9_TIMING_LAG = 11
    idx_POLYNOMIAL9_DUAL = 12
    idx_MAPOWER30 = 13
    idx_MAPOWER30_MAJOR = 14
    idx_HMAPOWER120_MAJOR = 15
    idx_MAINFEST_UPRISING_COEFFICIENT = 16
    idx_BOOTSTRAP_ENHANCED_TIMING_LAG = 17
    idx_MAINFEST_DOWNRISK_TIMING_LAG = 18

    ret_massive_fractal_trend = np.zeros((len(each_day_epoch),3))
    ret_massive_fractal_trend[:, 0] = each_day_epoch
    totals = len(features_np[0, :, idx_PEAK_LOW_TIMING_LAG])
    polynomial9_sum_yesterday = 0
    mapower_median_yesterday = 0
    mainfest_uprising_median_yesterday = 0
    bootstrap_enhanced_median_yesterday = 0
    for i in range(0, len(each_day_epoch)):
        mapower_median = np.median(features_np[i, :, idx_MAPOWER30]) + np.median(features_np[i, :, idx_MAPOWER30_MAJOR]) + np.median(features_np[i, :, idx_HMAPOWER120_MAJOR])
        mainfest_uprising_median = np.median(features_np[i, :, idx_MAINFEST_UPRISING_COEFFICIENT])
        bootstrap_enhanced_median = np.median(features_np[i, :, idx_BOOTSTRAP_ENHANCED_TIMING_LAG])
        day_pieces = np.where(features_np[i, :, idx_POLYNOMIAL9_TIMING_LAG] > 0)
        polynomial9_sum = np.sum(np.where(features_np[i, :, idx_POLYNOMIAL9_TIMING_LAG] > 0, 1, 0))
        polynomial9_dual_sum = np.sum(np.where(features_np[i, :, idx_POLYNOMIAL9_DUAL] > 0, 1, 0))

        if (np.median(features_np[i, :, idx_PEAK_LOW_TIMING_LAG]) > 0) and \
           (np.median(features_np[i, :, idx_HMAPOWER120_TIMING_LAG]) > 0) and \
           (np.median(features_np[i, :, idx_HMAPOWER120_QUARTER]) > 0) and \
           (np.median(features_np[i, :, idx_MAPOWER30_TIMING_LAG]) > 0) and \
           (np.median(features_np[i, :, idx_MAPOWER30_QUARTER]) > 0) and \
           (np.median(features_np[i, :, idx_MA90_CLEARANCE]) > 0) and \
           (np.median(features_np[i, :, idx_MA_CHANNEL]) > 0) and \
           ((np.median(features_np[i, :, idx_MACD_ZERO_TIMING_LAG]) > 0) or \
           ((np.median(features_np[i, :, idx_MA90_CLEARANCE]) < 0.618) and \
           (np.median(features_np[i, :, idx_ZEN_WAVELET_TIMING_LAG]) > 0))) and \
           (np.median(features_np[i, :, idx_DIF_ZERO_TIMING_LAG]) > 0):
            if ((int(mainfest_uprising_median) != 1) and (int(bootstrap_enhanced_median) != 1) and \
                (mapower_median_yesterday > mapower_median)) or \
                ((int(mainfest_uprising_median) > 20) and (int(bootstrap_enhanced_median) > 20)):
                pass
            else:
                ret_massive_fractal_trend[i, 1] = 1
        
        if ((polynomial9_sum / totals > 0.382) and (polynomial9_dual_sum / polynomial9_sum > 0.618) and \
            (polynomial9_sum_yesterday < polynomial9_sum / totals) and (mapower_median < 2.3)) or \
            ((polynomial9_sum / totals > 0.618) and (polynomial9_dual_sum / polynomial9_sum > 0.512) and \
            (polynomial9_sum_yesterday > 0.618) and (mapower_median < 1)) or \
            ((polynomial9_sum / totals > 0.618) and (polynomial9_dual_sum / polynomial9_sum > 0.618) and \
            (mapower_median < 1)) or \
            ((polynomial9_sum / totals > 0.618) and (polynomial9_dual_sum / polynomial9_sum > 0.512) and \
            (mapower_median < 0.618)) or \
            ((int(mainfest_uprising_median) == 1) and \
            (int(mainfest_uprising_median_yesterday) < 0.1) and (bootstrap_enhanced_median_yesterday < 0.1)) or \
            ((int(mainfest_uprising_median) == 1) and \
            (int(mainfest_uprising_median_yesterday) == 1) and (bootstrap_enhanced_median_yesterday < 0.1)) or \
            ((int(bootstrap_enhanced_median) == 1) and \
            ((mainfest_uprising_median - bootstrap_enhanced_median) > -0.1) and \
            ((mainfest_uprising_median - bootstrap_enhanced_median) < 6.1)):
            if ((int(mainfest_uprising_median) != 1) and (int(bootstrap_enhanced_median) != 1) and \
                (mapower_median_yesterday > mapower_median)) or \
                ((int(mainfest_uprising_median) > 20) and (int(bootstrap_enhanced_median) > 20)):
                pass
            else:
                if (ret_massive_fractal_trend[i, 1] == 0) :
                    ret_massive_fractal_trend[i, 1] = 2
                    #print(QA_util_print_timestamp(each_day[i]),
                    #polynomial9_sum,
                    #          '{:.2%}'.format(polynomial9_sum / totals),
                    #          '{:3d}'.format(polynomial9_dual_sum),
                    #          '{:.2%}'.format(polynomial9_dual_sum /
                    #          polynomial9_sum),
                    #          '{:.3f}'.format(np.median(features[i,
                    #          day_pieces, idx_BOLL_RAISED_TIMING_LAG])),
                    #          '{:.3f}'.format(np.quantile(features[i,
                    #          day_pieces, idx_BOLL_RAISED_TIMING_LAG], 0.75)),
                    #          '{:.3f}'.format(np.median(features[i, :,
                    #          idx_MAPOWER30])),
                    #          '{:.3f}'.format(np.median(features[i, :,
                    #          idx_MAPOWER30_MAJOR])),
                    #          '{:.3f}'.format(np.median(features[i, :,
                    #          idx_HMAPOWER120_MAJOR])),
                    #          '{:.1f}'.format(np.median(features[i, :,
                    #          idx_MAINFEST_UPRISING_COEFFICIENT])),
                    #          '{:.1f}'.format(np.median(features[i, :,
                    #          idx_BOOTSTRAP_ENHANCED_TIMING_LAG])),
                    #          '{:.3f}'.format(mapower_median), u'<<-- 建仓日')
                else:
                    ret_massive_fractal_trend[i, 1] = 3
                    #print(QA_util_print_timestamp(each_day[i]),
                    #polynomial9_sum,
                    #          '{:.2%}'.format(polynomial9_sum / totals),
                    #          '{:3d}'.format(polynomial9_dual_sum),
                    #          '{:.2%}'.format(polynomial9_dual_sum /
                    #          polynomial9_sum),
                    #          '{:.3f}'.format(np.median(features[i, :,
                    #          idx_BOLL_RAISED_TIMING_LAG])),
                    #          '{:.3f}'.format(np.quantile(features[i, :,
                    #          idx_BOLL_RAISED_TIMING_LAG], 0.75)),
                    #          '{:.3f}'.format(np.median(features[i, :,
                    #          idx_MAPOWER30])),
                    #          '{:.3f}'.format(np.median(features[i, :,
                    #          idx_MAPOWER30_MAJOR])),
                    #          '{:.3f}'.format(np.median(features[i, :,
                    #          idx_HMAPOWER120_MAJOR])),
                    #          '{:.1f}'.format(np.median(features[i, :,
                    #          idx_MAINFEST_UPRISING_COEFFICIENT])),
                    #          '{:.1f}'.format(np.median(features[i, :,
                    #          idx_BOOTSTRAP_ENHANCED_TIMING_LAG])),
                    #          '{:.3f}'.format(mapower_median), u'<<-- 建仓日')
        else:
            #print(QA_util_print_timestamp(each_day[i]), polynomial9_sum,
            #              '{:.2%}'.format(polynomial9_sum / totals),
            #              '{:3d}'.format(polynomial9_dual_sum),
            #              '{:.2%}'.format(polynomial9_dual_sum /
            #              polynomial9_sum),
            #              '{:.3f}'.format(np.median(features[i, :,
            #              idx_BOLL_RAISED_TIMING_LAG])),
            #              '{:.3f}'.format(np.quantile(features[i, :,
            #              idx_BOLL_RAISED_TIMING_LAG], 0.75)),
            #              '{:.3f}'.format(np.median(features[i, :,
            #              idx_MAPOWER30])),
            #              '{:.3f}'.format(np.median(features[i, :,
            #              idx_MAPOWER30_MAJOR])),
            #              '{:.3f}'.format(np.median(features[i, :,
            #              idx_HMAPOWER120_MAJOR])),
            #              '{:.1f}'.format(np.median(features[i, :,
            #              idx_MAINFEST_UPRISING_COEFFICIENT])),
            #              '{:.1f}'.format(np.median(features[i, :,
            #              idx_BOOTSTRAP_ENHANCED_TIMING_LAG])),
            #              '{:.3f}'.format(mapower_median))
            pass
        ret_massive_fractal_trend[i, 2] = mapower_median
        polynomial9_sum_yesterday = polynomial9_sum / totals
        mainfest_uprising_median_yesterday = mainfest_uprising_median
        bootstrap_enhanced_median_yesterday = bootstrap_enhanced_median
        mapower_median_yesterday = mapower_median
            #print(i, each_day[i], 'Yes!')
        #if (i % 100 == 0):
        #    print(i, each_day[i])
        #pass
    #print('sum', np.sum(ret_massive_fractal_trend[:, 1]))
    return ret_massive_fractal_trend


@nb.jit('f8[:,:](i8[:], f8[:,:,:])', nopython=True)
def calc_massive_uprising_func(each_day_epoch:np.ndarray, 
                               features_np:np.ndarray,) -> np.ndarray:
    """
    在版块行情特征数据中提取上升浪
    """
    idx_PEAK_LOW_TIMING_LAG = 0
    idx_HMAPOWER120_TIMING_LAG = 1
    idx_HMAPOWER120_QUARTER = 2
    idx_MAPOWER30_TIMING_LAG = 3
    idx_MAPOWER30_QUARTER = 4
    idx_MA90_CLEARANCE = 5
    idx_MA_CHANNEL = 6
    idx_MACD_ZERO_TIMING_LAG = 7
    idx_DIF_ZERO_TIMING_LAG = 8
    idx_ZEN_WAVELET_TIMING_LAG = 9
    idx_BOLL_RAISED_TIMING_LAG = 10
    idx_POLYNOMIAL9_TIMING_LAG = 11
    idx_POLYNOMIAL9_DUAL = 12
    idx_MAPOWER30 = 13
    idx_MAPOWER30_MAJOR = 14
    idx_HMAPOWER120_MAJOR = 15
    idx_MAINFEST_UPRISING_COEFFICIENT = 16
    idx_BOOTSTRAP_ENHANCED_TIMING_LAG = 17
    idx_MAINFEST_DOWNRISK_TIMING_LAG = 18

    ret_massive_fractal_trend = np.zeros((len(each_day_epoch),3))
    ret_massive_fractal_trend[:, 0] = each_day_epoch
    totals = len(features_np[0, :, idx_PEAK_LOW_TIMING_LAG])
    for i in range(0, len(each_day_epoch)):
        mapower_median = np.median(features_np[i, :, idx_MAPOWER30]) + np.median(features_np[i, :, idx_MAPOWER30_MAJOR]) + np.median(features_np[i, :, idx_HMAPOWER120_MAJOR])
        ret_massive_fractal_trend[i, 2] = mapower_median
        uprising_counts = np.where(features_np[i, :, idx_MAINFEST_DOWNRISK_TIMING_LAG] == 1)[0]
        if ((len(uprising_counts) * 4 / totals > 0.1236) and not ((totals < 100) and (len(uprising_counts) < 4))) or \
            ((totals < 100) and (len(uprising_counts) > 5)) or \
            ((len(uprising_counts) * 4 / totals > 0.0927) and (len(uprising_counts) > 5)):
            ret_massive_fractal_trend[i, 1] = 1
            #print(QA_util_print_timestamp(each_day[i]), u'<<-- 建仓日',
            #      'len {:d}, {:.02%}'.format(len(uprising_counts),
            #                                (len(uprising_counts) * 4) /
            #                                totals),
            #      uprising_counts)
        else:
            #print(QA_util_print_timestamp(each_day[i]),
            #      'len {:d}, {:.02%}'.format(len(uprising_counts),
            #                                (len(uprising_counts) * 4) /
            #                                totals),
            #      uprising_counts)
            pass

    return ret_massive_fractal_trend


def calc_massive_csindex_trend(features:pd.DataFrame=None,) -> pd.DataFrame:
    """
    """
    symbol_list = sorted(features.index.get_level_values(level=1).unique())
   
    column_list = [FLD.PEAK_OPEN,
                   FLD.ZEN_PEAK_TIMING_LAG,
                   FLD.ZEN_DASH_TIMING_LAG,
                   FLD.ZEN_WAVELET_TIMING_LAG,
                   FLD.ZEN_BOOST_TIMING_LAG,
                   FLD.RENKO_BOOST_S_TIMING_LAG,
                   FLD.RENKO_TREND_S_TIMING_LAG,
                   FLD.PEAK_LOW_TIMING_LAG,
                   FLD.HMAPOWER120_TIMING_LAG,
                   FLD.MAPOWER30_TIMING_LAG,
                   FLD.MACD_ZERO_TIMING_LAG,
                   FLD.DIF_ZERO_TIMING_LAG,
                   FLD.POLYNOMIAL9_TIMING_LAG,
                   FLD.MAPOWER30,
                   FLD.HMAPOWER120,]

    # 对齐数据，DataFrame转为3维array数组，用于cython或者jit加速运行
    try:
        features_aligned = features[column_list].unstack().ffill().bfill().stack()
        features_np = features_aligned.values.reshape(*features.index.levshape,-1)
        features_aligned['dt'] = pd.to_datetime(kline.data.index.get_level_values(level=0),).tz_localize('Asia/Shanghai')
        each_day = features_aligned['dt'].values
        symbol_list = features_aligned.index.get_level_values(level=1).unique()
        each_day_epoch = sorted(each_day.astype(np.int64) // 10 ** 9)

        ret_massive_fractal_trend = calc_massive_csindex_trend_func(np.array(each_day_epoch, dtype=np.int64), 
                                                                    features_np)
    except:
        each_day = features.index.get_level_values(level=0).unique()
        symbol_list = features.index.get_level_values(level=1).unique()
        each_day_epoch = sorted(each_day.astype(np.int64) // 10 ** 9)
        ret_massive_fractal_trend = None

    if (ret_massive_fractal_trend is None):
        features[FLD.MASSIVE_TREND] = np.nan
    else:
        ret_massive_trend_pd = pd.DataFrame(ret_massive_fractal_trend, 
                                            index=pd.to_datetime(ret_massive_fractal_trend[:, 0],
                                                                 unit='s'),
                                            columns=[AKA.DATETIME, FLD.MASSIVE_TREND, FLD.MAPOWER_MEDIAN])

        features[FLD.MASSIVE_TREND] = ret_massive_trend_pd.loc[ret_codelist_combo.index.get_level_values(level=0), 
                                                                      FLD.MASSIVE_TREND].values

    return features, ret_massive_fractal_trend


#@nb.jit('f8[:,:](i8[:], f8[:,:,:])', nopython=True)
def calc_massive_csindex_trend_func(each_day:np.ndarray, 
                                    features:np.ndarray,) -> np.ndarray:
    """
    在版块行情特征数据中提取上升浪
    """
    #print(features.shape)
    idx_PEAK_LOW_TIMING_LAG = 0
    idx_HMAPOWER120_TIMING_LAG = 1
    idx_HMAPOWER120_QUARTER = 2
    idx_MAPOWER30_TIMING_LAG = 3
    idx_MAPOWER30_QUARTER = 4
    idx_MA90_CLEARANCE = 5
    idx_MA_CHANNEL = 6
    idx_MACD_ZERO_TIMING_LAG = 7
    idx_DIF_ZERO_TIMING_LAG = 8
    idx_ZEN_WAVELET_TIMING_LAG = 9
    idx_PREDICT_GROWTH = 10
    idx_POLYNOMIAL9_TIMING_LAG = 11
    idx_POLYNOMIAL9_DUAL = 12
    idx_MAPOWER30 = 13
    idx_MAPOWER30_MAJOR = 14
    idx_HMAPOWER120_MAJOR = 15
    idx_MAINFEST_UPRISING_COEFFICIENT = 16
    idx_BOOTSTRAP_ENHANCED_TIMING_LAG = 17

    ret_massive_fractal_trend = np.zeros((len(each_day),3))
    ret_massive_fractal_trend[:, 0] = each_day
    totals = len(features[0, :, idx_PEAK_LOW_TIMING_LAG])
    polynomial9_sum_yesterday = 0
    mapower_median_yesterday = 0
    mainfest_uprising_median_yesterday = 0
    bootstrap_enhanced_median_yesterday = 0
    for i in range(0, len(each_day)):
        mapower_median = np.median(features[i, :, idx_MAPOWER30]) + np.median(features[i, :, idx_MAPOWER30_MAJOR]) + np.median(features[i, :, idx_HMAPOWER120_MAJOR])
        mainfest_uprising_median = np.median(features[i, :, idx_MAINFEST_UPRISING_COEFFICIENT])
        bootstrap_enhanced_median = np.median(features[i, :, idx_BOOTSTRAP_ENHANCED_TIMING_LAG])
        day_pieces = np.where(features[i, :, idx_POLYNOMIAL9_TIMING_LAG] > 0)
        polynomial9_sum = np.sum(np.where(features[i, :, idx_POLYNOMIAL9_TIMING_LAG] > 0, 1, 0))
        polynomial9_dual_sum = np.sum(np.where(features[i, :, idx_POLYNOMIAL9_DUAL] > 0, 1, 0))

        if (np.median(features[i, :, idx_PEAK_LOW_TIMING_LAG]) > 0) and \
           (np.median(features[i, :, idx_HMAPOWER120_TIMING_LAG]) > 0) and \
           (np.median(features[i, :, idx_HMAPOWER120_QUARTER]) > 0) and \
           (np.median(features[i, :, idx_MAPOWER30_TIMING_LAG]) > 0) and \
           (np.median(features[i, :, idx_MAPOWER30_QUARTER]) > 0) and \
           (np.median(features[i, :, idx_MA90_CLEARANCE]) > 0) and \
           (np.median(features[i, :, idx_MA_CHANNEL]) > 0) and \
           ((np.median(features[i, :, idx_MACD_ZERO_TIMING_LAG]) > 0) or \
           ((np.median(features[i, :, idx_MA90_CLEARANCE]) < 0.618) and \
           (np.median(features[i, :, idx_ZEN_WAVELET_TIMING_LAG]) > 0))) and \
           (np.median(features[i, :, idx_DIF_ZERO_TIMING_LAG]) > 0):
            if ((int(mainfest_uprising_median) != 1) and (int(bootstrap_enhanced_median) != 1) and \
                (mapower_median_yesterday > mapower_median)) or \
                ((int(mainfest_uprising_median) > 20) and (int(bootstrap_enhanced_median) > 20)):
                pass
            else:
                ret_massive_fractal_trend[i, 1] = 1
        
        if ((polynomial9_sum / totals > 0.382) and (polynomial9_dual_sum / polynomial9_sum > 0.618) and \
            (polynomial9_sum_yesterday < polynomial9_sum / totals) and (mapower_median < 2.3)) or \
            ((polynomial9_sum / totals > 0.618) and (polynomial9_dual_sum / polynomial9_sum > 0.512) and \
            (polynomial9_sum_yesterday > 0.618) and (mapower_median < 1)) or \
            ((polynomial9_sum / totals > 0.618) and (polynomial9_dual_sum / polynomial9_sum > 0.618) and \
            (mapower_median < 1)) or \
            ((polynomial9_sum / totals > 0.618) and (polynomial9_dual_sum / polynomial9_sum > 0.512) and \
            (mapower_median < 0.618)) or \
            ((int(mainfest_uprising_median) == 1) and \
            (int(mainfest_uprising_median_yesterday) < 0.1) and (bootstrap_enhanced_median_yesterday < 0.1)) or \
            ((int(mainfest_uprising_median) == 1) and \
            (int(mainfest_uprising_median_yesterday) == 1) and (bootstrap_enhanced_median_yesterday < 0.1)) or \
            ((int(bootstrap_enhanced_median) == 1) and \
            ((mainfest_uprising_median - bootstrap_enhanced_median) > -0.1) and \
            ((mainfest_uprising_median - bootstrap_enhanced_median) < 6.1)):
            if ((int(mainfest_uprising_median) != 1) and (int(bootstrap_enhanced_median) != 1) and \
                (mapower_median_yesterday > mapower_median)) or \
                ((int(mainfest_uprising_median) > 20) and (int(bootstrap_enhanced_median) > 20)):
                pass
            else:
                if (ret_massive_fractal_trend[i, 1] == 0) :
                    ret_massive_fractal_trend[i, 1] = 2
                    #print(QA_util_print_timestamp(each_day[i]),
                    #polynomial9_sum,
                    #          '{:.2%}'.format(polynomial9_sum / totals),
                    #          '{:3d}'.format(polynomial9_dual_sum),
                    #          '{:.2%}'.format(polynomial9_dual_sum /
                    #          polynomial9_sum),
                    #          '{:.3f}'.format(np.median(features[i,
                    #          day_pieces, idx_PREDICT_GROWTH])),
                    #          '{:.3f}'.format(np.quantile(features[i,
                    #          day_pieces, idx_PREDICT_GROWTH], 0.75)),
                    #          '{:.3f}'.format(np.median(features[i, :,
                    #          idx_MAPOWER30])),
                    #          '{:.3f}'.format(np.median(features[i, :,
                    #          idx_MAPOWER30_MAJOR])),
                    #          '{:.3f}'.format(np.median(features[i, :,
                    #          idx_HMAPOWER120_MAJOR])),
                    #          '{:.1f}'.format(np.median(features[i, :,
                    #          idx_MAINFEST_UPRISING_COEFFICIENT])),
                    #          '{:.1f}'.format(np.median(features[i, :,
                    #          idx_BOOTSTRAP_ENHANCED_TIMING_LAG])),
                    #          '{:.3f}'.format(mapower_median), u'<<-- 建仓日')
                else:
                    ret_massive_fractal_trend[i, 1] = 3
                    #print(QA_util_print_timestamp(each_day[i]),
                    #polynomial9_sum,
                    #          '{:.2%}'.format(polynomial9_sum / totals),
                    #          '{:3d}'.format(polynomial9_dual_sum),
                    #          '{:.2%}'.format(polynomial9_dual_sum /
                    #          polynomial9_sum),
                    #          '{:.3f}'.format(np.median(features[i, :,
                    #          idx_PREDICT_GROWTH])),
                    #          '{:.3f}'.format(np.quantile(features[i, :,
                    #          idx_PREDICT_GROWTH], 0.75)),
                    #          '{:.3f}'.format(np.median(features[i, :,
                    #          idx_MAPOWER30])),
                    #          '{:.3f}'.format(np.median(features[i, :,
                    #          idx_MAPOWER30_MAJOR])),
                    #          '{:.3f}'.format(np.median(features[i, :,
                    #          idx_HMAPOWER120_MAJOR])),
                    #          '{:.1f}'.format(np.median(features[i, :,
                    #          idx_MAINFEST_UPRISING_COEFFICIENT])),
                    #          '{:.1f}'.format(np.median(features[i, :,
                    #          idx_BOOTSTRAP_ENHANCED_TIMING_LAG])),
                    #          '{:.3f}'.format(mapower_median), u'<<-- 建仓日')
        else:
            #print(QA_util_print_timestamp(each_day[i]), polynomial9_sum,
            #              '{:.2%}'.format(polynomial9_sum / totals),
            #              '{:3d}'.format(polynomial9_dual_sum),
            #              '{:.2%}'.format(polynomial9_dual_sum /
            #              polynomial9_sum),
            #              '{:.3f}'.format(np.median(features[i, :,
            #              idx_PREDICT_GROWTH])),
            #              '{:.3f}'.format(np.quantile(features[i, :,
            #              idx_PREDICT_GROWTH], 0.75)),
            #              '{:.3f}'.format(np.median(features[i, :,
            #              idx_MAPOWER30])),
            #              '{:.3f}'.format(np.median(features[i, :,
            #              idx_MAPOWER30_MAJOR])),
            #              '{:.3f}'.format(np.median(features[i, :,
            #              idx_HMAPOWER120_MAJOR])),
            #              '{:.1f}'.format(np.median(features[i, :,
            #              idx_MAINFEST_UPRISING_COEFFICIENT])),
            #              '{:.1f}'.format(np.median(features[i, :,
            #              idx_BOOTSTRAP_ENHANCED_TIMING_LAG])),
            #              '{:.3f}'.format(mapower_median))
            pass
        ret_massive_fractal_trend[i, 2] = mapower_median
        polynomial9_sum_yesterday = polynomial9_sum / totals
        mainfest_uprising_median_yesterday = mainfest_uprising_median
        bootstrap_enhanced_median_yesterday = bootstrap_enhanced_median
        mapower_median_yesterday = mapower_median
            #print(i, each_day[i], 'Yes!')
        #if (i % 100 == 0):
        #    print(i, each_day[i])
        #pass
    #print('sum', np.sum(ret_massive_fractal_trend[:, 1]))
    return ret_massive_fractal_trend