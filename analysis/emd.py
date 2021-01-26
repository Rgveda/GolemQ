# coding:utf-8
# Author: 阿财（Rgveda@github）（11652964@qq.com）
# Created date: 2020-02-27
#
# The MIT License (MIT)
#
# Copyright (c) 2016-2019 yutiansut/QUANTAXIS
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
import numpy as np
import pandas as pd
import empyrical
#from PyEMD import EEMD, EMD, Visualisation

from GolemQ.analysis.timeseries import *
from GolemQ.utils.parameter import (
    AKA, 
    INDICATOR_FIELD as FLD, 
    TREND_STATUS as ST,
    FEATURES as FTR
)
from GolemQ.portfolio.utils import (
    calc_onhold_returns_v2,
)


"""
    # EEMD分解
    # 经验模式分解(empirical mode decomposition, EMD)方法是Huang提出的,
    # 它是一种新的时频分析方法, 而且是一种自适应的时频局部化分析方法:
    # ①IMF与采样频率相关;
    # ②它基于数据本身变化。
    # 这点是EMD分解优于快速傅立叶(FFT)变换和小波(Wavelet)变换方法的地方
"""

def calc_eemd_func(features,):
    '''
    # EEMD分解
    # 经验模式分解(empirical mode decomposition, EMD)方法是Huang提出的,
    # 它是一种新的时频分析方法, 而且是一种自适应的时频局部化分析方法:
    # ①IMF与采样频率相关;
    # ②它基于数据本身变化。
    # 这点是EMD分解优于快速傅立叶(FFT)变换和小波(Wavelet)变换方法的地方
    '''

    max_imf = 17
    S = features[FLD.MAXFACTOR].dropna().values
    T = range(len(S))
    eemd = EEMD() 
    eemd.trials = 50 
    eemd.noise_seed(12345)
    imfs = eemd.eemd(S, T, max_imf) 
    imfNo = imfs.shape[0]
    #emd = EMD()
    #emd.emd(S)
    #imfs, res = emd.get_imfs_and_residue()
    #imfNo = imfs.shape[0]
    tMin, tMax = np.min(T), np.max(T)

    # 补全NaN
    S = features[FLD.MAXFACTOR].values
    leakNum = len(S) - len(T)
    T = range(len(S))
    imfs = np.c_[np.full((imfNo, leakNum), np.nan), imfs]

    return imfs, imfNo


def calc_best_imf_periods(features, imfs, imfNo, 
                          symbol=None, display_name=None, taxfee=0.0003, annual=252, verbose=False):
    '''
    根据 imf 推断交易策略周期
    筛选出主升浪imf拟合函数和次级波浪imf拟合函数
    '''
    if (symbol is None):
        symbol = features.index.get_level_values(level=1)[0]
    if (display_name is None):
        display_name = symbol

    imf_periods = pd.DataFrame(columns=[AKA.CODE,
                                        'imf_num', 
                                        'imf', 
                                        'position',
                                        FLD.TURNOVER_RATIO,
                                        FLD.TURNOVER_RATIO_MEAN,
                                        'returns',
                                        FLD.TRANSACTION_RETURN_MEAN,
                                        FLD.ANNUAL_RETURN,
                                        FLD.ANNUAL_RETURN_MEAN])

    dea_zero_turnover_ratio_mean = features[FLD.DEA_ZERO_TURNOVER_RATIO].dropna().mean()
    best_imf4_candicate = None
    with np.errstate(invalid='ignore', divide='ignore'):
        for num in range(imfNo): 
            wavelet_cross = np.where(imfs[num] > np.r_[0, imfs[num, :-1]], 1, 0)

            # Turnover Analysis 简化计算，全仓操作 100%换手
            wavelet_turnover = np.where(wavelet_cross != np.r_[0, wavelet_cross[:-1]], 1, 0)
            wavelet_turnover_ratio = rolling_sum(wavelet_turnover, annual)
            wavelet_returns = features[FLD.PCT_CHANGE] * np.r_[0, wavelet_cross[:-1]]
            wavelet_annual_return = wavelet_returns.rolling(annual).apply(lambda x: 
                                                                      empyrical.annual_return(x, annualization=annual), 
                                                                      raw=True)

            turnover_ratio_mean = np.mean(wavelet_turnover_ratio[annual:])
            annual_return_mean = np.mean((wavelet_annual_return.values - wavelet_turnover_ratio * taxfee)[annual:])
            imf_periods = imf_periods.append(pd.Series({AKA.CODE:symbol,
                                                        'imf_num':num, 
                                                        'imf':imfs[num], 
                                                        ST.POSITION:wavelet_cross,
                                                        FLD.TURNOVER_RATIO:wavelet_turnover_ratio,
                                                        FLD.TURNOVER_RATIO_MEAN:turnover_ratio_mean,
                                                        'returns':wavelet_returns.values,
                                     FLD.TRANSACTION_RETURN_MEAN: annual_return_mean / (turnover_ratio_mean if (turnover_ratio_mean > 2) else 2 / 2),
                                     FLD.ANNUAL_RETURN:wavelet_annual_return.values - wavelet_turnover_ratio * taxfee,
                                     FLD.ANNUAL_RETURN_MEAN:annual_return_mean,}, 
                                              name=u"级别 {}".format(num + 1)))

            # 寻找 级别 4 imf (DEA 零轴 同步)
            wavelet_cross_timing_lag = calc_event_timing_lag(np.where(imfs[num] > np.r_[0, imfs[num, :-1]], 1, -1))
            if (dea_zero_turnover_ratio_mean > turnover_ratio_mean) and \
                (dea_zero_turnover_ratio_mean < (turnover_ratio_mean + 1)):
                # 我怀疑这是单边行情的特征，但是我没有证据
                best_imf4_candicate = u"级别 {}".format(num + 1)
            elif (dea_zero_turnover_ratio_mean < turnover_ratio_mean) and \
                (dea_zero_turnover_ratio_mean * 2 > turnover_ratio_mean):
                # 2倍采样率，理论上可以最大化似然 DEA波浪，未经数学证明
                best_imf4_candicate = u"级别 {}".format(num + 1)

            if (verbose):
                print('{} 级别{} 年化 returns:{:.2%}, 换手率 turnover:{:.0%}'.format(symbol, num + 1,
                                                                           wavelet_annual_return[-1].item(),
                                                                           wavelet_turnover_ratio[-1].item()))
    imf_periods = imf_periods.sort_values(by=FLD.ANNUAL_RETURN_MEAN, 
                                          ascending=False)
    #print(imf_periods[[FLD.ANNUAL_RETURN_MEAN,
    #                   FLD.TRANSACTION_RETURN_MEAN,
    #                   FLD.TURNOVER_RATIO_MEAN]])
    # 选择年化收益大于9%，并且按年化收益排序选出最高的4个imf波动周期，
    # 年化收益大于9.27%，单笔交易平均收益 > 3.82%，每年交易机会大于1次
    # 这将是策略进行操作的参照周期，
    # 其他的呢？对，没错，年化低于9%的垃圾标的你做来干啥？
    best_imf_periods = imf_periods.query('({}>0.0927 & {}>0.0382 & {}>3.82) | \
                                          ({}>0.168 & {}>0.0168 & {}>0.618) | \
                                          ({}>0.168 & {}>0.00618 & {}<82) | \
                                          ({}>0.382 & {}>0.618 & {}>0.618) | \
                                          ({}>0.927 & {}>3.82)'.format(FLD.ANNUAL_RETURN_MEAN, 
                                                                               FLD.TRANSACTION_RETURN_MEAN,
                                                                               FLD.TURNOVER_RATIO_MEAN,
                                                                               FLD.ANNUAL_RETURN_MEAN, 
                                                                               FLD.TRANSACTION_RETURN_MEAN,
                                                                               FLD.TURNOVER_RATIO_MEAN,
                                                                               FLD.ANNUAL_RETURN_MEAN, 
                                                                               FLD.TRANSACTION_RETURN_MEAN,
                                                                               FLD.TURNOVER_RATIO_MEAN,
                                                                               FLD.ANNUAL_RETURN_MEAN, 
                                                                               FLD.TURNOVER_RATIO_MEAN, 
                                                                               FLD.TRANSACTION_RETURN_MEAN,
                                                                               FLD.ANNUAL_RETURN_MEAN, 
                                                                               FLD.TURNOVER_RATIO_MEAN,)).head(4).copy()
    if (len(best_imf_periods) < 4):
        # 入选的级别数量不足，尝试补全
        print(symbol, u'入选的级别数量不足({} of 4)，尝试补全'.format(len(best_imf_periods)))
        rest_imf_periods = imf_periods.loc[imf_periods.index.difference(best_imf_periods.index), :].copy()
        rest_imf_periods = rest_imf_periods.sort_values(by=FLD.ANNUAL_RETURN_MEAN,
                                                        ascending=False)
        if (verbose):
            print(rest_imf_periods[[AKA.CODE,
                                    FLD.ANNUAL_RETURN_MEAN,
                                    FLD.TRANSACTION_RETURN_MEAN,
                                    FLD.TURNOVER_RATIO_MEAN]])
        if (len(best_imf_periods) < 3):
            best_imf_periods = best_imf_periods.append(imf_periods.loc[rest_imf_periods.index[[0, 1]], :])
        else:
            best_imf_periods = best_imf_periods.append(imf_periods.loc[rest_imf_periods.index[0], :])
        if (annual > 253):
            best_imf_periods = best_imf_periods.sort_values(by=FLD.TURNOVER_RATIO_MEAN,
                                                            ascending=False)
    if (verbose):
        print(best_imf_periods[[AKA.CODE,
                                FLD.ANNUAL_RETURN_MEAN,
                                FLD.TRANSACTION_RETURN_MEAN,
                                FLD.TURNOVER_RATIO_MEAN]])
    if (len(best_imf_periods) >= 2):
        if (annual > 253) and \
            (best_imf_periods.loc[best_imf_periods.index[0], FLD.TURNOVER_RATIO_MEAN] > 200):
            features[FTR.BEST_IMF1_TIMING_LAG] = calc_event_timing_lag(np.where(best_imf_periods.loc[best_imf_periods.index[1], 
                                                                                                     ST.POSITION] > 0,1,-1))
            features[FTR.BEST_IMF1] = best_imf_periods.loc[best_imf_periods.index[1], 'imf']
            features[FTR.BEST_IMF1_ANNUAL_RETURN] = best_imf_periods.loc[best_imf_periods.index[1], FLD.ANNUAL_RETURN]
            features[FTR.BEST_IMF1_NORM] = rolling_pctrank(features[FTR.BEST_IMF1].values, 84)

            features[FTR.BEST_IMF2_TIMING_LAG] = calc_event_timing_lag(np.where(best_imf_periods.loc[best_imf_periods.index[0], 
                                                                                                     ST.POSITION] > 0,1,-1))
            features[FTR.BEST_IMF2] = best_imf_periods.loc[best_imf_periods.index[0], 'imf']
            features[FTR.BEST_IMF2_ANNUAL_RETURN] = best_imf_periods.loc[best_imf_periods.index[0], FLD.ANNUAL_RETURN]
            features[FTR.BEST_IMF2_NORM] = rolling_pctrank(features[FTR.BEST_IMF2].values, 84)
        else:
            features[FTR.BEST_IMF1_TIMING_LAG] = calc_event_timing_lag(np.where(best_imf_periods.loc[best_imf_periods.index[0], 
                                                                                                     ST.POSITION] > 0,1,-1))
            features[FTR.BEST_IMF1] = best_imf_periods.loc[best_imf_periods.index[0], 'imf']
            features[FTR.BEST_IMF1_ANNUAL_RETURN] = best_imf_periods.loc[best_imf_periods.index[0], FLD.ANNUAL_RETURN]
            features[FTR.BEST_IMF1_NORM] = rolling_pctrank(features[FTR.BEST_IMF1].values, 84)

            features[FTR.BEST_IMF2_TIMING_LAG] = calc_event_timing_lag(np.where(best_imf_periods.loc[best_imf_periods.index[1], 
                                                                                                     ST.POSITION] > 0,1,-1))
            features[FTR.BEST_IMF2] = best_imf_periods.loc[best_imf_periods.index[1], 'imf']
            features[FTR.BEST_IMF2_ANNUAL_RETURN] = best_imf_periods.loc[best_imf_periods.index[1], FLD.ANNUAL_RETURN]
            features[FTR.BEST_IMF2_NORM] = rolling_pctrank(features[FTR.BEST_IMF2].values, 84)

    if (len(best_imf_periods) >= 3):
        if (best_imf4_candicate is None) or \
            (best_imf_periods.index[2] != best_imf4_candicate) or \
            (len(best_imf_periods) < 4):
            features[FTR.BEST_IMF3_TIMING_LAG] = calc_event_timing_lag(np.where(best_imf_periods.loc[best_imf_periods.index[2], 
                                                                                                     ST.POSITION] > 0,1,-1))
            features[FTR.BEST_IMF3] = best_imf_periods.loc[best_imf_periods.index[2], 'imf']
            features[FTR.BEST_IMF3_ANNUAL_RETURN] = best_imf_periods.loc[best_imf_periods.index[2], FLD.ANNUAL_RETURN]
        else:
            features[FTR.BEST_IMF3_TIMING_LAG] = calc_event_timing_lag(np.where(best_imf_periods.loc[best_imf_periods.index[3], 
                                                                                                     ST.POSITION] > 0,1,-1))
            features[FTR.BEST_IMF3] = best_imf_periods.loc[best_imf_periods.index[3], 'imf']
            features[FTR.BEST_IMF3_ANNUAL_RETURN] = best_imf_periods.loc[best_imf_periods.index[3], FLD.ANNUAL_RETURN]
            best_imf_periods.loc[best_imf_periods.index[2], :] = best_imf_periods.loc[best_imf_periods.index[3], :]
        features[FTR.BEST_IMF3_NORM] = rolling_pctrank(features[FTR.BEST_IMF3].values, 84)
    if (verbose):
        print(best_imf_periods[[AKA.CODE,
                                FLD.ANNUAL_RETURN_MEAN,
                                FLD.TRANSACTION_RETURN_MEAN,
                                FLD.TURNOVER_RATIO_MEAN]])
    if (len(best_imf_periods) >= 4) or \
        (best_imf4_candicate is not None):
        if (best_imf4_candicate is None):
            features[FTR.BEST_IMF4_TIMING_LAG] = calc_event_timing_lag(np.where(best_imf_periods.loc[best_imf_periods.index[3], 
                                                                                                     ST.POSITION] > 0,1,-1))
            features[FTR.BEST_IMF4] = best_imf_periods.loc[best_imf_periods.index[3], 'imf']
            features[FTR.BEST_IMF4_ANNUAL_RETURN] = best_imf_periods.loc[best_imf_periods.index[3], FLD.ANNUAL_RETURN]
            features[FLD.MACD_TREND_DENSITY] = best_imf_periods.loc[best_imf_periods.index[3], FLD.TURNOVER_RATIO]
            print(symbol, display_name, '1 震荡', 
                  round(best_imf_periods.loc[best_imf_periods.index[3], 
                                             FLD.TURNOVER_RATIO_MEAN], 3), 
                  round(dea_zero_turnover_ratio_mean, 3))
        else:
            features[FTR.BEST_IMF4_TIMING_LAG] = calc_event_timing_lag(np.where(imf_periods.loc[best_imf4_candicate, 
                                                                                                ST.POSITION] > 0,1,-1))
            features[FTR.BEST_IMF4] = imf_periods.loc[best_imf4_candicate, 'imf']
            features[FTR.BEST_IMF4_ANNUAL_RETURN] = imf_periods.loc[best_imf4_candicate, FLD.ANNUAL_RETURN]
            features[FLD.MACD_TREND_DENSITY] = imf_periods.loc[best_imf4_candicate]
            best_imf_periods.loc[best_imf_periods.index[3], :] = imf_periods.loc[best_imf4_candicate, :]
            print(symbol, best_imf4_candicate, 
                  '2 大概率主升浪' if (imf_periods.loc[best_imf4_candicate, FLD.TURNOVER_RATIO_MEAN] < 6.18) else '2 震荡', 
                  round(imf_periods.loc[best_imf4_candicate, 
                                        FLD.TURNOVER_RATIO_MEAN], 3), 
                  round(dea_zero_turnover_ratio_mean, 3),)
        features[FTR.BEST_IMF4_NORM] = rolling_pctrank(features[FTR.BEST_IMF4].values, 84)
    #print(best_imf_periods[[AKA.CODE,
    #                        FLD.ANNUAL_RETURN_MEAN,
    #                        FLD.TRANSACTION_RETURN_MEAN,
    #                        FLD.TURNOVER_RATIO_MEAN]])

    features[FTR.BEST_IMF1_RETURNS] = calc_onhold_returns_v2(features[FLD.PCT_CHANGE].values, 
                                                             features[FTR.BEST_IMF1_TIMING_LAG].values)
    features[FTR.BEST_IMF1_TRANSACTION_RETURNS] = np.where((features[FTR.BEST_IMF1_TIMING_LAG] <= 0) & \
        (features[FTR.BEST_IMF1_TIMING_LAG].shift() > 0),features[FTR.BEST_IMF1_RETURNS], 0)

    return features

    portfolio_briefs = pd.DataFrame(columns=['symbol',
                                             'name',
                                             'portfolio',
                                             'sharpe',
                                             'annual_return',
                                             'max_drawdown',
                                             'turnover_ratio'])
    if (FTR.BEST_IMF1_TIMING_LAG in features.columns) and (FTR.BEST_IMF2_TIMING_LAG in features.columns):
        wavelet_cross = np.where((features[FTR.BEST_IMF1_TIMING_LAG] + features[FTR.BEST_IMF2_TIMING_LAG]) > 0, 1, 0)
        features[FTR.BOOST_IMF_TIMING_LAG] = wavelet_cross
        portfolio_briefs = portfolio_briefs.append(calc_strategy_stats(symbol, 
                                                                       display_name, 
                                                                       'EEMD Boost', 
                                                                       wavelet_cross, 
                                                                       features), ignore_index=True)
    else:
        print('Code {}, {} EMD分析失败。\n'.format(symbol, display_name))
        QA.QA_util_log_info('Code {}, {} EMD分析失败。\n'.format(symbol, display_name))


