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
import pandas as pd
import scipy as sp
from sklearn import preprocessing as skp

from GolemQ.utils.parameter import (
    AKA, 
    INDICATOR_FIELD as FLD, 
    TREND_STATUS as ST,
    FEATURES as FTR,
    )


def prepar_feature_conditions(features,
                              model='xgboost_baseline'):
    if (model == 'xgboost_baseline') or \
        (model.startswith('xgboost_baseline')):
        features['FTR_01'] = features[FLD.DRAWDOWN_RATIO_MAJOR] + features[FLD.DRAWDOWN_RATIO] + features[FLD.MAPOWER30_MAJOR]
        features['FTR_02'] = features[FLD.MAPOWER30_MAJOR] + features[FLD.HMAPOWER120_MAJOR] + features[FLD.MAPOWER30]
        features['COND_01'] = (features[FLD.NEGATIVE_LOWER_PRICE_BEFORE] > features[FLD.MAPOWER30_PEAK_LOW_BEFORE])
        features['COND_02'] = (features[FLD.MAPOWER_BASELINE] > features[FLD.MAPOWER_BASELINE].shift(8))
        features['COND_03'] = (features[FLD.ATR_LB] > features[FLD.BOLL_LB])
        features['COND_04'] = ((features[FLD.ATR_LB] - features[FLD.BOLL_LB]) > (features[FLD.ATR_LB].shift(1) - features[FLD.BOLL_LB].shift(1))) 
        features['COND_05'] = (features[FLD.MA120] > features[FLD.MA90])
        features['COND_06'] = (features[FLD.MAPOWER30_PEAK_LOW_BEFORE] < features[FLD.MAPOWER30_PEAK_HIGH_BEFORE]) & \
                                ((features[FLD.MAPOWER30_PEAK_LOW_BEFORE] + features[FLD.DEADPOOL_REMIX_TIMING_LAG]) < 0) & \
                                ((features[FLD.MAPOWER_HMAPOWER120_TIMING_LAG] + features[FLD.DEADPOOL_REMIX_TIMING_LAG]) < 0)
        features['COND_07'] = (features[FLD.DEA] < features[FLD.MACD_PEAK_LOW_BACKWARD])
        features['COND_08'] = ((features[FLD.ATR_LB] > features[FLD.ATR_LB].shift(1)) & \
                                    (features[FLD.BOLL_LB] > features[FLD.BOLL_LB].shift(1)))
        features['COND_09'] = (features[FLD.POLYNOMIAL9] < features[FLD.ATR_LB])
        features['COND_10'] = ((features[FLD.BOLL_LB_HMA5_TIMING_LAG] + features[FLD.DEADPOOL_REMIX_TIMING_LAG]) > 0)
        features['COND_11'] = (features[FLD.HMAPOWER120_MAJOR] > features[FLD.MAPOWER30_MAJOR])
        features['COND_12'] = (features[FLD.ATR_LB] > features[FLD.ATR_LB].shift(1)) & \
                                    ((features[FLD.BOLL_LB] > features[FLD.BOLL_LB].shift(1)) | \
                                    (features[FLD.BOLL_DELTA] > 0))
    elif (model == 'xgboost_uprising') or \
        (model.startswith('xgboost_uprising')):
        features['FTR_01'] = features[FLD.DRAWDOWN_RATIO_MAJOR] + features[FLD.DRAWDOWN_RATIO] + features[FLD.MAPOWER30_MAJOR]
        features['FTR_02'] = features[FLD.MAPOWER30_MAJOR] + features[FLD.HMAPOWER120_MAJOR] + features[FLD.MAPOWER30]
        features['COND_01'] = (features[FLD.NEGATIVE_LOWER_PRICE_BEFORE] > features[FLD.MAPOWER30_PEAK_LOW_BEFORE])
        features['COND_02'] = (features[FLD.MAPOWER_BASELINE] > features[FLD.MAPOWER_BASELINE].shift(8))
        features['COND_03'] = (features[FLD.ATR_LB] > features[FLD.BOLL_LB])
        features['COND_04'] = ((features[FLD.ATR_LB] - features[FLD.BOLL_LB]) > (features[FLD.ATR_LB].shift(1) - features[FLD.BOLL_LB].shift(1))) 
        features['COND_05'] = (features[FLD.MA120] > features[FLD.MA90])
        features['COND_06'] = (features[FLD.MAPOWER30_PEAK_LOW_BEFORE] < features[FLD.MAPOWER30_PEAK_HIGH_BEFORE]) & \
                                ((features[FLD.MAPOWER30_PEAK_LOW_BEFORE] + features[FLD.DEADPOOL_REMIX_TIMING_LAG]) < 0) & \
                                ((features[FLD.MAPOWER_HMAPOWER120_TIMING_LAG] + features[FLD.DEADPOOL_REMIX_TIMING_LAG]) < 0)
        features['COND_07'] = (features[FLD.DEA] < features[FLD.MACD_PEAK_LOW_BACKWARD])
        features['COND_08'] = ((features[FLD.ATR_LB] > features[FLD.ATR_LB].shift(1)) & \
                              (features[FLD.BOLL_LB] > features[FLD.BOLL_LB].shift(1)))
        features['COND_09'] = (features[FLD.POLYNOMIAL9] < features[FLD.ATR_LB])
        features['COND_10'] = ((features[FLD.BOLL_LB_HMA5_TIMING_LAG] + features[FLD.DEADPOOL_REMIX_TIMING_LAG]) > 0)
        features['COND_11'] = (features[FLD.HMAPOWER120_MAJOR] > features[FLD.MAPOWER30_MAJOR])
        features['COND_12'] = (features[FLD.ATR_LB] > features[FLD.ATR_LB].shift(1)) & \
                                    ((features[FLD.BOLL_LB] > features[FLD.BOLL_LB].shift(1)) | \
                                    (features[FLD.BOLL_DELTA] > 0))
        features['COND_13'] = (features[FLD.MACD_ZERO_TIMING_LAG] > 0) & \
                              (features[FLD.HMAPOWER120_TIMING_LAG] < 0)
        features['COND_14'] = (features[FLD.MAINFEST_UPRISING_COEFFICIENT] > 0) & \
            (features[FLD.MAINFEST_UPRISING_COEFFICIENT] > features[FLD.DEA_ZERO_TIMING_LAG])
        features['COND_15'] = (features[FLD.MAPOWER30_TIMING_LAG_MAJOR] > 0) & \
            (features[FLD.MAPOWER30_TIMING_LAG_MAJOR] > features[FLD.DEA_ZERO_TIMING_LAG])
        features['COND_16'] = (features[FLD.HMAPOWER120_TIMING_LAG_MAJOR] > 0) & \
            (features[FLD.HMAPOWER120_TIMING_LAG_MAJOR] < np.minimum(features[FLD.MAPOWER30_PEAK_LOW_BEFORE],
                                                                     features[FLD.MAPOWER30_PEAK_LOW_BEFORE_MAJOR]))
        features['COND_17'] = (features[FLD.MA90] > features[FLD.MA120]) & \
                      (((features[FLD.MA90_CLEARANCE_TIMING_LAG] + features[FLD.MAINFEST_UPRISING_TIMING_LAG]) > 0) | \
                      (((features[FLD.MA90_TREND_TIMING_LAG] + features[FLD.MAINFEST_UPRISING_TIMING_LAG]) > 0) & \
                      (features[FLD.MA90_CLEARANCE] > -0.618) & \
                      (features[FLD.MA_CHANNEL] < 0.0382)))
        features['COND_18'] = ((features[FLD.MAPOWER30_PEAK_LOW_BEFORE] + features[FLD.DEADPOOL_REMIX_TIMING_LAG]) < 0)
        features['COND_19'] = ((features[FLD.HMAPOWER120_PEAK_LOW_BEFORE] + features[FLD.DEADPOOL_REMIX_TIMING_LAG]) < 0)
        
    return features


def feature_names(model='xgboost_baseline'):
    ret_feature_names = []
    if (model == 'xgboost_baseline') or \
        (model.startswith('xgboost_baseline')):
        ret_feature_names = [FLD.NEGATIVE_LOWER_PRICE_BEFORE,
            FLD.DEA_ZERO_TIMING_LAG,
            FLD.DIF_ZERO_TIMING_LAG,
            FLD.MACD_ZERO_TIMING_LAG,
            FLD.DEA_INTERCEPT_TIMING_LAG,
            FLD.MACD_INTERCEPT,
            FLD.CCI,
            FLD.RSI,
            FLD.PEAK_OPEN,
            FLD.MAXFACTOR,
            FLD.MAPOWER30,
            FLD.HMAPOWER120,
            FLD.MAPOWER30_MAJOR,
            FLD.HMAPOWER120_MAJOR,
            FLD.HMA10_CLEARANCE,
            FLD.HMA10_CLEARANCE_ZSCORE,
            FLD.PEAK_LOW_TIMING_LAG,
            FLD.BOLL_LB_HMA5_TIMING_LAG,
            FLD.BOLL_JX_RSI,
            FLD.BOLL_JX_MAXFACTOR,
            FLD.BOLL_JX_MAPOWER30,
            FLD.BOLL_JX_HMAPOWER120,
            FLD.DRAWDOWN_RATIO,
            FLD.DRAWDOWN_RATIO_MAJOR,
            FLD.MAPOWER30_TIMING_LAG_MAJOR,
            FLD.MAPOWER_BASELINE_TIMING_LAG_MAJOR,
            FLD.MAINFEST_UPRISING_COEFFICIENT,
            FTR.BOOTSTRAP_ENHANCED_TIMING_LAG,
            FLD.DEADPOOL_REMIX_TIMING_LAG,
            FLD.DEADPOOL_CANDIDATE_TIMING_LAG,
            FLD.MAPOWER30_TIMING_LAG,
            FLD.HMAPOWER120_TIMING_LAG,
            FLD.POLYNOMIAL9_TIMING_LAG,
            FLD.ZEN_WAVELET_TIMING_LAG,
            'FTR_01',
            'FTR_02',
            'COND_01',
            'COND_02',
            'COND_03',
            'COND_04',
            'COND_05',
            'COND_06',
            'COND_07',
            'COND_08',
            'COND_09',
            'COND_10',
            'COND_11',
            'COND_12',]
    elif (model == 'xgboost_uprising') or \
        (model.startswith('xgboost_uprising')):
        ret_feature_names = [FLD.NEGATIVE_LOWER_PRICE_BEFORE,
            FLD.DEA_ZERO_TIMING_LAG,
            FLD.DIF_ZERO_TIMING_LAG,
            FLD.MACD_ZERO_TIMING_LAG,
            FLD.DEA_INTERCEPT_TIMING_LAG,
            FLD.MACD_INTERCEPT,
            FLD.CCI,
            FLD.RSI,
            FLD.PEAK_OPEN,
            FLD.MAXFACTOR,
            FLD.MAPOWER30,
            FLD.HMAPOWER120,
            FLD.MAPOWER30_MAJOR,
            FLD.HMAPOWER120_MAJOR,
            FLD.HMA10_CLEARANCE,
            FLD.HMA10_CLEARANCE_ZSCORE,
            FLD.PEAK_LOW_TIMING_LAG,
            FLD.BIAS3,
            FLD.BIAS3_TREND_TIMING_LAG,
            FLD.LINEAREG_BAND_TIMING_LAG,
            FLD.BOLL_LB_HMA5_TIMING_LAG,
            FLD.BOLL_JX_RSI,
            FLD.BOLL_JX_MAXFACTOR,
            FLD.BOLL_JX_MAPOWER30,
            FLD.BOLL_JX_HMAPOWER120,
            FLD.DRAWDOWN_RATIO,
            FLD.DRAWDOWN_RATIO_MAJOR,
            FLD.MAPOWER30_TIMING_LAG_MAJOR,
            FLD.MAPOWER_BASELINE_TIMING_LAG_MAJOR,
            FLD.MAINFEST_UPRISING_COEFFICIENT,
            FTR.BOOTSTRAP_ENHANCED_TIMING_LAG,
            FLD.DEADPOOL_REMIX_TIMING_LAG,
            FLD.DEADPOOL_CANDIDATE_TIMING_LAG,
            FLD.MAPOWER30_TIMING_LAG,
            FLD.HMAPOWER120_TIMING_LAG,
            FLD.POLYNOMIAL9_TIMING_LAG,
            FLD.ZEN_WAVELET_TIMING_LAG,
            'FTR_01',
            'FTR_02',
            'COND_01',
            'COND_02',
            'COND_03',
            'COND_04',
            'COND_05',
            'COND_06',
            'COND_07',
            'COND_08',
            'COND_09',
            'COND_10',
            'COND_11',
            'COND_12',
            'COND_13',
            'COND_14',
            'COND_15',
            'COND_16',
            'COND_17',
            'COND_18',
            'COND_19',]
    return ret_feature_names


def winsorize_quantile(factor, up, down):
    '''
    参考 scipy.stats.mstats.winsorize(a, limits=None)

    Return a Winsorized version of the input array Parameters:
        a : sequence Input array
        limits : float 数据两端的 percentile 的值

    自实现分位数去极值
    大于（小于）分位值点 的 数据 用 分位值 替换
    '''
    # 求出分位数值：
    up_scale = np.percentile(factor, up)
    down_scale = np.percentile(factor, down)
    factor = np.where(factor > up_scale, up_scale, factor)
    factor = np.where(factor < down_scale, down_scale, factor)
    return factor
    from scipy.stats.mstats import winsorize
    sp.stats.mstats.winsorize(a, limits=None)


def winsorize_med(factor):
    '''
    实现3倍中位数绝对偏差去极值
    '''
    if (factor.max() <= 1) and (factor.min() >= -1):
        return factor

    # 1、找出因子的中位数
    me = np.median(factor)
    
    # 2、计算 | x - median |
    # 3、计算 MAD, median( | x - median| )
    mad = np.median(abs(factor - me))
    
    # 4、
    up = me + (3 * 1.4826 * mad)
    down = me - (3 * 1.4826 * mad)
    
    # 5、
    with np.errstate(invalid='ignore'):
        factor = np.where(factor > up,up,factor)
        factor = np.where(factor < down,down,factor)
    return factor


def winsorize_threesigma(factor):
    '''
    自实现正态分布去极值
    '''
    if (factor.max() <= 1) and (factor.min() >= -1):
        return factor

    mean = factor.mean()
    std = factor.std()
    
    up = mean + 3 * std
    down = mean - 3 * std
    
    with np.errstate(invalid='ignore'):
        factor = np.where(factor > up,up,factor)
        factor = np.where(factor < down,down,factor)
    return factor 


def standardize(s,ty=2):
    '''
    标准化函数
    s为Series数据
    ty为标准化类型:1 MinMax,2 Standard,3 maxabs 
    '''
    data = s.dropna().copy()
    if int(ty) == 1:
        re = (data - data.min()) / (data.max() - data.min())
    elif ty == 2:
        re = (data - data.mean()) / data.std()
    elif ty == 3:
        re = data / 10 ** np.ceil(np.log10(data.abs().max()))
    return re
    

def normalize(x):
    '''
    标准化函数
    s为Series数据
    ty为标准化类型:1 MinMax,2 Standard,3 maxabs 
    '''
    scaler = skp.StandardScaler()
    scaler.fit(x)
    x_norm = scaler.transform(x)

    return x_norm


def dev_features_rolling(ref_features,
                         columns,
                         ndays):
    """
    生成技术指标的滚动窗口特征
    """
    features = ref_features[columns].rolling(ndays).agg(['mean',
                                                         'max',
                                                         'min',
                                                         'std',
                                                         'var',
                                                         'median'])
    features.columns = ["_".join(col) for col in features.columns]
    features.columns = features.columns + '_rl_' + str(ndays) + 'T'
    ret_features = pd.merge(ref_features,
                   features,
                   left_index=True,
                   right_index=True,
                   how='inner')
    return ret_features


def dev_features_diff(ref_features,
                      columns,
                      ndays):
    """
    生成技术指标的差分特征
    """
    features = ref_features[columns].diff(ndays)
    features.columns = features.columns + '_diff_' + str(ndays) + 'T'
    ret_features = pd.merge(ref_features,
                   features,
                   left_index=True,
                   right_index=True,
                   how='inner')
    return ret_features


def dev_features_diff2(ref_features,
                       columns):
    """
    生成技术指标的差分特征 2
    """
    features = ref_features[columns].diff(1).diff(1)
    features.columns = features.columns + '_diff2'
    ret_features = pd.merge(ref_features,
                            features,
                            left_index=True,
                            right_index=True,
                            how='inner')
    return ret_features


def dev_features_lag(ref_features,
                     columns,
                     ndays):
    """
    生成技术指标的回溯(n天前)特征
    """
    features = ref_features[columns].shift(ndays)
    features.columns = features.columns + '_lag_' + str(ndays) + 'T'
    ret_features = pd.merge(ref_features,
                   features,
                   left_index=True,
                   right_index=True,
                   how='inner')
    return ret_features


def ts_rank(x):
    return pd.Series(x).rank().tail(1)


def ts_rankeq10(x):
    res = (x == 10) * 1
    return res.sum()
    
