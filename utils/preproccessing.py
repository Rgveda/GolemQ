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
    
