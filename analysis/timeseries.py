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
import numba as nb
import scipy.stats as scs
import scipy.signal as signal
#from scipy.signal import lfilter, lfilter_zi, filtfilt, butter, savgol_filter
try:
    import peakutils
except:
    #print('PLEASE run "pip install peakutils" to call these modules')
    pass
try:
    import QUANTAXIS as QA
    from QUANTAXIS.QAIndicator.base import *
    from QUANTAXIS.QAUtil.QADate_Adv import (
        QA_util_timestamp_to_str,
        QA_util_datetime_to_Unix_timestamp,
        QA_util_print_timestamp
    )
except:
    print('PLEASE run "pip install QUANTAXIS" before call GolemQ.analysis.timeseries modules')
    pass
from GolemQ.utils.parameter import (
    AKA, 
    INDICATOR_FIELD as FLD, 
    TREND_STATUS as ST
)

"""
时序信号处理，公共函数
时间序列分析工具函数
"""

from timeit import timeit, Timer
time_unit = None
units = {"nsec": 1e-9, u"μsec": 1e-6, "msec": 1e-3, "sec": 1.0}
precision = 3

def format_time(dt):
    unit = time_unit

    if unit is not None:
        scale = units[unit]
    else:
        scales = [(scale, unit) for unit, scale in units.items()]
        scales.sort(reverse=True)
        for scale, unit in scales:
            if dt >= scale:
                break

    return "%.*g %s" % (precision, dt / scale, unit)

def timeit_all(t, number=1000):
    repeat = 5
    raw_timings = t.repeat(repeat=repeat, number=number)
    timings = [dt / number for dt in raw_timings]

    best = min(timings)
    return "{:d} loop{:s}, best of {:d}: {:s} per loop".format(number, 
                                                            's' if number != 1 else '',
                                                            repeat, 
                                                            format_time(best))

class real_time_peak_detection():
    def __init__(self, array, lag, threshold, influence):
        self.y = list(array)
        self.length = len(self.y)
        self.lag = lag
        self.threshold = threshold
        self.influence = influence
        self.signals = [0] * len(self.y)
        self.filteredY = np.array(self.y).tolist()
        self.avgFilter = [0] * len(self.y)
        self.stdFilter = [0] * len(self.y)
        self.avgFilter[self.lag - 1] = np.mean(self.y[0:self.lag]).tolist()
        self.stdFilter[self.lag - 1] = np.std(self.y[0:self.lag]).tolist()

    def thresholding_algo(self, new_value):
        self.y.append(new_value)
        i = len(self.y) - 1
        self.length = len(self.y)
        if i < self.lag:
            return 0
        elif i == self.lag:
            self.signals = [0] * len(self.y)
            self.filteredY = np.array(self.y).tolist()
            self.avgFilter = [0] * len(self.y)
            self.stdFilter = [0] * len(self.y)
            self.avgFilter[self.lag - 1] = np.mean(self.y[0:self.lag]).tolist()
            self.stdFilter[self.lag - 1] = np.std(self.y[0:self.lag]).tolist()
            return 0

        self.signals += [0]
        self.filteredY += [0]
        self.avgFilter += [0]
        self.stdFilter += [0]

        if abs(self.y[i] - self.avgFilter[i - 1]) > self.threshold * self.stdFilter[i - 1]:
            if self.y[i] > self.avgFilter[i - 1]:
                self.signals[i] = 1
            else:
                self.signals[i] = -1

            self.filteredY[i] = self.influence * self.y[i] + (1 - self.influence) * self.filteredY[i - 1]
            self.avgFilter[i] = np.mean(self.filteredY[(i - self.lag):i])
            self.stdFilter[i] = np.std(self.filteredY[(i - self.lag):i])
        else:
            self.signals[i] = 0
            self.filteredY[i] = self.y[i]
            self.avgFilter[i] = np.mean(self.filteredY[(i - self.lag):i])
            self.stdFilter[i] = np.std(self.filteredY[(i - self.lag):i])

        return self.signals[i]


def thresholding_algo_old(y, lag, threshold, influence):
    signals = np.zeros(len(y))
    filteredY = np.array(y)
    avgFilter = [0] * len(y)
    stdFilter = [0] * len(y)
    avgFilter[lag - 1] = np.mean(y[0:lag])
    stdFilter[lag - 1] = np.std(y[0:lag])
    for i in range(lag, len(y)):
        if abs(y[i] - avgFilter[i - 1]) > threshold * stdFilter[i - 1]:
            if y[i] > avgFilter[i - 1]:
                signals[i] = 1
            else:
                signals[i] = -1

            filteredY[i] = influence * y[i] + (1 - influence) * filteredY[i - 1]
            avgFilter[i] = np.mean(filteredY[(i - lag + 1):i + 1])
            stdFilter[i] = np.std(filteredY[(i - lag + 1):i + 1])
        else:
            signals[i] = 0
            filteredY[i] = y[i]
            avgFilter[i] = np.mean(filteredY[(i - lag + 1):i + 1])
            stdFilter[i] = np.std(filteredY[(i - lag + 1):i + 1])

    return dict(signals = np.asarray(signals),
                avgFilter = np.asarray(avgFilter),
                stdFilter = np.asarray(stdFilter))


@nb.jit(nopython=True)
def thresholding_algo(y, lag, threshold, influence):
    """
    Robust peak detection algorithm (using z-scores)
    自带鲁棒性极值点识别，利用方差和ZSCORE进行时间序列极值检测。算法源自：
    https://stackoverflow.com/questions/22583391/
    本实现使用Numba JIT优化，比原版（上面）大约快了500倍。
    """
    ret_signals = np.zeros((3, len(y),))
    idx_signals = 0
    idx_avgFilter = 1
    idx_stdFilter = 2

    filteredY = np.copy(y)
    ret_signals[idx_avgFilter, lag - 1] = np.mean(y[0:lag])
    ret_signals[idx_stdFilter, lag - 1] = np.std(y[0:lag])
    for i in range(lag, len(y)):
        if abs(y[i] - ret_signals[idx_avgFilter, 
                                  i - 1]) > threshold * ret_signals[idx_stdFilter, 
                                                                    i - 1]:
            if y[i] > ret_signals[idx_avgFilter, i - 1]:
                ret_signals[idx_signals, i] = 1
            else:
                ret_signals[idx_signals, i] = -1

            filteredY[i] = influence * y[i] + (1 - influence) * filteredY[i - 1]
            ret_signals[idx_avgFilter, i] = np.mean(filteredY[(i - lag + 1):i + 1])
            ret_signals[idx_stdFilter, i] = np.std(filteredY[(i - lag + 1):i + 1])
        else:
            ret_signals[idx_signals, i] = 0
            filteredY[i] = y[i]
            ret_signals[idx_avgFilter, i] = np.mean(filteredY[(i - lag + 1):i + 1])
            ret_signals[idx_stdFilter, i] = np.std(filteredY[(i - lag + 1):i + 1])

    return ret_signals


def calc_peak_points(closep, lineareg_price):
    lag = 5
    threshold = 3.5
    influence = 0.5

    # Run algo with settings from above
    peak_point_candicate = []
    peak_point_candicate.append(thresholding_algo(closep, 
                                                  lag=lag, threshold=threshold, 
                                                  influence=influence)[0, :])

    peak_point_candicate.append(thresholding_algo(lineareg_price, 
                                                  lag=lag, threshold=threshold, 
                                                  influence=influence)[0, :])

    ret_peak_points = np.zeros(len(closep))
    for i in range(0, len(peak_point_candicate)):
        peak_point = peak_point_candicate[i]
        ret_peak_points = np.where(peak_point != 0, 
                                   peak_point * (9 - i), 
                                   ret_peak_points)
    return ret_peak_points


def QA_indicator_Get_Drawdown_Rate(stockPrice=pd.DataFrame, 
                                   peak_name='close'):
    """
    最大回撤
    """
    if(len(stockPrice) > 0):
        Peak_Price = stockPrice[peak_name].max()
        Drawdown = stockPrice.iat[-1, stockPrice.columns.get_loc(AKA.CLOSE)] - Peak_Price
        return Drawdown / Peak_Price
    else:
        return 0


def time_series_momemtum(price, n=24, rf=0.02):
    """
    时间序列动量指标
    Time Series Momentum strategy
    """
    return (price / price.shift(n) - 1) - rf


def find_peak_vextors(highp, lowp,):
    """
    使用 scipy.argrelextrema 算法进行

    Parameters
    ----------
    price : (N,) array_like
        传入需要查找极值点的价格-时间序列。
        The numerator coefficient vector of the filter.
    return_ref : bool or None, optional
        返回作为参照的平滑曲线，平滑曲线的目的是减少锯齿抖动，减少计算的极值点。
        Return the smoothed line for reference.
    offest : int or None, optional
        传递参数时可能会被 .dropna() 或者 price[t:0] 等切片手段移除 nparray 头部
        的 np.nan 元素，因为此函数返回的是向量节点的数组索引，为了跟原始参数对应，调用者
        可以指定一个补偿偏移量，在返回的最大最小值中每个索引都会追加这个偏移量。
        The number of elements index offest, for jump np.nan in price's head.

    Returns
    -------
    x_tp_min, x_tp_max : ndarray
        包含最大值和最少值索引的数组
        The min/max peakpoint's index in array.

    """
    # pass 1
    x_peak_lowP, x_peak_highP = signal.argrelextrema(lowp,np.less)[0], signal.argrelextrema(highp,np.greater)[0]

    ret_data_len = len(lowp)
    ret_peak_cross = np.zeros(ret_data_len)
    np.put(ret_peak_cross, x_peak_lowP, 1)
    np.put(ret_peak_cross, x_peak_highP, -1)

    return ret_peak_cross


def find_peak_vextors_smoothly(price, offest=0):
    """
    采用 savgol 信号滤波器，自适应寻找最佳极值点，决定平均周期分段数量。
    使用 scipy.argrelextrema 机器学习统计算法进行第二次分析

    Parameters
    ----------
    price : (N,) array_like
        传入需要查找极值点的价格-时间序列。
        The numerator coefficient vector of the filter.
    return_ref : bool or None, optional
        返回作为参照的平滑曲线，平滑曲线的目的是减少锯齿抖动，减少计算的极值点。
        Return the smoothed line for reference.
    offest : int or None, optional
        传递参数时可能会被 .dropna() 或者 price[t:0] 等切片手段移除 nparray 头部
        的 np.nan 元素，因为此函数返回的是向量节点的数组索引，为了跟原始参数对应，调用者
        可以指定一个补偿偏移量，在返回的最大最小值中每个索引都会追加这个偏移量。
        The number of elements index offest, for jump np.nan in price's head.

    Returns
    -------
    x_tp_min, x_tp_max : ndarray
        包含最大值和最少值索引的数组
        The min/max peakpoint's index in array.

    """
    # pass 0
    window_size, poly_order = 5, 1
    y = signal.savgol_filter(price, window_size, poly_order)

    # pass 1
    x_tp_min, x_tp_max = signal.argrelextrema(y, np.less)[0], signal.argrelextrema(y, np.greater)[0]

    return x_tp_min + offest, x_tp_max + offest


def find_peak_vextors_eagerly(price, smooth_ma5=[], offest=0):
    """
    （饥渴的）在 MACD 上坡的时候查找更多的极值点
    """
    xn = price

    # pass 0
    if (len(smooth_ma5) == len(price)):
        yy_sg = smooth_ma5
    else:
        yy_sg = np.r_[np.zeros(11), TA_HMA(xn, 10)[11:]]

    # pass 1
    x_tp_min, x_tp_max = signal.argrelextrema(yy_sg, np.less)[0], signal.argrelextrema(yy_sg, np.greater)[0]
    n = int(len(price) / (len(x_tp_min) + len(x_tp_max)))

    # peakutils 似乎一根筋只能查最大极值，通过曲线反相的方式查找极小点
    print('mean() here')
    mirrors = (yy_sg * -1) + np.mean(price) * 2
    print('mean() done')

    # pass 2 使用 peakutils 查找
    x_tp_max = peakutils.indexes(yy_sg, thres=0.01 / max(price), min_dist=n)
    x_tp_min = peakutils.indexes(mirrors, thres=0.01 / max(price), min_dist=n)

    return x_tp_min + offest, x_tp_max + offest


#def find_peak_vextors(price, return_ref=False, offest=0):
#    """
#    采用巴特沃斯信号滤波器，自适应寻找最佳极值点，决定平均周期分段数量。
#    使用 scipy.Gaussian 机器学习统计算法进行第二次分析
#    If you meet a Warning message, To slove this need upgrade scipy=>1.2.
#    but QUANTAXIS incompatible scipy=>1.2

#    Parameters
#    ----------
#    price : (N,) array_like
#        传入需要查找极值点的价格-时间序列。
#        The numerator coefficient vector of the filter.
#    return_ref : bool or None, optional
#        返回作为参照的平滑曲线，平滑曲线的目的是减少锯齿抖动，减少计算的极值点。
#        Return the smoothed line for reference.
#    offest : int or None, optional
#        传递参数时可能会被 .dropna() 或者 price[t:0] 等切片手段移除 nparray 头部
#        的 np.nan 元素，因为此函数返回的是向量节点的数组索引，为了跟原始参数对应，调用者
#        可以指定一个补偿偏移量，在返回的最大最小值中每个索引都会追加这个偏移量。
#        The number of elements index offest, for jump np.nan in price's head.

#    Returns
#    -------
#    x_tp_min, x_tp_max : ndarray
#        包含最大值和最少值索引的数组
#        The min/max peakpoint's index in array.

#    """
#    xn = price

#    # Create an order 3 lowpass butterworth filter.
#    b, a = butter(3, 0.05)

#    # Apply the filter to xn.  Use lfilter_zi to choose the initial condition
#    # of the filter.
#    zi = lfilter_zi(b, a)
#    z, _ = lfilter(b, a, xn, zi=zi * xn[0])

#    # Apply the filter again, to have a result filtered at an order
#    # the same as filtfilt.
#    z2, _ = lfilter(b, a, z, zi=zi * z[0])

#    # Use filtfilt to apply the filter.  If you meet a Warning need upgrade to
#    # scipy=>1.2 but QUANTAXIS incompatible scipy=>1.2
#    y = filtfilt(b, a, xn)

#    # pass 1
#    x_tp_min, x_tp_max = signal.argrelextrema(y, np.less)[0],
#    signal.argrelextrema(y, np.greater)[0]
#    n = int(len(price) / (len(x_tp_min) + len(x_tp_max))) * 2

#    # peakutils 似乎一根筋只能查最大极值，通过曲线反相的方式查找极小点
#    mirrors = (price * -1) + np.mean(price) * 2

#    # pass 2 使用 peakutils 查找
#    x_tp_max = peakutils.indexes(price, thres=0.01 / max(price), min_dist=n)
#    x_tp_min = peakutils.indexes(mirrors, thres=0.01 / max(price), min_dist=n)

#    if (return_ref):
#        return x_tp_min + offest, x_tp_max + offest, y
#    else:
#        return x_tp_min + offest, x_tp_max + offest
@nb.jit(nopython=True)
def Timeline_Integral(Tm:np.ndarray,) -> np.ndarray:
    """
    explanation:
        计算时域金叉/死叉信号的累积卷积和(死叉(1-->0)清零)，经测试for实现最快，比reduce快	

    params:
        * Tm ->:
            meaning:
            type: null
            optional: [null]

    return:
        np.array

    demonstrate:
        Not described

    output:
        Not described
    """
    T = np.zeros(len(Tm)).astype(np.int32)
    for i, Tmx in enumerate(Tm):
        T[i] = Tmx * (T[i - 1] + Tmx)
    return T


@nb.jit(nopython=True)
def Timeline_duration(Tm:np.ndarray,) -> np.ndarray:
    """
    explanation:
         计算时域金叉/死叉信号的累积卷积和(死叉(1-->0)不清零，金叉(0-->1)清零)		
         经测试for最快，比reduce快(无jit，jit的话for就更快了)

    params:
        * Tm ->:
            meaning: 数据
            type: null
            optional: [null]

    return:
        np.array
	
    demonstrate:
        Not described
	
    output:
        Not described
    """
    T = np.zeros(len(Tm)).astype(np.int32)
    for i, Tmx in enumerate(Tm):
        T[i] = (T[i - 1] + 1) if (Tmx != 1) else 0
    return T


@nb.jit(nopython=True)
def LIS(X:np.ndarray,) -> np.ndarray:
    """
    explanation:
        计算最长递增子序列		

    params:
        * X ->:
            meaning: 序列
            type: null
            optional: [null]

    return:
        (子序列开始位置, 子序列结束位置)

    demonstrate:
        Not described

    output:
        Not described
    """
    N = len(X)
    P = [0] * N
    M = [0] * (N + 1)
    L = 0
    for i in range(N):
        lo = 1
        hi = L
        while lo <= hi:
            mid = (lo + hi) // 2
            if (X[M[mid]] < X[i]):
                lo = mid + 1
            else:
                hi = mid - 1

        newL = lo
        P[i] = M[newL - 1]
        M[newL] = i

        if (newL > L):
            L = newL

    S = []
    pos = []
    k = M[L]
    for i in range(L - 1, -1, -1):
        S.append(X[k])
        pos.append(k)
        k = P[k]
    return S[::-1], pos[::-1]


@nb.jit(nopython=True)
def LDS(X:np.ndarray,) -> np.ndarray:
    """
    explanation:
        计算最长递减子序列		
        Longest decreasing subsequence
		
    params:
        * X ->:
            meaning: 序列
            type: null
            optional: [null]

    return:
         (子序列开始位置, 子序列结束位置)


    demonstrate:
        Not described

    output:
        Not described
    """
    N = len(X)
    P = [0] * N
    M = [0] * (N + 1)
    L = 0
    for i in range(N):
        lo = 1
        hi = L
        while lo <= hi:
            mid = (lo + hi) // 2
            if (X[M[mid]] > X[i]):
                lo = mid + 1
            else:
                hi = mid - 1

        newL = lo
        P[i] = M[newL - 1]
        M[newL] = i

        if (newL > L):
            L = newL

    S = []
    pos = []
    k = M[L]
    for i in range(L - 1, -1, -1):
        S.append(X[k])
        pos.append(k)
        k = P[k]
    return S[::-1], pos[::-1]


def indices_density_func(data, 
                         column_name, 
                         indices=None):
    """
    计算特定指标的“核”密度，用来反映走势趋势
    """
    if (ST.VERBOSE in data.columns):
        print('Phase indices_density_func:{}'.format(column_name),
              QA_util_timestamp_to_str())
    indices_column_cross_jx_before = '{}_JX_BF'.format(column_name)
    indices_column_cross_sx_before = '{}_SX_BF'.format(column_name)
    indices_column_median = '{}_MEDIAN'.format(column_name)
    indices_column_density = '{}_DENSITY'.format(column_name)
    indices_column_density_returns = '{}_DENSITY_RETURNS'.format(column_name)
    indices_column_tide_cross_jx = '{}_TIDE_JX'.format(column_name)
    indices_column_tide_cross_sx = '{}_TIDE_SX'.format(column_name)
    indices_column_median_cross_jx = '{}_MEDIAN_JX'.format(column_name)
    indices_column_median_cross_sx = '{}_MEDIAN_SX'.format(column_name)
    if ('PEAK_CROSS' not in indices.columns):
        # Todo: Cache indices in memory.
        highp = data.high.values
        lowp = data.high.values
        openp = data.open.values
        closep = data.close.values

        peak_lowV = lowp[signal.argrelextrema(lowp,np.less)]
        peak_lowP = signal.argrelextrema(lowp,np.less)[0]

        peak_highV = highp[signal.argrelextrema(highp,np.greater)]
        peak_highP = signal.argrelextrema(highp,np.greater)[0]

        #peak_cross = pd.DataFrame(columns=['PEAK_CROSS',
        #                                   'PEAK_CROSS_JX',
        #                                   'PEAK_CROSS_SX'],
        #                          index=data.index)
        #peak_cross.iloc[peak_lowP,
        #                [peak_cross.columns.get_loc('PEAK_CROSS'),
        #                 peak_cross.columns.get_loc('PEAK_CROSS_JX')]] = 1, 1
        #peak_cross.iloc[peak_highP,
        #                [peak_cross.columns.get_loc('PEAK_CROSS'),
        #                peak_cross.columns.get_loc('PEAK_CROSS_SX')]] = -1, 1
        peak_cross = np.zeros((len(lowp), 3),)
        np.put(peak_cross[:, [0,1]], peak_lowP, 1)
        np.put(peak_cross[:, 0], peak_highP, -1)
        np.put(peak_cross[:, 2], peak_highP, 1)
        indices['PEAK_CROSS'] = peak_cross[:, 0]
        indices['PEAK_CROSS_JX'] = peak_cross[:, 1]
        indices['PEAK_CROSS_SX'] = peak_cross[:, 2]
        indices['BASELINE_MEDIAN'] = 0.512

    if (indices_column_cross_jx_before not in indices.columns):
        # 计算金叉死叉点
        indices[indices_column_cross_jx_before] = np.where(np.nan_to_num(indices[column_name].values, nan=0) > 0, 1, 0)
        indices[indices_column_cross_sx_before] = np.where(np.nan_to_num(indices[column_name].values, nan=0) < 0, 1, 0)
        indices[indices_column_cross_jx_before] = (indices[indices_column_cross_jx_before].apply(int).diff() > 0).apply(int)
        indices[indices_column_cross_sx_before] = (indices[indices_column_cross_sx_before].apply(int).diff() > 0).apply(int)

    # 计算金叉死叉累计时间积分
    indices[indices_column_cross_jx_before] = np.where((np.nan_to_num(indices[column_name].values, nan=0) > 0) & \
        ((indices['PEAK_CROSS_JX'] == 1) | \
        (indices[indices_column_cross_jx_before] == 1)), 1, 0)
    indices[indices_column_cross_sx_before] = np.where((np.nan_to_num(indices[column_name].values, nan=0) < 0) & \
        ((indices['PEAK_CROSS_SX'] == 1) | \
        (indices[indices_column_cross_sx_before] == 1)), 1, 0)
    indices[indices_column_cross_jx_before] = Timeline_duration(indices[indices_column_cross_jx_before].values)
    indices[indices_column_cross_sx_before] = Timeline_duration(indices[indices_column_cross_sx_before].values)

    if (indices_column_median not in indices.columns):
        # 计算金叉死叉持续时间中位数
        indices[indices_column_median] = int(min(indices[indices_column_cross_jx_before].median(), 
                                                 indices[indices_column_cross_sx_before].median()))

    # 计算“核”密度
    sumJX = indices[indices_column_cross_jx_before].rolling(int(indices[indices_column_median].max() * 3.09)).sum()
    sumSX = indices[indices_column_cross_sx_before].rolling(int(indices[indices_column_median].max() * 3.09)).sum()
    indices[indices_column_density] = sumSX / (sumJX + sumSX)
    indices[indices_column_density_returns] = indices[indices_column_density].pct_change()

    indices_TIDE_DENSITY_Uptrend = np.where(indices[indices_column_density_returns] > 0, 1, 0)
    indices[indices_column_tide_cross_jx] = Timeline_Integral(indices_TIDE_DENSITY_Uptrend)
    indices_TIDE_DENSITY_Downtrend = np.where(indices[indices_column_density_returns] < 0, 1, 0)
    indices[indices_column_tide_cross_sx] = Timeline_Integral(indices_TIDE_DENSITY_Downtrend)

    indices[indices_column_median_cross_jx] = Timeline_Integral(np.where(indices[indices_column_density] > 0.512, 1, 0))
    indices[indices_column_median_cross_sx] = Timeline_Integral(np.where(indices[indices_column_density] < 0.512, 1, 0))

    return indices


def line_intersect(slope1, y1, slope2, y2):
    """
    计算两个指标趋向相交还是背离
    """
    if slope1 == slope2:
        print("These lines are parallel!!!")
        return None

    # Set both lines equal to find the intersection point in the x
    # direction
    x = (y2 - y1) / (slope1 - slope2)

    # Now solve for y -- use either line, because they are equal here
    y = slope1 * x + y1
    return x,y


def lineareg_intercept(slope1, y1, 
                       slope2, y2):
    """
    计算两个指标的相交点截距
    """
    # Set both lines equal to find the intersection point in the x
    # direction
    x = (y2 - y1) / (slope1 - slope2)

    # Now solve for y -- use either line, because they are equal here
    y = slope1 * x + y1
    return x


def timeline_corr_from_last_phase_func(data, 
                                       indices, 
                                       column_name, 
                                       bar_ind):
    """
    分析与上一个特定指标的持续相关性，用来反映持续走势趋势波段之间的相关性，
    将相关性连续的波段合为一个大上升浪作为时间积分统计
    """
    indices_state = column_name
    indices_cross_jx = '{}_JX'.format(column_name)
    indices_cross_sx = '{}_SX'.format(column_name)
    indices_cross_jx_before = '{}_JX_BF'.format(column_name)
    indices_cross_sx_before = '{}_SX_BF'.format(column_name)

    tick_index = indices.index.get_level_values(level=0).get_loc(bar_ind.name[0])
    indices_rs = bar_ind[indices_cross_jx]
    if (bar_ind[indices_state] > 0):
        last_indices_sx_length = indices.iat[tick_index - bar_ind[indices_cross_jx], 
                                             indices.columns.get_loc(indices_cross_sx)]
        last_indices_jx_start = indices.iat[tick_index - bar_ind[indices_cross_jx], 
                                            indices.columns.get_loc(indices_cross_jx_before)]
        last_indices_timeline_corr = (last_indices_jx_start - last_indices_sx_length) / last_indices_jx_start
        if ((last_indices_timeline_corr > 0.809) and \
            (last_indices_jx_start > 36)) or \
            ((last_indices_timeline_corr > 0.618) and \
            (bar_ind[FLD.MA90] > bar_ind[FLD.MA120]) and \
            (bar_ind[indices_cross_jx_before] < bar_ind[FLD.DEA_CROSS_JX_BEFORE])) or \
            ((bar_ind[FLD.MA120_SLOPE] > 0) and \
            (bar_ind[FLD.MA90_SLOPE] > bar_ind[FLD.MA120_SLOPE]) and \
            (bar_ind[FLD.MA90] > bar_ind[FLD.MA120]) and \
            (bar_ind[indices_cross_jx_before] < bar_ind[FLD.DEA_CROSS_JX_BEFORE])):
            #if (bar_ind.name[0] in pd.date_range('2020-04-22 12:00:00',
            #                            periods=96, freq='1H')):
            #    print(bar_ind.name[0], 'fooo1', 'last_tri_ma_jx_start',
            #    last_tri_ma_jx_start)
            indices_rs = last_indices_jx_start + bar_ind[indices_cross_jx]

    return indices_rs


def calc_event_timing_lag(vhma_directions):
    """
    计算事件的时间间隔
    """
    with np.errstate(invalid='ignore', divide='ignore'):
        vhma_trend_jx = Timeline_Integral(np.where(vhma_directions > 0, 1, 0))
        vhma_trend_sx = np.sign(vhma_directions) * Timeline_Integral(np.where(vhma_directions < 0, 1, 0))
    return np.array(vhma_trend_jx + vhma_trend_sx).astype(np.int32)


def calc_one_punch_impulse(features, 
                           ref_features=None):
    """
    “心电图”检测， n-1, n, -1, 1, 2, 3, 类似波形
    """
    features_shift_1 = np.r_[0, features[:-1]]
    features_shift_2 = np.r_[[0, 0], features[:-2]]
    features_shift_3 = np.r_[[0, 0, 0], features[:-3]]
    with np.errstate(invalid='ignore', divide='ignore'):
        if (ref_features is None):
            ret_one_impulse = np.where((features_shift_1 < 0) & \
                                   (features_shift_1 < 0.0002) & \
                                   (features_shift_1 > -1.68) & \
                                   ((features_shift_2 > 18) | \
                                   ((features_shift_2 >= -0.0002) & \
                                   (features_shift_2 <= 6))) & \
                                   (features >= -0.0002), 1, 
                                       np.where((features_shift_2 < features_shift_3) & \
                                           (features_shift_1 < features_shift_2) & \
                                           (features_shift_2 > features_shift_3) & \
                                           (features > features_shift_1) & \
                                           (features >= -0.0002), 666, 0))
        else:
            ret_one_impulse = np.where((ref_features > -0.0002) & \
                                   (features_shift_1 < 0.0002) & \
                                   (features_shift_1 > -1.68) & \
                                   ((features_shift_2 > 18) | \
                                   ((features_shift_2 >= -0.0002) & \
                                   (features_shift_2 <= 6))) & \
                                   (features >= -0.0002), 666, 0)

    return ret_one_impulse


#@nb.jit(nopython=True) # 经测试这样无 JIT 最快
def strided_app(a, L, S):  
    '''
    Pandas rolling for numpy
    # Window len = L, Stride len/stepsize = S
    '''
    nrows = ((a.size - L) // S) + 1
    n = a.strides[0]
    return np.lib.stride_tricks.as_strided(a, shape=(nrows,L), strides=(S * n,n))


#@nb.jit(nopython=True) # 经测试这样无 JIT 最快
def pctrank(x:np.ndarray,) -> np.ndarray:
    '''
    排序
    Panda rolling window percentile rank
    '''
    n = len(x)
    temp = x.argsort()
    ranks = np.empty(n)
    ranks[temp] = (np.arange(n) + 1) / n
    return ranks[-1]


#@nb.jit(nopython=True) # 经测试这样无 JIT 最快
def rank_order(x:np.ndarray,) -> np.ndarray:
    '''
    排序
    Panda rolling window percentile rank
    '''
    n = len(x)
    temp = x.argsort()
    ranks = np.empty(n)
    ranks[temp] = (np.arange(n) + 1)
    return ranks[-1]


#def rolling_pctrank(s:np.ndarray, w:int,) -> np.ndarray:
#    '''
#    排序
#    Panda rolling window percentile rank
#    '''
#    def pctrank(x:np.ndarray,) -> np.ndarray:
#        '''
#        排序
#        Panda rolling window percentile rank
#        '''
#        n = len(x)
#        temp = x.argsort()
#        ranks = np.empty(n)
#        ranks[temp] = (np.arange(n) + 1) / n
#        return ranks[-1]

#    x = strided_app(s, w, 1)
#    return np.r_[np.full(w - 1, np.nan),
#                 np.array(list(map(pctrank, x)))]
#@nb.jit(nopython=True) # 经测试这样无 JIT 最快
def rolling_pctrank(s:np.ndarray, w:int,) -> np.ndarray:
    '''
    排序
    Panda rolling window percentile rank
    '''
    x = strided_app(s, w, 1)
    ret_rolling_pctrank = np.empty((len(s)))
    ret_rolling_pctrank[w - 1:] = np.array(list(map(pctrank, x)))
    return ret_rolling_pctrank

#@nb.jit(nopython=True) # 经测试这样无 JIT 最快
def rolling_rank(s:np.ndarray, w:int,) -> np.ndarray:
    '''
    排序
    Panda rolling window rank order
    '''
    x = strided_app(s, w, 1)
    ret_rolling_rank = np.empty((len(s)))
    #return np.r_[np.full(w - 1, np.nan),
    #             np.array(list(map(rank_order, x)))]
    ret_rolling_rank[w - 1:] = np.array(list(map(rank_order, x)))
    return ret_rolling_rank


def euclidean_distance(vec1, vec2):
    """
    欧氏距离
    :param vec1:
    :param vec2:
    :return:
    """
    # return np.sqrt(np.sum(np.square(vec1 - vec2)))
    # return sum([(x - y) ** 2 for (x, y) in zip(vec1, vec2)]) ** 0.5
    return np.linalg.norm(vec1 - vec2, ord=2)


@nb.jit('f8[:](f8[:, :], i2)', nopython=True)
def rolling_euclidean_distance(s:np.ndarray, w:int=5,) -> np.ndarray:
    '''
    计算周线的欧式距离
    Panda rolling window euclidean distance
    '''
    rolling_euclidean_distance = np.zeros(len(s), dtype=np.float64)
    for i in range(w, len(s)):
        vec1 = s[(i - w):i, 0]
        vec2 = s[(i - w):i, 1]
        rolling_euclidean_distance[i] = np.linalg.norm(vec1 - vec2, ord=2)
    return rolling_euclidean_distance


def rolling_zscore(s:np.ndarray, w:int,) -> np.ndarray:
    '''
    标准分
    '''
    def last_zscore(sx):
        return scs.zscore(sx)[-1]

    if (len(s) > w):
        x = strided_app(s, w, 1)
        return np.r_[np.full(w - 1, np.nan), 
                     np.array(list(map(last_zscore, x)))]
    else:
        return scs.zscore(s)
        #return scs.zscore(s[:min(w, len(s))])


#@nb.jit(nopython=True)
def rolling_mean(s:np.ndarray, w:int,) -> np.ndarray:
    '''
    排序
    Pandas rolling window percentile rank
    '''
    if (len(s) > n):
        x = strided_app(s, w, 1)
        ret_rolling_mean = np.empty((len(s)))
        #return np.r_[np.full(w - 1, np.nan),
        #             np.array(list(map(np.mean, x)))]
        ret_rolling_mean[w - 1:] = np.array(list(map(np.mean, x)))
        return ret_rolling_mean
    else:
        return np.full(len(s), np.nan)


#@nb.jit(nopython=True)
def rolling_sum(a:np.ndarray, n:int=4) -> np.ndarray:
    '''
    pandas.DataFrame.rolling(4).sum()
    '''
    if (len(a) > n):
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        #rolling_sum = np.empty((len(a)))
        return np.r_[np.full(n - 1, np.nan), ret[n - 1:]]
        #rolling_sum[n-1:] = ret[n - 1:]
        #return rolling_sum
    else:
        return np.full(len(a), np.nan)


#@nb.jit(nopython=True)
def rolling_poly9(s:np.ndarray, w:int=252) -> np.ndarray:
    '''
    一次九项式滚动分解拟合
    '''
    x_index = range(252)
    def last_poly9(sx):
        p = np.polynomial.Chebyshev.fit(x_index, sx, 9)
        return p(x_index)[-1]

    if (len(s) > w):
        x = strided_app(s, w, 1)
        return np.r_[np.full(w - 1, np.nan), 
                     np.array(list(map(last_poly9, x)))]
    else:
        x_index = range(len(s))
        p = np.polynomial.Chebyshev.fit(x_index, s, 9)
        y_fit_n = p(x_index)
        return y_fit_n


#@nb.jit('f8[:](f8[:], i2)', nopython=True)
#def rolling_poly9_jit(s:np.ndarray, w:int=252) -> np.ndarray:
#    '''
#    标准分
#    '''
#    def last_poly9(sx, w):
#        x_index = range(w)
#        p = np.polynomial.Chebyshev.fit(x_index, sx, 9)
#        return p(x_index)[-1]

#    ret_rolling_poly9 = np.zeros((len(s),))
#    if (len(s) > w):
#        for i in range(w, len(s)):
#            sx = s[(i-w):i]
#            ret_rolling_poly9[i] = last_poly9(sx, w)
#    else:
#        x_index = range(len(s))
#        p = np.polynomial.Chebyshev.fit(x_index, s, 9)
#        y_fit_n = p(x_index)
#        return y_fit_n
