# coding:utf-8
#
# The MIT License (MIT)
#
# Copyright (c) 2016-2018 azai/GolemQ
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
基于 QUANTAXIS 的 DataStruct.add_func 使用，也可以单独使用处理 Kline，
使用机器学习算法统计分析走势
"""

import numpy as np
import numba as nb

try:
    import QUANTAXIS as QA
    from QUANTAXIS.QAIndicator.base import *
    from QUANTAXIS.QAUtil.QADate_Adv import (
        QA_util_timestamp_to_str,
        QA_util_datetime_to_Unix_timestamp,
        QA_util_print_timestamp
    )
except:
    print('PLEASE run "pip install QUANTAXIS" before call GolemQ.analysis.machinelearning modules')
    pass
from sklearn import mixture
from sklearn.cluster import AgglomerativeClustering
from sklearn.linear_model import LinearRegression

from GolemQ.analysis.timeseries import *
from GolemQ.indices.indices import *
from GolemQ.utils.parameter import (
    AKA,
    INDICATOR_FIELD as FLD,
    TREND_STATUS as ST
)

#import traceback
@nb.jit(nopython=True)
def features_formatter_k_means(closep:np.ndarray,) -> np.ndarray:
    """
    机器学习K-means聚类，数据预处理，jit优化能快几倍啊
    """
    X = []
    idx = []
    lag = 30
    for i in range(len(closep)):
        left = max(0,i - lag)
        right = min(len(closep) - 1, i + lag)
        l = max(0,i - 1)
        r = min(len(closep) - 1,i + 1)
        for j in range(left, right):
            minP = min(closep[left:right])
            maxP = max(closep[left:right])
            low = 1 if closep[i] <= closep[l] and closep[i] < closep[r] else 0
            high = 1 if closep[i] >= closep[l] and closep[i] > closep[r] else 0
        x = [i, closep[i], minP, maxP, low, high]
        X.append(x)
        idx.append(i)

    X = np.array(X)
    idx = np.array(idx)
    return X, idx.astype(np.int32)


@nb.jit(nopython=True)
def get_index_of_peaks(p:np.ndarray, 
                       yaxis_of_peaks:np.ndarray,) -> np.ndarray:
    """
    返回 极值 x轴 index索引
    """
    idx = np.empty(len(p))
    for i in range(len(p)):
        idx[i] = yaxis_of_peaks[p[i]]
    return idx.astype(np.int32)


def dpgmm_predict(data:pd.DataFrame, 
                  code=None,
                  indices:pd.DataFrame=None,) -> np.ndarray:
    """
     (假设)我们对这条行情的走势一无所知，使用机器学习可以快速的识别出走势，划分出波浪。
     DPGMM聚类 将整条行情大致分块，这个随着时间变化会有轻微抖动。
     所以不适合做精确买卖点控制。但是作为趋势判断已经足够了。
    """
    sub_offest = 0
    nextsub_first = sub_first = 0
    nextsub_last = sub_last = 1199
    
    ret_cluster_group = np.zeros(len(data.close.values))
    while (sub_first == 0) or (sub_first < len(data.close.values)):
        # 数量大于1200bar的话无监督聚类效果会变差，应该控制在1000~1500bar之间
        if (len(data.close.values) > 1200):
            highp = np.nan_to_num(data.high.values[sub_first:sub_last], nan=0)
            lowp = np.nan_to_num(data.low.values[sub_first:sub_last], nan=0)
            openp = np.nan_to_num(data.open.values[sub_first:sub_last], nan=0)
            closep = np.nan_to_num(data.close.values[sub_first:sub_last], nan=0)
            if (sub_last + 1200 < len(data.close.values)):
                nextsub_first = sub_first + 1199
                nextsub_last = sub_last + 1200
            else:
                nextsub_first = len(data.close.values) - 1200
                nextsub_first = 0 if (nextsub_first < 0) else nextsub_first
                nextsub_last = len(data.close.values) + 1
        else:
            highp = data.high.values
            lowp = data.low.values
            openp = data.open.values
            closep = data.close.values
            sub_last = nextsub_first = len(data.close.values)
            nextsub_last = len(data.close.values) + 1

        if (ST.VERBOSE in data.columns):
            print('slice:', {sub_first, sub_last}, 'total:{} netx_idx:{} unq:{}'.format(len(data.close.values), 
                                                                                        sub_offest, 
                                                                                        len(np.unique(ret_cluster_group))))

        # DPGMM 聚类
        X, idx = features_formatter_k_means(closep)
        dpgmm = mixture.BayesianGaussianMixture(n_components=min(len(closep) - 1, max(int(len(closep) / 10), 16)), 
                                                max_iter=1000, 
                                                covariance_type='spherical',
                                                weight_concentration_prior_type='dirichlet_process')

        # 训练模型不含最后一个点
        try:
            dpgmm.fit(X[:-1])
        except:
            print(u' {} 分析样本历史数据太少({})'.format(code, len(closep) - 1))

        try:
            y_t = dpgmm.predict(X)
        except:
            print(u'{} "{}" 前复权价格额能存在np.nan值，请等待收盘数据接收保存完毕。'.format(QA_util_timestamp_to_str(),
                                            data.index.get_level_values(level=1)[0]), 
                                            len(X), len(X[:-1]), X[-1:], data[-1:])
        
        ret_cluster_group[sub_first:sub_last] = y_t + int(sub_offest)
        sub_offest = int(np.max(ret_cluster_group) + 1)

        if (sub_last >= len(data.close.values)):
            break
        else:
            sub_first = min(nextsub_first, nextsub_last)
            sub_last = max(nextsub_first, nextsub_last)
        # DPGMM 聚类分析完毕

    return ret_cluster_group


def calc_slice_with_wave(ml_trend:pd.DataFrame,) -> pd.Series:
    """
    统计切片，计算长度
    """
    # 第一个bar自动成为首个趋势分段（可能DEA > 0）
    cluster_group_gap_idx = ml_trend.columns.get_loc(FLD.CLUSTER_GROUP_GAP)
    ml_trend.iat[0, cluster_group_gap_idx] = 1
    ml_trend.iat[-1, cluster_group_gap_idx] = 1

    ml_trend[FLD.CLUSTER_GROUP_TO] = np.where(ml_trend[FLD.CLUSTER_GROUP_GAP] == 1, 
                                              ml_trend[FLD.CLUSTER_GROUP_BEFORE].shift(1) + 1, 
                                              np.nan)

    cluster_group_to_idx = ml_trend.columns.get_loc(FLD.CLUSTER_GROUP_TO)
    ml_trend[FLD.CLUSTER_GROUP_TO] = ml_trend[FLD.CLUSTER_GROUP_TO].shift(-1)
    ml_trend.iat[-2, cluster_group_to_idx] = ml_trend.iat[-2, cluster_group_to_idx] + 1
    ml_trend[FLD.CLUSTER_GROUP_TO] = np.where(ml_trend[FLD.CLUSTER_GROUP_GAP] == 1, 
                                              np.nan, 
                                              ml_trend[FLD.CLUSTER_GROUP_TO])
    ml_trend[FLD.CLUSTER_GROUP_TO] = ml_trend[FLD.CLUSTER_GROUP_TO].bfill()
    ml_trend.iat[-1, cluster_group_gap_idx] = 0
    return ml_trend[FLD.CLUSTER_GROUP_TO]


def zen_in_wavelet_func(data:np.ndarray,) -> np.ndarray:
    """
    Find zen trend in only one wavelet.
    缠论 ——> 在一个波浪中。
    这是很有效的算法，能用缠论(或者随便什么你们喜欢称呼的名字)在波浪中找到趋势，
    问题是缠论有个毛病是波浪套波浪，波浪套波浪，而这个算法只会找出给定象限内的
    最大趋势（划重点），所以需要提前把一条K线“切”成（可以盈利或者我们自己交易
    系统所选择的波段区间内）最小波浪，然后函数送进来，Duang！趋势就判断出来了。
    """
    highp = data[:, 0]
    lowp = data[:, 1]
    openp = data[:, 2]
    closep = data[:, 3]
        
    ret_data_len = len(closep)
    ret_zen_wavelet_cross = np.zeros(ret_data_len)
    ret_zen_wavelet_cross_jx = np.zeros(ret_data_len)
    ret_zen_wavelet_cross_sx = np.zeros(ret_data_len)

    bV = lowp[signal.argrelextrema(lowp,np.less)]
    bP = signal.argrelextrema(lowp,np.less)[0]
    d,p = LIS(bV)
    if (len(bP) > 0):
        idx = get_index_of_peaks(np.array(p), bP)
        np.put(ret_zen_wavelet_cross, idx, 1)
        np.put(ret_zen_wavelet_cross_jx, idx, 1)

    qV = highp[signal.argrelextrema(highp,np.greater)]
    qP = signal.argrelextrema(highp,np.greater)[0]
    qd,qp = LDS(qV)
    if (len(qP) > 0):
        qidx = get_index_of_peaks(np.array(qp), qP)
        np.put(ret_zen_wavelet_cross, qidx, -1)
        np.put(ret_zen_wavelet_cross_sx, qidx, 1)

    zen_cross = np.c_[ret_zen_wavelet_cross,
                      ret_zen_wavelet_cross_jx,
                      ret_zen_wavelet_cross_sx,]

    return zen_cross


def calc_slice_with_cluster_grouped(ml_trend:pd.DataFrame, 
                                    margin:int=8) -> pd.DataFrame:
    """
    按处理过(少量人工监督) 无监督机器学习模型信号划分出了可能的存在
    所有完整（聚类或者波浪）趋势的区间，检索切片信息，按设置长度合并长度太小的分段。
    """
    # 长度小于36的切片合并
    ml_trend[FLD.CLUSTER_GROUP_GAP] = np.where((ml_trend[FLD.CLUSTER_GROUP_GAP] == 1) & \
                                             (ml_trend[FLD.CLUSTER_GROUP_TO] < margin), 
                                             0, ml_trend[FLD.CLUSTER_GROUP_GAP])
    ml_trend[FLD.CLUSTER_GROUP_BEFORE] = Timeline_duration(ml_trend[FLD.CLUSTER_GROUP_GAP].values)
    ml_trend[FLD.CLUSTER_GROUP_TO] = calc_slice_with_wave(ml_trend)

    # 现在机器学习程序自己划分出了可能的存在一个完整波浪趋势的区间
    # （不用人干预，太特么感动了！）
    trend_cluster_groups = ml_trend.loc[ml_trend[FLD.CLUSTER_GROUP_GAP].gt(0), 
                                        [ST.CLUSTER_GROUP,
                                         FLD.CLUSTER_GROUP_GAP,
                                         FLD.CLUSTER_GROUP_TO,
                                         FLD.CLUSTER_GROUP_BEFORE]].copy()

    # 接下来逐个进行分析，再把结果装配起来。对于 MultiIndex 获取日期 Indexer ID
    trend_cluster_groups[FLD.CLUSTER_GROUP_FROM] = trend_cluster_groups.apply(lambda x: 
                                                                            ml_trend.index.get_level_values(level=0).get_loc(x.name[0]), 
                                                                            axis=1).apply(int)

    trend_cluster_groups[FLD.CLUSTER_GROUP_TO] = (trend_cluster_groups[FLD.CLUSTER_GROUP_FROM] + trend_cluster_groups[FLD.CLUSTER_GROUP_TO])
    if (len(trend_cluster_groups[FLD.CLUSTER_GROUP_FROM]) > 1):
        try:
            trend_cluster_groups.iat[0, trend_cluster_groups.columns.get_loc(FLD.CLUSTER_GROUP_TO)] = trend_cluster_groups[FLD.CLUSTER_GROUP_FROM][1]
        except:
            code = ml_trend.index.get_level_values(level=1)[0]
            print('trend_cluster_groups len mismatch error!!!!')
            print(code, len(trend_cluster_groups), len(trend_cluster_groups[FLD.CLUSTER_GROUP_FROM]), 
              'columns:{},{}'.format(len(trend_cluster_groups), trend_cluster_groups.columns.get_loc(FLD.CLUSTER_GROUP_TO)))
            raise('trend_cluster_groups len mismatch error!!!!')
    trend_cluster_groups.iat[-1, trend_cluster_groups.columns.get_loc(FLD.CLUSTER_GROUP_TO)] = len(ml_trend)

    trend_cluster_groups[FLD.CLUSTER_GROUP_TO] = trend_cluster_groups[FLD.CLUSTER_GROUP_TO].apply(int)
    return trend_cluster_groups


def ml_trend_func(data:pd.DataFrame,) -> pd.DataFrame:
    """
    使用机器学习算法统计分析趋势，使用无监督学习的方法快速的识别出K线走势状态。
    简单(不需要太精确，精确的买卖点由个人的指标策略控制完成)划分出波浪区间。
    对不同的波浪就可以继续采用定制化的量化策略。
    """
    if (ST.VERBOSE in data.columns):
        print('Phase ml_trend_func', QA_util_timestamp_to_str())

    code = data.index.get_level_values(level=1)[0]
    #traceback.print_stack()

    # 统计学习方法分析大趋势：数据准备
    ml_trend = pd.DataFrame(columns=[ST.CLUSTER_GROUP, 
                                     FLD.ZEN_WAVELET_CROSS, 
                                     FLD.ZEN_WAVELET_CROSS_JX_BEFORE, 
                                     FLD.ZEN_WAVELET_CROSS_SX_BEFORE,], 
                            index=data.index)

    ml_trend[ST.CLUSTER_GROUP] = dpgmm_predict(data, code)
    if (ST.VERBOSE in data.columns):
        print(u'{} "{}" 自动划分为：{:d} 种形态走势'.format(QA_util_timestamp_to_str(),
                                                code, 
                                                len(ml_trend[ST.CLUSTER_GROUP].unique())))

    macd_cross = lineareg_band_cross_func(data)
    
    # 下降趋势检测，再次价格回归。波浪下降趋势一定DEA下沉到零轴下方，
    # 所以依次扫描所有聚类，在时间轴的近端发现有DEA下沉到零轴下方，
    # 作为一个分组进行统计学习判断聚类组合的趋势（目标区域的最小切片为一个波峰和一个波谷）。
    ml_trend[FLD.CLUSTER_GROUP_GAP] = (ml_trend[ST.CLUSTER_GROUP].diff() != 0).apply(int)
    ml_trend[FLD.CLUSTER_GROUP_BEFORE] = Timeline_duration(ml_trend[FLD.CLUSTER_GROUP_GAP].values)
    ml_trend[FLD.CLUSTER_GROUP_GAP] = np.where((ml_trend[FLD.CLUSTER_GROUP_GAP] == 1) & \
                                             (ml_trend[FLD.CLUSTER_GROUP_BEFORE].shift(1) < 3), 
                                             0, ml_trend[FLD.CLUSTER_GROUP_GAP])
    ml_trend[FLD.CLUSTER_GROUP_GAP] = np.where((ml_trend[FLD.CLUSTER_GROUP_GAP] == 1) & \
                                             (macd_cross[FLD.DEA] > 0), 
                                             0, ml_trend[FLD.CLUSTER_GROUP_GAP])
    ml_trend[FLD.CLUSTER_GROUP_BEFORE] = Timeline_duration(ml_trend[FLD.CLUSTER_GROUP_GAP].values)
    ml_trend[FLD.CLUSTER_GROUP_TO] = calc_slice_with_wave(ml_trend)

    trend_cluster_groups = calc_slice_with_cluster_grouped(ml_trend, margin=36)

    zen_cross_columns_idx = [ml_trend.columns.get_loc(FLD.ZEN_WAVELET_CROSS), 
                             ml_trend.columns.get_loc(FLD.ZEN_WAVELET_CROSS_JX_BEFORE),
                             ml_trend.columns.get_loc(FLD.ZEN_WAVELET_CROSS_SX_BEFORE),]

    if (ST.VERBOSE in data.columns):
        print("Phase for trend_cluster_groups ", QA_util_timestamp_to_str())

    zen_cross = None
    stack_debug = []
    for index, trend_cluster_group in trend_cluster_groups.iterrows():
        # 极少数情况可能会有2个相邻交易日的日期出现，返回是一个slice对象，
        # 这个问题是对MultiIndex.get_loc()引起的，获取第一个时间索引的方法是start，
        # 但是会导致不准确的索引值(同一个时间出现索引不唯一)，正确获取唯一值时间索引的办法是
        # ml_trend.index.get_level_values(level=0).get_loc(x.name[0])
        trend_cluster_group_range = range(int(trend_cluster_group[FLD.CLUSTER_GROUP_FROM]),
                                          int(trend_cluster_group[FLD.CLUSTER_GROUP_TO]))
        trend_cluster_data = data.iloc[trend_cluster_group_range, 
                                       [data.columns.get_loc(AKA.HIGH),
                                        data.columns.get_loc(AKA.LOW),
                                        data.columns.get_loc(AKA.OPEN),
                                        data.columns.get_loc(AKA.CLOSE)]]
        stack_debug.append(trend_cluster_group_range)

        if (zen_cross is None):
            zen_cross = zen_in_wavelet_func(trend_cluster_data.values)
        else:
            zen_cross = np.r_[zen_cross, zen_in_wavelet_func(trend_cluster_data.values)]
    try:
        ml_trend.iloc[:, zen_cross_columns_idx] = zen_cross
        ml_trend.iloc[:, zen_cross_columns_idx]
    except:
        print(code, len(ml_trend), len(zen_cross[:, 0]), 
              'columns:{},{}'.format(len(zen_cross_columns_idx), len(zen_cross[0, :])))
        print(trend_cluster_groups[[FLD.CLUSTER_GROUP_FROM,FLD.CLUSTER_GROUP_TO]],
              trend_cluster_groups[FLD.CLUSTER_GROUP_TO].shift(1),
              (trend_cluster_groups[FLD.CLUSTER_GROUP_FROM] - trend_cluster_groups[FLD.CLUSTER_GROUP_TO].shift(1)) < -0.001)
        print((trend_cluster_groups[FLD.CLUSTER_GROUP_FROM] > trend_cluster_groups[FLD.CLUSTER_GROUP_TO].shift(1)).sum())
        print(stack_debug)
        raise('Data len mismatch error!!!!')

    ml_trend[FLD.ZEN_WAVELET_CROSS_JX_BEFORE] = np.where(ml_trend[FLD.ZEN_WAVELET_CROSS] == 1, 1, 0)
    ml_trend[FLD.ZEN_WAVELET_CROSS_SX_BEFORE] = np.where(ml_trend[FLD.ZEN_WAVELET_CROSS] == -1, 1, 0)
    ml_trend.iat[0, ml_trend.columns.get_loc(FLD.ZEN_WAVELET_CROSS_JX_BEFORE)] = 1
    ml_trend.iat[0, ml_trend.columns.get_loc(FLD.ZEN_WAVELET_CROSS_SX_BEFORE)] = 1
    ml_trend[FLD.ZEN_WAVELET_CROSS_JX_BEFORE] = Timeline_duration(ml_trend[FLD.ZEN_WAVELET_CROSS_JX_BEFORE].values)
    ml_trend[FLD.ZEN_WAVELET_CROSS_SX_BEFORE] = Timeline_duration(ml_trend[FLD.ZEN_WAVELET_CROSS_SX_BEFORE].values)

    zen_wavelet_cross = np.where((ml_trend[FLD.ZEN_WAVELET_CROSS_SX_BEFORE] - ml_trend[FLD.ZEN_WAVELET_CROSS_JX_BEFORE]) > 0, 1, -1)
    zen_wavelet_jx = Timeline_Integral(np.where(zen_wavelet_cross > 0, 1, 0))
    zen_wavelet_sx = np.sign(zen_wavelet_cross) * Timeline_Integral(np.where(zen_wavelet_cross < 0, 1, 0))
    ml_trend[FLD.ZEN_WAVELET_TIMING_LAG] = zen_wavelet_jx + zen_wavelet_sx

    # 清理临时数据
    ml_trend = ml_trend.drop([FLD.CLUSTER_GROUP_GAP,
                              FLD.CLUSTER_GROUP_TO,
                              FLD.CLUSTER_GROUP_BEFORE, 
                              FLD.ZEN_TIDE_DENSITY_RETURNS], axis=1)

    ret_ml_trend = pd.concat([macd_cross, 
                              ml_trend], axis=1)

    if (ST.VERBOSE in data.columns):
        print("Phase ml_trend_func done", QA_util_timestamp_to_str())

    return ret_ml_trend
