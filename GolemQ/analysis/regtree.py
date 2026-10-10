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
#
"""回归树（CART）与 regtree 拟合线 —— 从旧树 `analysis/regtree.py` 搬运。

搬运时只动 import（逐条对照见下），算法与结构**逐字保留** ——
因为 `regtree_jit.py` 要拿它做**对拍**，改名/重构会让两边无法比对。

⚠️ **慢点是公认的**（旧树注释）：`calc_regtree_fractal_func` 的 docstring 写着
「超过 500 bar 计算速度会变得很慢（超过 5 秒），所以 `bar_limit` 默认限制为 300」。
热在 `choose_best_split_branch`：它对「每个特征 × 每个取值」都做一次
`calc_err_of_the_model`，而后者要建设计矩阵、`np.linalg.det` **再求逆**。
`regtree_jit.py` 就是冲这一点去的。

import 改接（老树 → 新树）
=========================
* ``GolemQ.utils.parameter`` → ``GolemQ.core.constants``（类名 ``INDICATOR_FIELD``→``FIELD``）
* ``GolemQ.analysis.timeseries import *`` → ``GolemQ.analysis.timing`` 的**三个**具名函数
  （`calc_event_timing_lag` ×11 / `rolling_sum` ×2 / `lineareg_intercept` ×1）——
  ⚠️ 这个清单是**扫出来的**不是猜的（比对面模块 43 个函数名 × 本文件的调用点），
  我第一版只导了 `rolling_sum`，漏掉的那个只在某条分支上才会 `NameError`）
* ``GolemQ.portfolio.utils`` → ``GolemQ.portfolio.returns``（新树没有 utils 杂物间）
* **删掉**：被注释掉的 ``regtree_cython`` 那一整段（Cython 方案**作废**，用户 2026-10-10 明确）、
  以及 ``import numba`` + ``@nb.jit sum1d``（**`sum1d` 只有定义、从无调用** —— 死代码；
  且新树 `analysis/` 不用 numba。**jit 版本在 `regtree_jit.py`**，不是在这里）
"""
import datetime
import pandas as pd
import numpy as np

try:
    import talib
except Exception:                      # noqa: BLE001
    print('PLEASE run "pip install TALIB" to call these modules')
from GolemQ.core.constants import (
    AKA,
    FIELD as FLD,
    TREND_STATUS as ST,
    FEATURES as FTR,
)
# ⚠️ 这三个是 `from GolemQ.analysis.timeseries import *` **真正**用到的全部 ——
# 星号导入收窄成具名，靠的不是「猜哪几个」，而是逐个比对面函数名（43 个）
# 与本文件的调用点。**漏一个就是运行时 NameError**，而且只在某条分支上才炸。
from GolemQ.analysis.timing import (
    calc_event_timing_lag,
    lineareg_intercept,
    rolling_sum,
)
from GolemQ.portfolio.returns import (
    calc_onhold_returns_np,
)

# ⚠️ **`np.mat` → `np.asmatrix`：这是 numpy 2.0 的官方迁移，不是重构。**
# numpy 2.0 **移除了 `np.mat`**（报错原文：`Use np.asmatrix instead`），而 `np.asmatrix`
# 产出**同一个 `np.matrix` 类型** ⇒ 下游的 `.I`（求逆）与 `*`（矩阵乘）语义**完全不变**。
# 实测本文件共 9 处（含 1 处在注释里）。**这不是可选项** —— 不换的话，
# `predict_regression_tree_branch` 一进来就抛 AttributeError，而它**被上层 `except` 吞掉**
# （且那句诊断打印还带 `len>460` 的门槛）⇒ **整段静默写出全 NaN**，跑起来 0.009s「成功」。

def choose_best_split_branch(seq_data:np.ndarray, rate, dur):
    """
    判断所有样本是否为同一分类
    """
    if len(set(seq_data[:,-1].T.tolist()[0])) == 1:
        return None, fit_lineareg_slope_of_the_model_leaf(seq_data)

    m,n = np.shape(seq_data)
    S = calc_err_of_the_model(seq_data) # 整体误差
    best_S = np.inf
    best_idx = 0
    best_val = 0
    for feat_idx in range(n - 1): # 遍历所有特征, 此处只有一个
        # 遍历特征中每种取值
        for split_val in set(seq_data[:,feat_idx].T.tolist()[0]):
            #print(split_val)
            mat0, mat1 = split_data_into_binary(seq_data, feat_idx, split_val)
            if (np.shape(mat0)[0] < dur) or (np.shape(mat1)[0] < dur):
                continue # 样本数太少, 前剪枝

             # 计算整体误差
            newS = calc_err_of_the_model(mat0) + calc_err_of_the_model(mat1)
            if newS < best_S:
                best_idx = feat_idx
                best_val = split_val
                best_S = newS

    if (S - best_S) < rate:
        # 如差误差下降得太少，则不切分
        return None, fit_lineareg_slope_of_the_model_leaf(seq_data)
    mat0, mat1 = split_data_into_binary(seq_data, best_idx, best_val)
    return best_idx, best_val


def split_data_into_binary(seq_data:np.ndarray, feature:int, value:float):
    """
    用feature把seq_data按value分成两个子集
    """
    mat0 = seq_data[np.nonzero(seq_data[:,feature] > value)[0],:]
    mat1 = seq_data[np.nonzero(seq_data[:,feature] <= value)[0],:]
    return mat0, mat1


def solve_lineareg_func(seq_data:np.ndarray):
    """
    求给定数据集的线性方程
    """
    m,n = np.shape(seq_data)
    X = np.asmatrix(np.ones((m,n))) # 第一行补1，线性拟合要求
    Y = np.asmatrix(np.ones((m,1)))
    X[:,1:n] = seq_data[:,0:n - 1]
    Y = seq_data[:,-1] # 这里约定 数据最后一列是 价格数据
    xTx = X.T * X
    if np.linalg.det(xTx) == 0.0:
        raise NameError('This matrix is singular, cannot do inverse,\ntry increasing dur')
    ws = xTx.I * (X.T * Y) # 公式推导较难理解
    return ws,X,Y


def fit_lineareg_slope_of_the_model_leaf(seq_data:np.ndarray):
    """
    求线性方程的参数
    """
    ws,X,Y = solve_lineareg_func(seq_data)
    return ws


def calc_err_of_the_model(seq_data:np.ndarray):
    """
    预测值和y的方差
    """
    ws, X, Y = solve_lineareg_func(seq_data)
    yaxis_height_at = X * ws
    return sum(np.power(Y - yaxis_height_at, 2))


# def choose_best_split_branch(seq_data:np.ndarray, rate, dur):
#     """
#     判断所有样本是否为同一分类
#     """
#     if len(set(seq_data[:,-1].T.tolist()[0])) == 1:
#         return None, fit_lineareg_slope_of_the_model_leaf(seq_data)

#     m,n = np.shape(seq_data)
#     S = calc_err_of_the_model(seq_data) # 整体误差
#     best_S = np.inf
#     best_idx = 0
#     best_val = 0
#     for feat_idx in range(n - 1): # 遍历所有特征, 此处只有一个
#         # 遍历特征中每种取值
#         for split_val in set(seq_data[:,feat_idx].T.tolist()[0]):
#             mat0, mat1 = split_data_into_binary(seq_data, feat_idx, split_val)
#             if (np.shape(mat0)[0] < dur) or (np.shape(mat1)[0] < dur):
#                 continue # 样本数太少, 前剪枝

#              # 计算整体误差
#             newS = calc_err_of_the_model(mat0) + calc_err_of_the_model(mat1)
#             if newS < best_S:
#                 best_idx = feat_idx
#                 best_val = split_val
#                 best_S = newS

#     if (S - best_S) < rate:
#         # 如差误差下降得太少，则不切分
#         return None, fit_lineareg_slope_of_the_model_leaf(seq_data)
#     mat0, mat1 = split_data_into_binary(seq_data, best_idx, best_val)
#     return best_idx, best_val


def tree_model_evaluation_func(model, branch_data):
    """
    预测评估函数,数据乘模型,模型是斜率和截距的矩阵
    """
    n = np.shape(branch_data)[1]
    X = np.asmatrix(np.ones((1,n + 1)))
    X[:,1:n + 1] = branch_data
    return float(X * model)


def is_tree_py(obj):
    """
    用字典保存的二叉树结构
    """
    return (type(obj).__name__ == 'dict')


def tree_model_forecast_decision_func_np(tree, branch_data, level=0):
    """
    预测/遍历整颗树的二元函数
    如果未预测，则预测树分支。
    如果已经预测，则遍历树分支 
    """
    if (level>99):
        print(u'tree_model_forecast_decision_func_np level gt {}.'.format(level))
    if not is_tree_py(tree):
        return tree_model_evaluation_func(tree, branch_data)
    if branch_data[tree['spInd']] > tree['spVal']:
        if is_tree_py(tree['left']):
            return tree_model_forecast_decision_func_np(tree['left'], branch_data, level=level+1)
        else:
            return tree_model_evaluation_func(tree['left'], branch_data)
    else:
        if is_tree_py(tree['right']):
            return tree_model_forecast_decision_func_np(tree['right'], branch_data, level=level+1)
        else:
            return tree_model_evaluation_func(tree['right'], branch_data)


def create_whole_forecast_tree(tree, seq_data, directions:bool=False):
    """
    对测试数据集预测一系列结果, 用于输出，
    Tree 结构中包含一些可变数据类型，所以不能完全扔进 regtree_cython.pyx 中
    """
    m = len(seq_data)
    if (directions == True):
        ret_tree_directions = np.zeros(m)
        ret_y = np.zeros(m)
    yaxis_height_at = np.asmatrix(np.zeros((m,1)))

    for i in range(m): # m是item个数
        yaxis_height_at[i,0] = tree_model_forecast_decision_func_np(tree, 
                                                                 np.asmatrix(seq_data[i]), level=0)
        if (directions == True):
            ret_y[i] = yaxis_height_at[i,0]
            if (i > 0):
                ret_tree_directions[i] = 1 if (ret_y[i] > ret_y[i - 1]) else -1

    if (directions == True):
        return ret_y, ret_tree_directions
    else:
        return yaxis_height_at


def predict_regression_tree_branch(seq_data, rate, dur, 
                                   level=0, code=''):
    """
    生成回归树, seq_data是数据, rate是误差下降, dur是叶节点的最小样本数
    寻找最佳划分点, feat为切分点, val为值
    """
    feat, val = choose_best_split_branch(seq_data, rate, dur)
    if feat == None:
        return val # 不再可分

    ret_tree = {}
    ret_tree['spInd'] = feat
    ret_tree['spVal'] = val

    # 把数据切给左右两树
    left_set, right_set = split_data_into_binary(seq_data, feat, val)
    if (level>99):
        print(u'\nCode:{}, predict_regression_tree_branch level gt {}. left_set,{} right_set:{}'.format(code,
                                                                            level,
                                                                            len(left_set), 
                                                                            len(right_set)))
    if ((len(left_set)==len(seq_data)) or (len(left_set)==len(seq_data)-1)) and (len(right_set)==0) and (level>9):
        # 不再可分
        ret_tree['left'] = left_set
        ret_tree['right'] = []
        return ret_tree

    ret_tree['left'] = predict_regression_tree_branch(left_set, rate, dur, 
                                                      level=level+1, code=code)
    ret_tree['right'] = predict_regression_tree_branch(right_set, rate, dur, 
                                                       level=level+1, code=code)

    return ret_tree


def fit_regtree_trend(tree, 
                      seq_data):
    """
    输出回归树预测
    """
    yaxis_height_at, tree_directions = create_whole_forecast_tree(tree, 
                                                                  seq_data.seq_index, 
                                                                  directions=True)

    ret_regtree_flu = np.c_[yaxis_height_at,
                            tree_directions,
                            tree_directions]

    with np.errstate(invalid='ignore', divide='ignore'):
        ret_regtree_flu[:, 2] = np.where((ret_regtree_flu[:, 1] > 0) | \
            ((rolling_sum(ret_regtree_flu[:, 1], 2) > 0) & \
            (rolling_sum(ret_regtree_flu[:, 1], 4) > 2)), 1, ret_regtree_flu[:, 2])

    return ret_regtree_flu


def calc_regtree_fractal_func(data, *args, **kwargs):
    '''
    快速计算regtree拟合线，因为超过500bar计算速度会变得很慢（超过5秒），所以 bar_limit 默认限制为 300
    '''
    # 针对多标的，拆分 indices 数据再自动合并
    code = data.index.get_level_values(level=1)[0]
    if ('features' in kwargs.keys()):
        features = kwargs['features'].loc[(slice(None), code), :]
    elif (len(args) > 0):
        features = args[0].loc[(slice(None), code), :]
    else:
        features = None
    bar_limit = kwargs['bar_limit'] if ('bar_limit' in kwargs.keys()) else 300
    rate = kwargs['rate'] if ('rate' in kwargs.keys()) else 25
    dur = kwargs['dur'] if ('dur' in kwargs.keys()) else 10

    #if (len(features) - bar_limit - 300) > 0:
    #    pre_ohlc_data = data[-bar_limit - 300:-bar_limit].copy()
    #    pre_price_dummy = features[FLD.HMA10][-bar_limit - 300:-bar_limit].copy()
    #    pre_price = np.c_[pre_ohlc_data.apply(lambda x:pre_ohlc_data.index.get_level_values(level=0).get_loc(x.name[0]), axis=1),
    #                      pre_price_dummy]
    #    pre_tree = predict_regression_tree_branch(np.asmatrix(pre_price), rate, dur, code=code)
    #    pre_ohlc_data['seq_index'] = pre_price[:, 0]
    #    pre_regtree_flu = fit_regtree_trend(pre_tree, pre_ohlc_data)

    ohlc_data = data[-bar_limit:].copy()
    price_dummy = features[FLD.HMA10][-bar_limit:].copy()
    price = np.c_[ohlc_data.apply(lambda x:ohlc_data.index.get_level_values(level=0).get_loc(x.name[0]), axis=1),
                  price_dummy]
    # ⚠️ **2026-10-10 唯一一处对旧实现的改动**：加一个 `builder=` 逃生口。
    # 默认仍是本模块的 `predict_regression_tree_branch`（**行为一字不变**），
    # 只为让 `regtree_jit.calc_regtree_fractal_jit` 能换成 jit 建树器 ——
    # 否则那份 60 行的编排就得**复制一遍**，正是本项目最忌的平行实现。
    builder = kwargs.get('builder') or predict_regression_tree_branch
    try:
        tree = builder(np.asmatrix(price), rate, dur,
                       level=0, code=code)
    except Exception as e:
        if (len(ohlc_data)>460):
            print(u'Code:{} length:{}'.format(code, len(ohlc_data)), e)
        tree = None
        features[FLD.REGTREE_PRICE] = np.nan
        features[FLD.REGTREE_TREND] = np.nan
        features[FLD.REGTREE_SLOPE] = np.nan
        features[FLD.REGTREE_TIMING_LAG] = np.nan
        features[FLD.REGTREE_TREND_RETURNS] = np.nan
        return features

    ohlc_data['seq_index'] = price[:, 0]
    regtree_flu = fit_regtree_trend(tree, ohlc_data)
    if (len(features) - bar_limit) > 0:
        #print(kwargs.keys(), len(features), bar_limit)
        #if (len(features) - bar_limit - 300) > 0:
        #    features[FLD.REGTREE_PRICE] = np.r_[np.empty(len(features) - bar_limit - 300), pre_regtree_flu[:, 0], regtree_flu[:, 0]]
        #else:
        #features[FLD.REGTREE_PRICE] = np.r_[np.empty(len(features) - bar_limit), regtree_flu[:, 0]]
        features[FLD.REGTREE_PRICE] = np.r_[features[FLD.MA30][: - bar_limit], regtree_flu[:, 0]]
        features[FLD.REGTREE_TREND] = np.r_[np.empty(len(features) - bar_limit), regtree_flu[:, 1]]
        features[FLD.REGTREE_TREND_RS] = np.r_[np.empty(len(features) - bar_limit), regtree_flu[:, 2]]
        features[FLD.REGTREE_TIMING_LAG] = np.r_[np.empty(len(features) - bar_limit), calc_event_timing_lag(features[FLD.REGTREE_TREND].tail(bar_limit))]
        features[FLD.REGTREE_TREND_RETURNS] = np.r_[np.empty(len(features) - bar_limit), calc_onhold_returns_np(ohlc_data.close.values,
                                                                                                                features[FLD.REGTREE_TREND].tail(bar_limit).values)]

    else:
        features[FLD.REGTREE_PRICE] = regtree_flu[:, 0]
        features[FLD.REGTREE_TREND] = regtree_flu[:, 1]
        features[FLD.REGTREE_TREND_RS] = regtree_flu[:, 2]
        features[FLD.REGTREE_TIMING_LAG] = calc_event_timing_lag(features[FLD.REGTREE_TREND])
        features[FLD.REGTREE_TREND_RETURNS] = calc_onhold_returns_np(ohlc_data.close.values,
                                                                     features[FLD.REGTREE_TREND].values)

    if (FLD.POLYNOMIAL9_TIMING_LAG in features.columns):
        features[FLD.POLY9_REGTREE_DIVERGENCE] = (features[FLD.POLYNOMIAL9] - features[FLD.REGTREE_PRICE]) / features[FLD.REGTREE_PRICE]
    if (FLD.ATR_LB in features.columns):
        features[FLD.ATR_LB_REGTREE_DIVERGENCE] = (features[FLD.ATR_LB] - features[FLD.REGTREE_PRICE]) / features[FLD.REGTREE_PRICE]
    #try:
    #    features[FLD.REGTREE_TREND_RETURNS] =
    #    calc_onhold_returns_np(ohlc_data.close.values,
    #                                                                 features[FLD.REGTREE_TREND].values)
    #except Exception as e:
    #    print(e)
    #    print(u'计算股票代码 {} 的利润数据出现意外 length:{}，忽略错误进行下一个分组...'.format(code,
    #    len(data)))

    features[FLD.REGTREE_SLOPE] = talib.LINEARREG_SLOPE(features[FLD.REGTREE_PRICE], timeperiod=14)
    features[FLD.REGTREE_MA90_INTERCEPT] = lineareg_intercept(features[FLD.REGTREE_SLOPE], 
                                                            features[FLD.REGTREE_PRICE], 
                                                            features[FLD.MA90_SLOPE], 
                                                            features[FLD.MA90])

    if (FLD.DEA_ZERO_TIMING_LAG in features.columns):
        deny_line = ((features[FLD.REGTREE_TIMING_LAG] > 0) & \
                    ((features[FLD.DEA_ZERO_TIMING_LAG] < 0) | \
                    (features[FLD.DEA_ZERO_TIMING_LAG] <= 6) & \
                    (features[FLD.MACD_DELTA] < 0)) & \
                    (features[FLD.REGTREE_MA90_INTERCEPT] > 9.27) & \
                    (features[FLD.REGTREE_PRICE] < features[FLD.MA90]) & \
                    (features[FLD.REGTREE_PRICE] < features[FLD.HMA5])) | \
                    ((features[FLD.REGTREE_TIMING_LAG] > 36) & \
                    (features[FLD.REGTREE_SLOPE] < 0.0008) & \
                    (features[FLD.REGTREE_PRICE] < features[FLD.HMA10]) & \
                    (features[FLD.REGTREE_PRICE] < features[FLD.HMA5])) | \
                    ((features[FLD.REGTREE_TIMING_LAG] > 18) & \
                    (features[FLD.REGTREE_SLOPE] < 0.002) & \
                    (features[FLD.REGTREE_PRICE] < features[FLD.HMA10]) & \
                    (features[FLD.REGTREE_PRICE] < features[FLD.HMA5])) | \
                    ((features[FLD.REGTREE_TIMING_LAG] > 18) & \
                    (features[FLD.REGTREE_TREND_RETURNS] < -0.0168) & \
                    (features[FLD.HMA10].diff(1) < 0))
        features[FLD.REGTREE_DENY_LINE] = deny_line

    #print(regtree_flu[-1, 1], regtree_flu[-1, 2])
    corrcoef = features[FLD.REGTREE_PRICE].tail(bar_limit).corr(features[FLD.HMA10].tail(bar_limit))
    features[FLD.REGTREE_CORRCOEF] = corrcoef

    #p = np.polynomial.Chebyshev.fit(ohlc_data['seq_index'], ohlc_data.close,
    #9)
    #y_fit_n = p(ohlc_data['seq_index'])
    #features[FLD.POLYNOMIAL9] = y_fit_n
    #features[FLD.POLYNOMIAL9_TIMING_LAG] =
    #calc_event_timing_lag(np.where(features[FLD.POLYNOMIAL9].diff(1) > 0, 1,
    #-1))

    if (FLD.COMBINE_DENSITY in features.columns):
        features[FLD.MAPOWER30_QUARTER] = np.where((features[FLD.MAPOWER30_TIMING_LAG] > 0) & \
            (features[FLD.MAPOWER30] < features[FLD.COMBINE_DENSITY]), 1,
                                                   np.where((features[FLD.MAPOWER30_TIMING_LAG] > 0) & \
            (features[FLD.MAPOWER30] > features[FLD.COMBINE_DENSITY]), 2,
                                                            np.where((features[FLD.MAPOWER30_TIMING_LAG] < 0) & \
            (features[FLD.MAPOWER30] > features[FLD.COMBINE_DENSITY]), -2, 
                                                                     np.where((features[FLD.MAPOWER30_TIMING_LAG] < 0) & \
            (features[FLD.MAPOWER30] < features[FLD.COMBINE_DENSITY]), -1, 0))))
        features[FLD.MAPOWER120_QUARTER] = np.where((features[FLD.MAPOWER120_TIMING_LAG] > 0) & \
            (features[FLD.MAPOWER120] < features[FLD.COMBINE_DENSITY]), 1,
                                                   np.where((features[FLD.MAPOWER120_TIMING_LAG] > 0) & \
            (features[FLD.MAPOWER120] > features[FLD.COMBINE_DENSITY]), 2,
                                                            np.where((features[FLD.MAPOWER120_TIMING_LAG] < 0) & \
            (features[FLD.MAPOWER120] > features[FLD.COMBINE_DENSITY]), -2, 
                                                                     np.where((features[FLD.MAPOWER120_TIMING_LAG] < 0) & \
            (features[FLD.MAPOWER120] < features[FLD.COMBINE_DENSITY]), -1, 0))))
        features[FLD.HMAPOWER120_QUARTER] = np.where((features[FLD.HMAPOWER120_TIMING_LAG] > 0) & \
            (features[FLD.HMAPOWER120] < features[FLD.COMBINE_DENSITY]), 1,
                                                   np.where((features[FLD.HMAPOWER120_TIMING_LAG] > 0) & \
            (features[FLD.HMAPOWER120] > features[FLD.COMBINE_DENSITY]), 2,
                                                            np.where((features[FLD.HMAPOWER120_TIMING_LAG] < 0) & \
            (features[FLD.HMAPOWER120] > features[FLD.COMBINE_DENSITY]), -2, 
                                                                     np.where((features[FLD.HMAPOWER120_TIMING_LAG] < 0) & \
            (features[FLD.HMAPOWER120] < features[FLD.COMBINE_DENSITY]), -1, 0))))

    return features


def calc_regtree_renko_fractal_func(data, *args, **kwargs):
    '''
    计算 regtree_renko 延长形态
    '''
    # 针对多标的，拆分 indices 数据再自动合并
    code = data.index.get_level_values(level=1)[0]
    if ('features' in kwargs.keys()):
        features = kwargs['features'].loc[(slice(None), code), :]
    elif (len(args) > 0):
        features = args[0].loc[(slice(None), code), :]
    else:
        features = None

    # 扩展一点点主升浪启动阶段
    #if (FLD.RENKO_BOOST_L_TIMING_LAG in features.columns):
    #    renko_bold = np.where((features[FLD.MAPOWER30_TIMING_LAG] < 0) & \
    #        (features[FLD.DEA_ZERO_TIMING_LAG] > 0) & \
    #        ((features[FLD.ZEN_WAVELET_TIMING_LAG] < 0) | \
    #        (((features[FLD.MAPOWER120] - features[FLD.MAPOWER30]) > 0.512) &
    #        \
    #        (features[FLD.DEA_SLOPE] < -0.008)) | \
    #        (features[FLD.LINEAREG_BAND_TIMING_LAG] < 0) | \
    #        (features[FLD.RENKO_TREND_L_TIMING_LAG] <
    #        features[FLD.RENKO_BOOST_L_TIMING_LAG])) & \
    #        (features[FLD.RENKO_TREND_L_TIMING_LAG] < 0) & \
    #        ~((features[FLD.MAPOWER30] < 0.0382) & \
    #        (features[FLD.RENKO_BOOST_L_TIMING_LAG] > 0)) & \
    #        ~((features[FLD.RENKO_BOOST_L_TIMING_LAG] > 0) & \
    #        (features[FLD.DIF_ZERO_TIMING_LAG] > 0) & \
    #        (features[FLD.PEAK_LOW_TIMING_LAG] <= 4) & \
    #        (features[FLD.MAPOWER120] > features[FLD.MAPOWER30]) & \
    #        (features[FLD.MAPOWER30] < features[FLD.CCI_NORM]) & \
    #        (features[FLD.ZEN_WAVELET_TIMING_LAG] >= -4)), -1,
    #                          np.where((features[FLD.MAPOWER30_TIMING_LAG] <
    #                          0) & \
    #                                   (features[FLD.DIF_ZERO_TIMING_LAG] < 0)
    #                                   & \
    #                                   (features[FLD.MACD_ZERO_TIMING_LAG] <
    #                                   features[FLD.DIF_ZERO_TIMING_LAG]) & \
    #                                   (features[FLD.MAPOWER30_TIMING_LAG] <
    #                                   features[FLD.DIF_ZERO_TIMING_LAG]) & \
    #                                   ((features[FLD.ZEN_WAVELET_TIMING_LAG]
    #                                   < 0) | \
    #                                   (features[FLD.LINEAREG_BAND_TIMING_LAG]
    #                                   < 0) | \
    #                                   (features[FLD.RENKO_TREND_L_TIMING_LAG]
    #                                   <
    #                                   features[FLD.RENKO_BOOST_L_TIMING_LAG]))
    #                                   & \
    #                                   (features[FLD.LINEAREG_BAND_TIMING_LAG]
    #                                   < 0) & \
    #                                   ~((features[FLD.MAPOWER30] < 0.0382) &
    #                                   \
    #                                   (features[FLD.RENKO_BOOST_L_TIMING_LAG]
    #                                   > 0)), -1,
    #                          np.where((features[FLD.MAPOWER30_TIMING_LAG] <
    #                          0) & \
    #                                   (features[FLD.DEA_ZERO_TIMING_LAG] > 0)
    #                                   & \
    #                                   ((features[FLD.ZEN_WAVELET_TIMING_LAG]
    #                                   < 0) | \
    #                                   (features[FLD.LINEAREG_BAND_TIMING_LAG]
    #                                   < 0) | \
    #                                   (features[FLD.RENKO_TREND_L_TIMING_LAG]
    #                                   <
    #                                   features[FLD.RENKO_BOOST_L_TIMING_LAG]))
    #                                   & \
    #                                   (features[FLD.LINEAREG_BAND_TIMING_LAG]
    #                                   < 0) & \
    #                                   ~((features[FLD.MAPOWER30] < 0.0382) &
    #                                   \
    #                                   (features[FLD.RENKO_BOOST_L_TIMING_LAG]
    #                                   > 0)) & \
    #                                   ~((features[FLD.RENKO_BOOST_L_TIMING_LAG]
    #                                   > 0) & \
    #                                   (features[FLD.DIF_ZERO_TIMING_LAG] > 0)
    #                                   & \
    #                                   (features[FLD.PEAK_LOW_TIMING_LAG] < 4)
    #                                   & \
    #                                   (features[FLD.MAPOWER120] >
    #                                   features[FLD.MAPOWER30]) & \
    #                                   (features[FLD.MAPOWER30] <
    #                                   features[FLD.CCI_NORM]) & \
    #                                   (features[FLD.ZEN_WAVELET_TIMING_LAG]
    #                                   >= -4)), -1,
    #                                   np.where((features[FLD.MAPOWER30_TIMING_LAG]
    #                                   < 0) & \
    #                                            (features[FLD.DEA_ZERO_TIMING_LAG]
    #                                            < 0) & \
    #                                            (((features[FLD.ZEN_WAVELET_TIMING_LAG]
    #                                            < 0) & \
    #                                            (features[FLD.DEA_ZERO_TIMING_LAG]
    #                                            <
    #                                            features[FLD.ZEN_WAVELET_TIMING_LAG]))
    #                                            | \
    #                                            ((features[FLD.LINEAREG_BAND_TIMING_LAG]
    #                                            < 0) & \
    #                                            (features[FLD.LINEAREG_PRICE]
    #                                            > features[FLD.HMA10]) & \
    #                                            (features[FLD.DEA_ZERO_TIMING_LAG]
    #                                            <
    #                                            features[FLD.LINEAREG_BAND_TIMING_LAG]))
    #                                            | \
    #                                            ((features[FLD.RENKO_BOOST_L_TIMING_LAG]
    #                                            < 0) & \
    #                                            (features[FLD.LINEAREG_PRICE]
    #                                            > features[FLD.HMA10]) & \
    #                                            (features[FLD.DEA_ZERO_TIMING_LAG]
    #                                            <
    #                                            features[FLD.RENKO_BOOST_L_TIMING_LAG]))
    #                                            & \
    #                                            (features[FLD.CCI_NORM] <
    #                                            features[FLD.MAPOWER30])) & \
    #                                            ~((features[FLD.MAPOWER30] <
    #                                            0.0382) & \
    #                                            (features[FLD.RENKO_BOOST_L_TIMING_LAG]
    #                                            > 0)), -1,
    #                                            np.where((features[FLD.DEA_ZERO_TIMING_LAG]
    #                                            < 0) & \
    #                                                     ((features[FLD.ZEN_WAVELET_TIMING_LAG]
    #                                                     < 0) | \
    #                                                     ((features[FLD.CCI_NORM].shift(1)
    #                                                     > 0.618) & \
    #                                                     (features[FLD.CCI_NORM].shift(1)
    #                                                     >
    #                                                     features[FLD.CCI_NORM])
    #                                                     & \
    #                                                     (features[FLD.CCI_NORM]
    #                                                     <
    #                                                     features[FLD.MAPOWER30])
    #                                                     & \
    #                                                     (features[FLD.DEA_ZERO_TIMING_LAG]
    #                                                     <
    #                                                     features[FLD.ZEN_WAVELET_TIMING_LAG]))
    #                                                     | \
    #                                                     (features[FLD.HMA10]
    #                                                     <
    #                                                     features[FLD.LINEAREG_PRICE]))
    #                                                     & \
    #                                                     (features[FLD.CCI_NORM]
    #                                                     <
    #                                                     features[FLD.MAPOWER30])
    #                                                     & \
    #                                                     ~((features[FLD.ATR_SuperTrend_TIMING_LAG]
    #                                                     > 0) & \
    #                                                     (features[FLD.MAPOWER30_TIMING_LAG]
    #                                                     > 0) & \
    #                                                     (features[FLD.ZEN_WAVELET_TIMING_LAG]
    #                                                     > 0) & \
    #                                                     (features[FLD.RENKO_TREND_L_TIMING_LAG]
    #                                                     +
    #                                                     features[FLD.ATR_SuperTrend_TIMING_LAG]
    #                                                     > 0)), -1, 1)))))
    #    features[FLD.RENKO_BOLD_TIMING_LAG] =
    #    calc_event_timing_lag(renko_bold)

        #renko_bold = np.where((features[FLD.MAPOWER30_TIMING_LAG] < -2) & \
        #                      (features[FLD.PEAK_LOW_TIMING_LAG] < -2) & \
        #                      (features[FLD.RENKO_BOLD_TIMING_LAG] == 1), -1,
        #                      renko_bold)
        #features[FLD.RENKO_BOLD_TIMING_LAG] =
        #calc_event_timing_lag(renko_bold)

        #features[FLD.RENKO_BOOST_L_TIMING_LAG] =
        #np.where((features[FLD.RENKO_BOOST_L_TIMING_LAG] > 0) & \
        #    (features[FLD.RENKO_TREND_L_TIMING_LAG] > 12) & \
        #    (features[FLD.COMBINE_DENSITY].rolling(4).mean() < 0.466) & \
        #    ((features[FLD.REGTREE_TIMING_LAG] <= -6) | \
        #    ((features[FLD.REGTREE_TIMING_LAG] < 0) & \
        #    (features[FLD.MA90_CLEARANCE] > 1.68))) & \
        #    (features[FLD.MACD] < features[FLD.DIF]), -1,
        #                                                 features[FLD.RENKO_BOOST_L_TIMING_LAG])

        #features[FLD.RENKO_TREND_L_TIMING_LAG] = np.where((features[FLD.MA90]
        #> features[FLD.MA120]) & \
        #    (features[FLD.MA90_CLEARANCE_TIMING_LAG] > 36) & \
        #    (features[FLD.REGTREE_PRICE] > features[FLD.MA90]) & \
        #    (features[FLD.REGTREE_TIMING_LAG] > 0) & \
        #    (features[FLD.DEA_ZERO_TIMING_LAG] > 0) & \
        #    (features[FLD.RENKO_BOOST_L_TIMING_LAG] ==
        #    features[FLD.RENKO_TREND_L_TIMING_LAG]), 1,
        #                                                 features[FLD.RENKO_TREND_L_TIMING_LAG])
        #features[FLD.RENKO_TREND_L_TIMING_LAG] = np.where(((features[FLD.MA90]
        #> features[FLD.MA120]) | \
        #    (features[FLD.COMBO_FLOW_TIMING_LAG] > 0)) & \
        #    ((features[FLD.DEA_ZERO_TIMING_LAG] < 0) | \
        #    (features[FLD.MACD_ZERO_TIMING_LAG] < 0)) & \
        #    (features[FLD.COMBINE_DENSITY].diff(1) > 0) & \
        #    (features[FLD.HMA10].diff(1) > 0) & \
        #    ((features[FLD.REGTREE_TIMING_LAG] > 0) | \
        #    (features[FLD.ZEN_WAVELET_TIMING_LAG] > 0) | \
        #    ((features[FLD.HMA10] > features[FLD.LINEAREG_PRICE]))), 1,
        #                                                 features[FLD.RENKO_TREND_L_TIMING_LAG])

        #features[FLD.RENKO_TREND_L_TIMING_LAG] =
        #np.where((features[FLD.DEA_ZERO_TIMING_LAG] < 0) & \
        #    (features[FLD.COMBINE_DENSITY] < 0.3) & \
        #    (features[FLD.HMA10].diff(1) > 0) & \
        #    (features[FLD.HMA10] > features[FLD.LINEAREG_PRICE]), 1,
        #                                                 features[FLD.RENKO_TREND_L_TIMING_LAG])

        #features[FLD.RENKO_TREND_L_TIMING_LAG] = np.where(((features[FLD.MA90]
        #> features[FLD.MA120]) | \
        #    ((features[FLD.RENKO_TREND_L_TIMING_LAG] > 0) | \
        #    (features[FLD.MACD_DELTA] > 0)) | \
        #    (features[FLD.COMBO_FLOW_TIMING_LAG] > 0)) & \
        #    ((features[FLD.DEA_ZERO_TIMING_LAG] < 0) | \
        #    (features[FLD.MACD_ZERO_TIMING_LAG] < 0)) & \
        #    ((features[FLD.COMBINE_DENSITY].diff(1) > 0) | \
        #    (features[FLD.MACD_DELTA].shift(1) > 0)) & \
        #    (features[FLD.HMA10].diff(1) > 0) & \
        #    ((features[FLD.REGTREE_TIMING_LAG] > 0) | \
        #    (features[FLD.ZEN_WAVELET_TIMING_LAG] > 0) | \
        #    ((features[FLD.HMA10] > features[FLD.LINEAREG_PRICE]))), 1,
        #                                                 features[FLD.RENKO_TREND_L_TIMING_LAG])

        #features[FLD.RENKO_TREND_L_TIMING_LAG] =
        #calc_event_timing_lag(np.where((features[FLD.RENKO_TREND_L_TIMING_LAG]
        #> 0), 1, -1))
        #features[FLD.RENKO_TREND_L_TIMING_LAG] =
        #calc_event_timing_lag(np.where((features[FLD.RENKO_TREND_L_TIMING_LAG]
        #> 0) & \
        #    ~((features[FLD.RENKO_TREND_L_TIMING_LAG] <= 6) & \
        #    (features[FLD.CCI].diff(2) < 0)), 1, -1))

    #checkpoint =
    #features.loc[features.index.get_level_values(level=0).intersection(pd.date_range('2020-09-16',
    #                                                                    periods=240,
    #                                                                    freq='30min')),
    #                          [FLD.RENKO_BOOST_L_TIMING_LAG,
    #                           FLD.COMBINE_DENSITY,
    #                           FLD.REGTREE_TIMING_LAG,
    #                           FLD.DEA_SLOPE]]
    #print(checkpoint)
    #checkpoint =
    #data.loc[data.index.get_level_values(level=0).intersection(pd.date_range('2020-11-18',
    #                                                                                      periods=240,
    #                                                                                      freq='30min')),
    #                          :]
    #print(checkpoint)
    return features


def calc_regtree_fractal_vXI(data, *args, **kwargs):
    '''
    快速计算regtree拟合线，因为超过500bar计算速度会变得很慢（超过5秒），所以 bar_limit 默认限制为 300
    '''
    # 针对多标的，拆分 indices 数据再自动合并
    code = data.index.get_level_values(level=1)[0]
    if ('features' in kwargs.keys()):
        features = kwargs['features'].loc[(slice(None), code), :]
    elif (len(args) > 0):
        features = args[0].loc[(slice(None), code), :]
    else:
        features = None
    bar_limit = kwargs['bar_limit'] if ('bar_limit' in kwargs.keys()) else 300
    rate = kwargs['rate'] if ('rate' in kwargs.keys()) else 25
    dur = kwargs['dur'] if ('dur' in kwargs.keys()) else 10

    ohlc_data = data[-bar_limit:].copy()
    price_dummy = features[FLD.HMA10][-bar_limit:].copy()
    price = np.c_[ohlc_data.apply(lambda x:ohlc_data.index.get_level_values(level=0).get_loc(x.name[0]), axis=1),
                  price_dummy]
    try:
        tree = predict_regression_tree_branch(np.asmatrix(price), rate, dur, 
                                              level=0)
    except Exception as e:
        if (len(ohlc_data)>460):
            print(u'Code:{} length:{}'.format(code, len(ohlc_data)), e)
        tree = None
        features[FLD.REGTREE_PRICE] = np.nan
        features[FLD.REGTREE_TREND] = np.nan
        features[FLD.REGTREE_SLOPE] = np.nan
        features[FLD.REGTREE_TIMING_LAG] = np.nan
        return features

    ohlc_data['seq_index'] = price[:, 0]
    regtree_flu = fit_regtree_trend(tree, ohlc_data)
    if (len(features) - bar_limit) > 0:
        features[FLD.REGTREE_PRICE] = np.r_[features[FLD.MA30][: - bar_limit], regtree_flu[:, 0]]
        features[FLD.REGTREE_TREND] = np.r_[np.empty(len(features) - bar_limit), regtree_flu[:, 1]]
        features[FLD.REGTREE_TREND_RS] = np.r_[np.empty(len(features) - bar_limit), regtree_flu[:, 2]]
        features[FLD.REGTREE_TIMING_LAG] = np.r_[np.empty(len(features) - bar_limit), calc_event_timing_lag(features[FLD.REGTREE_TREND].tail(bar_limit))]
    else:
        features[FLD.REGTREE_PRICE] = regtree_flu[:, 0]
        features[FLD.REGTREE_TREND] = regtree_flu[:, 1]
        features[FLD.REGTREE_TREND_RS] = regtree_flu[:, 2]
        features[FLD.REGTREE_TIMING_LAG] = calc_event_timing_lag(features[FLD.REGTREE_TREND])

    features[FLD.REGTREE_SLOPE] = talib.LINEARREG_SLOPE(features[FLD.REGTREE_PRICE], timeperiod=14)

    return features

    
def calc_regtree_fractal_vXIs(data, *args, **kwargs):
    '''
    快速计算regtree拟合线，因为超过500bar计算速度会变得很慢（超过5秒），所以 bar_limit 默认限制为 300
    此函数为返回数据为单精度浮点数版本
    '''
    # 针对多标的，拆分 indices 数据再自动合并
    code = data.index.get_level_values(level=1)[0]
    if ('features' in kwargs.keys()):
        features = kwargs['features'].loc[(slice(None), code), :]
    elif (len(args) > 0):
        features = args[0].loc[(slice(None), code), :]
    else:
        features = None
    bar_limit = kwargs['bar_limit'] if ('bar_limit' in kwargs.keys()) else 300
    rate = kwargs['rate'] if ('rate' in kwargs.keys()) else 25
    dur = kwargs['dur'] if ('dur' in kwargs.keys()) else 10

    ohlc_data = data[-bar_limit:].copy()
    if (FLD.HMA10 in features.columns):
        price_dummy = features[FLD.HMA10][-bar_limit:].copy()
    else:
        price_dummy = ohlc_data[AKA.CLOSE][-bar_limit:].copy()
    price = np.c_[ohlc_data.apply(lambda x:ohlc_data.index.get_level_values(level=0).get_loc(x.name[0]), axis=1),
                  price_dummy]

    features = features.reindex(columns=list(set([*features.columns,
                                                *[FLD.REGTREE_PRICE,
                                                FLD.REGTREE_TREND,
                                                FLD.REGTREE_SLOPE,
                                                FLD.REGTREE_TIMING_LAG,]])))
    try:
        tree = predict_regression_tree_branch(np.asmatrix(price), rate, dur, 
                                              level=0)
    except Exception as e:
        if (len(ohlc_data)>460):
            print(u'Code:{} length:{}'.format(code, len(ohlc_data)), e)
        tree = None
        return features

    ohlc_data['seq_index'] = price[:, 0]
    regtree_flu = fit_regtree_trend(tree, ohlc_data)
    if (len(features) - bar_limit) < 0:
        features[FLD.REGTREE_PRICE] = np.r_[ohlc_data[AKA.CLOSE][: - bar_limit-1], 
                                            regtree_flu[:, 0]].astype(np.float32)
        features[FLD.REGTREE_TREND] = np.r_[np.empty(len(features) - bar_limit), 
                                            regtree_flu[:, 1]].astype(np.int16)
        #features[FLD.REGTREE_TREND_RS] = np.r_[np.empty(len(features) - bar_limit), regtree_flu[:, 2]]
        features[FLD.REGTREE_TIMING_LAG] = np.r_[np.empty(len(features) - bar_limit), 
                                                 calc_event_timing_lag(features[FLD.REGTREE_TREND].tail(bar_limit))].astype(np.int16)
    else:
        features.loc[ohlc_data.index,
                FLD.REGTREE_PRICE] = regtree_flu[:, 0].astype(np.float32)
        features.loc[ohlc_data.index,
                FLD.REGTREE_TREND] = regtree_flu[:, 1].astype(np.int16)
        #features[FLD.REGTREE_TREND_RS] = regtree_flu[:, 2]
        features[FLD.REGTREE_TIMING_LAG] = calc_event_timing_lag(features[FLD.REGTREE_TREND]).astype(np.int16)

    features[FLD.REGTREE_SLOPE] = talib.LINEARREG_SLOPE(features[FLD.REGTREE_PRICE], timeperiod=14).astype(np.float32)

    return features