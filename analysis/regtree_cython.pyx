#coding=utf-8
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
# regtree_cython.pyx

import numpy as np
cimport numpy as np
cimport cython

#(np.ndarray[np.double, ndim=2], np.ndarray[np.float64_t, ndim=2])
cpdef tuple _split_data_into_binary(np.ndarray[np.float64_t, ndim=2] seq_data, int feature, float value):
    """
    用feature把seq_data按value分成两个子集
    """
    cdef np.ndarray[np.float64_t, ndim=2] mat0
    cdef np.ndarray[np.float64_t, ndim=2] mat1

    mat0 = seq_data[np.nonzero(seq_data[:,feature] > value)[0],:]
    mat1 = seq_data[np.nonzero(seq_data[:,feature] <= value)[0],:]

    return mat0, mat1


cpdef tuple _solve_lineareg_func(np.ndarray[np.float64_t, ndim=2] seq_data):
    """
    求给定数据集的线性方程
    """
    cdef int m
    cdef int n
    m,n = np.shape(seq_data)

    cdef np.ndarray[np.float64_t, ndim=2] X
    cdef np.ndarray[np.float64_t, ndim=2] Y
    X = np.mat(np.ones((m,n))) # 第一行补1，线性拟合要求
    Y = np.mat(np.ones((m,1)))
    X[:,1:n] = seq_data[:,0:n - 1]
    Y = seq_data[:,-1] # 这里约定 数据最后一列是 价格数据

    cdef np.ndarray[np.float64_t, ndim=2] xTx
    cdef np.ndarray[np.float64_t, ndim=2] ws
    xTx = X.T * X
    if np.linalg.det(xTx) == 0.0:
        raise NameError('This matrix is singular, cannot do inverse,\n\try increasing dur')
    ws = xTx.I * (X.T * Y) # 公式推导较难理解
    return ws,X,Y


cpdef np.ndarray[np.float64_t, ndim=2] _fit_lineareg_slope_of_the_model_leaf(np.ndarray[np.float64_t, ndim=2] seq_data):
    """
    求线性方程的参数
    """
    cdef np.ndarray[np.float64_t, ndim=2] X
    cdef np.ndarray[np.float64_t, ndim=2] Y
    cdef np.ndarray[np.float64_t, ndim=2] ws
    ws,X,Y = _solve_lineareg_func(seq_data)
    return ws


cpdef np.float64_t _calc_err_of_the_model(np.ndarray[np.float64_t, ndim=2] seq_data):
    """
    预测值和y的方差
    """
    cdef np.ndarray[np.float64_t, ndim=2] X
    cdef np.ndarray[np.float64_t, ndim=2] Y
    cdef np.ndarray[np.float64_t, ndim=2] ws
    cdef np.ndarray[np.float64_t, ndim=2] yaxis_height_at
    ws, X, Y = _solve_lineareg_func(seq_data)
    yaxis_height_at = X * ws
    return sum(np.power(Y - yaxis_height_at, 2))


cpdef tuple _choose_best_split_branch(np.ndarray[np.float64_t, ndim=2] seq_data, int rate, int dur):
    """
    判断所有样本是否为同一分类
    """
    if len(set(seq_data[:,-1].T.tolist()[0])) == 1:
        return None, _fit_lineareg_slope_of_the_model_leaf(seq_data)

    cdef int m
    cdef int n
    m,n = np.shape(seq_data)

    cdef np.float64_t S
    cdef np.float64_t best_S
    S = _calc_err_of_the_model(seq_data) # 整体误差
    best_S = np.inf

    cdef int best_idx
    cdef np.float64_t best_val
    best_idx = 0
    best_val = 0

    cdef np.ndarray[np.float64_t, ndim=2] mat0
    cdef np.ndarray[np.float64_t, ndim=2] mat1
    cdef np.float64_t newS
    for feat_idx in range(n - 1): # 遍历所有特征, 此处只有一个
        # 遍历特征中每种取值
        for split_val in set(seq_data[:,feat_idx].T.tolist()[0]):
            mat0, mat1 = _split_data_into_binary(seq_data, feat_idx, split_val)
            if (np.shape(mat0)[0] < dur) or (np.shape(mat1)[0] < dur): 
                continue # 样本数太少, 前剪枝

             # 计算整体误差
            newS = _calc_err_of_the_model(mat0) + calc_err_of_the_model(mat1)
            if newS < best_S: 
                best_idx = feat_idx
                best_val = split_val
                best_S = newS

    if (S - best_S) < rate:
        # 如差误差下降得太少，则不切分
        return None, _fit_lineareg_slope_of_the_model_leaf(seq_data)

    mat0, mat1 = _split_data_into_binary(seq_data, best_idx, best_val)
    return best_idx, best_val


cpdef np.float64_t _tree_model_evaluation_func(np.ndarray[np.float64_t, ndim=2] model, np.ndarray[np.float64_t, ndim=2] branch_data):
    """
    预测评估函数,数据乘模型,模型是斜率和截距的矩阵
    """
    cdef int n
    n = np.shape(branch_data)[1]

    cdef np.ndarray[np.float64_t, ndim=2] X
    X = np.mat(np.ones((1,n + 1)))
    X[:,1:n + 1] = branch_data
    return float(X * model)


cpdef bint _is_tree(dict obj):
    """
    用字典保存的二叉树结构
    """
    return (type(obj).__name__ == 'dict')


cpdef np.float64_t _tree_model_forecast_decision_func(dict tree, np.ndarray[np.float64_t, ndim=2] branch_data):
    """
    预测/遍历整颗树的二元函数
    如果未预测，则预测树分支。
    如果已经预测，则遍历树分支 
    """
    if not _is_tree(tree):
        return _tree_model_evaluation_func(tree, branch_data)
    if branch_data[tree['spInd']] > tree['spVal']:
        if _is_tree(tree['left']):
            return _tree_model_forecast_decision_func(tree['left'], branch_data)
        else:
            return _tree_model_evaluation_func(tree['left'], branch_data)
    else:
        if _is_tree(tree['right']):
            return _tree_model_forecast_decision_func(tree['right'], branch_data)
        else:
            return _tree_model_evaluation_func(tree['right'], branch_data)


cpdef dict _predict_regression_tree_branch(np.ndarray[np.float64_t, ndim=2] seq_data, int rate, int dur):
    """
    生成回归树, seq_data是数据, rate是误差下降, dur是叶节点的最小样本数
    寻找最佳划分点, feat为切分点, val为值
    """
    cdef int feat
    cdef np.float64_t val
    feat, val = _choose_best_split_branch(seq_data, rate, dur)
    if feat == None:
        return val # 不再可分

    ret_tree = {}
    ret_tree['spInd'] = feat
    ret_tree['spVal'] = val

    # 把数据切给左右两树
    cdef np.ndarray[np.float64_t, ndim=2] left_set
    cdef np.ndarray[np.float64_t, ndim=2] right_set
    left_set, right_set = _split_data_into_binary(seq_data, feat, val)
    ret_tree['left'] = _predict_regression_tree_branch(left_set, rate, dur)
    ret_tree['right'] = _predict_regression_tree_branch(right_set, rate, dur)

    return ret_tree


def split_data_into_binary(seq_data:np.ndarray, feature:int, value:float):
    return _split_data_into_binary(seq_data, feature, value)


def solve_lineareg_func(seq_data:np.ndarray):
    return _solve_lineareg_func(seq_data)


def fit_lineareg_slope_of_the_model_leaf(seq_data:np.ndarray):
    return _fit_lineareg_slope_of_the_model_leaf(seq_data)


def calc_err_of_the_model(seq_data:np.ndarray):
    return _calc_err_of_the_model(seq_data)


def choose_best_split_branch(seq_data:np.ndarray, rate, dur):
    return _choose_best_split_branch(seq_data, rate, dur)


def tree_model_evaluation_func(model, branch_data):
    return _tree_model_evaluation_func(model, branch_data)


def is_tree(obj):
    return _is_tree(obj)


def tree_model_forecast_decision_func(tree, branch_data):
    return _tree_model_forecast_decision_func(tree, branch_data)

def predict_regression_tree_branch(seq_data, rate, dur):
    return _predict_regression_tree_branch(seq_data, rate, dur)