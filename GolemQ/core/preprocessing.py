# coding:utf-8
#
# The MIT License (MIT)
#
# Copyright (c) 2020-2025 azai/GolemQ
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import json
import numpy as np
import pandas as pd
import warnings


def mask_sensitive_info(value: str, sensitive: bool = True) -> str:
    """对敏感信息进行脱敏处理"""
    if not value or not sensitive:
        return value

    length = len(value)
    if length <= 4:
        return '*' * length

    # 保留前1/6和后1/6，中间用星号替换
    keep_start = length // 6
    keep_end = length // 6
    masked_part = '*' * (length - keep_start - keep_end)
    return value[:keep_start] + masked_part + value[-keep_end:]


def GQ_util_to_json_from_pandas(data):
    """
    explanation:
        将pandas数据转换成json格式

    params:
        * data ->:
            meaning: pandas数据
            type: null
            optional: [null]

    return:
        dict

    demonstrate:
        Not described

    output:
        Not described
    """

    """需要对于datetime 和date 进行转换, 以免直接被变成了时间戳"""
    if 'datetime' in data.columns:
        data = data.reindex(columns=list(set([*data.columns,
                                             *['datetime']])))
        data['datetime'] = data.datetime.apply(str)
    if 'date' in data.columns:
        data = data.reindex(columns=list(set([*data.columns,
                                             *['date']])))
        data['date'] = data.date.apply(str)
    return json.loads(data.to_json(orient='records'))


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
        factor = np.where(factor > up, up, factor)
        factor = np.where(factor < down, down, factor)
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
        factor = np.where(factor > up, up, factor)
        factor = np.where(factor < down, down, factor)
    return factor 


def standardize(s, ty=2):
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
    

def fill_zero_features(features, all_columns, all_columns_dtype):
    """
    将所有DataFrame中的缺失列统一补齐。
    如果不补齐，pd.concat() 将会强制将部分数据缺失column列的数据类型转换为 np.float64
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        features = features.reindex(columns=list(set(all_columns)))
        for column in features.columns:
            if ((features[column].dtype==np.int32) or \
                (features[column].dtype==np.int64) or \
                (features[column].dtype==np.float64)) and \
                (column not in ['created_at', 'date_stamp', 'time_stamp']):
                if (features[column].dtype==np.float64):
                    features[column] = features[column].astype(np.float32)
                elif (features[column].dtype==np.int32) or \
                    (features[column].dtype==np.int64):
                    features[column] = np.nan_to_num(features[column], nan=65535).astype(np.int16)

    return features


def prefill_null_columns(
        features_list,
        columns,
        columns_dtype):
    """
    填满NULL列，否则pd.concat()将会导致转换成 np.float64 类型。
    """
    for i, features in enumerate(features_list):
        if (features is not None) and (len(features.columns)<len(columns_dtype)):
            code = features.index.get_level_values(level=1)[0]
            #print(u'code: {} length {} columns {} less than {}'.format(code, len(features), len(features.columns), len(all_columns_dtype)))
            features_list[i] = fill_zero_features(features, columns, columns_dtype)
    
    return features_list if (len(features_list)>1) else features_list[0]


def concat_return_features(
        codelist_candidate,
        eval_range,
        ret_features,
):

    codelist_len = len(codelist_candidate)
    batch_size = 618

    # 判断是否需要分段合并
    if (codelist_len < batch_size) or ((eval_range == 'etf') and (codelist_len < 2*batch_size)):
        # 小数据量直接合并
        ret_features_pd = pd.concat(ret_features, axis=0, sort=True).sort_index()
    else:
        # 通用分段合并逻辑
        chunks = []
        num_chunks = (len(ret_features) + batch_size - 1) // batch_size  # 计算需要分几段
        
        for i in range(num_chunks):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(ret_features))
            chunk = pd.concat(ret_features[start_idx:end_idx], axis=0, sort=True)
            chunks.append(chunk)
        
        ret_features_pd = pd.concat(chunks, axis=0, sort=True).sort_index()

    return ret_features_pd
