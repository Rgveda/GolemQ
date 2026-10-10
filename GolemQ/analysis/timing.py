# coding:utf-8
"""时域累积器与金叉/死叉时间间隔。

从哪来
======
逐字搬自旧树 `GolemQ_old/analysis/timeseries.py`（那一族函数原先住在那里），
因为**筹码分布**要用：`calc_feature_event_timing_lag` 是它的上游
（`ChipDistribution_jit` 里 `FLD.MA250_TIMING_LAG_MAJOR` 那一列）。

为什么**不并回** `analysis/timeseries.py`
========================================
那边现在只做**重采样**（`GQ_data_min_resample` / `GQ_data_min_to_day`，87 行）。
这一族是**逐 bar 累积**的另一件事（金叉/死叉的「距上一次多久」），并进来会把两个
关注点混在一个文件里，且立刻超过 300 行。

⚠️ 一个值得记的巧合：新树 `analysis/timeseries.py` 的 docstring 里写着
「原先的两个 stub（`Timeline_duration` / `align_kline_timeline`）已随
`services/persistence/` 的删除一并移除」。也就是说 `Timeline_duration` **曾经以
stub 形式存在、被当死代码删掉过**。现在真消费方来了，它**带着真实现回来**。

两条搬运动作（都影响可对拍性，别随手改）
======================================
1. **丢掉了 `@nb.jit` 装饰器**。旧树给 `calc_energy_f8/f4` 挂的是
   `@nb.jit('f4[:](f4[:])', nopython=True)`，而函数体返回 float64 —— **签名与实体
   不符**（旧树里疑似从未真的编译过）。新树 `analysis/` 全程不用 numba，`numba`
   也不在 `MIN_PACKAGES`（`cli/bootstrap.py`）里，故落成**纯 numpy/Python**，
   行为与翻译后的旧代码一致。
2. **函数名保留旧树原名**，含 CamelCase 的 `Timeline_*` —— 筹码分布是逐字搬的、
   按名调用；改名会让两边无法对拍。项目惯例是 snake_case，但这几个是**继承名**。
"""
from __future__ import annotations

import traceback

import numpy as np
import pandas as pd

__all__ = [
    'Timeline_Integral', 'Timeline_duration', 'calc_event_timing_lag',
    'calc_feature_event_timing_lag', 'calc_energy', 'calc_energy_f8',
    'calc_energy_f4', 'resample_multi_frequency_indices_func',
]


def Timeline_Integral(Tm: np.ndarray) -> np.ndarray:
    """时域金叉/死叉信号的累积和（**死叉 1→0 时清零**）。

    ``T[i] = Tm[i] * (T[i-1] + Tm[i])`` —— `Tm[i] == 0` 即清零。

    ⚠️ i=0 时 `T[-1]` 取的是数组**最后一个元素**（新数组全 0，结果正确）——
    依赖 numpy 的负索引，**保留原样**以免改口径。

    ⚠️ **只对 0/1 输入有意义** —— 它是给二值信号的。传别的值会得到公式产物：
    `Timeline_Integral([3]) == [9]`（因为 `3 * (0 + 3)`）。**别"顺手修"成
    `Tm[i] + T[i-1]`**：那会改口径，且旧树同公式。

    >>> Timeline_Integral(np.array([1, 1, 0, 1, 1], dtype=np.int32)).tolist()
    [1, 2, 0, 1, 2]
    >>> Timeline_Integral(np.array([3], dtype=np.int32)).tolist()   # 非二值的产物
    [9]
    """
    T = np.zeros(len(Tm)).astype(np.int32)
    for i, Tmx in enumerate(Tm):
        T[i] = Tmx * (T[i - 1] + Tmx)
    return T


def Timeline_duration(Tm: np.ndarray) -> np.ndarray:
    """时域累积和（**金叉 0→1 时清零**，与 :func:`Timeline_Integral` 相反）。

    ``T[i] = T[i-1] + 1 if Tm[i] != 1 else 0``。旧树注释：经测试 for 最快（比 reduce 快）。

    >>> Timeline_duration(np.array([0, 0, 1, 0, 0], dtype=np.int32)).tolist()
    [1, 2, 0, 1, 2]
    """
    T = np.zeros(len(Tm)).astype(np.int32)
    for i, Tmx in enumerate(Tm):
        T[i] = (T[i - 1] + 1) if (Tmx != 1) else 0
    return T


def calc_event_timing_lag(vhma_directions):
    """事件的时间间隔：金叉取**正**、死叉取**负**。

    >>> calc_event_timing_lag(np.array([1, 1, -1, -1, 1])).tolist()
    [1, 2, -1, -2, 1]
    """
    with np.errstate(invalid='ignore', divide='ignore'):
        vhma_trend_jx = Timeline_Integral(
            np.where(vhma_directions > 0, 1, 0).astype(np.int32))
        vhma_trend_sx = np.sign(vhma_directions) * Timeline_Integral(
            np.where(vhma_directions < 0, 1, 0).astype(np.int32))
        return np.array(vhma_trend_jx + vhma_trend_sx).astype(np.int32)


def calc_feature_event_timing_lag(features=None, column=None):
    """技术指标特征的金叉/死叉时序。

    旧树住在 `features/base.py`，但依赖链全在上面两个函数里 ⇒ 落在**同源**的这里，
    免得为 10 行代码复活一个 `features/base.py`。

    :param features: DataFrame，含 `column` 那一列
    :param column: 列名
    :return: int32 数组；``>0`` = 距上一次金叉的 bar 数，``<0`` = 距上一次死叉

    >>> import pandas as pd
    >>> df = pd.DataFrame({'x': [1.0, 2.0, 1.5, 0.5, 1.0]})
    >>> calc_feature_event_timing_lag(df, 'x').tolist()
    [-1, 1, -1, -2, 1]

    ⚠️ 第一个元素是 **-1 不是 +1**：`shift(1)` 给的是 NaN，而 `x[0] > NaN` 为 False
    ⇒ 金叉与死叉的累积都是 1 ⇒ `1 < 1` 为假 ⇒ 走 `-1` 分支。（本行期望值是**实测**的，
    不是推的 —— 我第一版就猜错成 +1。）
    """
    features_jx_before = Timeline_duration(
        np.where(features[column] > features[column].shift(1), 1, 0))
    features_sx_before = Timeline_duration(
        np.where(features[column] < features[column].shift(1), 1, 0))
    return calc_event_timing_lag(
        np.where((features_jx_before < features_sx_before), 1, -1))


def calc_energy_f8(signal):
    """`calc_energy` 的 float64 内核：同号累加、异号重启。

    >>> calc_energy_f8(np.array([1.0, 1.0, -1.0, 1.0])).tolist()
    [1.0, 2.0, -1.0, 1.0]
    """
    ret_energy = np.zeros(len(signal)).astype(np.float64)
    if (len(signal) > 0):
        ret_energy[0] = signal[0]
    for i in range(1, len(signal)):
        if (signal[i] > 0.0000000000000001) and (np.sign(ret_energy[i - 1]) == np.sign(signal[i])):
            ret_energy[i] = ret_energy[i - 1] + signal[i]
        elif (signal[i] < -0.0000000000000001) and (np.sign(ret_energy[i - 1]) == np.sign(signal[i])):
            ret_energy[i] = ret_energy[i - 1] + signal[i]
        elif (np.sign(ret_energy[i - 1]) != np.sign(signal[i])):
            ret_energy[i] = signal[i]
        elif (np.isnan(signal[i])):
            ret_energy[i] = signal[i]
    return ret_energy


def calc_energy_f4(signal):
    """`calc_energy` 的 float32 内核（逻辑同 :func:`calc_energy_f8`，只差 dtype）。"""
    ret_energy = np.zeros(len(signal)).astype(np.float32)
    if (len(signal) > 0):
        ret_energy[0] = signal[0]
    for i in range(1, len(signal)):
        if (signal[i] > 0.0000000000000001) and (np.sign(ret_energy[i - 1]) == np.sign(signal[i])):
            ret_energy[i] = ret_energy[i - 1] + signal[i]
        elif (signal[i] < -0.0000000000000001) and (np.sign(ret_energy[i - 1]) == np.sign(signal[i])):
            ret_energy[i] = ret_energy[i - 1] + signal[i]
        elif (np.sign(ret_energy[i - 1]) != np.sign(signal[i])):
            ret_energy[i] = signal[i]
        elif (np.isnan(signal[i])):
            ret_energy[i] = signal[i]
    return ret_energy


def calc_energy(signal: pd.Series):
    """信号的**绝对能量**（同号连续累加、异号重启），按 dtype 选内核。

    >>> calc_energy(np.array([1.0, 1.0, -1.0, 1.0])).tolist()
    [1.0, 2.0, -1.0, 1.0]
    """
    ret_energy = np.nan
    if (signal.dtype == np.float64):
        ret_energy = (calc_energy_f8(signal) if isinstance(signal, np.ndarray)
                      else calc_energy_f8(signal.values))
    elif (signal.dtype == np.float32):
        ret_energy = (calc_energy_f4(signal) if isinstance(signal, np.ndarray)
                      else calc_energy_f4(signal.values))
    else:
        try:
            signal = signal.astype(np.float32)
            ret_energy = (calc_energy_f4(signal) if isinstance(signal, np.ndarray)
                          else calc_energy_f4(signal.values))
        except Exception:              # noqa: BLE001
            print(u'Unsupport type:{}'.format(signal.dtype))
            traceback.print_exc()
    return ret_energy


def resample_multi_frequency_indices_func(data, *args, **kwargs):
    """把**另一个频率**算好的指标对齐到 `data` 的时间轴上（**单标的**）。

    `indices` 可按 code 拆一次；旧树注释：不要尝试传复杂标的。

    :param data: 目标帧，索引为 `(时间, code)` 两级
    :param indices: 已算好的指标帧（位置参数或 `indices=` 均可）
    :param frequency: ``'30min'`` 走**重采样**分支；``'1D'/'day'`` 等走 **ffill 对齐**分支
    """
    code = data.index.get_level_values(level=1)[0]
    if ('indices' in kwargs.keys()) and (kwargs['indices'] is not None):
        try:
            indices = kwargs['indices'].loc[(slice(None), code), :]
        except Exception as e:          # noqa: BLE001
            print(u"resample_multi_frequency_indices_func() invalid paramters: agrs[0] or kwargs['indices']")
            print(e)
            print(kwargs['indices'].index if isinstance(kwargs['indices'], pd.DataFrame)
                  else kwargs['indices'])
            traceback.print_exc()
            indices = None
    elif (len(args) > 0) and (args[0] is not None):
        try:
            indices = args[0].loc[(slice(None), code), :]
        except Exception:              # noqa: BLE001
            print(type(args[0]))
            indices = None
    else:
        print(u"resample_multi_frequency_indices_func() Missing paramters: agrs[0] or kwargs['indices']")
        indices = None

    frequency = kwargs['frequency'] if ('frequency' in kwargs.keys()) else '1h'

    if frequency in ['30min']:
        indices = indices.reset_index(level=[1], drop=False)
        if isinstance(indices.index, pd.PeriodIndex):
            indices.index = indices.index.to_timestamp('D')
        indices = indices.resample('30min').bfill()
        # ⚠️ 旧树按 pandas 版本分岔写 `'23.5H'` / `'23.5h'`；pandas 2.x 起小写是唯一
        # 无歧义的写法（大写 H 会告警/被弃用），故只留小写。
        from pandas.tseries.frequencies import to_offset
        indices.index = indices.index + to_offset('23.5h')
        indices = indices.reindex(data.index.get_level_values(level=0))
    else:
        indices = pd.DataFrame(indices, index=data.index)
        indices = indices.loc[data.index.get_level_values(level=0)].ffill()

    if (frequency in ['1D', 'W', 'w', '1d', 'day', '1day']):
        indices['date'] = pd.to_datetime(data.index.get_level_values(level=0),)
        indices['code'] = code
        indices = indices.set_index(['date', 'code'], drop=True)
    else:
        indices['datetime'] = pd.to_datetime(data.index.get_level_values(level=0),)
        indices['code'] = code
        indices = indices.set_index(['datetime', 'code'], drop=True)

    return indices
