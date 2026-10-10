# coding:utf-8
"""鲁棒性极值点识别（**PEAK_POINT**）—— 基于方差统计与 ZSCORE 排序。

从哪来
======
* :func:`thresholding_algo` —— 旧树 `analysis/timeseries.py`。算法源自
  https://stackoverflow.com/questions/22583391/ （Robust peak detection）。
  旧树注释：**「本实现使用 Numba JIT 优化，比原版大约快了 500 倍」** ——
  它是那一轮 jit 尝试里**真正成功**的一个（与 `timeseries` 里那批被注释掉的
  「经测试这样无 JIT 最快」形成对照）。
* :func:`calc_peak_point_v8` —— 旧树 `fractal/v8.py`（`v8_1.py` 有同名同实现）。
* :func:`calc_peak_points` —— 旧树 `analysis/timeseries.py`（`features/base.py`
  的 `calc_peak_point_vX` 与它同源）。

写到哪一列
==========
`TREND_STATUS.PEAK_POINT`（值 ``'PEAK_POINT'``）—— 老树 `cli/analysis.py` 的注释把它
读作「**主升浪买点**」（`PEAK_POINT > 0`）。⚠️ 新树原先**没有**这个常量，已按旧树真值补进
`core/constants.py`。

关于 numba
==========
旧树给 :func:`thresholding_algo` 挂了 `@nb.jit(nopython=True)` 并实测约 500×。
本模块**保留这个能力但不强制**：纯实现永远是参照，numba 在则用 jit 版，
**两者输出逐值一致**由 `test_peak.py` 钉住。理由与 `regtree_jit` 同：
`numba` 不在 `MIN_PACKAGES` 里（`cli/bootstrap.py`），缺它时本模块仍须可用。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from GolemQ.core.constants import AKA, TREND_STATUS as ST

__all__ = ['thresholding_algo', 'thresholding_algo_py', 'peak_status',
           'calc_peak_point_v8', 'calc_peak_points']


def _nb():
    import numba
    return numba


def peak_status() -> dict:
    """numba 在不在、jit 开没开（`DISABLE_JIT` 会关掉）。"""
    try:
        nb = _nb()
    except ImportError as exc:
        return {'jit': False, 'reason': '{}: {}'.format(type(exc).__name__, exc)}
    cfg = getattr(nb, 'config', None)
    return {'jit': not int(getattr(cfg, 'DISABLE_JIT', 0)), 'numba': nb.__version__}


def _kernels():
    """惰性建 jit 核（编译要几百毫秒，不该在 import 期付）。"""
    cached = getattr(_kernels, '_cache', None)
    if cached is not None:
        return cached
    njit = _nb().njit

    @njit(cache=True)
    def _th_jit(y, lag, threshold, influence):
        """见 :func:`thresholding_algo` 的说明（本函数是它的 jit 版）。"""
        ret_signals = np.zeros((3, len(y),))
        idx_signals = 0
        idx_avgFilter = 1
        idx_stdFilter = 2
        filteredY = np.copy(y)
        ret_signals[idx_avgFilter, lag - 1] = np.mean(y[0:lag])
        ret_signals[idx_stdFilter, lag - 1] = np.std(y[0:lag])
        for i in range(lag, len(y)):
            if abs(y[i] - ret_signals[idx_avgFilter, i - 1]) > \
                    threshold * ret_signals[idx_stdFilter, i - 1]:
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

    _kernels._cache = _th_jit
    return _th_jit


def thresholding_algo_py(y, lag, threshold, influence):
    """纯 numpy 参照实现（**jit 版必须与它逐值一致**）。"""
    y = np.asarray(y, dtype=np.float64)
    ret_signals = np.zeros((3, len(y),))
    idx_signals, idx_avg, idx_std = 0, 1, 2
    filteredY = np.copy(y)
    ret_signals[idx_avg, lag - 1] = np.mean(y[0:lag])
    ret_signals[idx_std, lag - 1] = np.std(y[0:lag])
    for i in range(lag, len(y)):
        if abs(y[i] - ret_signals[idx_avg, i - 1]) > threshold * ret_signals[idx_std, i - 1]:
            ret_signals[idx_signals, i] = 1 if y[i] > ret_signals[idx_avg, i - 1] else -1
            filteredY[i] = influence * y[i] + (1 - influence) * filteredY[i - 1]
        else:
            ret_signals[idx_signals, i] = 0
            filteredY[i] = y[i]
        ret_signals[idx_avg, i] = np.mean(filteredY[(i - lag + 1):i + 1])
        ret_signals[idx_std, i] = np.std(filteredY[(i - lag + 1):i + 1])
    return ret_signals


def thresholding_algo(y, lag, threshold, influence):
    """鲁棒极值点识别（z-score）。返回 **3×len(y)** 的数组。

    三行依次是 ``signals``（``1`` 上峰 / ``-1`` 下谷 / ``0`` 无）、
    ``avgFilter``、``stdFilter``。**调用方取 ``[0, :]`` 就是极值点序列。**

    :param y: 序列（价格）
    :param lag: 滑动窗口（旧树用 5）
    :param threshold: 偏离多少个标准差算极值（旧树用 3.5）
    :param influence: 极值点对滤波均值的衰减影响（旧树用 0.5）

    ⚠️ 前 `lag-1` 个位置的 `avgFilter`/`stdFilter` 是 **0**（只有第 `lag-1` 个被填），
    而 `signals` 前 `lag` 个恒为 0 —— 照抄旧实现，别"顺手补全"。

    >>> a = thresholding_algo(np.array([1., 1., 1., 1., 1., 9., 1., 1., 1., 1.]), 5, 3.5, 0.5)
    >>> a.shape
    (3, 10)
    >>> a[0].tolist()          # 第 6 个点（值 9）被识别成上峰
    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    """
    if peak_status().get('jit'):
        return _kernels()(np.asarray(y, dtype=np.float64), lag, threshold, influence)
    return thresholding_algo_py(y, lag, threshold, influence)


def calc_peak_point_v8(ohlc_data: pd.DataFrame, features: pd.DataFrame = None) -> pd.DataFrame:
    """**鲁棒性极值点识别**（基于方差统计与 ZSCORE 排序）—— 写 `ST.PEAK_POINT`。

    旧树 `fractal/v8.py` 与 `v8_1.py` 各有一份**逐字相同**的实现（那是老树里的
    平行实现；本模块**只留一份**）。

    参数写死为 `lag=5 / threshold=3.5 / influence=0.5`（旧树如此，未参数化 ——
    要调就先看清楚消费方）。

    :param ohlc_data: 含 `close` 列的行情帧
    :param features: 要写入的帧；``None`` 则新建一个（索引取 `ohlc_data.index`）
    :return: 带 `ST.PEAK_POINT` 列的帧

    >>> import pandas as pd
    >>> px = pd.DataFrame({'close': [1., 1., 1., 1., 1., 9., 1., 1., 1., 1.]})
    >>> calc_peak_point_v8(px)[ST.PEAK_POINT].tolist()
    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    """
    lag, threshold, influence = 5, 3.5, 0.5
    peak_point_candicate = thresholding_algo(
        ohlc_data[AKA.CLOSE].values, lag=lag, threshold=threshold,
        influence=influence)[0, :]

    if (features is None):
        features = pd.DataFrame(peak_point_candicate, index=ohlc_data.index,
                                columns=[ST.PEAK_POINT])
    else:
        features[ST.PEAK_POINT] = peak_point_candicate
    return features


def calc_peak_points(closep, lineareg_price):
    """**两个输入**的极值合成：收盘价算一遍、拟合价算一遍，**加权合成**。

    权重是 `9 - i`（第 1 个输入权 9、第 2 个权 8），且**非零者覆盖零值** ——
    也就是「后一个输入只在前面都没给出信号的位置上说话」。

    :return: 与 `closep` 等长的数组（``0`` = 无信号）

    >>> calc_peak_points(np.array([1., 1., 1., 1., 1., 9.]), np.zeros(6)).tolist()
    [0.0, 0.0, 0.0, 0.0, 0.0, 9.0]
    """
    lag, threshold, influence = 5, 3.5, 0.5
    peak_point_candicate = []
    peak_point_candicate.append(thresholding_algo(closep, lag=lag, threshold=threshold,
                                                  influence=influence)[0, :])
    peak_point_candicate.append(thresholding_algo(lineareg_price, lag=lag,
                                                  threshold=threshold,
                                                  influence=influence)[0, :])
    ret_peak_points = np.zeros(len(closep))
    for i in range(0, len(peak_point_candicate)):
        peak_point = peak_point_candicate[i]
        ret_peak_points = np.where(peak_point != 0,
                                   peak_point * (9 - i),
                                   ret_peak_points)
    return ret_peak_points
