# coding:utf-8
"""Poly9 家族的**编排层** —— 把 `rolling_poly9_v2` / `mapower` 串成指标列。

本模块与 `analysis/timeseries.py` / `analysis/quadrant.py` 的分工
==============================================================
* 那两处是**算术核**（`rolling_poly9_v2`、`mapower_np_vXIIs`）——纯序列进、纯序列出；
* 本模块做**编排**：从 `ohlc_data` 与 `features` 里取输入、拼 `s`、
  写回 `features` 的列。与 :mod:`analysis.stages` 是同一种角色。

搬了什么
=======
老树 `features/base.py:2091` 的 `calc_polynomial9_vXIs` —— 逐字搬。

⚠️ **`POLYNOMIAL9_NORM` 那一行老树不在这个函数里**：它在调用方
（`features/reviews.py:1606`，`calc_zen_wavelet_reality_major` 内联），
且带一个 `annual > 200` 的闸。本模块把它**提成具名函数**
:func:`calc_polynomial9_norm_vXIs` —— 因为那个闸是**策略**，散在调用方
会被各处抄一遍（本项目的老毛病）；提出来才能钉住、也才能被对拍覆盖。
**算术一字未改**（`range_list` 仍是 `5..29`）。

⚠️ `*_REAL` 那一族（`poly9MajR` / `poly9WekR` / …）**不在本模块** ——
它们是**周线/月线重采样后 rename**，属编排的更外层（老树在
`features/empirical.py:3838` 的 rename map 里）。等 `features/` 那两个 stub 解掉时再落。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from GolemQ.core.constants import AKA, FIELD as FLD
from .quadrant import mapower_np_vXIIs
from .timeseries import rolling_poly9_v2
from .timing import calc_event_timing_lag

__all__ = ['calc_polynomial9_vXIs', 'calc_polynomial9_norm_vXIs']

#: `rolling_poly9_v2` 的窗口 —— 老树在 `features/base.py:2110` 写死 252。
POLY9_WINDOW = 252

#: `HMA10` 的暖机长度 + 1 —— 拼 `s` 时**从这一位起**才用 HMA。
#: ⚠️ 这个 11 是 `TA_HMA(..., period=10)` 的暖机长度（见 `analysis/indicators.TA_HMA`），
#: **两者绑死**：换 HMA 周期就得改这里，否则会给 `rolling_poly9_v2` 喂进 NaN。
HMA_WARMUP = 11

#: `POLYNOMIAL9_NORM` 的闸：`annual` 大于它才算（老树写死在调用方里）。
NORM_ANNUAL_GATE = 200


def calc_polynomial9_vXIs(ohlc_data: pd.DataFrame, features: pd.DataFrame = None,
                          annual: int = 252, verbose: bool = False):
    """`POLYNOMIAL9` / `POLYNOMIAL9_DELTA` / `POLYNOMIAL9_TIMING_LAG` 三列。

    逐字搬自老树 `features/base.py:2091`。

    ⚠️ **幂等**：`if FLD.POLYNOMIAL9 not in features.columns` ——
    列已在就**整段跳过**（不是覆盖）。别改成无条件重算：调用方（`reviews.py` 那条链）
    靠这条来避免重复计算，改了会让每跑一次都重算一遍。

    **输入怎么拼**（这段是它最容易错的地方）：

        s = np.r_[close[0:11], HMA10[11:]]

    也就是：**前 11 根用原始收盘价、第 11 根起用 HMA10**。为什么是 11 ——
    `TA_HMA(period=10)` 的暖机长度正好 11，前 11 个是 NaN，**必须用收盘价顶掉**。
    拼完 `s` **不带 NaN**（有用例钉着）。`features[FLD.HMA10]` 长度 ≤ 12 时退化成纯收盘价。

    ⚠️ `annual` / `verbose` **在本函数里没有用到**（老树也没用）—— 保留是为了
    **签名兼容**（调用方按位置传）。`annual` 真正的用处在下游的
    :func:`calc_polynomial9_norm_vXIs`。

    ⚠️ **`features` 会被就地修改**（老树如此），且 `reindex` 会返回新对象 ——
    本函数返回它，**调用方必须接住返回值**。

    :param ohlc_data: 至少含 `close` 的帧（老树还从中取 index 的 code）
    :param features: 承载列的帧；`None` ⇒ 只打印一句就返回（**不是抛异常**，老树如此）
    :param annual: 见上（本函数不用）
    :param verbose: 见上（本函数不用）
    :returns: `features`（`features is None` 时原样返回 `None`）
    """
    if features is None:
        # ⚠️ 老树这里是**打印后返回**，不是抛。**保留返回 `None` 的语义**
        # （调用方按返回值判空）—— 只把「打印」这一半收进 `verbose`。
        # 这是本模块**唯一一处刻意偏离逐字照搬**：老树无条件 `print`，而子层裸
        # `print` 会打花 banner（`PITFALLS.md` **P22**）。信息一个字没少，
        # 只是默认不吵。
        if verbose:
            code = ohlc_data.index.get_level_values(level=1)[0]
            print(u'code:{} kline data length ({}) less than minial required.'
                  .format(code, len(ohlc_data)))
        return features

    if FLD.POLYNOMIAL9 not in features.columns:
        if len(features[FLD.HMA10]) > HMA_WARMUP + 1:
            s = np.r_[ohlc_data[AKA.CLOSE].values[0:HMA_WARMUP],
                      features[FLD.HMA10].values[HMA_WARMUP:]]
        else:
            s = ohlc_data[AKA.CLOSE].values

        features = features.reindex(columns=list(set([*features.columns,
                                                      *[FLD.POLYNOMIAL9,
                                                        FLD.POLYNOMIAL9_DELTA,
                                                        FLD.POLYNOMIAL9_TIMING_LAG]])))
        features[FLD.POLYNOMIAL9] = rolling_poly9_v2(s, POLY9_WINDOW).astype(np.float32)
        features[FLD.POLYNOMIAL9_DELTA] = features[FLD.POLYNOMIAL9].diff(1)
        features[FLD.POLYNOMIAL9_TIMING_LAG] = calc_event_timing_lag(
            np.where(features[FLD.POLYNOMIAL9_DELTA] > 0, 1, -1)).astype(np.int16)

    return features


def calc_polynomial9_norm_vXIs(features: pd.DataFrame, annual: int = 252):
    """`POLYNOMIAL9_NORM` —— MA 均线多头排列能量强度（`0..1`）。

    ⚠️ **老树没有这个函数**：那一行内联在 `features/reviews.py:1606`
    （`calc_zen_wavelet_reality_major` 里），本模块把它提出来只为让
    **那个 `annual > 200` 的闸**有个名字、能被钉住。**算术逐字照抄**
    （`mapower_np_vXIIs(POLYNOMIAL9, range_list=5..29)`）。

    ⚠️ 闸不满足时**不写该列**（老树如此），不是写 0。

    :param features: 需已含 `FLD.POLYNOMIAL9`（由 :func:`calc_polynomial9_vXIs` 产出）
    :param annual: 大于 :data:`NORM_ANNUAL_GATE` 才算
    :returns: `features`（就地加列）
    """
    if (annual > NORM_ANNUAL_GATE) and (FLD.POLYNOMIAL9 in features.columns):
        features[FLD.POLYNOMIAL9_NORM] = mapower_np_vXIIs(
            features[FLD.POLYNOMIAL9].values,
            range_list=np.array(range(5, 30)).astype(np.int16))
    return features
