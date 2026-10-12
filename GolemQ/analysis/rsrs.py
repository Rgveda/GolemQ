# coding:utf-8
"""RSRS（阻力支撑相对强度）**指标计算** —— 逐字搬自老树 `signal/rsrs.py`。

RSRS 是什么
==========
用**滚动 OLS** 把 `high` 对 `low` 回归：斜率 beta = 支撑把价格顶起来的强度，
R² = 这个线性关系的可靠度。再对 beta 做滚动标准分，得到三个派生量：

======================  ==========================================  ============
列                       含义                                          老树列名
======================  ==========================================  ============
`RSRS_BETA`             滚动 OLS 斜率（窗口 `N`=18）                    `RSRS_BETA`
`RSRS_BETA_R2`          该回归的 R²                                    `RSRS_BETA_R2`
`RSRS_ZSCORE`           beta 的滚动标准分（窗口 `M`=252）                `RSRS_ZSCORE`
`RSRS_ZSCORE_R2`        **标准分** = zscore × R²                        `RSRS_ZS_R2`
`RSRS_RIGHT`            **右偏标准分** = 标准分 × beta                   `RSRS_RIGHT`
======================  ==========================================  ============

搬了哪一条 / 没搬哪一条（D35「只搬一条」）
======================================
老树 `signal/rsrs.py` 995 行、14 个函数，其中 **8 个是择时信号/策略**
（`rsrs1_signal_func` / `rsrsN_np` / `RSRS1-4` / `combine_*`）—— **本轮只搬指标计算**（用户
2026-10-12 定）。另有一个 numpy 孪生 `pre_rsrs_zscore_np_func` **在老树就是坏的**：
它用了 `idx_RSRS_ZSCORE`，而那名字只在 :func:`pre_rsrs_data_func` 里是**局部变量**
⇒ 一调就 `NameError` —— **故不搬**（同 `GQ_fix_daily_metadata` 那类死孪生）。

解耦
====
* 删掉开头的 `if (ST.VERBOSE in data.columns): print(...)` 调试门控 ——
  新树**有意不补 `VERBOSE` 常量**（用户 2026-10-10），连带 `QA_util_timestamp_to_str` 一起走。
* 入参统一是 **`pd.DataFrame`**（含 `high` / `low` 列）—— 老树那份「支持 QUANTAXIS
  `DataStruct.add_func`」的用法不再支持。
* 老树 `from GolemQ.analysis.timeseries import *` 进来的 `rolling_zscore` 现由
  :mod:`GolemQ.analysis.timing` 提供。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.stats as scs
import statsmodels.api as sm

from GolemQ.analysis.timing import rolling_zscore
from GolemQ.core.constants import FIELD as FLD

__all__ = ['pre_rsrs_data_func']


def pre_rsrs_data_func(data, *args, **kwargs):
    """准备 RSRS 数据 —— 逐字搬自老树 `signal/rsrs.py:54`。

    窗口 `N`=18（回归）、`M`=252（标准分）。`N` / `M` 可由**位置参数**或关键字
    ``N=`` / ``M=`` 覆盖；给了 ``indices=`` 则把 5 列**写回**那个 DataFrame
    而不是新建一个。

    ⚠️ 回归是**逐 bar 一个 `statsmodels.OLS`**（`for i in range(len(highs))[N:]`）——
    慢，但**逐字照搬**：向量化会改数值（`statsmodels` 与手写最小二乘的最低位不同），
    而本项目的对拍判据是 `array_equal` **不容差**（`DECISIONS.md` D36）。

    >>> import numpy as np, pandas as pd
    >>> idx = pd.MultiIndex.from_product(
    ...     [pd.date_range('2020-01-01', periods=40, freq='D'), ['600519']],
    ...     names=['datetime', 'code'])
    >>> rng = np.random.default_rng(0)
    >>> ohlc = pd.DataFrame({'high': 100 + rng.normal(size=40).cumsum(),
    ...                      'low': 100 + rng.normal(size=40).cumsum()}, index=idx)
    >>> out = pre_rsrs_data_func(ohlc)
    >>> list(out.columns) == [FLD.RSRS_BETA, FLD.RSRS_BETA_R2, FLD.RSRS_ZSCORE,
    ...                       FLD.RSRS_ZSCORE_R2, FLD.RSRS_RIGHT]
    True
    >>> out.index.equals(ohlc.index)
    True
    """
    N, M = 18, 252
    indices = None
    code = data.index.get_level_values(level=1)[0]
    if (len(kwargs.keys()) > 0):
        N = kwargs['N'] if ('N' in kwargs.keys()) else N
        M = kwargs['M'] if ('M' in kwargs.keys()) else M
        indices = kwargs['indices'].loc[(slice(None), code), :] if ('indices' in kwargs.keys()) else None

    if (len(args) > 0):
        N = args[0] if (len(args) > 0) else N
        M = args[1] if (len(args) >= 1) else M
        indices = args[2] if (len(args) >= 2) else indices

    highs = data.high.values
    lows = data.low.values

    # 斜率
    idx_RSRS_BETA = 0
    idx_RSRS_BETA_R2 = 1
    idx_RSRS_ZSCORE = 2
    idx_RSRS_ZSCORE_R2 = 3
    idx_RSRS_RIGHT = 4
    ret_rsrs_beta = np.zeros((len(data), 5),)
    with np.errstate(invalid='ignore', divide='ignore'):
        for i in range(len(highs))[N:]:
            data_high = highs[i - N:i]
            data_low = lows[i - N:i]
            X = sm.add_constant(data_low)
            model = sm.OLS(data_high, X)
            results = model.fit()

            # beta = low
            if (len(results.params) > 1):
                ret_rsrs_beta[i, idx_RSRS_BETA] = results.params[1]
            else:
                ret_rsrs_beta[i, idx_RSRS_BETA] = results.params[0]
            ret_rsrs_beta[i, idx_RSRS_BETA_R2] = results.rsquared

    # 标准分
    ret_rsrs_beta[:, idx_RSRS_ZSCORE] = rolling_zscore(ret_rsrs_beta[:, idx_RSRS_BETA], M)
    ret_rsrs_beta[:min(M, len(highs)), idx_RSRS_ZSCORE] = scs.zscore(ret_rsrs_beta[:min(M, len(highs)), idx_RSRS_BETA])

    ret_rsrs_beta = np.nan_to_num(ret_rsrs_beta, nan=0)
    with np.errstate(invalid='ignore', divide='ignore', over='ignore'):
        ret_rsrs_beta[:, idx_RSRS_ZSCORE_R2] = ret_rsrs_beta[:, idx_RSRS_ZSCORE] * ret_rsrs_beta[:, idx_RSRS_BETA_R2]

        # 右偏标准分
        ret_rsrs_beta[:, idx_RSRS_RIGHT] = ret_rsrs_beta[:, idx_RSRS_ZSCORE_R2] * ret_rsrs_beta[i, idx_RSRS_BETA]

    if (indices is None):
        return pd.DataFrame(ret_rsrs_beta,
                            index=data.index,
                            columns=[FLD.RSRS_BETA,
                                     FLD.RSRS_BETA_R2,
                                     FLD.RSRS_ZSCORE,
                                     FLD.RSRS_ZSCORE_R2,
                                     FLD.RSRS_RIGHT])
    else:
        indices = indices.reindex(columns=[*indices.columns,
                                           *[FLD.RSRS_BETA,
                                             FLD.RSRS_BETA_R2,
                                             FLD.RSRS_ZSCORE,
                                             FLD.RSRS_ZSCORE_R2,
                                             FLD.RSRS_RIGHT]])

        indices.loc[:, [FLD.RSRS_BETA,
                        FLD.RSRS_BETA_R2,
                        FLD.RSRS_ZSCORE,
                        FLD.RSRS_ZSCORE_R2,
                        FLD.RSRS_RIGHT]] = ret_rsrs_beta

        return indices
