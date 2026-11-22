# coding: utf-8
"""`COST5_UPRISING_BEFORE` 的**化简移植**（用户 2026-11-21，D85）。

4.4 原式（`GolemQ_old/analysis/base.py:2309`）是一大串条件的
`Timeline_duration`，依赖 XGB echo / 1K stage / Prophet LB uprising / 周线 M9T /
ZEN OLS 等新树没有的生产者。本模块只保留**前两个 stage/MACD 分支**：

```
(RETURN_COST5 > 0.00618) & (RETURN_COST5 < 1.68)
& ((bstStgBf < max(MACD_CROSS_SX_BEFORE, MACD_CROSS_JX_BEFORE_MAJOR))
   | (bstStgBf < MACD_CROSS_SX_BEFORE_MAJOR))
```

其中 `MACD_CROSS_*` 是**零轴**交叉（旧树 `features/base.py:580-593`），
`_MAJOR` 是日线版；`bstStgBf = Timeline_duration(stage < -0.00168)`。

⚠️ 这是**化简版**，不是逐字复刻：丢掉了 XGB/1K/Prophet/周线/OLS 分支与排除项。
massive 侧聚合（众数 / 计数 / 中位数）不在本模块。

为什么不复用 `analysis/compact.py` 的同名列：
`compact.MACD_CROSS_SX_BEFORE` 是 **DEA/DIF 死叉**，与旧树这里的**MACD 零轴死叉**
不是同一个量；直接拿来会静默改变结果。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from GolemQ.analysis.timing import Timeline_duration
from GolemQ.core.constants import AKA, FIELD as FLD

__all__ = ['calc_cost5_uprising_before',
           'calc_cost5_uprising_from_stored']


def _zero_cross_before(series: pd.Series):
    """旧树 MACD 零轴交叉的 `(jx_before, sx_before)`（60min/日线同一公式）。"""
    s = pd.to_numeric(series, errors='coerce')
    prev = s.shift(1)
    jx = Timeline_duration(np.where((s > 0) & (prev < 0), 1, 0))
    sx = Timeline_duration(np.where((s < 0) & (prev > 0), 1, 0))
    return jx.astype(np.int32), sx.astype(np.int32)


def calc_cost5_uprising_before(features: pd.DataFrame,
                               macd_col: str = FLD.MACD,
                               macd_major_col: str = FLD.MACD_MAJOR,
                               stage_col: str = AKA.STAGE) -> pd.Series:
    """化简版 `cost5UprBf`（逐 bar 连续计数，条件不成立时回 0）。

    :param features: 需含 `RetCost5` / `MACD` / `macdMaj` / `stage`
        （`macdMaj` 由 `compact._derive_major` 提供；`bstStgBf` 由 `stage` 现算）。
    :returns: 与 `features` 同索引的 int32 Series；缺列时返回全 0。

    >>> import pandas as pd
    >>> f = pd.DataFrame({'RetCost5': [0.0, 0.01, 0.02, 0.02, 0.02],
    ...                   'MACD': [1, 1, -1, -1, -1],
    ...                   'macdMaj': [-1, -1, 1, 1, 1],
    ...                   'stage': [0, 0, 0, -1, -1]})
    >>> calc_cost5_uprising_before(f).tolist()
    [1, 2, 3, 0, 0]
    """
    idx = features.index
    if AKA.RETURN_COST5 not in features.columns:
        return pd.Series(np.zeros(len(idx), dtype=np.int32), index=idx)
    ret = pd.to_numeric(features[AKA.RETURN_COST5], errors='coerce')
    if (macd_col not in features.columns
            or macd_major_col not in features.columns
            or stage_col not in features.columns):
        return pd.Series(np.zeros(len(idx), dtype=np.int32), index=idx)

    _, sx = _zero_cross_before(features[macd_col])
    jx_maj, sx_maj = _zero_cross_before(features[macd_major_col])
    stage = pd.to_numeric(features[stage_col], errors='coerce')
    bst = Timeline_duration(np.where(stage < -0.00168, 1, 0)).astype(np.int32)

    cond = ((ret > 0.00618) & (ret < 1.68)
            & ((bst < np.maximum(sx, jx_maj)) | (bst < sx_maj)))
    return pd.Series(Timeline_duration(np.where(cond, 1, 0)).astype(np.int32),
                     index=idx)


def calc_cost5_uprising_from_stored(features: pd.DataFrame,
                                    day_ohlc: pd.DataFrame = None
                                    ) -> pd.Series:
    """**attach 用**：从落库列现算 `cost5UprBf`，不跑 ChipDist。

    数据全部来自库里已有的：
    * `cost5Prz`（**已还原的绝对成本价** = `cost5PrzD × as-of 收盘`）
      → ``RetCost5 = close / cost5Prz - 1``（用户 2026-11-21 指定）；
    * `stage`（60min 表）→ `bstStgBf`；
    * `close`（features 帧）与 `day_ohlc.close` → MACD / `macdMaj`。

    条件/交叉口径与 :func:`calc_cost5_uprising_before` 相同。
    """
    idx = features.index
    if ('cost5Prz' not in features.columns
            or AKA.STAGE not in features.columns
            or 'close' not in features.columns):
        return pd.Series(np.zeros(len(idx), dtype=np.int32), index=idx)

    from GolemQ.analysis.indicators import calc_macd_qa
    cost5 = pd.to_numeric(features['cost5Prz'], errors='coerce')
    close = pd.to_numeric(features['close'], errors='coerce')
    ret = close / cost5 - 1.0
    _, _, macd = calc_macd_qa(pd.to_numeric(features['close'],
                                            errors='coerce'))
    _, sx = _zero_cross_before(pd.Series(np.asarray(macd), index=idx))

    if day_ohlc is not None and len(day_ohlc):
        from GolemQ.analysis.compact import _map_causal
        _, _, dmacd = calc_macd_qa(pd.to_numeric(day_ohlc['close'],
                                                 errors='coerce'))
        dts = pd.DatetimeIndex(pd.to_datetime(
            day_ohlc.index.get_level_values(0)))
        day_series = pd.Series(np.asarray(dmacd), index=dts)
        day_series = day_series[~day_series.index.duplicated(keep='last')]
        macd_maj = _map_causal(day_series, idx)
    else:
        macd_maj = pd.Series(np.nan, index=idx)
    jx_maj, sx_maj = _zero_cross_before(macd_maj)

    stage = pd.to_numeric(features[AKA.STAGE], errors='coerce')
    bst = Timeline_duration(np.where(stage < -0.00168, 1, 0)).astype(np.int32)
    cond = ((ret > 0.00618) & (ret < 1.68)
            & ((bst < np.maximum(sx, jx_maj)) | (bst < sx_maj)))
    return pd.Series(Timeline_duration(np.where(cond, 1, 0)).astype(np.int32),
                     index=idx)
