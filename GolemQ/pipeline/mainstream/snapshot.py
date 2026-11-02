# coding: utf-8
"""Massive 快照聚合 —— **一次 groupby(bar)** 产出全部 MED/MODE/COMBO 列。

旧树对应：`GolemQ_old/analysis/chip_dist.py::calc_massive_snapshot_chip_dist`
（`groupby(level=[0]).apply(...)`，每根 bar 一行字典），再由
`calc_massive_chip_dist` 对其中若干 COMBO 列做 `calc_event_timing_lag`。

为什么这样组织
==============
「每加一列就写一个方法、再全表扫一次」会变成 N 次 500 万行扫描 —— 用户
2026-11-02 明确否掉这种做法。这里改成：

* **注册表** `SNAPSHOT_AGS`：一行一个契约列（目标名 / 源列 / 聚合方式 / 预处理）；
* **一次分组** `snapshot_frame()`：`groupby(ts).apply(aggregate_snapshot)`；
* 新增契约 = 往注册表加一行，不改循环。

目前可用的源列（来自 `*_reality_feat_60min`）：`stage`、`poly9NorMajR`、
`cost5PrzD/cost95PrzD`、以及虚拟列 `m9tCmpod`
（逐码合成的 M9T combo，见 `analysis.mainstream_timing.calc_m9t_combo`）。
缺 producer 的源列（stockScoreM15 / MAPOWER30_MAJOR / ATR_*_MAJOR / zenOLS /
MACD_CROSS_SX）等 mainstream 逐股特征栈移植后再往注册表里加。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Sequence

import numpy as np
import pandas as pd

__all__ = ['Agg', 'SNAPSHOT_AGS', 'aggregate_snapshot', 'snapshot_frame']


@dataclass(frozen=True)
class Agg:
    """一个快照契约列。

    :param target: 落库列名（沿用旧树 `MAS.*` 的名字）
    :param source: 逐股源列（虚拟列也可以）
    :param kind:  ``'median'`` / ``'mode'`` / ``'mean'`` / ``'count'``
    :param pre:   分组前对**整列**做的向量化预处理（如 `s - 1`）；None=不动
    :param drop_zero: 求 mode 前先把 0 去掉（旧树 M9T 系列的写法）
    """

    target: str
    source: str
    kind: str = 'median'
    pre: Optional[Callable[[pd.Series], pd.Series]] = None
    drop_zero: bool = False


#: 当前可算的契约列（按旧树 `MAS.*` 命名）。
SNAPSHOT_AGS: Sequence[Agg] = (
    Agg('STAGE_MODE', 'stage', 'mode'),
    Agg('POLYNOMIAL9_NORM_MAJOR_MEDIAN', 'poly9NorMajR', 'median'),
    # ⚠️ FBP_CLR_NORM(_MAJOR)_MEDIAN 已移除：`prophetClrNorm` 大量压在 1.0 上界，
    # 中位数恒为 1.0，不具备指向性（用户 2026-11-02）。
    # RETURN_COST5 = cost5/close - 1；库里 `cost5PrzD` 是 cost5/close 的比值
    Agg('RETURN_COST5_MEDIAN', 'cost5PrzD', 'median', pre=lambda s: s - 1.0),
    Agg('RETURN_COST95_MEDIAN', 'cost95PrzD', 'median', pre=lambda s: s - 1.0),
    # 逐码 M9T combo（COMPOUNDED）的众数 / 中位数
    Agg('MAGIC_NINE_TURNS_DUMMY_COMPOUNDED_MODE', 'm9tCmpod', 'mode',
        drop_zero=True),
    Agg('M9T_LAG_CMB_MED', 'm9tCmpod', 'median'),
)


def aggregate_snapshot(frame: pd.DataFrame, ags=SNAPSHOT_AGS) -> Dict[str, float]:
    """**一根 bar** 的横截面 → 全部契约列（一行字典）。"""
    out: Dict[str, float] = {}
    for ag in ags:
        if ag.source not in frame.columns:
            out[ag.target] = np.nan
            continue
        s = pd.to_numeric(frame[ag.source], errors='coerce')
        if ag.pre is not None:
            s = ag.pre(s)
        s = s.dropna()
        if ag.drop_zero:
            s = s[s != 0]
        if len(s) == 0:
            out[ag.target] = np.nan
            continue
        if ag.kind == 'median':
            out[ag.target] = float(s.median())
        elif ag.kind == 'mode':
            out[ag.target] = float(s.mode().iloc[0])
        elif ag.kind == 'mean':
            out[ag.target] = float(s.mean())
        elif ag.kind == 'count':
            out[ag.target] = float(len(s))
        else:                       # pragma: no cover 写错注册表要早炸
            raise ValueError(f'未知聚合方式: {ag.kind!r}')
    return out


def snapshot_frame(features: pd.DataFrame, ags=SNAPSHOT_AGS,
                   ts_col: str = 'ts') -> pd.DataFrame:
    """`(ts, code, 列...)` 长表 → 每根 bar 一行的契约列帧。

    >>> f = pd.DataFrame({
    ...     'ts': pd.to_datetime(['2024-01-02 10:30'] * 3),
    ...     'code': ['a', 'b', 'c'],
    ...     'poly9NorMajR': [0.2, 0.4, 0.6],
    ...     'cost5PrzD': [1.1, 1.2, 1.3],
    ...     'cost95PrzD': [1.4, 1.5, 1.6],
    ...     'stage': [1, 1, 2],
    ...     'm9tCmpod': [1, 1, -1]})
    >>> r = snapshot_frame(f)
    >>> round(float(r['POLYNOMIAL9_NORM_MAJOR_MEDIAN'].iloc[0]), 4)
    0.4
    >>> float(r['STAGE_MODE'].iloc[0])
    1.0
    """
    if features is None or len(features) == 0:
        return pd.DataFrame(columns=[a.target for a in ags])
    rows = features.groupby(ts_col, sort=True).apply(
        lambda x: aggregate_snapshot(x, ags))
    return pd.DataFrame(list(rows), index=rows.index)
