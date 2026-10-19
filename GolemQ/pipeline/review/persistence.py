# coding: utf-8
"""持久化检查 —— 以**目标表里已有哪些 review 行**为准，决定还算/还写哪些 bar。

与老树 `scribe/persistence.py::dataloader_persistence_check` 的对应关系
=====================================================================
老树按 `persistence_features` 里的已落库列算 `persistence_ratio` / `masked`，
只补缺失日期；本模块把它收敛成一件事：

1. 从目标 `*_reality_feat_*` 表读回该 code 的现有行（**只投影契约列**）；
2. 一行里**任一契约列非空** ⇒ 视为「这条 review 行已持久化」；
   ⚠️ 4.4 迁移来的 CV-only 行（`ClsMaj` 等）不算 —— 它们的 review 列全空；
3. 返回**源帧里还没持久化的目标时间**。调用方只对这些目标计算/写入，
   因而**不重复计算、不重复写入**。

为什么不用 `pipeline_watermarks` 做判据
=====================================
水位是**侧表**：手工删行、迁移、别的写口都会让它与表内容分叉。
本模块直接看表；水位只保留作观测（`status/error`），不参与目标选择。
"""
from __future__ import annotations

from typing import Iterable, Sequence

import pandas as pd

from GolemQ.pipeline.review.contract import FAMILY_COLUMNS


def _norm_keys(index, family: str) -> pd.DatetimeIndex:
    ts = pd.DatetimeIndex(pd.to_datetime(index))
    if family == 'day':
        return ts.normalize()
    return ts


def persisted_keys(code: str, family: str, market_type, db,
                   columns: Sequence[str] = None,
                   min_columns: int = 1,
                   start=None, end=None) -> set:
    """读回该 code 在目标表里**已持久化 review 行**的时间键集合。

    :param columns: 参与巡查的列；``None`` ⇒ 该族契约的**全部列**。
    :param min_columns: 一行里至少多少列非空才算「已持久化」（默认 1）。
        传更多的列要求更严格（例如要求 `zDLagR/zPLagR` 两列都在）。
    """
    from GolemQ.markets.StockCN.reality_save import fetch_reality_feat

    cols = list(columns or FAMILY_COLUMNS[family])
    got = fetch_reality_feat([code], start, end, family=family,
                             market_type=market_type, db=db,
                             columns=cols, restore_price=False)
    if got is None or len(got) == 0:
        return set()
    present = [c for c in cols if c in got.columns]
    if not present:
        return set()
    need = max(1, int(min_columns or 1))
    mask = got[present].notna().sum(axis=1).to_numpy() >= need
    keys = _norm_keys(got.index.get_level_values(0), family)
    return set(keys[mask])


def missing_targets(source_index, code: str, family: str, market_type, db,
                    force: bool = False, from_date: str = '',
                    recheck_days: int = 0,
                    columns: Sequence[str] = None,
                    min_columns: int = 1) -> pd.DatetimeIndex:
    """源帧里**尚未持久化**的目标时间（升序）。

    :param source_index: 源帧的全部时间戳（已按 `as_of` 截断）
    :param force: 忽略已持久化集合，全部重算
    :param from_date: 只算这一天之后（含下界之后）
    :param recheck_days: 把最近 N 个自然日内已持久化的行也纳入重算（修补迟到数据）
    :param columns: 参与巡查的列；``None`` ⇒ 该族契约的全部列
    :param min_columns: 一行至少多少列非空才算已持久化（见 :func:`persisted_keys`）
    """
    src = pd.DatetimeIndex(pd.to_datetime(source_index))
    if len(src) == 0:
        return src
    if force:
        return src

    start = str(src.min().date())
    end = str(src.max().date())
    have = persisted_keys(code, family, market_type, db,
                          columns=columns, min_columns=min_columns,
                          start=start, end=end)

    keep = []
    for t in src:
        k = t.normalize() if family == 'day' else t
        if from_date and k <= pd.Timestamp(from_date):
            continue
        if k not in have:
            keep.append(t)
    targets = pd.DatetimeIndex(keep)

    if recheck_days and len(have):
        cutoff = max(have) - pd.Timedelta(days=int(recheck_days))
        tail = pd.DatetimeIndex([t for t in src
                                 if (t.normalize() if family == 'day' else t) >= cutoff])
        targets = targets.union(tail)
    return pd.DatetimeIndex(sorted(set(targets)))
