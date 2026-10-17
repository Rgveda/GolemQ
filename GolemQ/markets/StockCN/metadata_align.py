# coding:utf-8
"""把 `stock_reality_feat_day` 的列**对齐到任意频率的 K 线**。

从哪来（以及**为什么不是照搬**）
================================
旧树 `cli/__main__.py:426` 的 `--stock-min-aligned` 分支调
`markets/StockCN/align.py::stock_min_aligned`。勘查结论：**照搬不可能，也不该照搬** ——

* 名字**名不副实**：全文**没有任何分钟数据**，`align.py` 里对齐的是**日线**
  （`(date, code)` 索引）。「股票分钟对齐功能」那句注释误导了整轮。
* 它依赖 **QUANTAXIS**（`QA_fetch_stock_list` / `QA_fetch_stock_day_adv`）、
  **xtquant**（`GQ_featch_stock_valuation_from_xtquant`，而 **MiniQMT 自 2026-10-01
  已停服**）与 **4.4 的 `stock_valuation`**（已随 4.4 停用）。
* 它同时**写** `stock_ranking` / `stock_valuation` —— 而新树那两列已经在
  `stock_reality_feat_day` 里了。

⇒ 用户 2026-10-11 定：**保留「对齐」这个能力，数据源换成 8.3 自己的
`stock_reality_feat_day`**。本模块就是那个能力，且**只读**（写元数据是
`--moneyflow` 与 `refresh_turnover` 的事）。

⚠️ **为什么落在 `markets/StockCN/` 而不叫 `align.py`**
====================================================
叫 `align.py` 会让人以为它是旧文件原样搬运（它不是）。放这里是因为它读的是
StockCN 的表；**不是** `analysis/` —— 它不产生特征，只是把已有列对齐到时间轴。

对齐键是 `(date, code)`，**不是 timestamp**
==========================================
分钟线一个交易日有 240 根 bar，而那天的元数据只有一行 ⇒ 必须**先按日折叠再连**。
⚠️ 这正是旧实现与「分钟对齐」这个名字之间的裂缝：它当年只对日线做，所以从没
暴露这个问题。本模块接受**任意频率**。

⚠️ `date_stamp` 一律用 :func:`metadata_save.day_stamp` 重算，**不取源端**
—— 源端三种口径并存（见 `DECISIONS.md` D34），照搬会差 8 小时而**静默查空**。
"""
from __future__ import annotations

import pandas as pd

from .metadata_save import REALITY_FEAT_DAY_COLL, _db, day_date, day_stamp

__all__ = ['align_metadata_frame', 'align_metadata_to_kline',
           'METADATA_ALIGN_COLUMNS']

#: 默认对齐哪几列（`None` = 表里除公共键外的**全部**列）。
#: ⚠️ 顺序即输出列序；想全要就传 `None`。
METADATA_ALIGN_COLUMNS = ('TurnoverRate', 'turnover', 'PERation', 'ZLCB', 'JGCYD',
                          'TotalScore', 'RankingUp', 'Ranking', 'Focus',
                          'ChangePercent', 'THS_Inflow', 'THS_Outflow',
                          'THS_Net', 'THS_Amount')

#: 表里的**公共键**（对齐时要排除，它们不是特征）
_BASE_KEYS = ('code', 'date', 'date_stamp', 'datetime', 'ts', 'created_at', '_id')


def _kline_dates(kline: pd.DataFrame) -> pd.Series:
    """K 线索引的 level 0 → `'YYYY-MM-DD'`。

    ⚠️ 分钟线的 level 0 是 **Timestamp**（`2026-10-09 10:30`），日线是当天零点。
    两者都取日期部分 —— **这就是「按日折叠」那一步**。
    """
    idx = kline.index
    level0 = idx.get_level_values(0) if isinstance(idx, pd.MultiIndex) else idx
    return pd.Series(level0, index=kline.index).map(day_date)


def align_metadata_frame(kline: pd.DataFrame, meta: pd.DataFrame,
                         columns=None) -> pd.DataFrame:
    """**纯函数**：把 `meta`（索引 `(date, code)`）对齐到 `kline` 的每一行。

    :param kline: 任意频率的 K 线帧，索引 `(时间, code)`
    :param meta: 元数据帧，索引 `(date_str, code)`，`date_str` 是 `'YYYY-MM-DD'`
    :param columns: 要哪些列；``None`` = `meta` 里除公共键外的**全部**
    :return: 与 `kline` **同索引同长度**的帧。**对不上的行是 NaN**，不是错位。

    >>> import pandas as pd
    >>> idx = pd.MultiIndex.from_tuples(
    ...     [(pd.Timestamp('2026-10-09 10:30'), '600519'),
    ...      (pd.Timestamp('2026-10-09 11:30'), '600519'),
    ...      (pd.Timestamp('2026-10-12 10:30'), '600519')], names=['ts', 'code'])
    >>> kline = pd.DataFrame({'close': [1200.0, 1205.0, 1210.0]}, index=idx)
    >>> midx = pd.MultiIndex.from_tuples(
    ...     [('2026-10-09', '600519'), ('2026-10-12', '600519')], names=['date', 'code'])
    >>> meta = pd.DataFrame({'TurnoverRate': [0.0028, 0.0031]}, index=midx)
    >>> out = align_metadata_frame(kline, meta)
    >>> out['TurnoverRate'].tolist()        # 同一天的两根 bar 拿到**同一个值**
    [0.0028, 0.0028, 0.0031]
    >>> list(out.index) == list(kline.index)    # 索引与长度都不变
    True

    对不上的日子留 NaN（**不是**前向填充 —— 那会凭空造出当天并不存在的元数据）：

    >>> align_metadata_frame(kline, meta.iloc[:1])['TurnoverRate'].tolist()
    [0.0028, 0.0028, nan]
    """
    cols = list(columns) if columns is not None else [
        c for c in meta.columns if c not in _BASE_KEYS]
    out = pd.DataFrame(index=kline.index, columns=cols, dtype='float64')
    if meta is None or meta.empty or not cols:
        return out

    key = pd.MultiIndex.from_arrays([_kline_dates(kline).values,
                                     kline.index.get_level_values(1)
                                     if isinstance(kline.index, pd.MultiIndex)
                                     else kline.index])
    lookup = {}
    for (day, code), row in meta.iterrows():
        lookup[(str(day), str(code))] = row
    for i, k in enumerate(key):
        row = lookup.get((str(k[0]), str(k[1])))
        if row is None:
            continue
        for c in cols:
            if c in row.index:
                out.iloc[i, out.columns.get_loc(c)] = row[c]
    return out


def align_metadata_to_kline(kline: pd.DataFrame, columns=None, database=None
                            ) -> pd.DataFrame:
    """读 `stock_reality_feat_day` 并把它对齐到 `kline`（见 :func:`align_metadata_frame`）。

    ⚠️ **只读**。不写库、不回填 —— 缺的日子就是缺（NaN），
    要补是 `--moneyflow` / `refresh_turnover` 的事。

    :param columns: ``None`` = :data:`METADATA_ALIGN_COLUMNS` 里**表里真有**的那些
    :return: 与 `kline` 同索引同长度的帧；库里一行都没有时返回**空的同形帧**
    """
    dates = sorted(set(_kline_dates(kline).dropna()))
    if not dates:
        return pd.DataFrame(index=kline.index, dtype='float64')
    codes = sorted(set(kline.index.get_level_values(1))) \
        if isinstance(kline.index, pd.MultiIndex) else None
    # ⚠️ 用 date_stamp 划范围是**这张表内部一致**的口径（D34 已把它重算成 8.3 口径），
    # 所以这里能安全地走索引；而搬 4.4 那次**不能**（源端三种口径混着）。
    q = {'date_stamp': {'$gte': day_stamp(dates[0]), '$lte': day_stamp(dates[-1])}}
    if codes:
        q['code'] = {'$in': [str(c) for c in codes]}
    want = list(columns) if columns is not None else list(METADATA_ALIGN_COLUMNS)
    proj = {'_id': 0, 'date': 1, 'code': 1}
    proj.update({c: 1 for c in want})
    rows = list(_db()[REALITY_FEAT_DAY_COLL].find(q, proj))
    if not rows:
        return pd.DataFrame(index=kline.index,
                            columns=[c for c in want], dtype='float64')
    meta = pd.DataFrame(rows).set_index(['date', 'code'])
    have = [c for c in want if c in meta.columns]
    return align_metadata_frame(kline, meta, columns=have)
