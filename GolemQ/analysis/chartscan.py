# coding: utf-8
"""ChartScan 三时钟 → 「上升主干」掩码与时序。

用户 2026-11-25 的算子
======================
``CHARTSCAN_UPRISING_TRUNK_TIMING_LAG``（`ChrtScanUprTukLag`）

.. code-block:: python

    calc_event_timing_lag(np.where(
        (ChartScan 60min Buy < Leadout Maj & < 日线 Sell)
        | (ChartScan 日线 Buy < Leadout Maj & < 日线 Sell), 1, -1))

* **ChartScan `{fam} Buy`** = 距最近一次 `Cls{fam} == 'Buy'` 的 **bar 数**
  （事件时间取 `Coord{fam}`，单位毫秒；见 :func:`cs_class_before`）；
* **Leadout Maj** = 日线 MAJOR 的 `MsLeadoutBfMaj` 按 `_map_causal` 映射到 60min
  （盘中用上一交易日、15:00 用当日）；
* 两条 `Buy` 各判一次，**取或**；再取 `calc_event_timing_lag`
  ⇒ **`+k` = 连续命中 k 根、`-k` = 连续未命中 k 根**（不会出现 0）。

来源
====
掩码本身原先只活在示例页 `examples/_app_common.py::add_chartscan_buy_bands`
（画「Buy < Leadout Maj & < 日线 Sell」的全高带）。这里把它的**判定**抽成纯函数，
示例页与 mainstream 共用一份 —— 平行实现只会分叉（CLAUDE.md 四问第 1 条）。

⚠️ **无未来函数**（用户 2026-11-26）：reality 行里的 `Coordinates` 是**框所在的
历史 bar**，普遍比该行的扫描时刻早 2~12 天。若拿它当事件时间，等于把「扫描时刻
才知道」的检出倒填回过去 —— 从框坐标到扫描时刻之间那段 bar 会用到当时不存在的信息。
本模块一律用**「这个框第一次被扫到」的行时间**当事件时间（按 `Coordinates` 去重），
`Coordinates` 只用来认框；日线族行按其 key（日期）记成**当日 15:00**（收盘后扫描）。

⚠️ **不造数**：ChartScan 是事件型稀疏数据。掩码的输入（`Leadout Maj` / 日线
`Sell` / 任一族的 `Buy`）还没就绪的 bar 一律 **NaN** —— 否则
`calc_event_timing_lag` 会在没有 ChartScan 数据的头部编造一串「连续命中/未命中」。
与 `pipeline/mainstream/pipeline.py::snapshot_contracts` 里对 M9T combo 的处理同源。
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from GolemQ.analysis.timing import calc_event_timing_lag
from GolemQ.models.alias import LTT

__all__ = ['CHARTSCAN_BUY_FAMILIES', 'calc_chartscan_uprising_trunk_lag',
           'chartscan_buy_mask', 'cs_class_before', 'leadout_major_distance']

#: 参与「上升主干」判定的两个 ChartScan 时钟（60min / 日线）。
#: 值就是 attach 后列名的中缀：`Cls60`/`Coord60`、`ClsDay`/`CoordDay`。
CHARTSCAN_BUY_FAMILIES = ('60', 'Day')


def _ts_index(index) -> pd.DatetimeIndex:
    """任意索引（含 `(ts, code)` 两层）→ `DatetimeIndex`。"""
    if isinstance(index, pd.MultiIndex):
        return pd.DatetimeIndex(pd.to_datetime(index.get_level_values(0)))
    return pd.DatetimeIndex(pd.to_datetime(index))


def _distance_from_events(times, idx: pd.DatetimeIndex
                          ) -> Optional[np.ndarray]:
    """**事件时间戳** → 每个 bar 距最近一次事件的 bar 数（此前无事件 ⇒ NaN）。"""
    if times is None or len(times) == 0:
        return None
    ev = np.searchsorted(idx.values, np.asarray(times), side='right') - 1
    ev = np.unique(ev[ev >= 0])
    if len(ev) == 0:
        return None
    last = np.searchsorted(ev, np.arange(len(idx)), side='right') - 1
    out = np.full(len(idx), np.nan)
    ok = last >= 0
    out[ok] = np.arange(len(idx))[ok] - ev[last[ok]]
    return out


def cs_class_before(feats: Optional[pd.DataFrame], index,
                    family: str, cls: str = 'Buy',
                    row_times=None) -> Optional[np.ndarray]:
    """距最近一次 ChartScan `cls`（``'Buy'`` / ``'Sell'``）的 bar 数。

    ⚠️ **事件时间 = 「这个框第一次被扫到」的时刻，不是框坐标**（用户 2026-11-26）。
    reality 行里的 `Coord{fam}` 是**框所在的历史 bar**，普遍比扫描时刻早 2~12 天；
    拿它当事件时间，等于把一个「扫描时刻才知道」的检出倒填回过去 ⇒ **未来函数**。
    这里按 `Coord` 去重：同一个框只在它**首次出现**的那次扫描上算一次事件。

    :param feats: 含 `Cls{fam}`/`Coord{fam}` 的帧（`attach_reality_feats` 之后，
        或 `fetch_reality_feat` 的原始帧）。**索引即扫描行时间**。
    :param index: 目标 bar 的时间索引（`DatetimeIndex` 或 `(ts, code)`）。
    :param family: ``'60'`` / ``'Day'`` / ``'15'``。
    :param cls: ``'Buy'`` 或 ``'Sell'``。
    :param row_times: 覆盖行时间（与 ``feats`` 等长 / 可下标）。``None`` ⇒ 用
        ``feats.index`` 的 level 0。
    :returns: 与 ``index`` 等长的 float 数组；**此前没有该事件 ⇒ NaN**。
        缺列 / 没有该事件 ⇒ ``None``（调用方据此判「不可用」，别再猜）。

    >>> import pandas as pd
    >>> ts = pd.DatetimeIndex(['2024-01-02 10:30', '2024-01-02 15:00'])
    >>> f = pd.DataFrame({'Cls60': ['', 'Buy'],
    ...                   'Coord60': [None, pd.Timestamp('2024-01-02 10:30').value // 10**6]},
    ...                  index=ts)
    >>> cs_class_before(f, ts, '60').tolist()
    [nan, 0.0]
    >>> # ↑ 15:00 那次扫描才报出「10:30 的框」⇒ 10:30 那根**还不知道**（不是 0）
    """
    col, ccol = f'Cls{family}', f'Coord{family}'
    if feats is None or col not in feats.columns or ccol not in feats.columns:
        return None
    sub = feats[[col, ccol]].dropna(subset=[col])
    ev_rows = sub[sub[col] == cls]
    if ev_rows.empty:
        return None
    _coord = pd.to_datetime(ev_rows[ccol].astype(float), unit='ms')
    if row_times is None:
        _seen = _ts_index(ev_rows.index)
    else:
        _rt = pd.Series(pd.to_datetime(np.asarray(row_times)),
                        index=feats.index)
        _seen = pd.DatetimeIndex(_rt.reindex(ev_rows.index).to_numpy())
    df = pd.DataFrame({'coord': _coord.to_numpy(), 'seen': _seen.to_numpy()})
    df = df.dropna()
    if df.empty:
        return None
    # 同一个框只看**首次出现**的扫描时间
    first_seen = df.groupby('coord')['seen'].min().sort_values()
    return _distance_from_events(first_seen.to_numpy(), _ts_index(index))


def leadout_major_distance(ms_major: Optional[pd.DataFrame], index
                           ) -> Optional[np.ndarray]:
    """日线 MAJOR 的 `MsLeadoutBfMaj` → **60min bar 数**距离。

    用户 2026-11-26 的口径：日线量**映射到 60min 后重新按 bar 数**算，不能直接拿
    日线的 `Timeline_duration`（单位是「几根日线」）去和 60min 的 `buy60`
    （单位「几根 60min」）比较 —— 同一段时间在两者上差约 4 倍，会让
    `buy < leadout` 长期假成立。

    事件日 = `MsLeadoutBfMaj < 1e-12`（该实现里事件当天就是 0），与
    `BarChartScan` 日线族**同一套因果映射**（事件日期 → 前一根已完成 bar 起算）。

    :returns: 与 ``index`` 等长的 float 数组；缺列 / 无事件 ⇒ ``None``。
    """
    if ms_major is None or len(ms_major) == 0:
        return None
    col = LTT.MAINSTREAM_LEADOUT_BEFORE_MAJOR
    if col not in ms_major.columns:
        return None
    s = pd.to_numeric(ms_major[col], errors='coerce')
    dates = _ts_index(s.index)[np.asarray(s.to_numpy(), dtype=float) < 1e-12]
    return _distance_from_events(dates.values, _ts_index(index))


def _mask_and_ready(feats, index, leadout_major):
    """内部：``(掩码 bool 数组 | None, 「输入已就绪」bool 数组)``。"""
    n = len(index)
    if leadout_major is None:
        return None, np.zeros(n, dtype=bool)
    lead = np.asarray(leadout_major, dtype=float)
    day_sell = cs_class_before(feats, index, 'Day', 'Sell')
    if day_sell is None:
        return None, np.zeros(n, dtype=bool)
    mask = np.zeros(n, dtype=bool)
    buy_any = np.zeros(n, dtype=bool)
    got = False
    for fam in CHARTSCAN_BUY_FAMILIES:
        buy = cs_class_before(feats, index, fam, 'Buy')
        if buy is None:
            continue
        got = True
        buy_any |= np.isfinite(buy)
        mask |= (np.isfinite(buy) & np.isfinite(lead) & np.isfinite(day_sell)
                 & (buy < lead) & (buy < day_sell))
    if not got:
        return None, np.zeros(n, dtype=bool)
    ready = np.isfinite(lead) & np.isfinite(day_sell) & buy_any
    return mask, ready


def chartscan_buy_mask(feats: Optional[pd.DataFrame], index,
                       leadout_major) -> Optional[np.ndarray]:
    """用户 2026-11-25 的**上升主干掩码**（bool 数组）。

    ``(60min Buy < LeadoutMaj & < DaySell) | (Day Buy < LeadoutMaj & < DaySell)``

    :param leadout_major: 与 ``index`` **等长**的 `MsLeadoutBfMaj` 值
        （调用方负责按 `_map_causal` 映射；本函数只做比较）。
    :returns: bool 数组；缺列 / 两支 Buy 都取不到 ⇒ ``None``。
    """
    return _mask_and_ready(feats, index, leadout_major)[0]


def calc_chartscan_uprising_trunk_lag(feats: Optional[pd.DataFrame], index,
                                      leadout_major) -> pd.Series:
    """``calc_event_timing_lag(np.where(mask, 1, -1))`` → `ChrtScanUprTukLag`。

    ``+k`` = 连续命中 k 根（ChartScan 买点比派发 Maj 与日线 Sell 都更近）；
    ``-k`` = 连续未命中 k 根（**不会出现 0**，用户 2026-11-26 更正算子）。

    掩码输入未就绪的 bar（ChartScan 首事件之前 / 日线 Sell 之前 / 缺
    `Leadout Maj`）**保持 NaN** —— 见模块 docstring 的「不造数」。

    缺 ChartScan 列 ⇒ 全 NaN 的 Series（不是空表、也不抛）。

    >>> import pandas as pd
    >>> ts = pd.DatetimeIndex(['2024-01-02 10:30', '2024-01-02 11:30',
    ...                        '2024-01-02 14:00', '2024-01-02 15:00'])
    >>> ms = lambda t: pd.Timestamp(t).value // 10**6
    >>> f = pd.DataFrame({'Cls60': ['', 'Buy', '', ''],
    ...                   'Coord60': [None, ms('2024-01-02 11:30'), None, None],
    ...                   'ClsDay': ['Sell', '', '', ''],
    ...                   'CoordDay': [ms('2024-01-02 10:30'), None, None, None]},
    ...                  index=ts)
    >>> calc_chartscan_uprising_trunk_lag(f, ts, [1.0, 2.0, 2.0, 2.0]).tolist()
    [nan, 1.0, 2.0, -1.0]
    """
    name = LTT.CHARTSCAN_UPRISING_TRUNK_TIMING_LAG
    mask, ready = _mask_and_ready(feats, index, leadout_major)
    if mask is None:
        return pd.Series(np.nan, index=index, name=name, dtype=float)
    lag = calc_event_timing_lag(np.where(mask, 1, -1)).astype(float)
    lag[~ready] = np.nan
    return pd.Series(lag, index=index, name=name)
