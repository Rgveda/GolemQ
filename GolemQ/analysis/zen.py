# coding: utf-8
"""缠论 Zen 特征（`zDLag*` / `zPLag*`）—— **基于新树 czsc 0.7.10 的因果版本**。

为什么不搬老树 `KlineAnalyze` 0.53
==================================
老树 `GolemQ_old/czsc/analyze.py`（1,286 行）自带一套笔/线段实现（0.53 时代），
`calc_zen_chopstickss` 依赖它的 `to_df()`。用户 2026-10-19 明确：**这个应被解耦、剔除**，
换成新树已装的 **czsc 0.7.10**。

本模块只保留老树真正用到的两段语义（逐字，不搬整类）：

1. **笔端点** → `ZEN_DASH`：`bi > 0 & fx_mark == 'd'` ⇒ +1，`... == 'g'` ⇒ -1；
2. **线段端点** → `ZEN_PEAK`：同一规则作用在 `xd` 上。
   线段算法（`get_potential_xd` + `_update_xd_list`，约 60 行）**单独抽出**，
   输入是 czsc 的笔端点列表，不再依赖 `KlineAnalyze`。

⚠️ **与老树的已知差异（设计变更，不是 bug）**：笔/线段由 czsc 0.7.10 生成，
与 0.53 的 `KlineAnalyze` 不保证逐笔相同 ⇒ `zDLag*`/`zPLag*` 可能在某些端点
相差 1 根或更早/更晚。对拍时归为 `design/czsc-version`，不要求逐值全等
（见 `docs/worklog` 与计划）。
"""
from __future__ import annotations

from typing import Iterable, List

import numpy as np
import pandas as pd

from czsc import CZSC
from czsc.objects import Freq, RawBar

from GolemQ.analysis.timing import Timeline_duration, calc_event_timing_lag
from GolemQ.core.constants import AKA, FIELD as FLD

__all__ = ['czsc_bi_points', 'xd_points_from_bi', 'calc_zen_timing_lag']

_MARK_D = ('D', 'd', '底分型')
_MARK_G = ('G', 'g', '顶分型')


def _mark_str(mark) -> str:
    name = getattr(mark, 'name', None)
    if name is None:
        name = str(mark)
    if name in _MARK_D:
        return 'd'
    if name in _MARK_G:
        return 'g'
    return 'o'


def _freq_of(ohlc: pd.DataFrame):
    """从索引间距推断 czsc 频率（15min / 60min / day）。"""
    try:
        ts = pd.DatetimeIndex(pd.to_datetime(ohlc.index.get_level_values(0)))
        gaps = ts.to_series().diff().dropna()
        if len(gaps) == 0:
            return Freq.F60
        med = gaps.median()
        if med <= pd.Timedelta(minutes=20):
            return Freq.F15
        if med <= pd.Timedelta(hours=2):
            return Freq.F60
        return Freq.D
    except Exception:           # noqa: BLE001 推断失败按 60min
        return Freq.F60


def _bars_of(ohlc: pd.DataFrame, code: str, freq) -> List[RawBar]:
    bars = []
    for i, (idx, row) in enumerate(ohlc.iterrows()):
        dt = idx[0] if isinstance(idx, tuple) else idx
        bars.append(RawBar(
            symbol=code, dt=dt, id=i, freq=freq,
            open=float(row[AKA.OPEN]), close=float(row[AKA.CLOSE]),
            high=float(row[AKA.HIGH]), low=float(row[AKA.LOW]),
            vol=float(row.get(AKA.VOLUME, 0) or 0),
            amount=float(row.get('amount', 0) or 0)))
    return bars


def czsc_bi_points(ohlc: pd.DataFrame, market_type=None) -> List[dict]:
    """用 czsc 0.7.10 生成**笔端点**列表（老树 `bi_points` 的等价形状）。

    每个点：``{'dt': Timestamp, 'fx_mark': 'd'|'g', 'bi': 价格}``。
    相邻笔共享端点，按 `dt` 去重（保留首个）。
    """
    if ohlc is None or len(ohlc) == 0:
        return []
    code = str(ohlc.index.get_level_values(1)[0])
    freq = _freq_of(ohlc)
    bars = _bars_of(ohlc, code, freq)
    czsc = CZSC(bars, max_bi_count=max(5000, len(bars) + 100), verbose=False)

    points = []
    seen = set()
    for bi in czsc.bi_list:
        for fx in (bi.fx_a, bi.fx_b):
            if fx.dt in seen:
                continue
            seen.add(fx.dt)
            points.append({'dt': fx.dt, 'fx_mark': _mark_str(fx.mark),
                           'bi': float(fx.fx)})
    points.sort(key=lambda x: x['dt'])
    return points


def _get_potential_xd(bi_points: List[dict]) -> List[dict]:
    """老树 `get_potential_xd` 逐字（输入改为 czsc 端点列表）。"""
    xd_p = []
    bi_d = [x for x in bi_points if x['fx_mark'] == 'd']
    bi_g = [x for x in bi_points if x['fx_mark'] == 'g']
    for i in range(1, len(bi_d) - 1):
        d1, d2, d3 = bi_d[i - 1: i + 2]
        if d1['bi'] > d2['bi'] < d3['bi']:
            xd_p.append(d2)
    for j in range(1, len(bi_g) - 1):
        g1, g2, g3 = bi_g[j - 1: j + 2]
        if g1['bi'] < g2['bi'] > g3['bi']:
            xd_p.append(g2)
    return sorted(xd_p, key=lambda x: x['dt'])


def xd_points_from_bi(bi_points: List[dict]) -> List[dict]:
    """老树 `KlineAnalyze._update_xd_list` 的纯函数版（逐字语义）。

    返回线段端点列表，元素含 ``dt/fx_mark/xd``（`xd` 为价格）。
    """
    if len(bi_points) < 4:
        return []

    xd_list = []
    for i in range(3):
        xd = dict(bi_points[i])
        xd['xd'] = xd.pop('bi')
        xd_list.append(xd)

    right_bi = [x for x in bi_points if x['dt'] >= xd_list[-1]['dt']]
    for xp in _get_potential_xd(right_bi):
        xd = dict(xp)
        xd['xd'] = xd.pop('bi')
        last_xd = xd_list[-1]
        if last_xd['fx_mark'] == xd['fx_mark']:
            if ((last_xd['fx_mark'] == 'd' and last_xd['xd'] > xd['xd']) or
                    (last_xd['fx_mark'] == 'g' and last_xd['xd'] < xd['xd'])):
                xd_list[-1] = xd
        else:
            if ((last_xd['fx_mark'] == 'd' and last_xd['xd'] > xd['xd']) or
                    (last_xd['fx_mark'] == 'g' and last_xd['xd'] < xd['xd'])):
                continue
            inside = [x for x in right_bi if last_xd['dt'] <= x['dt'] <= xd['dt']]
            if len(inside) < 4:
                continue
            xd_list.append(xd)
    return xd_list


def _states_from_points(index, points, value_key: str) -> tuple:
    """把端点列表还原成与 `index` 等长的 `(fx_mark, present)` 数组。"""
    n = len(index)
    marks = np.full(n, 'o', dtype=object)
    present = np.zeros(n, dtype=bool)
    pos = {}
    for i, t in enumerate(index.get_level_values(0)):
        pos.setdefault(t, i)
    for p in points:
        i = pos.get(p['dt'])
        if i is None:
            continue
        marks[i] = p['fx_mark']
        present[i] = p[value_key] > 0
    return marks, present


def calc_zen_timing_lag(ohlc: pd.DataFrame, features: pd.DataFrame = None,
                        market_type=None) -> pd.DataFrame:
    """`ZEN_DASH_TIMING_LAG` / `ZEN_PEAK_TIMING_LAG`（老树 `calc_zen_chopstickss` 的核）。

    逐字保留老树的最后两段：

    * `ZEN_DASH = (bi 存在 & fx_mark=='d') ? 1 : (bi 存在 & fx_mark=='g') ? -1 : 0`；
    * `ZEN_PEAK` 同式，作用在 `xd` 上；
    * 两者各自 `Timeline_duration` 后 `calc_event_timing_lag(where(jx < sx, 1, -1))`。

    :param ohlc: OHLC 帧（索引 `(dt, code)`），**调用方**负责先做因果切片。
    :param features: 承载列的帧；`None` ⇒ 新建。
    :returns: `features`（就地加两列）。
    """
    if features is None:
        features = pd.DataFrame(index=ohlc.index)
    if ohlc is None or len(ohlc) == 0:
        features[FLD.ZEN_DASH_TIMING_LAG] = np.int16(-1)
        features[FLD.ZEN_PEAK_TIMING_LAG] = np.int16(-1)
        return features

    bi_points = czsc_bi_points(ohlc, market_type=market_type)
    xd_points = xd_points_from_bi(bi_points)

    bi_marks, bi_present = _states_from_points(ohlc.index, bi_points, 'bi')
    xd_marks, xd_present = _states_from_points(ohlc.index, xd_points, 'xd')

    def _lag(marks, present):
        state = np.where(present & (marks == 'd'), 1,
                         np.where(present & (marks == 'g'), -1, 0))
        jx = Timeline_duration(np.where(state > 0, 1, 0))
        sx = Timeline_duration(np.where(state < 0, 1, 0))
        return calc_event_timing_lag(np.where(jx < sx, 1, -1)).astype(np.int16)

    features[FLD.ZEN_DASH_TIMING_LAG] = _lag(bi_marks, bi_present)
    features[FLD.ZEN_PEAK_TIMING_LAG] = _lag(xd_marks, xd_present)
    return features
