# coding: utf-8
"""多周期买点共振 —— 基础信号 `QuadMsUprBf*` + 权重 `QuadMsCredCpd`。

用户 2026-11-24 定名
====================
* **基础信号**（三周期同口径）：15min → `QuadMsUprBfMin`，
  60min → `QuadMsUprBf`，day → `QuadMsUprBfMaj`
  （全称 `QUADRANT_MAINSTREAM_UPRISING_BEFORE`，后缀 `_MINOR` / `_MAJOR`）。
* **权重**：`QuadMsCredCpd`（全称 `QUADRANT_MAINSTREAM_CREDIT_COMPOUNDED`）。

基础条件（三个周期各算一遍）：

1. `BS_BUY_BEFORE < BS_SELL_BEFORE`（买点在卖点之后/更快）
2. `QUAD_1..4` 中**恰好一个**为 1
3. `MACD > 0`（QUANTAXIS 口径柱）
4. `(BS_BUY_BEFORE + 9) >= MACD_CROSS_JX_BEFORE`
5. `BS_BUY_BEFORE < MACD_CROSS_SX_BEFORE`
6. `DEA_ZERO_TIMING_LAG < BS_BUY_BEFORE`

权重：

* 基础：15min = **1**，60min = **2**，day = **3**；
* 该周期当天任一命中 bar 同时满足
  `Mainstream_leadin_Before < MACD_CROSS_SX_BEFORE` 再 **+1**（每周期每天至多一次）；
* 三周期**按自然日**相加；`>= 4` 视为「共振」。

映射到 60min 沿用 `compact._map_causal` 的因果口径：
**盘中用上一交易日、15:00 用当日**。

⚠️ 8.3 没落 `MACD_CROSS_JX_BEFORE`，这里按 `CROSS(DIF, DEA)` 的镜像现算
（`MACD_CROSS_SX_BEFORE` 是 `CROSS(DEA, DIF)`，见 `compact._base_frame`）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from GolemQ.analysis.compact import _map_causal
from GolemQ.analysis.indicators import calc_macd_qa, cross_above
from GolemQ.analysis.quadrant_jit import (calc_bs_signals,
                                          calc_four_quadrant_boost,
                                          calc_mainstream_feats)
from GolemQ.analysis.timing import (Timeline_duration,
                                    calc_event_timing_lag)
from GolemQ.core.constants import AKA
from GolemQ.core.constants import FIELD as FLD
from GolemQ.models.alias import LTT

__all__ = ['BASE_COLUMNS', 'BASE_WEIGHTS', 'RESONANCE_THRESHOLD',
           'calc_mtf_buy_signals', 'calc_mtf_buy_resonance',
           'calc_mtf_buy_weight']

#: 各周期的基础权重（用户 2026-11-24）。
BASE_WEIGHTS = {'15min': 1, '60min': 2, 'day': 3}
#: 「共振」阈值：三周期权重相加 >= 4。
RESONANCE_THRESHOLD = 4

_QUADS = (LTT.QUADRANT_1, LTT.QUADRANT_2, LTT.QUADRANT_3, LTT.QUADRANT_4)

#: 各周期四量图的年化参数（与 `feature_flow` 的 q60/q15/qd 一致）。
_ANNUAL = (('15min', 4032), ('60min', 1008), ('day', 252))


def _level0(index) -> pd.DatetimeIndex:
    if isinstance(index, pd.MultiIndex):
        return pd.DatetimeIndex(pd.to_datetime(index.get_level_values(0)))
    return pd.DatetimeIndex(pd.to_datetime(index))


def _freq_state(ohlc: pd.DataFrame, annual: int):
    """单周期 → `(base, qual)` 两个 bool 序列（与 `ohlc` 同索引）。

    任何一步算不出来（短历史/缺列）⇒ 该周期全 False（不贡献权重），不抛。
    """
    if ohlc is None or len(ohlc) == 0:
        return None
    idx = ohlc.index
    try:
        close = pd.to_numeric(ohlc[AKA.CLOSE], errors='coerce')
        dif, dea, macd = calc_macd_qa(close)
        _dif = pd.Series(np.asarray(dif, dtype=float), index=idx)
        _dea = pd.Series(np.asarray(dea, dtype=float), index=idx)
        # `MACD_CROSS_SX_BEFORE` = DEA 上穿 DIF（死叉）；JX 取其镜像。
        sx = pd.Series(
            np.asarray(Timeline_duration(
                cross_above(_dea, _dif).to_numpy(np.int16)), dtype=float),
            index=idx)
        jx = pd.Series(
            np.asarray(Timeline_duration(
                cross_above(_dif, _dea).to_numpy(np.int16)), dtype=float),
            index=idx)
        dea_lag = pd.Series(
            np.asarray(calc_event_timing_lag(
                np.where(np.asarray(dea) > 0, 1,
                         np.where(np.asarray(dea) < 0, -1, 0))),
                dtype=float), index=idx)
        bs = calc_bs_signals(ohlc)
        code = ''
        if isinstance(idx, pd.MultiIndex) and 'code' in (idx.names or []):
            code = str(idx[0][1])
        q = calc_four_quadrant_boost(
            code, features=pd.DataFrame(index=idx), ohlc_data=ohlc,
            annual=int(annual))
        mf = calc_mainstream_feats(
            ohlc, pd.DataFrame({FLD.MACD: np.asarray(macd, dtype=float)},
                               index=idx))
        lead = pd.to_numeric(mf[LTT.MAINSTREAM_LEADIN_BEFORE], errors='coerce')

        bsb = pd.to_numeric(bs[LTT.BS_BUY_BEFORE], errors='coerce')
        bss = pd.to_numeric(bs[LTT.BS_SELL_BEFORE], errors='coerce')
        n_q1 = sum((pd.to_numeric(q[c], errors='coerce') == 1).astype(int)
                   for c in _QUADS if c in q.columns)
        base = ((bsb < bss)
                & (n_q1 == 1)
                & (pd.Series(np.asarray(macd, dtype=float), index=idx) > 0)
                & ((bsb + 9) >= jx)
                & (bsb < sx)
                & (dea_lag < bsb))
        qual = lead < sx
        return pd.DataFrame({'base': base.fillna(False).to_numpy(dtype=bool),
                             'qual': qual.fillna(False).to_numpy(dtype=bool)},
                            index=idx)
    except Exception:               # noqa: BLE001 短历史/缺列 ⇒ 该周期不贡献
        return pd.DataFrame({'base': np.zeros(len(idx), dtype=bool),
                             'qual': np.zeros(len(idx), dtype=bool)},
                            index=idx)


def _compose_day_weight(has15: bool, q15: bool, has60: bool, q60: bool,
                        hasd: bool, qd: bool) -> int:
    """一天的权重 = 各周期（有命中 ? 基础权重 : 0）+（有 qual ? 1 : 0）。

    >>> _compose_day_weight(True, False, True, False, False, False)   # 15m+60m
    3
    >>> _compose_day_weight(True, False, False, False, True, True)    # 15m+day(带 qual)
    5
    >>> _compose_day_weight(False, False, False, False, True, False)  # day 无 qual
    3
    >>> _compose_day_weight(False, False, False, False, True, True)   # day 带 qual
    4
    """
    w = 0
    for has, q, f in ((has15, q15, '15min'), (has60, q60, '60min'),
                      (hasd, qd, 'day')):
        if has:
            w += BASE_WEIGHTS[f] + (1 if q else 0)
    return w


#: 各周期基础信号 → 目标列名（用户 2026-11-24 定名：15min 后缀 `_MINOR`，
#: 60min 无后缀，day 后缀 `_MAJOR`）。
BASE_COLUMNS = {
    '15min': LTT.QUADRANT_MAINSTREAM_UPRISING_BEFORE_MINOR,
    '60min': LTT.QUADRANT_MAINSTREAM_UPRISING_BEFORE,
    'day': LTT.QUADRANT_MAINSTREAM_UPRISING_BEFORE_MAJOR,
}


def _state_map(f60, f15, fd, annual=_ANNUAL):
    """三周期 → `{周期名: (base, qual) 帧}`（每个周期只算一次）。"""
    frames = {'15min': f15, '60min': f60, 'day': fd}
    states = {}
    for name, ann in annual:
        st = _freq_state(frames[name], ann)
        if st is not None:
            states[name] = st
    return states


def _daily_weights_from_states(states) -> pd.Series:
    """`{周期名: 状态帧}` → 逐自然日权重（索引为 normalized 日期）。"""
    per = {}
    for name, st in states.items():
        dates = _level0(st.index).normalize()
        df = pd.DataFrame({'base': st['base'].to_numpy(dtype=bool),
                           'hit_qual': (st['base'] & st['qual']).to_numpy(
                               dtype=bool)}, index=dates)
        g = df.groupby(level=0)
        has = g['base'].any()
        has_q = g['hit_qual'].any()
        per[name] = pd.Series(
            np.where(has.to_numpy(), BASE_WEIGHTS[name], 0)
            + np.where(has_q.reindex(has.index).fillna(False).to_numpy(), 1, 0),
            index=has.index)
    if not per:
        return pd.Series(dtype=float)
    w = pd.concat(per, axis=1).fillna(0).sum(axis=1)
    w.index = pd.DatetimeIndex(w.index).normalize()
    w = w[~w.index.duplicated(keep='last')].sort_index()
    return w.astype(float)


def _daily_weights(f60, f15, fd, annual=_ANNUAL):
    """三帧 → 逐自然日权重（索引为 normalized 日期）。"""
    return _daily_weights_from_states(_state_map(f60, f15, fd, annual))


def _map_daily(daily: pd.Series, target_index) -> pd.Series:
    """日权重因果映射到 60min：盘中用上一交易日、15:00 用当日。"""
    if daily is None or len(daily) == 0:
        return pd.Series(np.nan, index=target_index)
    if isinstance(target_index, pd.MultiIndex):
        return _map_causal(daily, target_index)
    mi = pd.MultiIndex.from_arrays(
        [pd.DatetimeIndex(target_index), [''] * len(target_index)],
        names=['ts', 'code'])
    return pd.Series(_map_causal(daily, mi).to_numpy(), index=target_index)


def _base_map(states, target_index) -> pd.DataFrame:
    """三周期基础信号 → 对齐到 60min 的 bool 列（as-of 因果）。"""
    ts = _level0(target_index)
    out = pd.DataFrame(index=target_index)
    for name, st in states.items():
        base = pd.Series(st['base'].to_numpy(dtype=bool), index=_level0(st.index))
        base = base[~base.index.duplicated(keep='last')].sort_index()
        if name == 'day':
            mapped = _map_daily(base.astype(float), target_index)
            out[BASE_COLUMNS[name]] = mapped.fillna(0).astype(bool)
        else:
            mapped = base.reindex(ts, method='ffill')
            out[BASE_COLUMNS[name]] = mapped.fillna(False).astype(bool)
    return out


def calc_mtf_buy_signals(f60: pd.DataFrame, f15: pd.DataFrame,
                         fd: pd.DataFrame) -> pd.DataFrame:
    """三帧 → 对齐到 ``f60`` 索引的三个**基础信号**（bool）。

    列名（用户 2026-11-24）：`QuadMsUprBfMin` / `QuadMsUprBf` / `QuadMsUprBfMaj`
    —— 15min / 60min / day 各自的「买点 6 条件」是否成立（as-of 因果映射）。
    """
    return _base_map(_state_map(f60, f15, fd), f60.index)


def calc_mtf_buy_resonance(f60: pd.DataFrame, f15: pd.DataFrame,
                           fd: pd.DataFrame) -> pd.DataFrame:
    """三帧 → 三个基础信号 + `QuadMsCredCpd`（一次算完，避免重复 `_freq_state`）。"""
    states = _state_map(f60, f15, fd)
    out = _base_map(states, f60.index)
    daily = _daily_weights_from_states(states)
    out[LTT.QUADRANT_MAINSTREAM_CREDIT_COMPOUNDED] = _map_daily(
        daily, f60.index)
    return out


def calc_mtf_buy_weight(f60: pd.DataFrame, f15: pd.DataFrame,
                        fd: pd.DataFrame) -> pd.Series:
    """三帧 → 对齐到 ``f60`` 索引的 `QuadMsCredCpd`（日权重，因果映射）。

    返回的 Series 名字为 :attr:`LTT.QUADRANT_MAINSTREAM_CREDIT_COMPOUNDED`
    （`'QuadMsCredCpd'`）；`>= RESONANCE_THRESHOLD`（4）即用户口径的「共振」。

    >>> import pandas as pd
    >>> idx = pd.DatetimeIndex(['2024-01-02 15:00'])
    >>> f = pd.DataFrame({'open': [1.0], 'high': [1.0], 'low': [1.0],
    ...                   'close': [1.0], 'volume': [1.0]}, index=idx)
    >>> calc_mtf_buy_weight(f, f, f).name
    'QuadMsCredCpd'
    """
    return calc_mtf_buy_resonance(f60, f15, fd)[
        LTT.QUADRANT_MAINSTREAM_CREDIT_COMPOUNDED]
