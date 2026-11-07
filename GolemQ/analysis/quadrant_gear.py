# coding: utf-8
"""QuadrantGear —— 四象限「档位」模型（旧树 `analysis/quad_gear.py` 的重做）。

当前只落地 **O 档**（`QuadGear0`）：

.. code-block:: text

    QuadGear0 = (QLevMACDLagD < 0)
              & (QuadDwnRskLagMaj < 0)
              & (BSStarUpBfMaj > BSStarDwnBfMaj)
              & (BSBuBfMaj > BSSelBfMaj)
              & (MsLeadinBfMaj < MsLeadoutBfMaj)
              OR (
                (QLevMACDLagD > 0)
              & ((QuadDwnRskLagMaj + QLevMACDLagD) < 0)
              & (BSBuBfMaj < QLevMACDLagD)
              & (BSStarUpBfMaj > BSStarDwnBfMaj)
              & ((BSBuBfMaj > BSSelBfMaj)
                 | (QLevCharmOBf < QLevMACDLagD))
              & (MsLeadinBfMaj < MsLeadoutBfMaj))

`QuadGear0Bf` 是它的 `Timeline_duration`（0 = 当日触发）。**先合成最终
`QuadGear0`（两条 OR 分支），再算一次 `Timeline_duration`**；不要在第一分支后
先算一遍（会被覆盖）。

⚠️ **第五行必须是两个日线 MAJOR 列**（`MsLeadinBfMaj` / `MsLeadoutBfMaj`）。
`MsLeadinBf` / `MsLeadoutBf` 是 60min bar 计数，与日线计数**量纲不同**，直接比较
会得到错误的 O 档（2026-11-05 用户两次修订后定稿）。

⚠️ **默认 `fold=True`**：`MsLeadinBfMaj` / `MsLeadoutBfMaj` 的阈值按
D73 的折叠法算（`close > 10 → log10(close)`）；高价股不折叠时 leadout 几乎不
触发，会让 O 档永远不成立。

输入：
* `QLevMACDLagD` / `QuadDwnRskLagMaj` / 四个 `BS*_BEFORE_MAJOR` 必须已在帧里；
* `MsLeadinBfMaj` / `MsLeadoutBfMaj` 若未提供，给 `ohlc_day` 即可：本函数用
  `calc_mainstream_major_feats(ohlc_day, fold)` 现算，并按日期 ffill 贴回主频。

旧树 `quad_gear.py` 的 466 行 `where` 链**没有**逐字搬；这是按用户 2026-11-05
公式重写的 O 档。**只做算术，不写库、不被管线调用**。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from GolemQ.analysis.quadrant_jit import calc_mainstream_major_feats
from GolemQ.analysis.timing import Timeline_duration
from GolemQ.models.alias import LTT

__all__ = ['calc_quadrant_gear_o']


def _map_daily(series, index):
    """日线序列 → 主频：按日期取最后值并 ffill（盘中沿用上一交易日）。"""
    dates = pd.Index(pd.to_datetime(index.get_level_values(0))).normalize()
    src = series.copy()
    if isinstance(src.index, pd.MultiIndex):
        src.index = pd.Index(
            pd.to_datetime(src.index.get_level_values(0))).normalize()
    else:
        src.index = pd.Index(pd.to_datetime(src.index)).normalize()
    src = src[~src.index.duplicated(keep='last')]
    return pd.Series(src.reindex(dates).ffill().to_numpy(), index=index)


def calc_quadrant_gear_o(features: pd.DataFrame, *, ohlc_day=None,
                         fold: bool = True) -> pd.DataFrame:
    """在 `features` 上写 `QuadGear0` / `QuadGear0Bf`。

    >>> f = pd.DataFrame({
    ...     LTT.QUADRANT_LEVERAGE_MACD_TIMING_LAG_DUMMY: [-1.0, -1.0, -1.0],
    ...     LTT.QUADRANT_DOWNRISK_TIMING_LAG_MAJOR: [-1.0, -1.0, -1.0],
    ...     LTT.BS_STAR_UP_BEFORE_MAJOR: [2.0, 2.0, 1.0],
    ...     LTT.BS_STAR_DOWN_BEFORE_MAJOR: [1.0, 1.0, 2.0],
    ...     LTT.BS_BUY_BEFORE_MAJOR: [2.0, 2.0, 1.0],
    ...     LTT.BS_SELL_BEFORE_MAJOR: [1.0, 1.0, 2.0],
    ...     LTT.MAINSTREAM_LEADIN_BEFORE_MAJOR: [1.0, 1.0, 2.0],
    ...     LTT.MAINSTREAM_LEADOUT_BEFORE_MAJOR: [2.0, 2.0, 2.0],
    ...     LTT.QUADRANT_LEVERAGE_CHARM_O_BEFORE: [100.0, 100.0, 100.0],
    ...     LTT.QUADRANT_LEVERAGE_CHARM_I_BEFORE: [100.0, 100.0, 100.0]})
    >>> out = calc_quadrant_gear_o(f)
    >>> out[LTT.QUADRANT_GEAR_O].tolist()
    [1.0, 1.0, nan]
    >>> out[LTT.QUADRANT_GEAR_O_BEFORE].tolist()
    [0, 0, 1]
    """
    need = (LTT.QUADRANT_LEVERAGE_MACD_TIMING_LAG_DUMMY,
            LTT.QUADRANT_DOWNRISK_TIMING_LAG_MAJOR,
            LTT.BS_STAR_UP_BEFORE_MAJOR, LTT.BS_STAR_DOWN_BEFORE_MAJOR,
            LTT.BS_BUY_BEFORE_MAJOR, LTT.BS_SELL_BEFORE_MAJOR,
            LTT.QUADRANT_LEVERAGE_CHARM_O_BEFORE)
    missing = [c for c in need if c not in features.columns]
    if missing:
        raise KeyError(f'QuadGear0 缺列: {missing}')

    # 第五行：两个**日线** MAJOR 计数。给了 ohlc_day 就按 fold 现算；
    # 否则必须已有两列（60min 的 MsLeadinBf / MsLeadoutBf 量纲不同，不能用）。
    has_maj = (LTT.MAINSTREAM_LEADIN_BEFORE_MAJOR in features.columns
               and LTT.MAINSTREAM_LEADOUT_BEFORE_MAJOR in features.columns)
    if ohlc_day is not None:
        day = calc_mainstream_major_feats(ohlc_day, fold=fold)
        features[LTT.MAINSTREAM_LEADIN_BEFORE_MAJOR] = _map_daily(
            day[LTT.MAINSTREAM_LEADIN_BEFORE_MAJOR], features.index)
        features[LTT.MAINSTREAM_LEADOUT_BEFORE_MAJOR] = _map_daily(
            day[LTT.MAINSTREAM_LEADOUT_BEFORE_MAJOR], features.index)
    elif not has_maj:
        raise KeyError('QuadGear0 需要 MsLeadinBfMaj/MsLeadoutBfMaj，'
                       '或提供 ohlc_day 以 fold 现算')

    q = features[LTT.QUADRANT_LEVERAGE_MACD_TIMING_LAG_DUMMY]
    dwn = features[LTT.QUADRANT_DOWNRISK_TIMING_LAG_MAJOR]
    star_up = features[LTT.BS_STAR_UP_BEFORE_MAJOR]
    star_dwn = features[LTT.BS_STAR_DOWN_BEFORE_MAJOR]
    buy = features[LTT.BS_BUY_BEFORE_MAJOR]
    sell = features[LTT.BS_SELL_BEFORE_MAJOR]

    # 第一分支：QLev 为负
    base = ((q < -1e-12) & (dwn < -1e-12)
            & (star_up > star_dwn) & (buy > sell)
            & (features[LTT.MAINSTREAM_LEADIN_BEFORE_MAJOR]
               < features[LTT.MAINSTREAM_LEADOUT_BEFORE_MAJOR]))
    # 第二分支（2026-11-06 第六次修订）：QLev 为正，但下行风险把它压成净负，
    # BSBuBfMaj 已小到低于 QLev 的绝对值；方向条件用 Charm_O 兜底
    # （Charm_I 只保留为独立特征，不再进 O 档）。
    extra = ((q > 1e-12) & ((dwn + q) < -1e-12) & (buy < q)
             & (star_up > star_dwn)
             & ((buy > sell)
                | (features[LTT.QUADRANT_LEVERAGE_CHARM_O_BEFORE] < q))
             & (features[LTT.MAINSTREAM_LEADIN_BEFORE_MAJOR]
                < features[LTT.MAINSTREAM_LEADOUT_BEFORE_MAJOR]))

    # ⚠️ 先合成两条 OR，再算一次 Timeline_duration（第一分支后不要先算）。
    features[LTT.QUADRANT_GEAR_O] = np.where(base | extra, 1, np.nan)
    features[LTT.QUADRANT_GEAR_O_BEFORE] = Timeline_duration(
        np.where(features[LTT.QUADRANT_GEAR_O] > 1e-12, 1, 0))
    return features
