# coding: utf-8
"""15min minor 信号 —— 老树 `calc_zen_wavelet_reality_minor` 的完整 19 列。

来源：老树 `features/reviews.py:1745-2195`（`calc_zen_chopstickss` 的
`ZEN_DASH/ZEN_PEAK` → AROON → QA-MACD → 四量图 → BS 星线 → 主力吸筹），
最后 rename 成 `*_MINOR`。

* `MinorSignalsStep`：在 **15min 全帧**（固定锚点前缀）上算一次，输出 15min 族；
* `HourMinorSampleStep`：把其中 **17 个 minor 列**按精确时间戳采样到 60min 快照。
  ⚠️ 60min 的 `zDLagMinR/zPLagMinR` 由 `zen_snapshots` 产出，这里**不重复采**。

⚠️ AROON 的 up/down 交换是老树的真实 bug（`aroon_up=aroondown`），这里**照搬**
以保证对拍；修正它会让 `AroUprsgLagMin` 符号翻转，需单独拍板（已记入 D54 待确认清单）。
"""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
import talib

from GolemQ.analysis.indicators import calc_macd_qa
from GolemQ.analysis.quadrant import (calc_bs_signals, calc_four_quadrant_boost,
                                      calc_mainstream_feats)
from GolemQ.analysis.timing import calc_event_timing_lag
from GolemQ.analysis.zen import calc_zen_timing_lag
from GolemQ.core.constants import AKA, FEATURES, FIELD as FLD
from GolemQ.models.alias import LTT, TRD
from GolemQ.pipeline.steps import (FeatureStep, ReviewContext, TIER_RUNTIME,
                                   slice_asof)
from GolemQ.pipeline.review.contract import MINOR_15MIN_COLUMNS, MINOR_COLUMNS

#: 老树 15min 的 rename 表（源列 → MINOR 列）。
_RENAME = {
    FLD.ZEN_DASH_TIMING_LAG: FEATURES.ZEN_DASH_TIMING_LAG_MINOR_REAL,
    FLD.ZEN_PEAK_TIMING_LAG: FEATURES.ZEN_PEAK_TIMING_LAG_MINOR_REAL,
    LTT.ONE_QUADRANT_UPRISING_TIMING_LAG: LTT.ONE_QUADRANT_UPRISING_TIMING_LAG_MINOR,
    LTT.TWO_QUADRANT_UPRISING_TIMING_LAG: LTT.TWO_QUADRANT_UPRISING_TIMING_LAG_MINOR,
    LTT.THREE_QUADRANT_UPRISING_TIMING_LAG: LTT.THREE_QUADRANT_UPRISING_TIMING_LAG_MINOR,
    LTT.FOUR_QUADRANT_UPRISING_TIMING_LAG: LTT.FOUR_QUADRANT_UPRISING_TIMING_LAG_MINOR,
    LTT.QUADRANT_POWERLINE: LTT.QUADRANT_POWERLINE_MINOR,
    LTT.QUADRANT_POWERLINE_LEADIN_BEFORE: LTT.QUADRANT_POWERLINE_LEADIN_BEFORE_MINOR,
    LTT.QUADRANT_POWERLINE_LEADOUT_BEFORE: LTT.QUADRANT_POWERLINE_LEADOUT_BEFORE_MINOR,
    LTT.BS_BUY_BEFORE: LTT.BS_BUY_BEFORE_MINOR,
    LTT.BS_SELL_BEFORE: LTT.BS_SELL_BEFORE_MINOR,
    LTT.BS_STAR_UP_BEFORE: LTT.BS_STAR_UP_BEFORE_MINOR,
    LTT.BS_STAR_DOWN_BEFORE: LTT.BS_STAR_DOWN_BEFORE_MINOR,
    LTT.MAINSTREAM_LEADIN_VAR5: LTT.MAINSTREAM_LEADIN_VAR5_MINOR,
    LTT.MAINSTREAM_LEADIN_ENTRY: LTT.MAINSTREAM_LEADIN_ENTRY_MINOR,
    LTT.MAINSTREAM_LEADIN_BEFORE: LTT.MAINSTREAM_LEADIN_BEFORE_MINOR,
    LTT.MAINSTREAM_LEADINx3_BEFORE: LTT.MAINSTREAM_LEADINx3_BEFORE_MINOR,
    LTT.MAINSTREAM_LEADINx5_BEFORE: LTT.MAINSTREAM_LEADINx5_BEFORE_MINOR,
}


def minor_signals_step(ctx: ReviewContext, targets: pd.DatetimeIndex) -> pd.DataFrame:
    """在 15min 帧上算 17 个 minor 信号（一次算全帧，按 `as_of` 截断）。"""
    ohlc = slice_asof(ctx.frame('15min'), ctx.as_of)
    if ohlc is None or len(ohlc) < 30:
        return pd.DataFrame(columns=MINOR_15MIN_COLUMNS)

    feats = pd.DataFrame(index=ohlc.index)
    close = pd.to_numeric(ohlc[AKA.CLOSE], errors='coerce')
    high = pd.to_numeric(ohlc[AKA.HIGH], errors='coerce')
    low = pd.to_numeric(ohlc[AKA.LOW], errors='coerce')

    # 1) AROON（⚠️ 交换照搬老树）
    aroondown, aroonup = talib.AROON(high.to_numpy(np.float64),
                                     low.to_numpy(np.float64), timeperiod=36)
    aroon_up, aroon_down = aroondown, aroonup
    feats[TRD.AROON_UPRISING_TIMING_LAG_MINOR] = calc_event_timing_lag(
        np.where(aroon_up > aroon_down, 1, -1)).astype(np.int16)

    # 2) QUANTAXIS 口径 MACD（15min 链必须用它）
    _dif, _dea, hist = calc_macd_qa(close)
    feats[FLD.MACD] = hist.to_numpy()

    # 3) 四量图 → 4) BS 星线 → 5) 主力吸筹
    feats = calc_four_quadrant_boost(ctx.code, features=feats,
                                     ohlc_data=ohlc, annual=4032)
    feats = calc_bs_signals(ohlc, feats)
    feats = calc_mainstream_feats(ohlc, feats)

    # 5) zen（老树 `calc_zen_chopstickss` 的核，新树 czsc 0.7.10）
    feats = calc_zen_timing_lag(ohlc, features=feats,
                                market_type=ctx.market_type)

    feats = feats.rename(columns=_RENAME)
    out = pd.DataFrame(index=ohlc.index)
    for c in MINOR_15MIN_COLUMNS:
        out[c] = feats[c] if c in feats.columns else np.nan
    return out


def hour_minor_sample_step(ctx: ReviewContext, targets: pd.DatetimeIndex) -> pd.DataFrame:
    """把 15min minor 结果按**精确时间戳**采样到 60min 目标。"""
    src = ctx.outputs.get('minor_signals')
    if src is None or len(src) == 0:
        return pd.DataFrame(columns=MINOR_COLUMNS)
    # 按时间戳索引（单码），保留最后一根
    s = src.copy()
    s.index = pd.DatetimeIndex(pd.to_datetime(s.index.get_level_values(0)))
    s = s[~s.index.duplicated(keep='last')]
    rows = []
    idx = []
    for T in targets:
        if T in s.index:
            # ⚠️ 只采 **17 个 minor 列**：`minor_signals` 现在多产 2 个 zen 列，
            # 但 60min 的 `zDLagMinR/zPLagMinR` 由 `zen_snapshots` 负责。
            rows.append(s.loc[T, list(MINOR_COLUMNS)].to_numpy())
            idx.append((pd.Timestamp(T), ctx.code))
        else:
            rows.append(np.full(len(MINOR_COLUMNS), np.nan))
            idx.append((pd.Timestamp(T), ctx.code))
    if not idx:
        return pd.DataFrame(columns=MINOR_COLUMNS)
    return pd.DataFrame(rows,
                        index=pd.MultiIndex.from_tuples(idx, names=['ts', 'code']),
                        columns=list(MINOR_COLUMNS))


def build_minor_steps() -> List[FeatureStep]:
    return [
        FeatureStep(name='minor_signals', tier=TIER_RUNTIME,
                    input_freq='15min', output_family='15min',
                    columns=MINOR_15MIN_COLUMNS, func=minor_signals_step,
                    history='anchor'),
        FeatureStep(name='hour_minor_sample', tier=TIER_RUNTIME,
                    input_freq='15min', output_family='60min',
                    columns=MINOR_COLUMNS, func=hour_minor_sample_step,
                    deps=('minor_signals',), history='anchor'),
    ]
