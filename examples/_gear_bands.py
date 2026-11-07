# coding: utf-8
"""QuadGear0 触发区段（橙色半透明）—— Streamlit examples 共用。

`QuadGear0Bf` 是 O 档信号的 `Timeline_duration`：**0 = 触发**，1/2/… = 距上次
触发。因此 `QuadGear0Bf < 1e-12` 就是触发 bar。

信号固定按 **60min / 15min / 日线** 三频计算（`analysis/feature_flow.py`），
所以低频率图按**触发时间**画带；日线图按整日画。三帧只在这里取一次，
由 `st.cache_data` 按 symbol 缓存。

⚠️ 本模块只负责「取数 → Gear 信号 → 画带」，不参与中枢/笔/Renko。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from GolemQ import get_active_market
from GolemQ.analysis.ChipDistribution_jit import calc_stock_chip_distribution_jit
from GolemQ.analysis.feature_flow import quadrant_gear_features
from GolemQ.analysis.mainstream_timing import broadcast_daily_to_60min
from GolemQ.analysis.reality_attach import attach_reality_feats
from GolemQ.core.constants import AKA
from GolemQ.models.alias import LTT

#: Gear 只需要 MA120 / leadout 的暖机；不需要 mainstream 的 2×800 天 CZSC 窗口。
GEAR_LOOKBACK_DAYS = 900


def _frame(res):
    if isinstance(res, tuple):
        res = res[0]
    return getattr(res, 'data', None)


@st.cache_data(show_spinner=False, ttl=600)
def gear_o_before(symbol: str, lookback_days: int = GEAR_LOOKBACK_DAYS) -> pd.Series:
    """60min `QuadGear0Bf`（索引 = 60min 时间戳）。数据不足返回空 Series。"""
    market = get_active_market()
    end = pd.Timestamp.now()
    start = end - pd.Timedelta(days=int(lookback_days))
    start_day = start.strftime('%Y-%m-%d')
    end_min = end.strftime('%Y-%m-%d %H:%M:%S')
    end_day = end.strftime('%Y-%m-%d')

    f60 = _frame(market.get_kline_price_min(
        str(symbol), start=start_day, end=end_min, frequency='60min',
        realtime=False, verbose=False))
    f15 = _frame(market.get_kline_price_min(
        str(symbol), start=start_day, end=end_min, frequency='15min',
        realtime=False, verbose=False))
    fd = _frame(market.get_kline_price_v3(
        str(symbol), start=start_day, end=end_day, realtime=False,
        verbose=False))

    if any(x is None or len(x) < 262 for x in (f60, f15, fd)):
        return pd.Series(dtype=float, name=LTT.QUADRANT_GEAR_O_BEFORE)

    # Charm_O 要 `cost15Prz`：先 attach review；attach 也缺（如 000711 的
    # review 表没有 cost15PrzD）时，用日线 ChipDistribution 现算再贴回 60min。
    try:
        attached, _ = attach_reality_feats(f60.copy(), str(symbol),
                                           verbose=False)
        # attach 会把契约列**建出来但不一定有值**（000711 就是全 NaN）——
        # 必须看覆盖率，不能只看列是否存在。
        _c = attached.get(AKA.COST15_PRICE)
        if _c is not None and _c.notna().mean() > 0.5:
            f60 = attached
    except Exception:                        # noqa: BLE001 示例页缺列不致命
        pass
    if AKA.COST15_PRICE not in f60.columns:
        try:
            chip = calc_stock_chip_distribution_jit(
                features=pd.DataFrame(index=fd.index), ohlc_data=fd,
                annual=252, code=str(symbol), verbose=False)
            v = broadcast_daily_to_60min(chip[AKA.COST15_PRICE], f60.index)
            f60 = f60.copy()
            f60[AKA.COST15_PRICE] = pd.Series(
                np.asarray(v, dtype=float), index=f60.index)
        except Exception:                    # noqa: BLE001 缺换手率时退回 Charm_I
            pass

    out = quadrant_gear_features(str(symbol), f60, f15, fd)
    return out[LTT.QUADRANT_GEAR_O_BEFORE]


def trigger_runs(bf: pd.Series):
    """`QuadGear0Bf < 1e-12` 的连续段：`[(start_ts, end_ts), ...]`。"""
    if bf is None or len(bf) == 0:
        return []
    if isinstance(bf.index, pd.MultiIndex):
        idx = pd.DatetimeIndex(pd.to_datetime(bf.index.get_level_values(0)))
    else:
        idx = pd.DatetimeIndex(pd.to_datetime(bf.index))
    mask = pd.to_numeric(bf, errors='coerce').to_numpy() < 1e-12
    runs, start = [], None
    for i, on in enumerate(mask):
        if on and start is None:
            start = i
        elif not on and start is not None:
            runs.append((idx[start], idx[i - 1]))
            start = None
    if start is not None:
        runs.append((idx[start], idx[-1]))
    return runs


def add_gear_bands(fig, bf: pd.Series, *, frequency: str = '60min',
                   color: str = 'orange', opacity: float = 0.18):
    """按触发 bar 画橙色半透明竖带；返回 runs（供 caption 计数）。"""
    runs = trigger_runs(bf)
    if not runs:
        return runs
    pad = pd.Timedelta(hours=12) if frequency == 'day' else pd.Timedelta(minutes=30)
    for a, b in runs:
        fig.add_shape(type='rect', xref='x', yref='paper',
                      x0=a - pad, x1=b + pad, y0=0, y1=1,
                      fillcolor=color, opacity=opacity,
                      line=dict(width=0), layer='below')
    return runs
