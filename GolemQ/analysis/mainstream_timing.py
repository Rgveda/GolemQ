# coding: utf-8
"""Mainstream「全局智能仓位」的**逐股输入**（纯计算，不连库）。

对应旧树三处：

* 日线值 → 当日 4 根 60min 的广播：`GolemQ_old/models/mainstream.py:1194-1365`
  （`resample('30min').bfill()` → `+23.5h` → `reindex` 到 10:30/11:30/14:00/15:00）；
* `zDLagD`（缠论笔方向）：`GolemQ_old/analysis/mainstream.py:125-163`
  → 新树已有 `analysis/pivot.py::calc_zen_dash_timing_lag`，这里只负责造 CZSC；
* 两个前奏计数：`GolemQ_old/analysis/mainstream.py:348-360`。

⚠️ **后视镜**：旧口径的 `zDLagD`（方向写在笔起点 `fx_a`）不是因果的。
本模块提供三档：`calc_zen_dash_dummy`（旧树原样，仅对拍用）、
`calc_zen_dash_asof`（逐 bar 重建，慢，作基准）、
**`calc_zen_dash_causal`（新版无后视镜，管线默认）**。
「当日日线值贴到当日盘中 bar」仍照旧树保留（D79 已记录）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from GolemQ.analysis.pivot import calc_zen_dash_timing_lag
from GolemQ.analysis.timing import Timeline_duration, calc_event_timing_lag
from GolemQ.core.constants import FIELD as FLD
from GolemQ.models.alias import LTT

__all__ = ['broadcast_daily_to_60min', 'calc_zen_dash_dummy',
           'calc_zen_dash_asof', 'calc_zen_dash_causal', 'calc_prelude_before',
           'calc_m9t_combo']


def broadcast_daily_to_60min(daily: pd.Series, index60) -> pd.Series:
    """日线序列 → 60min 网格：**交易日 D 的值落到 D 当天 4 根 bar**。

    逐字复刻旧树的两步（`resample('30min').bfill()` + `+23.5h`）：

    * `bfill` 之后，D-1 白天的 30min 桶取到的是 **D 的值**；
    * 整体后移 23.5h 后，这些桶正好落在 **D 的 10:30/11:30/14:00/15:00**。

    序列**第一天**没有前驱可填 ⇒ NaN（旧树同样如此，下游 `>1e-19` 会判成 False）。

    >>> idx = pd.DatetimeIndex(
    ...     [f'2024-01-0{d} {t}' for d in (1, 2, 3) for t in
    ...      ('10:30', '11:30', '14:00', '15:00')])
    >>> daily = pd.Series([1.0, 2.0, 3.0],
    ...                   index=pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03']))
    >>> out = broadcast_daily_to_60min(daily, idx)
    >>> [None if pd.isna(v) else v for v in out.tolist()]
    [None, None, None, None, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0]
    """
    s = pd.Series(daily).copy()
    if isinstance(s.index, pd.MultiIndex):
        s.index = pd.DatetimeIndex(
            pd.to_datetime(s.index.get_level_values(0)))
    else:
        s.index = pd.DatetimeIndex(pd.to_datetime(s.index))
    s = s[~s.index.duplicated(keep='last')].sort_index()
    if len(s) == 0:
        return pd.Series(np.nan, index=pd.DatetimeIndex(index60), dtype=float)
    s = s.resample('30min').bfill()
    s.index = s.index + pd.Timedelta(hours=23.5)
    if isinstance(index60, pd.MultiIndex):
        target = pd.DatetimeIndex(pd.to_datetime(index60.get_level_values(0)))
    else:
        target = pd.DatetimeIndex(pd.to_datetime(index60))
    return s.reindex(target)


def _raw_bars(ohlc: pd.DataFrame, code: str, freq):
    """OHLC(`(dt, code)` 索引) → czsc `RawBar` 列表（60min 用 `Freq.F60`）。"""
    from czsc.objects import RawBar

    ts = pd.DatetimeIndex(pd.to_datetime(ohlc.index.get_level_values(0)))
    o = pd.to_numeric(ohlc['open'], errors='coerce').to_numpy(dtype=float)
    h = pd.to_numeric(ohlc['high'], errors='coerce').to_numpy(dtype=float)
    l = pd.to_numeric(ohlc['low'], errors='coerce').to_numpy(dtype=float)
    c = pd.to_numeric(ohlc['close'], errors='coerce').to_numpy(dtype=float)
    v = pd.to_numeric(ohlc.get('vol', ohlc.get('volume', 0)), errors='coerce')
    a = pd.to_numeric(ohlc.get('amount', 0), errors='coerce')
    v = v.to_numpy(dtype=float) if hasattr(v, 'to_numpy') else np.zeros(len(ts))
    a = a.to_numpy(dtype=float) if hasattr(a, 'to_numpy') else np.zeros(len(ts))
    bars = []
    for i, t in enumerate(ts):
        bars.append(RawBar(symbol=str(code), dt=t, id=i, freq=freq,
                           open=o[i], close=c[i], high=h[i], low=l[i],
                           vol=0.0 if not np.isfinite(v[i]) else v[i],
                           amount=0.0 if not np.isfinite(a[i]) else a[i]))
    return bars


def calc_zen_dash_dummy(ohlc: pd.DataFrame, code: str = '',
                        max_bi_count: int = 126) -> pd.Series:
    """`zDLagD`：缠论笔方向 → `calc_event_timing_lag`。

    `max_bi_count=126` 与旧树 `GolemQ_old/analysis/mainstream.py:107` 一致
    （不是 review 里用的 512/5000 —— 笔数上限不同会改变历史笔）。

    :param ohlc: 单只标的 60min OHLCV(+amount)，索引 `(dt, code)` 或 `dt`
    :returns: 与 `ohlc.index` 等长的 `int32` Series（列名 `zDLagD`）
    """
    empty = pd.Series(dtype=np.int32, name=FLD.ZEN_DASH_TIMING_LAG_DUMMY)
    if ohlc is None or len(ohlc) < 30:
        return empty
    from czsc import CZSC
    from czsc.objects import Freq

    czsc = CZSC(_raw_bars(ohlc, code, Freq.F60),
                max_bi_count=max_bi_count, verbose=False)
    out = calc_zen_dash_timing_lag(czsc.bi_list, ohlc.index)
    out.name = FLD.ZEN_DASH_TIMING_LAG_DUMMY
    return out


def calc_zen_dash_asof(ohlc: pd.DataFrame, code: str = '',
                       window: int = 2120, max_bi_count: int = 126) -> pd.Series:
    """**逐 bar as-of** 的 `zDLagD`。

    旧树每次运行只看到「截至运行时刻」的数据，笔的最后一笔可能随新 bar 改向；
    用整段序列算会把未来信息带回过去（实测 2026-06-30：12 只抽样里 9 只
    与 4.4 不一致，`zDLagD` 符号都翻了）。这里对每根 bar 只用它之前
    `window` 根重算一次 CZSC。

    ⚠️ 成本：每码约 `len(ohlc) × CZSC(window)`；60min 800 天 ≈ 2,100 根
    ⇒ 每码 20~40 s，全市场 24 workers 约 1.5~3 h。日常增量只算最后一根，秒级。

    :param window: 回看 bar 数（800 自然日 ≈ 2,120 根 60min）
    """
    if ohlc is None or len(ohlc) < 30:
        return pd.Series(dtype=np.int32, name=FLD.ZEN_DASH_TIMING_LAG_DUMMY)
    n = len(ohlc)
    out = np.full(n, -1, dtype=np.int32)
    for i in range(n):
        w = ohlc.iloc[max(0, i - window + 1): i + 1]
        s = calc_zen_dash_dummy(w, code, max_bi_count=max_bi_count)
        if len(s):
            out[i] = int(s.iloc[-1])
    return pd.Series(out, index=ohlc.index, dtype=np.int32,
                     name=FLD.ZEN_DASH_TIMING_LAG_DUMMY)


def calc_zen_dash_causal(ohlc: pd.DataFrame, code: str = '',
                         max_bi_count: int = 126) -> pd.Series:
    """**新版无后视镜**的 `zDLagD`：只认「当下已经确认的末笔」。

    与 `calc_zen_dash_asof`（逐 bar 重建 CZSC，慢但可作基准）语义相同 ——
    都是「截至该 bar 的 CZSC 状态」—— 这里改用 czsc 的逐 bar 增量更新
    （`CZSC.update`，与 `pivot.bi_confirm_map` 同一手法），一次线性回放。

    * 第 i 根 bar 的方向 = 该 bar 处理完之后 `bi_list[-1].direction`
      （空 ⇒ 0，按旧式 `>0 ? 1 : -1` 落成 -1）；
    * 再取 `calc_event_timing_lag`：同向连续计 ±1/±2…，换向重置。

    ⚠️ 旧树把方向写在 `bi.fx_a.dt`（笔**起点**）上，等于把后确认的信息
    带回到过去；本函数写在「确认时的当前 bar」，因此与 4.4 的历史值
    不逐值一致（D79 已定：新树口径自洽即可）。
    """
    empty = pd.Series(dtype=np.int32, name=FLD.ZEN_DASH_TIMING_LAG_DUMMY)
    if ohlc is None or len(ohlc) < 30:
        return empty
    from czsc import CZSC
    from czsc.objects import Freq

    bars = _raw_bars(ohlc, code, Freq.F60)
    n = len(bars)
    d = np.zeros(n, dtype=np.int8)
    czsc = CZSC(bars[:1], max_bi_count=max_bi_count, verbose=False)
    for i in range(1, n):
        czsc.update(bars[i])
        if czsc.bi_list:
            up = getattr(czsc.bi_list[-1].direction, 'name',
                         str(czsc.bi_list[-1].direction)) in ('Up', 'up', '向上')
            d[i] = 1 if up else -1
    out = calc_event_timing_lag(np.where(d > 0, 1, -1))
    return pd.Series(out, index=ohlc.index, dtype=np.int32,
                     name=FLD.ZEN_DASH_TIMING_LAG_DUMMY)


def calc_m9t_combo(m9t, m9t_lag_maj, m9t_lag_wek):
    """`MAGIC_NINE_TURNS_TIMING_LAG_COMBO`：三列合成的 event-lag（逐码）。

    旧树 `analysis/base.py:2348-2358`（`FLD.MAGIC_NINE_TURNS_COMPOUNDED` 同一式子）：

    ``sum = m9tLagMajR(nan→0) + m9t + m9tLagWekR(nan→0)``，`sum > +0.00168 → +1`、
    `sum < -0.00168 → -1`、其余 `0`，再取 `calc_event_timing_lag`。

    >>> calc_m9t_combo([1, 1, -1, -1], [0, 0, 0, 0], [0, 0, 0, 0]).tolist()
    [1, 2, -1, -2]
    """
    a = np.nan_to_num(np.asarray(m9t, dtype=float), nan=0.0)
    b = np.nan_to_num(np.asarray(m9t_lag_maj, dtype=float), nan=0.0)
    c = np.nan_to_num(np.asarray(m9t_lag_wek, dtype=float), nan=0.0)
    s = a + b + c
    return calc_event_timing_lag(
        np.where(s > 0.00168, 1, np.where(s < -0.00168, -1, 0)))


def calc_prelude_before(features: pd.DataFrame, window: int = 0) -> pd.DataFrame:
    """补 `QuadPreUprBf` / `QuadPreEpBf` 两列（旧树 `analysis/mainstream.py:348-360`）。

    需要四列在场：`QLevMACDLagD` / `QuadDwnRskLagMaj` / `QuadDwnRskLag` / `zDLagD`。

    >>> f = pd.DataFrame({
    ...     'QLevMACDLagD': [1, 1, -1, 1, 1],
    ...     'QuadDwnRskLagMaj': [1, 1, 1, -1, -1],
    ...     'QuadDwnRskLag': [1, 1, 1, -1, -1],
    ...     'zDLagD': [1, 1, 1, -1, -1]})
    >>> out = calc_prelude_before(f)
    >>> out['QuadPreUprBf'].tolist(), out['QuadPreEpBf'].tolist()
    ([0, 0, 1, 2, 3], [1, 2, 3, 0, 0])
    """
    need = (LTT.QUADRANT_LEVERAGE_MACD_TIMING_LAG_DUMMY,
            LTT.QUADRANT_DOWNRISK_TIMING_LAG_MAJOR,
            LTT.QUADRANT_DOWNRISK_TIMING_LAG,
            FLD.ZEN_DASH_TIMING_LAG_DUMMY)
    missing = [c for c in need if c not in features.columns]
    if missing:
        raise KeyError(f'calc_prelude_before 缺列: {missing}')

    up = ((features[LTT.QUADRANT_LEVERAGE_MACD_TIMING_LAG_DUMMY] > 1e-12) &
          (features[LTT.QUADRANT_DOWNRISK_TIMING_LAG_MAJOR] > 1e-12) &
          (features[LTT.QUADRANT_DOWNRISK_TIMING_LAG] > 1e-12) &
          (features[FLD.ZEN_DASH_TIMING_LAG_DUMMY] > 1e-12))
    ep = ((features[LTT.QUADRANT_DOWNRISK_TIMING_LAG_MAJOR] < -1e-12) &
          (features[LTT.QUADRANT_DOWNRISK_TIMING_LAG] < -1e-12) &
          (features[FLD.ZEN_DASH_TIMING_LAG_DUMMY] < -1e-12))
    up_dur = Timeline_duration(np.where(up.to_numpy(), 1, 0))
    ep_dur = Timeline_duration(np.where(ep.to_numpy(), 1, 0))
    if window and window > 0:
        # 旧树每次运行只看到「往前 800 天」的帧 ⇒ 计数在窗口头被截断。
        # 全帧 Timeline_duration 的等价修正是 `min(距上次事件, 距窗口头)`。
        n = len(up_dur)
        idx = np.arange(n)
        cap = np.minimum(idx + 1, window)
        up_dur = np.minimum(up_dur, cap)
        ep_dur = np.minimum(ep_dur, cap)
    features[LTT.QUADRANT_PRELUDE_UPRISING_BEFORE] = up_dur
    features[LTT.QUADRANT_PRELUDE_ENDPOINT_BEFORE] = ep_dur
    return features
