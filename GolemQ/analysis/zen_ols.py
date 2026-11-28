# coding: utf-8
"""`ZenOLSLag` —— **重新实现**的「波浪内最大趋势」时序（不搬老树那套依赖）。

老树口径（`GolemQ_old/models/zen.py::zen_trend_ml_vX`）是一条很长的链：
`MA30范数lag + DEA归零lag → 聚类 zen_mapower30_predict → 段边界 →
calc_slice_with_wavelet → calc_slice_with_cluster_grouped(margin=8) → 每段
find_peak_vextors + zen_in_ols_v2_func(LIS/LDS) → JX/SX duration → event lag`。

这里只保留**真正决定信号的那一步**（段内 LIS/LDS 找趋势主干），把**分段**换成
我们已经落库、且天然因果的 60min `ZenPivotLag`：

* `ZenPivotLag > 0` = 已在中枢内 N 根（盘整）
* `ZenPivotLag < 0` = 已离开中枢 N 根（趋势 ⇒ 老树说的「波浪」）

段内峰点支持两种口径（2026-11-22 待评估）：

* ``peak='scipy'``：老树口径，`scipy.argrelextrema` **相邻**极值（噪声大）；
* ``peak='czsc'``：CZSC 笔端点（与 `ZenPivotLag` 同源，且带**确认时间**，
  可以做到严格因果）。

⚠️ **老树自带未来函数**：它对整条历史一次算完，段尾信息回流到段首。
本模块默认 ``causal=True``：对每根 bar 只用「当前段起点 … 当前 bar」重算，
所以数值与老树**不同**（这是刻意的，见 D88 那轮踩过的坑）。

数据流::

    ZenPivotLag ──sign──> 段边界（NaN 按 ffill 延续，不制造新段）
    每段内（仅「中枢外」段）:
        low 的峰点 → LIS(最长递增子序列) → 标 +1
        high 的峰点 → LDS(最长递减子序列) → 标 -1
    JX = Timeline_duration(cross == +1)
    SX = Timeline_duration(cross == -1)
    ZenOLSLag = calc_event_timing_lag(where(JX < SX, +1, -1))
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

__all__ = ['lis_positions', 'lds_positions', 'find_peaks_scipy',
           'czsc_peak_markers', 'segment_ids', 'cross_marks',
           'calc_zen_ols_timing_lag', 'calc_zen_ols_timing_lag_streaming']


# --------------------------------------------------------------------------- #
# LIS / LDS（老树 `analysis/timeseries.py` 的耐心排序版，逐行照搬）
# --------------------------------------------------------------------------- #
def _lis_generic(x: np.ndarray, increasing: bool = True):
    """最长（递增/递减）子序列：返回 ``(子序列值, 原序列下标)``。"""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n == 0:
        return [], []
    sign = 1.0 if increasing else -1.0
    y = x * sign
    prev = [0] * n
    tail = [0] * (n + 1)          # tail[l] = 长度 l 的最小子序列末元素下标
    length = 0
    for i in range(n):
        lo, hi = 1, length
        while lo <= hi:
            mid = (lo + hi) // 2
            if y[tail[mid]] < y[i]:
                lo = mid + 1
            else:
                hi = mid - 1
        new_len = lo
        prev[i] = tail[new_len - 1]
        tail[new_len] = i
        if new_len > length:
            length = new_len
    vals, pos = [], []
    k = tail[length]
    for _ in range(length - 1, -1, -1):
        vals.append(x[k])
        pos.append(k)
        k = prev[k]
    return vals[::-1], pos[::-1]


def lis_positions(x) -> List[int]:
    """最长递增子序列的**原序列下标**（升序）。

    >>> lis_positions([3, 1, 2, 5, 4])      # 值 [1, 2, 4]，长度 3
    [1, 2, 4]
    >>> lis_positions([5, 4, 3])
    [2]
    """
    return list(_lis_generic(x, increasing=True)[1])


def lds_positions(x) -> List[int]:
    """最长递减子序列的**原序列下标**（升序）。

    >>> lds_positions([3, 1, 2, 5, 4])      # 值 [5, 4]，长度 2
    [3, 4]
    """
    return list(_lis_generic(x, increasing=False)[1])


# --------------------------------------------------------------------------- #
# 峰点：两种口径
# --------------------------------------------------------------------------- #
def find_peaks_scipy(high, low) -> np.ndarray:
    """老树口径：相邻极值。``+1`` = 低点（谷），``-1`` = 高点（峰）。

    ⚠️ `argrelextrema` 默认 `order=1` ⇒ **只要比左右各一个邻居极端就算**，
    噪声很大（每个小回调都能成为峰）。
    """
    from scipy import signal
    high = np.asarray(high, dtype=float)
    low = np.asarray(low, dtype=float)
    marks = np.zeros(len(low), dtype=np.int8)
    if len(low) >= 3:
        np.put(marks, signal.argrelextrema(low, np.less)[0], 1)
        np.put(marks, signal.argrelextrema(high, np.greater)[0], -1)
    return marks


def frame_context(ohlc: pd.DataFrame):
    """**一次** CZSC：同时产出「峰点标记」和「逐 bar 因果 `ZenPivotLag`」。

    为什么要自己算那条件分段用的 lag：review 里落库的 `ZenPivotLag` 是**逐目标
    2160 根窗口**算的，只在目标日有值；把它 ffill 到整帧，窗口外的 bar 全是
    「未知段」⇒ `ZenOLSLag` 会整列 NaN。

    这里用 `calc_pivots(strict=True)` + `causal_pivot_series(strict=True)`：
    分型只在**确认 bar** 之后才进入中枢计算，因此是**逐 bar 因果**的，
    也因此**前缀稳定**（不随本次跑了哪些 targets 而变）。

    :returns: ``(marks, known, pivot_lag)`` —— 三个与 `ohlc` 等长的数组
    """
    from GolemQ.analysis.pivot import (bi_confirm_map, bi_list_to_points,
                                       calc_pivots, causal_pivot_series)
    from GolemQ.analysis.timing import calc_event_timing_lag
    from GolemQ.analysis.zen import _bars_of, _freq_of

    n = len(ohlc)
    marks = np.zeros(n, dtype=np.int8)
    known = np.arange(n, dtype=np.int64)
    pivot_lag = np.zeros(n, dtype=float)
    if n == 0:
        return marks, known, pivot_lag

    code = str(ohlc.index.get_level_values(1)[0])
    ts = pd.DatetimeIndex(pd.to_datetime(ohlc.index.get_level_values(0)))
    try:
        from czsc import CZSC
        bars = _bars_of(ohlc, code, _freq_of(ohlc))
        czsc = CZSC(bars, max_bi_count=max(5000, len(bars) + 100), verbose=False)
        conf = bi_confirm_map(bars, max_bi_count=max(5000, len(bars) + 100))
        bi_list = list(czsc.bi_list)
    except Exception:                 # noqa: BLE001 数据太短/异常 ⇒ 全 0
        return marks, known, pivot_lag

    # ① 峰点（带确认时间，因果的关键）
    _SEG_TAIL = 2160        # 中枢 lag 每次只重算最近这么多根（见下面的注释）
    pos_of = {t: i for i, t in enumerate(ts)}
    for p in bi_list_to_points(bi_list, conf_map=conf):
        i = pos_of.get(pd.Timestamp(p['dt']))
        if i is None:
            continue
        k = pos_of.get(pd.Timestamp(p.get('known_dt') or p['dt']), i)
        marks[i] = 1 if p['fx_mark'] == 'd' else -1
        known[i] = max(i, k)

    # ② 逐 bar 因果中枢 lag（正值=在中枢内，负值=已离开 N 根）
    try:
        piv = calc_pivots(bi_list=bi_list, conf_map=conf, strict=True)
        zd = causal_pivot_series(piv, ts, strict=True)[0]
        pivot_lag = calc_event_timing_lag(
            np.where(~np.isnan(np.asarray(zd, dtype=float)), 1, -1)).astype(float)
    except Exception:                 # noqa: BLE001 无中枢 ⇒ 全 0（全都算「中枢外」）
        pivot_lag = np.zeros(n, dtype=float)
    return marks, known, pivot_lag


def czsc_peak_markers(ohlc: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """CZSC 口径的峰点标记，**带确认时间**。

    :returns: ``(marks, known)``，两个与 `ohlc` 等长的数组 ——
        ``marks[i]`` 是 ``+1``（底分型）/``-1``（顶分型）/``0``；
        ``known[i]`` 是该分型**被确认**的 bar 下标（≥ i）。
        在 ``t`` 时刻只知道 ``known <= t`` 的分型，这是因果性的关键。
    """
    marks, known, _ = frame_context(ohlc)
    return marks, known


# --------------------------------------------------------------------------- #
# 分段
# --------------------------------------------------------------------------- #
def _sign(v) -> int:
    """返回 -1 / 0 / +1（对 NaN 也安全）。"""
    try:
        f = float(v)
    except Exception:               # noqa: BLE001
        return 0
    if not np.isfinite(f) or f == 0:
        return 0
    return 1 if f > 0 else -1


#: 一侧峰点少于此数 ⇒ 认为「这一浪还没成形」，不出该方向的交叉
#: （用户 2026-11-23：2026-09-22 那段 75 根里只有 1 个高峰，却出了 +1，
#:  一个小时后段切到 205 根又出 -1，同一小时来回翻）。
MIN_PEAKS_SIDE = 3
#: 方向判定：链长要**长出这么多**才算赢（相等/接近 ⇒ 保持不动）。
#: 起因（用户 2026-11-23）：000711 在 2026-07-09 10:30 转多、15:00 又转空 ——
#: 同一段（中枢内、段长 100~107）里 lo=7 / hi=3~4，两边链长接近，逐根在 +1/-1
#: 之间跳。取 1 实测：翻转 205 → 135，且 09-20~10-05 的下跌仍跟得住；
#: 取 2 会把下跌跟丢（那段变成 +103）。
MARGIN = 1
#: 峰点不够时把取样窗按段长放大到这么多倍重取
WIDEN_FACTOR = 2


def _collect_peaks(marks, low, high, a, b):
    """取 ``[a, b]`` 内可用于 LIS/LDS 的峰点下标。"""
    lo = [k for k in range(a, b + 1)
          if marks[k] == 1 and np.isfinite(low[k])]
    hi = [k for k in range(a, b + 1)
          if marks[k] == -1 and np.isfinite(high[k])]
    return lo, hi


def segment_ids(pivot_lag) -> np.ndarray:
    """按 `ZenPivotLag` 的**符号**切段，返回逐 bar 的段编号。

    * ``> 0`` = 中枢内（盘整），``< 0`` = 中枢外（趋势）；
    * **NaN 用 ffill 延续上一段**（缺值不制造新段，用户 2026-11-22 口径）；
    * **开头的 NaN 记 -1**（那时还不知道符号；用 bfill 会吃未来值）。

    >>> import pandas as pd
    >>> segment_ids(pd.Series([5, 3, -2, -1, 4])).tolist()
    [0, 0, 1, 1, 2]
    >>> segment_ids(pd.Series([float('nan'), 2, float('nan'), -1])).tolist()
    [-1, 0, 0, 1]
    """
    s = pd.Series(pivot_lag, dtype=float).ffill()
    sign = np.sign(s.to_numpy(dtype=float))
    out = np.full(len(sign), -1, dtype=np.int64)
    cur, prev, started = 0, 0.0, False
    for i, v in enumerate(sign):
        if not np.isfinite(v) or v == 0:
            out[i] = cur if started else -1
            continue
        if not started:
            started, prev = True, v
        elif v != prev:
            cur += 1
            prev = v
        out[i] = cur
    return out


# --------------------------------------------------------------------------- #
# 主入口
# --------------------------------------------------------------------------- #
def _align_pivot_lag(pivot_lag, ohlc):
    """把调用方给的 `ZenPivotLag` 对齐到 `ohlc` 的逐 bar 序列。

    允许三种形态：**与 ohlc 同索引的 Series**、**扁平时间轴 Series**
    （review 里 `frame` 是按 targets 取的，传进来常是这种）、等长数组/列表。
    """
    n = len(ohlc)
    if pivot_lag is None:
        return None
    if isinstance(pivot_lag, pd.Series):
        try:
            if pivot_lag.index.equals(ohlc.index):
                return pivot_lag.to_numpy(dtype=float)
            ts = (pd.DatetimeIndex(pd.to_datetime(
                ohlc.index.get_level_values(0)))
                if isinstance(ohlc.index, pd.MultiIndex)
                else pd.DatetimeIndex(pd.to_datetime(ohlc.index)))
            re = pivot_lag.reindex(ts)
            if re.notna().any() or pivot_lag.isna().all():
                return re.to_numpy(dtype=float)
        except Exception:           # noqa: BLE001 落到等长数组分支
            pass
    arr = np.asarray(pivot_lag, dtype=float)
    if len(arr) == n:
        return arr
    raise ValueError(f'pivot_lag 长度 {len(arr)} 与 ohlc {n} 不一致')


def cross_marks(ohlc: pd.DataFrame, pivot_lag=None, peak: str = 'czsc',
                causal: bool = True, in_pivot_marks: bool = True,
                min_peak: int = 2) -> np.ndarray:
    """逐 bar 的交叉标记：``+1`` = 多头交叉，``-1`` = 空头交叉，``0`` = 无。

    单独抽出来是为了**可测**（`ZenOLSLag` 只是它的下游累积），
    也方便诊断脚本统计「到底标了几次」。
    """
    n = len(ohlc)
    if n == 0:
        return np.zeros(0, dtype=np.int8)
    high = pd.to_numeric(ohlc['high'], errors='coerce').to_numpy(dtype=float)
    low = pd.to_numeric(ohlc['low'], errors='coerce').to_numpy(dtype=float)

    if peak == 'czsc':
        marks, known, _pl_frame = frame_context(ohlc)
        if pivot_lag is None:
            pivot_lag = _pl_frame
    elif peak == 'scipy':
        marks = find_peaks_scipy(high, low)
        # ⚠️ `argrelextrema` 判 i 是不是极值**需要 i+1 那根** ⇒ 它在 i+1 才可知。
        # 当成「当场知道」会引入 1 根的未来函数（2026-11-22 的因果性用例抓到）。
        known = np.minimum(np.arange(n, dtype=np.int64) + 1, n - 1)
    else:
        raise ValueError(f"peak 只能是 'czsc' / 'scipy'，收到 {peak!r}")

    if pivot_lag is None:
        # 没给就自己算**帧级因果中枢 lag**（与峰点共用同一次 CZSC）
        _, _, _pl = frame_context(ohlc)
        seg = segment_ids(_pl)
        in_pivot = np.nan_to_num(_pl) > 0
    else:
        # 只 ffill：bfill 会拿未来的值补开头，那是未来函数
        pl = _align_pivot_lag(pivot_lag, ohlc)
        pl = pd.Series(pl).ffill().to_numpy(dtype=float)
        seg = segment_ids(pl)
        in_pivot = np.nan_to_num(pl) > 0

    bounds = []
    if n:
        st = 0
        for t in range(1, n):
            if seg[t] != seg[t - 1]:
                bounds.append((st, t - 1, bool(in_pivot[st])))
                st = t
        bounds.append((st, n - 1, bool(in_pivot[st])))
        # 开头的「未知段」不参与标记（那时还判不出符号）
        bounds = [x for x in bounds if seg[x[0]] >= 0]

    cross = np.zeros(n, dtype=np.int8)

    def _mark_idx(a, b, upto=None):
        up = b if upto is None else upto
        lo = [i for i in range(a, up + 1)
              if marks[i] == 1 and (not causal or known[i] <= up)
              and np.isfinite(low[i])]
        hi = [i for i in range(a, up + 1)
              if marks[i] == -1 and (not causal or known[i] <= up)
              and np.isfinite(high[i])]
        return lo, hi

    if not causal:
        for a, b, is_in in bounds:
            if is_in and not in_pivot_marks:
                continue
            lo, hi = _mark_idx(a, b)
            if len(lo) >= min_peak:
                for k in lis_positions(low[lo]):
                    cross[lo[k]] = 1
            if len(hi) >= min_peak:
                for k in lds_positions(high[hi]):
                    cross[hi[k]] = -1
    else:
        for a, b, is_in in bounds:
            if is_in and not in_pivot_marks:
                continue
            confirm_at = {}
            for i in range(a, b + 1):
                if marks[i] != 0:
                    confirm_at.setdefault(int(known[i]), []).append(i)
            for t in range(a, b + 1):
                newly = confirm_at.get(t)
                if not newly:
                    continue
                lo, hi = _mark_idx(a, b, upto=t)
                new_low = [i for i in newly if marks[i] == 1]
                new_high = [i for i in newly if marks[i] == -1]
                # 一侧峰点 < MIN_PEAKS_SIDE ⇒ 扩窗 2 倍重取（与流式同口径）
                if (new_low and len(hi) < MIN_PEAKS_SIDE) or                         (new_high and len(lo) < MIN_PEAKS_SIDE):
                    a1 = max(0, t - WIDEN_FACTOR * (t - a + 1) + 1)
                    lo, hi = _mark_idx(a1, b, upto=t)
                if new_low and len(hi) >= MIN_PEAKS_SIDE and len(lo) >= min_peak:
                    pos = lis_positions(low[lo])
                    if pos and lo[pos[-1]] in new_low:
                        cross[t] = 1
                elif new_high and len(lo) >= MIN_PEAKS_SIDE and len(hi) >= min_peak:
                    pos = lds_positions(high[hi])
                    if pos and hi[pos[-1]] in new_high:
                        cross[t] = -1
    return cross


def calc_zen_ols_timing_lag(ohlc: pd.DataFrame,
                            pivot_lag=None,
                            peak: str = 'czsc',
                            causal: bool = True,
                            in_pivot_marks: bool = True,
                            min_peak: int = 2) -> pd.Series:
    """逐 bar 的 `ZenOLSLag`。

    :param ohlc: 单标的多重索引帧（须含 open/high/low/close）
    :param pivot_lag: 与 `ohlc` 等长的 `ZenPivotLag`；``None`` ⇒ 全部当成「中枢外」
    :param peak: ``'czsc'`` / ``'scipy'``
    :param causal: ``True`` ⇒ 每根 bar 只用「段起点 … 当前 bar」重算 LIS/LDS
    :param in_pivot_marks: ``True``（默认，用户 2026-11-22）⇒ **中枢内也标交叉**
        （中枢里也试着用 LIS/LDS 判断二乘趋势）；``False`` ⇒ 只在中枢外标
    :param min_peak: 段内至少几个峰点才跑 LIS/LDS（少于此数无法判断趋势）
    :returns: 与 `ohlc` 等长的 float Series（`ZenOLSLag`）
    """
    from GolemQ.analysis.timing import (Timeline_duration,
                                         calc_event_timing_lag)

    n = len(ohlc)
    if n == 0:
        return pd.Series(dtype=float, index=ohlc.index)
    cross = cross_marks(ohlc, pivot_lag=pivot_lag, peak=peak, causal=causal,
                        in_pivot_marks=in_pivot_marks, min_peak=min_peak)

    jx = Timeline_duration((cross == 1).astype(np.int32))
    sx = Timeline_duration((cross == -1).astype(np.int32))
    lag = calc_event_timing_lag(np.where(jx < sx, 1, -1)).astype(np.float32)
    return pd.Series(lag, index=ohlc.index, name='ZenOLSLag')


def calc_zen_ols_timing_lag_streaming(ohlc: pd.DataFrame,
                                      in_pivot_marks: bool = True,
                                      min_peak: int = 2,
                                      debug_window=None,
                                      debug_out=None) -> pd.Series:
    """**流式**版 `ZenOLSLag`：一遍 `CZSC.update` 走到底，逐 bar 因果。

    与 :func:`calc_zen_ols_timing_lag`（逐 target 重建 2160 根窗口）的区别：

    * **口径**：这里的「历史」是从帧头累积到当前 bar（不再是 2160 滚动窗口）；
    * **成本**：O(总 bar 数) 而不是 O(目标数 × 窗口) —— 实测 9,800 根 ~2s，
      逐窗口版是 ~12 分钟；
    * **因果**：`CZSC.update` 在第 T 根只看过前 T 根 ⇒ 天然前缀稳定
      （用户 2026-11-23 选定）。

    实现要点：只在**分型确认**（`bi_list[-1].fx_b` 变化）那一刻重算本段前缀，
    平时 lag 只是逐根 +1。
    """
    from GolemQ.analysis.pivot import (calc_pivots, causal_pivot_series)
    from GolemQ.analysis.timing import (Timeline_duration,
                                         calc_event_timing_lag)
    from GolemQ.analysis.zen import raw_bars, _freq_of

    n = len(ohlc)
    if n == 0:
        return pd.Series(dtype=float, index=ohlc.index)
    code = str(ohlc.index.get_level_values(1)[0])
    ts = pd.DatetimeIndex(pd.to_datetime(ohlc.index.get_level_values(0)))
    bars = raw_bars(ohlc, code, _freq_of(ohlc), strict=False,
                    vol_keys=('volume', 'vol'))
    low = pd.to_numeric(ohlc['low'], errors='coerce').to_numpy(dtype=float)
    high = pd.to_numeric(ohlc['high'], errors='coerce').to_numpy(dtype=float)

    try:
        from czsc import CZSC
        from czsc.objects import Mark
    except Exception:                 # noqa: BLE001 无 czsc ⇒ 全 0
        return pd.Series(np.zeros(n, dtype=float), index=ohlc.index)

    _SEG_TAIL = 2160        # 中枢 lag 每次只重算最近这么多根（见下面的注释）
    pos_of = {t: i for i, t in enumerate(ts)}
    marks = np.zeros(n, dtype=np.int8)
    known = np.arange(n, dtype=np.int64)
    piv_lag = np.zeros(n, dtype=float)
    cross = np.zeros(n, dtype=np.int8)

    czsc = CZSC(bars[:1], max_bi_count=max(5000, n + 100), verbose=False)
    conf = {}
    prev_key = None
    # 增量维护：段编号 / 段起点 / 当前段内的低峰、高峰列表 / 上一根的符号
    _seg_id, s0 = 0, 0
    seg_lo, seg_hi = [], []
    _prev_sign = _sign(piv_lag[0]) if n else 0
    for i in range(1, n):
        czsc.update(bars[i])
        bi = czsc.bi_list
        _new_fx = None
        if bi:
            key = bi[-1].fx_b.dt
            if key != prev_key:
                prev_key = key
                conf[key] = bars[i].dt
                j0 = pos_of.get(pd.Timestamp(key))
                if j0 is not None:
                    marks[j0] = 1 if bi[-1].fx_b.mark == Mark.D else -1
                    known[j0] = i
                    _new_fx = j0
        if _new_fx is None:
            # 没有新分型 ⇒ 中枢状态不变，lag 逐根 +1（省掉每 bar 重算）
            if i > 0 and piv_lag[i - 1] != 0:
                piv_lag[i] = piv_lag[i - 1] + np.sign(piv_lag[i - 1])
        else:
            _a = max(0, i + 1 - _SEG_TAIL)
            try:
                piv = calc_pivots(bi_list=list(bi), conf_map=conf, strict=True)
                _zd = causal_pivot_series(piv, ts[_a:i + 1], strict=True)[0]
                piv_lag[_a:i + 1] = calc_event_timing_lag(
                    np.where(~np.isnan(np.asarray(_zd, dtype=float)), 1, -1))
            except Exception:         # noqa: BLE001 无中枢 ⇒ 保持上一状态
                pass

        # ② 每根 bar 都评估：段内已确认的峰点 + **当前 bar 自己**（用户 2026-11-23：
        # 「最后一 bar 也要同时加入计算作为高点或者低点」——否则最新那根永远不算峰，
        # 信号必然滞后一个分型确认周期）。
        # ⚠️ 段与峰点表都**增量维护**：`segment_ids()` 的写法是 O(前缀长度)，9,319 根
        # 逐 bar 调一次 = O(n²)（实测 3.7s → 30.7s）。这里只按上一根推一步。
        _sg = _sign(piv_lag[i])
        if _sg != 0 and _sg != _prev_sign:
            _seg_id += 1
            s0 = i
            seg_lo, seg_hi = [], []
            _prev_sign = _sg
        if _new_fx is not None:
            if marks[_new_fx] == 1:
                seg_lo.append(_new_fx)
            else:
                seg_hi.append(_new_fx)
        if not in_pivot_marks and piv_lag[i] > 0:
            continue
        # 当前 bar 同时作为**两侧候选**（用户 2026-11-23：「最后一 bar 也要加入」），
        # 但方向由**哪条链更长**决定 —— 不能简单让 +1 分支优先（实测正占比会
        # 从 44% 飙到 80%），也不能按「比上一根高低」判侧（实测翻转 1,683 次）。
        _lo_cand = np.isfinite(low[i])
        _hi_cand = np.isfinite(high[i])
        lo = seg_lo + ([i] if (_lo_cand and i not in seg_lo) else [])
        hi = seg_hi + ([i] if (_hi_cand and i not in seg_hi) else [])
        # 一侧峰点 < MIN_PEAKS_SIDE ⇒ 这一浪还没成形：把取样窗按段长放大
        # 2 倍重取，仍不足则**不出该方向**（用户 2026-11-23 口径）
        if len(hi) < MIN_PEAKS_SIDE or len(lo) < MIN_PEAKS_SIDE:
            _span = i - s0 + 1
            s1 = max(0, i - WIDEN_FACTOR * _span + 1)
            lo2, hi2 = _collect_peaks(marks, low, high, s1, i)
            if _lo_cand and i not in lo2:
                lo2 = lo2 + [i]
            if _hi_cand and i not in hi2:
                hi2 = hi2 + [i]
            if len(hi2) > len(hi) or len(lo2) > len(lo):
                lo, hi = lo2, hi2
        _up_ok = len(hi) >= MIN_PEAKS_SIDE and len(lo) >= min_peak
        _dn_ok = len(lo) >= MIN_PEAKS_SIDE and len(hi) >= min_peak
        _up_len = _dn_len = 0
        if _up_ok:
            pos = lis_positions(low[lo])
            if pos and lo[pos[-1]] == i:
                _up_len = len(pos)
        if _dn_ok:
            pos = lds_positions(high[hi])
            if pos and hi[pos[-1]] == i:
                _dn_len = len(pos)
        # 两边都成立时比链长；相等则保持不动（不制造来回翻）
        if _up_len > _dn_len + MARGIN:
            cross[i] = 1
        elif _dn_len > _up_len + MARGIN:
            cross[i] = -1
        if debug_window is not None and debug_out is not None:
            _t = ts[i]
            if debug_window[0] <= _t <= debug_window[1]:
                debug_out.append({'ts': _t, 'seg_start': ts[s0],
                                  'seg_len': i - s0 + 1, 'piv_lag': piv_lag[i],
                                  'n_lo': len(lo), 'n_hi': len(hi),
                                  'up_ok': _up_ok, 'dn_ok': _dn_ok,
                                  'up_len': _up_len, 'dn_len': _dn_len,
                                  'cross': int(cross[i])})

    jx = Timeline_duration((cross == 1).astype(np.int32))
    sx = Timeline_duration((cross == -1).astype(np.int32))
    lag = calc_event_timing_lag(np.where(jx < sx, 1, -1)).astype(np.float32)
    return pd.Series(lag, index=ohlc.index, name='ZenOLSLag')
