# coding: utf-8
"""60min 额外两列：样本熵 `se` + 因果中枢计数 `ZenPivotCnt`（D78，撤销 D74）。

口径（用户 2026-10-20 指定，2026-10-30 明确**回到 60min**）：

* `se`：60min 收盘价的 `antropy.sample_entropy`，只在 **15:00** 那根算，
  窗口为截至该 bar 的最后 **≤2160 根**，`round(..., 3)`；盘中/更早的 bar
  沿用**上一交易日 15:00** 的值（因果 ffill，**不**复刻旧树把当日 15:00 值
  回填到当日更早 bar 的未来函数）。
* `ZenPivotCnt`：截至当前 bar **已确认**的缠论中枢个数 —— CZSC 笔 + 确认时刻
  （`bi_confirm_map` / `calc_pivots` / `causal_pivot_count`）。

本模块只放**纯计算**；调用方是 `steps/zen.py`（与 zen60/minor 同趟算完，
避免再起一趟 step）。D74 曾把这两列挪到日族（日收盘 SE + 15:00 取样），
用户 2026-10-30 判定不妥，改回 60min 逐 bar 落 `*_reality_feat_60min`。
"""
from __future__ import annotations

from datetime import time as dtime

import numpy as np
import pandas as pd

from GolemQ.core.constants import AKA
from GolemQ.core.constants import FIELD as FLD

SE_WINDOW = 2160
CZSC_MAX_BI = 512
_DAY_END = dtime(15, 0)

__all__ = ['SE_WINDOW', 'sample_entropy_by_day', 'pivot_counts',
           'renko_by_day', 'native_by_day', 'pivot_stats', 'chip_cost_by_day']


def sample_entropy_by_day(ohlc: pd.DataFrame, targets=None):
    """返回 15:00 bar 的 `(timestamps, se)`；只算调用方已切好的帧。

    `targets` 给了就**只算目标实际会用到的那几个 15:00**：
    `se` 是「截至该 bar 最后 ≤2160 根」的样本熵，调用方按
    `searchsorted(T, side='right') - 1` 取值 —— 每个目标只依赖**它之前最近的
    一个 15:00**，所以无需把 2016 起全部 2,600 个 15:00 都算一遍
    （实测 15s/码 → ~3s/码，值逐位相同）。
    """
    try:
        import antropy as ant
    except Exception:               # noqa: BLE001 环境缺 antropy ⇒ 全 NaN
        return pd.DatetimeIndex([]), np.array([], dtype=float)
    if ohlc is None or len(ohlc) == 0:
        return pd.DatetimeIndex([]), np.array([], dtype=float)
    ts = pd.DatetimeIndex(pd.to_datetime(ohlc.index.get_level_values(0)))
    close = pd.to_numeric(ohlc[AKA.CLOSE], errors='coerce')
    bars15 = ts[ts.time == _DAY_END]
    if targets is not None and len(bars15):
        cut = pd.DatetimeIndex(pd.to_datetime(targets))
        pos = bars15.searchsorted(cut, side='right') - 1
        pos = np.unique(pos[pos >= 0])
        bars15 = bars15[pos]
    days, vals = [], []
    for t in bars15:
        win = close.loc[ts <= t].tail(SE_WINDOW).dropna().to_numpy()
        if len(win) < 2:
            continue
        try:
            v = ant.sample_entropy(win)
        except Exception:           # noqa: BLE001 常数序列等
            v = np.nan
        if np.isfinite(v):
            days.append(t)
            vals.append(round(float(v), 3))
    return pd.DatetimeIndex(days), np.asarray(vals, dtype=float)


def pivot_counts(ohlc: pd.DataFrame, targets, code: str) -> np.ndarray:
    """截至**当日 15:00**、用**最近 2160 根** 60min 算出的中枢个数；失败返回 NaN。

    口径（用户 2026-10-30 明确）：
    * 与 `se` **同一个窗口**：取目标日 15:00 往前 **≤2160 根**（2024 年的 bar
      ⇒ 数据从 2021 年底起）；
    * **as-of**：窗口里只放 ≤T 的 bar，从窗口第一根起按顺序建 CZSC，禁止
      「全历史跑一次再取 T」—— 实测 600519 2024-08-16 两种做法是 **47 vs 44**
      （全量跑完会改写早期笔/中枢），`--review` 就是为消除这种后视镜扰动；
    * 每天 4 根 60min 共用当天 15:00 的值（用户允许「跳 4」）。

    代价：每个目标日一遍 2160 根窗口；实测约 **25~40ms/日**。
    """
    if ohlc is None or len(ohlc) < 30 or targets is None or len(targets) == 0:
        return np.full(len(targets), np.nan)
    try:
        from czsc import CZSC
        from czsc.objects import Freq, RawBar

        from GolemQ.analysis.pivot import calc_pivots, causal_pivot_count
        ts = pd.DatetimeIndex(pd.to_datetime(ohlc.index.get_level_values(0)))
        cut = pd.DatetimeIndex(pd.to_datetime(targets))
        bars15 = ts[ts.time == _DAY_END]
        if len(bars15) == 0:
            return np.full(len(targets), np.nan)
        pos = bars15.searchsorted(cut, side='right') - 1
        need = sorted(set(bars15[pos[pos >= 0]]))
        if not need:
            return np.full(len(targets), np.nan)

        bars_all = []
        for i, (t, (_, row)) in enumerate(zip(ts, ohlc.iterrows())):
            bars_all.append(RawBar(
                symbol=code, dt=t, id=i, freq=Freq.F60,
                open=float(row.get('open', np.nan)),
                close=float(row.get('close', np.nan)),
                high=float(row.get('high', np.nan)),
                low=float(row.get('low', np.nan)),
                vol=float(row.get('volume', row.get('vol', 0)) or 0),
                amount=float(row.get('amount', 0) or 0),
            ))
        idx_of = {t: i for i, t in enumerate(ts)}
        by_day = {}
        for T in need:
            j = idx_of[T]
            w = bars_all[max(0, j - SE_WINDOW + 1): j + 1]
            czsc = CZSC(w[:1], max_bi_count=CZSC_MAX_BI, verbose=False)
            conf: dict = {}
            prev = None
            for bar in w[1:]:
                czsc.update(bar)
                if czsc.bi_list:
                    key = czsc.bi_list[-1].fx_b.dt
                    if key != prev:
                        conf[key] = bar.dt
                        prev = key
            piv = calc_pivots(bi_list=list(czsc.bi_list),
                              conf_map=conf, strict=True)
            by_day[T] = int(causal_pivot_count(
                piv, pd.DatetimeIndex([T]), strict=True)[0])
        days = pd.DatetimeIndex(sorted(by_day))
        p = days.searchsorted(cut, side='right') - 1
        out = np.full(len(targets), np.nan)
        ok = p >= 0
        out[ok] = np.array([by_day[d] for d in days[p[ok]]], dtype=float)
        out[cut < ts[0]] = np.nan
        return out
    except Exception:               # noqa: BLE001 CZSC 失败不该毁掉整条管线
        return np.full(len(targets), np.nan)


def renko_by_day(ohlc: pd.DataFrame, targets):
    """每个目标：用 **≤T 的最近一个 15:00 快照**、**2160 根窗口**跑一次 Renko。

    与 `se` / `ZenPivotCnt` **同窗口、同 as-of 纪律**（用户 2026-10-30：
    三者是一体的，Renko 不再放日线）。盘中 3 根沿用上一交易日 15:00 的值。

    返回 `(days, frame)`：`days` 是快照时间轴，`frame` 是 Renko 14 列（价格列
    仍是**绝对价**，落库前由调用方按目标行 close 转 `*D` 比值；不同目标行的
    close 不同，所以不能在快照里预先除）。
    """
    from GolemQ.analysis.renko_jit import (available,
                                           renko_trend_cross_func_jit)
    from GolemQ.pipeline.review.contract import (RENKO_DAY_COLUMNS,
                                                 RENKO_PRICE_DUMMIES)
    empty = pd.DataFrame(columns=list(RENKO_DAY_COLUMNS))
    if (not available() or ohlc is None or len(ohlc) < 30
            or targets is None or len(targets) == 0):
        return pd.DatetimeIndex([]), empty
    ts = pd.DatetimeIndex(pd.to_datetime(ohlc.index.get_level_values(0)))
    cut = pd.DatetimeIndex(pd.to_datetime(targets))
    bars15 = ts[ts.time == _DAY_END]
    if len(bars15) == 0:
        return pd.DatetimeIndex([]), empty
    pos = bars15.searchsorted(cut, side='right') - 1
    need = sorted(set(bars15[pos[pos >= 0]]))
    rows, days = [], []
    for T in need:
        w = ohlc.loc[ts <= T].tail(SE_WINDOW)
        try:
            rk = renko_trend_cross_func_jit(w)
        except Exception:            # noqa: BLE001 Renko 失败不该毁掉 review
            continue
        if rk is None or len(rk) == 0:
            continue
        last = rk.iloc[-1]
        # ⚠️ 价格列在契约里是 `*D`（比值），这里先放**绝对价**，
        # 由调用方按目标行 close 转比值（每行 close 不同）。
        row = {dummy: last.get(base, np.nan)
               for base, dummy in RENKO_PRICE_DUMMIES}
        for c in RENKO_DAY_COLUMNS:
            row.setdefault(c, last.get(c, np.nan))
        rows.append([row[c] for c in RENKO_DAY_COLUMNS])
        days.append(T)
    if not days:
        return pd.DatetimeIndex([]), empty
    return (pd.DatetimeIndex(days),
            pd.DataFrame(rows, index=pd.DatetimeIndex(days),
                         columns=list(RENKO_DAY_COLUMNS)))


def native_by_day(ohlc: pd.DataFrame, targets):
    """每个交易日 15:00：用 **≤T 的 2160 根 60min** 算一遍 60min 原生特征。

    用户 2026-10-30：交易日尾部按 60min 实际时间计算，日频版另行采样。
    产出 `HOUR_NATIVE_COLUMNS` 10 列（M9T 2 + Poly9 2 + RegTree 3 + Prophet 3）；
    价格量纲列按**快照所在 bar** 的 close 转比值。
    """
    from GolemQ.analysis.fbprophet import calc_fbprophet_bands  # noqa: F401
    from GolemQ.pipeline.review.contract import HOUR_NATIVE_COLUMNS
    from GolemQ.pipeline.review.coldstart import _core_daily, _prophet_columns
    from GolemQ.pipeline.review.steps.daily import _frame_features
    empty = pd.DataFrame(columns=list(HOUR_NATIVE_COLUMNS))
    if ohlc is None or len(ohlc) < 30 or targets is None or len(targets) == 0:
        return pd.DatetimeIndex([]), empty
    ts = pd.DatetimeIndex(pd.to_datetime(ohlc.index.get_level_values(0)))
    cut = pd.DatetimeIndex(pd.to_datetime(targets))
    bars15 = ts[ts.time == _DAY_END]
    if len(bars15) == 0:
        return pd.DatetimeIndex([]), empty
    pos = bars15.searchsorted(cut, side='right') - 1
    need = sorted(set(bars15[pos[pos >= 0]]))
    # STAGE 全帧一次算（与 `tools/backfill_stage.py` 同口径）：
    # 它自带 502 根定带窗口、逐 checkpoint 因果，不必按天重建 2160 窗口。
    from GolemQ.pipeline.review.coldstart import calc_stage_series
    _stage_all = calc_stage_series(ohlc)
    _ts_all = pd.DatetimeIndex(pd.to_datetime(ohlc.index.get_level_values(0)))
    stage_by_ts = dict(zip(_ts_all, pd.to_numeric(_stage_all,
                                                  errors='coerce').to_numpy()))
    days, rows = [], []
    for T in need:
        w = ohlc.loc[ts <= T].tail(SE_WINDOW)
        if len(w) < 30:
            continue
        try:
            feats = _frame_features(w, annual=4032)
            feats = _core_daily(w, feats)      # STAGE（+ garP 等伴随列）
            feats = _prophet_columns(w, feats)
        except Exception:           # noqa: BLE001 单点失败不毁整条链
            continue
        close_last = float(pd.to_numeric(w[AKA.CLOSE], errors='coerce').iloc[-1])
        last = feats.iloc[-1]
        row = {
            'stage': stage_by_ts.get(T, np.nan),
            'm9t': last.get('m9t', np.nan),
            'm9tLag': last.get('m9tLag', np.nan),
            'poly9D': (last.get('poly9', np.nan) / close_last
                       if close_last else np.nan),
            'poly9NorR': last.get('poly9pwr3', np.nan),
            'rTrePrcD': (last.get('regtreePrice', np.nan) / close_last
                         if close_last else np.nan),
            'rTreLagR': last.get('regtreeLag', np.nan),
            'rTreSlopD': (last.get('regtreeSlp', np.nan) / close_last
                          if close_last else np.nan),
            'prophetLbD': last.get(FLD.FBPROPHET_LB, np.nan),
            'prophetUbD': last.get(FLD.FBPROPHET_UB, np.nan),
            'prophetClrNorm': last.get(FLD.FBPROPHET_CLEARANCE_NORM, np.nan),
        }
        days.append(T)
        rows.append([row.get(c, np.nan) for c in HOUR_NATIVE_COLUMNS])
    if not days:
        return pd.DatetimeIndex([]), empty
    return (pd.DatetimeIndex(days),
            pd.DataFrame(rows, index=pd.DatetimeIndex(days),
                         columns=list(HOUR_NATIVE_COLUMNS)))


def pivot_stats(ohlc: pd.DataFrame, targets, code: str):
    """每个目标：`ZenPivotCnt` + `ZenPivotLag`（2160 根 as-of 窗口）。

    `ZenPivotLag` = 距**出中枢**（`ZenPivotZD` 变 NaN）以来的 lag —— 与
    `ZenPivotCnt` 用同一批 CZSC 中枢，`causal_pivot_series` 展开成逐 bar 的
    ZD 后做事件 lag（用户 2026-10-30：中枢还有一个 Lag，别漏）。
    """
    from GolemQ.analysis.pivot import (calc_pivots, causal_pivot_count,
                                       causal_pivot_series)
    from GolemQ.analysis.timing import calc_event_timing_lag
    from GolemQ.models.alias import ZEN
    cols = [ZEN.PIVOT_COUNT, ZEN.PIVOT_TIMING_LAG]
    empty = pd.DataFrame(columns=cols)
    if ohlc is None or len(ohlc) < 30 or targets is None or len(targets) == 0:
        return pd.DatetimeIndex([]), empty
    try:
        from czsc import CZSC
        from czsc.objects import Freq, RawBar
        ts = pd.DatetimeIndex(pd.to_datetime(ohlc.index.get_level_values(0)))
        cut = pd.DatetimeIndex(pd.to_datetime(targets))
        bars15 = ts[ts.time == _DAY_END]
        if len(bars15) == 0:
            return pd.DatetimeIndex([]), empty
        pos = bars15.searchsorted(cut, side='right') - 1
        need = sorted(set(bars15[pos[pos >= 0]]))
        bars_all = []
        for i, (t, (_, row)) in enumerate(zip(ts, ohlc.iterrows())):
            bars_all.append(RawBar(
                symbol=code, dt=t, id=i, freq=Freq.F60,
                open=float(row.get('open', np.nan)),
                close=float(row.get('close', np.nan)),
                high=float(row.get('high', np.nan)),
                low=float(row.get('low', np.nan)),
                vol=float(row.get('volume', row.get('vol', 0)) or 0),
                amount=float(row.get('amount', 0) or 0),
            ))
        idx_of = {t: i for i, t in enumerate(ts)}
        days, rows = [], []
        for T in need:
            j = idx_of[T]
            w = bars_all[max(0, j - SE_WINDOW + 1): j + 1]
            czsc = CZSC(w[:1], max_bi_count=CZSC_MAX_BI, verbose=False)
            conf: dict = {}
            prev = None
            for bar in w[1:]:
                czsc.update(bar)
                if czsc.bi_list:
                    key = czsc.bi_list[-1].fx_b.dt
                    if key != prev:
                        conf[key] = bar.dt
                        prev = key
            piv = calc_pivots(bi_list=list(czsc.bi_list), conf_map=conf,
                              strict=True)
            cnt = int(causal_pivot_count(piv, pd.DatetimeIndex([T]),
                                         strict=True)[0])
            dts = ts[max(0, j - SE_WINDOW + 1): j + 1]
            zd = causal_pivot_series(piv, dts, strict=True)[0]
            lag = int(calc_event_timing_lag(
                np.where(~np.isnan(np.asarray(zd, dtype=float)), 1, -1))[-1])
            days.append(T)
            rows.append([cnt, lag])
        if not days:
            return pd.DatetimeIndex([]), empty
        return (pd.DatetimeIndex(days),
                pd.DataFrame(rows, index=pd.DatetimeIndex(days), columns=cols))
    except Exception:               # noqa: BLE001
        return pd.DatetimeIndex([]), empty


def chip_cost_by_day(day: pd.DataFrame, targets, code: str, turnover=None):
    """ChipDist：**960 根日线窗口**、逐日 as-of，返回成本价（绝对价）四列。

    ⚠️ 两个关键点（2026-10-30 实测）：

    1. `ChipDistribution.get_data()` 只在**自己重新取数**时做 `adj` 缩放
       （按区间 ÷100/50/10/5）。直接传 960 窗口必须**自己补做**，否则
       分箱数爆炸（38s）且数值错（1177.81 vs 正确 1222）；
    2. 传 `len(data) > 1920` 会触发模块**重新取数**（`realtime=True`，
       含未来）—— 所以这里固定只传 960 根，绝不超过 1920。
    """
    from GolemQ.core.constants import FIELD as FLD
    cols = ('cost5Prz', 'cost15Prz', 'cost85Prz', 'cost95Prz')
    empty = pd.DataFrame(columns=list(cols))
    if day is None or len(day) < 30 or targets is None or len(targets) == 0:
        return pd.DatetimeIndex([]), empty
    day = day[~day.index.duplicated(keep='last')].sort_index()
    ts = pd.DatetimeIndex(pd.to_datetime(
        day.index.get_level_values(0))).normalize()
    day_idx = pd.DatetimeIndex(sorted(set(ts)))
    cut = pd.DatetimeIndex(pd.to_datetime(targets))

    need = set()
    for T in cut:
        d = T.normalize()
        if T.time() >= _DAY_END:
            key = d if d in day_idx else None
        else:
            prev = day_idx[day_idx < d]
            key = prev[-1] if len(prev) else None
        if key is not None:
            need.add(key)
    if not need:
        return pd.DatetimeIndex([]), empty

    from GolemQ.analysis.ChipDistribution_jit import (
        calc_stock_chip_distribution_jit)
    from GolemQ.core.constants import AKA
    by_day = {}
    for key in sorted(need):
        w = day.loc[ts <= key].tail(960).copy()
        if len(w) < 30:
            continue
        if turnover is not None:
            w[FLD.TURNOVER_RATE] = turnover.reindex(w.index)
            if w[FLD.TURNOVER_RATE].isna().any():
                # ⚠️ 缺**个别交易日**换手率时，从窗口里剔除那一天，不要整段
                # `continue`。960 根窗口少 1~2 天对成本分布影响可忽略；整段
                # 跳过会让该码所有 target 都没有 cost*PrzD（000711 因为
                # 2023-12-15 缺换手率，2026 年的四个成本列全空）。
                w = w[w[FLD.TURNOVER_RATE].notna()]
                if len(w) < 30:
                    continue
        close = pd.to_numeric(w[AKA.CLOSE], errors='coerce')
        rng = float(close.max() - close.min())
        adj = (100.0 if rng > 648 else 50.0 if rng > 262 else
               10.0 if rng > 83.2 else 5.0 if rng > 51.2 else 1.0)
        for c in (AKA.OPEN, AKA.HIGH, AKA.LOW, AKA.CLOSE):
            w[c] = pd.to_numeric(w[c], errors='coerce') / adj
        try:
            out = calc_stock_chip_distribution_jit(
                features=pd.DataFrame(index=w.index), ohlc_data=w, code=code)
            row = out.loc[(key, code)]
        except Exception:           # noqa: BLE001 单日失败留空
            continue
        by_day[key] = {c: (row.get(c, np.nan) * adj) for c in cols}
    if not by_day:
        return pd.DatetimeIndex([]), empty
    days = pd.DatetimeIndex(sorted(by_day))
    return days, pd.DataFrame([by_day[d] for d in days], index=days,
                              columns=list(cols))
