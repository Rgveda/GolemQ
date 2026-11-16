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

__all__ = ['SE_WINDOW', 'sample_entropy_by_day',
           'renko_by_day', 'native_by_day', 'pivot_stats', 'chip_cost_by_day',
           'build_hour_extra_frame', 'build_chip_frame', 'renko_ratio_frame',
           'asof_close_by_ts',
           'chip_ratios_from_abs', 'sample_day_frame']


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
    产出 `HOUR_NATIVE_COLUMNS`（M9T 2 + Poly9 2 + RegTree 3 + Prophet 3），**全部不带 `Maj`**。带 `Maj` 的日频版本由
    `day_core` 计算、`day_sample_frame` 按日采样，二者不是同一列。
    价格量纲列按**快照所在 bar** 的 close 转比值。
    """
    from GolemQ.analysis.fbprophet import calc_fbprophet_bands  # noqa: F401
    from GolemQ.pipeline.review.contract import HOUR_NATIVE_COLUMNS
    from GolemQ.pipeline.review.coldstart import _prophet_columns
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
    days, rows = [], []
    for T in need:
        w = ohlc.loc[ts <= T].tail(SE_WINDOW)
        if len(w) < 30:
            continue
        try:
            # 只算 60min 原生（无 Maj）列。日频 `*Maj*` 由 `day_core` 负责，
            # 再由 `day_sample_frame` 按日采样到 60min —— 这里**不要**调
            # `_core_daily`：它会逐日重算并生成本函数用不到的 Maj 列。
            # ⚠️ stage 族已改为 **4.4 搬运**（用户 2026-11-16），本函数不再产出。
            feats = _frame_features(w, annual=4032, with_zen=False)
            feats = _prophet_columns(w, feats)
        except Exception:           # noqa: BLE001 单点失败不毁整条链
            continue
        close_last = float(pd.to_numeric(w[AKA.CLOSE], errors='coerce').iloc[-1])
        last = feats.iloc[-1]
        row = {
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


# --------------------------------------------------------------------------- #
# 共享映射层（review step 与 backfill 工具**同一份**，别再各写一遍）
# --------------------------------------------------------------------------- #
def _target_positions(days, targets):
    """`days` 是**已排序**的「最后可用快照」轴；返回每个 target 的 `<=` 位置。"""
    days = pd.DatetimeIndex(pd.to_datetime(days))
    cut = pd.DatetimeIndex(pd.to_datetime(targets))
    if len(days) == 0:
        return np.full(len(cut), -1, dtype=int), days
    return days.searchsorted(cut, side='right') - 1, days


def map_last_scalar(days, values, targets):
    """把按日快照的 1-D 值映射到 target（每个 target 取 `<=` 的最后一个）。"""
    pos, _ = _target_positions(days, targets)
    vals = np.asarray(values, dtype=float)
    out = np.full(len(pos), np.nan)
    ok = (pos >= 0) & (pos < len(vals))
    out[ok] = vals[pos[ok]]
    return out


def map_last_rows(days, frame, targets):
    """把按日快照的 2-D 帧映射到 target（逐行取 `<=` 的最后一个）。

    ⚠️ **逐列转型**：stage 族里有 `garPD`（带的比值列表）与 `garPAnchor`
    （日期字符串），整帧 `to_numpy(dtype=float)` 会炸；数值列仍走 float，
    其余列按 object 原样搬运（下游按列语义消费）。
    """
    pos, _ = _target_positions(days, targets)
    ncols = 0 if frame is None else len(frame.columns)
    if frame is None or len(frame) == 0 or ncols == 0:
        return np.full((len(pos), ncols), np.nan)
    out = np.full((len(pos), ncols), np.nan, dtype=object)
    ok = (pos >= 0) & (pos < len(frame))
    for j in range(ncols):
        col = frame.iloc[:, j]
        try:
            arr = pd.to_numeric(col, errors='raise').to_numpy(dtype=float)
        except (TypeError, ValueError):
            arr = col.to_numpy(dtype=object)
        out[ok, j] = arr[pos[ok]]
    return out


def map_price_rows(days, frame, targets, close_by_ts, price_columns):
    """同 :func:`map_last_rows`，但 `price_columns` 按**该目标的 as-of 收盘**转比值。

    ⚠️ `close_by_ts` 必须是 :func:`asof_close_by_ts` 的结果（盘中 = 上一交易日
    收盘），不是目标 bar 自己的 close —— 否则同一份快照被 4 根 bar 各除一次，
    比值随盘中价格漂移（F8）。
    """
    out = map_last_rows(days, frame, targets)
    if frame is None or len(frame) == 0 or len(out) == 0:
        return out
    cols = list(frame.columns)
    price_pos = [j for j, c in enumerate(cols) if c in price_columns]
    if not price_pos:
        return out
    for i, T in enumerate(pd.DatetimeIndex(pd.to_datetime(targets))):
        cT = close_by_ts.get(pd.Timestamp(T), np.nan)
        if not (cT and np.isfinite(cT)):
            out[i, price_pos] = np.nan
        else:
            out[i, price_pos] = out[i, price_pos] / cT
    return out


def _close_by_ts(ohlc: pd.DataFrame):
    if ohlc is None or len(ohlc) == 0:
        return pd.Series(dtype=float)
    idx = pd.DatetimeIndex(pd.to_datetime(ohlc.index.get_level_values(0)))
    vals = pd.to_numeric(ohlc[AKA.CLOSE], errors='coerce').to_numpy()
    s = pd.Series(vals, index=idx)
    # 重复 bar 只保留最后一根（与 tool 的 `~duplicated(keep='last')` 一致）
    return s[~s.index.duplicated(keep='last')]


def asof_close_by_ts(ohlc: pd.DataFrame):
    """逐 bar close → **as-of 收盘**（盘中 = 上一交易日收盘）。F8。

    比值列的分母必须与分子的快照同一天锚；实现收口在
    `reality_save.asof_close`（restore 侧用的是同一份规则），这里只做
    「ohlc 帧 → close 序列」的转换，**不许再写一份**。
    """
    from GolemQ.markets.StockCN.reality_save import asof_close
    return asof_close(_close_by_ts(ohlc))


def renko_ratio_frame(ohlc: pd.DataFrame, targets, code: str) -> pd.DataFrame:
    """Renko 14 列 → 60min 目标行（6 个价格列 ÷ **as-of 收盘**）。

    review 与回填工具共用；`tools/backfill_hour_extra.py --ratios-only`
    就靠它重建存量比值。
    """
    from GolemQ.pipeline.review.contract import (RENKO_DAY_COLUMNS,
                                                 RENKO_PRICE_DUMMIES)
    targets = pd.DatetimeIndex(pd.to_datetime(targets))
    idx = pd.MultiIndex.from_arrays(
        [targets, [str(code)] * len(targets)], names=['ts', 'code'])
    rk_days, rk_df = renko_by_day(ohlc, targets)
    rk = map_price_rows(rk_days, rk_df, targets, asof_close_by_ts(ohlc),
                        {d for _b, d in RENKO_PRICE_DUMMIES})
    return pd.DataFrame(rk, index=idx, columns=list(RENKO_DAY_COLUMNS))


def build_hour_extra_frame(ohlc: pd.DataFrame, targets, code: str) -> pd.DataFrame:
    """`se` + `ZenPivotCnt/Lag` + Renko 14 + 60min 原生 10 列的**唯一构造器**。

    口径与 `steps/zen.py::zen_snapshots_step`、`tools/backfill_hour_extra.py`
    完全一致；两边都调这里，禁止再各自写 searchsorted 映射。
    """
    from GolemQ.pipeline.review.contract import HOUR_NATIVE_COLUMNS
    targets = pd.DatetimeIndex(pd.to_datetime(targets))
    idx = pd.MultiIndex.from_arrays(
        [targets, [str(code)] * len(targets)], names=['ts', 'code'])
    frame = pd.DataFrame(index=idx)

    se_days, se_vals = sample_entropy_by_day(ohlc, targets)
    frame['se'] = map_last_scalar(se_days, se_vals, targets)

    piv_days, piv_df = pivot_stats(ohlc, targets, code)
    piv = map_last_rows(piv_days, piv_df, targets)
    for j, c in enumerate(piv_df.columns):
        frame[c] = piv[:, j]

    rk = renko_ratio_frame(ohlc, targets, code)
    for c in rk.columns:
        frame[c] = rk[c].to_numpy()

    nat_days, nat_df = native_by_day(ohlc, targets)
    nat = map_last_rows(nat_days, nat_df, targets)
    for j, c in enumerate(HOUR_NATIVE_COLUMNS):
        frame[c] = nat[:, j]
    return frame


def chip_ratios_from_abs(chip_abs, targets, close_by_ts,
                         code: str = '') -> pd.DataFrame:
    """绝对成本（按日）→ 每个 target 的 `cost*PrzD` 比值（÷目标行 close）。

    `chip_abs` 的 key 是**归一化日**。⚠️ `close_by_ts` 必须是
    :func:`asof_close_by_ts` 的结果（盘中 = 上一交易日收盘，F8），
    否则同一份分子被 4 根 bar 各除一次。

    选中规则必须与 :func:`chip_cost_by_day` 一致（F2，2026-11-16 复审）：

    * **15:00** 目标 ⇒ 当日（当日收盘已成立）；
    * **盘中**（10:30 / 11:30 / 14:00）⇒ **严格早于当日的最后一个**可用日。

    以前一律 `searchsorted(normalize, side='right') - 1`：只要 `chip_abs` 里有
    当日 key（15:00 目标必然产生它），盘中三根也会取到**当日收盘**算出的成本
    ——未来函数。实测 000711 2025-12-08 10:30 的 `cost15Prz` 分子是 1.73（当日），
    应为前一交易日 1.72。
    """
    from GolemQ.pipeline.review.contract import HOUR_CHIP_COLUMNS
    targets = pd.DatetimeIndex(pd.to_datetime(targets))
    idx = pd.MultiIndex.from_arrays(
        [targets, [str(code)] * len(targets)], names=['ts', 'code'])
    vals = np.full((len(targets), len(HOUR_CHIP_COLUMNS)), np.nan)
    if chip_abs:
        days = pd.DatetimeIndex(sorted({pd.Timestamp(d).normalize()
                                        for d in chip_abs}))
        day = targets.normalize()
        at_close = np.array([t.time() >= _DAY_END for t in targets], dtype=bool)
        pos = np.empty(len(targets), dtype=int)
        pos[at_close] = days.searchsorted(day[at_close], side='right') - 1
        pos[~at_close] = days.searchsorted(day[~at_close], side='left') - 1
        srcs = ('cost5Prz', 'cost15Prz', 'cost85Prz', 'cost95Prz')
        for i, p in enumerate(pos):
            if p < 0:
                continue
            cT = (close_by_ts.get(pd.Timestamp(targets[i]), np.nan)
                  if close_by_ts is not None else np.nan)
            if not (cT and np.isfinite(cT)):
                continue
            row = chip_abs[days[p]]
            for k, src in enumerate(srcs):
                v = row.get(src, np.nan)
                vals[i, k] = (v / cT) if np.isfinite(v) else np.nan
    return pd.DataFrame(vals, index=idx, columns=list(HOUR_CHIP_COLUMNS))


def build_chip_frame(day: pd.DataFrame, targets, code: str,
                     turnover=None, close_by_ts=None) -> pd.DataFrame:
    """ChipDist 绝对成本 → 60min 目标行的 `HOUR_CHIP_COLUMNS`（唯一入口）。"""
    days, chip_df = chip_cost_by_day(day, targets, code, turnover=turnover)
    chip_abs = {pd.Timestamp(d).normalize(): chip_df.loc[d].to_dict()
                for d in days}
    return chip_ratios_from_abs(chip_abs, targets, close_by_ts, code=code)


def sample_day_frame(cache, targets, close_by_ts=None, chip_abs=None):
    """把按日缓存的日族行采样到 target（15:00 只认当日，盘中认上一交易日）。

    `cache` 的 key 是归一化日，value 是列名→值；`chip_abs` 可选，给了就按
    `chip_ratios_from_abs` 写 `cost*PrzD`（不再把日表的 ÷日close 值带到 60min）。
    """
    from GolemQ.pipeline.review.contract import (DAY_SAMPLE_COLUMNS,
                                                 HOUR_CHIP_COLUMNS)
    targets = pd.DatetimeIndex(pd.to_datetime(targets))
    idx = pd.MultiIndex.from_arrays(
        [targets, [''] * len(targets)], names=['ts', 'code'])
    nan_row = {c: np.nan for c in DAY_SAMPLE_COLUMNS}
    dates = pd.DatetimeIndex(sorted({pd.Timestamp(d).normalize()
                                     for d in (cache or {})}))
    rows = []
    for T in targets:
        d = T.normalize()
        if T.time() >= _DAY_END:
            key = d if d in (cache or {}) else None
        else:
            pos = int(dates.searchsorted(d, side='left')) - 1
            key = dates[pos] if pos >= 0 else None
        row = (cache or {}).get(key, nan_row) if key is not None else nan_row
        rows.append({c: row.get(c, np.nan) for c in DAY_SAMPLE_COLUMNS})
    out = pd.DataFrame(rows, index=idx, columns=list(DAY_SAMPLE_COLUMNS))
    if chip_abs:
        chip = chip_ratios_from_abs(chip_abs, targets, close_by_ts)
        for c in HOUR_CHIP_COLUMNS:
            if c in out.columns:
                out[c] = chip[c].to_numpy()
    return out
