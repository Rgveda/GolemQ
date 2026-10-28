# coding: utf-8
"""把 review / coldstart / ChartScan 的时间线附加到 60min 特征帧（**只读**）。

对应老树 `GolemQ_old/features/reviews.py:729 attach_reality_features` 的新树版本：
老树只处理一张 reality 表；新树拆成三族，并补上 ChartScan 三条时间线。

由 cookbook `calc_compact_features.ipynb` / `calc_compact_pivot_feats.ipynb`
的「附加 reality 特征」cell 抽出来 —— 调试稳定后不再把这段逻辑留在 notebook 里。

口径
====
* **60min coldstart 列**（`COLDSTART_COLUMNS`）：日派生连续量。先按日取最后值，
  再按**上一可用交易日**映射；周末/节后首日盘中不会整列变空。
* **day 列**（`DAY_COLUMNS`，含 `_REAL` 与 2026-10-28 移入的 `se` / `ZenPivotCnt`）：
  按 `_map_causal` 映射；盘中用上一可用交易日，15:00 只认当日行
  （当日缺行保持 NaN，不做值 ffill）。
* **ChartScan** 三族时间线：事件型稀疏数据，**精确时间戳 join**（15min 用
  `merge_asof` 取当前 60min bar 之前最后一个已完成值），不 ffill。
* **`DEA_NORM_WEK`**：新 review 契约没有这列；由日线 → 周线 DEA 归一（mapower
  5..30）现算，取每个已完成周的最后值，按「上一已完成周」落到日线再映射。
⚠️ 本模块**只读**：不写库、不写水位。**不在 notebook/attach 里重算 review 的
特征**（coldstart/day/zen 等一律以落库行为准；缺行就是 NaN，需跑 review 补齐）。
`DEA_NORM_WEK` 是新 review 契约没有的列，才在这里由日线/周线现算。
"""
from __future__ import annotations

from datetime import time as _dtime
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from GolemQ.analysis.compact import _map_causal
from GolemQ.analysis.indicators import calc_macd_qa
from GolemQ.analysis.quadrant_jit import mapower_np_vXIIs
from GolemQ.analysis.stages import (garident_bands_from_ratio,
                                    garident_bottom_roof)
from GolemQ.analysis.timeseries import weekly_ohlc
from GolemQ.core.constants import AKA, FIELD as FLD
from GolemQ.markets.StockCN.reality_save import (PRICE_RATIO_COLUMNS,
                                                 fetch_reality_feat)
from GolemQ.pipeline.review.contract import COLDSTART_COLUMNS, DAY_COLUMNS

__all__ = ['DEA_NORM_WEEKLY', 'attach_reality_feats']

#: 老树 `FLD.DEA_NORM_WEEKLY` 的落库名（新 review 契约没有这列，本模块现算）。
DEA_NORM_WEEKLY = 'DEA_NORM_WEK'

_CHART_COLS_60 = ('Class', 'Confidence', 'Coordinates', 'ChartScaned')
_RENAME_60 = {'Class': 'Cls60', 'Confidence': 'Conf60',
              'Coordinates': 'Coord60', 'ChartScaned': 'ChartScaned60'}
_RENAME_DAY = {'ClsMaj': 'ClsDay', 'ConfidenceMaj': 'ConfDay',
               'CoordinatesMaj': 'CoordDay', 'ChartScaned': 'ChartScanedDay'}
_KEEP_15 = ('ClsMin', 'ConfidenceMin', 'CoordinatesMin', 'ChartScaned')
_RENAME_15 = {'ClsMin': 'Cls15', 'ConfidenceMin': 'Conf15',
              'CoordinatesMin': 'Coord15', 'ChartScaned': 'ChartScaned15'}

#: Prophet 上下轨/净空是**连续量**；review 停更/缺日会在曲线上留下肉眼可见的断线。
#: 在**15:00 真值补齐之后**统一 ffill（只填 NaN，不覆盖刚算出的当日真值）。
_PROPHET_COLS = (
    FLD.FBPROPHET_LB_DUMMY, FLD.FBPROPHET_UB_DUMMY,
    FLD.FBPROPHET_CLEARANCE, FLD.FBPROPHET_CLEARANCE_NORM,
    FLD.FBPROPHET_LB_MAJOR_DUMMY, FLD.FBPROPHET_UB_MAJOR_DUMMY,
    FLD.FBPROPHET_CLEARANCE_MAJOR, FLD.FBPROPHET_CLEARANCE_NORM_MAJOR,
)


def _as_multi(df: pd.DataFrame) -> pd.DataFrame:
    """reality 帧的索引统一成 `(ts, code)` 两层。"""
    return df.set_axis(pd.MultiIndex.from_arrays(
        [pd.to_datetime(df.index.get_level_values(0)),
         df.index.get_level_values(1)], names=['ts', 'code']))


def _attach_60min(features: pd.DataFrame, r60: Optional[pd.DataFrame],
                  loaded: Dict[str, list]) -> pd.DataFrame:
    if r60 is None:
        return features
    r60 = _as_multi(r60)
    # coldstart 是日派生连续量：按日最后值 + 上一可用交易日映射
    cold = [c for c in COLDSTART_COLUMNS
            if c in r60.columns and c not in features.columns]
    if cold:
        tmp = r60[cold].copy()
        tmp['_date'] = pd.DatetimeIndex(
            pd.to_datetime(r60.index.get_level_values(0))).normalize()
        daily = tmp.groupby('_date').last()
        mapped = pd.DataFrame(index=features.index)
        for col in cold:
            mapped[col] = _map_causal(daily[col], features.index)
        features = pd.concat([features, mapped], axis=1)
        loaded['60min_cold'] = cold
    # `se` / `ZenPivotCnt` 已改由日族承载（D74），随 `_attach_day` 映射。
    # ChartScan 是稀疏事件：精确时间戳 join
    chart = [c for c in _CHART_COLS_60
             if c in r60.columns and c not in features.columns]
    add = r60[chart].rename(columns=_RENAME_60)
    features = features.join(add, how='left')
    loaded['60min_chart'] = list(add.columns)
    return features


def _attach_day(features: pd.DataFrame, rd: Optional[pd.DataFrame],
                loaded: Dict[str, list]) -> pd.DataFrame:
    if rd is None:
        return features
    want = [c for c in (*DAY_COLUMNS, 'ClsMaj', 'ConfidenceMaj',
                        'CoordinatesMaj', 'ChartScaned') if c in rd.columns]
    addd = pd.DataFrame(index=features.index)
    for col in want:
        s = rd[col].copy()
        s.index = pd.MultiIndex.from_arrays(
            [pd.to_datetime(s.index.get_level_values(0)),
             s.index.get_level_values(1)], names=['ts', 'code'])
        addd[col] = _map_causal(s, features.index)
    addd = addd.rename(columns=_RENAME_DAY)
    features = features.join(addd)
    loaded['day'] = list(addd.columns)
    return features


def _attach_15min(features: pd.DataFrame, r15: Optional[pd.DataFrame],
                  loaded: Dict[str, list]) -> pd.DataFrame:
    if r15 is None:
        return features
    r15 = _as_multi(r15)
    want = [c for c in _KEEP_15 if c in r15.columns]
    right = r15[want].reset_index()
    right = right[['ts'] + want].sort_values('ts')
    right['date'] = right['ts'].dt.normalize()
    ts_all = pd.DatetimeIndex(pd.to_datetime(features.index.get_level_values(0)))
    left = pd.DataFrame({'ts': ts_all})
    left['date'] = left['ts'].dt.normalize()
    merged = pd.merge_asof(left.sort_values('ts'), right,
                           on='ts', by='date', direction='backward')
    merged = merged.rename(columns=_RENAME_15)
    merged = merged.set_index('ts').reindex(ts_all)
    merged.index = features.index
    keep = [c for c in _RENAME_15.values() if c in merged.columns]
    features = features.join(merged[keep])
    loaded['15min'] = keep
    return features


def _attach_dea_norm_weekly(features: pd.DataFrame,
                            day_ohlc: Optional[pd.DataFrame],
                            loaded: Dict[str, list]) -> pd.DataFrame:
    """`DEA_NORM_WEK`：周线 DEA → mapower 归一；按上一已完成周映射回 60min。"""
    if day_ohlc is None or len(day_ohlc) < 30:
        return features
    wk = weekly_ohlc(day_ohlc)
    if wk is None or len(wk) < 26:
        return features
    close = pd.to_numeric(wk[AKA.CLOSE], errors='coerce')
    _dif, dea, _macd = calc_macd_qa(close)
    dea_norm = pd.Series(
        mapower_np_vXIIs(dea.to_numpy(np.float32),
                         range_list=np.arange(5, 30).astype(np.int16)),
        index=pd.DatetimeIndex(pd.to_datetime(
            wk.index.get_level_values(0))).normalize())
    wk_dates = dea_norm.index.sort_values()
    wk_vals = dea_norm.reindex(wk_dates).to_numpy()
    ts_all = pd.DatetimeIndex(pd.to_datetime(features.index.get_level_values(0)))
    day_dates = pd.DatetimeIndex(ts_all.normalize().unique()).sort_values()
    # 目标日的「本周起始日」；严格取更早的周 ⇒ 只用上一已完成周
    week_start = day_dates - pd.to_timedelta(day_dates.weekday, unit='D')
    pos = np.searchsorted(wk_dates.to_numpy(), week_start.to_numpy(),
                          side='left') - 1
    daily = pd.Series(
        np.where(pos >= 0, wk_vals[np.clip(pos, 0, None)], np.nan),
        index=day_dates)
    features = features.copy()          # 去碎片
    features[DEA_NORM_WEEKLY] = _map_causal(daily, features.index)
    loaded['weekly'] = [DEA_NORM_WEEKLY]
    return features


def _restore_garident_bands(features: pd.DataFrame,
                            day_ohlc: Optional[pd.DataFrame]) -> pd.DataFrame:
    """`garPD`（归一化带）* **锚 bar close** -> 无后缀 `garP` 价格带。

    ⚠️ 乘的是 `garPAnchor` 那天的 close，**不是当前 bar 的 close**（D49）。
    """
    if (FLD.GARIDENT_PRICE_DUMMY not in features.columns
            or FLD.GARIDENT_PRICE_ANCHOR not in features.columns):
        return features
    # anchor date -> close
    if day_ohlc is not None and len(day_ohlc):
        c = pd.to_numeric(day_ohlc[AKA.CLOSE], errors='coerce')
        d = pd.DatetimeIndex(pd.to_datetime(
            c.index.get_level_values(0))).normalize()
        anchor_close = pd.Series(c.to_numpy(), index=d).groupby(level=0).last()
    else:
        close = pd.to_numeric(features[AKA.CLOSE], errors='coerce')
        ts = pd.DatetimeIndex(pd.to_datetime(
            features.index.get_level_values(0)))
        mask = ts.time == _dtime(15, 0)
        anchor_close = pd.Series(
            close[mask].to_numpy(), index=ts[mask].normalize()
        ).groupby(level=0).last()

    def _miss(v):
        if v is None:
            return True
        if isinstance(v, (list, tuple, np.ndarray)):
            return False
        try:
            return bool(pd.isna(v))
        except (TypeError, ValueError):
            return False

    out = []
    for ratio, anchor in zip(features[FLD.GARIDENT_PRICE_DUMMY].values,
                             features[FLD.GARIDENT_PRICE_ANCHOR].values):
        if _miss(ratio) or _miss(anchor):
            out.append(None)
            continue
        k = anchor_close.get(pd.Timestamp(str(anchor)[:10]).normalize())
        out.append(None if k is None or pd.isna(k)
                   else garident_bands_from_ratio(ratio, float(k)))
    features = features.copy()
    features[FLD.GARIDENT_PRICE] = out
    # 绝对价层再往下取两条（D76）：btmP = 最早一段的底，
    # roofP = 最近一段的顶。**不落库、不归一化**，只给上层现用。
    pairs = [garident_bottom_roof(v) for v in out]
    features[FLD.BOTTOM_PRICE] = [p[0] for p in pairs]
    features[FLD.ROOF_PRICE] = [p[1] for p in pairs]
    return features


def _ffill_prophet(features: pd.DataFrame,
                    loaded: Dict[str, list]) -> pd.DataFrame:
    """Prophet 连续量缺口 ffill（只填 NaN，不覆盖已有值）。"""
    cols = [c for c in _PROPHET_COLS
            if c in features.columns and features[c].isna().any()]
    if not cols:
        return features
    features = features.copy()
    features[cols] = features[cols].ffill()
    loaded['ffill_prophet'] = cols
    return features


def _strip_price_suffix(features: pd.DataFrame,
                        loaded: Dict[str, list]) -> pd.DataFrame:
    """D / _DUMMY 已由 restore 生成无后缀 base；这里**删掉归一化列**并同步 loaded。

    最终 attach 帧只保留 `prophetLb` / `poly9Maj` / `garP` 这类**无后缀价格**；
    D/DUMMY 仍留在 MongoDB，不进入 features。
    """
    drop = []
    for col in (*PRICE_RATIO_COLUMNS, FLD.GARIDENT_PRICE_DUMMY):
        if col in features.columns:
            drop.append(col)
    if drop:
        features = features.drop(columns=drop)
    for key, cols in list(loaded.items()):
        base_cols = []
        for c in cols:
            if c.endswith('_DUMMY'):
                c = c[:-6]
            elif c in drop and c.endswith('D'):
                c = c[:-1]
            base_cols.append(c)
        loaded[key] = list(dict.fromkeys(base_cols))
    return features


def attach_reality_feats(features: pd.DataFrame,
                         code: Optional[str] = None,
                         *,
                         market_type=None,
                         start: Optional[str] = None,
                         end: Optional[str] = None,
                         db=None,
                         day_ohlc: Optional[pd.DataFrame] = None,
                         verbose: bool = False) -> Tuple[pd.DataFrame, Dict[str, list]]:
    """把 review / coldstart / ChartScan 时间线附加到 60min 特征帧。

    :param features: 60min 特征帧，索引 `(ts, code)`。
    :param code: 6 位代码；``None`` ⇒ 取 ``features`` 索引第二层第一个。
    :param market_type: 市场类型；``None`` ⇒ ``is_stock_cn(code)[1]``。
    :param start, end: reality 读取范围（``YYYY-MM-DD``）；``None`` ⇒ 特征帧首/末日期。
    :param db: 行情库实例；``None`` ⇒ ``get_active_market().get_database()``。
    :param day_ohlc: 可选日线 OHLC（用于 `DEA_NORM_WEK`）；
        ``None`` ⇒ 本函数自己按 5 年回看取一次。
    :returns: ``(features, loaded)``；``loaded`` 记录每个来源实际附加的列名。
    """
    if code is None:
        code = str(features.index.get_level_values(1)[0])
    code = str(code)[:6]
    ts_all = pd.DatetimeIndex(pd.to_datetime(features.index.get_level_values(0)))
    if start is None:
        start = str(ts_all.min().date())
    if end is None:
        end = str(ts_all.max().date())
    if market_type is None:
        from GolemQ.markets.StockCN.symbol import is_stock_cn
        market_type = is_stock_cn(code)[1]
    if db is None:
        from GolemQ import get_active_market
        db = get_active_market().get_database()

    def _fetch(family: str) -> Optional[pd.DataFrame]:
        return fetch_reality_feat(code, start, end, family=family,
                                  market_type=market_type, db=db,
                                  restore_price=False)

    loaded: Dict[str, list] = {}
    features = _attach_60min(features, _fetch('60min'), loaded)
    features = _attach_day(features, _fetch('day'), loaded)
    features = _attach_15min(features, _fetch('15min'), loaded)

    if day_ohlc is None:
        from GolemQ import get_active_market
        market = get_active_market()
        dstart = str((pd.Timestamp(ts_all.min()) - pd.Timedelta(days=365 * 5)).date())
        res, _ = market.get_kline_price_v3(code, start=dstart, end=end,
                                           realtime=False, verbose=False)
        day_ohlc = None if res is None else getattr(res, 'data', None)

    features = _attach_dea_norm_weekly(features, day_ohlc, loaded)
    features = _ffill_prophet(features, loaded)
    # 归一化列（D / _DUMMY）* close -> 无后缀价格列；attach 只读
    from GolemQ.markets.StockCN.reality_save import restore_price_columns
    if AKA.CLOSE in features.columns:
        features = restore_price_columns(features, features[AKA.CLOSE],
                                         PRICE_RATIO_COLUMNS)
    features = _restore_garident_bands(features, day_ohlc)
    if FLD.BOTTOM_PRICE in features.columns:
        loaded['garident'] = [FLD.GARIDENT_PRICE, FLD.BOTTOM_PRICE,
                              FLD.ROOF_PRICE]
    features = _strip_price_suffix(features, loaded)
    if verbose:
        print('[reality_attach] {}'.format({k: len(v) for k, v in loaded.items()}))
    return features, loaded
