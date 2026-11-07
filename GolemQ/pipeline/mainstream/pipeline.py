# coding: utf-8
"""Mainstream 管线（8.3）—— 目前只含「全局智能仓位」一段。

旧树对应：`GolemQ_old/benchmark/mainstream.py` 的
`workload → workload_p1（逐股特征）→ groupby(bar).calc_massive_ChartsScanAI_model`。
8.3 第一版**只移植最后一段**（Dempster–Shafer 策略仓位控制），
逐股特征在 `_code_masks` 里按需现算，不落 `stock_reality_feat_60min`。

⚠️ 口径（D79）：**新树自洽、不逐值复刻 4.4**。
`zDLagD` 用**新版无后视镜**（`calc_zen_dash_causal`：缠论笔确认才可见）；
「当日日线值贴到当日盘中 bar」照旧树保留（旧树既有前视）。
逐值对拍与结论见 `docs/decisions/D79.md`。
"""
from __future__ import annotations

from datetime import time as dtime
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd

from GolemQ.analysis.position import calc_position_flags, calc_position_series
from GolemQ.core.constants import FIELD as FLD
from GolemQ.markets.StockCN.date_utils import GQ_util_last_trade_as_of
from GolemQ.models.alias import LTT
from GolemQ.pipeline.base import BasePipeline, CodeResult

#: 构建主时间轴用的参考码（长历史、极少停牌）。三条并集覆盖 2016 以来全部交易日。
TIMELINE_REFS = ('600519', '000001', '300750')
ANCHOR_60MIN = '2016-01-01'
ANCHOR_15MIN = '2017-01-01'
#: 默认回看（自然日）与旧树 `models/mainstream.py` 的
#: `lookback_hours=19200`（=800 天）一致。实测 600519：800 天窗口与全历史
#: 在 2025-04-01 的四个输入**逐值相同**（137/45/0/-12 vs 4.4），而 15min
#: 从 2017 起读会把 I/O 放大 4~5 倍（5,580 码 × 37k 根）。
LOOKBACK_DAYS = 800
#: as-of 缠论的回看 bar 数（800 自然日 ≈ 530 交易日 × 4 根 = 2,120 根 60min）。
#: ⚠️ 别写成 770 —— 770 根只有约 9 个月，2026-06 的笔会整段错位。
ZEN_WINDOW_BARS = 2120
CHUNK_SIZE = 100


def _frame(res):
    if isinstance(res, tuple):
        res = res[0]
    return getattr(res, 'data', None)


def _dt_index(index) -> pd.DatetimeIndex:
    if isinstance(index, pd.MultiIndex):
        return pd.DatetimeIndex(pd.to_datetime(index.get_level_values(0)))
    return pd.DatetimeIndex(pd.to_datetime(index))


class MainstreamPipeline(BasePipeline):
    """全市场 Mainstream 聚合（当前只有仓位段）。"""

    def __init__(self, lookback_days: int = LOOKBACK_DAYS, verbose: bool = False,
                 timeout: int = 1800, zen_window: int = ZEN_WINDOW_BARS):
        super().__init__(name='mainstream', verbose=verbose, timeout=timeout)
        self.lookback_days = int(lookback_days or 0)
        self.zen_window = int(zen_window or ZEN_WINDOW_BARS)

    # ------------------------------------------------------------------ #
    def _start_dates(self, as_of: pd.Timestamp):
        """取数起点 = as_of − **2×** 回看。

        为什么是两倍：as-of 缠论要求**每根 bar 都有完整的 800 天前史**
        （旧树每次运行都是「截至运行时刻的 800 天帧」）；只取 800 天的话，
        窗口头部那些 bar 的 CZSC 只有几十根前史 —— 实测 2025-12-22 的
        `zDLagD` 会从 -1 漂成 -101。
        """
        if self.lookback_days > 0:
            day = (as_of - pd.Timedelta(days=2 * self.lookback_days)).strftime('%Y-%m-%d')
            return day, day
        return ANCHOR_60MIN, ANCHOR_15MIN

    def _fetch(self, code: str, as_of: pd.Timestamp):
        from GolemQ import get_active_market
        m = get_active_market()
        start60, start15 = self._start_dates(as_of)
        end = as_of.strftime('%Y-%m-%d %H:%M:%S')
        f60 = _frame(m.get_kline_price_min(code, start=start60, end=end,
                                           frequency='60min', realtime=False))
        f15 = _frame(m.get_kline_price_min(code, start=start15, end=end,
                                           frequency='15min', realtime=False))
        fd = _frame(m.get_kline_price_v3(code, start=start60,
                                         end=as_of.strftime('%Y-%m-%d'),
                                         realtime=False))
        return f60, f15, fd

    # ------------------------------------------------------------------ #
    def _features_from_frames(self, code: str, f60, f15, fd, *,
                              window: Optional[int] = None):
        """三帧 → 60min 特征帧。纯计算，不取数（供 `code_features` 与对拍）。

        走 `analysis/feature_flow.py` 的 Hamilton DAG：`q60/q15/qd` 各算一次，
        `quad_downrisk_major` 复用 `qd`，`zDLagD` 只在 prelude 被请求时执行。
        """
        if f60 is None or len(f60) < 262 or fd is None or len(fd) < 262:
            return None
        if f15 is None or len(f15) < 262:
            return None

        from GolemQ.analysis import feature_flow as ff
        dr = ff.build_driver()
        out = dr.execute(
            ['pwrln', 'pwrln_minor', 'pwrln_major', 'macd_delta_major',
             'qlev_frame', 'quad_downrisk', 'quad_downrisk_major',
             'zd_lag', LTT.QUADRANT_PRELUDE_UPRISING_BEFORE,
             LTT.QUADRANT_PRELUDE_ENDPOINT_BEFORE],
            inputs={'code': str(code), 'ohlc60': f60, 'ohlc15': f15,
                    'ohlc_day': fd,
                    'window': int(window or self.zen_window)})
        q = out['qlev_frame']
        return pd.DataFrame({
            LTT.QUADRANT_POWERLINE: out['pwrln'],
            LTT.QUADRANT_POWERLINE_MINOR: out['pwrln_minor'],
            LTT.QUADRANT_POWERLINE_MAJOR: out['pwrln_major'],
            FLD.MACD_DELTA_MAJOR: out['macd_delta_major'],
            LTT.QUADRANT_LEVERAGE_ENERGY_MINOR:
                q[LTT.QUADRANT_LEVERAGE_ENERGY_MINOR],
            LTT.QUADRANT_LEVERAGE_ENERGY: q[LTT.QUADRANT_LEVERAGE_ENERGY],
            LTT.QUADRANT_LEVERAGE_ENERGY_MAJOR:
                q[LTT.QUADRANT_LEVERAGE_ENERGY_MAJOR],
            LTT.QUADRANT_LEVERAGE_ENERGY_DUMMY:
                q[LTT.QUADRANT_LEVERAGE_ENERGY_DUMMY],
            LTT.QUADRANT_LEVERAGE_ENERGY_TIMING_LAG_DUMMY:
                q[LTT.QUADRANT_LEVERAGE_ENERGY_TIMING_LAG_DUMMY],
            LTT.QUADRANT_LEVERAGE_MACD_TIMING_LAG_DUMMY:
                q[LTT.QUADRANT_LEVERAGE_MACD_TIMING_LAG_DUMMY],
            LTT.QUADRANT_DOWNRISK_TIMING_LAG: out['quad_downrisk'],
            LTT.QUADRANT_DOWNRISK_TIMING_LAG_MAJOR: out['quad_downrisk_major'],
            FLD.ZEN_DASH_TIMING_LAG_DUMMY: out['zd_lag'],
            LTT.QUADRANT_PRELUDE_UPRISING_BEFORE:
                out[LTT.QUADRANT_PRELUDE_UPRISING_BEFORE],
            LTT.QUADRANT_PRELUDE_ENDPOINT_BEFORE:
                out[LTT.QUADRANT_PRELUDE_ENDPOINT_BEFORE],
        }, index=f60.index)

    def code_features(self, code: str, as_of: Optional[pd.Timestamp] = None):
        """单码 → 60min 特征帧（含 `QuadPreUprBf` / `QuadPreEpBf` / `zDLagD`）。

        供 `code_masks` 与逐值对拍共用（不另写一套）。
        """
        as_of = pd.Timestamp(as_of or GQ_util_last_trade_as_of())
        f60, f15, fd = self._fetch(code, as_of)
        return self._features_from_frames(code, f60, f15, fd)

    def code_masks(self, code: str, as_of: Optional[pd.Timestamp] = None):
        """单码 → `(ts, bull, bear)` 三个等长数组（只含可判定 bar）。

        不可判定的 bar（缺 `QuadDwnRskLagMaj` / `zDLagD` 等）**不投票**，
        与旧树的「NaN 落入 Ω」不同 —— 旧树 60min 特征行若存在，四列必有值；
        这里缺值意味着数据不足，计入 Ω 会把"没数据"当成"看不清"，宁可剔除。
        """
        feat = self.code_features(code, as_of)
        if feat is None:
            return None
        idx60 = _dt_index(feat.index)
        upr = feat[LTT.QUADRANT_PRELUDE_UPRISING_BEFORE].to_numpy(dtype=float)
        ep = feat[LTT.QUADRANT_PRELUDE_ENDPOINT_BEFORE].to_numpy(dtype=float)
        z = feat[FLD.ZEN_DASH_TIMING_LAG_DUMMY].to_numpy(dtype=float)
        valid = np.isfinite(upr) & np.isfinite(ep) & np.isfinite(z)
        bull = valid & (upr < ep)
        bear = valid & (ep < upr) & (z < -1e-12)
        return idx60.to_numpy(), bull, bear

    # ------------------------------------------------------------------ #
    def run_code(self, code: str, **kwargs) -> CodeResult:
        try:
            masks = self.code_masks(code, kwargs.get('as_of'))
            if masks is None:
                return CodeResult(code=code, status='no_data')
            ts, bull, bear = masks
            return CodeResult(code=code, status='ok',
                              result={'ts': ts, 'bull': bull, 'bear': bear})
        except Exception as exc:                    # noqa: BLE001
            return CodeResult(code=code, status='error',
                              error=f'{type(exc).__name__}: {exc}')

    # ------------------------------------------------------------------ #
    def market_timeline(self, as_of: pd.Timestamp) -> np.ndarray:
        """主时间轴：只保留 **10:30 / 11:30 / 14:00 / 15:00** 四根。

        ⚠️ `stock_60min` 里有杂散 13:00 行（实测 4 根），旧树只认四个标准
        stamp；不过滤会把脏 bar 当成有效快照（横截面成员数会明显偏少）。
        """
        stamps = {dtime(10, 30), dtime(11, 30), dtime(14, 0), dtime(15, 0)}
        ts = set()
        for ref in TIMELINE_REFS:
            f60, _, _ = self._fetch(ref, as_of)
            if f60 is not None and len(f60):
                idx = pd.DatetimeIndex(
                    pd.to_datetime(f60.index.get_level_values(0)))
                # ⚠️ `DatetimeIndex.time` 是 **ndarray**（datetime.time 对象），
                # 没有 `.isin`；用 strftime 得到 Index 再筛。
                keep = idx[idx.strftime('%H:%M').isin(
                    [t.strftime('%H:%M') for t in stamps])]
                ts |= set(keep.to_numpy())
        return np.array(sorted(ts), dtype='datetime64[ns]')

    @staticmethod
    def _index_close_map(code: str, timeline) -> np.ndarray:
        """指数 60min 收盘 → 与主时间轴对齐的数组（缺失为 NaN）。"""
        from GolemQ import get_active_market
        db = get_active_market().get_database()
        m = {pd.Timestamp(r['datetime']): float(r['close'])
             for r in db['index_60min'].find(
                 {'code': code}, {'datetime': 1, 'close': 1, '_id': 0})
             if r.get('close') is not None}
        return np.array([m.get(pd.Timestamp(t), np.nan) for t in timeline],
                        dtype=float)

    def snapshot_contracts(self) -> pd.DataFrame:
        """一次 groupby(bar) 产出全部快照契约列（MED/MODE/COMBO）。

        源列 = `*_reality_feat_60min` 的逐股列；`m9tCmpod` 是逐码合成的
        M9T combo（`calc_m9t_combo`），作为虚拟源列喂给注册表。
        """
        from GolemQ import get_active_market
        from GolemQ.analysis.mainstream_timing import calc_m9t_combo
        from GolemQ.analysis.timing import calc_event_timing_lag
        from GolemQ.pipeline.mainstream.snapshot import snapshot_frame
        db = get_active_market().get_database()
        cols = ('stage', 'poly9NorMajR', 'cost5PrzD', 'cost95PrzD',
                'm9t', 'm9tLagMajR', 'm9tLagWekR')
        # ⚠️ 是 **OR** 不是 AND：各列各取各的（`aggregate_snapshot` 内部再逐列
        # dropna）。用 AND 会把「只有 stage 没有 cost」的 bar 全丢掉 ⇒ 同一个
        # 契约列在不同时期来自不同聚合，且旧值因 write_nulls=False 不被覆盖。
        cur = db['stock_reality_feat_60min'].find(
            {'$or': [{c: {'$ne': None}} for c in cols]},
            {'code': 1, 'datetime': 1, **{c: 1 for c in cols}, '_id': 0})
        df = pd.DataFrame(list(cur))
        if df.empty:
            return pd.DataFrame()
        df['ts'] = pd.to_datetime(df['datetime'])
        parts = []
        _m9t_cols = ['m9t', 'm9tLagMajR', 'm9tLagWekR']
        for _code, x in df.groupby('code', sort=False):
            x = x.sort_values('ts')
            # ⚠️ 三列**齐全**才算 combo：缺列时 `nan→0` 会把「没有数据」算成
            # 「中性 0」，在聚合里变成假值（2024-02 实测）。
            _ok = x[_m9t_cols].notna().all(axis=1).to_numpy()
            _v = np.asarray(calc_m9t_combo(x['m9t'], x['m9tLagMajR'],
                                           x['m9tLagWekR']), dtype=float)
            _v[~_ok] = np.nan
            parts.append(pd.Series(_v, index=x.index))
        df['m9tCmpod'] = pd.concat(parts).sort_index()
        snap = snapshot_frame(df, ts_col='ts')
        # 旧树 chip_dist.py:1380 的再编码：市场级 COMBO 再取 event-lag
        if 'MAGIC_NINE_TURNS_DUMMY_COMPOUNDED_MODE' in snap.columns:
            _m = snap['MAGIC_NINE_TURNS_DUMMY_COMPOUNDED_MODE']
            # ⚠️ `np.where(NaN < x, -1, 1)` 会走 +1 分支 —— 必须显式把
            # 「众数为空」的 bar 也置空，否则 event-lag 会在无数据的 bar 上
            # 编造一串 +1/+2…
            _lag = calc_event_timing_lag(
                np.where(_m < -0.00168, -1, 1)).astype(float)
            _lag[pd.isna(_m).to_numpy()] = np.nan
            snap['MAGIC_NINE_TURNS_TIMING_LAG_COMPOUNDED'] = _lag
        return snap

    def _chunk_counts(self, codes: List[str], timeline: np.ndarray,
                      as_of) -> np.ndarray:
        counts = np.zeros((3, len(timeline)), dtype=np.int32)
        T = len(timeline)
        for code in codes:
            try:
                masks = self.code_masks(code, as_of)
            except Exception:                       # noqa: BLE001 单码失败不拖垮整片
                continue
            if masks is None:
                continue
            ts, bull, bear = masks
            pos = np.searchsorted(timeline, ts)
            ok = (pos < T) & (timeline[np.clip(pos, 0, T - 1)] == ts)
            if not ok.any():
                continue
            p = pos[ok]
            counts[0] += np.bincount(p[bull[ok]], minlength=T).astype(np.int32)
            counts[1] += np.bincount(p[bear[ok]], minlength=T).astype(np.int32)
            counts[2] += np.bincount(p, minlength=T).astype(np.int32)
        return counts

    def run_market(self, codes: Iterable[str], as_of=None,
                   pool_size: int = 0) -> pd.DataFrame:
        """全市场横截面 → 每根 60min bar 的仓位行。"""
        as_of = pd.Timestamp(as_of or GQ_util_last_trade_as_of())
        codes = [str(c) for c in codes]
        timeline = self.market_timeline(as_of)
        chunks = [codes[i:i + CHUNK_SIZE] for i in range(0, len(codes), CHUNK_SIZE)]
        if pool_size > 0 and len(chunks) > 1:
            from joblib import Parallel, delayed
            parts = Parallel(n_jobs=pool_size, backend='loky')(
                delayed(_chunk_counts_job)(self.lookback_days, chunk, timeline,
                                           as_of)
                for chunk in chunks)
        else:
            parts = [self._chunk_counts(c, timeline, as_of) for c in chunks]
        total = np.sum(parts, axis=0) if parts else np.zeros((3, len(timeline)),
                                                             dtype=np.int32)

        res = calc_position_series(total[2], total[0], total[1])
        # 快照契约（一次 groupby(bar)）：STAGE_MODE / 各 MEDIAN / COMBO
        snap = self.snapshot_contracts()
        _ts = pd.DatetimeIndex(timeline)

        def _snap_col(name):
            if snap is None or snap.empty or name not in snap.columns:
                return np.full(len(timeline), np.nan)
            return snap[name].reindex(_ts).to_numpy(dtype=float)

        stage_mode = _snap_col('STAGE_MODE')
        # 连续**交易日**的 -0.5 计数（雪球底 = 唯一的多日连续，见 D79）
        _is15 = _ts.strftime('%H:%M') == '15:00'
        sm_daily = pd.Series(stage_mode[_is15], index=_ts[_is15].normalize())
        sm_daily = sm_daily[~sm_daily.index.duplicated(keep='last')]
        streak_daily = {}
        _run = 0
        for _d, _v in sm_daily.items():
            _run = _run + 1 if (np.isfinite(_v) and _v < -1e-12) else 0
            streak_daily[_d] = _run
        days_norm = _ts.normalize()
        stage_streak = pd.Series(days_norm).map(streak_daily).to_numpy(
            dtype=float)
        flags = calc_position_flags(res['m_bull'], res['m_bear'],
                                    res['m_omega'], res['onhold'],
                                    stage_mode=stage_mode,
                                    stage_streak=stage_streak)
        # 清仓暴击的**新低附加条件**（2026-11-01 用户）：暴击时刻科创50 与
        # 创业板指数都必须低于「前一个零仓段的起点」。
        _prev = flags['ReClearPrevStart']
        if (_prev >= 0).any():
            _kc = self._index_close_map('000688', timeline)
            _cy = self._index_close_map('399006', timeline)
            _idx = np.arange(len(_prev))
            _pv = np.clip(_prev, 0, None)
            re_clear_strict = (flags['PosReClear'] & (_prev >= 0)
                               & (_kc[_idx] < _kc[_pv])
                               & (_cy[_idx] < _cy[_pv]))
        else:
            re_clear_strict = np.zeros(len(_prev), dtype=bool)

        out = pd.DataFrame({
            'datetime': pd.DatetimeIndex(timeline),
            'mberCnt': total[2],
            'PosBsln': res['m_bull'],
            'PosPad': res['m_omega'],
            'PosNegBsln': res['m_bear'],
            'MasPosOnhold': res['onhold'],
            'DS_BelBull': res['bel_bull'],
            'DS_BelBear': res['bel_bear'],
            'DS_PlBull': res['pl_bull'],
            'DS_PlBear': res['pl_bear'],
            'DS_UncertIntv': res['uncertainty'],
            'DS_State': res['ds_state'],
            # 事件标记（D79 增补）：满仓信号 / 满仓起点 / 清仓暴击
            'PosFull': flags['PosFull'].astype(np.int8),
            'PosFullOnset': flags['PosFullOnset'].astype(np.int8),
            'PosReClear': flags['PosReClear'].astype(np.int8),
            'PosReClearStrict': re_clear_strict.astype(np.int8),
            # 底部过滤器（2026-11-01）：空仓 + STAGE_MODE<=-0.5 ⇒ 试探仓
            'STAGE_MODE': stage_mode,
            'StageModeStreak': stage_streak.astype(np.int16),
            'PosBottom': flags['PosBottom'].astype(np.int8),
            'PosDeepBottom': flags['PosDeepBottom'].astype(np.int8),
            'PosProbe': flags['PosProbe'],
        })
        if snap is not None and not snap.empty:
            for _c in snap.columns:
                if _c == 'STAGE_MODE':
                    continue
                out[_c] = _snap_col(_c)
        return out[out['mberCnt'] > 0].reset_index(drop=True)


def _chunk_counts_job(lookback_days, codes, timeline, as_of):
    pipe = MainstreamPipeline(lookback_days=lookback_days)
    return pipe._chunk_counts(codes, timeline, as_of)


def run_mainstream(eval_range: str = 'all', codes=None, as_of: str = '',
                   pct_of_cores: float = 0.32,
                   lookback_days: int = LOOKBACK_DAYS,
                   verbose: bool = False):
    """解析 scope → 逐片算 → 返回 `(仓位 DataFrame, 统计)`。"""
    from GolemQ import get_active_market
    from GolemQ.analysis.cv.scopes import perpar_symbol_range

    if codes:
        codelist = [str(c) for c in codes if str(c)]
    elif str(eval_range).lower() in ('all', 'stock', 'stocks'):
        # mainstream 的横截面**只对同一族**有意义（ETF/指数要各自聚合），
        # 所以这里的 `all` = `stock_list` 全市场，而不是 scopes 的「股票∪ETF∪指数」。
        ds = get_active_market().scan_dataset()
        codelist = sorted({str(c) for c in ds.stock_list()['code']})
    else:
        got = perpar_symbol_range(eval_range,
                                  dataset=get_active_market().scan_dataset(),
                                  verbose=verbose)
        if got is False:
            raise ValueError(f'无法解析 eval_range={eval_range!r}')
        codelist = list(got)

    cpu = int(pct_of_cores) if pct_of_cores else 0
    if cpu > 1:
        import multiprocessing
        cpu = max(1, int((cpu / 100.0) * multiprocessing.cpu_count()))
    elif 0 < cpu <= 1:
        import multiprocessing
        cpu = max(1, int(cpu * multiprocessing.cpu_count()))

    pipe = MainstreamPipeline(lookback_days=lookback_days, verbose=verbose)
    df = pipe.run_market(codelist, as_of=(as_of or None), pool_size=cpu)
    return df, {'total_count': len(codelist), 'rows': len(df)}
