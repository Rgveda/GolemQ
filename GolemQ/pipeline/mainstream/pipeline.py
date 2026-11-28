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
from GolemQ.pipeline.base import BasePipeline, CodeResult, worker_count  # noqa: F401
from GolemQ.pipeline.mainstream.snapshot import M9T_SRC_COLS, SNAP_SRC_COLS

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

#: 快照契约的逐码源列 —— 在 `snapshot.py` 里由 `SNAPSHOT_AGS` 推导后导入
#: （别在本模块再写一份常量，平行实现只会分叉）。


def _frame(res):
    if isinstance(res, tuple):
        res = res[0]
    return getattr(res, 'data', None)


def _dt_index(index) -> pd.DatetimeIndex:
    if isinstance(index, pd.MultiIndex):
        return pd.DatetimeIndex(pd.to_datetime(index.get_level_values(0)))
    return pd.DatetimeIndex(pd.to_datetime(index))


def _stage(msg: str, t0: Optional[float] = None) -> float:
    """打印一条阶段进度（带**距上一阶段**的用时），返回新的计时起点。

    用户 2026-11-25：`Statistics` 打完到最后一整行 `[mainstream] ...` 之间原先
    有几分钟静默（横截面聚合 → 快照契约读库 → 清仓暴击的指数扫描 → 写库），
    看不出卡在哪一步。这里每过一站打一行，`flush=True` 保证管道重定向时也及时。
    """
    import time
    now = time.perf_counter()
    dt = '' if t0 is None else f'，{now - t0:.1f}s'
    print(f'[mainstream] {msg}{dt}', flush=True)
    return now


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
                              window: Optional[int] = None,
                              with_mtf_weight: bool = False,
                              with_bs_trunk: bool = True):
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
        _nodes = ['pwrln', 'pwrln_minor', 'pwrln_major', 'macd_delta_major',
                  'qlev_frame', 'quad_downrisk', 'quad_downrisk_major',
                  'zd_lag', LTT.QUADRANT_PRELUDE_UPRISING_BEFORE,
                  LTT.QUADRANT_PRELUDE_ENDPOINT_BEFORE]
        if with_bs_trunk:
            # ⚠️ `calc_bs_signals` 的 rolling 回归约 **0.21s/码**（实测 600519，
            # 占 `code_features` 的 1/3）—— `run_market` 的投票不用它，默认关。
            _nodes.append('bs_trunk')
        out = dr.execute(
            _nodes,
            inputs={'code': str(code), 'ohlc60': f60, 'ohlc15': f15,
                    'ohlc_day': fd,
                    'window': int(window or self.zen_window)})
        q = out['qlev_frame']
        frame = pd.DataFrame({
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
        if with_bs_trunk:
            # BS 主干（用户 2026-11-26）：共用 `quadrant_jit` 的规范实现
            frame[LTT.QUADRANT_BS_TRUNK] = out['bs_trunk'][LTT.QUADRANT_BS_TRUNK]
            frame[LTT.QUADRANT_BS_TRUNK_TIMING_LAG] = \
                out['bs_trunk'][LTT.QUADRANT_BS_TRUNK_TIMING_LAG]
        if with_mtf_weight:
            # 多周期买点：3 个基础信号（`QuadMsUprBf*`）+ 权重 `QuadMsCredCpd`
            # （全称见 LTT 常量注释）。
            # ⚠️ 与 feature_flow 的 q60/q15/qd 有重复计算（约 0.7s/码），
            # 默认**不**算；策略/回测需要时显式打开，避免拖慢 `run_market`。
            from GolemQ.analysis.mtf_buy_weight import calc_mtf_buy_resonance
            res = calc_mtf_buy_resonance(f60, f15, fd).reindex(f60.index)
            for _c in res.columns:
                frame[_c] = res[_c]
        return frame

    def code_features(self, code: str, as_of: Optional[pd.Timestamp] = None,
                      with_mtf_weight: bool = False,
                      with_chartscan: bool = False,
                      with_bs_trunk: bool = True):
        """单码 → 60min 特征帧（含 `QuadPreUprBf` / `QuadPreEpBf` / `zDLagD`
        / `QuadBsTuk(Lag)`）。

        供 `code_masks` 与逐值对拍共用（不另写一套）。

        :param with_mtf_weight: 是否附加多周期买点列（`QuadMsUprBf*` + `QuadMsCredCpd`）。
            默认 False —— 它约 **0.7s/码**，`run_market` 的投票不需要它。
        :param with_chartscan: 是否附加 `ChrtScanUprTukLag`
            （`CHARTSCAN_UPRISING_TRUNK_TIMING_LAG`）。要读 ChartScan 的
            `Cls*/Coord*`（两次 `fetch_reality_feat`，约 60ms/码）⇒ 默认关闭；
            独立入口仍是 :meth:`code_chartscan_trunk`。
        :param with_bs_trunk: 是否附加 `QuadBsTuk` / `QuadBsTukLag`
            （`QUADRANT_BS_TRUNK_TIMING_LAG`）。默认 **True**（纯 OHLC + 日线
            MAJOR，不碰库）；`run_market` 的投票不用它，那里显式关掉省 0.21s/码。
        """
        as_of = pd.Timestamp(as_of or GQ_util_last_trade_as_of())
        f60, f15, fd = self._fetch(code, as_of)
        frame = self._features_from_frames(code, f60, f15, fd,
                                           with_mtf_weight=with_mtf_weight,
                                           with_bs_trunk=with_bs_trunk)
        if with_chartscan and frame is not None:
            col = LTT.CHARTSCAN_UPRISING_TRUNK_TIMING_LAG
            frame[col] = self._chartscan_trunk(code, f60, fd)
        return frame

    def code_mtf_buy_weight(self, code: str,
                            as_of: Optional[pd.Timestamp] = None) -> pd.Series:
        """单码 → `QuadMsCredCpd`（对齐 60min 索引；日权重因果映射）。

        策略侧专用入口：不跑 feature_flow，只算三周期的买点权重
        （`analysis/mtf_buy_weight.py::calc_mtf_buy_weight`）。
        """
        from GolemQ.analysis.mtf_buy_weight import calc_mtf_buy_weight
        as_of = pd.Timestamp(as_of or GQ_util_last_trade_as_of())
        f60, f15, fd = self._fetch(code, as_of)
        if f60 is None or f15 is None or fd is None or len(f60) == 0:
            return pd.Series(dtype=float, name=LTT.QUADRANT_MAINSTREAM_CREDIT_COMPOUNDED)
        return calc_mtf_buy_weight(f60, f15, fd)

    # ------------------------------------------------------------------ #
    def _chartscan_feats(self, code: str, start, end):
        """ChartScan 的四列（`Cls60/Coord60` + `ClsDay/CoordDay`），**轻量读**。

        只用 `fetch_reality_feat` 投影需要的列（两次查询），**不**跑整帧
        `attach_reality_feats` —— 那会顺带算 `DEA_NORM_WEK` / 还原价格列等
        与本特征无关的东西（每码 0.3~1s）。60min 族在 60min 表、日线族在 day 表。
        """
        from GolemQ import get_active_market
        from GolemQ.markets.StockCN.reality_save import fetch_reality_feat
        from GolemQ.markets.StockCN.symbol import is_stock_cn
        db = get_active_market().get_database()
        mt = is_stock_cn(code)[1]
        parts = []
        for family, cls_col, coord_col, out in (
                ('60min', 'Class', 'Coordinates', ('Cls60', 'Coord60')),
                ('day', 'ClsMaj', 'CoordinatesMaj', ('ClsDay', 'CoordDay'))):
            df = fetch_reality_feat(code, start, end, family=family,
                                    market_type=mt,
                                    columns=[cls_col, coord_col],
                                    restore_price=False, db=db)
            if df is None or len(df) == 0:
                continue
            if cls_col in df.columns and coord_col in df.columns:
                # ⚠️ 索引 = **这一行的扫描时间**（`cs_class_before` 拿它当事件时间，
                # 才不是未来函数）。日族行按 key 是「日期」⇒ 扫描发生在收盘后，
                # 记成当日 15:00；记 00:00 会被 `searchsorted` 落到**前一交易日**
                # 的 15:00，等于提前一根。
                _ts = pd.DatetimeIndex(pd.to_datetime(
                    df.index.get_level_values(0)))
                if family == 'day':
                    _ts = _ts.normalize() + pd.Timedelta(hours=15)
                parts.append(pd.DataFrame(
                    {out[0]: df[cls_col].to_numpy(),
                     out[1]: df[coord_col].to_numpy()}, index=_ts))
        if not parts:
            return None
        return pd.concat(parts).sort_index()

    def _chartscan_trunk(self, code: str, f60, fd) -> pd.Series:
        """两帧 → `ChrtScanUprTukLag`（对齐 `f60.index`）。

        输入两条：
        * `Cls60/Coord60`、`ClsDay/CoordDay` —— :meth:`_chartscan_feats`
          （只读 `*_reality_feat_*` 的那 4 列，不重算 review）；
        * `MsLeadoutBfMaj` —— 日线 MAJOR 现算（`calc_mainstream_major_feats`），
          按 `_map_causal` 映射到 60min（盘中用上一交易日、15:00 用当日）。

        任何一条缺 ⇒ 全 NaN（不造数）。
        """
        from GolemQ.analysis.chartscan import calc_chartscan_uprising_trunk_lag
        name = LTT.CHARTSCAN_UPRISING_TRUNK_TIMING_LAG
        if f60 is None or len(f60) == 0:
            return pd.Series(dtype=float, name=name)
        empty = pd.Series(np.nan, index=f60.index, name=name, dtype=float)
        if fd is None or len(fd) == 0:
            return empty
        from GolemQ.analysis.chartscan import leadout_major_distance
        from GolemQ.analysis.quadrant_jit import calc_mainstream_major_feats
        try:
            ms_day = calc_mainstream_major_feats(fd)
            _col = LTT.MAINSTREAM_LEADOUT_BEFORE_MAJOR
            if ms_day is None or _col not in ms_day.columns:
                return empty
            # ⚠️ 日线量必须**映射到 60min 后重新按 bar 数**算（用户 2026-11-26）：
            # `MsLeadoutBfMaj` 是日线 `Timeline_duration`（单位「几根日线」），
            # 直接与 60min 的 `buy60`（「几根 60min」）比会让 `buy < leadout`
            # 长期假成立（同一段时间差约 4 倍）。
            lead = leadout_major_distance(ms_day, f60.index)
            if lead is None:
                return empty
            ts = _dt_index(f60.index)
            feats = self._chartscan_feats(code, str(ts.min().date()),
                                          str(ts.max().date()))
        except Exception:                       # noqa: BLE001 缺数据 ⇒ 该列不贡献
            return empty
        return calc_chartscan_uprising_trunk_lag(feats, f60.index, lead)

    def code_chartscan_trunk(self, code: str,
                             as_of: Optional[pd.Timestamp] = None) -> pd.Series:
        """单码 → `ChrtScanUprTukLag`（`CHARTSCAN_UPRISING_TRUNK_TIMING_LAG`）。

        用户 2026-11-26 更正后的算子：``calc_event_timing_lag(where((60min Buy
        < MsLeadoutBfMaj & < 日线 Sell) | (日线 Buy < 同条件), 1, -1))``。
        `+k` = 连续命中 k 根、`-k` = 连续未命中 k 根（不会出现 0）；
        ChartScan 输入未就绪 ⇒ NaN。
        """
        as_of = pd.Timestamp(as_of or GQ_util_last_trade_as_of())
        f60, _f15, fd = self._fetch(code, as_of)
        return self._chartscan_trunk(code, f60, fd)

    def code_masks_and_pwrln(self, code: str,
                             as_of: Optional[pd.Timestamp] = None):
        """单码 → `(ts, bull, bear, pwrln_major)`；`pwrln_major` 供 `MasQuadPwrlnMaj`。"""
        feat = self.code_features(code, as_of, with_bs_trunk=False)
        if feat is None:
            return None
        idx60 = _dt_index(feat.index)
        upr = feat[LTT.QUADRANT_PRELUDE_UPRISING_BEFORE].to_numpy(dtype=float)
        ep = feat[LTT.QUADRANT_PRELUDE_ENDPOINT_BEFORE].to_numpy(dtype=float)
        z = feat[FLD.ZEN_DASH_TIMING_LAG_DUMMY].to_numpy(dtype=float)
        valid = np.isfinite(upr) & np.isfinite(ep) & np.isfinite(z)
        bull = valid & (upr < ep)
        bear = valid & (ep < upr) & (z < -1e-12)
        pwrln = feat[LTT.QUADRANT_POWERLINE_MAJOR].to_numpy(dtype=float)
        return idx60.to_numpy(), bull, bear, pwrln

    def code_masks(self, code: str, as_of: Optional[pd.Timestamp] = None):
        """单码 → `(ts, bull, bear)` 三个等长数组（只含可判定 bar）。

        不可判定的 bar（缺 `QuadDwnRskLagMaj` / `zDLagD` 等）**不投票**，
        与旧树的「NaN 落入 Ω」不同 —— 旧树 60min 特征行若存在，四列必有值；
        这里缺值意味着数据不足，计入 Ω 会把"没数据"当成"看不清"，宁可剔除。
        """
        r = self.code_masks_and_pwrln(code, as_of)
        return None if r is None else r[:3]

    # ------------------------------------------------------------------ #
    def _snapshot_source(self, code: str, start=None,
                         end=None) -> pd.DataFrame:
        """单码的快照源列 → 小帧（`ts` + 8 源列 + `M9tTrkLag`）。

        旧树口径（`GolemQ_old/benchmark/mainstream.py::workload_p1`）：这些列在
        **逐码 worker**里 attach、`M9tTrkLag` 也在 worker 内行内算好，跑完
        `concat` 成 `ret_features_pd` 再一次性聚合 —— 不再由父进程全表扫
        `stock_reality_feat_60min`（18.2M 行、`datetime` 无索引，实测 135s）。

        :param start, end: 读取窗口（`timeline[0]` ~ `as_of`）。``None`` ⇒ 空帧
            （只在 `run_market` 里被调用，窗口由它给）。
        :returns: 列 `['ts', 8 源列, 'M9tTrkLag']` 的帧；缺数据 / 异常 ⇒ 空帧
            （该码不贡献，不抛）。
        """
        cols = list(SNAP_SRC_COLS)
        empty = pd.DataFrame(columns=['ts', *cols,
                                      LTT.MAGIC_NINE_TURNS_TRUNK_TIMING_LAG])
        if start is None or end is None:
            return empty
        try:
            from GolemQ import get_active_market
            from GolemQ.markets.StockCN.reality_save import fetch_reality_feat
            from GolemQ.markets.StockCN.symbol import is_stock_cn
            # ⚠️ 必须显式给 `market_type`：默认 None 会解析成
            # `index_reality_feat_60min`（实测返回 None）。
            #
            # ⚠️ **从 `ANCHOR_60MIN` 起读（暖机）**：`M9tTrkLag` 是
            # `calc_event_timing_lag` 的**有状态**结果（0 不清零），从窗口头起算
            # 会让窗口前段的计数比全历史起算少 —— 实测从 800 天窗口起算有
            # 35/2,129 根 COMPOUNDED_MODE 不一致。多读几年只多一点行数
            # （实测 600519：0.032s / 3,805 行）。
            df = fetch_reality_feat(
                code, ANCHOR_60MIN, end, family='60min',
                market_type=is_stock_cn(code)[1],
                columns=cols, restore_price=False,
                db=get_active_market().get_database())
        except Exception:                       # noqa: BLE001 缺数据 ⇒ 该码不贡献
            return empty
        if df is None or len(df) == 0 or 'datetime' not in df.columns:
            return empty
        out = df.reindex(columns=cols).apply(pd.to_numeric, errors='coerce')
        if out.empty:
            return empty
        # 逐码 M9T combo（三列齐全才算 —— 缺列时 `nan→0` 会算成「中性 0」）
        from GolemQ.analysis.mainstream_timing import calc_m9t_combo
        _m = out.reindex(columns=list(M9T_SRC_COLS))
        _ok = _m.notna().all(axis=1).to_numpy()
        _v = np.asarray(calc_m9t_combo(_m['m9t'], _m['m9tLagMajR'],
                                       _m['m9tLagWekR']), dtype=float)
        _v[~_ok] = np.nan
        out.insert(0, 'ts', pd.to_datetime(df['datetime']).to_numpy())
        out[LTT.MAGIC_NINE_TURNS_TRUNK_TIMING_LAG] = _v
        # 先算 combo（有状态）再切窗口 / 丢全 NaN 行 —— 顺序不能反
        keep = (out['ts'] >= pd.Timestamp(start)) & \
               (out['ts'] <= pd.Timestamp(end)) & \
               out[cols].notna().any(axis=1).to_numpy()
        if not keep.any():
            return empty
        return out[keep].reset_index(drop=True)

    @staticmethod
    def merge_snapshot_sources(results):
        """各码 worker 的 `snap_src` → ``(ret_features_pd, 贡献码数)``。

        * **不看 `r.ok`**：`no_data` / 报错的码也可能有 reality 源列，旧路径
          是全表聚合、与 OHLC 是否算得出来无关（评审 2026-11-25 指出）。
        * 不加 `code` 列、不 `.copy()`：`snapshot_frame` 只按 `ts` 分组，逐行
          字符串列在 5,600 码 × 2,600 行上就是几 GB。
        """
        parts = []
        for r in results:
            res = r.result if isinstance(r.result, dict) else None
            src = None if res is None else res.get('snap_src')
            if isinstance(src, pd.DataFrame) and not src.empty:
                parts.append(src)
        if not parts:
            return pd.DataFrame(columns=['ts', *SNAP_SRC_COLS,
                                         LTT.MAGIC_NINE_TURNS_TRUNK_TIMING_LAG]), 0
        return pd.concat(parts, ignore_index=True), len(parts)

    def snapshot_gap_codes(self, results) -> list:
        """本应当参与横截面、但**没有**产出 `snap_src` 的码（父进程补读）。

        旧路径聚合的是 `stock_reality_feat_60min` **全表**的码；逐码路径只有本次
        `run_codes` 的码 —— 差额来自「历史退市 / 不在 `stock_list` / 取数失败」。
        不补会让 `POLYNOMIAL9_NORM_MAJOR_MEDIAN`、`RETURN_COST*_MEDIAN` 等
        在部分 bar 上整体移位（2026-11-25 实测：5,636 vs 5,416 码，2,216 根 bar 不同）。
        ``distinct('code')`` 走索引，实测 0.05s。
        """
        from GolemQ import get_active_market
        try:
            have = set(get_active_market().get_database()[
                'stock_reality_feat_60min'].distinct('code'))
        except Exception:                       # noqa: BLE001 取不到就不补
            return []
        got = set()
        for r in results:
            res = r.result if isinstance(r.result, dict) else None
            src = None if res is None else res.get('snap_src')
            if isinstance(src, pd.DataFrame) and not src.empty:
                got.add(str(r.code))
        return sorted(have - got)

    def stored_stage_mode(self, timeline) -> np.ndarray:
        """窄口径：从 massive 表读回**既有**的 `STAGE_MODE`（不重算市场级快照）。

        `StageModeStreak` / `PosBottom` / `PosProbe` 都吃 `STAGE_MODE`；窄口径若
        把它当 NaN，这几个**会写库的**列就会从「按既有值推导」退化成
        `False` / `= onhold`，等于把历史改坏（评审 2026-11-25 指出）。
        """
        from GolemQ import get_active_market
        from GolemQ.core.constants import MARKET_TYPE
        ts = pd.DatetimeIndex(timeline)
        if len(ts) == 0:
            return np.full(0, np.nan)
        name = ('index_massive_reality_60min_models'
                if self.market_type == MARKET_TYPE.ETF_CN
                else 'stock_massive_reality_60min_models')
        lo, hi = int(ts[0].timestamp()), int(ts[-1].timestamp())
        try:
            cur = get_active_market().get_database()[name].find(
                {'time_stamp': {'$gte': lo, '$lte': hi}},
                {'time_stamp': 1, 'STAGE_MODE': 1, '_id': 0})
            _m = {int(d['time_stamp']): d.get('STAGE_MODE') for d in cur}
        except Exception:                       # noqa: BLE001 读不到 ⇒ 全 NaN
            _m = {}
        return np.array([_m.get(int(t.timestamp()), np.nan) for t in ts],
                        dtype=float)

    def run_code(self, code: str, **kwargs) -> CodeResult:
        # ⚠️ 快照源**先算**：`no_data` / 报错的码也要有现实列（旧路径是全表聚合，
        # 与这只码的 OHLC 是否算得出来无关）。快照源自己吞异常，不会污染 status。
        snap_src = self._snapshot_source(
            code, kwargs.get('snap_start'), kwargs.get('snap_end'))
        try:
            r = self.code_masks_and_pwrln(code, kwargs.get('as_of'))
            if r is None:
                return CodeResult(code=code, status='no_data',
                                  result={'snap_src': snap_src})
            ts, bull, bear, pwrln = r
            return CodeResult(code=code, status='ok',
                              result={'ts': ts, 'bull': bull, 'bear': bear,
                                      'pwrln_major': pwrln,
                                      'snap_src': snap_src})
        except Exception as exc:                    # noqa: BLE001
            return CodeResult(code=code, status='error',
                              error=f'{type(exc).__name__}: {exc}',
                              result={'snap_src': snap_src})

    # ------------------------------------------------------------------ #
    def market_timeline(self, as_of: pd.Timestamp) -> np.ndarray:
        """主时间轴：只保留 **10:30 / 11:30 / 14:00 / 15:00** 四根。

        ⚠️ `stock_60min` 里有杂散 13:00 行（实测 4 根），旧树只认四个标准
        stamp；不过滤会把脏 bar 当成有效快照（横截面成员数会明显偏少）。

        ⚠️ 返回的是**完整 2 倍回看**（≈4.4 年）—— 状态机（`StageModeStreak` /
        `PosReClear` 的「上一零仓段」）需要窗口之前的历史；**只落库最近
        `lookback_days`（800）天**在 `run_market` 末尾裁。窗口头部没有 233 根
        日线预热，Maj/仓位在那里不可信（2026-11-23 实测：2022-07-08 Maj -3.8
        vs 带预热回填 0.2）。
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

    @staticmethod
    def assert_snapshot_usable(snap, n_codes: int = 0, n_rows: int = 0) -> None:
        """全市场跑的**安全阀**：快照为空 ⇒ 中止，不写坏仓位列。

        空快照会让 `STAGE_MODE` 全 NaN ⇒ `StageModeStreak=0`、`PosBottom=False`、
        `PosProbe=onhold`，而这三列**是会写库的** —— 静默写坏比报错糟得多。
        """
        if (snap is None or not isinstance(snap, pd.DataFrame) or snap.empty
                or 'STAGE_MODE' not in snap.columns
                or pd.isna(snap['STAGE_MODE']).all()):
            raise RuntimeError(
                '[mainstream] 全市场快照契约算不出来（{} 码 / {:,} 行）：'
                '`STAGE_MODE` 全空。检查 `stock_reality_feat_60min` 是否有 '
                '`stage` 等源列 —— 宁可中止，也不要把 `StageModeStreak`/'
                '`PosBottom`/`PosProbe` 写成默认值。'.format(n_codes, n_rows))

    def snapshot_contracts(self,
                           features: Optional[pd.DataFrame] = None
                           ) -> pd.DataFrame:
        """**合并后的**逐码快照源表 → 每根 bar 一行的契约列（MED/MODE/COMBO）。

        :param features: `merge_snapshot_sources()` 的产物（长表，
            列 `ts` / `code` / `SNAP_SRC_COLS` / `M9tTrkLag`）。源列由逐码 worker
            读回并算好（`_snapshot_source`），本方法**不碰库**。
            ``None`` / 空帧 ⇒ 返回空帧。

        旧树对应：`calc_massive_snapshot_compact`（`features/empirical.py:2656`）
        —— 在 `ret_features_pd` 上按 bar 一次 `groupby` 出全部 Med/Mode。
        """
        from GolemQ.analysis.timing import calc_event_timing_lag
        from GolemQ.pipeline.mainstream.snapshot import snapshot_frame
        if features is None or len(features) == 0:
            return pd.DataFrame()
        if 'ts' not in features.columns:
            return pd.DataFrame()
        t0 = _stage('快照契约：{:,} 行 → 按 bar 聚合 Med/Mode…'
                    .format(len(features)))
        snap = snapshot_frame(features, ts_col='ts')
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
        _stage('快照契约：聚合完成', t0)
        return snap

    def run_market(self, codes: Iterable[str], as_of=None,
                   pool_size: int = 0, market_scope: bool = True) -> pd.DataFrame:
        """全市场横截面 → 每根 60min bar 的仓位行。

        **并行执行走 `BasePipeline.run_codes`**（与 compact / review 同一套）：
        一码一个任务、统一进度条、统一超时与统计。本类**不再自己写 `Parallel`** ——
        平行实现只会分叉（用户 2026-11-12 明确）。

        :param market_scope: 横截面是否**全市场**。`MasQuadPwrlnMajMed/Mode` 与
            全部**快照契约列**（`STAGE_MODE` / 各 MEDIAN / M9T COMBO）都是市场级
            序列，只对「全市场」有定义；`False`（如 `--eval fast` / 指定代码）
            时它们全部置空 —— ⚠️ 写库侧对 NaN 是**跳过不覆盖**
            （`save_position_series`），所以窄口径跑不会把历史的市场级值冲掉。
        """
        as_of = pd.Timestamp(as_of or GQ_util_last_trade_as_of())
        codes = [str(c) for c in codes]
        t0 = _stage(f'准备：{len(codes)} 码，as_of={as_of:%Y-%m-%d %H:%M}')
        timeline = self.market_timeline(as_of)
        t0 = _stage(
            '时间轴 {} 根 bar（{} ~ {}）'.format(
                len(timeline), timeline[0], timeline[-1]) if len(timeline)
            else '时间轴为空', t0)
        # 快照源列的读取窗口 = 时间轴窗口（时间轴之外的 bar 本来就会被 reindex 丢弃）
        results, _stats = self.run_codes(
            codes, pool_size=pool_size, verbose=self.verbose, as_of=as_of,
            snap_start=pd.Timestamp(timeline[0]) if len(timeline) else None,
            snap_end=as_of)
        t0 = _stage('逐码完成 {}/{} —— 聚合横截面…'.format(
            sum(1 for r in results if r.ok), len(results)), t0)

        total = np.zeros((3, len(timeline)), dtype=np.int32)
        T = len(timeline)
        # 横截面 `4QPwrlnMaj` —— 每根 bar 的**中位数**（`MasQuadPwrlnMaj`）
        pwrln_mat = np.full((len(results), T), np.nan, dtype=np.float32)
        for _i, r in enumerate(results):
            if not r.ok or not isinstance(r.result, dict):
                continue
            ts = np.asarray(r.result['ts'])
            bull = np.asarray(r.result['bull'], dtype=bool)
            bear = np.asarray(r.result['bear'], dtype=bool)
            pwr = np.asarray(r.result.get('pwrln_major', ()), dtype=float)
            if len(ts) == 0:
                continue
            pos = np.searchsorted(timeline, ts)
            ok = (pos < T) & (timeline[np.clip(pos, 0, T - 1)] == ts)
            if not ok.any():
                continue
            p = pos[ok]
            total[0] += np.bincount(p[bull[ok]], minlength=T)
            total[1] += np.bincount(p[bear[ok]], minlength=T)
            total[2] += np.bincount(p, minlength=T)
            if len(pwr) == len(ts):
                pwrln_mat[_i, p] = pwr[ok]

        with np.errstate(invalid='ignore'):
            import warnings as _w
            with _w.catch_warnings():
                _w.simplefilter('ignore', category=RuntimeWarning)
                mas_pwrln_maj_med = np.nanmedian(pwrln_mat, axis=0)
                _mat = pd.DataFrame(pwrln_mat)
                _mode = _mat.mode(axis=0, dropna=True)
                mas_pwrln_maj_mode = np.full(T, np.nan)
                if len(_mode):
                    mas_pwrln_maj_mode[np.asarray(_mode.columns)] = \
                        _mode.iloc[0].to_numpy(dtype=float)
            if not market_scope:
                # 市场级序列只对全市场横截面有定义；窄口径置空（写库跳过 NaN）。
                mas_pwrln_maj_med = np.full(T, np.nan)
                mas_pwrln_maj_mode = np.full(T, np.nan)

        res = calc_position_series(total[2], total[0], total[1])
        _ts = pd.DatetimeIndex(timeline)
        if market_scope:
            # 快照契约：逐码 worker 已经读回源列并算好 `M9tTrkLag`，这里只合并 + 聚合
            t0 = _stage(f'横截面聚合完成（{T} 根 bar）—— 合并逐码快照源…', t0)
            ret_features_pd, n_codes = self.merge_snapshot_sources(results)
            t0 = _stage(f'快照源合并：{n_codes} 码 / {len(ret_features_pd):,} 行', t0)
            # 补齐「表里有、本次没贡献」的码（退市 / 不在 stock_list / 取数失败）——
            # 旧路径是按全表聚合的，不补会让各 MEDIAN 在部分 bar 上整体移位。
            gap = self.snapshot_gap_codes(results)
            if gap:
                t0 = _stage('快照源补齐：{} 个码（表里有、本次未贡献）…'.format(
                    len(gap)), t0)
                _extra = [self._snapshot_source(c, _ts[0] if len(_ts) else None,
                                                as_of) for c in gap]
                _extra = [x for x in _extra if len(x)]
                if _extra:
                    ret_features_pd = pd.concat(
                        [ret_features_pd, *_extra], ignore_index=True)
                    n_codes += len(_extra)
                t0 = _stage(f'快照源补齐完成：合计 {n_codes} 码 / '
                            f'{len(ret_features_pd):,} 行', t0)
            snap = self.snapshot_contracts(ret_features_pd)
            self.assert_snapshot_usable(snap, n_codes, len(ret_features_pd))
            t0 = _stage('快照契约 {} 根 bar / {} 列'.format(
                0 if snap is None else len(snap),
                0 if snap is None else len(snap.columns)), t0)
        else:
            # 市场级序列（含快照契约）只对全市场横截面有定义 ⇒ 窄口径不算、置空，
            # 写库跳过 NaN ⇒ 库内既有值保持不动（用户 2026-11-25 确认）。
            snap = None
            t0 = _stage('窄口径（market_scope=False）：市场级快照列不重算，'
                        '改从 massive 表读回既有 `STAGE_MODE` 喂状态机', t0)

        def _snap_col(name):
            if snap is None or snap.empty or name not in snap.columns:
                return np.full(len(timeline), np.nan)
            return snap[name].reindex(_ts).to_numpy(dtype=float)

        # 窄口径也要有 `STAGE_MODE`（`StageModeStreak`/`PosBottom`/`PosProbe` 吃它，
        # 且这些列**会写库**）—— 用库里的既有值，不重算、不写回。
        stage_mode = (_snap_col('STAGE_MODE') if market_scope
                      else self.stored_stage_mode(timeline))
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
        t0 = _stage('仓位状态机完成 —— 清仓暴击指数扫描 / 组装输出…', t0)
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

        # `STAGE_MODE` 未知的 bar（窄口径库里还没有、或全市场缺 stage）⇒ 这四列
        # 也必须是**未知**（NaN，写库跳过），不能算成 0/False/onhold 写进历史。
        _sm_ok = np.isfinite(np.asarray(stage_mode, dtype=float))

        def _sm_known(v):
            return pd.Series(np.asarray(v, dtype=float)).where(_sm_ok)

        out = pd.DataFrame({
            'datetime': pd.DatetimeIndex(timeline),
            'mberCnt': total[2],
            'PosBsln': res['m_bull'],
            'PosPad': res['m_omega'],
            'PosNegBsln': res['m_bear'],
            'MasPosOnhold': res['onhold'],
            # 横截面 `4QPwrlnMaj` 的中位数 / 众数（用户 2026-11-21）
            'MasQuadPwrlnMajMed': mas_pwrln_maj_med,
            'MasQuadPwrlnMajMode': mas_pwrln_maj_mode,
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
            'StageModeStreak': _sm_known(stage_streak),
            'PosBottom': _sm_known(flags['PosBottom']),
            'PosDeepBottom': _sm_known(flags['PosDeepBottom']),
            'PosProbe': _sm_known(flags['PosProbe']),
        })
        if snap is not None and not snap.empty:
            for _c in snap.columns:
                if _c == 'STAGE_MODE':
                    continue
                out[_c] = _snap_col(_c)
        out = out[out['mberCnt'] > 0]
        # ⚠️ **只返回/落库最近 `lookback_days`（800）自然日**：时间轴与状态机
        # 在完整的 2 倍回看上算（窗口前史保留），但窗口头部没有 233 根日线预热，
        # Maj/仓位不可信 —— 不写库（更早的行保留既有值）。见 `market_timeline`。
        if self.lookback_days > 0 and len(out):
            floor = pd.Timestamp(as_of) - pd.Timedelta(
                days=int(self.lookback_days))
            out = out[pd.to_datetime(out['datetime']) >= floor]
        _stage('输出 {} 行（{} ~ {}）'.format(
            len(out), out['datetime'].iloc[0], out['datetime'].iloc[-1])
            if len(out) else '输出 0 行（没有可判定的 bar）', t0)
        return out.reset_index(drop=True)


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

    cpu = worker_count(pct_of_cores)

    # 只有「全市场股票池」的横截面才定义 `MasQuadPwrlnMajMed/Mode`；
    # `--eval fast|block|etc*|代码清单` 时置空，避免用十几个码的「中位数」
    # 覆盖掉历史的市场级值。
    full_market = (codes is None
                   and str(eval_range).lower() in ('all', 'stock', 'stocks'))

    t0 = _stage('scope={}：{} 码，{} 进程{}'.format(
        eval_range, len(codelist), cpu,
        '' if full_market else '，窄口径（市场级 Maj 列置空）'))
    pipe = MainstreamPipeline(lookback_days=lookback_days, verbose=verbose)
    df = pipe.run_market(codelist, as_of=(as_of or None), pool_size=cpu,
                         market_scope=full_market)
    _stage(f'run_market 完成：{len(df)} 行', t0)
    return df, {'total_count': len(codelist), 'rows': len(df)}
