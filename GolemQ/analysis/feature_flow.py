# coding: utf-8
"""QuadrantGear 的 Hamilton 数据流（列级 DAG，按需执行）。

为什么用 Hamilton：`:mod:`GolemQ.analysis.quadrant_gear` 的 O 档要 8 个输入，
其中日线四象限（`qd`）、日线 MACD、60min MACD 会被多条下游复用。手写链
（`MainstreamPipeline.code_features` 与示例各拼一遍）既容易分叉，也会把同一个
`calc_four_quadrant_boost(annual=252)` 算两次。

本模块把「算法」写成普通函数，函数**参数名即依赖节点名**：

.. code-block:: text

    q60 / q15 / qd          各频段四象限，各算一次
    pwrln(_minor/_major)    三个频段的 4QPwrln，各对齐一次
    macd60 / macd_day       各算一次
    qlev_frame              ← pwrln, pwrln_minor, pwrln_major, macd_delta_major
    qlev_macd_lag           ← qlev_frame
    quad_downrisk(_major)   ← q60 / qd（日线四象限不重算）
    zd_lag                  ← CZSC；只在 prelude 被请求时执行
    prelude_frame           ← QuadPreUprBf / QuadPreEpBf
    ms60 / ms_day          主力吸筹/派发，各算一次
    ms_leadin_maj           ← ms_day 的日线 leadin 贴回 60min
    ms_leadout_maj          ← ms_day 的日线 leadout 贴回 60min
    bs_day                 日线星线/买卖线（MAJOR）
    qlev_pwrln_upr_lag      ← QLevPwrlnUprLag（Charm_I 用）
    charm_frame             ← QLevCharmOBf / QLevCharmIBf
    quad_gear0              ← 上述 9 列（第五行含 Charm_O OR）

调用 `driver.execute(["QuadGear0"])` 时 Hamilton 只执行它的祖先；共享节点
（`qd` 等）只执行一次。需要跨运行缓存时再给 Builder 加 `.with_cache()`。

⚠️ **节点名必须是合法 Python 标识符**，所以 `4QPwrln*` 这类存储别名留在
四象限帧内部，不做节点名；`QuadGear0` / `QuadGear0Bf` 本身是合法标识符，
由 :func:`extract_columns` 直接暴露。

⚠️ **本模块只做算术，不取数、不写库**。三个 OHLC 帧由调用方注入；
`quadrant_gear_features` 是给示例/管线用的一次性入口。
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from hamilton.function_modifiers import extract_columns, hamilton_exclude

from GolemQ.analysis.indicators import calc_macd_qa
from GolemQ.analysis.mainstream_timing import (broadcast_daily_to_60min,
                                               calc_prelude_before,
                                               calc_zen_dash_causal)
from GolemQ.analysis.quadrant_gear import calc_quadrant_gear_o
from GolemQ.analysis.quadrant_jit import (calc_bs_before, calc_bs_signals,
                                          calc_four_quadrant_boost,
                                          calc_mainstream_feats,
                                          calc_quadrant_downrisk_major,
                                          calc_quadrant_leverage_charm,
                                          calc_quadrant_leverage_macd_timing_lag,
                                          calc_quadrant_powerline_uprisal_lag,
                                          calc_quadrant_stardown_count)
from GolemQ.core.constants import AKA, FIELD as FLD
from GolemQ.models.alias import LTT

__all__ = ['q60', 'q15', 'qd', 'macd60', 'macd_day_full', 'macd_day',
           'dea_day', 'dea_major', 'macd_delta_major',
           'pwrln', 'pwrln_minor', 'pwrln_major',
           'qlev_frame', 'qlev_macd_lag', 'quad_downrisk',
           'quad_downrisk_major', 'cost15_prz', 'qlev_pwrln_upr_lag',
           'ms60', 'ms_day', 'ms_leadout', 'ms_leadin_maj',
           'ms_leadout_maj', 'bs_day',
           'bs_star_up_bf_maj', 'bs_star_dwn_bf_maj', 'bs_bu_bf_maj',
           'bs_sel_bf_maj', 'charm_frame', 'zd_lag', 'prelude_frame',
           'quad_gear0', 'quadrant_gear_features', 'quadrant_gear_frame',
           'build_driver', 'NODE_ALIASES']

#: 节点名 → 存储别名。Hamilton 节点名用 snake_case；落库/展示时按这张表改名。
NODE_ALIASES = {
    'qlev_macd_lag': LTT.QUADRANT_LEVERAGE_MACD_TIMING_LAG_DUMMY,
    'quad_downrisk_major': LTT.QUADRANT_DOWNRISK_TIMING_LAG_MAJOR,
    'ms_leadin_maj': LTT.MAINSTREAM_LEADIN_BEFORE_MAJOR,
    'ms_leadout_maj': LTT.MAINSTREAM_LEADOUT_BEFORE_MAJOR,
    LTT.QUADRANT_LEVERAGE_CHARM_O_BEFORE:
        LTT.QUADRANT_LEVERAGE_CHARM_O_BEFORE,
    'bs_star_up_bf_maj': LTT.BS_STAR_UP_BEFORE_MAJOR,
    'bs_star_dwn_bf_maj': LTT.BS_STAR_DOWN_BEFORE_MAJOR,
    'bs_bu_bf_maj': LTT.BS_BUY_BEFORE_MAJOR,
    'bs_sel_bf_maj': LTT.BS_SELL_BEFORE_MAJOR,
}


# --------------------------------------------------------------------------- #
# 对齐工具：Hamilton 节点之间传值，索引必须回到主频
# --------------------------------------------------------------------------- #
def _dt_index(index) -> pd.DatetimeIndex:
    if isinstance(index, pd.MultiIndex):
        return pd.DatetimeIndex(pd.to_datetime(index.get_level_values(0)))
    return pd.DatetimeIndex(pd.to_datetime(index))


def _align_minor(series: pd.Series, index60) -> pd.Series:
    """15min 序列 → 60min 网格（同一时间戳取最后一根）。"""
    s = pd.Series(series).copy()
    s.index = _dt_index(s.index)
    s = s[~s.index.duplicated(keep='last')].sort_index()
    return pd.Series(s.reindex(_dt_index(index60)).to_numpy(),
                     index=index60, dtype=float)


def _align_daily(series: pd.Series, index60) -> pd.Series:
    """日线序列 → 60min 网格（交易日 D 的值落到 D 当天 4 根 bar）。"""
    v = broadcast_daily_to_60min(series, index60)
    return pd.Series(np.asarray(v, dtype=float), index=index60)


# --------------------------------------------------------------------------- #
# 各频段四象限：三个节点，各算一次
# --------------------------------------------------------------------------- #
def q60(ohlc60: pd.DataFrame, code: str) -> pd.DataFrame:
    """60min 四象限（annual=1008）。"""
    return calc_four_quadrant_boost(
        code=code, features=pd.DataFrame(index=ohlc60.index),
        ohlc_data=ohlc60, annual=1008)


def q15(ohlc15: pd.DataFrame, code: str) -> pd.DataFrame:
    """15min 四象限（annual=4032）。"""
    return calc_four_quadrant_boost(
        code=code, features=pd.DataFrame(index=ohlc15.index),
        ohlc_data=ohlc15, annual=4032)


def qd(ohlc_day: pd.DataFrame, code: str) -> pd.DataFrame:
    """日线四象限（annual=252）。

    ⚠️ `quad_downrisk_major` 直接复用本帧，**不会**再跑一遍
    `calc_four_quadrant_boost(annual=252)`。
    """
    return calc_four_quadrant_boost(
        code=code, features=pd.DataFrame(index=ohlc_day.index),
        ohlc_data=ohlc_day, annual=252)


# --------------------------------------------------------------------------- #
# MACD：60min / 日线各一次
# --------------------------------------------------------------------------- #
def macd60(ohlc60: pd.DataFrame) -> pd.Series:
    return calc_macd_qa(pd.to_numeric(ohlc60[AKA.CLOSE], errors='coerce'))[2]


def macd_day_full(ohlc_day: pd.DataFrame) -> pd.DataFrame:
    """日线 MACD 的 dif / dea / hist（一次 `calc_macd_qa`）。"""
    dif, dea, hist = calc_macd_qa(
        pd.to_numeric(ohlc_day[AKA.CLOSE], errors='coerce'))
    return pd.DataFrame({'dif': dif, 'dea': dea, 'hist': hist},
                        index=ohlc_day.index)


def macd_day(macd_day_full: pd.DataFrame) -> pd.Series:
    return macd_day_full['hist']


def dea_day(macd_day_full: pd.DataFrame) -> pd.Series:
    return macd_day_full['dea']


def dea_major(ohlc60: pd.DataFrame, dea_day: pd.Series) -> pd.Series:
    """日线 DEA → 60min（`DEA_MAJOR`）。"""
    return _align_daily(dea_day, ohlc60.index)


def macd_delta_major(ohlc60: pd.DataFrame, macd_day: pd.Series) -> pd.Series:
    """日线 MACD 柱差分 → 60min（`MACD_DELTA_MAJOR`）。"""
    return _align_daily(macd_day.diff(1), ohlc60.index)


# --------------------------------------------------------------------------- #
# QLevMACDLagD：用 q60 / q15 / qd 的同一份结果拼输入帧
# --------------------------------------------------------------------------- #
def pwrln(q60: pd.DataFrame, ohlc60: pd.DataFrame) -> pd.Series:
    """60min `4QPwrln`。"""
    return pd.Series(q60[LTT.QUADRANT_POWERLINE].to_numpy(),
                     index=ohlc60.index, name=LTT.QUADRANT_POWERLINE)


def pwrln_minor(q15: pd.DataFrame, ohlc60: pd.DataFrame) -> pd.Series:
    """15min `4QPwrln` → 60min（`4QPwrlnMin`）。"""
    return _align_minor(q15[LTT.QUADRANT_POWERLINE], ohlc60.index).rename(
        LTT.QUADRANT_POWERLINE_MINOR)


def pwrln_major(qd: pd.DataFrame, ohlc60: pd.DataFrame) -> pd.Series:
    """日线 `4QPwrln` → 60min（`4QPwrlnMaj`）。"""
    return _align_daily(qd[LTT.QUADRANT_POWERLINE], ohlc60.index).rename(
        LTT.QUADRANT_POWERLINE_MAJOR)


def qlev_frame(pwrln: pd.Series, pwrln_minor: pd.Series,
               pwrln_major: pd.Series, macd_delta_major: pd.Series,
               ohlc60: pd.DataFrame) -> pd.DataFrame:
    """QLev 链整帧：5 个能量中间列 + `QLevMACDLagD`（一次算完）。"""
    f = pd.DataFrame(index=ohlc60.index)
    f[LTT.QUADRANT_POWERLINE] = np.asarray(pwrln, dtype=float)
    f[LTT.QUADRANT_POWERLINE_MINOR] = np.asarray(pwrln_minor, dtype=float)
    f[LTT.QUADRANT_POWERLINE_MAJOR] = np.asarray(pwrln_major, dtype=float)
    f[FLD.MACD_DELTA_MAJOR] = np.asarray(macd_delta_major, dtype=float)
    # ⚠️ 该函数内部 reindex 后会返回新帧，必须接返回值（管线里也是这么用的）。
    return calc_quadrant_leverage_macd_timing_lag(f)


def qlev_macd_lag(qlev_frame: pd.DataFrame) -> pd.Series:
    """`QLevMACDLagD`（从 `qlev_frame` 取，不重算）。"""
    return qlev_frame[LTT.QUADRANT_LEVERAGE_MACD_TIMING_LAG_DUMMY]


def quad_downrisk(q60: pd.DataFrame, ohlc60: pd.DataFrame) -> pd.Series:
    """60min `QuadDwnRskLag`（非 MAJOR 版，prelude 要用）。"""
    return pd.Series(q60[LTT.QUADRANT_DOWNRISK_TIMING_LAG].to_numpy(),
                     index=ohlc60.index,
                     name=LTT.QUADRANT_DOWNRISK_TIMING_LAG)


def quad_downrisk_major(qd: pd.DataFrame, ohlc60: pd.DataFrame,
                        code: str) -> pd.Series:
    """`QuadDwnRskLagMaj`，复用 `qd`（日线四象限只算一次）。"""
    s = calc_quadrant_downrisk_major(
        code, None, ohlc60.index, feat_day=qd)
    return pd.Series(s.to_numpy(), index=ohlc60.index, name=s.name)


def cost15_prz(ohlc60: pd.DataFrame) -> pd.Series:
    """60min `cost15Prz`；帧里没有该列时返回 NaN（Charm_O 自动判假）。"""
    if AKA.COST15_PRICE in ohlc60.columns:
        v = pd.to_numeric(ohlc60[AKA.COST15_PRICE], errors='coerce')
        return pd.Series(v.to_numpy(), index=ohlc60.index,
                         name=AKA.COST15_PRICE)
    return pd.Series(np.nan, index=ohlc60.index, name=AKA.COST15_PRICE)


def qlev_pwrln_upr_lag(qlev_frame: pd.DataFrame,
                       bs_star_dwn_bf_maj: pd.Series,
                       dea_major: pd.Series,
                       ohlc60: pd.DataFrame) -> pd.Series:
    """`QLevPwrlnUprLag`（Charm_I 用）；复用 `qlev_frame`，不重算 QLev。"""
    f = pd.DataFrame(index=ohlc60.index)
    f[LTT.QUADRANT_LEVERAGE_ENERGY_TIMING_LAG_DUMMY] = np.asarray(
        qlev_frame[LTT.QUADRANT_LEVERAGE_ENERGY_TIMING_LAG_DUMMY],
        dtype=float)
    f[LTT.BS_STAR_DOWN_BEFORE_MAJOR] = np.asarray(
        bs_star_dwn_bf_maj, dtype=float)
    f = calc_quadrant_stardown_count(f)
    f[LTT.QUADRANT_POWERLINE_MAJOR] = np.asarray(
        qlev_frame[LTT.QUADRANT_POWERLINE_MAJOR], dtype=float)
    f[FLD.DEA_MAJOR] = np.asarray(dea_major, dtype=float)
    f = calc_quadrant_powerline_uprisal_lag(f)
    return f[LTT.QUADRANT_LEVERAGE_POWERLINE_UPRISING_TIMING_LAG]


def zd_lag(ohlc60: pd.DataFrame, code: str) -> pd.Series:
    """`zDLagD`（新版无后视镜）。只有下游请求 prelude 时才会被执行。"""
    return calc_zen_dash_causal(ohlc60, code)


@extract_columns(LTT.QUADRANT_PRELUDE_UPRISING_BEFORE,
                 LTT.QUADRANT_PRELUDE_ENDPOINT_BEFORE)
def prelude_frame(qlev_macd_lag: pd.Series, quad_downrisk_major: pd.Series,
                  quad_downrisk: pd.Series, zd_lag: pd.Series,
                  ohlc60: pd.DataFrame, window: int) -> pd.DataFrame:
    """`QuadPreUprBf` / `QuadPreEpBf`（一次 `calc_prelude_before`）。"""
    f = pd.DataFrame({
        LTT.QUADRANT_LEVERAGE_MACD_TIMING_LAG_DUMMY: qlev_macd_lag,
        LTT.QUADRANT_DOWNRISK_TIMING_LAG_MAJOR: quad_downrisk_major,
        LTT.QUADRANT_DOWNRISK_TIMING_LAG: quad_downrisk,
        FLD.ZEN_DASH_TIMING_LAG_DUMMY: zd_lag,
    }, index=ohlc60.index)
    out = calc_prelude_before(f, window=int(window))
    return out[[LTT.QUADRANT_PRELUDE_UPRISING_BEFORE,
                LTT.QUADRANT_PRELUDE_ENDPOINT_BEFORE]]


# --------------------------------------------------------------------------- #
# 主力吸筹/派发：60min 与日线各一次
# --------------------------------------------------------------------------- #
def ms60(ohlc60: pd.DataFrame, macd60: pd.Series) -> pd.DataFrame:
    """60min 主力吸筹 + 派发（fold=True，高价折叠）。"""
    f = pd.DataFrame(index=ohlc60.index)
    f[FLD.MACD] = np.asarray(macd60, dtype=float)
    return calc_mainstream_feats(ohlc60, f, fold=True)


def ms_day(ohlc_day: pd.DataFrame, macd_day: pd.Series) -> pd.DataFrame:
    """日线主力吸筹 + 派发（fold=True），供 `MsLeadoutBfMaj`。"""
    f = pd.DataFrame(index=ohlc_day.index)
    f[FLD.MACD] = np.asarray(macd_day, dtype=float)
    return calc_mainstream_feats(ohlc_day, f, fold=True)


def ms_leadout(ms60: pd.DataFrame) -> pd.Series:
    """60min `MsLeadoutBf`。"""
    return ms60[LTT.MAINSTREAM_LEADOUT_BEFORE]


def ms_leadin_maj(ohlc60: pd.DataFrame, ms_day: pd.DataFrame) -> pd.Series:
    """日线 `MsLeadinBf` → 60min（`MsLeadinBfMaj`，日线计数不换单位）。"""
    return _align_daily(ms_day[LTT.MAINSTREAM_LEADIN_BEFORE], ohlc60.index)


def ms_leadout_maj(ohlc60: pd.DataFrame, ms_day: pd.DataFrame) -> pd.Series:
    """日线 `MsLeadoutBf` → 60min（`MsLeadoutBfMaj`，日线计数不换单位）。"""
    return _align_daily(ms_day[LTT.MAINSTREAM_LEADOUT_BEFORE], ohlc60.index)


# --------------------------------------------------------------------------- #
# 日线星线/买卖线：一个节点出整帧，四个 MAJOR 列各贴回主频
# --------------------------------------------------------------------------- #
def bs_day(ohlc_day: pd.DataFrame) -> pd.DataFrame:
    """日线 `calc_bs_signals` + 重命名为 `*_MAJOR` + `calc_bs_before`。"""
    f = calc_bs_signals(ohlc_day, pd.DataFrame(index=ohlc_day.index))
    f[LTT.BS_BUY_MAJOR] = f[LTT.BS_BUY]
    f[LTT.BS_SELL_MAJOR] = f[LTT.BS_SELL]
    f[LTT.BS_STAR_UP_MAJOR] = f[LTT.BS_STAR_UP]
    f[LTT.BS_STAR_DOWN_MAJOR] = f[LTT.BS_STAR_DOWN]
    return calc_bs_before(f)


def bs_star_up_bf_maj(ohlc60: pd.DataFrame, bs_day: pd.DataFrame) -> pd.Series:
    return _align_daily(bs_day[LTT.BS_STAR_UP_BEFORE_MAJOR], ohlc60.index)


def bs_star_dwn_bf_maj(ohlc60: pd.DataFrame, bs_day: pd.DataFrame) -> pd.Series:
    return _align_daily(bs_day[LTT.BS_STAR_DOWN_BEFORE_MAJOR], ohlc60.index)


def bs_bu_bf_maj(ohlc60: pd.DataFrame, bs_day: pd.DataFrame) -> pd.Series:
    return _align_daily(bs_day[LTT.BS_BUY_BEFORE_MAJOR], ohlc60.index)


def bs_sel_bf_maj(ohlc60: pd.DataFrame, bs_day: pd.DataFrame) -> pd.Series:
    return _align_daily(bs_day[LTT.BS_SELL_BEFORE_MAJOR], ohlc60.index)


@extract_columns(LTT.QUADRANT_LEVERAGE_CHARM_O_BEFORE,
                 LTT.QUADRANT_LEVERAGE_CHARM_I_BEFORE)
def charm_frame(qlev_macd_lag: pd.Series, quad_downrisk_major: pd.Series,
                quad_downrisk: pd.Series, qlev_pwrln_upr_lag: pd.Series,
                cost15_prz: pd.Series,
                ohlc60: pd.DataFrame) -> pd.DataFrame:
    """`QLevCharmOBf` / `QLevCharmIBf`（QuadGear0 第五行的两个 OR 分支）。"""
    f = pd.DataFrame({
        LTT.QUADRANT_LEVERAGE_MACD_TIMING_LAG_DUMMY: qlev_macd_lag,
        LTT.QUADRANT_DOWNRISK_TIMING_LAG_MAJOR: quad_downrisk_major,
        LTT.QUADRANT_DOWNRISK_TIMING_LAG: quad_downrisk,
        LTT.QUADRANT_LEVERAGE_POWERLINE_UPRISING_TIMING_LAG:
            qlev_pwrln_upr_lag,
        AKA.COST15_PRICE: cost15_prz,
        AKA.CLOSE: pd.to_numeric(ohlc60[AKA.CLOSE], errors='coerce'),
    }, index=ohlc60.index)
    out = calc_quadrant_leverage_charm(f)
    return out[[LTT.QUADRANT_LEVERAGE_CHARM_O_BEFORE,
                LTT.QUADRANT_LEVERAGE_CHARM_I_BEFORE]]


# --------------------------------------------------------------------------- #
# 终点：QuadGear0 / QuadGear0Bf
# --------------------------------------------------------------------------- #
@extract_columns(LTT.QUADRANT_GEAR_O, LTT.QUADRANT_GEAR_O_BEFORE)
def quad_gear0(qlev_macd_lag: pd.Series, quad_downrisk_major: pd.Series,
               bs_star_up_bf_maj: pd.Series, bs_star_dwn_bf_maj: pd.Series,
               bs_bu_bf_maj: pd.Series, bs_sel_bf_maj: pd.Series,
               ms_leadin_maj: pd.Series, ms_leadout_maj: pd.Series,
               QLevCharmOBf: pd.Series,
               ohlc60: pd.DataFrame) -> pd.DataFrame:
    """O 档。输入列来自 DAG；`calc_quadrant_gear_o` 只做算术。"""
    f = pd.DataFrame({
        LTT.QUADRANT_LEVERAGE_MACD_TIMING_LAG_DUMMY: qlev_macd_lag,
        LTT.QUADRANT_DOWNRISK_TIMING_LAG_MAJOR: quad_downrisk_major,
        LTT.BS_STAR_UP_BEFORE_MAJOR: bs_star_up_bf_maj,
        LTT.BS_STAR_DOWN_BEFORE_MAJOR: bs_star_dwn_bf_maj,
        LTT.BS_BUY_BEFORE_MAJOR: bs_bu_bf_maj,
        LTT.BS_SELL_BEFORE_MAJOR: bs_sel_bf_maj,
        LTT.MAINSTREAM_LEADIN_BEFORE_MAJOR: ms_leadin_maj,
        LTT.MAINSTREAM_LEADOUT_BEFORE_MAJOR: ms_leadout_maj,
        LTT.QUADRANT_LEVERAGE_CHARM_O_BEFORE: QLevCharmOBf,
    }, index=ohlc60.index)
    out = calc_quadrant_gear_o(f, fold=True)
    return out[[LTT.QUADRANT_GEAR_O, LTT.QUADRANT_GEAR_O_BEFORE]]


# --------------------------------------------------------------------------- #
# 入口
# --------------------------------------------------------------------------- #
@hamilton_exclude
def build_driver():
    """构建本模块的 Hamilton Driver（每次调用新建，线程间不共享状态）。"""
    from hamilton import driver as hamilton_driver
    return hamilton_driver.Builder().with_modules(
        sys.modules[__name__]).build()


@hamilton_exclude
def quadrant_gear_features(code: str, ohlc60: pd.DataFrame,
                           ohlc15: pd.DataFrame, ohlc_day: pd.DataFrame,
                           *, driver=None) -> pd.DataFrame:
    """一次性入口：三帧 → 与 `ohlc60` 等长的 `QuadGear0` / `QuadGear0Bf`。

    只执行这两个节点的祖先；`qd`、`q60`、`q15`、`macd_day`、`ms60`、`ms_day`
    各算一次。不取数、不写库。
    """
    dr = driver or build_driver()
    out = dr.execute(
        [LTT.QUADRANT_GEAR_O, LTT.QUADRANT_GEAR_O_BEFORE],
        inputs={'code': str(code), 'ohlc60': ohlc60, 'ohlc15': ohlc15,
                'ohlc_day': ohlc_day})
    return pd.DataFrame(out, index=ohlc60.index)


@hamilton_exclude
def quadrant_gear_frame(code: str, ohlc60: pd.DataFrame,
                        ohlc15: pd.DataFrame, ohlc_day: pd.DataFrame,
                        *, driver=None) -> pd.DataFrame:
    """整条链的中间列（存储别名），供检查 / 训练 / 落库前预览。

    `NODE_ALIASES` 里的 8 个输入列 + `QuadGear0` / `QuadGear0Bf`，索引与
    `ohlc60` 对齐。同样是一次 `execute`，共享节点不会重算。
    """
    dr = driver or build_driver()
    nodes = list(NODE_ALIASES)
    out = dr.execute(
        nodes + [LTT.QUADRANT_GEAR_O, LTT.QUADRANT_GEAR_O_BEFORE],
        inputs={'code': str(code), 'ohlc60': ohlc60, 'ohlc15': ohlc15,
                'ohlc_day': ohlc_day})
    frame = {NODE_ALIASES[n]: out[n] for n in nodes}
    frame[LTT.QUADRANT_GEAR_O] = out[LTT.QUADRANT_GEAR_O]
    frame[LTT.QUADRANT_GEAR_O_BEFORE] = out[LTT.QUADRANT_GEAR_O_BEFORE]
    return pd.DataFrame(frame, index=ohlc60.index)
