# coding: utf-8
"""`examples/app_*.py` 的**共享骨架**（用户 2026-11-15：统一架构）。

六个 Streamlit 演示里，除 `app_chartscan.py`（ZMQ/YOLO 服务端演示）外，
其余五个的差别只是「**开哪些层**」—— 取数、attach、Mainstream feats、
K 线骨架、Mainstream 标记、ChartScan 带、Charm 点、上升系数链**全部相同**。
本模块就是这些公共件，app 里只保留「标题 + 侧边栏开关 + 自己的 layer」。

约定
====
* **不 import 时就画任何东西**（`set_page_config` / `st.title` 归各 app）；
* 所有 `st.cache_data` 都在这里（`load_base` / `_ohlc_of`），app 不再各缓存一份；
* 纯函数优先，方便 `streamlit.testing.AppTest` 与单元测试直接调。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from czsc import CZSC
from czsc.objects import Freq, RawBar

from GolemQ import get_active_market
from GolemQ.analysis import feature_flow as ff
from GolemQ.analysis.quadrant_jit import (calc_mainstream_feats,
                                          calc_mainstream_major_feats)
from GolemQ.analysis.reality_attach import attach_reality_feats
from GolemQ.core.constants import AKA, FIELD as FLD
from GolemQ.models.alias import LTT

from _gear_bands import trigger_runs

FREQS = {
    'day': (Freq.D, 'day', 900),
    '60min': (Freq.F60, 'min', 1500),
    '30min': (Freq.F30, 'min', 700),
    '15min': (Freq.F15, 'min', 360),
    '5min': (Freq.F5, 'min', 140),
}

MAX_BARS = 4096

#: 上升系数（`QUprICoefD`）。
COEF_COL = LTT.QUADRANT_UPRISING_I_COEF_DUMMY

#: review 的 as-of 特征只落 60min 表（D78）；其它频率只有 K 线。
RENKO_FREQ = '60min'

#: Charm_O 触发数 / Charm_I 触发数的**上限**（超过即视为口径异常，O 不画）。
CHARM_O_MAX_RATIO = 3.82


# --------------------------------------------------------------------------- #
# 侧边栏（公共三件）
# --------------------------------------------------------------------------- #
def sidebar_base(default_freq_index: int = 1, *, freqs=None,
                 with_history: bool = True, symbol_default: str = "000711",
                 symbol_help: str = "如 000711 / 600519 / 000858",
                 freq_help: str = None):
    """代码 / 频率 /（可选）加长历史 —— 所有 app 共用同一套侧边栏。

    :param freqs: 允许的频率清单；``None`` = `FREQS` 全部。`app_chartscan`
        只支持 ``day/60min/15min`` 三族，传 `FAMILIES` 进来。
    :param with_history: 画不画「加长历史」勾选（chartscan 不需要）。
    :returns: ``(symbol, frequency, long_history)``；`with_history=False` 时第三项为 None。
    """
    st.header("⚙️ 参数")
    symbol = st.text_input("股票代码", value=symbol_default,
                           help=symbol_help).strip()
    frequency = st.selectbox("K 线频率", list(freqs or FREQS),
                             index=default_freq_index, help=freq_help)
    long_history = (st.checkbox("加长历史（最多 4096 根）", value=False)
                    if with_history else None)
    return symbol, frequency, long_history


# --------------------------------------------------------------------------- #
# 上升系数链
# --------------------------------------------------------------------------- #
def _powerline(ohlc, code):
    """在给定帧上跑四量图，取 `4QPwrln`（电力线）。"""
    from GolemQ.analysis.quadrant_jit import calc_four_quadrant_boost
    f = calc_four_quadrant_boost(code, features=pd.DataFrame(index=ohlc.index),
                                 ohlc_data=ohlc, annual=252)
    return f[LTT.QUADRANT_POWERLINE]


@st.cache_data(show_spinner=False, ttl=600)
def _ohlc_of(symbol: str, kind: str, freq: str = None, start: str = None):
    """另取一份**别的频率**的 K 线帧（`kind` = ``'day'`` / ``'min'``）。

    ⚠️ `start` 是**暖机**：日线 = 首根 −252 天、15min = −192 天（老树口径）。
    """
    market = get_active_market()
    if kind == 'day':
        d, _ = market.get_kline_price_v3(symbol, start=start, realtime=False, verbose=False)
    else:
        d, _ = market.get_kline_price_min(symbol, start=start, frequency=freq,
                                          realtime=False, verbose=False)
    if d is None or len(getattr(d, 'data', ())) == 0:
        return None
    return d.data


def _qa_macd(ohlc):
    """**QUANTAXIS 口径**的 `(DIF, DEA, MACD柱)`（不是 talib.MACD）。"""
    from GolemQ.analysis.indicators import calc_macd_qa
    return calc_macd_qa(pd.to_numeric(ohlc[AKA.CLOSE], errors='coerce'))


def _macd_delta(ohlc):
    return _qa_macd(ohlc)[2].diff(1)


def _macd_dea(ohlc):
    return _qa_macd(ohlc)[1]


def _macd_cross_sx_before(ohlc):
    from GolemQ.analysis.timing import Timeline_duration
    dif, dea, _hist = _qa_macd(ohlc)
    cross = (dea > dif) & (dea.shift(1) <= dif.shift(1))
    return pd.Series(Timeline_duration(np.where(cross.fillna(False).to_numpy(), 1, 0)),
                     index=ohlc.index)


def _leadin_x5(ohlc):
    feats = pd.DataFrame(index=ohlc.index)
    feats[FLD.MACD] = _qa_macd(ohlc)[2].to_numpy()
    feats = calc_mainstream_feats(ohlc, feats)
    return feats[LTT.MAINSTREAM_LEADINx5_BEFORE]


def _align_by_timestamp(series, index):
    """同频/更高频按**精确时间戳**对齐（15min → 60min 不能按日期 ffill）。"""
    s = series.copy()
    if isinstance(s.index, pd.MultiIndex):
        s.index = pd.DatetimeIndex(pd.to_datetime(s.index.get_level_values(0)))
    else:
        s.index = pd.DatetimeIndex(pd.to_datetime(s.index))
    s = s[~s.index.duplicated(keep='last')]
    dates = pd.DatetimeIndex(pd.to_datetime(index.get_level_values(0)))
    return pd.Series(s.reindex(dates).to_numpy(), index=index)


def _align_by_date(series, index):
    """另一个频率按**日期**对齐并 ffill（日线/周线口径）。"""
    dates = pd.Index(pd.to_datetime(index.get_level_values(0))).normalize()
    src = series.copy()
    src.index = pd.Index(pd.to_datetime(src.index.get_level_values(0))).normalize()
    src = src[~src.index.duplicated(keep='last')]
    return pd.Series(src.reindex(dates).ffill().to_numpy(), index=index)


def calc_coef(code: str, ohlc, bi_list):
    """算 `LTT.QUADRANT_UPRISING_I_COEF_DUMMY` → ``(Series | None, 错误说明 | None)``。

    链路：`4QPwrln`（本频 / 15min / 日线）→ `QLevMACDLagD` → `QUprICoefD`
    （`analysis/quadrant_coef.py`）。`log10` 夹取要的两列
    （`4QuadZenDCnt` / `QLevPwrlnUprLag`）在这里按**最小表达式**现算。
    """
    from GolemQ.analysis.pivot import calc_zen_dash_timing_lag
    from GolemQ.analysis.quadrant_jit import (
        calc_bs_star_down, calc_quadrant_leverage_macd_timing_lag,
        calc_quadrant_powerline_uprisal_lag, calc_quadrant_stardown_count,
        calc_quadrant_zen_dash_count)
    from GolemQ.analysis.quadrant_coef import calc_uprising_i_coef
    from GolemQ.analysis.timing import Timeline_duration

    try:
        _first = pd.to_datetime(ohlc.index.get_level_values(0)[0])
        day_ohlc = _ohlc_of(code, 'day',
                            start=str((_first - pd.Timedelta(days=252)).date()))
        mn_ohlc = _ohlc_of(code, 'min', '15min',
                           start=str((_first - pd.Timedelta(days=192)).date()))
        if day_ohlc is None or mn_ohlc is None:
            return None, '取不到 15min / 日线的 K 线'

        f = pd.DataFrame(index=ohlc.index)
        f[LTT.QUADRANT_POWERLINE] = _powerline(ohlc, code).to_numpy()
        f[LTT.QUADRANT_POWERLINE_MAJOR] = _align_by_date(
            _powerline(day_ohlc, code), ohlc.index)
        f[LTT.QUADRANT_POWERLINE_MINOR] = _align_by_timestamp(
            _powerline(mn_ohlc, code), ohlc.index)
        f[FLD.MACD_DELTA_MAJOR] = _align_by_date(_macd_delta(day_ohlc), ohlc.index)
        f[FLD.DEA_MAJOR] = _align_by_date(_macd_dea(day_ohlc), ohlc.index)

        f[FLD.MACD_DELTA] = _macd_delta(ohlc).to_numpy()
        f[FLD.MACD_CROSS_SX_BEFORE] = _macd_cross_sx_before(ohlc).to_numpy()
        f[LTT.MAINSTREAM_LEADINx5_BEFORE_MINOR] = _align_by_timestamp(
            _leadin_x5(mn_ohlc), ohlc.index)

        _star = _align_by_date(calc_bs_star_down(day_ohlc).astype(float), ohlc.index)
        f[LTT.BS_STAR_DOWN_MAJOR] = _star.to_numpy()
        f[LTT.BS_STAR_DOWN_BEFORE_MAJOR] = Timeline_duration(
            np.where(f[LTT.BS_STAR_DOWN_MAJOR].to_numpy() > 1e-19, 1, 0))

        f[FLD.ZEN_DASH_TIMING_LAG_DUMMY] = calc_zen_dash_timing_lag(
            bi_list, ohlc.index).to_numpy()

        f[AKA.CLOSE] = pd.to_numeric(ohlc[AKA.CLOSE], errors='coerce')
        f = calc_quadrant_leverage_macd_timing_lag(f)
        f = calc_quadrant_stardown_count(f)
        f = calc_quadrant_powerline_uprisal_lag(f)
        f = calc_quadrant_zen_dash_count(f)
        f = calc_uprising_i_coef(f)
        return f[COEF_COL].astype(float), None
    except Exception as exc:                    # noqa: BLE001 演示里要给人看原因
        return None, '{}: {}'.format(type(exc).__name__, exc)


# --------------------------------------------------------------------------- #
# 取数 + attach + Mainstream + Charm
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False, ttl=600)
def load_base(symbol: str, frequency: str, long_history: bool,
              fold_mainstream: bool = True, with_coef: bool = True,
              with_dag: bool = True, with_czsc: bool = True):
    """取数 → attach → Mainstream（本频 + 日线 MAJOR）→ Charm/Rewind DAG → czsc 分笔。

    :param with_coef: 算不算上升系数（`QUprICoefD`，要另取两个频率，较慢）
    :param with_dag:  跑不跑 REWIND/STOPLOSS/Charm 的 feature_flow DAG
    :param with_czsc: 建不建 czsc 分笔（pivot app 画中枢用）

    → ``dict | None``：``df / name / ohlc / feats / ms / ms_major / bars /
    bi_list / coef / coef_err / feats_err / frequency``。``None`` 表示取不到 K 线。
    """
    freq_enum, kind, lookback_days = FREQS[frequency]
    market = get_active_market()
    if kind == 'day':
        start = str(pd.Timestamp.now() - pd.Timedelta(days=lookback_days))[:10] if long_history else None
        d, name = market.get_kline_price_v3(symbol, start=start, realtime=True, verbose=False)
    else:
        start = str(pd.Timestamp.now() - pd.Timedelta(days=lookback_days)) if long_history else None
        d, name = market.get_kline_price_min(symbol, start=start, frequency=frequency,
                                             realtime=True, verbose=False)

    if d is None or len(getattr(d, 'data', ())) == 0:
        return None
    ohlc = d.data.tail(MAX_BARS)
    if ohlc.empty:
        return {'df': pd.DataFrame(), 'name': name}

    _f = pd.DataFrame(index=ohlc.index)
    _f[FLD.MACD] = _qa_macd(ohlc)[2]
    ms = calc_mainstream_feats(ohlc, _f, fold=fold_mainstream)

    ms_major = None
    _day = None
    _ts0 = pd.Timestamp(pd.to_datetime(ohlc.index.get_level_values(0)).min())
    if frequency == '60min':
        _dstart = str((_ts0 - pd.Timedelta(days=900)).date())
        _res, _ = market.get_kline_price_v3(
            symbol, start=_dstart, end=str(pd.Timestamp.now().date()),
            realtime=True, verbose=False)
        _day = None if _res is None else getattr(_res, 'data', None)
        if _day is not None and len(_day):
            ms_major = calc_mainstream_major_feats(_day, fold=fold_mainstream)

    # attach（as-of 落库值；只有 60min 表有）
    feats = pd.DataFrame(index=ohlc.index)
    feats_err = None
    if frequency == RENKO_FREQ:
        try:
            feats, _loaded = attach_reality_feats(ohlc.copy(), symbol, verbose=False)
            if feats is not None and len(feats):
                feats = feats[~feats.index.duplicated(keep='last')]
                feats = feats.reindex(ohlc.index)
        except Exception as exc:                   # noqa: BLE001
            feats_err = f'{type(exc).__name__}: {exc}'

    # REWIND / STOPLOSS / Charm：复用 feature_flow 的共享节点。
    # ⚠️ 必须在 attach 之后：`Charm_O` 要 `cost15Prz`。
    if with_dag and frequency == RENKO_FREQ and _day is not None and len(_day):
        try:
            _r15, _ = market.get_kline_price_min(
                symbol,
                start=str((_ts0 - pd.Timedelta(days=192)).date()),
                end=str(pd.Timestamp.now().date()) + ' 23:59:59',
                frequency='15min', realtime=True, verbose=False)
            _m15 = None if _r15 is None else getattr(_r15, 'data', None)
            if _m15 is not None and len(_m15):
                _o = ohlc.copy()
                if AKA.COST15_PRICE in feats.columns:
                    _o[AKA.COST15_PRICE] = pd.to_numeric(
                        feats[AKA.COST15_PRICE], errors='coerce').to_numpy()
                _rw = ff.build_driver().execute(
                    ['ms_leadin_rewind_before_major',
                     'ms_leadout_stoploss_before_major',
                     'quad_downrisk_major',
                     'charm_frame'],
                    inputs={'code': symbol, 'ohlc60': _o,
                            'ohlc15': _m15, 'ohlc_day': _day})
                ms = ms.copy()
                ms[LTT.MAINSTREAM_LEADIN_REWIND_BEFORE_MAJOR] = _rw[
                    'ms_leadin_rewind_before_major']
                ms[LTT.MAINSTREAM_LEADOUT_STOPLOSS_BEFORE_MAJOR] = _rw[
                    'ms_leadout_stoploss_before_major']
                ms[LTT.QUADRANT_DOWNRISK_TIMING_LAG_MAJOR] = _rw[
                    'quad_downrisk_major']
                _charm = _rw['charm_frame']
                for _col in (LTT.QUADRANT_LEVERAGE_CHARM_O_BEFORE,
                             LTT.QUADRANT_LEVERAGE_CHARM_I_BEFORE):
                    ms[_col] = pd.Series(np.asarray(_charm[_col]),
                                         index=ohlc.index)
        except Exception as exc:                   # noqa: BLE001
            feats_err = f'{type(exc).__name__}: {exc}'

    # CCI_NORM：yellowgreen 带 / Charm 点的 y 基线
    try:
        import talib
        _h = pd.to_numeric(ohlc[AKA.HIGH], errors='coerce').to_numpy()
        _l = pd.to_numeric(ohlc[AKA.LOW], errors='coerce').to_numpy()
        _c = pd.to_numeric(ohlc[AKA.CLOSE], errors='coerce').to_numpy()
        _cci = talib.CCI(_h, _l, _c, timeperiod=14)
        feats = feats.copy()
        feats[FLD.CCI_NORM] = (_cci / 600.0 + 0.5)
    except Exception:                              # noqa: BLE001
        pass

    # czsc 分笔（pivot app 画中枢用；coef 链用 bi_list）
    bars, bi_list = None, None
    if with_czsc:
        bars = []
        for i, (idx, row) in enumerate(ohlc.iterrows()):
            dt = idx[0] if isinstance(idx, tuple) else idx
            bars.append(RawBar(symbol=symbol, dt=dt, id=i, freq=freq_enum,
                               open=float(row[AKA.OPEN]),
                               close=float(row[AKA.CLOSE]),
                               high=float(row[AKA.HIGH]),
                               low=float(row[AKA.LOW]),
                               vol=float(row.get(AKA.VOLUME, 0) or 0),
                               amount=float(row.get('amount', 0) or 0)))
        bi_list = CZSC(bars, max_bi_count=512, verbose=False).bi_list

    coef, coef_err = ((None, None) if not (with_coef and bi_list is not None)
                      else calc_coef(symbol, ohlc, bi_list))

    if bars is not None:
        df = pd.DataFrame({'dt': [b.dt for b in bars],
                           'open': [b.open for b in bars],
                           'high': [b.high for b in bars],
                           'low': [b.low for b in bars],
                           'close': [b.close for b in bars]})
    else:
        df = pd.DataFrame({
            'dt': pd.DatetimeIndex(pd.to_datetime(
                ohlc.index.get_level_values(0))),
            **{c: pd.to_numeric(ohlc[col], errors='coerce').to_numpy()
               for c, col in (('open', AKA.OPEN), ('high', AKA.HIGH),
                              ('low', AKA.LOW), ('close', AKA.CLOSE))}})
    return {'df': df, 'name': name, 'ohlc': ohlc, 'feats': feats, 'ms': ms,
            'ms_major': ms_major, 'bars': bars, 'bi_list': bi_list,
            'coef': coef, 'coef_err': coef_err, 'feats_err': feats_err,
            'frequency': frequency, 'day_ohlc': _day}


@st.cache_data(show_spinner=False, ttl=600)
def atr_major_bands(symbol: str, frequency: str, long_history: bool,
                    fold_mainstream: bool = True):
    """`ATR_LB_MAJ` / `ATR_UB_MAJ` —— compact 的日线派生列，按因果映射到当前帧。

    直接复用 `analysis.compact.calc_compact_features`（**不另写一份 ATR 公式**）；
    `include_rank=False` 跳过组合 rank（本页不用，且最慢）。
    **只有 60min** 才有日线帧 ⇒ 其它频率返回 None。
    """
    data = load_base(symbol, frequency, long_history, fold_mainstream)
    if data is None or data.get('day_ohlc') is None or data.get('ohlc') is None:
        return None
    from GolemQ.analysis.compact import calc_compact_features
    out = calc_compact_features(data['ohlc'], ohlc_day=data['day_ohlc'],
                                include_rank=False)
    return out[[FLD.ATR_LB_MAJOR, FLD.ATR_UB_MAJOR]]


# --------------------------------------------------------------------------- #
# 图骨架 / Mainstream 标记
# --------------------------------------------------------------------------- #
def make_figure(df) -> go.Figure:
    """K 线 + 公共 layout（各 app 再往上叠自己的 layer）。"""
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df['dt'], open=df['open'], high=df['high'], low=df['low'],
        close=df['close'],
        increasing_line_color='red', increasing_fillcolor='red',
        decreasing_line_color='green', decreasing_fillcolor='green',
        name='K线'))
    return fig


def finish_figure(fig) -> go.Figure:
    """公共 layout：高度 / 副轴 / 图例 / 边距。"""
    fig.update_layout(
        height=600, xaxis_rangeslider_visible=False, hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02),
        margin=dict(l=10, r=10, t=40, b=10),
        yaxis2=dict(title=COEF_COL, overlaying='y', side='right',
                    showgrid=False, zeroline=False))
    fig.update_xaxes(nticks=15)
    return fig


def major_values(ms, ms_major, col):
    """日线 MAJOR 列按日期**因果**映射到当前帧（15:00 用当日）。"""
    from GolemQ.analysis.compact import _map_causal
    s = pd.to_numeric(ms_major[col], errors='coerce').copy()
    s.index = pd.DatetimeIndex(
        pd.to_datetime(s.index.get_level_values(0))).normalize()
    s = s.groupby(level=0).last()
    return _map_causal(s, ms.index).to_numpy()


def trigger_mask(df, ms, ms_major, frequency, col, major_col):
    """Mainstream 触发掩码：60min 用日线 MAJOR（只取 15:00 那根），否则用本频。"""
    use_major = (frequency == '60min' and ms_major is not None
                 and len(ms_major) > 0)
    if use_major and major_col in ms_major.columns:
        ts = pd.DatetimeIndex(pd.to_datetime(df['dt']))
        return ((ts.hour == 15) & (ts.minute == 0)
                & (major_values(ms, ms_major, major_col) < 1e-12))
    if col not in ms.columns:
        return np.zeros(len(df), dtype=bool)
    return pd.to_numeric(ms[col], errors='coerce').to_numpy() < 1e-12


def _marker_y(df, feats, frequency, side):
    """标记 y：吸筹在 Prophet LB / low 下方，派发在 UB / high 上方。"""
    close = pd.to_numeric(df['close'], errors='coerce').to_numpy(float)
    low = pd.to_numeric(df['low'], errors='coerce').to_numpy(float)
    high = pd.to_numeric(df['high'], errors='coerce').to_numpy(float)
    lb = np.full(len(df), np.nan)
    ub = np.full(len(df), np.nan)
    if frequency == '60min' and feats is not None:
        for name, arr in (('prophetLb', lb), ('prophetUb', ub)):
            if name in feats.columns:
                arr[:] = pd.to_numeric(feats[name], errors='coerce').to_numpy()
    pad = np.maximum(close * 0.01, (ub - lb) * 0.15)
    pad = np.where(np.isfinite(pad) & (pad > 0), pad, close * 0.01)
    if side == 'up':
        base = np.maximum(np.where(np.isfinite(ub), ub, high), high)
        return base + pad
    base = np.minimum(np.where(np.isfinite(lb), lb, low), low)
    return base - pad


def add_mainstream_triggers(fig, data, show_leadin: bool, show_leadout: bool):
    """Mainstream leadin / leadout / rewind / stoploss 四个标记。"""
    df, feats = data['df'], data['feats']
    ms, ms_major, frequency = data['ms'], data['ms_major'], data['frequency']

    def _add(col, major_col, label, color, symbol, side):
        mask = trigger_mask(df, ms, ms_major, frequency, col, major_col)
        if mask.any():
            y = _marker_y(df, feats, frequency, side)
            fig.add_trace(go.Scatter(
                x=df['dt'].to_numpy()[mask], y=y[mask],
                mode='markers', name=label,
                marker=dict(symbol=symbol, size=10, color=color,
                            line=dict(width=1, color='white'))))

    if show_leadin:
        _add(LTT.MAINSTREAM_LEADIN_BEFORE, LTT.MAINSTREAM_LEADIN_BEFORE_MAJOR,
             'leadin（吸筹）', 'orange', 'triangle-up', 'down')
        _add(LTT.MAINSTREAM_LEADIN_REWIND_BEFORE_MAJOR, '',
             'leadin rewind（crimson）', 'crimson', 'triangle-up', 'down')
    if show_leadout:
        _add(LTT.MAINSTREAM_LEADOUT_BEFORE, LTT.MAINSTREAM_LEADOUT_BEFORE_MAJOR,
             'leadout（派发）', 'lime', 'triangle-down', 'up')
        _add(LTT.MAINSTREAM_LEADOUT_STOPLOSS_BEFORE_MAJOR, '',
             'leadout stoploss（olive）', 'olive', 'triangle-down', 'up')


# --------------------------------------------------------------------------- #
# ChartScan（Buy/Sell 三时钟）
# --------------------------------------------------------------------------- #
def cs_asof(feats, df, family):
    """图上每根 bar 的「最近一次 ChartScan 分类」（as-of，无后视镜）。"""
    col = f'Cls{family}'
    if feats is None or col not in feats.columns:
        return None
    s = feats[col].dropna()
    if s.empty:
        return None
    ts = pd.DatetimeIndex(pd.to_datetime(s.index.get_level_values(0)))
    order = np.searchsorted(ts.values,
                            pd.DatetimeIndex(pd.to_datetime(df['dt'])).values,
                            side='right') - 1
    out = np.full(len(df), '', dtype=object)
    ok = order >= 0
    out[ok] = s.to_numpy()[order[ok]]
    return out


def cs_before(feats, df, family, cls):
    """距最近一次 ChartScan `cls`（Buy/Sell）的图 bar 数（此前无该事件 ⇒ NaN）。"""
    col, ccol = f'Cls{family}', f'Coord{family}'
    if feats is None or col not in feats.columns or ccol not in feats.columns:
        return None
    sub = feats[[col, ccol]].dropna(subset=[col])
    ev_rows = sub[sub[col] == cls]
    if ev_rows.empty:
        return None
    times = pd.to_datetime(ev_rows[ccol].astype(float), unit='ms').dropna()
    if times.empty:
        return None
    idx = pd.DatetimeIndex(pd.to_datetime(df['dt']))
    ev = np.searchsorted(idx.values, times.values, side='right') - 1
    ev = np.unique(ev[ev >= 0])
    if len(ev) == 0:
        return None
    last = np.searchsorted(ev, np.arange(len(idx)), side='right') - 1
    out = np.full(len(idx), np.nan)
    ok = last >= 0
    out[ok] = np.arange(len(idx))[ok] - ev[last[ok]]
    return out


#: ChartScan 三时钟 → (族, 颜色, 名字)。Sell 画带；y 范围见 `add_chartscan_bands`。
CS_CLOCKS = (('15', 'lime', '15min'),
             ('60', 'olive', '60min'),
             ('Day', 'green', '日线'))


def _cci_bounds(feats):
    if feats is None or FLD.CCI_NORM not in feats.columns:
        return None
    cci = pd.to_numeric(feats[FLD.CCI_NORM], errors='coerce').dropna()
    if cci.empty:
        return None
    return float(cci.min()), float(cci.max())


def add_chartscan_sell_bands(fig, feats, df, echo=None):
    """三个时钟的 **Sell 状态带**（副轴 `y2`）。返回 ``[(name, 段数), ...]``。"""
    bounds = _cci_bounds(feats)
    if bounds is None:
        return []
    lo, hi = bounds
    ranges = {'15': (lo - 0.0618, hi - 0.0618),
              '60': (hi - 0.128, -0.168),
              'Day': (hi - 0.128, -0.168)}
    x = pd.DatetimeIndex(pd.to_datetime(df['dt']))
    pad = pd.Timedelta(minutes=30)
    out = []
    for fam, color, name in CS_CLOCKS:
        st_ = cs_asof(feats, df, fam)
        if st_ is None:
            continue
        band = pd.Series(np.where(st_ == 'Sell', 0.0, 1.0), index=x)
        runs = trigger_runs(band)
        if not runs:
            continue
        y0, y1 = ranges[fam]
        for a, b in runs:
            fig.add_shape(type='rect', xref='x', yref='y2',
                          x0=a - pad, x1=b + pad, y0=y0, y1=y1,
                          fillcolor=color, opacity=0.618,
                          line=dict(width=0), layer='below')
        out.append((name, len(runs)))
        if echo is not None:
            echo(f"ChartScan {name} Sell：{len(runs)} 段（{color}，副轴）")
    return out


def add_chartscan_buy_bands(fig, data, echo=None):
    """Buy 晚于 Leadout Maj **且**晚于日线 Sell 的全高带（orange / lightcoral）。"""
    feats, df = data['feats'], data['df']
    ms, ms_major, frequency = data['ms'], data['ms_major'], data['frequency']
    use_major = (frequency == '60min' and ms_major is not None
                 and len(ms_major) > 0)
    if not use_major or LTT.MAINSTREAM_LEADOUT_BEFORE_MAJOR not in ms_major.columns:
        return []
    leadout = major_values(ms, ms_major, LTT.MAINSTREAM_LEADOUT_BEFORE_MAJOR)
    day_sell = cs_before(feats, df, 'Day', 'Sell')
    if day_sell is None:
        return []
    pad = pd.Timedelta(minutes=30)
    out = []
    for fam, color, name in (('60', 'orange', '60min'),
                             ('Day', 'lightcoral', '日线')):
        buy = cs_before(feats, df, fam, 'Buy')
        if buy is None:
            continue
        m = (np.isfinite(buy) & np.isfinite(leadout) & np.isfinite(day_sell)
             & (buy < leadout) & (buy < day_sell))
        runs = trigger_runs(pd.Series(np.where(m, 0.0, 1.0), index=df['dt']))
        if not runs:
            continue
        for a, b in runs:
            fig.add_shape(type='rect', xref='x', yref='paper',
                          x0=a - pad, x1=b + pad, y0=0, y1=1,
                          fillcolor=color, opacity=0.22,
                          line=dict(width=0), layer='below')
        out.append((name, len(runs)))
        if echo is not None:
            echo(f"ChartScan {name} Buy < Leadout Maj & < 日线 Sell："
                 f"{len(runs)} 段（{color}，全高）")
    return out


# --------------------------------------------------------------------------- #
# Charm_O / Charm_I 触发点
# --------------------------------------------------------------------------- #
def add_charm_dots(fig, data, echo=None):
    """`QLevCharm{O,I}Bf < 1e-12` 的圆点（y = `CCI_NORM.min()-1`）。

    守卫：Charm_O 触发数 > ``CHARM_O_MAX_RATIO`` × Charm_I 时不画 O。
    """
    feats, ms, df = data['feats'], data['ms'], data['df']
    bounds = _cci_bounds(feats)
    if ms is None or bounds is None:
        return {}
    lo, _hi = bounds
    dot_y = lo - 1.0
    hits = {}
    for col in (LTT.QUADRANT_LEVERAGE_CHARM_O_BEFORE,
                LTT.QUADRANT_LEVERAGE_CHARM_I_BEFORE):
        if col not in ms.columns:
            continue
        v = pd.to_numeric(ms[col], errors='coerce').to_numpy()
        if len(v) != len(df):
            continue
        hits[col] = np.isfinite(v) & (v < 1e-12)
    zeros = np.zeros(0, dtype=bool)
    n_o = int(hits.get(LTT.QUADRANT_LEVERAGE_CHARM_O_BEFORE, zeros).sum())
    n_i = int(hits.get(LTT.QUADRANT_LEVERAGE_CHARM_I_BEFORE, zeros).sum())
    o_ok = n_o <= CHARM_O_MAX_RATIO * n_i

    def _draw(col, color, label):
        m = hits.get(col)
        if m is None or not m.any():
            return
        fig.add_trace(go.Scatter(
            x=df['dt'].to_numpy()[m], y=np.full(int(m.sum()), dot_y),
            mode='markers', name=f'{label} 触发',
            marker=dict(symbol='circle', size=7, color=color)))
        if echo is not None:
            echo(f"{label} 触发：{int(m.sum())} 处（{color}），"
                 f"y = `CCI_NORM.min()-1`。")

    if o_ok:
        _draw(LTT.QUADRANT_LEVERAGE_CHARM_O_BEFORE, 'purple', 'Charm_O')
    elif n_o and echo is not None:
        echo(f"Charm_O 触发 {n_o} 处 > {CHARM_O_MAX_RATIO}×Charm_I（{n_i}）"
             f"⇒ 口径异常，**不显示 Charm_O**。")
    _draw(LTT.QUADRANT_LEVERAGE_CHARM_I_BEFORE, 'red', 'Charm_I')
    return {'O': n_o, 'I': n_i}
