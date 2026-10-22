# coding: utf-8
"""缠论中枢 × RENKO 砖块界 × **上升系数** —— **叠加图**演示（新树 / MongoDB 8.3 版）

运行方式：

    streamlit run examples/app_pivot_renko_coef.py

与 `examples/app_pivot_renko.py` 的区别：**多画第三条曲线**
`LTT.QUADRANT_UPRISING_I_COEF_DUMMY`（上升系数），**叠在主图副轴**（同 `ax3 = ax1.twinx()`）、深红虚线
（`color='darkred'`, `linestyle='--'`, `lw=1`, `alpha=0.67`）。

三个东西各回答一个问题
======================
| | 来源 | 回答什么 |
|:--|:--|:--|
| **缠论中枢** | `analysis/pivot.py`（czsc 分笔 → 三笔重叠箱体） | **结构**：箱体上下沿 + 走势类型 |
| **RENKO 砖界** | `analysis/renko_jit.py`（固定砖高的砖块序列） | **趋势**：S 族 / L 族上下界，`RENKO_BAR` 只在两族同向时非零 |
| **上升系数** | `LTT.QUADRANT_UPRISING_I_COEF_DUMMY`（`QUprICoefD`） | 四量图链里的**优先级权重**（`portfolio/strategy.py` 的 `priority()` 用它排序） |

⚠️ 三者**没有共同的数学基础**，图上分歧是常态；本页**不做任何融合**。

上升系数怎么算（2026-10-19 二次核对后的**实际消费链**）
====================================================
老树对同一列有三条写法；用户看到的 `OmniPathUpR1CoefDmy` 是
`CANDIDATE × log10(有效 MACD lag)`（`mainstream.py:252-261` + `zen_pivot.py:272-285`），
**不是** `fourquad.py:717` 的参照序列回归：

```
本频 K线  ──calc_four_quadrant_boost──▶ 4QPwrln            ┐
15min     ──同上──────────────────────▶ 4QPwrln           ├─ QLevMACDLagD ──┐
日线      ──同上──────────────────────▶ 4QPwrln(+MACD柱)  ┘                 │
                                                                  ┌─────────┘
QLevMACDLagD ──getPoolRate──▶ QUprICoefCan ──× log10(lag)──▶ QUprICoefD
```

落地模块：**`analysis/quadrant_coef.py`**（`getPeak/ getPoolRate / getPoolRate_vX /
calc_wavlet_coef_of_fractal / calc_uprising_i_coef`）。

⚠️ `log10` 前的 lag 在老树里会**夹到 10**：`lag <= 10` 恒用 10；`lag > 10` 时
还要看 `4QuadZenDCnt` / `QLevPwrlnUprLag`。本页已按**最小表达式**现算这两列
（CZSC 笔方向 + 日线星线死叉 + 两条 event-lag，见 `_calc_coef`），
不再有「缺列 ⇒ 1.9% 偏差」那条。

⚠️ 上升系数**只画真实算出来的值**：算不出来就在页面上写清是哪一步（不造数）。

⚠️ `2026-10-19` 起该常量由 `TRD.OMNIPATH_UPRISING_I_COEF_DUMMY` **改名**而来，
值也从 `'OmniPathUpR1CoefDmy'` 改成 `'QUprICoefD'` —— 实测 4.4 / 8.3 **都没有行**
存过这一列，故无数据契约需要保留。
"""
import os
import sys

# --------------------------------------------------------------------------- #
# 让它能在「没装包、直接从源码跑」的情况下找到 GolemQ
# --------------------------------------------------------------------------- #
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from czsc import CZSC
from czsc.objects import Freq, RawBar

from GolemQ import get_active_market
from GolemQ.analysis import renko_jit as RENKO_JIT
from GolemQ.analysis.pivot import (
    bi_confirm_map, bi_list_to_points, calc_pivots, classify_pivots,
    pivots_to_df, plot_pivots_plotly,
)
from GolemQ.core.constants import AKA, FIELD as FLD
from GolemQ.models.alias import LTT

FREQS = {
    'day': (Freq.D, 'day', 900),
    '60min': (Freq.F60, 'min', 1500),
    '30min': (Freq.F30, 'min', 700),
    '15min': (Freq.F15, 'min', 360),
    '5min': (Freq.F5, 'min', 140),
}

MAX_BARS = 4096

#: 第三条曲线：上升系数（`QUprICoefD`）。
COEF_COL = LTT.QUADRANT_UPRISING_I_COEF_DUMMY

#: 走 jit 还是纯版 —— **指标卡上要如实标出来**（两者逐值相同，但速度差 ~6×）
_HAS_JIT = RENKO_JIT.available()
# D59 起 `renko_jit` 是唯一实现；`renko_trend_cross_func` 是 jit 版的兼容别名。
_renko_feats = RENKO_JIT.renko_trend_cross_func

st.set_page_config(page_title="中枢 × RENKO × 上升系数", layout="wide")
st.title("📦🧱📈 缠论中枢 × RENKO 砖块界 × 上升系数（叠加图）")

# --------------------------------------------------------------------------- #
# 侧边栏
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("⚙️ 参数")
    symbol = st.text_input("股票代码", value="000711", help="如 000711 / 600519 / 000858")
    frequency = st.selectbox("K 线频率", list(FREQS), index=1)
    long_history = st.checkbox("加长历史（最多 4096 根）", value=False)

    st.divider()
    st.caption("显示开关 —— 三种方法各自可关，便于对照")
    show_pivots = st.checkbox("缠论：中枢箱体", value=True)
    show_ggdd = st.checkbox("缠论：GG/DD 延伸边界", value=False)
    show_bi = st.checkbox("缠论：笔", value=False)
    show_renko_s = st.checkbox("RENKO：S 族砖界", value=True)
    show_renko_l = st.checkbox("RENKO：L 族砖界", value=True)
    show_signal = st.checkbox("RENKO：RENKO_BAR 信号", value=True)
    show_coef = st.checkbox("上升系数", value=True)

    st.divider()
    st.caption(f"RENKO 特征走 **{'jit' if _HAS_JIT else '纯 Python'}** 版"
               "；两版逐值相同，只是快慢差 ~6×。")


# --------------------------------------------------------------------------- #
# 数据加载
# --------------------------------------------------------------------------- #
def _powerline(ohlc, code):
    """在给定帧上跑四量图，取 `4QPwrln`（电力线）。

    `calc_four_quadrant_boost` 只依赖 `ohlc` 与 talib 指标，**不需要**外来的
    features —— 所以每个频率都能独立算一份。
    """
    from GolemQ.analysis.quadrant_jit import calc_four_quadrant_boost
    f = calc_four_quadrant_boost(code, features=pd.DataFrame(index=ohlc.index),
                                 ohlc_data=ohlc, annual=252)
    return f[LTT.QUADRANT_POWERLINE]


@st.cache_data(show_spinner=False, ttl=600)
def _ohlc_of(symbol: str, kind: str, freq: str = None, start: str = None):
    """另取一份**别的频率**的 K 线帧（`kind` = `'day'` / `'min'`）。

    ⚠️ `start` 是为了**暖机**：老树取日线是「60min 首根 **再往前 252 天**」
    （`models/mainstream.py:69`）、15min 是「**再往前 192 天**」
    （`features/reviews.py:2071`）。只取默认 800 天会与 60min 同起点，
    首日 `MACD_DELTA_MAJOR`/`powerline` 没有暖机 ⇒ 事件 lag 错位。
    """
    market = get_active_market()
    if kind == 'day':
        d, _ = market.get_kline_price_v3(symbol, start=start, realtime=False, verbose=False)
    else:
        d, _ = market.get_kline_price_min(symbol, start=start, frequency=freq,
                                          realtime=False, verbose=False)
    if d is None or len(getattr(d, 'data', ())) == 0:
        return None
    # ⚠️ **不要 tail(MAX_BARS)**：15min 的暖机窗口有 1 万多根，截到 4096 会
    # 把 60min 首根所在日期整段丢掉（对齐后全是 NaN）。MAX_BARS 只用于主图。
    return d.data


def _qa_macd(ohlc):
    """**QUANTAXIS 口径**的 `(DIF, DEA, MACD柱)` —— 老树主流链用的就是它。

    ⚠️ 不是 `talib.MACD`：QA 用 `ewm(adjust=True)` 且柱 `×2`（见
    `analysis/indicators.calc_macd_qa` 的 docstring）。用 talib 口径会让
    `MsLeadinx5BfMin` 有 1364/2075 根不同。
    """
    from GolemQ.analysis.indicators import calc_macd_qa
    return calc_macd_qa(pd.to_numeric(ohlc[AKA.CLOSE], errors='coerce'))


def _macd_delta(ohlc):
    """`MACD_DELTA` = **QA 口径 MACD 柱**的差分。"""
    return _qa_macd(ohlc)[2].diff(1)


def _macd_dea(ohlc):
    """`DEA`（QA 口径），用于日线 `DEA_MAJ`。"""
    return _qa_macd(ohlc)[1]


def _macd_cross_sx_before(ohlc):
    """`MACD_SX_BF` = `Timeline_duration(CROSS(DEA, DIF))`（老树 `indices.py:767-782`）。"""
    from GolemQ.analysis.timing import Timeline_duration
    dif, dea, _hist = _qa_macd(ohlc)
    cross = (dea > dif) & (dea.shift(1) <= dif.shift(1))
    return pd.Series(Timeline_duration(np.where(cross.fillna(False).to_numpy(), 1, 0)),
                     index=ohlc.index)


def _leadin_x5(ohlc):
    """15min 的 `MsLeadinx5Bf`（新树 `analysis/quadrant_jit.py::calc_mainstream_feats`）。"""
    from GolemQ.analysis.quadrant_jit import calc_mainstream_feats
    feats = pd.DataFrame(index=ohlc.index)
    feats[FLD.MACD] = _qa_macd(ohlc)[2].to_numpy()
    feats = calc_mainstream_feats(ohlc, feats)
    return feats[LTT.MAINSTREAM_LEADINx5_BEFORE]


def _align_by_timestamp(series, index):
    """把**同频/更高频**的序列按**精确时间戳**对齐到主索引。

    ⚠️ 15min → 60min **不能用按日期 ffill**：老树 `reviews.py:1876-1880` 是按
    `snapshot.name`（60min 的时间戳）去 15min 帧里取**同一时刻**那一行。
    按日期取当天最后一根会把 15:00 的值抹到 10:30/11:30/14:00 上 ——
    实测能让 `QLevPwrlnUprLag` 差出 148 根（2026-10-19）。
    """
    s = series.copy()
    if isinstance(s.index, pd.MultiIndex):
        s.index = pd.DatetimeIndex(pd.to_datetime(s.index.get_level_values(0)))
    else:
        s.index = pd.DatetimeIndex(pd.to_datetime(s.index))
    s = s[~s.index.duplicated(keep='last')]
    dates = pd.DatetimeIndex(pd.to_datetime(index.get_level_values(0)))
    return pd.Series(s.reindex(dates).to_numpy(), index=index)


def _align_by_date(series, index):
    """把**另一个频率**的序列按「日期」对齐到当前索引上（同 `fbprophet._reindex_by_date`）。"""
    dates = pd.Index(pd.to_datetime(index.get_level_values(0))).normalize()
    src = series.copy()
    src.index = pd.Index(pd.to_datetime(src.index.get_level_values(0))).normalize()
    src = src[~src.index.duplicated(keep='last')]
    # ⚠️ **ffill**：日线/周线的值要回填到其后的每一根 bar —— 老树合并周线帧
    # 也是 ffill（`features_week` → 日线帧）。只 reindex 会让非周五的 bar 全是 NaN。
    return pd.Series(src.reindex(dates).ffill().to_numpy(), index=index)


def _calc_coef(code: str, ohlc, bi_list):
    """算 `LTT.QUADRANT_UPRISING_I_COEF_DUMMY` → `(Series | None, 错误说明 | None)`。

    链路（`analysis/quadrant_coef.py` 的模块 docstring 有完整说明）：

        ohlc ──calc_four_quadrant_boost──▶ 4QPwrln（本频 / 15min / 日线）
             ──rename──▶ 4QPwrln / _MINOR / _MAJOR
             ──calc_quadrant_leverage_macd_timing_lag──▶ QLevMACDLagD
             ──calc_uprising_i_coef──▶ QUprICoefD

    ⚠️ 2026-10-19 起，`QUprICoefD` 的 `log10` 夹取还需要
    `4QuadZenDCnt` / `QLevPwrlnUprLag`；这两列在这里按**最小表达式**现算
    （ZEN 笔方向、星线死叉、两条 event-lag —— 不拖 ZEN/BS 整链）。
    """
    from GolemQ.analysis.pivot import calc_zen_dash_timing_lag
    from GolemQ.analysis.quadrant_jit import (
        calc_bs_star_down, calc_quadrant_leverage_macd_timing_lag,
        calc_quadrant_powerline_uprisal_lag, calc_quadrant_stardown_count,
        calc_quadrant_zen_dash_count)
    from GolemQ.analysis.quadrant_coef import calc_uprising_i_coef
    from GolemQ.analysis.timing import Timeline_duration

    try:
        # 暖机窗口照老树：日线 = 60min 首根 − 252 天；15min = − 192 天
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
        # ⚠️ 15min 用**精确时间戳**（不是按日期 ffill），见 `_align_by_timestamp`
        f[LTT.QUADRANT_POWERLINE_MINOR] = _align_by_timestamp(
            _powerline(mn_ohlc, code), ohlc.index)
        f[FLD.MACD_DELTA_MAJOR] = _align_by_date(_macd_delta(day_ohlc), ohlc.index)
        f[FLD.DEA_MAJOR] = _align_by_date(_macd_dea(day_ohlc), ohlc.index)

        # ---- 第二 OR 分支的三个 60min / 15min 输入 ----
        f[FLD.MACD_DELTA] = _macd_delta(ohlc).to_numpy()
        f[FLD.MACD_CROSS_SX_BEFORE] = _macd_cross_sx_before(ohlc).to_numpy()
        f[LTT.MAINSTREAM_LEADINx5_BEFORE_MINOR] = _align_by_timestamp(
            _leadin_x5(mn_ohlc), ohlc.index)

        # ---- 星线死叉（日线，展开到主频）→ 4QuadStDwnCnt ----
        _star = _align_by_date(calc_bs_star_down(day_ohlc).astype(float), ohlc.index)
        f[LTT.BS_STAR_DOWN_MAJOR] = _star.to_numpy()
        f[LTT.BS_STAR_DOWN_BEFORE_MAJOR] = Timeline_duration(
            np.where(f[LTT.BS_STAR_DOWN_MAJOR].to_numpy() > 1e-19, 1, 0))

        # ---- zDLagD：**只由 CZSC 笔方向决定**（不碰 ZEN 小波族）----
        f[FLD.ZEN_DASH_TIMING_LAG_DUMMY] = calc_zen_dash_timing_lag(
            bi_list, ohlc.index).to_numpy()

        f[AKA.CLOSE] = pd.to_numeric(ohlc[AKA.CLOSE], errors='coerce')
        f = calc_quadrant_leverage_macd_timing_lag(f)   # 含第二 OR 分支
        f = calc_quadrant_stardown_count(f)             # 一条 calc_energy
        f = calc_quadrant_powerline_uprisal_lag(f)      # 两条 event-lag
        f = calc_quadrant_zen_dash_count(f)             # 一条 calc_energy
        f = calc_uprising_i_coef(f)
        return f[COEF_COL].astype(float), None
    except Exception as exc:                    # noqa: BLE001 演示里要给人看原因
        return None, '{}: {}'.format(type(exc).__name__, exc)


@st.cache_data(show_spinner=False, ttl=600)
def load(symbol: str, frequency: str, long_history: bool):
    """取数 → RENKO 特征 → czsc 分笔 → 中枢 → （若已生产）上升系数。"""
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
        return None, name, None
    ohlc = d.data.tail(MAX_BARS)
    if ohlc.empty:
        return pd.DataFrame(), name, None

    # ---- RENKO 特征（同一份帧，x 轴与 K 线逐行对齐）----
    feats = _renko_feats(ohlc)

    # ---- czsc 分笔 → 中枢 ----
    bars = []
    for i, (idx, row) in enumerate(ohlc.iterrows()):
        dt = idx[0] if isinstance(idx, tuple) else idx
        bars.append(RawBar(symbol=symbol, dt=dt, id=i, freq=freq_enum,
                           open=float(row[AKA.OPEN]), close=float(row[AKA.CLOSE]),
                           high=float(row[AKA.HIGH]), low=float(row[AKA.LOW]),
                           vol=float(row.get(AKA.VOLUME, 0) or 0),
                           amount=float(row.get('amount', 0) or 0)))
    czsc = CZSC(bars, max_bi_count=512, verbose=False)
    conf_map = bi_confirm_map(bars, max_bi_count=512)
    pivots = calc_pivots(bi_list=czsc.bi_list, conf_map=conf_map, strict=True)
    cls = classify_pivots(pivots)
    bi_points = bi_list_to_points(czsc.bi_list)

    # ---- 上升系数（`analysis/quadrant_coef.py`）----
    # `QLevMACDLagD` 由 `4QPwrln` 的**三个频率**（15min / 本频 / 日线）合成
    # ⇒ 要另取两份帧。
    coef, coef_err = _calc_coef(symbol, ohlc, czsc.bi_list)

    dts = [b.dt for b in bars]
    return pd.DataFrame({'dt': dts,
                         'open': [b.open for b in bars],
                         'high': [b.high for b in bars],
                         'low': [b.low for b in bars],
                         'close': [b.close for b in bars]}), name, (feats, pivots, cls, bi_points, coef, coef_err)


try:
    df, name, bundle = load(symbol, frequency, long_history)
except Exception as exc:                            # noqa: BLE001
    st.error(f"数据加载失败：{exc}")
    st.stop()

if df is None:
    st.warning(f"{symbol} 在 {frequency} 上没有取到 K 线 —— 换只标的或换个频率试试。")
    st.stop()
if bundle is None or df.empty:
    st.warning(f"{symbol} 的 K 线不足 30 根 —— 三条方法都算不出来。")
    st.stop()

feats, pivots, cls, bi_points, coef, coef_err = bundle

# --------------------------------------------------------------------------- #
# 指标卡
# --------------------------------------------------------------------------- #
s_dir = int(feats[FLD.RENKO_TREND_S].astype(float).iloc[-1])
l_dir = int(feats[FLD.RENKO_TREND_L].astype(float).iloc[-1])
bar_sig = int(feats[FLD.RENKO_TREND].astype(float).iloc[-1])
_dir_word = {1: '🔴 上', -1: '🟢 下', 0: '— 平'}
_last_kind = cls['kinds'][-1] if cls.get('kinds') else '—'
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("📊 K 线数", f"{len(df):,}")
c2.metric("📦 中枢数", len(pivots))
c3.metric("🧭 走势类型", cls['kind'])
c4.metric("🧱 RENKO S / L", f"{_dir_word[s_dir]} / {_dir_word[l_dir]}")
# 末根常常是 NaN（分形长度不够/窗口越界）⇒ 取**最后一个非空值**
_coef_last = coef.dropna().tail(1) if coef is not None else None
c5.metric("📈 上升系数",
          f"{float(_coef_last.iloc[0]):.4f}" if _coef_last is not None and len(_coef_last)
          else '—')
_label = symbol if name == symbol else f'{name}（{symbol}）'
st.caption(f"标的：{_label} · {frequency} · 末个中枢：**{_last_kind}** · "
           f"末根 S={s_dir:+d} L={l_dir:+d} ⇒ RENKO_BAR={bar_sig:+d}")

# --------------------------------------------------------------------------- #
# 图 1：K 线 + 中枢 + RENKO（叠加）
# --------------------------------------------------------------------------- #
fig = go.Figure()
fig.add_trace(go.Candlestick(
    x=df['dt'], open=df['open'], high=df['high'], low=df['low'], close=df['close'],
    increasing_line_color='red', increasing_fillcolor='red',
    decreasing_line_color='green', decreasing_fillcolor='green',
    name='K线'))

if show_bi:
    fig.add_trace(go.Scatter(
        x=[p['dt'] for p in bi_points], y=[p['bi'] for p in bi_points],
        mode='lines+markers', name='笔',
        line=dict(color='royalblue', width=1.5),
        marker=dict(size=4, color='royalblue')))

if show_pivots:
    plot_pivots_plotly(fig, pivots, x_last=df['dt'].iloc[-1], show_ggdd=show_ggdd)

# RENKO S / L 两族砖界 —— 台阶线（`line_shape='hv'`）：砖界在两次分型之间恒定不变
_renko_bands = []
if show_renko_s:
    _renko_bands.append((FLD.RENKO_TREND_S_LB, FLD.RENKO_TREND_S_UB, 'royalblue', 'S'))
if show_renko_l:
    _renko_bands.append((FLD.RENKO_TREND_L_LB, FLD.RENKO_TREND_L_UB, 'darkorange', 'L'))
for lb, ub, col, tag in _renko_bands:
    for series, dash in ((lb, 'dot'), (ub, 'dash')):
        fig.add_trace(go.Scatter(
            x=df['dt'], y=feats[series].astype(float).values,
            mode='lines', name=series,
            line=dict(color=col, width=1, dash=dash), line_shape='hv',
            legendgroup=f'renko{tag}', legendgrouptitle_text=f'RENKO {tag} 族'))

if show_signal:
    sig = feats[FLD.RENKO_TREND].astype(float).values
    for val, col, nm, sym in ((1, 'red', 'RENKO_BAR = +1', 'triangle-up'),
                              (-1, 'green', 'RENKO_BAR = -1', 'triangle-down')):
        m = sig == val
        if m.any():
            fig.add_trace(go.Scatter(
                x=df['dt'][m], y=df['close'][m], mode='markers', name=nm,
                marker=dict(symbol=sym, size=8, color=col, opacity=0.75)))

# ---- 上升系数：**叠在主图的副轴**上（matplotlib 的 `ax3 = ax1.twinx()` 对应 Plotly 的 `yaxis='y2'`）----
# 参考 `OneWaveQuant/.../calc_4quad_poolcoef-bs-stage-of-zen.ipynb`：
#   ax3 = ax1.twinx(); ax3.plot(DATETIME_LABEL, features[COEF], color='darkred',
#                               linestyle='--', lw=1, alpha=0.67)
if show_coef and coef is not None:
    fig.add_trace(go.Scatter(
        x=df['dt'], y=coef.to_numpy(), mode='lines', name=COEF_COL,
        line=dict(color='darkred', dash='dash', width=1), opacity=0.67,
        yaxis='y2'))

fig.update_layout(
    height=600, xaxis_rangeslider_visible=False, hovermode='x unified',
    legend=dict(orientation='h', yanchor='bottom', y=1.02),
    margin=dict(l=10, r=10, t=40, b=10),
    # 副轴：叠在 y 上、放右侧 —— 与 `ax1.twinx()` 同一个意思
    yaxis2=dict(title=COEF_COL, overlaying='y', side='right',
                showgrid=False, zeroline=False))
fig.update_xaxes(nticks=15)
st.plotly_chart(fig, width='stretch')

# --------------------------------------------------------------------------- #
# 上升系数：**说明**（曲线已叠在主图副轴上，这里不再单画一张）
# --------------------------------------------------------------------------- #
st.subheader(f"📈 上升系数 `{COEF_COL}`")

if not show_coef:
    st.caption("已关闭（侧边栏）。")
elif coef is None:
    st.warning(
        f"**`{COEF_COL}` 这一次没算出来** —— `{coef_err}`。\n\n"
        "链路：`4QPwrln`（本频 / 15min / 日线）+ `MACD_DELTA_MAJOR`（日线）"
        " → `QLevMACDLagD` → `QUprICoefD`（`analysis/quadrant_coef.py`）。"
    )
else:
    _last = coef.dropna().tail(1)
    st.caption(
        f"已叠在**主图的右轴**上（matplotlib 的 `ax3 = ax1.twinx()` ≡ Plotly 的 "
        f"`yaxis='y2'`）：`darkred` 虚线 · `lw=1` · `alpha=0.67` —— 与 "
        "`calc_4quad_poolcoef-bs-stage-of-zen.ipynb` 的 `ax3.plot(...)` 同一口径。"
        f" 末个非空值 **{float(_last.iloc[0]):.4f}**（近 {int(coef.notna().sum())} 根有值）。"
    )

st.caption("⚠️ 三者**没有共同的数学基础**：中枢是「三笔重叠」的**结构**，RENKO 是「固定砖高」"
           "的**趋势**，上升系数是四量图链的**优先级权重**。图上分歧是常态，**本页不做融合**。")

# --------------------------------------------------------------------------- #
# 明细
# --------------------------------------------------------------------------- #
st.subheader("中枢明细")
st.dataframe(pivots_to_df(pivots, kind=cls['kinds']), width='stretch', hide_index=True)

with st.expander("RENKO 特征明细（末 200 行）"):
    cols = [c for c in (FLD.RENKO_PRICE_S, FLD.RENKO_TREND_S, FLD.RENKO_TREND_S_LB,
                        FLD.RENKO_TREND_S_UB, FLD.RENKO_TREND_L, FLD.RENKO_TREND_L_LB,
                        FLD.RENKO_TREND_L_UB, FLD.RENKO_TREND, FLD.RENKO_TREND_S_TIMING_LAG,
                        FLD.RENKO_BOOST_S_TIMING_LAG, FLD.RENKO_TREND_L_TIMING_LAG,
                        FLD.RENKO_BOOST_L_TIMING_LAG, COEF_COL) if c in feats.columns]
    st.dataframe(feats[cols].tail(200), width='stretch')
