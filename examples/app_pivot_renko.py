# coding: utf-8
"""缠论中枢 × RENKO 砖块界 —— **叠加图**演示（新树 / MongoDB 8.3 版）

运行方式：

    streamlit run examples/app_pivot_renko.py

⚠️ 文件名是 `app_pivot_renko`（**pivot**，与既有的 `examples/app_pivot.py` 一致）。
用户原话写的是「app_povit_renko」，按项目既有术语拼写订正。

这条演示要回答的问题是：**两条互不相干的方法在同一张图上说了什么**。

* **缠论中枢**（`analysis/pivot.py`）：从 `czsc` 的**笔**序列找出「至少三笔重叠」的
  箱体。它给的是**结构**（箱体上下沿 + 走势类型 盘整/上涨/下跌）。
* **RENKO 砖块界**（`*_reality_feat_60min` 的 Renko 14 列）：把价格压成固定砖高
  的砖块序列，给出 S 族（`renkoSbar` 及上下沿）与 L 族（`renkoLbar` 及上下沿）的
  上下界。它给的是**趋势**（`RENKO_BAR` 只在 S、L 同向时才给 ±1）。
  ⚠️ **只读 review 落库的 as-of 结果，不在本页重算**：`--review` 每个交易日 15:00
  用 ≤T 的 2160 根 60min 重算，盘中沿用上一交易日快照 —— 现场对整段历史跑
  `renko_trend_cross_func` 会把目标 bar 之后的数据带进来，是后视镜。

两者**没有共同的数学基础** —— 所以图上出现分歧是正常的，**分歧本身才是要看的东西**。
本页只把两条叠在一起、并把「末根一致吗」摆在指标卡上，**不做任何融合**。

数据链路：

    GolemQ.get_active_market().get_kline_price_min / get_kline_price_v3
        → markets/StockCN/kline83.py     # MongoDB 8.3 时序集合，已前复权
    czsc.CZSC                                # 分笔（bi_list）
    GolemQ.analysis.pivot                    # 中枢识别 / 走势分类 / 箱体绘图
    GolemQ.analysis.reality_attach           # attach `*_reality_feat_60min`
                                             #   → Renko 14 列（as-of，无后视镜）
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
from GolemQ.analysis.pivot import (
    bi_confirm_map, bi_list_to_points, calc_pivots, classify_pivots,
    pivots_to_df, plot_pivots_plotly,
)
from GolemQ.analysis.reality_attach import attach_reality_feats
from GolemQ.core.constants import AKA, FIELD as FLD

FREQS = {
    'day': (Freq.D, 'day', 900),
    '60min': (Freq.F60, 'min', 1500),
    '30min': (Freq.F30, 'min', 700),
    '15min': (Freq.F15, 'min', 360),
    '5min': (Freq.F5, 'min', 140),
}

MAX_BARS = 4096

#: review 的 Renko 只落 60min 表（D78）；其它频率只有中枢/笔，没有 Renko。
RENKO_FREQ = '60min'

st.set_page_config(page_title="缠论中枢 × RENKO 叠加", layout="wide")
st.title("📦🧱 缠论中枢 × RENKO 砖块界（叠加图）")

# --------------------------------------------------------------------------- #
# 侧边栏
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("⚙️ 参数")
    symbol = st.text_input("股票代码", value="000711", help="如 000711 / 600519 / 000858")
    frequency = st.selectbox("K 线频率", list(FREQS), index=1)
    long_history = st.checkbox("加长历史（最多 4096 根）", value=False)

    st.divider()
    st.caption("显示开关 —— 两条方法各自可关，便于对照")
    show_pivots = st.checkbox("缠论：中枢箱体", value=True)
    show_ggdd = st.checkbox("缠论：GG/DD 延伸边界", value=False)
    show_bi = st.checkbox("缠论：笔", value=False)
    _renko_freq_ok = frequency == RENKO_FREQ
    show_renko_s = st.checkbox("RENKO：S 族砖界", value=True,
                               disabled=not _renko_freq_ok)
    show_renko_l = st.checkbox("RENKO：L 族砖界", value=True,
                               disabled=not _renko_freq_ok)
    show_signal = st.checkbox("RENKO：RENKO_BAR 信号", value=True,
                              disabled=not _renko_freq_ok)

    st.divider()
    st.caption("RENKO 列来自 `*_reality_feat_60min`（`--review` 的 as-of 产出），"
               "本页**不重算**。落库只有 **60min**：两个交易时段之间的盘内 bar "
               "共用当日 15:00 的快照；其它频率只显示中枢/笔。")


# --------------------------------------------------------------------------- #
# 数据加载
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False, ttl=600)
def load(symbol: str, frequency: str, long_history: bool):
    """取数 → czsc 分笔 → 中枢 → RENKO 特征（全部在同一份帧上，故 x 轴天然对齐）。"""
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

    # ---- RENKO 列：attach `*_reality_feat_60min`（as-of，无后视镜）----
    # ⚠️ 只读落库值，**不**在页面上跑 `renko_trend_cross_func` —— 那是整段历史
    # 一次算完，会把目标 bar 之后的数据带进来。落库只有 60min 族（D78）。
    feats = pd.DataFrame(index=ohlc.index)
    renko_err = None
    if frequency == RENKO_FREQ:
        try:
            feats, _loaded = attach_reality_feats(ohlc.copy(), symbol,
                                                  verbose=False)
        except Exception as exc:                   # noqa: BLE001
            # 失败不毁中枢：留空 + 把原因带到页面上
            renko_err = f'{type(exc).__name__}: {exc}'
    renko_ok = (FLD.RENKO_TREND_S in feats.columns
                and pd.to_numeric(feats[FLD.RENKO_TREND_S],
                                  errors='coerce').notna().any())

    # ---- czsc 分笔 → 中枢 ----
    # ⚠️ `CZSC([])` 会抛 IndexError（czsc 0.7.10 的 analyze.py:228 直接取 bars[0].symbol）
    # —— 空结果 ≠ 错误，前面已判过空，这里才敢构造。
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

    dts = [b.dt for b in bars]
    return pd.DataFrame({'dt': dts,
                         'open': [b.open for b in bars],
                         'high': [b.high for b in bars],
                         'low': [b.low for b in bars],
                         'close': [b.close for b in bars]}), name, (
        feats, pivots, cls, bi_points, renko_ok, renko_err)


try:
    df, name, bundle = load(symbol, frequency, long_history)
except Exception as exc:                            # noqa: BLE001
    st.error(f"数据加载失败：{exc}")
    st.stop()

if df is None:
    st.warning(f"{symbol} 在 {frequency} 上没有取到 K 线 —— 换只标的或换个频率试试。")
    st.stop()
if bundle is None or df.empty:
    st.warning(f"{symbol} 的 K 线不足 30 根 —— 两条方法都算不出来。")
    st.stop()

feats, pivots, cls, bi_points, renko_ok, renko_err = bundle


def _last_int(col: str):
    """最后一个非 NaN 值（attach 的末几行可能还没落库）。"""
    if col not in feats.columns:
        return None
    s = pd.to_numeric(feats[col], errors='coerce').dropna()
    return None if s.empty else int(s.iloc[-1])


# --------------------------------------------------------------------------- #
# 指标卡 —— 两条方法各自的状态，并排看
# --------------------------------------------------------------------------- #
s_dir = _last_int(FLD.RENKO_TREND_S)
l_dir = _last_int(FLD.RENKO_TREND_L)
bar_sig = _last_int(FLD.RENKO_TREND)
_dir_word = {1: '🔴 上', -1: '🟢 下', 0: '— 平'}


def _dir_txt(v) -> str:
    return '—' if v is None else _dir_word.get(v, str(v))


_last_kind = cls['kinds'][-1] if cls.get('kinds') else '—'
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("📊 K 线数", f"{len(df):,}")
c2.metric("✏️ 笔数", len(bi_points))
c3.metric("📦 中枢数", len(pivots))
c4.metric("🧭 走势类型", cls['kind'])
c5.metric("🧱 RENKO S / L",
          f"{_dir_txt(s_dir)} / {_dir_txt(l_dir)}" if renko_ok else '—')
c6.metric("🎯 RENKO_BAR", _dir_txt(bar_sig) if renko_ok else '—')
# 展示名由读取器解析（`kline83._display_name`：stock_info → stock_list）。
# 解析不到时它**回落成代码**（真指数、未知代码、`name` 为空的标的）——
# 那种情况下别再印一次代码：`301308（301308）` 没有信息量。
_label = symbol if name == symbol else f'{name}（{symbol}）'
if renko_ok:
    _fmt = lambda v: f'{v:+d}' if v is not None else '—'
    st.caption(f"标的：{_label} · {frequency} · 末个中枢：**{_last_kind}** · "
               f"末根 S={_fmt(s_dir)} L={_fmt(l_dir)} ⇒ RENKO_BAR={_fmt(bar_sig)}"
               "（RENKO_BAR 只在 S、L **同向**时才非零；数据为 review 的 "
               "as-of 落库值）")
else:
    st.caption(f"标的：{_label} · {frequency} · 末个中枢：**{_last_kind}** · "
               "RENKO 列不可用（见下方说明）")
    if renko_err:
        st.warning(f"attach `*_reality_feat_60min` 失败：`{renko_err}`")
    elif frequency != RENKO_FREQ:
        st.info(f"RENKO 只落库 {RENKO_FREQ}；当前频率 `{frequency}` 只有中枢/笔。")
    else:
        st.info(f"`{symbol}` 的 `*_reality_feat_60min` 里没有 Renko 行 —— "
                "先跑 `--review` / `tools/backfill_renko.py`。")

# --------------------------------------------------------------------------- #
# 叠加图
# --------------------------------------------------------------------------- #
fig = go.Figure()

# K 线（红涨绿跌）
fig.add_trace(go.Candlestick(
    x=df['dt'], open=df['open'], high=df['high'], low=df['low'], close=df['close'],
    increasing_line_color='red', increasing_fillcolor='red',
    decreasing_line_color='green', decreasing_fillcolor='green',
    name='K线'))

# 笔
if show_bi:
    fig.add_trace(go.Scatter(
        x=[p['dt'] for p in bi_points], y=[p['bi'] for p in bi_points],
        mode='lines+markers', name='笔',
        line=dict(color='royalblue', width=1.5),
        marker=dict(size=4, color='royalblue')))

# 缠论中枢箱体（因果口径：自中枢成立 bar 起才画）
if show_pivots:
    plot_pivots_plotly(fig, pivots, x_last=df['dt'].iloc[-1], show_ggdd=show_ggdd)

# RENKO S / L 两族砖界 —— 落库的是还原后的有序上下沿（lb <= ub），直接用
# ⚠️ 台阶线（`line_shape='hv'`）才是对的画法：砖界在两次分型之间**恒定不变**，
# 用普通折线会把"没变化"画成斜线，看不出台阶。
# ⚠️ 这些列来自 `*_reality_feat_60min`，同一交易日 4 根共用一个 15:00 快照。
_renko_bands = []
if renko_ok and show_renko_s:
    _renko_bands.append((FLD.RENKO_TREND_S_LB, FLD.RENKO_TREND_S_UB, 'royalblue', 'S'))
if renko_ok and show_renko_l:
    _renko_bands.append((FLD.RENKO_TREND_L_LB, FLD.RENKO_TREND_L_UB, 'darkorange', 'L'))
for lb, ub, col, tag in _renko_bands:
    for series, dash in ((lb, 'dot'), (ub, 'dash')):
        if series not in feats.columns:
            continue
        fig.add_trace(go.Scatter(
            x=df['dt'], y=pd.to_numeric(feats[series], errors='coerce').values,
            mode='lines', name=series,
            line=dict(color=col, width=1, dash=dash), line_shape='hv',
            legendgroup=f'renko{tag}', legendgrouptitle_text=f'RENKO {tag} 族'))

# RENKO_BAR 信号
if renko_ok and show_signal and FLD.RENKO_TREND in feats.columns:
    sig = pd.to_numeric(feats[FLD.RENKO_TREND], errors='coerce').values
    close = pd.to_numeric(df['close'], errors='coerce').values
    for val, col, nm, sym in ((1, 'red', 'RENKO_BAR = +1', 'triangle-up'),
                              (-1, 'green', 'RENKO_BAR = -1', 'triangle-down')):
        m = sig == val
        if m.any():
            fig.add_trace(go.Scatter(
                x=df['dt'][m], y=close[m], mode='markers', name=nm,
                marker=dict(symbol=sym, size=8, color=col, opacity=0.75)))

fig.update_layout(
    height=680, xaxis_rangeslider_visible=False, hovermode='x unified',
    legend=dict(orientation='h', yanchor='bottom', y=1.02),
    margin=dict(l=10, r=10, t=40, b=10))
fig.update_xaxes(nticks=15)
st.plotly_chart(fig, width='stretch')

st.caption("⚠️ 两条方法**没有共同的数学基础**：中枢来自「三笔重叠」的**结构**，"
           "RENKO 来自「固定砖高的**趋势**」。图上的分歧是常态，**本页不做任何融合**。"
           "RENKO 是 review 的 as-of 落库值（每交易日 15:00、≤T 的 2160 根），"
           "不是本页重算 —— 所以同一交易日的 4 根是同一个台阶。")

# --------------------------------------------------------------------------- #
# 中枢明细
# --------------------------------------------------------------------------- #
st.subheader("中枢明细")
st.dataframe(pivots_to_df(pivots, kind=cls['kinds']), width='stretch', hide_index=True)

with st.expander("RENKO 特征明细（末 200 行）"):
    cols = [c for c in (FLD.RENKO_PRICE_S, FLD.RENKO_TREND_S, FLD.RENKO_TREND_S_LB,
                        FLD.RENKO_TREND_S_UB, FLD.RENKO_TREND_L, FLD.RENKO_TREND_L_LB,
                        FLD.RENKO_TREND_L_UB, FLD.RENKO_TREND, FLD.RENKO_TREND_S_TIMING_LAG,
                        FLD.RENKO_BOOST_S_TIMING_LAG, FLD.RENKO_TREND_L_TIMING_LAG,
                        FLD.RENKO_BOOST_L_TIMING_LAG) if c in feats.columns]
    if cols:
        st.dataframe(feats[cols].tail(200), width='stretch')
    else:
        st.caption("当前频率没有 Renko 落库列（只有 60min 表有）。")
