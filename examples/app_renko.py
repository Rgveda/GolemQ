# coding: utf-8
"""砖块图（**RENKO**）演示 —— 新树 / MongoDB 8.3 版

运行方式：

    streamlit run examples/app_renko.py

数据链路：

    GolemQ.get_active_market().get_kline_price_v3 / get_kline_price_min
        → markets/StockCN/kline83.py     # MongoDB 8.3 时序集合，已前复权
    GolemQ.analysis.renko.renko             # 砖块序列（真正的 Renko 图）
    GolemQ.analysis.renko.renko_trend_cross_func   # S/L 两族特征列

两个页签看的是**同一份数据的两个面**：

1. **砖块图** —— 按**砖序**画（x 轴是第几块砖，与时间无关）。这才是 Renko 的本体：
   价格每走够一个砖高才多一块，横盘时**不产生任何砖**，噪音被剔掉。
2. **特征叠加** —— 按**时间轴**画 S 族 / L 族的上界下界台阶线，以及两者的合成信号
   `RENKO_BAR`。

⚠️ **砖块图与特征叠加的「砖」不是同一批数**，这是刻意的：

* 砖块图用 `class renko`（`set_brick_size(auto=True)` → ATR 中位数定砖高），
  它给出**真正的砖块序列**（`renko_prices` + `renko_directions`）；
* 特征列走 `renko_trend_cross_func`，S 族用 numba 版 `renko_chart`（同一套砖高），
  L 族另起一套 —— **按 1200 bar 分窗、每窗用布伦特法搜自己的最优砖高**，
  所以两族的砖高**不同**，这也是它们叫 small / large 的原因。

⚠️ **别把 `source_aligned` 的两列当有序的上下界**：下跌砖上它返回的是
`[上一砖位, 上一砖位 - 砖高]`（**反序**）。本 demo 走的是 `renko_chart`，
它输出的 lb/ub **是有序的** —— 见 `analysis/renko.py` 的模块 docstring。
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

from GolemQ import get_active_market
from GolemQ.analysis.renko import renko, renko_trend_cross_func
from GolemQ.core.constants import AKA, FIELD as FLD

#: 频率 → (取数方式, 加长历史时回溯的自然日数)
#: ⚠️ **日线走 `get_kline_price_v3`，分钟走 `get_kline_price_min`** —— 两者签名不同，
#: 且日线无数据时返回 `None`（分钟返回空对象）。见 `PITFALLS.md` P3 的不对称契约。
FREQS = {
    'day': ('day', 900),
    '60min': ('min', 1500),
    '30min': ('min', 700),
    '15min': ('min', 360),
    '5min': ('min', 140),
    '1min': ('min', 40),
}

#: 演示要看的列（S 族 + L 族 + 合成）
SHOW_COLS = (FLD.RENKO_PRICE_S, FLD.RENKO_TREND_S, FLD.RENKO_TREND_S_LB,
             FLD.RENKO_TREND_S_UB, FLD.RENKO_PRICE_L, FLD.RENKO_TREND_L,
             FLD.RENKO_TREND_L_LB, FLD.RENKO_TREND_L_UB, FLD.RENKO_OPTIMAL,
             FLD.RENKO_TREND, FLD.RENKO_TREND_S_TIMING_LAG,
             FLD.RENKO_BOOST_S_TIMING_LAG, FLD.RENKO_TREND_L_TIMING_LAG,
             FLD.RENKO_BOOST_L_TIMING_LAG)

st.set_page_config(page_title="砖块图（RENKO）演示", layout="wide")
st.title("🧱 砖块图（RENKO）演示")

# --------------------------------------------------------------------------- #
# 侧边栏
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("⚙️ 参数")
    symbol = st.text_input("股票代码", value="000711", help="如 000711 / 600519 / 000858")
    frequency = st.selectbox("K 线频率", list(FREQS), index=1,
                             help="日线走 v3 读取器；1/5/15/30/60min 走分钟读取器")
    brick_mode = st.radio("砖高", ["自动（ATR 中位数）", "手动"], index=0,
                          help="自动 = talib.ATR(14) 的中位数，与旧树同口径")
    manual_brick = st.number_input("手动砖高（自动模式忽略）", min_value=0.01,
                                   value=1.0, step=0.1, format="%.2f")
    max_bars = st.slider("最多用多少根 K 线", 120, 4000, 1200, step=60,
                         help="L 族每 1200 bar 分一窗，根数越多越慢")
    st.caption("砖块图：价格每走够一个砖高才多一块，横盘不产生砖 —— 剔噪、提高信噪比")


# --------------------------------------------------------------------------- #
# 数据加载
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False, ttl=600)
def load(symbol: str, frequency: str, brick_mode: str, manual_brick: float,
         max_bars: int):
    """8.3 取数 → 砖块序列（class renko） + 特征列（renko_trend_cross_func）。"""
    kind, lookback_days = FREQS[frequency]
    market = get_active_market()
    if kind == 'day':
        start = str(pd.Timestamp.now() - pd.Timedelta(days=lookback_days))[:10]
        d, name = market.get_kline_price_v3(symbol, start=start,
                                            realtime=True, verbose=False)
    else:
        start = str(pd.Timestamp.now() - pd.Timedelta(days=lookback_days))
        d, name = market.get_kline_price_min(symbol, start=start,
                                             frequency=frequency,
                                             realtime=True, verbose=False)

    # ⚠️ 日线无数据返回 **None**（分钟返回空对象）—— 两种都要按「无数据」处理
    if d is None or len(getattr(d, 'data', ())) == 0:
        return None, name, None, None

    df = d.data.tail(int(max_bars)).copy()
    if len(df) < 30:
        # `renko_trend_cross_func` 在 <30 根时**只返回空列**（不计算），
        # 那是给调用方判断"要不要重算"用的形状契约 —— 演示里直接告诉用户更好
        return df, name, None, None

    # ---- 真正的砖块序列（砖块图那个页签用） ----
    hlc = df[[AKA.HIGH, AKA.LOW, AKA.CLOSE]].bfill().ffill().values
    obj = renko()
    if brick_mode == '自动（ATR 中位数）':
        brick = float(obj.set_brick_size(auto=True, HLC_history=hlc))
    else:
        brick = float(obj.set_brick_size(auto=False, brick_size=manual_brick))
    obj.build_history(hlc=hlc)
    bricks = pd.DataFrame({'price': obj.get_renko_prices(),
                           'direction': obj.get_renko_directions()})

    # ---- 特征列（特征叠加那个页签用） ----
    try:
        feats = renko_trend_cross_func(df)
    except Exception as exc:                       # noqa: BLE001
        # ⚠️ 那个函数里有个裸 `except`（保真搬运，见 analysis/renko.py 的「已知缺陷」2），
        # 极端输入下会以 NameError 收场 —— 演示里如实报出来，别静默。
        # ⚠️ 返回**字符串**而不是异常对象：`st.cache_data` 要 pickle 返回值，
        # 掺一个异常实例进去是自找麻烦。
        return df, name, bricks, None, f"{type(exc).__name__}: {exc}"
    return df, name, bricks, feats, None


try:
    df, name, bricks, feats, err = load(symbol, frequency, brick_mode,
                                        manual_brick, max_bars)
except Exception as exc:                            # noqa: BLE001
    st.error(f"数据加载失败：{exc}")
    st.stop()

if df is None:
    st.warning(f"{symbol} 在 {frequency} 上没有取到 K 线 —— 换只标的或换个频率试试。")
    st.stop()
if err:
    st.error(f"砖块特征计算失败：{err}")
    st.stop()
if feats is None or len(df) < 30:
    st.warning(f"只有 {len(df)} 根 K 线 —— 砖块特征至少需要 30 根。")
    st.stop()

# --------------------------------------------------------------------------- #
# 指标卡
# --------------------------------------------------------------------------- #
brick = float(bricks['price'].iloc[0]) if len(bricks) else float('nan')
s_dir = int(feats[FLD.RENKO_TREND_S].iloc[-1]) if len(feats) else 0
l_dir = int(feats[FLD.RENKO_TREND_L].iloc[-1]) if len(feats) else 0
_sig = {1: '🔴 上', -1: '🟢 下', 0: '— 无'}
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("📊 K 线数", f"{len(df):,}")
c2.metric("🧱 砖块数", f"{len(bricks) - 1:,}" if len(bricks) else "0")
c3.metric("📏 砖高", "—")
c4.metric("🅂 S 族", _sig.get(s_dir, s_dir))
c5.metric("🅻 L 族", _sig.get(l_dir, l_dir))
st.caption(f"标的：{name}（{symbol}）· {frequency} · 砖高 {brick:.4g}"
           f"（{'自动 ATR 中位数' if brick_mode.startswith('自动') else '手动'}）"
           f" · 砖块图 {len(bricks) - 1} 块 / 特征 {len(feats)} 行")

# --------------------------------------------------------------------------- #
# 两个页签
# --------------------------------------------------------------------------- #
tab_brick, tab_feat = st.tabs(["🧱 砖块图（按砖序）", "📈 特征叠加（按时间轴）"])

# ---- 页签 1：真正的 Renko 砖块图 ----
with tab_brick:
    if len(bricks) < 2:
        st.info("这一段没有产生砖块 —— 价格波动不足一个砖高。")
    else:
        # 砖 k（k>=1）：方向 +1 时砖身是 [price-brick, price]，方向 -1 时是 [price, price+brick]
        # （实测核对过：renko_prices/directions 直接给出真实砖序，见模块 docstring）
        px = bricks['price'].values[1:]
        dr = bricks['direction'].values[1:]
        bottom = np.where(dr > 0, px - brick, px)
        top = np.where(dr > 0, px, px + brick)
        x = np.arange(1, len(px) + 1)
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=x,
            # 开收按**砖的方向**摆，颜色才跟着方向走（否则下跌砖也会画成红的）
            open=np.where(dr > 0, bottom, top),
            close=np.where(dr > 0, top, bottom),
            high=top, low=bottom,
            increasing_line_color='red', increasing_fillcolor='red',
            decreasing_line_color='green', decreasing_fillcolor='green',
            name='砖块'))
        fig.update_layout(height=620, xaxis_rangeslider_visible=False,
                          hovermode='x unified', showlegend=False,
                          margin=dict(l=10, r=10, t=30, b=10),
                          xaxis_title='砖序号（与时间无关）', yaxis_title='价格')
        st.plotly_chart(fig, width='stretch')
        st.caption("⚠️ 横轴是**砖序号**，不是时间 —— 一根 K 线可能产生多块砖，"
                   "也可能连续多根都不产生砖（横盘）。这正是 Renko 剔噪的方式。")

# ---- 页签 2：时间轴上的特征叠加 ----
with tab_feat:
    f = feats
    # `renko_chart` 输出的 lb/ub **是有序的**（lb <= ub），直接用
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index.get_level_values(0),
        open=df[AKA.OPEN], high=df[AKA.HIGH], low=df[AKA.LOW], close=df[AKA.CLOSE],
        increasing_line_color='red', increasing_fillcolor='red',
        decreasing_line_color='green', decreasing_fillcolor='green',
        name='K线'))
    for lb, ub, col, nm in (
            (FLD.RENKO_TREND_S_LB, FLD.RENKO_TREND_S_UB, 'royalblue', 'S 族砖界'),
            (FLD.RENKO_TREND_L_LB, FLD.RENKO_TREND_L_UB, 'darkorange', 'L 族砖界')):
        for series, dash in ((lb, 'dot'), (ub, 'dot')):
            fig.add_trace(go.Scatter(
                x=f.index.get_level_values(0), y=f[series].astype(float),
                mode='lines', name=series, line=dict(color=col, width=1, dash=dash),
                line_shape='hv'))
    # 合成信号：S、L 同向才给 ±1
    sig = f[FLD.RENKO_TREND].astype(float)
    for val, col, nm in ((1, 'red', 'RENKO_BAR = +1'), (-1, 'green', 'RENKO_BAR = -1')):
        m = sig == val
        if m.any():
            fig.add_trace(go.Scatter(
                x=f.index.get_level_values(0)[m], y=df[AKA.CLOSE][m],
                mode='markers', name=nm,
                marker=dict(symbol='triangle-up' if val > 0 else 'triangle-down',
                            size=8, color=col)))
    fig.update_layout(height=620, xaxis_rangeslider_visible=False,
                      hovermode='x unified',
                      legend=dict(orientation='h', yanchor='bottom', y=1.02),
                      margin=dict(l=10, r=10, t=40, b=10))
    fig.update_xaxes(nticks=15)
    st.plotly_chart(fig, width='stretch')

    # ---------------------------------------------------------------------- #
    # 特征明细
    # ---------------------------------------------------------------------- #
    st.subheader("特征明细（末 200 行）")
    cols = [c for c in SHOW_COLS if c in f.columns]
    st.dataframe(f[cols].tail(200), width='stretch')
    st.caption("⚠️ `RENKO_PRICE_S/L` 是**砖位价**（不是收盘价），`RENKO_OPTIMAL` 只在"
               "每个 1200 bar 窗口的首行有值、其余为 0；这三列旧树也**写后无人读**"
               "（见 `analysis/renko.py` 的「写到哪些列」）。")
