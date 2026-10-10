# coding: utf-8
"""缠论盘整中枢（箱体）演示 —— **新树 / MongoDB 8.3 版**

运行方式：

    streamlit run examples/app_pivot.py

数据链路：

    GolemQ.fetch.kline.get_kline_price_min          # 门面，可指定 frequency
        → markets/StockCN/kline83.py                # MongoDB 8.3 时序集合，已前复权
    czsc.CZSC                                       # 分笔（bi_list）
    GolemQ.analysis.pivot                           # 中枢识别 / 走势分类 / 绘图

与旧树 `GolemQ_old/app_pivot.py` 的差别：
* 数据从 QUANTAXIS 4.4 的 `stock_min`（`type='60min'`）换成 8.3 的 `stock_*min`；
* **多了频率选择**（8.3 有 1/5/15/30/60min 五档，旧树门面拿不到这个参数）；
* 中枢那层来自 `GolemQ.analysis.pivot`（自旧树 `GolemQ_old/czsc/pivot.py` 搬运），
  其因果口径（无未来函数）与旧树一致 —— 箱体自**中枢成立 bar** 起才画。

⚠️ 中枢箱体**不铺满整段**：`ZG/ZD` 从中枢成立那一刻才出现，`GG/DD` 是逐根
刷新极值的台阶线。这是刻意的 —— 一上来就画到整段极值 = 未来函数。
"""
import os
import sys

# --------------------------------------------------------------------------- #
# 让它能在「没装包、直接从源码跑」的情况下找到 GolemQ
# --------------------------------------------------------------------------- #
# 旧树那段 hack 是删掉脚本目录以避开 `GolemQ/czsc` 遮蔽 pip 版 czsc；新树没有
# `GolemQ/czsc` 包，遮蔽问题不存在，故只保留「把仓库根放进 sys.path」这一半。
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from datetime import datetime, timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from czsc import CZSC
from czsc.objects import Freq, RawBar

from GolemQ.analysis.pivot import (
    bi_confirm_map, bi_list_to_points, calc_pivots, classify_pivots,
    pivots_to_df, plot_pivots_plotly,
)
from GolemQ.fetch.kline import get_kline_price_min

# 频率 → (czsc 枚举, 日线根数, 取「加长历史」时回溯的自然日数)
# 日线根数按 A 股一个交易日算：1min=240 / 5min=48 / 15min=16 / 30min=8 / 60min=4
FREQS = {
    '60min': (Freq.F60, 4, 1500),
    '30min': (Freq.F30, 8, 700),
    '15min': (Freq.F15, 16, 360),
    '5min': (Freq.F5, 48, 140),
    '1min': (Freq.F1, 240, 40),
}

MAX_BARS = 4096

st.set_page_config(page_title="盘整中枢（箱体）演示", layout="wide")
st.title("📦 缠论盘整中枢（箱体）演示")

# --------------------------------------------------------------------------- #
# 侧边栏
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("⚙️ 参数")
    symbol = st.text_input("股票代码", value="000711", help="如 000711 / 600519 / 000858")
    frequency = st.selectbox("K 线频率", list(FREQS), index=0,
                             help="8.3 里 1/5/15/30/60min 五档都有数据")
    show_pivots = st.checkbox("显示中枢箱体", value=True)
    show_ggdd = st.checkbox("显示 GG/DD 延伸边界", value=True)
    long_history = st.checkbox("加长历史（最多 4096 根）", value=False,
                               help="默认用读取器自带窗口；勾选后按频率回溯并截到 4096 根")
    st.caption("缠论定义：中枢 = 至少三个连续笔的重叠区间，即盘整箱体")


# --------------------------------------------------------------------------- #
# 数据加载（轻量，秒开）
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False, ttl=600)
def load_pivots(symbol: str, frequency: str = '60min', long_history: bool = False):
    """8.3 分钟线 → czsc 分笔 → 中枢识别。不跑任何重特征流水线。"""
    freq_enum, _, lookback_days = FREQS[frequency]
    if long_history:
        start = str(datetime.now() - timedelta(days=lookback_days))
        d, name = get_kline_price_min(symbol, start=start, frequency=frequency,
                                      realtime=True, verbose=False)
    else:
        d, name = get_kline_price_min(symbol, frequency=frequency,
                                      realtime=True, verbose=False)
    # ⚠️ **总是**截到最后 MAX_BARS 根：读取器的默认窗口是按**小时**算的
    # （60min 约 2000 根），切到 5min/1min 会变成两万根以上，画不动。
    ohlc = d.data.tail(MAX_BARS)

    # ⚠️ **必须先判空**：`CZSC([])` 会抛 `IndexError`（czsc 0.7.10 的
    # `analyze.py:228` 直接取 `bars[0].symbol`），把「这只票没有数据」报成异常。
    # 空结果 ≠ 错误 —— 返回空结构，让调用方按「无数据」处理。
    if ohlc.empty:
        return pd.DataFrame(), name, [], classify_pivots([]), 0, []

    bars = []
    for i, (idx, row) in enumerate(ohlc.iterrows()):
        # 8.3 读路径是 (ts, code) 两级索引，取 level 0
        dt = idx[0] if isinstance(idx, tuple) else idx
        bars.append(RawBar(symbol=symbol, dt=dt, id=i, freq=freq_enum,
                           open=float(row['open']), close=float(row['close']),
                           high=float(row['high']), low=float(row['low']),
                           vol=float(row['volume']),
                           amount=float(row.get('amount', 0) or 0)))

    df = pd.DataFrame([{'dt': b.dt, 'open': b.open, 'high': b.high,
                        'low': b.low, 'close': b.close} for b in bars])
    czsc = CZSC(bars, max_bi_count=512, verbose=False)
    conf_map = bi_confirm_map(bars, max_bi_count=512)
    pivots = calc_pivots(bi_list=czsc.bi_list, conf_map=conf_map, strict=True)
    cls = classify_pivots(pivots)
    bi_points = bi_list_to_points(czsc.bi_list)
    return df, name, pivots, cls, len(czsc.bi_list), bi_points


try:
    df, name, pivots, cls, bi_count, bi_points = load_pivots(symbol, frequency, long_history)
except Exception as e:
    st.error(f"数据加载失败：{e}")
    st.stop()

if df.empty:
    st.warning(f"{symbol} 在 {frequency} 上没有取到 K 线 —— 换只标的或换个频率试试。")
    st.stop()

# --------------------------------------------------------------------------- #
# 指标卡
# --------------------------------------------------------------------------- #
c1, c2, c3, c4 = st.columns(4)
c1.metric("📊 K 线数", f"{len(df):,}")
c2.metric("✏️ 笔数", bi_count)
c3.metric("📦 中枢数", len(pivots))
c4.metric("🧭 走势类型", cls['kind'])
st.caption(f"标的：{name}（{symbol}）· {frequency} · "
           f"上涨中枢 {cls['n_up']} 个 / 下跌中枢 {cls['n_down']} 个")

# --------------------------------------------------------------------------- #
# 绘图（Plotly，交互式）
# --------------------------------------------------------------------------- #
fig = go.Figure()

# K 线（红涨绿跌）
fig.add_trace(go.Candlestick(
    x=df['dt'], open=df['open'], high=df['high'], low=df['low'], close=df['close'],
    increasing_line_color='red', increasing_fillcolor='red',
    decreasing_line_color='green', decreasing_fillcolor='green',
    name='K线'))

# 笔（BI）
fig.add_trace(go.Scatter(
    x=[p['dt'] for p in bi_points],
    y=[p['bi'] for p in bi_points],
    mode='lines+markers', name='笔',
    line=dict(color='royalblue', width=1.5),
    marker=dict(size=4, color='royalblue'),
))

# 中枢箱体（因果口径：自中枢成立 bar 起才画）
if show_pivots:
    plot_pivots_plotly(
        fig, pivots, x_last=df['dt'].iloc[-1], show_ggdd=show_ggdd)

fig.update_layout(
    height=620,
    xaxis_rangeslider_visible=False,
    hovermode='x unified',
    margin=dict(l=10, r=10, t=40, b=10),
    legend=dict(orientation='h', yanchor='bottom', y=1.02),
)
fig.update_xaxes(nticks=15)

st.plotly_chart(fig, width='stretch')

# --------------------------------------------------------------------------- #
# 中枢明细表
# --------------------------------------------------------------------------- #
st.subheader("中枢明细")
st.dataframe(pivots_to_df(pivots, kind=cls['kinds']), width='stretch', hide_index=True)
