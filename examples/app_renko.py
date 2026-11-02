# coding: utf-8
"""砖块图（**RENKO**）演示 —— **attach `*_reality_feat_60min`（无后视镜）**

运行方式：

    streamlit run examples/app_renko.py

数据链路：

    GolemQ.get_active_market().get_kline_price_min(...)   # 60min K 线
        → markets/StockCN/kline83.py                      # MongoDB 8.3，已前复权
    GolemQ.analysis.reality_attach.attach_reality_feats   # 只读
        → `*_reality_feat_60min` 的 Renko 14 列（review 逐日 as-of 产出）

⚠️ **本页不再现场跑 `renko_trend_cross_func`。** 那套算法是「整段历史一次跑完」：
S 族的砖高取整段 ATR 中位数、L 族按 1200 bar 分窗各搜各的砖高，窗口里都包含
目标 bar **之后**的数据 —— 这正是 `docs/review_model_60min_feature_spec.md`
所说的后视镜。现在画的是 `--review` 落库的 as-of 结果：

* 每个交易日 **15:00** 用 **≤T 的最近 2160 根 60min** 重算一次 Renko；
* 盘中 3 根（10:30 / 11:30 / 14:00）沿用**上一交易日 15:00** 的快照；
* 所以同一交易日的 4 根共用一个值 —— 图上会看到台阶。

两个页签：

1. **砖块图** —— 把逐 bar 的 as-of 砖界压成砖序（快照变化点）。横轴是
   **快照砖序号**，不是真砖序列：一次跳多块砖只记终点（详见页内说明）。
2. **特征叠加** —— 时间轴上的 S/L 两族砖界台阶线 + `RENKO_BAR` 信号。

⚠️ `renkoSprice` / `renkoLprice` 是**带符号的砖位**（下跌砖为负），不是收盘价，
不能直接当价格画；本页用 `renkosBarLb/Ub`（还原后是有序上下沿）画砖。
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
from GolemQ.analysis.reality_attach import attach_reality_feats
from GolemQ.core.constants import AKA, FIELD as FLD

#: review 的 Renko 只落 60min 表（D78）；本页因此只做 60min。
RENKO_FREQ = '60min'
LOOKBACK_DAYS = 1500

#: 演示要看的列（S 族 + L 族 + 合成）
SHOW_COLS = (FLD.RENKO_PRICE_S, FLD.RENKO_TREND_S, FLD.RENKO_TREND_S_LB,
             FLD.RENKO_TREND_S_UB, FLD.RENKO_PRICE_L, FLD.RENKO_TREND_L,
             FLD.RENKO_TREND_L_LB, FLD.RENKO_TREND_L_UB, FLD.RENKO_OPTIMAL,
             FLD.RENKO_TREND, FLD.RENKO_TREND_S_TIMING_LAG,
             FLD.RENKO_BOOST_S_TIMING_LAG, FLD.RENKO_TREND_L_TIMING_LAG,
             FLD.RENKO_BOOST_L_TIMING_LAG)

st.set_page_config(page_title="砖块图（RENKO）演示", layout="wide")
st.title("🧱 砖块图（RENKO）演示 · as-of 落库版")

# --------------------------------------------------------------------------- #
# 侧边栏
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("⚙️ 参数")
    symbol = st.text_input("股票代码", value="000711",
                           help="如 000711 / 600519 / 000858").strip()
    max_bars = st.slider("最多显示多少根 60min", 120, 4000, 1200, step=60,
                         help="只影响显示窗口；值本身来自库里已算好的 as-of 行")
    st.caption("数据来自 `*_reality_feat_60min`：每个交易日 15:00 用 ≤T 的 "
               "2160 根重算，盘中沿用上一交易日快照 —— **无后视镜**。")


# --------------------------------------------------------------------------- #
# 数据加载
# --------------------------------------------------------------------------- #
def _snapshot_bricks(feats: pd.DataFrame) -> pd.DataFrame:
    """逐 bar 的 as-of 砖界 → 砖序（只在快照变化时记一块）。

    ⚠️ 用 `np.isclose` 而不是 `==`：落库时是「绝对价 ÷ 本 bar close」，还原时
    乘回**同一行**的 close；浮点往返在最后几位可能不相等（实测同一快照内
    `==` 偶尔为 False），直接比较会造出根本不存在的砖。
    """
    empty = pd.DataFrame(columns=['ts', 'lb', 'ub', 'direction'])
    need = (FLD.RENKO_TREND_S_LB, FLD.RENKO_TREND_S_UB, FLD.RENKO_TREND_S)
    if any(c not in feats.columns for c in need):
        return empty
    lb = pd.to_numeric(feats[FLD.RENKO_TREND_S_LB], errors='coerce')
    ub = pd.to_numeric(feats[FLD.RENKO_TREND_S_UB], errors='coerce')
    dr = pd.to_numeric(feats[FLD.RENKO_TREND_S], errors='coerce')
    m = (lb.notna() & ub.notna() & dr.notna()).to_numpy()
    if not m.any():
        return empty
    ts = pd.DatetimeIndex(pd.to_datetime(
        feats.index.get_level_values(0)))[m]
    sub = pd.DataFrame({'ts': ts.to_numpy(),
                        'lb': lb.to_numpy(dtype=float)[m],
                        'ub': ub.to_numpy(dtype=float)[m],
                        'direction': dr.to_numpy(dtype=float)[m]})
    same = (np.isclose(sub['lb'].to_numpy(), sub['lb'].shift().to_numpy(),
                       rtol=1e-9, atol=1e-6)
            & np.isclose(sub['ub'].to_numpy(), sub['ub'].shift().to_numpy(),
                         rtol=1e-9, atol=1e-6)
            & (sub['direction'].to_numpy()
               == sub['direction'].shift().to_numpy()))
    keep = np.r_[True, ~same[1:]]
    return sub.loc[keep].reset_index(drop=True)


@st.cache_data(show_spinner=False, ttl=600)
def load(symbol: str, max_bars: int):
    """60min K 线 → attach `*_reality_feat_60min` → Renko 砖序 + 特征帧。"""
    market = get_active_market()
    start = str(pd.Timestamp.now() - pd.Timedelta(days=LOOKBACK_DAYS))
    d, name = market.get_kline_price_min(symbol, start=start,
                                         frequency=RENKO_FREQ,
                                         realtime=True, verbose=False)
    if d is None or len(getattr(d, 'data', ())) == 0:
        return None, name, None, None, None

    df = d.data.tail(int(max_bars)).copy()
    if len(df) < 30:
        return df, name, None, None, None

    try:
        feats, loaded = attach_reality_feats(df.copy(), symbol, verbose=False)
    except Exception as exc:                       # noqa: BLE001
        # ⚠️ 返回**字符串**：`st.cache_data` 要 pickle 返回值，别掺异常实例。
        return df, name, None, None, f"{type(exc).__name__}: {exc}"

    if not loaded.get('60min_renko'):
        return df, name, None, feats, None
    return df, name, _snapshot_bricks(feats), feats, None


try:
    df, name, bricks, feats, err = load(symbol, max_bars)
except Exception as exc:                            # noqa: BLE001
    st.error(f"数据加载失败：{exc}")
    st.stop()

if df is None:
    st.warning(f"{symbol} 在 {RENKO_FREQ} 上没有取到 K 线 —— 换只标的试试。")
    st.stop()
if err:
    st.error(f"attach `*_reality_feat_60min` 失败：{err}")
    st.stop()
if feats is None or bricks is None:
    st.warning(f"{symbol} 的 `*_reality_feat_60min` 里没有 Renko 14 列 —— "
               "该标的还没跑过 review（或回填）。先跑 `--review` / "
               "`tools/backfill_renko.py`，别退回现场重算（那是有后视镜的）。")
    st.stop()
if len(bricks) == 0:
    st.info("这段没有产生 as-of 砖块 —— 价格波动不足一个砖高。")
    st.stop()

# --------------------------------------------------------------------------- #
# 指标卡
# --------------------------------------------------------------------------- #
def _last_int(frame: pd.DataFrame, col: str):
    if col not in frame.columns:
        return None
    s = pd.to_numeric(frame[col], errors='coerce').dropna()
    return None if s.empty else int(s.iloc[-1])


_s_dir = _last_int(feats, FLD.RENKO_TREND_S)
_l_dir = _last_int(feats, FLD.RENKO_TREND_L)
_sig = {1: '🔴 上', -1: '🟢 下', 0: '— 无'}
_brick_h = float((bricks['ub'] - bricks['lb']).median())
_last_ts = pd.to_datetime(
    feats.index.get_level_values(0))[feats[FLD.RENKO_TREND_S].notna().to_numpy()]
_last_ts = _last_ts.max() if len(_last_ts) else None

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("📊 K 线数", f"{len(df):,}")
c2.metric("🧱 as-of 砖块数", f"{max(len(bricks) - 1, 0):,}")
c3.metric("📏 中位砖高", f"{_brick_h:.4g}")
c4.metric("🅂 S 族", _sig.get(_s_dir, '—') if _s_dir is not None else '—')
c5.metric("🅻 L 族", _sig.get(_l_dir, '—') if _l_dir is not None else '—')
# 展示名由读取器解析（`kline83._display_name`：stock_info → stock_list）。
# 解析不到时它**回落成代码**（真指数、未知代码、`name` 为空的标的）——
# 那种情况下别再印一次代码：`301308（301308）` 没有信息量。
_label = symbol if name == symbol else f'{name}（{symbol}）'
st.caption(f"标的：{_label} · {RENKO_FREQ} · as-of 砖 {max(len(bricks) - 1, 0)} 块"
           f" · 末个快照 {_last_ts:%Y-%m-%d %H:%M}" if _last_ts is not None
           else f"标的：{_label} · {RENKO_FREQ}")

# --------------------------------------------------------------------------- #
# 两个页签
# --------------------------------------------------------------------------- #
tab_brick, tab_feat = st.tabs(["🧱 砖块图（as-of 砖序）",
                               "📈 特征叠加（按时间轴）"])

# ---- 页签 1：as-of 砖序 ----
with tab_brick:
    if len(bricks) <= 1:
        st.info("这段没有产生新的 as-of 砖块 —— 价格波动不足一个砖高。")
    else:
        x = np.arange(1, len(bricks) + 1)
        bottom = bricks['lb'].to_numpy(dtype=float)
        top = bricks['ub'].to_numpy(dtype=float)
        dr = bricks['direction'].to_numpy(dtype=float)
        fig = go.Figure()
        # ⚠️ **用 `Bar` + `base`，不要用 `Candlestick`**：砖块图要的是
        # **轴对齐矩形**（上下沿平直、左右沿竖直），`Bar` 的 `y` 是高度、
        # `base` 是下沿，出来的就是标准砖块。
        fig.add_trace(go.Bar(
            x=x,
            y=top - bottom,              # 高度 = 该砖的上下沿之差
            base=bottom,                 # 下沿
            width=1.0,                   # 相邻砖左右相接
            marker=dict(color=np.where(dr > 0, 'red', 'green'),
                        line=dict(color='white', width=1)),
            name='as-of 砖',
            customdata=np.stack([bottom, top, dr,
                                 bricks['ts'].dt.strftime('%Y-%m-%d %H:%M')],
                                axis=-1),
            hovertemplate=('砖 %{x}<br>下沿 %{customdata[0]:.2f}'
                           '<br>上沿 %{customdata[1]:.2f}'
                           '<br>方向 %{customdata[2]:.0f}'
                           '<br>快照 %{customdata[3]}<extra></extra>')))
        fig.update_layout(height=620, showlegend=False, bargap=0,
                          margin=dict(l=10, r=10, t=30, b=10),
                          xaxis_title='as-of 快照砖序号（与时间无关）',
                          yaxis_title='价格')
        st.plotly_chart(fig, width='stretch')
        st.caption("⚠️ 横轴是**快照砖序号**：每个交易日 15:00 才更新一次快照，"
                   "盘中 4 根沿用上一交易日；一次跳多块砖只记终点，"
                   "所以这里的块数会比「真 Renko 砖序列」少。数据是 review 的 "
                   "as-of 落库值，**不是**整段历史重算。")

# ---- 页签 2：时间轴上的特征叠加 ----
with tab_feat:
    f = feats
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
            if series not in f.columns:
                continue
            fig.add_trace(go.Scatter(
                x=f.index.get_level_values(0),
                y=pd.to_numeric(f[series], errors='coerce').to_numpy(),
                mode='lines', name=series, line=dict(color=col, width=1, dash=dash),
                line_shape='hv'))
    # 合成信号：S、L 同向才给 ±1
    sig = pd.to_numeric(f[FLD.RENKO_TREND], errors='coerce').to_numpy()
    close = pd.to_numeric(df[AKA.CLOSE], errors='coerce').to_numpy()
    x_all = f.index.get_level_values(0)
    for val, col, nm in ((1, 'red', 'RENKO_BAR = +1'), (-1, 'green', 'RENKO_BAR = -1')):
        m = sig == val
        if m.any():
            fig.add_trace(go.Scatter(
                x=x_all[m], y=close[m],
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
    st.caption("⚠️ `RENKO_PRICE_S/L` 是**带符号的砖位价**（下跌砖为负），"
               "不是收盘价；`RENKO_OPTIMAL` 只在每个窗口首行有值、其余为 0。"
               "全部 14 列都来自 `*_reality_feat_60min`（as-of），本页不重算。")
