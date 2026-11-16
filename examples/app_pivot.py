# coding: utf-8
"""缠论中枢 × Mainstream —— **叠加图**演示（新树 / MongoDB 8.3 版）

运行方式：

    streamlit run examples/app_pivot.py

与其它 `app_*` 一样只保留「标题 + 侧边栏开关 + 本页的 layer」；
取数 / attach / Mainstream / K 线骨架全在 `examples/_app_common.py`
（用户 2026-11-15：统一架构）。

缠论定义：中枢 = 至少三个连续笔的重叠区间（盘整箱体）；
60min 用**日线 MAJOR** 标记，其它频率用**本频**标记。
"""
import os
import sys

# --------------------------------------------------------------------------- #
# 让它能在「没装包、直接从源码跑」的情况下找到 GolemQ
# --------------------------------------------------------------------------- #
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
_EX = os.path.dirname(os.path.abspath(__file__))
if _EX not in sys.path:
    sys.path.insert(0, _EX)

import streamlit as st

from GolemQ.analysis.pivot import (bi_confirm_map, bi_list_to_points,
                                   calc_pivots, classify_pivots,
                                   pivots_to_df, plot_pivots_plotly)
from GolemQ.models.alias import LTT

import _app_common as app
from _gear_bands import add_gear_bands, gear_o_before

st.set_page_config(page_title="缠论中枢", layout="wide")
st.title("📦 缠论中枢 × Mainstream（叠加图）")

# --------------------------------------------------------------------------- #
# 侧边栏
# --------------------------------------------------------------------------- #
with st.sidebar:
    symbol, frequency, long_history = app.sidebar_base(default_freq_index=0)

    st.divider()
    show_pivots = st.checkbox("显示中枢箱体", value=True)
    show_ggdd = st.checkbox("显示 GG/DD 延伸边界", value=True)
    show_bi = st.checkbox("缠论：笔", value=False)
    show_gear_o = st.checkbox(
        "QuadGear0：橙色触发区段", value=True,
        help="QuadGear0Bf == 0（O 档触发）；信号固定为 60min 口径")

    st.divider()
    show_leadin = st.checkbox("Mainstream：吸筹 leadin 触发", value=True)
    show_leadout = st.checkbox("Mainstream：派发 leadout 触发", value=True)
    fold_mainstream = True

    st.caption("缠论定义：中枢 = 至少三个连续笔的重叠区间，即盘整箱体 · "
               "60min 用**日线 MAJOR** 标记，其它频率用**本频**标记")

# --------------------------------------------------------------------------- #
# 数据
# --------------------------------------------------------------------------- #
try:
    data = app.load_base(symbol, frequency, long_history, fold_mainstream)
except Exception as exc:                            # noqa: BLE001
    st.error(f"数据加载失败：{exc}")
    st.stop()

if data is None:
    st.warning(f"{symbol} 在 {frequency} 上没有取到 K 线 —— 换只标的或换个频率试试。")
    st.stop()
df = data.get('df')
if df is None or df.empty:
    st.warning(f"{symbol} 的 K 线不足 30 根 —— 算不出来。")
    st.stop()

feats, ms, ms_major = data['feats'], data['ms'], data['ms_major']
if data.get('feats_err'):
    st.warning(f"attach `*_reality_feat_60min` 失败：`{data['feats_err']}`")

conf_map = bi_confirm_map(data['bars'], max_bi_count=512)
pivots = calc_pivots(bi_list=data['bi_list'], conf_map=conf_map, strict=True)
cls = classify_pivots(pivots)
bi_points = bi_list_to_points(data['bi_list'])
gear_bf = gear_o_before(symbol) if show_gear_o else None

_use_major = (frequency == '60min' and ms_major is not None
              and len(ms_major) > 0)

# --------------------------------------------------------------------------- #
# 指标卡
# --------------------------------------------------------------------------- #
c1, c2, c3, c4 = st.columns(4)
c1.metric("📊 K 线数", f"{len(df):,}")
c2.metric("✏️ 笔数", len(bi_points))
c3.metric("📦 中枢数", len(pivots))
c4.metric("🧭 走势类型", cls['kind'])
_label = symbol if data['name'] == symbol else f"{data['name']}（{symbol}）"

_li = int(app.trigger_mask(df, ms, ms_major, frequency,
                           LTT.MAINSTREAM_LEADIN_BEFORE,
                           LTT.MAINSTREAM_LEADIN_BEFORE_MAJOR).sum())
_lo = int(app.trigger_mask(df, ms, ms_major, frequency,
                           LTT.MAINSTREAM_LEADOUT_BEFORE,
                           LTT.MAINSTREAM_LEADOUT_BEFORE_MAJOR).sum())
_mode = '日线 MAJOR' if _use_major else '本频'
st.caption(f"标的：{_label} · {frequency} · 末个中枢：**{cls['kinds'][-1] if cls.get('kinds') else '—'}**")
st.caption(f"Mainstream（{_mode}）：leadin {_li} 次 / leadout {_lo} 次"
           f"（fold={fold_mainstream}；标记在 Prophet LB 下方 / UB 上方）")

# --------------------------------------------------------------------------- #
# 图
# --------------------------------------------------------------------------- #
fig = app.make_figure(df)

if show_gear_o and gear_bf is not None:
    _runs = add_gear_bands(fig, gear_bf, frequency=frequency)
    if _runs:
        st.caption(f"橙色背景：`QuadGear0Bf < 1e-12`（O 档触发，"
                   f"{len(_runs)} 段；60min 口径）")

if show_bi:
    import plotly.graph_objects as go
    fig.add_trace(go.Scatter(
        x=[p['dt'] for p in bi_points], y=[p['bi'] for p in bi_points],
        mode='lines+markers', name='笔',
        line=dict(color='royalblue', width=1.5),
        marker=dict(size=4, color='royalblue')))

if show_pivots:
    plot_pivots_plotly(
        fig, pivots, x_last=df['dt'].iloc[-1], show_ggdd=show_ggdd)

app.add_mainstream_triggers(fig, data, show_leadin, show_leadout)
app.finish_figure(fig)
st.plotly_chart(fig, width='stretch')

# --------------------------------------------------------------------------- #
# 明细
# --------------------------------------------------------------------------- #
st.subheader("中枢明细")
st.dataframe(pivots_to_df(pivots, kind=cls['kinds']), width='stretch', hide_index=True)

with st.expander("Mainstream leadin / leadout 明细（末 200 行）"):
    _detail = ms_major if _use_major else ms
    _ms_cols = [c for c in (
        LTT.MAINSTREAM_LEADIN_BEFORE_MAJOR, LTT.MAINSTREAM_LEADOUT_BEFORE_MAJOR,
        LTT.MAINSTREAM_LEADIN_VAR5, LTT.MAINSTREAM_LEADIN_ENTRY,
        LTT.MAINSTREAM_LEADIN_SHAKEOUT, LTT.MAINSTREAM_LEADIN_BEFORE,
        LTT.MAINSTREAM_LEADINx3_BEFORE, LTT.MAINSTREAM_LEADINx5_BEFORE,
        LTT.MAINSTREAM_LEADOUT_VAR5, LTT.MAINSTREAM_LEADOUT_ENTRY,
        LTT.MAINSTREAM_LEADOUT_SHAKEOUT, LTT.MAINSTREAM_LEADOUT_BEFORE,
        LTT.MAINSTREAM_LEADOUTx3_BEFORE, LTT.MAINSTREAM_LEADOUTx5_BEFORE,
    ) if _detail is not None and c in _detail.columns]
    if _ms_cols:
        st.dataframe(_detail[_ms_cols].tail(200), width='stretch')
    else:
        st.caption("没有 Mainstream 列（数据不足）。")
