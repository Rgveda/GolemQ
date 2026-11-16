# coding: utf-8
"""缠论中枢 × RENKO 砖块界 —— **叠加图**演示（新树 / MongoDB 8.3 版）

运行方式：

    streamlit run examples/app_pivot_renko.py

与其它 `app_*` 一样只保留「标题 + 侧边栏开关 + 本页的 layer」；
取数 / attach / Mainstream / K 线骨架全在 `examples/_app_common.py`
（用户 2026-11-15：统一架构）。

两个东西各回答一个问题
======================
| | 来源 | 回答什么 |
|:--|:--|:--|
| **缠论中枢** | `analysis/pivot.py`（czsc 分笔 → 三笔重叠箱体） | **结构**：箱体上下沿 + 走势类型 |
| **RENKO 砖界** | `*_reality_feat_60min` 的 Renko 14 列（as-of 落库） | **趋势**：S/L 两族上下界；`RENKO_BAR` 只在两族同向时非零 |

⚠️ 两者**没有共同的数学基础**，图上分歧是常态。
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

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from GolemQ.analysis.pivot import (bi_confirm_map, bi_list_to_points,
                                   calc_pivots, classify_pivots,
                                   pivots_to_df, plot_pivots_plotly)
from GolemQ.core.constants import FIELD as FLD
from GolemQ.models.alias import LTT

import _app_common as app
from _gear_bands import add_gear_bands, gear_o_before

st.set_page_config(page_title="中枢 × RENKO", layout="wide")
st.title("📦🧱 缠论中枢 × RENKO 砖块界（叠加图）")

# --------------------------------------------------------------------------- #
# 侧边栏
# --------------------------------------------------------------------------- #
with st.sidebar:
    symbol, frequency, long_history = app.sidebar_base(default_freq_index=1)

    st.divider()
    st.caption("显示开关 —— 两层各自可关，便于对照")
    show_pivots = st.checkbox("缠论：中枢箱体", value=True)
    show_ggdd = st.checkbox("缠论：GG/DD 延伸边界", value=False)
    show_bi = st.checkbox("缠论：笔", value=False)
    _renko_freq_ok = frequency == app.RENKO_FREQ
    show_renko_s = st.checkbox("RENKO：S 族砖界", value=True,
                               disabled=not _renko_freq_ok)
    show_renko_l = st.checkbox("RENKO：L 族砖界", value=True,
                               disabled=not _renko_freq_ok)
    show_signal = st.checkbox("RENKO：RENKO_BAR 信号", value=True,
                              disabled=not _renko_freq_ok)
    show_gear_o = st.checkbox(
        "QuadGear0：橙色触发区段", value=True,
        help="QuadGear0Bf == 0（O 档触发）；信号固定为 60min 口径")

    st.divider()
    show_leadin = st.checkbox("Mainstream：吸筹 leadin 触发", value=True)
    show_leadout = st.checkbox("Mainstream：派发 leadout 触发", value=True)
    fold_mainstream = True

    st.divider()
    st.caption("RENKO 列来自 `*_reality_feat_60min`（`--review` 的 as-of 产出），"
               "本页**不重算**。落库只有 **60min**：两个交易时段之间的盘内 bar "
               "共用当日 15:00 的快照；其它频率只显示中枢/笔。")

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
    st.warning(f"{symbol} 的 K 线不足 30 根 —— 两层都算不出来。")
    st.stop()

feats, ms, ms_major = data['feats'], data['ms'], data['ms_major']
bars = data['bars']
if data.get('feats_err'):
    st.warning(f"attach `*_reality_feat_60min` 失败：`{data['feats_err']}`")

conf_map = bi_confirm_map(bars, max_bi_count=512)
pivots = calc_pivots(bi_list=data['bi_list'], conf_map=conf_map, strict=True)
cls = classify_pivots(pivots)
bi_points = bi_list_to_points(data['bi_list'])

renko_ok = (FLD.RENKO_TREND_S in feats.columns
            and pd.to_numeric(feats[FLD.RENKO_TREND_S],
                              errors='coerce').notna().any())
gear_bf = gear_o_before(symbol) if show_gear_o else None

_use_major = (frequency == '60min' and ms_major is not None
              and len(ms_major) > 0)


def _last_int(col: str):
    if col not in feats.columns:
        return None
    s = pd.to_numeric(feats[col], errors='coerce').dropna()
    return None if s.empty else int(s.iloc[-1])


# --------------------------------------------------------------------------- #
# 指标卡
# --------------------------------------------------------------------------- #
s_dir = _last_int(FLD.RENKO_TREND_S)
l_dir = _last_int(FLD.RENKO_TREND_L)
bar_sig = _last_int(FLD.RENKO_TREND)
_dir_word = {1: '🔴 上', -1: '🟢 下', 0: '— 平'}
_dir_txt = lambda v: '—' if v is None else _dir_word.get(v, str(v))

_last_kind = cls['kinds'][-1] if cls.get('kinds') else '—'
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("📊 K 线数", f"{len(df):,}")
c2.metric("✏️ 笔数", len(bi_points))
c3.metric("📦 中枢数", len(pivots))
c4.metric("🧭 走势类型", cls['kind'])
c5.metric("🧱 RENKO S / L",
          f"{_dir_txt(s_dir)} / {_dir_txt(l_dir)}" if renko_ok else '—')
c6.metric("🎯 RENKO_BAR", _dir_txt(bar_sig) if renko_ok else '—')

_label = symbol if data['name'] == symbol else f"{data['name']}（{symbol}）"
if renko_ok:
    _fmt = lambda v: f'{v:+d}' if v is not None else '—'
    st.caption(f"标的：{_label} · {frequency} · 末个中枢：**{_last_kind}** · "
               f"末根 S={_fmt(s_dir)} L={_fmt(l_dir)} ⇒ RENKO_BAR={_fmt(bar_sig)}"
               "（RENKO_BAR 只在 S、L **同向**时才非零；数据为 review 的 "
               "as-of 落库值）")
else:
    st.caption(f"标的：{_label} · {frequency} · 末个中枢：**{_last_kind}** · "
               "RENKO 列不可用（见下方说明）")
    if frequency != app.RENKO_FREQ:
        st.info(f"RENKO 只落库 {app.RENKO_FREQ}；当前频率 `{frequency}` 只有中枢/笔。")
    else:
        st.info(f"`{symbol}` 的 `*_reality_feat_60min` 里没有 Renko 行 —— "
                "先跑 `--review` / `tools/backfill_renko.py`。")

_li = int(app.trigger_mask(df, ms, ms_major, frequency,
                           LTT.MAINSTREAM_LEADIN_BEFORE,
                           LTT.MAINSTREAM_LEADIN_BEFORE_MAJOR).sum())
_lo = int(app.trigger_mask(df, ms, ms_major, frequency,
                           LTT.MAINSTREAM_LEADOUT_BEFORE,
                           LTT.MAINSTREAM_LEADOUT_BEFORE_MAJOR).sum())
_mode = '日线 MAJOR' if _use_major else '本频'
st.caption(f"Mainstream（{_mode}）：leadin {_li} 次 / leadout {_lo} 次"
           f"（fold={fold_mainstream}；标记在 Prophet LB 下方 / UB 上方）")

# --------------------------------------------------------------------------- #
# 图 1：K 线 × 中枢 × RENKO
# --------------------------------------------------------------------------- #
fig = app.make_figure(df)

if show_gear_o and gear_bf is not None:
    _runs = add_gear_bands(fig, gear_bf, frequency=frequency)
    if _runs:
        st.caption(f"橙色背景：`QuadGear0Bf < 1e-12`（O 档触发，"
                   f"{len(_runs)} 段；60min 口径）")

if show_bi:
    fig.add_trace(go.Scatter(
        x=[p['dt'] for p in bi_points], y=[p['bi'] for p in bi_points],
        mode='lines+markers', name='笔',
        line=dict(color='royalblue', width=1.5),
        marker=dict(size=4, color='royalblue')))

if show_pivots:
    plot_pivots_plotly(fig, pivots, x_last=df['dt'].iloc[-1], show_ggdd=show_ggdd)

# RENKO S / L 两族砖界（台阶线）
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

# RENKO_BAR 信号（两族同向）
if renko_ok and show_signal and FLD.RENKO_TREND in feats.columns:
    sig = pd.to_numeric(feats[FLD.RENKO_TREND], errors='coerce').values
    close = pd.to_numeric(df['close'], errors='coerce').values
    for val, col, nm, sym in ((1, 'red', 'RENKO_BAR = +1', 'triangle-up'),
                              (-1, 'green', 'RENKO_BAR = -1', 'triangle-down')):
        m = (sig == val)
        if m.any():
            fig.add_trace(go.Scatter(
                x=df['dt'][m], y=close[m], mode='markers', name=nm,
                marker=dict(symbol=sym, size=8, color=col, opacity=0.75)))

app.add_mainstream_triggers(fig, data, show_leadin, show_leadout)
app.finish_figure(fig)
st.plotly_chart(fig, width='stretch')

st.caption("⚠️ 中枢是「三笔重叠」的**结构**，RENKO 是「固定砖高」的**趋势** —— "
           "两者没有共同的数学基础，图上分歧是常态。RENKO 是 review 的 as-of "
           "落库值（每交易日 15:00、≤T 的 2160 根），不是本页重算。")

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
                        FLD.RENKO_BOOST_L_TIMING_LAG) if c in feats.columns]
    if cols:
        st.dataframe(feats[cols].tail(200), width='stretch')
    else:
        st.caption("当前频率没有 Renko 落库列（只有 60min 表有）。")

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
