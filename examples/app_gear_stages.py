# coding: utf-8
"""上升系数 × ChartScan —— **叠加图**演示（新树 / MongoDB 8.3 版）

运行方式：

    streamlit run examples/app_gear_stages.py

与其它 `app_*` 一样只保留「标题 + 侧边栏开关 + 本页的 layer」，
取数 / attach / Mainstream / K 线骨架 / ChartScan / Charm 全在
`examples/_app_common.py`（用户 2026-11-15：统一架构）。

本页**不画**：缠论（笔/线段/中枢）、QuadGear0 橙色区段、RENKO 砖界与三角点
（用户 2026-11-15 决定）。czsc 分笔仍在 `load_base` 里算 —— 上升系数要用 `bi_list`。

保留的两层
==========
| | 来源 | 回答什么 |
|:--|:--|:--|
| **上升系数** | `LTT.QUADRANT_UPRISING_I_COEF_DUMMY`（`QUprICoefD`） | 四量图链里的**优先级权重**（`portfolio/strategy.py` 的 `priority()` 用它排序） |
| **ChartScan / Charm** | `Cls60/ClsDay/Cls15` + `Coord*`；`QLevCharm{O,I}Bf` | Buy/Sell 三时钟带 + 触发点 |

⚠️ 上升系数**只画真实算出来的值**：算不出来就在页面上写清是哪一步（不造数）。
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

from GolemQ.core.constants import FIELD as FLD
from GolemQ.models.alias import LTT

import _app_common as app
from _gear_bands import trigger_runs

st.set_page_config(page_title="上升系数 × ChartScan", layout="wide")
st.title("📈🔍 上升系数 × ChartScan（叠加图）")

# --------------------------------------------------------------------------- #
# 侧边栏
# --------------------------------------------------------------------------- #
with st.sidebar:
    symbol, frequency, long_history = app.sidebar_base(default_freq_index=1)

    st.divider()
    st.caption("显示开关 —— 各层可单独关，便于对照")
    show_coef = st.checkbox("上升系数", value=True)
    show_chartscan = st.checkbox(
        "ChartScan：Buy/Sell 三时钟带", value=True,
        help="15min→lime / 60min→olive / 日线→green。Sell 状态画带（即老树的 "
             "STOPLOSS 区段）；数据来自 `*_reality_feat_60min` 的 "
             "`Cls60/ClsDay/Cls15` + `Coord*`。")
    show_stage_dots = st.checkbox(
        "bstStgBf/epStgBf 圆点（ATR_MAJ）", value=True,
        help="`bstStgBf < 1e-12` ⇒ 红圆 @ `ATR_LB_MAJ`；"
             "`epStgBf < 1e-12` ⇒ 蓝圆 @ `ATR_UB_MAJ`。"
             "ATR_MAJ 由 compact 的日线派生列因果映射而来。")
    show_charm = st.checkbox(
        "Charm_O/I 触发点", value=True,
        help="`QLevCharmOBf / QLevCharmIBf < 1e-12`（0 = 触发）⇒ 紫点 / 红点，"
             "y = `CCI_NORM.min()-1`。守卫：Charm_O 触发数 > 3.82×Charm_I 时"
             "视为口径异常，**O 不画**。")

    st.divider()
    show_leadin = st.checkbox("Mainstream：吸筹 leadin 触发", value=True)
    show_leadout = st.checkbox("Mainstream：派发 leadout 触发", value=True)
    fold_mainstream = True

    st.divider()
    st.caption("ChartScan / Charm / 上升系数都基于 `*_reality_feat_60min`"
               "（`--review` 的 as-of 产出）；本页**不重算**。")

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
    st.warning(f"{symbol} 的 K 线不足 —— 三条方法都算不出来。")
    st.stop()

feats = data['feats']
coef, coef_err = data['coef'], data['coef_err']
ms = data['ms']
if data.get('feats_err'):
    st.warning(f"attach `*_reality_feat_60min` 失败：`{data['feats_err']}`")

# --------------------------------------------------------------------------- #
# 指标卡
# --------------------------------------------------------------------------- #
c1, c2 = st.columns(2)
c1.metric("📊 K 线数", f"{len(df):,}")
_coef_last = coef.dropna().tail(1) if coef is not None else None
c2.metric("📈 上升系数",
          f"{float(_coef_last.iloc[0]):.4f}" if _coef_last is not None and len(_coef_last)
          else '—')
_label = symbol if data['name'] == symbol else f"{data['name']}（{symbol}）"
st.caption(f"标的：{_label} · {frequency}")

_li = int(app.trigger_mask(df, ms, data['ms_major'], frequency,
                           LTT.MAINSTREAM_LEADIN_BEFORE,
                           LTT.MAINSTREAM_LEADIN_BEFORE_MAJOR).sum())
_lo = int(app.trigger_mask(df, ms, data['ms_major'], frequency,
                           LTT.MAINSTREAM_LEADOUT_BEFORE,
                           LTT.MAINSTREAM_LEADOUT_BEFORE_MAJOR).sum())
_mode = ('日线 MAJOR' if (frequency == '60min' and data['ms_major'] is not None
                          and len(data['ms_major']) > 0) else '本频')
st.caption(f"Mainstream（{_mode}）：leadin {_li} 次 / leadout {_lo} 次"
           f"（fold={fold_mainstream}；标记在 Prophet LB 下方 / UB 上方）")

# --------------------------------------------------------------------------- #
# 图 1：K 线 × 上升系数 × ChartScan（叠加）
# --------------------------------------------------------------------------- #
fig = app.make_figure(df)

# QuadDwnRskLagMaj > 0 的黄色带：等价 matplotlib
# ax3.fill_between(x, CCI_NORM.max()-1, CCI_NORM.min()-1,
#                  color='yellowgreen', alpha=0.832)
if (LTT.QUADRANT_DOWNRISK_TIMING_LAG_MAJOR in ms.columns
        and FLD.CCI_NORM in feats.columns):
    _dwn = pd.to_numeric(
        ms[LTT.QUADRANT_DOWNRISK_TIMING_LAG_MAJOR], errors='coerce')
    _cci = pd.to_numeric(feats[FLD.CCI_NORM], errors='coerce').dropna()
    if len(_dwn) == len(df) and len(_cci):
        _band = pd.Series(
            np.where(_dwn.to_numpy() > 1e-12, 0.0, 1.0), index=ms.index)
        _runs = trigger_runs(_band)
        if _runs:
            _y0, _y1 = float(_cci.min() - 1), float(_cci.max() - 1)
            _pad = pd.Timedelta(minutes=30)
            for _a, _b in _runs:
                fig.add_shape(
                    type='rect', xref='x', yref='y',
                    x0=_a - _pad, x1=_b + _pad, y0=_y0, y1=_y1,
                    fillcolor='yellowgreen', opacity=0.832,
                    line=dict(width=0), layer='below')
            st.caption(f"yellowgreen 区域：`QuadDwnRskLagMaj > 1e-12`，"
                       f"{len(_runs)} 段；y 范围 = `CCI_NORM` min/max - 1。")

# ChartScan：Sell 三时钟带（副轴）+ Buy 晚于 Leadout/日线 Sell 的全高带
if show_chartscan:
    app.add_chartscan_sell_bands(fig, feats, df, echo=st.caption)
    app.add_chartscan_buy_bands(fig, data, echo=st.caption)

# Charm_O / Charm_I 触发点
if show_charm:
    app.add_charm_dots(fig, data, echo=st.caption)

# bstStgBf / epStgBf 触发圆点（用户 2026-11-16）：
# `bstStgBf < 1e-12` ⇒ 红圆 @ `ATR_LB_MAJ`；`epStgBf < 1e-12` ⇒ 蓝圆 @ `ATR_UB_MAJ`。
# ATR_MAJ 不在 reality 表里 —— 用 compact 的日线派生列（`atr_major_bands`）。
if show_stage_dots and frequency == app.RENKO_FREQ and feats is not None:
    _bands = app.atr_major_bands(symbol, frequency, long_history,
                                 fold_mainstream)
    if _bands is None:
        st.caption("ATR_MAJ 取不到（缺日线帧）—— 跳过 bstStgBf/epStgBf 圆点。")
    else:
        _x = df['dt'].to_numpy()
        for _col, _ycol, _color, _label in (
                (FLD.BOOTSTRAP_STAGE_BEFORE, FLD.ATR_LB_MAJOR, 'red', 'bstStgBf'),
                (FLD.ENDPOINT_STAGE_BEFORE, FLD.ATR_UB_MAJOR, 'blue', 'epStgBf')):
            if _col not in feats.columns or _ycol not in _bands.columns:
                continue
            _v = pd.to_numeric(feats[_col], errors='coerce').to_numpy()
            _y = pd.to_numeric(_bands[_ycol], errors='coerce').to_numpy()
            if len(_v) != len(df) or len(_y) != len(df):
                continue
            _hit = np.isfinite(_v) & (_v < 1e-12) & np.isfinite(_y)
            if not _hit.any():
                continue
            fig.add_trace(go.Scatter(
                x=_x[_hit], y=_y[_hit], mode='markers',
                name=f'{_label} 触发',
                marker=dict(symbol='circle', size=7, color=_color)))
            st.caption(f"{_label} 触发：{int(_hit.sum())} 处（{_color}），"
                       f"标在 `{_ycol}`。")

# Mainstream leadin / leadout
app.add_mainstream_triggers(fig, data, show_leadin, show_leadout)

# ---- 上升系数：**叠在主图的副轴**上（`ax3 = ax1.twinx()` ≡ `yaxis='y2'`）----
if show_coef and coef is not None:
    fig.add_trace(go.Scatter(
        x=df['dt'], y=coef.to_numpy(), mode='lines', name=app.COEF_COL,
        line=dict(color='darkred', dash='dash', width=1), opacity=0.67,
        yaxis='y2'))

app.finish_figure(fig)
st.plotly_chart(fig, width='stretch')

# --------------------------------------------------------------------------- #
# 上升系数：说明
# --------------------------------------------------------------------------- #
st.subheader(f"📈 上升系数 `{app.COEF_COL}`")

if not show_coef:
    st.caption("已关闭（侧边栏）。")
elif coef is None:
    st.warning(
        f"**`{app.COEF_COL}` 这一次没算出来** —— `{coef_err}`。\n\n"
        "链路：`4QPwrln`（本频 / 15min / 日线）+ `MACD_DELTA_MAJOR`（日线）"
        " → `QLevMACDLagD` → `QUprICoefD`（`analysis/quadrant_coef.py`）。"
    )
else:
    _last = coef.dropna().tail(1)
    st.caption(
        f"已叠在**主图的右轴**上（matplotlib 的 `ax3 = ax1.twinx()` ≡ Plotly 的 "
        f"`yaxis='y2'`）：`darkred` 虚线 · `lw=1` · `alpha=0.67`。"
        f" 末个非空值 **{float(_last.iloc[0]):.4f}**（近 {int(coef.notna().sum())} 根有值）。"
    )

st.caption("⚠️ 上升系数是四量图链的**优先级权重**，ChartScan 是 CV 的 Buy/Sell "
           "检出 —— 两者**没有共同的数学基础**，图上分歧是常态，本页**不做融合**。")

# --------------------------------------------------------------------------- #
# 明细
# --------------------------------------------------------------------------- #
with st.expander("Mainstream leadin / leadout 明细（末 200 行）"):
    _detail = data['ms_major'] if _mode == '日线 MAJOR' else ms
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
