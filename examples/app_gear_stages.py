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
from GolemQ.models.alias import LTT, ZEN

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
    show_cs_buy = st.checkbox(
        "ChartScan：Buy 全高带（orange / lightcoral）", value=False,
        help="`ChartScan 60min/日线 Buy < Leadout Maj & < 日线 Sell` 的两条全高带。"
             "它等价于 lag 副图里 `ChrtScanUprTukLag > 0`，也与 navy/crimson 仓位带"
             "大面积重合 ⇒ **默认关**（用户 2026-11-28）。")
    show_chartscan = st.checkbox(
        "ChartScan：Sell 三时钟带", value=True,
        help="15min→lime / 60min→olive / 日线→green。Sell 状态画带（即老树的 "
             "STOPLOSS 区段）；数据来自 `*_reality_feat_60min` 的 "
             "`Cls60/ClsDay/Cls15` + `Coord*`。")
    show_stage_dots = st.checkbox(
        "bstStgBf/epStgBf 圆点（ATR_MAJ）", value=True,
        help="`bstStgBf < 1e-12` ⇒ 红圆 @ `ATR_LB_MAJ`；"
             "`epStgBf < 1e-12` ⇒ 蓝圆 @ `ATR_UB_MAJ`。"
             "ATR_MAJ 由 compact 的日线派生列因果映射而来。")
    show_credit_star = st.checkbox(
        f"{LTT.QUADRANT_MAINSTREAM_CREDIT_COMPOUNDED} > 3 五角星（ATR_LB_MAJ）",
        value=True,
        help="多周期买点共振权重（15min 1 / 60min 2 / day 3，`MsLeadinBf` 补充 +1，"
             "按自然日相加）**> 3** 的 bar ⇒ 金色五角星 @ `ATR_LB_MAJ`。"
             "权重是**按自然日**汇总的，再因果映射到 bar（盘中用上一交易日、"
             "15:00 用当日）—— 所以同一天的多根 bar 常是同一个权重。"
             "只在 60min 有日线帧时可用；取值来源 `analysis/mtf_buy_weight.py`。")
    clear_thr = st.slider("仓位阈值：几条滞后同向", 2, 6, 4,
                          help="六条滞后（M9tTrkLag / ChrtScanUprTukLag / "
                               "QuadBsTukLag / ZenOLSLag / QuadDwnRskLagMaj / "
                               "QLevMACDLagD）里**至少这么多条为负** ⇒ navy 全高带（清仓）、"
                               "**至少这么多条为正** ⇒ crimson 全高带（满仓）；"
                               "NaN 不计票。默认 4（含 4）。")
    show_trunks = st.checkbox(
        "滞后类：M9tTrkLag / ChrtScanUprTukLag / QuadBsTukLag / ZenOLSLag /"
        " QuadDwnRskLagMaj / QLevMACDLagD",
        value=True,
        help="`QUADRANT_BS_TRUNK_TIMING_LAG`：正值 = 连续满足「买卖星线主干」的根数、"
             "负值 = 连续不满足（不会出现 0）。画在**右轴 y4**，本页只读、不重算。"
             "⚠️ `CHARTSCAN_UPRISING_TRUNK_TIMING_LAG` **不画** —— 它就是上面两条"
             "全高带（orange ∪ lightcoral）的数值化（正数=连续命中根数），"
             "再画一遍是重复。")
    show_ols = st.checkbox(
        f"ZEN.OLS {ZEN.OLS_TIMING_LAG}", value=True,
        help="段内 LIS/LDS 的「波浪趋势」时序（D89，review 落 60min 表）："
             "`>0` = 这一段内的上升趋势、`<0` = 下降趋势，数值是距最近一次"
             "方向变化的 bar 数。画在**右轴 y3**（青绿点线），本页只读、不重算。")
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

# ChartScan：Sell 三时钟带（副轴）/ Buy 全高带（两个独立开关）
if show_chartscan:
    app.add_chartscan_sell_bands(fig, feats, df, echo=st.caption)
# `add_chartscan_buy_bands`（orange / lightcoral）默认**不画**：它等价于
# `ChrtScanUprTukLag > 0`，且与 navy/crimson 仓位带大面积重合（用户 2026-11-28）。
if show_cs_buy:
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

# 多周期买点共振权重 `QuadMsCredCpd > 3` 的五角星（用户 2026-11-25）——
# 与 bstStgBf 红圆同一基线 `ATR_LB_MAJ`；只在 60min（日线帧）成立。
if show_credit_star:
    _cred = LTT.QUADRANT_MAINSTREAM_CREDIT_COMPOUNDED
    if frequency != app.RENKO_FREQ:
        st.caption(f"`{_cred}` 需要日线帧映射 —— 只在 60min 画五角星，当前 {frequency} 跳过。")
    else:
        _w = app.mtf_credit_weight(symbol, frequency, long_history,
                                   fold_mainstream)
        _bands = app.atr_major_bands(symbol, frequency, long_history,
                                     fold_mainstream)
        if _w is None:
            st.caption(f"`{_cred}` 取不到（缺 15min/日线帧或三周期都算不出来）—— 跳过五角星。")
        elif _bands is None or FLD.ATR_LB_MAJOR not in _bands.columns:
            st.caption("ATR_MAJ 取不到（缺日线帧）—— 跳过五角星。")
        else:
            _w = _w.reindex(data['ohlc'].index)   # 与 K 线同索引、同序
            _wv = pd.to_numeric(_w, errors='coerce').to_numpy()
            _yv = pd.to_numeric(_bands[FLD.ATR_LB_MAJOR],
                                errors='coerce').to_numpy()
            _xv = df['dt'].to_numpy()
            if len(_wv) != len(df) or len(_yv) != len(df):
                st.caption(f"`{_cred}` / `{FLD.ATR_LB_MAJOR}` 长度与 K 线不一致 "
                           f"（{len(_wv)} / {len(_yv)} vs {len(df)}）—— 跳过五角星。")
            else:
                _hit = np.isfinite(_wv) & (_wv > 3) & np.isfinite(_yv)
                if _hit.any():
                    fig.add_trace(go.Scatter(
                        x=_xv[_hit], y=_yv[_hit], mode='markers',
                        name=f'{_cred} > 3',
                        marker=dict(symbol='star', size=13, color='gold',
                                    line=dict(color='darkorange', width=1))))
                _vals = pd.Series(_wv).dropna()
                if len(_vals):
                    st.caption(
                        f"`{_cred} > 3`：{int(_hit.sum())} 处（金色五角星 @ "
                        f"`{FLD.ATR_LB_MAJOR}`）；有值 bar {int(len(_vals))}，"
                        f"末值 **{float(_vals.iloc[-1]):.0f}**。")
                else:
                    st.caption(f"`{_cred}` 全为空 —— 跳过五角星。")

# Mainstream leadin / leadout
app.add_mainstream_triggers(fig, data, show_leadin, show_leadout)

# ---- 上升系数：**叠在主图的副轴**上（`ax3 = ax1.twinx()` ≡ `yaxis='y2'`）----
if show_coef and coef is not None:
    fig.add_trace(go.Scatter(
        x=df['dt'], y=coef.to_numpy(), mode='lines', name=app.COEF_COL,
        line=dict(color='darkred', dash='dash', width=1), opacity=0.67,
        yaxis='y2'))

# ---- 滞后类序列：**单独画在「bar 序号等距」的副图上**
# 为什么不叠在主图的时间轴：60min bar 的 x 间距是 1h / 2.5h / 19.5h（隔夜）/
# 67.5h（周末）—— 每根 +1 的阶梯在时间轴上会被拉成「一段陡一段平」（用户 2026-11-26）。
# 按 bar 序号画，每根一格，正常的递进就是 45° 直线；有锯齿说明真的漏根/重复。
_lags = {}
_lag_caps = []          # (name, 末值说明文本)

if show_ols:
    _ols = None
    if feats is not None and ZEN.OLS_TIMING_LAG in getattr(feats, 'columns', ()):
        _ols = pd.to_numeric(feats[ZEN.OLS_TIMING_LAG], errors='coerce')
    if _ols is None or len(_ols) != len(df):
        st.caption(f"`{ZEN.OLS_TIMING_LAG}` 取不到（本频率没有 attach 该列，"
                   "或长度与 K 线不一致）—— 跳过。")
    else:
        _lags[ZEN.OLS_TIMING_LAG] = (_ols, 'teal', 'dot')
        _v = _ols.dropna()
        if len(_v):
            _lag_caps.append((ZEN.OLS_TIMING_LAG,
                              f"末值 **{float(_v.iloc[-1]):.0f}**（近 "
                              f"{int(len(_v))} 根有值；`>0` 上升段内、`<0` 下降段内）"))
        else:
            st.caption(f"`{ZEN.OLS_TIMING_LAG}` 全为空 —— review 还没跑出这列。")

if show_trunks:
    _tr = app.trunk_lags(symbol, frequency, long_history, fold_mainstream)
    if not _tr:
        st.caption("主干时序取不到（只在 60min、且需要 15min/日线帧）—— 跳过。")
    else:
        _styles = {LTT.QUADRANT_BS_TRUNK_TIMING_LAG: ('purple', 'solid'),
                   LTT.CHARTSCAN_UPRISING_TRUNK_TIMING_LAG: ('goldenrod', 'dash'),
                   LTT.MAGIC_NINE_TURNS_TRUNK_TIMING_LAG: ('royalblue', 'dashdot')}
        for _name, _s in _tr.items():
            _col, _dash = _styles.get(_name, ('gray', 'dot'))
            _v = pd.to_numeric(_s, errors='coerce')
            if len(_v) != len(df):
                st.caption(f"`{_name}` 长度 {len(_v)} 与 K 线 {len(df)} 不一致 —— 跳过。")
                continue
            _lags[_name] = (_v, _col, _dash)
            _d = _v.dropna()
            if len(_d):
                _lag_caps.append((_name, f"末值 **{float(_d.iloc[-1]):.0f}**"
                                         f"（近 {int(len(_d))} 根有值；正=连续成立、负=连续不成立）"))

# `QuadDwnRskLagMaj`（下行风险 lag，日线 MAJOR 映射到 60min 后**按 60min bar 逐根 ±1**）
_dwn = None
if ms is not None and LTT.QUADRANT_DOWNRISK_TIMING_LAG_MAJOR in getattr(
        ms, 'columns', ()):
    _dwn = pd.to_numeric(ms[LTT.QUADRANT_DOWNRISK_TIMING_LAG_MAJOR],
                         errors='coerce')
if _dwn is not None and len(_dwn) == len(df):
    _lags[LTT.QUADRANT_DOWNRISK_TIMING_LAG_MAJOR] = (_dwn, 'olive', 'dot')
    _d = _dwn.dropna()
    if len(_d):
        _lag_caps.append((LTT.QUADRANT_DOWNRISK_TIMING_LAG_MAJOR,
                          f"末值 **{float(_d.iloc[-1]):.0f}**（近 {int(len(_d))} 根"
                          f"有值；>0 上行风险段、<0 下行风险段）"))

# `QLevMACDLagD`（四量图杠杆 MACD 时序）：同样逐 bar ±1，一起放进 lag 副图
_macd_lag = None
if ms is not None and LTT.QUADRANT_LEVERAGE_MACD_TIMING_LAG_DUMMY in getattr(
        ms, 'columns', ()):
    _macd_lag = pd.to_numeric(
        ms[LTT.QUADRANT_LEVERAGE_MACD_TIMING_LAG_DUMMY], errors='coerce')
if _macd_lag is not None and len(_macd_lag) == len(df):
    _lags[LTT.QUADRANT_LEVERAGE_MACD_TIMING_LAG_DUMMY] = (
        _macd_lag, 'crimson', 'dot')
    _d = _macd_lag.dropna()
    if len(_d):
        _lag_caps.append((LTT.QUADRANT_LEVERAGE_MACD_TIMING_LAG_DUMMY,
                          f"末值 **{float(_d.iloc[-1]):.0f}**（近 {int(len(_d))} 根"
                          f"有值；四量图杠杆 MACD 时序）"))

# ---- 仓位信号（用户 2026-11-27/28）：六条滞后统计同向条数
#   **≥阈值条为负** ⇒ navy 全高带（清仓）；**≥阈值条为正** ⇒ crimson 全高带（满仓）。
#   两条不可能同时成立（6 条里 ≥4 负 + ≥4 正 需要 8 条）。NaN 不计票。
if len(_lags) >= 2:
    _neg = np.zeros(len(df), dtype=int)
    _pos = np.zeros(len(df), dtype=int)
    for _name, (_v, _c, _dash) in _lags.items():
        _arr = pd.to_numeric(_v, errors='coerce').to_numpy(dtype=float)
        _neg += (_arr < 0)
        _pos += (_arr > 0)
    _pad = pd.Timedelta(minutes=30)

    def _band(_mask, _color, _opacity):
        _runs = app.trigger_runs(pd.Series(np.where(_mask, 0.0, 1.0),
                                           index=df['dt']))
        for _a, _b in _runs:
            fig.add_shape(type='rect', xref='x', yref='paper',
                          x0=_a - _pad, x1=_b + _pad, y0=0, y1=1,
                          fillcolor=_color, opacity=_opacity,
                          line=dict(width=0), layer='below')
        return _runs

    _thr = int(clear_thr)
    _clear = _neg >= _thr
    _full = _pos >= _thr
    _runs_clear = _band(_clear, 'navy', 0.18) if _clear.any() else []
    _runs_full = _band(_full, 'crimson', 0.10) if _full.any() else []
    st.caption(
        f'**navy 全高带 = 清仓**：{len(_lags)} 条滞后里 ≥{_thr} 条为负 ⇒ '
        f'{len(_runs_clear)} 段 / {int(_clear.sum())} 根；'
        f'**crimson 全高带 = 满仓**：≥{_thr} 条为正 ⇒ '
        f'{len(_runs_full)} 段 / {int(_full.sum())} 根（NaN 不计票）')

if _lags:
    _x = np.arange(len(df))
    fig2 = go.Figure()
    for _name, (_v, _col, _dash) in _lags.items():
        fig2.add_trace(go.Scatter(
            x=_x, y=_v.to_numpy(), mode='lines', name=_name,
            line=dict(color=_col, dash=_dash, width=1.2), opacity=0.85,
            connectgaps=False, customdata=df['dt'].to_numpy(),
            hovertemplate='%{customdata|%Y-%m-%d %H:%M}  %{y:.0f}<extra></extra>'))
    _step = max(1, len(_x) // 10)
    _tick = list(_x[::_step])
    fig2.update_layout(
        height=260, margin=dict(l=10, r=10, t=10, b=10),
        hovermode='x unified', legend=dict(orientation='h', y=1.02),
        xaxis=dict(title='bar 序号（等距，鼠标悬停看时间）',
                   tickmode='array', tickvals=_tick,
                   ticktext=[str(pd.Timestamp(d))[:10]
                             for d in df['dt'].to_numpy()[_tick]]),
        yaxis=dict(title='lag', zeroline=True,
                   zerolinecolor='rgba(128,128,128,0.5)'))
    st.plotly_chart(fig2, width='stretch')
    st.caption('⚠️ 这几条都是**逐 bar 滞后量**：正常递进就是 ±1/根（45° 直线），'
               '大跳变 = 方向翻转（`calc_event_timing_lag` 重置）。'
               '按 bar 序号等距画，才不会被时间轴的空档（隔夜 19.5h、周末 67.5h）'
               '拉成一段陡一段平。')
    for _name, _txt in _lag_caps:
        st.caption(f"`{_name}`：{_txt}")

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
