#!/usr/bin/env python3
# coding: utf-8
"""全局智能仓位（mainstream）时间线 → **仓位推荐** + 图。

数据源：`stock_massive_reality_60min_models`（D79，市场级、每根 60min bar 一行）。

口径（D79）：

* ``PosBsln``      = m_bull —— 建议仓位（牛市占比）；
* ``PosNegBsln``   = m_bear —— 做空比；`> 0.618 且占优` ⇒ **强制清仓**；
* ``MasPosOnhold`` = 有效仓位（强制清仓后为 0）；
* ``PosFull``      = 满仓信号（`m_bull > 0.618` 且同时大于 m_bear / m_omega）；
* ``PosFullOnset`` = 满仓段第一根；``PosBottom`` = 空仓 + STAGE_MODE ≤ -0.5；
* ``PosReClear``   = 清仓暴击（清仓后 ≤4 根又再清仓）。

用法::

    python tools/plot_position.py                 # 全历史 + 最近 120 个交易日
    python tools/plot_position.py --zoom 60       # 缩放窗口改 60 天
    python tools/plot_position.py --out xxx.png

⚠️ 推荐的**不是预测**：它是 D79 事件规则在当前 bar 上的读数（空仓 / 满仓 /
按 m_bull 持有）。线条本身不含未来函数，但 `PosBottom` 依赖的 STAGE_MODE 是
横截面当日快照（D79 增补二）。
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

COLL = 'stock_massive_reality_60min_models'
STAMPS = ('10:30', '11:30', '14:00', '15:00')
BEAR_CLEAR = 0.618


def load_series(db=None) -> pd.DataFrame:
    """整张 massive 表 → DataFrame（datetime 索引），并把缺失 bar **补全到 4 根/日**。"""
    if db is None:
        from GolemQ import get_active_market
        db = get_active_market().get_database()
    cols = ['PosBsln', 'PosNegBsln', 'PosPad', 'MasPosOnhold', 'mberCnt',
            'PosFull', 'PosFullOnset', 'PosBottom', 'PosProbe', 'PosReClear']
    rows = list(db[COLL].find({}, dict({'_id': 0, 'datetime': 1},
                                       **{c: 1 for c in cols})))
    if not rows:
        raise SystemExit(f'{COLL} 是空的 —— 先跑 `--strategy mainstream`')
    df = pd.DataFrame(rows)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').set_index('datetime')

    # 补全：数据里偶有 3 根/日的日子（2026-10-09/29/30），按 4 个标准 stamp 建格，
    # 同日 ffill（市场级读数在当日内不会凭空变化；缺失 bar 是没算，不是 0）。
    days = df.index.normalize().unique()
    grid = pd.DatetimeIndex([d + pd.Timedelta(hours=int(s[:2]), minutes=int(s[3:]))
                             for d in days for s in STAMPS])
    return df.reindex(grid).ffill()


def recommend(row: pd.Series) -> str:
    """当前 bar 的**仓位推荐**（D79 事件规则，非预测）。"""
    if not np.isfinite(row.get('MasPosOnhold', np.nan)):
        return '无数据'
    if row.get('MasPosOnhold', 1) <= 0:
        return '空仓（做空比触发强制清仓）'
    if int(row.get('PosFull', 0) or 0) == 1:
        return '满仓（m_bull 占优 > 61.8%）'
    pos = float(row['MasPosOnhold'])
    if pos < 0.30:
        return '轻仓（{:.2%}）'.format(pos)
    if pos < 0.60:
        return '半仓（{:.2%}）'.format(pos)
    return '重仓（{:.2%}）'.format(pos)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description='mainstream 仓位时间线 + 推荐 + 图')
    p.add_argument('--zoom', type=int, default=120, help='放大窗口（交易日数）')
    p.add_argument('--out', default='', help='PNG 输出路径（默认 output/position_<日期>.png）')
    p.add_argument('--no-plot', action='store_true', help='只打印推荐，不画图')
    a = p.parse_args(argv)

    df = load_series()
    last = df.iloc[-1]
    print('数据 {} 根 / {} 个交易日：{} -> {}'.format(
        len(df), df.index.normalize().nunique(), df.index[0], df.index[-1]))
    print('最新 {}：仓位 {:.2%}  做空比 {:.2%}  有效仓位 {:.2%}'.format(
        df.index[-1], last['PosBsln'], last['PosNegBsln'], last['MasPosOnhold']))
    print('推荐：{}'.format(recommend(last)))
    print('事件：满仓 {} 次 / 空仓见底 {} 次 / 清仓暴击 {} 次（全历史）'.format(
        int(df['PosFullOnset'].fillna(0).sum()),
        int(df['PosBottom'].fillna(0).sum()),
        int(df['PosReClear'].fillna(0).sum())))

    # 近 5 个交易日（每日 15:00 收盘 bar）
    daily = df[df.index.strftime('%H:%M') == '15:00'].tail(5)
    print('\n近 5 日（15:00）：')
    for ts, r in daily.iterrows():
        print('  {}  仓位 {:5.2%}  做空比 {:5.2%}  有效 {:5.2%}  {}'.format(
            ts.date(), r['PosBsln'], r['PosNegBsln'], r['MasPosOnhold'],
            recommend(r)))

    if a.no_plot:
        return 0

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    fig, axes = plt.subplots(2, 1, figsize=(15, 10), sharey=False)
    for ax, data, title in (
            (axes[0], df, '全历史（2022-06 起）'),
            (axes[1], df.tail(a.zoom * 4), '最近 {} 个交易日'.format(a.zoom))):
        x = data.index
        ax.fill_between(x, 0, data['MasPosOnhold'].astype(float) * 100,
                        color='tab:blue', alpha=0.25, label='有效仓位（%）')
        ax.plot(x, data['PosBsln'].astype(float) * 100, color='tab:blue',
                lw=1.0, label='建议仓位 m_bull（%）')
        ax.plot(x, data['PosNegBsln'].astype(float) * 100, color='tab:red',
                lw=1.0, alpha=0.8, label='做空比 m_bear（%）')
        ax.axhline(BEAR_CLEAR * 100, color='tab:red', ls='--', lw=0.8,
                   label='强制清仓线 61.8%')

        def _marks(col, marker, color, label):
            m = data[col].fillna(0).astype(int) == 1
            if m.any():
                ax.scatter(x[m], np.full(int(m.sum()), -6), marker=marker,
                           s=40, color=color, label=label, zorder=5,
                           clip_on=False)

        _marks('PosFullOnset', '^', 'tab:green', '满仓信号')
        _marks('PosBottom', 'v', 'tab:orange', '空仓见底')
        _marks('PosReClear', 'x', 'tab:purple', '清仓暴击')
        ax.set_title(title)
        ax.set_ylim(-10, 105)
        ax.grid(alpha=0.3)
        ax.legend(loc='upper left', ncol=3, fontsize=8)

    fig.suptitle('mainstream 全局智能仓位 —— 最新 {}：{}'.format(
        df.index[-1], recommend(last)), fontsize=13)
    fig.tight_layout()

    out = a.out or os.path.join(
        _REPO, 'output', 'position_{}.png'.format(df.index[-1].strftime('%Y%m%d')))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=110)
    print('\n图已保存：{}'.format(out))
    return 0


if __name__ == '__main__':
    sys.exit(main())
