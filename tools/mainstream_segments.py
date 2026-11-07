# coding: utf-8
"""Mainstream leadin → leadout 的交易区间与区间收益。

规则
====
在一根 K 线上 `MsLeadin*Bf == 0`（刚触发）时**按收盘价买入**；此后第一根
`MsLeadout*Bf == 0` 的 K 线**按收盘价卖出**。持仓期间再出现的 leadin 忽略；
未等到 leadout 的按最后一根收盘价标记为「持仓中」（不计入统计）。

⚠️ **理想化口径**：同 bar 收盘成交、无手续费/印花税/滑点、无涨跌停与 T+1 限制。
回测数字只用于比较信号本身，不能当实盘收益。

用法
====
    python tools/mainstream_segments.py                      # 三只样例股
    python tools/mainstream_segments.py --codes 600519 --freq day
    python tools/mainstream_segments.py --signal Bf --since 2018-01-01
    python tools/mainstream_segments.py --no-fold            # 原公式（高价股几乎无信号）
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

from GolemQ import get_active_market
from GolemQ.analysis.indicators import calc_macd_qa
from GolemQ.analysis.quadrant_jit import calc_mainstream_feats
from GolemQ.core.constants import AKA, FIELD as FLD
from GolemQ.models.alias import LTT

SIGNALS = {
    'Bf': (LTT.MAINSTREAM_LEADIN_BEFORE, LTT.MAINSTREAM_LEADOUT_BEFORE),
    'x3': (LTT.MAINSTREAM_LEADINx3_BEFORE, LTT.MAINSTREAM_LEADOUTx3_BEFORE),
    'x5': (LTT.MAINSTREAM_LEADINx5_BEFORE, LTT.MAINSTREAM_LEADOUTx5_BEFORE),
}


def load_feats(code: str, freq: str, fold: bool):
    """取 K 线并算两族特征；返回 `(timestamps, close, features, name)`。

    无数据返回 `None`（调用方已判空）。`name` 是展示名，仅供表头。
    """
    market = get_active_market()
    start = '2015-01-01' if freq == 'day' else '2016-01-01'
    if freq == 'day':
        data, name = market.get_kline_price_v3(
            code, start=start, realtime=False, verbose=False)
    else:
        data, name = market.get_kline_price_min(
            code, start=start, frequency=freq, realtime=False, verbose=False)
    if data is None or data.data.empty:
        return None
    ohlc = data.data[~data.data.index.duplicated(keep='last')].sort_index()
    feats = pd.DataFrame(index=ohlc.index)
    feats[FLD.MACD] = calc_macd_qa(
        pd.to_numeric(ohlc[AKA.CLOSE], errors='coerce'))[2].to_numpy()
    feats = calc_mainstream_feats(ohlc, feats, fold=fold)
    ts = pd.DatetimeIndex(ohlc.index.get_level_values(0))
    close = pd.to_numeric(ohlc[AKA.CLOSE], errors='coerce').to_numpy()
    return ts, close, feats, name


def build_segments(ts, close, entry, exit_):
    """按「leadin 进 / leadout 出」配对；返回已平仓段与未平仓段。"""
    closed, i_entry = [], None
    for i in range(len(ts)):
        if i_entry is None and entry[i]:
            i_entry = i
        elif i_entry is not None and exit_[i]:
            closed.append((i_entry, i))
            i_entry = None
    return closed, ((i_entry, len(ts) - 1) if i_entry is not None else None)


def report(code, name, freq, signal, ts, close, feats, since, show=8):
    entry_col, exit_col = SIGNALS[signal]
    mask = ts >= pd.Timestamp(since)
    if not mask.any():
        print(f'\n=== {code} {name} [{freq} · {signal}]  无 ≥ {since} 的 K 线')
        return
    entry = (feats[entry_col].to_numpy() == 0) & mask
    exit_ = (feats[exit_col].to_numpy() == 0) & mask
    closed, open_seg = build_segments(ts, close, entry, exit_)

    print(f'\n=== {code} {name} [{freq} · {signal}]  '
          f'{ts[mask][0]:%Y-%m-%d} ~ {ts[-1]:%Y-%m-%d}')
    rets = []
    for idx, (a, b) in enumerate(closed):
        r = close[b] / close[a] - 1
        rets.append(r)
        if len(closed) - idx <= show:
            print(f'  {ts[a]:%Y-%m-%d %H:%M} → {ts[b]:%Y-%m-%d %H:%M} '
                  f'{b - a:>4}根 {r * 100:+7.2f}%  ({close[a]:.2f}→{close[b]:.2f})')
    if len(closed) > show:
        print(f'  …（前 {len(closed) - show} 段略）')
    if open_seg:
        a, b = open_seg
        print(f'  [持仓中] {ts[a]:%Y-%m-%d %H:%M} → {ts[b]:%Y-%m-%d %H:%M} '
              f'{b - a:>4}根 {(close[b] / close[a] - 1) * 100:+7.2f}%  '
              f'({close[a]:.2f}→{close[b]:.2f})')
    if not rets:
        print('  无已平仓区间')
        return
    arr = np.asarray(rets)
    bh = close[mask][-1] / close[mask][0] - 1
    print(f'  统计: {len(arr)}段  胜率 {(arr > 0).mean() * 100:.0f}%  '
          f'平均 {arr.mean() * 100:+.2f}%  中位 {np.median(arr) * 100:+.2f}%  '
          f'最好 {arr.max() * 100:+.1f}%  最差 {arr.min() * 100:+.1f}%  '
          f'复利 {(np.prod(1 + arr) - 1) * 100:+.1f}%  | 买入持有 {bh * 100:+.1f}%')


def main(argv=None):
    parser = argparse.ArgumentParser(description='Mainstream leadin→leadout 区间收益')
    parser.add_argument('--codes', default='000711,301308,601869')
    parser.add_argument('--freq', default='day,60min',
                        help='day / 60min / 30min / 15min，可逗号分隔')
    parser.add_argument('--signal', default='Bf,x3,x5')
    parser.add_argument('--since', default='2020-01-01')
    parser.add_argument('--show', type=int, default=8, help='每段列表显示最近 N 段')
    parser.add_argument('--no-fold', action='store_true',
                        help='关闭股价折叠（>10 取 log10）')
    args = parser.parse_args(argv)

    for code in [c.strip() for c in args.codes.split(',') if c.strip()]:
        for freq in [f.strip() for f in args.freq.split(',') if f.strip()]:
            loaded = load_feats(code, freq, fold=not args.no_fold)
            if loaded is None:
                print(f'\n=== {code} [{freq}] 无数据')
                continue
            ts, close, feats, name = loaded
            for signal in [s.strip() for s in args.signal.split(',') if s.strip()]:
                report(code, name, freq, signal, ts, close, feats,
                       args.since, show=args.show)


if __name__ == '__main__':
    sys.exit(main())
