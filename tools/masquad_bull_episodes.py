# coding: utf-8
"""MasQuad 版「两月牛」：周线 Powerline 双正段 + 日频仓位信号叠加。

对应文档 `docs/massive_timing.md` 第六节。只读，不写库。

判据
====
**段（趋势）**：`MasQuadPwrlnWekMed >= 0` 且 `MasQuadPwrlnWekMode >= 0`；
中间允许**不超过 `--gap` 个交易日**的负脉冲并入（用户 2026-11-22：牛市里
转负合计不到 10 天可视为连续）。短于 `--min-len` 个交易日的碎片丢弃。

**日频叠加**（2022-06 起才有 `MasPosOnhold`，之前只有周线）：

* `PosBottom` / `STAGE_MODE == -0.5` —— 底带；
* **`STAGE_MODE == -0.5` 且次日 `> -0.5`** —— 用户点名的「熊转牛分界」；
* `PosFull` / `PosFullOnset` —— 满仓确认；
* `PosReClear` —— 清仓暴击；
* `MasPosOnhold < 0.25` —— 弱仓。

用法::

    python tools/masquad_bull_episodes.py
    python tools/masquad_bull_episodes.py --csv output/masquad_bull.csv
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


def load_weekly(path: str) -> pd.DataFrame:
    w = pd.read_csv(path, parse_dates=['date']).set_index('date')
    # float32 尾巴会让 `>= 0` / `<= -2.8` 判错（D86 的坑），网格是 0.1
    for c in ('MasQuadPwrlnWekMed', 'MasQuadPwrlnWekMode'):
        w[c] = w[c].round(1)
    return w.sort_index()


def load_daily() -> pd.DataFrame:
    from GolemQ import get_active_market
    db = get_active_market().get_database()
    rows = list(db.stock_massive_reality_60min_models.find(
        {'datetime': {'$regex': ' 15:00:00$'}}, {'_id': 0}).sort('datetime', 1))
    m = pd.DataFrame(rows)
    m['date'] = pd.to_datetime(m['datetime']).dt.normalize()
    m = m.set_index('date').sort_index()
    if 'PosFull' in m.columns:
        # 库里的 `PosFullOnset` 不全（9 次只落 3 次），自己派生
        m['PosFullOnset'] = ((m['PosFull'] == 1)
                             & (m['PosFull'].shift(1) != 1)).astype(np.int8)
    return m


#: 基准指数（用户 2026-11-22 指定：不用沪深300）。
#: 基准指数（用户 2026-11-22 指定看这四条）。
#: ⚠️ 中证2000（932000）**库里一行都没有**，用中证1000（000852）替代；
#: 科创200（000699）只有 2022-12-30 起的数据，涉及它的 n 会明显偏小。
BENCHMARKS = (('沪深300', '000300'), ('科创200', '000699'),
              ('创业板', '399006'), ('中证1000', '000852'))


def load_index(code: str, start: str = '2017-12-01') -> pd.Series:
    from GolemQ import get_active_market
    db = get_active_market().get_database()
    idx = pd.DataFrame(list(db.index_day.find(
        {'code': code, 'date': {'$gte': start}},
        {'_id': 0, 'date': 1, 'close': 1}).sort('date', 1)))
    idx['date'] = pd.to_datetime(idx['date'])
    return idx.set_index('date')['close'].astype(float)


def build_episodes(med: pd.Series, mode: pd.Series,
                   gap: int = 10, min_len: int = 10):
    """双正段（允许 gap 个交易日的负脉冲）→ ``[(i, j), ...]`` 位置索引。"""
    neg = ((med < 0) | (mode < 0)).to_numpy()
    d = med.index
    eps, i, n = [], 0, len(neg)
    while i < n:
        if neg[i]:
            i += 1
            continue
        j = i
        while j + 1 < n:
            if not neg[j + 1]:
                j += 1
                continue
            k = j
            while k + 1 < n and neg[k + 1] and (k + 1 - j) <= gap:
                k += 1
            if k + 1 < n and not neg[k + 1]:
                j = k + 1
            else:
                break
        if j - i + 1 >= min_len:
            eps.append((i, j))
        i = j + 1
    return eps, d


def _count(frame, col, a, b) -> int:
    if frame is None or col not in frame:
        return 0
    sub = frame.loc[a:b]
    return int((sub[col] == 1).sum()) if len(sub) else 0


def build_table(eps, d, px: dict, med, mode, daily):
    rows = []
    for k, (i, j) in enumerate(eps, 1):
        a, b = d[i], d[j]
        base = px[next(iter(px))]
        prev = med.iloc[max(0, i - 1)]
        deep = 'Y' if (i > 0 and med.iloc[i - 1] <= -2.8
                       and mode.iloc[i - 1] <= -2.8) else ''
        row = {'序': k, '起': a.date(), '止': b.date(),
               '交易日': int(base.loc[a:b].shape[0]),
               '起点周线Med': float(prev), '起点深底': deep}
        for name in px:
            seg = px[name].loc[a:b].dropna()
            if len(seg) < 2:
                row[name] = np.nan
                row[f'{name}回撤'] = np.nan
                continue
            row[name] = round(float((seg.iloc[-1] / seg.iloc[0] - 1) * 100), 1)
            row[f'{name}回撤'] = round(float((seg / seg.cummax() - 1).min() * 100), 1)
        row.update({
            '段内PosBottom': _count(daily, 'PosBottom', a, b),
            '段内PosDeepBottom': _count(daily, 'PosDeepBottom', a, b),
            '段内PosFullOnset': _count(daily, 'PosFullOnset', a, b),
            '段内PosReClear': _count(daily, 'PosReClear', a, b),
            '段内PosReClearStrict': _count(daily, 'PosReClearStrict', a, b),
        })
        rows.append(row)
    return pd.DataFrame(rows)


def stage_signals(daily: pd.DataFrame):
    """底带首日 / 底带退出日（用户点名的「熊转牛分界」）。"""
    st = daily.STAGE_MODE
    enter = (st <= -0.5) & (st.shift(1) > -0.5)
    exit_ = (st > -0.5) & (st.shift(1) <= -0.5)
    return enter, exit_


def derived_clear_signals(daily: pd.DataFrame, weak: float = 0.25,
                          reopen_days: int = 3) -> dict:
    """**派生的清仓信号**（库里的 `PosReClear` 只在 60min 的 4 根内才算，
    跨一天就没了，所以口径要自己派生一份日频的）。

    :returns: ``{'zero': onhold==0, 'weak': onhold<weak,
                 'reclear_d': 零仓→≤reopen_days 个交易日重开→再次零仓}``
    """
    oh = daily.MasPosOnhold.astype(float)
    zero = oh <= 1e-12
    weak_ = oh < weak
    # 「高位跳水」：前 5 个交易日内还 ≥0.5，今天跌破 weak —— 这才是有效的清仓信号
    # （单纯的 onhold 低会在一整段低仓区里天天触发，没有信息量）
    prev5max = oh.shift(1).rolling(5, min_periods=1).max()
    # 只在**跌入**弱仓那一天报一次（否则会在低仓区连报 5 天）
    drop = weak_ & ~weak_.shift(1, fill_value=False) & (prev5max >= 0.5)
    reclear = pd.Series(False, index=oh.index)
    runs, i = [], 0
    v = zero.to_numpy()
    while i < len(v):
        if not v[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(v) and v[j + 1]:
            j += 1
        runs.append((i, j))
        i = j + 1
    for (a0, a1), (b0, b1) in zip(runs, runs[1:]):
        if b0 - a1 - 1 <= reopen_days:
            reclear.iloc[b0] = True
    return {'drop': drop, 'zero': zero, 'weak': weak_, 'reclear_d': reclear}


def report_exits(eps, d, px: dict, daily: pd.DataFrame) -> None:
    """段内出现清仓信号后「提前离场」相对「拿到段末」的差别。"""
    if 'MasPosOnhold' not in daily:
        print('\n[段内清仓信号] 无 MasPosOnhold（2022-06 之前），跳过')
        return
    sig = derived_clear_signals(daily.reindex(d))
    print('\n[段内清仓信号] 首个信号日 vs 拿到段末（仅 2022-06 之后的段）')
    print('  段起点      段止        信号      类型      '
          + '  '.join(f'{n:>22s}' for n in px))
    for i, j in eps:
        a, b = d[i], d[j]
        if a < pd.Timestamp('2022-06-16'):
            continue
        seg_sig = {k: s.loc[a:b] for k, s in sig.items()}
        first, kind = None, ''
        # 只用「高位跳水 / 日频再清仓」—— `onhold==0` 与 `weak` 在一整段低仓区
        # 天天触发，拿它们当离场会把大牛段砍掉（见文档 6.6）
        for k in ('reclear_d', 'drop'):
            s = seg_sig[k].iloc[1:]          # 段首那天不算
            if s.any():
                cand = s[s].index[0]
                if first is None or cand < first:
                    first, kind = cand, k
        if first is None:
            print(f'  {a.date()}  {b.date()}  （无信号）')
            continue
        cells = []
        for name, close in px.items():
            s = close.loc[a:b].dropna()
            if len(s) < 2 or first not in s.index:
                cells.append('   NA')
                continue
            hold = 100 * (s.iloc[-1] / s.iloc[0] - 1)
            out = 100 * (s.loc[first] / s.iloc[0] - 1)
            cells.append(f'拿住{hold:+5.1f} 离场{out:+5.1f}')
        print(f'  {a.date()}  {b.date()}  {first.date()}  {kind:<8s}  '
              + '  '.join(cells))


def rule_stats(px: dict, sig: pd.Series, label: str) -> None:
    s = sig.fillna(False)
    if not s.any():
        print(f'  {label}: 无事件')
        return
    parts = [f'  {label:<26s} n={int(s.sum()):3d}']
    for name, close in px.items():
        seg = []
        for h in (20, 40, 60):
            v = (close.shift(-h) / close - 1)[s].dropna()
            seg.append(f'{h}d {100 * (v > 0).mean():5.1f}%/{100 * v.median():+5.1f}%'
                       if len(v) else f'{h}d NA')
        parts.append(f'{name}: ' + ' '.join(seg))
    print('  '.join(parts))


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--wek', default='output/mas_quad_pwrln_wek_2018_2026.csv')
    p.add_argument('--gap', type=int, default=10)
    p.add_argument('--min-len', type=int, default=10)
    p.add_argument('--csv', default='')
    a = p.parse_args(argv)

    w = load_weekly(a.wek)
    px = {name: load_index(code).reindex(w.index)
          for name, code in BENCHMARKS}
    daily = load_daily()
    med, mode = w.MasQuadPwrlnWekMed, w.MasQuadPwrlnWekMode
    eps, d = build_episodes(med, mode, gap=a.gap, min_len=a.min_len)
    tbl = build_table(eps, d, px, med, mode, daily)

    print(f'[episodes] {len(tbl)} 段（gap={a.gap}, min_len={a.min_len}），'
          f'{d[0].date()} ~ {d[-1].date()}')
    print(tbl.to_string(index=False))
    for name in px:
        v = tbl[name].dropna()
        if len(v):
            print(f'  {name}: 正收益 {int((v > 0).sum())}/{len(v)} = '
                  f'{100 * (v > 0).mean():.1f}%，中位 {v.median():+.1f}%')
    print(f'  中位交易日 {tbl["交易日"].median():.0f}')

    print('\n[日频信号叠加]（2022-06 起；基准=科创50 / 创业板）')
    dsig = pd.Series(index=d, dtype=float)
    for c in ('STAGE_MODE', 'MasPosOnhold', 'PosBottom', 'PosFull', 'PosFullOnset',
              'PosReClear', 'PosReClearStrict', 'PosDeepBottom'):
        if c in daily:
            dsig[c] = daily[c].reindex(d)
    enter, exit_ = stage_signals(daily.reindex(d))
    rule_stats(px, enter, '底带首日 STAGE=-0.5')
    rule_stats(px, exit_, '底带退出日（次日>-0.5）')
    if 'PosFullOnset' in dsig:
        rule_stats(px, dsig.PosFullOnset == 1, 'PosFullOnset（派生）')
    if 'PosReClear' in dsig:
        rule_stats(px, dsig.PosReClear == 1, 'PosReClear')
    rule_stats(px, (med >= 0) & (mode >= 0), '双正（逐日）')
    rule_stats(px, med <= -2.8, '周线深底（逐日）')

    report_exits(eps, d, px, daily)

    if a.csv:
        os.makedirs(os.path.dirname(a.csv) or '.', exist_ok=True)
        tbl.to_csv(a.csv, index=False, encoding='utf-8')
        print(f'\n[out] {a.csv}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
