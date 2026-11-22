# coding: utf-8
"""Massive 择时报告：仓位事件 / STAGE_MODE 底带 / 周线 Powerline 共振。

对应文档 `docs/massive_timing.md`。只做**只读统计**，不写库。

⚠️ 基准指数用 **科创50（000688）+ 创业板指（399006）**，不用沪深300
（用户 2026-11-22 指定：模型的成分股偏中小盘/成长，宽基基准会把信号稀释掉）。
科创50 的行情从 **2019-12-31** 起，更早的日期在它那一列是 NaN。

数据来源
========
* `stock_massive_reality_60min_models` 的 **15:00** 行（全市场横截面，一日一行，
  2022-06-16 起）：`MasPosOnhold` / `PosFull` / `PosReClear` / `PosBottom` /
  `PosDeepBottom` / `STAGE_MODE` / `StageModeStreak`；
* `output/mas_quad_pwrln_wek_2018_2026.csv`：周线 `4QPwrln` 的截面中位数 / 众数
  （2018-01-02 起，D86 口径，只算成熟码）；
* `index_day` 的 000688 / 399006 收盘 —— 只用来算**事后**涨跌幅。

用法::

    python tools/massive_timing_report.py
    python tools/massive_timing_report.py --csv output/massive_timing_daily.csv
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

#: 基准指数（顺序即打印顺序）。
#: 基准指数（用户 2026-11-22 指定看这四条）。
#: ⚠️ 中证2000（932000）**库里一行都没有**，用中证1000（000852）替代；
#: 科创200（000699）只有 2022-12-30 起的数据，涉及它的 n 会明显偏小。
BENCHMARKS = (('沪深300', '000300'), ('科创200', '000699'),
              ('创业板', '399006'), ('中证1000', '000852'))
FWD = (5, 10, 20, 60)
FLAG_COLS = ('PosFullOnset', 'PosReClear', 'PosReClearStrict',
             'PosBottom', 'PosDeepBottom')

#: 用户口述的「两月牛」起止点（校准用）。
USER_EPISODES = {
    '#1 起': '2022-04-26', '#1 确认': '2022-04-28', '#1 离场': '2022-08-24',
    '#2 信号': '2022-10-14', '#2 起': '2022-10-31',
    '#3 起': '2022-12-23',
    '#4 起': '2023-06-13', '#4 离场': '2023-07-24',
    '#5 起': '2023-08-31', '#5 离场': '2023-09-27',
    '#6 起': '2023-10-31', '#6 离场': '2023-11-30',
    '#7 起': '2024-02-06', '#7 确认': '2024-02-07', '#7 离场': '2024-05-20',
    '#8 起': '2024-04-23', '#8 确认': '2024-05-20',
    '#9 起': '2024-09-18', '#9 确认': '2024-09-24', '#9 判断': '2024-09-27',
    '半波a': '2024-10-14', '半波b': '2025-01-14',
    '半波c': '2025-04-09', '半波d': '2025-06-23',
}


def load_index(code: str, start: str = '2017-12-01') -> pd.Series:
    """指数日线收盘。

    ⚠️ 基准序列**必须覆盖整个考察区间**。曾经把沪深300 取成 massive 帧里那份
    （只有 2022-06 起），早期日期前向收益算出 NaN 又被 `dropna()` 静默剔除，
    直接造出「深底 60d 胜率 57.3%」的假优势（全样本其实 48.4%）。
    """
    from GolemQ import get_active_market
    db = get_active_market().get_database()
    idx = pd.DataFrame(list(db.index_day.find(
        {'code': code, 'date': {'$gte': start}},
        {'_id': 0, 'date': 1, 'close': 1}).sort('date', 1)))
    idx['date'] = pd.to_datetime(idx['date'])
    return idx.set_index('date')['close'].astype(float)


def load_massive() -> pd.DataFrame:
    from GolemQ import get_active_market
    db = get_active_market().get_database()
    rows = list(db.stock_massive_reality_60min_models.find(
        {'datetime': {'$regex': ' 15:00:00$'}}, {'_id': 0}).sort('datetime', 1))
    df = pd.DataFrame(rows)
    df['date'] = pd.to_datetime(df['datetime']).dt.normalize()
    df = df.set_index('date').sort_index()
    if 'PosFull' in df.columns:
        # ⚠️ 库里 `PosFullOnset` 只落了 9 次里的 3 次（历史行没回填）。
        # 保留原始那份叫 `PosFullOnset_stored`，`PosFullOnset` 用派生值。
        if 'PosFullOnset' in df.columns:
            df['PosFullOnset_stored'] = df['PosFullOnset'].astype(float)
        df['PosFullOnset'] = ((df['PosFull'] == 1)
                              & (df['PosFull'].shift(1) != 1)).astype(np.int8)
    return df


def load_wek(path: str) -> pd.DataFrame:
    w = pd.read_csv(path, parse_dates=['date']).set_index('date').sort_index()
    # float32 尾巴会让 `>= 0` / `<= -2.8` 判错（D86 的坑），网格是 0.1
    for c in ('MasQuadPwrlnWekMed', 'MasQuadPwrlnWekMode'):
        w[c] = w[c].round(1)
    return w


def attach(df: pd.DataFrame, px: dict) -> pd.DataFrame:
    """挂上基准收盘与前向收益（列名 ``<基准>_f20`` 之类）。"""
    out = df.copy()
    for name, _ in BENCHMARKS:
        out[name] = px[_code(name)].reindex(out.index)
        for h in FWD:
            out[f'{name}_f{h}'] = out[name].shift(-h) / out[name] - 1.0
    return out


def _code(name: str) -> str:
    return dict(BENCHMARKS)[name]


def runs(mask: pd.Series):
    """连续 True 段 → ``[(start, end, n_days), ...]``。"""
    v = np.asarray(mask.fillna(False), dtype=bool)
    d = mask.index
    out, i = [], 0
    while i < len(v):
        if not v[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(v) and v[j + 1]:
            j += 1
        out.append((d[i], d[j], j - i + 1))
        i = j + 1
    return out


def stat_line(df: pd.DataFrame, mask, label: str,
              horizons=(10, 20, 60)) -> None:
    s = df[mask.fillna(False)]
    parts = [f'  {label:<26s} n={len(s):4d}']
    for name, _ in BENCHMARKS:
        seg = []
        for h in horizons:
            v = s[f'{name}_f{h}'].dropna()
            seg.append(f'{h}d {100 * (v > 0).mean():5.1f}%/{100 * v.median():+5.1f}%'
                       f'(n={len(v)})' if len(v) else f'{h}d NA')
        parts.append(f'{name}: ' + ' '.join(seg))
    print('  '.join(parts))


def report_events(df: pd.DataFrame) -> None:
    print('\n## 一、仓位事件（2022-06-16 起）')
    for col in FLAG_COLS:
        s = df[df[col] == 1]
        print(f'\n[{col}] {len(s)} 次')
        for d, r in s.iterrows():
            fwd = '  '.join(
                f'{n} 20d ' + ('NA' if pd.isna(r[f'{n}_f20'])
                               else f'{100 * r[f"{n}_f20"]:+5.1f}%')
                for n, _ in BENCHMARKS)
            print(f'  {d.date()}  onhold {r.MasPosOnhold:5.3f}  '
                  f'stage {r.STAGE_MODE:4.1f}  {fwd}')


def report_stage(df: pd.DataFrame, wek: pd.DataFrame) -> None:
    print('\n## 二、STAGE_MODE == -0.5 的事件区间')
    rs = runs(df.STAGE_MODE <= -0.4999)
    print(f'共 {len(rs)} 段，合计 {sum(n for _, _, n in rs)} 个交易日')
    for a, b, n in rs:
        r = df.loc[a]
        sub = wek.loc[:a] if len(wek) else pd.DataFrame()
        wm = (f'WekMed {sub.MasQuadPwrlnWekMed.iloc[-1]:5.1f}'
              if len(sub) else '')
        parts = []
        for name, _ in BENCHMARKS:
            v = r[f'{name}_f20']
            parts.append(f'{name} 20d ' + ('NA' if pd.isna(v) else f'{100 * v:+5.1f}%'))
        print(f'  {a.date()} ~ {b.date()}  {n}d  {wm}  ' + '  '.join(parts))
    for lbl, mask in (('底带首日', (df.STAGE_MODE <= -0.5)
                       & (df.STAGE_MODE.shift(1) > -0.5)),
                      ('底带退出（次日>-0.5）', (df.STAGE_MODE > -0.5)
                       & (df.STAGE_MODE.shift(1) <= -0.5)),
                      ('PosFullOnset（派生）', df.PosFullOnset == 1),
                      ('PosReClear', df.PosReClear == 1)):
        stat_line(df, mask, lbl)


def report_resonance(wek: pd.DataFrame) -> None:
    print('\n## 三、周线 Powerline 共振（2018-01-02 ~ 2026-11-20）')
    med, mode = wek.MasQuadPwrlnWekMed, wek.MasQuadPwrlnWekMode
    stat_line(wek, pd.Series(True, index=wek.index), 'ALL（基准）')
    stat_line(wek, med >= 0, 'Med>=0')
    stat_line(wek, (med >= 0) & (mode >= 0), 'Med>=0 & Mode>=0')
    stat_line(wek, med <= -2.8, 'Med<=-2.8')
    stat_line(wek, med >= 2.2, 'Med>=2.2')
    print('  — 转折型（只数第一次跨档）')
    stat_line(wek, (med <= -2.8) & (med.shift(1) > -2.8), '跌破 -2.8 首次')
    stat_line(wek, (med >= 0.2) & (med.shift(1) < 0.2), '跨上 0.2')
    stat_line(wek, (med >= 1.2) & (med.shift(1) < 1.2), '跨上 1.2')
    stat_line(wek, (med >= 2.2) & (med.shift(1) < 2.2), '跨上 2.2')


def report_user(df: pd.DataFrame, wek: pd.DataFrame) -> None:
    print('\n## 四、用户口述「两月牛」逐点校准')
    head = '  标签     日期        onhold stage  WekMed WekMode'
    for name, _ in BENCHMARKS:
        head += f'   {name} 20d/60d'
    print(head)
    for k, d in USER_EPISODES.items():
        t = pd.Timestamp(d)
        sub = df.loc[:t]
        if len(sub) == 0:
            print(f'  {k:<8s} {d}  早于数据起点（2022-06-16）')
            continue
        r = sub.iloc[-1]
        wk = wek.loc[:t] if len(wek) else pd.DataFrame()
        wm = wk.MasQuadPwrlnWekMed.iloc[-1] if len(wk) else np.nan
        wo = wk.MasQuadPwrlnWekMode.iloc[-1] if len(wk) else np.nan

        def _cell(name):
            a, b = r.get(f'{name}_f20'), r.get(f'{name}_f60')
            ca = f'{100 * a:+6.1f}%' if pd.notna(a) else '   NA  '
            cb = f'{100 * b:+6.1f}%' if pd.notna(b) else '   NA  '
            return f'{ca}/{cb}'

        line = (f'  {k:<8s} {d}  {r.MasPosOnhold:5.3f} {r.STAGE_MODE:4.1f}  '
                f'{wm:6.1f} {wo:6.1f}')
        for name, _ in BENCHMARKS:
            line += f'  {_cell(name)}'
        print(line)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--wek', default='output/mas_quad_pwrln_wek_2018_2026.csv')
    p.add_argument('--csv', default='', help='另存每日指标 + 前向收益')
    a = p.parse_args(argv)

    px = {code: load_index(code) for _, code in BENCHMARKS}
    wek_raw = load_wek(a.wek) if os.path.exists(a.wek) else None
    wek = attach(wek_raw, px) if wek_raw is not None else pd.DataFrame()
    df = attach(load_massive(), px)

    print(f'[massive] {len(df)} 个交易日 {df.index.min().date()} ~ '
          f'{df.index.max().date()}')
    if wek_raw is not None:
        print(f'[wek] {len(wek_raw)} 行 {wek_raw.index.min().date()} ~ '
              f'{wek_raw.index.max().date()}')
    for name, code in BENCHMARKS:
        s = px[code]
        print(f'[bench] {name}({code}) {s.index.min().date()} ~ '
              f'{s.index.max().date()}，{len(s)} 行')

    report_events(df)
    report_stage(df, wek)
    if len(wek):
        report_resonance(wek)
    report_user(df, wek)

    if a.csv:
        # 每日表：仓位三族 + 全部分档 + **七个事件标记** + 派生清仓信号 + 前向收益
        cols = ['mberCnt', 'PosBsln', 'PosPad', 'PosNegBsln', 'MasPosOnhold',
                'STAGE_MODE', 'StageModeStreak',
                # —— 事件标记（与文档 1.3 同名列）——
                'PosFull', 'PosFullOnset', 'PosFullOnset_stored',
                'PosReClear', 'PosReClearStrict',
                'PosBottom', 'PosDeepBottom', 'PosProbe']
        out = df[[c for c in cols if c in df.columns]].copy()
        if len(wek_raw):
            for c in ('MasQuadPwrlnWekMed', 'MasQuadPwrlnWekMode'):
                out[c] = wek_raw[c].reindex(out.index)
        # 派生清仓信号（文档 6.6）：zero / weak / drop / reclear_d
        from masquad_bull_episodes import derived_clear_signals
        for k, v in derived_clear_signals(df).items():
            out[f'd_{k}'] = v.astype(np.int8).reindex(out.index)
        for c in df.columns:
            if '_f' in c:                     # 四个基准的前向收益
                out[c] = df[c]
        out.index.name = 'date'
        out.to_csv(a.csv, encoding='utf-8')
        print(f'[out] {a.csv}  cols={len(out.columns)}  rows={len(out)}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
