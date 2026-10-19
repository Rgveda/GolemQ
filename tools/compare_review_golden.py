#!/usr/bin/env python3
# coding: utf-8
"""对比 review 金标准（4.4）与新实现输出，按列给容差与差异分类。

用法
====
    python tools/compare_review_golden.py --golden-dir GolemQ/test_cases/pipeline/fixtures \
        --new-dir /tmp/review_new --family day

目录约定：``<dir>/<family>/<code>.parquet``；键列：
* `day`：`date`（YYYY-MM-DD）
* `60min` / `15min` / `minor60`：`datetime`（优先）或 `time_stamp`

判定
====
* 整数/离散列（timing lag、信号）：**完全相等**或 `|diff| <= 1`（记 `near`）；
* 浮点列：`|a-b| <= atol + rtol*|b|`；
* 差异落在 `--dirty-dates` 日期（4.4 已知脏数据日）⇒ 归为 `old_data`；
* 列名在 `--known-design`（逗号分隔）⇒ 归为 `design`；
* 其余差异 ⇒ `unexplained`，进程**非零退出**（避免把「跑过了」读成「通过」）。

⚠️ 本工具只读；不连库、不写库。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

#: 4.4 已知脏数据日（复权换挡提前 / 分钟与日线不一致）。
DEFAULT_DIRTY_DATES = {'2024-10-24', '2026-06-01', '2026-09-30'}
#: 已知设计差异（czsc 版本 / 固定锚点 / 批次窗口）。
DEFAULT_DESIGN_COLUMNS = {
    'zDLagMajR', 'ZPLagMajR', 'zDLagWekR',
    'zDLagR', 'zPLagR', 'zDLagMinR', 'zPLagMinR',
    'm9tLagWekR',
    'MsLeadinx3BfMin', 'MsLeadinx5BfMin', '4QPwrlnLedInBfMin',
}


def _key(df: pd.DataFrame, family: str) -> pd.Series:
    if 'date' in df.columns:
        return pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
    if 'datetime' in df.columns:
        return pd.to_datetime(df['datetime'])
    if 'time_stamp' in df.columns:
        return pd.to_datetime(pd.to_numeric(df['time_stamp']), unit='s',
                              utc=True).dt.tz_convert('Asia/Shanghai').dt.tz_localize(None)
    raise KeyError(f'{family}: 找不到键列（date/datetime/time_stamp）')


def _compare_col(a: pd.Series, b: pd.Series, rtol: float, atol: float):
    """返回 (bad, near, maxdiff, kind)。"""
    a = pd.to_numeric(a, errors='coerce')
    b = pd.to_numeric(b, errors='coerce')
    m = a.notna() & b.notna()
    if not m.any():
        return 0, 0, None, 'empty'
    d = (a[m] - b[m]).abs()
    # 整数/离散：按 ±1 判 near
    if pd.api.types.is_integer_dtype(a) or pd.api.types.is_integer_dtype(b):
        bad = int((d > 1e-9).sum())
        near = int(((d > 1e-9) & (d <= 1)).sum())
        return bad, near, float(d.max()), 'int'
    bad = int((d > atol + rtol * b[m].abs()).sum())
    return bad, 0, float(d.max()), 'float'


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--golden-dir', required=True)
    ap.add_argument('--new-dir', required=True)
    ap.add_argument('--family', default='day')
    ap.add_argument('--rtol', type=float, default=1e-5)
    ap.add_argument('--atol', type=float, default=1e-6)
    ap.add_argument('--dirty-dates', default=','.join(sorted(DEFAULT_DIRTY_DATES)))
    ap.add_argument('--known-design', default=','.join(sorted(DEFAULT_DESIGN_COLUMNS)))
    args = ap.parse_args(argv)

    dirty = {d.strip() for d in args.dirty_dates.split(',') if d.strip()}
    design = {d.strip() for d in args.known_design.split(',') if d.strip()}
    gold = Path(args.golden_dir) / args.family
    new = Path(args.new_dir) / args.family
    if not gold.is_dir():
        print(f'[compare] 没有金标准目录: {gold}', file=sys.stderr)
        return 2

    unexplained_total = 0
    for gp in sorted(gold.glob('*.parquet')):
        npth = new / gp.name
        if not npth.exists():
            print(f'[compare] {args.family}/{gp.stem}: 缺新输出 {npth}')
            unexplained_total += 1
            continue
        g = pd.read_parquet(gp)
        n = pd.read_parquet(npth)
        g['_k'] = _key(g, args.family)
        n['_k'] = _key(n, args.family)
        common = sorted(set(g['_k']) & set(n['_k']))
        mg = g.set_index('_k').loc[common]
        mn = n.set_index('_k').loc[common]
        cols = [c for c in mg.columns if c in mn.columns and
                c not in ('code', 'date', 'datetime', 'time_stamp',
                          'date_stamp', 'created_at')]
        print(f'== {args.family}/{gp.stem}: rows={len(common)} cols={len(cols)}')
        for col in cols:
            bad, near, mx, kind = _compare_col(mg[col], mn[col], args.rtol, args.atol)
            if bad == 0:
                continue
            # 分类：脏日期 / 设计列 / 未解释
            bad_dates = set()
            if kind == 'int':
                a = pd.to_numeric(mg[col], errors='coerce')
                b = pd.to_numeric(mn[col], errors='coerce')
                bad_dates = set(pd.Index(common)[((a - b).abs() > 1e-9).to_numpy()])
            if col in design:
                tag = 'design'
            elif bad_dates and all(str(d)[:10] in dirty for d in bad_dates):
                tag = 'old_data'
            else:
                tag = 'unexplained'
                unexplained_total += 1
            print(f'    {col}: bad={bad} near(±1)={near} max={mx} -> {tag}')
    print(f'\n[compare] unexplained columns: {unexplained_total}')
    return 0 if unexplained_total == 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())
