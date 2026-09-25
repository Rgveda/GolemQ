# coding:utf-8
"""转储 `is_stock_cn` 的**全量行为基线** —— 重编排前的回归依据。

为什么要全量而不是抽样
====================
`is_stock_cn` 是**纯**函数（只做字符串号段判断，不碰 DB），所以扫完
``000000``~``999999`` 一百万条是**可行且便宜**的（实测约十余秒）。
有了这份基线，重编排后的 diff 就是**可证伪**的：
差异必须**恰好等于**我们有意改的那几段，多一条都是回归。

输出：``tools/_is_stock_cn_baseline.tsv``（``code<TAB>market_type<TAB>exchange``）
    只记 market_type / exchange 两项 —— 描述串**不参与**路由，且本次有意重写，
    把它纳入对比只会产生噪声。

用法::

    python tools/dump_is_stock_cn_baseline.py            # 写基线
    python tools/dump_is_stock_cn_baseline.py --diff     # 与现有基线比对
"""
from __future__ import annotations

import contextlib
import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from GolemQ.markets.StockCN.symbol import is_stock_cn  # noqa: E402

BASELINE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        '_is_stock_cn_baseline.tsv')


def sweep():
    """`{code: (market_type, exchange)}` —— 全部六位数字代码。

    ``is_stock_cn`` 对未知代码会 ``print`` 一行，一百万次会把终端淹掉 ——
    整段重定向到 StringIO 丢弃，但**不吞异常**。
    """
    out = {}
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink):
        for n in range(1000000):
            code = f'{n:06d}'
            _, market_type, exchange, _desc = is_stock_cn(code)
            out[code] = (market_type, exchange)
    return out


def load(path=BASELINE):
    with open(path, encoding='utf-8') as fh:
        out = {}
        for line in fh:
            code, mt, ex = line.rstrip('\n').split('\t')
            out[code] = (mt or '', ex or '')
    return out


def write(data, path=BASELINE):
    with open(path, 'w', encoding='utf-8') as fh:
        for code in sorted(data):
            mt, ex = data[code]
            fh.write(f'{code}\t{mt or ""}\t{ex or ""}\n')
    print(f'[baseline] 写入 {len(data)} 条 → {path}')


def _norm(pair):
    """``None`` 与 ``''`` 归一 —— 基线落盘时 ``None`` 被写成空串，
    不归一的话每一条「未知」都会伪装成差异（第一次跑就骗了我 81.7 万条）。"""
    return tuple('' if v is None else v for v in pair)


def diff(before, after):
    """按「旧值 → 新值」归并差异，**不逐条打印**一百万行。"""
    buckets = {}
    for code in sorted(before):
        b, a = _norm(before[code]), _norm(after.get(code, (None, None)))
        if b != a:
            buckets.setdefault((b, a), []).append(code)
    return buckets


def _summarize(codes):
    """把一批代码压成**连续号段区间**，便于人读（逐段列会刷屏几十行）。"""
    pref = sorted({c[:3] for c in codes})
    runs, start, prev = [], pref[0], pref[0]
    for p in pref[1:] + [None]:
        if p is None or int(p) != int(prev) + 1:
            runs.append(start if start == prev else f'{start}~{prev}')
            start = p
        prev = p if p is not None else prev
    if len(runs) > 10:
        runs = runs[:5] + [f'…({len(runs)} 段)'] + runs[-3:]
    return ', '.join(runs)


def main():
    if '--diff' in sys.argv:
        before = load()
        print(f'[baseline] 读入基线 {len(before)} 条')
        after = sweep()
        buckets = diff(before, after)
        if not buckets:
            print('[baseline] ✅ 与基线**逐条相同**')
            return 0
        total = 0
        for (b, a), codes in sorted(buckets.items(), key=lambda kv: -len(kv[1])):
            total += len(codes)
            print(f'\n[baseline] {b} → {a}  共 {len(codes)} 条')
            print(f'           {_summarize(codes)}')
        print(f'\n[baseline] 差异合计 {total} 条 / {len(buckets)} 类')
        return 1

    write(sweep())
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
