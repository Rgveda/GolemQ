# coding:utf-8
#
# The MIT License (MIT)
#
# Copyright (c) 2020-2025 azai/GolemQ(uant)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#

"""只读覆盖核对：**8.3 的 K 线有没有洞**（`--save tdx --save-coverage`）。

它回答的是增量**答不了**的问题：增量只补水位之后的洞，水位**之前**的历史空洞
（某只票某段整段缺失）它看不见。本模块逐 code 比「库里存在的交易日」与
「交易日历上应有的交易日」，只报**有缺口**的 code。

⚠️ 三条**实测**成本约束（`PITFALLS.md` P15），违反就会跑成分钟级：
* 时序集合上**绝不**无时间窗 `distinct('code')`/`aggregate`（实测 `stock_day` >120s 不返回、
  `stock_1min` 近 5 交易日窗 49.9s）。
* 每次查询**必须带 `code`**（metaField 剪枝）—— 实测 32–75 ms/code。
* **不逐 (code, day) 数行数**（`count_documents` 0.42s/次 × 百万组 = 不可行）；
  「存在但缺根」只在**最近一个交易日**用一次窄窗 `$group` 看（见 `roots_check`）。

⚠️ **扣停牌**是必须的：停牌日的 bar 已被 `GQ_purge_suspended` 移进 `stock_day_removed`
（**刻意留的空洞**），不扣就满屏假缺口。停牌名单取 :func:`maintenance.GQ_suspension_dates`
（实测 2.0s 量级）。
"""
from __future__ import annotations

from datetime import datetime

from .kline_doc import BARS_PER_DAY, trade_days_between
from .kline_save import (
    FREQUENCIES,
    TARGETS,
    _db,
    target_collection_name,
    universe,
)

__all__ = [
    'expected_dates',
    'format_kline_status',
    'kline_status',
    'roots_check',
]

#: 报告里每个 code 最多列几个缺口的日期样例。
_GAP_SAMPLES = 3


def expected_dates(first, last, suspended=()):
    """``[first, last]`` 内**应有 bar** 的交易日（含端点，扣掉 `suspended`）。

    >>> expected_dates('2026-10-08', '2026-10-09')
    ['2026-10-08', '2026-10-09']
    >>> expected_dates('2026-10-08', '2026-10-09', suspended=('2026-10-09',))
    ['2026-10-08']
    """
    skip = set(suspended)
    return [d for d in trade_days_between(first, last) if d not in skip]


def _last_trading_days(n):
    """最近 `n` 个交易日（升序）。"""
    days = trade_days_between('1990-01-01', datetime.now().strftime('%Y-%m-%d'))
    return days[-int(n):] if n else days


def kline_status(targets=None, frequencies=('day',), sample=None, window_days=None,
                 with_adj=False, verbose=True):
    """逐个集合核对覆盖率。

    ⚠️ **默认只核对日线**（`frequencies=('day',)`）：每集合约 3.5 min（5,574 code × 35ms）。
    分钟线的代价高一个量级（每个 code 要 distinct 一遍），必须**显式指定**，且
    建议配 `window_days`。

    :param window_days: 只核对**最近 N 个交易日**；``None`` = 该 code 的全历史。
        ⚠️ 分钟线用 `None` 会很慢（每个 code 要把全历史 distinct 一遍）。
    :param sample: 只随机抽 N 个 code（试跑用）；None = 全部
    :param with_adj: 额外核对 `stock_adj` 是否落后于 `stock_day`（逐 code 取两边末日期）
    :returns: ``{集合名: {'codes', 'checked', 'gap_codes', 'gaps': [...], 'universe': n}}``
    """
    from .maintenance import GQ_suspension_dates

    db = _db()
    targets = tuple(targets) if targets else TARGETS
    frequencies = tuple(frequencies) if frequencies else ('day',)
    # 停牌只对股票有意义（指数/ETF 不做停牌归档）
    suspended_pairs = GQ_suspension_dates(verbose=verbose) if 'stock' in targets else set()
    by_code: dict = {}
    for code, date in suspended_pairs:
        by_code.setdefault(code, set()).add(date)

    report = {}
    for target in targets:
        codes = universe(target, db=db)
        if sample:
            import random
            codes = random.sample(codes, min(int(sample), len(codes)))
        for frequency in frequencies:
            name = target_collection_name(target, frequency)
            coll = db[name]
            entry = {'universe': len(codes), 'codes': len(codes), 'checked': 0,
                     'gap_codes': 0, 'gaps': [], 'empty_codes': 0, 'last_date': None}
            report[name] = entry
            win = _last_trading_days(window_days)[0] if window_days else None
            for code in codes:
                q = {'code': code}
                if win:
                    q['date'] = {'$gte': win}
                dates = set(coll.distinct('date', q))
                if not dates:
                    entry['empty_codes'] += 1
                    continue
                first, last = min(dates), max(dates)
                entry['last_date'] = max(entry['last_date'] or last, last)
                lo = max(first, win) if win else first
                want = expected_dates(lo, last, by_code.get(code, ()))
                gap = sorted(set(want) - dates)
                entry['checked'] += 1
                if gap:
                    entry['gap_codes'] += 1
                    entry['gaps'].append({
                        'code': code, 'first': first, 'last': last,
                        'n_gap': len(gap), 'sample': gap[:_GAP_SAMPLES]})
            if verbose:
                print('[coverage] {}: 核 {} code（空 {}），有缺口 {} 个'.format(
                    name, entry['checked'], entry['empty_codes'], entry['gap_codes']))

    if with_adj:
        report['stock_adj'] = {'lagging': _adj_lagging(db, verbose=verbose)}
    return report


def _adj_lagging(db, verbose=True):
    """`stock_adj` 末日期**早于** `stock_day` 末日期的 code 清单。

    两张表都是「一码一份逐日序列」，所以「adj 落后」= 它的末日期比日线末日期旧
    （新除权事件没重算因子 → 该票的前复权会静默错价）。
    """
    out = []
    for code in universe('stock', db=db):
        adj = db['stock_adj'].find_one({'code': code}, {'_id': 0, 'date': 1},
                                       sort=[('ts', -1)])
        day = db['stock_day'].find_one({'code': code}, {'_id': 0, 'date': 1},
                                       sort=[('ts', -1)])
        if adj and day and adj.get('date') and day.get('date') and adj['date'] < day['date']:
            out.append({'code': code, 'adj_last': adj['date'], 'day_last': day['date']})
    if verbose:
        print('[coverage] stock_adj 落后于 stock_day 的 code：{} 个'.format(len(out)))
    return out


def roots_check(name='stock_1min', day=None, expected=None, verbose=True):
    """**「存在但缺根」**：只看某个交易日，逐 code 数行数，列出不等于应有根数的 code。

    这是唯一便宜的「缺根」检查方式（见模块 docstring 的成本约束）：
    单日 1min ≈ 5,574 × 240 ≈ 130 万文档，秒级；**不要**对多个交易日循环做。

    :param day: 交易日 `'YYYY-MM-DD'`；None = 最近一个交易日
    :param expected: 应有根数；None = 按集合名从 :data:`BARS_PER_DAY` 推
    """
    db = _db()
    day = day or _last_trading_days(1)[0]
    if expected is None:
        freq = name.split('_', 1)[1]
        expected = BARS_PER_DAY.get(freq)
    from .kline83 import bj_date
    lo = int(bj_date('{} 00:00:00'.format(day)).timestamp())
    cur = db[name].aggregate([
        {'$match': {'date_stamp': lo}},
        {'$group': {'_id': '$code', 'n': {'$sum': 1}}},
    ], allowDiskUse=True)
    rows = list(cur)
    bad = [{'code': r['_id'], 'n': r['n']} for r in rows if r['n'] != expected]
    if verbose:
        print('[coverage] {} {}：当天有数据的 code {} 个，应有 {} 根，不齐的 {} 个'.format(
            name, day, len(rows), expected, len(bad)))
    # ⚠️ `codes_seen` 必须一起返回：当天若**一个 code 都没有**，`bad` 为空是
    # **空洞的通过**（不是"都齐了"）—— 调用方要能看出这一点。
    return {'day': day, 'expected': expected, 'codes_seen': len(rows), 'bad': bad}


def format_kline_status(report):
    """把 `kline_status()` 的结果渲染成可读文本。"""
    lines = []
    for name, entry in report.items():
        if name == 'stock_adj':
            lag = entry.get('lagging', [])
            lines.append('[adj] stock_adj 落后 {} 个 code{}'.format(
                len(lag), ('（例：{}）'.format(lag[:3]) if lag else '')))
            continue
        lines.append('[{}] 核 {} / 空 {} / 有缺口 {}（末日 {}）'.format(
            name, entry.get('checked'), entry.get('empty_codes'),
            entry.get('gap_codes'), entry.get('last_date')))
        for g in entry.get('gaps', [])[:5]:
            lines.append('    {} {}~{} 缺 {} 天 例 {}'.format(
                g['code'], g['first'], g['last'], g['n_gap'], g['sample']))
    lines.append('⚠️ 只证明这些**日期存在**，不证明根数正确 —— 缺根看 roots_check()')
    return '\n'.join(lines)
