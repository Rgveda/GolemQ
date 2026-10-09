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

"""pytdx bar / xdxr → 8.3 时序文档的**纯函数**核心（写侧契约）。

与 :mod:`kline83` 的关系：那是 8.3 的**读**（按 `(code, ts)` 查 + MultiIndex），
本模块是同一份契约的**写**。规格书是 `MONGODB83.md` 的「五、数据契约」+ 存量实测。

本模块**不碰 DB、不碰网络** —— 所以能进 doctest（见 `test_cases/test_doctests.py`）。
取数在 :mod:`markets.StockCN.datasource.pytdx_kline`，落库在
:func:`GolemQ.datasource.writer.save_bar_chunk`，编排在 :mod:`markets.StockCN.kline_save`。

三条**实测**口径（逐条对拍过存量，改之前先看证据）
==============================================

**① 分钟 bar 的标签与存量**完全同标签**（`09:31 … 15:00`），不要做任何偏移。**
（一度按「盘口边界」推断出 +1 分钟，**对拍证伪**：拿 2026-09-30 全天 240 根
逐根比，`close` 240 根里 239 根逐值相同，只有存量缺最后一根 `15:00`。）

**② `vol` 的单位各家不同，必须按市场/频率换算**（否则静默差 100 倍）：

| 集合 | 换算 | 实测证据 |
|:--|:--|:--|
| `stock_day` / `etf_day` / `fund_day` | **不换算** | 31239 = 31239、6689451 = 6689451 |
| `stock_*min` / `etf_*min` / `fund_*min` | **÷100**（股 → 手） | 比中位数 = 100；÷100 后逐值相同 |
| `index_day` | **×100** | 4,385,300 → 438,530,412 |
| `index_*min` | **不可复现 → 不写**（见 `kline_save`） | 比值 14–18 **非常数**，且 close 精度来源不同（存量 3 位小数 / pytdx 2 位）|

**③ `amount` 的 `5.877471754e-39` 不是口径，是哨兵值。** 它出现在 **pytdx 的原包里**
（实测 `get_security_bars` 的零成交 bar 直接给这个数），存量里只是照抄：

| 集合 | 该 code 的哨兵行 / 总行 | 样例 |
|:--|--:|:--|
| `stock_1min` 600519 | 2,637 / 460,074（0.57%）| 2018-11-08 14:58 |
| `index_1min` 000001 | 138,375 / 460,070 | 2018-11-08 09:31（**vol 也是哨兵**）|
| `stock_day` 600519 | **0** / 6,010 | 全市场日线都不带 |

所以 `amount` **直传**，而 `vol` 过一个 `int()` —— 哨兵取整自然变 **0**（存量零成交行也是 0）。
**不要**反过来去"修"成哨兵，也不要把零成交行丢掉（`PITFALLS` P8b：0 成交有三种成因）。
"""
from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd

import polars as pl

from .constants import TRADE_DATE_SSE
# 时区换算的唯一入口 —— 绝不要在本模块自写第二份（`kline83.bj_date` 是共享的）
from .kline83 import bj_date

__all__ = [
    'AMOUNT_KEYS',
    'BARS_PER_DAY',
    'alive_threshold',
    'DAY_START',
    'last_closed_session_bar',
    'SESSION_CLOSE',
    'MINUTE_LABEL_OFFSET',
    'MIN_START_DEFAULT',
    'PAGE',
    'SESSION_READY_AT',
    'TDX_CATEGORY',
    'bar_date',
    'bar_doc',
    'bar_stamp',
    'bar_datetime',
    'normalize_vol',
    'normalize_amount',
    'bars_needed',
    'dedup_docs',
    'page_offsets',
    'trade_days_between',
    'xdxr_adj_events_changed',
    'xdxr_doc',
]

#: 各频率**一天的理论根数**。A 股一天 240 分钟（09:30–11:30 + 13:00–15:00），
#: 实测 8.3 `stock_1min` 的 2025-10-10 正好 240 根（09:31 … 15:00）。
BARS_PER_DAY = {'1min': 240, '5min': 48, '15min': 16, '30min': 8, '60min': 4}

#: 频率 → pytdx `get_security_bars` 的 `category`（实测取值，见 pytdx `params.py`）。
TDX_CATEGORY = {'day': 9, '1min': 8, '5min': 0, '15min': 1, '30min': 2, '60min': 3}

#: pytdx 单次请求上限（`params.MAX_KLINE_COUNT`，实测）。
PAGE = 800

#: 库里查不到任何 bar 时的起点：日线沿用老树 `save_x_func` 的 1990-01-01，
#: 分钟沿用老树 `--save-qmt-start` 的 2015-01-01（1min 首灌代价见 kline_save 的 help）。
DAY_START = '1990-01-01'
MIN_START_DEFAULT = '2015-01-01'

#: pytdx 的分钟标签与 8.3 存量**同标签**（实测逐根对拍）—— 偏移恒为 0。
#: 保留这个常量是为了让"有没有偏移"这件事在代码里显式可见、可被一行改掉。
MINUTE_LABEL_OFFSET = timedelta(0)

#: 时间序列集合里**不该出现**的键。`type` 在迁移时被剥掉（8.3 全部主集合都没有它，
#: 频率靠集合名区分）—— 写回去会让读侧多出一列、且与存量不一致。
FORBIDDEN_KEYS = ('type',)

#: **指数分钟**的 `vol` 换算系数。实测存量那批与 pytdx **不同源**（比值 14–18 且
#: 非常数、close 精度也不同），无法反推口径，故**照 pytdx 原值写**（系数 = 1）。
#: 将来若确认了权威口径，只改这一处即可；不要散落在调用点里。
INDEX_MIN_VOL_SCALE = 1

#: `stock_xdxr` 里与「除权」有关、会改变前复权因子的字段（其余字段变了不该触发重算）。
XDXR_ADJ_KEYS = ('date', 'fenhong', 'peigu', 'peigujia', 'songzhuangu')

AMOUNT_KEYS = ('amount',)

#: 开盘时刻（北京）。**分钟** bar 的 `ts` 口径是「那一刻」（实测：2025-10-10 第一根 09:31）。
SESSION_OPEN = '09:30:00'

#: 收盘时刻（北京）。**分钟** bar 里最后一根的 `ts` 就是它（实测 15:00）。
SESSION_CLOSE = '15:00:00'

#: **日线** bar 的 `ts` 口径是该日的**北京零点**（实测：`stock_day` 里 `date='2026-10-09'`
#: 的那根 `ts` = 北京 2026-10-09 00:00 —— 与 `time_stamp` 同值）。
DAY_BAR_AT = '00:00:00'

#: 各频率**要等到什么时刻**才认「今天这一批 bar 已经该有了」。
#:
#: ⚠️ **这个差异是承重的**（2026-10-09 修）：
#: * **日线要等到 15:00 之后** —— 盘中**不写当天日线**（`PITFALLS.md` P20：盘中那根是
#:   "当日累计"，写进去是错数据）。若盘中就拿「今天」当基准，:func:`alive_threshold`
#:   会给出「今天 00:00」，而库里**根本不该**有那根 ⇒ 探针**永远探不到前沿** ⇒
#:   **盘中每一轮都白扫全市场**（5,575 条连接）。
#: * **分钟只要开市**（09:30）就该有 —— 第一根落在 09:30/09:31。
#:
#: 收盘后两者都取「今天」，非交易日两者都退到上一交易日。
SESSION_READY_AT = {'day': '15:00:00'}


def bar_date(bar):
    """pytdx bar 的 ``'YYYY-MM-DD'``。

    **由 year/month/day 整数拼**，不解析 ``bar['datetime']`` 字符串 ——
    那个串的格式在不同 category 下并不一致，而整数三个字段是稳定的。

    >>> bar_date({'year': 2024, 'month': 1, 'day': 2})
    '2024-01-02'
    """
    return '{:04d}-{:02d}-{:02d}'.format(
        int(bar['year']), int(bar['month']), int(bar['day']))


def bar_stamp(bar, frequency):
    """pytdx bar → **北京口径的 unix 秒**（8.3 的 `time_stamp`）。

    * ``day``：**归零到当日 00:00**。存量实测日线 ``time_stamp == date_stamp``
      （最后一根 600519 的 ``ts`` 是 `16:00Z` = 次日北京 00:00 的反面，即当日零点）——
      不归零会得到 15:00，与存量差 15 小时，同一个 `(code, date)` 会写出两行。
    * 分钟：**就是 pytdx 的原标签**（实测逐根对拍过，不做偏移 —— 见模块 docstring ①）。

    >>> bar_stamp({'year': 2024, 'month': 1, 'day': 2}, 'day')
    1704124800
    >>> bar_stamp({'year': 2024, 'month': 1, 'day': 2, 'hour': 9, 'minute': 30}, 'day')
    1704124800
    >>> bar_stamp({'year': 2024, 'month': 1, 'day': 2, 'hour': 14, 'minute': 59}, '1min') - \
        bar_stamp({'year': 2024, 'month': 1, 'day': 2}, 'day')
    53940
    """
    d = bar_date(bar)
    if frequency == 'day':
        raw = '{} 00:00:00'.format(d)
    else:
        raw = '{} {:02d}:{:02d}:00'.format(
            d, int(bar['hour']), int(bar['minute']))
    t = pd.Timestamp(raw)
    if frequency != 'day':
        t = t + MINUTE_LABEL_OFFSET
    return int(bj_date(t).timestamp())


def bar_datetime(stamp):
    """unix 秒 → 北京时间的 ``'%Y-%m-%d %H:%M:%S'``（分钟集合的 `datetime` 列）。

    >>> bar_datetime(1704178800)
    '2024-01-02 15:00:00'
    """
    return pd.Timestamp(stamp, unit='s', tz='UTC').tz_convert(
        'Asia/Shanghai').strftime('%Y-%m-%d %H:%M:%S')


def normalize_vol(vol, *, market, frequency):
    """pytdx 的成交量 → **8.3 存量的单位**（实测表见模块 docstring 第 ② 条）。

    * 分钟：股票 / ETF / 基金 **÷100**（pytdx 给**股**，存量存**手**）
    * 日线：指数 **×100**（pytdx 给**手**，存量存**股**）
    * 其余原样

    哨兵 `5.877471754e-39`（pytdx 的零成交标记）取整后自然是 **0** —— 与存量一致。

    >>> normalize_vol(25500.0, market='stock', frequency='1min')
    255
    >>> normalize_vol(31239.0, market='stock', frequency='day')
    31239
    >>> normalize_vol(4385300.0, market='index', frequency='day')
    438530000
    >>> normalize_vol(5.877471754e-39, market='stock', frequency='1min')
    0
    >>> normalize_vol(None, market='stock', frequency='day')
    0
    """
    v = int(vol or 0)
    if frequency != 'day':
        return v // 100 if market in ('stock', 'etf', 'fund') else v * INDEX_MIN_VOL_SCALE
    return v * 100 if market == 'index' else v


def normalize_amount(amount):
    """pytdx 的成交额 → 8.3 的口径：**哨兵归零，其余直传**。

    pytdx 对零成交 bar 给 `5.877471754e-39`（float32 次正规量级），而 8.3 **近期**行
    存的是 `0.0`（对拍实测：重叠窗口里 618 行差异全是「哨兵 vs 0.0」）。
    存量的哨兵只出现在 2018-11-08/14、2024-09-27 等**早期异常日**，是那个年代
    的写入路径把源端标记照抄了进去 —— **不学它**，按近期约定归零。

    >>> normalize_amount(5.877471754e-39)
    0.0
    >>> normalize_amount(3.1580928e7)
    31580928.0
    >>> normalize_amount(None)
    0.0
    """
    if amount is None:
        return 0.0
    a = float(amount)
    return 0.0 if 0.0 < a < 1e-30 else a


def bar_doc(bar, code, *, market, frequency, up_count=None, down_count=None):
    """pytdx bar → 8.3 时序文档（逐字段对齐存量，见模块 docstring 与 `MONGODB83.md` §五）。

    * ``code`` 用**裸 6 位**（存量如此；读侧 `kline83._read_timeseries` 用 `str(c)[:6]`）
    * ``vol`` / ``date_stamp`` / ``time_stamp`` 一律 `int()` —— **不能过 numpy/pandas**，
      否则 int32 会被静默拓宽成 int64（存量是 int32，读侧按它算）
    * ``datetime`` **只有分钟集合**写；日线写进去就与存量不一致
    * ``up_count`` / ``down_count`` **只有 `index_*`**，且**有才写** ——
      存量 `index_*` 是异质的（同一集合里有的文档有、有的没有），补 0 就是造假
    * ``amount`` **直传**（哨兵来自源端本身，见模块 docstring ③）；``vol`` 过
      :func:`normalize_vol` 换算单位

    >>> d = bar_doc({'year': 2024, 'month': 1, 'day': 2, 'open': 1.0, 'high': 2.0,
    ...              'low': 0.5, 'close': 1.5, 'vol': 100, 'amount': 150.0},
    ...             '600519', market='stock', frequency='day')
    >>> sorted(d)
    ['amount', 'close', 'code', 'date', 'date_stamp', 'high', 'low', 'open', 'time_stamp', 'ts', 'vol']
    >>> d['vol'], d['date_stamp'] == d['time_stamp'], d['code']
    (100, True, '600519')

    >>> m = bar_doc({'year': 2024, 'month': 1, 'day': 2, 'hour': 9, 'minute': 30,
    ...              'open': 1.0, 'high': 2.0, 'low': 0.5, 'close': 1.5,
    ...              'vol': 0, 'amount': 0.0},
    ...             '600519', market='stock', frequency='1min')
    >>> m['datetime'], m['amount']
    ('2024-01-02 09:30:00', 0.0)
    """
    d = bar_date(bar)
    stamp = bar_stamp(bar, frequency)
    doc = {
        'code': str(code)[:6],
        'date': d,
        'date_stamp': int(bj_date('{} 00:00:00'.format(d)).timestamp()),
        'time_stamp': stamp,
        'ts': bj_date(pd.Timestamp(stamp, unit='s', tz='UTC').tz_convert(
            'Asia/Shanghai').tz_localize(None)),
        'open': float(bar['open']),
        'high': float(bar['high']),
        'low': float(bar['low']),
        'close': float(bar['close']),
        'vol': normalize_vol(bar['vol'], market=market, frequency=frequency),
        'amount': normalize_amount(bar.get('amount')),
    }
    if frequency != 'day':
        doc['datetime'] = bar_datetime(stamp)
    if market == 'index':
        # 有才写 —— 存量异质，补 0 会与存量不一致
        if up_count is not None:
            doc['up_count'] = int(up_count)
        if down_count is not None:
            doc['down_count'] = int(down_count)
    for k in FORBIDDEN_KEYS:
        doc.pop(k, None)
    return doc


def xdxr_doc(row, code, market='stock'):
    """pytdx `get_xdxr_info` 的一行 → 8.3 `stock_xdxr` 文档。

    字段名映射（存量实测 18 键）：``panqianliutong/panhouliutong`` →
    ``liquidity_before/after``、``qianzongguben/houzongguben`` → ``shares_before/after``，
    其余**同名直传**；``category_meaning`` 与 ``name`` 同值（存量如此）。

    ⚠️ 数值**不做 float32→float64 洗白**：存量的 ``peigujia=3.5599999428`` 正是
    源端 float32 的 artifact，直传才能与存量逐值相同。

    >>> r = {'year': 2024, 'month': 1, 'day': 2, 'category': 1, 'name': '除权除息',
    ...      'fenhong': 10.0, 'songzhuangu': 0.0, 'peigu': 0.0, 'peigujia': 0.0,
    ...      'panqianliutong': 100, 'panhouliutong': 200,
    ...      'qianzongguben': 300, 'houzongguben': 400}
    >>> d = xdxr_doc(r, '600519')
    >>> d['date'], d['liquidity_before'], d['shares_after'], d['category_meaning']
    ('2024-01-02', 100, 400, '除权除息')
    """
    d = bar_date(row)
    stamp = bar_stamp(row, 'day')
    doc = {
        'code': str(code)[:6],
        'date': d,
        'date_stamp': int(bj_date('{} 00:00:00'.format(d)).timestamp()),
        'time_stamp': stamp,
        'ts': bj_date(pd.Timestamp(stamp, unit='s', tz='UTC').tz_convert(
            'Asia/Shanghai').tz_localize(None)),
        'category': int(row.get('category') or 0),
        'name': row.get('name'),
        'category_meaning': row.get('name'),
    }
    for src_key, dst_key in (('panqianliutong', 'liquidity_before'),
                             ('panhouliutong', 'liquidity_after'),
                             ('qianzongguben', 'shares_before'),
                             ('houzongguben', 'shares_after')):
        if src_key in row:
            doc[dst_key] = row[src_key]
    for k in ('fenhong', 'songzhuangu', 'peigu', 'peigujia', 'suogu', 'fenshu',
              'xingquanjia'):
        if k in row:
            doc[k] = row[k]
    return doc


def dedup_docs(docs):
    """按 ``(code, ts)`` 去重（保留**先出现**的一行）。

    源端重复实测有过（迁移器也做过同一件事），而时序集合**不拦重复**（P14）——
    写进去就是两行，读侧再聚合就重复计数。

    >>> a = {'code': '600519', 'ts': 1, 'close': 1.0}
    >>> b = {'code': '600519', 'ts': 1, 'close': 2.0}
    >>> c = {'code': '600519', 'ts': 2, 'close': 3.0}
    >>> [x['close'] for x in dedup_docs([a, b, c])]
    [1.0, 3.0]
    """
    seen, out = set(), []
    for d in docs:
        key = (d.get('code'), d.get('ts'))
        if key in seen:
            continue
        seen.add(key)
        out.append(d)
    return out


def trade_days_between(start, end, calendar=None):
    """``[start, end]`` 内的交易日（**含端点**，闭区间）。

    :param calendar: 交易日列表（``'YYYY-MM-DD'`` 字符串）；None = `TRADE_DATE_SSE`

    >>> trade_days_between('2026-10-01', '2026-10-09')
    ['2026-10-08', '2026-10-09']
    >>> trade_days_between('2026-10-09', '2026-10-08')
    []
    """
    cal = TRADE_DATE_SSE if calendar is None else calendar
    if start is None or end is None:
        return []
    start, end = pd.Timestamp(start).strftime('%Y-%m-%d'), pd.Timestamp(end).strftime('%Y-%m-%d')
    if start > end:
        return []
    # 交易日历是 'YYYY-MM-DD' 字符串，字典序即时间序
    return [d for d in cal if start <= d <= end]


def alive_threshold(frequency, now=None):
    """「这个集合**该有的最早一根** bar 的 `ts`」—— 集合里存在 ``ts >= 它`` 的 bar，
    就说明**这个集合是活的**（有人取到过最近的数据）。

    用途见 :func:`kline_save.collection_frontier` 的说明：拿它**一次探针**判「集合停没停」，
    替代「按日历推演每一根分钟 bar 该不该存在」那套脆东西（半日市、节假日、14:57 竞价…）。
    日历只在这里、只以**日**为粒度出场。

    **基准日的取法**（两端都重要）：

    * 今天是交易日、且已过**该频率的 ready 时刻**（:data:`SESSION_READY_AT`：日线 15:00、
      分钟 09:30）⇒ 基准日 = **今天**。
      ⚠️ **分钟盘中绝不退到昨天** —— 否则上午还没人取过时，探针会说「集合是活的」，
      逐只判据又把所有人判成「等于集合前沿」⇒ **一个上午一根本都不取**（死锁）。
      ⚠️ **日线盘中也不认今天** —— 当天日线盘中不该存在，认了会永远探不到前沿。
    * 否则（非交易日 / 未到 ready 时刻）⇒ 基准日 = **上一个交易日**。

    :param frequency: ``'day'`` 用 :data:`DAY_BAR_AT`（北京零点），其余用 :data:`SESSION_OPEN`
    :param now: 裸的本地时间（与 `kline_save` 里既有的 `datetime.now()` 同一口径）；
        ``None`` = 现在
    :returns: **UTC-aware** datetime（与库里 `ts` 同系，可直接比）；``None`` =
        **日历答不出来 ⇒ 调用方不许短路**
    """
    now = datetime.now() if now is None else now
    base = _session_base(now, SESSION_READY_AT.get(frequency, SESSION_OPEN))
    if base is None:
        return None
    return bj_date('{} {}'.format(base, DAY_BAR_AT if frequency == 'day' else SESSION_OPEN))


def _session_base(now, ready_at):
    """基准日（``'YYYY-MM-DD'``）—— 「今天若已过 `ready_at` 则今天，否则上一交易日」。

    **日历判断只此一处**：:func:`alive_threshold` 与 :func:`last_closed_session_bar`
    都走它。别在别处再写一遍「今天算不算数」—— 那条规则今天已经踩过两次
    （日线盘中认今天 / 分钟退到昨天）。

    :returns: 日期字符串；日历覆盖之外 / 今天之前没有交易日 → ``None``（调用方不许跳）
    """
    today = now.strftime('%Y-%m-%d')
    if not TRADE_DATE_SSE or not (TRADE_DATE_SSE[0] <= today <= TRADE_DATE_SSE[-1]):
        return None                      # 日历覆盖之外：宁可每次都取，也别静默跳
    if TRADE_DATE_SSE[0] > today:
        return None                      # 早于日历起点（异常时钟）
    days = trade_days_between(DAY_START, today)
    if not days:
        return None
    if days[-1] == today and now.strftime('%H:%M:%S') >= ready_at:
        return today
    prior = [d for d in days if d < today]
    return prior[-1] if prior else None


def last_closed_session_bar(frequency, now=None):
    """**最近一次收盘**那一批 bar 里最早那根的 `ts`（UTC-aware）；答不出来 → ``None``。

    拿来探「**这一整批 bar 落库了没有**」—— 比 :func:`alive_threshold`（开盘口径）
    强：它问的是"**收盘那批**在不在"，而不只是"今天有人取过"。

    * ``day``：那根日线的 `ts` 是该日**北京零点**（:data:`DAY_BAR_AT`）
    * 分钟：最后一根分钟 bar 的 `ts` 就是**收盘那一刻**（:data:`SESSION_CLOSE` = 15:00）

    「最近一次收盘」= 今天 15:00 之后算今天，否则算上一交易日 —— 基准日走
    :func:`_session_base`（**与 :func:`alive_threshold` 同一份日历规则**）。
    """
    now = datetime.now() if now is None else now
    base = _session_base(now, SESSION_READY_AT['day'])      # 15:00 之后才认今天
    if base is None:
        return None
    return bj_date('{} {}'.format(base, DAY_BAR_AT if frequency == 'day' else SESSION_CLOSE))


def bars_needed(first_date, today, bars_per_day, calendar=None, extra_days=1):
    """「从 `first_date` 到 `today` 大约有多少根 bar」的**上界**（用来估首次翻页次数，不用于截断）。

    只多不少：多加 `extra_days` 天兜底（TDX 多播一根、起点落在盘中）。

    >>> bars_needed('2026-10-08', '2026-10-08', 240)
    480
    >>> bars_needed('2026-10-09', '2026-10-08', 240)
    240
    """
    days = trade_days_between(first_date, today, calendar=calendar)
    return (len(days) + extra_days) * bars_per_day


def page_offsets(total, page=PAGE, extra=1):
    """升序的 `start` 偏移列表（``0`` = 最新一根）。

    pytdx 的 `start` 是**从最新往旧的偏移**：`start=800` = 跳过最新 800 根。
    增量要的正好是最新的若干根 → 从 0 开始**升序**翻，**短页即止**。

    ⚠️ 老树是 ``(int(lens/800)-i)*800`` **降序**：`lens` 估小会**静默丢掉最老的一页**。
    这里 `total` 只是初始上限，估多估少都不丢数据。

    >>> page_offsets(0)
    []
    >>> page_offsets(800)
    [0]
    >>> page_offsets(801)[:2]
    [0, 800]
    """
    if total <= 0:
        return []
    return list(range(0, int(total), int(page)))


def xdxr_adj_events_changed(old_docs, new_docs):
    """两组 xdxr 文档里**会影响价格的事件集合**是否不同（决定要不要重写/重算 adj）。

    参与判据的只有两类：

    * ``category == 1``（除权除息）—— 比 :data:`XDXR_ADJ_KEYS` 那几个字段；
    * ``category == 11``（**扩缩股**，ETF 的份额折算）—— 比 ``date`` + ``suogu``。
      它**确实改变价格**（参考价 = 前收盘 / `suogu`），故必须参与；漏掉它的后果是实打实的：
      实测 159915 的份额折算事件因此**一行都没落库**，而 159110/159398 的存量
      `etf_adj` 一直留着 ~9900% 的假跳空。

    **不看** ``category == 5``（股东大会）之类的差异 —— 否则「多了一条股东大会」会触发
    全市场几千只票的前复权因子重算，白跑几小时。

    ⚠️ 本判据同时**把门着写入**：`save_xdxr_tdx` 判定「未变」时会**直接返回、不写库**。
    所以判据里漏掉一类事件 = 那类事件永远进不了库（不只是不重算）。

    >>> a = [{'date': '2024-01-02', 'fenhong': 10.0, 'category': 1}]
    >>> xdxr_adj_events_changed(a, list(a))
    False
    >>> xdxr_adj_events_changed(a, [dict(a[0], fenhong=11.0)])
    True
    >>> xdxr_adj_events_changed(a, a + [{'date': '2024-05-02', 'category': 5}])
    False
    >>> sp = [{'date': '2024-06-01', 'category': 11, 'suogu': 0.01}]
    >>> xdxr_adj_events_changed(a, a + sp)          # 新增份额折算 → 必须重算
    True
    >>> xdxr_adj_events_changed(a + sp, a + sp)
    False
    """

    def signature(docs):
        out = []
        for d in docs:
            cat = int(d.get('category') or 0)
            if cat == 1:
                out.append('1|' + '|'.join(
                    '{}={}'.format(k, d.get(k)) for k in XDXR_ADJ_KEYS))
            elif cat == 11:
                out.append('11|{}|{}'.format(d.get('date'), d.get('suogu')))
        return sorted(out)

    return signature(old_docs) != signature(new_docs)


#: `bar_doc` 里那些"不该出现"的键（见 :data:`FORBIDDEN_KEYS`）
_DOC_ORDER = ('code', 'date', 'date_stamp', 'time_stamp', 'ts', 'open', 'high',
              'low', 'close', 'vol', 'amount', 'datetime', 'up_count', 'down_count')


def bar_docs(bars, code, *, market, frequency):
    """**一批** pytdx bar → 8.3 文档（列式，polars）—— 与逐行 :func:`bar_doc` **等价**。

    为什么要它（实测，2026-10-08）
    ============================
    一次性 300 只 × 240 根 = 7.2 万根：

    ======================  ==============  ==========================
    构造方式                  每根            全市场 1min（134 万根）
    ======================  ==============  ==========================
    Python 逐行 `bar_doc`     **51.5 µs**     **68.9 s**
    polars 列式（本函数）      **4.3 µs**      **5.8 s**
    ======================  ==============  ==========================

    12 倍，而 69 秒在一次全量跑里是实打实的开销 —— 所以这条路径用列式。
    **`bar_doc` 仍是参照实现**：单测逐字段对拍两者（`test_kline_save` 的
    `test_bar_docs_matches_row_wise`），口径只有一处定义。

    :param bars: pytdx 的 `list[dict]`（可空）
    :returns: `list[dict]`，字段集与 :func:`bar_doc` 逐字相同

    >>> bars = [{'year': 2026, 'month': 10, 'day': 8, 'hour': 9, 'minute': 31,
    ...          'open': 10.0, 'high': 10.1, 'low': 9.9, 'close': 10.05,
    ...          'vol': 99300.0, 'amount': 123223312.0}]
    >>> d = bar_docs(bars, '600519', market='stock', frequency='1min')[0]
    >>> d['date'], d['datetime'], d['vol']          # 分钟线 vol ÷100（股→手）
    ('2026-10-08', '2026-10-08 09:31:00', 993)
    >>> bar_docs([], '600519', market='stock', frequency='day')
    []
    """
    if not bars:
        return []
    df = pl.DataFrame(bars)
    code6 = str(code)[:6]

    if 'year' not in df.columns:
        return []
    df = df.with_columns([
        pl.concat_str([
            pl.col('year').cast(pl.Utf8), pl.lit('-'),
            pl.col('month').cast(pl.Utf8).str.zfill(2), pl.lit('-'),
            pl.col('day').cast(pl.Utf8).str.zfill(2)]).alias('date'),
    ])
    # 北京零点 → epoch 秒（固定 +8 = Asia/Shanghai，无夏令时）
    midnight = (pl.col('date').str.to_datetime('%Y-%m-%d', time_zone='UTC')
                - pl.duration(hours=8))
    # polars 的 `dt.timestamp` 只收 ns/us/ms（本版本）→ 用 us 折成秒
    date_stamp = (midnight.dt.timestamp('us') // 1_000_000).cast(pl.Int64)
    if frequency == 'day':
        time_stamp = date_stamp
    else:
        time_stamp = (date_stamp + pl.col('hour').cast(pl.Int64) * 3600
                      + pl.col('minute').cast(pl.Int64) * 60)

    # vol：分钟线（股票/ETF/基金）÷100（pytdx 给股、存量存手）；指数日线 ×100。
    # 哨兵（float32 次正规量级）经 `cast(Int64)` 自然变 0 —— 与 `normalize_vol` 同口径。
    vol = pl.col('vol').cast(pl.Float64).fill_null(0.0).cast(pl.Int64)
    if frequency != 'day':
        vol = vol // 100 if market in ('stock', 'etf', 'fund') else vol * INDEX_MIN_VOL_SCALE
    elif market == 'index':
        vol = vol * 100

    # amount：哨兵（0 < a < 1e-30）归零，其余直传 —— 与 `normalize_amount` 同口径
    amount = pl.col('amount').cast(pl.Float64).fill_null(0.0)
    amount = (pl.when((amount > 0) & (amount < 1e-30)).then(0.0)
              .otherwise(amount)).cast(pl.Float64)

    exprs = [
        pl.lit(code6).alias('code'),
        pl.col('date'),
        date_stamp.cast(pl.Int32).alias('date_stamp'),
        time_stamp.cast(pl.Int32).alias('time_stamp'),
        (pl.from_epoch(time_stamp, time_unit='s').dt
         .replace_time_zone('UTC')).alias('ts'),
        pl.col('open').cast(pl.Float64), pl.col('high').cast(pl.Float64),
        pl.col('low').cast(pl.Float64), pl.col('close').cast(pl.Float64),
        vol.cast(pl.Int32).alias('vol'), amount.alias('amount'),
    ]
    if frequency != 'day':
        exprs.append(pl.concat_str([
            pl.col('date'), pl.lit(' '),
            pl.col('hour').cast(pl.Utf8).str.zfill(2), pl.lit(':'),
            pl.col('minute').cast(pl.Utf8).str.zfill(2), pl.lit(':00')]).alias('datetime'))
    if market == 'index':
        for k in ('up_count', 'down_count'):
            if k in df.columns:
                exprs.append(pl.col(k).cast(pl.Int32).alias(k))

    out = df.select(exprs).to_dicts()
    for d in out:                       # 统一键序，便于与 `bar_doc` 对拍
        for k in FORBIDDEN_KEYS:
            d.pop(k, None)
    return out
