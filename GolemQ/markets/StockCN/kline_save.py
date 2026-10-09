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

"""`--save tdx` 的 K 线主体：**增量水位 → pytdx 取数 → 落 8.3 时序集合**。

为什么这一层碰 DB 却在 `markets/StockCN/`（CLAUDE.md 的「DB 操作只能进 services/」）
====================================================================================
与 :mod:`markets.StockCN.refdata_save` / :mod:`markets.StockCN.kline83` 同一处境、
同一理由：这里读写的是 **8.3 A 股库的数据契约**（集合名、字段口径、单位换算），
换市场不成立。机制（先删后插、批大小）已抽到 :mod:`GolemQ.datasource.writer`，
所以本模块里没有一处裸 `insert_many` / `delete_many`。**不要**把这里的东西搬去
`services/`（那边是通用 CRUD，不认识 K 线文档），也不要在这里新增第二种写策略。

增量怎么算（用户 2026-10-08 定的口径：**从已有数据往后补**）
=========================================================
* **水位**：逐 code 取该集合里 `ts` 最大的那根（时序集合的 `(metaField, timeField)`
  就是为「per-series 取末点」建的，实测 12–29 ms/code）。
* **窗口起点** = **上一次写进去的那根 bar 本身**（`floor_ts`）—— **不是它所在那天的
  00:00**。粒度差一个量级：按天划时 `1min` 会把水位那天**已冻结的 240 根**全捞进来重写，
  按根划则每分钟线只重写 1 根。实测（每票）：`1min` **删 1.00** 根、`5min` **删 1.00** 根、
  `day` **删 1.00** 根。
  「上次挂在中途」由 **per-code 水位**天然兜住（各票从自己的最后一根续），
  不需要靠 margin；`margin_days > 0` 只用于「源端回补了更早历史」的**修补**，
  且每多一天就是每票多删多写一根 —— 见 `DEFAULT_MARGIN_DAYS` 的说明。
  写是「窗口整体替换」（时间序列不能 upsert），所以窗口越窄越省。
* **盘中 `day` 不更新**（项目所有者明确）：盘中最后一根 day bar 仍是昨天那根，
  盘中行情由 REALTIME 的 L1 tick 合成（另一条路径）。⚠️ 但 pytdx 的 day 接口
  盘中会返回「当日累计」bar → 盘中跑本命令会写进去，收盘后再跑即覆盖。
* 查不到任何 bar 的 code → 起点退到 `DAY_START`(1990-01-01) / `start_min`(2015-01-01)。
* **已知局限**：增量只能补**水位之后**的洞。水位**之前**的历史空洞（例如某只票
  2019 年整段缺失）不会被发现 —— 那要用 `kline_status.kline_status()` 的覆盖核对查。
"""
from __future__ import annotations

import threading
import time as _time
from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm
from datetime import datetime, timezone

import pandas as pd

from GolemQ.datasource.writer import replace_code_rows, save_bar_chunk

from .kline_doc import (
    alive_threshold,
    last_closed_session_bar,
    BARS_PER_DAY,
    bar_docs,
    bar_stamp,
    DAY_START,
    MIN_START_DEFAULT,
    TDX_CATEGORY,
    bar_date,
    bar_doc,
    bars_needed,
    dedup_docs,
    page_offsets,
    trade_days_between,
    xdxr_doc,
)
from .fq import xdxr_to_adj
from .kline83 import bj_date
from .datasource.pytdx_kline import bars_paged, xdxr_rows
from .datasource.pytdx_source import TdxSource, _tdx_market_of, tdx_market_of

__all__ = [
    'DEFAULT_JOBS',
    'DEFAULT_MARGIN_DAYS',
    'FREQUENCIES',
    'TARGETS',
    'floor_date',
    'last_bar',
    'save_kline_tdx',
    'save_adj',
    'verify_adj',
    'save_xdxr_tdx',
    'target_collection_name',
    'universe',
]

#: 标的族 → 8.3 里的集合前缀。`fund_*` / `future_*` **不在范围内**（老树 `save_x_func` 也没做）。
TARGETS = ('stock', 'index', 'etf')

#: 频率 → 集合后缀。日线是 `_day`，分钟是 `_Nmin`。
FREQUENCIES = ('day', '1min', '5min', '15min', '30min', '60min')

#: 并发线程数（= 每 task 一条 pytdx 连接）。老树分钟线用 4，实盘跑过。
DEFAULT_JOBS = 4

#: 增量窗口往前多算几个交易日。**默认 0 = 只覆盖「水位当天」那一根。**
#:
#: 为什么 0 就够：水位是 **per-code** 的（每个 code 从自己最后一根 bar 起算），
#: 所以「上次跑挂在中途」由它天然兜住；而「收盘后再跑一次把当日那根写全」
#: 只需要窗口从**水位当天**起算即可（见下条领域约定）。
#:
#: **领域约定（项目所有者 2026-10-08 明确）**：盘中 **`day` 不更新** ——
#: 所有行情系统都一样，盘中最后一根 day bar 仍是**昨天**那根；盘中的行情是
#: 用 **REALTIME 的 L1 tick 合成**的（那是另一条路径，与 `--save tdx` 无关）。
#:
#: ⚠️ 但**实测**：pytdx 的 `get_security_bars(category=9)` **盘中确实会返回一根
#: 「当日累计」的 day bar**（标 `15:00`，量与额是当天至今的累计）。所以
#: **盘中跑 `--save tdx` 会把未收盘的当日 bar 写进去** —— 收盘后再跑一次即被覆盖
#: （窗口从水位当天起算，正好覆盖它）。这不是"系统约定如此"，而是这条取数路径
#: 的行为，别混为一谈。
#:
#: ⚠️ **不要把它调大当默认**：删除窗每往前一天，**每只票每天就多删多写一根**。
#: 5,574 只 × 6 天 ≈ 每天 3.3 万行额外的 delete + insert，而收益只在
#: 「源端回补了更早的历史」这一种情形 —— 那是**修补**，用 `--save-margin-days` 显式开。
DEFAULT_MARGIN_DAYS = 0

#: 每处理多少 code 打一行进度。**逐条打就是在闪** —— 5,500 只 × 6 个频率根本没法看。
DEFAULT_PROGRESS_EVERY = 100

#: 标的宇宙的回看窗口：用「近 N 个交易日出现过的 code」当作指数/ETF 的名单来源。
_UNIVERSE_WINDOW_DAYS = 40


def _fmt_dur(seconds):
    """秒 → ``'1m23s'`` / ``'12.3s'``（进度行与 ETA 用）。

    >>> _fmt_dur(12.34)
    '12.3s'
    >>> _fmt_dur(83)
    '1m23s'
    """
    seconds = float(seconds)
    if seconds < 60:
        return '{:.1f}s'.format(seconds)
    return '{:.0f}m{:.0f}s'.format(seconds // 60, seconds % 60)


def progress_extra(last, stats, elapsed, done, total, today=None):
    """进度行里「除标签与百分比之外」的那段：当前标的+窗口、累计计数、用时与预计。

    单独一段是为了让起始行/结束行与集合内进度共用同一份措辞（措辞只写一次）。
    """
    parts = []
    # 起始行还没有"当前标的" —— 那就整段省掉，别显示成 ` |  →今天`（看着像坏了）
    if last:
        code, floor = last
        # 窗口两端都写 `Y-m-d`：**不要**写成 `2026-09-24→今天`（项目所有者明确）
        parts.append('{} {}~{}'.format(code, floor, today or ''))
    parts.extend(progress_summary(stats, elapsed, done, total))
    return ' | '.join(parts)


def progress_summary(stats, elapsed, done=0, total=0):
    """**汇总段**（不带"当前标的"）：写/删/空/跳过/重连/错 + 用时。

    完成行只该用它 —— 那里出现的 code 只是"最后处理的那只"，**不是计数**，
    读起来必然被当成累加器（项目所有者原话：「完成：920992 这个明显是一个累加的
    计数器，不对吧」）。而且各票窗口本就不同，窗口写在完成行上也没有意义。
    """
    parts = ['写 {} 删 {}'.format(stats['inserted'], stats['deleted'])]
    parts.append('空 {} 跳过 {} 重连 {} 错 {}'.format(
        stats['empty_codes'], stats['skipped_codes'], stats['reconnects'],
        len(stats['errors'])))
    if done and total:
        # 样本太少时 ETA 没有意义（每只票取数页数不同，前几只的均值会差很多倍）
        eta = _fmt_dur(elapsed / done * (total - done)) if done >= 5 else '—'
        parts.append('已用 {} 预计 {}'.format(_fmt_dur(elapsed), eta))
    else:
        parts.append('已用 {}'.format(_fmt_dur(elapsed)))
    return parts




def progress_line(name, done, total, last, stats, elapsed, every=None,
                  today=None):
    """单行进度：**集合 / 进度 / 当前 code 与窗口 / 累计行数 / 异常计数 / 已用与预计**。

    设计给"人在旁边看着"用：① 一行里给出**可判断对不对**的信息（窗口的 from→to
    能一眼看出增量算得对不对）；② **按批打印**（`every` 只），不做逐条刷屏 ——
    5,500 只 × 6 个频率逐条打就是在闪。

    >>> progress_line('stock_day', 1800, 5573, ('300333', '2026-09-30'), {'inserted': 21340,
    ...     'deleted': 8120, 'empty_codes': 12, 'skipped_codes': 0, 'reconnects': 0,
    ...     'errors': []}, 130.0, today='2026-10-09')[:52]
    '[stock_day] 1800/5573 32.3% | 300333 2026-09-30~2026'
    >>> progress_line('stock_day', 0, 10, None, {'inserted': 0, 'deleted': 0,
    ...     'empty_codes': 0, 'skipped_codes': 0, 'reconnects': 0, 'errors': []}, 0.0)
    '[stock_day] 0/10 0.0% | 写 0 删 0 | 空 0 跳过 0 重连 0 错 0 | 已用 0.0s 预计 —'
    """
    pct = (100.0 * done / total) if total else 100.0
    return '[{}] {}/{} {:.1f}% | {}'.format(
        name, done, total, pct,
        progress_extra(last, stats, elapsed, done, total, today=today))


def target_collection_name(target, frequency):
    """`('stock', '1min')` → `'stock_1min'`；`('index', 'day')` → `'index_day'`。

    >>> target_collection_name('stock', 'day')
    'stock_day'
    >>> target_collection_name('etf', '5min')
    'etf_5min'
    """
    return '{}_{}'.format(target, 'day' if frequency == 'day' else frequency)


def _db():
    """8.3 的 A 股历史库。**函数级导入**：`markets/StockCN/__init__.py` 会间接导入本模块，
    顶层 `from . import GOLEMQ_STOCK_CN` 会撞上半初始化的 ImportError
    （`kline83._collection` / `realtime._realtime_db` 同一写法、同一理由）。"""
    from . import GOLEMQ_STOCK_CN
    return GOLEMQ_STOCK_CN


def universe(target, db=None, codes=None, verbose=False):
    """该族要处理的 code 列表（**裸 6 位**）。

    * `stock`：8.3 `stock_list`（普通集合，毫秒级）—— 它**含北交所**（pytdx 自己枚举不到）。
    * `etf`：8.3 `etf_list`。
    * `index`：`index_day` **近 `_UNIVERSE_WINDOW_DAYS` 个交易日**出现过的 code
      （时序集合**绝不能**无时间窗 `distinct('code')` —— 实测 >120s 不返回）。
    """
    if codes:
        return [str(c)[:6] for c in codes]
    db = db if db is not None else _db()
    if target == 'stock':
        return sorted(db['stock_list'].distinct('code'))
    if target == 'etf':
        return sorted(db['etf_list'].distinct('code'))
    if target == 'index':
        since = trade_days_between(
            DAY_START, datetime.now().strftime('%Y-%m-%d'))[-_UNIVERSE_WINDOW_DAYS:]
        lo = bj_date('{} 00:00:00'.format(since[0])) if since else None
        q = {'date_stamp': {'$gte': int(lo.timestamp())}} if lo else {}
        return sorted(db['index_day'].distinct('code', q))
    raise ValueError('未知标的族: {!r}（应为 {}）'.format(target, TARGETS))


def last_bar(coll, code):
    """该 code 在该集合里的**最后一根 bar**：``(ts, date)``；没有则 ``(None, None)``。

    **按 `ts` 取**（timeField），不按 `date_stamp`：时序集合的加速只对
    `(metaField, timeField)` 生效，`date_stamp` 只有单键索引，按 code+它排序会退化。
    """
    doc = coll.find_one({'code': code}, {'_id': 0, 'ts': 1, 'date': 1},
                        sort=[('ts', -1)])
    if not doc:
        return None, None
    return doc.get('ts'), doc.get('date')


#: 前沿二分的**步长**（秒）—— 一分钟。
#:
#: ⚠️ **必须按「格点」二分，不能按连续时间二分。** 连续二分只会收敛到一个**区间**
#: （「已证明有数据的最大点」），于是**低估**真实前沿 —— 实测第一版就把 `stock_1min`
#: 的前沿报成 `06:59` 而真实最新是 `07:00`（少一分钟）。少一分钟看起来只是"保守"，
#: 实则会让判据**偏到另一侧**（见 :func:`collection_frontier` 的说明）。
#:
#: bar 的 `ts` 全都落在**整分钟**上（分钟 bar 取 `HH:MM`；日线 bar 取该日北京零点），
#: 所以只在 `60` 秒的整数倍上提问，就能**精确命中**那一根。
_FRONTIER_STEP_S = 60


def _as_utc(value):
    """裸 datetime → **UTC-aware**（pymongo 默认返回裸值，而其实值是 UTC）。

    ⚠️ 不补这一下，`.timestamp()` 会把它当**本地时间** —— 在 CST 机上整整差 8 小时，
    而且**不报错**。既有代码早就踩过同一件事，`floor_ts` 里同样是
    `last_ts.replace(tzinfo=timezone.utc)`。
    """
    if value is None or value.tzinfo is not None:
        return value
    return value.replace(tzinfo=timezone.utc)


def collection_frontier(coll, lo, hi):
    """二分找**集合里最后一个有数据的时刻** → unix 秒；定不了 → ``None``。

    谓词 ``P(X) = 存在 ts >= X 的 bar``，**只在 :data:`_FRONTIER_STEP_S` 的整数倍上提问**
    （见那个常量的说明：连续二分只会低估）。窗口一个交易日 ≤ 330 个格点 ⇒
    ``log2(330) ≈ 9`` 次探针。

    ⚠️ **必须用 `ts`（timeField）**。实测 `stock_1min`（21.4 亿行）：

    | 查谁 | 耗时 |
    |:--|--:|
    | `ts` 范围查询 | **0.0051 s** |
    | `time_stamp` 同样条件 | **44.63 s** |

    **8800 倍**，而且写错了**不报错** —— 只是每次探针白等 44 秒（9 次探针 ≈ 7 分钟）。
    时序集合的加速只对 `(metaField, timeField)` 生效。

    ⚠️ **精确性是承重的，两个方向都会坏事**：
    * **低估**（连续二分的老毛病）⇒ `ts >= frontier` 对**每一只票都成立** ⇒ 全都跳，
      短路看着"生效"了，实际**什么新数据都没取**；
    * **高估** ⇒ 连真在最新那根上的票也判成落后 ⇒ 每只都白取，等于没做短路。
    所以要的是「**恰好等于**集合里最新那根 bar 的时刻」。

    :param lo: **已知有数据**的下界（通常是 :func:`kline_doc.alive_threshold` 的产物）
    :param hi: 上界（通常是「现在」）
    :returns: unix 秒；``lo`` 处就没有数据（集合空 / 停住了）或探针出错 → ``None``
    """
    lo, hi = _as_utc(lo), _as_utc(hi)
    if lo is None or hi is None or lo >= hi:
        return None
    base = int(lo.timestamp())
    steps = int((hi - lo).total_seconds()) // _FRONTIER_STEP_S
    if steps < 1:
        return None

    def _has(at_sec):
        return coll.find_one({'ts': {'$gte': datetime.fromtimestamp(at_sec, timezone.utc)}},
                             {'_id': 0, 'ts': 1}, limit=1) is not None

    try:
        if not _has(base):
            return None          # 连下界都没有数据 = 集合是空的 / 停住了
        low, high = 0, steps     # 不变量：P(low) 真；P(high) 待定
        if _has(base + _FRONTIER_STEP_S * high):
            return base + _FRONTIER_STEP_S * high     # 一直有数据到窗口末尾
        while high - low > 1:                          # P(high) 假、P(low) 真
            mid = (low + high) // 2
            if _has(base + _FRONTIER_STEP_S * mid):
                low = mid
            else:
                high = mid
    except Exception:            # noqa: BLE001 探针失败 ⇒ 判不了 ⇒ 不跳（保守）
        return None
    return base + _FRONTIER_STEP_S * low


# ---------------------------------------------------------------------------
# 短路的三道判据之一：TTL（复用 refdata 的刷新闸，颗粒度 = 每个集合）
# ---------------------------------------------------------------------------
#: K 线各集合的刷新阈值 `(盘中, 盘后)`，小时。**与参考数据同口径**（`refdata_save.TTL_HOURS`）：
#: 盘中 5h / 盘后 24h。它**不是**「多久取一次数据」，而是「多久**强制全查一次**」——
#: 兜底用的：万一短路判据（集合自证 + 逐只水位）全都错了，这道闸会把每个集合
#: 至少完整扫一遍，错误不会无限累加。
KLINE_TTL_HOURS = (5, 24)

#: 签到名前缀。**每个集合一条记录**（26 个节点各自独立），不是按族、不是全局。
KLINE_CHECKIN_PREFIX = 'kline:'


def kline_checkin_name(collection) -> str:
    return KLINE_CHECKIN_PREFIX + str(collection)


def kline_ttl_hours(collection=None, now=None):
    """该集合此刻该用的强制全查间隔（小时）。盘中判定**复用** `GQ_util_if_tradetime`
    （它认真交易日历，且把 9:15 起的集合竞价算作盘中）—— 别再写第二套。"""
    open_h, closed_h = KLINE_TTL_HOURS
    from .date_utils import GQ_util_if_tradetime
    now = now if now is not None else datetime.now()
    return open_h if GQ_util_if_tradetime(now) else closed_h


def kline_sweep_age_hours(collection, now=None):
    """该集合距上次「完整扫过一遍」多少小时；从没扫过 → ``None``（调用方按「必须全查」）。"""
    from GolemQ.supervisor.function_checkin import (checkin_age_hours,
                                                    stable_caller_key)
    return checkin_age_hours(kline_checkin_name(collection),
                             caller_ip=stable_caller_key(), now=now)


def mark_kline_sweep(collection) -> bool:
    """记下「该集合刚刚完整扫完」。失败**不抛**（见 `mark_checkin` 的说明）。"""
    from GolemQ.supervisor.function_checkin import mark_checkin, stable_caller_key
    return mark_checkin(kline_checkin_name(collection), kline_ttl_hours(collection),
                        caller_ip=stable_caller_key())


def allow_shortcircuit(collection, now=None):
    """该集合本次**允不允许**短路 —— TTL 没过期才允许。

    从没扫过（``None``）＝**不允许**：第一次见到一个集合，先老老实实扫一遍再谈跳过。
    ⚠️ 这是判据③（TTL）。判据④「盘中禁用分钟短路」在 :func:`intraday_blocks_shortcircuit`。
    """
    age = kline_sweep_age_hours(collection, now=now)
    if age is None:
        return False
    return age < kline_ttl_hours(collection, now=now)


def hours_since_last_close(now=None):
    """距**最近一次收盘**（交易日 15:00）过去了多少小时；日历答不出来 → ``None``。"""
    now = now if now is not None else datetime.now()
    closed = last_closed_session_bar('1min', now)
    if closed is None:
        return None
    return (now.timestamp() - closed.timestamp()) / 3600.0


def fast_skip_reason(coll, collection, frequency, now=None):
    """**收盘后的快路径**：整个集合已整批落库 ⇒ 返回一句说明；否则 ``None``。

    成立时调用方可以**整段跳过**：不建连、**连逐只读都不做**（省的是 ~5,575 次
    `last_bar`，一轮全量约 40 秒）。**它比 TTL 硬，所以不看 TTL**（用户 2026-10-09 指出）：

    * TTL 只断言「最近查过」（5h / 24h）；
    * 本判据断言「**最近一个已收盘的交易日整批都查过了**」—— 更强，故 TTL 在此多余。

    三条判据：

    1. **不是「盘中 × 分钟」** —— ⚠️ 承重：盘中「最近一次收盘」是**昨天** 15:00，
       若不加这条，周一 10:00 会拿周五收盘当"已覆盖"⇒ 跳过周一的分钟线。
       `day` 不受影响（盘中本来就不写当天日线，跳是对的）。
    2. **上次全宇宙全查发生在最近一次收盘之后** —— 「全宇宙」是前提：记账只在
       `codes is None` 时发生（见 `save_kline_tdx`），所以这条等价于
       「**每一只 code 都在那次全查里被处理过**」。
    3. **探针**：库里真有 ``ts >= 最近一次收盘那一批的最早那根`` —— 记录说扫过
       ≠ 数据还在（误删、或那一轮有 code 失败）。~50ms，买的是这条保险。

    ⚠️ **已知残余**（写在这里免得日后当成 bug）：判据 2 无法区分「那次全查里某只 code
    失败了」。那只票会一直落后到**下一个交易日收盘**（TTL 到期后走常规路径时补上）。
    失败在**当时那一轮**是报出来的（`stats['errors']`），不是静默的。
    """
    now = now if now is not None else datetime.now()
    if intraday_blocks_shortcircuit(frequency, now):
        return None                       # ① 盘中×分钟：不许走快路径
    since_close = hours_since_last_close(now)
    if since_close is None:
        return None                       # 日历答不出来 ⇒ 保守
    age = kline_sweep_age_hours(collection, now=now)
    if age is None or age >= since_close:
        return None                       # 从没扫过 / 上次扫在收盘**之前** ⇒ 走常规路径
    threshold = last_closed_session_bar(frequency, now)
    if threshold is None:
        return None
    try:
        if coll.find_one({'ts': {'$gte': threshold}}, {'_id': 0, 'ts': 1},
                         limit=1) is None:
            return None                   # ③ 记录说扫过、库里没有那一批 ⇒ 不跳
    except Exception:                     # noqa: BLE001 探针失败 ⇒ 判不了 ⇒ 不跳
        return None
    return ('上次全查 {:.1f}h 前（在最近一次收盘之后）且库里已有收盘那批 '
            '→ 整段跳过（不建连、不逐只读）'.format(age))


def hours_since_last_open(now=None):
    """距**最近一次开盘**（交易日 09:30）过去了多少小时；日历答不出来 → ``None``。

    复用 :func:`kline_doc.alive_threshold('1min', ...)` —— 它给出的正是
    「最近一次开盘」（今天已开市 ⇒ 今天 09:30，否则上一交易日 09:30），
    **别再写第二套交易日历判断**。
    """
    now = now if now is not None else datetime.now()
    opened = alive_threshold('1min', now)
    if opened is None:
        return None                      # 日历覆盖之外 ⇒ 调用方不许跳
    return (now.timestamp() - opened.timestamp()) / 3600.0


def allow_xdxr_shortcircuit(collection, now=None):
    """复权（`*_xdxr`）的闸 —— 比 K 线**多一条**判据（`DECISIONS.md` D26）。

    K 线的闸只问「上次全查多久以前」；复权还要问「**跨没过一个开盘**」：

    * 上次全查发生在**最近一次开盘之前** ⇒ **不许跳** —— 除权信息是**日级**的、
      当天那条**开盘前就该拿到**，开盘后还拿着昨天扫的结果，说明今天那份没扫过；
    * 发生在**之后** ⇒ 可以跳（本交易日的数据这一轮已经拿到了）。

    ⚠️ 为什么不用"N 小时以内"：`(5, 24)` 的盘后口径会把**开盘前那一跑**跳掉 ——
    昨晚 17:30 扫过，今早 08:30 只隔 15h < 24h ⇒ 跳 ⇒ 当天除权信息漏到收盘后。
    实测这个时间线是真实的（用户 2026-10-09 指出）。
    ⚠️ **它保证不了"跑"、只保证"不跳"**：只在收盘后跑的工作流，开盘前永远没扫过，
    当天盘中仍是旧基准 —— 那要**排程上多一次盘前跑**，代码救不了。
    """
    if not allow_shortcircuit(collection, now=now):
        return False                     # TTL 闸（同 K 线）
    since_open = hours_since_last_open(now)
    if since_open is None:
        return False                     # 日历答不出来 ⇒ 全查（保守）
    age = kline_sweep_age_hours(collection, now=now)
    if age is None:
        return False                     # 从没扫过（`allow_shortcircuit` 已挡，双保险）
    return age < since_open              # 上次扫在开盘**之后** ⇒ 本交易日已扫


def intraday_blocks_shortcircuit(frequency, now=None):
    """**盘中 + 分钟频率 ⇒ 禁止短路**（判据④）。返回 True 表示「不许跳」。

    ⚠️ 这条是 2026-10-09 补的，堵的是一个**会静默冻住集合**的洞：

    短路判据②比的是**数据自证的前沿**（集合里最新那根 bar 在哪儿）。而**盘中**
    每分钟都有新数据 ⇒ 「全跳」之后**没有东西去写新 bar** ⇒ 前沿**自己不会前进** ⇒
    下一轮还是全跳 …… 集合就**冻在那一分钟**，直到 TTL 到期（最长 5h）才被强制全扫。
    判据①（探针）只挡得住「今天压根没取过」，挡不住「上午取过一次、然后停住」。

    **盘中禁用没有任何真实损失**：盘中本来每分钟都有新数据要取 ——「短路」在盘中
    字面意思就是"不取新数据"，那不是优化，那是 bug。

    `day` 不受影响：盘中**不写当天日线**（`PITFALLS.md` P20），所有 code 都停在前一
    个交易日的 bar 上，**短路是对的**（也正是盘中 `day` 能秒过的原因）。

    边界复用 :func:`date_utils.GQ_util_if_tradetime`（STOCK_CN）：**11:31–12:59 午休为
    假 ⇒ 午休仍可短路**（上午的分钟 bar 已定稿）、**15:00 之后为假 ⇒ 收盘后照常短路**。
    """
    if frequency == 'day':
        return False
    from .date_utils import GQ_util_if_tradetime
    return bool(GQ_util_if_tradetime(now if now is not None else datetime.now()))


def floor_date(last_date, margin_days=DEFAULT_MARGIN_DAYS, today=None):
    """窗口起点：水位日往前 `margin_days` 个交易日。

    >>> floor_date('2026-10-08', margin_days=1, today='2026-10-08')
    '2026-09-30'
    >>> floor_date(None, margin_days=5, today='2026-10-08')   # 没有水位 → 老树起点
    '1990-01-01'
    """
    if last_date is None:
        return DAY_START
    today = today or datetime.now().strftime('%Y-%m-%d')
    days = trade_days_between(DAY_START, last_date)
    if not days:
        return DAY_START
    idx = max(0, len(days) - 1 - int(margin_days))
    return days[idx]


def floor_ts(last_ts, last_date, margin_days=DEFAULT_MARGIN_DAYS, start_date=None):
    """增量窗口的起点 —— **上一次写进去的那根 bar 本身**，不是它所在那天的 00:00。

    为什么按「根」而不是按「天」
    ==========================
    已收盘的 bar 是**冻结数据**。按「天」划窗口时，`1min` 会把水位那天**已冻结的
    240 根**全捞进来重写 —— 实测那就是"每天多删多写 240 根"的来源。
    真正需要覆盖的只有**最后一根**（它可能是盘中写的、不完整）—— 所以日线与分钟线
    的删除量上界都是 **1 根**。

    ⚠️ **时区坑**：从 Mongo 读回来的 `ts` 是 **naive**（pymongo 默认 `tz_aware=False`），
    而 `kline_doc` 构造的 `ts` 是 **tz-aware UTC**。两者直接比较会得到错误结果，
    故这里统一补成 UTC-aware（存量 `ts` 的约定就是 UTC 瞬时）。

    :param margin_days: >0 时改为「往前 `margin_days` 个交易日的 00:00」——
        只在「源端回补更早历史」的**修补**场景用（见 `DEFAULT_MARGIN_DAYS`）。
    :param start_date: 没有水位时的起点日（`DAY_START` / `--save-start`）
    """
    if last_ts is None:
        return None if start_date is None else None    # 调用方直接用起点日
    if not margin_days:
        return (last_ts if last_ts.tzinfo is not None
                else last_ts.replace(tzinfo=timezone.utc))
    days = trade_days_between(DAY_START, str(last_date))
    if not days:
        return None
    idx = max(0, len(days) - 1 - int(margin_days))
    return bj_date('{} 00:00:00'.format(days[idx]))


def _fetch_docs(src, target, frequency, code, floor, today, bj_market=None,
                floor_bound=None, drop_today=False, verbose=False, note=None):
    """取一只票一个频率的 bar → 8.3 文档。返回 ``(docs, status, reconnects)``。

    ⚠️ **每 task 一条新连接**（PITFALLS P3b：pytdx 一次失败调用会毒死整条连接）。
    """
    # 族感知的 market 解析（`000001` 在 stock/index 里是两只标的 —— 见 `tdx_market_of`）：
    # 指数 `000001` 是**沪市**上证指数（market 1），而股票 `000001` 是深市平安银行（market 0）。
    # 用错会**静默串数据**（拿到 11.71 这种"看起来合理"的银行价写进 index_*）。
    #
    # ⚠️ **必须在建连之前**：这一句是**纯查表**（不碰网络），而号段表里没有的 code
    # （指数的 `810`/`899` 等）会返回 None。排在建连之后时，那些 code **每个频率白建
    # 一条连接**再返回 `'empty'` —— 2026-10-09 修，正好命中 `index_*` 段。
    market = tdx_market_of(code, target)
    if market is None:
        return [], 'empty', 0                      # 号段表里没有 → 跳过，**不猜**

    api = src.new_api()
    reconnects = [0]

    def _reconnect():
        reconnects[0] += 1
        return src.new_api()

    try:
        if market == 2 and bj_market:
            market = bj_market                     # 北交所：用开跑前的探针结果
        total = bars_needed(floor, today, BARS_PER_DAY.get(frequency, 1) or 1)
        bars, status, api = bars_paged(
            api, market, code, TDX_CATEGORY[frequency], page_offsets(total),
            is_index=(target == 'index'), reconnect=_reconnect, verbose=verbose,
            note=note)
        if status == 'aborted':
            return [], status, reconnects[0]
        # 边界按**根**（不是按天）：已收盘的 bar 冻结，不重写。
        # 用 `bar_stamp`（unix 秒）比，避免再造一个 datetime 出来比时区。
        bound_sec = int(floor_bound.timestamp()) if floor_bound is not None else None
        if drop_today:
            # 未收盘：今天的 bar 是**当累累计**，一律不要（未来日期同理）
            bars = [b for b in bars if bar_date(b) < today]
        if bound_sec is not None:
            bars = [b for b in bars if bar_stamp(b, frequency) >= bound_sec]
        else:
            bars = [b for b in bars if bar_date(b) >= floor]
        # 列式构造（`kline_doc.bar_docs`）：实测比逐行 `bar_doc` 快 12×
        # （51.5 µs/根 → 4.3 µs/根），等价性由单测逐字段对拍。
        return dedup_docs(bar_docs(bars, code, market=target, frequency=frequency)),             status, reconnects[0]
    finally:
        try:
            api.disconnect()
        except Exception:                      # noqa: BLE001
            pass


def save_kline_tdx(targets=None, frequencies=None, codes=None, start_min=MIN_START_DEFAULT,
                   jobs=DEFAULT_JOBS, margin_days=DEFAULT_MARGIN_DAYS,
                   dry_run=False, progress_every=DEFAULT_PROGRESS_EVERY, verbose=True,
                   echo=None, on_progress=None, force_refresh=False):
    """`--save tdx` 的 K 线主体：**增量**取数并落 8.3（可用零参调用 —— PITFALLS P10）。

    :param targets: `TARGETS` 的子集；None = 全部
    :param frequencies: `FREQUENCIES` 的子集；None = 全部
    :param codes: 限定 code（试跑/排障）；None = 按 `universe()`
    :param start_min: 分钟线**首次**灌库的起点（库里没有该 code 时才用）
    :param margin_days: 窗口往前多算的交易日数（见模块 docstring）
    :param dry_run: **只取数不写库**，并与库内重叠窗口**逐字段对拍** —— 映射表的唯一硬标准
    :param progress_every: 每处理这么多 code 打一行进度（0 = 只打首尾）
    :param echo: 输出汇。默认 ``print``；`--save` 传的是 `Banner.echo` ——
        **banner 活跃时下方的一切输出都必须走它**，否则 banner 的行数记账会错位。
    :param on_progress: 可选回调 ``f(集合名, 'start'|'done', stats)``，给 banner 点亮节点用。
    :param force_refresh: **关掉短路**，每个 code 都真去取（`--save-refresh`）。
        短路的三道判据见 :func:`_process_code`；这一条是人工旁路。
    :returns: ``{'kline': {集合名: {...}}, 'universe': {族: n}}``
    """
    say = echo or print
    src = TdxSource()
    db = _db()
    today = datetime.now().strftime('%Y-%m-%d')

    # ⚠️ **未收盘就不写「今天」的日线**：pytdx 的 day 接口盘中返回的是**当日累计**
    # bar（实测 2026-10-09 10:14：600519 拿到 `10-09 15:00` vol=8784，而全天约 2.5 万）
    # —— 写进去就是**错误数据**，不是"正常现象"。
    #
    # 领域约定：盘中 day 不更新（盘中行情由 REALTIME 的 L1 tick 合成），**收盘后**跑
    # 本命令才写当天那根。窗口从"水位那根 bar"起算，所以收盘后重跑会覆盖它。
    _now = datetime.now()
    market_closed = (_now.hour, _now.minute) >= (15, 0)
    drop_today = (not market_closed) and _now.date().isoformat() == today

    targets = tuple(targets) if targets else TARGETS
    frequencies = tuple(frequencies) if frequencies else FREQUENCIES
    report = {'kline': {}, 'universe': {}}
    bj_market = _resolve_bj_market(src, codes, verbose=verbose, echo=echo)

    for target in targets:
        cols = universe(target, db=db, codes=codes, verbose=verbose)
        report['universe'][target] = len(cols)
        if verbose:
            say('[kline] {} 标的数 {}'.format(target, len(cols)))
        for frequency in frequencies:
            name = target_collection_name(target, frequency)
            coll = db[name]
            stats = {'codes': len(cols), 'inserted': 0, 'deleted': 0,
                     'skipped_codes': 0, 'empty_codes': 0, 'reconnects': 0,
                     'errors': [], 'diffs': {}, 'skipped_fresh': 0}
            report['kline'][name] = stats

            # 短路的四道判据（见 `_process_code` 与 DECISIONS D25/D26）：
            #   ① 不是「盘中 × 分钟频率」—— 盘中每分钟都有新数据，短路 = 不取新数据，
            #      而且前沿自证不会自己前进 ⇒ 会把集合冻在那一分钟
            #      （判据①，:func:`intraday_blocks_shortcircuit`）。
            #   ② TTL 闸放行？（该集合上次**全查**未超时；`--save-refresh` 一并关掉）
            #   ③ 集合自证探针 —— 集合里有没有 ts >= 「活动阈值」的 bar？
            #      有 ⇒ 二分出**数据前沿**，逐只 code 拿自己的水位和它比。
            #   ④ 该 code 自己的水位 >= 前沿（在 `_process_code` 里）。
            # ⚠️ 探针用 `ts`（timeField），**绝不能用 `time_stamp`**（实测差 8800×）。
            # ---- 收盘后的**快路径** ----
            # 「该集合**最近一次收盘之后**全查过」⇒ 整批已落库 ⇒ 整段跳过，
            # **连逐只读都不做**（省一轮全量 ~40s 的 last_bar）。
            # ⚠️ 它**比 TTL 硬**（TTL 只说"最近查过"，它说"最近一个已收盘的交易日整批
            # 都查过了"），所以这条路径**不看 TTL** —— 用户 2026-10-09 指出。
            if not dry_run and not force_refresh:
                reason = fast_skip_reason(coll, name, frequency, now=_now)
                if reason is not None:
                    # ⚠️ 这里**不能用 `total`** —— 它在本函数下面才定义
                    # （2026-10-09 实测踩过：`UnboundLocalError: total`，
                    # 而 `run_save` 的兜底 except 会把它报成"被用户终止"）。
                    stats['skipped_fresh'] = len(cols)
                    if on_progress is not None:
                        on_progress(name, 'start', stats)
                        on_progress(name, 'done', stats)
                    if verbose:
                        say('[kline] {} {}'.format(name, reason))
                    continue

            allow_skip = (not dry_run) and (not force_refresh) \
                and (not intraday_blocks_shortcircuit(frequency, _now)) \
                and allow_shortcircuit(name, now=_now)
            frontier = None
            if allow_skip:
                threshold = alive_threshold(frequency, _now)
                if threshold is None:
                    allow_skip = False        # 日历答不出来 ⇒ 不许跳
                else:
                    frontier = collection_frontier(coll, threshold, datetime.now(timezone.utc))
                    if frontier is None:
                        # 集合停住了（没有 >= 阈值 的 bar）⇒ **强制全查**，
                        # 这正是探针自动治停滞、不会死锁的地方。
                        allow_skip = False
                        if verbose:
                            say('[kline] {} 停在水位之前（{}）→ 本次全查'
                                .format(name, threshold.strftime('%Y-%m-%d %H:%M')))
                    elif verbose:
                        say('[kline] {} 前沿 {} → 已到前沿的 code 直接跳过'
                            .format(name, datetime.fromtimestamp(
                                frontier, timezone.utc).strftime('%Y-%m-%d %H:%M')))


            prog = {'done': 0, 'last': None, 'drop_today': drop_today,
                    'lock': threading.Lock(),
                    'notes': [],                 # 内层诊断，**条关了才吐**（见下）
                    't0': _time.time(), 't0_print': _time.time()}
            total = len(cols)
            every = int(progress_every or 0)

            def _note(msg, _p=prog):
                """内层（每 code）诊断的**缓冲口**。

                ⚠️ 这里**不能直接 `say`/`print`**：本函数跑在 tqdm 进度条活着的期间，
                而 banner 的表头压在进度条上面 —— banner 靠「光标上移 N 行」重画，
                多出来的行会让 N 算错，**进度条和表头一起花**。所以先攒着，
                `bar.close()` 之后再交 `say` 吐（那时光标正好在 banner 之下）。
                """
                with _p['lock']:
                    _p['notes'].append(msg)
                    del _p['notes'][:-20]        # 只留最近 20 条，别在坏网络下涨爆

            def _one(code, _coll=coll, _t=target, _f=frequency, _s=stats, _p=prog):
                bar.set_description(str(code))       # 条上报**当前 code**（见下）
                last = _process_code(src, _coll, _t, _f, code, start_min,
                                     margin_days, today, bj_market, dry_run, _s,
                                     verbose, drop_today=_p['drop_today'],
                                     note=_note, frontier=frontier,
                                     allow_skip=allow_skip)
                with _p['lock']:
                    _p['done'] += 1
                    if last:
                        _p['last'] = last
                    # **不 gate 在 verbose 上**：进度是"跑起来有没有在动"的唯一反馈，
                    # 不带 -v 也必须有（`verbose` 只加细节，不加"有没有"）。
                    pass

            # ⚠️ 起始行/完成行**只在 `-v` 下打**（用户 2026-10-09 明确）。
            # 原来刻意不 gate 在 verbose 上（「进度是'跑起来有没有在动'的唯一反馈」），
            # 但**现在有 banner 了** —— 它才是那个反馈，这两行反而变成逐集合的噪声。
            # 注意 `on_progress` **不 gate**：banner 靠它点亮节点。
            if total and verbose:
                say(progress_line(name, 0, total, None, stats, 0.0))   # 起始行
            if on_progress is not None:
                on_progress(name, 'start', stats)
            # 进度条用 tqdm（`disable=None` = 非 TTY 自动关，重定向到日志时不灌控制字符）。
            # `desc` 报**当前 code**，不报固定的集合名 —— 集合名 banner 上已经有了，
            # 跑的是哪一只才有用（与 `pytdx_source.fetch_stock_info` 同一口径）。
            # ⚠️ **`leave=False` 是承重的**：条留在屏上，banner 的下一次原地重画
            # 就会错位（`core/presentation.Banner` 靠光标上移记账）。
            bar = tqdm(total=total, unit='code', disable=None, leave=False)
            with ThreadPoolExecutor(max_workers=max(1, int(jobs))) as pool:
                for _ in pool.map(_one, cols):
                    bar.update(1)
            bar.close()

            # 进度条关掉之后才吐内层诊断 —— 此时光标恰好回到 banner 之下，
            # `say`/`echo` 的光标上移记账才是准的（见 `_note` 的说明）。
            if verbose and prog['notes']:
                for _msg in prog['notes']:
                    say(_msg)

            if total and verbose:                 # 完成行同起始行，`-v` 才打
                say('[kline] {} 完成：{}'.format(
                    name, ' | '.join(progress_summary(
                        stats, _time.time() - prog['t0']))))
            # ⚠️ **只有「真完整扫过一遍」才记账** —— 记账的口径是「这个集合上次**全查**
            # 于何时」。两条都排除：
            #   ① 短路过的那一遍**不算**（它只探了前沿、按水位跳了一批 code）——
            #      给它记账会把 TTL 无限推后，那道兜底网**永远不触发**；
            #   ② **限定 universe 的跑法**（`--save-codes`）**不算** —— 「全查」的字面意思
            #      就是**查了全部**。K 线这边其实有逐 code 水位兜底（没数据的 code
            #      `ts is None`，绝不跳），但复权的闸是**集合级**的、没有这层兜底，
            #      所以这条规则两边必须一致：**别让一次小范围试跑把闸打开**。
            if total and not allow_skip and not dry_run and codes is None:
                mark_kline_sweep(name)
            if on_progress is not None:
                on_progress(name, 'done', stats)
    src.close()
    return report


def _process_code(src, coll, target, frequency, code, start_min, margin_days,
                  today, bj_market, dry_run, stats, verbose, drop_today=False,
                  note=None, frontier=None, allow_skip=False):
    """单只票单频率：水位 → 取数 → （对拍 | 落库）。

    :param frontier: 该集合的**数据前沿**（unix 秒，:func:`collection_frontier` 的产物）；
        ``None`` = 没探到 ⇒ 不跳。
    :param allow_skip: TTL 闸是否放行（:func:`allow_shortcircuit`）。两道都成立才跳。
    :returns: ``(code, floor)`` —— 进度行要用它显示"正在处理谁、窗口从哪天起"
        （窗口能让人一眼看出增量算得对不对）。**每条返回路径都要给。**
    """
    ts, last_date = last_bar(coll, code)
    # **短路**：数据已到集合前沿 ⇒ **一个连接都不建**（这是整件事的目的，见 DECISIONS D25）。
    # 判据缺一不可：`allow_skip`（= TTL 没过期 **且** 不是「盘中×分钟」，见
    # `save_kline_tdx` 那两处与 :func:`intraday_blocks_shortcircuit`）、
    # `frontier is not None`（集合自证探到了）、`ts is not None`（从没数据的要走全量回填，
    # **绝不能跳**）。
    if allow_skip and frontier is not None and ts is not None \
            and _as_utc(ts).timestamp() >= frontier:
        stats['skipped_fresh'] += 1
        return code, last_date or ''
    if ts is None:
        floor = DAY_START if frequency == 'day' else start_min
        floor_bound = bj_date('{} 00:00:00'.format(floor))
    else:
        floor_bound = floor_ts(ts, last_date, margin_days)
        # 时区换算**只走项目既有的那一条路**（pandas `Asia/Shanghai`）——
        # 我一度为了"不引入 pandas"用固定 +8 偏移，那等于**又开了一个口径**，
        # 违反 `kline83`/`kline_doc` 反复强调的「时区换算只应有一处」。
        floor = pd.Timestamp(floor_bound).tz_convert('Asia/Shanghai').strftime('%Y-%m-%d')
    try:
        docs, status, reconnects = _fetch_docs(src, target, frequency, code, floor,
                                               today, bj_market=bj_market,
                                               floor_bound=floor_bound,
                                               drop_today=(drop_today and frequency == 'day'),
                                               note=note)
        stats['reconnects'] += reconnects
    except Exception as exc:                   # noqa: BLE001 单只票失败不拖垮其余
        stats['errors'].append('{} {} 取数失败: {!r}'.format(code, frequency, exc))
        return code, floor
    if status == 'aborted':
        stats['skipped_codes'] += 1
        stats['errors'].append('{} {} 重试耗尽，**整只跳过**（不写半截）'.format(code, frequency))
        return code, floor
    if not docs:
        stats['empty_codes'] += 1
        return code, floor

    if dry_run:
        _accumulate_diffs(coll, code, docs, stats)
        return code, floor

    # 只 drop 掉本次要写的那几根（冻结日不碰）—— 见 `writer.save_bar_chunk` 的说明
    res = save_bar_chunk(coll, docs, code=code)
    stats['inserted'] += res['inserted']
    stats['deleted'] += res['deleted']
    return code, floor


def _naive_utc(x):
    """统一到**naive UTC**：写出去的 `ts` 是 tz-aware，而 pymongo 读回来是 naive
    （`tz_aware=False` 是默认）—— 不归一就永远比不中（同一个瞬间，两种表示）。"""
    if x is None:
        return None
    if getattr(x, 'tzinfo', None) is None:
        return x
    return x.astimezone(timezone.utc).replace(tzinfo=None)


def _accumulate_diffs(coll, code, docs, stats):
    """dry-run：把本次要写的文档与库内同 `(code, ts)` 的行**逐字段**比。

    报告**不是**简单的差异计数：数值字段给「差异行数 + 最大相对误差 + 一个样例」——
    否则 `amount` 差最后几位 float32 舍入（相对 1e-7）会和 `vol` 差 100 倍长得一模一样。
    `ts` 不参与比对（它是连接键，已按它配对；且写出去是 tz-aware、读回来是 naive）。
    """
    lo = bj_date(min(d['date'] for d in docs) + ' 00:00:00')
    stored = {_naive_utc(r['ts']): r for r in coll.find(
        {'code': code, 'ts': {'$gte': lo}})}
    for d in docs:
        old = stored.get(_naive_utc(d['ts']))
        if old is None:
            continue
        stats['compared'] = stats.get('compared', 0) + 1
        for k, v in d.items():
            if k in ('_id', 'ts'):
                continue
            old_v = old.get(k)
            if old_v == v:
                continue
            if old_v is None or v is None:
                # **缺键单独记**：我方 day 行没有 `datetime`（存量没有）、反之亦然。
                # 早先把它并进 `_rel_diff` 会得到 max_rel=1，把一条精度差 1e-6 的
                # 正常对拍读成"完全不一致"（实测踩过 —— 报告误导比没有报告更坏）。
                m = stats.setdefault('missing_keys', {})
                m[k] = m.get(k, 0) + 1
                continue
            entry = stats['diffs'].setdefault(
                k, {'n': 0, 'max_rel': 0.0, 'sample': None})
            entry['n'] += 1
            rel = _rel_diff(v, old_v)
            if rel >= entry['max_rel']:
                entry['max_rel'] = rel
                entry['sample'] = (v, old_v)


def _rel_diff(a, b):
    """相对误差（两个都是数值才算；非数值字段只记"不同"）。"""
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        denom = max(abs(float(a)), abs(float(b)))
        return abs(float(a) - float(b)) / denom if denom else 0.0
    return 1.0


def _resolve_bj_market(src, codes, verbose=True, echo=None):
    """北交所（`82`/`92` 开头）该用哪个 market 号 —— **探针**，探不到就跳过它们。

    pytdx 的北交所行情要不要单独 market 号、这台服务器支不支持，**事先不知道**；
    而探针本身有代价（PITFALLS P3b：`get_security_list(2, 0)` 返回 None 会毒死连接），
    所以只用**一条独立连接、用完即弃**。

    :returns: 可用的 market 号（2 或 1）；都不行 → None（调用方跳过 82/92 的 code）
    """
    from .datasource.pytdx_kline import probe_market_bars
    say = echo or print
    sample = None
    if codes:
        sample = next((c for c in codes if _tdx_market_of(str(c)) == 2), None)
    if sample is None:
        sample = '920992'                  # 已知在 8.3 里有数据的北交所标的
    for market in (2, 1):
        api = src.new_api()
        try:
            if probe_market_bars(api, sample, market, TDX_CATEGORY['day']):
                if verbose:
                    say('[kline] 北交所探针：market={} 可用（样本 {}）'.format(market, sample))
                return market
        except Exception:                  # noqa: BLE001
            pass
        finally:
            try:
                api.disconnect()
            except Exception:              # noqa: BLE001
                pass
    if verbose:
        say('[kline] ⚠️ 北交所探针失败：market=2 与 1 都取不到 bar —— '
            '本次**跳过全部 82/92 开头的标的**（它们会停在最后一次成功写入的日期）')
    return None


def save_xdxr_tdx(codes=None, jobs=DEFAULT_JOBS, recompute_adj=True, verbose=True,
                  echo=None, target='stock'):
    """`{target}_xdxr`：逐 code 取**全历史**事件，**整 code 替换**（见 `replace_code_rows`）。

    触发 `{target}_adj` 重算的依据是 :func:`kline_doc.xdxr_adj_events_changed`
    （只看除权事件，不看「多了一条股东大会」）。

    :param target: ``'stock'`` 或 ``'etf'`` —— 决定写哪个集合
        （`stock_xdxr` / `etf_xdxr`）、用哪套 market 路由
        （:func:`tdx_market_of` 的族语义，ETF 是 ``5x``/``15x``/``16x``），
        以及宇宙从哪取（`stock_list` / `etf_list`）。
        ⚠️ **ETF 的 `etf_xdxr` 曾由 QMT 侧写**（`GQ_SU_save_etf_xdxr_qmt`），
        2026-10-09 起可由 pytdx 直供 —— **实测 `get_xdxr_info` 对 ETF 完全可用**
        （510300 给 15 条含 `除权除息`+`扩缩股`），且 `xdxr_doc` 转出的形态与
        `stock_xdxr` 逐字段相同。唯一缺的是 QMT 的 `dr` 字段（本路径不产出，
        复权因子改用 :func:`fq.xdxr_to_adj` 从事件自算，见 :func:`save_adj`）。
    :returns: ``{'codes':…, 'updated':…, 'events_changed':[…], 'errors':[…] }``

    ⚠️ **`recompute_adj` 是个死形参**（2026-10-09 核实：全树只有签名这一处出现）——
    它**不触发**复权重算，真正的重算由**调用方**在拿到 `events_changed` 之后
    单独调 :func:`save_adj`（见 `cli/commands/save.py`）。留着会让读者以为
    「xdxr 自己会连带重算」，故在此点名。**没删**是因为删它属接口变更，需所有者点头。
    """
    src = TdxSource()
    db = _db()
    say = echo or print
    coll = db['{}_xdxr'.format(target)]
    cols = universe(target, db=db, codes=codes)
    stats = {'codes': len(cols), 'updated': 0, 'unchanged': 0, 'skipped_codes': 0,
             'events_changed': [], 'errors': []}

    def _one(code):
        bar.set_description(str(code))
        # 每 task 一条新连接（PITFALLS P3b）—— 共享连接在多线程下等于把毒化扩散给所有线程
        api = src.new_api()
        try:
            rows = xdxr_rows(api, tdx_market_of(code, target), code)
        except Exception as exc:               # noqa: BLE001
            stats['errors'].append('{} xdxr 取数失败: {!r}'.format(code, exc))
            return
        if rows is None:
            stats['skipped_codes'] += 1
            stats['errors'].append('{} xdxr 连接已废，跳过'.format(code))
            return
        docs = dedup_docs([xdxr_doc(r, code) for r in rows])
        if not docs:
            return
        old = list(coll.find({'code': code}, {'_id': 0}))
        from .kline_doc import xdxr_adj_events_changed
        if not xdxr_adj_events_changed(old, docs):
            stats['unchanged'] += 1
            return
        stats['events_changed'].append(code)
        res = replace_code_rows(coll, docs, code=code)
        stats['updated'] += res['inserted']

    #: 进度条的口径与 K 线段**逐字相同**：`desc` = 当前 code、`leave=False`（条不留在屏上，
    #: banner 才能原地重画，见 `PITFALLS.md` P22）、`disable=None`（非 TTY 自动关）。
    #: ⚠️ 下面那句 `[xdxr] 更新 …` 的 `say` 必须在 `bar.close()` **之后** ——
    #: 条活着时写 stdout 会把 banner 的行数记账打乱。
    bar = tqdm(total=len(cols), unit='code', disable=None, leave=False)
    try:
        with ThreadPoolExecutor(max_workers=max(1, int(jobs))) as pool:
            for _ in pool.map(_one, cols):
                bar.update(1)
    finally:
        bar.close()
    src.close()
    if verbose:
        say('[xdxr] 更新 {} 行 / {} 只票事件有变 / {} 只未变 / {} 只跳过 / {} 错'.format(
            stats['updated'], len(stats['events_changed']), stats['unchanged'],
            stats['skipped_codes'], len(stats['errors'])))
    return stats


def adj_docs(code, dates, adj):
    """因子序列 → `stock_adj` 文档（形态与日线一致：`time_stamp == date_stamp` = 当日零点）。"""
    docs = []
    for d, a in zip(dates, adj):
        stamp = int(bj_date('{} 00:00:00'.format(str(d))).timestamp())
        docs.append({
            'code': code, 'date': str(d), 'date_stamp': stamp, 'time_stamp': stamp,
            'ts': datetime.fromtimestamp(stamp, tz=timezone.utc), 'adj': float(a),
        })
    return docs


def save_adj(codes, verbose=True, echo=None, target='stock'):
    """对给定 code **整条重算**前复权因子并落 `{target}_adj`。

    为什么必须整条重算（而不是只补新事件）：`{target}_adj` 是**前复权**因子，
    **最新一根恒为 1.0** —— 一旦有新事件，整条历史的基准都要重标。局部更新会留下
    「一半旧基准 + 一半新基准」，`to_qfq()` 会**静默**输出错价。

    三道护栏（前复权因子写坏 = 全链路静默错价，宁可拒绝）：
    1. **空则不写**（`replace_code_rows` 内建）；
    2. **不变量**：末日因子必须 == 1.0、所有因子必须 > 0，不满足就**拒绝该 code 并记录**；
    3. **只对事件变化的 code 调**（判据在 :func:`kline_doc.xdxr_adj_events_changed`）——
       调用方负责；本函数不自己判断，因为「事件变没变」要在写 `{target}_xdxr` **之前**取旧值。

    :param target: ``'stock'``（读 `stock_day`/`stock_xdxr` → 写 `stock_adj`）
        或 ``'etf'``（读 `etf_day`/`etf_xdxr` → 写 `etf_adj`）。**同一条公式**
        （:func:`fq.xdxr_to_adj`）服务两者 —— ETF 只是多一类
        ``category == 11 扩缩股`` 事件，那条 fq 里已按乘性口径处理。
    :returns: ``{'codes':…, 'written':…, 'refused':[…], 'skipped':[…] }``
    """
    db = _db()
    say = echo or print
    coll = db['{}_adj'.format(target)]
    day = db['{}_day'.format(target)]
    out = {'codes': len(codes or []), 'written': 0, 'refused': [], 'skipped': []}

    #: 进度条口径与 K 线 / xdxr 段一致（`desc` = 当前 code、`leave=False`、`disable=None`）。
    #: ⚠️ 循环体里有**三处 `continue`**（行数不足 / 因子全空 / 不变量不满足），
    #: 所以 `bar.update(1)` 必须放进 `finally` —— 否则这三种 code 会被**漏计**，
    #: 条永远到不了 100%（跑完停在 90% 那种，看着像卡住）。
    bar = tqdm(total=len(codes or []), unit='code', disable=None, leave=False)
    try:
        for code in (codes or []):
            bar.set_description(str(code))
            try:
                rows = sorted(day.find({'code': code}, {'_id': 0, 'date': 1, 'close': 1}),
                              key=lambda r: r['date'])
                if len(rows) < 2:
                    out['skipped'].append(code)
                    continue
                xdxr = list(db['{}_xdxr'.format(target)].find({'code': code}, {'_id': 0}))
                s = xdxr_to_adj([r['date'] for r in rows], [r['close'] for r in rows],
                                xdxr).dropna()
                if not len(s):
                    out['skipped'].append(code)
                    continue
                if abs(float(s.iloc[-1]) - 1.0) > 1e-9 or bool((s <= 0).any()):
                    out['refused'].append({
                        'code': code,
                        'reason': '不变量不成立：末日因子 {!r}'.format(float(s.iloc[-1]))})
                    continue
                out['written'] += replace_code_rows(
                    coll, adj_docs(code, list(s.index), list(s.values)),
                    code=code)['inserted']
            finally:
                bar.update(1)
    finally:
        bar.close()

    if verbose:
        say('[adj] 处理 {} code：写 {} 行，跳过 {}，**拒绝 {}**（不变量不满足）'.format(
            out['codes'], out['written'], len(out['skipped']), len(out['refused'])))
        for r in out['refused'][:5]:
            say('    ⚠️ {}'.format(r))
    return out


def verify_adj(codes, tol=1e-9, verbose=True):
    """把**重算结果**与存量 `stock_adj` 逐值比 —— 移植保真度的一次性核对。

    只对**事件没有变化**的 code 有意义（那种情形下重算必须与存量逐值相同）。
    实测抽样 120 只：**120 只全部 < 1e-9**（对齐细节见 `fq.xdxr_to_adj` 的说明）。

    :returns: ``{'checked':…, 'exact':…, 'worst':[(code, max相对差), …]}``
    """
    db = _db()
    day = db['stock_day']
    out = {'checked': 0, 'exact': 0, 'worst': []}
    for code in codes or []:
        rows = sorted(day.find({'code': code}, {'_id': 0, 'date': 1, 'close': 1}),
                      key=lambda r: r['date'])
        stored = {r['date']: r['adj'] for r in
                  db['stock_adj'].find({'code': code}, {'_id': 0, 'date': 1, 'adj': 1})}
        if len(rows) < 2 or not stored:
            continue
        xdxr = list(db['stock_xdxr'].find({'code': code}, {'_id': 0}))
        s = xdxr_to_adj([r['date'] for r in rows], [r['close'] for r in rows], xdxr)
        common = [d for d in s.index if d in stored and stored[d]]
        if not common:
            continue
        rel = max(abs(float(s[d]) - stored[d]) / abs(stored[d]) for d in common)
        out['checked'] += 1
        if rel <= tol:
            out['exact'] += 1
        else:
            out['worst'].append((code, rel))
    out['worst'].sort(key=lambda x: -x[1])
    if verbose:
        say('[adj] 保真度核对：{} 只可比，逐值相同 {} 只；偏差最大 {}'.format(
            out['checked'], out['exact'], out['worst'][:5]))
    return out
