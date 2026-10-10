# coding:utf-8
"""`stock_metadata_day` —— 日频元数据的**统一落点**。

为什么要有一张统一的表
======================
老树是「临时命名、一次一列」：`stock_ranking` / `stock_valuation` /
`stock_wencai_metadata` / `stock_metadata` …各自一张。后果是**加一列就要加一张集合**，
消费方要 join 五张表才拼出一个标的的当日画像。

本模块把它收成一张：**一行 = 一标的 × 一交易日**，列就是字段。加一列 = 加一个字段。

命名法（2026-10-10 定）
======================
* ``stock_metadata_day``   —— **日频**（本模块）
* ``stock_metadata_60min`` —— 小时频，**相当于旧库的 `stock_metadata`**

⚠️ 与本项目其它表的四处**刻意不同**
==================================
1. **普通集合，不是时间序列**。时间序列**不支持单文档 upsert**、**不能建唯一索引**
   （`PITFALLS.md` P14 实测；8.3 里 `stock_adj`/`stock_day` 的 `code_1_ts_1` 都是
   `unique=False`）—— 而本表要的正是「加一列就 upsert 一下」的幂等。
   日频的量也让时间序列的收益归零（≈136 万行/年），且 metadata 是**永久历史**、
   不需要 TTL。
2. **唯一键 `(code, date_stamp)`，没有 `revision`**。旧库 `stock_metadata_day` 的
   唯一索引是 ``(code, revision, date_stamp)`` 且 `revision` 恒为 `'chip_dist'`
   —— 那是历史包袱，本轮**丢弃**（用户 2026-10-10 明确）。
3. **`date_stamp` 是秒**（int），与 8.3 存量一致（`stock_day` 实测 `1791388800`）。
   写成毫秒会让唯一键差 1000 倍且**不报错**。
4. **同一事实存两列、不拼成一条序列** —— 见下。

⚠️ 换手率为什么是**两列**（而不是一列）
======================================
4.4 里有两个源都有换手率，**它们不是同一个量**（实测 2026-09 后 15,918 组同码同日）：

===============  ====================  ===================  ==========
源                字段                  覆盖                  口径
===============  ====================  ===================  ==========
`stock_ranking`   `TurnoverRate`        **2021 → 2026**       东财（按**自由流通股本**）
`stock_valuation` `turnover`            **1999 → 2026**       baostock（按**流通股本**）
===============  ====================  ===================  ==========

两者的相对差**中位 25.1% / p90 58.0%**，且**完全相等的 0 条**；
东财那列还被四舍五入到 **0.01% 精度**（0.0268 / 0.0081 …），baostock 是全精度。

⇒ **把两源接在时间轴上会在 2021 年留下一个定义性台阶**。对筹码分布来说，
衰减因子 `(1−换手率)^n` 逐日累积，接缝会变成**静默的系统性拐点**。
所以：**拼字段，不拼序列** —— 消费方显式选一列，口径切换成为看得见的决定。

* `TurnoverRate` ← 东财 `stock_ranking`（**旧树筹码分布吃的就是它**，对齐老结果用它）
* `turnover`     ← baostock `stock_valuation`（1999 起，全精度）

**两列都保留源端的原字段名**（用户 2026-10-10）—— 除了一致性，还有个硬理由：
`ChipDistribution.calcuChip` 读的就是 `FIELD.TURNOVER_RATE`（`'TurnoverRate'`），
**保持同名 ⇒ 那一侧代码一行都不用改**。两个源的名字本来就不同（`TurnoverRate` /
`turnover`），涵盖自带来源信息，不需要我另编 `_bs` 之类的后缀。

哪一列没有，**就是没有这个字段**（不写 0 —— 0 是「当天真的零换手」的意思）。

三条硬规矩（写坏了都不报错，所以写在这里）
==========================================
1. **只 `$set` 自己那几列**，永远不许整文档 replace —— 用的是
   :func:`GolemQ.datasource.writer.upsert_fields`，**不是** `save_collection`
   （后者走 `ReplaceOne`，会把别列的成果整片抹掉；实测对照见该函数 docstring）。
2. `date_stamp` 一律 `int()`，**换手率一律 `float()`** —— 源端若给成
   float 的 stamp（pandas 常干这事），存进去与存量的 int32 不相等，唯一键会插重复行。
3. **> ``MAX_TURNOVER_RATIO`` 的值判为脏数据**（丢 + 计数 + 报出），见其说明。
"""
from __future__ import annotations

import time

from GolemQ.core.constants import AKA, FIELD

from .kline83 import bj_date

#: 统一的日频元数据集合。库句柄见 :func:`_db`（不在本模块硬编码库名）。
METADATA_DAY_COLL = 'stock_metadata_day'

#: 换手率：东财口径（4.4 `stock_ranking.TurnoverRate`）。**旧树筹码分布的输入**。
TURNOVER_DC = FIELD.TURNOVER_RATE

#: 换手率：baostock 口径（4.4 `stock_valuation.turnover`）。历史长、全精度。
TURNOVER_BS = AKA.TURNOVER

#: 唯一键。**没有 `revision`** —— 见模块 docstring 第 2 条。
UNIQUE_KEYS = ('code', 'date_stamp')

#: 换手率的**物理上限**（小数）。用户 2026-10-10 给的判据：日换手率极限约 1.03–1.08，
#: **超过它基本就是「百分比没换算成小数」直接落库了**（例如 2.68 而非 0.0268）。
#:
#: 实测两个源**抽样 30 万条一条都没触发**（`stock_ranking` max = 0.8499、
#: `stock_valuation` max = 0.8634）—— 所以它是**护栏**，不是当前的数据清理。
#: 真触发时**丢该行并计数报出**，而不是写进去：写进去就是静默污染分布。
MAX_TURNOVER_RATIO = 1.08

#: 源清单：``(4.4 的集合名, 源字段名, 本表字段名, 源 date 的格式)``。
#: 顺序**不代表优先级** —— 两列互不覆盖，各写各的。
#:
#: 第 4 项（``'datetime'`` / ``'date'``）必须显式声明：两个源的 `date` 字符串
#: **格式不同**（`stock_ranking` 带时分秒、`stock_valuation` 是裸日期），
#: 而范围过滤是**字符串比较** —— 给裸日期补 `' 00:00:00'` 会把当天整段排掉。
SOURCES = (
    ('stock_ranking', 'TurnoverRate', TURNOVER_DC, 'datetime'),
    ('stock_valuation', 'turnover', TURNOVER_BS, 'date'),
)


def _date_range(fmt: str, since=None, until=None) -> dict:
    """按**源端 date 的格式**造范围条件。

    ⚠️ **不用 `date_stamp` 过滤** —— 实测 `stock_valuation` 里两套口径混着
    （2026-09-01：5131 行真实 UTC / 61 行墙上时间），按 stamp 划边界会**静默漏掉**
    边界那天的另一套。`date` 字段两种口径下都是同一个日历日，没有这个问题。

    >>> _date_range('datetime', since='2026-09-01', until='2026-09-30')
    {'$gte': '2026-09-01 00:00:00', '$lte': '2026-09-30 23:59:59'}
    >>> _date_range('date', since='2026-09-01', until='2026-09-30')
    {'$gte': '2026-09-01', '$lte': '2026-09-30'}
    >>> _date_range('date')
    {}
    """
    rng = {}
    if since:
        d = day_date(since)
        rng['$gte'] = '{} 00:00:00'.format(d) if fmt == 'datetime' else d
    if until:
        d = day_date(until)
        rng['$lte'] = '{} 23:59:59'.format(d) if fmt == 'datetime' else d
    return rng


def day_stamp(day) -> int:
    """``'YYYY-MM-DD'``（或带时分秒）→ **本表** `date_stamp`（秒，int）。

    ⚠️ 口径 = **8.3 自己的那套**：北京零点的**真实 UTC 时刻**（= 北京 00:00
    == UTC 前一日 16:00），由 :func:`kline83.bj_date` 给出。实测锚点：

    ==========================================  ============
    `stock_day` / `stock_adj` 里 ``date='2026-10-08'`` 的 `date_stamp`  1791388800
    本函数 ``day_stamp('2026-10-08')``                                  1791388800
    ==========================================  ============

    **为什么以 8.3 而不是源端为准**（2026-10-10 实测，三种口径并存）：

    =========================================  ==========================
    数据                                        口径
    =========================================  ==========================
    **8.3** `stock_day` / `stock_adj`            **真实 UTC**（本函数）
    4.4 `stock_ranking`（东财那列）              墙上时间当 UTC（差 +8h）
    4.4 `stock_valuation`（baostock 那列）       **两套混着**（2026-09-01：5131 真实UTC / 61 墙上）
    =========================================  ==========================

    本表住在 8.3，**与邻居一致**才能 join / 比较；照搬源端会把那个混乱带进来，
    而且 `date` 与 `date_stamp` 会**不是同一天**（valuation 那 1.2%）。用户
    2026-10-10 定：按 8.3 口径重算。`date` 字段原样保留，溯源不受影响。

    >>> day_stamp('2026-10-08')           # 与 stock_day 逐秒相同
    1791388800
    >>> day_stamp('2026-10-08 15:00:00')  # 时分秒被丢弃，只取当日
    1791388800
    """
    return int(bj_date('{} 00:00:00'.format(day_date(day))).timestamp())


def day_date(date_str) -> str:
    """源端日期 → ``'%Y-%m-%d'``。

    4.4 的 `stock_ranking.date` 带时分秒（``'2018-12-20 00:00:00'``），
    而 `stock_metadata_day.date` 不带（``'2017-01-04'``）。**两个格式混进同一列
    会让按 date 的 join 静默失配**，所以在入口统一。

    >>> day_date('2018-12-20 00:00:00')
    '2018-12-20'
    >>> day_date('2017-01-04')
    '2017-01-04'
    >>> day_date(None) is None
    True
    """
    if date_str is None:
        return None
    return str(date_str)[:10]


def metadata_day_doc(code, day, field, rate, created_at=None) -> dict:
    """一行元数据 → `stock_metadata_day` 文档（**只带一列载荷**）。

    字段集**照 4.4 的 `stock_metadata_day`**（`code` / `date` / `date_stamp` /
    `datetime` / `created_at`）+ 载荷列。额外多一个 `ts`（UTC-aware datetime）——
    8.3 其它表都有它、读层按它取轴，**这一处是对旧树的增补**，不是照抄。

    ⚠️ **`date_stamp` 由 `day` 重算，不取源端的值** —— 源端三种口径并存
    （见 :func:`day_stamp`），照搬会让 `date` 与 `date_stamp` 不是同一天。

    >>> d = metadata_day_doc('600519', '2026-10-09', 'TurnoverRate', 0.0123, created_at=0)
    >>> sorted(d)      # ⚠️ 大写 T 的码点小于小写 ⇒ `TurnoverRate` 排**最前**
    ['TurnoverRate', 'code', 'created_at', 'date', 'date_stamp', 'datetime', 'ts']
    >>> d['code'], d['date'], d['TurnoverRate']
    ('600519', '2026-10-09', 0.0123)
    >>> d['date_stamp'] == day_stamp('2026-10-09')   # 与 8.3 口径一致（北京时间当天的真实时刻）
    True
    >>> d['ts'].isoformat()          # 北京 00:00 == UTC 前一日 16:00
    '2026-10-08T16:00:00+00:00'

    带时分秒的 `day` 也收（源端两种格式），只取日期部分：

    >>> metadata_day_doc('600519', '2026-10-09 15:00:00', 'TurnoverRate', 0.1, 0)['date']
    '2026-10-09'
    """
    day = day_date(day)
    return {
        'code': str(code),
        'date': day,
        'date_stamp': day_stamp(day),
        'datetime': '{} 00:00:00'.format(day),
        'ts': bj_date('{} 00:00:00'.format(day)),
        field: float(rate),
        'created_at': int(created_at if created_at is not None else time.time()),
    }


def _db():
    """8.3 的库句柄。

    函数级导入（不是模块级）：`markets/StockCN/__init__.py` 里的
    `GOLEMQ_STOCK_CN` **定义在文件中部**，本模块在那一刻可能被间接导入 ——
    模块级 `from . import GOLEMQ_STOCK_CN` 会撞上半初始化的 `ImportError`。
    同层邻居（`kline_save.py` 的 `_db` / `refdata.py` 的 `_stock_cn_db`）都这么写。
    """
    from . import GOLEMQ_STOCK_CN
    return GOLEMQ_STOCK_CN


def turnover_rows(rows, src_field, dst_field, created_at=None) -> tuple:
    """某个源的源行 → `stock_metadata_day` 文档列表。

    **纯函数**（不碰 DB），所以整条搬运的正确性可以在没有 4.4 的情况下测。

    :param rows: 源行，每条至少含 ``code`` / ``date`` / `src_field`
        ⚠️ **要 `date`、不要 `date_stamp`** —— 本表的 stamp 由 `date` 按 8.3 口径重算
        （源端三种口径并存，见 :func:`day_stamp`）。
    :param src_field: 源端字段名（如 ``'TurnoverRate'``）
    :param dst_field: 本表字段名（如 ``'TurnoverRate'``）
    :param created_at: 统一写入的 `created_at`（秒）。``None`` = 当前时刻
    :return: ``(docs, skipped)`` —— `skipped` 是缺字段 / 非数字 / 越界而**被丢弃**的行数。
        它必须被计数并报出来：静默丢行会让「搬完了」这句话没有依据。

    缺值丢掉（**不写 0** —— 0 是「当天真的零换手」）：

    >>> rows = [
    ...   {'code': '600519', 'date': '2026-10-09 00:00:00', 'TurnoverRate': 0.0123},
    ...   {'code': '000001', 'date': '2026-10-09', 'TurnoverRate': None},
    ...   {'code': '000002', 'date': '2026-10-09', 'TurnoverRate': 'x'},
    ... ]
    >>> docs, skipped = turnover_rows(rows, 'TurnoverRate', 'TurnoverRate', created_at=0)
    >>> len(docs), skipped
    (1, 2)
    >>> docs[0]['code'], docs[0]['date'], docs[0]['TurnoverRate']
    ('600519', '2026-10-09', 0.0123)

    超过 :data:`MAX_TURNOVER_RATIO` 的判为「百分比没换算就落库了」，**丢并计数**：

    >>> turnover_rows([{'code': '1', 'date': '2026-10-09',
    ...                 'turnover': 2.68}], 'turnover', 'turnover')
    ([], 1)

    缺 `date` 的行也丢（stamp 推不出来）：

    >>> turnover_rows([{'code': '1', 'TurnoverRate': 0.01}], 'TurnoverRate', 'TurnoverRate')
    ([], 1)
    """
    docs, skipped = [], 0
    for r in rows:
        code = r.get('code')
        day = r.get('date')
        rate = r.get(src_field)
        if not code or not day or rate is None:
            skipped += 1
            continue
        try:
            rate = float(rate)
        except (TypeError, ValueError):
            skipped += 1
            continue
        # 换手率不为负；> 上限判为「百分比没换算」（见 MAX_TURNOVER_RATIO 的说明）
        if rate < 0 or rate > MAX_TURNOVER_RATIO:
            skipped += 1
            continue
        docs.append(metadata_day_doc(code, day, dst_field, rate, created_at))
    return docs, skipped


def _flush_batch(out, dst, buf, src_field, dst_field, upsert_fields, created_at) -> None:
    """把一批源行转文档并 upsert 进目标，累计统计。"""
    docs, skipped = turnover_rows(buf, src_field, dst_field, created_at=created_at)
    out['skipped'] += skipped
    if not docs:
        return
    stats = upsert_fields(dst, docs, list(UNIQUE_KEYS))
    out['upserted'] += stats['upserted']
    out['modified'] += stats['modified']
    out['written'] += len(docs)


def save_turnover(since=None, until=None, batch: int = 5000, verbose: bool = True,
                  echo=None, created_at=None, dry_run: bool = False) -> dict:
    """把 4.4 两源的换手率搬进 8.3 `stock_metadata_day`（**两列，各写各的**）。

    **一次性搬运**（走 :mod:`GolemQ.core.migrate44` 那条只读通道）。
    ⚠️ 运行时路径**不许**调它 —— 它连 4.4。

    为什么从 4.4 搬而不是爬 akshare：同源、一次搬完、且两个源**都已经是小数**，
    不需要任何换算。爬 akshare 要 5,573 次请求。

    :param since: 只搬 ``date >= since``（``'YYYY-MM-DD'``）。增量用。
    :param until: 只搬 ``date <= until``。
    :param batch: 每次从 4.4 读多少行。
    :param created_at: 统一写入的 `created_at`；``None`` = 当前时刻。
    :param dry_run: 只读源、只统计，**不写 8.3**（先量规模与丢弃量的用）。
    :returns: ``{源名: {'read','written','skipped','upserted','modified'}}``
    """
    from GolemQ.core.migrate44 import db44
    from GolemQ.datasource.writer import upsert_fields

    say = echo or print
    g44 = db44('golemq')
    dst = None if dry_run else _db()[METADATA_DAY_COLL]

    out = {}
    for coll_name, src_field, dst_field, date_fmt in SOURCES:
        src = g44[coll_name]
        # 每个源**按自己的 date 格式**造范围条件（见 `_date_range` 的说明）
        rng = _date_range(date_fmt, since, until)
        query = {'date': rng} if rng else {}
        stats = {'read': 0, 'written': 0, 'skipped': 0, 'upserted': 0, 'modified': 0}
        # ⚠️ 不取源端 `date_stamp` —— 本表的 stamp 由 `date` 按 8.3 口径重算
        proj = {'_id': 0, 'code': 1, 'date': 1, src_field: 1}
        cursor = src.find(query, proj).batch_size(batch)
        buf = []
        try:
            for r in cursor:
                stats['read'] += 1
                buf.append(r)
                if len(buf) >= batch:
                    if dst is not None:
                        _flush_batch(stats, dst, buf, src_field, dst_field,
                                     upsert_fields, created_at)
                    buf = []
            if buf and dst is not None:
                _flush_batch(stats, dst, buf, src_field, dst_field,
                             upsert_fields, created_at)
        finally:
            cursor.close()
        out[coll_name] = stats
        if verbose:
            say('[metadata_day] {}{} → {}: 读 {} / 写 {} / 丢 {} / upserted {} / modified {}'.format(
                coll_name, '.{}'.format(src_field), dst_field,
                stats['read'], stats['written'], stats['skipped'],
                stats['upserted'], stats['modified']))
    return out


def refresh_turnover(days: int = 7, verbose: bool = True, echo=None, **kwargs) -> dict:
    """**增量刷新**最近 `days` 个自然日的换手率（供 `--save` 调用）。

    ⚠️ **它连 4.4**（走 :mod:`GolemQ.core.migrate44`）。这条只允许出现在
    **取数侧**（`--save` / 一次性搬运），库代码与交易运行时不许调它 ——
    见 `core/migrate44.py` 的模块文档与 `DECISIONS.md` D34。

    为什么用 4.4 而不是 东财/baostock 的在线接口做刷新：**后两者都有访问频次限制**
    （用户 2026-10-10 明确），而 4.4 是同局域网的 Mongo 读、没有频次问题。
    等 4.4 下线，把 :data:`SOURCES` 指到在线适配器即可 —— **换源只改那一处**。

    :param days: 回溯的自然日数（含今天）。默认 7 天：够覆盖一个长假 + 补漏，
        且每天的 upsert 是幂等的，重叠跑没有代价。
    :returns: ``save_turnover`` 的统计；**4.4 不可达时**返回
        ``{'unavailable': <原因>}`` 而不是抛 —— 换手率是加分项，
        不该让整个 `--save` 失败（调用方负责把它说出来）。
    """
    from datetime import date, timedelta

    since = (date.today() - timedelta(days=days)).isoformat()
    try:
        return save_turnover(since=since, verbose=verbose, echo=echo, **kwargs)
    except Exception as exc:                      # noqa: BLE001
        reason = '{}: {}'.format(type(exc).__name__, exc)
        say = echo or print
        if verbose:
            say('[metadata_day] 4.4 不可达，跳过换手率刷新 —— {}'.format(reason))
        return {'unavailable': reason}
