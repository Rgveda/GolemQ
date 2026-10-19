# coding:utf-8
"""`stock_reality_feat_day` —— 日频元数据的**统一落点**。

为什么要有一张统一的表
======================
老树是「临时命名、一次一列」：`stock_ranking` / `stock_valuation` /
`stock_wencai_metadata` / `stock_metadata` …各自一张。后果是**加一列就要加一张集合**，
消费方要 join 五张表才拼出一个标的的当日画像。

本模块把它收成一张：**一行 = 一标的 × 一交易日**，列就是字段。加一列 = 加一个字段。

命名法（2026-10-10 定）
======================
* ``stock_reality_feat_day``   —— **日频**（本模块）
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

from GolemQ.core.constants import AKA, FIELD, MARKET_TYPE

from .kline83 import bj_date
from .reality_save import reality_feat_name

#: 日频元数据**落哪张表** —— ⚠️ **唯一出处是 `reality_save.reality_feat_name`**
#: （那个函数有 doctest，且 `*_reality_feat_*` 六族的命名规则都归它管）。
#: **本模块不再自己拼这个名字** —— 拼法分叉的症状是**静默写到另一张表**。
#:
#: ⚠️ **2026-10-16 起改名**（原名 `METADATA_DAY_COLL`，值 `'stock_metadata_day'`）：
#: 日频元数据与日频 reality 特征**同住这一张表**（`DECISIONS.md` D50）。
#: 改名的理由：常量名与值必须一致 —— 一个叫 `REALITY_FEAT_DAY_COLL` 却等于
#: `'stock_reality_feat_day'` 的常量就是**撒谎的常量**，比不改更坏。
REALITY_FEAT_DAY_COLL = reality_feat_name('day', MARKET_TYPE.STOCK_CN)

#: 合并**之前**的旧集合名。
#: ⚠️ **该集合已于 2026-10-17 drop（D50）—— 它现在不存在了。**
#: 本常量保留只是给 :func:`merge_metadata_into_feat` / :func:`verify_metadata_merge`
#: 作记录；**重跑那两个函数会读到空表**。新代码里**不许**用它 ——
#: 读要读 :data:`REALITY_FEAT_DAY_COLL`。
LEGACY_METADATA_DAY_COLL = 'stock_metadata_day'

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
    而 `stock_reality_feat_day.date` 不带（``'2017-01-04'``）。**两个格式混进同一列
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
    """一行元数据 → `stock_reality_feat_day` 文档（**只带一列载荷**）。

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
    return metadata_day_doc_multi(code, day, {field: rate}, created_at=created_at)


def metadata_day_doc_multi(code, day, fields, created_at=None) -> dict:
    """一行**多列**元数据 → 文档。:func:`metadata_day_doc` 是它的单列特例。

    存在的理由：`stock_reality_feat_day` 是「一行 = 一标的 × 一交易日，**列就是字段**」，
    而爬来的数据一次就有多列（千股千评一次 9 列）。单列版要写 9 次文档、
    `$set` 9 遍同一行的公共键 —— 多列版一条文档解决。

    ⚠️ **值为 `None` / NaN 的列直接跳过，不写 `None`** —— 与换手率那条
    「哪一列没有就不写，**不写 0**」是同一个口径：写 `None` 会让下游分不清
    「没这个数据」与「这个数据是空」。

    ⚠️ 公共键的形状**必须与 :func:`metadata_day_doc` 逐字一致**（`code` / `date` /
    `date_stamp` / `datetime` / `ts` / `created_at`）—— 单列版已委托到本函数，
    所以这条由它的 doctest 自动守住。

    >>> d = metadata_day_doc_multi('301308', '2026-10-09', {'PERation': 4.38, 'Ranking': 1302}, created_at=0)
    >>> d['PERation'], d['Ranking'], d['date_stamp'] == day_stamp('2026-10-09')
    (4.38, 1302.0, True)
    >>> 'ZLCB' in metadata_day_doc_multi('301308', '2026-10-09', {'ZLCB': None}, 0)   # None 不写
    False
    """
    day = day_date(day)
    doc = {
        'code': str(code),
        'date': day,
        'date_stamp': day_stamp(day),
        'datetime': '{} 00:00:00'.format(day),
        'ts': bj_date('{} 00:00:00'.format(day)),
        'created_at': int(created_at if created_at is not None else time.time()),
    }
    for field, value in (fields or {}).items():
        if value is None:
            continue
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if value != value:          # NaN（唯一不自等的浮点）
            continue
        doc[field] = value
    return doc


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
    """某个源的源行 → `stock_reality_feat_day` 文档列表。

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


def _flush_batch(out, db, buf, src_field, dst_field, created_at) -> None:
    """把一批源行转文档，经**统一写口** `save_reality_feat` 落库。"""
    import pandas as pd

    from .reality_save import save_reality_feat

    docs, skipped = turnover_rows(buf, src_field, dst_field, created_at=created_at)
    out['skipped'] += skipped
    if not docs:
        return
    frame = pd.DataFrame(docs).set_index(['date', 'code'])
    stats = save_reality_feat(frame, 'day', MARKET_TYPE.STOCK_CN,
                              verbose=False, created_at=created_at, db=db)
    out['upserted'] += stats['upserted']
    out['modified'] += stats['modified']
    out['written'] += len(docs)


def save_turnover(since=None, until=None, batch: int = 5000, verbose: bool = True,
                  echo=None, created_at=None, dry_run: bool = False) -> dict:
    """把 4.4 两源的换手率搬进 8.3 `stock_reality_feat_day`（**两列，各写各的**）。

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

    say = echo or print
    g44 = db44('golemq')
    dst = None if dry_run else _db()

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
                                     created_at)
                    buf = []
            if buf and dst is not None:
                _flush_batch(stats, dst, buf, src_field, dst_field,
                             created_at)
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

    ⚠️ **不再连 4.4**（2026-10-18 起，用户：「stock_cn 中使用 4.4 旧树的方法都切换成 8.3」）。
    ==========================================================================
    过去这里走 :data:`SOURCES` → `save_turnover` → :mod:`GolemQ.core.migrate44`
    **从 4.4 读**换手率。现在改成 **8.3 现算**：直接调 :func:`fill_turnover`
    （`成交量 ÷ 流通股本`，纯 8.3 数据：`stock_day.volume` + `stock_info.liutongguben`）。

    口径没变：:func:`turnover_from_volume` 与 baostock 的 `turnover` **逐位相同**
    （对拍表见该函数），落的是同一列 `turnover`。只是**来源从 4.4 换成计算** ——
    于是 `--save` 不再依赖 4.4 可达。

    :param days: 回溯的自然日数（含今天）。默认 7 天：够覆盖一个长假 + 补漏，
        且每天的 upsert 是幂等的，重叠跑没有代价。
    :returns: :func:`fill_turnover` 的统计（``{'codes','filled','skipped',
        'upserted','modified'}``）；**出错时**返回 ``{'unavailable': <原因>}``
        而不是抛 —— 换手率是加分项，不该让整个 `--save` 失败。
    """
    from datetime import date, timedelta

    since = (date.today() - timedelta(days=days)).isoformat()
    try:
        # `codes=None` ⇒ 全市场（`refdata.GQ_fetch_stock_list`）
        return fill_turnover(None, start=since, verbose=verbose, echo=echo, **kwargs)
    except Exception as exc:                      # noqa: BLE001
        reason = '{}: {}'.format(type(exc).__name__, exc)
        say = echo or print
        if verbose:
            say('[metadata_day] 换手率现算失败，跳过 —— {}'.format(reason))
        return {'unavailable': reason}


# --------------------------------------------------------------------------- #
# 补齐换手率：`成交量 / 流通股本` **现算**
# --------------------------------------------------------------------------- #
#: 补算换手率的**公式出处**：老树 `analysis/ChipDistribution_jit.py` 的三处内联
#: （`:785` / `:819` / `:865`）与 `markets/StockCN/realtime.py` 的实时分支。
#: 2026-10-13 起**只此一份** —— 筹码分布那段自补逻辑已删（`DECISIONS.md` D44），
#: `realtime.py` 改调本函数。
#:
#: ⚠️ **老树那三处公式多除了一个 100，是错的**（见 :func:`turnover_from_volume` 的
#: 「量纲」一节）。这里给的是**修正后**的式子 —— 不是照搬。
#:
#: 口径：按**流通**股本 —— 与表里 `turnover`（baostock）**同一个量**，与
#: `TurnoverRate`（东财，按**自由**流通股本）**不是**。所以补算值写 `turnover` 那列。
FILL_THRESHOLD = 0.96


def turnover_from_volume(vol, liutongguben, threshold: float = FILL_THRESHOLD):
    """`成交量(手) / 流通股本(股)` → 换手率。**纯函数**。

    量纲（实测，不是推的）
    ====================
    `kline83` 的日线 `volume` 以**手**计（`amount / volume` 恒是股价的 100 倍），
    而 `stock_info.liutongguben` 以**股**计（600519 = 1,250,081,562.5）⇒ 换算成股是 ``vol * 100``。

    用 600519 六个交易日对拍 baostock 的 `turnover`（`stock_reality_feat_day`）逐位相同：

    ==========  =======  ==========  ==================
    日           volume   本函数       baostock `turnover`
    ==========  =======  ==========  ==================
    2026-09-28   28218    0.002257     0.0022572926
    2026-09-29   26366    0.002109     0.0021091423
    2026-09-30   38330    0.003066     0.0030662799
    ==========  =======  ==========  ==================

    ⚠️ **老树写的是 `(vol * 100 / liutongguben) / 100`，多除了一个 100 ⇒ 恒小 100 倍**
    （同一天它给 2.3e-05，而真值是 0.002257）。三处内联（`ChipDistribution_jit.py:785`
    / `:819` / `:865`）与 `realtime.py` 那份**全都是这个式子**。没被发现是因为
    进 `get_data` 的唯一路径 `calc_stock_chip_distribution_jit` **全树无人调用**
    （2026-10-13 核；那段自补逻辑随之删掉，见 `DECISIONS.md` D44）。
    **这里不照搬它** —— 照搬等于把一条已知错误落进库里。

    :param threshold: 超阈值就判「`vol` 已经是**股**」（而不是手），改用 ``vol / liutongguben``。
        老树的阈值不一致（`:786` 用 `0.96`、`:820`/`:866` 用 `1.0618`、`realtime.py` 用 `0.96`），
        这里取 `0.96` —— 与「缺今天」那条主路径、以及实时分支一致。

    >>> turnover_from_volume(28218, 1250081562.5)          # 600519 / 2026-09-28
    0.002257

    `vol` 已经是**股**时（值会大 100 倍）触发修正：

    >>> turnover_from_volume(1e6, 1e9, threshold=0.01)     # 0.1 超阈值 ⇒ 去掉 ×100
    0.001

    股本取不到时**返 `None`，不编一个数出来**（调用方负责跳过）：

    >>> turnover_from_volume(1e7, 0), turnover_from_volume(1e7, float('nan'))
    (None, None)
    """
    if not liutongguben or liutongguben <= 0 or liutongguben != liutongguben:
        return None
    rate = round(vol * 100 / liutongguben, 6)      # vol 以**手**计
    if rate > threshold:                           # vol 以**股**计 ⇒ 去掉那个 ×100
        rate = round(vol / liutongguben, 6)
    return rate


def fill_turnover(codes, start=None, end=None, verbose: bool = True, echo=None,
                  created_at=None) -> dict:
    """把「成交量 ÷ 流通股本」**现算**的换手率补进 `stock_reality_feat_day`。

    为什么要它：`analysis/ChipDistribution_jit.py` 现在**不再自己补算**换手率
    （缺就抛，见 `DECISIONS.md` D44），补齐这件事归这里。最典型的场景是
    「`--save` 先写 K 线、后写元数据」之间那段窗口 —— 盘中跑筹码分布必然缺当日换手率。

    只补**真的缺**的日子：`TurnoverRate`（东财）与 `turnover`（baostock）**任一列有值**
    就算覆盖（两列口径与覆盖期都不同，取并集才是「这天有换手率」的正解）。

    **补算值写 `turnover` 那列**（不是 `TurnoverRate`），两个理由：

    1. **口径同源** —— :func:`turnover_from_volume` 按**流通**股本，与 baostock 的
       `turnover` 是同一个量（实测逐位相同，见该函数的对拍表）；写进按**自由**流通股本
       的东财列就是混口径。
    2. **筹码分布先读它** —— `ChipDistribution.get_data` 的**第一读**取的就是 `turnover`
       （再 rename 进内存的 `TurnoverRate`），第二读才拿东财列补缺口。

    日后 `refresh_turnover` 从 4.4 / 在线源拿到真值会自然覆盖它。

    :param codes: 6 位代码或列表。**`None` ⇒ 全市场**（`refdata.GQ_fetch_stock_list`）——
        「全市场是哪些」属于市场层，不该由 CLI 或调用方自己拼。显式给**空列表**仍报错：
        那是调用方的 bug，不是「全市场」的意思。
    :param start,end: ``'YYYY-MM-DD'``（**含**当日）。``None`` = 不设边界。
    :param created_at: 统一写入的 `created_at`（秒）；``None`` = 当前时刻。
    :returns: ``{'codes','filled','skipped','upserted','modified'}``
        （`filled` 是**写进去的天数**，`skipped` 是缺量/缺股本/越界而没写的天数）
    """
    from GolemQ.core.market_registry import get_active_market
    import pandas as pd

    from GolemQ.datasource.writer import upsert_fields  # noqa: F401 其他写口仍在用

    from .refdata import GQ_fetch_stock_info, GQ_fetch_stock_list, GQ_fetch_stock_reality_feat_day
    from .reality_save import save_reality_feat

    say = echo or print
    if codes is None:
        df = GQ_fetch_stock_list()
        codes = [] if df is None or len(df) == 0 else [str(c)[:6] for c in df['code']]
        if not codes:
            raise ValueError('全市场名单为空 —— 先跑一次 `--save tdx` 灌 `stock_list`')
    if isinstance(codes, str):
        codes = [codes]
    codes = [str(c)[:6] for c in codes]
    if not codes:
        raise ValueError('fill_turnover 的 `codes` 是空列表：不知道补谁就无从补起'
                         '（要全市场请传 `None`）')

    market = get_active_market()
    db = _db()
    out = {'codes': len(codes), 'filled': 0, 'skipped': 0, 'upserted': 0, 'modified': 0}

    for code in codes:
        # ① 现有覆盖：两列任一有值就算这天有了
        have = GQ_fetch_stock_reality_feat_day(code, start=start, end=end)
        covered = set()
        cols = [c for c in (TURNOVER_DC, TURNOVER_BS)
                if have is not None and len(have) > 0 and c in have.columns]
        if cols:
            ok = have[cols].notna().any(axis=1).to_numpy()
            covered = {str(d)[:10] for d in have.index.get_level_values(0)[ok]}

        # ② 流通股本：补算的分母，取不到就只能整只跳过（不编数）
        info = GQ_fetch_stock_info([code])
        guben = None
        if info is not None and len(info) > 0 and 'liutongguben' in info.columns:
            guben = info['liutongguben'].iloc[0]
        if guben is None or guben != guben or guben <= 0:
            out['skipped'] += 1
            if verbose:
                say(f'[fill_turnover] {code}: 取不到 liutongguben，跳过')
            continue

        # ③ 日线（`realtime=True` —— 要补的常常正是**今天**那根）
        result, _name = market.get_kline_price_v3(
            code, start=start, end=end, verbose=False, realtime=True)
        frame = None if result is None else getattr(result, 'data', None)
        if frame is None or len(frame) == 0:
            out['skipped'] += 1
            if verbose:
                say(f'[fill_turnover] {code}: 取不到日线，跳过')
            continue

        rows = []
        for row in frame.reset_index().to_dict('records'):
            day = str(row.get('ts'))[:10]
            if day in covered:
                continue
            rate = turnover_from_volume(row.get(AKA.VOLUME), guben)
            if rate is None or rate < 0 or rate > MAX_TURNOVER_RATIO:
                out['skipped'] += 1
                continue
            rows.append({'date': day, 'code': code, TURNOVER_BS: rate})

        if rows:
            # 统一入口：`save_reality_feat`（D55）
            df_rows = pd.DataFrame(rows).set_index(['date', 'code'])
            stats = save_reality_feat(df_rows, 'day', MARKET_TYPE.STOCK_CN,
                                      verbose=False, created_at=created_at, db=db)
            out['filled'] += len(rows)
            out['upserted'] += stats['upserted']
            out['modified'] += stats['modified']
        if verbose:
            say('[fill_turnover] {}: 补 {} 天 / 未覆盖共 {} 天'.format(
                code, len(rows), len(covered)))
    return out


def merge_metadata_into_feat(batch: int = 5000, limit=None, dry_run: bool = False,
                             verbose: bool = True, echo=None) -> dict:
    """把 :data:`LEGACY_METADATA_DAY_COLL` 的行**原样**搬进 :data:`REALITY_FEAT_DAY_COLL`。

    **一次性**（`DECISIONS.md` D50）。⚠️ 这是 **8.3 内部**的合并，**不连 4.4** ——
    与同目录的 :func:`save_turnover`（4.4 → 8.3）不是一回事。

    幂等
    ====
    只 `$set`（走 :func:`GolemQ.datasource.writer.upsert_fields`），键是
    :data:`UNIQUE_KEYS` = `(code, date_stamp)` —— 两表的唯一索引**同名同键**
    （实测都是 `code_1_date_stamp_1`），所以**重跑不会插重复行**。
    **中途挂了直接再跑一遍即可**，不需要断点续传。

    ⚠️ **绝不能用 `save_collection`**（`ReplaceOne` 整文档替换）—— 目标表里
    已有 reality 特征列（`ClsMaj` 等，2026-10-13 起由 `reality_save` 写），
    整文档替换会把它们**静默抹掉**。这正是 `upsert_fields` 存在的理由。

    :param batch: 每批 upsert 多少行
    :param limit: 只搬前 N 行（**试跑用**；`None` = 全量）
    :param dry_run: 只读源、只计数，**不写目标表**
    :param verbose: 报进度
    :param echo: 输出汇（默认 `print`）—— 见 CLI 那条「banner 活跃时输出要走 echo」
    :returns: ``{'read': n, 'written': n, 'skipped': n, 'upserted': n, 'modified': n}``
    """
    import pandas as pd

    say = echo or print
    db = _db()
    src = db[LEGACY_METADATA_DAY_COLL]
    from .reality_save import save_reality_feat

    stats = {'read': 0, 'written': 0, 'skipped': 0, 'upserted': 0, 'modified': 0}
    # ⚠️ `no_cursor_timeout=True`：全量 1,700 万行要跑几十分钟，默认游标
    # （MongoDB 10 分钟空闲超时）会**在跑到一半时**把游标掐掉。
    # ⚠️ **只投影掉 `_id`**，其余字段**一个不改名、不换算** —— 这是"原样搬"。
    cursor = src.find({}, {'_id': 0}, no_cursor_timeout=True)
    if limit:
        cursor = cursor.limit(int(limit))
    try:
        chunk = []
        for doc in cursor:
            stats['read'] += 1
            if not all(k in doc for k in UNIQUE_KEYS):
                stats['skipped'] += 1      # 缺键 ⇒ 落进去会插出无键文档
                continue
            chunk.append(doc)
            if len(chunk) >= batch:
                stats['written'] += len(chunk)
                if not dry_run:
                    r = save_reality_feat(
                        pd.DataFrame(chunk).set_index(['date', 'code']),
                        'day', MARKET_TYPE.STOCK_CN, verbose=False,
                        write_nulls=True, db=db)
                    stats['upserted'] += r['upserted']
                    stats['modified'] += r['modified']
                chunk = []
                if verbose and stats['written'] % (batch * 20) == 0:
                    say('[merge] 已搬 {:,} / 读 {:,}'.format(
                        stats['written'], stats['read']))
        if chunk:
            stats['written'] += len(chunk)
            if not dry_run:
                r = save_reality_feat(
                    pd.DataFrame(chunk).set_index(['date', 'code']),
                    'day', MARKET_TYPE.STOCK_CN, verbose=False,
                    write_nulls=True, db=db)
                stats['upserted'] += r['upserted']
                stats['modified'] += r['modified']
    finally:
        cursor.close()

    if verbose:
        say('[merge] {} → {}{}：读 {:,} / 写 {:,} / 丢弃 {:,}'.format(
            LEGACY_METADATA_DAY_COLL, REALITY_FEAT_DAY_COLL,
            '（**干跑，没有写库**）' if dry_run else '',
            stats['read'], stats['written'], stats['skipped']))
    if stats['skipped']:
        say('[merge] ⚠️ 丢弃 {} 行（缺 {}）'.format(stats['skipped'], UNIQUE_KEYS))
    return stats


def verify_metadata_merge(sample: int = 20000, verbose: bool = True, echo=None) -> dict:
    """**只读**核对：抽样比对源表与目标表在 `(code, date_stamp)` 上是否逐字段相同。

    ⚠️ 合并跑完**必须**跑这个再谈 drop —— 「搬完了」这句话得有依据。

    抽样的取法：**从源表尾部取** `sample` 行（`$natural:-1`）。为什么不用随机：
    尾部是最近写入的、也是最可能被后搬覆盖的；而随机抽样在 1,700 万行上要走全表。
    两种都能漏，但尾部抽样**至少覆盖到边界那一批**。

    :returns: ``{'checked': n, 'missing': n, 'mismatch': n, 'examples': [...]}``
    """
    say = echo or print
    db = _db()
    src, dst = db[LEGACY_METADATA_DAY_COLL], db[REALITY_FEAT_DAY_COLL]

    out = {'checked': 0, 'missing': 0, 'mismatch': 0, 'examples': []}
    # 源表**不含** `_id`/`ts` 之外的目标表独有列比对 —— 这里只比**源表有的字段**
    for doc in src.find({}, {'_id': 0}).sort('$natural', -1).limit(int(sample)):
        key = {k: doc[k] for k in UNIQUE_KEYS if k in doc}
        if len(key) != len(UNIQUE_KEYS):
            continue
        out['checked'] += 1
        got = dst.find_one(key, {'_id': 0})
        if got is None:
            out['missing'] += 1
            if len(out['examples']) < 5:
                out['examples'].append(('missing', key))
            continue
        for field, want in doc.items():
            if field not in got:
                out['mismatch'] += 1
                if len(out['examples']) < 5:
                    out['examples'].append(('field-absent', key, field))
                break
            have = got[field]
            # ⚠️ 浮点**按位**比：搬家不该动任何一个 bit。差一个 bit 就是搬错了。
            if isinstance(want, float) or isinstance(have, float):
                same = (want == have) or (want != want and have != have)
            else:
                same = want == have
            if not same:
                out['mismatch'] += 1
                if len(out['examples']) < 5:
                    out['examples'].append(('value', key, field, want, have))
                break
    if verbose:
        say('[verify] 抽 {:,} 行：缺 {:,} / 不符 {:,}'.format(
            out['checked'], out['missing'], out['mismatch']))
        for ex in out['examples']:
            say('    {}'.format(ex))
    return out
