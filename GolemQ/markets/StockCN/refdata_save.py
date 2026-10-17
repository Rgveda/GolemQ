# coding:utf-8
"""A 股参考集合的取数与落库编排 —— 对应 CLI 的 `--save <SOURCE>` 的参考数据段。

为什么在 `markets/StockCN/` 而不是 `pipeline/`
=============================================
这 5 个集合（`stock_list`/`stock_info`/`etf_list`/`stock_block`/`financial`）
**全是 A 股特有概念**：源是通达信/东财/新浪，schema 是 A 股口径，目标是 A 股库。
它们的获取逻辑属于市场本身，不属于通用数据加工。

`pipeline/` 的实际内容是 `base.py` + 三个 benchmark —— 那是**数据加工流水线**
（特征计算、基准对比），不是数据获取。CLI 里已有的先例也印证这一点：
`--stock-min-aligned` 委托给的是 `markets/StockCN/align.py`。

CLAUDE.md 那条「CLI 只做参数校验再委托」的重点是**别把业务逻辑留在 CLI 层**，
而非「必须放 pipeline」。

为什么不并进 `refdata.py`
=========================
`refdata.py` 是**读取叶子**，只依赖 `core.settings` 与 `pandas`，谁都能轻量
import 它。本模块要用 `datasource/` 整包与 `writer`，并进去会让读路径被迫
拖进整个适配层与网络依赖。读写分开。

设计要点
========
**取数与落库分离。** 适配器（`datasource/`）只产出行、不碰 DB；落库由
`datasource/writer.py` 完成。本模块把两者接起来并决定用哪个源。

**不支持的集合要显式报错，不能静默跳过。** 用户点的是"保存全部"，若某个
集合没有可用源，静默跳过会让人以为存过了 —— 这正是本次重构要消除的那类失败。
故返回结果里逐集合标注 `status`（ok / skipped / failed）。
"""
from __future__ import annotations

import datetime as dt

from . import GOLEMQ_STOCK_CN
from GolemQ.datasource.base import DataSourceNotAvailable, UnsupportedCollection
from GolemQ.datasource.writer import save_block_collection, save_collection

# 实现层：导入即触发 A 股各适配器自注册，并给出该市场的源优先级
from .datasource import COLLECTION_SOURCE_PRIORITY, get_source

__all__ = [
    'ALL_REF_COLLECTIONS',
    'REFDATA_BY_SOURCE',
    'UNIQUE_KEYS',
    'DELTA_DELETE',
    'MIN_STOCK_CODES',
    'SNAPSHOT_COLLECTIONS',
    'SNAPSHOT_BLOCKED_PREFIXES',
    'snapshot_universe',
    'save_refdata',
    'save_index_list_from44',
    'refresh_stock_codes_cache',
    'refdata_status',
    'format_status',
]

#: **6 个**参考集合
#: ⚠️ 这份清单与 `refdata.REF_COLLECTIONS` 是**两份平行清单**（那里管「读」、这里管「写」）——
#: 两边必须同步。`index_list` 是 2026-10-11 一起加进去的第 6 个。
ALL_REF_COLLECTIONS = ('stock_list', 'stock_info', 'etf_list', 'index_list',
                       'stock_block', 'financial')

#: `--save <SOURCE>` 里各源**真正能供**的集合。**按源不同** —— 别硬凑成一个元组：
#: `qmt` 适配器压根没声明 `etf_list`，列进去只会每次报一遍 `UnsupportedCollection`。
#:
#: ⚠️ `etf_list` 为什么在 tdx 侧却由 **akshare** 供（实测 2026-10-09）：
#: `tdxaidata` 的 `get_trackzs_etf_info` 返回 **0 行 + `[错误码 2] 股票代码错误`**，
#: 那条路是坏的；可用源只有 akshare —— **一次调用、1694 行、1.4 秒**。
#: 别把它和 `financial` 混为一谈：当初「避开 akshare」避的是 `financial` 的**逐只**
#: 路径（5000+ 只 × 30s ≈ 46.5 小时），本命令从不碰它。
#: 另注：akshare 只出现在 `etf_list` / `financial` 两个集合的优先级里，所以
#: `--save` **不传** `exclude_sources` —— 对另外三个集合那本就是个空操作。
REFDATA_BY_SOURCE = {
    'pytdx': ('stock_list', 'stock_info', 'stock_block', 'etf_list', 'index_list'),
    'qmt': ('stock_list', 'stock_info', 'stock_block'),
}

#: 各参考集合的**刷新间隔（小时）**：``(盘中, 盘后)`` —— **按集合细分，不是一个全局值**
#: （用户 2026-10-09 定）。起算点是**上一次成功完成**的时刻（签到表 `GOLEMQ.function_checkins` 的 `last_call_timestamp`），
#: 不是本次启动时刻 —— 否则「跑挂了 / 跑了一半」会被当成刚刷新过。
#:
#: 为什么这样分：`stock_list` / `stock_block` / `etf_list` 是**名单**类，盘中有新上市、
#: 停牌、调样，故盘中收紧到 5h；而 `stock_info` 装的是**股本、上市日、省份/行业**，
#: **日内根本不变**，盘中也没必要去敲源 —— 恒定 24h。
TTL_HOURS = {
    'stock_list': (5, 24),
    'stock_block': (5, 24),
    'etf_list': (5, 24),
    'index_list': (5, 24),          # 同属**名单**类（盘中会调样/新发）
    'stock_info': (24, 24),
    'financial': (24, 24),          # 季频；`--save` 不做它，留在这里只是表完整
}

#: 表里没有的集合用这个。
DEFAULT_TTL_HOURS = (5, 24)

#: 刷新闸的**签到名前缀**：每个集合在 `supervisor` 的签到表里占一行
#: （`GOLEMQ.function_checkins`，键是 `(function_name, caller_ip)`）。
#: **不再自建 `refdata_meta`** —— 签到表本来就是干这个的（`function_checkin.py` 的
#: `expired_timestamp` 语义与 TTL 完全同构），再养一张表就是第二个轮子。
TTL_CHECKIN_PREFIX = 'refdata:'


def refdata_ttl_hours(collection=None, now=None) -> int:
    """**该集合**此刻该用的刷新间隔（小时）—— 查 :data:`TTL_HOURS` 的 ``(盘中, 盘后)``。

    「盘中」判定**复用** :func:`date_utils.GQ_util_if_tradetime`（别再写第二套）——
    它认的是真交易日历 `TRADE_DATE_SSE`，且把 9:15 起的集合竞价算作盘中。
    ``(x, x)`` 这种前后相同的表项**直接返回、不查日历**（`stock_info` 走这条）。

    >>> import datetime as _dt
    >>> refdata_ttl_hours('stock_list', _dt.datetime(2026, 10, 9, 10, 30))   # 周五盘中
    5
    >>> refdata_ttl_hours('stock_list', _dt.datetime(2026, 10, 9, 20, 0))    # 周五盘后
    24
    >>> refdata_ttl_hours('stock_list', _dt.datetime(2026, 10, 10, 10, 30))  # 周六
    24
    >>> refdata_ttl_hours('stock_info', _dt.datetime(2026, 10, 9, 10, 30))   # 恒定 24
    24
    >>> refdata_ttl_hours('不存在的集合', _dt.datetime(2026, 10, 9, 20, 0))    # 走默认
    24
    """
    open_h, closed_h = TTL_HOURS.get(collection, DEFAULT_TTL_HOURS)
    if open_h == closed_h:
        return open_h                      # 与盘中无关，不必去查交易日历
    from .date_utils import GQ_util_if_tradetime
    now = now if now is not None else dt.datetime.now()
    return open_h if GQ_util_if_tradetime(now) else closed_h


def _checkin_name(collection) -> str:
    return TTL_CHECKIN_PREFIX + str(collection)


def _caller_key() -> str:
    """刷新闸签到用的 `caller_ip`。

    2026-10-09 起**上移**到 :func:`supervisor.function_checkin.stable_caller_key` ——
    kline 的刷新闸要用同一个键，而那段「为什么不走 `resolve_caller_ip`」的警告是承重的，
    两处各写一份就是又一个平行实现。这里保留这个名字只是别名。
    """
    from GolemQ.supervisor.function_checkin import stable_caller_key
    return stable_caller_key()


def mark_refdata_success(collection, echo=None):
    """把「该集合**刚刚成功完成**」记进 `supervisor` 的签到表。**只该在真写完之后调**。

    `expired_time` 给的是该集合**此刻**的 TTL —— 签到表把 `expired_timestamp` 存成
    ``now + TTL``，所以起算点天然就是「成功完成这一刻」，正是要的语义。
    （这也是为什么**不拿** `checkin_function` 的 `allowed` 当判据：那个是
    「调用即记账」的名额模型，跑挂了也会把时间戳推后。）

    ⚠️ **记账失败绝不抛**（那一层在 :func:`checkin_function.mark_checkin` 里保证）：
    签到表在**运维库**（`GOLEMQ`）里，连不上 / 索引建不出来都有可能。这是记账，不是数据
    —— 为它把一次**成功的取数**变成失败是本末倒置。失败只是刷不成新、下次照常重取。

    :param echo: 输出汇（默认 `print`）。⚠️ **不能写死 `print`**：本函数在
        `save_refdata` 的集合循环里被调，那一刻 banner **正活着** —— 直接 `print`
        会让它的行数记账少算一行、下次重画整块写花（`PITFALLS.md` P22）。
        它是**故障报告**（不是闸的细节），所以**非 verbose 也要打**，只是要走 `echo`。
    """
    from GolemQ.supervisor.function_checkin import mark_checkin
    if not mark_checkin(_checkin_name(collection), refdata_ttl_hours(collection),
                        caller_ip=_caller_key()):
        (echo or print)('[refdata] ⚠️ {} 的签到时刻没记上 —— 刷新闸对它下次仍会重取'
                        .format(collection))


def _resolve_ttl_hours(ttl_hours, collection):
    """把 `save_refdata` 的 `ttl_hours` 形参解析成**该集合**的阈值（小时）。

    允许三种形态：``None``（不做闸）、**数字**（所有集合同一阈值 —— 测试与简单调用用）、
    **可调用** ``f(集合名) -> 小时``（CLI 传的就是 :func:`refdata_ttl_hours`，
    这样阈值才**按集合**分）。
    """
    if ttl_hours is None:
        return None
    return ttl_hours(collection) if callable(ttl_hours) else ttl_hours


def refdata_age_hours(collection, now=None):
    """该集合距上次**成功完成**过去了多少小时；**从没成功过 → None**（调用方按「该取」处理）。

    只读：走 `supervisor.function_checkin.checkin_age_hours`，**不签到、不写库**。
    """
    from GolemQ.supervisor.function_checkin import checkin_age_hours
    return checkin_age_hours(_checkin_name(collection), caller_ip=_caller_key(), now=now)

#: 各集合的 upsert 键。`stock_block` 是复合键 —— 一只股票属于多个板块。
UNIQUE_KEYS = {
    'stock_list': ['code'],
    'stock_info': ['code'],
    'etf_list': ['code'],
    'index_list': ['code'],
    'financial': ['code', 'report_date'],
}

#: 允许 delta 删除的集合 —— **不是所有集合都能删差量**。
#: `stock_list`/`stock_info` 的源是全市场权威名单，本次没出现即为退市，可删。
#: `etf_list` 是快照非权威名单，删差量会误删停牌/新上市未收录的品种。
#: `financial` 是历史累积，本次没拉到不代表没有，**绝不能删**。
#: 允许做**差量删除**的集合（语义：「本次取到的就是全部，其余删掉」）。
#:
#: ⚠️ **`stock_list` 刻意不在列**：主源 pytdx **设计上就枚举不全**（拿不到北交所
#: 那 351 只 `92xxxx`，见 `datasource/__init__.py` 的优先级说明），而差量删除会把
#: "没取到"当成"已退市"**删掉**。实测踩过：切成 pytdx 主源的那一次，`deleted=351`。
#: 失去的只是"自动清退市代码"这一点便利，换来的是"取数不全也不会静默删数据"。
#: ⚠️ **`index_list` 同理不入列**（同 `stock_list` 的理由）：pytdx 枚举不全
#: （1221 条 vs 4.4 的 1291 条），差量删除会把"没取到"当成"已下架"删掉。
DELTA_DELETE = {'stock_info'}


def _pick_source(collection: str, source: str = None, verbose: bool = False,
                 exclude=()):
    """选源。显式指定则用之；否则按优先级取第一个 `available()` 的。

    :param exclude: 优先级的**排除名单**。给「命令名即源名」的场景用 ——
        例如 `--save tdx` 不该拖 akshare（慢、有频率限制、会打 tqdm），
        而 `etf_list`/`financial` 的优先级首位恰好是它。
    """
    if source:
        s = get_source(source)
        if not s.supports(collection):
            raise UnsupportedCollection(
                f'{source} 不提供 {collection}（可提供 {s.collections or "无"}）')
        if not s.available():
            raise DataSourceNotAvailable(
                f'{source} 当前不可用：'
                f'{getattr(s, "unavailable_reason", lambda: "原因未知")()}')
        return s
    for name in COLLECTION_SOURCE_PRIORITY.get(collection, []):
        if name in exclude:
            continue
        s = get_source(name)
        if s.available():
            return s
    return None


def _supplement_stock_list(rows: list, primary_name: str, verbose: bool,
                           exclude=(), echo=None) -> list:
    """从下一个可用源给 `stock_list` **补字段、也补行**。

    两类差异，都源于同一个接口边界：
    * **补字段**：主源缺 `name`/`pre_close` 时填空（**不覆盖主源给的值** —— 主源是权威）。
    * **补行**：主源**枚举不到的标的**（pytdx 拿不到北交所 `92xxxx`，
      `get_security_list(2, 0)` 返回 None 且会毒死连接）由补充源补上。

    ⚠️ **补行是必须的**：`stock_list` 带 `delete_delta_key`，主源取到多少就是"全集"，
    少的那 351 只会被**差量删除**。实测 pytdx 5,228 / tdxaidata 5,579，差的正好是
    北交所那 351 只。
    """
    # ⚠️ **不要因为"主源字段齐全"就提前返回**：还要补主源**枚举不到**的标的
    # （pytdx 拿不到北交所）。我曾在这里留了个 `if not need: return rows`，
    # 结果 pytdx 当主源时 351 只北交所被差量删除 —— 见下面 `DELTA_DELETE` 的说明。
    say = echo or print
    for name in COLLECTION_SOURCE_PRIORITY.get('stock_list', []):
        if name == primary_name or name in exclude:
            continue
        try:
            s = get_source(name)
            if not s.available():
                continue
            fetched = s.fetch('stock_list')
        except Exception as exc:      # noqa: BLE001 补充源失败不该毁掉主结果
            if verbose:
                say(f'[refdata] 补充源 {name} 失败: {exc!r}')
            continue

        # 建索引时**显式处理同码**：裸 6 位代码在跨市场时可能重复
        # （沪市指数 000001 ↔ 深市股票 000001）。静默用后写覆盖先写，
        # 会把指数的名字填进股票记录 —— 实测踩过。
        extra: dict = {}
        dup = 0
        for r in fetched:
            c = r['code']
            if c in extra:
                dup += 1
                continue
            extra[c] = r
        if dup and verbose:
            say(f'[refdata] 补充源 {name} 有 {dup} 个重复 code，已保留首条')

        filled = 0
        have = {r['code'] for r in rows}
        added = 0
        for r in rows:
            e = extra.get(r['code'])
            if not e:
                continue
            for k in ('name', 'pre_close'):
                if not r.get(k) and e.get(k):
                    r[k] = e[k]
                    filled += 1
        for c, e in extra.items():          # 主源枚举不到的标的：整行补上
            if c not in have:
                rows.append(e)
                added += 1
        if verbose:
            say(f'[refdata] 以 {name} 补充 {filled} 个字段、{added} 行（主源枚举不到）')
        break
    return rows


def _fill_stock_info_name(rows: list, verbose: bool = False, echo=None) -> list:
    """给 `stock_info` 的行补 `name` —— 名字在同库的 `stock_list` 里，**不留 NULL 列**。

    为什么需要：pytdx 的 `get_finance_info` **不给名字**（源适配器刻意不重复取），
    而 4.4 的 `stock_info` 干脆没有 `name` 键。8.3 这边既然同库就有 `stock_list`
    可查，就别留一列全 NULL —— 实测项目所有者在 Compass 里看到那张表时，
    `Name` 列整列是空的。

    **只补缺名字的行**，不覆盖源已经给的值（与 `_supplement_stock_list` 同一条纪律）。
    """
    need = [r for r in rows if not r.get('name')]
    if not need:
        return rows
    names = {d['code']: d.get('name') for d in GOLEMQ_STOCK_CN['stock_list'].find(
        {}, {'_id': 0, 'code': 1, 'name': 1})}
    filled = 0
    for r in need:
        n = names.get(r.get('code'))
        if n:
            r['name'] = n
            filled += 1
    if verbose:
        (echo or print)(f'[refdata] stock_info: 以 stock_list 补了 {filled} 个 name')
    return rows


def _authoritative_codes(verbose: bool = False, echo=None) -> list:
    """以 `stock_list` 为准的标的宇宙。

    **为什么不能各源自己枚举**：`codelist=None` 时，若让每个适配器用自己的
    `fetch_stock_list()`，各源会静默覆盖**不同的标的宇宙** —— 因为它们枚举能力不同。
    实测过的后果：`stock_info` 只覆盖 5226 只，而 `stock_list` 有 5574 只，
    **差的 348 只全是北交所**（pytdx 枚举不到北交所，但它其实**取得到**北交所的财务）。

    `stock_list` 是集合里唯一的「全市场名单」，且它的主源 tdxaidata 覆盖北交所。
    故逐只查询的集合（`stock_info` / `financial`）一律以它为准。
    """
    codes = sorted(GOLEMQ_STOCK_CN['stock_list'].distinct('code'))
    if verbose:
        (echo or print)(f'[refdata] 以 stock_list 为准的标的宇宙: {len(codes)} 只')
    return codes


#: 码表缓存**拒绝写入**的下限。`stock_list` 取到的码数低于它就不覆盖缓存文件。
#: 理由：缓存是 `easyquotation` 全市场快照的**枚举源** —— 表空了，
#: `market_snapshot()` 就一台标的都不取，而症状是**实时集合每天是空的**，
#: 不报错、不告警（订阅器照样在跑、照样签到）。宁可留着旧表。
MIN_STOCK_CODES = 1000

#: 进快照码表的参考集合。**顺序即拼接顺序**（最终会去重排序）。
#: * `stock_list` —— 5,580 只（含北交所；腾讯不供那 352 只，但不值得为它们特判）
#: * `etf_list` —— 1,696 只，实测腾讯 **1,696/1,696 全供**
#: * `index_list` —— 1,291 只，实测腾讯供 **590**（见 `SNAPSHOT_BLOCKED_PREFIXES`）
SNAPSHOT_COLLECTIONS = ('stock_list', 'etf_list', 'index_list')

#: 腾讯**不供**的号段 —— 实测（2026-10-15）`880xxx` **689/689 全返回
#: `v_pv_none_match`**：那是**通达信自编指数**（总市值、平均股价这类），
#: 不是交易所发布的指数，故腾讯的行情端没有它们。
#: 留着它们每轮白占 689 个名额（约一整页 800），而且**永远不会返回数据**。
#:
#: ⚠️ 这是**实测出来的排除**，不是"大概没有"。若哪天腾讯供了，去掉这个元组即可 ——
#: 但**加之前先重新量一遍**：同族的 `399xxx` 供 360/365、`395xxx` 供 14/21，
#: 说明"深市指数"不是整体供或不供，得逐族看。
SNAPSHOT_BLOCKED_PREFIXES = ('880',)


def snapshot_universe(verbose: bool = False, echo=None) -> list:
    """8.3 的 `stock_list` ∪ `etf_list` ∪ `index_list` → **带交易所前缀**的码表。

    为什么前缀**必须**带上（这条是陷阱，不是风格）
    ============================================
    `easyquotation` 的 ``_gen_stock_prefix`` 对**裸 6 位**码用
    `helpers.get_stock_type()` 猜交易所。猜法对股票和 ETF 都对，**对指数是错的**：

    * `000001` 在 `stock_list` 里是**平安银行**（深市），在 `index_list` 里是
      **上证指数**（沪市）—— 两个集合**共用这个号**。
    * `get_stock_type('000001')` 判 `sz` ⇒ 去问腾讯要 **`sz000001` = 平安银行**，
      而 `index_list` 那行要的是 **`sh000001`**。**静默取错标的**，不报错。

    带上前缀之后两个都能取对（实测 `normalize_code('sh000001')` → `000001.XSHG`、
    `normalize_code('sz000001')` → `000001.XSHE`，互不干扰）。

    ⚠️ 前缀**顺带修好了另一个坑**：ETF 的 `158xxx` 那批，**裸码**过
    `normalize_code` 会打警告并**原样返回 6 位**（`_write_ts_rows` 要求带后缀，
    裸码会让两条流在同一 `(code, ts)` 上错配）；**带前缀**就正常归一化成
    `158000.XSHE`（实测）。

    `sse` 取**库里的字段**而不是再猜一遍号段 —— 三个集合都带这个字段，
    它是源端（pytdx/akshare）给的，比我们的启发式权威。

    :returns: 去重排序后的列表，形如 ``['sh000001', 'sh600519', 'sz000001', ...]``
    """
    say = echo or print
    out, per = set(), {}
    for name in SNAPSHOT_COLLECTIONS:
        n = 0
        for r in GOLEMQ_STOCK_CN[name].find({}, {'_id': 0, 'code': 1, 'sse': 1}):
            code, sse = r.get('code'), r.get('sse')
            if not code or not sse:
                # `sse` 缺失的行**跳过而不是猜**：猜错就是静默取错标的（见 docstring）
                continue
            if code.startswith(SNAPSHOT_BLOCKED_PREFIXES):
                continue
            out.add(sse + code)
            n += 1
        per[name] = n
    codes = sorted(out)
    if verbose:
        say('[snapshot] 码表宇宙：{} 只 —— {}'.format(
            len(codes), ' + '.join(f'{k} {v}' for k, v in per.items())))
    return codes


def refresh_stock_codes_cache(verbose: bool = False, echo=None) -> dict:
    """把 8.3 `stock_list` 的码表**导出**到 easyquotation 的本地缓存文件。

    为什么需要这一步
    ================
    `easyquotation` 的全市场快照（`BaseQuotation.market_snapshot`，即
    `--sub l1_tencent` 的数据源）枚举的不是什么"白名单"，
    而是 `~/.GolemQ/datastore/cache/stock_cn/stock_codes.conf` 这份**本地码表**。
    它此前**全树无人写** —— 唯一的上游刷新口 `helpers.update_stock_codes()`
    零调用。后果是**新股永远收不到、退市股永远在收，且完全静默**：

    * 实测量（2026-10-15，**改之前**）：缓存里 **5,186** 码，8.3 `stock_list` 有 **5,580** 码，
      差的正是 2026 年新上的那批；缓存里还留着 `300114`（已退市/换码）。

    为什么不复用 `helpers.update_stock_codes()`（**判成「新增」的留痕**）
    ==================================================================
    那个函数敲的是 `shdjt.com`，实测 **19.1 秒**才回来（15s 超时下时好时坏，
    `https://` 那道门直接拒绝连接）。而 `--save tdx` 刚刚把 `stock_list` 刷进
    8.3 —— 那份才是**本次的权威名单**（pytdx 主源 + `_supplement_stock_list`
    用 tdxaidata 补上 pytdx 枚举不到的北交所）。从库里导出 = **零网络**、
    与本流程同源、且绝不会与刚写进去的名单打架。故新增本函数，
    并把 `update_stock_codes()` 的写法收敛到 `helpers.save_stock_codes`（保留为上游 API）。

    为什么是**三个集合的并集**
    ========================
    见 :func:`snapshot_universe` —— `stock_list` ∪ `etf_list` ∪ `index_list`，
    一律**带交易所前缀**（不带前缀会让 `000001` 静默取成平安银行而不是上证指数）。
    实测腾讯：ETF **1,696/1,696 全供**、指数供 **590/1,291**（`880xxx` 那 689 只
    是通达信自编，不供）。

    写盘时机（调用方的契约）
    ======================
    **只在「本次真的取到并写成功」且「码表有增减」时才落盘** ——
    每日跑的大多数天码表是不变的，无谓的写盘只是多一次原子替换。
    判据由调用方给（见 `cli/commands/save.py` 的 `status == 'ok'`），
    本函数自己再做一次增删判定。

    :returns: ``{'changed': bool, 'total': n, 'added': [...], 'removed': [...],
        'path': str, 'refused': str|None}``
    """
    # 唯一出处是 `helpers.STOCK_CODE_PATH` —— 不在这里复制一份路径常量。
    # 路径一旦分叉，症状是**订阅器读的是另一张表**（静默读到空/读到旧表）。
    # 用**模块属性**而不是 `from ... import STOCK_CODE_PATH`：后者在 import 期
    # 就把值抄走了，测试没法替换。
    from .easyquotation import helpers

    say = echo or print
    out = {'changed': False, 'total': 0, 'added': [], 'removed': [],
           'path': helpers.STOCK_CODE_PATH, 'refused': None}

    codes = snapshot_universe(verbose=verbose, echo=echo)
    out['total'] = len(codes)
    if len(codes) < MIN_STOCK_CODES:
        out['refused'] = ('码表宇宙只有 {} 只（< {}），拒绝覆盖码表缓存'
                          .format(len(codes), MIN_STOCK_CODES))
        say(f'[stock_codes] {out["refused"]}')
        return out

    current = set(helpers.get_stock_codes())
    new = set(codes)
    added, removed = sorted(new - current), sorted(current - new)
    if not added and not removed:
        if verbose:
            say(f'[stock_codes] 码表无增减（{len(codes)} 只），不写盘')
        return out

    helpers.save_stock_codes(codes)
    out.update(changed=True, added=added, removed=removed)
    # ⚠️ **只在 `-v` 下打**（用户 2026-10-15 定）—— 与参考数据的刷新闸那句、
    # 逐集合状态表**同一口径**：「更新了几只」是实现细节，不是每次运行都要看的东西。
    # 非 `-v` 下**成功就是静默的**。
    # ⚠️ 但**拒写**（上面 `refused`）与**失败**（调用方 `except`）**不跟 `-v` 走** ——
    # 那两种情况意味着"这次没写/写坏了"，静默会让人以为写成功了。
    if verbose:
        say('[stock_codes] 码表缓存已更新：{} 只（+{} / -{}）→ {}'.format(
            len(codes), len(added), len(removed), out['path']))
        for label, diff in (('新增', added), ('移除', removed)):
            if diff:
                head = ','.join(diff[:20]) + ('...' if len(diff) > 20 else '')
                say(f'    {label}: {head}')
    return out


def save_index_list_from44(dry_run: bool = False, verbose: bool = True,
                          echo=None) -> dict:
    """把 4.4 的 `quantaxis.index_list` **补**进 8.3 的 `index_list`（**只补缺**）。

    **为什么需要这一步**：`index_list` 在 4.4 里是 1291 行，迁移时**整张漏了**
    （8.3 此前根本没有这张表），于是 `kline83._display_name` 对真指数一直回落成
    代码 —— 用户 2026-10-11 实报「标的：301308（301308）」时顺带发现的。

    现在 `--save tdx` 能从 pytdx 取到 **1221** 条（见
    `pytdx_source.fetch_index_list`，零误选），但那 70 条**只在 4.4 里**的
    指数（TDX 已下架）取不回来 —— 本函数就是补那一段。

    ⚠️ **只补缺、不覆盖**（区别于 `save_refdata` 的 upsert）：
    4.4 的 `pre_close` 是 2026 年那一版、早已过期，拿它盖掉 pytdx 刚取的新值
    是净损失。只补缺还让本函数的**执行顺序无关** —— `--save tdx` 之前跑、
    之后跑，结果一样。

    :returns: ``{'read':…, 'written':…, 'exists':…, 'skipped':…}``
    """
    from GolemQ.core.migrate44 import db44
    say = echo or print
    coll = GOLEMQ_STOCK_CN['index_list']
    have = {d['code'] for d in coll.find({}, {'_id': 0, 'code': 1})}
    rows = list(db44('quantaxis').index_list.find({}, {'_id': 0}))
    out = {'read': len(rows), 'written': 0, 'exists': 0, 'skipped': 0}
    docs = []
    for r in rows:
        code = r.get('code')
        if not code:
            out['skipped'] += 1
            continue
        if code in have:
            out['exists'] += 1
            continue
        # 键序与 `pytdx_source.TdxSource._row` 对齐；`source` 标明来路
        docs.append({'code': str(code),
                     'volunit': r.get('volunit'),
                     'decimal_point': r.get('decimal_point'),
                     'name': r.get('name'),
                     'pre_close': r.get('pre_close'),
                     'sse': r.get('sse'),
                     'sec': r.get('sec') or 'index_cn',
                     'source': 'quantaxis-4.4'})
    if dry_run:
        out['written'] = len(docs)
        say('[index_list] 干跑：读 {} / 可补 {} / 已存在 {} / 丢弃 {}（【没有写库】）'
            .format(out['read'], len(docs), out['exists'], out['skipped']))
        return out
    if docs:
        coll.insert_many(docs, ordered=False)
    out['written'] = len(docs)
    if verbose:
        say('[index_list] 4.4 补缺：读 {} / 写入 {} / 已存在 {} / 丢弃 {}'.format(
            out['read'], out['written'], out['exists'], out['skipped']))
    return out


def save_refdata(collections=None, source: str = None, codelist=None,
                 exclude_sources=None, verbose: bool = True,
                 on_progress=None, echo=None, ttl_hours=None) -> dict:
    """把参考集合取回并落库到 8.3 的 `golemq_stock_cn`。

    :param collections: 要保存的集合名列表；None = 全部 5 个
    :param source: 强制使用的源名；None = 按优先级自动选
    :param codelist: 仅供逐只查询的集合（`stock_info`/`financial`）限定范围
    :param exclude_sources: 不参与优先级选择的源名；见 :func:`_pick_source`
    :param on_progress: 可选回调 ``f(集合名, entry)``，**每个集合处理完调一次**，
        ok / skipped / failed 都调（含各条 `continue` 出口）。给调用方画进度用 ——
        CLI 的 `--save` 拿它点亮 banner。默认 None = 不做任何事。
        **本模块不认识任何 UI**：回调由调用方注入，见模块 docstring 的
        「读写分开 / 不拖进界面依赖」那条纪律。
    :param ttl_hours: 新鲜度闸。距**上次成功完成**不足这么多小时就**跳过该集合、不碰源**。
        三种形态：``None``（默认，不做闸，行为与从前逐字相同）、**数字**（所有集合同一
        阈值）、**可调用** ``f(集合名) -> 小时``。CLI 传的是 :func:`refdata_ttl_hours`
        —— **阈值是按集合分的**（名单类盘中 5h/盘后 24h，`stock_info` 恒定 24h），
        为的是压掉重复的 HTTP 查询。
        ⚠️ 被判过期而跳过时 ``entry['cached'] = True`` —— 数据**是好的、只是没重取**，
        调用方据此把它当作「已就绪」而不是「没取到」。
    :returns: ``{集合名: {'status':..., 'rows':..., 'source':..., 'detail':...}}``
    """
    targets = list(collections) if collections else list(ALL_REF_COLLECTIONS)
    say = echo or print
    report: dict = {}

    for coll_name in targets:
        entry = {'status': 'skipped', 'rows': 0, 'source': None, 'detail': '',
                 'cached': False}
        report[coll_name] = entry

        # 整段包在 try/finally 里，只为让 `on_progress` **每条出口都回调** ——
        # 下面有 4 条 `continue`（跳过/无源/取数失败/落库失败），把回调写在末尾
        # 就会漏掉它们，banner 上于是留下永远点不亮的节点。
        try:
            # —— 新鲜度闸：距上次**成功完成**不足**该集合**的 TTL 就跳过，连源都不选 ——
            # TTL **按集合**取（`TTL_HOURS`：名单类盘中 5h/盘后 24h，`stock_info` 恒定 24h）。
            # 起算点是「上次成功完成」而不是「上次启动」：跑挂了 / 跑一半不算数。
            ttl = _resolve_ttl_hours(ttl_hours, coll_name)
            if ttl:
                age = refdata_age_hours(coll_name)
                if age is not None and age < ttl:
                    entry['cached'] = True
                    entry['detail'] = ('未过期：{:.1f}h 前成功完成（< TTL {}h），本次不重取'
                                       .format(age, ttl))
                    if verbose:
                        say('[refdata] {}: 跳过 —— {}'.format(coll_name, entry['detail']))
                    continue

            if on_progress is not None:
                on_progress(coll_name, 'start', entry)       # banner 置「正在读取」
            try:
                src = _pick_source(coll_name, source=source, verbose=verbose,
                                   exclude=exclude_sources or ())
            except (UnsupportedCollection, DataSourceNotAvailable) as exc:
                entry['detail'] = str(exc)
                if verbose:
                    say(f'[refdata] {coll_name}: 跳过 —— {exc}')
                continue

            if src is None:
                entry['detail'] = ('无可用源。已注册的源里，能提供该集合的都不可用；'
                                   '见各自适配器的 unavailable_reason()')
                if verbose:
                    say(f'[refdata] {coll_name}: 跳过 —— {entry["detail"]}')
                continue

            entry['source'] = src.name
            try:
                kwargs = {}
                if coll_name in ('stock_info', 'financial'):
                    # 逐只查询的集合：范围一律以 stock_list 为准，不让各源按自己的
                    # 枚举能力定宇宙 —— 否则会静默漏掉某些市场（实测漏过北交所 348 只）。
                    scope = codelist if codelist is not None else _authoritative_codes(
                        verbose, echo=echo)
                    kwargs['codelist'] = scope
                rows = src.fetch(coll_name, **kwargs)
            except Exception as exc:      # noqa: BLE001 逐集合隔离，一个失败不拖垮其余
                entry['status'] = 'failed'
                entry['detail'] = f'{type(exc).__name__}: {exc}'
                if verbose:
                    say(f'[refdata] {coll_name}: 失败 —— {entry["detail"]}')
                continue

            if coll_name == 'stock_list' and rows:
                rows = _supplement_stock_list(rows, src.name, verbose,
                                              exclude=exclude_sources or (),
                                              echo=echo)
            elif coll_name == 'stock_info' and rows:
                rows = _fill_stock_info_name(rows, verbose=verbose, echo=echo)

            coll = GOLEMQ_STOCK_CN[coll_name]
            try:
                if coll_name == 'stock_block':
                    stats = save_block_collection(coll, rows, verbose=verbose)
                else:
                    delta_key = {'stock_list': 'code', 'stock_info': 'code'}.get(coll_name)
                    # ⚠️ **传了 `codelist` 就绝不删差量**（承重）。
                    # 删差量的语义是「本次取到的就是全部」；而 `codelist` 是调用方
                    # **故意只取一部分**（试跑、排障、补齐某几只）。两者相乘 = 把没取到
                    # 的那些标的**静默删掉** —— 实测踩过：`--save-codes` 跑一次
                    # 就把 `stock_info` 从 5,574 行删到 250 行。
                    if codelist is not None and delta_key:
                        delta_key = None
                        if verbose:
                            say(f'[refdata] {coll_name}: 本次传了 codelist（部分取数），'
                                  f'**跳过差量删除**，不会动其他标的')
                    stats = save_collection(
                        coll, rows, UNIQUE_KEYS[coll_name],
                        delete_delta_key=delta_key, verbose=verbose)
            except Exception as exc:      # noqa: BLE001
                entry['status'] = 'failed'
                entry['detail'] = f'落库失败 {type(exc).__name__}: {exc}'
                if verbose:
                    say(f'[refdata] {coll_name}: {entry["detail"]}')
                continue

            entry['rows'] = len(rows)
            entry['status'] = 'skipped' if stats.get('skipped') else 'ok'
            entry['detail'] = str(stats)
            # 只记「真成功」的时刻 —— `skipped`（源返回空/未写入）不算数，
            # 否则一次空跑就会把这个集合冻住一整个 TTL。
            if entry['status'] == 'ok':
                mark_refdata_success(coll_name, echo=echo)
            if verbose:
                say(f'[refdata] {coll_name}: {entry["status"]} {len(rows)} 行 (源 {src.name})')
        finally:
            if on_progress is not None:
                on_progress(coll_name, 'done', entry)

    return report


def refdata_status() -> dict:
    """各集合当前的源可用性与库存量 —— 排障用。"""
    out = {}
    for name in ALL_REF_COLLECTIONS:
        try:
            count = GOLEMQ_STOCK_CN[name].estimated_document_count()
        except Exception:  # noqa: BLE001
            count = None
        avail = []
        for s_name in COLLECTION_SOURCE_PRIORITY.get(name, []):
            s = get_source(s_name)
            avail.append({'source': s_name, 'available': s.available()})
        out[name] = {'docs': count, 'sources': avail}
    return out


def format_status(report: dict = None) -> str:
    """把 `save_refdata` 的结果或 `refdata_status()` 渲染成可读文本。"""
    lines = []
    if report is not None:
        lines.append(f'{"集合":<12}{"状态":<9}{"行数":>8}  源 / 说明')
        for name, e in report.items():
            lines.append(f'{name:<12}{e["status"]:<9}{e["rows"]:>8}  '
                         f'{e.get("source") or "-"} {e.get("detail", "")[:60]}')
    else:
        ts = dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        lines.append(f'参考集合状态 @ {ts}')
        for name, e in refdata_status().items():
            srcs = ', '.join(f'{s["source"]}{"" if s["available"] else "(不可用)"}'
                             for s in e['sources']) or '无'
            lines.append(f'  {name:<12} 库存 {e["docs"] if e["docs"] is not None else "?":>8}  {srcs}')
    return '\n'.join(lines)
