# coding:utf-8
"""资金流 / 千股千评 → **`stock_reality_feat_day`**（旧 CLI `--moneyflow` 的搬运）。

从哪来
======
旧树 `cli/__main__.py:514` 的 ``elif (cmdline_args.moneyflow in ['eastmoney', 'ths']):``
分支。⚠️ 那个分支**名字骗人**：取值列了 `eastmoney` / `ths`，**分支体里只有东财**
（`ths` 是个假别名，和 eastmoney 并进同一段代码）。真正的同花顺实现**从来不存在**。

写到哪
======
`stock_reality_feat_day`（**不是**旧树的 `stock_ranking` / `stock_valuation` / `stock_moneyflow`）——
用户 2026-10-11 明确要求。落库走 :func:`GolemQ.datasource.writer.upsert_fields`：
**只 `$set` 自己的列，绝不整文档 replace**（那张表多列共用一个 `(code, date_stamp)` 文档）。

⚠️ 两个源的可用性是**实测过的**，不是照旧树采信（2026-10-11）
=========================================================

**死的**（❌）—— 旧树那个**逐只**资金流爬虫打的就是第一个：

* ``push2his.eastmoney.com/.../fflow/daykline`` —— 开代理时 502 / ProxyError，
  **关代理后 RemoteDisconnected**
* ``push2.eastmoney.com/api/qt/clist``（批量快照）—— 同上

⚠️ **代理开与关都测过**（2026-10-11 用户关代理后复测）：失败形态变了三次
（502 / ProxyError / RemoteDisconnected），**结论一次没变** —— 那批接口**是真死，
不是被代理挡的**。这条复测很重要：只看到 "ProxyError" 会误判成"修好代理就能用"。

**活的**（✅）：

* ``datacenter-web.eastmoney.com``（千股千评）—— 开关代理**两种状态下都 200**
* ``ak.stock_comment_em()`` —— (5200, 14) / ~3s
* ``ak.stock_fund_flow_individual('即时')``（同花顺）—— (5210, 10) / ~9s

⇒ 旧树那个逐只资金流爬虫**搬不了**（打的是已死的 `push2his`），
搬过来就是一个跑不通的东西。资金流改走**同花顺的当日全市场快照** ——
而 `ths` 本来就在旧 flag 的取值里，不算另起炉灶。

⚠️ **同花顺只有「当日」，没有历史** ⇒ 它**补不了过去**，只能从今天往后攒。
CLI 的输出里会写明这一点。

两处**实测钉住**的口径（别"顺手改"）
====================================
1. **换手率 ÷100、写进现有的 `TurnoverRate` 列**。千股千评给的是百分数
   （`600519` = `0.28`），÷100 后与**库内同源同列**逐值相同 ——
   实测 `600519 → 0.0028` / `000001 → 0.0056` / `301308 → 0.0361`，
   与 `stock_reality_feat_day` 里 4.4 迁移来的值**完全一致**（同一个东财源）。
   所以这不是"覆盖"，是**刷新**（那一列从"一次性搬运"变成"每次爬取都更新"）。
2. **日期取「最后交易日」，不取 `today()`**。同花顺的 `即时` 在**周六**给的是
   **最后交易日**的数据（实测 `600519` 最新价 `1263.0` == 库内 `2026-10-09` 的
   `close`），拿 `today()` 当 `date` 会**整体错位一天**。
"""
from __future__ import annotations

import time

import pandas as pd

from GolemQ.core.constants import FIELD as FLD

from .metadata_save import (REALITY_FEAT_DAY_COLL, _db, day_date, day_stamp,
                            metadata_day_doc_multi)

__all__ = ['save_moneyflow', 'rows_eastmoney', 'rows_ths', 'parse_cn_amount',
           'parse_pct', 'to_code', 'last_trade_day', 'MONEYFLOW_SOURCES']

#: 本模块负责的列（**只 `$set` 这些**）。别的列（换手率的 baostock 那列、
#: 别处写的任何列）不归本模块管 —— `upsert_fields` 的 `payload_keys` 用它做白名单。
EM_PAYLOAD = ('code', 'date', 'date_stamp', 'datetime', 'ts', 'created_at',
              FLD.TURNOVER_RATE, FLD.PE_RATION, FLD.ZLCB, FLD.JGCYD,
              FLD.TOTAL_SCORE, FLD.RANKING_UP, FLD.RANKING, FLD.FOCUS,
              FLD.CHANGE_PERCENT)

THS_PAYLOAD = ('code', 'date', 'date_stamp', 'datetime', 'ts', 'created_at',
               FLD.THS_INFLOW, FLD.THS_OUTFLOW, FLD.THS_NET, FLD.THS_AMOUNT)

MONEYFLOW_SOURCES = ('eastmoney', 'ths')


# --------------------------------------------------------------------------- #
# 纯函数（标量解析）—— 都要能单测，别藏进循环里
# --------------------------------------------------------------------------- #
def to_code(value) -> str:
    """任意形态的标的代码 → **6 位字符串**。

    ⚠️ 同花顺那列 `股票代码` 是 **int**（`300530`），直接 `str()` 会把
    `000001` 变成 **`'1'`** —— 静默写错行，且不报错。

    >>> to_code(300530), to_code('000001'), to_code(1), to_code('600519.SH')
    ('300530', '000001', '000001', '600519')
    """
    return str(value).strip().split('.')[0].zfill(6)


def parse_cn_amount(text):
    """同花顺的**带单位**金额 → 浮点。

    ⚠️ 那两个源给的金额是**字符串**（`'3.72亿'` / `'2098.28万'`），不是数字 ——
    直接 `float()` 会抛，`pd.to_numeric` 会全变 NaN。取不到数时返回 `None`
    （调用方据此跳过该列，**不写 0**）。

    >>> round(parse_cn_amount('3.72亿')), round(parse_cn_amount('2098.28万')), round(parse_cn_amount('7.23亿'))
    (372000000, 20982800, 723000000)
    >>> round(parse_cn_amount('-2098.28万')), parse_cn_amount('1234'), parse_cn_amount('-')
    (-20982800, 1234.0, None)

    ⚠️ 别把上面写成 `20982800.0` —— 实测是 `20982800.000000004`（`2098.28 * 1e4`
    在二进制里就不精确）。**这是浮点现实，不是 bug**，所以这里用 `round()` 判等。
    """
    if text is None:
        return None
    s = str(text).strip().replace(',', '')
    if s in ('', '-', '--', 'None', 'nan'):
        return None
    unit = 1.0
    for suffix, factor in (('万亿', 1e12), ('亿', 1e8), ('万', 1e4), ('元', 1.0)):
        if s.endswith(suffix):
            unit = factor
            s = s[:-len(suffix)]
            break
    try:
        return float(s) * unit
    except ValueError:
        return None


def parse_pct(text):
    """`'20.02%'` → `0.2002`（百分数 → 小数）。取不到返回 `None`。

    >>> parse_pct('20.02%'), parse_pct('-1.5%'), parse_pct('-')
    (0.2002, -0.015, None)
    >>> round(parse_pct(0.56), 12)      # ⚠️ 0.56/100 在二进制里是 0.005600000000000001
    0.0056
    """
    if text is None:
        return None
    s = str(text).strip().replace(',', '')
    if s in ('', '-', '--', 'None', 'nan'):
        return None
    try:
        return float(s.rstrip('%')) / 100.0
    except ValueError:
        return None


def last_trade_day() -> str:
    """8.3 口径的**最后交易日**（`'YYYY-MM-DD'`）—— 同花顺的日期靠它。

    ⚠️ 用日历，**不能用 `today()`**：见模块 docstring 第 2 条实测。
    """
    from .date_utils import GQ_util_get_last_day
    return str(GQ_util_get_last_day())[:10]


# --------------------------------------------------------------------------- #
# 取数（薄封装，网络都在这两个函数里）
# --------------------------------------------------------------------------- #
def fetch_eastmoney_comment() -> pd.DataFrame:
    """东财千股千评（`ak.stock_comment_em`）—— 实测 5200 行 / 11 次分页 / ~3s。

    ⚠️ akshare 内部**已经把 11 次分页打完**才返回；它打的是
    `datacenter-web.eastmoney.com`（实测 200），**不是**已死的 `push2*`。
    """
    import akshare as ak
    return ak.stock_comment_em()


def fetch_ths_fundflow() -> pd.DataFrame:
    """同花顺全市场**当日**资金流（`ak.stock_fund_flow_individual('即时')`）。

    实测 5210 行 / 105 次分页 / ~8.7s。⚠️ **只有当日快照，没有历史。**
    """
    import akshare as ak
    return ak.stock_fund_flow_individual(symbol='即时')


# --------------------------------------------------------------------------- #
# 行构造（纯函数）
# --------------------------------------------------------------------------- #
def rows_eastmoney(df: pd.DataFrame, created_at=None) -> list:
    """千股千评帧 → `stock_reality_feat_day` 文档列表。

    列映射见模块 docstring；`date` 取源端的**交易日**列（它是 `datetime.date`）。
    ⚠️ `最新价` / `名称` **不落**：价格 K 线里有，名称在 `stock_list` 里，
    往这张表再存一份只会多两列"写后无人读"。
    """
    mapping = (('换手率', FLD.TURNOVER_RATE, parse_pct),
               ('市盈率', FLD.PE_RATION, None),
               ('主力成本', FLD.ZLCB, None),
               ('机构参与度', FLD.JGCYD, None),
               ('综合得分', FLD.TOTAL_SCORE, None),
               ('上升', FLD.RANKING_UP, None),
               ('目前排名', FLD.RANKING, None),
               ('关注指数', FLD.FOCUS, None),
               ('涨跌幅', FLD.CHANGE_PERCENT, parse_pct))
    stamp = int(created_at if created_at is not None else time.time())
    docs = []
    for _, r in df.iterrows():
        day = day_date(r.get('交易日'))
        code = to_code(r.get('代码'))
        if not day or not code:
            continue
        fields = {}
        for src, dst, conv in mapping:
            if src not in df.columns:
                continue
            raw = r.get(src)
            try:
                val = conv(raw) if conv else float(raw)
            except (TypeError, ValueError):
                val = None
            if val is not None:
                fields[dst] = val
        docs.append(metadata_day_doc_multi(code, day, fields, stamp))
    return docs


def rows_ths(df: pd.DataFrame, day: str, created_at=None) -> list:
    """同花顺资金流帧 → `stock_reality_feat_day` 文档列表（`day` 由调用方给）。

    ⚠️ **换手率那一列丢弃**：这是**第三个**换手率口径（既不是东财千股千评的、
    也不是 baostock 的）。表里已经有两列，再加一列只会是第三列死列。
    同 `涨跌幅` 也丢 —— 与千股千评重复，且口径未必一致。
    """
    mapping = (('流入资金', FLD.THS_INFLOW), ('流出资金', FLD.THS_OUTFLOW),
               ('净额', FLD.THS_NET), ('成交额', FLD.THS_AMOUNT))
    stamp = int(created_at if created_at is not None else time.time())
    docs = []
    for _, r in df.iterrows():
        code = to_code(r.get('股票代码'))
        if not code:
            continue
        fields = {}
        for src, dst in mapping:
            if src not in df.columns:
                continue
            val = parse_cn_amount(r.get(src))
            if val is not None:
                fields[dst] = val
        docs.append(metadata_day_doc_multi(code, day, fields, stamp))
    return docs


# --------------------------------------------------------------------------- #
# 落库
# --------------------------------------------------------------------------- #
def save_moneyflow(source: str = 'eastmoney', created_at=None, verbose=True,
                   echo=None, dry_run=False) -> dict:
    """取数 → 组行 → `upsert_fields` 落 `stock_reality_feat_day`。

    :param source: ``'eastmoney'``（千股千评）或 ``'ths'``（资金流）
    :param dry_run: 只取数并统计，**不写库**
    :returns: ``{'source':…, 'day':…, 'read':…, 'written':…, 'upserted':…,
        'modified':…, 'dry_run':…}``

    ⚠️ **只 `$set` 本模块负责的列**（`payload_keys` 白名单）—— 同一文档里
    别人写的列（baostock 的那列换手率、以后的任何派生列）**必须原样活着**。
    这是那张表的硬规矩，也是 `PITFALLS` 里"整文档替换会静默抹掉别列成果"那条。
    """
    say = echo or print
    if source not in MONEYFLOW_SOURCES:
        raise ValueError('未知源 {!r}；可用 {}'.format(source, MONEYFLOW_SOURCES))

    if source == 'eastmoney':
        raw = fetch_eastmoney_comment()
        docs = rows_eastmoney(raw, created_at=created_at)
        payload = EM_PAYLOAD
        day = docs[0]['date'] if docs else None
    else:
        day = last_trade_day()
        raw = fetch_ths_fundflow()
        docs = rows_ths(raw, day, created_at=created_at)
        payload = THS_PAYLOAD

    out = {'source': source, 'day': day, 'read': len(raw), 'written': len(docs),
           'upserted': 0, 'modified': 0, 'dry_run': bool(dry_run)}
    if dry_run:
        say('[moneyflow] {} 干跑：读 {} 行 / 可写 {} 行（交易日 {}）（【没有写库】）'
            .format(source, out['read'], out['written'], day))
        return out
    if not docs:
        say('[moneyflow] {}：**没取到任何行**，不写库'.format(source))
        return out

    from GolemQ.core.constants import MARKET_TYPE
    from .reality_save import save_reality_feat

    # 统一入口（D55）：只把**本模块 payload 白名单**里的列交给 save_reality_feat，
    # 别的列不会被写、更不会被清。
    frame = pd.DataFrame(docs)
    # ⚠️ **必须去重**：`payload` 本身已含 `code`/`date`，再拼一次会让
    # `frame.loc[:, keep]` 出现**重复列**，`set_index(['date','code'])`
    # 就会拿 4 列去建 2 层索引 ⇒
    # `ValueError: Length mismatch: Expected axis has N elements, new values have 2`。
    keep = [c for c in dict.fromkeys(('code', 'date', *payload))
            if c in frame.columns]
    frame = frame.loc[:, keep].set_index(['date', 'code'])
    # ⚠️ 同花顺快照**源端会重复**（实测 2026-10-20：3,300 行只有 2,237 个代码，
    # 1,063 行是分页重叠的完全重复行）。目标表唯一键是 `(code, date_stamp)`，
    # 重复送进去只会多写一遍（最后一条覆盖），并让 `written` 虚高。
    frame = frame[~frame.index.duplicated(keep='last')]
    stats = save_reality_feat(frame, 'day', MARKET_TYPE.STOCK_CN,
                              verbose=False, db=_db())
    out.update({k: stats.get(k, 0)
                for k in ('written', 'upserted', 'modified')})
    if verbose:
        say('[moneyflow] {} → {}：交易日 {}，读 {} 行 / 写 {} 文档 '
            '/ 新增 {} / 更新 {}'.format(source, REALITY_FEAT_DAY_COLL, day,
                                         out['read'], out['written'],
                                         out['upserted'], out['modified']))
    return out
