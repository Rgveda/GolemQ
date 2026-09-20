# coding:utf-8
"""A 股 K 线读取器 —— 走 MongoDB 8.3 时序库（``golemq_stock_cn``）。

为什么存在这一层
====================================================================
分钟线已从旧 GolemQ 系统的 MongoDB 4.4（``stock_min`` / ``index_min``）迁移到
8.3 的时序集合，但 ``services/persistence/*`` 仍从 stub 模块 ``GolemQ.fetch.kline``
导入（返回空结果）。本模块提供真实读路径，返回形态与既有调用点对齐，
从而修复那条「空数据 → IndexError → 完整性监控恒报 0%」的静默失败链。

数据布局
====================================================================
::

    golemq_stock_cn
      ├─ stock_1min | stock_5min | stock_15min | stock_30min | stock_60min
      └─ index_1min | index_5min | index_15min | index_30min | index_60min

集合名由 ``f'{market}_{frequency}'`` 推导，``market ∈ {'stock','index'}``。
时序规格 ``{timeField:'ts', metaField:'code', granularity:'minutes'|'hours'}``。

字段：``ts``(UTC Date) ``code`` ``datetime`` ``date`` ``date_stamp``(int32)
``time_stamp``(int32) ``open/close/high/low``(float) ``vol``(**int32**) ``amount``(float)；
``index_*`` 另有 ``up_count`` / ``down_count``。

.. warning::
   ``vol`` 必须是 **int32**，绝不能是 int16 —— 实测 ``stock_1min`` 单日最大
   ``vol`` 达 35,455,100、``index_1min`` 达 131,132,928，而 int16 上限仅 32,767。
   用 int16 会把成交量截断成垃圾数据。

为什么按 ``ts`` 查而不是 ``time_stamp``
====================================================================
时序集合的加速只对 ``timeField``(``ts``) 与 ``metaField``(``code``) 生效 ——
规划器靠它们做**分桶剪枝**。``time_stamp``(int32) 虽是 unix 秒、与时区无关，
但规划器不知道它与时间单调，用它过滤只能走普通二级索引，拿不到时序加速。
故本模块一律按 ``(code, ts)`` 查、按 ``ts`` 排序。

时区口径
====================================================================
全模块时区转换收敛到 :func:`_bj_date` 一处。``start`` / ``end`` 一律按
**北京时间**解释；返回的索引是**naive 北京时间**（与 QUANTAXIS 及下游
``services/persistence/`` 的处理口径一致）。

⚠️ 绝不要用 naive ``datetime`` 直接查 ``ts`` —— pymongo 会把 naive 当 **UTC**，
区间静默偏 8 小时。

调用契约（**空结果的处理刻意不对称，勿"修正"**）
====================================================================
* :func:`get_kline_price_v3`（日线）找不到数据时返回 ``None``。
  调用方 ``services/persistence/_daily.py:105`` 有 ``if data_baseline is None``
  分支，``_stock.py:87`` 也预先初始化了 ``kline_daily_baseline = None`` ——
  返回 ``None`` 命中两条既有保护。

* :func:`get_kline_price_min`（分钟线）找不到数据时返回**空 KlineResult**，
  不返回 ``None``。因为 ``_stock.py:138`` 是 ``kline_hour_baseline =
  hour_baseline.data`` —— 既无 ``None`` 判断、也未预先初始化该变量；
  返回 ``None`` 会抛 ``AttributeError`` 逃出 ``try``，最终在 ``:288`` 的
  ``return ... kline_hour_baseline ...`` 处炸成 ``UnboundLocalError``。

8.3 尚未有日线集合，故日线路径当前恒返回 ``None`` —— 这是**预期状态**，
不是缺陷。接口先在这里备好，日线迁移完成后无需改动本模块或 service 层。
"""
from __future__ import annotations

import datetime as dt

import pandas as pd

from GolemQ.core.constants import MARKET_TYPE

from . import DATABASE_STOCK_CN
from .symbol import is_stock_cn

__all__ = ['KlineResult', 'get_kline_price_min', 'get_kline_price_v3']


class KlineResult:
    """与 QUANTAXIS ``QA_DataStruct_*`` 对齐的最小接口 —— 调用方只取 ``.data``。

    刻意在本模块内定义而非复用 ``GolemQ.fetch.kline`` 的同名类：那个包整体是
    stub，本模块不依赖它，以免 stub 被删除时连带受影响。
    """

    def __init__(self, data=None):
        self.data = data if data is not None else pd.DataFrame()


def _bj_date(x):
    """北京时间的裸值 → **UTC-aware datetime**（给 ``ts`` 字段用）。

    本模块时区处理的唯一入口。裸时间一律先 ``tz_localize('Asia/Shanghai')``
    再转 UTC，避免 pymongo 把 naive 当 UTC 造成的静默 8 小时偏移。
    """
    if x is None:
        return None
    t = pd.Timestamp(x)
    if t.tzinfo is None:
        t = t.tz_localize('Asia/Shanghai')
    return t.tz_convert('UTC').to_pydatetime()


def _market_prefix(codelist, market_type=None):
    """决定读 ``stock_*`` 还是 ``index_*``。

    代码本身有歧义（``000001`` 既可能是上证指数也可能是平安银行），故复用
    既有的 :func:`is_stock_cn` 分类器，而不是自己猜。``market_type`` 显式传入
    时以传入值为准。
    """
    if market_type is None:
        probe = codelist[0] if isinstance(codelist, (list, tuple, set)) else codelist
        try:
            _, market_type, _, _ = is_stock_cn(probe)
        except Exception:
            market_type = MARKET_TYPE.STOCK_CN
    # ETF 基金在迁移时与指数共用 index_* 集合
    return 'index' if market_type in (MARKET_TYPE.INDEX_CN, MARKET_TYPE.FUND_CN) else 'stock'


def _read_timeseries(codelist, start, end, frequency, market,
                     default_days=800, limit=None):
    """按 ``(code, ts)`` 查询时序集合，返回 naive 北京时间索引的 DataFrame。

    返回空 DataFrame 表示无数据。查询强制带 code —— ``stock_1min`` 有十余亿行，
    任何全表扫描都不可接受。
    """
    name = f'{market}_{frequency}'
    coll = DATABASE_STOCK_CN[name]

    if isinstance(end, str) and len(end.strip()) == 10:
        # 只给到日则补到当天 23:59:59，否则 start=end='2017-03-14' 会塌成零长区间
        end = end.strip() + ' 23:59:59'
    hi = _bj_date(end) if end is not None else dt.datetime.now(dt.timezone.utc)
    lo = _bj_date(start) if start is not None else hi - dt.timedelta(days=default_days)

    codes = codelist if isinstance(codelist, (list, tuple, set)) else [codelist]
    codes = [str(c) for c in codes]

    query = {'code': {'$in': codes}, 'ts': {'$gte': lo, '$lte': hi}}
    cur = coll.find(query, {'_id': 0}).sort('ts', 1)
    if limit:
        cur = cur.limit(int(limit))

    df = pd.DataFrame(list(cur))
    if df.empty:
        return df

    # 迁移把成交量存为 ``vol``，但 QUANTAXIS 口径与全部下游消费方用的是
    # ``volume`` —— 如 markets/StockCN/fetch.py:829 的 ``data_min.data['volume']``
    # 与 GolemQ_old/analysis/ChipDistribution.py:65 的 ``VOLUME = 'volume'``。
    # 不在这里改名，真实消费方会静默拿不到该列。
    if 'vol' in df.columns and 'volume' not in df.columns:
        df = df.rename(columns={'vol': 'volume'})

    # ts 是 UTC-aware → 转北京后去掉时区，与下游的 naive 口径对齐
    ts_bj = pd.to_datetime(df['ts'], utc=True).dt.tz_convert('Asia/Shanghai')
    df['ts'] = ts_bj.dt.tz_localize(None)
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
    return df


def _to_kline_frame(df):
    """扁平表 → 以 ``(时间, code)`` 为 MultiIndex 的 DataFrame。

    调用方依赖此形状：``services/persistence/_stock.py:119`` 与
    ``_daily.py:122`` 都用 ``.index.get_level_values(level=0)`` 取时间轴，
    ``_review.py:113`` 还用 ``level=1`` 取 code。
    """
    if df.empty:
        # 空表也保留两层 MultiIndex 结构。若返回裸的 pd.DataFrame()（单层空
        # Index），下游的 index.get_level_values(level=1) 会抛
        # "Too many levels: Index has only 1 level, not 2" —— 那是形状错误，
        # 会被误读成 bug；而形状正确的空表会让调用方既有的 len(...) < 1
        # 判空逻辑正常工作，如实表达"这只标的没有数据"。
        return pd.DataFrame(
            index=pd.MultiIndex.from_arrays([[], []], names=['ts', 'code'])
        )
    return df.set_index(['ts', 'code']).sort_index()


def get_kline_price_min(codelist, start=None, market_type=None, frequency='60min',
                        verbose=True, end=None, realtime=True):
    """分钟线读取（8.3 时序）。返回 ``(KlineResult, codename)``。

    ``realtime=True`` 时**不做实时补数** —— 迁移数据已包含当日，实时拼接属另行
    实现，此处保留参数只为与 ``markets/StockCN/fetch.py:721`` 的同名函数签名
    兼容，使 service 层可以平滑替换。

    无数据时返回**空** ``KlineResult``（详见模块 docstring 的契约说明）。
    """
    market = _market_prefix(codelist, market_type)
    try:
        df = _read_timeseries(codelist, start, end, frequency, market)
    except Exception:
        if verbose:
            print(f'GolemQ Error: 读取 {market}_{frequency} 失败，'
                  f'code={codelist} start={start} end={end}')
        raise
    codename = codelist[0] if isinstance(codelist, (list, tuple, set)) else codelist
    return KlineResult(_to_kline_frame(df)), codename


def get_kline_price_v3(codelist, start=None, market_type=None, verbose=True,
                       end=None, realtime=None):
    """日线读取（8.3 时序，**尚未就绪**）。返回 ``(KlineResult | None, codename)``。

    8.3 上还没有 ``stock_day`` / ``index_day`` 集合 —— 日线迁移未完成。集合名
    仍按 ``f'{market}_{frequency}'`` 推导、``frequency='day'``，因此日线一旦迁移
    完成，本函数**无需改动**即可工作。

    在就绪之前，无数据时返回 ``None`` 并打印一条明确日志，而非静默返回空表 ——
    否则就是重演本模块所修复的那条静默失败链。
    """
    market = _market_prefix(codelist, market_type)
    frequency = 'day'
    try:
        df = _read_timeseries(codelist, start, end, frequency, market)
    except Exception:
        if verbose:
            print(f'GolemQ Error: 读取 {market}_{frequency} 失败，'
                  f'code={codelist} start={start} end={end}')
        raise

    codename = codelist[0] if isinstance(codelist, (list, tuple, set)) else codelist
    if df.empty:
        if verbose:
            print(f'GolemQ Warning: {market}_{frequency} 在 8.3 无数据 '
                  f'（日线迁移未完成），返回 None。code={codelist}')
        return None, codename
    return KlineResult(_to_kline_frame(df)), codename
