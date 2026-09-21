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

日线集合**已就绪**（2026-09-20 实测：``stock_day`` 17,871,122 行、
``index_day`` 8,602,127 行），故 :func:`get_kline_price_v3` 与
:func:`GQ_fetch_stock_day_adv` 的日线路径已是活路径。早先「日线迁移未完成、
恒返回 ``None``」的说法已作废 —— 当时确实如此，现在不是。
"""
from __future__ import annotations

import datetime as dt

import pandas as pd

from GolemQ.core.constants import MARKET_TYPE

from .datastruct import apply_qfq, frame_to_datastruct
from .etf_fq import GQ_apply_etf_qfq
from .symbol import is_stock_cn

__all__ = [
    'KlineResult',
    'FREQUENCY_ALIASES',
    'normalize_frequency',
    'bj_date',
    'market_prefix',
    'read_min_frame',
    'GQ_fetch_stock_day_adv',
    'get_kline_price_min',
    'get_kline_price_v3',
]

# 日线的默认回看窗口。``_read_timeseries`` 的 800 天默认是给分钟线定的；
# 日线上 ``start=None`` 时 800 天只够 3 年，会静默截断到 1990 年至今的一小段。
# A 股最早的数据是 1991-12-23（实测 ``stock_day`` 里 ``000001`` 的首行）。
_DAY_DEFAULT_DAYS = 365 * 40


class KlineResult:
    """与 QUANTAXIS ``QA_DataStruct_*`` 对齐的最小接口 —— 调用方只取 ``.data``。

    刻意在本模块内定义而非复用 ``GolemQ.fetch.kline`` 的同名类：那个包整体是
    stub，本模块不依赖它，以免 stub 被删除时连带受影响。
    """

    def __init__(self, data=None):
        self.data = data if data is not None else pd.DataFrame()


def bj_date(x):
    """北京时间的裸值 → **UTC-aware datetime**（给 ``ts`` 字段用）。

    时区处理的唯一入口（**公开**，因为实时落库那条路也要用同一个换算 ——
    `realtime.py` 的 `_l2_row` / L1 订阅器都调它）。裸时间一律先
    ``tz_localize('Asia/Shanghai')`` 再转 UTC，避免 pymongo 把 naive 当 UTC
    造成的静默 8 小时偏移。
    """
    if x is None:
        return None
    t = pd.Timestamp(x)
    if t.tzinfo is None:
        t = t.tz_localize('Asia/Shanghai')
    return t.tz_convert('UTC').to_pydatetime()


# 旧名保留为别名（本模块历史上叫 _bj_date）。
_bj_date = bj_date


#: 频率别名 → 8.3 集合名里的规范频率。集合名是 ``f'{market}_{frequency}'``，
#: 所以规范值必须是 ``1min``/``5min``/``15min``/``30min``/``60min``。
FREQUENCY_ALIASES = {
    '1m': '1min', '1min': '1min',
    '5m': '5min', '5min': '5min',
    '15m': '15min', '15min': '15min',
    '30m': '30min', '30min': '30min',
    '60m': '60min', '60min': '60min',
}


def normalize_frequency(frequence):
    """频率别名 → 规范频率；不认识的值抛 ``ValueError``。

    **幂等**：规范值传进来还是它自己，所以各层都可以放心再调一次。

    旧实现把这张表抄了三份（`fetch.py` 的 `GQ_fetch_stock_min` 与
    `GQ_fetch_stock_min_adv`、`quotes.py` 的 `get_kline_quotes_min`），三份的
    失败处理还不一样：`GQ_fetch_stock_min` 只打印一行就**继续拿原值拼集合名**
    （拼出不存在的集合 → 静默空结果），`_adv` 返回 `None`，`quotes.py` 直接沿用
    原值。收敛到一处后统一**抛错** —— 未知频率是编程错误，不该退化成空数据。
    """
    key = str(frequence).strip().lower()
    if key not in FREQUENCY_ALIASES:
        raise ValueError(
            f'未知的频率 {frequence!r}；支持 '
            f'{sorted(set(FREQUENCY_ALIASES))}')
    return FREQUENCY_ALIASES[key]


def market_prefix(codelist, market_type=None):
    """决定读 ``stock_*`` 还是 ``index_*``。

    代码本身有歧义（``000001`` 既可能是上证指数也可能是平安银行），故复用
    既有的 :func:`is_stock_cn` 分类器，而不是自己猜。``market_type`` 显式传入
    时以传入值为准。

    公开这个函数是因为**装 K 线数据的容器类型也由它决定**：ETF 与指数共用
    ``index_*`` 集合，因此拿到的是指数类容器（没有 ``to_qfq``，复权归
    ``etf_fq.py``）。`fetch.py` 的 ``GQ_fetch_stock_min_adv`` 据此选容器类。
    """
    if market_type is None:
        probe = codelist[0] if isinstance(codelist, (list, tuple, set)) else codelist
        try:
            _, market_type, _, _ = is_stock_cn(probe)
        except Exception:
            market_type = MARKET_TYPE.STOCK_CN
    # ETF 基金在迁移时与指数共用 index_* 集合
    return 'index' if market_type in (MARKET_TYPE.INDEX_CN, MARKET_TYPE.FUND_CN) else 'stock'


# 旧名，模块内部仍在用；新代码用 `market_prefix`。
_market_prefix = market_prefix


def _collection(name):
    """取 8.3 库里的集合句柄。**函数级导入是刻意的。**

    ``markets/StockCN/__init__.py:55`` 先 ``from .quotes import StockCNQuotes``，
    到 ``:70`` 才定义 ``DATABASE_STOCK_CN``。``quotes.py`` 会在模块级导入本模块，
    所以顶层 ``from . import DATABASE_STOCK_CN`` 会抛
    ``ImportError: cannot import name ... from partially initialized module``。
    ``datastruct.py`` 出于同一原因也这样写。
    """
    from . import DATABASE_STOCK_CN
    return DATABASE_STOCK_CN[name]


def _read_timeseries(codelist, start, end, frequency, market,
                     default_days=800, limit=None):
    """按 ``(code, ts)`` 查询时序集合，返回 naive 北京时间索引的 DataFrame。

    返回空 DataFrame 表示无数据。查询强制带 code —— ``stock_1min`` 有十余亿行，
    任何全表扫描都不可接受。
    """
    name = f'{market}_{frequency}'
    coll = _collection(name)

    if isinstance(end, str) and len(end.strip()) == 10:
        # 只给到日则补到当天 23:59:59，否则 start=end='2017-03-14' 会塌成零长区间
        end = end.strip() + ' 23:59:59'
    hi = bj_date(end) if end is not None else dt.datetime.now(dt.timezone.utc)
    lo = bj_date(start) if start is not None else hi - dt.timedelta(days=default_days)

    codes = codelist if isinstance(codelist, (list, tuple, set)) else [codelist]
    # 一律截到 6 位。集合里存的是 6 位代码，而调用方常传带后缀的形式
    # （`normalize_code()` 给的是 `'600519.XSHG'`）—— 直接拿去查会**静默读空**，
    # 症状是「这只票没有分钟数据」。`code[:6]` 是全树既有的惯例写法。
    # 注意这也让 `market_prefix` 对带后缀代码的判断保持一致（它由 `is_stock_cn`
    # 处理，同样只看前 6 位）。
    codes = [str(c)[:6] for c in codes]

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
    result = KlineResult(_to_kline_frame(df))
    _apply_adjustments(result, market, codelist, verbose=verbose)
    return result, codename


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
            print(f'GolemQ Warning: {market}_{frequency} 在 8.3 无数据，'
                  f'返回 None。code={codelist} start={start} end={end}')
        return None, codename
    result = KlineResult(_to_kline_frame(df))
    _apply_adjustments(result, market, codelist, verbose=verbose)
    return result, codename


def _apply_adjustments(result, market, codelist, verbose=False):
    """给读取结果套上该市场的复权。两条**互斥**路径，与老树同构。

    - ``market == 'stock'`` → 股票前复权（因子表 ``stock_adj``）。
      老树在股票分支里调的是 QUANTAXIS 的 ``to_qfq()``（日线 `kline.py:906`、
      分钟 `:1484`）；重构只把 ETF 那条接了过来，于是**股票返回不复权价** ——
      见 ``MIGRATION_STATUS.md`` HIGH #11。
    - ``market == 'index'`` → ETF 前复权（因子表 ``etf_adj``）。真指数在
      ``etf_adj`` 里没有行，是 no-op，且一次 Mongo 都不查。

    互斥由 ``market`` 保证：ETF 经 :func:`market_prefix` 归类为 ``'index'``，
    所以股票分支不会碰到 ETF 数据、反之亦然。**每条路径只应用一次** —— 重复
    应用会二次缩放（`etf_fq.py` 的 docstring 记着这个坑）。

    内存量级与改动前一致：股票因子是一次 ``$in`` 区间查询后整体 join，老树的
    ``to_qfq()`` 也是这么做的（``_QA_fetch_stock_adj`` + ``join``）。
    """
    if result is None or len(getattr(result, 'data', ())) == 0:
        return result
    if market == 'stock':
        result.data = apply_qfq(result.data, verbose=verbose)
        try:
            result.if_fq = 'qfq'
        except Exception:  # noqa: BLE001
            pass
    GQ_apply_etf_qfq(result, codelist=codelist, verbose=verbose)
    return result


def read_min_frame(codelist, start=None, end=None, frequence='1min',
                   market_type=None):
    """分钟线**扁平帧**（8.3 时序）。列含 ``datetime`` / ``code`` / ``volume``。

    与 :func:`get_kline_price_min` 的区别：那个返回 ``(KlineResult, codename)``
    给门面用，本函数返回裸 DataFrame 给需要自己摆索引的调用方（如
    ``fetch.py`` 的 ``GQ_fetch_stock_min``）。时间一律是 **naive 北京时间**，
    耗时区换算在 :func:`_read_timeseries` 里一处完成。

    无数据返回**空 DataFrame**（不是 ``None``）—— 由调用方按各自契约处理：
    ``GQ_fetch_stock_min`` 把空表映射成 ``None``，``_adv`` 再映射成空容器。

    市场（``stock_*`` / ``index_*`` 集合）由 :func:`market_prefix` 判定；
    ETF 与指数共用 ``index_*``，所以「按代码段猜」会读空。
    """
    frequency = normalize_frequency(frequence)
    market = market_prefix(codelist, market_type)
    return _read_timeseries(codelist, start, end, frequency, market)


def GQ_fetch_stock_day_adv(codelist, start=None, end=None, market_type=None,
                           verbose=False):
    """日线 → ``GQ_DataStruct_*``。**替代 QUANTAXIS 的 ``QA_fetch_stock_day_adv``。**

    存在的理由：``quotes.py`` 与 ``fetch.py`` 直接调 QUANTAXIS 的那个函数，而它
    读的是 4.4 的 ``quantaxis`` 库。本函数读 8.3 的 ``stock_day`` / ``index_day``
    —— 实测**两个库的同名集合逐行相同**（``600519`` 各 6006 行、``000001`` 各
    8252 行，日期范围一致、收盘价最大差 0），所以换读 8.3 不是换数据源。

    返回类型按标的自动选：股票 → ``GQ_DataStruct_Stock_day``（有 ``to_qfq``）；
    指数 / ETF → ``GQ_DataStruct_Index_day``（**没有** ``to_qfq``，与 QUANTAXIS
    一致，见 ``datastruct.py`` 的模块说明）。ETF 与指数共用 ``index_*`` 集合，
    分类交给既有的 :func:`is_stock_cn`，不自己猜代码段。

    **无数据返回 ``None``**，与 QUANTAXIS 一致 —— 调用方有 ``is None`` 分支
    （``fetch.py:1264``、``quotes.py:64``），见 ``markets/base_market.py``
    关于空值契约刻意不对称的说明。
    """
    market = _market_prefix(codelist, market_type)
    df = _read_timeseries(codelist, start, end, 'day', market,
                          default_days=_DAY_DEFAULT_DAYS)
    if df.empty:
        if verbose:
            print(f'GolemQ Warning: {market}_day 无数据，返回 None。'
                  f'code={codelist} start={start} end={end}')
        return None
    return frame_to_datastruct(df, market=market, frequency='day')
