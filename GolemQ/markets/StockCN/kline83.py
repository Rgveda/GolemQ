# coding:utf-8
"""A 股 K 线读取器 —— 走 MongoDB 8.3 时序库（``golemq_stock_cn``）。

为什么存在这一层
====================================================================
分钟线已从旧 GolemQ 系统的 MongoDB 4.4（``stock_min`` / ``index_min``）迁移到
8.3 的时序集合。本模块提供真实读路径（2026-09 迁移时它替换掉了原先那个返回
空结果的 stub，从而修复「空数据 → IndexError → 完整性监控恒报 0%」的静默失败链）。
⚠️ 2026-10-10 起 `GolemQ/fetch/` 整包已删，取数入口是
``GolemQ.get_active_market().get_kline_price_min(...)``。

数据布局
====================================================================
::

    golemq_stock_cn
      ├─ stock_1min | stock_5min | stock_15min | stock_30min | stock_60min
      ├─ index_1min | index_5min | index_15min | index_30min | index_60min
      └─ etf_1min   | etf_5min   | etf_15min   | etf_30min   | etf_60min

集合名由 ``f'{market}_{frequency}'`` 推导，``market ∈ {'stock','index','etf'}``
（由 :func:`market_prefix` 判定，见其 docstring 与 doctest）。

⚠️ ``etf_*`` 是 2026-09 从 ``index_*`` **拆出来**的 —— 此前 ETF 与真指数共用
``index_*``（老树 `save_qa.py` 刻意「与 QUANTAXIS 一致」）。``etf_*`` 与
``index_*`` 字段同形。
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

from GolemQ.core.constants import MARKET_TYPE, epoch_divisor

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

    刻意在本模块内定义而不是从别处复用：本模块是**叶子**，不该依赖任何上层门面
    —— 上层被改/被删（`GolemQ/fetch/` 就于 2026-10-10 整包删除）时，读路径不该受影响。
    """

    def __init__(self, data=None):
        self.data = data if data is not None else pd.DataFrame()


def bj_date(x):
    """北京时间的裸值 → **UTC-aware datetime**（给 ``ts`` 字段用）。

    时区处理的唯一入口（**公开**，因为实时落库那条路也要用同一个换算 ——
    `realtime.py` 的 L1 订阅器调它）。裸时间一律先
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


def epoch_seconds(series):
    """**tz-aware** 时间 Series → 秒级 epoch（`int64`）。**与 pandas 版本无关**。

    ⚠️ 为什么要这个函数（2026-10-17 实测踩到）
    =========================================
    **pandas 3.0 把默认 datetime 精度从 ns 改成 µs**（本机 3.0.6 实测
    `pd.to_datetime(...).dtype == datetime64[us]`）。于是老写法

    ```python
    series.astype(np.int64) // 10 ** 9
    ```

    在 pandas 3 上算出来**少 1000 倍**（拿到的是微秒再除 1e9），而且**不报错**
    —— 只是把 `date_stamp` / `time_stamp` 这一类**键**写成另一个数量级。
    实测当场红了 `reality_feat_rows` 的 doctest（`1767596` vs `1767596400`）。

    **除数不写死、也不是一个全局标量**：用 `constants.epoch_divisor(series.dtype)`
    **按这条序列自己的精度**查表（s / ms / us / ns ⇒ 1 / 1e3 / 1e6 / 1e9）。
    单标量会算错 —— pandas 3 会按输入推不同精度：`pd.to_datetime(date 对象)` 得到
    `datetime64[s]`，而字符串/Series 得到 `datetime64[us]`，两者并存（实测）。
    全树做「datetime → 秒」只准走这里（有 `test_epoch_divisor` 守着）。

    tz-aware → 先 `tz_convert('UTC')` 再 `tz_localize(None)`（**顺序不能反**：
    `tz_localize(None)` 只丢时区、保留墙上时间）。

    >>> import pandas as pd
    >>> s = pd.Series(pd.to_datetime(['2026-01-05 15:00'])).dt.tz_localize('Asia/Shanghai')
    >>> int(epoch_seconds(s).iloc[0])          # 北京 15:00 = 07:00Z
    1767596400
    >>> epoch_seconds(s).dtype.name
    'int64'

    **`Series` 与 `DatetimeIndex` 都收**（`reality_save` 那边 `day_dt` 是 index、
    `full_dt` 是 Series；`.dt` 只有 Series 有 —— 所以先统一成 Series）：

    >>> idx = pd.DatetimeIndex(['2026-01-05 15:00']).tz_localize('Asia/Shanghai')
    >>> int(epoch_seconds(idx).iloc[0])
    1767596400
    """
    if not isinstance(series, pd.Series):
        series = pd.Series(series)
    naive = series.dt.tz_convert('UTC').dt.tz_localize(None)
    return naive.astype('int64') // epoch_divisor(naive.dtype)


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
    """决定读哪一族集合 —— ``'stock'`` / ``'index'`` / ``'etf'``。

    代码本身有歧义（``000001`` 既可能是上证指数也可能是平安银行），故复用
    既有的 :func:`is_stock_cn` 分类器，而不是自己猜。``market_type`` 显式传入
    时以传入值为准。

    公开这个函数是因为**装 K 线数据的容器类型也由它决定**：`fetch.py` 的
    ``GQ_fetch_stock_min_adv`` 据此选容器类。

    ⚠️ **ETF 2026-09 起独立成 ``etf_*``**。此前 ETF 被归为 ``INDEX_CN``，
    与真指数**共用** ``index_*``（老树 `save_qa.py` 刻意「与 QUANTAXIS 一致」），
    因此拿到的是指数类容器（没有 ``to_qfq``）。现在 ETF 有自己的集合与容器。

    ⚠️ **``FUND_CN`` 仍走 ``index_*``** —— 那是 ``50x`` 封基/LOF/分级，
    本次**有意不动**（它们的行情本就不在 ``etf_*``）。别顺手"统一"进来。

    >>> market_prefix('600519')                    # 股票
    'stock'
    >>> market_prefix('200037')                    # 深市B股，曾被判成 ETF
    'stock'
    >>> market_prefix('510300')                    # ETF
    'etf'
    >>> market_prefix('399001')                    # 真指数
    'index'
    >>> market_prefix('150001')                    # 分级基金 —— 基金，不是 ETF
    'index'
    >>> market_prefix('000001', market_type=MARKET_TYPE.ETF_CN)   # 显式传入优先
    'etf'
    """
    if market_type is None:
        probe = codelist[0] if isinstance(codelist, (list, tuple, set)) else codelist
        try:
            _, market_type, _, _ = is_stock_cn(probe)
        except Exception:
            market_type = MARKET_TYPE.STOCK_CN
    if market_type == MARKET_TYPE.ETF_CN:
        return 'etf'
    # 真指数与基金（50x 封基/LOF/分级）仍共用 index_* —— 见上方的 ⚠️
    if market_type in (MARKET_TYPE.INDEX_CN, MARKET_TYPE.FUND_CN):
        return 'index'
    return 'stock'


# 旧名，模块内部仍在用；新代码用 `market_prefix`。
_market_prefix = market_prefix


def _collection(name):
    """取 8.3 库里的集合句柄。**函数级导入是刻意的。**

    ``markets/StockCN/__init__.py:55`` 先 ``from .quotes import StockCNQuotes``，
    到 ``:70`` 才定义 ``GOLEMQ_STOCK_CN``。``quotes.py`` 会在模块级导入本模块，
    所以顶层 ``from . import GOLEMQ_STOCK_CN`` 会抛
    ``ImportError: cannot import name ... from partially initialized module``。
    ``datastruct.py`` 出于同一原因也这样写。
    """
    from . import GOLEMQ_STOCK_CN
    return GOLEMQ_STOCK_CN[name]


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


def _display_name(code, market):
    """标的的**展示名**（`'301308'` → `'江波龙'`）—— 查不到就回**代码本身**，绝不抛。

    ⚠️ **这是补一个移植缺口，不是新功能**：旧树这里调的是
    ``codename = GQ_fetch_stock_name(codelist)``（**真名字**，见
    ``GolemQ_old/fetch/kline.py:247``），重构时被简化成 ``codelist[0]`` ——
    于是凡是把第二个返回值当**名字**显示的地方，全印成 ``301308（301308）``。
    实证：`ChipDistribution_jit.py:695` 的接收变量名就叫 ``stock_name_faked``。

    三个约束：

    1. **不许抛**。名字是**展示用品**，查不到不该让取数失败。
    2. **逐个回落**，每一步都是按 `code` 索引的小查询 —— 不在全表上找。
    3. **按 `market` 选链**：``'stock'`` → ``stock_info`` → ``stock_list``；
       ``'index'``（ETF 与真指数**共用** `index_*` 集合）→ ``etf_list``。

    ⚠️ **实测（2026-10-11）：`stock_list` 这一跳今天一次都不会触发。**
    两张表都是 **5579 行、351 个 `name` 为空/null**，且「只在 `stock_list` 里有」
    的代码 **0 个**、「`stock_info` 取不到而 `stock_list` 能取到」的 **0 个** ——
    覆盖与缺口**完全一致**。留着它是**防两张表将来分叉**（不同源刷新时很容易），
    代价只在首次落空时多一次查询。
    ⇒ 那 351 个代码**无论走哪条链都会回落成代码本身**，三个 demo 上仍显示
    ``代码（代码）``。要根治得先去补**数据源**，不是在这里加分支。

    ⚠️ **`index` 分支是两条**：真指数在 `index_list`（2026-10-11 新建的表，
    此前 8.3 根本没有 —— 4.4 那张 1291 行的漏迁了），ETF 在 `etf_list`。
    两者**码段不重叠**，所以查两次是确定的、不是碰运气。
    """
    from . import refdata          # 函数级导入 —— 与 `datastruct._adj_coll` 同一手法
    readers = ([refdata.GQ_fetch_stock_info, refdata.GQ_fetch_stock_list]
               if market == 'stock'
               else [refdata.GQ_fetch_index_list, refdata.GQ_fetch_etf_list])
    for fetch in readers:
        try:
            df = fetch(codes=[code])
            if not df.empty and 'name' in df.columns:
                name = df['name'].iloc[0]
                if isinstance(name, str) and name.strip():
                    return name.strip()
        except Exception:                   # noqa: BLE001 —— 见上面第 1 条
            continue
    return code


def get_kline_price_min(codelist, start=None, market_type=None, frequency='60min',
                        verbose=True, end=None, realtime=True, if_fq='qfq'):
    """分钟线读取（8.3 时序）。返回 ``(KlineResult, codename)``。

    ⚠️ **`codename` 是标的的展示名**（``'301308'`` → ``'江波龙'``），**不是代码** ——
    由 :func:`_display_name` 解析，查不到时**回落成代码本身**（真指数与未知代码
    都走这条）。旧树返回的也是名字（`GQ_fetch_stock_name`），重构时被简化成代码，
    2026-10-11 补回。

    ``realtime=True``（**默认**）→ **带 REALTIME**：读到的历史再与
    ``realtime_YYYY-MM-DD`` 的 L1 tick 合成（见 :func:`_merge_realtime`）。

    ⚠️ **2026-10-09 改**：此前这个形参是**空转**的（旧 docstring 写「不做实时补数，
    保留参数只为签名兼容」）—— 用户要做分析、明确要「带 REALTIME」，故接通。
    不需要补数时**代价只有一次 `list_collection_names()`**（~1ms），见 `_merge_realtime` 的两道门。

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
    # 展示名（旧树给的是真名字，见 :func:`_display_name`）
    codename = _display_name(codelist[0] if isinstance(codelist, (list, tuple, set))
                             else codelist, market)
    result = KlineResult(_to_kline_frame(df))
    _apply_adjustments(result, market, codelist, if_fq=if_fq, verbose=verbose)
    if realtime:
        # **带 REALTIME**（用户 2026-10-09：「我要做分析了，取行情，8.3库，带 REALTIME」）。
        # ⚠️ 本函数那个 `realtime=True` **从"空转"变成"真的合并"** —— 但不需要补数时
        # 代价只有一次 `list_collection_names()`（见 :func:`_merge_realtime` 的两道门）。
        _merge_realtime(result, codelist, frequency, verbose=verbose)
    return result, codename


def get_kline_price_v3(codelist, start=None, market_type=None, verbose=True,
                       end=None, realtime=True, if_fq='qfq'):
    """日线读取（8.3 时序）。返回 ``(KlineResult | None, codename)``。

    ⚠️ **`codename` 是标的的展示名**（``'301308'`` → ``'江波龙'``），**不是代码**；
    查不到时回落成代码本身。详见 :func:`get_kline_price_min` 的同一条说明。

    ⚠️ **先前这里写着「尚未就绪 / 日线未迁移」—— 那条已作废**（2026-10-08 实测：
    `stock_day` 有 **17,893,343** 行、`index_day` 4,564,152 行，日线是迁全了的；
    当时那个"未迁移"的结论来自时序集合上**不可靠的计数方式**，见 `PITFALLS.md` P15）。

    集合名按 ``f'{market}_{frequency}'`` 推导、``frequency='day'``。
    无数据时返回 ``None`` 并打印明确日志，不静默返回空表 —— 后者是重演本模块
    所修复的那条静默失败链。

    ``realtime=True``（**默认**）→ **带 REALTIME**：与 ``realtime_YYYY-MM-DD``
    的 L1 tick 合成（见 :func:`_merge_realtime`）。

    ⚠️ **默认值 2026-10-09 由 `None` 改成 `True`**：`base_market.py` 的抽象声明与
    市场方法（`markets/StockCN/__init__.py`）**一直写的就是 `True`**，而实现写
    `None` —— 以前是空转所以看不出来，接通后必须对齐。改它是**行为保持**的：
    市场方法**显式传** `realtime=realtime`，所以现有调用方本来就一直在传 `True`。
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

    # 展示名（旧树给的是真名字，见 :func:`_display_name`）
    codename = _display_name(codelist[0] if isinstance(codelist, (list, tuple, set))
                             else codelist, market)
    if df.empty:
        if verbose:
            print(f'GolemQ Warning: {market}_{frequency} 在 8.3 无数据，'
                  f'返回 None。code={codelist} start={start} end={end}')
        return None, codename
    result = KlineResult(_to_kline_frame(df))
    _apply_adjustments(result, market, codelist, if_fq=if_fq, verbose=verbose)
    if realtime:
        _merge_realtime(result, codelist, 'day', verbose=verbose)
    return result, codename


def _merge_realtime(result, codelist, frequency, verbose=False):
    """**带 REALTIME**：把 ``realtime_YYYY-MM-DD`` 的 L1 tick 合成进结果（就地改 `result`）。

    这是「历史 + 盘中」那条路的**读侧入口**（写侧是 `--sub`；合成是 `realtime.py`
    的两个 `GQ_fetch_*_realtime_adv`）。此前 `get_kline_price_min` / `get_kline_price_v3`
    的 `realtime` 形参是**空转**的（前一版的 docstring 明说「不做实时补数」）。

    ⚠️ **两道门，都是为了「不需要补数时与不带 REALTIME 同速」**（用户 2026-10-09 要求）：

    1. **集合不存在就直接返回** —— `realtime_ts_collection` 是
       `create_collection` 包了 `try`，**读它会把不存在的集合建出来**（读路径的
       写副作用）；一次 `list_collection_names()`（~1ms）就能挡掉，且顺带省掉
       后面整条 tick 读。
    2. 合并器**自带判据**（「历史最后一根落后最后交易日 >10h（盘中）/ >40h（盘前）
       才去读 tick」），所以正常情况下它连 tick 集合都不读。

    ⚠️ **实测边界**：本机 `golemq_stock_cn_realtime` 现在是**空的**（`--sub` 尚未跑过，
    周一起才有数据），所以目前只验到「门挡掉、结果不变、耗时不增」；
    **tick 合成那一半要等实时库里有数据才验得了**。
    """
    from . import GOLEMQ_STOCK_CN_REALTIME as rt_db
    from .date_utils import GQ_util_get_last_day
    from .realtime import realtime_collection_name

    day = GQ_util_get_last_day()
    name = realtime_collection_name(day)
    try:
        if name not in rt_db.list_collection_names():
            if verbose:
                print(f'[realtime] {name} 不存在（今天还没跑 --sub）→ 不合并')
            return result
    except Exception:      # noqa: BLE001 实时库连不上不该毁掉一次成功的历史读
        if verbose:
            print(f'[realtime] 探测 {name} 失败 → 不合并')
        return result

    # 延迟 import：`realtime.py` 与本模块没有互相 import，但它拖进的东西不少
    # （requests / tqdm / easyquotation …），读历史时不必付这份 import 代价。
    try:
        if frequency == 'day':
            from .realtime import GQ_fetch_stock_day_realtime_adv
            return GQ_fetch_stock_day_realtime_adv(codelist, result, verbose=verbose)
        from .realtime import GQ_fetch_stock_min_realtime_adv
        return GQ_fetch_stock_min_realtime_adv(codelist, result, frequency,
                                               verbose=verbose)
    except Exception as exc:      # noqa: BLE001 合并失败不该把读到手的历史也丢了
        if verbose:
            print(f'[realtime] 合并失败（{type(exc).__name__}: {exc}）→ 返回纯历史')
        return result


def _apply_adjustments(result, market, codelist, if_fq='qfq', verbose=False):
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
    if if_fq == 'bfq':
        # **不复权**：两支都跳（一次 Mongo 都不查）。容器的 `if_fq` 本就是 'bfq' 默认值。
        return result
    if market == 'stock':
        result.data = apply_qfq(result.data, if_fq=if_fq, verbose=verbose)
        try:
            result.if_fq = if_fq
        except Exception:  # noqa: BLE001
            pass
    GQ_apply_etf_qfq(result, codelist=codelist, if_fq=if_fq, verbose=verbose)
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
