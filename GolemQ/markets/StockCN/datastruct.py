# coding:utf-8
"""QUANTAXIS ``QA_DataStruct_*`` 的替身 —— 只覆盖**实际被调用**的接口。

## 为什么要有这个模块

`markets/StockCN` 里有两个消费方用 `QA_DataStruct_*` 包 K 线 DataFrame：
`fetch.py`（`get_kline_price_v3` 家族）与 `quotes.py`（`StockCNQuotes`）。
解耦要求去掉这个依赖，而**实测调用面比预想小得多**：

============ ==================================================================
`.data`      读写（`quotes.py` 往里加 `FULL_SYMBOL` / `MARKET_TYPE` 列）
`.to_qfq()`  **仅股票**。见下
`.select_code()`  `realtime.py:630/632/633/877`
`isinstance` 区分「股票」与「指数/ETF」（`fetch.py:1116/1449`）
============ ==================================================================

## 两条照搬 QUANTAXIS 的形状，不要「改进」

**① 必须有类层级，不能塌缩成一个类。** 上面那两个 `isinstance` 就是靠类型
区分股票与指数，来决定去 `GQ_fetch_stock_name` 还是 `GQ_fetch_etf_name` 取名字。

**② `to_qfq()` 只挂在 Stock 类上。** 实测 `QA_DataStruct_Index_day` /
`QA_DataStruct_Index_min` **没有** `to_qfq`。ETF 在迁移里与指数共用 `index_*`
集合，因此走的是指数分支，拿不到 `to_qfq` —— 老树的 ETF 复权由
`GQ_apply_etf_qfq`（`GolemQ_old/markets/StockCN/etf_fq.py`）单独负责，**该模块
在本树尚未移植**。不要把 `to_qfq` 提到基类来「补全」，那会改变 ETF 的行为。

## 为什么放在 `markets/StockCN` 而不是根目录

复权因子来自 A 股特有的 `stock_adj` 集合，换市场不适用。与 `symbol.py` /
`scribe.py` / `kline83.py` 同层。

## 调用面之外的东西一律没实现

QUANTAXIS 的 datastruct 有几十个方法（`select_time` / `get_bar` / `resample`
…），本模块**一个都没搬**。用到就报 `AttributeError` —— 静默返回空值会让
「没实现」与「真的没数据」无从区分，这是本树反复踩过的坑。

## 一处**刻意不照搬**的行为：替身不会吞掉重复行

QUANTAXIS 的基类构造 `_quotation_base.__init__` 里有这么一行::

    self.data = DataFrame.drop_duplicates().sort_index()

`drop_duplicates()` **不带参数时只看列值、不看索引**，而此刻列已被裁到
`open/high/low/close/volume/amount`、`date`/`code` 进了索引。于是**任何两根
OHLCV 完全相同的 K 线，后一根会被静默删掉**。

实测（`000001`，全历史 8252 行）::

    1993-06-04  29.00 29.2 28.9 29.10 15060.0 43824600.0   ← 与 06-03 完全相同
    1998-06-20  18.91 19.0  18.6 18.79 24556.0 46049048.0   ← 与 06-19 完全相同

这两根被 QUANTAXIS 吞掉 → 它的帧只有 8250 行。**它们是真实交易日**
（1993-06-04 在 `TRADE_DATE_SSE` 里），老树同样会丢（同一个库），所以这是
QUANTAXIS 自身的缺陷、**不是重构回归**。替身不复制它：本模块返回完整的
8252 行。这条差异会让新旧路径的行数对不上，**是预期的**，不是 bug。
"""

import numpy as np
import pandas as pd

__all__ = [
    'GQ_DataStruct',
    'GQ_DataStruct_Stock_day',
    'GQ_DataStruct_Stock_min',
    'GQ_DataStruct_Index_day',
    'GQ_DataStruct_Index_min',
    'GQ_DataStruct_Stock_block',
    'apply_qfq',
    'frame_to_datastruct',
]

# QUANTAXIS 的日线 datastruct 对外只暴露这 6 列（实测 `QA_fetch_stock_day_adv`
# 的 `.data.columns`）。8.3 的 `stock_day` 还有 `ts` / `date_stamp` /
# `time_stamp`，替身里一律不保留 —— 消费方按列名取值，多出来的列只会让人
# 误以为可以依赖。
_OHLCV_COLUMNS = ('open', 'high', 'low', 'close', 'volume', 'amount')

# 复权的日期级字段名。日线是 `date`，分钟线是 `datetime`。
_DATE_LEVELS = ('date', 'datetime')


def _adj_collection():
    """`stock_adj` 集合。**函数级导入**是刻意的。

    `markets/StockCN/__init__.py:55` 先 `from .quotes import StockCNQuotes`，
    到 `:70` 才定义 `DATABASE_STOCK_CN` —— 本模块在那一刻被间接导入，模块级
    `from . import DATABASE_STOCK_CN` 会直接 `ImportError`。
    """
    from . import DATABASE_STOCK_CN
    return DATABASE_STOCK_CN['stock_adj']


def _row_dates(data):
    """每行的所属日期（`'YYYY-MM-DD'` 字符串），找不到返回 `None`。

    日线的 index level 0 就是日期；分钟线是 `datetime`，取 `'%Y-%m-%d'`
    —— 复权系数逐日恒定，日内所有 bar 同系数。
    """
    idx = data.index
    nlevels = idx.nlevels if isinstance(idx, pd.MultiIndex) else 1
    for lv in range(nlevels):
        vals = idx.get_level_values(lv)
        if pd.api.types.is_datetime64_any_dtype(vals):
            return pd.Series(
                pd.DatetimeIndex(vals).strftime('%Y-%m-%d'), index=data.index)
    for name in _DATE_LEVELS:
        if name in data.columns:
            return data[name].astype(str).str[:10]
    return None


def _row_codes(data):
    """每行的 6 位代码；从 index 的 `code` level 或 `code` 列取，都没有则 `None`。"""
    idx = data.index
    if isinstance(idx, pd.MultiIndex) and 'code' in (idx.names or []):
        return pd.Series(
            idx.get_level_values('code').astype(str).str[:6], index=data.index)
    if 'code' in data.columns:
        return data['code'].astype(str).str[:6]
    return None


def _adj_frame(codes, dmin, dmax):
    """`stock_adj` 在 `[dmin, dmax]` 内的 `(date, code) -> adj` 因子表。

    `date` 在这个集合里是**字符串** `'YYYY-MM-DD'`，所以范围查询必须传字符串
    —— 传 `datetime` 会静默返回空集（不抛异常）。见 `PITFALLS.md`。
    """
    cur = _adj_collection().find(
        {'code': {'$in': list(codes)}, 'date': {'$gte': dmin, '$lte': dmax}},
        {'_id': 0, 'code': 1, 'date': 1, 'adj': 1},
    )
    df = pd.DataFrame(list(cur))
    if df.empty:
        return df
    df['code'] = df['code'].astype(str).str[:6]
    return df


def apply_qfq(data, verbose=False):
    """就地把 OHLC 乘上前复权因子，**返回新的 DataFrame**（不改入参）。

    `adj` 列的来源是 8.3 的 `stock_adj` —— 预计算的**乘法**因子：最新交易日恒为
    `1.0`，越早越小。实测 8 只股票 × 全历史 **40,192 行**与 QUANTAXIS 的
    `to_qfq()` 逐值一致（最大差 0），验证脚本见 commit。

    三条与 QUANTAXIS 的**刻意差异**：

    1. **按标的分别 ffill。** QUANTAXIS 在 `(date, code)` 索引上直接
       `ffill()`，跨标的也会前向填充 —— 多标的帧里某只票缺一行因子，会拿
       **上一只票**的系数。那显然是错的，本函数用 `groupby('code').ffill()`。
    2. **ffill 之后仍为 NaN 的补 `1.0`。** 完全没有因子的标的（例如 `stock_adj`
       尚未覆盖的新票）在 QUANTAXIS 那边会 `close * NaN` 把整列抹成 NaN。
       老树 `GQ_apply_etf_qfq` 的契约就是「查不到的按 1.0 处理，绝不给 NaN」，
       这里沿用同一条。注意这**只影响 QUANTAXIS 本来就产出 NaN 的情形**，
       不是数值口径的改动。
    3. **`volume` 不动。** 与 QUANTAXIS 一致：只乘 `open/high/low/close`。
    """
    if data is None or len(data) == 0:
        return data

    dates = _row_dates(data)
    codes = _row_codes(data)
    if dates is None or codes is None:
        if verbose:
            print('GQ Warning: 无法从帧中判定 (date, code)，跳过复权')
        return data

    dmin, dmax = dates.min(), dates.max()
    adj = _adj_frame(sorted(set(codes)), dmin, dmax)
    if adj.empty:
        if verbose:
            print(f'GQ Warning: stock_adj 在 [{dmin}, {dmax}] 无因子，按 1.0 处理')
        out = data.copy()
        out['adj'] = 1.0
        return out

    key = pd.DataFrame({'date': dates.values, 'code': codes.values})
    merged = key.merge(adj, on=['date', 'code'], how='left')
    if len(merged) != len(data):
        # (date, code) 在 stock_adj 里应当唯一；重复会让下面直接错位。
        raise ValueError(
            f'stock_adj 出现重复的 (date, code)：期望 {len(data)} 行，得到 {len(merged)} 行')

    factor = merged.groupby('code', sort=False)['adj'].ffill()
    out = data.copy()
    out['adj'] = factor.fillna(1.0).values
    for col in ('open', 'high', 'low', 'close'):
        if col in out.columns:
            out[col] = out[col] * out['adj']
    return out


class GQ_DataStruct:
    """K 线容器。`data` 是以 `(时间, code)` 为 MultiIndex 的 DataFrame。

    刻意不做 `__getattr__` 转发：QUANTAXIS 的 datastruct 会把未知属性丢给
    `self.data`，于是**拼错的属性名会静默变成 DataFrame 的列查找**。本树刚在
    `_StubMeta` 上吃过这个亏（见 `PITFALLS.md` P1），不再复制。
    """

    type = None
    if_fq = 'bfq'

    def __init__(self, data=None, type=None, if_fq='bfq'):
        self.data = data if data is not None else pd.DataFrame()
        if type is not None:
            self.type = type
        self.if_fq = if_fq

    def new(self, data, type=None, if_fq='bfq'):
        # 必须用 `self.__class__` 而不是 `type(self)`：子类把 `.type` 定义成
        # **类属性**（QUANTAXIS 的对外接口），它在类体里遮蔽了内建 `type`，
        # `type(self)` 会变成「用字符串调用」并抛 TypeError。
        return self.__class__(
            data, type if type is not None else self.type, if_fq)

    def __len__(self):
        return len(self.data)

    @property
    def index(self):
        """透出 `data.index`。

        `realtime.py:877` 用 `select_code(...).index.get_level_values(level=0)`
        取时间轴，所以这个属性是**被依赖的接口**，不是便利方法。
        """
        return self.data.index

    @property
    def code(self):
        return _row_codes(self.data)

    @property
    def date(self):
        dates = _row_dates(self.data)
        if dates is None:
            return pd.DatetimeIndex([])
        return pd.DatetimeIndex(pd.to_datetime(dates))

    def select_code(self, code):
        """取单只标的，返回同类型对象；标的不存在抛 `ValueError`。

        与 QUANTAXIS 一致抛 `ValueError`（它内部捕 `KeyError` 后改写），而不是
        返回空帧 —— 空帧会被调用方的 `len(...) < 30` 判成「数据不足」，
        把「代码写错了」伪装成「数据少」。
        """
        data = self.data
        idx = data.index
        if isinstance(idx, pd.MultiIndex) and 'code' in (idx.names or []):
            try:
                sel = data.loc[(slice(None), code), :]
            except KeyError:
                raise ValueError(f'GQ CANNOT FIND THIS CODE {code}')
        else:
            sel = data
        return self.new(sel, self.type, self.if_fq)

    def __repr__(self):
        return (f'<{self.__class__.__name__} type={self.type} '
                f'if_fq={self.if_fq} rows={len(self)}>')


class _QfqMixin:
    """`to_qfq()` —— 只给 Stock 类混入（见模块 docstring 第 ② 条）。"""

    def to_qfq(self, verbose=False):
        """前复权。**幂等**：已经复权过的对象原样返回。

        与 QUANTAXIS 的 `if self.if_fq == 'bfq'` 分支同构 —— 它的 `else` 分支只
        记一行日志然后 `return self`，所以重复调用不会二次乘因子。
        """
        if self.if_fq != 'bfq':
            if verbose:
                print(f'GQ Warning: if_fq={self.if_fq}，不重复复权')
            return self
        if len(self.data) == 0:
            return self
        return self.new(apply_qfq(self.data, verbose=verbose), self.type, 'qfq')


class GQ_DataStruct_Stock_day(_QfqMixin, GQ_DataStruct):
    """个股日线。"""
    type = 'stock_day'


class GQ_DataStruct_Stock_min(_QfqMixin, GQ_DataStruct):
    """个股分钟线。"""
    type = 'stock_min'


class GQ_DataStruct_Index_day(GQ_DataStruct):
    """指数 / ETF 日线。**没有 `to_qfq`** —— 与 QUANTAXIS 一致。"""
    type = 'index_day'


class GQ_DataStruct_Index_min(GQ_DataStruct):
    """指数 / ETF 分钟线。**没有 `to_qfq`** —— 与 QUANTAXIS 一致。"""
    type = 'index_min'


class GQ_DataStruct_Stock_block:
    """板块成分容器 —— QUANTAXIS `QA_DataStruct_Stock_block` 的替身。

    ⚠️ 那个类**不在** `QUANTAXIS/QAData/QADataStruct.py` 里，而在
    `QAData/QABlockStruct.py` —— 按名字在 QADataStruct 里找会找不到。
    索引是 `(blockname, code)` 两层，因为板块的唯一键就是这两者的组合。

    消费方（`fetch.py` 的板块取数）只用三样：`.block_name` /
    `.get_block(names).code` / `.get_blocklist(names).code`。

    **两处刻意分歧**：

    1. 板块名找不到时**抛 `ValueError`** 并列出缺的名字。QUANTAXIS 的
       `.loc[(names, slice(None))]` 会抛裸 `KeyError`；两者都报错，本实现把
       「哪几个名字不存在」写进消息里 —— 静默返回空帧才是真危险。
    2. `get_blocklist` 是 `get_block` 的**别名**。QUANTAXIS **从来没有**这个
       方法（全包 grep 为 0），唯一调用点 `fetch.py:480` 在死函数
       `prepare_symbol_range` 里，从未执行过 —— 也就是说那行如果跑了，
       原本会 `AttributeError`。这里给个别名让本地化的代码自洽，
       **不改变任何可达路径的行为**。
    """

    def __init__(self, data=None):
        self.data = data if data is not None else pd.DataFrame()

    def new(self, data):
        return GQ_DataStruct_Stock_block(data)

    def __len__(self):
        return len(self.data)

    @property
    def block_name(self):
        """全部板块名，**排序后**返回（QUANTAXIS 的 `index.levels[0]` 本就是有序的）。"""
        if len(self.data) == 0:
            return []
        return sorted(self.data.index.get_level_values(0).unique().tolist())

    @property
    def code(self):
        """成分代码去重后排序返回。"""
        if len(self.data) == 0:
            return []
        return sorted(self.data.index.get_level_values(1).unique().tolist())

    def get_block(self, blockname):
        """取若干板块的全部成分。`blockname` 是单个名字或名字的可迭代对象。"""
        names = [blockname] if isinstance(blockname, str) else list(blockname)
        idx = self.data.index
        present = set(idx.get_level_values(0)) if len(self.data) else set()
        missing = [n for n in names if n not in present]
        if missing:
            raise ValueError(f'GQ 板块不存在: {missing}')
        sel = self.data[idx.get_level_values(0).isin(names)]
        return self.new(sel)

    def get_blocklist(self, blockname):
        """`get_block` 的别名 —— QUANTAXIS 无此方法，见类 docstring 分歧 ②。"""
        return self.get_block(blockname)

    def __repr__(self):
        return f'<GQ_DataStruct_Stock_block blocks={len(self.block_name)} rows={len(self)}>'


def frame_to_datastruct(df, market='stock', frequency='day'):
    """8.3 读取器产出的扁平表 → 对应的容器类型。

    `df` 需带 `date`（字符串）与 `code` 列；本函数负责选出 QUANTAXIS 口径的
    6 列并建 `(date, code)` MultiIndex。空表返回 `None`（日线调用方依赖
    `is None` 分支，见 `markets/base_market.py` 的空值契约说明）。
    """
    if df is None or len(df) == 0:
        return None
    keep = [c for c in _OHLCV_COLUMNS if c in df.columns]
    if not keep:
        raise ValueError(f'帧里没有任何 OHLCV 列：{list(df.columns)}')
    out = df.copy()
    out['date'] = pd.to_datetime(out['date'])
    out = out.set_index(['date', 'code']).sort_index()[keep]
    cls = {
        ('stock', 'day'): GQ_DataStruct_Stock_day,
        ('stock', 'min'): GQ_DataStruct_Stock_min,
        ('index', 'day'): GQ_DataStruct_Index_day,
        ('index', 'min'): GQ_DataStruct_Index_min,
    }[(market, frequency)]
    return cls(out)
