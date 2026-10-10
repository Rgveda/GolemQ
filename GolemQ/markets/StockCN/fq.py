# coding:utf-8
"""复权的**纯函数**核心 —— 股票与 ETF 共用。

为什么有这一层
==============
股票与 ETF 的复权原本是两份平行实现（`datastruct.apply_qfq` 与
`etf_fq.GQ_apply_etf_qfq`），连 `_row_dates` / `_row_codes` / `_adj_collection`
都各有一份。但两者的**差异全在策略，不在机制**：

============ ========================== ================================
            股票                         ETF
============ ========================== ================================
因子表       ``stock_adj``（密集）        ``etf_adj``（**稀疏**，294/1674）
缺失处理     先按标的 ffill，再填 1.0      **只填 1.0，绝不 ffill**
============ ========================== ================================

机制只有一句话：**按 `(code, date)` 把因子对到行上，乘 OHLC**。所以这里把它
抽成两个纯函数 —— :func:`align_factors`（对齐 + 缺失策略）与
:func:`multiply_ohlc`（乘法）—— 两处调用方各自传入自己的策略。

⚠️ **缺失策略是参数，不是实现细节，不能「统一」掉。**
``etf_adj`` 只落有除权事件的 code，若改成 ffill，会把最后一次事件前的系数一路
拖到最新日期而**静默算错价**；而股票的 ``stock_adj`` 按标的密集、个别日期可能
缺行，**需要** ffill。两种行为都是承重契约，见 `etf_fq.py` 的模块说明。

本模块**不碰数据库**（因子的取数留在两个来源模块里），所以是纯函数、带 doctest。
"""

from typing import Optional

import pandas as pd

__all__ = [
    'index_field',
    'row_dates',
    'row_codes',
    'flatten_factor_map',
    'factor_dict_from_frame',
    'align_factors',
    'multiply_ohlc',
]

#: 参与复权的价格列（成交量/成交额不复权）
PRICE_COLUMNS = ('open', 'high', 'low', 'close')

#: `align_factors` 里拼接复合键的分隔符。**不能用 `|` 以外的字符去「美化」** ——
#: 它只要求不与 6 位代码/`YYYY-MM-DD` 冲突，而 `|` 两者都不含。
_KEY_SEP = '|'


def index_field(data: pd.DataFrame, name: str) -> Optional[pd.Series]:
    """从 (Multi)Index 里取某个 level 作为 Series；没有则返回 ``None``。

    >>> import pandas as pd
    >>> idx = pd.MultiIndex.from_tuples(
    ...     [(pd.Timestamp('2024-01-02'), '600519')], names=['date', 'code'])
    >>> df = pd.DataFrame({'close': [10.0]}, index=idx)
    >>> list(index_field(df, 'code'))
    ['600519']
    >>> index_field(df, 'nope') is None
    True
    """
    idx = data.index
    if not isinstance(idx, pd.MultiIndex):
        return None
    try:
        names = list(idx.names or [])
    except Exception:  # noqa: BLE001
        return None
    if name not in names:
        return None
    return pd.Series(idx.get_level_values(name), index=data.index)


def row_dates(data: pd.DataFrame) -> Optional[pd.Series]:
    """每行的所属日期（``'YYYY-MM-DD'`` 字符串），找不到返回 ``None``。

    日线的 index level 0 就是日期；分钟线是 ``datetime``，取 ``'%Y-%m-%d'``
    —— 复权系数逐日恒定，日内所有 bar 同系数。索引里没有 datetime 层时退而看
    ``date`` 列。**顺序不能反**：索引是权威，列可能是迁移残留。

    >>> import pandas as pd
    >>> idx = pd.MultiIndex.from_tuples(
    ...     [(pd.Timestamp('2024-01-02 10:30'), '600519')], names=['ts', 'code'])
    >>> df = pd.DataFrame({'close': [10.0]}, index=idx)
    >>> list(row_dates(df))
    ['2024-01-02']
    """
    idx = data.index
    nlevels = idx.nlevels if isinstance(idx, pd.MultiIndex) else 1
    for lv in range(nlevels):
        vals = idx.get_level_values(lv)
        if pd.api.types.is_datetime64_any_dtype(vals):
            return pd.Series(
                pd.DatetimeIndex(vals).strftime('%Y-%m-%d'), index=data.index)
    for name in ('date', 'datetime'):
        if name in data.columns:
            return data[name].astype(str).str[:10]
    return None


def row_codes(data: pd.DataFrame) -> Optional[pd.Series]:
    """每行的 6 位代码；索引的 ``code`` 层优先，其次 ``code`` 列，都没有则 ``None``。

    >>> import pandas as pd
    >>> idx = pd.MultiIndex.from_tuples([('2024-01-02', '600519.XSHG')], names=['date', 'code'])
    >>> df = pd.DataFrame({'close': [10.0]}, index=idx)
    >>> list(row_codes(df))
    ['600519']
    """
    codes = index_field(data, 'code')
    if codes is not None:
        return codes.astype(str).str[:6]
    if 'code' in data.columns:
        return data['code'].astype(str).str[:6]
    return None


def flatten_factor_map(factor_map) -> dict:
    """``{code: {date: adj}}`` → ``{code|date: adj}``，供 :func:`align_factors`。

    拼字符串是刻意的：`Series.map` 走字典 C 查找，避免逐行 python 循环。

    >>> flatten_factor_map({'600519': {'2024-01-02': 0.9}})
    {'600519|2024-01-02': 0.9}
    """
    return {f'{c}{_KEY_SEP}{d}': v
            for c, dd in (factor_map or {}).items() for d, v in dd.items()}


def factor_dict_from_frame(df: pd.DataFrame, code_col: str = 'code',
                           date_col: str = 'date',
                           value_col: str = 'adj') -> dict:
    """因子 DataFrame → ``{code|date: adj}``。

    **重复的 `(code, date)` 直接抛错**：重复键会让对齐静默错位 —— 老实现用
    `merge` 的 `len` 比对来拦这件事，换成字典会把这个保护丢掉，所以显式检查。

    >>> import pandas as pd
    >>> factor_dict_from_frame(pd.DataFrame(
    ...     {'code': ['600519'], 'date': ['2024-01-02'], 'adj': [0.9]}))
    {'600519|2024-01-02': 0.9}
    """
    if df is None or len(df) == 0:
        return {}
    keys = df[code_col].astype(str).str[:6] + _KEY_SEP + df[date_col].astype(str).str[:10]
    if keys.duplicated().any():
        dup = sorted(set(keys[keys.duplicated()]))[:5]
        raise ValueError(
            f'因子表出现重复的 (code, date)：{dup} —— 对齐会静默错位，拒绝继续')
    return dict(zip(keys, df[value_col].astype(float)))


def align_factors(data: pd.DataFrame, flat: dict, *,
                  ffill_by_code: bool, fill: float = 1.0,
                  dates=None, codes=None) -> pd.Series:
    """把因子对齐到 `data` 的每一行，返回与 ``data.index`` 同索引的 Series。

    :param flat: ``{code|date: adj}``（见 :func:`flatten_factor_map`）
    :param ffill_by_code: **缺失策略**，两边不同、且都承重（见模块 docstring）。
        ``True``（股票）：先**按标的**前向填充 —— 按标的而非整帧，是因为
        QUANTAXIS 在 `(date, code)` 索引上直接 ffill，多标的帧里某只票缺一行会
        拿**上一只票**的系数，那是错的。``False``（ETF）：**绝不 ffill**。
    :param fill: ffill 之后仍缺的填这个值（默认 1.0 = no-op）。绝不给 NaN ——
        那会把整列价格抹掉。
    :param dates, codes: 预设的行级日期/代码 Series。**单标的帧**（索引里没有
        `code` 层、也没有 `code` 列）推不出代码，此时由调用方传入 —— ETF 侧就是
        这种情形，它从 `codelist` 参数知道代码。

    >>> import pandas as pd
    >>> idx = pd.MultiIndex.from_tuples(
    ...     [(pd.Timestamp('2024-01-02'), '600519'),
    ...      (pd.Timestamp('2024-01-03'), '600519')], names=['date', 'code'])
    >>> df = pd.DataFrame({'close': [10.0, 11.0]}, index=idx)
    >>> list(align_factors(df, {'600519|2024-01-02': 0.5}, ffill_by_code=True))
    [0.5, 0.5]
    >>> list(align_factors(df, {'600519|2024-01-02': 0.5}, ffill_by_code=False))
    [0.5, 1.0]
    """
    if dates is None:
        dates = row_dates(data)
    if codes is None:
        codes = row_codes(data)
    if dates is None or codes is None:
        raise ValueError('无法从帧中判定 (date, code)，不能复权')

    key = codes.astype(str) + _KEY_SEP + dates.astype(str)
    factor = key.map(flat)
    if ffill_by_code:
        factor = pd.Series(factor.to_numpy(), index=data.index).groupby(
            codes.to_numpy(), sort=False).ffill()
    return factor.fillna(fill)


def multiply_ohlc(data: pd.DataFrame, factor, *,
                  keep_factor_column: bool = True,
                  factor_name: str = 'adj') -> pd.DataFrame:
    """把 OHLC 乘上因子，返回**新帧**（不改入参）。

    **只乘 OHLC**，`volume`/`vol`/`amount` 不动 —— 与 QUANTAXIS 股票 `to_qfq()`
    的口径一致，也是老树 `GQ_apply_etf_qfq` 的契约。

    :param keep_factor_column: 股票侧保留 `adj` 列（QUANTAXIS 的 `to_qfq()` 会
        留下它），ETF 侧不留（老树的 `GQ_apply_etf_qfq` 没有这一列）——
        两边的可观测行为已按各自基准验证过，故保留为参数而不是「统一」。

    >>> import pandas as pd
    >>> df = pd.DataFrame({'open': [1.0], 'close': [2.0], 'volume': [100.0]})
    >>> multiply_ohlc(df, pd.Series([0.5]), keep_factor_column=False).to_dict('list')
    {'open': [0.5], 'close': [1.0], 'volume': [100.0]}
    """
    factor = pd.Series(factor)
    out = data.copy()
    for col in PRICE_COLUMNS:
        if col in out.columns:
            out[col] = out[col].astype('float64') * factor.to_numpy()
    if keep_factor_column:
        out[factor_name] = factor.to_numpy()
    return out


#: `stock_xdxr` 里参与前复权计算的事件字段（`category == 1` 即除权除息）。
XDXR_FIELDS = ('fenhong', 'peigu', 'peigujia', 'songzhuangu')


def xdxr_to_adj(dates, close, xdxr):
    """日线 + 除权除息事件 → **前复权因子**（`stock_adj` 的 `adj` 列）。

    公式逐字照抄 QUANTAXIS `QAData/data_fq.py::_QA_data_stock_to_fq` 的 **qfq 分支**：

        preclose = (close.shift(1) * 10 - fenhong + peigu * peigujia) / (10 + peigu + songzhuangu)
        adj      = (preclose.shift(-1) / close).fillna(1)[::-1].cumprod()

    ⚠️ **一个必须照做的对齐细节**：事件日期**不一定落在日线里**（实测 600519 的
    2006-05-19 除权日，日线整段 2006-05-18~05-24 是空的）。这种事件要**挪到
    下一个存在的交易日**再去算 —— 直接 `join` 会把它丢掉，结果就是**整段历史
    差一个因子**（实测 600519 差 2 倍、600601 差 10 倍）。

    ⚠️ **与 QUANTAXIS 的一处刻意偏离：无事日的比率强制为精确 1.0。**
    公式里 `preclose.shift(-1) / close` 在**无除权日**数学上**恒为 1**（那种日子
    `preclose == close.shift(1) * 10 / 10 == close.shift(1)`），但 `x * 10 / 10`
    在 IEEE754 下**可能差 1 ulp**，于是比率 ≈1 却非 1 —— **`cumprod` 把这点残差
    一路连乘累积**。实测 `000711` 的无事段 1590 行漂到 `1 ± 2.9e-15`，
    且**相邻日期的比特各不相同**。

    后果不是"精度差一点"：因子因此**不再是分段常数**，会把原始行情里
    **本来严格相等的价格撬开**（低价股同日 high 重复取值极多），czsc 分型的
    **平局判定**随之翻转 —— 实测同一份 K 线、笔 59→61、中枢 4→6、
    走势 盘整→上涨趋势（`PITFALLS.md` P30）。

    故本函数把无事日的比率**显式置为精确 1.0**。这是**恒等变换**
    （那些日子的真实比率就是 1），只是抹掉浮点残差。改后因子恢复
    **分段常数**、末日**精确 1.0**；改前末日是 `1.0000000000000029` ——
    一个前复权因子**大于 1** 本身就是警号。

    ⚠️ **事件日的比率不动** —— 那是一次**真实除法**（如 `close*10/20`），
    1 ulp 是这类计算的固有属性（10 送 10 会得到 `0.49999999999999994`），
    **别去"修"它**：事件日只算一次，其后的 `cumprod` 乘的全是精确 1.0，
    **分段常数**照样成立。要抹平它反而会引入一个新错误。
    （两类比率的区分及其理由，见 `PITFALLS.md` P30 的末节。）

    实证（2026-10-10，全市场 400 只抽样）：改前 313 只有"≈1.0 噪声行"、
    改后与 4.4 存量（QUANTAXIS 生成）在**无事段逐值相同**。

    :param dates: 交易日 `'YYYY-MM-DD'` 列表（**升序**，与 `close` 等长）
    :param close: 对应的**不复权**收盘价
    :param xdxr: 该 code 的 `stock_xdxr` 文档列表（只用 `category == 1` 的行）
    :returns: `pd.Series`，索引 = `dates`（升序），值 = 因子；**最新一根恒为 1.0**

    >>> s = xdxr_to_adj(['2024-01-01', '2024-01-02', '2024-01-03'], [10.0, 11.0, 12.0], [])
    >>> [round(float(x), 6) for x in s]
    [1.0, 1.0, 1.0]
    >>> ev = [{'date': '2024-01-03', 'category': 1, 'fenhong': 0.0, 'peigu': 0.0,
    ...        'peigujia': 0.0, 'songzhuangu': 10.0}]
    >>> [round(float(x), 6) for x in xdxr_to_adj(
    ...     ['2024-01-01', '2024-01-02', '2024-01-03'], [10.0, 11.0, 12.0], ev)]
    [0.5, 0.5, 1.0]

    **扩缩股**（``category == 11``，ETF 的份额折算）走**另一条乘性口径**：
    ``参考价 = 前收盘 / suogu``。实测 ``suogu`` 就是除权日价格跳变的比值本身
    （159901 2010-11-22 ``suogu=5.0``：前收 3.966 → 当日开 0.797，比值 4.976）：

    >>> ev = [{'date': '2024-01-03', 'category': 11, 'suogu': 5.0}]
    >>> [round(float(x), 6) for x in xdxr_to_adj(
    ...     ['2024-01-01', '2024-01-02', '2024-01-03'], [10.0, 10.0, 2.0], ev)]
    [0.2, 0.2, 1.0]

    **无事段的因子必须是"精确"1.0，不是 ≈1.0** —— 这条钉的正是上面那处偏离。
    改成 `>= 1e-15` 之类的容差写就抓不住它了（改前这里漂到 2.9e-15）：

    >>> days = ['2024-%02d-%02d' % (m, d) for m in range(1, 13) for d in range(1, 29)]
    >>> closes = [10.0 + i * 0.37 for i in range(len(days))]     # 336 个交易日，零事件
    >>> max(abs(float(x) - 1.0) for x in xdxr_to_adj(days, closes, []))
    0.0
    """
    idx = [str(d) for d in dates]
    df = pd.DataFrame({'close': list(close)}, index=pd.Index(idx, name='date'))

    slots: dict = {}
    event_days: set = set()          # 真正改变了 `preclose` 的日子（两类事件合并）
    for r in (xdxr or []):
        if int(r.get('category') or 0) != 1:
            continue
        pos = [i for i, d in enumerate(idx) if d >= str(r.get('date'))]
        if not pos:
            continue            # 事件晚于最后一根 bar：不影响任何已有行
        event_days.add(idx[pos[0]])
        slot = slots.setdefault(idx[pos[0]], {k: 0.0 for k in XDXR_FIELDS})
        for k in XDXR_FIELDS:
            slot[k] += r.get(k) or 0.0

    ev = (pd.DataFrame(slots).T.reindex(columns=list(XDXR_FIELDS)) if slots
          else pd.DataFrame(columns=list(XDXR_FIELDS), index=pd.Index([], name='date')))
    data = df.join(ev, how='left').astype(float).fillna(0.0)
    data['preclose'] = (
        data['close'].shift(1) * 10 - data['fenhong']
        + data['peigu'] * data['peigujia']
    ) / (10 + data['peigu'] + data['songzhuangu'])

    # 扩缩股（ETF 份额折算）：**乘性**，与上面那条加性公式不同形态，故单独盖掉
    # 该交易日的 `preclose`（其余日子 preclose == 前收盘，乘 1 无影响）。
    # 事件日同样要**挪到下一个存在的交易日**（与上面同一套对齐理由）。
    for r in (xdxr or []):
        if int(r.get('category') or 0) != 11:
            continue
        suogu = r.get('suogu')
        if not suogu:
            continue                      # 缺 suogu 的扩缩股无从计算，宁可不动
        pos = [i for i, d in enumerate(idx) if d >= str(r.get('date'))]
        if not pos:
            continue
        day = idx[pos[0]]
        prev = data['close'].shift(1).loc[day]
        if prev == prev and prev > 0:     # 非 NaN
            data.loc[day, 'preclose'] = float(prev) / float(suogu)
            event_days.add(day)           # 只有真改了 preclose 才算事件日

    # `ratio[t] = preclose[t+1] / close[t]`：**下一根不是事件日**时它数学上恒为 1
    # （`preclose[t+1] == close[t] * 10 / 10`），这里把浮点残差抹掉 —— 见 docstring
    # 「与 QUANTAXIS 的一处刻意偏离」。不这么做，`cumprod` 会把 1 ulp 的残差连乘
    # 放大成 1e-15 量级的**逐日不同**的噪声，进而撬开价格的相等关系。
    ratio = data['preclose'].shift(-1) / data['close']
    next_is_event = pd.Series(data.index, index=data.index).shift(-1).isin(event_days)
    ratio[~next_is_event.values] = 1.0

    data['adj'] = ratio.fillna(1)[::-1].cumprod()
    return data['adj']
