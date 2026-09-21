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
