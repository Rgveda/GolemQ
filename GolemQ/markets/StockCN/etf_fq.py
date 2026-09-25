# coding:utf-8
#
# The MIT License (MIT)
#
# Copyright (c) 2020-2025 azai/GolemQ(uant)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#

"""ETF 复权（除权）读取侧支持。

**本模块是回迁**：老树 `GolemQ_old/markets/StockCN/etf_fq.py`（260 行）在重构中
**整个没迁过来**，新树只在 `fetch.py` 留了一段「检测到大跳空就 print 一句
『ETF 需要人工复权』」。后果记在 `MIGRATION_STATUS.md` 的 HIGH #9：拉 ETF
（如 510300）日线跨除权日时，老代码给**连续的前复权 OHLC**，新代码给**不复权
数据**，约 10% 的假跳空会污染 ETF 回测。逻辑，而不是格式，丢了。

为什么自建这张表
====================================================================
ETF 与真指数**共用** `index_day`/`index_min` 集合，而 QUANTAXIS 不提供指数/ETF
复权 —— `QA_DataStruct_Index_day` 没有 `to_qfq()`，`stock_xdxr`/`stock_adj` 与
`_QA_fetch_stock_adj` 也只服务股票。所以 `etf_adj` 由
`GolemQ.gateway.xtquant.save_qa.GQ_SU_save_etf_xdxr_qmt` 写入：

``etf_xdxr``
    除权除息事件。字段与 ``stock_xdxr`` 同形（``fenhong``/``peigu``/``peigujia``/
    ``songzhuangu``，均按「每 10 份」），**另存 QMT 的 ``dr``（除权因子）与 xtdata 原值**。
    唯一索引 ``(code, date)``。

``etf_adj``
    逐日**等比前复权**系数，由 ``dr`` 推导 —— ``adj(d) = Π(1/dr)`` 对所有权除权日
    > d 的事件。实测精确对齐 QMT ``dividend_type='front_ratio'``（510300/510050/
    510880/159941/000858 误差 0）。唯一索引 ``(code, date)``。

稀疏约定（与读法的硬约束）
====================================================================
``etf_adj`` **只落有除权事件的 code**；无事件的 code 一行不落，其系数由读取侧按
**1.0** 处理，等于 no-op。2026-09-21 实测：397,499 行 / **294** 只 code
（老树注释记的是 293，多出来的一只未追查）。

对有事件的 code，``etf_adj`` 覆盖其**完整**交易日序列 —— 这是刻意的：若有人把这里
的合并改成 QUANTAXIS 那套 ``join + ffill``，缺 code / 缺日期会得到 **NaN（看得见的
失败）**；而如果只落 ``adj≠1`` 的行，``ffill`` 会把最后一次事件前的系数一路拖到最新
日期，**静默算错价**。所以：**本模块一律用 ``fillna(1.0)``，不要用 ffill。**

**只乘内存里返回的 OHLC，绝不改写库中的不复权行情。** 真指数在 ``etf_adj`` 里没有行，
``GQ_apply_etf_qfq`` 也不会为它们查库，因此指数行为与改动前逐行一致。
"""

from typing import Dict

import pandas as pd

from GolemQ.core.constants import MARKET_TYPE

from .fq import (
    align_factors,
    flatten_factor_map,
    multiply_ohlc,
    row_codes as _row_codes,
    row_dates as _row_dates,
)
from .symbol import is_stock_cn

# 参与复权的价格列已随乘法收敛到 `fq.PRICE_COLUMNS`（本模块不再自己乘）。


def _adj_collection():
    """``etf_adj`` 集合句柄。**函数级导入是刻意的。**

    ``markets/StockCN/__init__.py:55`` 经 ``from .quotes import StockCNQuotes``
    间接走到本模块，而 ``DATABASE_STOCK_CN`` 到 ``:70`` 才定义 —— 顶层
    ``from . import DATABASE_STOCK_CN`` 会抛
    ``ImportError: cannot import name ... from partially initialized module``。
    ``datastruct.py`` / ``kline83.py`` / ``refdata.py`` 出于同一原因也这样写。

    ``etf_adj`` 在 ``quantaxis``(4.4) 与 ``golemq_stock_cn``(8.3) 里**逐行相同**
    （2026-09-21 实测：两边都是 397,499 行 / 294 只 code），故取 8.3 ——
    与 D5「参考集合的家是 golemq_stock_cn」一致。
    """
    from . import DATABASE_STOCK_CN
    return DATABASE_STOCK_CN['etf_adj']


def GQ_is_etf(code) -> bool:
    """判断 6 位代码是否为场内 ETF。

    判据是 :func:`is_stock_cn` 返回的 **``MARKET_TYPE.ETF_CN``** —— 单一真值来源。

    ⚠️ **原实现嗅探描述串**（``is_stock_cn(code)[3].endswith('ETF基金')``）。
    那是在 ETF 被归进 ``INDEX_CN``、与真指数同类型时期的**唯一**区分手段，
    但极脆：描述串是展示文本，任何人改一下措辞（或改成更规范的说法），
    `GQ_is_etf` 就静默返回 ``False`` → **ETF 复权全线停摆，且不报任何错**。
    现在类型是权威的，描述串回归纯展示。

    :param code: 6 位代码，如 ``'510300'``；带交易所标记也容忍
        （``'510300.XSHG'`` / ``'sh.510300'``，由 ``is_stock_cn`` 归一）。
    :return: ``True`` 为 ETF；任何异常或未知代码返回 ``False``（保守：不当作 ETF）。
    """
    try:
        return is_stock_cn(str(code))[1] == MARKET_TYPE.ETF_CN
    except Exception:  # noqa: BLE001
        return False


def GQ_fetch_etf_adj(codelist, start=None, end=None,
                     adj_collection=None) -> Dict[str, Dict[str, float]]:
    """读取逐日复权系数，返回 ``{code: {date: adj}}``。

    单次区间查询，仿 QUANTAXIS ``_QA_fetch_stock_adj``。``date`` 是 ``'YYYY-MM-DD'``
    字符串（字典序即可比大小）—— 注意 ``etf_adj`` 里 ``date`` **就是字符串**，
    范围查询必须传字符串，传 ``datetime`` 会静默返回空集（见 ``PITFALLS.md``）。

    **缺失即 1.0**：返回的字典里没有的 code、或 code 里没有的日期，语义都是
    ``adj = 1.0``（no-op）—— 见模块 docstring 的「稀疏约定」。调用方必须用
    ``fillna(1.0)`` 兜底，**不要 ffill**。

    :param codelist: 6 位 code（``str``）或其列表；超过 6 位按前 6 位截断。
    :param start: 起始日期 ``'YYYY-MM-DD'``（含）；``None`` = 不设下界。
    :param end: 结束日期 ``'YYYY-MM-DD'``（含）；``None`` = 不设上界。
    :param adj_collection: 覆盖默认集合（测试/迁移用）。
    :return: ``{code: {date: adj}}``；无数据时返回 ``{}``。
    """
    codes = [codelist] if isinstance(codelist, str) else list(codelist or [])
    codes = sorted({str(c)[:6] for c in codes})
    if not codes:
        return {}
    if adj_collection is None:
        adj_collection = _adj_collection()
    query = {'code': {'$in': codes}}
    rng = {}
    if start is not None:
        rng['$gte'] = str(start)[:10]
    if end is not None:
        rng['$lte'] = str(end)[:10]
    if rng:
        query['date'] = rng
    out: Dict[str, Dict[str, float]] = {}
    for doc in adj_collection.find(
            query, {'_id': 0, 'code': 1, 'date': 1, 'adj': 1}, batch_size=10000):
        try:
            out.setdefault(str(doc['code'])[:6], {})[str(doc['date'])[:10]] = float(doc['adj'])
        except (KeyError, TypeError, ValueError):
            continue
    return out


def GQ_apply_etf_qfq(data_day, codelist=None, verbose: bool = False):
    """把 ETF 的 OHLC 换成前复权价，并**原样返回同一个对象**。

    （实现上由「就地改 `data` 的列」改为「换一个新帧并挂回 `data_day.data`」
    —— 对外可观测行为不变：同一个 `data_day` 出去，`.data` 是复权后的帧。
    复权那几步已收敛到 `fq.py` 的共用纯函数，见其模块说明。）

    行为契约：

    - **只处理 ETF**。真指数、股票直接原样返回，且**一次 Mongo 都不查**。
    - **只乘 OHLC**（``open``/``high``/``low``/``close``）；``volume``/``vol``/
      ``amount`` 不动 —— 与 QUANTAXIS 股票 ``to_qfq()`` 的口径一致。
    - 系数表里查不到的日期按 **1.0** 处理（``fillna``），绝不给 NaN，避免把最新一根
      K 线抹掉。这是 ``etf_adj`` 稀疏约定成立的前提。
    - 任何异常（集合不存在、表为空、字段缺失等）都退化为「不复权」并打印一行告警，
      行为与改动前完全一致。
    - 幂等：对同一个 frame 调两次，第二次原样返回、不重复乘因子。
    - 成功后置 ``data_day.if_fq = 'qfq'``，并在 ``data_day.etf_qfq_bars`` 上写下
      被调整的 K 线根数（0 表示本区间一根都没被调整，供调用方判断「复权是否真的发生」）。

    ⚠️ **幂等是靠下面那个 ``if_fq`` 早退实现的，不是天然的。** 老树那份 docstring
    也写了「幂等」，但**从未实现** —— 它同样置 ``if_fq = 'qfq'`` 却从不读它。
    2026-09-21 实测老逻辑连调两次会把因子乘两遍（510300：第二次/第一次 = 0.931873，
    正是它的除权因子）。本模块补上早退，使契约成真；**改动前请先想清楚有没有路径
    会依赖「调两次会乘两遍」**（没有 —— 没有任何调用方这么做）。

    调用位置很关键：必须在实时行情覆盖（``GQ_fetch_stock_day_realtime_adv``）**之前**。
    最新一根的系数按构造恒为 1，所以覆盖上去的实时（不复权）价不会被错误缩放。

    :param data_day: ``GQ_DataStruct_Index_day`` / ``GQ_DataStruct_Index_min`` /
        ``KlineResult`` 等，需要有 ``.data``（DataFrame，index 为
        ``(date|datetime, code)`` MultiIndex 或单标的 DatetimeIndex）。
    :param codelist: 6 位 code 或列表；``None`` 时从 index 的 ``code`` level/列推断。
    :param verbose: 打印一行复权摘要（调整根数）。
    :return: 传入的 ``data_day``（原地修改）；失败时原样返回。
    """
    try:
        # 幂等早退。老树缺这一步（见 docstring 的 ⚠️），重复调用会把因子乘两遍。
        if getattr(data_day, 'if_fq', 'bfq') == 'qfq':
            return data_day

        data = getattr(data_day, 'data', None)
        if data is None or len(data) == 0:
            return data_day

        if codelist is None:
            codes = None
        elif isinstance(codelist, str):
            codes = [codelist[:6]]
        else:
            codes = [str(c)[:6] for c in codelist]
        codes_row = _row_codes(data)
        if codes is None:
            if codes_row is None:
                return data_day
            codes = sorted(set(codes_row))
        codes = [c for c in codes if GQ_is_etf(c)]
        if not codes:
            return data_day

        # 供调用方判断「复权到底有没有发生」：0 表示该区间 ETF 一根都没被调整
        # （没跑过 etf_xdxr，或区间内本无除权事件）。非 ETF 不会被标记。
        try:
            data_day.etf_qfq_bars = 0
        except Exception:  # noqa: BLE001
            pass

        dates = _row_dates(data)
        if dates is None:
            return data_day
        adj_map = GQ_fetch_etf_adj(codes, start=dates.min(), end=dates.max())
        if not adj_map:
            return data_day

        # 对齐与乘法走 `fq.py` 的共用核心（与股票侧同一实现）。
        # 这里的两处差异**都是承重契约，不是实现细节**：
        #   * `ffill_by_code=False` —— 稀疏约定：查不到 = 1.0（no-op），
        #     **绝不 ffill**（ffill 会把最后一次事件前的系数拖到最新日期而静默算错价）
        #   * `keep_factor_column=False` —— 老树的 `GQ_apply_etf_qfq` 不产生 `adj` 列
        if codes_row is None:
            if len(codes) != 1:
                return data_day
            # 单标的帧推不出代码，用 `codelist` 兜底
            codes_row = pd.Series([codes[0]] * len(data), index=data.index)
        factor = align_factors(data, flatten_factor_map(adj_map),
                               ffill_by_code=False,
                               dates=dates, codes=codes_row)

        if factor.eq(1.0).all():
            return data_day                       # 该区间没有除权事件

        n_bars = int((~factor.eq(1.0)).sum())
        # 注意：由「就地改 `data` 的列」改为「换一个新帧并挂回 `data_day.data`」，
        # 对外行为一致（函数返回的是同一个 `data_day`）。
        data_day.data = multiply_ohlc(data, factor, keep_factor_column=False)

        try:
            data_day.etf_qfq_bars = n_bars
        except Exception:  # noqa: BLE001
            pass
        if verbose:
            print(f'[etf:qfq] {codes[0] if len(codes) == 1 else codes[0:5]} '
                  f'{dates.min()}~{dates.max()} 已前复权 ({n_bars}/{len(factor)} 根调整)')
        try:
            data_day.if_fq = 'qfq'
        except Exception:  # noqa: BLE001
            pass
    except Exception as e:  # noqa: BLE001
        print(f'[etf:qfq] {codelist} 复权失败，按不复权返回: {e!r}')
    return data_day
