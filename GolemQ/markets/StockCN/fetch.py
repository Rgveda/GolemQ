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
# all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#

# from datetime import datetime, timedelta
# from functools import lru_cache
import numpy as np
import pandas as pd
# from .constants import TRADE_DATE_SSE
from GolemQ.core import GQ_util_code_tolist
from .date_utils import (
    GQ_util_date_valid,
    GQ_util_date_stamp,
    GQ_util_get_last_day,
)
from GolemQ.core.preprocessing import (
    GQ_util_to_json_from_pandas,
)
from GolemQ.core.constants import (
    MARKET_TYPE,
    AKA,
    FIELD as FLD,
)
from datetime import (
    datetime as dt,
    timedelta,
    timezone,
)
import traceback
from func_timeout import func_set_timeout
from .symbol import (
    normalize_code,
    is_stock_cn,
)
# 时间戳格式化已本地化（GQ_util_timestamp_to_str 实测与 QA 版同值）。
# 原先包 try/except 是因为「没装 QUANTAXIS 就不 import」—— 现已无必要，
# 而且那层保护会把真正的导入错误吞掉。
from GolemQ.markets.StockCN.date_utils import GQ_util_timestamp_to_str
from .datastruct import (
    GQ_DataStruct_ETF_day,
    GQ_DataStruct_ETF_min,
    GQ_DataStruct_Index_min,
    GQ_DataStruct_Index_day,
    GQ_DataStruct_Stock_day,
    GQ_DataStruct_Stock_min,
    GQ_DataStruct_Stock_block,
)
import warnings
from .kline83 import (
    GQ_fetch_stock_day_adv,
    market_prefix,
    normalize_frequency,
    read_min_frame,
)
from .etf_fq import GQ_apply_etf_qfq
from .refdata import (
    GQ_fetch_stock_block,
    GQ_fetch_stock_list,
)
from .realtime import (
    GQ_fetch_stock_day_realtime_adv,
)



def GQ_fetch_stock_min(
    code,
    start,
    end,
    format='numpy',
    frequence='1min',
    market_type=None
):
    """A 股分钟线（MongoDB 8.3 时序库）。返回以 ``datetime`` 为索引的扁平帧。

    本函数是**重写**，不是改库名。改动前的每一处都是缺陷：

    1. **读 8.3 的分频集合**（``stock_1min`` / ``index_5min`` …），不再读
       ``DATABASE.stock_min`` —— 那个集合在 ``golemq`` 库里**不存在**，实测本函数
       恒返回 ``None``（空游标 → ``res.vol`` 抛 ``AttributeError`` → 被下面那个
       光秃秃的 ``except`` 吞成 ``None``）。
    2. **去掉 ``collections=`` 参数**。它让调用方能指错库，而实测正是指错的；
       集合改由 :func:`kline83.market_prefix` 按标的判定 —— ETF 与指数共用
       ``index_*``，「按代码段猜」会读空。需要显式指定市场时传 ``market_type``。
    3. **频率归一化收敛到** :func:`kline83.normalize_frequency`（原来同一张别名表
       抄了三个模块，且未知频率只打印一行就继续拿原值拼集合名 → 静默空结果；
       现在统一抛 ``ValueError``）。
    4. **按 ``(code, ts)`` 查询**，不再用 ``time_stamp`` + ``type`` 字段 ——
       ``ts`` 是时序集合的 ``timeField``，规划器靠它分桶剪枝（见 ``kline83``
       模块说明）；8.3 的频率在**集合名**里，没有 ``type`` 字段。
    5. **``format='numpy'`` 且无数据时返回 ``None``**，不再是 ``np.asarray(None)``
       —— 那是个 0 维 object 数组，既不是 None 也不能当帧用。

    ⚠️ **一处刻意不照搬：零成交量 bar 保留。** QUANTAXIS 的 ``QA_fetch_stock_min``
    里有 ``.query('volume>1')``，会**丢掉所有零成交的分钟**。本函数**不过滤**，
    理由有三，**详见 ``PITFALLS.md`` P8b**（通达信 0 成交分钟的三种成因）：

    1. **收盘集合竞价**（14:57–15:00）本就没有连续成交 —— 实测 600519 在
       2026-09-01~18 的 19 根 0 量 bar **全部落在 14:58/14:59**，每只票每天都
       会有。`volume>1` 等于**每天删掉所有股票的收盘竞价分钟**。
    2. **封死涨跌停**的票在无人卖出的分钟同样 0 成交，那是真实行情。
    3. 停牌造成的伪 0（老 4.4 里是 `1e-34` 级 float）**才是该剔的那个**，
       但它在迁移到 8.3 时被 cast 成 int，**判别特征已丢失**，只能靠
       「整日全部 bar 皆为 0 且价格持平」识别 —— 那是**消费方**的判断，
       不是读取器能替它做的。

    所以：读取器如实返回库里存的，由消费方决定丢什么 —— ``quotes.py`` 的
    分钟路径与 ``get_kline_price_min`` 都有专门的 zero_trading 处理（删 4 根一组的
    午休段、修正 13:00 时间戳等），**它们需要看见这些 bar 才能做判断**。
    这是全树「适配器只管取数，编排层决定范围」的同一条分工（``PITFALLS.md`` P1）。

    :param format: ``'pd'`` 返回 DataFrame（唯一有消费方的取值），另有
        ``json`` / ``numpy`` / ``list``；未知取值返回 ``None``。
    :return: 帧（列含 ``datetime`` / ``code`` / ``volume`` / OHLC / ``amount``）；
        **无数据或 format 非法时返回 ``None``**。
    """
    try:
        res = read_min_frame(code, start, end, frequence, market_type=market_type)
    except ValueError as e:
        print(f'GolemQ Error GQ_fetch_stock_min: {e}')
        return None
    if res is None or len(res) == 0:
        return None

    res = (res.drop_duplicates(['datetime', 'code'])
              .set_index('datetime', drop=False)
              .sort_index())

    if format in ['P', 'p', 'pandas', 'pd']:
        return res
    elif format in ['json', 'dict']:
        return GQ_util_to_json_from_pandas(res)
    # 多种数据格式
    elif format in ['n', 'N', 'numpy']:
        return np.asarray(res)
    elif format in ['list', 'l', 'L']:
        return np.asarray(res).tolist()
    else:
        print(
            "QA Error QA_fetch_stock_min format parameter %s is none of  \"P, p, pandas, pd , json, dict , n, N, numpy, list, l, L, !\" "
            % format
        )
        return None
    

def GQ_fetch_stock_min_adv(
        code,
        start,
        end=None,
        frequence='1min',
        if_drop_index=True,
        verbose=False,):
    '''
    '获取股票分钟线'
    :param code:  字符串str eg 600085
    :param start: 字符串str 开始日期 eg 2011-01-01
    :param end:   字符串str 结束日期 eg 2011-05-01
    :param frequence: 字符串str 分钟线的类型 支持 1min 1m 5min 5m 15min 15m 30min 30m 60min 60m 类型
    :param if_drop_index: Ture False ， dataframe drop index or not
    :return: 股票为 ``GQ_DataStruct_Stock_min``；ETF/指数为
        ``GQ_DataStruct_Index_min``（**与 QUANTAXIS 一致：指数类没有
        ``to_qfq``**，复权由 ``etf_fq.py`` 负责）
    '''
    # 别名归一化收敛到 kline83 一处（原来三个模块各抄一份，失败处理还各不相同）。
    # 本函数保留「非法输入返回 None」的既有契约 —— 它的调用方（quotes.py）都有
    # `is None` 分支；quotes.py 自己也会先归一化一次，非法频率在那边就抛错了。
    try:
        frequence = normalize_frequency(frequence)
    except ValueError as e:
        if (verbose):
            print(f'GolemQ Error GQ_fetch_stock_min_adv: {e}')
        return None

    # __data = [] 未使用

    end = start if end is None else end
    if len(start) == 10:
        start = '{} 09:30:00'.format(start)

    if len(end) == 10:
        end = '{} 15:00:00'.format(end)

    if start == end:
        # 🛠 todo 如果相等，根据 frequence 获取开始时间的 时间段 QA_fetch_stock_min， 不支持start
        # end是相等的
        if (verbose):
            print(f"GolemQ Error GQ_fetch_stock_min_adv parameter code={code:%s}, start={start:%s}, end={end:%s} is equal, should have time span! ")
        return None

    # 🛠 todo 报告错误 如果开始时间 在 结束时间之后
    res = GQ_fetch_stock_min(code, start, end, format='pd', frequence=frequence)

    if res is None:
        if (verbose):
            print(f"QA Error GQ_fetch_stock_min_adv parameter code={code:%s}, start={start:%s}, end={end:%s} frequence={frequence:%s} call GQ_fetch_stock_min return None")
        return None
    else:
        res_set_index = res.set_index(['datetime', 'code'], drop=if_drop_index)
        # 容器类型跟随**数据所属的市场**，而不是函数名里的 "stock"：ETF 与指数
        # 共用 index_* 集合，拿它们的数据装进 Stock 容器会让
        # `isinstance(data_min, GQ_DataStruct_Index_min)` 恒为假 —— `fetch.py`
        # 靠那两个 isinstance 决定去取股票名还是 ETF 名。
        return _min_container(res_set_index, code)


def _min_container(res_set_index, code):
    """帧 → 容器：股票走 ``Stock_min``，ETF 走 ``ETF_min``，真指数走 ``Index_min``。

    容器类型跟随**数据所属的市场**，而不是函数名里的 "stock"：拿 ETF 的数据装进
    Stock 容器会让 ``isinstance`` 判定失真 —— ``fetch.py`` 靠它决定去取股票名
    还是 ETF 名。

    ⚠️ **三分支缺一不可。** 2026-09 ETF 独立成 ``etf_*`` 之前，这里只有
    「index / 否则 stock」两分支，ETF 与真指数共用 ``index_*``。若只改
    ``market_prefix`` 而不改这里，ETF 会**落到 ``Stock_min``** ——
    那会拿到 ``to_qfq()``（按 ``stock_adj`` 错乘 ETF 价格）且被当股票取名，
    **全程不报错**。这是本次改动里最容易漏的一处。
    """
    probe = code[0] if isinstance(code, (list, tuple, set)) else code
    prefix = market_prefix(probe)
    if prefix == 'etf':
        return GQ_DataStruct_ETF_min(res_set_index)
    if prefix == 'index':
        return GQ_DataStruct_Index_min(res_set_index)
    return GQ_DataStruct_Stock_min(res_set_index)


def GQ_fetch_index_min_adv(
        code,
        start,
        end=None,
        frequence='1min',
        if_drop_index=True,
        verbose=False,):
    """指数 / ETF 分钟线。**与 ``GQ_fetch_stock_min_adv`` 是同一条路径。**

    两者本就只差容器的选法，而容器现在由 :func:`kline83.market_prefix` 按代码
    判定（ETF 与指数共用 ``index_*`` 集合）。保留这个名字是因为调用点
    （``pipeline/base.py``）按市场分支书写，直接用「股票函数」读数据会让读者
    以为写错了。

    原实现走 QUANTAXIS 的 ``QA_fetch_index_min_adv``（4.4 库）。
    """
    return GQ_fetch_stock_min_adv(code, start, end=end, frequence=frequence,
                                  if_drop_index=if_drop_index, verbose=verbose)
