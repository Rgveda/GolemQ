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

import pandas as pd
import numpy as np
from datetime import datetime as dt
from datetime import timedelta, timezone
from .constants import TRADE_DATE_SSE
from functools import lru_cache
from GolemQ.core.constants import MARKET_TYPE
import time


def GQ_util_if_trade(day):
    """
    得到前 n 个交易日 (不包含当前交易日)
    '日期是否交易'
    查询上面的 交易日 列表
    :param day: 类型 str eg: 2018-11-11
    :return: Boolean 类型
    """
    if day in TRADE_DATE_SSE:
        return True
    else:
        return False


@lru_cache(maxsize=128)
def GQ_util_get_last_day(ts: dt = None, n: int = 0) -> dt:
    """
    获取最后一个交易日(含当天)
    Get last trading day (including today if trading day)

    Args:
        ts: Target date (default: today)
        n: Offset (0 = last trading day, 1 = previous, etc)

    Returns:
        datetime of the requested trading day
    """
    if ts is None:
        ts = dt.today()

    time_diff_condition = (ts - pd.to_datetime(TRADE_DATE_SSE)) > timedelta(hours=9.5)
    tradedate_until_today = pd.to_datetime(list(filter(
        None,
        np.where(time_diff_condition, TRADE_DATE_SSE, None)
    )))

    if (n > 0) and (n + 1 < len(tradedate_until_today)):
        return tradedate_until_today[-1 - n]

    return tradedate_until_today[-1]


def GQ_util_if_tradetime(
    _time=dt.now(), market=MARKET_TYPE.STOCK_CN, code=None
):
    """
    explanation:
        时间是否交易

    params:
        * _time->
            含义: 指定时间
            类型: datetime
            参数支持: []
        * market->
            含义: 市场
            类型: int
            参数支持: [MARKET_TYPE.STOCK_CN]
        * code->
            含义: 代码
            类型: str
            参数支持: [None]
    """
    _time = dt.strptime(str(_time)[0:19], "%Y-%m-%d %H:%M:%S")
    if market is MARKET_TYPE.STOCK_CN:
        if GQ_util_if_trade(str(_time.date())[0:10]):
            if _time.hour in [10, 13, 14]:
                return True
            elif (
                _time.hour in [9] and _time.minute >= 15
            ):  # 修改成9:15 加入 9:15-9:30的盘前竞价时间
                return True
            elif _time.hour in [11] and _time.minute <= 30:
                return True
            else:
                return False
        else:
            return False
    elif market is MARKET_TYPE.FUTURE_CN:
        date_today = str(_time.date())
        date_yesterday = str((_time - timedelta(days=1)).date())

        is_today_open = GQ_util_if_trade(date_today)
        is_yesterday_open = GQ_util_if_trade(date_yesterday)

        # 考虑周六日的期货夜盘情况
        if not is_today_open:  # 可能是周六或者周日
            if not is_yesterday_open or (
                _time.hour > 2 or _time.hour == 2 and _time.minute > 30
            ):
                return False

        shortName = ""  # i , p
        for i in range(len(code)):
            ch = code[i]
            if ch.isdigit():  # ch >= 48 and ch <= 57:
                break
            shortName += code[i].upper()

        period = [[9, 0, 10, 15], [10, 30, 11, 30], [13, 30, 15, 0]]

        if shortName in ["IH", "IF", "IC"]:
            period = [[9, 30, 11, 30], [13, 0, 15, 0]]
        elif shortName in ["T", "TF"]:
            period = [[9, 15, 11, 30], [13, 0, 15, 15]]

        if 0 <= _time.weekday() <= 4:
            for i in range(len(period)):
                p = period[i]
                hour_cond_a = (_time.hour > p[0] or
                               (_time.hour == p[0] and _time.minute >= p[1]))
                hour_cond_b = (_time.hour < p[2] or
                               (_time.hour == p[2] and _time.minute < p[3]))
                if hour_cond_a and hour_cond_b:
                    return True

        # 最新夜盘时间表_2019.03.29
        nperiod = [
            [["AU", "AG", "SC"], [21, 0, 2, 30]],
            [["CU", "AL", "ZN", "PB", "SN", "NI"], [21, 0, 1, 0]],
            [["RU", "RB", "HC", "BU", "FU", "SP"], [21, 0, 23, 0]],
            [
                [
                    "A",
                    "B",
                    "Y",
                    "M",
                    "JM",
                    "J",
                    "P",
                    "I",
                    "L",
                    "V",
                    "PP",
                    "EG",
                    "C",
                    "CS",
                ],
                [21, 0, 23, 0],
            ],
            [["SR", "CF", "RM", "MA", "TA", "ZC", "FG", "IO", "CY"],
             [21, 0, 23, 30]],
        ]

        for i in range(len(nperiod)):
            for j in range(len(nperiod[i][0])):
                if nperiod[i][0][j] == shortName:
                    p = nperiod[i][1]
                    condA = _time.hour > p[0] or (
                        _time.hour == p[0] and _time.minute >= p[1]
                    )
                    condB = _time.hour < p[2] or (
                        _time.hour == p[2] and _time.minute < p[3]
                    )
                    # in one day
                    if p[2] >= p[0]:
                        if (
                            (_time.weekday() >= 0 and _time.weekday() <= 4)
                            and condA
                            and condB
                        ):
                            return True
                    else:
                        if (
                            (_time.weekday() >= 0 and _time.weekday() <= 4) and condA
                        ) or (
                            (_time.weekday() >= 1 and _time.weekday() <= 5) and condB
                        ):
                            return True
                    return False
        return False


def GQ_util_date_valid(date):
    """
    explanation:
        判断字符串格式(1982-05-11)

    params:
        * date->
            含义: 日期
            类型: str
            参数支持: []

    return:
        bool
    """
    try:
        time.strptime(date, "%Y-%m-%d")
        return True
    except Exception:
        return False


def get_15min_aligned_timestamp(
    current_time=dt.now().time(),
):
    # 增加15分钟对齐的时间戳
    if dt.strptime('09:30', '%H:%M').time() >= current_time:
        last_day = GQ_util_get_last_day()
    else:
        last_day = GQ_util_get_last_day()

    # 交易时间段的15分钟间隔
    trading_intervals_15min = [
        '09:45:00', '10:00:00', '10:15:00', '10:30:00', '10:45:00', '11:00:00', '11:15:00', '11:30:00',
        '13:15:00', '13:30:00', '13:45:00', '14:00:00', '14:15:00', '14:30:00', '14:45:00', '15:00:00'
    ]

    # 找到当前时间应该对齐到的15分钟间隔
    if dt.strptime('09:30', '%H:%M').time() <= current_time <= dt.strptime('12:59', '%H:%M').time():
        # 上午交易时间
        for interval in trading_intervals_15min[:9]:  # 上午的间隔
            interval_time = dt.strptime(interval, '%H:%M:%S').time()
            if current_time <= interval_time:
                return f'{last_day:%Y-%m-%d} {interval}'
        return f'{last_day:%Y-%m-%d} 11:30:00'  # 默认上午最后一个

    elif dt.strptime('13:00', '%H:%M').time() <= current_time <= dt.strptime('15:00', '%H:%M').time():
        # 下午交易时间
        for interval in trading_intervals_15min[9:]:  # 下午的间隔
            interval_time = dt.strptime(interval, '%H:%M:%S').time()
            if current_time <= interval_time:
                return f'{last_day:%Y-%m-%d} {interval}'
        return f'{last_day:%Y-%m-%d} 15:00:00'  # 默认下午最后一个

    else:
        # 非交易时间，使用最后交易时间
        return f'{last_day:%Y-%m-%d} 15:00:00'


def get_60min_aligned_timestamp(
    current_time=dt.now().time(),
):
    # 修正时间判断逻辑
    if dt.strptime('09:00', '%H:%M').time() >= current_time:
        curr_timestamp = f'{GQ_util_get_last_day():%Y-%m-%d} 15:00:00'
    elif dt.strptime('09:00', '%H:%M').time() <= current_time <= dt.strptime('09:30', '%H:%M').time():
        curr_timestamp = f'{GQ_util_get_last_day():%Y-%m-%d} 15:00:00'
    elif dt.strptime('09:30', '%H:%M').time() <= current_time <= dt.strptime('10:30', '%H:%M').time():
        curr_timestamp = f'{GQ_util_get_last_day():%Y-%m-%d} 10:30:00'
    elif dt.strptime('10:30', '%H:%M').time() < current_time <= dt.strptime('12:59', '%H:%M').time():
        curr_timestamp = f'{GQ_util_get_last_day():%Y-%m-%d} 11:30:00'
    elif dt.strptime('13:00', '%H:%M').time() <= current_time <= dt.strptime('14:00', '%H:%M').time():
        curr_timestamp = f'{GQ_util_get_last_day():%Y-%m-%d} 14:00:00'
    elif dt.strptime('14:00', '%H:%M').time() < current_time <= dt.strptime('15:00', '%H:%M').time():
        curr_timestamp = f'{GQ_util_get_last_day():%Y-%m-%d} 15:00:00'
    else:
        # 非交易时间，使用最后交易时间
        curr_timestamp = f'{GQ_util_get_last_day():%Y-%m-%d} 15:00:00'

    return curr_timestamp


def GQ_util_date_stamp(date):
    """
    explanation:
        转换日期时间字符串为浮点数的时间戳

    params:
        * date->
            含义: 日期时间
            类型: str
            参数支持: []

    return:
        time
    """
    if not date:
        return date
    datestr = pd.Timestamp(date).strftime("%Y-%m-%d")
    date = time.mktime(time.strptime(datestr, '%Y-%m-%d'))
    return date


def GQ_util_time_stamp(time_):
    """
    explanation:
       转换日期时间的字符串为浮点数的时间戳

    params:
        * time_->
            含义: 日期时间
            类型: str
            参数支持: ['2018-01-01 00:00:00']

    return:
        time
    """
    if len(str(time_)) == 10:
        # yyyy-mm-dd格式
        return time.mktime(time.strptime(time_, '%Y-%m-%d'))
    elif len(str(time_)) == 16:
        # yyyy-mm-dd hh:mm格式
        return time.mktime(time.strptime(time_, '%Y-%m-%d %H:%M'))
    else:
        timestr = str(time_)[0:19]
        return time.mktime(time.strptime(timestr, '%Y-%m-%d %H:%M:%S'))


def GQ_util_timestamp_to_str(ts_epoch=None, local_tz=None):
    """时间戳 → ``'%Y-%m-%d %H:%M:%S'`` 字符串（**默认 UTC+8**）。

    行为对齐 ``QUANTAXIS.QAUtil.QADate_Adv.QA_util_timestamp_to_str`` ——
    已实测两者对同一输入返回同值。传入 ``None`` 取当前时间。

    ⚠️ 默认时区是 **UTC+8 而非 UTC**。这是 A 股价量数据的口径，不是笔误：
    对齐原实现，改成本地时区会让日志时间戳在非中国时区的机器上静默偏移。

    >>> GQ_util_timestamp_to_str(1704124800)      # 2024-01-02 00:00 UTC+8
    '2024-01-02 00:00:00'
    >>> GQ_util_timestamp_to_str(dt(2024, 1, 2, 10, 30, 0))
    '2024-01-02 10:30:00'
    >>> len(GQ_util_timestamp_to_str())           # 不给参数 → 当前时间
    19
    """
    if local_tz is None:
        local_tz = timezone(timedelta(hours=8))

    if ts_epoch is None:
        ts_epoch = dt.now(timezone(timedelta(hours=8)))

    if isinstance(ts_epoch, dt):
        return ts_epoch.astimezone(local_tz).strftime('%Y-%m-%d %H:%M:%S')

    # 数值时间戳
    return dt.fromtimestamp(float(ts_epoch), local_tz).strftime('%Y-%m-%d %H:%M:%S')


def GQ_util_get_pre_trade_date(cursor_date=None, n: int = 1) -> str:
    """前 n 个交易日。行为对齐 ``QUANTAXIS.QAUtil.QADate_trade.QA_util_get_pre_trade_date``。

    ⚠️ **两处反直觉，都是原样对齐，不要"改对"**：

    1. **非交易日向后找，不是向前。** 名字叫 ``pre``，但 2024-01-06（周六）
       返回的是 **2024-01-08（下周一）**，不是上一个交易日。
    2. **默认 ``n=1``**，即默认不含当天 —— 与其他日期助手的 ``n=0`` 默认不同。

    >>> GQ_util_get_pre_trade_date('2024-01-02', 0)   # 交易日本身
    '2024-01-02'
    >>> GQ_util_get_pre_trade_date('2024-01-02', 1)   # 前一个交易日
    '2023-12-29'
    >>> GQ_util_get_pre_trade_date('2024-01-06', 0)   # 周六 → 下周一（不是上一个）
    '2024-01-08'
    """
    sse = TRADE_DATE_SSE
    if not cursor_date:
        cursor_date = dt.today().strftime('%Y-%m-%d')
    else:
        cursor_date = pd.Timestamp(cursor_date).strftime('%Y-%m-%d')

    if cursor_date in sse:
        return sse[sse.index(cursor_date) - n]

    # 原实现走 QA_util_get_real_date(cursor_date, towards=1)，
    # 语义即「向后找第一个交易日」—— 照搬，不改成向前。
    for day in sse:
        if day > cursor_date:
            return sse[sse.index(day) - n]
    raise ValueError(f'{cursor_date} 之后没有交易日了（TRADE_DATE_SSE 覆盖不足？）')
