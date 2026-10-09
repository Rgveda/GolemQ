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
from datetime import (
    datetime as dt,
    timedelta,
    timezone,
)
import datetime
from .date_utils import (
    GQ_util_get_last_day,
    GQ_util_if_tradetime,
)
from .symbol import (
    is_stock_cn,
    normalize_code,
)
import traceback
from GolemQ.core.constants import (
    AKA,
    FIELD as FLD,
)
import time as timer
import time
import requests
import json
import time as ts
from GolemQ.supervisor.heartbeat import HeartbeatModule
try:
    from .easyquotation import use as eq_use
    easyquotation_not_install = False
except ImportError:
    easyquotation_not_install = True
import pymongo
from pymongo import UpdateOne
from collections import deque
# 注意：这里**不要**再 import `QAREALTIME` —— 它是 4.4 老库的句柄，
# 实时读写已全部切到 8.3 的 `golemq_stock_cn_realtime`（见 `_realtime_db`）。
from tqdm import tqdm
import urllib3
import threading
from GolemQ.core.constants import MARKET_TYPE
from GolemQ.analysis.timeseries import (
    GQ_data_min_resample,
    GQ_data_min_to_day,
)
from .refdata import GQ_fetch_stock_info
# 时区换算的唯一入口（`ts` 字段口径）。`kline83` 只用函数级包导入，
# 不会与本模块成环。
from .kline83 import bj_date
from .date_utils import (
    GQ_util_if_tradetime as QA_util_if_tradetime,
    GQ_util_get_pre_trade_date as QA_util_get_pre_trade_date,
)
from func_timeout import func_set_timeout
from pandas.tseries.frequencies import to_offset


def formater_l1_tick(code: str, l1_tick: dict) -> dict:
    """
    处理分发 Tick 数据，新浪和tdx l1 tick差异字段格式化处理
    """
    if ((len(code) == 6) and code.startswith('00')):
        l1_tick['code'] = normalize_code(code, l1_tick['now'])
    else:
        l1_tick['code'] = normalize_code(code)

    if ('time' in l1_tick):
        # Sina snapshot
        l1_tick['servertime'] = l1_tick['time']
        if ('date' in l1_tick):
            l1_tick['datetime'] = '{} {}'.format(l1_tick['date'], l1_tick['time'])
        elif ('timetag' in l1_tick):
            # 解析原始时间字符串
            tick_datetime = dt.strptime(l1_tick['timetag'], '%Y%m%d %H:%M:%S')

            # 格式化为目标格式
            l1_tick['datetime'] = tick_datetime.strftime('%Y-%m-%d %H:%M:%S')
        if ('lastPrice' in l1_tick):
            l1_tick['price'] = l1_tick['lastPrice']
        else:
            l1_tick['price'] = l1_tick['now']
        l1_tick['vol'] = l1_tick['volume']
        if ('date' in l1_tick):
            del l1_tick['date']
        del l1_tick['time']
        if ('lastPrice' in l1_tick):
            del l1_tick['lastPrice']
        if ('now' in l1_tick):
            del l1_tick['now']
        if ('nam' in l1_tick):
            del l1_tick['name']
        del l1_tick['volume']
    else:
        # Tecenct snapshot
        l1_tick['servertime'] = '{}'.format(l1_tick['datetime'].time())
        l1_tick['datetime'] = '{}'.format(l1_tick['datetime'])
        l1_tick['price'] = l1_tick['now']
        l1_tick['vol'] = l1_tick['成交量(手)']
        l1_tick['amount'] = l1_tick['成交额(万)']
        if ('now' in l1_tick):
            del l1_tick['now']
        if ('nam' in l1_tick):
            del l1_tick['name']
        del l1_tick['volume']
    return l1_tick


#: 实时落库的 `source` 取值 —— 同一个日集合里区分数据流，**且是去重键的一部分**
#: （见 :func:`_write_ts_rows`：删除条件不带它，两条流会互删对方刚写的行）。
REALTIME_SOURCE_TENCENT_L1 = 'tencent_l1'
REALTIME_SOURCE_TENCENT_L2 = 'tencent_l2'
REALTIME_SOURCE_QMT = 'qmt'

#: MiniQMT 自 2026-10-01 起停服（监管），替代方案（cfquant）尚未落地。
#: ``False`` = 不订阅、不取数、不刷错 —— QMT 那条路的**结构保留**，
#: 将来接上替代源时把它改回 ``True``，并让 :func:`_l2_rows_from_qmt` 对接新源。
QMT_REALTIME_ENABLED = False


def realtime_collection_name(day=None):
    """当日实时集合名 —— ``realtime_YYYY-MM-DD``。

    格式**必须**与 `markets/StockCN/tools.py` 的
    ``f"realtime_{target_date.strftime('%Y-%m-%d')}"``（purge 那条）逐字符一致：
    purge 按名字找集合，找不到只是**安静地**计入 miss，连续 14 次就退出 ——
    名字分叉的症状是**保留策略静默失效**（磁盘无声地涨），不是报错。

    ⚠️ 不要把 ``datetime`` 直接 ``format`` 进名字：本模块的 ``dt`` 是
    ``datetime`` **类**，``'realtime_{}'.format(dt.today())`` 会拼出
    ``'realtime_2026-10-08 01:00:27.222907'``（老树那里用的是 ``date.today()``，
    重构时被换掉 —— 那条读路径因此**从来没命中过任何集合**）。

    :param day: ``date`` / ``datetime`` / ``'YYYY-MM-DD'`` / ``Timestamp``；``None`` = 今天
    :returns: ``'realtime_YYYY-MM-DD'``
    """
    day = dt.today() if day is None else day
    return 'realtime_{}'.format(pd.Timestamp(day).strftime('%Y-%m-%d'))


def collections_of_today(database):
    """当天实时集合 —— **普通集合**，QMT / 老 L1 路径的写法。

    ⚠️ **当前无调用者**：唯一使用方是 `gateway/xtquant/realtime.py`，而 MiniQMT
    自 2026-10-01 起停服（见 :data:`QMT_REALTIME_ENABLED`）。函数与那三个索引
    **保留**，是为了将来接替代源时有个明确的对接口；不要因为"没人用"就删掉。

    ⚠️ 它建**普通**集合 + ``(code, datetime)`` 索引 + upsert 写入，与现在的
    **时间序列**存储**不兼容**（实测时间序列不支持唯一索引、不能 upsert）。
    复活这条路时必须改走 :func:`realtime_ts_collection` + :func:`_write_ts_rows`，
    否则第一次写就会抛 ``Cannot perform a non-multi update``。
    """
    collection = database.get_collection(realtime_collection_name())
    collection.create_index([('code', pymongo.ASCENDING)])
    collection.create_index([('datetime', pymongo.ASCENDING)])
    collection.create_index(
        [("code", pymongo.ASCENDING),
            ("datetime", pymongo.ASCENDING)],  # unique=True,
    )
    return collection


def formater_l1_ticks(l1_ticks: dict, codelist: list = None, stacks=None, symbol_list=None) -> dict:
    """
    处理 l1 ticks 数据
    """
    if (stacks is None):
        l1_ticks_data = []
        symbol_list = []
    else:
        l1_ticks_data = stacks

    for code, l1_tick_values in l1_ticks.items():
        # l1_tick = namedtuple('l1_tick', l1_ticks[code])
        # formater_l1_tick_jit(code, l1_tick)
        if (codelist is None) or \
                (code in codelist):
            l1_tick = formater_l1_tick(code, l1_tick_values)
            if (l1_tick['code'] not in symbol_list):
                l1_ticks_data.append(l1_tick)
                symbol_list.append(l1_tick['code'])

    return l1_ticks_data, symbol_list


def sub_l1_from_tencent(database_realtime=None):
    """从腾讯获取 L1 数据（成交快照，**含五档**）。

    `database_realtime` 默认 8.3 的 `golemq_stock_cn_realtime`。**必须有默认值** ——
    CLI 的 `--sub` 是 `subscriber_func()` 无参调用，原先这个位置参数是必填，
    于是文档里写的 `python -m GolemQ.cli --sub l1_tencent` 一跑就
    `TypeError`（被 CLI 的 except 吞成一行"发生错误"）。

    **落库形态**：写 8.3 的 `golemq_stock_cn_realtime.realtime_YYYY-MM-DD`，
    **按日**的**时间序列**集合（`timeField='ts'`、`metaField='code'`、
    `granularity='seconds'`）。与 `golemq_stock_cn` 的 `stock_1min` 同规格，
    按 `(code, ts)` 查能走分桶剪枝。

    为什么按日、且名字必须是 `realtime_YYYY-MM-DD`：保留策略是**按集合名**删的
    （`markets/StockCN/tools.py` 的 purge，删 14 天前的整日集合）。名字一旦分叉，
    purge 不会报错 —— 它只是连续 miss 后安静退出，**保留策略静默失效**。

    ⚠️ 时间序列**不能 upsert、也没有唯一索引**（实测），故写入一律走
    :func:`_write_ts_rows`（它对首次见到的 `(source, code)` 先删后插，细节见那里）。
    """
    urllib3.util.connection.DEFAULT_MAX_POOL_SIZE = 100

    if database_realtime is None:
        database_realtime = _realtime_db()

    if (easyquotation_not_install is True):
        print(u'PLEASE run "pip install easyquotation" before call GolemQ.cli.sub modules')
        return

    # 创建心跳监控实例
    module = HeartbeatModule(
        module_name="sub_l1_from_tencent",
        instance_id=f"sub_l1_from_tencent_{dt.now().strftime('%Y%m%d_%H%M%S')}",
        timeout_seconds=30,  # 5分钟超时
    )

    if (module.mutex(
            verbose=True)):
        return False
    else:
        # 开始模块执行记录
        module.start(
            initial_message="L1数据订阅模块启动"
        )

    # 主函数逻辑
    quotation = eq_use('tencent')  # 新浪 ['sina'] 腾讯 ['tencent', 'qq']

    sleep_time = 2.0
    sleep = int(sleep_time)
    _time1 = dt.now()
    collection_day = dt.now().date()
    collection = realtime_ts_collection(
        database_realtime, realtime_collection_name(collection_day))
    last_ts: dict = {}
    get_once = True

    # 初始化一个队列，用于存储最后两次的数据
    l1_ticks_data_last = deque(maxlen=2)
    sync_count = 0
    current_date = datetime.datetime.now()
    start_time = current_date.replace(hour=9, minute=15, second=0, microsecond=0)
    end_time = current_date.replace(hour=15, minute=15, second=0, microsecond=0)

    while (start_time < datetime.datetime.now() < end_time) or get_once:
        # 开盘/收盘时间
        end_time = dt.strptime(str(dt.now().date()) + ' 15:15',
                               '%Y-%m-%d %H:%M')
        _time = dt.now()

        # 跨日切换：集合名按「当天」取，长跑进程过零点后自动落进新集合。
        # 原来那个哨兵判断（`GQ_util_if_tradetime(_time) and dt.now() < 今天01:00`）
        # **逻辑上不可达** —— 交易时间必然在 09:15 之后，不可能同时早于当天 01:00，
        # 所以它从来没有跨日能力，只会每轮吞掉一次取样、并打印一句错的
        # 「Not Trading time」。不要照它改回来。
        if dt.now().date() != collection_day:
            collection_day = dt.now().date()
            collection = realtime_ts_collection(
                database_realtime, realtime_collection_name(collection_day))

        # 心跳签到
        if (sync_count % 8 == 1):
            module.checkin(
                message=f"处理L1数据，当前时间: {_time.strftime('%Y-%m-%d %H:%M:%S')}"
            )

        symbol_list = []
        l1_ticks_data = []
        if GQ_util_if_tradetime(_time) or \
                (get_once):  # 如果在交易时间
            l1_ticks = quotation.market_snapshot(prefix=True)
            l1_ticks_data, symbol_list = formater_l1_ticks(l1_ticks)

            # 获取第二遍，包含上证指数信息
            l1_ticks = quotation.market_snapshot(prefix=False)
            l1_ticks_data, symbol_list = formater_l1_ticks(
                l1_ticks,
                stacks=l1_ticks_data,
                symbol_list=symbol_list)

            # 将新数据转换为 DataFrame 并设置索引
            l1_ticks_data_idx = pd.DataFrame(
                [{'code': l1_tick['code'],
                  'datetime': l1_tick['datetime']} for l1_tick in l1_ticks_data],
            ).set_index(['datetime', 'code'], drop=False)

            # 获取最后两次数据的合集
            if l1_ticks_data_last:
                l1_ticks_data_last_combined = pd.concat(l1_ticks_data_last)
            else:
                l1_ticks_data_last_combined = pd.DataFrame(
                    columns=[
                        'datetime',
                        'code']).set_index([
                            'datetime',
                            'code'], drop=False, )

            # 找出新数据中不重复的部分
            l1_ticks_data_neo = l1_ticks_data_idx.index.difference(l1_ticks_data_last_combined.index)

            if (len(l1_ticks_data_neo) > 0):
                # 补 `ts`（UTC-aware，时间序列的 timeField）后写库。
                # 行内**同时保留 `datetime`**（北京时字符串）—— 下游
                # `GQ_data_tick_resample_1min` 的重采样按它取时刻，别删。
                for l1_tick in l1_ticks_data:
                    l1_tick['ts'] = bj_date(l1_tick.get('datetime'))
                _write_ts_rows(collection, l1_ticks_data, last_ts,
                               REALTIME_SOURCE_TENCENT_L1)
            if (get_once is not True):
                print(
                    u'Trading time now 现在是中国A股交易时间 {}\n'
                    u'Processing ticks data cost:{:.3f}s'.format(
                        dt.now().strftime("%Y-%m-%d %H:%M:%S"),
                        (dt.now() - _time).total_seconds()))
            if ((dt.now() - _time).total_seconds() < sleep):
                timer.sleep(sleep - (dt.now() - _time).total_seconds())
            print('Program Last Time {:.3f}s'.format((dt.now() - _time1).total_seconds()))
            get_once = False
        else:
            print(u'Not Trading time 现在是中国A股收盘时间 {}'.format(_time.strftime("%Y-%m-%d %H:%M:%S")))
            timer.sleep(sleep)
            
        sync_count = sync_count + 1
    
    # 每天下午5点，代码就会执行到这里，如有必要，再次执行收盘行情下载，也就是 QUANTAXIS/save X
    save_time = dt.strptime(str(dt.now().date()) + ' 17:00', '%Y-%m-%d %H:%M')
    if (dt.now() > end_time) and \
            (dt.now() < save_time):
        # 收盘时间 下午16:00到17:00 更新收盘数据
        # 我不建议整合，因为少数情况会出现 程序执行阻塞 block，
        # 本进程被阻塞后无人干预第二天影响实盘行情接收。
        # save_x_func()
        pass
    timer.sleep(15)

    # 标记模块执行完成
    module.complete(
        completion_message="L1数据订阅模块正常结束"
    )

    # While循环每天下午5点自动结束，在此等待13小时，大概早上六点结束程序自动重启
    print(u'While循环每天下午5点自动结束，在此等待半小时，大概早上六点结束程序自动重启，这样只要窗口不关，永远每天自动收取 tick')
    timer.sleep(1800)


def _realtime_db():
    """8.3 的实时库句柄。**函数级导入是刻意的。**

    `markets/StockCN/__init__.py:55` 经 `.quotes` 间接导入本模块，而
    `GOLEMQ_STOCK_CN_REALTIME` 到 `:70` 之后才定义 —— 顶层 `from . import`
    会抛 partially-initialized 的 ImportError（`datastruct`/`kline83`/`refdata`
    出于同一原因都这样写）。
    """
    from . import GOLEMQ_STOCK_CN_REALTIME
    return GOLEMQ_STOCK_CN_REALTIME


def realtime_ts_collection(database, name):
    """取（必要时创建）实时用的**时间序列**集合。

    为什么是时间序列 + `ts`
    ======================
    与 `golemq_stock_cn` 的 `stock_1min` 等同一规格：`timeField='ts'`、
    `metaField='code'`，`granularity='seconds'`（tick 级）。这样按
    `(code, ts)` 区间查能走**分桶剪枝**，与历史行情的读法一致。

    ⚠️ 两条**实测约束**（MongoDB 8.3.2）决定了写入方式
    ================================================
    * ``Unique indexes are not supported on time-series collections``
    * 因而 ``Cannot perform a non-multi update on a time-series collection``
      —— **不能 upsert**

    所以写入一律 `insert_many`，**去重靠调用方按 `ts` 判新**（见
    :func:`_write_ts_rows`），不靠库。这与旧的按日集合 + 唯一索引 upsert 是
    不同的机制，不要照旧写法改回来。

    :param name: 一律来自 :func:`realtime_collection_name`（``realtime_YYYY-MM-DD``）
        —— **不要在这里或调用处自己拼字符串**，名字格式是 purge 的契约
        （见 :func:`realtime_collection_name` 的说明）。
    """
    try:
        return database.create_collection(
            name, timeseries={'timeField': 'ts', 'metaField': 'code',
                              'granularity': 'seconds'})
    except Exception:                 # 已存在（或并发创建）——直接用
        return database[name]


def _write_ts_rows(collection, rows, last_ts, source):
    """把 `rows` 写进当日时间序列集合 —— **先删后插**，因而幂等。

    时间序列不能 upsert、也没有唯一索引（见 :func:`realtime_ts_collection`），
    所以「同一个 tick 被写两遍」只能在这里挡，两件事各挡一半：

    * **进程内**：`ts <= last_ts[code]` 的行直接丢 —— 老写法靠它挡住
      「每 2 秒把整份快照再存一遍」（同一秒内重复取到的快照）；
    * **跨进程 / 重启后**：`last_ts` 是内存态，进程重启就没了，而时间序列
      **不拦重复**（实测同一 ``(code, ts)`` 插两次就是两行）—— 故对**本进程
      首次见到的** ``(source, code)`` 先用 ``delete_many`` 清掉同 ``ts`` 的行再插。

    ⚠️ **`source` 必须进删除条件**：L1（腾讯全量快照）与 L2（腾讯五档）
    落在**同一个日集合**里，同一 ``(code, ts)`` 会有两行（L1 是 L2 的真超集，
    这是刻意接受的重复）。删除条件不带 `source`，L1 那轮就会把 L2 刚写的
    行删掉，反之亦然。

    ⚠️ **code 一律保持 :func:`normalize_code` 的带后缀形式**（``600519.XSHG``），
    调用方不要为了"省事"截成 6 位：``000001`` 是上证指数与平安银行共用的号，
    靠 ``pre_close`` 启发式区分，截断会把两只标的并成同一个 metaField 值。

    `last_ts` 由调用方持有（``{(source, code): 最近写入的 ts}``），本函数就地更新它。
    **键要含 `source`**：同一个循环里可能先后写两条流（见 `sub_l2_from_tencent`），
    只按 `code` 记的话，第二条流的第一行会被当成"重复"丢掉。
    缺 `code` 或 `ts` 的行直接丢（`ts` 是 timeField，缺了 insert 会报错）。

    :param source: :data:`REALTIME_SOURCE_TENCENT_L1` 等；写进每行的 `source` 字段
    :returns: 实际写入的行数
    """
    fresh, new_keys = [], set()
    for r in rows:
        code, ts = r.get('code'), r.get('ts')
        if not code or ts is None:
            continue
        key = (source, code)
        prev = last_ts.get(key)
        if prev is not None and ts <= prev:
            continue
        if prev is None:
            new_keys.add(key)
        last_ts[key] = ts
        r['source'] = source
        fresh.append(r)
    if not fresh:
        return 0
    # ⚠️ **只删「本进程首次见到的 (source, code)」的行**，不要每轮无条件删。
    # 每轮无条件删是**跑不动**的：实测 4900 个 metaField 值的 `delete_many`
    # 约 **3.7 秒**（时序删除是「解压桶 → 摘测量 → 回写桶」，而命中的正是当前
    # 热桶），而 L1 订阅器每 **2 秒**一轮。
    # 收窄后语义不变：进程内的重复已由上面的 `last_ts` 挡住，跨进程的重复只可能
    # 出现在**重启后的第一批**（那时 `last_ts` 为空 → 整批都是新 key → 全量删一次）。
    #
    # ⚠️ 判据是**按 key（而不是按行）**：同一批里同一个 code 若有多个 ts，
    # 它们**都要**进删除条件。按行判会漏掉"该 code 的第二行及以后"——
    # 那一版实测过：模拟重启后库内从 5 行涨到 9 行（幂等被破坏）。
    if new_keys:
        collection.delete_many({
            'code': {'$in': sorted({c for _src, c in new_keys})},
            'ts': {'$in': sorted({r['ts'] for r in fresh
                                  if (source, r['code']) in new_keys})},
            'source': source,
        })
    collection.insert_many(fresh, ordered=False)
    return len(fresh)


#: L2 五档盘口的字段名 —— 与 `easyquotation/tencent.py` 的解析结果同形。
L2_DEPTH_FIELDS = tuple(
    '%s%d%s' % (side, i, suffix)
    for side in ('bid', 'ask')
    for i in range(1, 6)
    for suffix in ('', '_volume')
)

#: 腾讯快照里顺带留下的标量字段（都是响应里现成的，不额外请求）。
L2_EXTRA_FIELDS = ('涨停价', '跌停价', '均价', '委差', '量比')


def _l2_row(code, ts, source, **vals):
    """一行 L2。`ts` 必须是 **UTC-aware**（时间序列的 timeField）。

    两份时间的口径收敛在这里：`ts` 是唯一来源，`datetime` 由它反推成北京时间的
    可读字符串。**不要**让调用方各传各的 —— 那正是 `kline83` 模块反复强调的
    「时区换算只应有一处」。

    `code` 必须是 :func:`normalize_code` 的**带后缀**形式：L1 行存的就是那个形式，
    两条流落在同一个日集合里，按 code 查时才能一起命中。裸 6 位（``'600519'``）
    会让 ``$in ['600519.XSHG']`` **静默读空**。

    **`None` 不进库**：「字段缺失」与「值为 0」必须能区分 —— 盘口全 0
    （集合竞价前）是有意义的状态，不能与没取到混为一谈。
    """
    row = {'code': code, 'source': source}
    if ts is not None:
        row['ts'] = ts
        row['datetime'] = pd.Timestamp(ts).tz_convert('Asia/Shanghai').strftime(
            '%Y-%m-%d %H:%M:%S')
    for k, v in vals.items():
        if v is not None and v == v:          # 排除 None 与 NaN
            row[k] = v
    return row


def _l2_rows_from_tencent(quotation):
    """腾讯全市场快照 → L2 行（含五档深度）。

    两次 `market_snapshot`（`prefix=True/False`）的合计约 4,900 只，
    与 L1 订阅器同一份数据源 —— 区别只在本函数**把深度显式取出来**，
    按 3 秒节奏再落一遍（L1 行是它的真超集，重复是刻意接受的：
    两条流靠 `source` 区分，见 :func:`_write_ts_rows`）。

    `code` 走 :func:`normalize_code`（带后缀），与 L1 行、与读取器的查询同形。
    `pre_close` 喂 `tick['now']` 是为了 ``000001`` 那条启发式与 L1 一致
    （上证指数与平安银行共用这个号）。
    """
    rows, seen = [], set()
    for prefix in (True, False):
        for code, tick in quotation.market_snapshot(prefix=prefix).items():
            c6 = str(code)[-6:]
            if c6 in seen:
                continue
            seen.add(c6)
            vals = {'price': tick.get('now'), 'volume': tick.get('volume')}
            for f in L2_DEPTH_FIELDS:
                if f in tick:
                    vals[f] = tick[f]
            for f in L2_EXTRA_FIELDS:
                if f in tick:
                    vals[f] = tick[f]
            # 腾讯给的是 naive 北京时间字符串 → 交给 bj_date 换算（唯一入口）
            rows.append(_l2_row(normalize_code(c6, tick.get('now')),
                                bj_date(tick.get('datetime')),
                                REALTIME_SOURCE_TENCENT_L2, **vals))
    return rows


def _l2_qmt_xt_codes(codelist):
    """GolemQ 6 位代码 → QMT 的 `510300.SH` 形式。

    解析会走 `symbol.is_stock_cn`，而它对**每一个不认识的代码**都打印一行
    （`00158061 True None SZ 深交所未知代码` 之类）。ETF 名单里有大量它不认识的
    号段（158xxx/150xxx 等分级基金），实测一次解析刷出**几百行**，把本订阅器
    自己的进度输出整个淹掉。故这里用 `suppress_stdout_stderr` 包住解析 ——
    `crawler.py` 用它是同一个理由（遮 QUANTAXIS 的 banner）。
    """
    from GolemQ.core.presentation import suppress_stdout_stderr
    from .datasource.qmt_source import GQ_qmt_resolve_xt_code
    out = []
    with suppress_stdout_stderr():
        for c in codelist:
            try:
                out.append(GQ_qmt_resolve_xt_code(c))
            except Exception:
                continue
    return out


def _l2_rows_from_qmt(xt_codes, subscribe: bool = True):
    """MiniQMT → 五档行（ETF 走这条）。

    ⚠️ **MiniQMT 自 2026-10-01 起停服，目前取不到任何数据**（监管）；
    调用点已在 :func:`sub_l2_from_tencent` 用 :data:`QMT_REALTIME_ENABLED` 关掉。
    本函数**结构保留**，等接上替代源（cfquant）后对接即可。

    ⚠️ **必须先 `subscribe_quote(period='tick')`**：未订阅的代码
    `get_full_tick` 只返回陈旧缓存（实测盘口全 0、`time` 停在几分钟前），
    看起来像「盘口是空的」，实则是没订阅。这是这条路上唯一的坑。

    订阅是**持续生效**的，所以订阅一次即可：`sub_l2_from_tencent` 在进入循环前
    订阅一次（`subscribe=True`），循环里只读（`subscribe=False`）——
    否则每轮都重订阅，而订阅后立刻读**当轮仍拿不到数据**（第一轮必空）。

    :param xt_codes: QMT 形式代码（`'510300.SH'`），用 :func:`_l2_qmt_xt_codes` 转
    """
    import xtquant.xtdata as xtdata
    from .datasource.qmt_source import GQ_qmt_to_qa_code

    if subscribe:
        for xc in xt_codes:
            try:
                xtdata.subscribe_quote(xc, period='tick')
            except Exception:
                pass
    ticks = xtdata.get_full_tick(xt_codes) or {}

    rows = []
    for xc, t in ticks.items():
        if not t:
            continue
        bid = list(t.get('bidPrice') or [])
        ask = list(t.get('askPrice') or [])
        bvol = list(t.get('bidVol') or [])
        avol = list(t.get('askVol') or [])
        vals = {'price': t.get('lastPrice'), 'volume': t.get('volume'),
                'amount': t.get('amount')}
        # ⚠️ 本机 QMT 客户端**不提供盘口深度**：`get_full_tick` 的 bid/ask 恒为 0，
        # `get_fullspeed_orderbook` 直接报「当前客户端未支持此功能，请更新客户端或
        # 升级投研版」，`get_l2_quote` 返回空（2026-09-21 实测，ETF 与股票都一样；
        # 只有价格是活的）。**全 0 的深度一律不写** —— 写进去会被读成「盘口是空的」，
        # 而真相是「这个源给不了」，两者必须能区分。
        if any(bid) or any(ask) or any(bvol) or any(avol):
            for i in range(5):
                vals['bid%d' % (i + 1)] = bid[i] if i < len(bid) else None
                vals['bid%d_volume' % (i + 1)] = bvol[i] if i < len(bvol) else None
                vals['ask%d' % (i + 1)] = ask[i] if i < len(ask) else None
                vals['ask%d_volume' % (i + 1)] = avol[i] if i < len(avol) else None
        else:
            vals['depth'] = 'unavailable'
        # QMT 的 `time` 是**绝对时间**（epoch 毫秒，UTC 基线），不是北京时间裸值
        # —— 所以**不能**过 `bj_date`（那会把它当北京时再 +8 小时）。
        # 用 `pd.Timestamp` 而不是 `datetime`：本模块 `dt` 已经被
        # `from datetime import datetime as dt` 占成**类**了，`dt.datetime` 不存在。
        ms = t.get('time')
        ts = (pd.Timestamp(int(ms), unit='ms', tz='UTC').to_pydatetime()
              if ms else None)
        rows.append(_l2_row(normalize_code(GQ_qmt_to_qa_code(xc)), ts,
                            REALTIME_SOURCE_QMT, **vals))
    return rows


def sub_l2_from_tencent(database_realtime=None, sleep_time: float = 3.0,
                        etf_codelist=None, verbose: bool = True):
    """L2（五档盘口）订阅：全市场走腾讯；ETF 那条留给 MiniQMT（**已停服**）。

    为什么 3 秒
    ==========
    新浪那条 L2 已加 **30 秒**请求限制，无法连续取 —— 这就是本订阅器存在的理由。
    腾讯这条可以 3 秒一轮（L1 订阅器跑的是 2 秒），故默认 `sleep_time=3.0`。

    落库
    ====
    与 L1 一样写 8.3 的 `golemq_stock_cn_realtime.realtime_YYYY-MM-DD`
    （**按日**的**时间序列**集合），且**两条流共用同一个日集合**，靠行内
    `source` 区分：``tencent_l1`` / ``tencent_l2`` / ``qmt``。

    ⚠️ 腾讯 L2 存的字段是 L1 行的**真子集**（`_l2_rows_from_tencent` 只取价格、
    成交量与五档，而 L1 存的是整份快照、**本来就含五档**）—— 这份重复是
    **刻意保留**的：它是 3 秒节奏的独立流。**不要**为了"省磁盘"把它删掉。
    两条流互不覆盖，靠的是 :func:`_write_ts_rows` 的删除条件带 `source`。

    ⚠️ 时间序列**不支持唯一索引、也不能 upsert**（实测），写入见 :func:`_write_ts_rows`。

    两个源的分工
    ============
    * **腾讯**：全市场约 4,900 只，一次 `market_snapshot` 拿全部五档。
    * **MiniQMT**：仅 ETF（`etf_codelist`，默认取 `etf_list`）。
      ⚠️ **自 2026-10-01 起停服**（监管），目前**取不到任何数据** —— 由
      :data:`QMT_REALTIME_ENABLED` 关掉取数与订阅，等替代源（cfquant）落地后再恢复。
      恢复时注意：必须先 `subscribe_quote(period='tick')`，否则拿到的是陈旧缓存。
    * 每行带 `source` 字段标明来路（它同时是去重键的一部分）。

    :param database_realtime: 目标库；默认 8.3 的 `golemq_stock_cn_realtime`
        （`DATABASE.GolemQ_StockCN_REALTIME` 在 8.3 服务器上不存在，实测 0 集合，
        故不用它 —— 见 `markets/StockCN/__init__.py` 里的同一处标注）。
    :param sleep_time: 轮询间隔（秒），默认 3.0。
    :param etf_codelist: ETF 代码列表；None = 取 `etf_list` 全量（QMT 停服期忽略）。
    :param verbose: 打印每轮摘要。
    """
    urllib3.util.connection.DEFAULT_MAX_POOL_SIZE = 100

    if database_realtime is None:
        database_realtime = _realtime_db()
    if easyquotation_not_install is True:
        print(u'PLEASE run "pip install easyquotation" before call GolemQ.cli.sub modules')
        return

    module = HeartbeatModule(
        module_name="sub_l2_from_tencent",
        instance_id=f"sub_l2_from_tencent_{dt.now().strftime('%Y%m%d_%H%M%S')}",
        timeout_seconds=30,
    )
    if module.mutex(verbose=True):
        return False
    module.start(initial_message="L2盘口订阅模块启动")

    quotation = eq_use('tencent')

    if etf_codelist is None:
        if not QMT_REALTIME_ENABLED:
            # MiniQMT 自 2026-10-01 停服：取 etf_list 只为喂 QMT 那条，直接跳过 ——
            # 否则每轮都会去取 1674 只 ETF 名单、再因取不到数据刷错。
            # 腾讯那条本来就不需要它（它自己一次快照拿全市场）。
            print('[l2:qmt] MiniQMT 停服中（2026-10-01 起），只订阅腾讯源')
            etf_codelist = []
        else:
            # ETF 那份走 QMT：默认取 etf_list 全量。取不到就只跑腾讯那条，
            # 但要**说清**，不能静默少一半数据。
            try:
                from .refdata import GQ_fetch_etf_list
                etf_codelist = [str(c) for c in GQ_fetch_etf_list()['code'].tolist()]
            except Exception as exc:      # noqa: BLE001
                etf_codelist = []
                print(f'[l2:qmt] 取 etf_list 失败，本轮只订阅腾讯源: {exc!r}')

    # QMT 的订阅放**后台线程**：实测订阅 1,674 只 ETF 要 **58 秒**，放在主循环前
    # 会把第一轮数据推迟近一分钟，而腾讯那条 0.6 秒就拿到了。
    #
    # 后台订阅还有个好处：本条路径对本机**暂时拿不到深度**（见 `_l2_rows_from_qmt`
    # 的实测说明），将来客户端升级、深度可用时会自然补上，不必回头改结构。
    # 价格则**不需要订阅**就是活的（实测），所以推迟订阅不损失任何已有数据。
    #
    # `QMT_REALTIME_ENABLED` 为假时这里恒为空列表，下面三处调用点
    # （后台订阅线程、循环里的取数）自然短路 —— 结构不动，只是不发请求。
    xt_etf_codes = (_l2_qmt_xt_codes(etf_codelist)
                    if (etf_codelist and QMT_REALTIME_ENABLED) else [])
    if xt_etf_codes:
        def _subscribe_etf_bg(codes):
            try:
                _l2_rows_from_qmt(codes, subscribe=True)
                print(f'[l2:qmt] 后台订阅完成 {len(codes)} 只 ETF')
            except Exception as exc:  # noqa: BLE001
                print(f'[l2:qmt] 后台订阅失败: {exc!r}')
        threading.Thread(target=_subscribe_etf_bg, args=(xt_etf_codes,),
                         daemon=True).start()

    collection_day = dt.now().date()
    collection = realtime_ts_collection(
        database_realtime, realtime_collection_name(collection_day))
    last_ts: dict = {}
    get_once = True
    sync_count = 0
    total = 0

    while (GQ_util_if_tradetime(dt.now())) or get_once:
        _time = dt.now()

        # 跨日切换（与 `sub_l1_from_tencent` 同一写法）
        if dt.now().date() != collection_day:
            collection_day = dt.now().date()
            collection = realtime_ts_collection(
                database_realtime, realtime_collection_name(collection_day))

        if (sync_count % 8 == 1):
            module.checkin(message=f"L2盘口 已写 {total} 行")

        rows = []
        try:
            rows.extend(_l2_rows_from_tencent(quotation))
        except Exception as exc:      # noqa: BLE001 单源失败不拖垮另一源
            print(f'[l2:tencent] 本轮失败: {exc!r}')

        if xt_etf_codes:
            try:
                rows.extend(_l2_rows_from_qmt(xt_etf_codes, subscribe=False))
            except Exception as exc:  # noqa: BLE001
                print(f'[l2:qmt] 本轮失败: {exc!r}')

        if rows:
            try:
                # 一轮里可能混着两个源的行（腾讯盘口 + QMT），而 `_write_ts_rows`
                # 的删除条件带 `source` —— 必须**按 source 分组**分别调用，
                # 否则一组会把另一组刚写的同 (code, ts) 行删掉。
                groups: dict = {}
                for r in rows:
                    groups.setdefault(r.get('source'), []).append(r)
                n = sum(_write_ts_rows(collection, group, last_ts, src)
                        for src, group in groups.items())
                total += n
                if verbose:
                    print(f'[l2] {_time.strftime("%H:%M:%S")} 写 {n}/{len(rows)} 行 '
                          f'(累计 {total})')
            except Exception as exc:  # noqa: BLE001
                print(f'[l2] 写入失败: {exc!r}')

        sync_count += 1
        timer.sleep(sleep_time)

    module.complete(completion_message="L2盘口订阅模块正常结束")


def GQ_fetch_stock_realtime_adv(
    code=None,
    num=1,
    collections=None,
    verbose=True,
    suffix=False,
    day=None,
    source=REALTIME_SOURCE_TENCENT_L1,
):
    '''返回当日的上下五档, code可以是股票可以是list, num是每个股票获取的数量

    :param code: 6 位或带后缀的代码，str 或 list
    :param num: **每个标的**要的行数 —— ⚠️ 见下面「num 的真实语义」
    :param collections: 已建好的集合句柄；None = 按 `day` 取 8.3 的
        ``golemq_stock_cn_realtime.realtime_YYYY-MM-DD``
    :param suffix: 股票代码是否带沪深交易所后缀
    :param day: 目标交易日；None = 今天
    :param source: 只取该条数据流的行（默认 L1）。传 ``None`` 表示不过滤 ——
        ⚠️ 那时同一 ``(code, ts)`` 会有 L1 与 L2 两行，下游的
        ``drop_duplicates(['datetime','code'])`` 会**任意**留一行
    :return: DataFrame，``set_index(['datetime','code'])``（无 `_id` 列）

    num 的真实语义
    =============
    单条查询 + ``limit(num * len(code))`` 的实际语义是「**跨全部 code 的**最近
    ``num*len(code)`` 行」，不是"每个 code 各 num 行"（冷门标的会被热门挤掉）。
    三个真实调用点（`GQ_fetch_stock_day_realtime_adv` / `..._min_realtime_adv`）
    都传 ``num=8000``，而单标的单日 tick 数 < 8000，所以 limit **从不截断** ——
    实际拿到的是"当天全部行"，正是重采样需要的。将来若真需要 per-code N 行，
    得改成每 code 一次查询或 ``$group`` + ``$top``。
    '''
    if collections is None:
        collections = realtime_ts_collection(
            _realtime_db(), realtime_collection_name(day))

    if code is not None:
        # code 必须转换成list 去查询数据库，因为五档数据用一个collection保存了股票，指数及基金，所以强制必须使用标准化代码
        if isinstance(code, str):
            code = [normalize_code(code)]
        elif isinstance(code, list):
            code = [normalize_code(symbol) for symbol in code]
            pass
        else:
            print("QA Error GQ_fetch_stock_realtime_adv parameter code is not List type or String type")
        if (verbose):
            print(
                "GQ_fetch_stock_realtime_adv Ckpo 1",
                code)
        # 用 `ts`（timeField）查与排序 —— 它是**分桶剪枝的唯一开关**。
        # `datetime` 是北京时**字符串**，字典序恰好等于时间序，但规划器不认它：
        # 拿它当条件或排序键，查询会退化成把当天的桶整个翻一遍。
        # 区间即便集合本身就是一天也要显式给（剪枝只看条件，不看集合名）。
        day_str = pd.Timestamp(
            day if day is not None else dt.today()).strftime('%Y-%m-%d')
        query = {
            'code': {'$in': code},
            'ts': {'$gte': bj_date('{} 00:00:00'.format(day_str)),
                   '$lte': bj_date('{} 23:59:59'.format(day_str))},
        }
        if source is not None:
            query['source'] = source
        items_from_collections = [
            item for item in collections.find(query,
                limit=num * len(code),
                sort=[('ts',
                       pymongo.DESCENDING)])
        ]
        if (items_from_collections is None) or \
            (len(items_from_collections) == 0):
            if verbose:
                print("GolemQ Error GQ_fetch_stock_realtime_adv find parameter code={} num={} collection={} return None".format(
                    code,
                    num,
                    collections))
            return None
        data = pd.DataFrame(items_from_collections)
        if (suffix is False):
            # 返回代码数据中是否包含交易所代码
            data['code'] = data.apply(lambda x: x.at['code'][:6], axis=1)
        data_set_index = data.set_index(['datetime',
                                         'code'],
                                        drop=False).drop(['_id'],
                                                         axis=1)
        return data_set_index
    else:
        print("QA Error GQ_fetch_stock_realtime_adv parameter code is None")


def GQ_data_tick_resample_1min(tick, type_='1min', if_drop=True, stack_vol=True):
    """
    tick 采样为 分钟数据
    1. 仅使用将 tick 采样为 1 分钟数据
    2. 仅测试过，与通达信 1 分钟数据达成一致
    3. 经测试，可以匹配 QUANTAXIS 的 ``QA_fetch_get_stock_transaction`` 得到的
       数据，其他类型数据未测试。（那个函数读的是秒级成交明细；本树没有对应
       数据源，所以下面这段 demo 现在只能作形状参考，**不能直接运行**。）
    demo:
    df = <秒级成交明细帧，列含 price / vol / date>
    df_min = GQ_data_tick_resample_1min(df)
    """
    tick = tick.assign(amount=tick.price * tick.vol)
    resx = pd.DataFrame()
    _dates = set(tick.date)

    for date in sorted(list(_dates)):
        _data = tick.loc[tick.date == date]
        # morning min bar
        if (stack_vol):
            _data1 = _data[time(9,
                                25):time(11,
                                         30)].resample(type_,
                                             closed='left',
                                             offset="30min",).apply({
                                                 'price': 'ohlc',
                                                 'vol': 'last',
                                                 'code': 'last',
                                                 'amount': 'last'
                                             })
            _data1.index = _data1.index + to_offset(type_)
        else:
            _data1 = _data[time(9,
                                25):time(11,
                                         30)].resample(type_,
                                             closed='left',
                                             offset="30min",).apply({
                                                 'price': 'ohlc',
                                                 'vol': 'last',
                                                 'code': 'last',
                                                 'amount': 'last'
                                             })
            # print( _data1.index)
            _data1.index = _data1.index + to_offset(type_)
        _data1.columns = _data1.columns.droplevel(0)
        # do fix on the first and last bar
        # 某些股票某些日期没有集合竞价信息，譬如 002468 在 2017 年 6 月 5 日的数据
        if len(_data.loc[time(9, 25):time(9, 25)]) > 0:
            _data1.loc[time(9,
                            31):time(9,
                                     31),
                       'open'] = _data1.loc[time(9,
                                                 26):time(9,
                                                          26),
                                            'open'].values
            _data1.loc[time(9,
                            31):time(9,
                                     31),
                       'high'] = _data1.loc[time(9,
                                                 26):time(9,
                                                          31),
                                            'high'].max()
            _data1.loc[time(9,
                            31):time(9,
                                     31),
                       'low'] = _data1.loc[time(9,
                                                26):time(9,
                                                         31),
                                           'low'].min()
            _data1.loc[time(9,
                            31):time(9,
                                     31),
                       'vol'] = _data1.loc[time(9,
                                                26):time(9,
                                                         31),
                                           'vol'].sum()
            _data1.loc[time(9,
                            31):time(9,
                                     31),
                       'amount'] = _data1.loc[time(9,
                                                   26):time(9,
                                                            31),
                                              'amount'].sum()
        ## 通达信分笔数据有的有 11:30 数据，有的没有

        _data1 = _data1.loc[time(9, 31):time(11, 30)]

        # afternoon min bar
        if (stack_vol):
            _data2 = _data[time(13,
                                0):time(15,
                                        0)].resample(type_,
                                             closed='left',
                                             offset="30min",).apply({
                                                 'price': 'ohlc',
                                                 'vol': 'last',
                                                 'code': 'last',
                                                 'amount': 'last',
                                             })
            _data1.index = _data1.index + to_offset(type_)
        else:
            # 新浪l1快照数据不需要累加成交量 -- 阿财 2020/12/29
            _data2 = _data[time(13,
                                0):time(15,
                                        0)].resample(type_,
                                             closed='left',
                                             offset="30min",).apply({
                                                 'price': 'ohlc',
                                                 'vol': 'last',
                                                 'code': 'last',
                                                 'amount': 'last',
                                             })
            _data1.index = _data1.index + to_offset(type_)

        _data2.columns = _data2.columns.droplevel(0)
        # 沪市股票在 2018-08-20 起，尾盘 3 分钟集合竞价
        if (pd.Timestamp(date) < pd.Timestamp('2018-08-20')) and (tick.code.iloc[0][0] == '6'):
            # 避免出现 tick 数据没有 1:00 的值
            if len(_data.loc[time(13, 0):time(13, 0)]) > 0:
                _data2.loc[time(15,
                                0):time(15,
                                        0),
                           'high'] = _data2.loc[time(15,
                                                     0):time(15,
                                                             1),
                                                'high'].max()
                _data2.loc[time(15,
                                0):time(15,
                                        0),
                           'low'] = _data2.loc[time(15,
                                                    0):time(15,
                                                            1),
                                               'low'].min()
                _data2.loc[time(15,
                                0):time(15,
                                        0),
                           'close'] = _data2.loc[time(15,
                                                      1):time(15,
                                                              1),
                                                 'close'].values
        else:
            # 避免出现 tick 数据没有 15:00 的值
            if len(_data.loc[time(13, 0):time(13, 0)]) > 0:
                if (len(_data2.loc[time(15, 1):time(15, 1)]) > 0):
                    _data2.loc[time(15,
                                    0):time(15,
                                            0)] = _data2.loc[time(15,
                                                                  1):time(15,
                                                                          1)].values
                else:
                    # 这种情况下每天下午收盘后15:00已经具有tick值，不需要另行额外填充
                    #  -- 阿财 2020/05/27
                    pass
        _data2 = _data2.loc[time(13, 1):time(15, 0)]
        resx = pd.concat(
            [resx, _data1, _data2],
            axis=0,
            sort=True).sort_index()
    resx['vol'] = resx['vol']
    resx['volume'] = resx['vol']
    resx['type'] = '1min'
    if if_drop:
        resx = resx.dropna()
    return resx.reset_index().drop_duplicates().set_index(['datetime', 'code'])


def GQ_fetch_stock_day_realtime_adv(
    codelist,
    data_day,
    market_type: str = MARKET_TYPE.STOCK_CN,
    verbose: bool = True
):
    """
    查询日线实盘数据，支持多股查询
    """
    if codelist is not None:
        # codelist 必须转换成list 去查询数据库
        if isinstance(codelist, str):
            codelist = [codelist]
        elif isinstance(codelist, list):
            pass
        else:
            print("QA Error GQ_fetch_stock_day_realtime_adv parameter codelist is not List type or String type")
    start_time = dt.strptime(str(dt.now().date()) + ' 09:15', '%Y-%m-%d %H:%M')    
    offset_time = pd.to_datetime(GQ_util_get_last_day())-data_day.data.index.get_level_values(level=0)[-1].to_pydatetime()
    if (len(data_day.data.index.get_level_values(level=0)) == 0):
        print(u'K线数据长度为零：', codelist)
    elif ((dt.now() > start_time) and (offset_time > timedelta(hours=10))) or \
        ((dt.now() < start_time) and (offset_time > timedelta(hours=40))):
        log_msg = u'时间戳差距超过：{} 尝试查找日线实盘数据....{}'.format(
            dt.now() - data_day.data.index.get_level_values(level=0)[-1].to_pydatetime(),
            codelist)
        if (isinstance(verbose, tqdm)):
            verbose.write(log_msg)
        elif (verbose is True):
            print(log_msg)

        try:
            # 取「上一个交易日（含今天，9:30 后算今天）」的**当日集合**。
            # 老写法是 `QAREALTIME` + `'realtime_{}'.format(dt.today())`，两处都错：
            # ① 读的是 4.4 老库（写入端早在 8.3）；② `dt` 在本模块是 `datetime`
            # **类**，`dt.today()` 会拼出 `'realtime_2026-10-08 01:00:27.222907'`
            # （带时分秒），那个集合**不可能存在** —— 所以这条路从来没读到过东西。
            # 另外 `dt.today() - timedelta(hours=24)` 给的是「昨天同一时刻」，
            # 周一凌晨会得到周日、节假日同样错；`GQ_util_get_last_day()` 才认交易日。
            data_realtime = GQ_fetch_stock_realtime_adv(
                codelist, num=8000,
                verbose=verbose,
                suffix=False,
                day=GQ_util_get_last_day())
        except Exception:
            # 兜底的 `GQ_data_tick_resample_1min(codelist, ...)` 实参形态是错的
            # （那个函数要的是 tick 帧、不是代码列表，必然 AttributeError 且异常
            # 会从这个 except 块里逃出去）。与 `GQ_fetch_stock_min_realtime_adv`
            # 的那处先例保持一致：置 None，由下面 `is not None` 的守卫跳过。
            data_realtime = None
        if (data_realtime is not None) and \
            (len(data_realtime) > 0):
            # 合并实盘实时数据
            data_realtime = data_realtime.drop_duplicates(([
                "datetime",
                'code'])).set_index([
                    "datetime",
                    'code'],
                    drop=False)
            data_realtime = data_realtime.reset_index(level=[1], drop=True)
            data_realtime['date'] = pd.to_datetime(data_realtime['datetime']).dt.strftime('%Y-%m-%d')
            data_realtime['datetime'] = pd.to_datetime(data_realtime['datetime'])
            for code in codelist:
                # 顺便检查股票行情长度，发现低于30天直接砍掉。
                try:
                    if (len(data_day.select_code(code[:6])) < 30):
                        print(u'{} 行情只有{}天数据，新股或者数据不足，不进行择时分析。'.format(code, 
                                                                   len(data_day.select_code(code[:6]))))
                        data_day.data.drop(data_day.select_code(code).data, 
                                           inplace=True)
                        continue
                except Exception:
                    pass
                    continue

                # *** 注意，QA_data_tick_resample_1min 函数不支持多标的 *** 需要循环处理
                data_realtime_code = data_realtime[data_realtime['code'].eq(code[:6])]
                if (len(data_realtime_code) > 0):
                    data_realtime_code = data_realtime_code.set_index(['datetime']).sort_index()
                    if ('volume' in data_realtime_code.columns) and \
                        ('vol' not in data_realtime_code.columns):
                        # 我也不知道为什么要这样转来转去，但是各家(新浪，pytdx)l1数据就是那么不统一
                        data_realtime_code.rename(
                            columns={"volume": "vol"},
                            inplace=True)
                    elif ('volume' in data_realtime_code.columns):
                        data_realtime_code['vol'] = np.where(np.isnan(data_realtime_code['vol']), 
                                                             data_realtime_code['volume'], 
                                                             data_realtime_code['vol'])

                    # 一分钟数据转出来了
                    try:
                        data_realtime_1min = GQ_data_tick_resample_1min(
                            data_realtime_code, 
                            type_='1min',
                            stack_vol=False)
                        # data_realtime_1min['vol']
                        if (market_type == MARKET_TYPE.STOCK_CN):
                            vol = data_realtime_1min.tail(1)["volume"].item()
                            stock_info = GQ_fetch_stock_info([code[:6]])
                            total_volume = stock_info['liutongguben'].iloc[0]
                            if total_volume and total_volume > 0:
                                turnover_rate = round((vol * 100 / total_volume) / 100, 6)
                                if (turnover_rate > 0.96):
                                    turnover_rate = round((vol * 100 / total_volume) / 10000, 6)
                            else:
                                turnover_rate = None
                    except Exception:
                        if (len(data_realtime_code) > 3.82):
                            print('fooo1 GQ_fetch_stock_day_realtime_adv', code)
                            print(data_realtime_code)
                            traceback.print_exc()
                            # raise ('foooo1{}'.format(code))
                    data_realtime_1day = GQ_data_min_to_day(data_realtime_1min)
                    data_realtime_1day = data_realtime_1day.rename_axis('date')
                    if (len(data_realtime_1day) > 0):
                        # 转成日线数据
                        data_realtime_1day.rename(
                            columns={"vol": "volume"},
                            inplace=True)

                        # 假装复了权，我建议复权那几天直接量化处理，复权几天内对策略买卖点影响很大
                        data_realtime_1day['adj'] = 1.0
                        if (market_type == MARKET_TYPE.STOCK_CN):
                            try:
                                data_realtime_1day[FLD.TURNOVER_RATE] = turnover_rate
                            except Exception:
                                print(turnover_rate)
                                traceback.print_exc()
                        data_realtime_1day['date'] = pd.to_datetime(data_realtime_1day.index)
                        data_realtime_1day = data_realtime_1day.set_index(
                            ['date', 'code'], drop=True).sort_index()

                        # 当早盘集合竞价未出现成交，9:30分的Open和Low报价会是0元，特别处理
                        pre_close = data_day.data[AKA.CLOSE].tail(1).item()
                        if (data_realtime_1day[AKA.OPEN].head(1).item() < 0.001):
                            data_realtime_1day.loc[data_realtime_1day.index.get_level_values(level=0)[0],
                                                   AKA.OPEN] = pre_close
                            data_realtime_1day.loc[data_realtime_1day.index.get_level_values(level=0)[0],
                                                   AKA.LOW] = min(data_realtime_1day[AKA.OPEN].head(1).item(), 
                                                                  data_realtime_1day[AKA.HIGH].head(1).item(), 
                                                                  data_realtime_1day[AKA.CLOSE].head(1).item())

                        if (data_day.data.index.get_level_values(level=0)[-1] != data_realtime_1day.index.get_level_values(level=0)[-1]):
                            # 成功获取到 l1 实盘数据，获取主力资金流向
                            if (data_realtime_1day.index.get_level_values(level=0)[-1] > dt.now()):
                                print(u'尝试追加资金流向数据，股票代码：{} 时间：{} 价格：{}'.format(
                                    data_realtime_1day.index[0][1],
                                    data_realtime_1day.index[-1][0],
                                    data_realtime_1day[AKA.CLOSE].iloc[-1]))

                            log_msg = u'追加实时实盘数据 {}，股票代码：{} 时间：{} 价格：{}'.format(
                                len(data_realtime_1day), 
                                data_realtime_1day.index[0][1],
                                data_realtime_1day.index[-1][0],
                                data_realtime_1day[AKA.CLOSE].iloc[-1])
                            if (isinstance(verbose, tqdm)):
                                verbose.write(log_msg)
                            elif (verbose is True):
                                print(log_msg)

                            data_day.data = pd.concat(
                                [data_day.data,
                                 data_realtime_1day],
                                axis=0,
                                sort=True).sort_index()

    return data_day


def GQ_fetch_stock_min_realtime_adv(
    codelist,
    data_min,
    frequency, 
    verbose=False
):
    """
    查询A股的指定小时/分钟线线实盘数据
    """

    if codelist is not None:
        # codelist 必须转换成list 去查询数据库
        if isinstance(codelist, str):
            codelist = [codelist]
        elif isinstance(codelist, list):
            pass
        else:
            if verbose:
                print("QA Error GQ_fetch_stock_min_realtime_adv parameter codelist is not List type or String type")

    if data_min is None:
        if verbose:
            print(u'代码：{} 今天停牌或者已经退市*'.format(codelist))  
        return None

    try:
        foo = (dt.now() - data_min.data.index.get_level_values(level=0)[-1].to_pydatetime())
    except Exception:
        log_msg = u'代码：{} 今天停牌或者已经退市**'.format(codelist)
        if (isinstance(verbose, tqdm)):
            verbose.write(log_msg)
        elif (verbose):
            print(log_msg)
        return None
    
    if (verbose):
        print(
            f'Code: {codelist} 行 {len(data_min.data)}........... Ckpo 3.1.1\n',
            data_min.data.query("volume < 1").tail(20))
    
    start_time = dt.strptime(str(dt.now().date()) + ' 09:15', '%Y-%m-%d %H:%M')
    offset_time = pd.to_datetime(GQ_util_get_last_day())-data_min.data.index.get_level_values(level=0)[-1].to_pydatetime()
    if ((dt.now() > start_time) and (offset_time > timedelta(hours=10))) or \
        ((dt.now() < start_time) and (offset_time > timedelta(hours=24))) or \
        (((dt.now() - data_min.data.index.get_level_values(level=0)[-1].to_pydatetime()) > timedelta(hours=18.25)) and \
        ((dt.now() - pd.to_datetime(GQ_util_get_last_day()))<timedelta(hours=30))):
        log_msg = u'时间戳差距超过：{} 尝试查找分钟线实盘数据.... {}'.format(dt.now() - data_min.data.index.get_level_values(level=0)[-1].to_pydatetime(),
                    codelist)
        if (isinstance(verbose, tqdm)):
            verbose.write(log_msg)
        elif (verbose):
            print(log_msg)

        # 上一个交易日（含今天，9:30 后算今天）的当日集合 —— 理由见
        # `GQ_fetch_stock_day_realtime_adv` 里的同一处说明（旧写法既读错库、
        # 又把时分秒拼进了集合名，所以从来没读到过东西）。
        day = GQ_util_get_last_day()
        for code in codelist:
            # print(u'查询实盘数据。', code, )
            try:
                data_realtime = GQ_fetch_stock_realtime_adv(
                    code, num=8000,
                    verbose=verbose, suffix=False,
                    day=day)
            except Exception:
                # 原先这里兜底调 QUANTAXIS 的 QA_fetch_stock_realtime_adv。
                # 已移除 —— 且有实测依据：那个函数读 QUANTAXIS 包内 `DATABASE`
                # 的 realtime_* 集合，而它解析到 `quantaxis` 库，**该库里
                # realtime_* 集合数为 0**（2026-09-21 实测）。也就是说这个兜底
                # **只能返回 None**，移除它对任何原本能工作的情形都没有行为影响。
                data_realtime = None

            if (data_realtime is not None) and \
                (len(data_realtime) > 0):
                # 合并实盘实时数据
                data_realtime = data_realtime.drop_duplicates(([
                    "datetime",
                    'code'])).set_index([
                        "datetime",
                        'code'],
                        drop=False)

                data_realtime = data_realtime.reset_index(level=[1], drop=True)
                data_realtime['date'] = pd.to_datetime(data_realtime['datetime']).dt.strftime('%Y-%m-%d')
                data_realtime['datetime'] = pd.to_datetime(data_realtime['datetime'])

                # *** 注意，QA_data_tick_resample_1min 函数不支持多标的 *** 需要循环处理
                # 可能出现8位六位股票代码兼容问题
                data_realtime_code = data_realtime[data_realtime['code'].eq(code[:6])]

                if (len(data_realtime_code) > 0):
                    data_realtime_code = data_realtime_code.set_index(['datetime']).sort_index()
                    if ('volume' in data_realtime_code.columns) and \
                        ('vol' not in data_realtime_code.columns):
                        # 我也不知道为什么要这样转来转去，但是各家(新浪，pytdx)l1数据就是那么不统一
                        data_realtime_code.rename(
                            columns={"volume": "vol"},
                            inplace=True)
                    elif ('volume' in data_realtime_code.columns):
                        data_realtime_code['vol'] = np.where(np.isnan(data_realtime_code['vol']), 
                                                             data_realtime_code['volume'], 
                                                             data_realtime_code['vol'])

                    # 将l1 Tick数据重采样为1分钟
                    try:
                        data_realtime_1min = GQ_data_tick_resample_1min(
                            data_realtime_code, 
                            type_='1min',
                            stack_vol=False)
                    except Exception:
                        if verbose:
                            print('fooo1 GQ_fetch_stock_min_realtime_adv', code)
                            print(data_realtime_code)
                        pass
                        # raise('foooo1{}'.format(code))

                    if (len(data_realtime_1min) == 0):
                        pass
                        return data_min

                    # 一分钟数据转出来了，重采样为指定小时/分钟线数据
                    data_realtime_1min = data_realtime_1min.reset_index([1], drop=False)
                    data_realtime_mins = GQ_data_min_resample(data_realtime_1min, 
                                                                 type_=frequency)

                    if (len(data_realtime_mins) > 0):
                        # 转成指定分钟线数据
                        data_realtime_mins.rename(
                            columns={"vol": "volume"},
                            inplace=True)

                        # 假装复了权，我建议复权那几天直接量化处理，复权几天内对策略买卖点影响很大
                        data_realtime_mins['adj'] = 1.0

                        # 当早盘集合竞价未出现成交，9:30分的Open和Low报价会是0元，特别处理
                        pre_close = data_min.data[AKA.CLOSE].tail(1).item()
                        if (data_realtime_mins[AKA.OPEN].head(1).item() < 0.001):
                            data_realtime_mins.loc[data_realtime_mins.index.get_level_values(level=0)[0],
                                                   AKA.OPEN] = pre_close
                            data_realtime_mins.loc[data_realtime_mins.index.get_level_values(level=0)[0],
                                                   AKA.LOW] = min(data_realtime_1min[AKA.LOW][1:].min(), 
                                                                  data_realtime_1min[AKA.HIGH][1:].min(), 
                                                                  data_realtime_1min[AKA.CLOSE][1:].min(),)

                        if (data_min.select_code(code[:6]).index.get_level_values(level=0)[-1] != data_realtime_mins.index.get_level_values(level=0)[-1]):
                            log_msg = u'追加实时实盘数据 {}，股票代码：{}({}) 开盘 {}：价格：{}'.format(
                                len(data_realtime_mins),
                                code, data_realtime_mins.index[0][1],
                                data_realtime_mins.index[-1][0],
                                data_realtime_mins[AKA.OPEN].iloc[0],
                                data_realtime_mins[AKA.CLOSE].iloc[-1])
                            if (isinstance(verbose, tqdm)):
                                verbose.write(log_msg)
                            elif (verbose):
                                print(log_msg)
                                
                            if (verbose):
                                print(
                                    f'Code: {codelist} 行 {len(data_min.data)}........... Ckpo 3.1.5\n', 
                                    data_min.data.query("volume < 1").tail(20))
                                
                            data_min.data = pd.concat([data_min.data, data_realtime_mins], axis=0, sort=True)

                            # 根据索引去重，只保留每个索引的第一个条目
                            data_min.data = data_min.data[~data_min.data.index.duplicated(keep='first')]

                        # Amount, Volume 计算不对
        if (verbose):
            print(
                f'Code: {codelist} 行 {len(data_min.data)}........... Ckpo 3.1.6\n', 
                data_min.data.query("volume < 1").tail(20))
    else:
        log_msg = u'没有时间差{}'.format(dt.now() - data_min.data.index.get_level_values(level=0)[-1].to_pydatetime())
        if (isinstance(verbose, tqdm)):
            verbose.write(log_msg)
        elif (verbose is True):
            print(log_msg)
            print((dt.now() - data_min.data.index.get_level_values(level=0)[-1].to_pydatetime()),
                  (dt.now() - pd.to_datetime(GQ_util_get_last_day())))
            
    if (verbose):
        print(
            f'Code: {codelist} 行 {len(data_min.data)}........... Ckpo 3.1.3\n', 
            data_min.data.query("volume < 1").tail(20))
    data_min.data = data_min.data.sort_index()
    if (verbose):
        print(
            f'Code: {codelist} 行 {len(data_min.data)}........... Ckpo 3.1.4\n', 
            data_min.data.query("volume < 1").tail(20))
    
    return data_min


def GQ_fetch_index_min_realtime_adv(codelist,
                                    data_min,
                                    frequency, 
                                    verbose=True):
    """
    查询指数和ETF的分钟线实盘数据
    """
    # 将l1 Tick数据重采样为1分钟
    data_realtime_1min = data_realtime_1min.reset_index(level=[1], drop=False)

    # 检查 1min数据是否完整，如果不完整，需要从腾讯财经获取1min K线
    #if ():


    data_realtime_5min = GQ_data_min_resample(data_realtime_1min, 
                                                 type_='5min')
    print(data_realtime_5min)

    data_realtime_15min = GQ_data_min_resample(data_realtime_1min, 
                                                  type_='15min')
    print(data_realtime_15min)

    data_realtime_30min = GQ_data_min_resample(data_realtime_1min, 
                                                  type_='30min')
    print(data_realtime_30min)
    data_realtime_1hour = GQ_data_min_resample(data_realtime_1min,
                                                 type_='60min')
    print(data_realtime_1hour)
    return data_min


def stock_individual_fund_flow_push(stock: str = "600094",
                                    market: str = "sh",) -> pd.DataFrame:
    """
    东方财富网-数据中心-实时资金流向
    http://data.eastmoney.com/zjlx/detail.html
    :param stock: 股票代码
    :type stock: str
    :param market: 股票市场; 上海证券交易所: sh, 深证证券交易所: sz
    :type market: str
    :return: 今天个股的资金流数据
    :rtype: pandas.DataFrame
    """
    market_map = {"sh": 1, "sz": 0}
    url = "http://push2.eastmoney.com/api/qt/ulist.np/get"
    try:
        UTC_dummy = datetime.UTC
    except Exception:
        UTC_dummy = timezone.utc
    params = {
        "fltt": "2",
        "secids": f"{market_map[market]}.{stock}",
        "fields": "f62,f184,f66,f69,f72,f75,f78,f81,f84,f87,f64,f65,f70,f71,f76,f77,f82,f83,f164,f166,f168,f170,f172,f252,f253,f254,f255,f256,f124,f6,f278,f279,f280,f281,f282",
        "cb": "jQuery112304684874904250048_{:d}".format(int(dt.now(UTC_dummy).timestamp())),
        "ut": "b2884a393a59ad64002292a3e90d46a5",
        "_": int(ts.time() * 1000),
    }
    r = requests.get(url, params=params)
    text_data = r.text
    json_data = json.loads(text_data[text_data.find("{") : -2])

    try:
        temp_df = pd.DataFrame(json_data["data"]["diff"])
    except Exception:
        # ETF基金可能已经退市

        return None

    temp_df.columns = ["f6",
        u"主力净流入-净额",      # f62
        u"超大单流入",         # f64
        u"超大单流出",         # f65
        u"超大单净流入-净额",     # f66
        u"超大单净流入-净占比",   # f69
        u"大单流入",  # 'f70'
        u"大单流出",     # 'f71'
        u"大单净流入-净额",    # 'f72'
        u"大单净流入-净占比",    # 'f75'
        u"中单流入",    # 'f76'
        u"中单流出",    # 'f77': 140798103.0,
        u"中单净流入-净额",    # 'f78': 6168170.0,
        u"中单净流入-净占比",    # 'f81': 1.34,
        u"小单流入",    # 'f82': 189769712.0,
        u"小单流出",    # 'f83': 150377047.0,
        u"小单净流入-净额",    # 'f84': 39392665.0,
        u"小单净流入-净占比",    # 'f87': 8.56,
        "f124",    # 'f124': 1611811131,
        "f164",    # 'f164': 85855248.0,
        "f166",    # 'f166': 24503470.0,
        "f168",    # 'f168': 61351778.0,
        "f170",    # 'f170': -56766522.0,
        "f172",    # 'f172': -29088728.0,
        u"主力净流入-净占比",    # 'f184': -9.9,
        "f252",    # 'f252': 64801488.0,
        "f253",    # 'f253': 35072981.0,
        "f254",    # 'f254': 29728507.0,
        "f255",    # 'f255': -108915608.0,
        "f256",    # 'f256': 44114119.0,
        "f278",    # 'f278': 124515993.0,
        "f279",    # 'f279': 42538342.0,
        "f280",    # 'f280': 81977651.0,
        "f281",    # 'f281': -67973143.0,
        "f282",    # 'f282': -56542852.0,
    ]
    temp_df = temp_df[[u"主力净流入-净额",      # f62
            u"超大单流入",         # f64
            u"超大单流出",         # f65
            u"超大单净流入-净额",     # f66
            u"超大单净流入-净占比",   # f69
            u"大单流入",  # 'f70'
            u"大单流出",     # 'f71'
            u"大单净流入-净额",    # 'f72'
            u"大单净流入-净占比",    # 'f75'
            u"中单流入",    # 'f76'
            u"中单流出",    # 'f77': 140798103.0,
            u"中单净流入-净额",    # 'f78': 6168170.0,
            u"中单净流入-净占比",    # 'f81': 1.34,
            u"小单流入",    # 'f82': 189769712.0,
            u"小单流出",    # 'f83': 150377047.0,
            u"小单净流入-净额",    # 'f84': 39392665.0,
            u"小单净流入-净占比",    # 'f87': 8.56,
            u"主力净流入-净占比",]].astype(np.float64)
    return temp_df


@func_set_timeout(5)
def get_moneyflow_from_eastmoney_push(code, date_epoch=dt.now().date()) -> pd.DataFrame:
    """
    从东方财富抓取股票资金流向
    """
    if (is_stock_cn(code)[2] == 'SH'):
        stock_individual_fund_flow_df = stock_individual_fund_flow_push(stock=code, 
                                                                        market="sh")
    else:
        stock_individual_fund_flow_df = stock_individual_fund_flow_push(stock=code, 
                                                                        market="sz")

    date_epoch = dt.now().date()
    if not QA_util_if_tradetime(dt.now()):
        date_epoch = QA_util_get_pre_trade_date('{}'.format(dt.today()), n=0)

    stock_individual_fund_flow_df['code'] = code
    stock_individual_fund_flow_df['date'] = pd.to_datetime(date_epoch)
    stock_individual_fund_flow_df['date_stamp'] = pd.to_datetime(stock_individual_fund_flow_df['date']).astype(np.int64) // 10 ** 9
    # stock_individual_fund_flow_df['date_stamp'] = pd.to_datetime(stock_individual_fund_flow_df['date']).view("int64") // 10 ** 9

    stock_individual_fund_flow_df[u"主力净流入-净占比"] = stock_individual_fund_flow_df[u"主力净流入-净占比"].astype(np.float64) / 100
    stock_individual_fund_flow_df[u"小单净流入-净占比"] = stock_individual_fund_flow_df[u"小单净流入-净占比"].astype(np.float64) / 100
    stock_individual_fund_flow_df[u"中单净流入-净占比"] = stock_individual_fund_flow_df[u"中单净流入-净占比"].astype(np.float64) / 100
    stock_individual_fund_flow_df[u"大单净流入-净占比"] = stock_individual_fund_flow_df[u"大单净流入-净占比"].astype(np.float64) / 100
    stock_individual_fund_flow_df[u"超大单净流入-净占比"] = stock_individual_fund_flow_df[u"超大单净流入-净占比"].astype(np.float64) / 100
    stock_individual_fund_flow_df = stock_individual_fund_flow_df.set_index(['date', 'code'], drop=False)

    return stock_individual_fund_flow_df


if __name__ == '__main__':
    """
    用法示范
    """
    # 函数级导入：`fetch.py` 模块级导入本模块，本模块若在顶层反向导入 `fetch`
    # 就成环。这里只在 `__main__` 里用一次，放进来最省事。
    from .fetch import GQ_fetch_stock_min_adv

    codelist = ['600157', '300263']
    data_min = GQ_fetch_stock_min_adv(
        codelist,
        '2008-01-01',
        '{}'.format(dt.today(),),
        frequence='15min')

    data_min = GQ_fetch_stock_min_realtime_adv(
        codelist, data_min,
        frequency='15min')
