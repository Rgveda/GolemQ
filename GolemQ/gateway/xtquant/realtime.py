# coding:utf-8
#
# The MIT License (MIT)
#
# Copyright (c) 2020-2025 azai/GolemQ
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#


import datetime
from xtquant import xtdata
from datetime import (
    datetime as dt,
    timedelta,
)
from collections import deque
from QUANTAXIS.QAUtil.QADate_trade import (
        QA_util_if_tradetime as GQ_util_if_tradetime,
)
import pandas as pd
from pymongo import UpdateOne
import time as timer
from GolemQ.markets.StockCN import (
    formater_l1_ticks,
    collections_of_today,
)
from GolemQ.supervisor.heartbeat import HeartbeatModule
from GolemQ.agents import (
    send_serverchan_message,
)
from GolemQ.supervisor import (
    checkin_function
)


def sub_l1_from_xtquant(
    database_realtime,
    stock_list_codes: list = [],
):
    """
    从讯投获取L1数据，大约3秒钟更新一次
    """
    # 创建心跳监控实例
    module = HeartbeatModule(
        module_name="sub_l1_from_xtquant",
        instance_id=f"sub_l1_from_xtquant_{dt.now().strftime('%Y%m%d_%H%M%S')}",
        timeout_seconds=30,  # 5分钟超时
    )

    if (module.mutex(
            verbose=True)):
        return False
    else:
        # 开始模块执行记录
        module.start(
            initial_message="讯投L1数据订阅模块启动"
        )

    sleep_time = 2.0
    sleep = int(sleep_time)
    _time1 = dt.now()
    collection = collections_of_today(database=database_realtime)
    get_once = True

    # 开盘/收盘时间
    day_changed_time = dt.strptime(
        str(dt.now().date()) + ' 01:00',
        '%Y-%m-%d %H:%M')

    # 初始化一个队列，用于存储最后两次的数据
    l1_ticks_data_last = deque(maxlen=2)
    sync_count = 0
    current_date = datetime.datetime.now()
    start_time = current_date.replace(hour=9, minute=15, second=0, microsecond=0)
    end_time = current_date.replace(hour=15, minute=15, second=0, microsecond=0)

    while (start_time < datetime.datetime.now() < end_time) or get_once:
        # 开盘/收盘时间
        end_time = dt.strptime(
            str(dt.now().date()) + ' 15:15',
            '%Y-%m-%d %H:%M')
        day_changed_time = dt.strptime(
            str(dt.now().date()) + ' 01:00',
            '%Y-%m-%d %H:%M')
        _time = dt.now()

        # 心跳签到
        if (sync_count % 8 == 1):
            module.checkin(
                message=f"处理讯投L1数据，当前时间: {_time.strftime('%Y-%m-%d %H:%M:%S')}"
            )

        if GQ_util_if_tradetime(_time) and \
                (dt.now() < day_changed_time):
            # 日期变更，写入表也会相应变更，这是为了防止用户永不退出一直执行
            print(u'当前日期更新~！ {} '.format(dt.today()))
            collection = collections_of_today(database=database_realtime)
            print(u'Not Trading time 现在是中国A股收盘时间 {}'.format(_time.strftime("%Y-%m-%d %H:%M:%S")))
            timer.sleep(sleep)
            sync_count = sync_count + 1
            continue

        symbol_list = []
        l1_ticks_data = []
        if GQ_util_if_tradetime(_time) or \
                (get_once):  # 如果在交易时间
            try:
                l1_ticks = xtdata.get_full_tick(stock_list_codes)
                l1_ticks_data, symbol_list = formater_l1_ticks(l1_ticks)
            except Exception as e:
                print(f'[{dt.now().strftime("%Y-%m-%d %H:%M:%S")}] 讯投MiniQMT进程意外终止。发送消息通知 {e}')

                # 检查调用频率（15分钟限制）
                checkin_result = checkin_function(
                    function_name="send_serverchan_message(xtquant)",
                    expired_time=timedelta(minutes=15)
                )
                if checkin_result["allowed"]:
                    send_serverchan_message(
                        title='阿财的量化交易系统正在偷懒',
                        content='讯投MiniQMT进程意外终止，请检查讯投MiniQMT程序是否崩溃。如果双击讯投QMT图标启动程序依然不能解决问题，请联系IT老司机！')
                    timer.sleep(15)
                sync_count = sync_count + 1

                continue

            # 将新数据转换为 DataFrame 并设置索引
            l1_ticks_data_idx = pd.DataFrame(
                [{
                    'code': l1_tick['code'],
                    'datetime': l1_tick['datetime']
                } for l1_tick in l1_ticks_data],
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
                # 查询是否新 tick
                li_ticks_datetime = sorted(list(set([l1_tick['datetime'] for l1_tick in l1_ticks_data])))
                li_ticks_code = sorted(list(set([l1_tick['code'] for l1_tick in l1_ticks_data])))
                if (len(li_ticks_datetime) > 1):
                    query_id = {
                        '$or': [{"code": {'$in': li_ticks_code}, 'datetime': li_tick_datetime} for li_tick_datetime in li_ticks_datetime],
                    }
                else:
                    query_id = {
                        "code": {
                            '$in': list(set([l1_tick['code'] for l1_tick in l1_ticks_data]))
                        },
                        "datetime": sorted(list(set([l1_tick['datetime'] for l1_tick in l1_ticks_data])))[-1]
                    }

                # 检查是否有重复数据
                refcount = collection.count_documents(query_id)
                if (refcount > 0):
                    # 使用 bulk_write 进行批量 upsert 操作
                    bulk_operations = []
                    for l1_tick in l1_ticks_data:
                        update_query = {
                            "code": l1_tick['code'],
                            "datetime": l1_tick['datetime']
                        }
                        update_data = {
                            "$set": l1_tick
                        }
                        bulk_operations.append(
                            UpdateOne(
                                update_query,
                                update_data,
                                upsert=True))

                    if bulk_operations:
                        collection.bulk_write(bulk_operations)
                else:
                    # 新 tick，插入记录
                    collection.insert_many(l1_ticks_data)
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

    # 标记模块执行完成
    module.complete(
        completion_message="讯投L1数据订阅模块正常结束"
    )

    # While循环每天下午5点自动结束，在此等待13小时，大概早上六点结束程序自动重启
    print(u'While循环每天下午5点自动结束，在此等待半小时，大概早上六点结束程序自动重启，这样只要窗口不关，永远每天自动收取 tick')
    timer.sleep(1800)
