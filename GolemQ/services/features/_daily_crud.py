# coding:utf-8
#
# The MIT License (MIT)
#
# Copyright (c) 2018-2020 azai/Rgveda/GolemQuant
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

import numpy as np
import pandas as pd
import json
import traceback
import time
import pymongo
from pymongo import UpdateOne
from datetime import datetime as dt
from GolemQ.core.gq_logging import GQ_util_log_info
from GolemQ.core.symbol import GQ_util_code_tolist
from GolemQ.markets.StockCN.date_utils import (
    GQ_util_date_valid,
    GQ_util_date_stamp,
    GQ_util_time_stamp,
)
from GolemQ.core.settings import DATABASE as DATABASE_GolemQ
from GolemQ.core.constants import AKA
from GolemQ.core.preprocessing import GQ_util_to_json_from_pandas
from GolemQ.core import GQ_util_code_tolist
from func_timeout import func_set_timeout
def GQ_fix_daily_metadata(
    code: str,
    start: str,
    end: str,
    collections=DATABASE_GolemQ.stock_diagnosis
) -> pd.DataFrame:
    """
    获取并修正date字段错误。

    参数:
        code (str): 股票代码或代码列表
        start (str): 开始日期，格式为 'YYYY-MM-DD'
        end (str): 结束日期，格式为 'YYYY-MM-DD'
        collections: MongoDB 集合对象，默认为 DATABASE_GolemQ.stock_diagnosis

    返回:
        pd.DataFrame: 修正后的复盘数据
    """

    start = str(start)[0:10]
    end = str(end)[0:10]

    # 检查并转换代码格式
    code = GQ_util_code_tolist(code)
    if GQ_util_date_valid(end):
        cursor = collections.find({
                'code': {
                    '$in': code
                },
                "date_stamp": {
                    "$gte": GQ_util_date_stamp(start),
                    "$lte": GQ_util_date_stamp(end) if int(
                        pd.to_datetime(end).to_pydatetime().timestamp()) <= GQ_util_date_stamp(
                            end) else int(pd.to_datetime(end).to_pydatetime().timestamp()),
                }
            },
            {"_id": 0},
            batch_size=10000)
        
        res = pd.DataFrame([item for item in cursor])
        try:
            # 筛选出int64类型的时间戳
            int64_dates = res[res['date'].apply(lambda x: isinstance(x, int))]

            # 将int64类型的时间戳转换为'%Y-%m-%d'格式的字符串
            int64_dates['date'] = int64_dates['date'].apply(lambda x: pd.to_datetime(x, unit='ms').strftime('%Y-%m-%d'))

            # 检查 int64_dates['date'] / 1000 是否等于 int64_dates['date_stamp']
            mismatched_dates = int64_dates[int64_dates['date'].apply(lambda: (x/1000) != int64_dates['date_stamp'])]
            if not mismatched_dates.empty:
                print("Mismatched dates:")
                print(mismatched_dates[['date', 'date_stamp']])

            # 将更新后的数据写回MongoDB
            for index, row in int64_dates.iterrows():
                collections.update_one(
                    {'code': row['code'], 'date_stamp': row['date_stamp']},
                    {'$set': {'date': row['date']}}
                )

            # 返回修正后的数据
            return int64_dates.index

        except Exception as e:
            print(f'GQ_fix_daily_metadata: {code} Error: {e}')
            return None

    return None


def GQ_update_daily_metadata(
    features: pd.DataFrame = None,
    collections=DATABASE_GolemQ.stock_metadata_day
):
    features['code'] = features.index.get_level_values(level=1)
    features['date'] = pd.to_datetime(features.index.get_level_values(level=0).date).tz_localize('Asia/Shanghai')
    
    # GMT+0 String 转换为 UTC Timestamp
    features["date_stamp"] = pd.to_datetime(features['date']).astype(np.int64) // 10 ** 9
    features['updated_at'] = int(time.mktime(dt.now().utctimetuple()))
    features['date'] = features['date'].dt.strftime('%Y-%m-%d')
    
    data = features
    bulk_operations = []
    
    if len(data) == 0:
        print(u'GQ_update_daily_metadata() Data len equals zero.')

    for idx, each_tick in data.iterrows():
        update_query = {
            'code': each_tick['code'],
            'date_stamp': int(each_tick['date_stamp']),
        }
        update_data = {
            '$set': json.loads(each_tick.to_json(orient='index')),  # each_tick.to_dict(),  # 直接使用 to_dict() 转换为字典
        }
        bulk_operations.append(UpdateOne(update_query, update_data, upsert=True))
        if (int(each_tick['date_stamp']) < 0):
            print(update_query, update_data)
    try:
        if bulk_operations:
            #pass
            collections.bulk_write(bulk_operations)
    except:
        traceback.print_exc()
        
        
def GQ_remove_daily_metadata(
    features: pd.DataFrame = None,
    collections=DATABASE_GolemQ.stock_metadata_day
):
    features['code'] = features.index.get_level_values(level=1)
    features['date'] = pd.to_datetime(features.index.get_level_values(level=0).date).tz_localize('Asia/Shanghai')
    
    # GMT+0 String 转换为 UTC Timestamp
    features["date_stamp"] = pd.to_datetime(features['date']).astype(np.int64) // 10 ** 9
    features['date'] = features['date'].dt.strftime('%Y-%m-%d')
    
    data = features
    bulk_operations = []
    
    if len(data) == 0:
        print(u'GQ_remove_daily_metadata() Data len equals zero.')

    for idx, each_tick in data.iterrows():
        update_query = {
            'code': each_tick['code'],
            'date_stamp': int(each_tick['date_stamp']),
        }
        
        # 生成 remove_data 字典
        remove_data = {col: 1 for col in features.columns if col not in ['code', 'date', 'date_stamp']}
        # print(remove_data, features.columns)
        bulk_operations.append(UpdateOne(update_query, {'$unset': remove_data}))
        if (int(each_tick['date_stamp']) < 0):
            print(update_query, remove_data)
    try:
        if bulk_operations:
            collections.bulk_write(bulk_operations)
    except:
        traceback.print_exc()


