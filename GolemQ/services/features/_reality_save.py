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
import pymongo
from pymongo import UpdateOne
from datetime import datetime as dt
from QUANTAXIS.QAUtil import QA_util_log_info, QA_util_code_tolist, QA_util_date_valid, QA_util_date_stamp, QA_util_time_stamp
from GolemQ.core.settings import DATABASE as DATABASE_GolemQ
from GolemQ.core.constants import AKA
from GolemQ.core.preprocessing import GQ_util_to_json_from_pandas
from GolemQ.core import GQ_util_code_tolist
from func_timeout import func_set_timeout
def GQ_save_metadata_reality(
    features: pd.DataFrame = None,
    collections=DATABASE_GolemQ.stock_metadata,
):
    '''
    保存同花顺的牛股诊股
    :param collections: 数据库 collection 存储名称
    :return:
    '''
    coll = collections
    coll.create_index(
        [("time_stamp",
          pymongo.ASCENDING),],
        unique=False)
    coll.create_index(
        [("code",
          pymongo.ASCENDING),
         ("time_stamp",
          pymongo.ASCENDING),],
        unique=True)
    coll.create_index(
        [("code",
          pymongo.ASCENDING),
         ("date_stamp",
          pymongo.ASCENDING),],
        unique=False)

    refcount = 0
    try:
        features['code'] = features.index.get_level_values(level=1)

        features['date'] = pd.to_datetime(features.index.get_level_values(level=0).date,).tz_localize('Asia/Shanghai')
        features['datetime'] = pd.to_datetime(features.index.get_level_values(level=0),).tz_localize('Asia/Shanghai')

        # GMT+0 String 转换为 UTC Timestamp
        features['time_stamp'] = features['datetime'].astype(np.int64) // 10 ** 9
        features["date_stamp"] = features['date'].astype(np.int64) // 10 ** 9
        features['created_at'] = int(time.mktime(dt.now().utctimetuple()))

        features['date'] = features['date'].dt.strftime('%Y-%m-%d')
        features['datetime'] = features['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')

        # print(features.columns)
        each_day = sorted(features.index.get_level_values(level=0).unique())
        codelist = sorted(features.index.get_level_values(level=1).unique())

        # Convert all numpy.float32 to Python float
        data = features

        # 查询是否新 tick
        if (len(each_day) > 1):
            query_id = {
                '$or': [{
                    'code': codelist[0] if (len(codelist) == 1) else {'$in': codelist},
                    'time_stamp': int(each_tick_datetime)
                } for each_tick_datetime in features['time_stamp'].values],
            }
        elif (len(each_day) > 0):
            query_id = {
                'code': {
                    '$in': codelist,
                },
                'time_stamp': int(features['time_stamp'].tail(1).item()),
            }
        else:
            return None

        # print(query_id)
        # 检查是否有重复数据
        refcount = coll.count_documents(query_id)
        if (refcount > 0):
            # print('Delete', refcount, len(l1_ticks_data))
            # 使用 bulk_write 进行批量 upsert 操作
            bulk_operations = []

            if (len(data) == 0):
                print(u'GQ_save_metadata_reality() {} Data len equals zero.'.format(codelist))

            for idx, each_tick in data.iterrows():
                update_query = {
                    'code': each_tick['code'],
                    'time_stamp': int(each_tick['time_stamp']),
                }
                update_data = {
                    '$set': json.loads(each_tick.to_json(orient='index')),  # each_tick.to_dict(),  # 直接使用 to_dict() 转换为字典
                }
                bulk_operations.append(
                    UpdateOne(
                        update_query,
                        update_data,
                        upsert=True))
            try:
                if bulk_operations:
                    coll.bulk_write(bulk_operations)
            except Exception:
                try:
                    for idx, each_tick in data.iterrows():
                        bulk_operations = []
                        update_query = {
                            'code': each_tick['code'],
                            'time_stamp': int(each_tick['time_stamp']),
                        }
                        update_data = {
                            '$set': json.loads(each_tick.to_json(orient='index')),  # each_tick.to_dict(),  # 直接使用 to_dict() 转换为字典
                        }
                        bulk_operations.append(
                            UpdateOne(
                                update_query,
                                update_data,
                                upsert=True))
                        coll.bulk_write(bulk_operations)
                except Exception as e:
                    print(e)
                    print(update_data)

            # print('bulk_write:UpdateOne', refcount, update_query, update_data)
        else:
            # 新 tick，插入记录
            # print('insert_many', refcount)
            data_json = GQ_util_to_json_from_pandas(data)
            coll.insert_many(data_json)
    except Exception as e:
        traceback.print_exception(type(e), e, sys.exc_info()[2])
        print(e)
        if (refcount == 0):
            print(query_id)
        return None
    # print(u'Code: {} 保存review缓存成功!'.format(codelist))
    return features


