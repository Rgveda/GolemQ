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
import sys
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
def GQ_save_daily_metadata_reality(
    features: pd.DataFrame = None,
    collections=DATABASE_GolemQ.stock_metadata_day,
):
    '''
    保存同花顺的牛股诊股
    :param collections: 数据库 collection 存储名称
    :return:
    '''
    coll = collections
    coll.create_index(
        [('code',
        pymongo.ASCENDING),
        ("date_stamp",
        pymongo.ASCENDING)],
        unique=True)
    coll.create_index(
        [("date_stamp",
        pymongo.ASCENDING)],
        unique=False)

    codelist = list(features.index.get_level_values(level=1).unique())
    if (AKA.CODE not in features.columns):
        features = features.reindex(
            columns=list(set([
                *features.columns,
                *[AKA.CODE,]])))
        features[AKA.CODE] = features.index.get_level_values(level=1)
    if ("time_stamp" in features.columns):
        features = features.drop(columns=["time_stamp",])
    if (AKA.DATETIME not in features.columns):
        features = features.reindex(
            columns=list(set([
                *features.columns,
                *[AKA.DATE,
                AKA.DATETIME,]])))
    features[AKA.DATE] = pd.to_datetime(features.index.get_level_values(level=0).date,).tz_localize('Asia/Shanghai')
    features[AKA.DATETIME] = pd.to_datetime(features.index.get_level_values(level=0),).tz_localize('Asia/Shanghai')
    features = features.drop_duplicates(
        ([AKA.DATE, AKA.CODE])).set_index(
            [AKA.DATE],
            drop=False)
    
    # GMT+0 String 转换为 UTC Timestamp
    # features["date_stamp"] = pd.to_datetime(features['date']).view(np.int64) // 10 ** 9
    features["date_stamp"] = pd.to_datetime(features['date']).astype(np.int64) // 10 ** 9
    features['created_at'] = int(time.mktime(dt.now().utctimetuple()))
    
    features['date'] = features['date'].dt.strftime('%Y-%m-%d')
    features['datetime'] = features['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')
    try:
        # print(features.columns)
        each_day = sorted(features.index.get_level_values(level=0).unique())

        # Convert all numpy.float32 to Python float
        data = features

        # 查询是否新 tick
        if (len(each_day) > 1):
            query_id = {
                '$or': [{
                    'code': codelist[0] if (len(codelist) == 1) else {'$in': codelist},
                    'date_stamp': int(each_tick_datetime)
                } for each_tick_datetime in features['date_stamp'].values],
            }
        elif (len(each_day) > 0):
            query_id = {
                'code': {
                    '$in': codelist,
                },
                'date_stamp': int(features['date_stamp'].tail(1).item()),
            }
        else:
            return None

        refcount = coll.count_documents(query_id)
        if (refcount > 0):
            bulk_operations = []

            if (len(data) == 0):
                print(u'GQ_save_daily_metadata_reality() {} Data len equals zero.'.format(codelist))

            for idx, each_tick in data.iterrows():
                update_query = {
                    'code': each_tick['code'],
                    'date_stamp': int(each_tick['date_stamp']),
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
                traceback.print_exc()
                try:
                    for idx, each_tick in data.iterrows():
                        bulk_operations = []
                        update_query = {
                            'code': each_tick['code'],
                            'date_stamp': int(each_tick['date_stamp']),
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

        else:
            data_json = GQ_util_to_json_from_pandas(data)
            coll.insert_many(data_json)
    except Exception as e:
        traceback.print_exception(type(e), e, sys.exc_info()[2])
        print(e)
        if (refcount == 0):
            print(query_id)
        return None
    return features


