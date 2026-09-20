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
from datetime import (
    datetime as dt
)
import time
import numpy as np
import pandas as pd

try:
    # import QUANTAXIS as QA
    from QUANTAXIS.QAUtil import (
        # DATABASE,
        QA_util_log_info, 
        QA_util_code_tolist,
        QA_util_date_valid,
        QA_util_date_stamp,
        QA_util_time_stamp,
    )
except Exception:
    print('PLEASE run "pip install QUANTAXIS" before call GolemQ.fetch.StockCN_realtime modules')
    pass
    from QUANTAXIS.QAUtil import (
        # DATABASE,
        QA_util_log_info, 
        QA_util_code_tolist,
        QA_util_date_valid,
        QA_util_date_stamp,
        QA_util_time_stamp,
    )
try:
    from GolemQ.core.settings import (
        DATABASE as DATABASE_GolemQ,
    )
    from GolemQ.core.constants import (
        AKA,
    )
except Exception:
    class AKA():
        """
        常量，专有名称指标，定义成常量可以避免直接打字符串造成的拼写错误。
        """

        # 蜡烛线指标
        CODE = 'code'
        NAME = 'name'
        OPEN = 'open'
        HIGH = 'high'
        LOW = 'low'
        CLOSE = 'close'
        VOLUME = 'volume'
        VOL = 'vol'
        DATETIME = 'datetime'
        DATE = 'date'
        LAST_CLOSE = 'last_close'
        PRE_CLOSE = 'pre_close'

        CAPITALIZATION = 'capitalization'

        def __setattr__(self, name, value):
            raise Exception(u'Const Class can\'t allow to change property\' value.')
            return super().__setattr__(name, value)
    from GolemQ.core.settings import (
        DATABASE as DATABASE_GolemQ,
    )
from GolemQ.core.preprocessing import (
    GQ_util_to_json_from_pandas,
)
from GolemQ.markets.StockCN import (
    is_stock_cn,
)
import traceback
import json
import sys
import pymongo
from pymongo import UpdateOne
from func_timeout import func_set_timeout
from GolemQ.core import GQ_util_code_tolist


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


def GQ_fetch_daily_metadata_reality(
    code: str,
    start: str,
    end: str,
    collections=DATABASE_GolemQ.stock_diagnosis
) -> pd.DataFrame:
    """'获取 A股 复盘数据'，
    包括 fbprophet 价格区间预测，20/60日主力成本，
    机构研报信息，ROE/PE 财务数据等等，
    复盘数据在 每天凌晨，中午休息获取，计算和更新

    Returns:
        [type] -- [description]
    """

    start = str(start)[0:10]
    end = str(end)[0:10]

    # code checking
    if (code is not None):
        code = GQ_util_code_tolist(code)
        if QA_util_date_valid(end):
            query_id = {
                    'code': {
                        '$in': code
                    },
                    "date_stamp": {
                        "$gte": QA_util_date_stamp(start),
                        "$lte": QA_util_date_stamp(end) if int(
                            pd.to_datetime(end).to_pydatetime().timestamp()) <= QA_util_date_stamp(
                                end) else int(pd.to_datetime(end).to_pydatetime().timestamp()),
                    }
                }
            cursor = collections.find(query_id,
                {"_id": 0},
                batch_size=10000)
    else:
        if QA_util_date_valid(end):
            query_id = {
                    "date_stamp": {
                        "$gte": QA_util_date_stamp(start),
                        "$lte": QA_util_date_stamp(end) if int(
                            pd.to_datetime(end).to_pydatetime().timestamp()) <= QA_util_date_stamp(
                                end) else int(pd.to_datetime(end).to_pydatetime().timestamp()),
                    }
                }
            cursor = collections.find(query_id,
                {"_id": 0},
                batch_size=10000)        
        # res=[QA_util_dict_remove_key(data, '_id') for data in cursor]

    if QA_util_date_valid(end):
        res = pd.DataFrame([item for item in cursor])
        try:
            res = res.assign(
                date=pd.to_datetime(res.date)).drop_duplicates(([
                        'date',
                        'code'])).set_index([
                            'date',
                            'code'], drop=False)
        except Exception:
            # print(u'GQ_fetch_stock_diagnosis:{} Error:'.format(code), e, res.columns)
            res = None
            pass

        return res
    else:
        QA_util_log_info(f'GolemQ Error GQ_fetch_daily_metadata_reality data parameter start={start:%s} end={end:%s} is not right')
    return None


def GQ_fetch_hourly_metadata_reality(
    code: str,
    start: str,
    end: str,
    verbose: bool = False,
    collections=DATABASE_GolemQ.stock_diagnosis
) -> pd.DataFrame:
    """'获取 A股 复盘数据'，
    包括 fbprophet 价格区间预测，20/60日主力成本，
    机构研报信息，ROE/PE 财务数据等等，
    复盘数据在 每天凌晨，中午休息获取，计算和更新

    Returns:
        [type] -- [description]
    """

    start = str(start)[0:19]
    end = str(end)[0:19]

    # code checking
    code = QA_util_code_tolist(code)
    if (code is not None):
        code = QA_util_code_tolist(code)
        if QA_util_date_valid(str(end)[0:10]):
            cursor = collections.find({
                    'code': {
                        '$in': code
                    },
                    "time_stamp":
                        {
                        "$gte": QA_util_time_stamp(start),
                        "$lte": QA_util_time_stamp(end) if int(
                            pd.to_datetime(
                                end).to_pydatetime().timestamp()) <= QA_util_time_stamp(
                                    end) else int(pd.to_datetime(end).to_pydatetime().timestamp()),
                        }
                },
                {"_id": 0},
                batch_size=10000)
        #res=[QA_util_dict_remove_key(data, '_id') for data in cursor]
    else:
        if QA_util_date_valid(str(end)[0:10]):
            cursor = collections.find({
                    "time_stamp":
                        {
                        "$gte": QA_util_time_stamp(start),
                        "$lte": QA_util_time_stamp(end) if int(
                            pd.to_datetime(
                                end).to_pydatetime().timestamp()) <= QA_util_time_stamp(
                                    end) else int(pd.to_datetime(end).to_pydatetime().timestamp()),
                        }
                },
                {"_id": 0},
                batch_size=10000)
    
    if QA_util_date_valid(str(end)[0:10]):
        res = pd.DataFrame([item for item in cursor])
        try:
            res = res.assign(datetime=pd.to_datetime(res.datetime)).drop_duplicates((['datetime',
                                'code'])).set_index(['datetime',
                                'code'],
                                    drop=False)
        except Exception as e:
            if (verbose): 
                print(u'GQ_fetch_hourly_metadata_reality{} Error:'.format(code), e, res.columns)
                print(collections, res)
                traceback.print_exc()
            if (len(res.columns)==0) and \
                (len(res.index)==0):
                pass
            elif (is_stock_cn(code)[1] == QA.MARKET_TYPE.INDEX_CN):
                traceback.print_exc()
            elif (code[0].startswith('159')):
                traceback.print_exception(type(e), e, sys.exc_info()[2])  
            res = None
            pass
            
        return res
    else:
        QA_util_log_info(f'GolemQ Error GQ_fetch_hourly_metadata_reality data parameter start={start:%s} end={end:%s} is not right')
    return None


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
    code = QA_util_code_tolist(code)
    if QA_util_date_valid(end):
        cursor = collections.find({
                'code': {
                    '$in': code
                },
                "date_stamp": {
                    "$gte": QA_util_date_stamp(start),
                    "$lte": QA_util_date_stamp(end) if int(
                        pd.to_datetime(end).to_pydatetime().timestamp()) <= QA_util_date_stamp(
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


def GQ_update_hourly_metadata(
    features: pd.DataFrame = None,
    collections=DATABASE_GolemQ.stock_metadata_60min,
):
    coll=collections
    coll.create_index([("time_stamp",
                    pymongo.ASCENDING),],
                unique=False)
    coll.create_index([("code",
                    pymongo.ASCENDING),
                    ("time_stamp",
                    pymongo.ASCENDING),],
            unique=True)
    coll.create_index([("code",
                    pymongo.ASCENDING),
                    ("date_stamp",
                    pymongo.ASCENDING),],
            unique=False)
    
    features['code'] = features.index.get_level_values(level=1)

    features['date'] = pd.to_datetime(features.index.get_level_values(level=0).date,).tz_localize('Asia/Shanghai')
    features['datetime'] = pd.to_datetime(features.index.get_level_values(level=0),).tz_localize('Asia/Shanghai')

    # GMT+0 String 转换为 UTC Timestamp
    features['time_stamp'] = features['datetime'].astype(np.int64) // 10 ** 9
    features["date_stamp"] = features['date'].astype(np.int64) // 10 ** 9
    features['updated_at'] = int(time.mktime(dt.now().utctimetuple()))

    features['date'] = features['date'].dt.strftime('%Y-%m-%d')
    features['datetime'] = features['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')

    #print(features.columns)
    each_day = sorted(features.index.get_level_values(level=0).unique())
    codelist = sorted(features.index.get_level_values(level=1).unique())

    # Convert all numpy.float32 to Python float
    data = features
    bulk_operations = []
    
    if len(data) == 0:
        print(u'GQ_update_hourly_metadata() Data len equals zero.')

    for idx, each_tick in data.iterrows():
        update_query = {
            'code': each_tick['code'],
            'time_stamp': int(each_tick['time_stamp']),
        }
        update_data = {
            '$set': json.loads(each_tick.to_json(orient='index')),  # each_tick.to_dict(),  # 直接使用 to_dict() 转换为字典
        }
        bulk_operations.append(UpdateOne(update_query, update_data, upsert=True))
        if (int(each_tick['time_stamp']) < 0):
            print(update_query, update_data)
    try:
        if bulk_operations:
            #pass
            collections.bulk_write(bulk_operations)
    except:
        traceback.print_exc()

        
def GQ_remove_hourly_metadata(
    features: pd.DataFrame = None,
    collections=DATABASE_GolemQ.stock_metadata_60min,
):
    features['code'] = features.index.get_level_values(level=1)

    features['date'] = pd.to_datetime(features.index.get_level_values(level=0).date,).tz_localize('Asia/Shanghai')
    features['datetime'] = pd.to_datetime(features.index.get_level_values(level=0),).tz_localize('Asia/Shanghai')

    # GMT+0 String 转换为 UTC Timestamp
    features['time_stamp'] = features['datetime'].astype(np.int64) // 10 ** 9
    features["date_stamp"] = features['date'].astype(np.int64) // 10 ** 9

    features['date'] = features['date'].dt.strftime('%Y-%m-%d')
    features['datetime'] = features['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    data = features
    bulk_operations = []
    
    if len(data) == 0:
        print(u'GQ_remove_hourly_metadata() Data len equals zero.')

    for idx, each_tick in data.iterrows():
        update_query = {
            'code': each_tick['code'],
            'time_stamp': int(each_tick['time_stamp']),
        }
        
        # 生成 remove_data 字典
        remove_data = {col: 1 for col in features.columns if col not in ['code', 'date', 'datetime', "date_stamp", 'time_stamp']}
        # print(remove_data, features.columns)
        bulk_operations.append(UpdateOne(update_query, {'$unset': remove_data}))
        if (int(each_tick['time_stamp']) < 0):
            print(update_query, remove_data)
    try:
        if bulk_operations:
            collections.bulk_write(bulk_operations)
    except:
        traceback.print_exc()
        

def GQ_move_hourly_metadata(
    features,
    verbose=False,
    collections=DATABASE_GolemQ.stock_metadata_60min,
    collections_to=DATABASE_GolemQ.stock_metadata_15min,):
    """
    save current day's stock_min data
    """
    coll = collections
    coll_t=collections_to
    
    features['date'] = pd.to_datetime(features.index.get_level_values(level=0).date,).tz_localize('Asia/Shanghai')
    features['datetime'] = pd.to_datetime(features.index.get_level_values(level=0),).tz_localize('Asia/Shanghai')

    # GMT+0 String 转换为 UTC Timestamp
    features['time_stamp'] = features['datetime'].astype(np.int64) // 10 ** 9
    features["date_stamp"] = features['date'].astype(np.int64) // 10 ** 9
    
    #print(features.columns)
    each_day = sorted(features.index.get_level_values(level=0).unique())
    codelist = sorted(features.index.get_level_values(level=1).unique())
    
    try:
        # GMT+0 String 转换为 UTC Timestamp
        if (len(each_day) > 1):
            query_id = {
                '$or': [{'code': codelist[0] if (len(codelist)==1) else {'$in': codelist},
                        'time_stamp': int(each_tick_datetime)} for each_tick_datetime in features['time_stamp'].values],
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
        
        try:
            refcount = coll.count_documents(query_id)
        except:
            print(query_id)
            traceback.print_exc()

        if refcount > 0:
            if ('CoordinatesMin' in features.columns):
                features['CoordinatesMin'] = features['CoordinatesMin'].dt.tz_localize('Asia/Shanghai').apply(lambda x: x.timestamp() * 1000 if x is not pd.NaT else np.nan)
            if ('Coordinates' in features.columns):
                features['Coordinates'] = features['Coordinates'].dt.tz_localize('Asia/Shanghai').apply(lambda x: x.timestamp() * 1000 if x is not pd.NaT else np.nan)
            if ('CoordinatesNano' in features.columns):
                features['CoordinatesNano'] = features['CoordinatesNano'].dt.tz_localize('Asia/Shanghai').apply(lambda x: x.timestamp() * 1000 if x is not pd.NaT else np.nan)
            GQ_update_hourly_metadata(
                features=features,
                collections=collections_to,
            )
            try:
                coll.delete_many(query_id)
                if (verbose):
                    print(f'Code: {codelist} @{features.index.get_level_values(level=0)[-1]} removed OK.... \n',)
            except Exception as e:
                traceback.print_exception(type(e), e, sys.exc_info()[2])
                print(e)
        else:
            # 新 tick，插入记录
            if (verbose):
                print(f'code: {codelist} not kline\n',)
    except Exception as e:
        traceback.print_exception(type(e), e, sys.exc_info()[2])
        print(e)
        return None
    
    return


@func_set_timeout(5)
def GQ_save_stock_valuation(
    stock_valuation_detail,
    collections=DATABASE_GolemQ.stock_valuation,
    verbose=False
):
    '''
    保存股价估值信息等数据
    :param collections: 数据库 collection 存储名称
    :return:
    '''
    coll = collections
    coll.create_index([(
        "date_stamp",
        pymongo.ASCENDING)],
        unique=False)
    coll.create_index([(
        AKA.CODE,
        pymongo.ASCENDING), (
        "date_stamp",
        pymongo.ASCENDING)],
        unique=True)

    refcount = 0
    if (len(stock_valuation_detail) > 0):
        try:
            stock_valuation_detail = stock_valuation_detail.assign(
                date_stamp=pd.to_datetime(stock_valuation_detail['date']).astype(np.int64) // 10 ** 9
                # date_stamp=pd.to_datetime(stock_valuation_detail['date']).view("int64") // 10 ** 9
                
            )
            stock_valuation_detail['date'] = pd.to_datetime(stock_valuation_detail['date'])
            stock_valuation_detail = stock_valuation_detail.set_index(['date', AKA.CODE], drop=False)
            stock_valuation_detail['date'] = pd.to_datetime(stock_valuation_detail['date']).dt.strftime('%Y-%m-%d')

            each_day = sorted(stock_valuation_detail.index.get_level_values(level=0).unique())
            codelist = sorted(stock_valuation_detail.index.get_level_values(level=1).unique())

            # Convert all numpy.float32 to Python float
            data = stock_valuation_detail

            # 查询是否新 tick
            if (len(each_day) > 1):
                query_id = {
                    '$or': [{
                        'code': codelist[0] if (len(codelist) == 1) else {'$in': codelist},
                        'date_stamp': int(each_tick_datetime)} for each_tick_datetime in stock_valuation_detail['date_stamp'].values],
                }
            elif (len(each_day) > 0):
                query_id = {
                    'code': {
                        '$in': codelist,
                    },
                    'date_stamp': int(stock_valuation_detail['date_stamp'].tail(1).item()),
                }
            else:
                return None

            refcount = coll.count_documents(query_id)
            if (refcount > 0):
                # 使用 bulk_write 进行批量 upsert 操作
                bulk_operations = []

                if (len(data) == 0):
                    print(u'GQ_save_stock_valuation() {} Data len equals zero.'.format(codelist))

                for idx, each_tick in data.iterrows():
                    update_query = {
                        'code': each_tick['code'],
                        'date_stamp': int(each_tick['date_stamp']),
                    }
                    update_data = {
                        '$set': json.loads(each_tick.to_json(orient='index')),     # each_tick.to_dict(),  # 直接使用 to_dict() 转换为字典
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
                                'date_stamp': int(each_tick['date_stamp']),
                            }
                            update_data = {
                                '$set': json.loads(each_tick.to_json(orient='index')),    # each_tick.to_dict(),  # 直接使用 to_dict() 转换为字典
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
                # 新 tick，插入记录
                data_json = GQ_util_to_json_from_pandas(data)
                coll.insert_many(data_json)
        except Exception as e:
            traceback.print_exception(type(e), e, sys.exc_info()[2])
            print(e)
            if (refcount == 0):
                print(query_id)
            return None

    return stock_valuation_detail

