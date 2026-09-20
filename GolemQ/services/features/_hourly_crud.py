# coding:utf-8
"""小时级 metadata 的 CRUD —— ``services/features`` 包的 hourly 分册。

对应日线侧的 ``_daily_crud.py``（``GQ_fix/update/remove_daily_metadata``）。

``GQ_move_hourly_metadata`` 会调本模块的 ``GQ_update_hourly_metadata`` 把记录写到
另一个集合再删原集合里的 —— 所以这两个函数必须同处一个模块、共享同一份实现，
拆到两个文件只会多一层无谓的间接。

**三个函数都是从原 ``services/features.py`` 逐字节搬运过来的**，未做任何改写；
该模块已退休（它曾把本包整体遮蔽，见 ``services/features/__init__.py`` 的说明）。
"""
import numpy as np
import pandas as pd
import json
import sys
import time
import traceback
import pymongo
from pymongo import UpdateOne
from datetime import datetime as dt
from GolemQ.core.settings import DATABASE as DATABASE_GolemQ

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
