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
    datetime as dt,
    timedelta,
)
import time
import numpy as np
import pandas as pd
import traceback
import json
import sys
import pymongo
from pymongo import UpdateOne
from GolemQ.core.constants import (
    AKA, MARKET_TYPE,
    FIELD as FLD,
)
from GolemQ.core.preprocessing import GQ_util_to_json_from_pandas
from GolemQ.core.symbol import GQ_util_code_tolist
from GolemQ.core.settings import DATABASE as DATABASE_GolemQ
from .date_utils import (
    GQ_util_date_valid,
    GQ_util_date_stamp,
    GQ_util_time_stamp,
    GQ_util_get_last_day,
    get_15min_aligned_timestamp,
    get_60min_aligned_timestamp,
)
from .symbol import (
    is_stock_cn,
    GQ_fetch_etf_list,
)
from .constants import TRADE_DATE_SSE
from GolemQ.core.settings import DATABASE
from GolemQ.supervisor import (
    checkin_function
)
import requests
try:
    import akshare as ak
except ImportError:
    # 获取Python主版本和次版本
    major, minor = sys.version_info[:2]
    if (major == 3 and minor > 11) or major > 3:
        print('PLEASE run "pip install akshare" before call GolemQ.cli modules')
    pass


def QA_fetch_trade_date():
    '获取交易日期'
    return TRADE_DATE_SSE


def QA_fetch_stock_list(collections=DATABASE.stock_list):
    '获取股票列表'

    return pd.DataFrame([item for item in collections.find()]).drop(
        '_id',
        axis=1,
        inplace=False
    ).set_index(
        'code',
        drop=False
    )


def QA_fetch_index_list(collections=DATABASE.index_list):
    '获取指数列表'
    return pd.DataFrame([item for item in collections.find()]).drop(
        '_id',
        axis=1,
        inplace=False
    ).set_index(
        'code',
        drop=False
    )


def QA_fetch_stock_terminated(collections=DATABASE.stock_terminated):
    '获取股票基本信息 , 已经退市的股票列表'
    # 🛠todo 转变成 dataframe 类型数据
    return pd.DataFrame([item for item in collections.find()]).drop(
        '_id',
        axis=1,
        inplace=False
    ).set_index(
        'code',
        drop=False
    )


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
                print(u'save_stock_review() {} Data len equals zero.'.format(codelist))

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
        [
            ('code', pymongo.ASCENDING),
            ("date_stamp", pymongo.ASCENDING)
        ],
        unique=True)

    code = features.index.get_level_values(level=1)[0]
    if (AKA.CODE not in features.columns):
        features[AKA.CODE] = features.index.get_level_values(level=1)

    features['date'] = pd.to_datetime(features.index.get_level_values(level=0).date,).tz_localize('Asia/Shanghai')
    features['datetime'] = pd.to_datetime(features.index.get_level_values(level=0),).tz_localize('Asia/Shanghai')
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
        data = features
        query_id = {
            "code":
                {
                    "$in": list(set(features['code'].to_list()))
                },
            "date_stamp":
                {
                    "$in": features['date_stamp'].to_list()
                },
        }

        refcount = coll.count_documents(query_id)
        if (refcount > 0):
            bulk_operations = []

            if (len(data) == 0):
                print(u'save_metadata_reality_major() {} Data len equals zero.'.format(code))

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
        if GQ_util_date_valid(end):
            query_id = {
                    'code': {
                        '$in': code
                    },
                    "date_stamp": {
                        "$gte": GQ_util_date_stamp(start),
                        "$lte": GQ_util_date_stamp(end) if int(
                            pd.to_datetime(end).to_pydatetime().timestamp()) <= GQ_util_date_stamp(
                                end) else int(pd.to_datetime(end).to_pydatetime().timestamp()),
                    }
                }
            cursor = collections.find(
                query_id,
                {"_id": 0},
                batch_size=10000)
    else:
        if GQ_util_date_valid(end):
            query_id = {
                    "date_stamp": {
                        "$gte": GQ_util_date_stamp(start),
                        "$lte": GQ_util_date_stamp(end) if int(
                            pd.to_datetime(end).to_pydatetime().timestamp()) <= GQ_util_date_stamp(
                                end) else int(pd.to_datetime(end).to_pydatetime().timestamp()),
                    }
                }
            cursor = collections.find(
                query_id,
                {"_id": 0},
                batch_size=10000)
        # res=[QA_util_dict_remove_key(data, '_id') for data in cursor]

    if GQ_util_date_valid(end):
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
        print(f'GolemQ Error GQ_fetch_daily_metadata_reality data parameter start={start:%s} end={end:%s} is not right')
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
    code = GQ_util_code_tolist(code)
    if (code is not None):
        code = GQ_util_code_tolist(code)
        if GQ_util_date_valid(str(end)[0:10]):
            cursor = collections.find({
                    'code': {
                        '$in': code
                    },
                    "time_stamp": {
                        "$gte": GQ_util_time_stamp(start),
                        "$lte": GQ_util_time_stamp(end) if int(
                            pd.to_datetime(
                                end).to_pydatetime().timestamp()) <= GQ_util_time_stamp(
                                    end) else int(pd.to_datetime(end).to_pydatetime().timestamp()),
                    }
                },
                {"_id": 0},
                batch_size=10000)
        # res=[QA_util_dict_remove_key(data, '_id') for data in cursor]
    else:
        if GQ_util_date_valid(str(end)[0:10]):
            cursor = collections.find({
                    "time_stamp": {
                        "$gte": GQ_util_time_stamp(start),
                        "$lte": GQ_util_time_stamp(end) if int(
                            pd.to_datetime(
                                end).to_pydatetime().timestamp()) <= GQ_util_time_stamp(
                                    end) else int(pd.to_datetime(end).to_pydatetime().timestamp()),
                    }
                },
                {"_id": 0},
                batch_size=10000)

    if GQ_util_date_valid(str(end)[0:10]):
        res = pd.DataFrame([item for item in cursor])
        try:
            res = res.assign(
                datetime=pd.to_datetime(
                    res.datetime)).drop_duplicates(([
                        'datetime',
                        'code'])).set_index([
                            'datetime',
                            'code'],
                            drop=False)
        except Exception as e:
            if (verbose):
                print(u'GQ_fetch_hourly_metadata_reality{} Error:'.format(code), e, res.columns)
                print(collections, res)
                traceback.print_exc()
            if (len(res.columns) == 0) and (len(res.index) == 0):
                pass
            elif (is_stock_cn(code)[1] == MARKET_TYPE.INDEX_CN):
                traceback.print_exc()
            elif (code[0].startswith('159')):
                traceback.print_exception(type(e), e, sys.exc_info()[2])
            res = None
            pass

        return res
    else:
        print(f'GolemQ Error GQ_fetch_hourly_metadata_reality data parameter start={start:%s} end={end:%s} is not right')
    return None


def GQ_get_etf_list(
    verbose: bool = False
) -> pd.DataFrame:
    """获取A股全部ETF列表

    从./datastore/metadata/stock_cn/etf/目录读取所有Excel文件，
    提取基金代码、基金简称、基金资产净值(元)数据

    Returns:
        pd.DataFrame: 包含所有ETF信息的DataFrame，列包括:
            - code: 基金代码
            - name: 基金简称
            - net_asset_value: 基金资产净值(元)
    """
    # 导入频率控制模块
    try:
        # 检查调用频率（15分钟限制）
        checkin_result = checkin_function(
            function_name="GQ_get_etf_list",
            expired_time=timedelta(minutes=15)
        )
        if (verbose):
            print(checkin_result)
        if checkin_result["allowed"]:
            if (verbose):
                print('rounte 3')
            # 获取ETF分类列表
            etf_category_df = ak.fund_etf_category_sina(symbol="ETF基金")
            # 可以方便地保存到本地
            etf_category_df = etf_category_df.rename(
                columns={
                    '代码': AKA.CODE,
                    '名称': AKA.NAME,
                    # '涨跌额': FLD.PCT_CHANGE,
                    '昨收': AKA.LAST_CLOSE,
                    '成交量': AKA.VOLUME,
                    '成交额': AKA.AMOUNT,
                    '换手率': FLD.TURNOVER_RATE,
                    '最新价': 'price',
                    '涨跌幅': FLD.PCT_CHANGE,
                    '最高': AKA.HIGH,
                    '最低': AKA.LOW,
                    '今开': AKA.OPEN,
                    '总市值': AKA.CAPITALIZATION,
                }
            )
            etf_category_df['sec'] = 'etf_cn'
            etf_category_df['sse'] = etf_category_df[AKA.CODE].str[:2]
            etf_category_df[AKA.CODE] = etf_category_df[AKA.CODE].str[2:]
            etf_category_df['decimal_point'] = 3
            etf_category_df['volunit'] = 100

            try:
                pandas_data = GQ_util_to_json_from_pandas(etf_category_df)

                if len(pandas_data) > 0:
                    coll = DATABASE.etf_list
                    coll.create_index('code')

                    # 准备批量操作
                    operations = []
                    for item in pandas_data:
                        operation = UpdateOne(
                            {'code': item['code']},  # 查询条件
                            {
                                '$set': item,
                                '$setOnInsert': {'create_time': dt.now()}
                            },
                            upsert=True
                        )
                        operations.append(operation)

                    # 执行批量操作
                    if operations:
                        result = coll.bulk_write(operations)
                        if (verbose):
                            print(f"成功处理 {len(operations)} 条ETF数据")
                            print(f"匹配: {result.matched_count}, 修改: {result.modified_count}, 插入: {result.upserted_count}")
            except Exception as e:
                print(f"Error in bulk updating ETF list: {e}")
        elif not checkin_result["allowed"]:
            if (verbose):
                print('rounte 6', )
            # print('访问频率受控制', checkin_result)
            etf_category_df = GQ_fetch_etf_list()
            etf_category_df = etf_category_df[(etf_category_df[AKA.VOLUME] > 0.927)]
    except ImportError:
        # 如果监控模块不可用，跳过频率检查
        traceback.print_exc()
        pass
    except Exception as e:
        # 频率限制异常，直接抛出
        traceback.print_exc()
        raise e

    return etf_category_df


def GQ_stock_a_spot_em(
    market_type: str = MARKET_TYPE.STOCK_CN,
    collections=DATABASE_GolemQ.stock_a_snapshot,
    verbose=False
):
    """
    获取A股实时行情数据并保存到不同时间周期的数据集合中

    这个函数是为了解决每天实盘换手率的问题，通过时间对齐机制确保数据的一致性：
    - 按小时级别对齐时间戳，用于60分钟数据存储
    - 按15分钟对齐时间戳，用于15分钟数据存储
    - 同时保存日级别数据用于日线分析

    频率控制：同一个IP地址每15分钟只能调用一次

    Args:
        market_type: 市场类型，默认为A股市场
        collections: 数据库集合，默认为股票快照集合

    Returns:
        DataFrame: 处理后的股票快照数据
    """
    def rewrite_snapshot_daily(stock_cn_snapshot_pd):
        stock_cn_snapshot_daily = stock_cn_snapshot_pd.copy()
        stock_cn_snapshot_daily[AKA.DATE] = pd.to_datetime(GQ_util_get_last_day())
        stock_cn_snapshot_daily[AKA.CODE] = stock_cn_snapshot_daily.index.get_level_values(level=1)
        stock_cn_snapshot_daily = stock_cn_snapshot_daily.set_index(
            [AKA.DATE,
             AKA.CODE],
            drop=True)
        # print(stock_cn_snapshot_daily[[AKA.NAME, AKA.VOLUME, AKA.AMOUNT, FLD.TURNOVER_RATE, 'price']].tail(60))
        GQ_save_daily_metadata_reality(
            stock_cn_snapshot_daily,
            collections=DATABASE_GolemQ.stock_a_snapshot_day if (market_type == MARKET_TYPE.STOCK_CN) else DATABASE_GolemQ.etf_a_snapshot_day,
        )
        return stock_cn_snapshot_daily

    now = dt.now()
    current_time = now.time()
    curr_timestamp = get_60min_aligned_timestamp(current_time)
    curr_timestamp_15min = get_15min_aligned_timestamp(current_time)
    # print(curr_timestamp, curr_timestamp_15min)
    if (verbose):
        print(DATABASE_GolemQ.stock_a_snapshot_15min if (market_type == MARKET_TYPE.STOCK_CN) else DATABASE_GolemQ.etf_a_snapshot_15min)
    stock_hourly_feats_snapshot = GQ_fetch_hourly_metadata_reality(
        code=None,
        start=curr_timestamp_15min,
        end=curr_timestamp_15min,
        collections=DATABASE_GolemQ.stock_a_snapshot_15min if (market_type == MARKET_TYPE.STOCK_CN) else DATABASE_GolemQ.etf_a_snapshot_15min,
    )
    if (stock_hourly_feats_snapshot is not None) and (len(stock_hourly_feats_snapshot) > 1008):
        if (verbose):
            print('rounte 1')
        stock_cn_snapshot = stock_hourly_feats_snapshot
        stock_cn_snapshot_daily = GQ_fetch_daily_metadata_reality(
            code=None,
            start=curr_timestamp[0:10],
            end=curr_timestamp[0:10],
            collections=DATABASE_GolemQ.stock_a_snapshot_day if (market_type == MARKET_TYPE.STOCK_CN) else DATABASE_GolemQ.etf_a_snapshot_day,
        )
        if (stock_cn_snapshot_daily is None) or (len(stock_hourly_feats_snapshot) > len(stock_cn_snapshot_daily)):
            stock_cn_snapshot_daily = rewrite_snapshot_daily(stock_hourly_feats_snapshot)
                
        if (verbose):
            print(
                DATABASE_GolemQ.stock_a_snapshot_day if (market_type == MARKET_TYPE.STOCK_CN) else DATABASE_GolemQ.etf_a_snapshot_day,
                stock_cn_snapshot_daily,
                stock_hourly_feats_snapshot,)
    else:
        if (verbose):
            print('rounte 2')
        # 导入频率控制模块
        try:
            # 检查调用频率（15分钟限制）
            checkin_result = checkin_function(
                function_name="GQ_stock_a_spot_em" if (market_type == MARKET_TYPE.STOCK_CN) else "GQ_stock_a_spot_em",
                expired_time=timedelta(minutes=10)
            )
            if (verbose):
                print(checkin_result)
            if checkin_result["allowed"]:
                if (verbose):
                    print('rounte 3')
                try:
                    stock_cn_snapshot = ak.stock_zh_a_spot_em() if (market_type == MARKET_TYPE.STOCK_CN) else ak.fund_etf_spot_em()
                except requests.exceptions.ConnectionError:
                    stock_cn_snapshot_daily = GQ_fetch_daily_metadata_reality(
                        code=None,
                        start=curr_timestamp[0:10],
                        end=curr_timestamp[0:10],
                        collections=DATABASE_GolemQ.stock_a_snapshot_day if (market_type == MARKET_TYPE.STOCK_CN) else DATABASE_GolemQ.etf_a_snapshot_day,
                    )
                    if (verbose):
                        print('rounte 9\n', stock_cn_snapshot_daily)
                    return stock_cn_snapshot_daily

                _ = stock_cn_snapshot[stock_cn_snapshot['最新价'].isna()]
                if (market_type == MARKET_TYPE.STOCK_CN):
                    if (verbose):
                        print('rounte 4')
                    # stock_cn_snapshot = stock_cn_snapshot[~stock_cn_snapshot['最新价'].isna()].set_index('序号')
                    stock_cn_snapshot = stock_cn_snapshot.rename(
                        columns={
                            '代码': AKA.CODE,
                            '名称': AKA.NAME,
                            '昨收': AKA.LAST_CLOSE,
                            '成交量': AKA.VOLUME,
                            '成交额': AKA.AMOUNT,
                            '换手率': FLD.TURNOVER_RATE,
                            '最新价': 'price',
                            '涨跌幅': FLD.PCT_CHANGE,
                            '最高': AKA.HIGH,
                            '最低': AKA.LOW,
                            '今开': AKA.OPEN,
                        }
                    )
                    if (1.0 <= stock_cn_snapshot[FLD.TURNOVER_RATE].max() <= 100):
                        stock_cn_snapshot[FLD.TURNOVER_RATE] = stock_cn_snapshot[FLD.TURNOVER_RATE].astype(np.float32) / 100
                        stock_cn_snapshot[FLD.PCT_CHANGE] = stock_cn_snapshot[FLD.PCT_CHANGE].astype(np.float32) / 100
                else:
                    if (verbose):
                        print('rounte 5')
                    # stock_cn_snapshot = stock_cn_snapshot[~stock_cn_snapshot['最新价'].isna()]
                    stock_cn_snapshot = stock_cn_snapshot.rename(
                        columns={
                            '代码': AKA.CODE,
                            '名称': AKA.NAME,
                            '昨收': AKA.LAST_CLOSE,
                            '成交量': AKA.VOLUME,
                            '成交额': AKA.AMOUNT,
                            '换手率': FLD.TURNOVER_RATE,
                            '最新价': 'price',
                            '涨跌幅': FLD.PCT_CHANGE,
                            '最高': AKA.HIGH,
                            '最低': AKA.LOW,
                            '今开': AKA.OPEN,
                            '总市值': AKA.CAPITALIZATION,
                        }
                    )
                    if (stock_cn_snapshot is not None) and (FLD.TURNOVER_RATE in stock_cn_snapshot.columns):
                        if (1.0 <= stock_cn_snapshot[FLD.TURNOVER_RATE].max() <= 100):
                            stock_cn_snapshot[FLD.TURNOVER_RATE] = stock_cn_snapshot[FLD.TURNOVER_RATE].astype(np.float32) / 100
                    if (1.0 <= stock_cn_snapshot[FLD.PCT_CHANGE].max() <= 100):
                        stock_cn_snapshot[FLD.PCT_CHANGE] = stock_cn_snapshot[FLD.PCT_CHANGE].astype(np.float32) / 100

                stock_cn_snapshot[AKA.DATETIME] = pd.to_datetime(curr_timestamp_15min)
                stock_cn_snapshot = stock_cn_snapshot.set_index(
                    [AKA.DATETIME,
                     AKA.CODE],
                    drop=True)
                GQ_save_metadata_reality(
                    stock_cn_snapshot,
                    collections=DATABASE_GolemQ.stock_a_snapshot_15min if (market_type == MARKET_TYPE.STOCK_CN) else DATABASE_GolemQ.etf_a_snapshot_15min,
                )

                stock_cn_snapshot[AKA.DATETIME] = pd.to_datetime(curr_timestamp)
                stock_cn_snapshot[AKA.CODE] = stock_cn_snapshot.index.get_level_values(level=1)
                stock_cn_snapshot = stock_cn_snapshot.set_index(
                    [AKA.DATETIME,
                     AKA.CODE],
                    drop=True)
                GQ_save_metadata_reality(
                    stock_cn_snapshot,
                    collections=DATABASE_GolemQ.stock_a_snapshot_60min if (market_type == MARKET_TYPE.STOCK_CN) else DATABASE_GolemQ.etf_a_snapshot_60min,
                )

                stock_cn_snapshot_daily = rewrite_snapshot_daily(stock_cn_snapshot)
            elif not checkin_result["allowed"]:
                if (verbose):
                    print('rounte 6', DATABASE_GolemQ.stock_a_snapshot_day if (market_type == MARKET_TYPE.STOCK_CN) else DATABASE_GolemQ.etf_a_snapshot_day)
                # print('访问频率受控制', checkin_result)
                stock_cn_snapshot_daily = GQ_fetch_daily_metadata_reality(
                    code=None,
                    start=curr_timestamp[0:10],
                    end=curr_timestamp[0:10],
                    collections=DATABASE_GolemQ.stock_a_snapshot_day if (market_type == MARKET_TYPE.STOCK_CN) else DATABASE_GolemQ.etf_a_snapshot_day,
                )
                if (stock_cn_snapshot_daily is not None) and (len(stock_cn_snapshot_daily) > 1008):
                    return stock_cn_snapshot_daily
                else:
                    if (market_type == MARKET_TYPE.STOCK_CN):
                        return QA_fetch_stock_list()
                    else:
                        return GQ_get_etf_list()
                if checkin_result["reason"] == "expired_time_not_reached":
                    print(
                        f"调用频率限制：同一个IP每15分钟只能调用一次GQ_stock_a_spot_em函数。"
                        f"当前频率：{checkin_result.get('calls_per_minute', 0):.2f}次/分钟，"
                        f"限制：4次/小时")

        except ImportError:
            # 如果监控模块不可用，跳过频率检查
            traceback.print_exc()
            pass
        except Exception as e:
            # 频率限制异常，直接抛出
            traceback.print_exc()
            raise e
        
    return stock_cn_snapshot_daily


def GQ_etf_a_spot_em(
        collections=DATABASE_GolemQ.stock_a_snapshot,
):
    # `GQ_stock_a_spot_em` 内部按 `market_type == STOCK_CN ? stock_* : etf_*`
    # 选快照列，所以这里传的其实是「**不是股票**」的标记。原先借用
    # `INDEX_CN`；ETF 2026-09 起已是独立类型，直接用 `ETF_CN` 更直白。
    stock_cn_snapshot_daily = GQ_stock_a_spot_em(
        market_type=MARKET_TYPE.ETF_CN,
        verbose=False,
    )

    return stock_cn_snapshot_daily


def GQ_fetch_stock_moneyflow(
        code: str,
        start: str,
        end: str,
        offset: str = '',
        format: str = 'pd',
        collections=DATABASE_GolemQ.stock_moneyflow,
        verbose: bool = True) -> pd.DataFrame:
    """'获取股票资金流向'

    Returns:
        [type] -- [description]
    """

    start = str(start)[0:10]
    end = str(end)[0:10]
    # code= [code] if isinstance(code,str) else code

    # code checking
    code = GQ_util_code_tolist(code)
    if GQ_util_date_valid(end):
        cursor = collections.find({
                'code': {
                    '$in': code
                },
                "date_stamp": {
                        "$lte": GQ_util_date_stamp(end) if int(pd.to_datetime(end).to_pydatetime().timestamp()) <= GQ_util_date_stamp(end) else int(pd.to_datetime(end).to_pydatetime().timestamp()),
                        "$gte": GQ_util_date_stamp(start)
                }
            },
            {"_id": 0},
            batch_size=10000)
        # res=[QA_util_dict_remove_key(data, '_id') for data in cursor]

        res = pd.DataFrame([item for item in cursor])
        try:
            res = res.assign(
                date=pd.to_datetime([datestring[0:10] for _, datestring in res.date.items()])).drop_duplicates(([
                    'date', 'code'])).set_index([
                        'date', 'code'],
                        drop=False)
            res = res.loc[:, [
                u"主力净流入-净额",
                u"小单净流入-净额",
                u"中单净流入-净额",
                u"大单净流入-净额",
                u"超大单净流入-净额",
                u"主力净流入-净占比",
                u"小单净流入-净占比",
                u"中单净流入-净占比",
                u"大单净流入-净占比",
                u"超大单净流入-净占比",
                u"收盘价",
                u"涨跌幅",]]
        except Exception as e:
            if (verbose):
                print(u'GQ_fetch_stock_moneyflow Code:{}'.format(code), e)
            
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
            print("GolemQ Error GQ_fetch_stock_moneyflow format parameter %s is none of  \"P, p, pandas, pd , json, dict , n, N, numpy, list, l, L, !\" " % format)
            return None
    else:
        print(
            'GolemQ Error GQ_fetch_stock_moneyflow data parameter start=%s end=%s is not right' % (
                start,
                end))
    return None
