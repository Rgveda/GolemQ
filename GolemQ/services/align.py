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
import os
import time
import numpy as np
import pandas as pd
import QUANTAXIS as QA
try:
    from GolemQ.core.constants import (
        AKA,
        FIELD as FLD,
    )
except Exception:
    # 自适应脚本的根目录，默认在GolemQ平级目录下，
    # 如果不在，则自动调整PYTHONPATH路径。
    import sys
    p = os.path.abspath(r'..\..\.')
    sys.path.insert(1, p)
from GolemQ.core.constants import (
    AKA,
    FIELD as FLD,
)
from GolemQ.core.base import (
    GQ_util_get_last_day,
)
from GolemQ.markets.StockCN.base import (
    resample_features_frequency,
)
from GolemQ.core.settings import (
    DATABASE as DATABASE_GolemQ,
)
from GolemQ.services.features import (
    GQ_save_metadata_reality,
    GQ_save_daily_metadata_reality,
    GQ_fetch_daily_metadata_reality,
    GQ_fetch_hourly_metadata_reality,
    GQ_fix_daily_metadata,
    GQ_remove_daily_metadata,
    GQ_update_daily_metadata,
    GQ_update_hourly_metadata,
    GQ_remove_hourly_metadata,
    GQ_move_hourly_metadata,
    GQ_save_stock_valuation,
)
from GolemQ.core.settings import (
    DATABASE as DATABASE_GolemQ,
)
import traceback
import json
import pymongo
from pymongo import UpdateOne
from GolemQ.models.alias import (
    LTT,
)
import pymongo
from GolemQ.core.gq_logging import GQ_util_log_info
from GolemQ.core.symbol import GQ_util_code_tolist
from GolemQ.markets.StockCN.date_utils import (
    GQ_util_date_valid,
    GQ_util_date_stamp,
    GQ_util_time_stamp,
)
from GolemQ.core.preprocessing import (
    GQ_util_to_json_from_pandas,
)
from GolemQ.core.base import (
    GQ_util_get_last_day,
)


def save_symbol_checkpoint_log(
    log_context,
    catalog=None,
    collections=DATABASE_GolemQ.symbol_checkpoints_log
):
    """
    保存当天股票走势分组聚类
    :param collections: 数据库 collection 存储名称
    :return:
    """
    coll = collections
    coll.create_index([(AKA.FROZEN,
                    pymongo.ASCENDING),],
                    unique=False)
    coll.create_index([(AKA.CATALOG,
                    pymongo.ASCENDING),
                    ("date_stamp",
                    pymongo.ASCENDING),],
                    unique=False)
    coll.create_index([(AKA.CODE,
                    pymongo.ASCENDING),
                    (AKA.CATALOG,
                    pymongo.ASCENDING),
                    ("date_stamp",
                    pymongo.ASCENDING),],
                    unique=True)

    try:
        # GMT+0 String 转换为 UTC Timestamp
        log_context["date_stamp"] = log_context[AKA.DATE].astype(np.int64) // 10 ** 9
        log_context['created_at'] = int(time.mktime(dt.now().utctimetuple()))
        if (AKA.FROZEN in log_context.columns):
            log_context[AKA.FROZEN] = log_context[AKA.FROZEN].astype(np.int64) // 10 ** 9

        log_context[AKA.DATE] = log_context[AKA.DATE].dt.strftime('%Y-%m-%d')

        # print(massive_views.columns)
        each_day = sorted(log_context.index.get_level_values(level=0).unique())
        data = log_context
        start = log_context[AKA.DATE].head(1).item()
        end = log_context[AKA.DATE].tail(1).item()
        codelist = sorted(log_context[AKA.CODE].unique())
        catalog = log_context[AKA.CATALOG].tail(1).item() if AKA.CATALOG in log_context.columns else catalog
        if (len(log_context)==1):
            query_id = {
                AKA.CODE: codelist[0],
                AKA.CATALOG: catalog,
                "date_stamp": log_context["date_stamp"].tail(1).item(),
            }
        else:
            query_id = {
                AKA.CODE: {
                    '$in': codelist
                },
                AKA.CATALOG: catalog,
                "date_stamp":
                        {
                            "$gte": int(GQ_util_date_stamp(start)),
                            "$lte": int(GQ_util_date_stamp(end)),
                        },
                }
        #print(query_id)
        refcount = coll.count_documents(query_id)

        if refcount > 0:
            if (len(data) > 1):
                # 原设计删掉前一交易日后的重复数据，考虑到除权计算，改为全部更新
                refcount = coll.count_documents(query_id)
                coll.delete_many(query_id)

                # 作为差量更新，只更新最后一天的数据
                #data = data.query("date_stamp>={}".format(GQ_util_time_stamp(end))).copy()
                if (len(data) == 0):
                    print(u'{} Data len equals zero.'.format(codelist))
                data_json = GQ_util_to_json_from_pandas(data)
                try:
                    coll.insert_many(data_json)
                except Exception as e:
                    print(u'{} Duplicate key error.'.format(codelist))
                    print(e)
            else:
                # 持续接收行情，更新记录
                if ('created_at' in data.columns):
                    data.drop('created_at', axis=1, inplace=True)
                data_json = GQ_util_to_json_from_pandas(data)
                coll.replace_one(query_id, data_json[0])
        else:
            # 新 tick，插入记录
            try:
                data_json = GQ_util_to_json_from_pandas(data)
                coll.insert_many(data_json)
            except Exception as e:
                print(query_id)
                data_slice = log_context.loc[(each_day[-1], slice(None)), :]
                data_slice_json = GQ_util_to_json_from_pandas(data_slice)
                coll.insert_many(data_slice_json)
    except Exception as e:
        traceback.print_exception(type(e), e, sys.exc_info()[2])
        print(e)
        return None
    # print(u'Code: {} 保存review缓存成功!'.format(codelist))
    return log_context


def symbol_checkpoint_log(
    logs:str, 
    symbol=None, 
    FrozenExpired=pd.to_datetime(GQ_util_get_last_day())+timedelta(days=-1),
    length=0,
    missing_kline_index=None,
    frequency=None,
    catalog='CHECKPOINT',
    collections=DATABASE_GolemQ.symbol_checkpoints_log ):
    
    symbol_checkpoints_log={AKA.CODE:symbol,
                            AKA.DATE:pd.to_datetime(GQ_util_get_last_day()),
                            AKA.CATALOG:catalog,
                            'logs':logs,
                            AKA.LENGTH:length,
                            AKA.FROZEN:FrozenExpired,}
    
    if (frequency is not None) and \
        (frequency in ['1min', '5min', '15min', '30min', '60min', '120min']):
        symbol_checkpoints_log['type']=frequency

    if (missing_kline_index is not None) and \
        (len(missing_kline_index)>0):
        try:
            symbol_checkpoints_log['missing_kline_index']=pd.to_datetime(missing_kline_index).strftime("%Y-%m-%d %H:%M:%S")
        except AttributeError as e:
            # 对比检查baostock Kline 数据，确保它的K线数据正常。
            traceback.print_exc()
            symbol_checkpoints_log['missing_kline_index']=pd.to_datetime(missing_kline_index).strftime("%Y-%m-%d %H:%M:%S")
        except:
            # 出现异常说明 15分钟线可能是多余的
            print(missing_kline_index)
            symbol_checkpoints_log['missing_kline_index']=pd.to_datetime(missing_kline_index).strftime("%Y-%m-%d %H:%M:%S")

    try:
        # if (missing_kline_index is not None) and \
        #     (len(missing_kline_index)>0):
        # log_context=pd.DataFrame([symbol_checkpoints_log], 
        #                         index=[0])
        # else:
        log_context=pd.DataFrame(symbol_checkpoints_log, 
                                index=[0])
        log_context=log_context.set_index([AKA.DATE,
                                        AKA.CODE], 
                                        drop=False)
    except Exception as e:
        print('\n symbol_checkpoint_log() got an Error:', e, symbol_checkpoints_log)
        traceback.print_exc()
    
    save_symbol_checkpoint_log(
        log_context=log_context,
        catalog=catalog,
        collections=collections)

    return 


def GQ_fetch_checkpoint_symbols(
    FrozenExpired: str = None,
    collections=DATABASE_GolemQ.symbol_checkpoints_log,
) -> pd.DataFrame:
    """
    保存当天股票走势分组聚类
    :param collections: 数据库 collection 存储名称
    :return:
    """
    coll = collections
    coll.create_index([(AKA.FROZEN,
                    pymongo.ASCENDING),],
                    unique=False)
    coll.create_index([(AKA.CATALOG,
                    pymongo.ASCENDING),
                    ("date_stamp",
                    pymongo.ASCENDING),],
                    unique=False)
    coll.create_index([(AKA.CODE,
                    pymongo.ASCENDING),
                    (AKA.CATALOG,
                    pymongo.ASCENDING),
                    ("date_stamp",
                    pymongo.ASCENDING),],
                    unique=True)

    FrozenExpired = str(FrozenExpired)[0:19]

    if GQ_util_date_valid(str(FrozenExpired)[0:10]):
        cursor = collections.find({
                AKA.FROZEN:
                    {
                    "$gte": GQ_util_date_stamp(FrozenExpired)
                    }
            },
            {"_id": 0},
            batch_size=10000)
        #res=[QA_util_dict_remove_key(data, '_id') for data in cursor]

        res = pd.DataFrame([item for item in cursor])
        try:
            res = res.assign(date=pd.to_datetime(res.date)).drop_duplicates(([AKA.DATE,
                                AKA.CODE])).set_index([AKA.DATE,
                                AKA.CODE],
                                    drop=False)
        except Exception as e:
            if (len(res.columns)>0):
                print(u'\nGQ_fetch_checkpoint_symbols:{} Error:'.format(FrozenExpired), e, res.columns)
            res = None
            pass
            
        return res
    else:
        print(f'\nQA Error GQ_fetch_checkpoint_symbols data parameter FrozenExpired={FrozenExpired} is not right')
    return None


def calc_stock_hourly_kline_align(
    code:str,
    freq: str = '60min',
    market_type = QA.MARKET_TYPE.STOCK_CN,
    collections=DATABASE_GolemQ.stock_reality_feat_60min,
):
    start_date=pd.to_datetime(GQ_util_get_last_day()).tz_localize('Asia/Shanghai')-timedelta(days=1680)
    end_date=pd.to_datetime(GQ_util_get_last_day()).tz_localize('Asia/Shanghai')

    return stock_hourly_feats


def calc_stock_metadata_missing_queries(
    features: pd.DataFrame = None,
    verbose: bool = False,
    collections=DATABASE_GolemQ.stock_wencai_metadata_missing_queries,
):

    # 获取包含空值的完整行数据
    null_rows = features[
        (features[LTT.QUADRANT_LEVERAGE_MACD_TIMING_LAG_DUMMY] > 1e-12) &
        (features[LTT.QUADRANT_LEVERAGE_MACD_TIMING_LAG_DUMMY] < features[FLD.MACD_CROSS_SX_BEFORE_MAJOR]) &
        (features[LTT.QUADRANT_LEVERAGE_MACD_LEADIN_UB] > features[AKA.CLOSE]) &
        ((features[AKA.DDE_NUMBER_OF_INDIVIDUAL_INVESTORS].isnull()) |
        (features[AKA.HOT_RANK].isnull()) |
        (features[AKA.BUY_SIGNAL].isnull()) |
        (features[AKA.PCT_RANKING].isnull()))
    ]

    # 生成查询字符串列表，并去重
    query_strings = []
    seen_dates = set()

    for idx in null_rows.index:
        # 提取日期和股票代码
        datetime_str = idx[0]  # datetime部分
        code_symbol = idx[1]   # 股票代码部分

        # 格式化日期（假设需要转换为YYYY-MM-DD格式）
        if hasattr(datetime_str, 'strftime'):
            day_slice = datetime_str.strftime('%Y-%m-%d')
        else:
            day_slice = str(datetime_str).split(' ')[0]  # 如果是字符串，取日期部分

        # 创建唯一标识符（日期+股票代码）
        unique_key = f"{day_slice}_{code_symbol}"

        # 如果这个日期+股票代码组合还没见过，则添加
        if unique_key not in seen_dates:
            seen_dates.add(unique_key)

            # 生成查询字符串
            query_string = '{}日{}的DDE散户数量,{}日{}的个股热度排名,{}日{}的个股热度,{}日{}的换手率,{}日{}的资金流向,{}日{}的支撑压力解读'.format(
                day_slice, code_symbol,
                day_slice, code_symbol,
                day_slice, code_symbol,
                day_slice, code_symbol,
                day_slice, code_symbol,
                day_slice, code_symbol
            )

            query_strings.append({
                AKA.DATE: pd.to_datetime(day_slice), 
                AKA.CODE: code_symbol,
                'query': query_string,
                # 'acquired': 0,
                # 'retried': 0,
            })

    stock_missing_metadata_pd = pd.DataFrame(
        query_strings).set_index(
        [AKA.DATE, AKA.CODE], drop=False)

    GQ_save_stock_valuation(
        stock_missing_metadata_pd,
        collections=collections,
        verbose=verbose,)

    return stock_missing_metadata_pd


def kline_missing_checkpoints(
        features: pd.DataFrame,
        ohlc_data: pd.DataFrame,
        checkpoint_remark: str = '',
        verbose=False,):
    """
    检查 kline 是否缺失，CLOSE 字段 是否为np.nan
    """
    join_columns = list(set([
        AKA.OPEN,
        AKA.CLOSE,
        AKA.LOW,
        AKA.HIGH,
        AKA.VOLUME,
        AKA.AMOUNT,]))

    if (AKA.NAME in features.columns):
        missing_name_idx = features.index.difference(
            features.dropna(
                subset=[AKA.NAME,],
                axis=0,
                how="all").index).get_level_values(level=0)
        missing_name = features.dropna(
            subset=[AKA.NAME],
            axis=0,
            how="all",)[AKA.NAME].tail(1).item()
        
        if (len(missing_name_idx)>0) and \
            ((GQ_util_get_last_day()-missing_name_idx[-1])<timedelta(days=21)):
            features.loc[missing_name_idx,
                        AKA.NAME]=missing_name

    if (features[AKA.CLOSE].tail(9).isnull().values.any()==True) or \
        (features[AKA.CLOSE].head(9).isnull().values.any()==True):
        missing_kline_idx=features.index.difference(features.dropna(subset=[AKA.CLOSE],
                                                                    axis=0,
                                                                    how="all",).index)
        try:
            features.loc[missing_kline_idx,
                        join_columns]=ohlc_data.loc[missing_kline_idx, 
                                                    join_columns]
        except Exception as e:
            codelist=features.index.get_level_values(level=1)[0]
            log_msg='Code:{} Ohlc data missing: {}'.format(codelist, e)
            try:
                symbol_checkpoint_log(
                    logs=log_msg, 
                    symbol=codelist, 
                    length=len(ohlc_data) if (ohlc_data is not None) else None,
                    FrozenExpired=pd.to_datetime(GQ_util_get_last_day())+timedelta(days=9),
                    catalog=AKA.CHECKPOINT,)
            except Exception:
                print(u'\n{}'.format(log_msg))
                if (len(missing_kline_idx.difference(ohlc_data.index))>0) and \
                    (len(missing_kline_idx.difference(ohlc_data.index))<21):
                    traceback.print_exc()
                    pass
                else:
                    traceback.print_exc()
            finally:
                return features
        if (len(checkpoint_remark)>0):
            code = features.index.get_level_values(level=1)[0]
            log_msg='Code: {} Missing kline {} at checkpoint: {}'.format(code,
                                                                len(missing_kline_idx),
                                                                checkpoint_remark)
            try:
                symbol_checkpoint_log(logs=log_msg, 
                                symbol=code, 
                                length=len(ohlc_data) if (ohlc_data is not None) else None,
                                catalog=AKA.CHECKPOINT,)
            except Exception:
                print(u'\n{}'.format(log_msg))
                if (len(missing_kline_idx.difference(ohlc_data.index))>0) and \
                    (len(missing_kline_idx.difference(ohlc_data.index))<21):
                    pass
                else:
                    traceback.print_exc()
            finally:
                pass
        if (verbose):
            print(features[[AKA.OPEN,
                        AKA.CLOSE,
                        AKA.LOW,
                        AKA.HIGH,
                        AKA.VOLUME,
                        AKA.AMOUNT,]].tail(20))
            
    return features




                
