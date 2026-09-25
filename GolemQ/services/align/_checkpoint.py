# coding:utf-8
"""检查点日志（checkpoint log）的写入与查询。

从 `services/align.py` 拆出（该文件 492 行，超 `services/` 300 行上限）。
**拆分为纯机械搬运，不改行为** —— 除一处例外，见下。

三个函数的职责
==============
``save_symbol_checkpoint_log``   写：建索引 + 按 `(code, catalog, date_stamp)` upsert
``symbol_checkpoint_log``        组装一行日志后交给上面那个写（**唯一有外部调用者的**）
``GQ_fetch_checkpoint_symbols``  按 `FrozenExpired` 区间查回

⚠️ 搬运时**发现并修掉**的一处（唯一的行为变更）
==============================================
原 `align.py:195` 在 `except` 里调 `sys.exc_info()`，但 `import sys` 只写在
`:43` 一个**永不执行的 `except` 分支**里（那个 `try` 的 import 实际总成功）。
于是 `sys` 在模块级**根本没定义** → `save_symbol_checkpoint_log` 的
**错误处理路径自己会抛 `NameError`**，把原始异常吞掉。
本模块在模块级正常 `import sys`，使该错误路径如它所愿地打印回溯。

（原文件的 `try/except` 还有个更小的问题：它在 `except` 里改 `sys.path` 之后，
`:46` 又**无条件**import 同一个东西 —— 兜底形同虚设。搬运时已删掉这层无意义包装。）
"""
from __future__ import annotations

import sys
import time
import traceback
from datetime import (
    datetime as dt,
    timedelta,
)

import numpy as np
import pandas as pd
import pymongo

from GolemQ.core.constants import (
    AKA,
    FIELD as FLD,
)
from GolemQ.core.preprocessing import (
    GQ_util_to_json_from_pandas,
)
from GolemQ.core.settings import (
    DATABASE as DATABASE_GolemQ,
)
from GolemQ.markets.StockCN.date_utils import (
    GQ_util_date_stamp,
    GQ_util_date_valid,
    GQ_util_get_last_day,
)

__all__ = [
    'save_symbol_checkpoint_log',
    'symbol_checkpoint_log',
    'GQ_fetch_checkpoint_symbols',
]


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
