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
import datetime
import time
import numpy as np
import pandas as pd
import pymongo

try:
    import QUANTAXIS as QA
    from QUANTAXIS.QAUtil import (
        QASETTING, 
        DATABASE, 
        QA_util_date_stamp,
        QA_util_date_valid,
        QA_util_log_info, 
        QA_util_to_json_from_pandas,
        QA_util_dict_remove_key,
        QA_util_code_tolist,
    )
    from QUANTAXIS.QAUtil.QAParameter import ORDER_DIRECTION
    from QUANTAXIS.QAData.QADataStruct import (
        QA_DataStruct_Index_min, 
        QA_DataStruct_Index_day, 
        QA_DataStruct_Stock_day, 
        QA_DataStruct_Stock_min
    )
    from QUANTAXIS.QAUtil.QADate_Adv import (
        QA_util_timestamp_to_str,
        QA_util_datetime_to_Unix_timestamp,
        QA_util_print_timestamp
    )
except:
    print('PLEASE run "pip install QUANTAXIS" before call GolemQ.fetch.portfolio modules')
    pass

from GolemQ.utils.parameter import (
    AKA, 
    INDICATOR_FIELD as FLD, 
    TREND_STATUS as ST, 
    FEATURES as FTR,)


def GQSignal_fetch_position_singal_day(start,
                                       end,
                                       frequence='day',
                                       market_type=QA.MARKET_TYPE.STOCK_CN,
                                       portfolio='myportfolio',
                                       getting_trigger=True,
                                       format='numpy',
                                       ui_log=None, 
                                       ui_progress=None):
    """
    '获取特定买入信号的股票指标日线'

    Keyword Arguments:
        client {[type]} -- [description] (default: {DATABASE})
    """
    start = str(start)[0:10]
    end = str(end)[0:10]
    #code= [code] if isinstance(code,str) else code

    client = QASETTING.client[AKA.SYSTEM_NAME]
    # 同时写入横表和纵表，减少查询困扰
    #coll_day = client.get_collection(
    #        'indices_{}'.format(datetime.date.today()))
    try:
        if (market_type == QA.MARKET_TYPE.STOCK_CN):
            #coll_indices = client.stock_cn_indices_min
            coll_indices = client.get_collection('stock_cn_indices_{}'.format(portfolio))
        elif (market_type == QA.MARKET_TYPE.INDEX_CN):
            #coll_indices = client.index_cn_indices_min
            coll_indices = client.get_collection('index_cn_indices_{}'.format(portfolio))
        elif (market_type == QA.MARKET_TYPE.FUND_CN):
            #coll_indices = client.future_cn_indices_min
            coll_indices = client.get_collection('fund_cn_indices_{}'.format(portfolio))
        elif (market_type == QA.MARKET_TYPE.FUTURE_CN):
            #coll_indices = client.future_cn_indices_min
            coll_indices = client.get_collection('future_cn_indices_{}'.format(portfolio))
        elif (market_type == QA.MARKET_TYPE.CRYPTOCURRENCY):
            #coll_indices = client.cryptocurrency_indices_min
            coll_indices = client.get_collection('cryptocurrency_indices_{}'.format(portfolio))
        else:
            QA_util_log_info('WTF IS THIS! \n ', ui_log=ui_log)
            return False
    except Exception as e:
        QA_util_log_info(e)
        QA_util_log_info('WTF IS THIS! \n ', ui_log=ui_log)
        return False

    if QA_util_date_valid(end):
        if (getting_trigger):
            # 按“信号”查询
            cursor = coll_indices.find({
                    ST.TRIGGER_R5: {
                        '$gt': 0
                    },
                    "date_stamp":
                        {
                            "$lte": QA_util_date_stamp(end),
                            "$gte": QA_util_date_stamp(start)
                        }
                },
                {"_id": 0},
                batch_size=10000)
        else:
            # 按“持有状态”查询
            cursor = coll_indices.find({
                    ST.POSITION_R5: {
                        '$gt': 0
                    },
                    "date_stamp":
                        {
                            "$lte": QA_util_date_stamp(end),
                            "$gte": QA_util_date_stamp(start)
                        }
                },
                {"_id": 0},
                batch_size=10000)

        #res=[QA_util_dict_remove_key(data, '_id') for data in cursor]
        res = pd.DataFrame([item for item in cursor])
        #print(len(res), start, end)
        try:
            res = res.assign(date=pd.to_datetime(res.date)).drop_duplicates((['date',
                                'code'])).set_index(['date',
                                'code'],
                                    drop=False)
        except:
            res = None

        if (res is not None):
            try:
                codelist = QA.QA_fetch_stock_name(res[AKA.CODE].tolist())
                res['name'] = res.apply(lambda x:codelist.at[x.get(AKA.CODE), 'name'], axis=1)
            except:
                res['name'] = res['code']
        if format in ['P', 'p', 'pandas', 'pd']:
            return res
        elif format in ['json', 'dict']:
            return QA_util_to_json_from_pandas(res)
        # 多种数据格式
        elif format in ['n', 'N', 'numpy']:
            return numpy.asarray(res)
        elif format in ['list', 'l', 'L']:
            return numpy.asarray(res).tolist()
        else:
            print("QA Error GQSignal_fetch_position_singal_day format parameter %s is none of  \"P, p, pandas, pd , json, dict , n, N, numpy, list, l, L, !\" " % format)
            return None
    else:
        QA_util_log_info('QA Error GQSignal_fetch_position_singal_day data parameter start=%s end=%s is not right' % (start,
               end))


def GQSignal_fetch_mainfest_singal_day(start,
                                       end,
                                       frequence='day',
                                       market_type=QA.MARKET_TYPE.STOCK_CN,
                                       portfolio='myportfolio',
                                       getting_trigger=True,
                                       format='numpy',
                                       ui_log=None, 
                                       ui_progress=None):
    """
    '获取主升浪买入信号的股票指标日线'

    Keyword Arguments:
        client {[type]} -- [description] (default: {DATABASE})
    """
    start = str(start)[0:10]
    end = str(end)[0:10]
    #code= [code] if isinstance(code,str) else code

    client = QASETTING.client[AKA.SYSTEM_NAME]
    # 同时写入横表和纵表，减少查询困扰
    #coll_day = client.get_collection(
    #        'indices_{}'.format(datetime.date.today()))
    try:
        if (market_type == QA.MARKET_TYPE.STOCK_CN):
            #coll_indices = client.stock_cn_indices_min
            coll_indices = client.get_collection('stock_cn_indices_{}'.format(portfolio))
        elif (market_type == QA.MARKET_TYPE.INDEX_CN):
            #coll_indices = client.index_cn_indices_min
            coll_indices = client.get_collection('index_cn_indices_{}'.format(portfolio))
        elif (market_type == QA.MARKET_TYPE.FUND_CN):
            #coll_indices = client.future_cn_indices_min
            coll_indices = client.get_collection('fund_cn_indices_{}'.format(portfolio))
        elif (market_type == QA.MARKET_TYPE.FUTURE_CN):
            #coll_indices = client.future_cn_indices_min
            coll_indices = client.get_collection('future_cn_indices_{}'.format(portfolio))
        elif (market_type == QA.MARKET_TYPE.CRYPTOCURRENCY):
            #coll_indices = client.cryptocurrency_indices_min
            coll_indices = client.get_collection('cryptocurrency_indices_{}'.format(portfolio))
        else:
            QA_util_log_info('WTF IS THIS! \n ', ui_log=ui_log)
            return False
    except Exception as e:
        QA_util_log_info(e)
        QA_util_log_info('WTF IS THIS! \n ', ui_log=ui_log)
        return False

    if QA_util_date_valid(end):
        if (getting_trigger):
            # 按主升浪买入“信号”查询
            cursor = coll_indices.find({ 
                '$and': [{ '$or': [{
                            FLD.BOOTSTRAP_COMBO_TIMING_LAG:{
                                '$gt':0
                                } 
                            }, 
                        {
                            FLD.BOOTSTRAP_GROUND_ZERO_MINOR_TIMING_LAG:{
                                '$gt':0
                                }
                            }]
                     },
                    #{ FLD.BOOTSTRAP_COMBO_RETURNS:{
                    #    '$gt':0.00618
                    #    }
                    # },
                    { '$or': [{ FLD.BOOTSTRAP_COMBO_RETURNS:{
                        '$gt':-0.0927
                        } 
                     }, { FLD.BOOTSTRAP_COMBO_MINOR_RETURNS:{
                        '$gt':-0.0927
                        } 
                     }]},
                    { '$or': [{ ST.TRIGGER_R5:{'$gt':0}}, { ST.TRIGGER_RPS:{'$gt':0}}]},
                    { "date_stamp":
                        {
                            "$lte": QA_util_date_stamp(end),
                            "$gte": QA_util_date_stamp(start)
                        }},]
                },
                {"_id": 0},
                batch_size=10000)
        else:
            # 按“持有状态”查询
            cursor = coll_indices.find({
                    ST.POSITION_R5: {
                        '$gt': 0
                    },
                    "date_stamp":
                        {
                            "$lte": QA_util_date_stamp(end),
                            "$gte": QA_util_date_stamp(start)
                        }
                },
                {"_id": 0},
                batch_size=10000)

        #res=[QA_util_dict_remove_key(data, '_id') for data in cursor]
        res = pd.DataFrame([item for item in cursor])
        #print(len(res), start, end)
        try:
            res = res.assign(date=pd.to_datetime(res.date)).drop_duplicates((['date',
                                'code'])).set_index(['date',
                                'code'],
                                    drop=False)
        except:
            res = None

        if (res is not None):
            try:
                codelist = QA.QA_fetch_stock_name(res[AKA.CODE].tolist())
                res['name'] = res.apply(lambda x:codelist.at[x.get(AKA.CODE), 'name'], axis=1)
            except:
                res['name'] = res['code']
        if format in ['P', 'p', 'pandas', 'pd']:
            return res
        elif format in ['json', 'dict']:
            return QA_util_to_json_from_pandas(res)
        # 多种数据格式
        elif format in ['n', 'N', 'numpy']:
            return numpy.asarray(res)
        elif format in ['list', 'l', 'L']:
            return numpy.asarray(res).tolist()
        else:
            print("QA Error GQSignal_fetch_position_singal_day format parameter %s is none of  \"P, p, pandas, pd , json, dict , n, N, numpy, list, l, L, !\" " % format)
            return None
    else:
        QA_util_log_info('QA Error GQSignal_fetch_position_singal_day data parameter start=%s end=%s is not right' % (start,
               end))


def GQSignal_fetch_bootstrap_singal_day(start,
                                       end,
                                       frequence='day',
                                       market_type=QA.MARKET_TYPE.STOCK_CN,
                                       portfolio='myportfolio',
                                       getting_trigger=True,
                                       format='numpy',
                                       ui_log=None, 
                                       ui_progress=None):
    """
    '获取主升浪买入信号的股票指标日线'

    Keyword Arguments:
        client {[type]} -- [description] (default: {DATABASE})
    """
    start = str(start)[0:10]
    end = str(end)[0:10]
    #code= [code] if isinstance(code,str) else code

    client = QASETTING.client[AKA.SYSTEM_NAME]
    # 同时写入横表和纵表，减少查询困扰
    #coll_day = client.get_collection(
    #        'indices_{}'.format(datetime.date.today()))
    try:
        if (market_type == QA.MARKET_TYPE.STOCK_CN):
            #coll_indices = client.stock_cn_indices_min
            coll_indices = client.get_collection('stock_cn_indices_{}'.format(portfolio))
        elif (market_type == QA.MARKET_TYPE.INDEX_CN):
            #coll_indices = client.index_cn_indices_min
            coll_indices = client.get_collection('index_cn_indices_{}'.format(portfolio))
        elif (market_type == QA.MARKET_TYPE.FUND_CN):
            #coll_indices = client.future_cn_indices_min
            coll_indices = client.get_collection('fund_cn_indices_{}'.format(portfolio))
        elif (market_type == QA.MARKET_TYPE.FUTURE_CN):
            #coll_indices = client.future_cn_indices_min
            coll_indices = client.get_collection('future_cn_indices_{}'.format(portfolio))
        elif (market_type == QA.MARKET_TYPE.CRYPTOCURRENCY):
            #coll_indices = client.cryptocurrency_indices_min
            coll_indices = client.get_collection('cryptocurrency_indices_{}'.format(portfolio))
        else:
            QA_util_log_info('WTF IS THIS! \n ', ui_log=ui_log)
            return False
    except Exception as e:
        QA_util_log_info(e)
        QA_util_log_info('WTF IS THIS! \n ', ui_log=ui_log)
        return False

    if QA_util_date_valid(end):
        if (getting_trigger):
            # 按主升浪买入“信号”查询
            cursor = coll_indices.find({ 
                '$and': [
                    { '$or' :[ 
                        { ST.CLUSTER_GROUP_TOWARDS:{'$lt':0} },
                        { ST.CLUSTER_GROUP_TOWARDS_MINOR:{'$lt':0} },
                            ]},
                    { '$or' :[ 
                        { FTR.UPRISING_RAIL_TIMING_LAG:{'$gt':0} },
                        { FLD.BOOTSTRAP_GROUND_ZERO_TIMING_LAG:{'$gt':0} },
                            ]},
                    { ST.BOOTSTRAP_GROUND_ZERO:{'$gt':0} },
                    { "date_stamp":
                        {
                            "$lte": QA_util_date_stamp(end),
                            "$gte": QA_util_date_stamp(start)
                        }},]
                },
                {"_id": 0},
                batch_size=10000)
        else:
            # 按“持有状态”查询
            cursor = coll_indices.find({
                    ST.POSITION_R5: {
                        '$gt': 0
                    },
                    "date_stamp":
                        {
                            "$lte": QA_util_date_stamp(end),
                            "$gte": QA_util_date_stamp(start)
                        }
                },
                {"_id": 0},
                batch_size=10000)

        #res=[QA_util_dict_remove_key(data, '_id') for data in cursor]
        res = pd.DataFrame([item for item in cursor])
        #print(len(res), start, end)
        try:
            res = res.assign(date=pd.to_datetime(res.date)).drop_duplicates((['date',
                                'code'])).set_index(['date',
                                'code'],
                                    drop=False)
        except:
            res = None

        if (res is not None):
            try:
                codelist = QA.QA_fetch_stock_name(res[AKA.CODE].tolist())
                res['name'] = res.apply(lambda x:codelist.at[x.get(AKA.CODE), 'name'], axis=1)
            except:
                res['name'] = res['code']
        if format in ['P', 'p', 'pandas', 'pd']:
            return res
        elif format in ['json', 'dict']:
            return QA_util_to_json_from_pandas(res)
        # 多种数据格式
        elif format in ['n', 'N', 'numpy']:
            return numpy.asarray(res)
        elif format in ['list', 'l', 'L']:
            return numpy.asarray(res).tolist()
        else:
            print("QA Error GQSignal_fetch_position_singal_day format parameter %s is none of  \"P, p, pandas, pd , json, dict , n, N, numpy, list, l, L, !\" " % format)
            return None
    else:
        QA_util_log_info('QA Error GQSignal_fetch_position_singal_day data parameter start=%s end=%s is not right' % (start,
               end))


def GQSignal_fetch_code_singal_day(code,
                              start,
                              end,
                              frequence='day',
                              market_type=QA.MARKET_TYPE.STOCK_CN,
                              portfolio='myportfolio', 
                              format='numpy',
                              ui_log=None, 
                              ui_progress=None):
    """
    获取指定代码股票日线指标/策略信号数据

    Keyword Arguments:
        client {[type]} -- [description] (default: {DATABASE})
    """
    start = str(start)[0:10]
    end = str(end)[0:10]
    #code= [code] if isinstance(code,str) else code

    client = QASETTING.client[AKA.SYSTEM_NAME]
    # 同时写入横表和纵表，减少查询困扰
    #coll_day = client.get_collection(
    #        'indices_{}'.format(datetime.date.today()))
    try:
        if (market_type == QA.MARKET_TYPE.STOCK_CN):
            #coll_indices = client.stock_cn_indices_min
            coll_indices = client.get_collection('stock_cn_indices_{}'.format(portfolio))
        elif (market_type == QA.MARKET_TYPE.INDEX_CN):
            #coll_indices = client.index_cn_indices_min
            coll_indices = client.get_collection('index_cn_indices_{}'.format(portfolio))
        elif (market_type == QA.MARKET_TYPE.FUND_CN):
            #coll_indices = client.future_cn_indices_min
            coll_indices = client.get_collection('fund_cn_indices_{}'.format(portfolio))
        elif (market_type == QA.MARKET_TYPE.FUTURE_CN):
            #coll_indices = client.future_cn_indices_min
            coll_indices = client.get_collection('future_cn_indices_{}'.format(portfolio))
        elif (market_type == QA.MARKET_TYPE.CRYPTOCURRENCY):
            #coll_indices = client.cryptocurrency_indices_min
            coll_indices = client.get_collection('cryptocurrency_indices_{}'.format(portfolio))
        else:
            QA_util_log_info('WTF IS THIS! \n ', ui_log=ui_log)
            return False
    except Exception as e:
        QA_util_log_info(e)
        QA_util_log_info('WTF IS THIS! \n ', ui_log=ui_log)
        return False

    # code checking
    print(code)
    code = QA_util_code_tolist(code)
    print(code)

    if QA_util_date_valid(end):
        cursor = coll_indices.find({
                'code': {
                    '$in': code
                },
                "date_stamp":
                    {
                        "$lte": QA_util_date_stamp(end),
                        "$gte": QA_util_date_stamp(start)
                    }
            },
            {"_id": 0},
            batch_size=10000)
        #res=[QA_util_dict_remove_key(data, '_id') for data in cursor]

        res = pd.DataFrame([item for item in cursor])
        try:
            res = res.assign(date=pd.to_datetime(res.date)).drop_duplicates((['date',
                                'code'])).set_index(['date',
                                'code'], drop=False)
            res.sort_index(inplace=True)
        except:
            res = None
        if format in ['P', 'p', 'pandas', 'pd']:
            return res
        elif format in ['json', 'dict']:
            return QA_util_to_json_from_pandas(res)
        # 多种数据格式
        elif format in ['n', 'N', 'numpy']:
            return numpy.asarray(res)
        elif format in ['list', 'l', 'L']:
            return numpy.asarray(res).tolist()
        else:
            print("QA Error GQSignal_fetch_singal_day format parameter %s is none of  \"P, p, pandas, pd , json, dict , n, N, numpy, list, l, L, !\" " % format)
            return None
    else:
        QA_util_log_info('QA Error GQSignal_fetch_singal_day data parameter start=%s end=%s is not right' % (start,
               end))


def GQSignal_fetch_block_cunsum_day(codelist,
                                    start,
                                    end,
                                    frequency='day',
                                    market_type=QA.MARKET_TYPE.STOCK_CN,
                                    portfolio='myportfolio', 
                                    format='numpy',
                                    ui_log=None, 
                                    ui_progress=None):
    """
    '获取结构化行情中板块股票指标日线处于上升形态的数量'

    Keyword Arguments:
        client {[type]} -- [description] (default: {DATABASE})
    """
    start = str(start)[0:10]
    end = str(end)[0:10]
    #code= [code] if isinstance(code,str) else code

    client = QASETTING.client[AKA.SYSTEM_NAME]
    # 同时写入横表和纵表，减少查询困扰
    #coll_day = client.get_collection(
    #        'indices_{}'.format(datetime.date.today()))
    try:
        if (market_type == QA.MARKET_TYPE.STOCK_CN):
            #coll_indices = client.stock_cn_indices_min
            coll_indices = client.get_collection('stock_cn_indices_{}'.format(portfolio))
        elif (market_type == QA.MARKET_TYPE.INDEX_CN):
            #coll_indices = client.index_cn_indices_min
            coll_indices = client.get_collection('index_cn_indices_{}'.format(portfolio))
        elif (market_type == QA.MARKET_TYPE.FUND_CN):
            #coll_indices = client.future_cn_indices_min
            coll_indices = client.get_collection('fund_cn_indices_{}'.format(portfolio))
        elif (market_type == QA.MARKET_TYPE.FUTURE_CN):
            #coll_indices = client.future_cn_indices_min
            coll_indices = client.get_collection('future_cn_indices_{}'.format(portfolio))
        elif (market_type == QA.MARKET_TYPE.CRYPTOCURRENCY):
            #coll_indices = client.cryptocurrency_indices_min
            coll_indices = client.get_collection('cryptocurrency_indices_{}'.format(portfolio))
        else:
            QA_util_log_info('WTF IS THIS! \n ', ui_log=ui_log)
            return False
    except Exception as e:
        QA_util_log_info(e)
        QA_util_log_info('WTF IS THIS! \n ', ui_log=ui_log)
        return False

    # code checking
    code = QA_util_code_tolist(codelist)

    if QA_util_date_valid(end):
        cursor = coll_indices.aggregate([{
            '$match': {
                'code': {
                    '$in': code
                },
                'date_stamp': {
                    "$lte": QA_util_date_stamp(end),
                    "$gte": QA_util_date_stamp(start)
                    }
                } 
             },
             {"$group" : {
                 '_id': {
                     'date': "$date",
                     },
                 'date': {
                     '$first': "$date"},
                 'date_stamp': {
                     '$first': "$date_stamp"},
                 FLD.FLU_POSITIVE: {                            # 上涨趋势通道个股家数
                     '$sum': '${}'.format(FLD.FLU_POSITIVE)},  
                 FLD.ML_FLU_TREND: {                            # “技术调整”个股家数
                     '$sum': '${}'.format(FLD.ML_FLU_TREND)}, 
                 ST.BOOTSTRAP_I: {
                     '$sum': { 
                         '$cond': ['${}'.format(ST.BOOTSTRAP_I), 1, 0] }},
                 ST.DEADPOOL: {
                     '$sum': { 
                         '$cond': ['${}'.format(ST.DEADPOOL), 1, 0] }},
                 FLD.FLU_POSITIVE_MASTER: {
                     '$sum': { 
                         '$cond': ['${}'.format(FLD.FLU_POSITIVE_MASTER), 1, 0] }},
                 FLD.FLU_NEGATIVE_MASTER: {
                     '$sum': { 
                         '$cond': ['${}'.format(FLD.FLU_NEGATIVE_MASTER), 1, 0] }},
                 ST.VOLUME_FLOW_BOOST: {
                     '$sum': '${}'.format(ST.VOLUME_FLOW_BOOST)},  
                 ST.VOLUME_FLOW_BOOST_BONUS: {
                     '$sum': { 
                         '$cond': ['${}'.format(ST.VOLUME_FLOW_BOOST_BONUS), 1, 0] }},
                 ST.CLUSTER_GROUP_TOWARDS: {
                     '$sum': '${}'.format(ST.CLUSTER_GROUP_TOWARDS)},
                 FLD.ATR_Stopline: {
                     '$sum': '${}'.format(FLD.ATR_Stopline)},
                 FLD.ATR_SuperTrend: {
                     '$sum': '${}'.format(FLD.ATR_SuperTrend)},
                 FLD.BOLL_RENKO_TIMING_LAG: {
                     '$sum': { 
                         '$cond': [{ '$gte': ['${}'.format(FLD.BOLL_RENKO_TIMING_LAG), 0] }, 1, -1] }},
                         #['${}'.format(FLD.BOLL_RENKO_TIMING_LAG), 1, -1] }},
                 FLD.BOOTSTRAP_GROUND_ZERO_TIMING_LAG: {
                     '$sum': {
                         '$cond': [{ '$gte': ['${}'.format(FLD.BOOTSTRAP_GROUND_ZERO_TIMING_LAG), 0] }, 1, -1] }},
                         #'$cond':
                         #['${}'.format(FLD.BOOTSTRAP_GROUND_ZERO_TIMING_LAG),
                         #1, -1] }},
                 FLD.BOLL_RENKO_MINOR_TIMING_LAG: {
                     '$sum': { 
                         '$cond': [{ '$gte': ['${}'.format(FLD.BOLL_RENKO_MINOR_TIMING_LAG), 0] }, 1, -1] }},
                         #['${}'.format(FLD.BOLL_RENKO_TIMING_LAG), 1, -1] }},
                 FLD.BOOTSTRAP_GROUND_ZERO_MINOR_TIMING_LAG: {
                     '$sum': {
                         '$cond': [{ '$gte': ['${}'.format(FLD.BOOTSTRAP_GROUND_ZERO_MINOR_TIMING_LAG), 0] }, 1, -1] }},
                         #'$cond':
                         #['${}'.format(FLD.BOOTSTRAP_GROUND_ZERO_ZERO__TIMING_LAG),
                         #1, -1] }},
                 FLD.PEAK_OPEN: {
                     '$avg': '${}'.format(FLD.PEAK_OPEN)},
                 FLD.PEAK_OPEN_MINOR: {
                     '$avg': '${}'.format(FLD.PEAK_OPEN_MINOR)},
                 FLD.DEA: {
                     '$avg': '${}'.format(FLD.DEA_NORM)},
                 FLD.MACD: {
                     '$avg': '${}'.format(FLD.MACD_NORM)},
                 FLD.MACD_DELTA: {
                     '$avg': '${}'.format(FLD.MACD_DELTA_NORM)},
                 FLD.MAXFACTOR_CROSS: {
                     '$sum': '${}'.format(FLD.MAXFACTOR_CROSS)},
                 FLD.DUAL_CROSS: {
                     '$sum': '${}'.format(FLD.DUAL_CROSS)},
                 #ST.TRIGGER_R5: {
                 #    '$sum': '${}'.format(ST.TRIGGER_R5)},
                 FLD.TALIB_PATTERNS: {
                     '$sum': '${}'.format(FLD.TALIB_PATTERNS)},
                 FLD.ADXm_Trend: {
                     '$sum': '${}'.format(FLD.ADXm_Trend)},
                 FLD.Volume_HMA5: {
                     '$sum': '${}'.format(FLD.Volume_HMA5)},
                 FLD.RENKO_TREND_S: {
                     '$sum': '${}'.format(FLD.RENKO_TREND_S)},
                 FLD.RENKO_TREND_L: {
                     '$sum': '${}'.format(FLD.RENKO_TREND_L)},
                 'total': {                                   # 全市场个股家数
                     '$sum': 1}                       
                 }
              },
              {'$sort':{"_id.date":1}}])

        try:
            res = pd.DataFrame([QA_util_dict_remove_key(item, '_id') for item in cursor])
            res = res.assign(date=pd.to_datetime(res.date)).drop_duplicates((['date'])).set_index('date',
                                    drop=False)
        except:
            res = None
        if format in ['P', 'p', 'pandas', 'pd']:
            return res
        elif format in ['json', 'dict']:
            return QA_util_to_json_from_pandas(res)
        # 多种数据格式
        elif format in ['n', 'N', 'numpy']:
            return numpy.asarray(res)
        elif format in ['list', 'l', 'L']:
            return numpy.asarray(res).tolist()
        else:
            print("QA Error GQSignal_fetch_block_cunsum_day format parameter %s is none of  \"P, p, pandas, pd , json, dict , n, N, numpy, list, l, L, !\" " % format)
            return None
    else:
        QA_util_log_info('QA Error GQSignal_fetch_block_cunsum_day data parameter start=%s end=%s is not right' % (start,
               end))
    

def GQSignal_fetch_flu_cunsum_day(start,
                                end,
                                frequency='day',
                                market_type=QA.MARKET_TYPE.STOCK_CN,
                                portfolio='myportfolio', 
                                format='numpy',
                                ui_log=None, 
                                ui_progress=None):
    """
    '获取全市场股票指标日线处于上升形态的数量'

    Keyword Arguments:
        client {[type]} -- [description] (default: {DATABASE})
    """
    start = str(start)[0:10]
    end = str(end)[0:10]
    #code= [code] if isinstance(code,str) else code

    client = QASETTING.client[AKA.SYSTEM_NAME]
    # 同时写入横表和纵表，减少查询困扰
    #coll_day = client.get_collection(
    #        'indices_{}'.format(datetime.date.today()))
    try:
        if (market_type == QA.MARKET_TYPE.STOCK_CN):
            #coll_indices = client.stock_cn_indices_min
            coll_indices = client.get_collection('stock_cn_indices_{}'.format(portfolio))
        elif (market_type == QA.MARKET_TYPE.INDEX_CN):
            #coll_indices = client.index_cn_indices_min
            coll_indices = client.get_collection('index_cn_indices_{}'.format(portfolio))
        elif (market_type == QA.MARKET_TYPE.FUND_CN):
            #coll_indices = client.future_cn_indices_min
            coll_indices = client.get_collection('fund_cn_indices_{}'.format(portfolio))
        elif (market_type == QA.MARKET_TYPE.FUTURE_CN):
            #coll_indices = client.future_cn_indices_min
            coll_indices = client.get_collection('future_cn_indices_{}'.format(portfolio))
        elif (market_type == QA.MARKET_TYPE.CRYPTOCURRENCY):
            #coll_indices = client.cryptocurrency_indices_min
            coll_indices = client.get_collection('cryptocurrency_indices_{}'.format(portfolio))
        else:
            QA_util_log_info('WTF IS THIS! \n ', ui_log=ui_log)
            return False
    except Exception as e:
        QA_util_log_info(e)
        QA_util_log_info('WTF IS THIS! \n ', ui_log=ui_log)
        return False

    if QA_util_date_valid(end):
        cursor = coll_indices.aggregate([{
            '$match': {
                'date_stamp': {
                    "$lte": QA_util_date_stamp(end),
                    "$gte": QA_util_date_stamp(start)
                    }
                } 
             },
             {"$group" : {
                 '_id': {
                     'date': "$date",
                     },
                 'date': {
                     '$first': "$date"},
                 'date_stamp': {
                     '$first': "$date_stamp"},
                 FLD.FLU_POSITIVE: {                            # 上涨趋势通道个股家数
                     '$sum': '${}'.format(FLD.FLU_POSITIVE)},  
                 FLD.ML_FLU_TREND: {                            # “技术调整”个股家数
                     '$sum': '${}'.format(FLD.ML_FLU_TREND)}, 
                 ST.BOOTSTRAP_I: {
                     '$sum': { 
                         '$cond': ['${}'.format(ST.BOOTSTRAP_I), 1, 0] }},
                 ST.DEADPOOL: {
                     '$sum': { 
                         '$cond': ['${}'.format(ST.DEADPOOL), 1, 0] }},
                 FLD.FLU_POSITIVE_MASTER: {
                     '$sum': { 
                         '$cond': ['${}'.format(FLD.FLU_POSITIVE_MASTER), 1, 0] }},
                 FLD.FLU_NEGATIVE_MASTER: {
                     '$sum': { 
                         '$cond': ['${}'.format(FLD.FLU_NEGATIVE_MASTER), 1, 0] }},
                 ST.VOLUME_FLOW_BOOST: {
                     '$sum': '${}'.format(ST.VOLUME_FLOW_BOOST)},  
                 ST.VOLUME_FLOW_BOOST_BONUS: {
                     '$sum': { 
                         '$cond': ['${}'.format(ST.VOLUME_FLOW_BOOST_BONUS), 1, 0] }},
                 ST.CLUSTER_GROUP_TOWARDS: {
                     '$sum': '${}'.format(ST.CLUSTER_GROUP_TOWARDS)},
                 FLD.ATR_Stopline: {
                     '$sum': '${}'.format(FLD.ATR_Stopline)},
                 FLD.ATR_SuperTrend: {
                     '$sum': '${}'.format(FLD.ATR_SuperTrend)},
                 FLD.BOLL_RENKO_TIMING_LAG: {
                     '$sum': { 
                         '$cond': [{ '$gte': ['${}'.format(FLD.BOLL_RENKO_TIMING_LAG), 0] }, 1, -1] }},
                         #['${}'.format(FLD.BOLL_RENKO_TIMING_LAG), 1, -1] }},
                 FLD.BOOTSTRAP_GROUND_ZERO_TIMING_LAG: {
                     '$sum': {
                         '$cond': [{ '$gte': ['${}'.format(FLD.BOOTSTRAP_GROUND_ZERO_TIMING_LAG), 0] }, 1, -1] }},
                         #'$cond':
                         #['${}'.format(FLD.BOOTSTRAP_GROUND_ZERO_TIMING_LAG),
                         #1, -1] }},
                 FLD.BOLL_RENKO_MINOR_TIMING_LAG: {
                     '$sum': { 
                         '$cond': [{ '$gte': ['${}'.format(FLD.BOLL_RENKO_MINOR_TIMING_LAG), 0] }, 1, -1] }},
                         #['${}'.format(FLD.BOLL_RENKO_TIMING_LAG), 1, -1] }},
                 FLD.BOOTSTRAP_GROUND_ZERO_MINOR_TIMING_LAG: {
                     '$sum': {
                         '$cond': [{ '$gte': ['${}'.format(FLD.BOOTSTRAP_GROUND_ZERO_MINOR_TIMING_LAG), 0] }, 1, -1] }},
                         #'$cond':
                         #['${}'.format(FLD.BOOTSTRAP_GROUND_ZERO_ZERO__TIMING_LAG),
                         #1, -1] }},
                 FLD.PEAK_OPEN: {
                     '$avg': '${}'.format(FLD.PEAK_OPEN)},
                 FLD.PEAK_OPEN_MINOR: {
                     '$avg': '${}'.format(FLD.PEAK_OPEN_MINOR)},
                 FLD.DEA: {
                     '$avg': '${}'.format(FLD.DEA_NORM)},
                 FLD.MACD: {
                     '$avg': '${}'.format(FLD.MACD_NORM)},
                 FLD.MACD_DELTA: {
                     '$avg': '${}'.format(FLD.MACD_DELTA_NORM)},
                 FLD.MAXFACTOR_CROSS: {
                     '$sum': '${}'.format(FLD.MAXFACTOR_CROSS)},
                 FLD.DUAL_CROSS: {
                     '$sum': '${}'.format(FLD.DUAL_CROSS)},
                 #ST.TRIGGER_R5: {
                 #    '$sum': '${}'.format(ST.TRIGGER_R5)},
                 FLD.TALIB_PATTERNS: {
                     '$sum': '${}'.format(FLD.TALIB_PATTERNS)},
                 FLD.ADXm_Trend: {
                     '$sum': '${}'.format(FLD.ADXm_Trend)},
                 FLD.Volume_HMA5: {
                     '$sum': '${}'.format(FLD.Volume_HMA5)},
                 FLD.RENKO_TREND_S: {
                     '$sum': '${}'.format(FLD.RENKO_TREND_S)},
                 FLD.RENKO_TREND_L: {
                     '$sum': '${}'.format(FLD.RENKO_TREND_L)},
                 'total': {                                   # 全市场个股家数
                     '$sum': 1}                       
                 }
              },
              {'$sort':{"_id.date":1}}])

        try:
            res = pd.DataFrame([QA_util_dict_remove_key(item, '_id') for item in cursor])
            res = res.assign(date=pd.to_datetime(res.date)).drop_duplicates((['date'])).set_index('date',
                                    drop=False)
        except:
            res = None
        if format in ['P', 'p', 'pandas', 'pd']:
            return res
        elif format in ['json', 'dict']:
            return QA_util_to_json_from_pandas(res)
        # 多种数据格式
        elif format in ['n', 'N', 'numpy']:
            return numpy.asarray(res)
        elif format in ['list', 'l', 'L']:
            return numpy.asarray(res).tolist()
        else:
            print("QA Error GQSignal_fetch_flu_cunsum_day format parameter %s is none of  \"P, p, pandas, pd , json, dict , n, N, numpy, list, l, L, !\" " % format)
            return None
    else:
        QA_util_log_info('QA Error GQSignal_fetch_flu_cunsum_day data parameter start=%s end=%s is not right' % (start,
               end))


def GQSignal_fetch_flu_cunsum_min(start,
                                end,
                                frequency='60min',
                                market_type=QA.MARKET_TYPE.STOCK_CN,
                                portfolio='myportfolio', 
                                format='numpy',
                                ui_log=None, 
                                ui_progress=None):
    """
    '获取全市场股票指标日线处于上升形态的数量'

    Keyword Arguments:
        client {[type]} -- [description] (default: {DATABASE})
    """
    start = str(start)[0:10]
    end = str(end)[0:10]
    #code= [code] if isinstance(code,str) else code

    client = QASETTING.client[AKA.SYSTEM_NAME]
    # 同时写入横表和纵表，减少查询困扰
    #coll_day = client.get_collection(
    #        'indices_{}'.format(datetime.date.today()))
    try:
        if (market_type == QA.MARKET_TYPE.STOCK_CN):
            #coll_indices = client.stock_cn_indices_min
            coll_indices = client.get_collection('stock_cn_indices_{}'.format(portfolio))
        elif (market_type == QA.MARKET_TYPE.INDEX_CN):
            #coll_indices = client.index_cn_indices_min
            coll_indices = client.get_collection('index_cn_indices_{}'.format(portfolio))
        elif (market_type == QA.MARKET_TYPE.FUND_CN):
            #coll_indices = client.future_cn_indices_min
            coll_indices = client.get_collection('fund_cn_indices_{}'.format(portfolio))
        elif (market_type == QA.MARKET_TYPE.FUTURE_CN):
            #coll_indices = client.future_cn_indices_min
            coll_indices = client.get_collection('future_cn_indices_{}'.format(portfolio))
        elif (market_type == QA.MARKET_TYPE.CRYPTOCURRENCY):
            #coll_indices = client.cryptocurrency_indices_min
            coll_indices = client.get_collection('cryptocurrency_indices_{}'.format(portfolio))
        else:
            QA_util_log_info('WTF IS THIS! \n ', ui_log=ui_log)
            return False
    except Exception as e:
        QA_util_log_info(e)
        QA_util_log_info('WTF IS THIS! \n ', ui_log=ui_log)
        return False

    if QA_util_date_valid(end):
        cursor = coll_indices.aggregate([{
            '$match': {
                'time_stamp': {
                    "$lte": QA_util_date_stamp(end),
                    "$gte": QA_util_date_stamp(start)
                    },
                'type': frequency,
                } 
             },
             {"$group" : {
                 '_id': {
                     'datetime': "$datetime",
                     },
                 'datetime': {
                     '$first': "$datetime"},
                 'time_stamp': {
                     '$first': "$time_stamp"},
                 FLD.FLU_POSITIVE: {                            # 上涨趋势通道个股家数
                     '$sum': '${}'.format(FLD.FLU_POSITIVE)},  
                 FLD.ML_FLU_TREND: {                            # “技术调整”个股家数
                     '$sum': '${}'.format(FLD.ML_FLU_TREND)}, 
                 ST.BOOTSTRAP_I: {
                     '$sum': { 
                         '$cond': ['${}'.format(ST.BOOTSTRAP_I), 1, 0] }},
                 ST.DEADPOOL: {
                     '$sum': { 
                         '$cond': ['${}'.format(ST.DEADPOOL), 1, 0] }},
                 FLD.FLU_POSITIVE_MASTER: {
                     '$sum': { 
                         '$cond': ['${}'.format(FLD.FLU_POSITIVE_MASTER), 1, 0] }},
                 FLD.FLU_NEGATIVE_MASTER: {
                     '$sum': { 
                         '$cond': ['${}'.format(FLD.FLU_NEGATIVE_MASTER), 1, 0] }},
                 ST.VOLUME_FLOW_BOOST: {
                     '$sum': '${}'.format(ST.VOLUME_FLOW_BOOST)},  
                 ST.VOLUME_FLOW_BOOST_BONUS: {
                     '$sum': { 
                         '$cond': ['${}'.format(ST.VOLUME_FLOW_BOOST_BONUS), 1, 0] }},
                 ST.CLUSTER_GROUP_TOWARDS: {
                     '$sum': '${}'.format(ST.CLUSTER_GROUP_TOWARDS)},
                 FLD.ATR_Stopline: {
                     '$sum': '${}'.format(FLD.ATR_Stopline)},
                 FLD.ATR_SuperTrend: {
                     '$sum': '${}'.format(FLD.ATR_SuperTrend)},
                 FLD.DEA: {
                     '$sum': '${}'.format(FLD.DEA)},
                 FLD.MAXFACTOR_CROSS: {
                     '$sum': '${}'.format(FLD.MAXFACTOR_CROSS)},
                 FLD.TALIB_PATTERNS: {
                     '$sum': '${}'.format(FLD.TALIB_PATTERNS)},
                 FLD.ADXm_Trend: {
                     '$sum': '${}'.format(FLD.ADXm_Trend)},
                 FLD.Volume_HMA: {
                     '$sum': '${}'.format(FLD.Volume_HMA)},
                 FLD.RENKO_TREND_S: {
                     '$sum': '${}'.format(FLD.RENKO_TREND_S)},
                 FLD.RENKO_TREND_L: {
                     '$sum': '${}'.format(FLD.RENKO_TREND_L)},
                 'total': {                                   # 全市场个股家数
                     '$sum': 1}                                             
                 }
              },
              {'$sort':{"_id.datetime":1}}])
        try:
            res = pd.DataFrame([QA_util_dict_remove_key(item, '_id') for item in cursor])
            res = res.assign(datetime=pd.to_datetime(res.datetime)).drop_duplicates((['datetime'])).set_index('datetime',
                                    drop=False)
        except:
            res = None
        if format in ['P', 'p', 'pandas', 'pd']:
            return res
        elif format in ['json', 'dict']:
            return QA_util_to_json_from_pandas(res)
        # 多种数据格式
        elif format in ['n', 'N', 'numpy']:
            return numpy.asarray(res)
        elif format in ['list', 'l', 'L']:
            return numpy.asarray(res).tolist()
        else:
            print("QA Error GQSignal_fetch_flu_cunsum_min format parameter %s is none of  \"P, p, pandas, pd , json, dict , n, N, numpy, list, l, L, !\" " % format)
            return None
    else:
        QA_util_log_info('QA Error GQSignal_fetch_flu_cunsum_min data parameter start=%s end=%s is not right' % (start,
               end))


def GQSignal_fetch_position_singal_min(start,
                                end,
                                frequence,
                                market_type=QA.MARKET_TYPE.STOCK_CN,
                                portfolio='myportfolio',
                                format='numpy',
                                ui_log=None, 
                                ui_progress=None):
    """
    在数据库中保存所有计算出来的指标信息，用于汇总评估和筛选数据——分钟线
    save stock_indices, state

    Keyword Arguments:
        client {[type]} -- [description] (default: {DATABASE})
    """
    def _check_index(coll_indices):
        coll_indices.create_index([("code",
                     pymongo.ASCENDING),
                    ("type",
                     pymongo.ASCENDING),
                    (FLD.DATETIME,
                     pymongo.ASCENDING),],
                unique=True)
        coll_indices.create_index([("code",
                     pymongo.ASCENDING),
                    ("type",
                     pymongo.ASCENDING),
                    ("time_stamp",
                     pymongo.ASCENDING),],
                unique=True)
        coll_indices.create_index([(FLD.DATETIME,
                     pymongo.ASCENDING),
                    ("type",
                     pymongo.ASCENDING),
                    (ST.TRIGGER_R5,
                     pymongo.ASCENDING),],)
        coll_indices.create_index([("type",
                     pymongo.ASCENDING),
                    ("time_stamp",
                     pymongo.ASCENDING),
                    (ST.TRIGGER_R5,
                     pymongo.ASCENDING),],)
        coll_indices.create_index([(FLD.DATETIME,
                     pymongo.ASCENDING),
                    ("type",
                     pymongo.ASCENDING),
                    (FLD.FLU_POSITIVE,
                     pymongo.ASCENDING),],)
        coll_indices.create_index([("type",
                     pymongo.ASCENDING),
                    ("time_stamp",
                     pymongo.ASCENDING),
                    (FLD.FLU_POSITIVE,
                     pymongo.ASCENDING),],)
        coll_indices.create_index([("code",
                     pymongo.ASCENDING),
                    ("type",
                     pymongo.ASCENDING),
                    (FLD.DATETIME,
                     pymongo.ASCENDING),
                     (ST.CANDIDATE,
                     pymongo.ASCENDING),],
                unique=True)
        coll_indices.create_index([("code",
                     pymongo.ASCENDING),
                    ("type",
                     pymongo.ASCENDING),
                    ("time_stamp",
                     pymongo.ASCENDING),
                     (ST.CANDIDATE,
                     pymongo.ASCENDING),],
                unique=True)

    def _formatter_data(indices, frequence):
        frame = indices.reset_index(1, drop=False)
        # UTC时间转换为北京时间
        frame['date'] = pd.to_datetime(frame.index,).tz_localize('Asia/Shanghai')
        frame['date'] = frame['date'].dt.strftime('%Y-%m-%d')
        frame['datetime'] = pd.to_datetime(frame.index,).tz_localize('Asia/Shanghai')
        frame['datetime'] = frame['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')
        # GMT+0 String 转换为 UTC Timestamp
        frame['time_stamp'] = pd.to_datetime(frame['datetime']).astype(np.int64) // 10 ** 9
        frame['type'] = frequence
        frame['created_at'] = int(time.mktime(datetime.datetime.now().utctimetuple()))
        frame = frame.tail(len(frame) - 150)
        return frame

    client = QASETTING.client[AKA.SYSTEM_NAME]

    # 同时写入横表和纵表，减少查询困扰
    #coll_day = client.get_collection(
    #        'indices_{}'.format(datetime.date.today()))
    try:
        if (market_type == QA.MARKET_TYPE.STOCK_CN):
            #coll_indices = client.stock_cn_indices_min
            coll_indices = client.get_collection('stock_cn_indices_{}'.format(portfolio))
        elif (market_type == QA.MARKET_TYPE.INDEX_CN):
            #coll_indices = client.index_cn_indices_min
            coll_indices = client.get_collection('index_cn_indices_{}'.format(portfolio))
        elif (market_type == QA.MARKET_TYPE.FUND_CN):
            #coll_indices = client.future_cn_indices_min
            coll_indices = client.get_collection('fund_cn_indices_{}'.format(portfolio))
        elif (market_type == QA.MARKET_TYPE.FUTURE_CN):
            #coll_indices = client.future_cn_indices_min
            coll_indices = client.get_collection('future_cn_indices_{}'.format(portfolio))
        elif (market_type == QA.MARKET_TYPE.CRYPTOCURRENCY):
            #coll_indices = client.cryptocurrency_indices_min
            coll_indices = client.get_collection('cryptocurrency_indices_{}'.format(portfolio))
        else:
            QA_util_log_info('WTF IS THIS! \n ', ui_log=ui_log)
            return False
    except Exception as e:
        QA_util_log_info(e)
        QA_util_log_info('WTF IS THIS! \n ', ui_log=ui_log)
        return False      

    _check_index(coll_indices)
    data = _formatter_data(indices, frequence)
    err = []

    # 查询是否新 tick
    query_id = {
        "code": code,
        'type': frequence,
        "time_stamp": {
            '$in': data['time_stamp'].tolist()
        }
    }
    refcount = coll_indices.count_documents(query_id)
    if refcount > 0:
        if (len(data) > 1):
            # 删掉重复数据
            coll_indices.delete_many(query_id)
            data = QA_util_to_json_from_pandas(data)
            coll_indices.insert_many(data)
        else:
            # 持续更新模式，更新单条记录
            data.drop('created_at', axis=1, inplace=True)
            data = QA_util_to_json_from_pandas(data)
            coll_indices.replace_one(query_id, data[0])
    else:
        # 新 tick，插入记录
        data = QA_util_to_json_from_pandas(data)
        coll_indices.insert_many(data)
    return True

