# coding:utf-8
#
# The MIT License (MIT)
#
# Copyright (c) 2020-2025 azai/GolemQ(uant)
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

"""
爬虫模块, 爬取A股指数成分, 财报, 研报和资金流向等财务信息
"""

import time
import sys
from datetime import (
    datetime as dt,
    timedelta,
)
import pandas as pd
import numpy as np
from GolemQ.core.constants import (
    AKA,
    FIELD as FLD,
)
from GolemQ.markets.StockCN import (
    is_stock_cn,
)
from GolemQ.core.presentation import (
    suppress_stdout_stderr,
)
try:
    with suppress_stdout_stderr():
        import QUANTAXIS as QA
except Exception:
    print('QUANTAXIS not installed.')
    pass

import pymongo
import traceback
from GolemQ.core.settings import (
    DATABASE as DATABASE_GolemQ,
)
from func_timeout import func_set_timeout
import baostock as bs
from GolemQ.services.features import (
    GQ_save_stock_valuation,
    GQ_fetch_daily_metadata_reality,
)
from .date_utils import GQ_util_get_last_day


@func_set_timeout(9)
def GQ_SU_crawl_stock_valuation(
    code,
    start=None, end=None,
    collections=DATABASE_GolemQ.stock_valuation,
    verbose=False,
):
    start = '{}'.format(dt.now() - timedelta(hours=19200)) if (start is None) else '{}'.format(pd.to_datetime(start) - timedelta(hours=8.5))
    end = '{}'.format(dt.now() + timedelta(hours=16)) if (end is None) else '{}'.format(pd.to_datetime(end) + timedelta(hours=16))
    stock_valuation_pd = GQ_fetch_daily_metadata_reality(
        code=code,
        start=start, end=end,
        collections=collections)
    data_day = QA.QA_fetch_stock_day_adv(
        code,
        start=start,
        end=end,)
    features_dummy = data_day.data.copy()
    column_list = [
        FLD.PE_RATION,
        FLD.TURNOVER_RATE,]
    if (stock_valuation_pd is None) and \
        (features_dummy is not None) and \
        (len(features_dummy) > 0):
        start = '{}'.format(features_dummy.index.get_level_values(level=0)[0] - timedelta(hours=8.5))
        end = '{}'.format(features_dummy.index.get_level_values(level=0)[-1] + timedelta(hours=16))
        stock_valuation_pd = GQ_fetch_daily_metadata_reality(
            code=code,
            start=start, end=end,
            collections=collections)
        
    if (stock_valuation_pd is not None) and \
        (len(stock_valuation_pd) > 0):
        stock_valuation_pd = stock_valuation_pd.rename(
            columns={
                u'peTTM': FLD.PE_RATION,
                u'turnover': FLD.TURNOVER_RATE
            }
        )
        stock_valuation_idx = features_dummy.index.intersection(stock_valuation_pd.index)
        features_dummy = features_dummy.reindex(
            columns=list(set([
                *features_dummy.columns,
                *column_list])))
        features_dummy.loc[
            stock_valuation_idx,
            column_list] = stock_valuation_pd.loc[
                stock_valuation_idx,
                column_list]
    else:
        if (verbose):
            print(f'Phase 2.1: stock_valuation_pd is {stock_valuation_pd}', )
        features_dummy = features_dummy.reindex(
            columns=list(set([
                *features_dummy.columns,
                *column_list])))
    missing_stock_valuation_idx = features_dummy.index.difference(
        features_dummy.dropna(
            subset=[FLD.TURNOVER_RATE],
            axis=0,
            how="all").index)
    if (verbose):
        print(
            f'Phase 2: 缺 {len(missing_stock_valuation_idx)} 天的换手率', missing_stock_valuation_idx,
            f"首:{missing_stock_valuation_idx.get_level_values(level=0)[0]} 尾:{missing_stock_valuation_idx.get_level_values(level=0)[0]}")
    
    action = False
    baostock_st = False
    if ((stock_valuation_pd is not None) and \
        (len(stock_valuation_pd) > 0)) or \
        ((stock_valuation_pd is None) and \
        (features_dummy is not None) and \
        (len(features_dummy) > 0)):
        if (verbose):
            print(f'\nCode {code} missing length:{len(missing_stock_valuation_idx)} {missing_stock_valuation_idx}')
        if (stock_valuation_pd is not None) and \
            ((pd.to_datetime(GQ_util_get_last_day())-stock_valuation_pd.index.get_level_values(level=0)[-1]) > timedelta(days=1)):
            start = pd.to_datetime(missing_stock_valuation_idx.get_level_values(level=0)[0]-timedelta(hours=8.5))
            if ((len(missing_stock_valuation_idx)) < 200):
                end = pd.to_datetime(missing_stock_valuation_idx.get_level_values(level=0)[-1])+timedelta(hours=16)
            else:    
                end = pd.to_datetime(GQ_util_get_last_day())+timedelta(hours=16)
            action = True
        elif ((stock_valuation_pd is None) and \
            (features_dummy is not None) and \
            (len(features_dummy) > 0)):
            start = pd.to_datetime(missing_stock_valuation_idx.get_level_values(level=0)[0] - timedelta(hours=8.5))
            end = pd.to_datetime(missing_stock_valuation_idx.get_level_values(level=0)[-1]) + timedelta(hours=16)
            action = True

    else:
        start = pd.to_datetime(GQ_util_get_last_day())-timedelta(hours=16)
        end = pd.to_datetime(GQ_util_get_last_day())+timedelta(hours=16)
        action = True

    if (action):
        try:
            baostock_st = True
            ret_valuation = GQ_featch_stock_valuation_from_baostock(
                code, 
                start='{}'.format(start)[:10], 
                end='{}'.format(end)[:10],)
            baostock_st = False
            if (verbose):
                print(
                    'GQ_featch_stock_valuation_from_baostock', code, 
                    '{}'.format(start)[:10], 
                    '{}'.format(end)[:10],)
                print(ret_valuation)

            if (isinstance(ret_valuation, tuple)):
                print('query_history_k_data_plus respond error_code:'+ret_valuation[0])
                print('query_history_k_data_plus respond error_msg:'+ret_valuation[1], code)
            else:
                if (verbose) and \
                    (pd.to_datetime(missing_stock_valuation_idx.get_level_values(level=0)[-1]) - \
                    pd.to_datetime(missing_stock_valuation_idx.get_level_values(level=0)[0]) > timedelta(days=8.5)):
                    if (verbose):
                        print(ret_valuation.index.intersection(missing_stock_valuation_idx))
                    ret_valuation = ret_valuation.loc[missing_stock_valuation_idx, :]
                    if (verbose):
                        print(ret_valuation)
                try:
                    if (verbose):
                        print(f'{start} {end} \n{ret_valuation.head(5)}')
                    GQ_save_stock_valuation(
                        ret_valuation.dropna(
                            subset=['turnover'],
                            axis=0,
                            how="all"))
                except Exception:
                    print('{}'.format(ret_valuation))
                    pass
        except Exception as e:
            # traceback.print_exc()
            if ((pd.to_datetime(end)-pd.to_datetime(start)) < timedelta(days=84)):
                if (baostock_st is False):
                    traceback.print_exception(type(e), e, sys.exc_info()[2])
                    print(u'Failed:{}\n'.format(code), e)
                time.sleep(0.15)


def GQ_featch_stock_valuation_from_baostock(
    code,
    start,
    end,
):
    ret_is_stock_cn, market_type, market_alias, market_remark = is_stock_cn(code)
    if (ret_is_stock_cn):
        alias_code = '{}.{}'.format(market_alias.lower(), code)
    elif (isinstance(code, str)):
        alias_code = code
    else:
        print("code 必须是字符串类型", code)
        alias_code = list(code)[0]
    if (not alias_code.startswith('sh') and not alias_code.startswith('sz')):
        print('股票代码未标识sh或sz', alias_code)
        return None, alias_code
    elif (len(alias_code)!=9):
        print('股票代码应为9位, 请检查。格式示例: sh.600000', alias_code)
        return None, alias_code
    rs = bs.query_history_k_data_plus(
        alias_code,
        "date,code,close,turn,tradestatus,pctChg,peTTM,pbMRQ,psTTM,pcfNcfTTM,isST",
        start_date=start, end_date=end,
        frequency="d", adjustflag="3")
    # print('query_history_k_data_plus respond error_code:'+rs.error_code)
    # print('query_history_k_data_plus respond  error_msg:'+rs.error_msg)

    # 打印结果集 #
    result_list = []
    while (rs.error_code == '0') & rs.next():
        # 获取一条记录，将记录合并在一起
        result_list.append(rs.get_row_data())
    result = pd.DataFrame(
        result_list, columns=rs.fields,
    )
    if (AKA.CODE not in result.columns):
        result[AKA.CODE] = code
    else:
        result[AKA.CODE] = [code[-6:] for _,code in result[AKA.CODE].items()]
    if ('date' in result.columns) and (len(result) > 0):
        result = result.rename(columns={'turn':AKA.TURNOVER})
        result[AKA.TURNOVER] = np.where(result[AKA.TURNOVER] == '', 
                                        np.nan, result[AKA.TURNOVER])
        result[AKA.TURNOVER] = result[AKA.TURNOVER].astype(np.float32)/100
        result[AKA.CLOSE] = result[AKA.CLOSE].astype(np.float64)
        result['tradestatus'] = result['tradestatus'].astype(np.int16)
        result['pctChg'] = np.where(
            result['pctChg'] == '',
            np.nan, result['pctChg']).astype(np.float32)
        result['peTTM'] = np.where(
            result['peTTM'] == '',
            np.nan, result['peTTM']).astype(np.float32)
        result['pbMRQ'] = np.where(
            result['pbMRQ'] == '',
            np.nan, result['pbMRQ']).astype(np.float32)
        result['psTTM'] = np.where(
            result['psTTM'] == '',
            np.nan, result['psTTM']).astype(np.float32)
        result['pcfNcfTTM'] = np.where(
            result['pcfNcfTTM'] == '',
            np.nan, result['pcfNcfTTM']).astype(np.float32)
        result['isST'] = result['isST'].astype(np.float32)
        result[AKA.DATE] = pd.to_datetime(result[AKA.DATE].to_list())
        result = result.drop_duplicates(([
            'date',
            'code'])).set_index([
                  'date',
                  'code'],
                  drop=False)

        return result
    else:
        print((alias_code,
        "date,code,close,turn,tradestatus,pctChg,peTTM,pbMRQ,psTTM,pcfNcfTTM,isST",
        start, end, 
        "d", "3"),
        ('date' in result.columns), len(result), (len(result) > 0), result.columns)
    
    return rs.error_code, rs.error_msg


def GQ_remove_stock_valuation(
        stock_valuation_detail,
        collections=DATABASE_GolemQ.stock_valuation):
    '''
    保存股价估值信息等数据
    :param collections: 数据库 collection 存储名称
    :return:
    '''
    coll = collections
    coll.create_index([
            ('code', pymongo.ASCENDING),
            ("date_stamp", pymongo.ASCENDING)
        ], unique=True)

    # ret_stock_diagnosis_detail['date'] = stock_valuation_detail['datetime'].dt.date
    # stock_valuation_detail['date_stamp'] = pd.to_datetime(stock_valuation_detail['date']).view("int64") // 10 ** 9
    stock_valuation_detail = stock_valuation_detail.assign(
        date_stamp=pd.to_datetime(stock_valuation_detail['date']).view("int64") // 10 ** 9
    )
    # print(ret_stock_diagnosis_detail)

    stock_valuation_detail = stock_valuation_detail.set_index(['date', 'code'], drop=False)

    try:
        data = stock_valuation_detail

        # 查询是否新数据
        if (len(data.index.get_level_values(level=1).unique()) == 1) and \
            (len(data.index.get_level_values(level=0).unique()) > len(data.index.get_level_values(level=1).unique())):
            query_id = {
                "code": data.iloc[0][AKA.CODE],
                'date_stamp': {
                    '$in': list(set(data['date_stamp'].to_list()))
                }
            }
            code = data.iloc[0][AKA.CODE]
        elif (len(data.index.get_level_values(level=0).unique()) == 1) and \
            (len(data.index.get_level_values(level=1).unique()) > len(data.index.get_level_values(level=0).unique())):
            query_id = {
                "code": {
                    '$in': list(data.index.get_level_values(level=1).unique())
                },
                'date_stamp': data['date_stamp'].astype(np.int32).tail(1).item()
            }
            code = list(data.index.get_level_values(level=1).unique())
        else:
            query_id = {
                "code": {
                    '$in': list(data.index.get_level_values(level=1).unique())
                },
                'date_stamp': {
                    '$in': list(set(data['date_stamp'].astype(np.int32).to_list()))
                }
            }
            code = list(data.index.get_level_values(level=1).unique())
        # print('dupl:',  list(data.index.get_level_values(level=1).unique()))
        refcount = coll.count_documents(query_id)

        if refcount > 0:
            if (len(data) > 1):
                # 删掉重复数据
                coll.delete_many(query_id)
    except Exception as e:
        if (data is not None):
            code = data.iloc[0].code
            traceback.print_exception(type(e), e, sys.exc_info()[2])
            print(u'Failed:{}\n'.format(code), e)

    return stock_valuation_detail


