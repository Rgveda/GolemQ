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


