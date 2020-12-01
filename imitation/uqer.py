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
"""
转换优矿API，本地数据接口来自QUANTAXIS，方便在QUANTAXIS中使用UQER策略，
"""
import pandas as pd
import numpy as np
from dateutil.parser import parse
from datetime import date, timedelta
import datetime
import tushare as ts

from QUANTAXIS.QAUtil.QADate_trade import (
    QA_util_get_pre_trade_date,
    QA_util_get_real_date,
    trade_date_sse
)
from GolemQ.utils.symbol import (
    EXCHANGE as EXG
)

class uqer_api():
    """
    在QUANTAXIS中模拟 uqer DataAPI 接口
    """
    # uqer A股交易日期
    _tradeDate_sse = pd.DataFrame(columns=['exchangeID', 'isOpen', 'calendarDate', 'isMonthEnd'])

    def __init__(self, *args, **kwargs):
        self._tradeDate_sse['calendarDate'] = trade_date_sse
        self._tradeDate_sse['exchangeID'] = EXG.XSHG
        self._tradeDate_sse['isOpen'] = 1
        self._tradeDate_sse = self._tradeDate_sse.assign(date=pd.to_datetime(self._tradeDate_sse.calendarDate)).drop_duplicates((['date'])).set_index(['date'],
                                   drop=False)
        tradeDate_sse_month_day = self._tradeDate_sse.index.month.values

        # 月末最后一天，优矿系统的专有特征
        self._tradeDate_sse['isMonthEnd'] = np.where(tradeDate_sse_month_day != np.r_[tradeDate_sse_month_day[1:], 
                                                                                      tradeDate_sse_month_day[-1]], 1, 0)
        return super().__init__(*args, **kwargs)

    def _selects_trade_date(self, start, end):
        if end is not None:
            return self._tradeDate_sse.loc[(slice(pd.Timestamp(start), pd.Timestamp(end))), :]
        else:
            return self._tradeDate_sse.loc[(slice(pd.Timestamp(start), None)), :]

    def TradeCalGet(self, 
                    exchangeCD=u"XSHG",
                    beginDate=None,
                    endDate=None,
                    field=u"isOpen,calendarDate",
                    pandas=1):
        """

        """

        # Alias variant names from differnent quantive system code
        start = beginDate
        end = endDate

        if ((beginDate is None) and (endDate is None)) or \
            ((beginDate == '') and (endDate == '')):
            ret_trade_cal = pd.DataFrame(trade_date_sse, columns=['calendarDate'])
            ret_trade_cal['isOpen'] = 1
            return ret_trade_cal
        else:
            try:
                return self._selects_trade_date(start, end).copy()
            except:
                raise ValueError('QA CANNOT GET TRADE DATE /START {}/END {} '.format(start,
                        end))


    def IdxCloseWeightGet(self,
                          secID=u"",
                          ticker='000300',
                          beginDate=None,
                          endDate=None,
                          field=u"",
                          pandas="1"):
        """
        获取每月的中证指标调整，成分股和权重
        """
        ret_indices = pd.DataFrame(columns=['secID',
      	    'effDate',
      	    'secShortName',
      	    'ticker',
      	    'consID',
      	    'consShortName',
      	    'consTickerSymbol',
      	    'consExchangeCD',
      	    'weight',])
        if ((beginDate is None) and (endDate is None)) or \
            ((beginDate == '') and (endDate == '')):
            csindex500 = ts.get_hs500s()
            print(csindex500)
        else:
            pass

        return ret_indices


DataAPI = uqer_api()

if __name__ == '__main__':
    # 可以从优矿提取下一年度的交易日期信息，直接导入到QA中
    #trade_cal =
    #DataAPI.TradeCalGet(exchangeCD=u"XSHG",beginDate=u"2020-12-31",endDate=u"",field=u"isOpen,calendarDate",pandas="1")
    #trade_cal = trade_cal[trade_cal['isOpen'] == 1]
    #lt = trade_cal['calendarDate'].tolist()
    #lt = ['2020-12-31', '2021-01-04', '2021-01-05', '2021-01-06',
    #'2021-01-07', '2021-01-08', '2021-01-11', '2021-01-12', '2021-01-13',
    #'2021-01-14', '2021-01-15', '2021-01-18', '2021-01-19', '2021-01-20',
    #'2021-01-21', '2021-01-22', '2021-01-25', '2021-01-26', '2021-01-27',
    #'2021-01-28', '2021-01-29', '2021-02-01', '2021-02-02', '2021-02-03',
    #'2021-02-04', '2021-02-05', '2021-02-08', '2021-02-09', '2021-02-10',
    #'2021-02-18', '2021-02-19', '2021-02-22', '2021-02-23', '2021-02-24',
    #'2021-02-25', '2021-02-26', '2021-03-01', '2021-03-02', '2021-03-03',
    #'2021-03-04', '2021-03-05', '2021-03-08', '2021-03-09', '2021-03-10',
    #'2021-03-11', '2021-03-12', '2021-03-15', '2021-03-16', '2021-03-17',
    #'2021-03-18', '2021-03-19', '2021-03-22', '2021-03-23', '2021-03-24',
    #'2021-03-25', '2021-03-26', '2021-03-29', '2021-03-30', '2021-03-31',
    #'2021-04-01', '2021-04-02', '2021-04-06', '2021-04-07', '2021-04-08',
    #'2021-04-09', '2021-04-12', '2021-04-13', '2021-04-14', '2021-04-15',
    #'2021-04-16', '2021-04-19', '2021-04-20', '2021-04-21', '2021-04-22',
    #'2021-04-23', '2021-04-26', '2021-04-27', '2021-04-28', '2021-04-29',
    #'2021-04-30', '2021-05-04', '2021-05-05', '2021-05-06', '2021-05-07',
    #'2021-05-10', '2021-05-11', '2021-05-12', '2021-05-13', '2021-05-14',
    #'2021-05-17', '2021-05-18', '2021-05-19', '2021-05-20', '2021-05-21',
    #'2021-05-24', '2021-05-25', '2021-05-26', '2021-05-27', '2021-05-28',
    #'2021-05-31', '2021-06-01', '2021-06-02', '2021-06-03', '2021-06-04',
    #'2021-06-07', '2021-06-08', '2021-06-09', '2021-06-10', '2021-06-11',
    #'2021-06-15', '2021-06-16', '2021-06-17', '2021-06-18', '2021-06-21',
    #'2021-06-22', '2021-06-23', '2021-06-24', '2021-06-25', '2021-06-28',
    #'2021-06-29', '2021-06-30', '2021-07-01', '2021-07-02', '2021-07-05',
    #'2021-07-06', '2021-07-07', '2021-07-08', '2021-07-09', '2021-07-12',
    #'2021-07-13', '2021-07-14', '2021-07-15', '2021-07-16', '2021-07-19',
    #'2021-07-20', '2021-07-21', '2021-07-22', '2021-07-23', '2021-07-26',
    #'2021-07-27', '2021-07-28', '2021-07-29', '2021-07-30', '2021-08-02',
    #'2021-08-03', '2021-08-04', '2021-08-05', '2021-08-06', '2021-08-09',
    #'2021-08-10', '2021-08-11', '2021-08-12', '2021-08-13', '2021-08-16',
    #'2021-08-17', '2021-08-18', '2021-08-19', '2021-08-20', '2021-08-23',
    #'2021-08-24', '2021-08-25', '2021-08-26', '2021-08-27', '2021-08-30',
    #'2021-08-31', '2021-09-01', '2021-09-02', '2021-09-03', '2021-09-06',
    #'2021-09-07', '2021-09-08', '2021-09-09', '2021-09-10', '2021-09-13',
    #'2021-09-14', '2021-09-15', '2021-09-16', '2021-09-17', '2021-09-20',
    #'2021-09-22', '2021-09-23', '2021-09-24', '2021-09-27', '2021-09-28',
    #'2021-09-29', '2021-09-30', '2021-10-08', '2021-10-11', '2021-10-12',
    #'2021-10-13', '2021-10-14', '2021-10-15', '2021-10-18', '2021-10-19',
    #'2021-10-20', '2021-10-21', '2021-10-22', '2021-10-25', '2021-10-26',
    #'2021-10-27', '2021-10-28', '2021-10-29', '2021-11-01', '2021-11-02',
    #'2021-11-03', '2021-11-04', '2021-11-05', '2021-11-08', '2021-11-09',
    #'2021-11-10', '2021-11-11', '2021-11-12', '2021-11-15', '2021-11-16',
    #'2021-11-17', '2021-11-18', '2021-11-19', '2021-11-22', '2021-11-23',
    #'2021-11-24', '2021-11-25', '2021-11-26', '2021-11-29', '2021-11-30',
    #'2021-12-01', '2021-12-02', '2021-12-03', '2021-12-06', '2021-12-07',
    #'2021-12-08', '2021-12-09', '2021-12-10', '2021-12-13', '2021-12-14',
    #'2021-12-15', '2021-12-16', '2021-12-17', '2021-12-20', '2021-12-21',
    #'2021-12-22', '2021-12-23', '2021-12-24', '2021-12-27', '2021-12-28',
    #'2021-12-29', '2021-12-30', '2021-12-31']
    pass