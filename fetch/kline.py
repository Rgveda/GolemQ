# coding:utf-8
#
# The MIT License (MIT)
#
# Copyright (c) 2016-2018 yutiansut/QUANTAXIS
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
from datetime import datetime as dt, timezone, timedelta
import os

try:
    import QUANTAXIS as QA
    from QUANTAXIS.QAIndicator.talib_numpy import *
    from QUANTAXIS.QAUtil.QADate_Adv import (
        QA_util_timestamp_to_str,
        QA_util_datetime_to_Unix_timestamp,
        QA_util_print_timestamp
    )
    from QUANTAXIS.QAUtil.QACode import (
        QA_util_code_tostr
        )
    from QUANTAXIS.QAData.QADataStruct import (
        QA_DataStruct_Index_min, 
        QA_DataStruct_Index_day, 
        QA_DataStruct_Stock_day, 
        QA_DataStruct_Stock_min,
        QA_DataStruct_CryptoCurrency_day,
        QA_DataStruct_CryptoCurrency_min,
        )
except:
    print('PLEASE run "pip install QUANTAXIS" before call GolemQ.fetch.kline modules')
    pass

from GolemQ.utils.parameter import (
    AKA, 
    INDICATOR_FIELD as FLD, 
    TREND_STATUS as ST,
    FEATURES as FTR,
    )

from GolemQ.utils.symbol import (
    is_stock_cn, 
    is_furture_cn,
    is_cryptocurrency,
)
from GolemQ.fetch.StockCN_realtime import (
    GQ_fetch_stock_day_realtime_adv,
    GQ_fetch_stock_min_realtime_adv,
    #GQ_fetch_index_day_realtime_adv,
    #GQ_fetch_index_min_realtime_adv,
)
from GolemQ.utils.path import (
    mkdirs,
)

def get_kline_price(codelist, start=None, market_type=None, verbose=True):
    """
    写这个函数的目的就是不用去考虑乱七八糟币种和市场种类，直接怼一个或者几个代码就能读取到合适的数据
    """
    if (market_type is None):
        if (isinstance(codelist, str)):
            # 判断是单一标的
            if (is_stock_cn(codelist)[1] == QA.MARKET_TYPE.STOCK_CN):
                market_type_desc = 'A股'
                market_type = QA.MARKET_TYPE.STOCK_CN
            elif (is_cryptocurrency(codelist)[1] == QA.MARKET_TYPE.CRYPTOCURRENCY):
                market_type_desc = '数字货币'
                market_type = QA.MARKET_TYPE.CRYPTOCURRENCY
            elif (is_stock_cn(codelist)[1] == QA.MARKET_TYPE.INDEX_CN):
                market_type_desc = 'A股指数'
                market_type = QA.MARKET_TYPE.INDEX_CN
            elif (is_stock_cn(codelist)[1] == QA.MARKET_TYPE.FUND_CN):
                market_type_desc = 'A股ETF基金'
                market_type = QA.MARKET_TYPE.INDEX_CN
            else:
                if verbose:
                    print(is_stock_cn(codelist))
        else:
            # 判断是多标的
            if (is_stock_cn(codelist[0])[1] == QA.MARKET_TYPE.STOCK_CN):
                market_type_desc = 'A股'
                market_type = QA.MARKET_TYPE.STOCK_CN
            elif (is_cryptocurrency(codelist[0])[1] == QA.MARKET_TYPE.CRYPTOCURRENCY):
                market_type_desc = '数字货币'
                market_type = QA.MARKET_TYPE.CRYPTOCURRENCY
            elif (is_stock_cn(codelist[0])[1] == QA.MARKET_TYPE.INDEX_CN):
                market_type_desc = 'A股指数'
                market_type = QA.MARKET_TYPE.INDEX_CN
            elif (is_stock_cn(codelist[0])[1] == QA.MARKET_TYPE.FUND_CN):
                market_type_desc = 'A股ETF基金'
                market_type = QA.MARKET_TYPE.INDEX_CN
            else:
                if verbose:
                    print(is_stock_cn(codelist))

            #raise Exception(u'多标的我还没时间实现')
    else:
        if (market_type == QA.MARKET_TYPE.STOCK_CN):
            market_type_desc = 'A股'
        elif (market_type == QA.MARKET_TYPE.CRYPTOCURRENCY):
            market_type_desc = '数字货币'
        elif (market_type == QA.MARKET_TYPE.FUND_CN):
            market_type_desc = 'A股ETF基金'
        elif (market_type == QA.MARKET_TYPE.INDEX_CN):
            market_type_desc = 'A股指数'

    if verbose:
        print(u'{} 开始读取{}日K线历史数据'.format(QA_util_timestamp_to_str()[2:16], 
                                                market_type_desc), 
                codelist if isinstance(codelist, str) else codelist[0:10])
    #data_day = QA.QA_fetch_stock_min_adv(codelist,
    #                                      '2018-11-01',
    #                                      '{}'.format(datetime.date.today()),
    #                                      frequence=frequence)
    if (market_type == QA.MARKET_TYPE.STOCK_CN):
        start = '{}'.format(datetime.date.today() - timedelta(days=2500)) if (start is None) else start
        data_day = QA.QA_fetch_stock_day_adv(codelist,
            start=start,
            end='{}'.format(datetime.date.today() + timedelta(days=1)),).to_qfq()
        #data_day = QA.QA_fetch_stock_day_adv(codelist,
        #                                  '2006-01-01',
        #                                  '{}'.format(datetime.date.today(),)).to_qfq()

        if (np.isnan(data_day).any() == True):
            # 在下载数据的时候，有时候除权后莫名其妙丢数据了，我只能拿没除权的数据补
            predict_null = pd.isnull(data_day.data[AKA.CLOSE])
            data_null = data_day.data[predict_null == True]
            data_day.data.loc[data_null.index, :] = QA.QA_fetch_stock_day_adv(codelist,
                                                '{}'.format(data_null.index.get_level_values(level=0).values[0]),
                                                '{}'.format(datetime.date.today(),)).data

        data_day = GQ_fetch_stock_day_realtime_adv(codelist, data_day, verbose=verbose)
        if verbose:
            data_day.data[ST.VERBOSE] = True
    elif (market_type == QA.MARKET_TYPE.CRYPTOCURRENCY):
        start = '{}'.format(datetime.datetime.now() - timedelta(hours=3600)) if (start is None) else start
        data_day = QA.QA_fetch_cryptocurrency_min_adv(code=codelist,
                start=start,
                end='{}'.format(datetime.datetime.now(timezone(timedelta(hours=8))) + timedelta(minutes=1)),
                frequence='60min')
        #data_hour = data_day =
        #QA.QA_fetch_cryptocurrency_day_adv(code=codelist,
        #        start='2018-01-15',
        #        end='{}'.format(datetime.datetime.now(timezone(timedelta(hours=8)))
        #        + timedelta(minutes=1)),
        #        )
        #data_day =
        #QA.QA_DataStruct_CryptoCurrency_min(data_day.resample('4h'))
        if verbose:
            data_day.data[ST.VERBOSE] = True
    elif (market_type == QA.MARKET_TYPE.INDEX_CN):
        start = '{}'.format(datetime.date.today() - timedelta(days=2500)) if (start is None) else start
        data_day = QA.QA_fetch_index_day_adv(codelist,
            start=start,
            end='{}'.format(datetime.date.today() + timedelta(days=1)),)
        data_day = GQ_fetch_stock_day_realtime_adv(codelist, data_day, verbose=verbose)
        if verbose:
            data_day.data[ST.VERBOSE] = True

    if verbose:
        print('Code:{}, last time:{:} total bars:{:d}'.format(data_day.data.index.get_level_values(level=1)[-1], data_day.data.index.get_level_values(level=0)[-1], len(data_day.data)))

    if verbose:
        print(u'{} 读取{}日K线历史数据完毕'.format(QA_util_timestamp_to_str()[2:16],
                                        market_type_desc), 
                codelist[-10:])

    if (isinstance(data_day, QA_DataStruct_Stock_min) or \
        isinstance(data_day, QA_DataStruct_Stock_day)):
        codename = QA.QA_fetch_stock_name(codelist)
        if (isinstance(codelist, str)):
            pass
        elif (len(codelist) != len(codename)):
            # 需要更新股票列表数据
            miss_codelist = [item for item in codelist if item not in codename[AKA.CODE].tolist()]
            if verbose:
                print(u'需要更新{}列表数据'.format(market_type_desc), 'miss_codelist', miss_codelist)
            codename = codename.reindex([*codename.index,
                                         *miss_codelist])
            #print(len(codename), codename)
    elif (isinstance(data_day, QA_DataStruct_Index_min) or \
        isinstance(data_day, QA_DataStruct_Index_day)):
        if (market_type_desc == 'A股ETF基金'):
            codename = QA.QA_fetch_etf_name(codelist)
            if (isinstance(codelist, str)):
                pass
            elif (len(codelist) != len(codename)):
                # 需要更新股票列表数据
                miss_codelist = [item for item in codelist if item not in codename[AKA.CODE].tolist()]
                if verbose:
                    print(u'需要更新{}列表数据'.format(market_type_desc), 'miss_codelist', miss_codelist)
                codename = codename.reindex([*codename.index,
                                             *miss_codelist])
        else:
            codename = QA.QA_fetch_index_name(codelist)
            if (isinstance(codelist, str)):
                pass
            elif (len(codelist) != len(codename)):
                # 需要更新股票列表数据
                miss_codelist = [item for item in codelist if item not in codename[AKA.CODE].tolist()]
                if verbose:
                    print(u'需要更新{}列表数据'.format(market_type_desc), 'miss_codelist', miss_codelist)
                codename = codename.reindex([*codename.index,
                                             *miss_codelist])
    elif isinstance(codelist, list):
        if (len(codelist) == 1):
            codename = codelist[0]
        else:
            codename = '{}'.format(codelist)
    else:
        codename = codelist if isinstance(codelist, str) else codelist.item()

    return data_day, codename


def get_kline_price_min(codelist, 
                        start=None, 
                        market_type=None, 
                        frequency='60min',
                        verbose=True):
    """
    写这个函数的目的就是不用去考虑乱七八糟币种和市场种类，直接怼一个或者几个代码就能读取到合适的数据
    """
    if (market_type is None):
        if (isinstance(codelist, str)):
            # 判断是单一标的
            if (is_stock_cn(codelist)[1] == QA.MARKET_TYPE.STOCK_CN):
                market_type_desc = 'A股'
                market_type = QA.MARKET_TYPE.STOCK_CN
            elif (is_cryptocurrency(codelist)[1] == QA.MARKET_TYPE.CRYPTOCURRENCY):
                market_type_desc = '数字货币'
                market_type = QA.MARKET_TYPE.CRYPTOCURRENCY
            elif (is_stock_cn(codelist)[1] == QA.MARKET_TYPE.INDEX_CN):
                market_type_desc = 'A股指数'
                market_type = QA.MARKET_TYPE.INDEX_CN
            elif (is_stock_cn(codelist)[1] == QA.MARKET_TYPE.FUND_CN):
                market_type_desc = 'A股ETF基金'
                market_type = QA.MARKET_TYPE.INDEX_CN
            else:
                if verbose:
                    print(is_stock_cn(codelist))
        else:
            # 判断是多标的
            if (is_stock_cn(codelist[0])[1] == QA.MARKET_TYPE.STOCK_CN):
                market_type_desc = 'A股'
                market_type = QA.MARKET_TYPE.STOCK_CN
            elif (is_cryptocurrency(codelist[0])[1] == QA.MARKET_TYPE.CRYPTOCURRENCY):
                market_type_desc = '数字货币'
                market_type = QA.MARKET_TYPE.CRYPTOCURRENCY
            elif (is_stock_cn(codelist[0])[1] == QA.MARKET_TYPE.INDEX_CN):
                market_type_desc = 'A股指数'
                market_type = QA.MARKET_TYPE.INDEX_CN
            elif (is_stock_cn(codelist[0])[1] == QA.MARKET_TYPE.FUND_CN):
                market_type_desc = 'A股ETF基金'
                market_type = QA.MARKET_TYPE.INDEX_CN
            else:
                if verbose:
                    print(is_stock_cn(codelist))
            #raise Exception(u'多标的我还没时间实现')
    else:
        if (market_type == QA.MARKET_TYPE.STOCK_CN):
            market_type_desc = 'A股'
        elif (market_type == QA.MARKET_TYPE.CRYPTOCURRENCY):
            market_type_desc = '数字货币'
        elif (market_type == QA.MARKET_TYPE.FUND_CN):
            market_type_desc = 'A股ETF基金'
        elif (market_type == QA.MARKET_TYPE.INDEX_CN):
            market_type_desc = 'A股指数'

    if verbose:
        print(u'{} 开始读取{}分钟K线历史数据'.format(QA_util_timestamp_to_str()[2:16], 
                                                 market_type_desc), 
                codelist if isinstance(codelist, str) else codelist[0:10])

    if (market_type == QA.MARKET_TYPE.STOCK_CN):
        start = '{}'.format(datetime.datetime.now() - timedelta(hours=19200)) if (start is None) else start

        start_time = dt.strptime(str(dt.now().date()) + ' 09:15', 
                                 '%Y-%m-%d %H:%M')
        if (dt.now() > start_time):
            end='{}'.format(dt.now(timezone(timedelta(hours=8))) + timedelta(minutes=1))
        else:
            end='{}'.format(dt.strptime(str(dt.now(timezone(timedelta(hours=8))).date() - timedelta(hours=24)) + ' 16:30', 
                           '%Y-%m-%d %H:%M'))
        data_min = GQ_fetch_stock_min_adv(code=codelist,
                start=start,
                end=end,
                frequence=frequency)

        data_min = GQ_fetch_stock_min_realtime_adv(codelist, data_min, frequency=frequency, verbose=verbose)
        if (data_min is None):
            if verbose:
                print(market_type, codelist)
            pass
        #print(data_min.data.tail(10))
        if verbose:
            data_min.data[ST.VERBOSE] = True
    elif (market_type == QA.MARKET_TYPE.CRYPTOCURRENCY):
        start = '{}'.format(datetime.datetime.now() - timedelta(hours=5400)) if (start is None) else start
        data_min = QA.QA_fetch_cryptocurrency_min_adv(code=codelist,
                start=start,
                end='{}'.format(datetime.datetime.now(timezone(timedelta(hours=8))) + timedelta(minutes=1)),
                frequence=frequency)
        if verbose:
            data_min.data[ST.VERBOSE] = True
    elif (market_type == QA.MARKET_TYPE.INDEX_CN):
        start = '{}'.format(datetime.datetime.now() - timedelta(hours=19200)) if (start is None) else start
        data_min = QA.QA_fetch_index_min_adv(codelist,
            start=start,
            end='{}'.format(datetime.datetime.now(timezone(timedelta(hours=8))) + timedelta(minutes=1)),
            frequence=frequency)
        data_min = GQ_fetch_stock_min_realtime_adv(codelist, data_min, frequency=frequency, verbose=verbose)
        if (data_min is None):
            if verbose:
                print(market_type, codelist)
            pass

        if verbose:
            data_min.data[ST.VERBOSE] = True
    else:
        if verbose:
            print(u'Not Supported code:', codelist)
        return None, None

    if verbose:
        print('last time:{:} total bars:{:d}'.format(data_min.data.index.get_level_values(level=0)[-1], len(data_min.data)))

    if verbose:
        print(u'{} 读取{}分钟K线历史数据完毕'.format(QA_util_timestamp_to_str()[2:16],
                                        market_type_desc), 
                codelist[-10:])

    try:
        if (isinstance(data_min, QA_DataStruct_Stock_min) or \
            isinstance(data_min, QA_DataStruct_Stock_day)):
            codename = QA.QA_fetch_stock_name(codelist)
        elif (isinstance(data_min, QA_DataStruct_Index_min) or \
            isinstance(data_min, QA_DataStruct_Index_day)):
            if (market_type_desc == 'A股ETF基金'):
                codename = QA.QA_fetch_etf_name(codelist)
            else:
                codename = QA.QA_fetch_index_name(codelist)
        elif isinstance(codelist, list):
            if (len(codelist) == 1):
                codename = codelist[0]
            else:
                codename = '{}'.format(codelist)
        else:
            codename = codelist if isinstance(codelist, str) else codelist.item()
    except:
        if verbose:
            print(u'Unsupported code:{}'.format(codelist))
        return None, None

    #print(data_min.data.tail(10))
    return data_min, codename


def GQ_fetch_stock_min_adv(code,
    start,
    end=None,
    frequence='1min',
    if_drop_index=True,
    verbose=False,):
    '''
    '获取股票分钟线'
    :param code:  字符串str eg 600085
    :param start: 字符串str 开始日期 eg 2011-01-01
    :param end:   字符串str 结束日期 eg 2011-05-01
    :param frequence: 字符串str 分钟线的类型 支持 1min 1m 5min 5m 15min 15m 30min 30m 60min 60m 类型
    :param if_drop_index: Ture False ， dataframe drop index or not
    :param collections: mongodb 数据库
    :return: QA_DataStruct_Stock_min 类型
    '''
    if frequence in ['1min', '1m']:
        frequence = '1min'
    elif frequence in ['5min', '5m']:
        frequence = '5min'
    elif frequence in ['15min', '15m']:
        frequence = '15min'
    elif frequence in ['30min', '30m']:
        frequence = '30min'
    elif frequence in ['60min', '60m']:
        frequence = '60min'
    else:
        if (verbose):
            print("QA Error QA_fetch_stock_min_adv parameter frequence=%s is none of 1min 1m 5min 5m 15min 15m 30min 30m 60min 60m" % frequence)
        return None

    # __data = [] 未使用

    end = start if end is None else end
    if len(start) == 10:
        start = '{} 09:30:00'.format(start)

    if len(end) == 10:
        end = '{} 15:00:00'.format(end)

    if start == end:
        # 🛠 todo 如果相等，根据 frequence 获取开始时间的 时间段 QA_fetch_stock_min， 不支持start
        # end是相等的
        if (verbose):
            print("QA Error QA_fetch_stock_min_adv parameter code=%s , start=%s, end=%s is equal, should have time span! " % (code,
                   start,
                   end))
        return None

    # 🛠 todo 报告错误 如果开始时间 在 结束时间之后

    res = QA.QA_fetch_stock_min(code, start, end, format='pd', frequence=frequence)
    if res is None:
        if (verbose):
            print("QA Error QA_fetch_stock_min_adv parameter code=%s , start=%s, end=%s frequence=%s call QA_fetch_stock_min return None" % (code,
                   start,
                   end,
                   frequence))
        return None
    else:
        res_set_index = res.set_index(['datetime', 'code'], drop=if_drop_index)
        # if res_set_index is None:
        #     print("QA Error QA_fetch_stock_min_adv set index 'datetime, code'
        #     return None")
        #     return None
        return QA_DataStruct_Stock_min(res_set_index)


def GQ_fetch_stock_min(symbol, frequence, path='cache', verbose=False):
    if (isinstance(symbol, list)):
        symbol = symbol[0]

    mkdirs(os.path.join(path, 'stock'))
    res = pd.read_hdf(os.path.join(path, 'stock', '{}_{}_kline.hdf'.format(symbol, frequence)), key='df', mode='r')
    return res


def GQ_fetch_stock_features_min(symbol, frequence, path='cache', verbose=False):
    if (isinstance(symbol, list)):
        symbol = symbol[0]

    mkdirs(os.path.join(path, 'stock'))
    res = pd.read_hdf(os.path.join(path, 'stock', '{}_{}_features.hdf'.format(symbol, frequence)), key='df', mode='r')
    return res


def GQ_fetch_index_min(symbol, frequence, path='cache', verbose=False):
    if (isinstance(symbol, list)):
        symbol = symbol[0]

    mkdirs(os.path.join(path, 'index'))
    res = pd.read_hdf(os.path.join(path, 'index', '{}_{}_kline.hdf'.format(symbol, frequence)), key='df', mode='r')
    return res


def GQ_fetch_index_features_min(symbol, frequence, path='cache', verbose=False):
    if (isinstance(symbol, list)):
        symbol = symbol[0]

    mkdirs(os.path.join(path, 'index'))
    res = pd.read_hdf(os.path.join(path, 'index', '{}_{}_features.hdf'.format(symbol, frequence)), key='df', mode='r')
    return res