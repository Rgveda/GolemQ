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


import os
from datetime import (
    datetime as dt, timedelta
)
from GolemQ.core.settings import (
    DATABASE_QA as DATABASE,
)
import pandas as pd
from GolemQ.core import (
    GQ_util_code_tolist,
    GQ_util_code_tostr,
    get_pickle_filename,
    mkdirs, cache_path,
    load_snapshot_cache,
    save_snapshot_cache,
)
from GolemQ.core.constants import (
    MARKET_TYPE,
    AKA,
)
from functools import lru_cache
from QUANTAXIS.QAFetch.QAQuery import (
    QA_fetch_stock_list
)
import pymongo
from GolemQ.markets.StockCN.date_utils import (
    GQ_util_get_last_day
)
import traceback


class EXCHANGE():
    XSHG = 'XSHG'
    SSE = 'XSHG'
    SH = 'XSHG'
    XSHE = 'XSHE'
    SZ = 'XSHE'
    SZE = 'XSHE'
    BJ = 'XBJE'

    def __setattr__(self, name, value):
        raise Exception(u'Const Class can\'t allow to change property value.')
        return super().__setattr__(name, value)


focus_block = [
    'MSCI中国', 'MSCI成份', 'MSCI概念', '三网融合',
    '上证180', '上证380', '沪深300', '上证380',
    '深证300', '上证50', '上证电信', '电信等权',
    '上证100', '上证150', '沪深300', '中证100', '深证100',
    '中证500', '全指消费', '中小板指', '创业板指',
    '综企指数', '1000可选', '国证食品', '深证可选',
    '深证消费', '深成消费', '中证酒指数', '中证白酒指数',
    '行业龙头', '白酒', '证券', '消费100',
    '消费电子', '消费金融', '富时A50', '银行',
    '中小银行', '证券', '军工', '白酒', '啤酒',
    '医疗器械', '医疗器械服务', '医疗改革', '医药商业',
    '医药电商', '中药', '消费100', '消费电子',
    '消费金融', '黄金', '黄金概念', '4G5G',
    '5G概念', '生态农业', '生物医药', '生物疫苗',
    '机场航运', '数字货币', '文化传媒'
]


def normalize_code(symbol, pre_close=None, market_type=None):
    """
    归一化证券代码

    :param code 如000001
    :return 证券代码的全称 如000001.XSHE
    """
    if (isinstance(symbol, list)):
        return [normalize_code(each_symbo) for each_symbo in symbol]
    elif (not isinstance(symbol, str)):
        return symbol
    else:
        symbol = symbol.lstrip().rstrip()

    if (symbol.startswith('sz') and (len(symbol) == 8)):
        ret_normalize_code = '{}.{}'.format(symbol[2:8], EXCHANGE.SZ)
    elif (symbol.endswith('SZ') and (len(symbol) == 9)):
        ret_normalize_code = '{}.{}'.format(symbol[0:6], EXCHANGE.SZ)
    elif (symbol.startswith('sh') and (len(symbol) == 8)):
        ret_normalize_code = '{}.{}'.format(symbol[2:8], EXCHANGE.SH)
    elif (symbol.endswith('SH') and (len(symbol) == 9)):
        ret_normalize_code = '{}.{}'.format(symbol[0:6], EXCHANGE.SH)
    elif (symbol.startswith('00') and (len(symbol) == 6)):
        if ((pre_close is not None) and (pre_close > 2000)) or \
           ((market_type is not None) and
           (market_type == MARKET_TYPE.INDEX_CN)):
            # 推断是上证指数
            ret_normalize_code = '{}.{}'.format(symbol, EXCHANGE.SH)
        else:
            ret_normalize_code = '{}.{}'.format(symbol, EXCHANGE.SZ)
    elif ((symbol.startswith('399') or symbol.startswith('159') or
           symbol.startswith('150')) and (len(symbol) == 6)):
        ret_normalize_code = '{}.{}'.format(symbol, EXCHANGE.SZ)
    elif ((len(symbol) == 6) and (symbol.startswith('399') or
          symbol.startswith('159') or symbol.startswith('150') or
          symbol.startswith('16') or symbol.startswith('18') or
          symbol.startswith('20'))):
        ret_normalize_code = '{}.{}'.format(symbol, EXCHANGE.SZ)
    elif ((len(symbol) == 6) and (symbol.startswith('82') or
          symbol.startswith('83') or symbol.startswith('87') or
          symbol.startswith('88') or
          symbol.startswith('43') or symbol.startswith('92'))):
        ret_normalize_code = '{}.{}'.format(symbol, EXCHANGE.BJ)
    elif ((len(symbol) == 6) and (symbol.startswith('50') or
          symbol.startswith('51') or symbol.startswith('52') or
          symbol.startswith('53') or symbol.startswith('56') or
          symbol.startswith('55') or symbol.startswith('56') or
          symbol.startswith('58') or symbol.startswith('60') or
          symbol.startswith('688') or symbol.startswith('900') or
          symbol.startswith('689') or symbol.startswith('588') or
          symbol == '751038')):
        ret_normalize_code = '{}.{}'.format(symbol, EXCHANGE.SH)
    elif ((len(symbol) == 6) and (symbol.startswith('000') or
          symbol.startswith('001') or symbol.startswith('002') or
          symbol.startswith('200') or symbol.startswith('300') or
          symbol.startswith('301') or symbol.startswith('302') or
          symbol.startswith('303'))):
        ret_normalize_code = '{}.{}'.format(symbol, EXCHANGE.SZ)
    elif ((len(symbol) == 6) and (symbol[:3] in ['000', '001', '002',
                                                 '200', '300', '301',
                                                 '302', '303'])):
        ret_normalize_code = '{}.{}'.format(symbol, EXCHANGE.SZ)
    elif symbol.startswith('XSHG'):
        ret_normalize_code = '{}.{}'.format(symbol[5:], EXCHANGE.SH)
    elif symbol.startswith('XSHE'):
        ret_normalize_code = '{}.{}'.format(symbol[5:], EXCHANGE.SZ)
    elif (symbol.endswith('XSHG') or symbol.endswith('XSHE')):
        ret_normalize_code = symbol
    else:
        print(u'normalize_code():', len(symbol), symbol)
        ret_normalize_code = symbol

    return ret_normalize_code


def is_stock_cn(code):
    """
    判断 股票代码，市场来源，板块
    1- sh
    0 -sz
    """
    symbol = code
    if (isinstance(code, list)):
        for ecach_code in code:
            return is_stock_cn(ecach_code)

    code = str(code)
    if (len(code) == 0):
        print(symbol, u'长度为零')
        return False, None, None, None
    if code[:2] in ["83", "87", "82", "88", "43", '92', ]:
        return True, MARKET_TYPE.STOCK_CN, 'BJ', '北证A股'
    elif code[0] in ['5', '6', '9'] or \
        code[:3] in ["009", "126", "110", "201", "202", "203", "204",
                     '688', '689'] or \
        (code.startswith('XSHG')) or \
        (code.startswith('sh')) or \
        (code.startswith('SH')) or \
            (code.endswith('XSHG')):
        if (code.startswith('XSHG')) or \
                (code.endswith('XSHG')):
            if (len(code.split('.')) > 1):
                try_split_codelist = code.split('.')
                if (try_split_codelist[0] == 'XSHG') and \
                        (len(try_split_codelist[1]) == 6):
                    code = try_split_codelist[1]
                elif (try_split_codelist[1] == 'XSHG') and \
                        (len(try_split_codelist[0]) == 6):
                    code = try_split_codelist[0]
                if (code.startswith("00000")) or \
                        (code.startswith("000")):
                    return True, MARKET_TYPE.INDEX_CN, 'SH', '上交所指数'
        if (len(code) == 8):
            code = code[-6:]
        elif (len(code) == 11) and code.startswith('XSHG'):
            code = code[-6:]
        elif (len(code) == 11) and code.endswith('XSHG'):
            code = code[:6]
        if code.startswith('60'):
            return True, MARKET_TYPE.STOCK_CN, 'SH', '上交所A股'
        elif (code.startswith('688')) or \
                (code.startswith('689')):
            return True, MARKET_TYPE.STOCK_CN, 'SH', '上交所科创板'
        elif code.startswith('900'):
            return True, MARKET_TYPE.STOCK_CN, 'SH', '上交所B股'
        elif code.startswith('50'):
            return True, MARKET_TYPE.FUND_CN, 'SH', '上交所传统封闭式基金'
        elif (code.startswith('51')) or \
            (code.startswith('52')) or \
            (code.startswith('53')) or \
            (code.startswith('55')) or \
            (code.startswith('56')) or \
            (code.startswith('58')) or \
                (code.startswith('5880')):
            return True, MARKET_TYPE.INDEX_CN, 'SH', '上交所ETF基金'  # QA 把ETF归类为INDX_CN
        else:
            print(code, True, None, 'SH', '上交所未知代码')
            return True, None, 'SH', '上交所未知代码'
    elif code[0] in ['0', '2', '3'] or \
        code[:2] in ['15', '16', '18'] or \
        code[:3] in ['000', '001', '002', '200', '300', '159'] or \
        (code.startswith('XSHE')) or \
        (code.startswith('sz')) or \
        (code.startswith('SZ')) or \
            (code.endswith('XSHE')):
        if (len(code) == 8):
            code = code[-6:]
        elif (len(code) == 11) and code.startswith('XSHE'):
            code = code[-6:]
        elif (len(code) == 11) and code.endswith('XSHE'):
            code = code[:6]
        if (code.startswith('000')) or \
                (code.startswith('001')):
            if (code in ['000003', '000112', '000300', '000132', '000133']):
                return True, MARKET_TYPE.INDEX_CN, 'SH', '中证指数'
            else:
                return True, MARKET_TYPE.STOCK_CN, 'SZ', '深交所主板'
        if code.startswith('002'):
            return True, MARKET_TYPE.STOCK_CN, 'SZ', '深交所中小板'
        elif code.startswith('003'):
            return True, MARKET_TYPE.STOCK_CN, 'SZ', '中广核？？'
        elif (code.startswith('159')) or \
            (code.startswith('150')) or \
            (code.startswith('160')) or \
            (code.startswith('180')) or \
                (code.startswith('20')):
            return True, MARKET_TYPE.INDEX_CN, 'SZ', '深交所ETF基金'  # QA 把ETF归类为INDX_CN
        elif code.startswith('200'):
            return True, MARKET_TYPE.STOCK_CN, 'SZ', '深交所B股'
        elif code.startswith('399'):
            return True, MARKET_TYPE.INDEX_CN, 'SZ', '中证指数'
        elif (code.startswith('300')) or \
            (code.startswith('301')) or \
                (code.startswith('302')):
            return True, MARKET_TYPE.STOCK_CN, 'SZ', '深交所创业板'
        elif (code.startswith('XSHE')) or \
                (code.endswith('XSHE')):
            pass
        else:
            print(code, True, None, 'SZ', '深交所未知代码')
            return True, None, 'SZ', '深交所未知代码'
    elif code[:2] in ["83", "87", "43"]:
        return True, MARKET_TYPE.STOCK_CN, 'BJ', '北交所主板'
    else:
        print(code, isinstance(code, list), '不知道')
        return False, None, None, None


def is_furture_cn(code):
    if code[:2] in ['IH', 'IF', 'IC', 'TF', 'JM', 'PP', 'EG', 'CS',
                    'AU', 'AG', 'SC', 'CU', 'AL', 'ZN', 'PB', 'SN', 'NI',
                    'RU', 'RB', 'HC', 'BU', 'FU', 'SP',
                    'SR', 'CF', 'RM', 'MA', 'TA', 'ZC', 'FG', 'IO', 'CY']:
        return True, MARKET_TYPE.FUTURE_CN, 'NA', '中国期货'
    elif code[:1] in ['A', 'B', 'Y', 'M', 'J', 'P', 'I',
                      'L', 'V', 'C', 'T']:
        return True, MARKET_TYPE.FUTURE_CN, 'NA', '中国期货'
    else:
        return False, None, None, None


def is_cryptocurrency(code):
    code = str(code)
    if (code.startswith('HUOBI')) or code.startswith('huobi') or \
            code.endswith('husd') or code.endswith('HUSD'):
        return True, MARKET_TYPE.CRYPTOCURRENCY, 'huobi.pro', '数字货币'
    elif code.endswith('bnb') or code.endswith('BNB') or \
            code.startswith('BINANCE') or code.startswith('binance'):
        return True, MARKET_TYPE.CRYPTOCURRENCY, 'Binance', '数字货币'
    elif code.startswith('BITMEX') or code.startswith('bitmex'):
        return True, MARKET_TYPE.CRYPTOCURRENCY, 'Bitmex', '数字货币'
    elif code.startswith('OKEX') or code.startswith('OKEx') or \
        code.startswith('okex') or code.startswith('OKCoin') or \
            code.startswith('okcoin') or code.startswith('OKCOIN'):
        return True, MARKET_TYPE.CRYPTOCURRENCY, 'OKEx', '数字货币'
    elif code.startswith('BITFINEX') or code.startswith('bitfinex') or \
            code.startswith('Bitfinex'):
        return True, MARKET_TYPE.CRYPTOCURRENCY, 'Bitfinex', '数字货币'
    elif (code[:-7] in ['adausdt', 'bchusdt', 'bsvusdt', 'btcusdt',
                        'btchusd', 'eoshusd', 'eosusdt', 'etcusdt',
                        'etchusd', 'ethhusd', 'ethusdt', 'ltcusdt',
                        'trxusdt', 'xmrusdt', 'xrpusdt', 'zecusdt']) or \
        (code[:-8] in ['atomusdt', 'algousdt', 'dashusdt', 'dashhusd',
                       'hb10usdt']) or \
            (code[:-6] in ['hthusd', 'htusdt']):
        return True, MARKET_TYPE.CRYPTOCURRENCY, 'huobi.pro', '数字货币'
    elif code.endswith('usd') or code.endswith('USD'):
        return True, MARKET_TYPE.CRYPTOCURRENCY, 'NA', '数字货币'
    elif code.endswith('usdt') or code.endswith('USDT'):
        return True, MARKET_TYPE.CRYPTOCURRENCY, 'NA', '数字货币'
    elif code[:3] in ['BTC', 'btc', 'ETH', 'eth',
                      'EOS', 'eos', 'ADA', 'ada',
                      'BSV', 'bsv', 'BCH', 'bch',
                      'xmr', 'XMR', 'LTC', 'ltc',
                      'xrp', 'XRP', 'ZEC', 'zec',
                      'trx', 'TRX', 'ZEC', 'zec']:
        return True, MARKET_TYPE.CRYPTOCURRENCY, 'NA', '数字货币'
    else:
        return False, None, None, None


def GQ_fetch_stock_info(code, collections=DATABASE.stock_info, ):
    code = GQ_util_code_tolist(code)
    try:
        data = pd.DataFrame(
            [
                item for item in collections
                .find({'code': {
                    '$in': code
                }},
                      {"_id": 0},
                      batch_size=10000)
            ]
        )
        # data['date'] = pd.to_datetime(data['date'], utc=False)
        return data.set_index('code', drop=False)
    except Exception:
        # QA_util_log_info(e)
        return None


def GQ_fetch_stock_name(code, collections=DATABASE.stock_list, ):
    """
    获取股票名称
    """
    if isinstance(code, str):
        try:
            res = collections.find_one({'code': code})
            return res['name']
        except Exception:
            # QA_util_log_info(e)
            return code
    elif isinstance(code, list):
        code = GQ_util_code_tolist(code)
        data = pd.DataFrame(
            [
                item for item in collections
                .find({'code': {
                    '$in': code
                }},
                      {"_id": 0},
                      batch_size=10000)
            ]
        )
        # data['date'] = pd.to_datetime(data['date'], utc=False)
        return data.set_index('code', drop=False)


def GQ_fetch_index_name(code, collections=DATABASE.etf_list, ):
    """
    获取股票名称
    """
    if isinstance(code, str):
        try:
            res = collections.find_one({'code': code})
            return res['name']
        except Exception:
            # QA_util_log_info(e)
            return code
    elif isinstance(code, list):
        code = GQ_util_code_tolist(code)
        data = pd.DataFrame(
            [
                item for item in collections
                .find({'code': {
                    '$in': code
                }},
                      {"_id": 0},
                      batch_size=10000)
            ]
        )
        # data['date'] = pd.to_datetime(data['date'], utc=False)
        return data.set_index('code', drop=False)


def GQ_fetch_etf_name(
    code,
    collections=DATABASE.etf_list
):
    """
    获取ETF名称
    """
    if isinstance(code, str):
        try:
            return collections.find_one({'code': code})['name']
        except Exception:
            # QA_util_log_info(e)
            return code
    elif isinstance(code, list):
        code = GQ_util_code_tolist(code)
        data = pd.DataFrame(
            [
                item for item in collections
                .find({'code': {
                    '$in': code
                }},
                      {"_id": 0},
                      batch_size=10000)
            ]
        )
        # data['date'] = pd.to_datetime(data['date'], utc=False)
        return data.set_index('code', drop=False)


def get_codelist(codepool):
    """
    将各种‘随意’写法的A股股票代码列表，转换为6位数字list标准规格，
    可以是“,”斜杠，“，”，可以是“、”，或者其他类似全角或者半角符号分隔。
    """
    if (isinstance(codepool, str)):
        codelist = [code.strip() for code in codepool.splitlines()]
    elif (isinstance(codepool, list)):
        codelist = codepool
    else:
        print(u'Unsolved stock_cn code/symbol string:{}'.format(codepool))

    ret_codelist = []
    for code in codelist:
        if (len(code) > 6):
            try_split_codelist = code.split('/')
            if (len(try_split_codelist) > 1):
                ret_codelist.extend(try_split_codelist)
            elif (len(code.split(' ')) > 1):
                try_split_codelist = code.split(' ')
                ret_codelist.extend(try_split_codelist)
            elif (len(code.split('、')) > 1):
                try_split_codelist = code.split('、')
                ret_codelist.extend(try_split_codelist)
            elif (len(code.split('，')) > 1):
                try_split_codelist = code.split('，')
                ret_codelist.extend(try_split_codelist)
            elif (len(code.split(',')) > 1):
                try_split_codelist = code.split(',')
                ret_codelist.extend(try_split_codelist)
            elif (code.startswith('XSHE')) or \
                (code.endswith('XSHE')):
                ret_codelist.append('{}.XSHE'.format(GQ_util_code_tostr(code)))
                pass
            elif (code.startswith('XSHG')) or \
                (code.endswith('XSHG')):
                # Ztry_split_codelist = code.split('.')
                ret_codelist.append('{}.XSHG'.format(GQ_util_code_tostr(code)))
            else:
                if (GQ_util_code_tostr(code)):
                    pass
                print(u'Unsolved stock_cn code/symbol string:{}'.format(code))
        else:
            ret_codelist.append(code)

    # 去除空字符串
    ret_codelist = list(filter(None, ret_codelist))
    # print(ret_codelist)

    # 清除尾巴
    ret_codelist = [code.strip(',') for code in ret_codelist]
    ret_codelist = [code.strip('\'') for code in ret_codelist]
    ret_codelist = [code.strip('’') for code in ret_codelist]
    ret_codelist = [code.strip('‘') for code in ret_codelist]

    # 去除重复代码
    ret_codelist = list(set(ret_codelist))

    return ret_codelist


def stock_cn_blacklist():
    """
    股票黑名单，这些票没有当前行情数据，容错会影响策略计算速度
    blacklist：退市而依然在tushare或者tdx的股票列表中
    banedlist：完全不知道代码出处
    brandnew ：新股未挂牌就已经出现在tushare或者tdx的股票列表中，
               需要新股模块定时更新状态，在适当的时候（挂牌交易累积120小时后）正常加入策略
    """
    ret_blacklist = ['000018', '000792', '000670', '000760',
                     '002260', '002359', '002450', '002710',
                     '002720', '002711', '300362', '300156',
                     '600485', '603302',]
    ret_banedlist = ['602227', '693690', '688688',]

    # 新股发行流程中，需要代码定期检查是否挂牌交易(存在暴雷可能性)
    ret_brandnew_list = ['301005', '301001', '600905', '688067',
                         '688681', '688269', '301010', '301012',
                         '301003', '688625', '301007', '688216',
                         '301009', '688621', '603529', '601156',
                         '688425', '300728', '605028', '688131',
                         '605319', '688597', '688517', '601206',
                         '300984', '001208', '688319', '605259',
                         '301002', '301011', '601528', '301008',
                         '688700', '301006', '603302', '001207',
                         '688276', '688690', '688345', '601665',
                         '301004', '301013', '300991', '688682',
                         '688260']
    return list(set(ret_blacklist + ret_banedlist + ret_brandnew_list))


def GQ_util_firstDayTrading(codelist: list):
    """
    explanation:
        获取交易品种的第一个上市日期，或第一个交易日。支持混合股票,index,etf		

    params:
        * codelist ->:
            meaning: stcok/index/etf 代码列表
            type: list
            optional: [null]

    return:
        pandas.DataFrame: the code with its first trading date

    demonstrate:
        QA_util_firstDayTrading(['600066','510050','000300'])

    output:
        Not described
    """

    coll_stock_day = DATABASE.stock_day
    coll_index_day = DATABASE.index_day
    coll_stock_day.create_index(
    [("code", pymongo.ASCENDING),
     ("date_stamp", pymongo.ASCENDING)]
    )
    coll_index_day.create_index(
    [("code",
      pymongo.ASCENDING),
     ("date_stamp",
      pymongo.ASCENDING)]
    )

    dates = []
    for code in codelist:
        # print('{} is ref is {}, ref2 is {}'.format(code, ref.count(), ref2.count()))
        if coll_stock_day.count_documents({"code": code}) > 0:
            ref = coll_stock_day.find({"code": code})
            start_date = ref[0]['date']
            dates.append(start_date)
        elif coll_index_day.count_documents({"code": code}) > 0:
            ref2 = coll_index_day.find({'code': code})
            start_date = ref2[0]['date']
            dates.append(start_date)
        else:
            dates.append(None)
            # raise ValueError('{} 没有数据'.format(code))

    return pd.DataFrame({
        AKA.CODE: codelist,
        AKA.IPO_DATE: dates}).set_index(
            [AKA.CODE],
            drop=False)


@lru_cache()
def GQ_fetch_stock_list():
    """
    """
    cachefile_base = 'codelist_firstDayTrading.pickle'
    cachefile_code_list_firstDayTrading = get_pickle_filename(cachefile_base)

    stock_items = QA_fetch_stock_list()
    code_list = list(set([stock['code'] for index, stock in stock_items.iterrows()]))

    try:
        code_list_firstDayTrading = load_snapshot_cache(
            mkdirs(
                cache_path(
                    'stock_cn',
                    portable=False
                )
            ),
            cachefile_code_list_firstDayTrading
        )
        brandnew_stock_pd = set(stock_items[AKA.CODE].to_list()).difference(code_list_firstDayTrading[AKA.CODE].to_list())
        if (len(brandnew_stock_pd) > 0):
            code_list_firstDayTrading = pd.concat([code_list_firstDayTrading,
                                                stock_items[(stock_items[AKA.CODE].isin(list(brandnew_stock_pd)))]], 
                                                axis=0)
        # code_list_firstDayTrading.loc[stock_items[(stock_items['pre_close']>0.1)].index, 
        #                             [AKA.NAME,
        #                             'pre_close']]=stock_items.loc[stock_items[(stock_items['pre_close']>0.1)].index, 
        #                                                         [AKA.NAME,
        #                                                         'pre_close']]
        # Example: Directly assign values to the DataFrame
        code_list_firstDayTrading.loc[stock_items[(stock_items['pre_close'] > 0.1)].index, [AKA.NAME, 'pre_close']] = stock_items.loc[stock_items[(stock_items['pre_close'] > 0.1)].index, [AKA.NAME, 'pre_close']]

        code_list_firstDayTrading.loc[stock_items[(stock_items['pre_close'] > 0.1)].index,
                                    AKA.EXPIRED_TIME]=pd.to_datetime(dt.now().date())+timedelta(hours=33.5)
        pre_ipo_stock_pd=code_list_firstDayTrading[(code_list_firstDayTrading[AKA.IPO_DATE].isnull())].copy()
        pre_ipo_stocklist=pre_ipo_stock_pd[(pd.to_datetime(dt.now())-pre_ipo_stock_pd[AKA.EXPIRED_TIME])>timedelta(seconds=5)][AKA.CODE].to_list()
        if (len(pre_ipo_stocklist)>0.000168):
            pre_ipo_codelist_firstDayTrading=GQ_util_firstDayTrading(pre_ipo_stocklist)
            pre_ipo_stock_pd[AKA.EXPIRED_TIME]=None
            pre_ipo_stock_pd.loc[pre_ipo_codelist_firstDayTrading[AKA.IPO_DATE].isnull().index, 
                                AKA.EXPIRED_TIME]=pd.to_datetime(dt.now().date())+timedelta(hours=33.5)
            code_list_firstDayTrading.loc[pre_ipo_stock_pd.index, :]=pre_ipo_stock_pd
            
            # print(code_list_firstDayTrading[code_list_firstDayTrading['date'].isnull()])
            save_snapshot_cache(cache_path('stock_cn', 
                                           portable=False), 
                                cachefile_code_list_firstDayTrading, 
                                code_list_firstDayTrading)
        
    except Exception:
        if os.access(os.path.join(cache_path('stock_cn', portable=False), 
                                            cachefile_code_list_firstDayTrading), 
                     os.F_OK):
            traceback.print_exc()
        else:
            code_list_firstDayTrading=GQ_util_firstDayTrading(code_list)
            code_list_firstDayTrading[AKA.EXPIRED_TIME]=pd.to_datetime(GQ_util_get_last_day())+timedelta(hours=33.5)
            if (len(code_list_firstDayTrading)==len(stock_items)):
                code_list_firstDayTrading=stock_items.join(code_list_firstDayTrading.drop(columns=stock_items.columns.intersection(code_list_firstDayTrading.columns)))
                code_list_firstDayTrading.loc[code_list_firstDayTrading[AKA.IPO_DATE].isnull().index, 
                                              AKA.EXPIRED_TIME]=pd.to_datetime(dt.now().date())+timedelta(hours=33.5)

            save_snapshot_cache(cache_path('stock_cn', 
                                           portable=False), 
                                cachefile_code_list_firstDayTrading, 
                                code_list_firstDayTrading)
        
    return code_list_firstDayTrading


@lru_cache()
def GQ_fetch_etf_list():
    """
    """
    cachefile_base = 'etflist_firstDayTrading.pickle'
    cachefile_code_list_firstDayTrading = get_pickle_filename(cachefile_base)
    stock_items = QA_fetch_stock_list()
    code_list = list(set([stock['code'] for index, stock in stock_items.iterrows()]))
    
    try:
        code_list_firstDayTrading = load_snapshot_cache(mkdirs(cache_path('stock_cn', 
                                                                portable=False)),
                                                                cachefile_code_list_firstDayTrading)
        brandnew_stock_pd = set(stock_items[AKA.CODE].to_list()).difference(code_list_firstDayTrading[AKA.CODE].to_list())
        if (len(brandnew_stock_pd) > 0):
            code_list_firstDayTrading = pd.concat([code_list_firstDayTrading,
                                                stock_items[(stock_items[AKA.CODE].isin(list(brandnew_stock_pd)))]], 
                                                axis=0)
        # code_list_firstDayTrading.loc[stock_items[(stock_items['pre_close']>0.1)].index, 
        #                             [AKA.NAME,
        #                             'pre_close']]=stock_items.loc[stock_items[(stock_items['pre_close']>0.1)].index, 
        #                                                         [AKA.NAME,
        #                                                         'pre_close']]
        # Example: Directly assign values to the DataFrame
        code_list_firstDayTrading.loc[stock_items[(stock_items['pre_close'] > 0.1)].index, [AKA.NAME, 'pre_close']] = stock_items.loc[stock_items[(stock_items['pre_close'] > 0.1)].index, [AKA.NAME, 'pre_close']]

        code_list_firstDayTrading.loc[stock_items[(stock_items['pre_close']>0.1)].index,
                                    AKA.EXPIRED_TIME]=pd.to_datetime(dt.now().date())+timedelta(hours=33.5)
        pre_ipo_stock_pd=code_list_firstDayTrading[(code_list_firstDayTrading[AKA.IPO_DATE].isnull())].copy()
        pre_ipo_stocklist=pre_ipo_stock_pd[(pd.to_datetime(dt.now())-pre_ipo_stock_pd[AKA.EXPIRED_TIME])>timedelta(seconds=5)][AKA.CODE].to_list()
        if (len(pre_ipo_stocklist)>0.000168):
            pre_ipo_codelist_firstDayTrading=GQ_util_firstDayTrading(pre_ipo_stocklist)
            pre_ipo_stock_pd[AKA.EXPIRED_TIME]=None
            pre_ipo_stock_pd.loc[pre_ipo_codelist_firstDayTrading[AKA.IPO_DATE].isnull().index, 
                                AKA.EXPIRED_TIME]=pd.to_datetime(dt.now().date())+timedelta(hours=33.5)
            code_list_firstDayTrading.loc[pre_ipo_stock_pd.index, :]=pre_ipo_stock_pd
            
            #print(code_list_firstDayTrading[code_list_firstDayTrading['date'].isnull()])
            save_snapshot_cache(cache_path('stock_cn', 
                                           portable=False), 
                                cachefile_code_list_firstDayTrading, 
                                code_list_firstDayTrading)
        
    except Exception:
        if os.access(os.path.join(cache_path('stock_cn', portable=False), 
                                            cachefile_code_list_firstDayTrading), 
                     os.F_OK):
            traceback.print_exc()
        else:
            code_list_firstDayTrading=GQ_util_firstDayTrading(code_list)
            code_list_firstDayTrading[AKA.EXPIRED_TIME]=pd.to_datetime(GQ_util_get_last_day())+timedelta(hours=33.5)
            if (len(code_list_firstDayTrading)==len(stock_items)):
                code_list_firstDayTrading=stock_items.join(code_list_firstDayTrading.drop(columns=stock_items.columns.intersection(code_list_firstDayTrading.columns)))
                code_list_firstDayTrading.loc[code_list_firstDayTrading[AKA.IPO_DATE].isnull().index, 
                                              AKA.EXPIRED_TIME]=pd.to_datetime(dt.now().date())+timedelta(hours=33.5)

            save_snapshot_cache(cache_path('stock_cn', 
                                           portable=False), 
                                cachefile_code_list_firstDayTrading, 
                                code_list_firstDayTrading)
        
    return code_list_firstDayTrading
