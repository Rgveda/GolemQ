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
import QUANTAXIS as QA
try:
    import QUANTAXIS as QA
except:
    print('PLEASE run "pip install QUANTAXIS" before call GolemQ.utils.symbol modules')
    pass

from QUANTAXIS.QAUtil.QACode import (QA_util_code_tostr)
from GolemQ.utils.const import _const
    
class EXCHANGE(_const):
    XSHG = 'XSHG'
    SSE = 'XSHG'
    SH = 'XSHG'
    XSHE = 'XSHE'
    SZ = 'XSHE'
    SZE = 'XSHE'


def normalize_code(symbol, pre_close=None):
    """
    归一化证券代码

    :param code 如000001
    :return 证券代码的全称 如000001.XSHE
    """
    if (not isinstance(symbol, str)):
        return symbol

    if (symbol.startswith('sz') and (len(symbol) == 8)):
        ret_normalize_code = '{}.{}'.format(symbol[2:8], EXCHANGE.SZ)
    elif (symbol.startswith('sh') and (len(symbol) == 8)):
        ret_normalize_code = '{}.{}'.format(symbol[2:8], EXCHANGE.SH)
    elif (symbol.startswith('00') and (len(symbol) == 6)):
        if ((pre_close is not None) and (pre_close > 2000)):
            # 推断是上证指数
            ret_normalize_code = '{}.{}'.format(symbol, EXCHANGE.SH)
        else:
            ret_normalize_code = '{}.{}'.format(symbol, EXCHANGE.SZ)
    elif ((symbol.startswith('399') or symbol.startswith('159') or \
        symbol.startswith('150')) and (len(symbol) == 6)):
        ret_normalize_code = '{}.{}'.format(symbol, EXCHANGE.SH)
    elif ((len(symbol) == 6) and (symbol.startswith('399') or \
        symbol.startswith('159') or symbol.startswith('150') or \
        symbol.startswith('16') or symbol.startswith('184801') or \
        symbol.startswith('201872'))):
        ret_normalize_code = '{}.{}'.format(symbol, EXCHANGE.SZ)
    elif ((len(symbol) == 6) and (symbol.startswith('50') or \
        symbol.startswith('51') or symbol.startswith('60') or \
        symbol.startswith('688') or symbol.startswith('900') or \
        (symbol == '751038'))):
        ret_normalize_code = '{}.{}'.format(symbol, EXCHANGE.SH)
    elif ((len(symbol) == 6) and (symbol[:3] in ['000', '001', '002',
                                                 '200', '300'])):
        ret_normalize_code = '{}.{}'.format(symbol, EXCHANGE.SZ)
    elif symbol.startswith('XSHG'):
        ret_normalize_code = '{}.{}'.format(symbol[5:], EXCHANGE.SH)
    elif symbol.startswith('XSHE'):
        ret_normalize_code = '{}.{}'.format(symbol[5:], EXCHANGE.SZ)
    elif (symbol.endswith('XSHG') or symbol.endswith('XSHE')):
        ret_normalize_code = symbol
    else:
        print(u'normalize_code():', symbol)
        ret_normalize_code = symbol

    return ret_normalize_code


def is_stock_cn(code):
    """
    1- sh
    0 -sz
    """
    code = str(code)
    if code[0] in ['5', '6', '9'] or \
        code[:3] in ["009", "126", "110", "201", "202", "203", "204", '688'] or \
        (code.startswith('XSHG')) or \
        (code.endswith('XSHG')):
        if (code.startswith('XSHG')) or \
             (code.endswith('XSHG')):
            if (len(code.split('.')) > 1):
                try_split_codelist = code.split('.')
                if (try_split_codelist[0] == 'XSHG') and (len(try_split_codelist[1]) == 6):
                    code = try_split_codelist[1]
                elif (try_split_codelist[1] == 'XSHG') and (len(try_split_codelist[0]) == 6):
                    code = try_split_codelist[0]
                if (code[:5] in ["00000"]) or \
                    (code[:3] in ["000"]):
                    return True, QA.MARKET_TYPE.INDEX_CN, 'SH', '上交所指数'
        if code.startswith('60') == True:
            return True, QA.MARKET_TYPE.STOCK_CN, 'SH', '上交所A股'
        elif code.startswith('688') == True:
            return True, QA.MARKET_TYPE.STOCK_CN, 'SH', '上交所科创板'
        elif code.startswith('900') == True:
            return True, QA.MARKET_TYPE.STOCK_CN, 'SH', '上交所B股'
        elif code.startswith('50') == True:
            return True, QA.MARKET_TYPE.FUND_CN, 'SH', '上交所传统封闭式基金'
        elif code.startswith('51') == True:
            return True, QA.MARKET_TYPE.FUND_CN, 'SH', '上交所ETF基金'
        else:
            print(code, True, None, 'SH', '上交所未知代码')
            return True, None, 'SH', '上交所未知代码'
    elif code[0] in ['0', '2', '3'] or \
        code[:3] in ['000', '001', '002', '200', '300', '159'] or \
        (code.startswith('XSHE')) or \
        (code.endswith('XSHE')):
        if (code.startswith('000') == True) or \
            (code.startswith('001') == True):
            if (code in ['000003', '000112', '000300', '000132', '000133']):
                return True, QA.MARKET_TYPE.INDEX_CN, 'SH', '中证指数'
            else:
                return True, QA.MARKET_TYPE.STOCK_CN, 'SZ', '深交所主板'
        if code.startswith('002') == True:
            return True, QA.MARKET_TYPE.STOCK_CN, 'SZ', '深交所中小板'
        elif code.startswith('003') == True:
            return True, QA.MARKET_TYPE.STOCK_CN, 'SZ', '中广核？？'
        elif code.startswith('159') == True:
            return True, QA.MARKET_TYPE.FUND_CN, 'SZ', '深交所ETF基金'
        elif code.startswith('200') == True:
            return True, QA.MARKET_TYPE.STOCK_CN, 'SZ', '深交所B股'
        elif code.startswith('399') == True:
            return True, QA.MARKET_TYPE.INDEX_CN, 'SZ', '中证指数'
        elif code.startswith('300') == True:
            return True, QA.MARKET_TYPE.STOCK_CN, 'SZ', '深交所创业板'
        elif (code.startswith('XSHE')) or \
            (code.endswith('XSHE')):
             pass
        else:
            print(code, True, None, 'SZ', '深交所未知代码')
            return True, None, 'SZ', '深交所未知代码'
    else:
        print(code, '不知道')
        return False, None, None, None

def is_furture_cn(code):
    if code[:2] in ['IH', 'IF', 'IC', 'TF', 'JM', 'PP', 'EG', 'CS',
         'AU', 'AG', 'SC', 'CU', 'AL', 'ZN', 'PB', 'SN', 'NI',
         'RU', 'RB', 'HC', 'BU', 'FU', 'SP',
         'SR', 'CF', 'RM', 'MA', 'TA', 'ZC', 'FG', 'IO', 'CY']:
         return True, QA.MARKET_TYPE.FUTURE_CN, 'NA', '中国期货'
    elif code[:1] in ['A', 'B', 'Y', 'M', 'J', 'P', 'I',
                    'L', 'V', 'C', 'T']:
        return True, QA.MARKET_TYPE.FUTURE_CN, 'NA', '中国期货'
    else:
        return False, None, None, None


def is_cryptocurrency(code):
    code = str(code)
    if (code.startswith('HUOBI') == True) or code.startswith('huobi') == True or \
        code.endswith('husd') == True or code.endswith('HUSD') == True:
        return True, QA.MARKET_TYPE.CRYPTOCURRENCY, 'huobi.pro', '数字货币'
    elif code.endswith('bnb') == True or code.endswith('BNB') == True or \
        code.startswith('BINANCE') == True or code.startswith('binance') == True:
        return True, QA.MARKET_TYPE.CRYPTOCURRENCY, 'Binance', '数字货币'
    elif code.startswith('BITMEX') == True or code.startswith('bitmex') == True:
        return True, QA.MARKET_TYPE.CRYPTOCURRENCY, 'Bitmex', '数字货币'
    elif code.startswith('OKEX') == True or code.startswith('OKEx') == True or \
        code.startswith('okex') == True or code.startswith('OKCoin') == True or \
        code.startswith('okcoin') == True or code.startswith('OKCOIN') == True:
        return True, QA.MARKET_TYPE.CRYPTOCURRENCY, 'OKEx', '数字货币'
    elif code.startswith('BITFINEX') == True or code.startswith('bitfinex') == True or \
        code.startswith('Bitfinex') == True:
        return True, QA.MARKET_TYPE.CRYPTOCURRENCY, 'Bitfinex', '数字货币'
    elif (code[:-7] in ['adausdt', 'bchusdt', 'bsvusdt', 'btcusdt', 'btchusd', 
                     'eoshusd', 'eosusdt', 'etcusdt', 'etchusd', 'ethhusd', 
                     'ethusdt', 'ltcusdt', 'trxusdt', 'xmrusdt', 'xrpusdt', 
                     'zecusdt']) or \
        (code[:-8] in ['atomusdt', 'algousdt', 'dashusdt', 'dashhusd', 'hb10usdt']) or \
        (code[:-6] in ['hthusd', 'htusdt']):
        return True, QA.MARKET_TYPE.CRYPTOCURRENCY, 'huobi.pro', '数字货币'
    elif code.endswith('usd') == True or code.endswith('USD'):
        return True, QA.MARKET_TYPE.CRYPTOCURRENCY, 'NA', '数字货币'
    elif code.endswith('usdt') == True or code.endswith('USDT') == True:
        return True, QA.MARKET_TYPE.CRYPTOCURRENCY, 'NA', '数字货币'
    elif code[:3] in ['BTC', 'btc', 'ETH', 'eth',
                      'EOS', 'eos', 'ADA','ada',
                      'BSV', 'bsv', 'BCH', 'bch', 
                      'xmr', 'XMR', 'LTC', 'ltc',
                      'xrp', 'XRP', 'ZEC', 'zec',
                      'trx', 'TRX', 'ZEC', 'zec']:
        return True, QA.MARKET_TYPE.CRYPTOCURRENCY, 'NA', '数字货币'
    else:
        return False, None, None, None
    

def get_display_name_list(codepool):
    """
    将各种‘随意’写法的A股股票名称列表，转换为6位数字list标准规格，
    可以是“,”斜杠，“，”，可以是“、”，或者其他类似全角或者半角符号分隔。
    """
    if (isinstance(codepool, str)):
        codelist = [code.strip() for code in codepool.splitlines()]
    elif (isinstance(codepool, list)):
        codelist = codepool
    else:
        print(u'Unsolved stock_cn code/symbol string:{}'.format(codepool))

    ret_displaynamelist = []
    for display_name in codelist:
        if (len(display_name) > 6):
            try_split_codelist = display_name.split('/')
            if (len(try_split_codelist) > 1):
                ret_displaynamelist.extend(try_split_codelist)
            elif (len(display_name.split(' ')) > 1):
                try_split_codelist = display_name.split(' ')
                ret_displaynamelist.extend(try_split_codelist)
            elif (len(display_name.split('、')) > 1):
                try_split_codelist = display_name.split('、')
                ret_displaynamelist.extend(try_split_codelist)
            elif (len(display_name.split('，')) > 1):
                try_split_codelist = display_name.split('，')
                ret_displaynamelist.extend(try_split_codelist)
            elif (len(display_name.split(',')) > 1):
                try_split_codelist = display_name.split(',')
                ret_displaynamelist.extend(try_split_codelist)
            elif (display_name.startswith('XSHE')) or \
                (display_name.endswith('XSHE')):
                ret_displaynamelist.append('{}.XSHE'.format(QA_util_code_tostr(display_name)))
                pass
            elif (display_name.startswith('XSHG')) or \
                (display_name.endswith('XSHG')):
                #Ztry_split_codelist = code.split('.')
                ret_displaynamelist.append('{}.XSHG'.format(QA_util_code_tostr(display_name)))
            else:
                if (QA_util_code_tostr(display_name)):
                    pass
                print(u'Unsolved stock_cn code/symbol string:{}'.format(display_name))
        else:
            ret_displaynamelist.append(display_name)

    # 去除空字符串
    ret_displaynamelist = list(filter(None, ret_displaynamelist))

    # 清除尾巴
    ret_displaynamelist = [code.strip(',') for code in ret_displaynamelist]
    ret_displaynamelist = [code.strip('\'') for code in ret_displaynamelist]

    # 去除重复代码
    ret_displaynamelist = list(set(ret_displaynamelist))

    # 用中文名反查股票代码
    stock_list = QA.QA_fetch_stock_list()
    dict_stock_list = {stock['name'].replace(' ', ''):code for code, stock in stock_list.T.iteritems()}
    ret_symbol_list = [dict_stock_list[display_name] for display_name in ret_displaynamelist]

    return ret_displaynamelist, ret_symbol_list


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
                ret_codelist.append('{}.XSHE'.format(QA_util_code_tostr(code)))
                pass
            elif (code.startswith('XSHG')) or \
                (code.endswith('XSHG')):
                #Ztry_split_codelist = code.split('.')
                ret_codelist.append('{}.XSHG'.format(QA_util_code_tostr(code)))
            else:
                if (QA_util_code_tostr(code)):
                    pass
                print(u'Unsolved stock_cn code/symbol string:{}'.format(code))
        else:
            ret_codelist.append(code)

    # 去除空字符串
    ret_codelist = list(filter(None, ret_codelist))
    #print(ret_codelist)

    # 清除尾巴
    ret_codelist = [code.strip(',') for code in ret_codelist]
    ret_codelist = [code.strip('\'') for code in ret_codelist]
    ret_codelist = [code.strip('’') for code in ret_codelist]
    ret_codelist = [code.strip('‘') for code in ret_codelist]
    
    # 去除重复代码
    ret_codelist = list(set(ret_codelist))

    return ret_codelist


def get_block_symbols(blockname, stock_cn_block=None):
    """
    自定义结构板块，用于分析板块轮动，结构化行情和国家队护盘行情
    例如证券软件的“军工”板块股票超过200支过于庞大，在这里进行精选和过滤
    """
    if (stock_cn_block is None):
        stock_cn_block = QA.QA_fetch_stock_block_adv() 

    # 追加指标/行业ETF基金代码和上下游相关企业
    blockset = {'军工': ['150182', '512660', '512560', '512710'],
                '银行': ['512800', '515020', '159933', '512730', '515820', '515280'],
                '医疗': ['512170',],
                '医药': ['512170',],
                '黄金概念': ['518880', '159934'],
                '黄金': ['518880', '159934'],
                '证券': ['512000', '512880'],
                '酒': ['512690'],
                '白酒': ['512690'],
                '文化传媒': ['512980', '159805'],
                '传媒': ['512980', '159805'],
                '疫苗': ['600529'],
                '生物疫苗': ['600529'],
                }

    # 这几类上市公司主营业务比较确定，不需要额外过滤
    blockset['银行'].extend(stock_cn_block.get_block(['银行', '中小银行']).code)
    blockset['证券'].extend(stock_cn_block.get_block(['证券']).code)

    blockset['酒'].extend(stock_cn_block.get_block(['白酒', '啤酒']).code)
    blockset['白酒'] = blockset['酒']

    blockset['机场航运'] = stock_cn_block.get_block(['机场航运', '航运']).code
    blockset['航运'] = stock_cn_block.get_block(['机场航运', '航运']).code

    # 这几类鱼龙混杂，不少是“号称”，必须使用行业龙头/机构持股/指标成份过滤基本面
    lockheed = list(set(stock_cn_block.get_block(['军工']).code).intersection(stock_cn_block.get_block(['国防军工']).code))
    blockset['军工'].extend(list(set(lockheed).intersection(stock_cn_block.get_block(['行业龙头', 
                                                                                 '保险重仓', 
                                                                                 'MSCI成份',
                                                                                 'QFII重仓',
                                                                                 '信托重仓',
                                                                                 '券商重仓',
                                                                                 '基金重仓',
                                                                                 '社保重仓',
                                                                                 '证金持股']).code)))

    # 这个问题是“生物疫苗”，和"医疗器械"是否算医疗类
    medic = stock_cn_block.get_block(['生物医药', '医疗器械', '医疗改革', '医药商业']).code
    blockset['医疗'].extend(list(set(medic).intersection(stock_cn_block.get_block(['行业龙头', 
                                                                                 '保险重仓', 
                                                                                 'MSCI成份',
                                                                                 'QFII重仓',
                                                                                 '信托重仓',
                                                                                 '券商重仓',
                                                                                 '基金重仓',
                                                                                 '社保重仓',
                                                                                 '证金持股']).code)))
    blockset['医药'] = blockset['医疗']

    vaccine = stock_cn_block.get_block(['生物疫苗']).code
    blockset['疫苗'].extend(list(set(vaccine).intersection(stock_cn_block.get_block(['行业龙头', 
                                                                                 '保险重仓', 
                                                                                 'MSCI成份',
                                                                                 'QFII重仓',
                                                                                 '信托重仓',
                                                                                 '券商重仓',
                                                                                 '基金重仓',
                                                                                 '社保重仓',
                                                                                 '证金持股']).code)))
    blockset['生物疫苗'] = blockset['疫苗']

    culture = stock_cn_block.get_block(['文化传媒', '传媒', '传播与文化产业',]).code
    blockset['文化传媒'].extend(list(set(culture).intersection(stock_cn_block.get_block(['行业龙头', 
                                                                                 '保险重仓', 
                                                                                 'MSCI成份',
                                                                                 'QFII重仓',
                                                                                 '信托重仓',
                                                                                 '券商重仓',
                                                                                 '基金重仓',
                                                                                 '社保重仓',
                                                                                 '证金持股']).code)))
    blockset['传媒'] = blockset['文化传媒']

    gold = stock_cn_block.get_block(['黄金概念', '黄金']).code
    blockset['黄金'].extend(list(set(gold).intersection(stock_cn_block.get_block(['行业龙头', 
                                                                                 '保险重仓', 
                                                                                 'MSCI成份',
                                                                                 'QFII重仓',
                                                                                 '信托重仓',
                                                                                 '券商重仓',
                                                                                 '基金重仓',
                                                                                 '社保重仓',
                                                                                 '证金持股']).code)))
    blockset['黄金概念'] = blockset['黄金']

    if (blockname in blockset.keys()):
        return list(set(blockset[blockname]))
    else:
        return stock_cn_block.get_block(blockname).code


def perpar_symbol_range(eval_range):
    """
    返回预设的标的合集
    """
    if (eval_range == 'all'):
        codelist_candidate = QA.QA_fetch_stock_list()
        if (len(codelist_candidate) > 0):
            codelist_candidate = codelist_candidate[AKA.CODE].tolist()
        else:
            return False
    elif (eval_range == 'etf'):
        codelist_candidate = ['159919', '159997', '159805', '159987', 
               '159952', '159920', '518880', '159934', 
               '159985', '515050', '159994', '159941', 
               '512800', '515000', '512170', '512980', 
               '510300', '513100', '510900', '512690', 
               '510050', '159916', '512910', '510310', 
               '512090', '513050', '513030', '513500', 
               '159905', '159949', '510330', '510500', 
               '510180', '159915', '510810', '159901', 
               '512710', '510850', '512500', '512000',
               '513900', '513090']
    elif (eval_range == 'csindex'):
        codelist_candidate = ['000001.XSHG', 
                      '000002.XSHG', '000003.XSHG', '000004.XSHG', '000849.XSHG'
                      '000005.XSHG', '000006.XSHG', '000007.XSHG', '000009.XSHG',
                      '000009.XSHG', '000010.XSHG', '000015.XSHG', '000016.XSHG',  
                      '000036.XSHG', '000037.XSHG', '000038.XSHG', '000039.XSHG',  
                      '000040.XSHG', '000300.XSHG', '000112.XSHG', '000133.XSHG', 
                      '000903.XSHG', '000905.XSHG', '000906.XSHG', '000993.XSHG', 
                      '000989.XSHG', '000990.XSHG', '000991.XSHG', '000992.XSHG', 
                      '399001', '399006', '399997', '399987', 
                      '399396', '399384', '399684', '399616',
                      '399240', '399976', '399005', '000934.XSHG',
                      '399248', '399811', '399810', '000863.XSHG',
                      '399989', '399935', '399808', '399932', 
                      '399804', '399928', '399998', '399934',
                      '399967', '399986', '399933', '399233',
                      '399699', '399959',]
    elif (eval_range == 'hs300'):
        blockname = ['沪深300']
        blockname = list(set(blockname))
        codelist_candidate = QA.QA_fetch_stock_block_adv().get_block(blockname).code + perpar_symbol_range('test')
        codelist_candidate = list(set(codelist_candidate))
    elif (eval_range == 'test'):
        codepool = ['600104', '300015', '600612', '300750',
                '600585', '000651', '600436', '002475',
                '600030', '300760', '000895', '600066',
                '000661', '600887', '600352', '002352',
                '000568', '002714', '002415', '002594',
                '603713', '000858', '601138', '300122',
                '002179', '601888', '002557', '600036',
                '002271', '600298', '600276', '600547',
                '300146', '600660', '600161', '601318',
                '002050', '600900', '300498', '603515',
                '002007', '600600', '300059', '601933',
                '002258', '300715', '603899', '603444',
                '600031', '000876', '600332', '601877',
                '603288', '603520', '000333', '600563',
                '603259', '603517', '600309', '002230',
                '600009', '600519', '603486', '601100',
                '300144', '000538', '600486', '002705',
                '600570', '603129', '000963', '600738', 
                '600529', 
                '002129', '002041', '603816', '600009',
                '300677', '002304', '600893', '603185']
        codelist_candidate = get_codelist(codepool)
    elif (eval_range == 'sz150'):
        blockname = ['上证150', '上证50', '深证300']
        blockname = list(set(blockname))
        codelist_candidate = QA.QA_fetch_stock_block_adv().get_block(blockname).code + perpar_symbol_range('test')
        codelist_candidate = list(set(codelist_candidate))
    elif (eval_range == 'zz500'):
        blockname = ['中证500']
        blockname = list(set(blockname))
        codelist_candidate = QA.QA_fetch_stock_block_adv().get_block(blockname).code + perpar_symbol_range('test')
        codelist_candidate = list(set(codelist_candidate))
    else:
        eval_range = 'blocks'
        #blockname = ['中证500', '创业板50', '上证50', '上证150', '深证300']
        blockname = ['MSCI中国', 'MSCI成份', 'MSCI概念', '三网融合',
                    '上证180', '上证380', '沪深300', '上证380', 
                    '深证300', '上证50', '上证电信', '电信等权', 
                    '上证100', '上证150', '沪深300', '中证100',
                    '中证500', '全指消费', '中小板指', '创业板指',
                    '综企指数', '1000可选', '国证食品', '深证可选',
                    '深证消费', '深成消费', '中证酒', '中证白酒',
                    '行业龙头', '白酒', '证券', '消费100', 
                    '消费电子', '消费金融', '富时A50', '银行', 
                    '中小银行', '证券', '军工', '白酒', '啤酒', 
                    '医疗器械', '医疗器械服务', '医疗改革', '医药商业', 
                    '医药电商', '中药', '消费100', '消费电子', 
                    '消费金融', '黄金', '黄金概念', '4G5G', 
                    '5G概念', '生态农业', '生物医药', '生物疫苗',
                    '机场航运', '数字货币', '文化传媒']
        blockname = list(set(blockname))
        codelist_candidate = QA.QA_fetch_stock_block_adv().get_block(blockname).code
        #codelist_candidate = [code for code in codelist_candidate if not
        #code.startswith('300')]
        codelist_candidate = list(set(codelist_candidate))
        print('批量评估板块成分股：{} Total:{}'.format(blockname, 
                                                len(codelist_candidate)))

    return codelist_candidate


def stock_list_terminated():
    """
    获取退市股代码列表
    """
    import akshare as ak
    stock_info_sz_delist_df = ak.stock_info_sz_delist(indicator=u"终止上市公司")
    print(stock_info_sz_delist_df)

    stock_info_sh_delist_df = ak.stock_info_sh_delist(indicator=u"终止上市公司")
    print(stock_info_sh_delist_df)


def get_sz150s():
    from GolemQ.utils.clawer import get_csindex_stocklist_from_sina
    stocklist = get_csindex_stocklist_from_sina()
    stocklist['blockname'] = '上证150'
    #print(stocklist)
    return stocklist


if __name__ == '__main__':
    from QUANTAXIS.QACmd import QA_SU_save_stock_block
    from QUANTAXIS.QAFetch import QA_fetch_get_stock_block
    from QUANTAXIS.QAUtil import (
        DATABASE,
        QASETTING,
        QA_util_log_info, 
        QA_util_log_debug, 
        QA_util_log_expection,
        QA_util_to_json_from_pandas
    )    

    client=DATABASE
    coll = client.stock_block
    coll.create_index('code')

    #stock_block = QA_fetch_get_stock_block('tdx')
    #print(stock_block)
    #coll.insert_many(
    #    QA_util_to_json_from_pandas(stock_block)
    #)
    #from QUANTAXIS.QAFetch.QATushare import QA_fetch_get_stock_block as QATushare_fetch_get_stock_block
    #codelist = QATushare_fetch_get_stock_block()
    #print(codelist)
    #client=DATABASE
    #coll = client.stock_block
    #coll.create_index('code')
    #coll.insert_many(
    #    QA_util_to_json_from_pandas(codelist)
    #)
    #codelist = get_sz150s()
    #print(codelist)
    #coll.insert_many(
    #    QA_util_to_json_from_pandas(codelist)
    #)
    #QA_SU_save_stock_block('tdx')    
    blockname = ['中证500']
    blockname = list(set(blockname))
    codelist_candidate = QA.QA_fetch_stock_block_adv().get_block(blockname).code
    codelist_candidate = list(set(codelist_candidate))
    print(blockname,
          codelist_candidate)
    
    blockname = ['创业板50']
    blockname = list(set(blockname))
    codelist_candidate = QA.QA_fetch_stock_block_adv().get_block(blockname).code
    codelist_candidate = list(set(codelist_candidate))
    print(blockname,
          codelist_candidate)

    blockname = ['上证50']
    blockname = list(set(blockname))
    codelist_candidate = QA.QA_fetch_stock_block_adv().get_block(blockname).code
    codelist_candidate = list(set(codelist_candidate))
    print(blockname,
          codelist_candidate)

    blockname = ['上证150']
    blockname = list(set(blockname))
    codelist_candidate = QA.QA_fetch_stock_block_adv().get_block(blockname).code
    codelist_candidate = list(set(codelist_candidate))
    print(blockname,
          codelist_candidate)

    blockname = ['深证300']
    blockname = list(set(blockname))
    codelist_candidate = QA.QA_fetch_stock_block_adv().get_block(blockname).code
    codelist_candidate = list(set(codelist_candidate))
    print(blockname,
          codelist_candidate)
    