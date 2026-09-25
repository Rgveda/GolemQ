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
def _fetch_stock_list_mongo(collections=None):
    """股票列表（**GolemQ 自己的实现，不再经 QUANTAXIS**）。

    读的是 ``DATABASE.stock_list``（= ``DATABASE_QA`` = 4.4 ``quantaxis``），
    与 QUANTAXIS 的 ``QA_fetch_stock_list()`` **同一个集合** ——
    所以这是等价替换，不改变数据来源。

    ⚠️ 本函数与 ``scribe.QA_fetch_stock_list`` 是同一个读法的两份实现，
    原因是 **import 成环**：``scribe.py:54`` 反向 import 了本模块的
    ``is_stock_cn``，本模块若再 import ``scribe`` 就成环。
    等参考集合迁到 8.3 后，两者都应改指向 ``markets/StockCN/refdata.py``
    （那是叶子模块，无环），届时可合并。
    """
    coll = collections if collections is not None else DATABASE.stock_list
    return pd.DataFrame([item for item in coll.find()]).drop(
        '_id', axis=1, inplace=False).set_index('code', drop=False)
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
        return [normalize_code(each_symbol) for each_symbol in symbol]
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


# ---------------------------------------------------------------------------
# A 股号段表 —— `is_stock_cn` 的**唯一分类依据**
# ---------------------------------------------------------------------------
# 每条 ``(前缀, market_type, 交易所别名, 中文描述)``。匹配时按**前缀长度降序**，
# 长的先命中 —— 这解决了 ``200`` 被 ``20`` 抢先、``399`` 被 ``39`` 抢先这类
# **包含关系**。原实现靠 ``elif`` 的书写顺序来保证，极脆：深市 B 股分支
# （``200``）就是因此被 ``20`` 压死了多年（见 ``PITFALLS.md`` P12）。
#
# ⚠️ **ETF 的号段口径以交易所规则为准，不是以"看起来像"为准。** 原实现把
# 深市 ``150``(分级子份额) / ``16x``(LOF) / ``180``(REITs) / ``20``(B股) 一股脑
# 判成「深交所ETF基金」，5 段里错了 4 段。核实来源与结论见
# ``MIGRATION_STATUS.md`` 的「ETF 独立成 ETF_CN」记录。
#
# 沪市：50x 基金(封基/LOF/分级) · 51x–58x ETF · 60x/688/689 股票 · 900 B股
_SH_SEGMENTS = (
    ('688', MARKET_TYPE.STOCK_CN, 'SH', '上交所科创板'),
    ('689', MARKET_TYPE.STOCK_CN, 'SH', '上交所科创板存托凭证'),
    ('900', MARKET_TYPE.STOCK_CN, 'SH', '上交所B股'),
    ('60', MARKET_TYPE.STOCK_CN, 'SH', '上交所主板'),
    ('50', MARKET_TYPE.FUND_CN, 'SH', '上交所基金(封基/LOF/分级)'),
    ('51', MARKET_TYPE.ETF_CN, 'SH', '上交所ETF'),
    ('52', MARKET_TYPE.ETF_CN, 'SH', '上交所ETF'),
    ('53', MARKET_TYPE.ETF_CN, 'SH', '上交所单市场股票ETF'),
    ('55', MARKET_TYPE.ETF_CN, 'SH', '上交所单市场债券ETF'),
    ('56', MARKET_TYPE.ETF_CN, 'SH', '上交所跨市场股票ETF'),
    ('58', MARKET_TYPE.ETF_CN, 'SH', '上交所ETF(含科创板ETF)'),
)

# 深市：159/158 ETF · 150 分级子份额 · 16x LOF · 180 REITs · 184 封基
#       000/001/002/003/300–303 股票 · 200 B股 · 28 B股配股权证 · 399 指数
_SZ_SEGMENTS = (
    ('159', MARKET_TYPE.ETF_CN, 'SZ', '深交所ETF'),
    ('158', MARKET_TYPE.ETF_CN, 'SZ', '深交所ETF'),
    ('150', MARKET_TYPE.FUND_CN, 'SZ', '深交所分级基金子份额'),
    ('184', MARKET_TYPE.FUND_CN, 'SZ', '深交所封闭式基金'),
    ('180', MARKET_TYPE.FUND_CN, 'SZ', '深交所基础设施基金(REITs)'),
    # B股是**整个 ``20`` 段**（200–209），不只 ``200``——深交所规则「B股首二位为 20」
    ('20', MARKET_TYPE.STOCK_CN, 'SZ', '深交所B股'),
    ('399', MARKET_TYPE.INDEX_CN, 'SZ', '中证指数'),
    ('300', MARKET_TYPE.STOCK_CN, 'SZ', '深交所创业板'),
    ('301', MARKET_TYPE.STOCK_CN, 'SZ', '深交所创业板'),
    ('302', MARKET_TYPE.STOCK_CN, 'SZ', '深交所创业板'),
    ('303', MARKET_TYPE.STOCK_CN, 'SZ', '深交所创业板'),
    ('002', MARKET_TYPE.STOCK_CN, 'SZ', '深交所中小板'),
    ('003', MARKET_TYPE.STOCK_CN, 'SZ', '深交所主板'),
    ('000', MARKET_TYPE.STOCK_CN, 'SZ', '深交所主板'),
    ('001', MARKET_TYPE.STOCK_CN, 'SZ', '深交所主板'),
    ('16', MARKET_TYPE.FUND_CN, 'SZ', '深交所LOF'),
    ('28', MARKET_TYPE.STOCK_CN, 'SZ', '深交所B股配股权证'),
)

# 北交所/新三板：920 北交所（2025-10-09 起存量已全部切换为该段）·
#                43/83/87/88 股转系统股票 · 82 优先股
# ⚠️ 原实现把 ``82`` 也写成「北证A股」—— 官方口径 ``82``/``820`` 是**优先股**。
_BJ_SEGMENTS = (
    ('92', MARKET_TYPE.STOCK_CN, 'BJ', '北交所'),
    ('88', MARKET_TYPE.STOCK_CN, 'BJ', '全国股转系统股票'),
    ('87', MARKET_TYPE.STOCK_CN, 'BJ', '全国股转系统股票'),
    ('83', MARKET_TYPE.STOCK_CN, 'BJ', '全国股转系统股票'),
    ('43', MARKET_TYPE.STOCK_CN, 'BJ', '全国股转系统股票'),
    ('82', MARKET_TYPE.STOCK_CN, 'BJ', '全国股转系统优先股'),
)

#: 深市 ``000`` 段里**其实是指数**的少数代码（沪深300 等）。硬编码沿用原样 ——
#: 注意它们返回 ``'SH'`` 交易所别名，这是原行为，**勿"修正"**成 ``'SZ'``。
_SZ_INDEX_CODES = frozenset(
    {'000003', '000112', '000300', '000132', '000133'})


def _by_length(segments):
    """按前缀**长度降序**排好，供 :func:`_match_segment` 顺序匹配。

    在模块加载时做一次（而不是每次调用排序）—— ``is_stock_cn`` 在热路径上。
    """
    return tuple(sorted(segments, key=lambda seg: -len(seg[0])))


_SH_SEGMENTS = _by_length(_SH_SEGMENTS)
_SZ_SEGMENTS = _by_length(_SZ_SEGMENTS)
_BJ_SEGMENTS = _by_length(_BJ_SEGMENTS)


def _match_segment(segments, bare):
    """在号段表里按最长前缀匹配，命中返回 ``(market_type, 交易所, 描述)``。

    >>> _match_segment(_SH_SEGMENTS, '510300')
    ('etf_cn', 'SH', '上交所ETF')
    >>> _match_segment(_SZ_SEGMENTS, '200037')     # B股，不再被 '20' 抢走
    ('stock_cn', 'SZ', '深交所B股')
    >>> _match_segment(_SZ_SEGMENTS, '159915')
    ('etf_cn', 'SZ', '深交所ETF')
    >>> _match_segment(_SZ_SEGMENTS, '160105')     # LOF，不是 ETF
    ('fund_cn', 'SZ', '深交所LOF')
    """
    for prefix, market_type, exchange, desc in segments:
        if bare.startswith(prefix):
            return market_type, exchange, desc
    return None


#: 交易所 token（**大小写不敏感**）→ `is_stock_cn` 返回的交易所别名。
#:
#: - ``SH`` / ``XSHG`` —— 上交所。``XSHG`` 取「上**海**」拼音里的 g
#:   （``XSHG`` = **Shan(g)hai**），好与深圳区分。
#: - ``SZ`` / ``XSHE`` —— 深交所。``XSHE`` 取「深**圳**」的 e
#:   （**Sh(e)nzhen**）。
#: - ``BJ`` —— 北交所。
#:
#: ⚠️ **只收真实存在的写法。** 曾考虑再加一个 ``CZ`` 当深交所别名（源自一次
#: 口头笔误），查证后确认**两棵树里都零出现**，故不加 —— 多容忍一个不存在的
#: token，只会让**打错的代码被静默当成深交所**；正确行为是判成不认识。
_EXCHANGE_TOKENS = {
    'SH': 'SH', 'XSHG': 'SH',
    'SZ': 'SZ', 'XSHE': 'SZ',
    'BJ': 'BJ',
}

#: 没有显式交易所时，靠号段就能断定属于沪市的 3 位前缀（债券/回购等）。
_SH_FORCED_3 = ('009', '126', '110', '201', '202', '203', '204')
#: 没有显式交易所时，靠号段就能断定属于深市的 3 位前缀。
_SZ_FORCED_3 = ('000', '001', '002', '200', '300', '159')


def _split_cn_code(raw):
    """把各种带交易所标记的写法拆成 ``(裸6位代码, 交易所别名或 None)``。

    **长度 >6 的判断只在这里做一次** —— 原实现把长度判断散在沪/深两个分支里
    各写一遍（``len==8`` / ``len==11 & startswith`` / ``len==11 & endswith``），
    而且两处**只认自己的交易所**：于是 ``bj430489`` 谁都进不去，直接返回
    ``False``（"不是 A 股"）。

    支持的形态（长度 >6 的全部保留，见 ``MISSING``/原契约）：

    >>> _split_cn_code('600519')          # 裸 6 位
    ('600519', None)
    >>> _split_cn_code('sh600519')        # 紧贴前缀
    ('600519', 'SH')
    >>> _split_cn_code('sh.600000')       # 点号前缀（树里 qmt_source 用这种）
    ('600000', 'SH')
    >>> _split_cn_code('600519.XSHG')     # 点号后缀
    ('600519', 'SH')
    >>> _split_cn_code('bj430489')        # 北交所 —— 原实现认不出来
    ('430489', 'BJ')
    >>> _split_cn_code('XSHG600519')      # QUANTAXIS 的前缀式
    ('600519', 'SH')
    >>> _split_cn_code('cz000001')        # 不存在的 token 不猜
    ('cz000001', None)
    """
    s = str(raw).strip()
    if not s or len(s) <= 6:
        return s, None

    if '.' in s:
        left, right = s.rsplit('.', 1)
        alias = _EXCHANGE_TOKENS.get(right.upper())
        if alias and len(left) >= 6:
            return left[-6:], alias
        alias = _EXCHANGE_TOKENS.get(left.upper())
        if alias and len(right) >= 6:
            return right[-6:], alias
        return s, None

    alias = _EXCHANGE_TOKENS.get(s[:2].upper())
    if alias and len(s) - 2 >= 6:
        return s[-6:], alias

    alias = _EXCHANGE_TOKENS.get(s[:4].upper())
    if alias and len(s) >= 10:
        return s[4:10], alias

    for suffix in ('XSHG', 'XSHE'):
        if s.upper().endswith(suffix) and len(s) >= 6:
            return s[:-4][-6:], _EXCHANGE_TOKENS[suffix]

    return s, None


def _classify_sh(bare):
    """沪市号段 → 四元组。``000`` 段**只在有显式沪市标记时**才算指数。"""
    if bare.startswith('000'):
        return True, MARKET_TYPE.INDEX_CN, 'SH', '上交所指数'
    hit = _match_segment(_SH_SEGMENTS, bare)
    if hit is not None:
        return True, hit[0], hit[1], hit[2]
    print(bare, True, None, 'SH', '上交所未知代码')
    return True, None, 'SH', '上交所未知代码'


def _classify_sz(bare):
    """深市号段 → 四元组。"""
    if bare in _SZ_INDEX_CODES:
        return True, MARKET_TYPE.INDEX_CN, 'SH', '中证指数'
    hit = _match_segment(_SZ_SEGMENTS, bare)
    if hit is not None:
        return True, hit[0], hit[1], hit[2]
    print(bare, True, None, 'SZ', '深交所未知代码')
    return True, None, 'SZ', '深交所未知代码'


def _classify_bj(bare):
    """北交所 / 新三板号段 → 四元组。"""
    hit = _match_segment(_BJ_SEGMENTS, bare)
    if hit is not None:
        return True, hit[0], hit[1], hit[2]
    print(bare, True, None, 'BJ', '北交所未知代码')
    return True, None, 'BJ', '北交所未知代码'


def is_stock_cn(code):
    """判断 股票代码，市场来源，板块。

    返回 ``(是否A股, market_type, 交易所别名, 中文描述)`` —— **四元组形状与
    位置语义是全树契约**（十余处按位置解包），不得改动。

    ``market_type`` 见 :class:`GolemQ.core.constants.MARKET_TYPE`；
    **ETF 现为独立的 ``ETF_CN``**（不再混进 ``INDEX_CN``）。

    长度 >6 的写法（``sh600519`` / ``sh.600000`` / ``600519.XSHG`` /
    ``bj430489`` / ``cz000001``）由 :func:`_split_cn_code` 统一拆解，
    **显式交易所标记优先于号段推断**。

    >>> is_stock_cn('600519')[1]
    'stock_cn'
    >>> is_stock_cn('510300')[1]                   # ETF 是与指数并列的一等类型
    'etf_cn'
    >>> is_stock_cn('000300')[1]                   # 真指数仍是索引
    'index_cn'
    >>> is_stock_cn('200037')[3]                   # 深市B股，曾被 '20' 误判成 ETF
    '深交所B股'
    >>> is_stock_cn('160105')[1]                   # LOF —— 基金，不是 ETF
    'fund_cn'
    >>> is_stock_cn('820001')[3]                   # 优先股，原写成「北证A股」
    '全国股转系统优先股'
    >>> is_stock_cn('bj430489')[2]                 # 带交易所标记的写法不该丢
    'BJ'
    >>> is_stock_cn('sh.600000')[1]
    'stock_cn'
    """
    symbol = code
    if (isinstance(code, list)):
        for each_code in code:
            return is_stock_cn(each_code)

    code = str(code)
    if (len(code) == 0):
        print(symbol, u'长度为零')
        return False, None, None, None

    bare, hint = _split_cn_code(code)

    # 显式交易所标记优先 —— 它专门用来消解 000xxx 这类固有歧义
    if hint == 'SH':
        return _classify_sh(bare)
    if hint == 'SZ':
        return _classify_sz(bare)
    if hint == 'BJ':
        return _classify_bj(bare)

    # —— 北交所 / 新三板：原实现放在最前，保持同样的优先级 ——
    if bare[:2] in ('92', '88', '87', '83', '82', '43'):
        return _classify_bj(bare)

    if bare[0] in ('5', '6', '9') or bare[:3] in _SH_FORCED_3 + ('688', '689'):
        return _classify_sh(bare)

    if bare[0] in ('0', '2', '3') or bare[:2] in ('15', '16', '18') \
            or bare[:3] in _SZ_FORCED_3:
        return _classify_sz(bare)

    print(code, isinstance(code, list), '不知道')
    return False, None, None, None


def is_future_cn(code):
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


def GQ_fetch_index_name(code, collections=DATABASE.index_list, ):
    """指数名称。QA 版 `QA_fetch_index_name` 的忠实替身（同一读法、同一集合）。

    ⚠️ 默认集合原为 `DATABASE.etf_list` —— 那是从下面 `GQ_fetch_etf_name`
    复制粘贴来的错误。**`etf_list` 里没有真指数**：实测 `000300` 在
    `golemq_stock_cn.etf_list` 命中 0 行（该集合 `sec` 恒为 `etf_cn`），而在
    `DATABASE.index_list` 里是 `'沪深300'`（1291 行，`sec='index_cn'`）。
    用错集合的症状是**静默退化**：查不到就 `return code`，指数名变成一串数字。

    该默认值此前从未生效（本函数零调用者），所以这是一处潜在缺陷修复，
    不改变任何既有行为。
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
    ret_bannedlist = ['602227', '693690', '688688',]

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
    return list(set(ret_blacklist + ret_bannedlist + ret_brandnew_list))


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

    stock_items = _fetch_stock_list_mongo()
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
    stock_items = _fetch_stock_list_mongo()
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
