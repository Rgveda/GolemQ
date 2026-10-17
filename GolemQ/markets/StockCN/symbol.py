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

#: 「``etf``」 scan scope 的码表 —— 旧树 `GolemQ/utils/symbol.py:724-786` 那份硬编码表的
#: **逐字搬运**（用 `ast.literal_eval` 取同一个字面量，再按旧代码的 `each_code[0:6]`
#: 归一 ⇒ 506 个）。2026-10-13 从 `markets/StockCN/scopes.py` 挪到这里 —— 它是 **A 股名单
#: 字面量**，与 `focus_block` 同类；而 scope 的**解析**已经搬到跨市场的
#: :mod:`GolemQ.analysis.cv.scopes`，那边只经 `ScanDataset` 收它。
#:
#: ⚠️ **两处旧树的形状，刻意保留**：
#: 1. `'159530''159531'`（旧 `:770` **少一个逗号**）在 Python 里是**字符串拼接** ⇒
#:    实际项是 `'159530159531'`，`[0:6]` 后是 `159530` ⇒ **`159531` 从来不在清单里**。
#:    照旧（改它 = 加一只票 = 与老结果不可对拍）。
#: 2. 表里**混着 30 个真指数**（`399001` 等）—— 调用方**不能**给这批码传单一
#:    `market_type`（传 `INDEX_CN` 会让 ETF 去读 `index_*` 集合而**静默读空**），
#:    见 `GolemQ.analysis.cv.scopes` 的模块文档。
ETF_LEGACY_CODES = frozenset("""
    159001 159003 159005 159150 159300 159306 159307 159310 159321 159350 159501 159502 159503
    159505 159506 159507 159508 159509 159510 159511 159512 159513 159515 159516 159517 159518
    159519 159520 159521 159522 159523 159525 159526 159527 159528 159529 159530 159533 159535
    159536 159537 159538 159539 159540 159541 159542 159543 159545 159546 159547 159549 159551
    159552 159553 159555 159556 159557 159558 159559 159560 159561 159562 159563 159565 159566
    159567 159568 159570 159571 159572 159573 159575 159576 159577 159581 159582 159586 159588
    159589 159591 159592 159593 159595 159596 159597 159599 159601 159602 159603 159605 159606
    159607 159608 159609 159610 159611 159612 159613 159615 159616 159617 159618 159619 159620
    159621 159622 159623 159625 159627 159628 159629 159630 159631 159632 159633 159635 159636
    159637 159638 159639 159640 159641 159642 159643 159645 159646 159647 159649 159650 159651
    159652 159653 159655 159656 159657 159658 159659 159660 159661 159662 159663 159665 159666
    159667 159669 159670 159671 159672 159673 159675 159676 159677 159678 159679 159680 159681
    159682 159683 159685 159686 159687 159688 159689 159690 159691 159692 159693 159695 159696
    159697 159698 159699 159701 159703 159706 159707 159708 159709 159711 159712 159713 159715
    159716 159717 159718 159719 159720 159721 159723 159725 159726 159728 159729 159730 159731
    159732 159735 159736 159738 159739 159740 159741 159742 159743 159745 159747 159748 159750
    159751 159752 159755 159756 159757 159758 159760 159761 159763 159766 159767 159768 159769
    159770 159773 159775 159776 159777 159778 159779 159780 159781 159782 159783 159786 159787
    159788 159790 159791 159792 159793 159795 159796 159797 159798 159799 159801 159804 159805
    159806 159807 159808 159810 159811 159812 159813 159814 159816 159819 159820 159821 159822
    159823 159824 159825 159827 159828 159830 159831 159834 159835 159836 159837 159838 159839
    159840 159841 159842 159843 159845 159847 159848 159849 159850 159851 159852 159855 159856
    159857 159858 159859 159861 159862 159863 159864 159865 159866 159867 159869 159870 159871
    159872 159873 159875 159876 159877 159880 159881 159883 159885 159886 159887 159888 159889
    159890 159891 159892 159895 159896 159898 159899 159901 159902 159903 159905 159906 159907
    159908 159909 159910 159912 159913 159915 159916 159918 159919 159920 159922 159923 159925
    159928 159929 159930 159931 159933 159934 159935 159936 159937 159938 159939 159940 159941
    159943 159944 159945 159948 159949 159952 159954 159956 159957 159958 159959 159960 159961
    159964 159965 159966 159967 159968 159969 159970 159971 159972 159973 159974 159975 159976
    159977 159980 159981 159982 159985 159987 159990 159991 159992 159993 159994 159995 159996
    159997 159998 160119 399001 399005 399006 399233 399240 399248 399306 399380 399384 399396
    399616 399684 399699 399804 399808 399810 399811 399928 399932 399933 399934 399935 399959
    399967 399976 399986 399987 399989 399997 399998 510010 510020 510030 510050 510060 510090
    510100 510130 510150 510160 510170 510180 510190 510200 510210 510230 510270 510290 510300
    510310 510330 510350 510360 510370 510380 510410 510500 510810 510850 510900 512000 512010
    512040 512090 512170 512200 512260 512400 512500 512580 512660 512670 512690 512710 512720
    512760 512770 512800 512880 512890 512910 512950 512980 513010 513030 513050 513090 513100
    513500 513900 515000 515050 515070 515120 515180 515210 515220 515300 515520 515650 515700
    515790 515810 515880 516010 516020 516060 516150 516780 516880 518880 588000 588080
""".split())


def etf_scope_codes() -> list:
    """`etf` scan scope 的码表（排序后的列表）。**纯函数**。

    >>> codes = etf_scope_codes()
    >>> len(codes)
    506
    >>> '510300' in codes, '399001' in codes      # 混着真指数（见 ETF_LEGACY_CODES）
    (True, True)
    >>> '159531' in codes                         # 旧树的语法疣把它吃掉了
    False
    """
    return sorted(ETF_LEGACY_CODES)


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

