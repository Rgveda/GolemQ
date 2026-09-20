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

# from datetime import datetime, timedelta
# from functools import lru_cache
import numpy as np
import pandas as pd
# from .constants import TRADE_DATE_SSE
from GolemQ.core.settings import DATABASE
from GolemQ.core import GQ_util_code_tolist
from .date_utils import (
    GQ_util_date_valid,
    GQ_util_date_stamp,
    GQ_util_time_stamp,
    GQ_util_get_last_day,
)
from GolemQ.core.preprocessing import (
    GQ_util_to_json_from_pandas,
)
from GolemQ.core.constants import (
    MARKET_TYPE,
    AKA,
    FIELD as FLD,
)
from datetime import (
    datetime as dt,
    timedelta,
    timezone,
)
import traceback
from func_timeout import func_set_timeout
from .symbol import (
    normalize_code,
    is_stock_cn,
    is_cryptocurrency,
    GQ_fetch_stock_name,
    GQ_fetch_etf_name,
)
try:
    from QUANTAXIS.QAUtil.QADate_Adv import (
        QA_util_timestamp_to_str,
    )
except Exception:
    print('PLEASE run "pip install QUANTAXIS" before call GolemQ.analysis.machinelearning modules')
    pass
from QUANTAXIS.QAData.QADataStruct import (
        QA_DataStruct_Index_min,
        QA_DataStruct_Index_day,
        QA_DataStruct_Stock_day,
        QA_DataStruct_Stock_min,
)
import warnings
from QUANTAXIS.QAFetch.QAQuery_Advance import (
    QA_fetch_stock_block_adv,
    QA_fetch_stock_day_adv,
)
from QUANTAXIS.QAFetch.QAQuery import (
    QA_fetch_stock_list
)
from .realtime import (
    GQ_fetch_stock_day_realtime_adv,
)
from .scribe import (
    GQ_fetch_stock_moneyflow,
)


def GQ_fetch_stock_list_day(
    start,
    end,
    code=None,
    collections=DATABASE.stock_day
):
    """'获取股票日线清单'

    Returns:
        [type] -- [description]

    """
    start = str(start)[0:10]
    end = str(end)[0:10]

    if GQ_util_date_valid(end):
        if (code is None):
            cursor = collections.find(
                {
                    "date_stamp":
                        {
                            "$lte": GQ_util_date_stamp(end),
                            "$gte": GQ_util_date_stamp(start)
                        }
                },
                {"_id": 0},
                batch_size=10000
            )
            # res=[QA_util_dict_remove_key(data, '_id') for data in cursor]
        else:
            # code checking
            code = GQ_util_code_tolist(code)
            cursor = collections.find(
                {
                    "code": {
                        "$in": code,
                    },
                    "date_stamp":
                        {
                            "$lte": GQ_util_date_stamp(end),
                            "$gte": GQ_util_date_stamp(start)
                        }
                },
                {"_id": 0},
                batch_size=10000
            )            

        res = pd.DataFrame([item for item in cursor])
        try:
            res = res.assign(
                volume=res.vol,
                date=pd.to_datetime(res.date, utc=False)
            ).drop_duplicates(([
                'date',
                'code'])).set_index([
                    'date',
                    'code'],
                    drop=False)
            res = res.loc[:,
                          [
                              'code',
                              'open',
                              'high',
                              'low',
                              'close',
                              'volume',
                              'amount',
                              'date'
                          ]]
        except Exception:
            res = None

        return res

    return None


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


def prepare_symbol_range(eval_range, verbose=True):
    """
    返回预设的标的合集
    """
    if (eval_range == 'all'):
        codelist_candidate = QA_fetch_stock_list()
        if (len(codelist_candidate) > 0):
            codelist_candidate = codelist_candidate[AKA.CODE].tolist()
        else:
            return False
    elif (eval_range == 'etc') or \
        (eval_range == 'other'):
        codelist_candidate = QA_fetch_stock_list()
        if (len(codelist_candidate) > 0):
            codelist_candidate = codelist_candidate[AKA.CODE].tolist()
        else:
            return False
        blockname = focus_block

        blk = QA_fetch_stock_block_adv(collections=DATABASE.stock_block)
        blockname_exodus = list(set(blockname).difference(set(blk.block_name)))
        blockname = list(set(blk.block_name).intersection(set(blockname)))
        if (len(blockname_exodus) > 0):
            print(f'系统预设的A股板块: {blockname_exodus} 在通达信板块分类中已被移除')
        codelist_candidate_revese = blk.get_block(blockname).code
        codelist_candidate_revese = list(set(codelist_candidate_revese))

        # 求差集，不在指数股名单之中
        codelist_candidate = list(set(codelist_candidate).difference(set(codelist_candidate_revese)))
    elif (eval_range == 'etc_1') or \
        (eval_range == 'other_1'):
        codelist_candidate = QA_fetch_stock_list()
        if (len(codelist_candidate) > 0):
            codelist_candidate = codelist_candidate[AKA.CODE].tolist()
        else:
            return False
        blockname = focus_block

        blk = QA_fetch_stock_block_adv(collections=DATABASE.stock_block)
        blockname_exodus = list(set(blockname).difference(set(blk.block_name)))
        blockname = list(set(blk.block_name).intersection(set(blockname)))
        if (len(blockname_exodus) > 0):
            print(f'系统预设的A股板块: {blockname_exodus} 在通达信板块分类中已被移除')
        codelist_candidate_revese = blk.get_block(blockname).code
        codelist_candidate_revese = list(set(codelist_candidate_revese))

        # 求差集，不在指数股名单之中
        codelist_candidate = list(set(codelist_candidate).difference(set(codelist_candidate_revese)))
        if (len(codelist_candidate) > 1700):
            # 股票代码均匀分组法，此算法通过股票代码后2位之和取模分为2组，3组，4组，5组均可。理论最多可以将股票池均分成2~18组。
            # codelist_candidate = list(filter(None, [print(code, int(code[5])) for code in codelist_candidate]))
            codelist_candidate = list(filter(None, [code if ((int(code[4]) + int(code[5])) % 2) == 1 else None for code in codelist_candidate]))
    elif (eval_range == 'etc_2') or \
        (eval_range == 'other_2'):
        codelist_candidate = QA_fetch_stock_list()
        if (len(codelist_candidate) > 0):
            codelist_candidate = codelist_candidate[AKA.CODE].tolist()
        else:
            return False
        blockname = focus_block

        blk = QA_fetch_stock_block_adv(collections=DATABASE.stock_block)
        blockname_exodus = list(set(blockname).difference(set(blk.block_name)))
        blockname = list(set(blk.block_name).intersection(set(blockname)))
        if (len(blockname_exodus) > 0):
            print(f'系统预设的A股板块: {blockname_exodus} 在通达信板块分类中已被移除')
        codelist_candidate_revese = blk.get_block(blockname).code
        codelist_candidate_revese = [code[1:7] if (len(code) == 7) else code for code in codelist_candidate_revese]
        codelist_candidate_revese = list(set(codelist_candidate_revese))
        # 求差集，不在指数股名单之中
        codelist_candidate = list(set(codelist_candidate).difference(set(codelist_candidate_revese)))
        if (len(codelist_candidate) > 1700):
            # codelist_candidate = list(filter(None, [print(code, int(code[5])) for code in codelist_candidate]))
            codelist_candidate = list(filter(None, [code if ((int(code[4]) + int(code[5])) % 2) == 0 else None for code in codelist_candidate]))
    elif (isinstance(eval_range, str)) and (eval_range.lower() == 'etf'):
        codelist_candidate = GQ_get_etf_list()

        if (len(codelist_candidate) > 0):
            codelist_candidate = list(set([each_code['code'][0:6] for idx, each_code in codelist_candidate.iterrows()]))
        elif (isinstance(eval_range, str)) and (eval_range.lower() == 'etf'):
            codelist_candidate = list(set([
                '516860', '159851',
                '159919', '159997', '159805', '159987', 
                '159952', '159920', '518880', '159934', '159828',
                '159985', '515050', '159994', '159941', 
                '512800', '515000', '512170', '512980', 
                '510300', '513100', '510900', '512690', 
                '510050', '159916', '512910', '510310', 
                '512090', '513050', '513030', '513500', 
                '159905', '159949', '510330', '510500', 
                '510180', '159915', '510810', '159901', '512950',
                '512710', '510850', '512500', '512000',
                '513900', '513090', '159977',    # 清盘 '159803',
                '515650', '159928', '159902', '588080',
                '512010', '512720', '512400', '159807',
                '512580', '515700', '515880', '159995',
                '512200', '512880', '512660', '512800',
                '512760', '512980', '518880', '588000', 
                '159870', '516020', '515210',  # '511880', 银华日利
                '515220', '159880', '159871',  # '516680',
                '159945', '159876', '510410', '512400',
                '515210', '510170', '159930', '516780',
                '516150', '159839', '159980', '159835',
                '516060', '159849', '159858', '515120',
                '159857', '515790', '516880',  # '515870',
                '159806', '159825', '510210', '516010',
                '159619', '513010', '159792', '515070',
                '515520.XSHG', '512890.XSHG', '515180.XSHG',
                '515300.XSHG', '159916.XSHE', '512910.XSHG',
                '510310.XSHG', '512260.XSHG', '515810.XSHG',
                '159949.XSHE', '515000.XSHG', '512770.XSHG',
                '512040.XSHG', '510900.XSHG', '513050.XSHG',
                '159941.XSHE', '518880.XSHG', '159938.XSHE',
                '159944.XSHE', '159945.XSHE', '512580.XSHG',
                '512670.XSHG', '512800.XSHG', '512880.XSHG',
                '512980.XSHG', '512760.XSHG', '515880.XSHG',
                '512720.XSHG', '512690.XSHG', '515700.XSHG',
                '515650.XSHG', '512760.XSHG', '399001',
                '399006', '399997', '399987', '399306',
                '399396', '399384', '399684', '399616',
                '399240', '399976', '399005', '399380',
                '399248', '399811', '399810',
                '399989', '399935', '399808', '399932',
                '399804', '399928', '399998', '399934',
                '399967', '399986', '399933', '399233',
                '399699', '399959', '159611', '159523', '159525', '159526', '159527',
                '159528', '159529',
                '159530''159531', '159533', '159535', '159536', '159537', '159538', '159539', '159540',
                '159541', '159542', '159543', '159545', '159546', '159547', '159549', '159551',
                '159552', '159553', '159555', '159556', '159557', '159558',
                '159559', '159560', '159561', '159562', '159563', '159565', '159566', '159567', '159568', '159570', '159571', '159572', '159573', '159575', '159576', 
                '159577', '159581', '159582', '159586', '159588', '159589', '159591', '159592', '159593', '159595', '159596', '159597', '159599', '159601', '159602',
                '159603', '159605', '159606', '159607', '159608', '159609', '159610', '159611', '159612', '159613', '159615', '159616', '159617', '159618', '159619', 
                '159620', '159621', '159622', '159623', '159625', '159627', '159628', '159629', '159630', '159631', '159632', '159633', '159635', '159636', '159637',
                '159638', '159639', '159640', '159641', '159642', '159643', '159645', '159646', '159647', '159649', '159650', '159651', '159652', '159653', '159655', 
                '159656', '159657', '159658', '159659', '159660', '159661', '159662', '159663', '159665', '159666', '159667', '159669', '159670', '159671', '159672',
                '159638', '159639', '159640', '159641', '159642', '159643', '159645', '159646', '159647', '159649', '159650', '159651', '159652', '159653', '159655', 
                '159656', '159657', '159658', '159659', '159660', '159661', '159662', '159663', '159665', '159666', '159667', '159669', '159670', '159671', '159672',
                '159673', '159675', '159676', '159677', '159678', '159679', '159680', '159681', '159682', '159683', '159685', '159686', '159687', '159688', '159689', 
                '159690', '159691', '159692', '159693', '159695', '159696', '159697', '159698', '159699', '159701', '159703', '159706', '159707', '159708', '159709',
                '159711', '159712', '159713', '159715', '159716', '159717', '159718', '159719', '159720', '159721', '159723', '159725', '159726', '159728', '159729', 
                '159730', '159731', '159732', '159735', '159736', '159738', '159739', '159740', '159741', '159742', '159743', '159745', '159747', '159748', '159750',
                '159751', '159752', '159755', '159756', '159757', '159758', '159760', '159761', '159763', '159766', '159767', '159768', '159769', '159770', '159773', 
                '159775', '159776', '159777', '159778', '159779', '159780', '159781', '159782', '159783', '159786', '159787', '159788', '159790', '159791', '159792',
                '159793', '159795', '159796', '159797', '159798', '159799', '159801', '159804', '159805', '159806', '159807', '159808', '159810', '159811', '159812', 
                '159813', '159814', '159816', '159819', '159820', '159821', '159822', '159823', '159824', '159825', '159827', '159828', '159830', '159831', '159834',
                '159835', '159836', '159837', '159838', '159839', '159840', '159841', '159842', '159843', '159845', '159847', '159848', '159849', '159850', '159851', 
                '159852', '159855', '159856', '159857', '159858', '159859', '159861', '159862', '159863', '159864', '159865', '159866', '159867', '159869', '159870',
                '159871', '159872', '159873', '159875', '159876', '159877', '159880', '159881', '159883', '159885', '159886', '159887', '159888', '159889', '159890', 
                '159891', '159892', '159895', '159896', '159898', '159899', '159901', '159902', '159903', '159905', '159906', '159907', '159908', '159909', '159910',
                '159912', '159913', '159915', '159916', '159918', '159919', '159920', '159922', '159923', '159925', '159928', '159929', '159930', '159931', '159933', 
                '159934', '159935', '159936', '159937', '159938', '159939', '159940', '159941', '159943', '159944', '159945', '159948', '159949', '159952', '159954',
                '159956', '159957', '159958', '159959', '159960', '159961', '159964', '159965', '159966', '159967', '159968', '159969', '159970', '159971', '159972', 
                '159973', '159974', '159975', '159976', '159977', '159980', '159981', '159982', '159985', '159987', '159990', '159991', '159992', '159993', '159994',
                '159995', '159996', '159997', '159998', '160119', '510010', '510020', '510030', '510050', '510060', '510090', '510100', '510130', '510150', '510160', 
                '510170', '510180', '510190', '510200', '510210', '510230', '510270', '510290', '510300', '510310', '510330', '510350', '510360', '510370', '510380',
                '159001', '159003', '159005', '159150', '159300', '159306', '159307', '159310', '159321', '159350', '159501', '159502', '159503', '159505', '159506', 
                '159507', '159508', '159509', '159510', '159511', '159512', '159513', '159515', '159516', '159517', '159518', '159519', '159520', '159521', '159522']))
                
        codelist_candidate = list(set([each_code[0:6] for each_code in codelist_candidate]))
    elif (eval_range == 'csindex'):
        codelist_candidate = set([
            '000001.XSHG',
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
            '399699', '399959',])
        # codelist_candidate = [code[1:7] if(len(code)==7) else code for code in codelist_candidate]
        codelist_candidate = list(set(codelist_candidate))
    elif (eval_range == '399006'):
        blockname = set([
            '创业板', '创业板指', '创业300', '创业创新', '创业大盘', '创业板50',
            '创业板指', '创业蓝筹',])
        # blockname = list(set(blockname))
        all_stock_blocks = QA_fetch_stock_block_adv(collections=DATABASE.stock_block)
        codelist_candidate = all_stock_blocks.get_block(list(blockname.intersection(all_stock_blocks.block_name))).code

        codelist_candidate = [code[1:7] if (len(code) == 7) else (code.strip('33,') if (len(code) == 9) else code) for code in codelist_candidate]
        codelist_candidate = list(set(codelist_candidate))
    elif (eval_range == '399001'):
        blockname = set([
            '深证300', '深证可选', '深证消费', '深成消费', '深证100', '深证价值',
            '深证创新', '深证成指', '深证成长', '深证治理', '深证红利',])
        # blockname = list(set(blockname))
        all_stock_blocks = QA_fetch_stock_block_adv(collections=DATABASE.stock_block)
        codelist_candidate = all_stock_blocks.get_block(list(blockname.intersection(all_stock_blocks.block_name))).code
        codelist_candidate = [code[1:7] if (len(code) == 7) else code for code in codelist_candidate]
        codelist_candidate = list(set(codelist_candidate))
    elif (eval_range == '000001.XSHG'):
        blockname = set([
            '上证综指', '上证180', '上证380', '上证50', '上证100', '上证150', '上证中盘',
            '上证创新', '上证治理', '上证混改', '上证红利', '上证超大',])
        # blockname = list(set(blockname))
        all_stock_blocks = QA_fetch_stock_block_adv(collections=DATABASE.stock_block)
        codelist_candidate = all_stock_blocks.get_block(list(blockname.intersection(all_stock_blocks.block_name))).code
        codelist_candidate = [code[1:7] if (len(code) == 7) else code for code in codelist_candidate]
        codelist_candidate = list(set(codelist_candidate))
    elif (eval_range == '000688.XSHG'):
        blockname = set(['科创版指', '科创50', '科创信息',])
        # blockname = list(set(blockname))
        all_stock_blocks = QA_fetch_stock_block_adv(collections=DATABASE.stock_block)
        codelist_candidate = all_stock_blocks.get_block(list(blockname.intersection(all_stock_blocks.block_name))).code
        codelist_candidate = [code[1:7] if (len(code) == 7) else code for code in codelist_candidate]
        codelist_candidate = list(set(codelist_candidate))
    elif (eval_range == 'hs300'):
        blockname = set(['沪深300'])
        # blockname = list(set(blockname))
        all_stock_blocks = QA_fetch_stock_block_adv(collections=DATABASE.stock_block)
        codelist_candidate = all_stock_blocks.get_block(list(blockname.intersection(all_stock_blocks.block_name))).code
        codelist_candidate = [code[1:7] if (len(code) == 7) else code for code in codelist_candidate]
        codelist_candidate = list(set(codelist_candidate))
    elif (eval_range == 'test'):
        # '600104', '300015', '600612', '300750',
        #        '600585', '000651', '600436', '002475',
        #        '600030', '300760', '000895', '600066',
        #        '000661', '600887', '600352', '002352',
        #        '000568', '002714', '002415', '002594',
        #        '603713', '000858', '601138', '300122',
        #        '002179', '601888', '002557', '600036',
        #        '002271', '600298', '600276', '600547',
        #        '300146', '600660', '600161', '601318',
        #        '002050', '600900', '300498', '603515',
        #        '002007', '600600', '300059', '601933',
        #        '002258', '300715', '603899', '603444',
        #        '600031', '000876', '600332', '601877',
        #        '603288', '603520', '000333', '600563',
        #        '603259', '603517', '600309', '002230',
        #        '600009', '600519', '603486', '601100',
        #        '300144', '000538', '600486', '002705',
        #        '600570', '603129', '000963', '600738',
        #        '600529', '603733', '002466', '603056',
        #        '002129', '002041', '603816', '600009',
        #        '300677', '002304', '600893', '603185',
        codepool = ['300809', '600452', '000993', '000027',
                    '600172', '601567', '000625', '600744',
                    '300672', '603020', '600088', '600587',
                    '601999', '300273', '300755', '000997',
                    '002383', '002577', '002195', '300247',
                    '300722', '600596', '600331', '000923',
                    '601011', '600792', '002385', '600759',
                    '300157', '600007', '002581', '300309',
                    '300056', '300153', '300108', '300424',
                    '600318', '002670', '601628', '300141',
                    '300397', '600839', '603608', '603661',
                    '601066', '300190', '002622', '603169',
                    '002046', '000762', '000966', '600338',
                    '600956', '000683', '002658', '603123',
                    '601952', '600581', '600137', '300547',
                    '300309', '300490', '000066', '300123',
                    '300506', '002617', '300539', '603799',
                    '002497', '601579', '603169', '002046',
                    '000762', '000966', '003035', '000978',
                    '600338', '600956',
                    '000683', '002658',
                    '603123', '601952',
                    '600581', '000012', 
                    '000983', '002195', 
                    '300205', '002249',
                    '600126', '603518', 
                    '603900', '603933', 
                    '601066', '002316', '002436', '600327',
                    '600321', '300093', '300919', '603879',
                    '002426', '002104', 
                    '600198', '601699', '603348',
                    '600744', '300735', '002545',
                    '600381', '602227', '002386',
                    '693690', '300843', '000037',
                    '600355', '300582', '002472',
                    '603611', '688551', '600505',
                    '300453', '002082', '600722',
                    '600321', '605186', '002796', '002140', '688129', '603318', '002115']
        codelist_candidate = get_codelist(codepool)
        codelist_candidate = [code[1:7] if (len(code) == 7) else code for code in codelist_candidate]
        codelist_candidate = list(set(codelist_candidate))
    elif (eval_range == 'sz150'):
        blockname = set(['上证150', '上证50', '深证300'])
        # blockname = list(set(blockname))
        all_stock_blocks = QA_fetch_stock_block_adv(collections=DATABASE.stock_block)
        codelist_candidate = all_stock_blocks.get_block(list(blockname.intersection(all_stock_blocks.block_name))).code
        codelist_candidate = [code[1:7] if (len(code) == 7) else code for code in codelist_candidate]
        codelist_candidate = list(set(codelist_candidate))
    elif (eval_range == 'zz500'):
        blockname = set(['中证500'])
        # blockname = list(set(blockname))
        all_stock_blocks = QA_fetch_stock_block_adv(collections=DATABASE.stock_block)
        codelist_candidate = all_stock_blocks.get_blocklist((blockname.intersection(all_stock_blocks.block_name))).code
        codelist_candidate = [code[1:7] if (len(code) == 7) else code for code in codelist_candidate]
        codelist_candidate = list(set(codelist_candidate))
    elif (eval_range == 'zz100'):
        blockname = set(['中证100'])
        # blockname = list(set(blockname))
        all_stock_blocks = QA_fetch_stock_block_adv(collections=DATABASE.stock_block)
        codelist_candidate = all_stock_blocks.get_block(list(blockname.intersection(all_stock_blocks.block_name))).code
        codelist_candidate = [code[1:7] if (len(code) == 7) else code for code in codelist_candidate]
        codelist_candidate = list(set(codelist_candidate))
    elif (isinstance(eval_range, str) and (eval_range.startswith('BK'))) or \
        (eval_range == 'sector') or \
        (eval_range == 'industry'):
        codelist_candidate = get_stock_sector_members()
    elif (isinstance(eval_range, list)):
        eval_range = [code[1:7] if (len(code) == 7) else code for code in eval_range]
        codelist_candidate = list(set(eval_range))
    elif (eval_range == 'gcd'):
        codelist_candidate = [
            '600511', '600967', '000998', '300212', '002189', '000799', '600744', '000008', 
            '601991', '001979', '002415', '002916', '300003', '000778', '600107', '600161', 
            '600850', '600406', '601939', '002265', '601319', '002427', '003035', '000969', 
            '600038', '600879', '600101', '601669', '000822', '000590', '002268', '600131', 
            '000819', '000039', '000826', '601156', '600765', '600845', '300516', '600808', 
            '600528', '600581', '002232', '600536', '002368', '601006', '600688', '600271', 
            '000875', '600198', '600710', '601399', '601005', '000553', '600495', '600968', 
            '300678', '600876', '300523', '002059', '600886', '000738', '002013', '600737', 
            '600775', '000068', '600871', '600435', '002939', '000550', '600180', '000663', 
            '000504', '002037', '000151', '300797', '002819', '002140', '600644', '601919', 
            '002302', '600206', '600148', '000400', '300388', '601989', '600990', '000815', 
            '600498', '000032', '600497', '600178', '000595', '300070', '000026', '601872', 
            '000788', '600313', '300788', '600448', '601858', '002544', '600444', '002046', 
            '600905', '600021', '000800', '600391', '000930', '002423', '000519', '300425', 
            '000591', '300291', '600506', '601618', '600030', '600316', '601117', '601985', 
            '603128', '600501', '600582', '600372', '600855', '601179', '300077', '301215', 
            '600583', '601975', '600037', '601918', '600339', '600508', '000807', '000537', 
            '600176', '002320', '000902', '600476', '603698', '300962', '600312', '600977', 
            '002462', '600138', '600184', '601016', '000021', '601808', '603860', '000786', 
            '600862', '600795', '001289', '002017', '601881', '600050', '000708', '000566', 
            '002338', '000928', '600505', '002030', '601798', '600882', '600328', '600195', 
            '002128', '000050', '600811', '600118', '600610', '300597', '600179', '600088', 
            '600027', '600390', '300527', '002389', '002601', '000793', '002226', '600420', 
            '601888', '600058', '000617', '000737', '600433', '600707', '300832', '600560', 
            '300073', '600007', '000862', '600962', '600062', '600116', '000990', '600877', 
            '600787', '601607', '600830', '600552', '600900', '600379', '601333', '600961', 
            '603927', '601816', '600029', '300034', '002246', '600698', '000958', '300334', 
            '600764', '601857', '600378', '601818', '000882', '000166', '300455', '000720', 
            '000710', '600292', '600396', '600268', '002401', '002786', '600999', '601988', 
            '000159', '003009', '601718', '600517', '601226', '601166', '600100', '000851', '601111', '601999', '601800', '601068', '600055', '300469', '600345', 
            '002039', '600071', '000762', '000065', '000698', '003031', '600750', '601236', '000777', '000657', '600299', '601901', '000629', '000963', '601728', 
            '600562', '300374', '301137', '000554', '002419', '002935', '600019', '000966', '000925', '000825', '000927', '601965', '000069', '600081', '000059', 
            '600099', '001965', '601328', '601658', '600489', '600335', '600230', '600115', '003816', '603000', '000626', '600150', '000028', '000625', '600449', 
            '600685', '002297', '300161', '301058', '600896', '600011', '600760', '601288', '000831', '600094', '000066', '601390', '600640', '002167', '603126', 
            '600135', '002190', '000901', '000951', '600863', '600158', '603013', '601628', '600320', '002163', '600480', '601998', '600916', '600598', '000727', 
            '601088', '000031', '600500', '601600', '600129', '601766', '000751', '000411', '300557', '000898', '601106', '600831', '600026', '601198', '600776', 
            '002063', '601611', '600452', '600742', '002116', '300105', '002643', '001914', '002051', '300140', '300205', '600601', '600730', '000503', '300864', 
            '601949', '603888', '000520', '600072', '603026', '000920', '300087', '300747', '000526', '000717', '600006', '600282', '000768', '001872', '600980', '600025', '600458', '600061', '000547', '000731', '600875', '600028', '000016', '600486', '600482', '000410', '600127', '000677', '600343', '000761', '002145', '300197', '000830', '300847', '601038', '601868', '300188', '002066', '000881', '000099', '000798', '600056', '000852', '000635', '002106', '600428', '000877', '000767', '601995', '600151', '601866', '600361', '600523', '600579', '603060', '300446', '600212', '002057', '600893', '000883', '601186', '000988', '002049', '601880', '000968', '601968', '300396', '002025', '601788', '600125', '000922', '601608', '600392', '600171', '000333', '002305', '000758', '000953', '600262', '600657', '000736', '600846', '600622', '600283', '600236', '002777', '000999', '301090', '600048', '600973', '600970', '601668', '600647', '300294', '000733', '000423', '000666', '600636', '600550', '601898', '000938', '002405', '600720', '600726', '601336', '002281', '000985', '601598', '002080', '002205', '300225', '000908', '603019', '300114', '300268', '600469', '600797', '600705', '002179', '600995', '002258', '000915', '600963', '601698', '000878', '600624']
    elif (eval_range =='gcd2'):
        codelist_candidate = ['002441', '002011', '000599', '601111', '300839', '000025', '000572', '600368', '002691', '600501', '600345', '002659', '000070', '002905', '601298', '600825', '600965', '600756', '002556', '000778', '603040', '002699', '002194', '003022', '000762', '600583', '300021', '600836', '002908', '000016', '002213', '300772', '002343', '000816', '601918', '600339', '002576', '002543', '000913', '002676', '300619', '603665', '601669', '002320', '603855', '002638', '605086', '600343', '300442', '002480', '603843', '002137', '600269', '002120', '601789', '000702', '002290', '601038', '603919', '601568', '000546', '002990', '002066', '605399', '600113', '000521', '600397', '600866', '600085', '300200', '600121', '601808', '600012', '002589', '300022', '603860', '002935', '600862', '600251', '000852', '603127', '300209', '300401', '603029', '603883', '603225', '002420', '002702', '603990', '002181', '600784', '600817', '000559', '300654', '601798', '600081', '601006', '600151', '600156', '300662', '600099', '601928', '603233', '600495', '001965', '600968', '600190', '600662', '002096', '000883', '601811', '300550', '300010', '600386', '600179', '600035', '002630', '002554', '600016', '603535', '603988', '300270', '603918', '000705', '000062', '002923', '603786', '000582', '600125', '300848', '603289', '600279', '603979', '605577', '002523', '000034', '600559', '600997', '002660', '603126', '603565', '000020', '600135', '600560', '000703', '002406', '300484', '300990', '002209', '600323', '603033', '600033', '603013', '600815', '600320', '000430', '300547', '600618', '603012', '601101', '002570', '600348', '603100', '600830', '300048', '002629', '600552', '002090', '000610', '600735', '603789', '300586', '600990', '300012', '300057', '600408', '600066', '002860', '002862', '603316', '601699', '002201', '600418', '002571', '601799', '002696', '002782', '600178', '300761', '600053', '002514', '300903', '600098', '000301', '600679', '000978', '000911', '002973', '603223', '601008', '601857', '600803', '002519', '002971', '300651', '600297', '300518', '600444', '300299', '603505', '600757', '002888', '603822', '300799', '002350', '300105', '600626', '000800', '603536', '002933', '002418', '600740', '002786', '600469', '300121', '600730', '601326', '000930', '002179', '601988', '600660', '603590', '300425', '300409', '600995', '000791', '600963', '003009']
    else:
        eval_range = 'blocks'
        # blockname = ['中证500', '创业板50', '上证50', '上证150', '深证300']
        all_stock_blocks = QA_fetch_stock_block_adv(collections=DATABASE.stock_block) 
        blockname = focus_block

        blockname_exodus = list(set(blockname).difference(set(all_stock_blocks.block_name)))
        blockname = list(set(all_stock_blocks.block_name).intersection(set(blockname)))
        if (len(blockname_exodus) > 0):
            print(f'系统预设的A股板块: {blockname_exodus} 在通达信板块分类中已被移除')
        codelist_candidate = all_stock_blocks.get_block(blockname).code
        codelist_candidate = [code[1:7] if (len(code) == 7) else code for code in codelist_candidate]
        # codelist_candidate = [code for code in codelist_candidate if not
        # code.startswith('300')]
        codelist_candidate = list(set(codelist_candidate))
        if (verbose):
            print('批量评估板块成分股：{} Total:{}'.format(
                blockname,
                len(codelist_candidate)))

    # 求差集，股票不在黑名单之中
    codelist_candidate = list(set(codelist_candidate).difference(set(stock_cn_blacklist())))
    return codelist_candidate


def GQ_fetch_stock_min(
    code,
    start,
    end,
    format='numpy',
    frequence='1min',
    collections=DATABASE.stock_min
):
    '获取股票分钟线'
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
        print(
            "QA Error QA_fetch_stock_min parameter frequence=%s is none of 1min 1m 5min 5m 15min 15m 30min 30m 60min 60m"
            % frequence
        )

    # code checking
    code = GQ_util_code_tolist(code)

    cursor = collections.find(
        {
            'code': {
                '$in': code
            },
            "time_stamp":
                {
                    "$gte": GQ_util_time_stamp(start),
                    "$lte": GQ_util_time_stamp(end)
                },
            'type': frequence
        },
        {"_id": 0},
        batch_size=10000
    )

    res = pd.DataFrame([item for item in cursor])
    try:
        if (frequence in ['15min', '15m']) or \
            (frequence in ['5min', '5m']) or \
            (frequence in ['1min', '1m']) or \
            (frequence in ['30min', '30m']) or \
            (frequence in ['60min', '60m']):
            res = res.assign(
                volume=res.vol,
                datetime=pd.to_datetime(res.datetime, utc=False)
            ).drop_duplicates([
                'datetime',
                'code']).set_index(
                    'datetime',
                    drop=False
                )
        else:
            res = res.assign(
                volume=res.vol,
                datetime=pd.to_datetime(res.datetime, utc=False)
            ).query('volume>1').drop_duplicates(['datetime',
                                                'code']).set_index(
                                                    'datetime',
                                                    drop=False
                                                )
        # return res
    except Exception:
        res = None
    if format in ['P', 'p', 'pandas', 'pd']:
        return res
    elif format in ['json', 'dict']:
        return GQ_util_to_json_from_pandas(res)
    # 多种数据格式
    elif format in ['n', 'N', 'numpy']:
        return np.asarray(res)
    elif format in ['list', 'l', 'L']:
        return np.asarray(res).tolist()
    else:
        print(
            "QA Error QA_fetch_stock_min format parameter %s is none of  \"P, p, pandas, pd , json, dict , n, N, numpy, list, l, L, !\" "
            % format
        )
        return None
    

def GQ_fetch_stock_min_adv(
        code,
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
            print(f"GolemQ Error GQ_fetch_stock_min_adv parameter frequence={frequence:%s} is none of 1min 1m 5min 5m 15min 15m 30min 30m 60min 60m")
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
            print(f"GolemQ Error GQ_fetch_stock_min_adv parameter code={code:%s}, start={start:%s}, end={end:%s} is equal, should have time span! ")
        return None

    # 🛠 todo 报告错误 如果开始时间 在 结束时间之后
    res = GQ_fetch_stock_min(code, start, end, format='pd', frequence=frequence)
    
    if res is None:
        if (verbose):
            print(f"QA Error GQ_fetch_stock_min_adv parameter code={code:%s}, start={start:%s}, end={end:%s} frequence={frequence:%s} call QA_fetch_stock_min return None")
        return None
    else:
        res_set_index = res.set_index(['datetime', 'code'], drop=if_drop_index)
        # if res_set_index is None:
        #     print("QA Error QA_fetch_stock_min_adv set index 'datetime, code'
        #     return None")
        #     return None
        return QA_DataStruct_Stock_min(res_set_index)
    

@func_set_timeout(12)
def get_kline_price_min(
    codelist,
    start=None,
    market_type=None,
    frequency='60min',
    verbose=True,
    end=None,
    realtime=True,
):
    """
    写这个函数的目的就是不用去考虑乱七八糟币种和市场种类，直接怼一个或者几个代码就能读取到合适的数据
    """
    if (market_type is None):
        _, market_type, market_words, market_type_desc = is_stock_cn(codelist)
        if (isinstance(codelist, str)):
            # 判断是单一标的
            if (market_type == MARKET_TYPE.STOCK_CN):
                market_type_desc = 'A股'
                market_type = MARKET_TYPE.STOCK_CN
            elif (market_type == MARKET_TYPE.INDEX_CN):
                if (market_type_desc.endswith('ETF基金')):
                    market_type_desc = 'A股ETF基金'
                    market_type = MARKET_TYPE.INDEX_CN
                else:
                    market_type_desc = 'A股指数'
                    market_type = MARKET_TYPE.INDEX_CN
            elif (is_cryptocurrency(codelist)[1] == MARKET_TYPE.CRYPTOCURRENCY):
                market_type_desc = '数字货币'
                market_type = MARKET_TYPE.CRYPTOCURRENCY
            elif (market_type == MARKET_TYPE.FUND_CN):
                market_type_desc = 'A股ETF基金'
                market_type = MARKET_TYPE.INDEX_CN
            else:
                if verbose:
                    print(is_stock_cn(codelist))
        else:
            # 判断是多标的
            if (market_type == MARKET_TYPE.STOCK_CN):
                market_type_desc = 'A股'
                market_type = MARKET_TYPE.STOCK_CN
            elif (market_type == MARKET_TYPE.INDEX_CN):
                if (market_type_desc.endswith('ETF基金')):
                    market_type_desc = 'A股ETF基金'
                    market_type = MARKET_TYPE.INDEX_CN
                else:
                    market_type_desc = 'A股指数'
                    market_type = MARKET_TYPE.INDEX_CN
            elif (is_cryptocurrency(codelist[0])[1] == MARKET_TYPE.CRYPTOCURRENCY):
                market_type_desc = '数字货币'
                market_type = MARKET_TYPE.CRYPTOCURRENCY
            elif (market_type == MARKET_TYPE.FUND_CN):
                market_type_desc = 'A股ETF基金'
                market_type = MARKET_TYPE.INDEX_CN
            else:
                if verbose:
                    print(is_stock_cn(codelist))
            # raise Exception(u'多标的我还没时间实现')
    else:
        if (market_type == MARKET_TYPE.STOCK_CN):
            market_type_desc = 'A股'
        elif (market_type == MARKET_TYPE.CRYPTOCURRENCY):
            market_type_desc = '数字货币'
        elif (market_type == MARKET_TYPE.FUND_CN):
            market_type_desc = 'A股ETF基金'
        elif (market_type == MARKET_TYPE.INDEX_CN):
            market_type_desc = 'A股指数'

    if verbose:
        print(
            u'{} 开始读取{}分钟K线历史数据'.format(
                QA_util_timestamp_to_str()[2:16],
                market_type_desc),
            codelist if isinstance(codelist, str) else codelist[0:10])

    if (market_type == MARKET_TYPE.STOCK_CN):
        if (frequency == '60min'):
            start = '{}'.format(dt.now() - timedelta(hours=19200)) if (start is None) else start
        else:
            start = '{}'.format(dt.now() - timedelta(hours=19200*4)) if (start is None) else start

        start_time = dt.strptime(str(dt.now().date()) + ' 09:15',
                                 '%Y-%m-%d %H:%M')
        if (dt.now() > start_time):
            end = '{}'.format(dt.now(timezone(timedelta(hours=8))) + timedelta(minutes=1)) if (end is None) else end
        else:
            end = '{}'.format(
                dt.strptime(
                    str(dt.now(timezone(timedelta(hours=8))).date() - timedelta(hours=24)) + ' 16:30', 
                    '%Y-%m-%d %H:%M')) if (end is None) else end

        data_min = GQ_fetch_stock_min_adv(
            codelist,
            start=start,
            end=end,
            frequence=frequency)

        if (data_min is not None):
            if (verbose):
                print(
                    f'Code: {codelist} 行 {len(data_min.data)}........... Ckpo 0\n',
                    data_min.data.query("volume < 1").tail(20),
                    data_min.data[[AKA.CLOSE]].head(10),
                    data_min.data[[AKA.CLOSE]].tail(15))
            
            # 计算每一行 close, open, high, low 为 NaN 的总和
            nan_sum = data_min.data[['close', 'open', 'high', 'low']].isna().sum(axis=1)

            # 筛选出 close 为 NaN 或 close < 0.168 的行，并结合 nan_sum > 2 的条件
            zero_trading = data_min.data[(data_min.data['close'].isna() | (data_min.data['volume'] < 0.168)) & (nan_sum > 2)]

            # 如果 zero_trading 长度为零
            if (len(zero_trading) == 0):
                # 筛选出 volume < 1 的行
                zero_trading = data_min.data.query("volume < 1").copy()
                
                # 留下同一日 date 超过1条记录的子集
                zero_trading['date'] = zero_trading.index.get_level_values(level=0).date
                date_counts = zero_trading['date'].value_counts()
                valid_dates = date_counts[date_counts > 1].index
                zero_trading = zero_trading[zero_trading['date'].isin(valid_dates)]
            
            # 判断是否长度为4的整倍数
            if len(zero_trading) > 0:
                if len(zero_trading) % 4 == 0:
                    if (verbose):
                        print(f"Code {codelist} Freq:{frequency} zero_trading 的长度 {len(zero_trading)} 是4的整倍数")
                    
                    # 在 res 中 drop zero_trading
                    data_min.data = data_min.data.drop(zero_trading.index)
                elif any(time.time() == pd.Timestamp("13:00:00").time() for time in zero_trading.index.get_level_values(level=0)):
                    if (verbose):
                        print(f"Code {codelist} Freq:{frequency} zero_trading 的索引包含时间 13:00:00")
                    
                    # 获取包含 "13:00:00" 的日期
                    dates_with_13 = zero_trading.index[zero_trading.index.get_level_values(level=0).time == pd.Timestamp("13:00:00").time()].get_level_values(level=0).date
                    
                    # 删除这些日期的所有行
                    # 提取 level=0 的日期部分，并转换为 datetime.date
                    level_0_dates = [zero_trading_date.date() in dates_with_13 for zero_trading_date in zero_trading.index.get_level_values(level=0)]
                    
                    # 使用 drop 方法删除匹配的行
                    data_min.data = data_min.data.drop(zero_trading[level_0_dates].index)
                else:
                    if (verbose):
                        print(f"Code {codelist} Freq:{frequency} zero_trading 的长度 {len(zero_trading)} 不是4的整倍数, 有感知, 但是无法确认.")

            if (verbose):
                print(
                    f'Code: {codelist} 行 {len(data_min.data)}........... Ckpo 1\n', 
                    data_min.data.query("volume < 1").tail(20),
                    data_min.data[[AKA.CLOSE]].head(10),
                    data_min.data[[AKA.CLOSE]].tail(15))

            # 当QA未全部更新完当天"前复权"xdxr价格数据的时候会导致最后一天K线空白
            # 一般的说最后一天价格数据与前复权价格相同，不需要乘复权系数，直接覆盖nan值即可。

            # # 当Volume<1时候，QA复权函数会产生“NAN”的OHLC价格，
            # # 这里从外部复权函数外部修正这个问题 Part1：记录未复权价格
            # fqadj_chkpoi_idx = pd.Index([])
            # if (len(data_min.data.query("volume<1")) > 0):
            #     fqadj_chkpoi_idx = data_min.data[
            #         (data_min.data['volume'] > 1) & \
            #         (data_min.data['volume'].shift(-1) < 1)].index
            #     fqadj_chkpoi_snapshot=data_min.data[((data_min.data['volume']>1) & \
            #                                         (data_min.data['volume'].shift(-1)<1)) | \
            #                                         (data_min.data['volume']<1)].copy()

            # data_min_rollback = data_min.data.tail(4).copy()
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=FutureWarning)
                data_min = data_min.to_qfq()

            if (verbose):
                print(
                    f'Code: {codelist} 行 {len(data_min.data)}........... Ckpo 2\n', 
                    data_min.data.query("volume < 1").tail(20),
                    data_min.data[[AKA.CLOSE]].head(10),
                    data_min.data[[AKA.CLOSE]].tail(15))
                
        if (verbose):
            if (data_min is not None):
                print(
                    f'Code: {codelist} 行 {len(data_min.data)}........... Ckpo 3\n', 
                    data_min.data.query("volume < 1").tail(20),
                    data_min.data[[AKA.CLOSE]].head(10),
                    data_min.data[[AKA.CLOSE]].tail(15))
            else:
                print(f'Code: {codelist} has non-data........... Ckpo 3\n')

        if ((GQ_util_get_last_day()-pd.to_datetime(end).tz_localize(None)) > timedelta(days=1)):
            if (verbose):
                print(u'Code: {} 跳过实时行情读取 {}...........\n'.format(
                    normalize_code(
                        codelist,
                        market_type=market_type),
                    GQ_util_get_last_day()))
            pass
        else:
            if (realtime):
                if (verbose):
                    if (data_min is not None):
                        print(
                            f'Code: {codelist} 行 {len(data_min.data)}........... Ckpo 3.1\n', 
                            data_min.data.query("volume < 1").tail(20),
                            data_min.data[[AKA.CLOSE]].head(10),
                            data_min.data[[AKA.CLOSE]].tail(15))
                    else:
                        print(f'Code: {codelist} has non-data........... Ckpo 3.1\n')

                data_min = GQ_fetch_stock_min_realtime_adv(
                    normalize_code(
                        codelist, 
                        market_type=market_type),
                    data_min,
                    frequency=frequency, 
                    verbose=verbose)
                if (verbose):
                    print(u'Code: {} STOCK_CN 实时行情读取 {}...........\n'.format(
                        normalize_code(
                            codelist, 
                            market_type=market_type),
                        GQ_util_get_last_day(), ),
                        data_min.index if (data_min is not None) else "无数据 None")
                if (verbose):
                    if (data_min is not None):
                        print(
                            f'Code: {codelist} 行 {len(data_min.data)}........... Ckpo 3.2\n', 
                            data_min.data.query("volume < 1").tail(20),
                            data_min.data[[AKA.CLOSE]].head(10),
                            data_min.data[[AKA.CLOSE]].tail(15))
                    else:
                        print(f'Code: {codelist} has non-data........... Ckpo 3.2\n')

        if (data_min is None):
            if verbose:
                print(market_type, codelist)
            pass
        else:
            if (len(codelist) == 1):
                data_min.data[AKA.FULL_SYMBOL] = normalize_code(codelist[0], market_type=market_type)
                data_min.data[AKA.MARKET_TYPE] = market_type
        if (verbose):
            if (data_min is not None):
                print(f'Code: {codelist} 行 {len(data_min.data)}........... Ckpo 4\n', data_min.data.query("volume < 1").tail(20))
            else:
                print(f'Code: {codelist} has non-data........... Ckpo 4\n')
    elif (market_type == MARKET_TYPE.CRYPTOCURRENCY):
        start = '{}'.format(dt.now() - timedelta(hours=5400)) if (start is None) else start
        end = '{}'.format(dt.now(timezone(timedelta(hours=8))) + timedelta(minutes=1)) if (end is None) else end
        data_min = QA.QA_fetch_cryptocurrency_min_adv(
            code=codelist,
            start=start,
            end=end,
            frequence=frequency)
        # if verbose:
        #     data_min.data[ST.VERBOSE] = True
    elif (market_type == MARKET_TYPE.INDEX_CN) or \
        (market_type == MARKET_TYPE.FUND_CN):
        start = '{}'.format(datetime.datetime.now() - timedelta(hours=19200)) if (start is None) else start
        end = '{}'.format(datetime.datetime.now(timezone(timedelta(hours=8))) + timedelta(minutes=1)) if (end is None) else end
        if (isinstance(codelist, str)):
            data_min = QA.QA_fetch_index_min_adv(
                codelist[:6],
                start=start,
                end=end,
                frequence=frequency)
        else:
            data_min = QA.QA_fetch_index_min_adv(
                [code[:6] for code in codelist],
                start=start,
                end=end,
                frequence=frequency)
        # print(u'开始读取', codelist)
        if (data_min is not None):
            # 计算每一行 close, open, high, low 为 NaN 的总和
            nan_sum = data_min.data[['close', 'open', 'high', 'low']].isna().sum(axis=1)

            # 筛选出 close 为 NaN 或 close < 0.168 的行，并结合 nan_sum > 2 的条件
            zero_trading = data_min.data[(data_min.data['close'].isna() | (data_min.data['volume'] < 0.168)) & (nan_sum > 2)]

            # 如果 zero_trading 长度为零
            if (len(zero_trading) == 0):
                # 筛选出 volume < 1 的行
                zero_trading = data_min.data.query("volume < 1").copy()

                # 留下同一日 date 超过1条记录的子集
                zero_trading['date'] = zero_trading.index.get_level_values(level=0).date
                date_counts = zero_trading['date'].value_counts()
                valid_dates = date_counts[date_counts > 1].index
                zero_trading = zero_trading[zero_trading['date'].isin(valid_dates)]

            # 判断是否长度为4的整倍数
            if len(zero_trading) > 0:
                if len(zero_trading) % 4 == 0:
                    if (verbose):
                        print(f"Code {codelist} Freq:{frequency} zero_trading 的长度 {len(zero_trading)} 是4的整倍数")

                    # 在 res 中 drop zero_trading
                    data_min.data = data_min.data.drop(zero_trading.index)
                elif any(time.time() == pd.Timestamp("13:00:00").time() for time in zero_trading.index.get_level_values(level=0)):
                    if (verbose):
                        print(f"Code {codelist} Freq:{frequency} zero_trading 的索引包含时间 13:00:00")

                    # 获取包含 "13:00:00" 的日期
                    dates_with_13 = zero_trading.index[zero_trading.index.get_level_values(level=0).time == pd.Timestamp("13:00:00").time()].get_level_values(level=0).date

                    # 删除这些日期的所有行
                    # 提取 level=0 的日期部分，并转换为 datetime.date
                    level_0_dates = [zero_trading_date.date() in dates_with_13 for zero_trading_date in zero_trading.index.get_level_values(level=0)]

                    # 使用 drop 方法删除匹配的行
                    data_min.data = data_min.data.drop(zero_trading[level_0_dates].index)
                else:
                    if (verbose):
                        print(f"Code {codelist} Freq:{frequency} zero_trading 的长度 {len(zero_trading)} 不是4的整倍数, 有感知, 但是无法确认.")

            # 基金存在复权问题，未解决。
            # data_min = data_min.to_qfq()
            pass

        if ((GQ_util_get_last_day()-pd.to_datetime(end).tz_localize(None))>timedelta(days=1)):
            pass
        else:
            if (realtime):
                data_min = GQ_fetch_stock_min_realtime_adv(
                    normalize_code(
                        codelist, 
                        market_type=market_type), 
                    data_min, 
                    frequency=frequency, 
                    verbose=verbose)
                if (verbose):
                    print(u'Code: {} INDEX_CN 实时行情读取 {}...........\n'.format(
                        normalize_code(
                            codelist, 
                            market_type=market_type), 
                        GQ_util_get_last_day(), ), 
                        data_min.index)
        if (data_min is None):
            if verbose:
                print(market_type, codelist)
            pass
        else:
            if (len(codelist) == 1):
                data_min.data[AKA.FULL_SYMBOL] = normalize_code(codelist[0], market_type=market_type)
                data_min.data[AKA.MARKET_TYPE] = market_type

            # if verbose:
            #     data_min.data[ST.VERBOSE] = True
    else:
        if verbose:
            print(u'Not Supported code:', codelist)
        return None, None

    if (verbose) and (data_min is not None):
        print('last time:{:%Y-%m-%d %H:%M} total bars:{:d}'.format(data_min.data.index.get_level_values(level=0)[-1], len(data_min.data)))
    
    # 移除 open, high, low, close 列全为 NaN 的行
    if (frequency in ['15min', '5min', '30min', '1min']) and \
       (len(data_min.data.query("volume<1")) == data_min.data[['open', 'high', 'low', 'close']].isna().all(axis=1).sum()):
        if (verbose):
            # 保存操作前的全部索引
            original_indices = data_min.data.index.tolist()

            # 执行dropna操作
            data_min.data = data_min.data.dropna(subset=['open', 'high', 'low', 'close'], how='all')

            # 获取操作后剩余的索引
            remaining_indices = data_min.data.index.tolist()

            # 计算被删除的索引（差集）
            dropped_indices = list(set(original_indices) - set(remaining_indices))

            print(f"被删除的行索引: {dropped_indices}")

        data_min.data = data_min.data.dropna(subset=['open', 'high', 'low', 'close'], how='all')

    if verbose:
        print(f'Code: {codelist} 行 {len(data_min.data)}........... Ckpo 8\n', data_min.data.query("volume < 1").tail(10))

    # 使用修正函数
    if (data_min is not None):
        data_min.data = fix_incorrect_1300_time(data_min.data)

    if verbose:
        print(u'{} get_kline_price_min() 读取{}分钟K线历史数据完毕'.format(QA_util_timestamp_to_str()[2:16],
                                        market_type_desc), 
                codelist[-10:])
        if (data_min is not None) and (len(data_min.data.query("volume<1")) > 0):
            print(
                u'\nzero_trading Checkpoint: \n',
                data_min.data.query("volume<1").tail(20))
            
        if (data_min is not None):
            print(f'Code: {codelist} 行 {len(data_min.data)}........... Ckpo 9\n', data_min.data.query("volume < 1").tail(20))
        else:
            print(f'Code: {codelist} has non-data........... Ckpo 6\n')        
    try:
        if (isinstance(data_min, QA_DataStruct_Stock_min) or \
            isinstance(data_min, QA_DataStruct_Stock_day)):
            codename = GQ_fetch_stock_name(codelist)
        elif (isinstance(data_min, QA_DataStruct_Index_min) or \
            isinstance(data_min, QA_DataStruct_Index_day)):
            if (market_type_desc == 'A股ETF基金'):
                if (isinstance(codelist, str)):
                    codename = GQ_fetch_etf_name(codelist[:6])
                else:
                    codename = GQ_fetch_etf_name([code[:6] for code in codelist])
            else:
                if (isinstance(codelist, str)):
                    codename = QA.QA_fetch_index_name(codelist[:6])
                else:
                    codename = QA.QA_fetch_index_name([code[:6] for code in codelist])
        elif isinstance(codelist, list):
            if (len(codelist) == 1):
                codename = codelist[0]
            else:
                codename = '{}'.format(codelist)
        else:
            codename = codelist if isinstance(codelist, str) else codelist.item()
    except Exception:
        if (data_min is not None):
            # 破通达信的更新股票列表数据又双挂了
            if (len(codelist) == 1):
                codename = codelist[0]
            else:
                codename = '{}'.format(codelist)
        else:
            if verbose:
                traceback.print_exc()
                print(u'Unsupported code:{}'.format(codelist))
            return None, None

    # print(data_min.data.tail(10))
    if isinstance(codename, str):
        # 需要更新股票列表数据
        pass
    else:
        if (isinstance(codename, pd.DataFrame)) and \
            (len(codename)>1):
            codename=codename.tail(1).copy()        
        elif isinstance(codename, list):
            codename={AKA.CODE:codelist,
                    AKA.NAME:codelist,
                    'market_type_desc':market_type_desc,
                    'market_type':market_type,}
        try:
            codename['market_type_desc'] = market_type_desc
        except Exception:
            codename['market_type_desc'] = codelist
        codename['market_type'] = market_type
    return data_min, codename


@func_set_timeout(8)
def get_kline_price_v3(
    codelist,
    start=None,
    market_type=None,
    verbose=True,
    end=None,
    realtime=None
):
    """
    写这个函数的目的就是不用去考虑乱七八糟币种和市场种类，直接怼一个或者几个代码就能读取到合适的数据
    v2 增加主力资金流向数据
    """
    if (verbose):
        print('realtime:', realtime)
        
    if (market_type is None):
        if (isinstance(codelist, str)):
            # 判断是单一标的
            _, market_type, market_words, market_type_desc = is_stock_cn(codelist)
            if (market_type == MARKET_TYPE.STOCK_CN):
                market_type_desc = 'A股'
                market_type = MARKET_TYPE.STOCK_CN
            elif (market_type_desc.endswith('ETF基金')):
                market_type_desc = 'A股ETF基金'
                market_type = MARKET_TYPE.INDEX_CN
            elif (is_cryptocurrency(codelist)[1] == MARKET_TYPE.CRYPTOCURRENCY):
                market_type_desc = '数字货币'
                market_type = MARKET_TYPE.CRYPTOCURRENCY
            elif (market_type == MARKET_TYPE.INDEX_CN):
                market_type_desc = 'A股指数'
                market_type = MARKET_TYPE.INDEX_CN
            elif (market_type == MARKET_TYPE.FUND_CN):
                market_type_desc = 'A股ETF基金'
                market_type = MARKET_TYPE.INDEX_CN
            else:
                if verbose:
                    print('is_stock_cn():', is_stock_cn(codelist))
        else:
            # 判断是多标的
            _, market_type, market_words, market_type_desc = is_stock_cn(codelist[0])
            if (market_type == MARKET_TYPE.STOCK_CN):
                market_type_desc = 'A股'
                market_type = MARKET_TYPE.STOCK_CN
            elif (market_type_desc.endswith('ETF基金')):
                market_type_desc = 'A股ETF基金'
                market_type = MARKET_TYPE.INDEX_CN
            elif (is_cryptocurrency(codelist[0])[1] == MARKET_TYPE.CRYPTOCURRENCY):
                market_type_desc = '数字货币'
                market_type = MARKET_TYPE.CRYPTOCURRENCY
            elif (market_type == MARKET_TYPE.INDEX_CN):
                market_type_desc = 'A股指数'
                market_type = MARKET_TYPE.INDEX_CN
            elif (market_type == MARKET_TYPE.FUND_CN):
                market_type_desc = 'A股ETF基金'
                market_type = MARKET_TYPE.INDEX_CN
            else:
                if verbose:
                    print('is_stock_cn():', is_stock_cn(codelist))
            # raise Exception(u'多标的我还没时间实现')
    else:
        if (market_type == MARKET_TYPE.STOCK_CN):
            market_type_desc = 'A股'
        elif (market_type == MARKET_TYPE.CRYPTOCURRENCY):
            market_type_desc = '数字货币'
        elif (market_type == MARKET_TYPE.FUND_CN):
            market_type_desc = 'A股ETF基金'
        elif (market_type == MARKET_TYPE.INDEX_CN):
            market_type_desc = 'A股指数'

    if verbose:
        print(u'{} 开始读取{}日K线历史数据'.format(
            QA_util_timestamp_to_str()[2:16], 
            market_type_desc),
            codelist if isinstance(codelist, str) else codelist[0:10])
    if (market_type == MARKET_TYPE.STOCK_CN):
        start = '{}'.format(dt.today() - timedelta(days=2560)) if (start is None) else start
        end = '{}'.format(dt.today() + timedelta(days=1)) if (end is None) else end
        # offest = (dt.today() + timedelta(days=1)) - pd.to_datetime(end)
        if (isinstance(codelist, str)):
            data_day = QA_fetch_stock_day_adv(
                codelist[:6],
                start=start,
                end=end,)
            short_code = codelist[:6]
        else:
            data_day = QA_fetch_stock_day_adv(
                [code[:6] for code in codelist],
                start=start,
                end=end,)
            short_code = codelist[0][:6] if (len(codelist) == 1) else [code[:6] for code in codelist]

        if (data_day is not None):
            # 计算每一行 close, open, high, low 为 NaN 的总和
            nan_sum = data_day.data[['close', 'open', 'high', 'low']].isna().sum(axis=1)

            # 筛选出 close 为 NaN 或 close < 0.168 的行，并结合 nan_sum > 2 的条件
            zero_trading = data_day.data[(data_day.data['close'].isna() | (data_day.data['volume'] < 0.168)) & (nan_sum > 2)]

            # 在 res 中 drop zero_trading
            data_day.data = data_day.data.drop(zero_trading.index)

            zero_trading_raw = data_day.data[(data_day.data['volume'] < 0.168)]
            if (len(zero_trading_raw) > 0):
                if (len(zero_trading_raw)/len(data_day.data) < 0.0168):
                    if (verbose):
                        print(f'Code: {codelist} has zero_trading kline... \n{zero_trading_raw[["close", "open", "high", "low", "volume"]]} ')
                    data_day.data = data_day.data.drop(zero_trading_raw.index)
                else:
                    print(f'Code: {codelist} has too much zero_trading kline... \n{zero_trading_raw[["close", "open", "high", "low", "volume"]]} ')

            data_day = data_day.to_qfq()
            if (len(codelist) == 1):
                data_day.data[AKA.FULL_SYMBOL] = normalize_code(codelist[0], market_type=market_type)
                data_day.data[AKA.MARKET_TYPE] = market_type
        else:
            # 退市股
            if (isinstance(codelist, list)):
                try:
                    log_msg = u'退市股 {}'.format(codelist[0])
                    symbol_checkpoint_log(
                        logs=log_msg,
                        symbol=codelist[0],
                        length=None,
                        FrozenExpired=pd.to_datetime(GQ_util_get_last_day())+timedelta(days=63),
                        catalog=AKA.SHORT,)
                except Exception:
                    print(u'\n{}'.format(log_msg))
                    traceback.print_exc()
                codename = GQ_fetch_stock_name(codelist[0])
            else:
                try:
                    log_msg = u'退市股 {}'.format(codelist)
                    symbol_checkpoint_log(
                        logs=log_msg,
                        symbol=codelist,
                        length=None,
                        FrozenExpired=pd.to_datetime(GQ_util_get_last_day())+timedelta(days=63),
                        catalog=AKA.SHORT,)
                except Exception:
                    print(u'\n{}'.format(log_msg))
                    traceback.print_exc()
                codename = GQ_fetch_stock_name(codelist)
            return data_day, codelist

        try:
            if (data_day.data[[AKA.OPEN, AKA.CLOSE]].tail(10).isnull().values.any() == True):
                # 在下载数据的时候XRDR数据不全，有时候除权后最尾部莫名其妙丢数据了，只能拿没除权的数据补
                predict_null = pd.isnull(data_day.data[AKA.CLOSE])
                data_null = data_day.data[predict_null is True]
                data_day.data.loc[data_null.index, :] = QA_fetch_stock_day_adv(
                    codelist,
                    '{}'.format(data_null.index.get_level_values(level=0).values[0]),
                    '{}'.format(data_null.index.get_level_values(level=0).values[-1]),).data
        except Exception as e:
            print(u'get_kline_price_v3 Code:{}'.format(codelist), e)
            print(type(data_day), len(data_day.data))
            pass

        if ((GQ_util_get_last_day()-pd.to_datetime(end)) > timedelta(days=1)):
            pass
        else:
            if (realtime):
                data_day = GQ_fetch_stock_day_realtime_adv(
                    codelist,
                    data_day,
                    market_type=market_type,
                    verbose=verbose)
            else:
                # print('offest', offest, timedelta(days=1), (offest<timedelta(days=2)))
                pass

        try:
            # 临时读取资金流向
            if (realtime):
                stock_individual_fund_flow_df = GQ_fetch_stock_moneyflow(
                    short_code,
                    start=start,
                    end='{}'.format(pd.to_datetime(end).tz_localize('Asia/Shanghai') + \
                                    timedelta(hours=16)),
                    format='pd')
            else:
                stock_individual_fund_flow_df = GQ_fetch_stock_moneyflow(
                    short_code,
                    start=start,
                    end='{}'.format(pd.to_datetime(end).tz_localize('Asia/Shanghai') + \
                                    timedelta(hours=16)),
                    verbose=verbose,
                    format='pd')
            if (stock_individual_fund_flow_df is not None) and \
                (u"主力净流入-净额" in stock_individual_fund_flow_df.columns):
                data_day.data[FLD.MONEYFLOW_VOLUME] = stock_individual_fund_flow_df[u"主力净流入-净额"].astype('float') + \
                np.where(
                    (stock_individual_fund_flow_df[u"超大单净流入-净额"].astype('float') > 1e6),
                    stock_individual_fund_flow_df[u"超大单净流入-净额"].astype('float'), 0)
                data_day.data[FLD.MONEYFLOW_VOLUME_MINOR] = stock_individual_fund_flow_df[u"小单净流入-净额"].astype('float')
                data_day.data[FLD.MONEYFLOW_PERCENT] = stock_individual_fund_flow_df[u"主力净流入-净占比"].astype('float') + \
                np.where(
                    (stock_individual_fund_flow_df[u"超大单净流入-净额"].astype('float') > 1e6),
                    stock_individual_fund_flow_df[u"超大单净流入-净占比"].astype('float'), 0)
                data_day.data[FLD.MONEYFLOW_PERCENT_MINOR] = stock_individual_fund_flow_df[u"小单净流入-净占比"].astype('float')
        except Exception as e:
            print(u'get_kline_price_v3 -> GQ_fetch_stock_moneyflow Code:{}'.format(codelist), e)
            err = traceback.format_exc()
            print(err)
            pass

    elif (market_type == MARKET_TYPE.CRYPTOCURRENCY):
        start = '{}'.format(dt.now() - timedelta(hours=3600)) if (start is None) else start
        data_day = QA.QA_fetch_cryptocurrency_min_adv(
            code=codelist,
            start=start,
            end='{}'.format(dt.now(timezone(timedelta(hours=8))) + timedelta(minutes=1)),
            frequence='60min')
    elif (market_type == MARKET_TYPE.INDEX_CN) or \
        (market_type == MARKET_TYPE.FUND_CN):
        start = '{}'.format(dt.today() - timedelta(days=2500)) if (start is None) else start
        end = '{}'.format(dt.today() + timedelta(days=1)) if (end is None) else end
        if (isinstance(codelist, str)):
            data_day = QA.QA_fetch_index_day_adv(
                codelist[:6],
                start=start,
                end=end,)
        else:
            data_day = QA.QA_fetch_index_day_adv(
                [code[:6] for code in codelist],
                start=start,
                end=end,)
        if (data_day is None):
            print(u'没有K线数据。\n', codelist)
            return data_day, None
        else:
            # 计算每一行 close, open, high, low 为 NaN 的总和
            nan_sum = data_day.data[['close', 'open', 'high', 'low']].isna().sum(axis=1)

            # 筛选出 close 为 NaN 或 close < 0.168 的行，并结合 nan_sum > 2 的条件
            zero_trading = data_day.data[(data_day.data['close'].isna() | (data_day.data['volume'] < 0.168)) & (nan_sum > 2)]

            # 在 res 中 drop zero_trading
            data_day.data = data_day.data.drop(zero_trading.index)

            # QA 不支持 ETF 复权
            data_day.data[FLD.PCT_CHANGE_MAJOR] = np.log(data_day.data[AKA.CLOSE] / data_day.data[AKA.CLOSE].shift(1))
            fq_ckpo = data_day.data.query(f'({FLD.PCT_CHANGE_MAJOR} > 0.10832) | ({FLD.PCT_CHANGE_MAJOR} < -0.10832)')
            if (len(fq_ckpo) > 1e-12) and ((len(fq_ckpo) == 1) or (len(data_day.data.query(f'({FLD.PCT_CHANGE_MAJOR} > 0.20) | ({FLD.PCT_CHANGE_MAJOR} < -0.20)')))):
                if (is_stock_cn(codelist)[1] == MARKET_TYPE.INDEX_CN):
                    pass
                else:
                    print(f'\nETF {codelist[0]} 需要人工复权：\n', fq_ckpo)

            # data_day = data_day.to_qfq()
            if (len(codelist)==1):
                data_day.data[AKA.FULL_SYMBOL] = normalize_code(codelist[0], market_type=market_type)
                data_day.data[AKA.MARKET_TYPE] = market_type
        if (verbose):
            print('realtime:', realtime)
        if ((GQ_util_get_last_day()-pd.to_datetime(end)) > timedelta(days=1)):
            pass
        else:
            if (realtime):
                if (isinstance(codelist, str)):
                    data_day = GQ_fetch_stock_day_realtime_adv(
                        codelist[:6], data_day,
                        market_type=market_type, verbose=verbose)
                else:
                    data_day = GQ_fetch_stock_day_realtime_adv(
                        [code[:6] for code in codelist], data_day,
                        market_type=market_type, verbose=verbose)
    else:
        data_day = None
        print('data_day', codelist, market_type)

    if verbose:
        print('Code:{}, last time:{:%Y-%m-%d} total bars:{:d}'.format(data_day.data.index.get_level_values(level=1)[-1], 
                                                              data_day.data.index.get_level_values(level=0)[-1], 
                                                              len(data_day.data)))
    
    if (isinstance(data_day, QA_DataStruct_Stock_min) or \
        isinstance(data_day, QA_DataStruct_Stock_day)):
        try:
            codename = GQ_fetch_stock_name(codelist)
        except Exception:
            codename = [codelist]
        if (isinstance(codelist, str) or isinstance(codelist, list)):
            pass
        elif (len(codelist) != len(codename.drop_duplicates())):
            # 需要更新股票列表数据
            if (isinstance(codename, pd.DataFrame)):
                miss_codelist = [item for item in codelist if item not in codename[AKA.CODE].tolist()]
            else:
                miss_codelist = codelist
            if verbose:
                print(u'需要更新{}列表数据'.format(market_type_desc), 'miss_codelist', miss_codelist)
            if (isinstance(codename, pd.DataFrame)):
                codename = codename.reindex([*codename.index,
                                             *miss_codelist])
            # print(len(codename), codename)
    elif (isinstance(data_day, QA_DataStruct_Index_min) or \
        isinstance(data_day, QA_DataStruct_Index_day)):
        if (market_type_desc == 'A股ETF基金'):
            try:
                if (isinstance(codelist, str)):
                    codename = GQ_fetch_etf_name(codelist[:6])
                else:
                    codename = GQ_fetch_etf_name([code[:6] for code in codelist])
            except Exception:
                codename = []
            if (isinstance(codelist, str)):
                pass
            elif (len(codelist) != len(codename.drop_duplicates())):
                # 需要更新股票列表数据
                if (isinstance(codename, pd.DataFrame)):
                    miss_codelist = [item for item in codelist if item not in codename[AKA.CODE].tolist()]
                else:
                    miss_codelist = codelist
                if verbose:
                    print(u'需要更新{}列表数据'.format(market_type_desc), 'miss_codelist', miss_codelist)
                if (isinstance(codename, pd.DataFrame)):
                    codename = codename.reindex([*codename.index,
                                                 *miss_codelist])
        else:
            try:
                if (isinstance(codelist, list)):
                    all_etf_list = GQ_get_etf_list()
                    codename = codelist
                    if (len(codelist) == 1):
                        if (len(all_etf_list.query(f'code=={codelist[0]}')) > 0):
                            codename = all_etf_list.query(f'code=={codelist[0]}')['name'].item()
                elif (isinstance(codelist, str)):
                    print(codelist, market_type_desc)
                    codename = QA.QA_fetch_index_name(codelist[:6])       
                else:
                    print(codelist, market_type_desc)
                    codename = QA.QA_fetch_index_name([code[:6] for code in codelist])
            except Exception:
                traceback.print_exc()
                if (isinstance(codelist, str)):
                    codename = [codelist]
                else:
                    codename = codelist
            if (isinstance(codelist, str)):
                pass
            elif (isinstance(codename, list)):
                pass
            elif (len(codelist) != len(codename.drop_duplicates())):
                # 需要更新股票列表数据
                if (isinstance(codename, pd.DataFrame)):
                    miss_codelist = [item for item in codelist if item not in codename[AKA.CODE].tolist()]
                else:
                    miss_codelist = codelist
                if verbose:
                    print(u'需要更新{}列表数据'.format(market_type_desc), 'miss_codelist', miss_codelist)
                if (isinstance(codename, pd.DataFrame)):
                    codename = codename.reindex([*codename.index,
                                                 *miss_codelist])
    elif isinstance(codelist, list):
        if (len(codelist) == 1):
            codename = codelist[0]
        else:
            codename = '{}'.format(codelist)
    else:
        codename = codelist if isinstance(codelist, str) else codelist.item()

    # 计算每一行 close, open, high, low 为 NaN 的总和
    nan_sum = data_day.data[['close', 'open', 'high', 'low']].isna().sum(axis=1)

    # 筛选出 close 为 NaN 或 close < 0.168 的行，并结合 nan_sum > 2 的条件
    zero_trading = data_day.data[(data_day.data['close'].isna() | (data_day.data['volume'] < 0.168)) & (nan_sum > 2)]
    # 在 res 中 drop zero_trading
    data_day.data = data_day.data.drop(zero_trading.index)
    
    if verbose:
        if (len(zero_trading) > 0):
            print(f'\nget_kline_price_v3 detected zero_trading:{nan_sum}')
            print(zero_trading)

        print(u'{} get_kline_price_v3() 读取{}日K线历史数据完毕'.format(QA_util_timestamp_to_str()[2:16],
                                        market_type_desc), 
                codelist[-10:], '查询 {} 名称'.format(market_type_desc))
    
    if isinstance(codename, str):
        # 需要更新股票列表数据
        pass
    else:
        if (isinstance(codename, pd.DataFrame)) and \
            (len(codename)>1):
            codename=codename.tail(1).copy()
        elif isinstance(codename, list) and \
            (len(codename)>0):
            if (verbose):
                print(u'Code:{}, Code name:{}'.format(codelist, codename), )
            codename={AKA.CODE:codelist,
                    AKA.NAME:codename,
                    'market_type_desc':market_type_desc,
                    'market_type':market_type,}
        else:
            codename={AKA.CODE:codelist,
                    AKA.NAME:codelist,
                    'market_type_desc':market_type_desc,
                    'market_type':market_type,}
        try:
            codename['market_type_desc'] = market_type_desc
        except Exception:
            codename['market_type_desc'] = codelist
        codename['market_type'] = market_type
    return data_day, codename

