# coding:utf-8
# Author: 阿财（Rgveda@github）（4910163#qq.com）
# Created date: 2020-02-27
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
import datetime as dt
import numpy as np
import pandas as pd

try:
    import talib
except:
    print('PLEASE install TALIB to call these methods')
    pass

try:
    import QUANTAXIS as QA
    from QUANTAXIS.QAUtil.QAParameter import ORDER_DIRECTION
    from QUANTAXIS.QAData.QADataStruct import (
        QA_DataStruct_Index_min, 
        QA_DataStruct_Index_day, 
        QA_DataStruct_Stock_day, 
        QA_DataStruct_Stock_min,
        QA_DataStruct_CryptoCurrency_day,
        QA_DataStruct_CryptoCurrency_min,
        )
    from QUANTAXIS.QAIndicator.talib_numpy import *
    from QUANTAXIS.QAUtil.QADate_Adv import (
        QA_util_timestamp_to_str,
        QA_util_datetime_to_Unix_timestamp,
        QA_util_print_timestamp
    )
    from QUANTAXIS.QAUtil.QALogs import (
        QA_util_log_info, 
        QA_util_log_debug, 
        QA_util_log_expection
    )
except:
    print('PLEASE run "pip install QUANTAXIS" before call GolemQ.indices.fractal modules')
    pass

"""
talib K线分型库
"""

fractal_patterns = [{'code': 'CDL2CROWS',
                    'func': talib.CDL2CROWS,
                    'name': 'Two Crows',
                    'desc': '两只乌鸦',
                    'info': '三日K线模式，第一天长阳，第二天高开收阴，第三天再次高开继续收阴，收盘比前一日收盘价低，预示股价下跌。',
                    'trend': -1,
                    'remark': '看跌',},

                    {'code': 'Three Black Crows',
                    'func': talib.CDL3BLACKCROWS,
                    'name': 'Three Black Crows',
                    'desc': '三只乌鸦',
                    'info': '三日K线模式，连续三根阴线，每日收盘价都下跌且接近最低价，每日开盘价都在上根K线实体内，预示股价下跌。',
                    'trend': -1,
                    'remark': '看跌',},

                    {'code': 'CDL3INSIDE',
                    'func': talib.CDL3INSIDE,
                    'name': 'Three Inside Up/Down',
                    'desc': '三内部上涨和下跌',
                    'info': '三日K线模式，母子信号+长K线，以三内部上涨为例，K线为阴阳阳，第三天收盘价高于第一天开盘价，第二天K线在第一天K线内部，预示着股价上涨。',
                    'trend': 0,
                    'remark': '双向',},

                    {'code': 'CDL3LINESTRIKE',
                    'func': talib.CDL3LINESTRIKE,
                    'name': 'Three-Line Strike',
                    'desc': '三线打击',
                    'info': '四日K线模式，前三根阳线，每日收盘价都比前一日高，开盘价在前一日实体内，第四日市场高开，收盘价低于第一日开盘价，预示股价下跌。',
                    'trend': -1,
                    'remark': '看跌',},

                    {'code': 'CDL3OUTSIDE',
                    'func': talib.CDL3OUTSIDE,
                    'name': 'Three Outside Up/Down',
                    'desc': '三外部上涨和下跌',
                    'info': '三日K线模式，与三内部上涨和下跌类似，K线为阴阳阳，但第一日与第二日的K线形态相反，以三外部上涨为例，第一日K线在第二日K线内部，预示着股价上涨。',
                    'trend': 0,
                    'remark': '双向',},

                    {'code': 'CDL3STARSINSOUTH',
                    'func': talib.CDL3STARSINSOUTH ,
                    'name': 'Three Stars In The South',
                    'desc': '南方三星',
                    'info': '三日K线模式，与大敌当前相反，三日K线皆阴，第一日有长下影线，第二日与第一日类似，K线整体小于第一日，第三日无下影线实体信号，成交价格都在第一日振幅之内，预示下跌趋势反转，股价上升。',
                    'trend': 0,
                    'remark': '双向',},

                    {'code': 'CDL3WHITESOLDIERS',
                    'func': talib.CDL3WHITESOLDIERS,
                    'name': 'Three Advancing White Soldiers',
                    'desc': '三个白兵',
                    'info': '三日K线模式，三日K线皆阳，每日收盘价变高且接近最高价，开盘价在前一日实体上半部，预示股价上升。',
                    'trend': 0,
                    'remark': '双向',},

                    {'code': 'CDLABANDONEDBABY',
                    'func': talib.CDLABANDONEDBABY,
                    'name': 'Abandoned Baby',
                    'desc': '弃婴',
                    'info': '三日K线模式，第二日价格跳空且收十字星（开盘价与收盘价接近，最高价最低价相差不大），预示趋势反转，发生在顶部下跌，底部上涨。',
                    'trend': 0,
                    'remark': '双向',},

                    {'code': 'CDLADVANCEBLOCK',
                    'func': talib.CDLADVANCEBLOCK,
                    'name': 'Advance Block',
                    'desc': '大敌当前',
                    'info': '三日K线模式，三日都收阳，每日收盘价都比前一日高，开盘价都在前一日实体以内，实体变短，上影线变长。',
                    'trend': 0,
                    'remark': '双向',},

                    {'code': 'CDLBELTHOLD',
                    'func': talib.CDLBELTHOLD,
                    'name': 'Belt-hold',
                    'desc': '捉腰带线',
                    'info': '两日K线模式，下跌趋势中，第一日阴线，第二日开盘价为最低价，阳线，收盘价接近最高价，预示价格上涨。',
                    'trend': 0,
                    'remark': '双向',},

                    {'code': 'CDLBREAKAWAY',
                    'func': talib.CDLBREAKAWAY,
                    'name': 'Breakaway',
                    'desc': '脱离',
                    'info': '五日K线模式，以看涨脱离为例，下跌趋势中，第一日长阴线，第二日跳空阴线，延续趋势开始震荡，第五日长阳线，收盘价在第一天收盘价与第二天开盘价之间，预示价格上涨。',
                    'trend': 0,
                    'remark': '双向',},

                    {'code': 'CDLCLOSINGMARUBOZU',
                    'func': talib.CDLCLOSINGMARUBOZU,
                    'name': 'Closing Marubozu',
                    'desc': '收盘缺影线',
                    'info': '一日K线模式，以阳线为例，最低价低于开盘价，收盘价等于最高价，预示着趋势持续。',
                    'trend': 0,
                    'remark': '双向',},

                    {'code': 'CDLCONCEALBABYSWALL',
                    'func': talib.CDLCONCEALBABYSWALL,
                    'name': 'Concealing Baby Swallow',
                    'desc': '藏婴吞没',
                    'info': '四日K线模式，下跌趋势中，前两日阴线无影线，第二日开盘、收盘价皆低于第二日，第三日倒锤头，第四日开盘价高于前一日最高价，收盘价低于前一日最低价，预示着底部反转。',
                    'trend': 0,
                    'remark': '双向',},

                    {'code': 'CDLCOUNTERATTACK',
                    'func': talib.CDLCOUNTERATTACK,
                    'name': 'Counterattack',
                    'desc': '反击线',
                    'info': '二日K线模式，与分离线类似。',
                    'trend': 0,
                    'remark': '双向',},

                    {'code': 'CDLDARKCLOUDCOVER',
                    'func': talib.CDLDARKCLOUDCOVER,
                    'name': 'Dark Cloud Cover',
                    'desc': '乌云压顶',
                    'info': '二日K线模式，第一日长阳，第二日开盘价高于前一日最高价，收盘价处于前一日实体中部以下，预示着股价下跌。',
                    'trend': 0,
                    'remark': '双向',},

                    {'code': 'CDLDOJI',
                    'func': talib.CDLDOJI,
                    'name': 'Doji',
                    'desc': '十字',
                    'info': '一日K线模式，开盘价与收盘价基本相同。',
                    'trend': 0,
                    'remark': '双向',},

                    {'code': 'CDLDOJISTAR',
                    'func': talib.CDLDOJISTAR,
                    'name': 'Doji Star',
                    'desc': '十字星',
                    'info': '一日K线模式，开盘价与收盘价基本相同，上下影线不会很长，预示着当前趋势反转。',
                    'trend': 0,
                    'remark': '双向',},

                    {'code': 'CDLDRAGONFLYDOJI',
                    'func': talib.CDLDRAGONFLYDOJI,
                    'name': 'Dragonfly Doji',
                    'desc': '蜻蜓十字/T形十字',
                    'info': '一日K线模式，开盘后价格一路走低，之后收复，收盘价与开盘价相同，预示趋势反转。',
                    'trend': 0,
                    'remark': '双向',},

                    {'code': 'CDLENGULFING',
                    'func': talib.CDLENGULFING,
                    'name': 'Engulfing Pattern',
                    'desc': '吞噬模式',
                    'info': '两日K线模式，分多头吞噬和空头吞噬，以多头吞噬为例，第一日为阴线，第二日阳线，第一日的开盘价和收盘价在第二日开盘价收盘价之内，但不能完全相同。',
                    'trend': 0,
                    'remark': '双向',},

                    {'code': 'CDLEVENINGDOJISTAR',
                    'func': talib.CDLEVENINGDOJISTAR,
                    'name': 'Evening Doji Star',
                    'desc': '十字暮星',
                    'info': '三日K线模式，基本模式为暮星，第二日收盘价和开盘价相同，预示顶部反转。',
                    'trend': 0,
                    'remark': '双向',},

                    {'code': 'CDLEVENINGSTAR',
                    'func': talib.CDLEVENINGSTAR,
                    'name': 'Evening Star',
                    'desc': '暮星',
                    'info': '三日K线模式，与晨星相反，上升趋势中，第一日阳线，第二日价格振幅较小，第三日阴线，预示顶部反转。',
                    'trend': 0,
                    'remark': '双向',},

                    {'code': 'CDLGAPSIDESIDEWHITE',
                    'func': talib.CDLGAPSIDESIDEWHITE,
                    'name': 'Up/Down-gap side-by-side white lines',
                    'desc': '向上/下跳空并列阳线',
                    'info': '二日K线模式，上升趋势向上跳空，下跌趋势向下跳空，第一日与第二日有相同开盘价，实体长度差不多，则趋势持续。',
                    'trend': 0,
                    'remark': '双向',},

                    {'code': 'CDLGRAVESTONEDOJI',
                    'func': talib.CDLGRAVESTONEDOJI,
                    'name': 'Gravestone Doji',
                    'desc': '墓碑十字/倒T十字',
                    'info': '一日K线模式，开盘价与收盘价相同，上影线长，无下影线，预示底部反转。',
                    'trend': 0,
                    'remark': '双向',},

                    {'code': 'CDLHAMMER',
                    'func': talib.CDLHAMMER,
                    'name': 'Hammer',
                    'desc': '锤头',
                    'info': '一日K线模式，实体较短，无上影线，下影线大于实体长度两倍，处于下跌趋势底部，预示反转。',
                    'trend': 0,
                    'remark': '双向',},

                    {'code': 'CDLHANGINGMAN',
                    'func': talib.CDLHANGINGMAN,
                    'name': 'Hanging Man',
                    'desc': '上吊线',
                    'info': '一日K线模式，形状与锤子类似，处于上升趋势的顶部，预示着趋势反转。',
                    'trend': 0,
                    'remark': '双向',},

                    {'code': 'CDLHARAMI',
                    'func': talib.CDLHARAMI,
                    'name': 'Harami Pattern',
                    'desc': '母子线',
                    'info': '二日K线模式，分多头母子与空头母子，两者相反，以多头母子为例，在下跌趋势中，第一日K线长阴，第二日开盘价收盘价在第一日价格振幅之内，为阳线，预示趋势反转，股价上升。',
                    'trend': 0,
                    'remark': '双向',},

                    {'code': 'CDLHARAMICROSS',
                    'func': talib.CDLHARAMICROSS,
                    'name': 'Harami Cross Pattern',
                    'desc': '十字孕线',
                    'info': '二日K线模式，与母子县类似，若第二日K线是十字线，便称为十字孕线，预示着趋势反转。',
                    'trend': 0,
                    'remark': '双向',},

                    {'code': 'CDLHIGHWAVE',
                    'func': talib.CDLHIGHWAVE,
                    'name': 'High-Wave Candle',
                    'desc': '风高浪大线',
                    'info': '三日K线模式，具有极长的上/下影线与短的实体，预示着趋势反转。',
                    'trend': 0,
                    'remark': '双向',},

                    {'code': 'CDLHIKKAKE',
                    'func': talib.CDLHIKKAKE,
                    'name': 'Hikkake Pattern',
                    'desc': '陷阱',
                    'info': '三日K线模式，与母子类似，第二日价格在前一日实体范围内，第三日收盘价高于前两日，反转失败，趋势继续。',
                    'trend': 0,
                    'remark': '双向',},

                    {'code': 'CDLHIKKAKEMOD',
                    'func': talib.CDLHIKKAKEMOD,
                    'name': 'Modified Hikkake Pattern',
                    'desc': '修正陷阱',
                    'info': '三日K线模式，与陷阱类似，上升趋势中，第三日跳空高开；下跌趋势中，第三日跳空低开，反转失败，趋势继续。',
                    'trend': 0,
                    'remark': '双向',},

                    {'code': 'CDLHOMINGPIGEON',
                    'func': talib.CDLHOMINGPIGEON,
                    'name': 'Homing Pigeon',
                    'desc': '家鸽',
                    'info': '二日K线模式，与母子线类似，不同的的是二日K线颜色相同，第二日最高价、最低价都在第一日实体之内，预示着趋势反转。',
                    'trend': 0,
                    'remark': '双向',},

                    {'code': 'CDLIDENTICAL3CROWS',
                    'func': talib.CDLIDENTICAL3CROWS,
                    'name': 'Identical Three Crows',
                    'desc': '三胞胎乌鸦',
                    'info': '三日K线模式，上涨趋势中，三日都为阴线，长度大致相等，每日开盘价等于前一日收盘价，收盘价接近当日最低价，预示价格下跌。',
                    'trend': 0,
                    'remark': '双向',},

                    {'code': 'CDLINNECK',
                    'func': talib.CDLINNECK,
                    'name': 'In-Neck Pattern',
                    'desc': '颈内线',
                    'info': '二日K线模式，下跌趋势中，第一日长阴线，第二日开盘价较低，收盘价略高于第一日收盘价，阳线，实体较短，预示着下跌继续。',
                    'trend': 0,
                    'remark': '双向',},

                    {'code': 'CDLINVERTEDHAMMER',
                    'func': talib.CDLINVERTEDHAMMER,
                    'name': 'Inverted Hammer',
                    'desc': '倒锤头',
                    'info': '一日K线模式，上影线较长，长度为实体2倍以上，无下影线，在下跌趋势底部，预示着趋势反转。',
                    'trend': 0,
                    'remark': '双向',},

                    {'code': 'CDLKICKING',
                    'func': talib.CDLKICKING,
                    'name': 'Kicking',
                    'desc': '反冲形态',
                    'info': '二日K线模式，与分离线类似，两日K线为秃线，颜色相反，存在跳空缺口。',
                    'trend': 0,
                    'remark': '双向',},

                    {'code': 'CDLKICKINGBYLENGTH',
                    'func': talib.CDLKICKINGBYLENGTH,
                    'name': 'Kicking - bull/bear determined by the longer marubozu',
                    'desc': '由较长缺影线决定的反冲形态',
                    'info': '二日K线模式，与反冲形态类似，较长缺影线决定价格的涨跌。',
                    'trend': 0,
                    'remark': '双向',},

                    {'code': 'CDLLADDERBOTTOM',
                    'func': talib.CDLLADDERBOTTOM,
                    'name': 'Ladder Bottom',
                    'desc': '梯底',
                    'info': '五日K线模式，下跌趋势中，前三日阴线，开盘价与收盘价皆低于前一日开盘、收盘价，第四日倒锤头，第五日开盘价高于前一日开盘价，阳线，收盘价高于前几日价格振幅，预示着底部反转。',
                    'trend': 0,
                    'remark': '双向',},

                    {'code': 'CDLLONGLEGGEDDOJI',
                    'func': talib.CDLLONGLEGGEDDOJI,
                    'name': 'Long Legged Doji',
                    'desc': '长脚十字',
                    'info': '一日K线模式，开盘价与收盘价相同居当日价格中部，上下影线长，表达市场不确定性。',
                    'trend': 0,
                    'remark': '双向',},

                    {'code': 'CDLLONGLINE',
                    'func': talib.CDLLONGLINE,
                    'name': 'Long Line Candle',
                    'desc': '长蜡烛',
                    'info': '一日K线模式，K线实体长，无上下影线。',
                    'trend': 0,
                    'remark': '双向',},

                    {'code': 'CDLMARUBOZU',
                    'func': talib.CDLMARUBOZU,
                    'name': 'Marubozu',
                    'desc': '光头光脚/缺影线',
                    'info': '一日K线模式，上下两头都没有影线的实体，阴线预示着熊市持续或者牛市反转，阳线相反。',
                    'trend': 0,
                    'remark': '双向',},

                    {'code': 'CDLMATCHINGLOW',
                    'func': talib.CDLMATCHINGLOW,
                    'name': 'Matching Low',
                    'desc': '相同低价',
                    'info': '二日K线模式，下跌趋势中，第一日长阴线，第二日阴线，收盘价与前一日相同，预示底部确认，该价格为支撑位。',
                    'trend': 0,
                    'remark': '双向',},

                    {'code': 'CDLMATHOLD',
                    'func': talib.CDLMATHOLD,
                    'name': 'Mat Hold',
                    'desc': '铺垫',
                    'info': '五日K线模式，上涨趋势中，第一日阳线，第二日跳空高开影线，第三、四日短实体影线，第五日阳线，收盘价高于前四日，预示趋势持续。',
                    'trend': 0,
                    'remark': '双向',},

                    {'code': 'CDLMORNINGDOJISTAR',
                    'func': talib.CDLMORNINGDOJISTAR,
                    'name': 'Morning Doji Star',
                    'desc': '十字晨星',
                    'info': '三日K线模式，基本模式为晨星，第二日K线为十字星，预示底部反转。',
                    'trend': 0,
                    'remark': '双向',},

                    {'code': 'CDLMORNINGSTAR',
                    'func': talib.CDLMORNINGSTAR,
                    'name': 'Morning Star',
                    'desc': '晨星',
                    'info': '三日K线模式，下跌趋势，第一日阴线，第二日价格振幅较小，第三天阳线，预示底部反转。',
                    'trend': 0,
                    'remark': '双向',},

                    {'code': 'CDLONNECK',
                    'func': talib.CDLONNECK,
                    'name': 'On-Neck Pattern',
                    'desc': '颈上线',
                    'info': '二日K线模式，下跌趋势中，第一日长阴线，第二日开盘价较低，收盘价与前一日最低价相同，阳线，实体较短，预示着延续下跌趋势。',
                    'trend': 0,
                    'remark': '双向',},

                    {'code': 'CDLPIERCING',
                    'func': talib.CDLPIERCING,
                    'name': 'Piercing Pattern',
                    'desc': '刺透形态',
                    'info': '两日K线模式，下跌趋势中，第一日阴线，第二日收盘价低于前一日最低价，收盘价处在第一日实体上部，预示着底部反转。',
                    'trend': 0,
                    'remark': '双向',},

                    {'code': 'CDLRICKSHAWMAN',
                    'func': talib.CDLRICKSHAWMAN,
                    'name': 'Rickshaw Man',
                    'desc': '黄包车夫',
                    'info': '一日K线模式，与长腿十字线类似，若实体正好处于价格振幅中点，称为黄包车夫。',
                    'trend': 0,
                    'remark': '双向',},

                    {'code': 'CDLRISEFALL3METHODS',
                    'func': talib.CDLRISEFALL3METHODS,
                    'name': 'Rising/Falling Three Methods',
                    'desc': '上升/下降三法',
                    'info': '五日K线模式，以上升三法为例，上涨趋势中，第一日长阳线，中间三日价格在第一日范围内小幅震荡，第五日长阳线，收盘价高于第一日收盘价，预示股价上升。',
                    'trend': 0,
                    'remark': '双向',},

                    {'code': 'CDLSEPARATINGLINES',
                    'func': talib.CDLSEPARATINGLINES,
                    'name': 'Separating Lines',
                    'desc': '分离线',
                    'info': '二日K线模式，上涨趋势中，第一日阴线，第二日阳线，第二日开盘价与第一日相同且为最低价，预示着趋势继续。',
                    'trend': 0,
                    'remark': '双向',},

                    {'code': 'CDLSHOOTINGSTAR',
                    'func': talib.CDLSHOOTINGSTAR,
                    'name': 'Shooting Star',
                    'desc': '射击之星',
                    'info': '一日K线模式，上影线至少为实体长度两倍，没有下影线，预示着股价下跌',
                    'trend': 0,
                    'remark': '双向',},

                    {'code': 'CDLSHORTLINE',
                    'func': talib.CDLSHORTLINE,
                    'name': 'Short Line Candle',
                    'desc': '短蜡烛',
                    'info': '一日K线模式，实体短，无上下影线。',
                    'trend': 0,
                    'remark': '双向',},

                    {'code': 'CDLSPINNINGTOP',
                    'func': talib.CDLSPINNINGTOP,
                    'name': 'Spinning Top',
                    'desc': '纺锤',
                    'info': '一日K线，实体小。',
                    'trend': 0,
                    'remark': '双向',},

                    {'code': 'CDLSTALLEDPATTERN',
                    'func': talib.CDLSTALLEDPATTERN,
                    'name': 'Stalled Pattern',
                    'desc': '停顿形态',
                    'info': '三日K线模式，上涨趋势中，第二日长阳线，第三日开盘于前一日收盘价附近，短阳线，预示着上涨结束。',
                    'trend': 0,
                    'remark': '双向',},

                    {'code': 'CDLSTICKSANDWICH',
                    'func': talib.CDLSTICKSANDWICH,
                    'name': 'Stick Sandwich',
                    'desc': '条形三明治',
                    'info': '三日K线模式，第一日长阴线，第二日阳线，开盘价高于前一日收盘价，第三日开盘价高于前两日最高价，收盘价于第一日收盘价相同。',
                    'trend': 0,
                    'remark': '双向',},

                    {'code': 'CDLTAKURI',
                    'func': talib.CDLTAKURI,
                    'name': 'Takuri (Dragonfly Doji with very long lower shadow)',
                    'desc': '探水竿',
                    'info': '一日K线模式，大致与蜻蜓十字相同，下影线长度长。',
                    'trend': 0,
                    'remark': '双向',},

                    {'code': 'CDLTASUKIGAP',
                    'func': talib.CDLTASUKIGAP,
                    'name': 'Tasuki Gap',
                    'desc': '跳空并列阴阳线',
                    'info': '三日K线模式，分上涨和下跌，以上升为例，前两日阳线，第二日跳空，第三日阴线，收盘价于缺口中，上升趋势持续。',
                    'trend': 0,
                    'remark': '双向',},

                    {'code': 'CDLTHRUSTING',
                    'func': talib.CDLTHRUSTING,
                    'name': 'Thrusting Pattern',
                    'desc': '插入',
                    'info': '二日K线模式，与颈上线类似，下跌趋势中，第一日长阴线，第二日开盘价跳空，收盘价略低于前一日实体中部，与颈上线相比实体较长，预示着趋势持续。',
                    'trend': 0,
                    'remark': '双向',},

                    {'code': 'CDLTRISTAR',
                    'func': talib.CDLTRISTAR,
                    'name': 'Tristar Pattern',
                    'desc': '三星',
                    'info': '三日K线模式，由三个十字组成，第二日十字必须高于或者低于第一日和第三日，预示着反转。',
                    'trend': 0,
                    'remark': '双向',},

                    {'code': 'CDLUNIQUE3RIVER',
                    'func': talib.CDLUNIQUE3RIVER,
                    'name': 'Unique 3 River',
                    'desc': '奇特三河床',
                    'info': '三日K线模式，下跌趋势中，第一日长阴线，第二日为锤头，最低价创新低，第三日开盘价低于第二日收盘价，收阳线，收盘价不高于第二日收盘价，预示着反转，第二日下影线越长可能性越大。',
                    'trend': 0,
                    'remark': '双向',},

                    {'code': 'CDLUPSIDEGAP2CROWS',
                    'func': talib.CDLUPSIDEGAP2CROWS,
                    'name': 'Upside Gap Two Crows',
                    'desc': '向上跳空的两只乌鸦',
                    'info': '三日K线模式，第一日阳线，第二日跳空以高于第一日最高价开盘，收阴线，第三日开盘价高于第二日，收阴线，与第一日比仍有缺口。',
                    'trend': 0,
                    'remark': '双向',},

                    {'code': 'CDLXSIDEGAP3METHODS',
                    'func': talib.CDLXSIDEGAP3METHODS,
                    'name': 'Upside/Downside Gap Three Methods',
                    'desc': '上升/下降跳空三法',
                    'info': '五日K线模式，以上升跳空三法为例，上涨趋势中，第一日长阳线，第二日短阳线，第三日跳空阳线，第四日阴线，开盘价与收盘价于前两日实体内，第五日长阳线，收盘价高于第一日收盘价，预示股价上升。',
                    'trend': 0,
                    'remark': '双向',},]


def talib_patterns_func(data, *args, **kwargs):
    """
    计算Talib Pattern Recognition Functions, 统计出累加值
    累加值大的，有极大几率为短期最佳买入点和卖出点
    支持QA add_func，第二个参数 默认为 indices= 为已经计算指标
    理论上这个函数只计算单一标的，不要尝试传递复杂标的，indices会尝试拆分。
    """
    # 针对多标的，拆分 indices 数据再自动合并
    code = data.index.get_level_values(level=1)[0]
    if ('indices' in kwargs.keys()):
        indices = kwargs['indices'].loc[(slice(None), code), :]
    elif (len(args) > 0):
        indices = args[0].loc[(slice(None), code), :]
    else:
        indices = None

    open = data.open.values
    high = data.high.values
    low = data.low.values
    close = data.close.values
    ret_cumsum = None
    ret_labels = None

    for fractal_pattern in fractal_patterns:
        integer = fractal_pattern['func'](open, high, low, close)

        if (ret_cumsum is None):
            ret_cumsum = integer
            ret_labels = integer
        else:
            ret_cumsum = ret_cumsum + integer
            ret_labels = np.c_[ret_labels, integer]

    if (indices is None):
        indices = pd.DataFrame(ret_cumsum, 
                        columns=['talib_patterns'], 
                        index=data.index)
    else:
        indices['talib_patterns'] = ret_cumsum

    indices['dcperiod'] = talib.HT_DCPERIOD(data.close.values)
    return indices


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    pd.set_option('display.max_rows', 120)

    # 暗色主题
    plt.style.use('Solarize_Light2')

    # 正常显示中文字体
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']

    # 股票代码，如果选股以后：我们假设有这些代码
    codelist = ['159919', '159908', '159902', '510900', 
                '513100', '512980', '515000', '512800', 
                '512170', '510300', '159941', '512690',
                '159928']
    codelist = ['510300']
    # 获取ETF基金中文名称，只是为了看得方便，交易策略并不需要ETF基金中文名称
    stock_names = QA.QA_fetch_etf_name(codelist)
    codename = [stock_names.at[code, 'name'] for code in codelist]
    codename_by_symbol = {codelist[i]:codename[i] for i in range(len(codelist))}

    # 读取 ETF基金 日线，存在index_day中
    data_day = QA.QA_fetch_index_day_adv(codelist,
        start='2014-01-01',
        end='{}'.format(dt.date.today()))

    ## 股票代码，我直接用我的选股程序获取选股列表。这段别人运行不了，所以注释掉了
    #position_signals = position(portfolio='sharpe_scale_patterns_day',
    #                            frequency='day',
    #                            market_type=QA.MARKET_TYPE.STOCK_CN,
    #                            verbose=False)
    #codelist = position_signals.index.get_level_values(level=1).to_list()

    ## 获取股票中文名称，只是为了看得方便，交易策略并不需要股票中文名称
    #stock_names = QA.QA_fetch_stock_name(codelist)
    #codename = [stock_names.at[code, 'name'] for code in codelist]
    #print(codename)

    #data_day = QA.QA_fetch_stock_day_adv(codelist,
    #    start='2014-01-01',
    #    end='{}'.format(datetime.date.today())).to_qfq()

    print('总共有{}个talib K线形态进行识别。'.format(len(fractal_patterns)))

    indices = data_day.add_func(talib_patterns_func)

    indices['talib_patterns_norm'] = indices['talib_patterns'] / 20
    indices['close'] = data_day.data.close

    #indices['dcphase'] = talib.HT_DCPHASE(data_day.data.close)
    #indices['inhpase'], indices['quadrature'] =
    #talib.HT_PHASOR(data_day.data.close)
    #indices['sine'], indices['leadsine'] = sine, leadsine =
    #talib.HT_SINE(data_day.data.close)
    #indices['trendmode'] = talib.HT_TRENDMODE(data_day.data.close)

    #indices[['close','dcperiod','dcphase','inhpase',
    #    'quadrature','sine','leadsine','trendmode']
    #    ].plot(figsize=(20,18), subplots=True, layout=(4,2))
    #plt.subplots_adjust(wspace=0, hspace=0.2)
    indices[['dcperiod', 'talib_patterns_norm']].tail(180).plot()
    print(indices[['dcperiod', 'talib_patterns']].tail(120))
    plt.show()