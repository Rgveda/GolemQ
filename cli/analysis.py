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
import datetime
import numpy as np
import pandas as pd

try:
    import talib
except:
    print('PLEASE run "pip install TALIB" to call these modules')
    pass
try:
    import QUANTAXIS as QA
    from QUANTAXIS.QAUtil.QAParameter import ORDER_DIRECTION
    from QUANTAXIS.QAData.QADataStruct import (
        QA_DataStruct_Index_min, 
        QA_DataStruct_Index_day, 
        QA_DataStruct_Stock_day, 
        QA_DataStruct_Stock_min
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
    from QUANTAXIS.QAUtil.QADate_trade import (
        QA_util_get_pre_trade_date,
        QA_util_get_real_date,
        trade_date_sse
    )
except:
    print('PLEASE run "pip install QUANTAXIS" before call GolemQ.cli.show_number modules')
    pass

from GolemQ.fetch.Portfolio_signals import (
    GQSignal_fetch_position_singal_day, 
    GQSignal_fetch_position_singal_min,
    GQSignal_fetch_mainfest_singal_day,
    GQSignal_fetch_flu_cunsum_day,
    GQSignal_fetch_flu_cunsum_min
)
from GolemQ.fetch.base import (
    market_brief,
)
from GolemQ.utils.parameter import (
    AKA, 
    INDICATOR_FIELD as FLD,
    TREND_STATUS as ST
)

from GolemQ.analysis.timeseries import (
    Timeline_Integral,
)

def position(portfolio:str='myportfolio',
             frequency:str='day',
             market_type:str=QA.MARKET_TYPE.STOCK_CN,
             cumsum_signals:pd.DataFrame=None,
             verbose:bool=True):
    """
    读取今日操作 Number
    """
    nodata, n = True, 0
    last_tradedate = QA_util_get_pre_trade_date('{}'.format(datetime.date.today()), n=n)
    while (nodata):
        position_signals = GQSignal_fetch_position_singal_day(start=last_tradedate,
            end='{}'.format(datetime.date.today() + datetime.timedelta(days=1)),
            market_type=QA.MARKET_TYPE.STOCK_CN,
            portfolio=portfolio,
            format='pd')

        mainfest_signals = GQSignal_fetch_mainfest_singal_day(start=last_tradedate,
            end='{}'.format(datetime.date.today() + datetime.timedelta(days=1)),
            market_type=QA.MARKET_TYPE.STOCK_CN,
            portfolio=portfolio,
            format='pd')

        if (n > 10):
            if (verbose == True):
                print(u'最近10个交易日内没有找到任何全市场趋势买卖分析数据....\n在QUANTAXIS 中 save X 保存数据，跑全市场策略，然后再来这里查看输出结果。')
            nodata = False
            return None
        elif (position_signals is not None) and \
            (len(position_signals) >= 0):
            nodata = False
        elif (position_signals is None) or \
            (len(position_signals) == 0):
            if (n == 0):
                if (verbose == True):
                    print(u"""最后一个交易日为：{} ，但是 “show me a number” 模块没有查询到分析计算数据。\n
                        在QUANTAXIS 中 save X 保存数据，跑全市场策略，然后再来这里查看输出结果。\n
                        尝试往前回溯更多的交易日。""".format(last_tradedate))
            n = n + 1
            last_tradedate = QA_util_get_pre_trade_date('{}'.format(datetime.date.today()), n=n)
 
    position_signals[FLD.ZEN_WAVELET_MINOR_TIMING_LAG] = np.nan_to_num(position_signals[FLD.ZEN_WAVELET_MINOR_TIMING_LAG], nan=0)
    position_signals[FLD.BOOTSTRAP_COMBO_MINOR_TIMING_LAG] = np.nan_to_num(position_signals[FLD.BOOTSTRAP_COMBO_MINOR_TIMING_LAG], nan=0)
    position_signals[u'缠周期'] = position_signals.apply(lambda x: '{:d}/{:d}'.format(int(x[FLD.ZEN_WAVELET_TIMING_LAG]),
                                                                                     int(x[FLD.ZEN_WAVELET_MINOR_TIMING_LAG])), 
                                                         axis=1)
    position_signals[u'挂单价'] = position_signals.apply(lambda x: '{:.2f}/{:.2f}'.format(x[FLD.RENKO_TREND_S_UB],
                                                                                         x[FLD.RENKO_TREND_S_UB]), 
                                                         axis=1)
    position_signals[u'突破天'] = position_signals.apply(lambda x: '{:d}/{:d}'.format(x[FLD.HYBIRD_TRI_MA_JX_RS],
                                                                                     int(x[FLD.BOOTSTRAP_GROUND_ZERO_TIMING_LAG])), 
                                                         axis=1)
    position_signals[u'赚钱指数'] = position_signals.apply(lambda x: '{:.2f}/{:.2f}'.format(x[FLD.SHARPE_RATIO],
                                                                                          x[FLD.SORTINO_RATIO]), 
                                                         axis=1)
    position_signals['RENKO/OPT'] = position_signals.apply(lambda x: '{:d}/{:d}'.format(int(x[FLD.RENKO_TREND_S_TIMING_LAG]),
                                                                                        int(x[FLD.BOLL_RENKO_TIMING_LAG])), 
                                                         axis=1)
    position_signals[u'启动天'] = position_signals.apply(lambda x: '{:.1f}/{:.1f}'.format(x[FLD.BOOTSTRAP_COMBO_TIMING_LAG],
                                                                                         x[FLD.BOOTSTRAP_COMBO_MINOR_TIMING_LAG]), 
                                                         axis=1)
    position_signals[u'力量/仓位'] = position_signals.apply(lambda x: '{:.1%}/{:.1%}'.format(x[FLD.COMBINE_DENSITY],
                                                                                            x[FLD.LEVERAGE_ONHOLD]), 
                                                         axis=1)
    position_signals[u'买点/偏离'] = position_signals.apply(lambda x: '{:d}/{:.1%}'.format(x[ST.TRIGGER_RPS],
                                                                                          x[FLD.BIAS3_ZSCORE]), 
                                                         axis=1)
    position_signals[ST.CANDIDATE] = position_signals.apply(lambda x: '{}/{}'.format(x[ST.CANDIDATE],
                                                                                          x[ST.CANDIDATE_MINOR]), 
                                                         axis=1)

    mainfest_signals[FLD.ZEN_WAVELET_MINOR_TIMING_LAG] = np.nan_to_num(mainfest_signals[FLD.ZEN_WAVELET_MINOR_TIMING_LAG], nan=0)
    mainfest_signals[u'缠周期'] = mainfest_signals.apply(lambda x: '{:d}/{:d}'.format(int(x[FLD.ZEN_WAVELET_TIMING_LAG]),
                                                                                     int(x[FLD.ZEN_WAVELET_MINOR_TIMING_LAG])), 
                                                         axis=1)
    mainfest_signals[u'挂单价'] = mainfest_signals.apply(lambda x: '{:.2f}/{:.2f}'.format(x[FLD.RENKO_TREND_S_UB],
                                                                                         x[FLD.RENKO_TREND_S_UB]), 
                                                         axis=1)
    mainfest_signals[u'突破天'] = mainfest_signals.apply(lambda x: '{:d}/{:d}'.format(x[FLD.HYBIRD_TRI_MA_JX_RS],
                                                                                     int(x[FLD.BOOTSTRAP_GROUND_ZERO_TIMING_LAG])), 
                                                         axis=1)
    mainfest_signals[u'赚钱指数'] = mainfest_signals.apply(lambda x: '{:.2f}/{:.2f}'.format(x[FLD.SHARPE_RATIO],
                                                                                          x[FLD.SORTINO_RATIO]), 
                                                         axis=1)
    mainfest_signals['RENKO/OPT'] = mainfest_signals.apply(lambda x: '{:d}/{:d}'.format(int(x[FLD.RENKO_TREND_S_TIMING_LAG]),
                                                                                        int(x[FLD.BOLL_RENKO_TIMING_LAG])), 
                                                         axis=1)
    mainfest_signals[u'启动天'] = mainfest_signals.apply(lambda x: '{:.1f}/{:.1f}'.format(x[FLD.BOOTSTRAP_COMBO_TIMING_LAG],
                                                                                         x[FLD.BOOTSTRAP_COMBO_MINOR_TIMING_LAG]), 
                                                         axis=1)
    mainfest_signals[u'力量/仓位'] = mainfest_signals.apply(lambda x: '{:.1%}/{:.1%}'.format(x[FLD.COMBINE_DENSITY],
                                                                                              x[FLD.LEVERAGE_ONHOLD]), 
                                                         axis=1)
    mainfest_signals[u'买点/偏离'] = mainfest_signals.apply(lambda x: '{:d}/{:.1%}'.format(x[ST.TRIGGER_RPS],
                                                                                          x[FLD.BIAS3_ZSCORE]), 
                                                         axis=1)
    mainfest_signals[ST.CANDIDATE] = mainfest_signals.apply(lambda x: '{}/{}'.format(x[ST.CANDIDATE],
                                                                                          x[ST.CANDIDATE_MINOR]), 
                                                         axis=1)

    retreat_all = False
    if (cumsum_signals is None) and (verbose == False):
        return position_signals


    pd.set_option('display.float_format',lambda x : '%.3f' % x)
    pd.set_option('display.max_columns', 22)
    pd.set_option("display.max_rows", 300)
    pd.set_option('display.width', 240)  # 设置打印宽度
    pd.set_option('display.unicode.ambiguous_as_wide', True)
    pd.set_option('display.unicode.east_asian_width', True)

    cumsum_signals[ST.CLUSTER_GROUP_TOWARDS_RS] = cumsum_signals[ST.CLUSTER_GROUP_TOWARDS].diff().rolling(4).sum()
    #print(cumsum_signals[['missive_dif', FLD.MAXFACTOR_CROSS,
    #ST.CLUSTER_GROUP_TOWARDS_RS, 'massive_flu', 'master_stream',
    #'massive_trend']].head(180))
    #print(cumsum_signals[['missive_dif', FLD.MAXFACTOR_CROSS,
    #ST.CLUSTER_GROUP_TOWARDS_RS, 'massive_flu', 'master_stream',
    #'massive_trend']].tail(180))
    if (cumsum_signals.at[position_signals.index.get_level_values(level=0)[0], 'massive_flu'] < 0):
        retreat_all = True
        #print(u'大盘趋势进入下跌浪，建议个股只卖不买。除超强势个股（上涨天数超过400天以上）')

    position_dual_punch = position_signals[(position_signals[ST.TRIGGER_RPS].gt(80) & \
                                            position_signals[ST.TRIGGER_RPS].lt(85))].copy()
    print('position_dual_punch:', len(position_dual_punch), position_dual_punch.index.get_level_values(level=1).unique().values)

    mainfest_punch = mainfest_signals[(mainfest_signals[ST.TRIGGER_R5].gt(0) | \
                                       mainfest_signals[ST.TRIGGER_RPS].gt(0))].copy()
    print('mainfest_punch:', len(mainfest_punch), mainfest_punch.index.get_level_values(level=1).unique().values)
    #each_day = sorted(position_signals.index.get_level_values(level=0).unique())
    #for i in range(1, 6):
    #    print(position_dual_punch.loc[(each_day[-i], slice(None)), [AKA.NAME, ]])
    
    position_singals_fade = position_signals.query('({}<0 & {}<0 & {}<0) | \
                                                   ({}<1.5 & {}<1.5)'.format(FLD.BOOTSTRAP_COMBO_TIMING_LAG,
                                        FLD.BOOTSTRAP_GROUND_ZERO_TIMING_LAG, 
                                        ST.POSITION_ONHOLD, 
                                        FLD.SHARPE_RATIO,
                                        FLD.SORTINO_RATIO))
    position_signals_final = position_signals.drop(position_singals_fade.index)
    position_signals_best = position_signals_final.loc[position_signals[FLD.LEVERAGE_ONHOLD].gt(0.99) | \
                                                 (position_signals[ST.TRIGGER_RPS].gt(80) & \
                                                 position_signals[ST.TRIGGER_RPS].lt(85)) | \
                                                 (position_signals[FLD.BOLL_RENKO_TIMING_LAG].gt(0) & \
                                                 position_signals[FLD.BOLL_RENKO_TIMING_LAG].le(4)) | \
                                                 (position_signals[FLD.BOLL_RENKO_TIMING_LAG].gt(0) & \
                                                 position_signals[ST.TRIGGER_RPS].gt(0)) | \
                                                 (position_signals[ST.TRIGGER_RPS].gt(0) & \
                                                 position_signals[FLD.COMBINE_DENSITY].gt(0.5)) | \
                                                 (position_signals[ST.TRIGGER_RPS].gt(0) & \
                                                 position_signals[FLD.NEGATIVE_LOWER_PRICE].gt(0) & \
                                                 position_signals[FLD.COMBINE_DENSITY].gt(0.382)) | \
                                                 (position_signals[ST.TRIGGER_RPS].gt(0) & \
                                                 position_signals[FLD.COMBO_FLOW].gt(0)), :].copy()
    position_signals_best = position_signals_best.sort_values(by=ST.TRIGGER_RPS, 
                                                              ascending=False)
    #position_dual_punch = position_signals_best[(position_signals[ST.TRIGGER_RPS].gt(80) & \
    #                                             position_signals[ST.TRIGGER_RPS].lt(85))].copy()
    #print('position_dual_punch:', len(position_dual_punch))

    if (verbose == True):
        print(u'\n根据最后一个交易日K线走势基于策略“{}”筛选出来....'.format(portfolio))
        print(u'\n第零组：大盘趋势进入上涨形态第{}天，以下个股处于底部启动形态，***抢反弹高风险***'.format(cumsum_signals.at[position_signals.index.get_level_values(level=0)[0], 'max_trend_jx_count']) if (not retreat_all) else u'第零组：大盘趋势进入下跌形态第{}天（抢反弹高风险）'.format(cumsum_signals.at[position_signals.index.get_level_values(level=0)[0], 'max_trend_sx_count']))

    position_signals_grade0 = position_signals_best.loc[((position_signals_best[FLD.BOOTSTRAP_GROUND_ZERO_MINOR_TIMING_LAG].gt(0) & \
                                                        position_signals_best[FLD.BOOTSTRAP_COMBO_TIMING_LAG].gt(0)) | \
                                                        (position_signals_best[FLD.DEA].lt(0))) & \
                                                        (position_signals_best[FLD.NEGATIVE_LOWER_PRICE].gt(0) & \
                                                        position_signals_best[FLD.BOOTSTRAP_COMBO_TIMING_LAG].gt(0)), :].copy()

    if (verbose == True):
        print(position_signals_grade0[[AKA.NAME,
                                        u'力量/仓位',
                                        u'买点/偏离',
                                        u'赚钱指数',
                                        ST.CANDIDATE,
                                        u'突破天',
                                        u'启动天',
                                        u'缠周期',
                                        u'挂单价']].head(100))

    if (verbose == True):
        print(u'\n根据最后一个交易日K线走势基于策略“{}”筛选出来....'.format(portfolio))
        print(u'\n第一组：大盘趋势进入上涨形态第{}天，以下个股处于底部启动形态，***推荐***'.format(cumsum_signals.at[position_signals.index.get_level_values(level=0)[0], 'max_trend_jx_count']) if (not retreat_all) else u'第一组：大盘趋势进入下跌形态第{}天'.format(cumsum_signals.at[position_signals.index.get_level_values(level=0)[0], 'max_trend_sx_count']))

    position_signals_grade1 = position_signals_best.loc[((position_signals_best[FLD.BOOTSTRAP_GROUND_ZERO_MINOR_TIMING_LAG].gt(0) & \
                                                        position_signals_best[FLD.BOOTSTRAP_COMBO_TIMING_LAG].gt(0)) | \
                                                        (position_signals_best[FLD.DEA].lt(0))) & \
                                                        position_signals_best[ST.BOOTSTRAP_I].gt(0), :].copy()
    
    if (verbose == True):
        print(position_signals_grade1[[AKA.NAME,
                                        u'力量/仓位',
                                        u'买点/偏离',
                                        u'赚钱指数',
                                        ST.CANDIDATE,
                                        u'突破天',
                                        u'启动天',
                                        u'缠周期',
                                        u'挂单价']].head(100))

    position_signals_final = position_signals_best.drop(position_signals_grade1.index).drop(position_signals_grade0.index)

    if (verbose == True):
        print(u'\n第二组：大盘趋势进入上涨形态第{}天，以下个股处于启动形态，***次要推荐买入***'.format(cumsum_signals.at[position_signals.index.get_level_values(level=0)[0], 'max_trend_jx_count']) if (not retreat_all) else u'第二组：大盘趋势进入下跌形态第{}天'.format(cumsum_signals.at[position_signals.index.get_level_values(level=0)[0], 'max_trend_sx_count']))
    #position_signals_grade2 = position_signals_final.query('{}=="{}" &
    #{}>0'.format(ST.CANDIDATE,
    #                                                     ST.BOOTSTRAP_II,
    #                                                     FLD.HYBIRD_TRI_MA_JX_RS,)).loc[:,[AKA.NAME,
    #                                    FLD.HYBIRD_TRI_MA_JX,
    #                                    FLD.HYBIRD_TRI_MA_JX_RS,
    #                                    u'建仓价位',
    #                                    FLD.VOLUME_FLOW_TRI_CROSS_JX,
    #                                    ST.CANDIDATE]]
    position_signals_grade2 = position_signals_final.loc[((position_signals_best[FLD.BOOTSTRAP_GROUND_ZERO_MINOR_TIMING_LAG].gt(0) & \
                                                        position_signals_best[FLD.BOOTSTRAP_COMBO_TIMING_LAG].gt(0)) | \
                                                        (position_signals_best[FLD.DEA].lt(0))) & \
                                                        position_signals_final[ST.BOOTSTRAP_II].gt(0) & \
                                                         ((position_signals[FLD.COMBINE_DENSITY].gt(0.512) & \
                                                         position_signals[FLD.BIAS3].gt(0) & \
                                                         (position_signals[FLD.BIAS3].shift(1).lt(0) | \
                                                         position_signals[FLD.BIAS3].shift(2).lt(0) | \
                                                         position_signals[FLD.BIAS3].shift(2).lt(0) | \
                                                         (position_signals[FLD.ZSCORE_21].shift(1) < position_signals[FLD.ZSCORE_21]) | \
                                                         (position_signals[FLD.ZSCORE_84].shift(1) < position_signals[FLD.ZSCORE_84]))) | \
                                                         (position_signals[FLD.BOLL_RENKO_TIMING_LAG].gt(0) & \
                                                         position_signals[FLD.BOLL_RENKO_TIMING_LAG].le(4)) | \
                                                         (position_signals[FLD.BOLL_RENKO_TIMING_LAG].gt(0) & \
                                                         position_signals[ST.TRIGGER_RPS].gt(0))), 
                                                 :].copy()
    position_signals_grade2.sort_values(by=ST.TRIGGER_RPS, 
                                      inplace=True,
                                      ascending=False)

    if (verbose == True):
        print(position_signals_grade2[[AKA.NAME,
                                        u'力量/仓位',
                                        u'买点/偏离',
                                        u'赚钱指数',
                                        ST.CANDIDATE,
                                        u'突破天',
                                        u'启动天',
                                        u'缠周期',
                                        u'挂单价']].head(100))

    position_signals_final = position_signals_final.drop(position_signals_grade2.index)

    if (verbose == True):
        print(u'\n第三组：反弹形态最好，***潜力反弹个股***' if (not retreat_all) else u'第三组：大盘趋势进入下跌形态第{}天'.format(cumsum_signals.at[position_signals.index.get_level_values(level=0)[0], 'max_trend_sx_count']))

    position_signals_grade3 = position_signals_final.loc[((position_signals_best[FLD.BOOTSTRAP_GROUND_ZERO_MINOR_TIMING_LAG].gt(0) & \
                                                        position_signals_best[FLD.BOOTSTRAP_COMBO_TIMING_LAG].gt(0)) | \
                                                        (position_signals_best[FLD.DEA].lt(0))) & \
                                                        position_signals[ST.TRIGGER_RPS].gt(0) & \
                                                         ((position_signals[FLD.BOLL_RENKO_TIMING_LAG].gt(0) & \
                                                         position_signals[FLD.BOLL_RENKO_TIMING_LAG].le(4)) | \
                                                         (position_signals[FLD.BOLL_RENKO_TIMING_LAG].gt(0) & \
                                                         position_signals[ST.TRIGGER_RPS].gt(0))), 
                                                 :].copy()

    position_signals_grade3.sort_values(by=ST.TRIGGER_RPS, 
                                        inplace=True,
                                        ascending=False)
    if (verbose == True):
        print(position_signals_grade3[[AKA.NAME,
                                        u'力量/仓位',
                                        u'买点/偏离',
                                        u'赚钱指数',
                                        ST.CANDIDATE,
                                        u'突破天',
                                        u'启动天',
                                        u'缠周期',
                                        u'挂单价']].head(100))

    position_signals_final = position_signals_final.drop(position_signals_grade3.index)
    position_signals_final.sort_values(by=ST.TRIGGER_RPS, 
                                      inplace=True,
                                      ascending=False)
    position_signals_grade4 = position_signals_final.loc[((position_signals_best[FLD.BOOTSTRAP_GROUND_ZERO_MINOR_TIMING_LAG].gt(0) & \
                                                        position_signals_best[FLD.BOOTSTRAP_COMBO_TIMING_LAG].gt(0)) | \
                                                        (position_signals_best[FLD.DEA].lt(0))) & \
                                                        (position_signals[FLD.BOOTSTRAP_COMBO_TIMING_LAG].gt(0) | \
                                                         position_signals[FLD.BOOTSTRAP_GROUND_ZERO_TIMING_LAG].gt(0)), 
                                                 :].copy()
    if (verbose == True):
        print(u'\n第四组：这些个股也许还没发现价值所在，看清楚之前，***强烈不推荐买入，但可以持有***' if (not retreat_all) else u'第四组：大盘趋势进入下跌形态第{}天，建议此类个股只卖不买。'.format(cumsum_signals.at[position_signals.index.get_level_values(level=0)[0], 'max_trend_sx_count']))
        print(position_signals_grade4[[AKA.NAME,
                                        u'力量/仓位',
                                        u'买点/偏离',
                                        u'赚钱指数',
                                        ST.CANDIDATE,
                                        u'突破天',
                                        u'启动天',
                                        u'缠周期',
                                        u'挂单价']].head(100))

    if (verbose == True):
        print(u'\n第五组：检查有没有走宝，***强烈不推荐买入，但可以持有***' if (not retreat_all) else u'第五组：大盘趋势进入下跌形态第{}天，检查有没有走宝。'.format(cumsum_signals.at[position_signals.index.get_level_values(level=0)[0], 'max_trend_sx_count']))
        print(position_dual_punch.loc[position_dual_punch[FLD.SORTINO_RATIO].gt(0.5), [AKA.NAME,
                                        u'力量/仓位',
                                        u'买点/偏离',
                                        u'赚钱指数',
                                        ST.CANDIDATE,
                                        u'突破天',
                                        u'启动天',
                                        u'缠周期',
                                        u'挂单价']].head(250))

    if (verbose == True):
        print(u'\n第六组：主升浪可以放心操作 ***主升浪板块***' if (not retreat_all) else u'第六组：大盘趋势进入下跌形态第{}天，主升浪板块可以放心操作。'.format(cumsum_signals.at[position_signals.index.get_level_values(level=0)[0], 'max_trend_sx_count']))
        mainfest_signals.sort_values(by=FLD.COMBINE_DENSITY, 
                                      inplace=True,
                                      ascending=False)

        print(mainfest_signals.loc[mainfest_signals[ST.TRIGGER_RPS].gt(0), [AKA.NAME,
                                        u'力量/仓位',
                                        u'买点/偏离',
                                        u'赚钱指数',
                                        ST.CANDIDATE,
                                        u'突破天',
                                        u'启动天',
                                        u'缠周期',
                                        u'挂单价']].head(250))

        print(u'\n第六组：主升浪可以放心操作 ***主升浪买点*** 总数：{:d}'.format(len(mainfest_signals.loc[mainfest_signals[ST.PEAK_POINT].gt(0)])))
        print(mainfest_signals.loc[mainfest_signals[ST.PEAK_POINT].gt(0), [AKA.NAME,
                                        u'力量/仓位',
                                        u'买点/偏离',
                                        u'赚钱指数',
                                        ST.PEAK_POINT,
                                        FLD.BOLL_SX_DRAWDOWN,
                                        ST.CANDIDATE,
                                        u'突破天',
                                        u'启动天',
                                        u'缠周期',
                                        u'挂单价']].head(250))

    #open = position_signals[AKA.OPEN]
    #close = position_signals[AKA.CLOSE]
    #var = close - open
    #vol= position_signals[AKA.VOLUME]
    ##position_signals.replace([np.inf, -np.inf], np.nan, inplace=True)
    ##codelist = position_signals.index.get_level_values(level=1).unique()
    ##each_day = sorted(position_signals.index.get_level_values(level=0).unique())
    ##close_prices = np.vstack([position_signals.loc[(each_day[-10:], code), [AKA.OPEN]].fillna(method='ffill') for code in codelist])
    ##open_prices = np.vstack([position_signals.loc[(each_day[-10:], code), [AKA.CLOSE]].fillna(method='ffill') for code in codelist])
    ### 每日价格变换可能承载我们所需信息dict_data[s_key].fillna(method='ffill')
    ##variation = close_prices - open_prices
    ##vol = np.vstack([position_signals.loc[(each_day[-10:], code), [AKA.VOLUME]].fillna(method='ffill') for code in codelist])

    #from sklearn import covariance, cluster

    ##x = var/var.std(0)
    #x = var * vol * 10e-9 / var.std(0)
    #print(x)
    #edge_model = covariance.GraphicalLassoCV(cv=5)   
    #edge_model.fit(x)
    #centers, labels = cluster.affinity_propagation(edge_model.covariance_)

    #n_labels = labels.max()
    #print('Centers : \n', ', '.join(np.array(dict_stock.values())[centers]))

    #for i in range(n_labels + 1):
    #    print('Cluster %i: %s' % ((i + 1), ', '.join([dict_stock[key]
    #                               for key in var.columns[labels == i]])))


    return position_signals


def show_me_number(portfolio='sharpe_onewavelet_day',):
    """
    每天早晨给一个神秘Number
    """
    print(u'近期大盘趋势预测{}：'.format(portfolio))
    cumsum_signals = market_brief(start='2019-01-01',
                                  portfolio=portfolio,
             frequency='day',
             market_type=QA.MARKET_TYPE.STOCK_CN,)

    cumsum_signals['flu/master_ratio'] = cumsum_signals.apply(lambda x:'{:.3f}/{:.3f}'.format(x.at['flu_ratio'], 
                                                                                   x.at['master_ratio']), axis=1)
    cumsum_signals['{}/total'.format(ST.CLUSTER_GROUP)] = cumsum_signals.apply(lambda x:'{}/{}'.format(int(x.at[ST.CLUSTER_GROUP_TOWARDS]), 
                                                                                        x.at['total']), axis=1)
    #cumsum_signals['{}/total'.format(ST.TRIGGER_R5)] =
    #cumsum_signals.apply(lambda x:'{}'.format(int(x.at[ST.TRIGGER_R5]),
    #                                                                                  x.at['total']),
    #                                                                                  axis=1)
    cumsum_signals['ATR_Stplin\SupTrd'] = cumsum_signals.apply(lambda x:'{}/{}'.format(x.at[FLD.ATR_Stopline], 
                                                                                   x.at[FLD.ATR_SuperTrend]), axis=1)
    cumsum_signals['FLU_POS/ML_FLU'] = cumsum_signals.apply(lambda x:'{}/{}'.format(x.at[FLD.FLU_POSITIVE], 
                                                                                  x.at[FLD.ML_FLU_TREND]), axis=1)
    cumsum_signals['trend(MT/MS)'] = cumsum_signals.apply(lambda x:'{}:{}/{}'.format(x.at['massive_flu'],
                                                                                     x.at['massive_trend'], 
                                                                                     x.at['master_stream']), axis=1)
    cumsum_signals['BOOT/DEAD/MINOR'] = cumsum_signals.apply(lambda x:'{}/{}/{}/{}'.format(x.at[FLD.BOLL_RENKO_TIMING_LAG], 
                                                                               x.at[FLD.BOOTSTRAP_GROUND_ZERO_TIMING_LAG],
                                                                               x.at[FLD.BOLL_RENKO_MINOR_TIMING_LAG], 
                                                                               x.at[FLD.BOOTSTRAP_GROUND_ZERO_MINOR_TIMING_LAG]), axis=1)
    cumsum_signals['FLU_MS/NEG'] = cumsum_signals.apply(lambda x:'{}/{}'.format(x.at[FLD.FLU_POSITIVE_MASTER], 
                                                                                   x.at[FLD.FLU_NEGATIVE_MASTER]), axis=1)

    cumsum_signals['DEA/MACD/DELTA'] = cumsum_signals.apply(lambda x:'{}/{}/{}'.format(round(x.at[FLD.DEA], 2), 
                                                                                       round(x.at[FLD.MACD], 2), 
                                                                                       round(x.at[FLD.MACD_DELTA], 2)), axis=1)
    cumsum_signals['VOL_FLOW/BONUS'] = cumsum_signals.apply(lambda x:'{:d}/{:d}'.format(x.at[ST.VOLUME_FLOW_BOOST], 
                                                                                   x.at[ST.VOLUME_FLOW_BOOST_BONUS]), axis=1)
    #cumsum_signals['MASSIVE_DIF/FLU'] = cumsum_signals.apply(lambda
    #x:'{:d}/{:d}'.format(int(x.at['missive_dif']),
    #                                                                                     int(x.at['massive_flu'])),
    #                                                                                     axis=1)
    cumsum_signals['MFT/DUAL_CROSS'] = cumsum_signals.apply(lambda x:'{:d}/{:d}'.format(int(x.at[FLD.MAXFACTOR_CROSS]), 
                                                                                   x.at[FLD.DUAL_CROSS]), axis=1)
    cumsum_signals['RENKO/OPT'] = cumsum_signals.apply(lambda x:'{:d}/{:d}'.format(int(x.at[FLD.RENKO_TREND_S]), 
                                                                                   int(x.at[FLD.RENKO_TREND_L])), axis=1)
    volume_flow_boost_jx = Timeline_Integral((cumsum_signals[ST.VOLUME_FLOW_BOOST] > 0).values)
    volume_flow_boost_sx = Timeline_Integral((cumsum_signals[ST.VOLUME_FLOW_BOOST] < 0).values)
    massive_trend_jx = Timeline_Integral((cumsum_signals['massive_flu'] > 0).values)
    massive_trend_sx = Timeline_Integral((cumsum_signals['massive_flu'] < 0).values)
    maxfactor_cross_jx = Timeline_Integral((cumsum_signals[FLD.MAXFACTOR_CROSS] > 0).values)
    maxfactor_cross_sx = Timeline_Integral((cumsum_signals[FLD.MAXFACTOR_CROSS] < 0).values)
    max_trend_jx_count = np.maximum(volume_flow_boost_jx, massive_trend_jx, maxfactor_cross_jx)
    max_trend_sx_count = np.maximum(volume_flow_boost_sx, massive_trend_sx, maxfactor_cross_sx)
    cumsum_signals['max_trend_jx_count'] = np.where(max_trend_sx_count == 0, max_trend_jx_count, 
                                                    np.where(max_trend_jx_count == 0 , 0, -1))
    cumsum_signals['max_trend_sx_count'] = np.where(max_trend_jx_count == 0, max_trend_sx_count, 
                                                    np.where(max_trend_sx_count == 0 , 0, -1))
    #print(cumsum_signals[['max_trend_jx_count', 'max_trend_sx_count']])
    pd.set_option('display.max_columns', 10)
    pd.set_option('display.max_rows', 180)
    pd.set_option('display.width', 180)  # 设置打印宽度
    print(cumsum_signals[cumsum_signals.columns.drop(['massive_trend', 
                                                      'master_stream',
                                                      'flu_ratio',
                                                      'master_ratio',
                                                      'missive_dif',
                                                      'massive_flu',
                                                      'max_trend_jx_count',
                                                      'max_trend_sx_count',
                                                      'flu/master_ratio',
                                                      'FLU_POS/ML_FLU',
                                                      FLD.BOLL_RENKO_TIMING_LAG, 
                                                      FLD.BOOTSTRAP_GROUND_ZERO_TIMING_LAG,
                                                      FLD.BOLL_RENKO_MINOR_TIMING_LAG,
                                                      FLD.BOOTSTRAP_GROUND_ZERO_MINOR_TIMING_LAG,
                                                      FLD.PEAK_OPEN_ZSCORE,
                                                      FLD.PEAK_OPEN,
                                                      FLD.PEAK_OPEN_MINOR,
                                                      ST.CLUSTER_GROUP_TOWARDS, 
                                                      'total',
                                                      'ATR_Stplin\SupTrd',
                                                      'RENKO/OPT',
                                                      FLD.DEA,
                                                      FLD.MACD,
                                                      FLD.MACD_DELTA,
                                                      FLD.ATR_SuperTrend,
                                                      FLD.ATR_Stopline,
                                                      FLD.ADXm_Trend,
                                                      FLD.Volume_HMA5,
                                                      FLD.TALIB_PATTERNS,
                                                      FLD.RENKO_TREND_S,
                                                      FLD.RENKO_TREND_L,
                                                      ST.BOOTSTRAP_I, 
                                                      ST.DEADPOOL,
                                                      FLD.ATR_Stopline,
                                                      FLD.ATR_SuperTrend,
                                                      FLD.FLU_POSITIVE_MASTER, 
                                                      FLD.FLU_NEGATIVE_MASTER,
                                                      ST.VOLUME_FLOW_BOOST,
                                                      ST.VOLUME_FLOW_BOOST_BONUS,
                                                      FLD.MAXFACTOR_CROSS,
                                                      FLD.DUAL_CROSS,
                                                      'date_stamp',
                                                      'date',
                                                      FLD.FLU_POSITIVE, 
                                                      FLD.ML_FLU_TREND])].tail(42))

    #print('\nflu_ratio.percentile(19%)——>{:.3g},
    #\nflu_ratio.percentile(38%)——>{:.3g},\nflu_ratio.percentile(50%)——>{:.3g},\nflu_ratio.percentile(61%)——>{:.3g},\nflu_ratio.percentile(81%)——>{:.3g},\n'.format(cumsum_singals['flu_ratio'].quantile(0.191),
    #      cumsum_singals['flu_ratio'].quantile(0.382),
    #      cumsum_singals['flu_ratio'].median(),
    #      cumsum_singals['flu_ratio'].quantile(0.618),
    #      cumsum_singals['flu_ratio'].quantile(0.809),))

    #print('\nML_FLU_TREND.percentile(19%)——>{:d},
    #\nML_FLU_TREND.percentile(38%)——>{:d},\nML_FLU_TREND.percentile(50%)——>{:d},\nML_FLU_TREND.percentile(61%)——>{:d},\nML_FLU_TREND.percentile(81%)——>{:d},\n'.format(int(cumsum_singals[FLD.ML_FLU_TREND].quantile(0.191)),
    #      int(cumsum_singals[FLD.ML_FLU_TREND].quantile(0.382)),
    #      int(cumsum_singals[FLD.ML_FLU_TREND].median()),
    #      int(cumsum_singals[FLD.ML_FLU_TREND].quantile(0.618)),
    #      int(cumsum_singals[FLD.ML_FLU_TREND].quantile(0.809)),))

    #massive_trend = []
    #massive_trend.append(cumsum_singals['massive_trend'].iat[-1]）
    #massive_trend[1] = ((cumsum_singals['flu_ratio'].pct_change() <
    #0).rolling(4).sum().iat[-1] < 0)
    #massive_trend[2] =
    #cumsum_singals[FLD.ML_FLU_TREND].iat[-1]<cumsum_singals[FLD.ML_FLU_TREND].quantile(0.382)
    #print(massive_trend)

    position(portfolio=portfolio,
             frequency='day',
             market_type=QA.MARKET_TYPE.STOCK_CN,
             cumsum_signals=cumsum_signals)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import mpl_finance as mpf
    import matplotlib.dates as mdates

    codelist = ['HUOBI.btcusdt']
    code = codelist[0]
    data_day = QA.QA_fetch_cryptocurrency_min_adv(code=codelist,
        start='2019-12-10',
        end='{}'.format(datetime.date.today()),
        frequence='60min')

    cumsum_signals = market_brief(portfolio='gostrike_day',
                                  start='2019-12-10',
                                  frequency='60min',
             market_type=QA.MARKET_TYPE.CRYPTOCURRENCY,)

    print('\nflu_ratio.percentile(19%)——>{:.3g}, \nflu_ratio.percentile(38%)——>{:.3g},\nflu_ratio.percentile(50%)——>{:.3g},\nflu_ratio.percentile(61%)——>{:.3g},\nflu_ratio.percentile(81%)——>{:.3g},\n'.format(cumsum_signals['flu_ratio'].quantile(0.191), 
          cumsum_signals['flu_ratio'].quantile(0.382),
          cumsum_signals['flu_ratio'].median(),
          cumsum_signals['flu_ratio'].quantile(0.618),
          cumsum_signals['flu_ratio'].quantile(0.809),))

    print('\nML_FLU_TREND.percentile(19%)——>{:d}, \nML_FLU_TREND.percentile(38%)——>{:d},\nML_FLU_TREND.percentile(50%)——>{:d},\nML_FLU_TREND.percentile(61%)——>{:d},\nML_FLU_TREND.percentile(81%)——>{:d},\n'.format(int(cumsum_signals[FLD.ML_FLU_TREND].quantile(0.191)), 
          int(cumsum_signals[FLD.ML_FLU_TREND].quantile(0.382)),
          int(cumsum_signals[FLD.ML_FLU_TREND].median()),
          int(cumsum_signals[FLD.ML_FLU_TREND].quantile(0.618)),
          int(cumsum_signals[FLD.ML_FLU_TREND].quantile(0.809)),))

    cumsum_signals['FLU_POS/ML_FLU'] = cumsum_signals.apply(lambda x:'{}/{}'.format(x.at[FLD.FLU_POSITIVE], 
                                                                                    x.at[FLD.ML_FLU_TREND]), axis=1)
    cumsum_signals['trend(MT/MS)'] = cumsum_signals.apply(lambda x:'{}/{}'.format(x.at['massive_trend'], 
                                                                                  x.at['master_stream']), axis=1)
    cumsum_signals['BOOT/DEAD'] = cumsum_signals.apply(lambda x:'{}/{}'.format(x.at[ST.BOOTSTRAP_I], 
                                                                               x.at[ST.DEADPOOL]), axis=1)
    cumsum_signals['FLU_MS/NEG'] = cumsum_signals.apply(lambda x:'{}/{}'.format(x.at[FLD.FLU_POSITIVE_MASTER], 
                                                                                x.at[FLD.FLU_NEGATIVE_MASTER]), axis=1)
    cumsum_signals['DEA/MACD/DELTA'] = cumsum_signals.apply(lambda x:'{}/{}/{}'.format(x.at[FLD.DEA], 
                                                                                       x.at[FLD.MACD], 
                                                                                       x.at[FLD.MACD_DELTA]), axis=1)
    #cumsum_singals =
    #cumsum_singals[cumsum_singals.columns.drop(['massive_trend',
    #                                                  'master_stream',
    #                                                  ST.BOOTSTRAP_I,
    #                                                  ST.DEADPOOL,
    #                                                  FLD.FLU_POSITIVE_MASTER,
    #                                                  FLD.FLU_NEGATIVE_MASTER,
    #                                                  'date_stamp',
    #                                                  'date',
    #                                                  FLD.FLU_POSITIVE,
    #                                                  FLD.ML_FLU_TREND])]

    #print(cumsum_singals.loc[cumsum_singals.index.intersection(pd.date_range('2019-01-01',
    #                                                                         periods=50,
    #                                                                         freq='1D')),
    #                         cumsum_singals.columns.drop(['massive_trend',
    #                                                  'master_stream',
    #                                                  ST.BOOTSTRAP_I,
    #                                                  ST.DEADPOOL,
    #                                                  FLD.FLU_POSITIVE_MASTER,
    #                                                  FLD.FLU_NEGATIVE_MASTER,
    #                                                  'time_stamp',
    #                                                  'datetime',
    #                                                  FLD.FLU_POSITIVE,
    #                                                  FLD.ML_FLU_TREND])])

    #print(cumsum_singals.loc[cumsum_singals.index.intersection(pd.date_range('2019-11-01',
    #                                                                         periods=50,
    #                                                                         freq='1D')),
    #                         cumsum_singals.columns.drop(['massive_trend',
    #                                                  'master_stream',
    #                                                  ST.BOOTSTRAP_I,
    #                                                  ST.DEADPOOL,
    #                                                  FLD.FLU_POSITIVE_MASTER,
    #                                                  FLD.FLU_NEGATIVE_MASTER,
    #                                                  'time_stamp',
    #                                                  'datetime',
    #                                                  FLD.FLU_POSITIVE,
    #                                                  FLD.ML_FLU_TREND])])
    #print(cumsum_singals.loc[cumsum_singals.index.intersection(pd.date_range('2019-11-20',
    #                                                                         periods=50,
    #                                                                         freq='1D')),
    #                         cumsum_singals.columns.drop(['massive_trend',
    #                                                  'master_stream',
    #                                                  ST.BOOTSTRAP_I,
    #                                                  ST.DEADPOOL,
    #                                                  FLD.FLU_POSITIVE_MASTER,
    #                                                  FLD.FLU_NEGATIVE_MASTER,
    #                                                  'time_stamp',
    #                                                  'datetime',
    #                                                  FLD.FLU_POSITIVE,
    #                                                  FLD.ML_FLU_TREND])])
    #print(cumsum_singals.loc[cumsum_singals.index.intersection(pd.date_range('2020-01-20',
    #                                                                         periods=50,
    #                                                                         freq='1D')),
    #                         cumsum_singals.columns.drop(['massive_trend',
    #                                                  'master_stream',
    #                                                  ST.BOOTSTRAP_I,
    #                                                  ST.DEADPOOL,
    #                                                  FLD.FLU_POSITIVE_MASTER,
    #                                                  FLD.FLU_NEGATIVE_MASTER,
    #                                                  'date_stamp',
    #                                                  'date',
    #                                                  FLD.FLU_POSITIVE,
    #                                                  FLD.ML_FLU_TREND])])

    #print(cumsum_singals.tail(20).loc[:,
    #cumsum_singals.columns.drop(['massive_trend',
    #                                                  'master_stream',
    #                                                  ST.BOOTSTRAP_I,
    #                                                  ST.DEADPOOL,
    #                                                  FLD.FLU_POSITIVE_MASTER,
    #                                                  FLD.FLU_NEGATIVE_MASTER,
    #                                                  'date_stamp',
    #                                                  'date',
    #                                                  FLD.FLU_POSITIVE,
    #                                                  FLD.ML_FLU_TREND])])

    # 暗色主题
    plt.style.use('Solarize_Light2')

    # 正常显示中文字体
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    fig = plt.figure(figsize = (22,9))
    codename = 'BTC'
    code = QA.MARKET_TYPE.CRYPTOCURRENCY
    fig.suptitle(u'阿财的 {:s}（{:s}）机器学习大趋势判断'.format(codename, code), fontsize=16)
    ax1 = plt.subplot2grid((4,3),(0,0), rowspan=3, colspan=3)
    ax2 = plt.subplot2grid((4,3),(3,0), rowspan=1, colspan=3, sharex=ax1)
    ax3 = ax1.twinx()
    mpf.candlestick2_ochl(ax1,
                          data_day.data.open,
                          data_day.data.close,
                          data_day.data.high,
                          data_day.data.low,
                          width=0.6, colorup='r', colordown='green',
                          alpha=0.3)

    DATETIME_LABEL = cumsum_signals.index.to_series().apply(lambda x: x.strftime("%Y-%m-%d %H:%M")[2:16])
    plt.plot(DATETIME_LABEL, 
             cumsum_signals['flu_ratio'], lw=0.75, c='limegreen', alpha=0.6)
    #plt.plot(DATETIME_LABEL,
    #         np.where(cumsum_singals['massive_trend'] >= 0,
    #                  cumsum_singals['flu_ratio'], np.nan),
    #         lw=0.75, c='crimson', alpha=0.6)
    plt.plot(DATETIME_LABEL,
             cumsum_signals['master_ratio'],
             lw=1, c='crimson', alpha=0.6)
    #plt.plot(DATETIME_LABEL, cumsum_singals['master_stream'] *
    #cumsum_singals['master_ratio'], lw=0.75, c='deepskyblue', alpha=0.6)
    #plt.plot(DATETIME_LABEL, cumsum_singals['massive_trend'] *
    #cumsum_singals['flu_ratio'], lw=0.75, c='crimson', alpha=0.6)
    #plt.plot(DATETIME_LABEL,
    #         np.where((cumsum_singals['master_stream'] >= 0),
    #                  cumsum_singals['flu_ratio'] +
    #                  cumsum_singals['master_ratio'], np.nan),
    #         lw=1, c='crimson', alpha=0.6)

    ax1.set_xticks(range(0, len(DATETIME_LABEL), 
                         round(len(DATETIME_LABEL) / 12)))
    ax1.set_xticklabels(DATETIME_LABEL[::round(len(DATETIME_LABEL) / 12)])

    #ax1.plot(DATETIME_LABEL, regtree_flu['regtree_price'], linewidth=0.75,
    #color='deepskyblue')
    #ax1.plot(DATETIME_LABEL,
    #         np.where((regtree_flu['tree_direction_rs'] == True),
    #                  regtree_flu['regtree_price'],
    #                  np.nan),
    #         linewidth=0.75, color='crimson')

    plt.show()
