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
import numpy as np
import pandas as pd

try:
    import QUANTAXIS as QA
except:
    print('PLEASE run "pip install QUANTAXIS" before use these modules')
    pass

from GolemQ.analysis.timeseries import *
from GolemQ.fetch.Portfolio_signals import (
    GQSignal_fetch_position_singal_day, 
    GQSignal_fetch_position_singal_min,
    GQSignal_fetch_flu_cunsum_day,
    GQSignal_fetch_flu_cunsum_min,
    GQSignal_fetch_block_cunsum_day,
)
from GolemQ.utils.parameter import (
    AKA, 
    INDICATOR_FIELD as FLD,
    TREND_STATUS as ST,
    FEATURES as FTR,
)

def predict_cumsum_trend(cumsum_singals):    
    """
    推断群体趋势走势
    """
    cumsum_singals['flu_ratio'] = cumsum_singals[FLD.FLU_POSITIVE] / cumsum_singals['total']
    cumsum_singals['master_ratio'] = cumsum_singals[FLD.FLU_POSITIVE_MASTER] / cumsum_singals['total']
    volume_flow_boost_dif = cumsum_singals[ST.VOLUME_FLOW_BOOST].diff() + cumsum_singals[ST.VOLUME_FLOW_BOOST_BONUS].diff()
    atr_supertrend_dif = cumsum_singals[FLD.ATR_Stopline].diff() + cumsum_singals[FLD.ATR_SuperTrend].diff()

    cumsum_diff_c00 = np.where(cumsum_singals['flu_ratio'].diff() > 0, 1, 
                               np.where(cumsum_singals['flu_ratio'].diff() < 0, -1, 0))
    cumsum_diff_c01 = np.where(cumsum_singals['master_ratio'].diff() > 0, 1, 
                               np.where(cumsum_singals['master_ratio'].diff() < 0, -1, 0))
    cumsum_diff_c02 = np.where(cumsum_singals[FLD.MAXFACTOR_CROSS].diff() < 0, 1, 
                               np.where(cumsum_singals[FLD.MAXFACTOR_CROSS].diff() > 0, -1, 0))
    cumsum_diff_c03 = np.where(cumsum_singals[FLD.DUAL_CROSS].diff() < 0, 1, 
                               np.where(cumsum_singals[FLD.DUAL_CROSS].diff() > 0, -1, 0))
    cumsum_diff_c04 = np.where(cumsum_singals[ST.VOLUME_FLOW_BOOST].diff() > 0, 1, 
                               np.where(cumsum_singals[ST.VOLUME_FLOW_BOOST].diff() < 0, -1, 0))
    cumsum_diff_c05 = np.where(cumsum_singals[ST.VOLUME_FLOW_BOOST_BONUS].diff() > 0, 1, 
                               np.where(cumsum_singals[ST.VOLUME_FLOW_BOOST_BONUS].diff() < 0, -1, 0))
    cumsum_diff_c06 = np.where(cumsum_singals[ST.BOOTSTRAP_I].diff() > 0, 1, 
                               np.where(cumsum_singals[ST.BOOTSTRAP_I].diff() < 0, -1, 0))
    cumsum_diff_c07 = np.where(cumsum_singals[ST.DEADPOOL].diff() < 0, 1, 
                               np.where(cumsum_singals[ST.DEADPOOL].diff() > 0, -1, 0))
    cumsum_diff_c08 = np.where(cumsum_singals[FLD.TALIB_PATTERNS].diff() > 0, 1, 
                               np.where(cumsum_singals[FLD.TALIB_PATTERNS].diff() < 0, -1, 0))
    cumsum_diff_c09 = np.where(cumsum_singals[FLD.FLU_POSITIVE].diff() > 0, 1, 
                               np.where(cumsum_singals[FLD.FLU_POSITIVE].diff() < 0, -1, 0))
    cumsum_diff_c10 = np.where(cumsum_singals[FLD.ML_FLU_TREND].diff() > 0, 1, 
                               np.where(cumsum_singals[FLD.ML_FLU_TREND].diff() < 0, -1, 0))
    cumsum_diff_c11 = np.where(cumsum_singals[FLD.ADXm_Trend].diff() > 0, 1, 
                               np.where(cumsum_singals[FLD.ADXm_Trend].diff() < 0, -1, 0))
    cumsum_diff_c12 = np.where(cumsum_singals[FLD.Volume_HMA5].diff() > 0, 1, 
                               np.where(cumsum_singals[FLD.Volume_HMA5].diff() < 0, -1, 0))

    cumsum_singals['missive_dif'] = np.c_[cumsum_diff_c00 + \
                       cumsum_diff_c01 + \
                       cumsum_diff_c02 + \
                       cumsum_diff_c03 + \
                       cumsum_diff_c04 + \
                       cumsum_diff_c05 + \
                       cumsum_diff_c06 + \
                       cumsum_diff_c07 + \
                       cumsum_diff_c08 + \
                       cumsum_diff_c09 + \
                       cumsum_diff_c10 + \
                       cumsum_diff_c11 + \
                       cumsum_diff_c12]

    cumsum_singals['massive_trend'] = np.where((cumsum_singals['flu_ratio'] < max(0.382, cumsum_singals['flu_ratio'].quantile(0.382))) & \
                                               (atr_supertrend_dif < 0), -1,
                                               np.where(volume_flow_boost_dif / cumsum_singals['total'] < -0.168, -2,
                                                        np.where((cumsum_singals[ST.VOLUME_FLOW_BOOST].diff() < 0) & \
                                                                 (cumsum_singals[ST.VOLUME_FLOW_BOOST] < 0) & \
                                                                 ((cumsum_singals[FLD.MAXFACTOR_CROSS] < 0) | \
                                                                 (cumsum_singals['missive_dif'].shift(1) < 0) | \
                                                                 (cumsum_singals[FLD.MAXFACTOR_CROSS].diff() / cumsum_singals['total'] < -0.168)), -3,
                                                        np.where(volume_flow_boost_dif / cumsum_singals['total'] > 0.309, 1,
                                                                 np.where((volume_flow_boost_dif.rolling(2).sum() / cumsum_singals['total'] > 0.309) & \
                                                                          (volume_flow_boost_dif / cumsum_singals['total'] > 0.168), 2,
                                                                          np.where(((cumsum_singals[ST.VOLUME_FLOW_BOOST] + cumsum_singals[ST.VOLUME_FLOW_BOOST_BONUS]) / cumsum_singals['total'] > 0.0618) & \
                                                                                   (cumsum_singals[ST.VOLUME_FLOW_BOOST] > 0) & \
                                                                                   (volume_flow_boost_dif > 0), 3,
                                               np.where((cumsum_singals['flu_ratio'] < max(0.512, cumsum_singals['flu_ratio'].quantile(0.618))) & \
                                                   (cumsum_singals['flu_ratio'].rolling(4).mean() < max(0.512, cumsum_singals['flu_ratio'].quantile(0.618))) & \
                                                   (cumsum_singals[FLD.ML_FLU_TREND].rolling(4).mean() < 0) & \
                                                   (cumsum_singals[FLD.ML_FLU_TREND] < cumsum_singals[FLD.ML_FLU_TREND].median()) & \
                                                   ((cumsum_singals[FLD.MAXFACTOR_CROSS] < 0) | \
                                                   (cumsum_singals['missive_dif'].shift(1) < 0) | \
                                                   (cumsum_singals[FLD.MAXFACTOR_CROSS].diff() / cumsum_singals['total'] < -0.168)), -4,
                                               np.where((cumsum_singals['flu_ratio'] < cumsum_singals['flu_ratio'].quantile(0.382)) & \
                                                   ((cumsum_singals[FLD.MAXFACTOR_CROSS] < 0) | \
                                                   (cumsum_singals['missive_dif'].shift(1) < 0) | \
                                                   (cumsum_singals[FLD.MAXFACTOR_CROSS].diff() / cumsum_singals['total'] < -0.168)), -5,
                                                        np.where((cumsum_singals['flu_ratio'] > cumsum_singals['flu_ratio'].quantile(0.809)) & \
                                                            (cumsum_singals[FLD.ML_FLU_TREND] > cumsum_singals[FLD.ML_FLU_TREND].median()), 4,
                                               np.where((cumsum_singals[FLD.FLU_POSITIVE] <= cumsum_singals[FLD.FLU_POSITIVE].shift(1)) & \
                                                        (cumsum_singals[FLD.ML_FLU_TREND] < cumsum_singals[FLD.ML_FLU_TREND].shift(1)) & \
                                                   ((cumsum_singals[FLD.MAXFACTOR_CROSS] < 0) | \
                                                   (cumsum_singals['missive_dif'].shift(1) < 0) | \
                                                   (cumsum_singals[FLD.MAXFACTOR_CROSS].diff() / cumsum_singals['total'] < -0.168)), -6,
                                               np.where((cumsum_singals[FLD.FLU_POSITIVE].pct_change() < 0) & \
                                                        (cumsum_singals[FLD.ML_FLU_TREND].pct_change() < 0), -7,
                                                        np.where((cumsum_singals[FLD.FLU_POSITIVE].pct_change() > 0) & \
                                                                 (cumsum_singals['missive_dif'] > 0) & \
                                                                 (cumsum_singals[ST.VOLUME_FLOW_BOOST] > 0), 5, 
                                                                 np.where((cumsum_singals[FLD.ML_FLU_TREND].pct_change() > 0) & \
                                                                          (cumsum_singals['missive_dif'] > 0) & \
                                                                          (cumsum_singals[ST.VOLUME_FLOW_BOOST] > 0), 6,
                                                                          np.where((cumsum_singals[ST.VOLUME_FLOW_BOOST] < 0), -8, 
                                                                                   np.where(cumsum_singals['missive_dif'] > 0, 9, 
                                                                                    np.where((cumsum_singals['missive_dif'] < 0) & \
                                                   ((cumsum_singals[FLD.MAXFACTOR_CROSS] < 0) | \
                                                   (cumsum_singals['missive_dif'].shift(1) < 0) | \
                                                   (cumsum_singals[FLD.MAXFACTOR_CROSS].diff() / cumsum_singals['total'] < -0.168)), -9,
                                                                                             np.where((cumsum_singals[FLD.MAXFACTOR_CROSS].diff() < 0) & \
                                                                                                     ((cumsum_singals['missive_dif'].shift(1) < 0) | \
                                                                                                     (cumsum_singals['missive_dif'] < 0)), -11, 0)))))))))))))))))

    cumsum_singals['master_stream'] = np.where((cumsum_singals['master_ratio'] < max(0.382, cumsum_singals['master_ratio'].quantile(0.382))) & \
                                               (atr_supertrend_dif < 0), -1,
                                               np.where(volume_flow_boost_dif / cumsum_singals['total'] < -0.168, -2,
                                                        np.where((cumsum_singals[ST.VOLUME_FLOW_BOOST].diff() < 0) & \
                                                                 (cumsum_singals[ST.VOLUME_FLOW_BOOST] < 0) & \
                                                                 ((cumsum_singals[FLD.MAXFACTOR_CROSS] < 0) | \
                                                                 (cumsum_singals['missive_dif'].shift(1) < 0) | \
                                                                 (cumsum_singals[FLD.MAXFACTOR_CROSS].diff() / cumsum_singals['total'] < -0.168)), -3,
                                                        np.where(volume_flow_boost_dif / cumsum_singals['total'] > 0.309, 1,
                                                                 np.where((volume_flow_boost_dif.rolling(2).sum() / cumsum_singals['total'] > 0.309) & \
                                                                          (volume_flow_boost_dif / cumsum_singals['total'] > 0.168), 2,
                                                                          np.where(((cumsum_singals[ST.VOLUME_FLOW_BOOST] + cumsum_singals[ST.VOLUME_FLOW_BOOST_BONUS]) / cumsum_singals['total'] > 0.0618) & \
                                                                                   (cumsum_singals[ST.VOLUME_FLOW_BOOST] > 0) & \
                                                                                   (volume_flow_boost_dif > 0), 3,
                                               np.where((cumsum_singals['master_ratio'] < max(0.512, cumsum_singals['master_ratio'].quantile(0.618))) & \
                                                   (cumsum_singals['master_ratio'].rolling(4).mean() < max(0.512, cumsum_singals['master_ratio'].quantile(0.618))) & \
                                                   (cumsum_singals['massive_trend'].rolling(4).mean() < 0) & \
                                                   (cumsum_singals['massive_trend'] < cumsum_singals['massive_trend'].median()) & \
                                                   ((cumsum_singals[FLD.MAXFACTOR_CROSS] < 0) | \
                                                   (cumsum_singals['missive_dif'].shift(1) < 0) | \
                                                   (cumsum_singals[FLD.MAXFACTOR_CROSS].diff() / cumsum_singals['total'] < -0.168)), -4,
                                               np.where((cumsum_singals['master_ratio'] < cumsum_singals['master_ratio'].quantile(0.382)) & \
                                                        ((cumsum_singals[FLD.MAXFACTOR_CROSS] < 0) | \
                                                        (cumsum_singals['missive_dif'].shift(1) < 0) | \
                                                        (cumsum_singals[FLD.MAXFACTOR_CROSS].diff() / cumsum_singals['total'] < -0.168)), -5,
                                                        np.where((cumsum_singals['master_ratio'] > cumsum_singals['master_ratio'].quantile(0.809)) & \
                                                            (cumsum_singals['massive_trend'] > cumsum_singals['massive_trend'].median()), 4,
                                               np.where((cumsum_singals[FLD.FLU_POSITIVE_MASTER] <= cumsum_singals[FLD.FLU_POSITIVE_MASTER].shift(1)) & \
                                                        (cumsum_singals['massive_trend'] < cumsum_singals['massive_trend'].shift(1)) & \
                                                        ((cumsum_singals[FLD.MAXFACTOR_CROSS] < 0) | \
                                                        (cumsum_singals['missive_dif'].shift(1) < 0) | \
                                                        (cumsum_singals[FLD.MAXFACTOR_CROSS].diff() / cumsum_singals['total'] < -0.168)), -6,
                                               np.where((cumsum_singals[FLD.FLU_POSITIVE_MASTER].pct_change() < 0) & \
                                                        (cumsum_singals['massive_trend'].pct_change() < 0), -7,
                                                        np.where((cumsum_singals[FLD.FLU_POSITIVE_MASTER].pct_change() > 0) & \
                                                                 (cumsum_singals['missive_dif'] > 0) & \
                                                                 (cumsum_singals[ST.VOLUME_FLOW_BOOST] > 0), 5, 
                                                                 np.where((cumsum_singals['massive_trend'].pct_change() > 0) & \
                                                                          (cumsum_singals['missive_dif'] > 0) & \
                                                                          (cumsum_singals[ST.VOLUME_FLOW_BOOST] > 0), 6, 
                                                                          np.where((cumsum_singals[ST.VOLUME_FLOW_BOOST] < 0), -8, 
                                                                                   np.where(cumsum_singals['missive_dif'] > 0, 9, 
                                                                                    np.where((cumsum_singals['missive_dif'] < 0) & \
                                                                                    ((cumsum_singals[FLD.MAXFACTOR_CROSS] < 0) | \
                                                                                    (cumsum_singals['missive_dif'].shift(1) < 0) | \
                                                                                    (cumsum_singals[FLD.MAXFACTOR_CROSS].diff() / cumsum_singals['total'] < -0.168)), -9, 
                                                                                             np.where((cumsum_singals[FLD.MAXFACTOR_CROSS].diff() < 0) & \
                                                                                                     ((cumsum_singals['missive_dif'].shift(1) < 0) | \
                                                                                                     (cumsum_singals['missive_dif'] < 0)), -11, 0)))))))))))))))))

    # 在特殊情况下需要快速技术指标 “全体表决”
    cumsum_singals['massive_flu'] = np.where((cumsum_singals['master_stream'] / cumsum_singals['massive_trend']) < 0.0002, 
                                             np.where(cumsum_singals['missive_dif'] > 0, 10,
                                                      np.where(cumsum_singals['missive_dif'] < 0, 
                                                      np.where((cumsum_singals[FLD.MAXFACTOR_CROSS] > 0) & \
                                                               (cumsum_singals[FLD.MAXFACTOR_CROSS].shift(1) > 0), 12, -10), 0)), cumsum_singals['massive_trend'])
    cumsum_singals['massive_flu'] = np.where((cumsum_singals['massive_flu'] == -4) & \
                                             (cumsum_singals[FLD.MAXFACTOR_CROSS] > 0) & \
                                             (cumsum_singals[FLD.MAXFACTOR_CROSS].shift(1) > 0) & \
                                             (cumsum_singals['massive_trend'].shift(2) > 0) & \
                                             ((cumsum_singals[FLD.MAXFACTOR_CROSS].diff() > 0) | \
                                             (cumsum_singals['missive_dif'] > 0)), 13, cumsum_singals['massive_flu'])
    cumsum_singals['massive_flu'] = np.where((cumsum_singals['massive_flu'].shift(1) < 0) & \
                                             (cumsum_singals['massive_flu'].shift(2) > 0) & \
                                             (cumsum_singals[FLD.MAXFACTOR_CROSS] < 0) & \
                                             (cumsum_singals[FLD.MAXFACTOR_CROSS].shift(1) < 0) & \
                                             (cumsum_singals[FLD.DUAL_CROSS].shift(1) > cumsum_singals[FLD.DUAL_CROSS]), -99,
                                             cumsum_singals['massive_flu'])

    cumsum_singals['massive_flu'] = np.where((cumsum_singals['massive_flu'] <= 0) & \
                                             ((cumsum_singals['massive_flu'].shift(1) > 0) | \
                                             ((cumsum_singals[FLD.MAXFACTOR_CROSS] / cumsum_singals['total'] > 0.809) & \
                                             (cumsum_singals[FLD.MAXFACTOR_CROSS].shift(1) / cumsum_singals['total'] > 0.809)) | \
                                             ((cumsum_singals['massive_flu'].shift(2) > 0) & \
                                             (cumsum_singals[FLD.MAXFACTOR_CROSS] / cumsum_singals['total'] > 0.809))) & \
                                             (cumsum_singals[FLD.DEA] > 0) & \
                                             (cumsum_singals[FLD.MACD_DELTA] > 0) & \
                                             (cumsum_singals[FLD.MAXFACTOR_CROSS] > 0) & \
                                             ((cumsum_singals[FLD.MAXFACTOR_CROSS] / cumsum_singals['total'] > 0.618) | \
                                             (cumsum_singals[FLD.BOOTSTRAP_GROUND_ZERO_MINOR_TIMING_LAG] > 0)), 11,
                                             cumsum_singals['massive_flu'])

    cumsum_singals['massive_flu'] = np.where((cumsum_singals['massive_flu'] <= 0) & \
                                             ((cumsum_singals['massive_flu'].shift(1) > 0) | \
                                             (cumsum_singals[FLD.MACD] > 0)) & \
                                             (cumsum_singals['massive_flu'].shift(2) > 0) & \
                                             (cumsum_singals[FLD.MACD_DELTA].shift(1) > 0) & \
                                             (cumsum_singals[FLD.MACD_DELTA].shift(2) > 0) & \
                                             (cumsum_singals[FLD.DEA] > 0) & \
                                             (cumsum_singals[FLD.MACD_DELTA] > 0) & \
                                             (cumsum_singals[FLD.MAXFACTOR_CROSS] > 0) & \
                                             (cumsum_singals[FLD.DUAL_CROSS] > 0) & \
                                             (cumsum_singals[ST.CLUSTER_GROUP_TOWARDS] < 0), 12,
                                             cumsum_singals['massive_flu'])

    cumsum_singals['massive_flu'] = np.where((cumsum_singals['massive_flu'] >= 0) & \
                                             (cumsum_singals[FLD.MAXFACTOR_CROSS] < 0) & \
                                             (cumsum_singals[FLD.DUAL_CROSS] < 0) & \
                                             ((cumsum_singals[FLD.MACD_DELTA] < 0) | \
                                             ((cumsum_singals[FLD.MACD_DELTA].shift(1) < 0) & \
                                             (cumsum_singals[FLD.MAXFACTOR_CROSS].shift(1) < 0))) & \
                                             (cumsum_singals[FLD.MACD] < 0) & \
                                             (((cumsum_singals[FLD.BOLL_RENKO_TIMING_LAG] < 0) & \
                                             (cumsum_singals[FLD.BOOTSTRAP_GROUND_ZERO_TIMING_LAG] < 0)) | \
                                             ((cumsum_singals[FLD.BOLL_RENKO_MINOR_TIMING_LAG] < 0) & \
                                             (cumsum_singals[FLD.BOOTSTRAP_GROUND_ZERO_MINOR_TIMING_LAG] < 0))), -98,
                                             cumsum_singals['massive_flu'])

    cumsum_singals['massive_flu'] = np.where((cumsum_singals['massive_flu'] >= 0) & \
                                             (cumsum_singals[FLD.MAXFACTOR_CROSS] < 0) & \
                                             ((cumsum_singals[FLD.MACD_DELTA] < 0) | \
                                             ((cumsum_singals[FLD.MACD_DELTA].shift(1) < 0) & \
                                             (cumsum_singals[FLD.MAXFACTOR_CROSS].shift(1) < 0))) & \
                                             ((cumsum_singals[FLD.DUAL_CROSS] < 0) | \
                                             (cumsum_singals[FLD.MACD] < 0)) & \
                                             (cumsum_singals[FLD.BOLL_RENKO_TIMING_LAG] < 0) & \
                                             (cumsum_singals[FLD.BOOTSTRAP_GROUND_ZERO_TIMING_LAG] < 0) & \
                                             (cumsum_singals[FLD.BOLL_RENKO_MINOR_TIMING_LAG] < 0) & \
                                             (cumsum_singals[FLD.BOOTSTRAP_GROUND_ZERO_MINOR_TIMING_LAG] < 0), -97,
                                             cumsum_singals['massive_flu'])

    peak_open_zscore_21 = rolling_pctrank(cumsum_singals[FLD.PEAK_OPEN].values, w=21)
    cumsum_singals[FLD.PEAK_OPEN_ZSCORE] = peak_open_zscore_21
    #cumsum_singals[FLD.PEAK_OPEN_MINOR] = np.where((cumsum_singals['massive_flu'] <= 0) & \
    #                                         ((cumsum_singals[FLD.PEAK_OPEN].shift(1) + \
    #                                         cumsum_singals[FLD.PEAK_OPEN].shift(2) + \
    #                                         cumsum_singals[FLD.PEAK_OPEN].shift(3) + \
    #                                         cumsum_singals[FLD.PEAK_OPEN].shift(4)) > 0) & \
    #                                         (cumsum_singals[FLD.PEAK_OPEN] < -0.0168) & \
    #                                         (cumsum_singals[FLD.DEA] > 0) & \
    #                                         (cumsum_singals[FLD.PEAK_OPEN_ZSCORE] < 0.1236) & \
    #                                         ((cumsum_singals[ST.CLUSTER_GROUP_TOWARDS] / cumsum_singals['total']) > 0.168), 13, 0)

    # 强势抢反弹
    cumsum_singals['massive_flu'] = np.where((cumsum_singals['massive_flu'] <= 0) & \
                                             ((cumsum_singals[FLD.PEAK_OPEN].shift(1) + \
                                             cumsum_singals[FLD.PEAK_OPEN].shift(2) + \
                                             cumsum_singals[FLD.PEAK_OPEN].shift(3) + \
                                             cumsum_singals[FLD.PEAK_OPEN].shift(4)) > 0) & \
                                             (cumsum_singals[FLD.PEAK_OPEN] < -0.0168) & \
                                             (cumsum_singals[FLD.DEA] > 0) & \
                                             (cumsum_singals[FLD.PEAK_OPEN_ZSCORE] < 0.1236) & \
                                             ((cumsum_singals[ST.CLUSTER_GROUP_TOWARDS] / cumsum_singals['total']) > 0.168), 13,
                                             cumsum_singals['massive_flu'])

    return cumsum_singals


def market_brief(start='2008-01-01',
                 portfolio='myportfolio',
                 frequency='day',
                 market_type=QA.MARKET_TYPE.STOCK_CN,):
    """
    大盘趋势预测
    """
    if (frequency == 'day'):
        cumsum_singals = GQSignal_fetch_flu_cunsum_day(start=start,
            end='{}'.format(datetime.date.today() + datetime.timedelta(days=1)),
            market_type=market_type,
            portfolio=portfolio,
            format='pd')
    else:
        cumsum_singals = GQSignal_fetch_flu_cunsum_min(start=start,
            end='{}'.format(datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))) + datetime.timedelta(minutes=1)),
            frequency=frequency,
            market_type=market_type,
            portfolio=portfolio,
            format='pd')

    return predict_cumsum_trend(cumsum_singals)


def block_brief(blockname=None,
                codelist=None,
                start='2008-01-01',
                portfolio='myportfolio',
                frequency='day',
                market_type=QA.MARKET_TYPE.STOCK_CN,
                verbose=False):
    """
    板块趋势预测
    """
    if (codelist is None):
        blockname = list(set(blockname))
        codelist = QA.QA_fetch_stock_block_adv().get_block(blockname).code

    if (verbose):
        print('批量评估板块成分股：{} Total:{}'.format(blockname, 
                                                   len(codelist)), codelist)

    if (frequency == 'day'):
        cumsum_singals = GQSignal_fetch_block_cunsum_day(codelist, start=start,
            end='{}'.format(datetime.date.today() + datetime.timedelta(days=1)),
            market_type=market_type,
            portfolio=portfolio,
            format='pd')
    else:
        cumsum_singals = GQSignal_fetch_flu_cunsum_min(codelist, start=start,
            end='{}'.format(datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))) + datetime.timedelta(minutes=1)),
            frequency=frequency,
            market_type=market_type,
            portfolio=portfolio,
            format='pd')

    return predict_cumsum_trend(cumsum_singals)

