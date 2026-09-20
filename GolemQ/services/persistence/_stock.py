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
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
Stock persistence check for hourly/daily feature completeness.

Validates data integrity across multiple timeframes (m9t, stage, hourly,
15min, daily) for individual stock reality features.
"""

import numpy as np
import pandas as pd
from datetime import (
    datetime as dt,
    timezone, timedelta
)
import datetime
import traceback
import random
from GolemQ.core.constants import MARKET_TYPE, FREQUENCE
from GolemQ.markets.StockCN.date_utils import GQ_util_timestamp_to_str

from GolemQ.core.settings import DATABASE as DATABASE_GolemQ
from GolemQ.core.constants import (
    AKA,
    FEATURES as FTR,
    FIELD as FLD,
)
from GolemQ.features.empirical import load_massive_reviews
from GolemQ.features.reviews import attach_reality_features
from GolemQ.fetch.kline import (
    get_kline_price_min,
    get_kline_price_v3,
)
from GolemQ.core.base import GQ_util_get_last_day
from GolemQ.services.align import symbol_checkpoint_log
from GolemQ.services.persistence._schema import (
    reality_columns_of_persistence,
    calc_masked_tail_missing_index,
)


def dataloader_persistence_check(symbol=None,
                                 offset: str = '',
                                 market_type=MARKET_TYPE.STOCK_CN,
                                 verbose: bool = False,
                                 peek_column: list = [FLD.RENKO_TREND_S_TIMING_LAG],
                                 debug: bool = False, ):
    """
    持久化指标数据加载和缺漏检查
    """
    if (verbose):
        print(u'dataloader_persistence_check start')
    persistence_ratio = {'symbol': symbol,
                         'halt': False,
                         'daily': {'total': 0, 'persistence': 0, 'ratio': 0.0, 'masked': [], },
                         'hourly': {'total': 0, 'persistence': 0, 'ratio': 0.0, 'masked': [], },
                         '15min': {'total': 0, 'persistence': 0, 'ratio': 0.0, 'masked': [], },
                         'stage': {'total': 0, 'persistence': 0, 'ratio': 0.0, 'masked': [], },
                         'm9t': {'total': 0, 'persistence': 0, 'ratio': 0.0, 'masked': [], },
                         }
    persistence_features = pd.DataFrame(columns=reality_columns_of_persistence())
    each_day = []
    kline_hour_baseline = None
    if (len(offset) == 0):
        data_baseline, codename = get_kline_price_v3(symbol,
                                                     verbose=verbose,
                                                     realtime=True)
    else:
        start = '{}'.format(pd.to_datetime(offset).tz_localize('Asia/Shanghai') -
                            datetime.timedelta(days=980))
        end = '{}'.format(pd.to_datetime(offset).tz_localize('Asia/Shanghai') +
                          timedelta(hours=16))
        data_baseline, codename = get_kline_price_v3(symbol,
                                                     start=start,
                                                     end=end,
                                                     verbose=verbose,
                                                     realtime=False)

    if (data_baseline is None):
        log_msg = 'Code:{}, daily kline is None!'.format(symbol, )
        code = symbol[0] if (isinstance(symbol, list)) else symbol
        try:
            symbol_checkpoint_log(logs=log_msg,
                                  symbol=code,
                                  length=None,
                                  FrozenExpired=pd.to_datetime(GQ_util_get_last_day()) + timedelta(days=63) if code.startswith('83') or
                                  code.startswith('87') else pd.to_datetime(GQ_util_get_last_day()) + timedelta(days=9),
                                  catalog=AKA.SHORT, )
        except:
            print(u'\n{}'.format(log_msg))
        finally:
            return persistence_ratio, None, persistence_features

    kline_daily_baseline = data_baseline.data
    each_day = sorted(kline_daily_baseline.index.get_level_values(level=0).unique())
    persistence_ratio['daily']['total'] = len(each_day)

    try:
        each_day = sorted(kline_daily_baseline.index.get_level_values(level=0).unique())

        start = pd.to_datetime(each_day[0]) - timedelta(hours=8.3)
        end = pd.to_datetime(each_day[-1]) + timedelta(hours=16)
        if (len(offset) == 0):
            hour_baseline, codename_faked = get_kline_price_min(symbol,
                                                                start='{}'.format(start),
                                                                verbose=verbose,
                                                                realtime=True)
        else:
            hour_baseline, codename_faked = get_kline_price_min(symbol,
                                                                start='{}'.format(start),
                                                                end='{}'.format(end),
                                                                verbose=verbose,
                                                                realtime=False)
        kline_hour_baseline = hour_baseline.data

        if (market_type == MARKET_TYPE.STOCK_CN):
            persistence_features = attach_reality_features(features_dummy=kline_hour_baseline,
                                                           annual=1008,
                                                           collections=DATABASE_GolemQ.stock_reality_features, )
        elif (market_type == MARKET_TYPE.INDEX_CN):
            persistence_features = attach_reality_features(features_dummy=kline_hour_baseline,
                                                           annual=1008,
                                                           collections=DATABASE_GolemQ.index_reality_features, )

        persistence_features = persistence_features.reindex(columns=list(set([*persistence_features.columns,
                                                                               *[AKA.STAGE,
                                                                                 FLD.RENKO_TREND_S_TIMING_LAG,
                                                                                 FLD.MAGIC_NINE_TURNS_BASELINE,
                                                                                 FTR.ZEN_PEAK_TIMING_LAG_MAJOR_REAL,
                                                                                 FTR.ZEN_PEAK_TIMING_LAG_REAL,
                                                                                 FTR.ZEN_DASH_TIMING_LAG_MINOR_REAL,
                                                                                 FTR.ZEN_PEAK_TIMING_LAG_MINOR_REAL, ]])))
        if (debug):
            persistence_features.drop(index=persistence_features.tail(random.randint(5, 90)).index, inplace=True)

        try:
            persistence_features_daily = persistence_features.dropna(axis=0,
                                                                      subset=[FTR.ZEN_PEAK_TIMING_LAG_MAJOR_REAL],
                                                                      how="all")
            persistence_features_daily = persistence_features_daily.dropna(axis=0,
                                                                           subset=[FLD.MAXFACTOR_MAJOR],
                                                                           how="all")
            persistence_features_daily = persistence_features_daily.dropna(axis=0,
                                                                           subset=[FLD.MAGIC_NINE_TURNS_MAJOR],
                                                                           how="all")
        except:
            persistence_features_daily = pd.DataFrame(columns=persistence_features.columns)
        try:
            persistence_features_hourly = persistence_features.dropna(axis=0,
                                                                       subset=[FLD.MAGIC_NINE_TURNS_MAJOR],
                                                                       how="all")
            persistence_features_hourly = persistence_features_hourly.dropna(axis=0,
                                                                             subset=[FLD.POLYNOMIAL9],
                                                                             how="all")
        except:
            persistence_features_hourly = pd.DataFrame(columns=persistence_features.columns)
        try:
            persistence_features_stage = persistence_features.dropna(axis=0,
                                                                      subset=[AKA.STAGE],
                                                                      how="all")
        except:
            persistence_features_stage = pd.DataFrame(columns=persistence_features.columns)
        try:
            persistence_features_15min = persistence_features.dropna(axis=0,
                                                                      subset=[FTR.ZEN_PEAK_TIMING_LAG_MINOR_REAL],
                                                                      how="all")
        except:
            persistence_features_15min = pd.DataFrame(columns=persistence_features.columns)
        try:
            persistence_features_m9t = persistence_features.dropna(axis=0,
                                                                    subset=peek_column,
                                                                    how="all")
        except:
            persistence_features_m9t = pd.DataFrame(columns=persistence_features.columns)
        persistence_ratio['m9t']['distal_missing'] = True if (len(persistence_features_m9t) < 1) else (False if (hour_baseline.index.get_level_values(level=0)[0] -
                                                                                                                  persistence_features_m9t.index.get_level_values(level=0)[0]) < timedelta(hours=6) else True)
        persistence_ratio['m9t']['tail_missing'] = True if (len(persistence_features_m9t) < 1) else (False if (hour_baseline.index.get_level_values(level=0)[-1] -
                                                                                                                 persistence_features_m9t.index.get_level_values(level=0)[-1]) < timedelta(hours=6) else True)
        persistence_ratio['m9t']['total'] = len(kline_hour_baseline)
        persistence_ratio['m9t']['persistence'] = len(persistence_features_m9t)
        persistence_ratio['m9t']['ratio'] = round(len(persistence_features_m9t) / len(kline_hour_baseline), 4)
        persistence_ratio['m9t']['unmask'] = kline_hour_baseline.index.difference(persistence_features_m9t.index)
        persistence_ratio['m9t']['masked'] = calc_masked_tail_missing_index(persistence_ratio['m9t']['unmask'],
                                                                             baseline=hour_baseline,
                                                                             persistence_ratio=persistence_ratio['m9t']['ratio'])
        persistence_ratio['stage']['distal_missing'] = True if (len(persistence_features_stage) < 1) else (False if (hour_baseline.index.get_level_values(level=0)[0] -
                                                                                                                     persistence_features_stage.index.get_level_values(level=0)[0]) < timedelta(hours=6) else True)
        persistence_ratio['stage']['tail_missing'] = True if (len(persistence_features_stage) < 1) else (False if (hour_baseline.index.get_level_values(level=0)[-1] -
                                                                                                                    persistence_features_stage.index.get_level_values(level=0)[-1]) < timedelta(hours=6) else True)
        persistence_ratio['stage']['total'] = len(kline_hour_baseline)
        persistence_ratio['stage']['persistence'] = len(persistence_features_stage)
        persistence_ratio['stage']['ratio'] = round(len(persistence_features_stage) / len(kline_hour_baseline), 4)
        persistence_ratio['stage']['unmask'] = kline_hour_baseline.index.difference(persistence_features_stage.index)
        persistence_ratio['stage']['masked'] = calc_masked_tail_missing_index(persistence_ratio['stage']['unmask'],
                                                                               baseline=hour_baseline,
                                                                               persistence_ratio=persistence_ratio['stage']['ratio'])

        persistence_ratio['hourly']['distal_missing'] = True if (len(persistence_features_hourly) < 1) else (False if (hour_baseline.index.get_level_values(level=0)[0] -
                                                                                                                       persistence_features_hourly.index.get_level_values(level=0)[0]) < timedelta(hours=6) else True)
        persistence_ratio['hourly']['tail_missing'] = True if (len(persistence_features_hourly) < 1) else (False if (hour_baseline.index.get_level_values(level=0)[-1] -
                                                                                                                      persistence_features_hourly.index.get_level_values(level=0)[-1]) < timedelta(hours=6) else True)
        persistence_ratio['hourly']['total'] = len(kline_hour_baseline)
        persistence_ratio['hourly']['persistence'] = len(persistence_features_hourly)
        persistence_ratio['hourly']['ratio'] = round(len(persistence_features_hourly) / len(kline_hour_baseline), 4)
        if (len(persistence_features_daily) > 20) and \
           (len(persistence_features_hourly) > 0) and \
           (len(persistence_features_daily) / len(persistence_features_hourly) < 0.382):
            persistence_ratio['hourly']['unmask'] = pd.Index(list(set([*kline_hour_baseline.index.difference(persistence_features_hourly.index),
                                                                        *kline_hour_baseline.index.difference(persistence_features_daily.index), ])))
        else:
            persistence_ratio['hourly']['unmask'] = kline_hour_baseline.index.difference(persistence_features_hourly.index)

        persistence_ratio['hourly']['masked'] = calc_masked_tail_missing_index(persistence_ratio['hourly']['unmask'],
                                                                                baseline=hour_baseline,
                                                                                persistence_ratio=persistence_ratio['hourly']['ratio'])
        persistence_ratio['15min']['distal_missing'] = True if (len(persistence_features_15min) < 1) else (False if (hour_baseline.index.get_level_values(level=0)[0] -
                                                                                                                     persistence_features_15min.index.get_level_values(level=0)[0]) < timedelta(hours=6) else True)
        persistence_ratio['15min']['tail_missing'] = True if (len(persistence_features_15min) < 1) else (False if (hour_baseline.index.get_level_values(level=0)[-1] -
                                                                                                                    persistence_features_15min.index.get_level_values(level=0)[-1]) < timedelta(hours=6) else True)
        persistence_ratio['15min']['total'] = len(kline_hour_baseline)
        persistence_ratio['15min']['persistence'] = len(persistence_features_15min)
        persistence_ratio['15min']['ratio'] = round(len(persistence_features_15min) / len(kline_hour_baseline), 4)
        persistence_ratio['15min']['unmask'] = kline_hour_baseline.index.difference(persistence_features_15min.index)
        persistence_ratio['15min']['masked'] = calc_masked_tail_missing_index(persistence_ratio['15min']['unmask'],
                                                                               baseline=hour_baseline,
                                                                               persistence_ratio=persistence_ratio['stage']['ratio'])

        persistence_ratio['daily']['distal_missing'] = True if (len(persistence_features_daily) < 1) else (False if (hour_baseline.index.get_level_values(level=0)[0] -
                                                                                                                     persistence_features_daily.index.get_level_values(level=0)[0]) < timedelta(hours=6) else True)
        persistence_ratio['daily']['tail_missing'] = True if (len(persistence_features_daily) < 1) else (False if (hour_baseline.index.get_level_values(level=0)[-1] -
                                                                                                                    persistence_features_daily.index.get_level_values(level=0)[-1]) < timedelta(hours=6) else True)
        persistence_ratio['daily']['total'] = len(kline_hour_baseline)
        persistence_ratio['daily']['persistence'] = len(persistence_features_daily)
        persistence_ratio['daily']['ratio'] = round(len(persistence_features_daily) / len(kline_hour_baseline), 4)
        unmask_daily_index = pd.to_datetime(kline_hour_baseline.index.difference(persistence_features_daily.index).get_level_values(level=0).date, )
        persistence_ratio['daily']['unmask'] = unmask_daily_index
        persistence_ratio['daily']['masked'] = calc_masked_tail_missing_index(persistence_ratio['daily']['unmask'],
                                                                               baseline=hour_baseline,
                                                                               persistence_ratio=persistence_ratio['daily']['ratio'])

        if (verbose):
            print(u'[{}] {} 先验数据前置妥善率：({}) {:.2%}, ({}) {:.2%}, ({}) {:.2%}, ({}) {:.2%}, ({}) {:.2%} Total:{}(H:{})'.format(GQ_util_timestamp_to_str(),
                                                                                                                                       symbol,
                                                                                                                                       (len(persistence_features[FTR.ZEN_PEAK_TIMING_LAG_MAJOR_REAL].dropna())),
                                                                                                                                       persistence_ratio['daily']['ratio'],
                                                                                                                                       (len(persistence_features[FTR.ZEN_PEAK_TIMING_LAG_REAL].dropna())),
                                                                                                                                       persistence_ratio['hourly']['ratio'],
                                                                                                                                       (len(persistence_features[FTR.ZEN_PEAK_TIMING_LAG_MINOR_REAL].dropna())),
                                                                                                                                       persistence_ratio['15min']['ratio'],
                                                                                                                                       (len(persistence_features[AKA.STAGE].dropna())),
                                                                                                                                       persistence_ratio['stage']['ratio'],
                                                                                                                                       peek_column, persistence_ratio['m9t']['ratio'],
                                                                                                                                       len(persistence_features),
                                                                                                                                       persistence_ratio['hourly']['total'], ), )
            if (persistence_ratio['15min']['ratio'] < 0.382):
                print(persistence_features_15min[[FTR.ZEN_DASH_TIMING_LAG_MINOR_REAL,
                                                  FTR.ZEN_PEAK_TIMING_LAG_MINOR_REAL, ]])
    except Exception as e:
        if (len(each_day) > 0):
            traceback.print_exc()
            print(u'code:{}, kline hourly length: {}. get a Error!'.format(symbol,
                                                                           len(kline_daily_baseline)), e)

    return persistence_ratio, kline_hour_baseline, persistence_features
