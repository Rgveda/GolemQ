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
Stock review data persistence check.

Loads persistent feature data for individual stocks and checks data completeness
against hourly kline baselines.
"""

import numpy as np
import pandas as pd
from datetime import (
    datetime as dt,
    timezone, timedelta
)
import datetime
import traceback
import QUANTAXIS as QA
from QUANTAXIS.QAUtil.QADate_Adv import (
    QA_util_timestamp_to_str,
)

from GolemQ.core.settings import DATABASE as DATABASE_GolemQ
from GolemQ.core.constants import (
    AKA,
    FIELD as FLD,
    STATE as STE,
)
from GolemQ.features.empirical import load_massive_reviews
from GolemQ.markets.StockCN.kline83 import get_kline_price_min
from GolemQ.services.persistence._schema import (
    stock_review_columns_of_persistence,
    calc_masked_tail_missing_index,
)


def dataloader_review_check_reflush(persistence_ratio: pd.DataFrame = None,
                                    persistence_features_hourly: pd.DataFrame = None,
                                    hour_baseline: pd.DataFrame = None):
    persistence_ratio['hourly']['distal_missing'] = True if (len(persistence_features_hourly) < 1) else (False if (hour_baseline.index.get_level_values(level=0)[0] -
                                                                                                                   persistence_features_hourly.index.get_level_values(level=0)[0]) < timedelta(hours=6) else True)
    persistence_ratio['hourly']['tail_missing'] = True if (len(persistence_features_hourly) < 1) else (False if (hour_baseline.index.get_level_values(level=0)[-1] -
                                                                                                                  persistence_features_hourly.index.get_level_values(level=0)[-1]) < timedelta(hours=6) else True)
    persistence_ratio['hourly']['total'] = len(hour_baseline)
    persistence_ratio['hourly']['persistence'] = len(persistence_features_hourly)
    persistence_ratio['hourly']['ratio'] = round(len(persistence_features_hourly) / len(hour_baseline), 4)
    persistence_ratio['hourly']['unmask'] = hour_baseline.index.difference(persistence_features_hourly.index)
    persistence_ratio['hourly']['masked'] = calc_masked_tail_missing_index(persistence_ratio['hourly']['unmask'],
                                                                            baseline=hour_baseline,
                                                                            persistence_ratio=persistence_ratio['hourly']['ratio'])
    return persistence_ratio


def dataloader_review_check(symbol=None,
                            features: pd.DataFrame = None,
                            offset: str = '',
                            market_type=QA.MARKET_TYPE.STOCK_CN,
                            verbose: bool = False,
                            peek_column=[STE.MACD_COMPOUDED_BAND_RATIO],
                            collections=DATABASE_GolemQ.stock_ta_reviews, ):
    """
    持久化指标数据加载和缺漏检查
    """
    if (verbose):
        print(u'dataloader_review_check start')
    persistence_ratio = {'symbol': symbol,
                         'hourly': {'total': 0, 'persistence': 0, 'ratio': 0.0, 'masked': [], },
                         }
    persistence_features = pd.DataFrame(columns=stock_review_columns_of_persistence())
    persistence_features_hourly = pd.DataFrame(columns=stock_review_columns_of_persistence())
    each_day = []
    hour_baseline = None
    data_baseline = None

    if (features is not None):
        codelist = sorted(features.index.get_level_values(level=1).unique())
        each_day = sorted(features.index.get_level_values(level=0).unique())
        persistence_ratio['symbol'] = symbol
        persistence_features = load_massive_reviews(symbol,
                                                    start='{}'.format(each_day[0]),
                                                    end='{}'.format(each_day[-1]),
                                                    compact=True,
                                                    collections=collections, )
        hour_baseline = features
    elif (len(offset) == 0):
        try:
            data_baseline, codename = get_kline_price_min(symbol,
                                                          verbose=verbose,
                                                          realtime=True)
            hour_baseline = data_baseline.data
            codelist = sorted(hour_baseline.index.get_level_values(level=1).unique())
            each_day = sorted(hour_baseline.index.get_level_values(level=0).unique())
            persistence_features = load_massive_reviews(codelist,
                                                        start='{}'.format(each_day[0]),
                                                        end='{}'.format(each_day[-1]),
                                                        compact=True,
                                                        collections=collections, )
        except:
            if (data_baseline is not None):
                print(u'function dataloader_review_check(symbol=\'{}\') got a Error. '.format(symbol))
                traceback.print_exc()
            else:
                # 申购，未上市
                pass
    else:
        start = '{}'.format(pd.to_datetime(offset).tz_localize('Asia/Shanghai') -
                            datetime.timedelta(days=980))
        end = '{}'.format(pd.to_datetime(offset).tz_localize('Asia/Shanghai') +
                          timedelta(hours=16))
        data_baseline, codename = get_kline_price_min(symbol,
                                                      start=start,
                                                      end=end,
                                                      realtime=False,
                                                      verbose=verbose)
        if (data_baseline is not None):
            hour_baseline = data_baseline.data
        persistence_features = load_massive_reviews(symbol,
                                                    start=start,
                                                    end=end,
                                                    compact=True,
                                                    collections=collections, )

    if (hour_baseline is None):
        if (verbose):
            print(u'\nCode:{}, hourly kline is None!'.format(symbol, ))
        return persistence_ratio, None, None

    each_day = sorted(hour_baseline.index.get_level_values(level=0).unique())
    try:
        persistence_features_hourly = persistence_features.dropna(axis=0,
                                                                   subset=peek_column,
                                                                   how="all")
        if (AKA.CLOSE in persistence_features_hourly.columns):
            persistence_features_hourly = persistence_features_hourly.dropna(axis=0,
                                                                             subset=[AKA.CLOSE],
                                                                             how="all")
        if (AKA.LOW in persistence_features_hourly.columns):
            persistence_features_hourly = persistence_features_hourly.dropna(axis=0,
                                                                             subset=[AKA.LOW],
                                                                             how="all")
        if (FLD.DIF_NORM_MAJOR in persistence_features_hourly.columns):
            persistence_features_hourly = persistence_features_hourly.dropna(axis=0,
                                                                             subset=[FLD.DIF_NORM_MAJOR],
                                                                             how="all")
        if (FLD.DEA_NORM_MAJOR in persistence_features_hourly.columns):
            persistence_features_hourly = persistence_features_hourly.dropna(axis=0,
                                                                             subset=[FLD.DEA_NORM_MAJOR],
                                                                             how="all")
        if (FLD.MA5_MAJOR in persistence_features_hourly.columns):
            persistence_features_hourly = persistence_features_hourly.dropna(axis=0,
                                                                             subset=[FLD.MA5_MAJOR],
                                                                             how="all")
        if (FLD.MA10_MAJOR in persistence_features_hourly.columns):
            persistence_features_hourly = persistence_features_hourly.dropna(axis=0,
                                                                             subset=[FLD.MA10_MAJOR],
                                                                             how="all")
        if (FLD.FBPROPHET_LB_MAJOR in persistence_features_hourly.columns):
            persistence_features_hourly = persistence_features_hourly.dropna(axis=0,
                                                                             subset=[FLD.FBPROPHET_LB_MAJOR],
                                                                             how="all")
        if (FLD.FBPROPHET_LB in persistence_features_hourly.columns):
            persistence_features_hourly = persistence_features_hourly.dropna(axis=0,
                                                                             subset=[FLD.FBPROPHET_LB],
                                                                             how="all")
        if (FLD.ATR_Stopline_PRICE_MAJOR in persistence_features_hourly.columns):
            persistence_features_hourly = persistence_features_hourly.dropna(axis=0,
                                                                             subset=[FLD.ATR_Stopline_PRICE_MAJOR],
                                                                             how="all")
        if (FLD.ATR_Stopline_PRICE_WEEKLY in persistence_features_hourly.columns):
            persistence_features_hourly = persistence_features_hourly.dropna(axis=0,
                                                                             subset=[FLD.ATR_Stopline_PRICE_WEEKLY],
                                                                             how="all")
    except:
        if (persistence_features is not None):
            print(u'code:{}, persistence_features is None!'.format(symbol, ))
            persistence_features_hourly = pd.DataFrame(columns=persistence_features.columns)
        else:
            persistence_features = pd.DataFrame(columns=stock_review_columns_of_persistence())

        if (len(persistence_features) - len(persistence_features_hourly) > 262):
            traceback.print_exc()

    persistence_ratio = dataloader_review_check_reflush(persistence_ratio=persistence_ratio,
                                                        persistence_features_hourly=persistence_features_hourly,
                                                        hour_baseline=hour_baseline, )

    if (verbose):
        print(u'[{}] {} 先验数据前置妥善率：{:.2%},{:.2%},{:.2%} Total:{}'.format(QA_util_timestamp_to_str(),
                                                                                  symbol, persistence_ratio['hourly']['ratio'], ), )

    return persistence_ratio, hour_baseline, persistence_features
