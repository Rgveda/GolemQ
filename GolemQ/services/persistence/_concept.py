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
Concept and massive data persistence checks.

Validates data completeness for stock-concept features and market-wide
massive reality models.
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
import QUANTAXIS as QA
from QUANTAXIS.QAUtil.QADate_Adv import (
    QA_util_timestamp_to_str,
)

from GolemQ.core.settings import DATABASE as DATABASE_GolemQ
from GolemQ.core.constants import (
    AKA,
    FIELD as FLD,
)
from GolemQ.models.massive import MAS
from GolemQ.features.empirical import load_massive_reviews
from GolemQ.markets.StockCN.kline83 import get_kline_price_min
# TODO: 概念 K 线尚无真实实现 —— GolemQ.fetch.concept 仍是 stub，
# 真实版本只存在于 GolemQ_old/fetch/concept.py:865（读 4.4）。待移植。
from GolemQ.fetch.concept import get_stock_concept_kline
from GolemQ.analysis.timeseries import (
    Timeline_duration,
    align_kline_timeline,
)
from GolemQ.services.persistence._schema import (
    concept_review_columns_of_persistence,
    stock_review_columns_of_persistence,
    massive_review_columns_of_persistence,
    calc_masked_tail_missing_index,
)


def dataloader_concept_check(symbol=None,
                             offset: str = '',
                             market_type=QA.MARKET_TYPE.STOCK_CN,
                             verbose: bool = False,
                             peek_column: list = [MAS.CONCEPT_MACD_COMPOUDED_RATIO_MEDIAN],
                             debug: bool = False, ):
    """
    持久化指标数据加载和缺漏检查
    """
    persistence_ratio = {'symbol': symbol,
                         'hourly': {'total': 0, 'persistence': 0, 'ratio': 0.0, 'masked': [], },
                         'kline': {'total': 0, 'persistence': 0, 'ratio': 0.0, 'masked': [], },
                         }
    persistence_features = pd.DataFrame(columns=concept_review_columns_of_persistence())
    persistence_features_hourly = pd.DataFrame(columns=concept_review_columns_of_persistence())
    each_day = []
    hour_baseline = None

    if (len(offset) == 0):
        try:
            ret_concept_kline = get_stock_concept_kline(symbol,
                                                        freq=QA.FREQUENCE.HOUR, )
            hour_baseline = ret_concept_kline
            codelist = sorted(hour_baseline.index.get_level_values(level=1).unique())
            each_day = sorted(hour_baseline.index.get_level_values(level=0).unique())
            persistence_features = load_massive_reviews(symbol,
                                                        start='{}'.format(each_day[0]),
                                                        end='{}'.format(each_day[-1]),
                                                        compact=True,
                                                        collections=DATABASE_GolemQ.concept_shaplet_reviews, )
            if (debug):
                persistence_features.drop(index=persistence_features.tail(random.randint(5, 90)).index, inplace=True)
        except:
            print(u'function dataloader_review_check(symbol=\'{}\') got a Error. '.format(symbol))
            traceback.print_exc()
    else:
        start = '{}'.format(pd.to_datetime(offset).tz_localize('Asia/Shanghai') -
                            datetime.timedelta(days=980))
        end = '{}'.format(pd.to_datetime(offset).tz_localize('Asia/Shanghai') +
                          timedelta(hours=16))
        ret_concept_kline = get_stock_concept_kline(symbol,
                                                    start=start,
                                                    end=end,
                                                    freq=QA.FREQUENCE.HOUR, )
        persistence_features = load_massive_reviews(symbol,
                                                    start=start,
                                                    end=end,
                                                    compact=True,
                                                    collections=DATABASE_GolemQ.concept_shaplet_reviews, )

    if (hour_baseline is None):
        print(u'code:{}, hourly kline is None!'.format(symbol, ))
        return persistence_ratio, None, pd.DataFrame(columns=concept_review_columns_of_persistence())

    each_day = sorted(hour_baseline.index.get_level_values(level=0).unique())
    try:
        persistence_features_hourly = persistence_features.dropna(axis=0,
                                                                   subset=peek_column,
                                                                   how="all")
    except:
        if (persistence_features is not None):
            persistence_features_hourly = pd.DataFrame(columns=persistence_features.columns)
        else:
            persistence_features_hourly = pd.DataFrame(columns=stock_review_columns_of_persistence())
    try:
        persistence_features_kline = persistence_features.dropna(axis=0,
                                                                  subset=[AKA.CLOSE],
                                                                  how="all")
    except:
        if (persistence_features is not None):
            persistence_features_kline = pd.DataFrame(columns=persistence_features.columns)
        else:
            persistence_features_kline = pd.DataFrame(columns=stock_review_columns_of_persistence())

    persistence_ratio['hourly']['distal_missing'] = True if (len(persistence_features_hourly) < 1) else (False if (hour_baseline.index.get_level_values(level=0)[0] -
                                                                                                                   persistence_features_hourly.index.get_level_values(level=0)[0]) < timedelta(hours=6) else True)
    persistence_ratio['hourly']['tail_missing'] = True if (len(persistence_features_hourly) < 1) else (False if (hour_baseline.index.get_level_values(level=0)[-1] -
                                                                                                                  persistence_features_hourly.index.get_level_values(level=0)[-1]) < timedelta(hours=6) else True)
    if (persistence_features is not None):
        persistence_ratio['hourly']['total'] = len(list(set([*hour_baseline.index.get_level_values(level=0),
                                                              *persistence_features.index.get_level_values(level=0)])))
    else:
        persistence_ratio['hourly']['total'] = len(hour_baseline.index.get_level_values(level=0))
    persistence_ratio['hourly']['persistence'] = len(persistence_features_hourly)
    persistence_ratio['hourly']['ratio'] = round(len(persistence_features_hourly) / persistence_ratio['hourly']['total'], 4)
    persistence_ratio['hourly']['unmask'] = hour_baseline.index.difference(persistence_features_hourly.index).get_level_values(level=0)
    persistence_ratio['hourly']['masked'] = calc_masked_tail_missing_index(persistence_ratio['hourly']['unmask'],
                                                                            baseline=hour_baseline,
                                                                            persistence_ratio=persistence_ratio['hourly']['ratio'])
    persistence_ratio['kline']['distal_missing'] = True if (len(persistence_features_kline) < 1) else (False if (hour_baseline.index.get_level_values(level=0)[0] -
                                                                                                                   persistence_features_kline.index.get_level_values(level=0)[0]) < timedelta(hours=6) else True)
    persistence_ratio['kline']['tail_missing'] = True if (len(persistence_features_kline) < 1) else (False if (hour_baseline.index.get_level_values(level=0)[-1] -
                                                                                                                  persistence_features_kline.index.get_level_values(level=0)[-1]) < timedelta(hours=6) else True)
    if (persistence_features is not None):
        persistence_ratio['kline']['total'] = len(list(set([*hour_baseline.index.get_level_values(level=0),
                                                             *persistence_features.index.get_level_values(level=0)])))
    else:
        persistence_ratio['kline']['total'] = len(hour_baseline.index.get_level_values(level=0))
    persistence_ratio['kline']['persistence'] = len(persistence_features_kline)
    persistence_ratio['kline']['ratio'] = round(len(persistence_features_kline) / persistence_ratio['kline']['total'], 4)
    if (persistence_features is not None):
        if ((hour_baseline.index.get_level_values(level=0)[-1] -
             persistence_features.index.get_level_values(level=0)[-1]) > timedelta(hours=0.25)):
            persistence_ratio['kline']['unmask'] = hour_baseline.index.difference(persistence_features_kline.index).get_level_values(level=0)
            persistence_ratio['kline']['masked'] = calc_masked_tail_missing_index(persistence_ratio['kline']['unmask'],
                                                                                   baseline=persistence_features,
                                                                                   persistence_ratio=persistence_ratio['kline']['ratio'])
        else:
            persistence_ratio['kline']['unmask'] = persistence_features.index.difference(persistence_features_kline.index).get_level_values(level=0)
            persistence_ratio['kline']['masked'] = calc_masked_tail_missing_index(persistence_ratio['kline']['unmask'],
                                                                                   baseline=persistence_features,
                                                                                   persistence_ratio=persistence_ratio['kline']['ratio'])
    else:
        persistence_ratio['kline']['unmask'] = hour_baseline.index.difference(persistence_features_kline.index).get_level_values(level=0)
        persistence_ratio['kline']['masked'] = calc_masked_tail_missing_index(persistence_ratio['kline']['unmask'],
                                                                               baseline=hour_baseline,
                                                                               persistence_ratio=persistence_ratio['kline']['ratio'])
    if (len(persistence_ratio['kline']['masked']) > 0.168):
        persistence_ratio['hourly']['masked'] = sorted(list(set([*persistence_ratio['hourly']['masked'],
                                                                  *persistence_ratio['kline']['masked']])))
    if (verbose):
        print(u'[{}] {} 板块走势先验数据妥善率：{:.2%},{:.2%},{:.2%} Total:{}'.format(QA_util_timestamp_to_str(),
                                                                                      symbol, persistence_ratio['hourly']['ratio'], ), )

    return persistence_ratio, hour_baseline, persistence_features


def dataloader_massive_check(revision: str = 'compact',
                             eval_range: str = 'fast',
                             offset: str = '',
                             market_type=QA.MARKET_TYPE.STOCK_CN,
                             peek_column=[MAS.MACD_COMPOUDED_BAND_RATIO_MEDIAN],
                             verbose: bool = False,
                             collections=DATABASE_GolemQ.stock_massive_models, ):
    """
    持久化指标数据加载和缺漏检查
    """
    from GolemQ.features.empirical import load_massive_model

    persistence_ratio = {'symbol': eval_range,
                         'hourly': {'total': 0, 'persistence': 0, 'ratio': 0.0, 'masked': [], },
                         }
    persistence_features = pd.DataFrame(columns=massive_review_columns_of_persistence())
    each_day = []
    kline_hour_baseline = None
    if (len(offset) == 0):
        persistence_features = load_massive_model(revision=revision,
                                                  eval_range=eval_range,
                                                  collections=collections, )
        kline_hour, display_name = get_kline_price_min(['399001',
                                                        '399006'],
                                                       verbose=verbose,
                                                       realtime=True)
        timeline_aligned, renew_ohlc, dropindex = align_kline_timeline(kline_hour.data,
                                                                       freq='30min',
                                                                       annual=1008)
        if (not timeline_aligned):
            kline_hour_baseline = renew_ohlc
        else:
            kline_hour_baseline = kline_hour.data
    else:
        start = '{}'.format(pd.to_datetime(offset).tz_localize('Asia/Shanghai') -
                            datetime.timedelta(days=980))
        end = '{}'.format(pd.to_datetime(offset).tz_localize('Asia/Shanghai') +
                          timedelta(hours=16))
        persistence_features = load_massive_model(revision=revision,
                                                  eval_range=eval_range,
                                                  start=start,
                                                  end=end,
                                                  collections=collections, )
        kline_hour, display_name = get_kline_price_min(['399001',
                                                        '399006'],
                                                       start='{}'.format(start),
                                                       end='{}'.format(end),
                                                       verbose=verbose,
                                                       realtime=False)
        timeline_aligned, renew_ohlc, dropindex = align_kline_timeline(kline_hour.data,
                                                                       freq='30min',
                                                                       annual=1008)
        if (not timeline_aligned):
            kline_hour_baseline = renew_ohlc
        else:
            kline_hour_baseline = kline_hour.data

    if (persistence_features is None):
        persistence_features = pd.DataFrame(columns=massive_review_columns_of_persistence())

    each_day = sorted(persistence_features.index.get_level_values(level=0).unique())

    try:
        persistence_features_hourly = persistence_features.dropna(axis=0,
                                                                   subset=peek_column,
                                                                   how="all")
    except:
        persistence_features_hourly = pd.DataFrame(columns=persistence_features.columns)

    try:
        persistence_ratio['hourly']['distal_missing'] = True if (len(persistence_features_hourly) < 1) else (False if (kline_hour_baseline.index.get_level_values(level=0)[0] -
                                                                                                                       persistence_features_hourly.index.get_level_values(level=0)[0]) < timedelta(hours=6) else True)
        persistence_ratio['hourly']['tail_missing'] = True if (len(persistence_features_hourly) < 1) else (False if (kline_hour_baseline.index.get_level_values(level=0)[-1] -
                                                                                                                      persistence_features_hourly.index.get_level_values(level=0)[-1]) < timedelta(hours=6) else True)
        persistence_ratio['hourly']['total'] = len(kline_hour_baseline.index.get_level_values(level=0).unique())
        persistence_ratio['hourly']['persistence'] = len(persistence_features_hourly)
        persistence_ratio['hourly']['ratio'] = round(len(persistence_features_hourly) / len(kline_hour_baseline.index.get_level_values(level=0).unique()), 4)
        persistence_ratio['hourly']['unmask'] = kline_hour_baseline.index.get_level_values(level=0).unique().difference(persistence_features_hourly.index)
        persistence_ratio['hourly']['masked'] = calc_masked_tail_missing_index(persistence_ratio['hourly']['unmask'],
                                                                                baseline=kline_hour_baseline,
                                                                                persistence_ratio=persistence_ratio['hourly']['ratio'])
        if (verbose):
            print(u'[{}] {} 大盘势态先验数据前置妥善率：{:.2%}, Total:{}'.format(QA_util_timestamp_to_str(),
                                                                                  eval_range,
                                                                                  persistence_ratio['hourly']['ratio'],
                                                                                  persistence_ratio['hourly']['total'], ), )
    except Exception as e:
        if (len(each_day) > 0):
            traceback.print_exc()
            print(u'code:{}, kline hourly length: {}. get a Error!'.format(eval_range,
                                                                           len(kline_hour_baseline)), e)

    if (MAS.STAGE_MODE in persistence_features.columns) and \
       (MAS.BOOTSTRAP_STAGE_MODE_BEFORE not in persistence_features.columns):
        if (persistence_features[MAS.STAGE_MODE].dtype == object):
            persistence_feats_mas_stage_mode = pd.Series([each[0] for i, each in persistence_features[MAS.STAGE_MODE].items()],
                                                         index=persistence_features.index)
            persistence_features[MAS.BOOTSTRAP_STAGE_MODE_BEFORE] = Timeline_duration(np.where((persistence_feats_mas_stage_mode < -0.000168), 1, 0))
        else:
            persistence_features[MAS.BOOTSTRAP_STAGE_MODE_BEFORE] = Timeline_duration(np.where((persistence_features[MAS.STAGE_MODE] < -0.000168), 1, 0))

    return persistence_ratio, kline_hour_baseline, persistence_features
