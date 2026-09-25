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
Daily persistence check for stock reality features.

Validates daily feature data completeness including chip distribution
(获利盘) persistence cache data.
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
from GolemQ.features.empirical import (
    save_stock_metadata,
    GQ_fetch_stock_metadata_major,
)
from GolemQ.fetch.kline import get_kline_price_v3
from GolemQ.core.base import GQ_util_get_last_day
from GolemQ.services.align import symbol_checkpoint_log
from GolemQ.services.persistence._schema import (
    daily_columns_of_persistence,
    calc_masked_tail_missing_index,
)


def dataloader_persistence_daily_check(
        symbol=None,
        offset: str = '',
        market_type=MARKET_TYPE.STOCK_CN,
        verbose: bool = False,
        peek_column: list = [FLD.STOCK_SCORE_M15],
        debug: bool = False,
        collections=DATABASE_GolemQ.stock_reality_feat_day,
):
    """
    持久化指标数据加载和缺漏检查
    """
    if (verbose):
        print(u'dataloader_persistence_daily_check() start')
    persistence_ratio = {'symbol': symbol,
                         'halt': False,
                         'daily': {'total': 0, 'persistence': 0, 'ratio': 0.0, 'masked': [], },
                         'stage': {'total': 0, 'persistence': 0, 'ratio': 0.0, 'masked': [], },
                         'm9t': {'total': 0, 'persistence': 0, 'ratio': 0.0, 'masked': [], },
                         }
    persistence_features = pd.DataFrame(columns=daily_columns_of_persistence())
    each_day = []
    kline_daily_baseline = None
    if (len(offset) == 0):
        data_baseline, codename = get_kline_price_v3(
            symbol,
            verbose=verbose,
            realtime=True)
    else:
        start = '{}'.format(pd.to_datetime(offset).tz_localize('Asia/Shanghai') -
                            datetime.timedelta(days=980))
        end = '{}'.format(pd.to_datetime(offset).tz_localize('Asia/Shanghai') +
                          timedelta(hours=16))
        data_baseline, codename = get_kline_price_v3(
            symbol,
            start=start,
            end=end,
            verbose=verbose,
            realtime=False)

    if (data_baseline is None):
        log_msg = 'Code:{}, daily kline is None!'.format(symbol, )
        code = symbol[0] if (isinstance(symbol, list)) else symbol
        try:
            symbol_checkpoint_log(
                logs=log_msg,
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
        # 加载（获利盘）持久化缓存数据
        stock_metadata_pd = GQ_fetch_stock_metadata_major(
            code=symbol,
            verbose=verbose,
            start=start, end=end,
            collections=collections,
        )
        if (stock_metadata_pd is not None) and \
           (FLD.MAXFACTOR in stock_metadata_pd.columns) and \
           (FLD.MAXFACTOR_MAJOR not in stock_metadata_pd.columns):
            stock_metadata_pd = stock_metadata_pd.rename(columns={FLD.MACD_ZERO_TIMING_LAG: FLD.MACD_ZERO_TIMING_LAG_MAJOR,
                                                                   FLD.BOLL_LB_HMA5_TIMING_LAG: FLD.BOLL_LB_HMA5_TIMING_LAG_MAJOR,
                                                                   FLD.DEA_NORM: FLD.DEA_NORM_MAJOR,
                                                                   FLD.DIF_NORM: FLD.DIF_NORM_MAJOR,
                                                                   FLD.MTM_NORM: FLD.MTM_NORM_MAJOR,
                                                                   FLD.MAPOWER30: FLD.MAPOWER30_MAJOR,
                                                                   FLD.MAXFACTOR: FLD.MAXFACTOR_MAJOR,
                                                                   FLD.ATR_Stopline_TIMING_LAG: FLD.ATR_Stopline_TIMING_LAG_MAJOR,
                                                                   FLD.ATR_SuperTrend_TIMING_LAG: FLD.ATR_SuperTrend_TIMING_LAG_MAJOR, })
            save_stock_metadata(stock_metadata_pd,
                                collections=collections)

        try:
            if (stock_metadata_pd is not None) and \
               (market_type == MARKET_TYPE.STOCK_CN):
                persistence_features = data_baseline.data.join(stock_metadata_pd[stock_metadata_pd.columns.difference(data_baseline.data.columns)])
            elif (stock_metadata_pd is not None) and \
                    (market_type == MARKET_TYPE.INDEX_CN or
                     market_type == MARKET_TYPE.ETF_CN):
                # ETF 2026-09 起是独立类型（原为 INDEX_CN），这里与指数同支
                # 以保住它改动前的行为 —— 那时它走的就是这一支。
                persistence_features = data_baseline.data.join(stock_metadata_pd[stock_metadata_pd.columns.difference(data_baseline.data.columns)])
            else:
                if (market_type == MARKET_TYPE.STOCK_CN):
                    print(u'\nCode:{} missing stock_metadata: stock_reality_feat_day'.format(symbol, ))
                elif (market_type == MARKET_TYPE.INDEX_CN) or \
                        (market_type == MARKET_TYPE.ETF_CN):
                    print(u'\nCode:{} missing index_metadata: index_reality_feat_day'.format(symbol, ))
        except Exception as e:
            print('code:{}'.format(symbol), e, '\n')
            print(stock_metadata_pd.columns.difference(data_baseline.data.columns), '\n\n', )
            print(stock_metadata_pd.columns.intersection(data_baseline.data.columns), '\n\n\n\n\n')
            persistence_features = data_baseline.data.join(stock_metadata_pd)

        persistence_features = persistence_features.reindex(columns=list(set([*persistence_features.columns,
                                                                               *[FTR.ZEN_PEAK_TIMING_LAG_MAJOR_REAL,
                                                                                 FLD.MAXFACTOR_MAJOR,
                                                                                 FTR.MAGIC_NINE_TURNS_MAJOR_REAL, ]])))
        if (debug):
            persistence_features.drop(index=persistence_features.tail(random.randint(5, 90)).index, inplace=True)

        try:
            persistence_features_daily = persistence_features.dropna(axis=0,
                                                                      subset=[FTR.ZEN_PEAK_TIMING_LAG_MAJOR_REAL],
                                                                      how="all")
            persistence_features_daily = persistence_features_daily.dropna(axis=0,
                                                                           subset=[FLD.MAXFACTOR_MAJOR],
                                                                           how="all")
            if (len(persistence_features_daily) == 0) and \
               (verbose):
                print(FLD.MAXFACTOR_MAJOR, 'len(persistence_features_daily)=={}'.format(len(persistence_features_daily)))
            persistence_features_daily = persistence_features_daily.dropna(axis=0,
                                                                           subset=[FTR.MAGIC_NINE_TURNS_MAJOR_REAL],
                                                                           how="all")
            if (len(persistence_features_daily) == 0) and \
               (verbose):
                print(FTR.MAGIC_NINE_TURNS_MAJOR_REAL, 'len(persistence_features_daily)=={}'.format(len(persistence_features_daily)))
        except:
            if (FLD.MAXFACTOR_MAJOR not in persistence_features.columns) or \
               (FTR.MAGIC_NINE_TURNS_MAJOR_REAL not in persistence_features.columns):
                traceback.print_exc()
            persistence_features_daily = pd.DataFrame(columns=persistence_features.columns)

        try:
            persistence_features_stage = persistence_features.dropna(axis=0,
                                                                      subset=[AKA.STAGE],
                                                                      how="all")
        except:
            persistence_features_stage = pd.DataFrame(columns=persistence_features.columns)
        try:
            persistence_features_m9t = persistence_features.dropna(axis=0,
                                                                    subset=peek_column,
                                                                    how="all")
        except:
            persistence_features_m9t = pd.DataFrame(columns=persistence_features.columns)
        persistence_ratio['m9t']['distal_missing'] = True if (len(persistence_features_m9t) < 1) else (False if (data_baseline.index.get_level_values(level=0)[0] -
                                                                                                                  persistence_features_m9t.index.get_level_values(level=0)[0]) < timedelta(hours=6) else True)
        persistence_ratio['m9t']['tail_missing'] = True if (len(persistence_features_m9t) < 1) else (False if (data_baseline.index.get_level_values(level=0)[-1] -
                                                                                                                 persistence_features_m9t.index.get_level_values(level=0)[-1]) < timedelta(hours=6) else True)
        persistence_ratio['m9t']['total'] = len(kline_daily_baseline)
        persistence_ratio['m9t']['persistence'] = len(persistence_features_m9t)
        persistence_ratio['m9t']['ratio'] = round(len(persistence_features_m9t) / len(kline_daily_baseline), 4)
        persistence_ratio['m9t']['unmask'] = kline_daily_baseline.index.difference(persistence_features_m9t.index)
        persistence_ratio['m9t']['masked'] = calc_masked_tail_missing_index(persistence_ratio['m9t']['unmask'],
                                                                             baseline=data_baseline,
                                                                             persistence_ratio=persistence_ratio['m9t']['ratio'])
        persistence_ratio['stage']['distal_missing'] = True if (len(persistence_features_stage) < 1) else (False if (data_baseline.index.get_level_values(level=0)[0] -
                                                                                                                     persistence_features_stage.index.get_level_values(level=0)[0]) < timedelta(hours=6) else True)
        persistence_ratio['stage']['tail_missing'] = True if (len(persistence_features_stage) < 1) else (False if (data_baseline.index.get_level_values(level=0)[-1] -
                                                                                                                    persistence_features_stage.index.get_level_values(level=0)[-1]) < timedelta(hours=6) else True)
        persistence_ratio['stage']['total'] = len(kline_daily_baseline)
        persistence_ratio['stage']['persistence'] = len(persistence_features_stage)
        persistence_ratio['stage']['ratio'] = round(len(persistence_features_stage) / len(kline_daily_baseline), 4)
        persistence_ratio['stage']['unmask'] = kline_daily_baseline.index.difference(persistence_features_stage.index)
        persistence_ratio['stage']['masked'] = calc_masked_tail_missing_index(persistence_ratio['stage']['unmask'],
                                                                               baseline=data_baseline,
                                                                               persistence_ratio=persistence_ratio['stage']['ratio'])

        persistence_ratio['daily']['distal_missing'] = True if (len(persistence_features_daily) < 1) else (False if (data_baseline.index.get_level_values(level=0)[0] -
                                                                                                                     persistence_features_daily.index.get_level_values(level=0)[0]) < timedelta(hours=6) else True)
        persistence_ratio['daily']['tail_missing'] = True if (len(persistence_features_daily) < 1) else (False if (data_baseline.index.get_level_values(level=0)[-1] -
                                                                                                                    persistence_features_daily.index.get_level_values(level=0)[-1]) < timedelta(hours=6) else True)
        persistence_ratio['daily']['total'] = len(kline_daily_baseline)
        persistence_ratio['daily']['persistence'] = len(persistence_features_daily)
        persistence_ratio['daily']['ratio'] = round(len(persistence_features_daily) / len(kline_daily_baseline), 4)
        unmask_daily_index = pd.to_datetime(kline_daily_baseline.index.difference(persistence_features_daily.index).get_level_values(level=0).date, )
        persistence_ratio['daily']['unmask'] = unmask_daily_index
        persistence_ratio['daily']['masked'] = calc_masked_tail_missing_index(persistence_ratio['daily']['unmask'],
                                                                               baseline=data_baseline,
                                                                               persistence_ratio=persistence_ratio['daily']['ratio'])

        if (verbose):
            print(u'[{}] {} 先验数据前置妥善率：({}) {:.2%}, ({}) {:.2%} Total:{}(H:{})'.format(GQ_util_timestamp_to_str(),
                                                                                                symbol,
                                                                                                (len(persistence_features[FTR.ZEN_PEAK_TIMING_LAG_MAJOR_REAL].dropna())),
                                                                                                persistence_ratio['daily']['ratio'],
                                                                                                peek_column, persistence_ratio['m9t']['ratio'],
                                                                                                len(persistence_features),
                                                                                                persistence_ratio['daily']['total'], ), )
    except Exception as e:
        if (len(each_day) > 0):
            traceback.print_exc()
            print(u'code:{}, kline daily length: {}. get a Error!'.format(symbol,
                                                                          len(kline_daily_baseline)), e)
    return persistence_ratio, kline_daily_baseline, persistence_features
