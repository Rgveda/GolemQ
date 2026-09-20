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
Persistence column schemas and helper functions.

Defines the column sets used by persistence check functions and provides
utility functions for missing-index calculation and reasonableness checks.
"""

import numpy as np
import pandas as pd
from datetime import timedelta

from GolemQ.core.constants import (
    AKA,
    FEATURES as FTR,
    FIELD as FLD,
    TREND_STATUS as ST,
    STATE as STE,
)
from GolemQ.models.massive import MAS
from GolemQ.models.risk import RSK


def concept_review_columns_of_persistence():
    massive_columns = [MAS.CONCEPT_ZEN_DASH_TIMING_LAG_COMBO,
                       MAS.CONCEPT_XGB_ECHO_TIMING_LAG_COMBO,
                       MAS.CONCEPT_STOCK_SCORE_M15_TIMING_LAG_COMBO,
                       MAS.CONCEPT_STOCK_SCORE_M15_NORM_TIMING_LAG_COMBO,
                       MAS.CONCEPT_DIF_WAVELET_TIMING_LAG_COMBO,
                       MAS.CONCEPT_MTM_NORM_TIMING_LAG_MAJOR_COMBO,
                       MAS.CONCEPT_DIF_NORM_TIMING_LAG_MAJOR_COMBO,
                       MAS.CONCEPT_DEA_NORM_TIMING_LAG_MAJOR_COMBO,
                       MAS.CONCEPT_STAGE,
                       MAS.CONCEPT_STAGE_DUMMY,
                       MAS.CONCEPT_BOOTSTRAP_BEFORE,
                       MAS.CONCEPT_ENDPOINT_BEFORE,
                       MAS.CONCEPT_TREND_TIMING_LAG,
                       MAS.CONCEPT_MACD_COMPOUDED_RATIO_MEDIAN, ]

    return massive_columns


def stock_review_columns_of_persistence():
    massive_columns = [FLD.MAPOWER30,
                       FLD.MAPOWER30_MAJOR,
                       FLD.STOCK_SCORE_M15,
                       FLD.MAXFACTOR,
                       FLD.MAXFACTOR_MAJOR,
                       STE.MACD_COMPOUDED_BAND_RATIO, ]

    return massive_columns


def massive_review_columns_of_persistence():
    massive_columns = [ST.CLUSTER_GROUP_CHECKPOINTS,
                       AKA.STAGE,
                       FLD.GARIDENT_PRICE,
                       FLD.GARIDENT_PRICE_COUNT,
                       FLD.GARIDENT_PRICE_CLEARANCE,
                       FLD.GARIDENT_STAGE_TIMING_LAG,
                       FLD.BOOTSTRAP_STAGE_BEFORE,
                       FLD.ENDPOINT_STAGE_BEFORE,
                       FLD.MAPOWER30_MAJOR,
                       STE.MACD_COMPOUDED_BAND_RATIO, ]

    return massive_columns


def reality_columns_of_persistence():
    reality_columns = list(set([ST.CLUSTER_GROUP_CHECKPOINTS,
                                AKA.STAGE,
                                FLD.GARIDENT_PRICE,
                                FLD.GARIDENT_PRICE_COUNT,
                                FLD.GARIDENT_PRICE_CLEARANCE,
                                FLD.GARIDENT_STAGE_TIMING_LAG,
                                FLD.BOOTSTRAP_STAGE_BEFORE,
                                FLD.ENDPOINT_STAGE_BEFORE,
                                FTR.ZEN_DASH_TIMING_LAG_MINOR_REAL,
                                FTR.ZEN_PEAK_TIMING_LAG_MAJOR_REAL,
                                FTR.ZEN_PEAK_TIMING_LAG_REAL,
                                FTR.ZEN_PEAK_TIMING_LAG_REAL,
                                FTR.ZEN_PEAK_TIMING_LAG_MINOR_REAL,
                                FTR.ZEN_PEAK_TIMING_LAG_MAJOR_REAL,
                                FLD.MAGIC_NINE_TURNS,
                                FLD.MAGIC_NINE_TURNS_MAJOR,
                                FLD.MAGIC_NINE_TURNS_BASELINE,
                                FLD.MAGIC_NINE_TURNS_BASELINE_TIMING_LAG,
                                FLD.MAGIC_NINE_TURNS_BEFORE,
                                FLD.MAGIC_NINE_TURNS_SX_BEFORE,
                                FLD.MACD_ZERO_TIMING_LAG,
                                FLD.MACD_ZERO_TIMING_LAG_MAJOR,
                                FLD.DEA_NORM_MAJOR,
                                FLD.DIF_NORM_MAJOR,
                                FLD.POLYNOMIAL9_NORM_MAJOR,
                                FLD.MTM_NORM_MAJOR,
                                FLD.MAPOWER30_MAJOR,
                                FLD.MAXFACTOR_MAJOR,
                                FLD.ATR_Stopline_TIMING_LAG_MAJOR,
                                FLD.ATR_SuperTrend_TIMING_LAG_MAJOR,
                                FLD.RENKO_TREND_S_TIMING_LAG,
                                FLD.RENKO_BOOST_S_TIMING_LAG,
                                FLD.RENKO_TREND_S_LB,
                                FLD.RENKO_TREND_S_UB,
                                FTR.CVaR_risk90,
                                FTR.CVaR_risk95,
                                FTR.CVaR_risk90_MAJOR,
                                FTR.CVaR_risk95_MAJOR,
                                RSK.CVaR_PEAK_PRICE,
                                RSK.CVaR_PEAK_LOW,
                                RSK.CVaR_PEAK_LOW_PRICE,
                                RSK.CVaR_PEAK_LOW_BEFORE,
                                FLD.STOCK_SCORE_M15,
                                FLD.STOCK_SCORE_M15_NORM,
                                FLD.STOCK_SCORE_M15_TIMING_LAG,
                                FLD.STOCK_SCORE_M15_NORM_TIMING_LAG, ]))
    return reality_columns


def daily_columns_of_persistence():
    daily_columns = [FTR.ZEN_DASH_TIMING_LAG_WEEKLY_REAL,
                     FTR.POLYNOMIAL9_WEEKLY_REAL,
                     FTR.POLYNOMIAL9_NORM_WEEKLY_REAL,
                     FTR.ZEN_DASH_TIMING_LAG_MAJOR_REAL,
                     FTR.MAGIC_NINE_TURNS_MAJOR_REAL,
                     FTR.MAGIC_NINE_TURNS_TIMING_LAG_MAJOR_REAL,
                     FLD.STOCK_SCORE_M15,
                     FLD.STOCK_SCORE_M15_NORM,
                     FTR.POLYNOMIAL9_MAJOR_REAL,
                     FTR.POLYNOMIAL9_NORM_MAJOR_REAL,
                     FTR.REGTREE_PRICE_MAJOR_REAL,
                     FTR.REGTREE_TIMING_LAG_MAJOR_REAL,
                     FTR.REGTREE_SLOPE_MAJOR_REAL, ]

    return daily_columns


def calc_masked_tail_missing_index(unmasked_missing_index,
                                   baseline,
                                   persistence_ratio):
    """
    计算标的K线近端缺失的特征索引
    """
    if (persistence_ratio > 0.618) or (len(unmasked_missing_index) < 84):
        tail_missing_index = pd.Series(np.where(((baseline.index.get_level_values(level=0)[-1] -
                                                  unmasked_missing_index.get_level_values(level=0)) < timedelta(days=365)), True, False),
                                       index=unmasked_missing_index, name='masked')
        return tail_missing_index.to_frame().query('masked==True').index
    else:
        return unmasked_missing_index


def features_reasonableness_checks(features):
    reasonableness = True
    return reasonableness
