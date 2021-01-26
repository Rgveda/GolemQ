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
"""
这里定义的是一些分类功能
"""
import pandas as pd
import numpy as np
from GolemQ.utils.parameter import (
    AKA, 
    INDICATOR_FIELD as FLD, 
    TREND_STATUS as ST,
    FEATURES as FTR,
)


def find_fratcal_fullstack(features:pd.DataFrame=None,) -> pd.DataFrame:
    """
    分析并且排除掉咸鱼票
    """
    if (FLD.FULLSTACK_TIMING_LAG in features.columns):
        ret_codelist_fullstack = features.query('({}>0) & ({}>0)'.format(FLD.FULLSTACK_TIMING_LAG,
                                                                         FLD.MAINFEST_DOWNRISK_TIMING_LAG)).copy()
        codelist_fullstack = [index[1] for index, symbol in ret_codelist_fullstack.iterrows()]
    else:
        codelist_fullstack = []

    if (len(ret_codelist_fullstack) > 0):
        ret_features = features.drop(ret_codelist_fullstack.index)
    else:
        ret_features = features
    return codelist_fullstack, ret_codelist_fullstack, ret_features


def find_fratcal_halfstack(features:pd.DataFrame=None,) -> pd.DataFrame:
    """
    分析并且排除掉咸鱼票
    """
    if (FLD.HALFSTACK_TIMING_LAG in features.columns):
        ret_codelist_halfstack = features.query('({}>0) & ({}>0)'.format(FLD.HALFSTACK_TIMING_LAG,
                                                                         FLD.MAINFEST_DOWNRISK_TIMING_LAG)).copy()
        codelist_halfstack = [index[1] for index, symbol in ret_codelist_halfstack.iterrows()]
    else:
        codelist_halfstack = []

    if (len(ret_codelist_halfstack) > 0):
        ret_features = features.drop(ret_codelist_halfstack.index)
    else:
        ret_features = features
    return codelist_halfstack, ret_codelist_halfstack, ret_features


def find_fratcal_downrisk(features:pd.DataFrame=None,) -> pd.DataFrame:
    """
    分析并且排除掉主跌浪票
    """
    if (FLD.MAINFEST_DOWNRISK_TIMING_LAG in features.columns):
        ret_codelist_downrisk = features.query('({}<0)'.format(FLD.MAINFEST_DOWNRISK_TIMING_LAG)).copy()
        codelist_downrisk = [index[1] for index, symbol in ret_codelist_downrisk.iterrows()]
    else:
        codelist_downrisk = []

    if (len(ret_codelist_downrisk) > 0):
        ret_features = features.drop(ret_codelist_downrisk.index)
    else:
        ret_features = features
    return codelist_downrisk, ret_codelist_downrisk, ret_features


def tag_fratcal_saltedfish(features:pd.DataFrame=None,) -> pd.DataFrame:
    """
    标记咸鱼票
    """
    features[FTR.STRATEGY_TYPE] = np.where(((features[FLD.REGTREE_TIMING_LAG] < -400) & \
                                           (features[FLD.HMA10] < 9.27)) | \
                                           ((features[FLD.REGTREE_TIMING_LAG] > 400) & \
                                           (features[FLD.HMA10] < 9.27)), 
                                           'saltedfish,' + features[FTR.STRATEGY_TYPE], 
                                           features[FTR.STRATEGY_TYPE])
    return features


def find_fratcal_saltedfish(features:pd.DataFrame=None,) -> pd.DataFrame:
    """
    分析并且排除掉咸鱼票
    """
    if (FLD.REGTREE_TIMING_LAG in features.columns):
        ret_codelist_slatfish = features.query('(({}<-400) & ({}<9.27)) | (({}>400) & ({}<9.27))'.format(FLD.REGTREE_TIMING_LAG,
                                                                                                         FLD.HMA10,
                                                                                                         FLD.REGTREE_TIMING_LAG,
                                                                                                         FLD.HMA10)).copy()
    else:
        codelist_slatfish = []
        ret_codelist_slatfish = features.head(2).copy()
        ret_codelist_slatfish = ret_codelist_slatfish.drop(ret_codelist_slatfish.index)
        return codelist_slatfish, ret_codelist_slatfish, features

    codelist_slatfish = [index[1] for index, symbol in ret_codelist_slatfish.iterrows()]
    if (len(ret_codelist_slatfish) > 0):
        ret_features = features.drop(ret_codelist_slatfish.index)
    else:
        ret_features = features
    return codelist_slatfish, ret_codelist_slatfish, ret_features


def tag_fratcal_weakshort(features:pd.DataFrame=None,) -> pd.DataFrame:
    """
    标记弱势票
    """
    if (FLD.REGTREE_TIMING_LAG in features.columns):
        regtree_timing_lag_threshold = features[FLD.REGTREE_TIMING_LAG].quantile(0.25)
        features[FTR.STRATEGY_TYPE] = np.where((features[FLD.REGTREE_TIMING_LAG] < regtree_timing_lag_threshold) & \
                                               (features[FLD.DRAWDOWN_RATIO_MAJOR] < 0.84) & \
                                               (features[FLD.BOLL_LB_HMA5_TIMING_LAG] < features[FLD.MACD_CROSS_JX_BEFORE]) & \
                                               (features[FLD.BOLL_LB_HMA5_TIMING_LAG] < features[FLD.MACD_CROSS_JX_BEFORE]), 
                                               'weakshort,' + features[FTR.STRATEGY_TYPE], 
                                               features[FTR.STRATEGY_TYPE])
    else:
        features[FTR.STRATEGY_TYPE] = np.where((features[FLD.REGTREE_TIMING_LAG] < -72) & \
                                               (features[FLD.DRAWDOWN_RATIO_MAJOR] < 0.84) & \
                                               (features[FLD.BOLL_LB_HMA5_TIMING_LAG] < features[FLD.MACD_CROSS_JX_BEFORE]) & \
                                               (features[FLD.BOLL_LB_HMA5_TIMING_LAG] < features[FLD.MACD_CROSS_JX_BEFORE]),
                                               'weakshort,' + features[FTR.STRATEGY_TYPE], 
                                               features[FTR.STRATEGY_TYPE])
    return features


def find_fratcal_weakshort(features:pd.DataFrame=None,) -> pd.DataFrame:
    """
    分析并且排除掉弱势票
    """
    if (FLD.REGTREE_TIMING_LAG in features.columns):
        regtree_timing_lag_threshold = features[FLD.REGTREE_TIMING_LAG].quantile(0.25)
        if (regtree_timing_lag_threshold < -36):
            ret_codelist_short = features.query('({}<{}) & ({}<0.84) & ({}<{}) & ({}<{})'.format(FLD.REGTREE_TIMING_LAG,
                                                                                                  regtree_timing_lag_threshold,
                                                                        FLD.DRAWDOWN_RATIO_MAJOR,
                                                                        FLD.BOLL_LB_HMA5_TIMING_LAG,
                                                                        FLD.MACD_CROSS_JX_BEFORE,
                                                                        FLD.BOLL_LB_HMA5_TIMING_LAG,
                                                                        FLD.MACD_CROSS_JX_BEFORE,)).copy()
        else:
            ret_codelist_short = features.query('({}<-72) & ({}<0.84) & ({}<{}) & ({}<{})'.format(FLD.REGTREE_TIMING_LAG,
                                                                        FLD.DRAWDOWN_RATIO_MAJOR,
                                                                        FLD.BOLL_LB_HMA5_TIMING_LAG,
                                                                        FLD.MACD_CROSS_JX_BEFORE,
                                                                        FLD.BOLL_LB_HMA5_TIMING_LAG,
                                                                        FLD.MACD_CROSS_JX_BEFORE,)).copy()
    codelist_short = [index[1] for index, symbol in ret_codelist_short.iterrows()]
    if (len(ret_codelist_short) > 0):
        ret_features = features.drop(ret_codelist_short.index)
    else:
        ret_features = features
    return codelist_short, ret_codelist_short, ret_features


def find_fratcal_bootstrap_ground_zero(features:pd.DataFrame=None,) -> pd.DataFrame:
    """
    分析并且提取出买点票
    """
    if (ST.BOOTSTRAP_GROUND_ZERO in features.columns):
        ret_codelist_ground_zero = features.query('({}>0) & ({}>0.0002)'.format(FLD.BOOTSTRAP_GROUND_ZERO_TIMING_LAG,
                                                                                FLD.ZEN_DASH_RETURNS)).copy()
        codelist_ground_zero = [index[1] for index, symbol in ret_codelist_ground_zero.iterrows()]
    else:
        codelist_ground_zero = []

    if (len(ret_codelist_ground_zero) > 0):
        ret_features = features.drop(ret_codelist_ground_zero.index)
    else:
        ret_features = features
    return codelist_ground_zero, ret_codelist_ground_zero, ret_features


def find_fratcal_bootstrap_combo(features:pd.DataFrame=None,) -> pd.DataFrame:
    """
    分析并且提取出买点票
    """
    if (ST.BOOTSTRAP_COMBO in features.columns):
        ret_codelist_combo = features.query('({}==1) | ({}==2) | ({}==-1) | ({}==-2) | ({}==-3) | ({}==-4)'.format(FLD.BOOTSTRAP_COMBO_TIMING_LAG,
                                                                                                                   FLD.BOOTSTRAP_COMBO_TIMING_LAG,
                                                                                                                   FLD.BOOTSTRAP_COMBO_TIMING_LAG,
                                                                                                                   FLD.BOOTSTRAP_COMBO_TIMING_LAG,
                                                                                                                   FLD.BOOTSTRAP_COMBO_TIMING_LAG,
                                                                                                                   FLD.BOOTSTRAP_COMBO_TIMING_LAG,)).copy()

    #ret_codelist_treasure = features.query('({}>0) &
    #({}>{})'.format(FLD.DEA_ZERO_TIMING_LAG,
    #                                                                 FLD.MAINFEST_UPRISING_COEFFICIENT,
    #                                                                 FLD.DEA_ZERO_TIMING_LAG))
    #ret_codelist_treasure1 = features.query('({}>0) &
    #({}>{})'.format(FLD.DEA_ZERO_TIMING_LAG,
    #                                                                  FLD.MAPOWER30_TIMING_LAG_MAJOR,
    #                                                                  FLD.DEA_ZERO_TIMING_LAG))
    #ret_codelist_treasure2 = features.query('({}>0) & ({}>0) & ({}>0) &
    #({}>{})'.format(FLD.DEA_ZERO_TIMING_LAG,
    #                                                                                FLD.MAINFEST_UPRISING_COEFFICIENT,
    #                                                                                FLD.MAINFEST_UPRISING_TIMING_LAG,
    #                                                                                FLD.HMAPOWER120_TIMING_LAG_MAJOR,
    #                                                                                FLD.DEA_ZERO_TIMING_LAG))
    #ret_codelist_treasure = pd.concat([ret_codelist_treasure,
    #ret_codelist_treasure1, ret_codelist_treasure2],
    #                                    axis=0, sort=True).drop_duplicates()
    #ret_codelist_treasure = ret_codelist_treasure.query('(({}>{}) | ({}>{})) &
    #({}>{}) & ({}>{}) & ({}>{}) & ~(({}<0) & ({}>0.618)) & ~(({}<0) &
    #({}<-0.5)) & ~(({}<-36) &
    #({}>{}))'.format(FLD.HMAPOWER120_TIMING_LAG_MAJOR,
    #                                                                                FLD.DEA_ZERO_TIMING_LAG,
    #                                                                                FLD.MAPOWER30_TIMING_LAG_MAJOR,
    #                                                                                FLD.DEA_ZERO_TIMING_LAG,
    #                                                                                FLD.REGTREE_TIMING_LAG,
    #                                                                                FLD.DEA_ZERO_TIMING_LAG,
    #                                                                                FLD.ATR_LB,
    #                                                                                FLD.POLYNOMIAL9_CHECKOUT_ATR_LB,
    #                                                                                FLD.ATR_LB,
    #                                                                                FLD.MAPOWER30_CHECKOUT_ATR_LB,
    #                                                                                FLD.MAPOWER30_TIMING_LAG_MAJOR,
    #                                                                                FLD.MAPOWER30_CHECKOUT_MAPOWER30_MAJOR,
    #                                                                      FLD.REGTREE_TIMING_LAG,
    #                                                                      ST.PREDICT_LONG,
    #                                                                      FLD.DEADPOOL_REMIX_TIMING_LAG,
    #                                                                      FLD.POLYNOMIAL9_CHECKOUT_ATR_LB,
    #                                                                      FLD.ATR_LB,))

    #ret_polynomial9_empirical = features.query('({}>0) & ({}<{}) & ({}>-200) &
    #~(({}<0) & ({}<-0.5)) & ~(({}<-36) &
    #({}>{}))'.format(FTR.POLYNOMIAL9_EMPIRICAL_TIMING_LAG,
    #                                                                      FTR.POLYNOMIAL9_EMPIRICAL_TIMING_LAG,
    #                                                                      FLD.MACD_CROSS_SX_BEFORE,
    #                                                                      FLD.REGTREE_TIMING_LAG,
    #                                                                      FLD.REGTREE_TIMING_LAG,
    #                                                                      ST.PREDICT_LONG,
    #                                                                      FLD.DEADPOOL_REMIX_TIMING_LAG,
    #                                                                      FLD.POLYNOMIAL9_CHECKOUT_ATR_LB,
    #                                                                      FLD.ATR_LB,))

    #ret_codelist_combo = pd.concat([ret_codelist_combo,
    #                                ret_codelist_treasure,
    #                                ret_polynomial9_empirical],
    #                                axis=0, sort=True).drop_duplicates()
    codelist_combo = [index[1] for index, symbol in ret_codelist_combo.iterrows()]
    if (len(ret_codelist_combo) > 0):
        ret_features = features.drop(ret_codelist_combo.index)
    else:
        ret_features = features
    return codelist_combo, ret_codelist_combo, ret_features


def find_action_long(features:pd.DataFrame=None,) -> pd.DataFrame:
    """
    分析并且提取出买点票
    """
    ret_codelist_action = features.query('({}<{}) & ((({}>0) | ({}>0)) & ~(({}<0) & ({}<0)) & (({}==1) | ({}==1) | ({}>0) | ({}>0))) | (({}>0) & ({}<0))'.format(FLD.HMA10,
                                                                                                                                                                 FLD.REGTREE_PRICE,
                                                                FTR.BOOTSTRAP_CONFIRM_TIMING_LAG,
                                                                FTR.POLYNOMIAL9_DUAL_TIMING_LAG,
                                                                FTR.POLYNOMIAL9_DUAL_TIMING_LAG,
                                                                FLD.POLYNOMIAL9_TIMING_LAG,
                                                                FTR.POLYNOMIAL9_DUAL_TIMING_LAG,
                                                                FLD.POLYNOMIAL9_TIMING_LAG,
                                                                ST.HYPER_PUNCH,
                                                                ST.BOOTSTRAP_I,
                                                                ST.PREDICT_LONG,
                                                                FLD.MACD_ZERO_TIMING_LAG)).copy()
    ret_codelist_action1 = features.query('({}>0) & ({}>0) & ({}<=6) & ({}<0.0168)'.format(ST.ZEN_BUY3,
                                                                                           FLD.BOOTSTRAP_COMBO_TIMING_LAG,
                                                                                 FLD.BOOTSTRAP_COMBO_TIMING_LAG,
                                                    FLD.POLY9_REGTREE_DIVERGENCE,)).copy()
    ret_codelist_action2 = features.query('({}>0) & ((({}>0) & ({}<=8)) | (({}>0) & ({}<=6))) & ({}<0.0168)'.format(ST.ZEN_BUY3,
                                                                                           FLD.ZEN_DASH_TIMING_LAG,
                                                                                 FLD.ZEN_DASH_TIMING_LAG,
                                                                                 FLD.ZEN_PEAK_TIMING_LAG,
                                                                                 FLD.ZEN_PEAK_TIMING_LAG,
                                                    FLD.POLY9_REGTREE_DIVERGENCE,)).copy()
    ret_polynomial9_empirical = features.query('({}>0) & ({}<61) & ({}>-200) & ~(({}<0) & ({}<-0.5)) & ~(({}<-36) & ({}>{}))'.format(FTR.POLYNOMIAL9_EMPIRICAL,
                                                                         FTR.POLYNOMIAL9_EMPIRICAL,
                                                                          FLD.REGTREE_TIMING_LAG,
                                                                          FLD.REGTREE_TIMING_LAG,
                                                                          ST.PREDICT_LONG,
                                                                          FLD.DEADPOOL_REMIX_TIMING_LAG,
                                                                          FLD.POLYNOMIAL9_CHECKOUT_ATR_LB,
                                                                          FLD.ATR_LB,))
    ret_polynomial9_empirical1 = features.query('({}>0) & ({}>33) & (({}<{}) | (({} < -12) & ({} < 35) & ({} < -150))) & ({}>-200) & ~(({}<0) & ({}<-0.5)) & ~(({}<-36) & ({}>{}))'.format(FTR.POLYNOMIAL9_EMPIRICAL,
                                                                   FTR.POLYNOMIAL9_EMPIRICAL,
                                                                   FLD.POLYNOMIAL9_CHECKOUT_ATR_LB,
                                                                   FLD.ATR_LB,
                                                                   FLD.POLYNOMIAL9_TIMING_GAP,
                                                                   FLD.BOLL_JX_RSI,
                                                                   FLD.BOLL_JX_MAXFACTOR,
                                                                          FLD.REGTREE_TIMING_LAG,
                                                                          FLD.REGTREE_TIMING_LAG,
                                                                          ST.PREDICT_LONG,
                                                                          FLD.DEADPOOL_REMIX_TIMING_LAG,
                                                                          FLD.POLYNOMIAL9_CHECKOUT_ATR_LB,
                                                                          FLD.ATR_LB,))
    ret_codelist_action = pd.concat([ret_codelist_action, 
                                     ret_codelist_action1,
                                     ret_codelist_action2,
                                     ret_polynomial9_empirical,
                                     ret_polynomial9_empirical1,],
                                    axis=0, sort=True).drop_duplicates()
    ret_codelist_action = ret_codelist_action.query('({}>0) | ({}>0) | ({}<0)'.format(FTR.POLYNOMIAL9_EMPIRICAL,
                                                                                      ST.ZEN_BUY3,
                                                                                      FLD.MACD_ZERO_TIMING_LAG)) 
    codelist_action = [index[1] for index, symbol in ret_codelist_action.iterrows()]
    if (len(ret_codelist_action) > 0):
        ret_features = features.drop(ret_codelist_action.index)
    else:
        ret_features = features
    return codelist_action, ret_codelist_action, ret_features


def tag_action_sellshort(features:pd.DataFrame=None,) -> pd.DataFrame:
    """
    标记卖点票
    """
    features[FTR.STRATEGY_TYPE] = np.where((features[ST.BOOTSTRAP_ENDPOINTS] < 0) | \
                                           (features[FLD.BOOTSTRAP_ENDPOINTS_BEFORE] < 4), 
                                           'sell,' + features[FTR.STRATEGY_TYPE], 
                                           features[FTR.STRATEGY_TYPE])
    return features


def find_action_buylong(features:pd.DataFrame=None,) -> pd.DataFrame:
    """
    分析并且提取出买点票
    """
    ret_codelist_action = features.query('''(({} > 0) & \
({}>0)) | \
(({} > 0) & \
({} > 0) & \
({} > 0) & \
({} <= 12) & \
({} < 0) & \
(({} - {}) < 0.382)) | \
({}>0)'''.format(FLD.BOOTSTRAP_GROUND_ZERO_TIMING_LAG,
        ST.BOOTSTRAP_GROUND_ZERO,
        FTR.POLYNOMIAL9_EMPIRICAL,
        FLD.BOOTSTRAP_GROUND_ZERO_TIMING_LAG,
        FLD.ZEN_DASH_TIMING_LAG,
        FLD.ZEN_DASH_TIMING_LAG,
        FLD.MACD_ZERO_TIMING_LAG,
        FLD.DRAWDOWN_RATIO,
        FLD.DRAWDOWN_RATIO_MAJOR,
        ST.BOOTSTRAP_I)).copy()

    codelist_action = [index[1] for index, symbol in ret_codelist_action.iterrows()]
    if (len(ret_codelist_action) > 0):
        ret_features = features.drop(ret_codelist_action.index)
    else:
        ret_features = features
    return codelist_action, ret_codelist_action, ret_features


def find_action_sellshort(features:pd.DataFrame=None,) -> pd.DataFrame:
    """
    分析并且提取出卖点票
    """
    if (ST.BOOTSTRAP_ENDPOINTS in features.columns):
        ret_codelist_short = features.query('''({} > 0)'''.format(ST.BOOTSTRAP_ENDPOINTS,)).copy()
    elif (FLD.BOOTSTRAP_GROUND_ZERO_TIMING_LAG in features.columns):
        ret_codelist_short = features.query('''
(({} == -1) & \
        ~(({} > 0) & \
        ({} > 0))) | \
    (({} == -1) & \
        ({} == -1)) | \
        (({} > 0) & \
        ({} < 0) & \
        ({} >= -6) & \
        ({} < 0) & \
        ~(({} > 0) & \
        ({} > 0))) | \
        (((({} == -1) & \
        ({} < 0)) | \
        ((({} < 0) & \
        ({} == -1))) | \
        (({} == 0) & \
        ({} < 0) & \
        ({} < 0))) & \
        ({} < 0))
        '''.format(FLD.BOOTSTRAP_GROUND_ZERO_TIMING_LAG,
                   FLD.ZEN_DASH_TIMING_LAG,
                   FLD.ZEN_PEAK_TIMING_LAG,
                   FLD.ZEN_PEAK_TIMING_LAG,
                   FLD.ZEN_DASH_TIMING_LAG,
                   FLD.BOOTSTRAP_GROUND_ZERO_TIMING_LAG,
                   ST.BOOTSTRAP_GROUND_ZERO,
                   FLD.ZEN_DASH_TIMING_LAG,
                   FLD.ZEN_DASH_TIMING_LAG,
                   FLD.ZEN_DASH_TIMING_LAG,
                   FLD.ZEN_PEAK_TIMING_LAG,
                   FLD.ZEN_DASH_TIMING_LAG,
                   FLD.ZEN_DASH_TIMING_LAG_MAJOR,
                   FLD.ZEN_DASH_TIMING_LAG,
                   FLD.ZEN_DASH_TIMING_LAG_MAJOR,
                   FLD.MAPOWER30_PEAK_HIGH_BEFORE,
                   FLD.ZEN_PEAK_TIMING_LAG,
                   FLD.ZEN_DASH_TIMING_LAG_MAJOR,
                   FLD.ZEN_PEAK_TIMING_LAG_MAJOR)).copy()
    else:
        ret_codelist_short = features.query('').copy()
    codelist_short = [index[1] for index, symbol in ret_codelist_short.iterrows()]
    if (len(ret_codelist_short) > 0):
        ret_features = features.drop(ret_codelist_short.index)
    else:
        ret_features = features
    return codelist_short, ret_codelist_short, ret_features


def find_fratcal_short(features:pd.DataFrame=None,) -> pd.DataFrame:
    """
    分析并且提取出卖点票
    """
    ret_codelist_short = features.query('({} < 0)'.format(FLD.BOOTSTRAP_GROUND_ZERO_TIMING_LAG)).copy()
    codelist_short = [index[1] for index, symbol in ret_codelist_short.iterrows()]
    if (len(ret_codelist_short) > 0):
        ret_features = features.drop(ret_codelist_short.index)
    else:
        ret_features = features
    return codelist_short, ret_codelist_short, ret_features


def find_action_hyper_punch(features:pd.DataFrame=None,) -> pd.DataFrame:
    """
    分析并且提取出买点票
    """
    ret_codelist_hyper_punch = features.query('({}>0) & ({}>{}) & ({}>{}) & ({}>{})'.format(FLD.MAPOWER30_TIMING_LAG_MAJOR,
                                                                                            FLD.MAPOWER30_TIMING_LAG_MAJOR,
                                                                                            FLD.HMAPOWER120_TIMING_LAG_MAJOR,
                                                                                            FLD.ATR_LB,
                                                                                            FLD.POLYNOMIAL9_CHECKOUT_ATR_LB,
                                                                                    FLD.ZEN_PEAK_TIMING_LAG,
                                                                                    FLD.MAPOWER30_TIMING_LAG_MAJOR,))
    ret_codelist_hyper_punch1 = features.query('({}>0) & ({}>{}) & ({}>{}) & ({}>{}) & ({}>{}) & ({}>{})'.format(FLD.MAPOWER30_TIMING_LAG_MAJOR,
                                                                                                                 FLD.MAPOWER30_TIMING_LAG_MAJOR,
                                                                                                                 FLD.HMAPOWER120_TIMING_LAG_MAJOR,
                                                                                                                FLD.ATR_LB,
                                                                                                                FLD.POLYNOMIAL9_CHECKOUT_ATR_LB,
                                                                                                                FLD.ATR_LB,
                                                                                                                FLD.MAPOWER30_CHECKOUT_ATR_LB,
                                                                                    FLD.MAPOWER30_PEAK_LOW_BEFORE,
                                                                                    FLD.MAPOWER30_TIMING_LAG_MAJOR,
                                                                                    FLD.BOLL_LB_HMA5_TIMING_LAG,
                                                                                    FLD.MAPOWER30_TIMING_LAG_MAJOR,))
    ret_codelist_hyper_punch2 = features.query('({}>0) & ({}>{}) & (({}+{})<0) & ({}>{}) & ({}>{}) & ({}>{})'.format(FLD.MAPOWER30_TIMING_LAG_MAJOR,
                                                                                                                    FLD.MAPOWER30_TIMING_LAG_MAJOR,
                                                                                                                    FLD.HMAPOWER120_TIMING_LAG_MAJOR,
                                                                                                                FLD.MAPOWER30_PEAK_LOW_BEFORE,
                                                                                                                FLD.POLYNOMIAL9_CHECKOUT_TIMING_LAG,
                                                                                                                FLD.ATR_LB,
                                                                                                                FLD.MAPOWER30_CHECKOUT_ATR_LB,
                                                                                    FLD.MAPOWER30_PEAK_LOW_BEFORE,
                                                                                    FLD.MAPOWER30_TIMING_LAG_MAJOR,
                                                                                    FLD.BOLL_LB_HMA5_TIMING_LAG,
                                                                                    FLD.MAPOWER30_TIMING_LAG_MAJOR,))
    ret_codelist_hyper_punch = pd.concat([ret_codelist_hyper_punch, 
                                          ret_codelist_hyper_punch1, 
                                          ret_codelist_hyper_punch2,],
                                         axis=0, sort=True).drop_duplicates()
    codelist_hyper_punch = [index[1] for index, symbol in ret_codelist_hyper_punch.iterrows()]
    if (len(ret_codelist_hyper_punch) > 0):
        ret_features = features.drop(ret_codelist_hyper_punch.index)
    else:
        ret_features = features
    return codelist_hyper_punch, ret_codelist_hyper_punch, ret_features
