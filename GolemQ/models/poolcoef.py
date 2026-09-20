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
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
import numpy as np
import pandas as pd

try:
    from GolemQ.core.constants import (
        AKA,
        FIELD as FLD,
    )
except ImportError:
    class AKA:
        OPEN = 'open'
        HIGH = 'high'
        LOW = 'low'
        CLOSE = 'close'
        COST5_PRICE = 'COST5_PRICE'
        COST15_PRICE = 'COST15_PRICE'
        SAMPLE_ENTROPY_DUMMY = 'SED'
        RETURN_COST95 = 'RETURN_COST95'

    class FLD:
        ZEN_DASH_TIMING_LAG_DUMMY = 'ZEN_DASH_TIMING_LAG_DUMMY'
        RENKO_BOOST_S_TIMING_LAG = 'RENKO_BOOST_S_TIMING_LAG'
        MACD_MAJOR = 'MACD_MAJOR'
        MACD_CROSS_SX_BEFORE_MAJOR = 'MACD_CROSS_SX_BEFORE_MAJOR'
        CHARTSCAN_BUY_BEFORE_MAJOR_DUMMY = 'CHARTSCAN_BUY_BEFORE_MAJOR_DUMMY'
        CHARTSCAN_BUY_BEFORE_DUMMY = 'CHARTSCAN_BUY_BEFORE_DUMMY'
        CHARTSCAN_BUY_BEFORE_MINOR_DUMMY = 'CHARTSCAN_BUY_BEFORE_MINOR_DUMMY'
        DEA_ZERO_TIMING_LAG_MAJOR = 'DEA_ZERO_TIMING_LAG_MAJOR'

try:
    from GolemQ.models.alias import (
        TRD,
        LTT,
    )
except ImportError:
    class TRD:
        UNIVERSAL_CLEARANCE_ENERGY = 'UNIVERSAL_CLEARANCE_ENERGY'
        OMNIPATH_UPRISING_I_COEF_CANDIDATE = 'OMNIPATH_UPRISING_I_COEF_CANDIDATE'
        OMNIPATH_UPRISING_I_COEF_UPRISING_TIMING_LAG = 'OMNIPATH_UPRISING_I_COEF_UPRISING_TIMING_LAG'

    class LTT:
        QUADRANT_LEVERAGE_MACD_TIMING_LAG_DUMMY = 'QUADRANT_LEVERAGE_MACD_TIMING_LAG_DUMMY'
        QUADRANT_PUSH_CREDIT = 'QUADRANT_PUSH_CREDIT'
        QUADRANT_TRIGGER_CREDIT = 'QUADRANT_TRIGGER_CREDIT'
        QUADRANT_ZEN_DASH_COUNT = 'QUADRANT_ZEN_DASH_COUNT'
        QUADRANT_M9T_DEADPOOL_COUNT = 'QUADRANT_M9T_DEADPOOL_COUNT'
        QUADRANT_LEVERAGE_POWERLINE_UPRISING_TIMING_LAG = 'QUADRANT_LEVERAGE_POWERLINE_UPRISING_TIMING_LAG'
        MAINSTREAM_LEADIN_BEFORE_MINOR = 'MAINSTREAM_LEADIN_BEFORE_MINOR'
        MAINSTREAM_LEADIN_BEFORE = 'MAINSTREAM_LEADIN_BEFORE'
        MAINSTREAM_LEADIN_BEFORE_MAJOR = 'MAINSTREAM_LEADIN_BEFORE_MAJOR'
        MAINSTREAM_LEADIN_VAR5 = 'MAINSTREAM_LEADIN_VAR5'
        MAINSTREAM_LEADIN_ENTRY = 'MAINSTREAM_LEADIN_ENTRY'
        ONE_QUADRANT_UPRISING_TIMING_LAG_MAJOR = 'ONE_QUADRANT_UPRISING_TIMING_LAG_MAJOR'
        TWO_QUADRANT_UPRISING_TIMING_LAG_MAJOR = 'TWO_QUADRANT_UPRISING_TIMING_LAG_MAJOR'
        BS_STAR_UP_BEFORE_MAJOR = 'BS_STAR_UP_BEFORE_MAJOR'
        BS_STAR_DOWN_BEFORE_MAJOR = 'BS_STAR_DOWN_BEFORE_MAJOR'

try:
    from GolemQ.portfolio.base import PFL
except ImportError:
    class PFL:
        POSITION_LEVEL = 'POSITION_LEVEL'


def calc_4Quad_push_credit(features):
    """
    Calculate quadrant push credit scores.

    Parameters
    ----------
    features : pd.DataFrame
        Multi-index DataFrame with feature columns.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame with QUADRANT_PUSH_CREDIT column added.
    """
    uninon_buy_ckpo = features[
        (features[LTT.QUADRANT_LEVERAGE_MACD_TIMING_LAG_DUMMY] > 1e-2) & \
        ((features[LTT.MAINSTREAM_LEADIN_BEFORE_MINOR] < 32) | \
        ((features[LTT.QUADRANT_LEVERAGE_MACD_TIMING_LAG_DUMMY] > 1e-12) & \
        (features[LTT.QUADRANT_ZEN_DASH_COUNT] > 0.618) & \
        (features[LTT.QUADRANT_ZEN_DASH_COUNT] > features[LTT.QUADRANT_M9T_DEADPOOL_COUNT] + 1.68)) | \
        ((features[LTT.QUADRANT_LEVERAGE_MACD_TIMING_LAG_DUMMY] > 1e-2) & \
        (features[TRD.UNIVERSAL_CLEARANCE_ENERGY] > features[TRD.OMNIPATH_UPRISING_I_COEF_CANDIDATE]) & \
        ((features[TRD.UNIVERSAL_CLEARANCE_ENERGY] * 1.68 > features[LTT.QUADRANT_LEVERAGE_MACD_TIMING_LAG_DUMMY]) | \
        ((features[FLD.ZEN_DASH_TIMING_LAG_DUMMY] > 1e-12) & \
        (features[TRD.UNIVERSAL_CLEARANCE_ENERGY] * 1.68 > features[FLD.ZEN_DASH_TIMING_LAG_DUMMY]))) & \
        (((features[LTT.QUADRANT_LEVERAGE_POWERLINE_UPRISING_TIMING_LAG] > 1e-12) & \
        (features[LTT.QUADRANT_LEVERAGE_MACD_TIMING_LAG_DUMMY] > features[LTT.QUADRANT_LEVERAGE_POWERLINE_UPRISING_TIMING_LAG])) | \
        (features[LTT.QUADRANT_LEVERAGE_MACD_TIMING_LAG_DUMMY] > 3))) | \
        ((features[LTT.QUADRANT_LEVERAGE_MACD_TIMING_LAG_DUMMY] > 1e-2) & \
        (features[FLD.RENKO_BOOST_S_TIMING_LAG].shift(1) < -1e-12) & \
        (features[FLD.RENKO_BOOST_S_TIMING_LAG] == 1) & \
        (((features[FLD.RENKO_BOOST_S_TIMING_LAG].shift(1) + features[LTT.QUADRANT_LEVERAGE_MACD_TIMING_LAG_DUMMY].shift(1)) < -1e-12) | \
        ((features[FLD.ZEN_DASH_TIMING_LAG_DUMMY] > 1e-12) & \
        (features[FLD.MACD_MAJOR].diff(4) > 1e-12) & \
        ((features[FLD.RENKO_BOOST_S_TIMING_LAG].shift(1) + features[FLD.ZEN_DASH_TIMING_LAG_DUMMY].shift(1)) < -1e-12))))) & \
        ((features[AKA.COST15_PRICE] > features[AKA.LOW]) | \
        ((features[FLD.DEA_ZERO_TIMING_LAG_MAJOR] > features[FLD.MACD_CROSS_SX_BEFORE_MAJOR]) & \
        ((features[AKA.COST15_PRICE] + features[LTT.MAINSTREAM_LEADIN_VAR5]) > features[AKA.CLOSE])) | \
        ((features[FLD.DEA_ZERO_TIMING_LAG_MAJOR] > features[FLD.MACD_CROSS_SX_BEFORE_MAJOR]) & \
        ((features[AKA.COST5_PRICE] + features[LTT.MAINSTREAM_LEADIN_VAR5]) > features[AKA.LOW])) | \
        ((features[FLD.DEA_ZERO_TIMING_LAG_MAJOR] > features[FLD.MACD_CROSS_SX_BEFORE_MAJOR]) & \
        ((features[AKA.COST15_PRICE] + features[LTT.MAINSTREAM_LEADIN_ENTRY]) > features[AKA.CLOSE])) | \
        ((features[FLD.DEA_ZERO_TIMING_LAG_MAJOR] > features[FLD.MACD_CROSS_SX_BEFORE_MAJOR]) & \
        ((features[AKA.COST5_PRICE] + features[LTT.MAINSTREAM_LEADIN_ENTRY]) > features[AKA.LOW])))
    ].copy()

    print(len(uninon_buy_ckpo) / len(features))

    if (len(uninon_buy_ckpo) / len(features) > 0.168):
        uninon_buy_ckpo = features[
            (features[LTT.QUADRANT_LEVERAGE_MACD_TIMING_LAG_DUMMY] > 1e-2) & \
            ((features[LTT.MAINSTREAM_LEADIN_BEFORE_MINOR] < 32) | \
            ((features[LTT.QUADRANT_LEVERAGE_MACD_TIMING_LAG_DUMMY] > 1e-12) & \
            (features[LTT.QUADRANT_ZEN_DASH_COUNT] > 0.618) & \
            (features[LTT.QUADRANT_ZEN_DASH_COUNT] > features[LTT.QUADRANT_M9T_DEADPOOL_COUNT] + 1.68)) | \
            ((features[LTT.QUADRANT_LEVERAGE_MACD_TIMING_LAG_DUMMY] > 1e-2) & \
            (features[TRD.UNIVERSAL_CLEARANCE_ENERGY] > features[TRD.OMNIPATH_UPRISING_I_COEF_CANDIDATE]) & \
            ((features[TRD.UNIVERSAL_CLEARANCE_ENERGY] * 1.68 > features[LTT.QUADRANT_LEVERAGE_MACD_TIMING_LAG_DUMMY]) | \
            ((features[FLD.ZEN_DASH_TIMING_LAG_DUMMY] > 1e-12) & \
            (features[TRD.UNIVERSAL_CLEARANCE_ENERGY] * 1.68 > features[FLD.ZEN_DASH_TIMING_LAG_DUMMY]))) & \
            (((features[LTT.QUADRANT_LEVERAGE_POWERLINE_UPRISING_TIMING_LAG] > 1e-12) & \
            (features[LTT.QUADRANT_LEVERAGE_MACD_TIMING_LAG_DUMMY] > features[LTT.QUADRANT_LEVERAGE_POWERLINE_UPRISING_TIMING_LAG])) | \
            (features[LTT.QUADRANT_LEVERAGE_MACD_TIMING_LAG_DUMMY] > 3))) | \
            ((features[LTT.QUADRANT_LEVERAGE_MACD_TIMING_LAG_DUMMY] > 1e-2) & \
            (features[FLD.RENKO_BOOST_S_TIMING_LAG].shift(1) < -1e-12) & \
            (features[FLD.RENKO_BOOST_S_TIMING_LAG] == 1) & \
            (((features[FLD.RENKO_BOOST_S_TIMING_LAG].shift(1) + features[LTT.QUADRANT_LEVERAGE_MACD_TIMING_LAG_DUMMY].shift(1)) < -1e-12) | \
            ((features[FLD.ZEN_DASH_TIMING_LAG_DUMMY] > 1e-12) & \
            (features[FLD.MACD_MAJOR].diff(4) > 1e-12) & \
            ((features[FLD.RENKO_BOOST_S_TIMING_LAG].shift(1) + features[FLD.ZEN_DASH_TIMING_LAG_DUMMY].shift(1)) < -1e-12))))) & \
            ((features[AKA.COST15_PRICE] > features[AKA.LOW]) | \
            ((features[FLD.DEA_ZERO_TIMING_LAG_MAJOR] > features[FLD.MACD_CROSS_SX_BEFORE_MAJOR]) & \
            ((features[AKA.COST15_PRICE] + features[LTT.MAINSTREAM_LEADIN_VAR5]) > features[AKA.CLOSE])) | \
            ((features[FLD.DEA_ZERO_TIMING_LAG_MAJOR] > features[FLD.MACD_CROSS_SX_BEFORE_MAJOR]) & \
            ((features[AKA.COST5_PRICE] + features[LTT.MAINSTREAM_LEADIN_VAR5]) > features[AKA.LOW])) | \
            ((features[FLD.DEA_ZERO_TIMING_LAG_MAJOR] > features[FLD.MACD_CROSS_SX_BEFORE_MAJOR]) & \
            ((features[AKA.COST15_PRICE] + features[LTT.MAINSTREAM_LEADIN_ENTRY]) > features[AKA.CLOSE])) | \
            ((features[FLD.DEA_ZERO_TIMING_LAG_MAJOR] > features[FLD.MACD_CROSS_SX_BEFORE_MAJOR]) & \
            ((features[AKA.COST5_PRICE] + features[LTT.MAINSTREAM_LEADIN_ENTRY]) > features[AKA.LOW]))) & \
            (((features[LTT.QUADRANT_LEVERAGE_MACD_TIMING_LAG_DUMMY] > 1e-2) & \
            (features[TRD.UNIVERSAL_CLEARANCE_ENERGY] > features[TRD.OMNIPATH_UPRISING_I_COEF_CANDIDATE]) & \
            ((features[TRD.UNIVERSAL_CLEARANCE_ENERGY] * 1.68 > features[LTT.QUADRANT_LEVERAGE_MACD_TIMING_LAG_DUMMY]) | \
            ((features[FLD.ZEN_DASH_TIMING_LAG_DUMMY] > 1e-12) & \
            (features[TRD.UNIVERSAL_CLEARANCE_ENERGY] * 1.68 > features[FLD.ZEN_DASH_TIMING_LAG_DUMMY]))) & \
            (((features[LTT.QUADRANT_LEVERAGE_POWERLINE_UPRISING_TIMING_LAG] > 1e-12) & \
            (features[LTT.QUADRANT_LEVERAGE_MACD_TIMING_LAG_DUMMY] > features[LTT.QUADRANT_LEVERAGE_POWERLINE_UPRISING_TIMING_LAG])) | \
            (features[LTT.QUADRANT_LEVERAGE_MACD_TIMING_LAG_DUMMY] > 3))) | \
            (features[PFL.POSITION_LEVEL] > 0.168) | \
            ((features[LTT.QUADRANT_LEVERAGE_MACD_TIMING_LAG_DUMMY] < (features[TRD.OMNIPATH_UPRISING_I_COEF_UPRISING_TIMING_LAG] + features[FLD.MACD_CROSS_SX_BEFORE_MAJOR])) & \
            (features[TRD.OMNIPATH_UPRISING_I_COEF_UPRISING_TIMING_LAG] > 1e-12)) | \
            ((features[LTT.QUADRANT_LEVERAGE_MACD_TIMING_LAG_DUMMY] > 1e-2) & \
            (features[FLD.RENKO_BOOST_S_TIMING_LAG].shift(1) < -1e-12) & \
            (features[FLD.RENKO_BOOST_S_TIMING_LAG] == 1) & \
            (((features[FLD.RENKO_BOOST_S_TIMING_LAG].shift(1) + features[LTT.QUADRANT_LEVERAGE_MACD_TIMING_LAG_DUMMY].shift(1)) < -1e-12) | \
            ((features[FLD.ZEN_DASH_TIMING_LAG_DUMMY] > 1e-12) & \
            (features[FLD.MACD_MAJOR].diff(4) > 1e-12) & \
            ((features[FLD.RENKO_BOOST_S_TIMING_LAG].shift(1) + features[FLD.ZEN_DASH_TIMING_LAG_DUMMY].shift(1)) < -1e-12)))))
        ].copy()
        print(len(uninon_buy_ckpo) / len(features))

        if (len(uninon_buy_ckpo) / len(features) < 0.168):
            uninon_buy_ckpo[LTT.QUADRANT_PUSH_CREDIT] = 1
        else:
            uninon_buy_ckpo[LTT.QUADRANT_PUSH_CREDIT] = 0
    else:
        uninon_buy_ckpo[LTT.QUADRANT_PUSH_CREDIT] = 2

    uninon_buy_ckpo[LTT.QUADRANT_PUSH_CREDIT] = uninon_buy_ckpo[LTT.QUADRANT_PUSH_CREDIT].astype(np.int16)

    uninon_buy_ckpo[LTT.QUADRANT_PUSH_CREDIT] = np.where(
        (uninon_buy_ckpo[LTT.MAINSTREAM_LEADIN_VAR5] > uninon_buy_ckpo[TRD.OMNIPATH_UPRISING_I_COEF_CANDIDATE]) | \
        (uninon_buy_ckpo[LTT.MAINSTREAM_LEADIN_ENTRY] > uninon_buy_ckpo[TRD.OMNIPATH_UPRISING_I_COEF_CANDIDATE]),
        np.where(
            ((uninon_buy_ckpo[LTT.MAINSTREAM_LEADIN_VAR5] > uninon_buy_ckpo[AKA.CLOSE]) | \
            (uninon_buy_ckpo[LTT.MAINSTREAM_LEADIN_ENTRY] > uninon_buy_ckpo[AKA.CLOSE])) & \
            (uninon_buy_ckpo[AKA.SAMPLE_ENTROPY_DUMMY] < 0.0928),
            uninon_buy_ckpo[LTT.QUADRANT_PUSH_CREDIT] + 2,
            uninon_buy_ckpo[LTT.QUADRANT_PUSH_CREDIT] + 1,
        ),
        uninon_buy_ckpo[LTT.QUADRANT_PUSH_CREDIT]
    )
    uninon_buy_ckpo[LTT.QUADRANT_PUSH_CREDIT] = np.where(
        (uninon_buy_ckpo[TRD.UNIVERSAL_CLEARANCE_ENERGY] > uninon_buy_ckpo[TRD.OMNIPATH_UPRISING_I_COEF_CANDIDATE]),
        uninon_buy_ckpo[LTT.QUADRANT_PUSH_CREDIT] + 1,
        uninon_buy_ckpo[LTT.QUADRANT_PUSH_CREDIT]
    )
    uninon_buy_ckpo[LTT.QUADRANT_PUSH_CREDIT] = np.where(
        (uninon_buy_ckpo[LTT.QUADRANT_LEVERAGE_MACD_TIMING_LAG_DUMMY] > 1e-12) & \
        (uninon_buy_ckpo[LTT.QUADRANT_ZEN_DASH_COUNT] > 0.618) & \
        (uninon_buy_ckpo[LTT.QUADRANT_ZEN_DASH_COUNT] > uninon_buy_ckpo[LTT.QUADRANT_M9T_DEADPOOL_COUNT] + 1.68),
        uninon_buy_ckpo[LTT.QUADRANT_PUSH_CREDIT] + 1,
        uninon_buy_ckpo[LTT.QUADRANT_PUSH_CREDIT]
    )
    uninon_buy_ckpo[LTT.QUADRANT_PUSH_CREDIT] = np.where(
        (uninon_buy_ckpo[LTT.QUADRANT_LEVERAGE_MACD_TIMING_LAG_DUMMY] > uninon_buy_ckpo[LTT.QUADRANT_LEVERAGE_POWERLINE_UPRISING_TIMING_LAG]) & \
        (uninon_buy_ckpo[LTT.QUADRANT_LEVERAGE_POWERLINE_UPRISING_TIMING_LAG] > 1e-12) & \
        ((uninon_buy_ckpo[FLD.CHARTSCAN_BUY_BEFORE_MAJOR_DUMMY] < (uninon_buy_ckpo[LTT.QUADRANT_LEVERAGE_POWERLINE_UPRISING_TIMING_LAG] + 4)) | \
        (uninon_buy_ckpo[FLD.CHARTSCAN_BUY_BEFORE_DUMMY] < (uninon_buy_ckpo[LTT.QUADRANT_LEVERAGE_POWERLINE_UPRISING_TIMING_LAG] + 4)) | \
        (uninon_buy_ckpo[FLD.CHARTSCAN_BUY_BEFORE_MINOR_DUMMY] < (uninon_buy_ckpo[LTT.QUADRANT_LEVERAGE_POWERLINE_UPRISING_TIMING_LAG] + 4))),
        uninon_buy_ckpo[LTT.QUADRANT_PUSH_CREDIT] + 1,
        uninon_buy_ckpo[LTT.QUADRANT_PUSH_CREDIT]
    )
    uninon_buy_ckpo[LTT.QUADRANT_PUSH_CREDIT] = np.where(
        (uninon_buy_ckpo[LTT.QUADRANT_LEVERAGE_MACD_TIMING_LAG_DUMMY] > 1e-12) & \
        (uninon_buy_ckpo[TRD.OMNIPATH_UPRISING_I_COEF_UPRISING_TIMING_LAG] > 1e-12) & \
        (uninon_buy_ckpo[LTT.ONE_QUADRANT_UPRISING_TIMING_LAG_MAJOR] >= uninon_buy_ckpo[TRD.OMNIPATH_UPRISING_I_COEF_UPRISING_TIMING_LAG]) & \
        (uninon_buy_ckpo[TRD.OMNIPATH_UPRISING_I_COEF_UPRISING_TIMING_LAG] > uninon_buy_ckpo[LTT.TWO_QUADRANT_UPRISING_TIMING_LAG_MAJOR]) & \
        (uninon_buy_ckpo[TRD.OMNIPATH_UPRISING_I_COEF_UPRISING_TIMING_LAG] > uninon_buy_ckpo[FLD.DEA_ZERO_TIMING_LAG_MAJOR]),
        uninon_buy_ckpo[LTT.QUADRANT_PUSH_CREDIT] + 1,
        uninon_buy_ckpo[LTT.QUADRANT_PUSH_CREDIT]
    )
    uninon_buy_ckpo[LTT.QUADRANT_PUSH_CREDIT] = np.where(
        (uninon_buy_ckpo[LTT.QUADRANT_LEVERAGE_MACD_TIMING_LAG_DUMMY] > 1e-12) & \
        (uninon_buy_ckpo[LTT.QUADRANT_LEVERAGE_POWERLINE_UPRISING_TIMING_LAG] > 1e-12) & \
        (uninon_buy_ckpo[LTT.ONE_QUADRANT_UPRISING_TIMING_LAG_MAJOR] >= uninon_buy_ckpo[LTT.QUADRANT_LEVERAGE_POWERLINE_UPRISING_TIMING_LAG]) & \
        (uninon_buy_ckpo[LTT.QUADRANT_LEVERAGE_POWERLINE_UPRISING_TIMING_LAG] >= uninon_buy_ckpo[LTT.TWO_QUADRANT_UPRISING_TIMING_LAG_MAJOR]) & \
        (uninon_buy_ckpo[FLD.DEA_ZERO_TIMING_LAG_MAJOR] > uninon_buy_ckpo[LTT.TWO_QUADRANT_UPRISING_TIMING_LAG_MAJOR]) & \
        (uninon_buy_ckpo[LTT.ONE_QUADRANT_UPRISING_TIMING_LAG_MAJOR] > uninon_buy_ckpo[FLD.DEA_ZERO_TIMING_LAG_MAJOR]),
        uninon_buy_ckpo[LTT.QUADRANT_PUSH_CREDIT] + 1,
        uninon_buy_ckpo[LTT.QUADRANT_PUSH_CREDIT]
    )
    uninon_buy_ckpo[LTT.QUADRANT_PUSH_CREDIT] = np.where(
        (uninon_buy_ckpo[LTT.QUADRANT_LEVERAGE_MACD_TIMING_LAG_DUMMY] > 1e-12) & \
        (uninon_buy_ckpo[TRD.OMNIPATH_UPRISING_I_COEF_UPRISING_TIMING_LAG] > 1e-12) & \
        ((uninon_buy_ckpo[LTT.ONE_QUADRANT_UPRISING_TIMING_LAG_MAJOR] + \
         uninon_buy_ckpo[TRD.OMNIPATH_UPRISING_I_COEF_UPRISING_TIMING_LAG] + \
         uninon_buy_ckpo[LTT.TWO_QUADRANT_UPRISING_TIMING_LAG_MAJOR] + \
         uninon_buy_ckpo[LTT.QUADRANT_LEVERAGE_POWERLINE_UPRISING_TIMING_LAG]) / (uninon_buy_ckpo[LTT.QUADRANT_LEVERAGE_MACD_TIMING_LAG_DUMMY] * 4) > 0.832),
        uninon_buy_ckpo[LTT.QUADRANT_PUSH_CREDIT] + 1,
        uninon_buy_ckpo[LTT.QUADRANT_PUSH_CREDIT]
    )
    uninon_buy_ckpo[LTT.QUADRANT_PUSH_CREDIT] = np.where(
        (uninon_buy_ckpo[LTT.QUADRANT_LEVERAGE_MACD_TIMING_LAG_DUMMY] > 1e-12) & \
        (uninon_buy_ckpo[TRD.OMNIPATH_UPRISING_I_COEF_UPRISING_TIMING_LAG] > 1e-12) & \
        ((uninon_buy_ckpo[TRD.OMNIPATH_UPRISING_I_COEF_UPRISING_TIMING_LAG] >= uninon_buy_ckpo[LTT.QUADRANT_LEVERAGE_MACD_TIMING_LAG_DUMMY]) | \
        ((uninon_buy_ckpo[TRD.OMNIPATH_UPRISING_I_COEF_UPRISING_TIMING_LAG] - uninon_buy_ckpo[TRD.OMNIPATH_UPRISING_I_COEF_UPRISING_TIMING_LAG]) < 20) & \
        ((uninon_buy_ckpo[TRD.OMNIPATH_UPRISING_I_COEF_UPRISING_TIMING_LAG] >= uninon_buy_ckpo[LTT.ONE_QUADRANT_UPRISING_TIMING_LAG_MAJOR]))),
        np.where(
            ((uninon_buy_ckpo[TRD.OMNIPATH_UPRISING_I_COEF_CANDIDATE] > 0.618) & \
            (uninon_buy_ckpo[AKA.SAMPLE_ENTROPY_DUMMY] < 0.0928)) | \
            (uninon_buy_ckpo[TRD.OMNIPATH_UPRISING_I_COEF_CANDIDATE] > 0.75),
            uninon_buy_ckpo[LTT.QUADRANT_PUSH_CREDIT] + 2,
            uninon_buy_ckpo[LTT.QUADRANT_PUSH_CREDIT] + 1
        ),
        uninon_buy_ckpo[LTT.QUADRANT_PUSH_CREDIT]
    )
    uninon_buy_ckpo[LTT.QUADRANT_PUSH_CREDIT] = np.where(
        (uninon_buy_ckpo[LTT.QUADRANT_LEVERAGE_MACD_TIMING_LAG_DUMMY] > 1e-12) & \
        (uninon_buy_ckpo[LTT.MAINSTREAM_LEADIN_BEFORE_MINOR] > uninon_buy_ckpo[AKA.CLOSE]) & \
        (uninon_buy_ckpo[LTT.MAINSTREAM_LEADIN_BEFORE_MINOR] > uninon_buy_ckpo[FLD.MACD_CROSS_SX_BEFORE_MAJOR]) & \
        (uninon_buy_ckpo[FLD.CHARTSCAN_BUY_BEFORE_DUMMY] > uninon_buy_ckpo[AKA.CLOSE]),
        uninon_buy_ckpo[LTT.QUADRANT_PUSH_CREDIT] + 1,
        uninon_buy_ckpo[LTT.QUADRANT_PUSH_CREDIT]
    )
    uninon_buy_ckpo[LTT.QUADRANT_PUSH_CREDIT] = np.where(
        (uninon_buy_ckpo[LTT.QUADRANT_LEVERAGE_MACD_TIMING_LAG_DUMMY] > 1e-12) & \
        (uninon_buy_ckpo[LTT.MAINSTREAM_LEADIN_BEFORE_MINOR] > uninon_buy_ckpo[AKA.CLOSE]) & \
        (uninon_buy_ckpo[LTT.MAINSTREAM_LEADIN_BEFORE_MINOR] > uninon_buy_ckpo[FLD.MACD_CROSS_SX_BEFORE_MAJOR]) & \
        (uninon_buy_ckpo[FLD.CHARTSCAN_BUY_BEFORE_MINOR_DUMMY] > uninon_buy_ckpo[AKA.CLOSE]),
        uninon_buy_ckpo[LTT.QUADRANT_PUSH_CREDIT] + 1,
        uninon_buy_ckpo[LTT.QUADRANT_PUSH_CREDIT]
    )
    uninon_buy_ckpo[LTT.QUADRANT_PUSH_CREDIT] = np.where(
        (uninon_buy_ckpo[LTT.QUADRANT_LEVERAGE_MACD_TIMING_LAG_DUMMY] > 1e-12) & \
        (uninon_buy_ckpo[LTT.MAINSTREAM_LEADIN_BEFORE] > uninon_buy_ckpo[AKA.CLOSE]) & \
        (uninon_buy_ckpo[FLD.CHARTSCAN_BUY_BEFORE_MINOR_DUMMY] > uninon_buy_ckpo[FLD.MACD_CROSS_SX_BEFORE_MAJOR]) & \
        (uninon_buy_ckpo[FLD.CHARTSCAN_BUY_BEFORE_MINOR_DUMMY] > uninon_buy_ckpo[AKA.CLOSE]),
        uninon_buy_ckpo[LTT.QUADRANT_PUSH_CREDIT] + 1,
        uninon_buy_ckpo[LTT.QUADRANT_PUSH_CREDIT]
    )
    uninon_buy_ckpo[LTT.QUADRANT_PUSH_CREDIT] = np.where(
        (uninon_buy_ckpo[LTT.QUADRANT_LEVERAGE_MACD_TIMING_LAG_DUMMY] > 1e-12) & \
        (uninon_buy_ckpo[AKA.LOW] < uninon_buy_ckpo[AKA.COST5_PRICE]),
        uninon_buy_ckpo[LTT.QUADRANT_PUSH_CREDIT] + 1,
        uninon_buy_ckpo[LTT.QUADRANT_PUSH_CREDIT]
    )
    uninon_buy_ckpo[LTT.QUADRANT_PUSH_CREDIT] = np.where(
        (uninon_buy_ckpo[LTT.QUADRANT_LEVERAGE_MACD_TIMING_LAG_DUMMY] > (uninon_buy_ckpo[LTT.QUADRANT_LEVERAGE_POWERLINE_UPRISING_TIMING_LAG] + 84)) & \
        (uninon_buy_ckpo[LTT.QUADRANT_LEVERAGE_POWERLINE_UPRISING_TIMING_LAG] > 1e-12) & \
        (uninon_buy_ckpo[LTT.QUADRANT_LEVERAGE_POWERLINE_UPRISING_TIMING_LAG] == uninon_buy_ckpo[LTT.ONE_QUADRANT_UPRISING_TIMING_LAG_MAJOR]) & \
        (uninon_buy_ckpo[AKA.SAMPLE_ENTROPY_DUMMY] > 0.0618),
        uninon_buy_ckpo[LTT.QUADRANT_PUSH_CREDIT] + 1,
        uninon_buy_ckpo[LTT.QUADRANT_PUSH_CREDIT]
    )
    uninon_buy_ckpo[LTT.QUADRANT_PUSH_CREDIT] = np.where(
        (uninon_buy_ckpo[LTT.QUADRANT_LEVERAGE_MACD_TIMING_LAG_DUMMY] > 1e-12) & \
        (uninon_buy_ckpo[TRD.OMNIPATH_UPRISING_I_COEF_CANDIDATE] < 0.382) & \
        (uninon_buy_ckpo[AKA.SAMPLE_ENTROPY_DUMMY] > 0.0618),
        uninon_buy_ckpo[LTT.QUADRANT_PUSH_CREDIT] - 1,
        uninon_buy_ckpo[LTT.QUADRANT_PUSH_CREDIT]
    )
    uninon_buy_ckpo[LTT.QUADRANT_PUSH_CREDIT] = np.where(
        (uninon_buy_ckpo[LTT.QUADRANT_LEVERAGE_MACD_TIMING_LAG_DUMMY] > 1e-12) & \
        (uninon_buy_ckpo[LTT.QUADRANT_TRIGGER_CREDIT] > 5.333) & \
        (uninon_buy_ckpo[LTT.QUADRANT_ZEN_DASH_COUNT] > 0.382) & \
        (uninon_buy_ckpo[AKA.SAMPLE_ENTROPY_DUMMY] > 0.0618),
        uninon_buy_ckpo[LTT.QUADRANT_PUSH_CREDIT] + 1,
        uninon_buy_ckpo[LTT.QUADRANT_PUSH_CREDIT]
    )
    uninon_buy_ckpo[LTT.QUADRANT_PUSH_CREDIT] = np.where(
        (uninon_buy_ckpo[LTT.QUADRANT_LEVERAGE_MACD_TIMING_LAG_DUMMY] > 1e-12) & \
        (uninon_buy_ckpo[PFL.POSITION_LEVEL] > 0.168) & \
        (uninon_buy_ckpo[AKA.SAMPLE_ENTROPY_DUMMY] > 0.0618),
        uninon_buy_ckpo[LTT.QUADRANT_PUSH_CREDIT] + 1,
        uninon_buy_ckpo[LTT.QUADRANT_PUSH_CREDIT]
    )
    uninon_buy_ckpo[LTT.QUADRANT_PUSH_CREDIT] = np.where(
        (uninon_buy_ckpo[LTT.QUADRANT_LEVERAGE_MACD_TIMING_LAG_DUMMY] > 1e-12) & \
        (uninon_buy_ckpo[LTT.QUADRANT_TRIGGER_CREDIT] > 3.82) & \
        (uninon_buy_ckpo[LTT.QUADRANT_ZEN_DASH_COUNT] > 0.382) & \
        (uninon_buy_ckpo[LTT.QUADRANT_ZEN_DASH_COUNT] < 1.68) & \
        (uninon_buy_ckpo[TRD.UNIVERSAL_CLEARANCE_ENERGY] > uninon_buy_ckpo[TRD.OMNIPATH_UPRISING_I_COEF_CANDIDATE]) & \
        ((uninon_buy_ckpo[TRD.UNIVERSAL_CLEARANCE_ENERGY] * 1.68 > uninon_buy_ckpo[LTT.QUADRANT_LEVERAGE_MACD_TIMING_LAG_DUMMY]) | \
        ((uninon_buy_ckpo[FLD.ZEN_DASH_TIMING_LAG_DUMMY] > 1e-12) & \
        (uninon_buy_ckpo[TRD.UNIVERSAL_CLEARANCE_ENERGY] * 1.68 > uninon_buy_ckpo[FLD.ZEN_DASH_TIMING_LAG_DUMMY]))) & \
        (((uninon_buy_ckpo[LTT.QUADRANT_LEVERAGE_POWERLINE_UPRISING_TIMING_LAG] > 1e-12) & \
        (uninon_buy_ckpo[LTT.QUADRANT_LEVERAGE_MACD_TIMING_LAG_DUMMY] > uninon_buy_ckpo[LTT.QUADRANT_LEVERAGE_POWERLINE_UPRISING_TIMING_LAG])) | \
        (uninon_buy_ckpo[LTT.QUADRANT_LEVERAGE_MACD_TIMING_LAG_DUMMY] > 3)) & \
        (uninon_buy_ckpo[AKA.SAMPLE_ENTROPY_DUMMY] > 0.0618),
        uninon_buy_ckpo[LTT.QUADRANT_PUSH_CREDIT] + 1,
        uninon_buy_ckpo[LTT.QUADRANT_PUSH_CREDIT]
    )
    uninon_buy_ckpo[LTT.QUADRANT_PUSH_CREDIT] = np.where(
        (uninon_buy_ckpo[LTT.QUADRANT_LEVERAGE_MACD_TIMING_LAG_DUMMY] > 1e-12) & \
        (uninon_buy_ckpo[LTT.BS_STAR_UP_BEFORE_MAJOR] < uninon_buy_ckpo[LTT.QUADRANT_LEVERAGE_MACD_TIMING_LAG_DUMMY]) & \
        (uninon_buy_ckpo[LTT.BS_STAR_UP_BEFORE_MAJOR] < uninon_buy_ckpo[FLD.DEA_ZERO_TIMING_LAG_MAJOR]) & \
        (uninon_buy_ckpo[LTT.BS_STAR_UP_BEFORE_MAJOR] < uninon_buy_ckpo[LTT.ONE_QUADRANT_UPRISING_TIMING_LAG_MAJOR]) & \
        (uninon_buy_ckpo[LTT.BS_STAR_UP_BEFORE_MAJOR] < uninon_buy_ckpo[LTT.BS_STAR_DOWN_BEFORE_MAJOR]) & \
        (uninon_buy_ckpo[AKA.SAMPLE_ENTROPY_DUMMY] > 0.0618),
        uninon_buy_ckpo[LTT.QUADRANT_PUSH_CREDIT] + 1,
        uninon_buy_ckpo[LTT.QUADRANT_PUSH_CREDIT]
    )
    uninon_buy_ckpo[LTT.QUADRANT_PUSH_CREDIT] = np.where(
        (uninon_buy_ckpo[LTT.QUADRANT_LEVERAGE_MACD_TIMING_LAG_DUMMY] > 1e-12) & \
        (uninon_buy_ckpo[FLD.MACD_CROSS_SX_BEFORE_MAJOR] < uninon_buy_ckpo[FLD.DEA_ZERO_TIMING_LAG_MAJOR]) & \
        (uninon_buy_ckpo[AKA.RETURN_COST95] < 0.168) & \
        (uninon_buy_ckpo[AKA.SAMPLE_ENTROPY_DUMMY] < 0.0928),
        uninon_buy_ckpo[LTT.QUADRANT_PUSH_CREDIT] - 5,
        uninon_buy_ckpo[LTT.QUADRANT_PUSH_CREDIT]
    )
    uninon_buy_ckpo[LTT.QUADRANT_PUSH_CREDIT] = uninon_buy_ckpo[LTT.QUADRANT_PUSH_CREDIT].astype(np.int16)

    return uninon_buy_ckpo
