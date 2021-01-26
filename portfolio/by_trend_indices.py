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
import numpy as np

from GolemQ.portfolio.utils import (
    calc_leverages,
)

from GolemQ.utils.parameter import (
    AKA, 
    INDICATOR_FIELD as FLD, 
    TREND_STATUS as ST
)

def smart_positions_v1_func(data, indices=None):
    """
    计算杠杆，仓位 v1
    通过趋势判断和技术指标计算持仓杠杆和仓位
    """
    onebite = 0.0309
    if (FLD.RSI_DELTA not in indices.columns):
        indices[FLD.RSI_DELTA] = indices[FLD.RSI].diff()
    leverage_delta = np.where((indices[FLD.LINEAREG_CROSS] > 0) & \
                              (indices[FLD.PCT_CHANGE] > 0) & \
                              (indices[FLD.RSI_DELTA] > 0), onebite, 
                        np.where((indices[FLD.LINEAREG_CROSS] < 0) & \
                              (indices[FLD.PCT_CHANGE] < 0) & \
                              (indices[FLD.RSI_DELTA] < 0), -onebite, 0))

    leverage_delta = np.where((indices[FLD.MAXFACTOR_CROSS] > 0) & \
                              (indices[FLD.PCT_CHANGE] > 0) & \
                              (indices[FLD.RSI_DELTA] > 0), leverage_delta + onebite, 
                        np.where((indices[FLD.MAXFACTOR_CROSS] < 0) & \
                              (indices[FLD.PCT_CHANGE] < 0) & \
                              (indices[FLD.RSI_DELTA] < 0), leverage_delta - onebite, leverage_delta))

    leverage_delta = np.where((indices[FLD.DUAL_CROSS] > 0) & \
                              (indices[FLD.PCT_CHANGE] > 0) & \
                              (indices[FLD.RSI_DELTA] > 0), leverage_delta + onebite, 
                              np.where((indices[FLD.DUAL_CROSS] < 0) & \
                                       (indices[FLD.PCT_CHANGE] < 0) & \
                                       (indices[FLD.RSI_DELTA] < 0), leverage_delta - onebite, leverage_delta))

    leverage_delta = np.where((indices[FLD.ATR_Stopline] > 0) & \
                              (indices[FLD.PCT_CHANGE] > 0) & \
                              (indices[FLD.RSI_DELTA] > 0), leverage_delta + onebite, 
                        np.where((indices[FLD.ATR_Stopline] < 0) & \
                              (indices[FLD.PCT_CHANGE] < 0) & \
                              (indices[FLD.RSI_DELTA] < 0), leverage_delta - onebite, leverage_delta))

    leverage_delta = np.where((indices[FLD.ATR_SuperTrend] > 0) & \
                              (indices[FLD.PCT_CHANGE] > 0) & \
                              (indices[FLD.RSI_DELTA] > 0), leverage_delta + onebite, 
                        np.where((indices[FLD.ATR_SuperTrend] < 0) & \
                              (indices[FLD.PCT_CHANGE] < 0) & \
                              (indices[FLD.RSI_DELTA] < 0), leverage_delta - onebite, leverage_delta))

    leverage_delta = np.where((indices[FLD.Volume_HMA5] > 0) & \
                              (indices[FLD.PCT_CHANGE] > 0) & \
                              (indices[FLD.RSI_DELTA] > 0), leverage_delta + onebite, 
                        np.where((indices[FLD.Volume_HMA5] < 0) & \
                              (indices[FLD.PCT_CHANGE] < 0) & \
                              (indices[FLD.RSI_DELTA] < 0), leverage_delta - onebite, leverage_delta))
   
    leverage_delta = np.where((indices[FLD.COMBO_FLOW] > 0) & \
                              (indices[FLD.PCT_CHANGE] > 0) & \
                              (indices[FLD.RSI_DELTA] > 0), leverage_delta + onebite, 
                        np.where((indices[FLD.COMBO_FLOW] < 0) & \
                              (indices[FLD.PCT_CHANGE] < 0) & \
                              (indices[FLD.RSI_DELTA] < 0), leverage_delta - onebite, leverage_delta))
    
    leverage_delta = np.where((indices[FLD.RENKO_TREND_L] > 0) & \
                              (indices[FLD.PCT_CHANGE] > 0) & \
                              (indices[FLD.RSI_DELTA] > 0), leverage_delta + onebite, 
                        np.where((indices[FLD.RENKO_TREND_L] < 0) & \
                              (indices[FLD.PCT_CHANGE] < 0) & \
                              (indices[FLD.RSI_DELTA] < 0), leverage_delta - onebite, leverage_delta))

    leverage_delta = np.where((indices[FLD.RENKO_TREND_S] > 0) & \
                              (indices[FLD.PCT_CHANGE] > 0) & \
                              (indices[FLD.RSI_DELTA] > 0), leverage_delta + onebite, 
                        np.where((indices[FLD.RENKO_TREND_S] < 0) & \
                              (indices[FLD.PCT_CHANGE] < 0) & \
                              (indices[FLD.RSI_DELTA] < 0), leverage_delta - onebite, leverage_delta))

    leverage_delta = np.where((indices[FLD.RENKO_TREND] > 0) & \
                              (indices[FLD.PCT_CHANGE] > 0) & \
                              (indices[FLD.RSI_DELTA] > 0), leverage_delta + onebite, 
                        np.where((indices[FLD.RENKO_TREND] < 0) & \
                              (indices[FLD.PCT_CHANGE] < 0) & \
                              (indices[FLD.RSI_DELTA] < 0), leverage_delta - onebite, leverage_delta))
    
    leverage_delta = np.where((indices[ST.BOOTSTRAP_I] == True) & \
                              (indices[FLD.PCT_CHANGE] > 0) & \
                              (indices[FLD.RSI_DELTA] > 0), leverage_delta + 2 * onebite, 
                        np.where((indices[ST.BOOTSTRAP_II] == True) & \
                              (indices[FLD.PCT_CHANGE] < 0) & \
                              (indices[FLD.RSI_DELTA] < 0), leverage_delta + 1.236 * onebite, 
                              np.where((indices[ST.DEADPOOL] == True) & \
                                       (indices[FLD.PCT_CHANGE] < 0) & \
                                       (indices[FLD.RSI_DELTA] < 0), leverage_delta - 2 * onebite, leverage_delta)))

    leverage_delta = np.where((indices[FLD.VOLUME_FLOW_TRI_CROSS_JX] > 0) & \
                              (indices[FLD.PCT_CHANGE] > 0) & \
                              (indices[FLD.RSI_DELTA] > 0), leverage_delta + onebite, 
                        np.where((indices[FLD.VOLUME_FLOW_TRI_CROSS_SX] < 0) & \
                              (indices[FLD.PCT_CHANGE] < 0) & \
                              (indices[FLD.RSI_DELTA] < 0), leverage_delta - onebite, leverage_delta))

    leverage_delta = np.where((indices[ST.VOLUME_FLOW_BOOST] > 0) & \
                              (indices[FLD.PCT_CHANGE] > 0) & \
                              (indices[FLD.RSI_DELTA] > 0), leverage_delta + onebite, 
                              np.where((indices[ST.VOLUME_FLOW_BOOST] < 0) & \
                                       (indices[FLD.PCT_CHANGE] < 0) & \
                                       (indices[FLD.RSI_DELTA] < 0), leverage_delta - onebite, 
                                       np.where((indices[ST.VOLUME_FLOW_BOOST_BONUS] == True) & \
                                                (indices[FLD.PCT_CHANGE] > 0) & \
                                                (indices[FLD.RSI_DELTA] > 0), leverage_delta + onebite, leverage_delta)))

    leverage_delta = np.where((indices[FLD.FLU_POSITIVE] > 0) & \
                              (indices[FLD.PCT_CHANGE] > 0) & \
                              (indices[FLD.RSI_DELTA] > 0), leverage_delta + onebite, 
                        np.where((indices[FLD.FLU_NEGATIVE] > 0) & \
                              (indices[FLD.PCT_CHANGE] < 0) & \
                              (indices[FLD.RSI_DELTA] < 0), leverage_delta - onebite, leverage_delta))

    leverage_delta = np.where((indices[FLD.ML_FLU_TREND] > 0) & \
                              (indices[FLD.PCT_CHANGE] > 0) & \
                              (indices[FLD.RSI_DELTA] > 0), leverage_delta + onebite, 
                              np.where((indices[FLD.ML_FLU_TREND] < 0) & \
                                       (indices[FLD.PCT_CHANGE] < 0) & \
                                       (indices[FLD.RSI_DELTA] < 0), leverage_delta - onebite, leverage_delta))     

    leverage_delta = np.where((indices[FLD.FLU_POSITIVE_MASTER] > 0) & \
                              (indices[FLD.PCT_CHANGE] > 0) & \
                              (indices[FLD.RSI_DELTA] > 0), leverage_delta + onebite, 
                              np.where((indices[FLD.FLU_NEGATIVE_MASTER] > 0) & \
                                       (indices[FLD.PCT_CHANGE] < 0) & \
                                       (indices[FLD.RSI_DELTA] < 0), leverage_delta - onebite, leverage_delta))     
     
    leverage_delta = np.where((indices[ST.CLUSTER_GROUP_TOWARDS] > 0) & \
                              (indices[FLD.PCT_CHANGE] > 0) & \
                              (indices[FLD.RSI_DELTA] > 0), leverage_delta + onebite, 
                              np.where((indices[ST.CLUSTER_GROUP_TOWARDS] < 0) & \
                                       (indices[FLD.PCT_CHANGE] < 0) & \
                                       (indices[FLD.RSI_DELTA] < 0), leverage_delta - onebite, leverage_delta))          
     
    leverage_delta = np.where((indices[FLD.TALIB_PATTERNS] > 0) & \
                              (indices[FLD.PCT_CHANGE] > 0) & \
                              (indices[FLD.RSI_DELTA] > 0), leverage_delta + onebite, 
                              np.where((indices[FLD.TALIB_PATTERNS] < 0) & \
                                       (indices[FLD.PCT_CHANGE] < 0) & \
                                       (indices[FLD.RSI_DELTA] < 0), leverage_delta - onebite, leverage_delta))

    leverage_delta = np.where((indices[FLD.MA90] < indices[FLD.MA120]) & \
                              (indices[FLD.MA30] < indices[FLD.MA90]) & \
                              ((indices[FLD.DEA] < 0) | \
                              (indices[FLD.COMBINE_DENSITY] < 0.5)) & \
                              (((indices[FLD.MACD_DELTA] < 0) & \
                              (indices[FLD.BOLL_CROSS_SX_BEFORE] < indices[FLD.BOLL_CROSS_JX_BEFORE])) | \
                              ((indices[FLD.PCT_CHANGE] < 0) & \
                              (indices[FLD.RSI_DELTA] < 0))), -abs(leverage_delta) - onebite, leverage_delta)

    leverage_delta = np.where((indices[FLD.DEA_SLOPE].rolling(4).mean() < 0) & \
                              (indices[FLD.MACD].rolling(4).mean() < 0) & \
                              (indices[FLD.COMBINE_DENSITY_SMA] < 0.512) & \
                              ~(indices[FLD.BOOTSTRAP_I_BEFORE] > indices[FLD.MACD_CROSS_SX_BEFORE]), leverage_delta - 2 * onebite, leverage_delta)

    indices[FLD.LEVERAGE_DELTA] = leverage_delta
   
    ## 计算累积仓位
    #pd.set_option('display.float_format',lambda x : '%.3f' % x)
    #pd.set_option('display.max_columns', 10)
    #pd.set_option("display.max_rows", 120)
    #pd.set_option('display.width', 180) # 设置打印宽度

    if (FLD.LEVERAGE_NORM not in indices.columns):
        indices = indices.reindex(columns=[*indices.columns,
                                           *[FLD.LEVERAGE_NORM, 
                                             FLD.LEVERAGE_PRE,]])
        
    indices[[FLD.LEVERAGE_NORM, 
             FLD.LEVERAGE_PRE]] = calc_leverages(leverage_delta,
                                                 upper_limit=1.0, 
                                                 lower_limit=0.0)

    return indices