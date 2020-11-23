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

try:
    import talib
except:
    print('PLEASE run "pip install TALIB" to call these modules')
    pass
try:
    import QUANTAXIS as QA
    from QUANTAXIS.QAUtil.QAParameter import ORDER_DIRECTION
    from QUANTAXIS.QAData.QADataStruct import (QA_DataStruct_Index_min, 
                                               QA_DataStruct_Index_day, 
                                               QA_DataStruct_Stock_day, 
                                               QA_DataStruct_Stock_min)
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
except:
    print('PLEASE run "pip install QUANTAXIS" before call GolemQ.cli.okex modules')
    pass

from GolemQ.utils.parameter import (
    AKA, 
    INDICATOR_FIELD as FLD, 
    TREND_STATUS as ST
)


def realtime_rolling_by_vol():
    return False


