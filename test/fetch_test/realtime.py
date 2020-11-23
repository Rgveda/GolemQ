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
from numpy import *
import numpy as np
import pandas as pd

from GolemQ.utils.symbol import (
    get_codelist,
)
from GolemQ.fetch.kline import (
    get_kline_price,
    get_kline_price_min,
)

if __name__ == '__main__':
    pd.set_option('display.float_format',lambda x : '%.3f' % x)
    pd.set_option('display.max_columns', 20)
    pd.set_option("display.max_rows", 300)
    pd.set_option('display.width', 240)  # 设置打印宽度
    pd.set_option('display.unicode.ambiguous_as_wide', True)
    pd.set_option('display.unicode.east_asian_width', True)

    #codepool = ['159919', '159997', '159805', '159987', 
    #           '159952', '159920', '518880', '159934', 
    #           '159985', '515050', '159994', '159941', 
    #           '512800', '515000', '512170', '512980', 
    #           '510300', '513100', '510900', '512690', 
    #           '510050', '159916', '512910', '510310', 
    #           '512090', '513050', '513030', '513500', 
    #           '159905', '159949', '510330', '510500', 
    #           '510180', '159915', '510810', '159901', 
    #           '512710', '510850', '512500', '512000',]

    # 我改进了 get_codelist 这个函数，为的是别人QQ发过来的一大串股票代码，
    # 我不用整理出上面的list，直接灌QA进去也能读取行情。
    codepool = ''' 601801,   
   603028    600519,   '''

    codelist_candidate = get_codelist(codepool)
    codelist = codelist_candidate
    #codelist = [code for code in codelist_candidate if not
    #code.startswith('300')]
    #codelist = ['300386', '300402']
    print(len(codelist), codelist)

    verbose = True # 如果不要罗里吧嗦，就设置为False

    # 计算策略主时间周期：日线
    data_day, codename = get_kline_price(codelist, verbose=verbose)
    if (len(data_day.data) < 300):
        print(u'新股或者次新股，{}'.format(codename))

    print(data_day.data.tail(12))