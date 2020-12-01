#coding=utf-8
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
# sub_cython.pyx


import numpy as np
cimport numpy as np
cimport cython


cdef const char* EXCHANGE_SH "XSHG"
cdef const char* EXCHANGE_SZ "XSHE"


cdef struct l1_snapshot:
    char* code
    float price
    float vol
    char* servertime
    char* datetime


cpdef unsigned char[:] _normalize_code(unsigned char[:] symbol):
    if (not isinstance(symbol, str)):
        return symbol

    if (symbol.startswith('sz') and (len(symbol) == 8)):
        ret_normalize_code = '{}.{}'.format(symbol[2:8], EXCHANGE_SZ)
    elif (symbol.startswith('sh') and (len(symbol) == 8)):
        ret_normalize_code = '{}.{}'.format(symbol[2:8], EXCHANGE_SH)
    elif ((symbol.startswith('399') or symbol.startswith('159') or \
        symbol.startswith('150')) and (len(symbol) == 6)):
        ret_normalize_code = '{}.{}'.format(symbol, EXCHANGE_SH)
    elif ((len(symbol) == 6) and (symbol.startswith('399') or \
        symbol.startswith('159') or symbol.startswith('150') or \
        symbol.startswith('16') or symbol.startswith('184801') or \
        symbol.startswith('201872'))):
        ret_normalize_code = '{}.{}'.format(symbol, EXCHANGE_SZ)
    elif ((len(symbol) == 6) and (symbol.startswith('50') or \
        symbol.startswith('51') or symbol.startswith('60') or \
        symbol.startswith('688') or symbol.startswith('900') or \
        (symbol == '751038'))):
        ret_normalize_code = '{}.{}'.format(symbol, EXCHANGE_SH)
    elif ((len(symbol) == 6) and (symbol[:3] in ['000', '001', '002', '200', '300'])):
        ret_normalize_code = '{}.{}'.format(symbol, EXCHANGE_SZ)
    else:
        print(symbol)
        ret_normalize_code = symbol

    return ret_normalize_code


cpdef unsigned char[:] _normalize_code_with_closep(unsigned char[:] symbol, np.float64_t pre_close):
    if (not isinstance(symbol, str)):
        return symbol

    if (symbol.startswith('sz') and (len(symbol) == 8)):
        ret_normalize_code = '{}.{}'.format(symbol[2:8], EXCHANGE_SZ)
    elif (symbol.startswith('sh') and (len(symbol) == 8)):
        ret_normalize_code = '{}.{}'.format(symbol[2:8], EXCHANGE_SH)
    elif (symbol.startswith('00') and (len(symbol) == 6)):
        if ((pre_close is not None) and (pre_close > 2000)):
            # 推断是上证指数
            ret_normalize_code = '{}.{}'.format(symbol, EXCHANGE_SH)
        else:
            ret_normalize_code = '{}.{}'.format(symbol, EXCHANGE_SZ)
    elif ((symbol.startswith('399') or symbol.startswith('159') or \
        symbol.startswith('150')) and (len(symbol) == 6)):
        ret_normalize_code = '{}.{}'.format(symbol, EXCHANGE_SH)
    elif ((len(symbol) == 6) and (symbol.startswith('399') or \
        symbol.startswith('159') or symbol.startswith('150') or \
        symbol.startswith('16') or symbol.startswith('184801') or \
        symbol.startswith('201872'))):
        ret_normalize_code = '{}.{}'.format(symbol, EXCHANGE_SZ)
    elif ((len(symbol) == 6) and (symbol.startswith('50') or \
        symbol.startswith('51') or symbol.startswith('60') or \
        symbol.startswith('688') or symbol.startswith('900') or \
        (symbol == '751038'))):
        ret_normalize_code = '{}.{}'.format(symbol, EXCHANGE_SH)
    elif ((len(symbol) == 6) and (symbol[:3] in ['000', '001', '002', '200', '300'])):
        ret_normalize_code = '{}.{}'.format(symbol, EXCHANGE_SZ)
    else:
        print(symbol)
        ret_normalize_code = symbol

    return ret_normalize_code


cpdef dict _formatter_l1_snapshot(unsigned char[:] code, dict l1_tick):
    """
    处理分发 Tick 数据，新浪和tdx l1 tick差异字段格式化处理
    """
    if ((len(code) == 6) and code.startswith('00')):
        l1_tick['code'] = _normalize_code_with_closep(code, l1_tick['now'])
    else:
        l1_tick['code'] = _normalize_code(code)
    l1_tick['servertime'] = l1_tick['time']
    l1_tick['datetime'] = '{} {}'.format(l1_tick['date'], l1_tick['time'])
    l1_tick['price'] = l1_tick['now']
    l1_tick['vol'] = l1_tick['volume']
    del l1_tick['date']
    del l1_tick['time']
    del l1_tick['now']
    del l1_tick['name']
    del l1_tick['volume']

    return l1_tick


cpdef list _formater_l1_snapshots(dict l1_ticks):
    """
    处理 l1 ticks 数据
    """
    cpdef list l1_ticks_data
    #cdef int i, j
    #cdef int end_index = len(l1_ticks)
    #cdef l1_snapshot[:] l1_ticks_data

    for code, l1_tick_values in l1_ticks.items():
        l1_tick = _formatter_l1_snapshot(code, l1_tick_values)
        l1_ticks_data.append(l1_tick)
        #i = i+1

    return l1_ticks_data


def formatter_l1_snapshot(unsigned char[:] code, l1_tick:dict):
    return _formatter_l1_snapshot(code, l1_tick)

