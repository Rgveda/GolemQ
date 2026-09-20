# coding:utf-8
#
# The MIT License (MIT)
#
# Copyright (c) 2020-2026 acai/GolemQ
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


def GQ_util_code_tostr(code):
    """
    explanation:
        将所有沪深股票从数字转化到6位的代码,因为有时候在csv等转换的时候,诸如 000001的股票会变成office强制转化成数字1,
        同时支持聚宽股票格式,掘金股票代码格式,Wind股票代码格式,天软股票代码格式

    params:
        * code ->
            含义: 代码
            类型: str
            参数支持: []
    """
    if isinstance(code, int):
        return "{:>06d}".format(code)
    if isinstance(code, str):
        # 聚宽股票代码格式 '600000.XSHG'
        # 掘金股票代码格式 'SHSE.600000'
        # Wind股票代码格式 '600000.SH'
        # 天软股票代码格式 'SH600000'
        if len(code) == 6:
            return code
        if len(code) == 8:
            # 天软数据
            return code[-6:]
        if len(code) == 9:
            return code[:6]
        if len(code) == 11:
            if code[0] in ["S"]:
                return code.split(".")[1]
            return code.split(".")[0]
        raise ValueError(f"错误的股票代码格式 '{code}'")
    if isinstance(code, list):
        return GQ_util_code_tostr(code[0])


def GQ_util_code_tolist(code, auto_fill=True):
    """
    explanation:
        将转换code==> list

    params:
        * code ->
            含义: 代码
            类型: str
            参数支持: []
        * auto_fill->
            含义: 是否自动补全(一般是用于股票/指数/etf等6位数,期货不适用) (default: {True})
            类型: bool
            参数支持: [True]
    """

    if isinstance(code, str):
        if auto_fill:
            return [GQ_util_code_tostr(code)]
        else:
            return [code]

    elif isinstance(code, list):
        if auto_fill:
            return [GQ_util_code_tostr(item) for item in code]
        else:
            return [item for item in code]
