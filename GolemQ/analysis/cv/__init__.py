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

#: CV（图形识别）那条链里**跨市场**的部分。
#:
#: 目前只有 :mod:`GolemQ.analysis.cv.scopes` —— 「把 scan scope 名解析成一份代码清单」。
#: 它不 import 任何市场包：数据落点由 `dataset` 参数注入（`DECISIONS.md` D44）。
#:
#: ⚠️ CV 的**其余**部分仍在 `markets/StockCN/chartscan/` —— 那里是 A 股专有的
#: （15min 网格、`10:30/11:30/14:00/15:00` 收盘时刻、`stock_*`/`index_*` 结果族）。
#: 别以为「CV 都在 analysis/cv 下」。
