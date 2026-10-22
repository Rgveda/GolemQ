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

#: CV（图形识别）整条链 —— **跨市场共用**。
#:
#: * :mod:`GolemQ.analysis.cv.scopes`：scan scope → 代码清单（数据落点由 `dataset` 注入）；
#: * `client` / `server` / `predict` / `render` / `model`：原
#:   `markets/StockCN/chartscan/`，2026-10-21 整体搬入 `analysis/cv/`。
#:
#: ⚠️ 落库用的行情库**不在本包内取**：所有写/读函数都接收 `db` 参数，
#: 由 CLI / server / client 从 **Active Market** 的 `get_database()` 取实例后注入
#: （「库名不外泄」）。模块级仍不 import torch/ultralytics —— 只在 `load_model()` /
#: 出图时惰性导入。

from .model import load_model, weights_path
from .predict import (GQ_markup_ChartScaned, GQ_markup_daily_ChartScaned,
                      GQ_predict_ChartScanAI, calc_ChartScanAI_reality,
                      calc_ChartScaned_persistence)
from .render import GQ_generate_chart_stock, calculate_i

__all__ = [
    'weights_path', 'load_model',
    'GQ_generate_chart_stock', 'calculate_i',
    'GQ_predict_ChartScanAI', 'calc_ChartScaned_persistence',
    'calc_ChartScanAI_reality', 'GQ_markup_ChartScaned',
    'GQ_markup_daily_ChartScaned',
]
