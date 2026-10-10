# coding=utf-8
#
# The MIT License (MIT)
#
# Copyright (c) 2020 azai/GolemQ
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

# --------------------------------------------------------------------------- #
# BLAS / OpenMP 线程数上限 —— **必须在 numpy 首次 import 之前**设置
# --------------------------------------------------------------------------- #
# ⚠️ **本段是从老树搬回来的，别删**（`GolemQ_old/__init__.py`，老树日志
# `GolemQ_old/docs/claude_change_log.md:3182` 记着定位过程）。新树重构时漏了它，
# 后果实测重现：`--save tdx` 跑到 76% 时 **17.33s/code**（正常是亚秒级），
# 且迟迟不结束 —— 那是**提交量（commit）撞上限后在拿页面文件硬撑**。
#
# 机制：numpy 自带的 OpenBLAS 会**按线程**预留大块提交量。实测（老树，32 逻辑核）
# `import numpy` 后单进程提交量直接 **11.69 GB**（物理只 0.05 GB，约 366MB/线程）；
# 压到 1 之后 **0.02 GB**。joblib/loky 的**每个 worker 都是独立 python 进程**、
# 各自承担这份预留 → 24 workers ≈ **280 GB**，撞的是 Windows 的 **commit limit**
# （物理内存 + 页面文件，**不是**物理内存），报
# ``OSError [WinError 1455] ERROR_COMMITMENT_LIMIT``。
#
# 用 `setdefault`：单进程做数值密集计算时，可在启动前自行 `export` 覆盖
# （例如 `OPENBLAS_NUM_THREADS=4`）。
import os as _os

#: 要压住的线程数环境变量。**测试也读这一份**（`test_cases/test_blas_guard.py`），
#: 加变量时只改这里 —— 别在两处各写一遍。
BLAS_THREAD_VARS = ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'NUMBA_NUM_THREADS',
                    'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')

for _v in BLAS_THREAD_VARS:
    _os.environ.setdefault(_v, '1')
del _os, _v

# Package exports
from . import core
from . import agents
"""GolemQ - A Python package for quantum-inspired algorithms."""

# ⚠️ **子包与 QUANTAXIS 句柄一律惰性加载**（PEP 562 模块级 `__getattr__`，见文件末尾）。
#
# 原来这里急切 `from . import analysis / services / pipeline / supervisor`
# 与 `from .core.settings import DATABASE, DATABASE_ASYNC` —— 后果是
# **任何** `import GolemQ.<任意子模块>` 都会把半个项目拉进来：实测
# `import GolemQ.markets.StockCN.kline_save` 因此加载 **3,679** 个模块 / 3.0s，
# 其中 statsmodels / matplotlib / joblib / tushare / QUANTAXIS **一个都用不到**。
# 那正是「解耦解了个寂寞」的观感来源（`PITFALLS.md` P18）。

# 市场注册表、默认/当前激活市场、市场类型解析 —— 实现在 core/market_registry.py，
# 此处再导出以保持既有 `from GolemQ import GQMARKETS` 的写法可用。
#
# ⚠️ **这里是取行情唯一的入口**：2026-10-10 起 `GolemQ/fetch/` 整包已删（旧门面
# 只是 `get_active_market().xxx(...)` 的一层改名），故取数一律写成
#     from GolemQ import get_active_market
#     res, code = get_active_market().get_kline_price_min('600519', frequency='60min')
from .core.market_registry import (  # noqa: F401
    DEFAULT_MARKET,
    GQMARKETS,
    GQSUBSCRIBER,
    MARKET_TYPE_TO_MARKET,
    active_market_name,
    get_active_market,
    get_default_market,
    get_market,
    register_market,
    register_subscriber,
    resolve_market,
    set_active_market,
)

__version__ = "0.1.1"
__all__ = ['core', 'agents', 'analysis',
           'services', 'pipeline', 'supervisor',
           'GQMARKETS', 'GQSUBSCRIBER', 'DEFAULT_MARKET',
           'get_active_market', 'set_active_market', 'get_market',
           'active_market_name', 'register_market', 'register_subscriber']


#: 惰性加载的子包 —— `from GolemQ import analysis` 与 `GolemQ.analysis` 都照旧可用，
#: 只是"什么时候真的导入"推迟到第一次被引用。
_LAZY_SUBPACKAGES = ('agents', 'analysis', 'datasource', 'markets', 'models',
                     'pipeline', 'services', 'supervisor')


def __getattr__(name):
    """模块级惰性属性（PEP 562）。见文件头部那段说明。"""
    if name in _LAZY_SUBPACKAGES:
        import importlib
        return importlib.import_module('.' + name, __name__)
    raise AttributeError('module {!r} has no attribute {!r}'.format(__name__, name))
