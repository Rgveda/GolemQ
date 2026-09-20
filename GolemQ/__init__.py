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

# Package exports
from . import core
from . import agents
# from . import models
from . import analysis
from . import services as services
from . import pipeline
from . import supervisor

"""GolemQ - A Python package for quantum-inspired algorithms."""
from .core.settings import DATABASE, DATABASE_ASYNC

# 市场注册表与「当前激活市场」—— 实现在 core/market_registry.py，此处再导出
# 以保持既有 `from GolemQ import GQMARKETS` 的写法可用。
from .core.market_registry import (  # noqa: F401
    DEFAULT_MARKET,
    GQMARKETS,
    GQSUBSCRIBER,
    active_market_name,
    get_active_market,
    get_market,
    register_market,
    register_subscriber,
    set_active_market,
)

__version__ = "0.1.1"
__all__ = ['core', 'agents', 'analysis',
           'services', 'pipeline', 'supervisor',
           'DATABASE', 'DATABASE_ASYNC',
           'GQMARKETS', 'GQSUBSCRIBER', 'DEFAULT_MARKET',
           'get_active_market', 'set_active_market', 'get_market',
           'active_market_name', 'register_market', 'register_subscriber']
