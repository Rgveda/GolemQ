# coding:utf-8
#
# The MIT License (MIT)
#
# Copyright (c) 2020-2025 azai/GolemQ
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

"""**4.4 搬迁源通道：一次性专用，搬完即弃。**

⚠️⚠️ **本模块只允许被 `--migrate-*` 这类一次性命令引用。**
**运行时代码（读行情、写实时、跑回测、订阅器）一律禁止 import 它** ——
`D12` 定的规矩是「QUANTAXIS 已从全树剔除，4.4 只作一次性迁移源」，
运行时不连 4.4（`Project.md` 的原则 1）。违反这条会让"已剔除 QUANTAXIS"
变成空话：`golemq` 里那批 `stock_a_snapshot*`/`stock_metadata*` 在 8.3 没有落点，
运行时代码一旦指向 4.4，症状是**读一半、空一半**而且不报错。

为什么不用 `~/.QUANTAXIS/setting/config.ini` 里的地址
================================================================
那正是被剔除的耦合（QUANTAXIS 配置驱动 + 它那个 `QA_Setting` 会拖进整棵依赖树）。
这里改成**常量 + 环境变量**，地址是实测值，可被 `GQ_MIGRATE44_URI` 覆盖。
"""
from __future__ import annotations

import os

from GolemQ.core.mongo import GQ_util_mongodb_client

__all__ = ['MIGRATE44_URI', 'client44', 'db44']

#: 4.4 服务器地址。默认值是**实测**的（`192.168.50.210:27017` = MongoDB 4.4.30，
#: 与 8.3 的 `57017` 同一台主机、不同实例）。运维换地址用环境变量或
#: CLI 的 `--migrate-source-uri`，不要改代码里的这行。
MIGRATE44_URI = os.getenv('GQ_MIGRATE44_URI', 'mongodb://192.168.50.210:27017')

_CLIENT = None


def client44(uri: str = None):
    """4.4 的客户端（进程内复用；`uri` 只在首次生效）。"""
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = GQ_util_mongodb_client(uri or MIGRATE44_URI)
    return _CLIENT


def db44(name: str, uri: str = None):
    """4.4 上的某个库（`'golemq'` / `'quantaxis'`）。

    :param uri: 显式地址（CLI 的 `--migrate-source-uri`）；None = 用 `MIGRATE44_URI`

    >>> from GolemQ.core import migrate44
    >>> migrate44.MIGRATE44_URI.startswith('mongodb://')
    True
    """
    if uri:
        return GQ_util_mongodb_client(uri)[name]
    return client44()[name]
