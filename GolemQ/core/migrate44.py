# coding:utf-8
"""**一次性**搬运用的 4.4 只读通道。

⚠️⚠️ **本模块只允许被一次性搬运命令引用。运行时代码禁止 import 它。**
（判据：`test_cases/test_no_quantaxis.py` 那类守卫的同一精神 —— 运行时只连 8.3。）

为什么它又回来了（D33 的修正）
==============================
2026-10-10 的 **D33** 把这条通道整条删了，理由是「4.4 → 8.3 搬运完成」。
**那个前提是错的**：当时只核了 `financial` 与关注列表，没核元数据类集合 ——
而 `golemq.stock_ranking`（5,809,496 行 / 2012-02-08 ~ 至今 / 5,573 只，
含筹码分布要用的 `TurnoverRate`）**从没搬过**。见 `DECISIONS.md` 的 D33 修正条。

所以：**通道恢复，但用途限定** —— 只读、只服务一次性搬运、不进任何运行时路径。

地址**只从配置文件来**
======================
4.4 的地址是**部署事实**，不是代码常量：写在
`~/.GolemQ/settings/config.ini` 的 ``[MONGODB44] uri``。
代码里**没有任何主机/端口默认值**；没配就报错并给出要补的 INI 片段。

```ini
[MONGODB44]
uri = mongodb://<host>:<port>
```
"""
from __future__ import annotations

__all__ = ['mongo44_uri', 'client44', 'db44']


def mongo44_uri(uri=None) -> str:
    """4.4 的 uri：显式参数优先，否则读 `config.ini` 的 ``[MONGODB44] uri``。

    ⚠️ **没有默认地址**。换电脑/换实例只改配置文件，不改代码。

    >>> mongo44_uri('mongodb://10.0.0.8:27017')
    'mongodb://10.0.0.8:27017'
    """
    if uri:
        return uri
    from GolemQ.core.settings import GQSETTING
    uri = GQSETTING.get_config('MONGODB44', 'uri', default_value='')
    if not uri:
        raise RuntimeError(
            '未配置 4.4：请在 ~/.GolemQ/settings/config.ini 增加\n'
            '[MONGODB44]\n'
            'uri = mongodb://<host>:<port>\n'
            '（代码不提供默认主机/端口。）')
    return uri


def client44(uri=None, **kwargs):
    """4.4 的 `MongoClient`。`kwargs` 直通 `pymongo.MongoClient`。

    ⚠️ **默认 30s 超时会让「4.4 没开」这件事卡半分钟** —— 调用方若只是探活，
    请传 ``serverSelectionTimeoutMS=1500``。
    """
    from GolemQ.core.mongo import GQ_util_mongodb_client
    return GQ_util_mongodb_client(mongo44_uri(uri), **kwargs)


def db44(name: str, uri=None, **kwargs):
    """4.4 上名为 `name` 的库句柄（如 ``'golemqq'`` / ``'quantaxis'``）。"""
    return client44(uri, **kwargs)[name]
