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

地址**不写死**
==============
4.4 与 8.3 是**同一台机上的两个实例**（实测：8.3 = `192.168.50.210:57017`，
4.4 = 同 IP `:27017`）。所以地址由 8.3 的 uri **推导**：换端口即可。

* 好处：换机器/换端口只改 `~/.GolemQ/settings/config.ini` 一处，
  **不会留下一份会腐烂的常量**，也不需要环境变量（D33 删掉的那个 `GQ_MIGRATE44_URI`
  就不必复活）。
* ⚠️ 同 IP 只是**当前**的部署事实，不是协议。若哪天 4.4 搬到了别的机器，
  推导就会指向错误的主机 —— 所以 :func:`mongo44_uri` 接受显式覆盖，且
  连不上时报错要**说清它推的是哪个地址**。
"""
from __future__ import annotations

from urllib.parse import quote_plus

from pymongo.uri_parser import parse_uri

__all__ = ['MONGO44_PORT', 'mongo44_uri', 'client44', 'db44']

#: 4.4 实例的端口。与 8.3（`config.ini` 里那个 57017）同机不同端口。
MONGO44_PORT = 27017


def mongo44_uri(uri=None, port: int = MONGO44_PORT) -> str:
    """8.3 的 uri → 4.4 的 uri（同主机，端口换成 `port`）。

    保留凭证与库名；**丢弃** 8.3 uri 上的查询选项（4.4 是另一个实例，
    `replicaSet` / `authSource` 之类未必通用 —— 传进来反而可能连不上）。
    需要选项时显式覆盖整个 uri。

    :param uri: 8.3 的 uri。``None`` 则读 `~/.GolemQ/settings/config.ini`。
    :param port: 4.4 的端口，默认 :data:`MONGO44_PORT`。

    实测的那一组（本机真值）：

    >>> mongo44_uri('mongodb://192.168.50.210:57017')
    'mongodb://192.168.50.210:27017'

    带凭证与库名时逐项保留、只换端口：

    >>> mongo44_uri('mongodb://u:p%40ss@10.0.0.8:57017/golemq')
    'mongodb://u:p%40ss@10.0.0.8:27017/golemq'

    端口由参数给定（4.4 若搬去别的端口，改这一处）：

    >>> mongo44_uri('mongodb://h:57017', port=27018)
    'mongodb://h:27018'
    """
    if uri is None:
        from GolemQ.core.settings import GQSETTING
        uri = GQSETTING.get_config('MONGODB', 'uri')

    parsed = parse_uri(uri)
    nodelist = parsed.get('nodelist') or []
    if not nodelist:
        raise ValueError(f'无法从 uri 解析出主机：{uri!r}')
    host = nodelist[0][0]

    auth = ''
    if parsed.get('username'):
        auth = quote_plus(parsed['username'])
        if parsed.get('password'):
            auth += ':' + quote_plus(parsed['password'])
        auth += '@'
    db = ''
    if parsed.get('database'):
        db = '/' + parsed['database']
    return f'mongodb://{auth}{host}:{port}{db}'


def client44(uri=None, port: int = MONGO44_PORT, **kwargs):
    """4.4 的 `MongoClient`。`kwargs` 直通 `pymongo.MongoClient`。

    ⚠️ **默认 30s 超时会让「4.4 没开」这件事卡半分钟** —— 调用方若只是探活，
    请传 ``serverSelectionTimeoutMS=1500``（`core/mongo.py` 的 docstring 同此提醒）。
    """
    from GolemQ.core.mongo import GQ_util_mongodb_client
    return GQ_util_mongodb_client(mongo44_uri(uri, port), **kwargs)


def db44(name: str, uri=None, port: int = MONGO44_PORT, **kwargs):
    """4.4 上名为 `name` 的库句柄（如 ``'golemq'`` / ``'quantaxis'``）。"""
    return client44(uri, port, **kwargs)[name]
