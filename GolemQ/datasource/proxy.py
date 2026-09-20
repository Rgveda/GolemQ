# coding:utf-8
"""代理注入点。

**代理池本身没有实现，这是刻意的。**
================================
一个可靠的免费代理池需要持续维护与可用性验证；仓促造一个只会把随机失败引入
取数链路，排障时还会被误当成上游限频。当前策略是 **限频优先 + 代理接口预留**
（见 `throttle.py`）。

日后若确实需要，实现方式是给本模块补一个 pool provider：由它提供
``next_url()``，`apply()` 每取一次就换一个。注入点已经是单一位置，届时不必
改动任何数据源适配器。

⚠️ 代理只对 HTTP 源有效，**对 pytdx 是空的**
==========================================
pytdx 走的是通达信私有 TCP 协议（``TdxHq_API.connect(ip, port)``），
**不是 HTTP**，requests 的 ``proxies`` 对它没有任何作用。

pytdx 的等价物是**服务器地址列表**（那是一个中继/轮换，不是代理），应当作为
独立的 ``hosts`` 配置暴露 —— **不要把它和 HTTP 代理混为一谈**，否则会给人
「代理已覆盖所有源」的错觉，而实际上 TCP 那条路从未被代理过。
"""
from __future__ import annotations

import os


class ProxyConfig:
    """从配置读出的代理设置。未配置即禁用。"""

    def __init__(self, url: str = None):
        self.url = (url or '').strip() or None

    @classmethod
    def from_settings(cls, section: str = 'DATASOURCE') -> 'ProxyConfig':
        url = ''
        try:
            from GolemQ.core.settings import GQSETTING
            url = GQSETTING.get_config(section, 'proxy', '') or ''
        except Exception:
            # 配置读取失败按「未配置」处理 —— 无代理可直连，不该因此起不来
            url = ''
        return cls(url)

    @property
    def enabled(self) -> bool:
        return self.url is not None

    def apply(self, session):
        """把代理挂到 requests.Session。未配置则原样返回（直连）。"""
        if self.url:
            session.proxies.update({'http': self.url, 'https': self.url})
        return session

    def apply_env(self):
        """给不暴露 session 的库（如 akshare）用。仅在启用时设置环境变量。

        akshare 内部自建请求、不接外部 session，所以只能经环境变量影响它。
        这是同一个注入点的另一种落法，不是第二个机制。
        """
        if self.url:
            os.environ.setdefault('HTTP_PROXY', self.url)
            os.environ.setdefault('HTTPS_PROXY', self.url)
        return self.enabled
