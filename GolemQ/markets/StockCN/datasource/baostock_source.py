# coding:utf-8
"""baostock 数据源适配器 —— **骨架，当前不可用**。

⚠️ 状态：`available()` 恒返回 False，原因见下
=============================================
2026-09-20 实测，**本机无法连接 baostock 服务器**：

    bs.login() -> 服务器连接失败，请稍后再试。
                  [WinError 10057] 由于套接字没有连接并且(当使用一个 sendto
                  调用发送数据报套接字时)没有提供地址，发送或接收数据的请求
                  没有被接受。
                  login: 10002007 网络接收错误。

``query_profit_data`` / ``query_balance_data`` / ``query_cash_flow_data`` /
``query_growth_data`` 均随之失败。**是网络层不通，不是接口不存在** ——
这四个接口在已装的 baostock 0.8.9 里都存在。

因此本模块**不实现 fetch**：写了也是无法验证的代码。等网络可达后再补，
接口已按 `DataSource` 契约留好位置。

它能供什么（据接口签名，未实测）
================================
================  =========================================================
`stock_list`      ``query_all_stock``
`stock_info`      ``query_stock_basic``
`financial`       ``query_profit_data`` / ``query_balance_data`` /
                  ``query_cash_flow_data`` / ``query_growth_data``（季频）
`stock_block`     ❌ 无板块接口
`etf_list`        ⚠️ 无专用 ETF 清单
================  =========================================================

注意：`financial` 现已由 akshare 提供且**已实测可用**，所以即便 baostock
网络恢复，它更适合作为 `financial` 的**备选源**而非主源。

⚠️ 一旦启用，必须走限频
=======================
baostock 有访问频率控制，且其接口是**逐只查询**（无批量），默认间隔取 30s。
"""
from __future__ import annotations

from GolemQ.datasource.base import (
    FINANCIAL,
    STOCK_INFO,
    STOCK_LIST,
    DataSource,
    DataSourceNotAvailable,
    UnsupportedCollection,
    register,
)

#: 实测失败时的错误码，写在这里便于日后比对是否同一原因
_LOGIN_ERROR_CODE = '10002007'


@register
class BaostockSource(DataSource):
    name = 'baostock'
    collections = (STOCK_LIST, STOCK_INFO, FINANCIAL)
    #: 有频率控制 + 逐只查询 —— 沿用 30s 默认间隔
    default_interval = 30.0

    def available(self) -> bool:
        """当前恒 False。

        不做「包在不在」的判断，因为包**在**（0.8.9）—— 不可用的是**网络**。
        如实返回 False，比返回 True 然后在 fetch 里炸更不容易误导调用方。
        网络恢复后把这里改成探测 ``bs.login()`` 的返回码即可。
        """
        return False

    def unavailable_reason(self) -> str:
        return (f'baostock 服务器不可达（登录返回 {_LOGIN_ERROR_CODE} '
                f'网络接收错误）；接口存在但网络层不通，未实现 fetch。')

    def fetch(self, collection: str, **kwargs) -> list:
        if not self.supports(collection):
            raise UnsupportedCollection(
                f'{self.name} 不提供 {collection}（可提供 {self.collections}）'
            )
        raise DataSourceNotAvailable(self.unavailable_reason())
