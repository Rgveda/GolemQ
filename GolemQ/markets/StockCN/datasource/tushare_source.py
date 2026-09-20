# coding:utf-8
"""tushare 数据源适配器 —— **骨架，缺 token 未启用**。

⚠️ 状态：无 token 时 `available()` 返回 False
==============================================
tushare 1.4.21 已安装，但其所有接口都要求 `pro_api(token)`。当前配置里
**没有 TUSHARE 段**，故无法调用。

配置怎么加（有个坑）
====================
`core/settings.py` 只把 `MONGODB` / `DINGTALK` / `SERVERCHAN` / `XTQUANT`
四个段当作 INI 支持（`:103` 读、`:132` 写），**其余段落入 Mongo 集合
`GolemQ.settings`**。所以 `GQSETTING.get_config('TUSHARE', 'token')` 现在
**能用**，但值存在 Mongo 里而不是 `config.ini`。

token 属凭证，放 INI 更合适 —— 需要把 `'TUSHARE'` 加进那两处的元组。
本模块不擅自改配置层，留给配置整理时一并做。

它能供什么（据接口能力，未实测）
================================
================  =========================================================
`stock_list`      ``stock_basic``
`stock_info`      ``stock_basic``（含 ``total_share``/``float_share``/``list_date``）
`etf_list`        ``fund_basic(market='E')``
`financial`       ``income`` / ``balancesheet`` / ``cashflow`` / ``fina_indicator``
`stock_block`     ``index_member`` / ``ths_index``（板块名空间与 QMT/TDX 不同）
================  =========================================================

tushare 是**字段最规范**的源，尤其 `financial`（英文字段名，无需中文键映射）。
但需凭证，且按积分限频。

⚠️ 一旦启用，必须走限频
=======================
tushare 按积分限制每分钟调用次数，默认间隔取 30s（可经
``[DATASOURCE] tushare_interval`` 覆盖）。
"""
from __future__ import annotations

from .base import (
    ETF_LIST,
    FINANCIAL,
    STOCK_BLOCK,
    STOCK_INFO,
    STOCK_LIST,
    DataSource,
    DataSourceNotAvailable,
    UnsupportedCollection,
    register,
)


@register
class TushareSource(DataSource):
    name = 'tushare'
    collections = (STOCK_LIST, STOCK_INFO, ETF_LIST, FINANCIAL, STOCK_BLOCK)
    #: 按积分限频 —— 沿用 30s 默认间隔
    default_interval = 30.0

    def token(self):
        try:
            from GolemQ.core.settings import GQSETTING
            return (GQSETTING.get_config('TUSHARE', 'token', '') or '').strip()
        except Exception:  # noqa: BLE001
            return ''

    def available(self) -> bool:
        """包在 + 有 token 才算可用。缺任一项返回 False，不抛异常。"""
        try:
            import tushare  # noqa: F401
        except ImportError:
            return False
        return bool(self.token())

    def unavailable_reason(self) -> str:
        try:
            import tushare  # noqa: F401
        except ImportError:
            return 'tushare 未安装'
        return ('未配置 tushare token。当前 GQSETTING.get_config("TUSHARE","token") '
                '为空；注意该段落入的是 Mongo 集合 GolemQ.settings 而非 config.ini，'
                '要放进 INI 需把 "TUSHARE" 加进 core/settings.py:103 与 :132 的元组。')

    def fetch(self, collection: str, **kwargs) -> list:
        if not self.supports(collection):
            raise UnsupportedCollection(
                f'{self.name} 不提供 {collection}（可提供 {self.collections}）'
            )
        if not self.available():
            raise DataSourceNotAvailable(self.unavailable_reason())
        self.gate()
        raise DataSourceNotAvailable('tushare 适配器尚未实现（仅有骨架）')
