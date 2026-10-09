# coding:utf-8
"""tushare 数据源适配器 —— **骨架，缺 token 未启用**。

⚠️ 状态：无 token 时 `available()` 返回 False
==============================================
tushare 1.4.21 已安装，但其所有接口都要求 `pro_api(token)`。

配置怎么加
==========
在 `~/.GolemQ/settings/config.ini` 里填：

```ini
[TUSHARE]
token = 你的 token
```

✅ **`'TUSHARE'` 已在 `core.settings.INI_ONLY_SECTIONS` 里**（2026-10-10 加）——
该段只走 INI，**不会**落到 Mongo 集合 `GolemQ.settings`。凭证放 INI 是刻意的，
也正因为进了那个元组才成立。（在此之前它虽然"能用"，但值会写进 Mongo。）
`[TDXAIDATA]` 同批加的（它原先也漏在外面，能读只是"INI 命中在前"的巧合。）

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

from GolemQ.datasource.base import (
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
        # ⚠️ 这段措辞 2026-10-10 改过：原文还在讲「要把 TUSHARE 加进 settings.py 的
        # 两处元组」—— 那件事**已经做了**（现在是 `core.settings.INI_ONLY_SECTIONS`），
        # 留着会把人引向一个不存在的待办（行号也早变了）。
        return ('未配置 tushare token —— 在 ~/.GolemQ/settings/config.ini 的 '
                '[TUSHARE] token 填上即可（该段属 INI_ONLY_SECTIONS，只读 INI，'
                '不落 Mongo 集合）')

    def fetch(self, collection: str, **kwargs) -> list:
        if not self.supports(collection):
            raise UnsupportedCollection(
                f'{self.name} 不提供 {collection}（可提供 {self.collections}）'
            )
        if not self.available():
            raise DataSourceNotAvailable(self.unavailable_reason())
        self.gate()
        raise DataSourceNotAvailable('tushare 适配器尚未实现（仅有骨架）')
