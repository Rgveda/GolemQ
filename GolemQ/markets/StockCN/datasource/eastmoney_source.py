# coding:utf-8
"""东方财富数据源适配器 —— **骨架，尚未实现**。

⚠️ 现有 eastmoney 代码不服务于这 5 个集合
=========================================
新树里已有直连东方财富的代码，但用途不同：

* `markets/StockCN/realtime.py:957`  `stock_individual_fund_flow_push`
* `markets/StockCN/realtime.py:1052` `get_moneyflow_from_eastmoney_push`

两者取的都是**个股资金流**（主力/超大/大/中/小单），**不属于这 5 个参考集合**。
所以本适配器**不能靠包装现有代码完成**，需要新的接口实现。

网络连通性不是问题 —— 上述两处在用，说明 `push2.eastmoney.com` 可达。
缺的是接口实现本身。

它可能供什么（据公开接口，未实测）
==================================
================  =============================================================
`stock_block`     ``push2.eastmoney.com`` 的概念/行业板块成分接口，可能是**最合适**
                    的一个 —— 东财的板块分类与 QMT/TDX 是三套不同的名空间，互为补充
`etf_list`        ETF 全量列表接口
`stock_list`      A 股列表接口（但 pytdx 已覆盖且无需派生字段）
`stock_info`      ❌ 无股本
`financial`       ⚠️ 有，但 akshare 已实测可用且更规整
================  =============================================================

⚠️ 东财有访问频率控制
=====================
与 akshare/baostock 同档，默认间隔 30s。

**未实现 fetch** —— 接口踏勘需要先确认返回结构与分页方式，不能照猜测写。
能力声明留空，踏勘后再填；届时自动获得注册/限频/代理/降级的整套能力。
"""
from __future__ import annotations

from .base import DataSource, DataSourceNotAvailable, register


@register
class EastmoneySource(DataSource):
    name = 'eastmoney'
    #: 接口未踏勘，不声明 —— 声明了就是承诺
    collections = ()
    #: 有频率控制
    default_interval = 30.0

    def available(self) -> bool:
        """网络可达性与接口实现是两回事。

        现有 moneyflow 代码能通，只说明域名可达，**不代表本适配器的集合可取**。
        故返回 False 表示「本适配器不可用」，而不是「东财不可达」。
        """
        return False

    def unavailable_reason(self) -> str:
        return ('eastmoney 适配器尚未实现。现有 eastmoney 代码（realtime.py:957/1052）'
                '取的是个股资金流，不属于这 5 个参考集合，无法直接包装；'
                '需按 push2.eastmoney.com 的板块/ETF 接口另做踏勘与实现。')

    def fetch(self, collection: str, **kwargs) -> list:
        raise DataSourceNotAvailable(self.unavailable_reason())
