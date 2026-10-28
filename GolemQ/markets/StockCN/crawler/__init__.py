# coding:utf-8
"""外部网页/接口快照的抓取层 —— **只收"取数"型模块**。

边界（为什么只有这两个）
========================
`markets/StockCN/` 根目录的 `*_save.py` 是**落库编排**：写哪张表、增量水位、
payload 白名单（`kline_save` / `refdata_save` / `metadata_save` / `reality_save`）。

本包收的是另一类：**从外部站点抓当日快照**、顺带写进 `stock_reality_feat_day`
的模块。它们的主要身份是取数，不是写库，名字也不该带 `_save`：

* :mod:`.moneyflow` —— 东财千股千评（akshare）+ 同花顺当日资金流；
* :mod:`.diagnosis` —— 同花顺「牛叉诊股」逐只 HTML 抓取。

⚠️ 数据源适配器（pytdx/QMT/baostock）在 ``datasource/``，不在这里；
``fetch.py`` / ``realtime.py`` 是行情读取，也不在这里。
"""
from .diagnosis import (DIAG_DIMENSIONS, DIAG_PAYLOAD, fetch_diagnosis,
                        parse_diagnosis, rows_diagnosis, save_diagnosis)
from .moneyflow import (MONEYFLOW_SOURCES, last_trade_day, parse_cn_amount,
                        parse_pct, rows_eastmoney, rows_ths, save_moneyflow,
                        to_code)

__all__ = [
    'save_moneyflow', 'MONEYFLOW_SOURCES', 'rows_eastmoney', 'rows_ths',
    'parse_cn_amount', 'parse_pct', 'to_code', 'last_trade_day',
    'save_diagnosis', 'parse_diagnosis', 'fetch_diagnosis', 'rows_diagnosis',
    'DIAG_DIMENSIONS', 'DIAG_PAYLOAD',
]
