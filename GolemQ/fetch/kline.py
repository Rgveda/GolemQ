# coding:utf-8
"""K 线获取的**门面** —— 不含任何市场知识，一律调度到市场实现。

接口形态
========
    get_kline_price_min(symbol, ...)                        # 隐含默认市场
    get_kline_price_min(symbol, ..., market=MARKET_TYPE.STOCK_CN)   # 显式指定

**不传 `market` 用当前激活市场**（`core.market_registry`），传了则单次覆盖 ——
后者让你不必为了查一次别的市场而改动全局状态。

调度链
======
    services/persistence/*              调用方
            ↓
    fetch/kline.py                      门面（本模块）：定义契约 + 调度
            ↓
    BaseMarket.get_kline_price_min()    抽象声明
            ↓
    markets/StockCN/__init__.py         市场实现
            ↓
    markets/StockCN/kline83.py          A 股的 8.3 时序读法

`market` 参数的一个前提
======================
`MARKET_TYPE` 同时装了**两个层级**的东西：

* **市场归属** —— `STOCK_CN` / `STOCK_HK` / `STOCK_US` / `CRYPTOCURRENCY`
* **品种类型** —— `INDEX_CN` / `ETF_CN` / `FUND_CN` / `BOND_CN` / `FUTURE_CN` /
  `OPTION_CN` / `STOCKOPTION_CN` / `STOCK_CN_B` / `STOCK_CN_D`

**`INDEX_CN`（A股指数）仍然属于 `StockCN` 市场**，不是另一个市场。故本模块的
`market` 参数按**市场归属**分发；品种类型由市场内部判定
（`markets/StockCN/kline83.py` 用 `is_stock_cn()` 决定读 `stock_*` 还是 `index_*`）。

这张映射表就是「层级」这一事实的落点 —— 新增市场时在这里加一行。

空值契约（沿用既有约定，勿改）
==============================
* `get_kline_price_v3`（日线）无数据 → 返回 `None`，命中 `_daily.py:105` 的既有分支
* `get_kline_price_min`（分钟）无数据 → 返回**空对象**，因 `_stock.py:138` 无 `None`
  保护且未预初始化该变量，返回 `None` 会一路变成 `UnboundLocalError`

详见 `markets/StockCN/MONGODB83.md`。
"""
from __future__ import annotations

from GolemQ.core.constants import MARKET_TYPE
from GolemQ.core.market_registry import get_active_market, get_market

__all__ = ['get_kline_price_min', 'get_kline_price_v3', 'resolve_market',
           'MARKET_TYPE_TO_MARKET']

#: `MARKET_TYPE` → 市场包名。
#:
#: **注意层级**：左侧多数是「品种类型」而非「市场」，它们同属一个市场包 ——
#: A股的指数/ETF/基金/债券/期货/期权都由 `StockCN` 提供，只是集合不同。
#: 新增市场时在此加一行（如 `MARKET_TYPE.STOCK_US: 'StockUS'`）。
MARKET_TYPE_TO_MARKET = {
    # —— A 股市场下的各品种 ——
    MARKET_TYPE.STOCK_CN: 'StockCN',
    MARKET_TYPE.STOCK_CN_B: 'StockCN',
    MARKET_TYPE.STOCK_CN_D: 'StockCN',
    MARKET_TYPE.INDEX_CN: 'StockCN',
    MARKET_TYPE.ETF_CN: 'StockCN',
    MARKET_TYPE.FUND_CN: 'StockCN',
    MARKET_TYPE.BOND_CN: 'StockCN',
    MARKET_TYPE.FUTURE_CN: 'StockCN',
    MARKET_TYPE.OPTION_CN: 'StockCN',
    MARKET_TYPE.STOCKOPTION_CN: 'StockCN',
    # —— 其他市场（待实现）——
    MARKET_TYPE.STOCK_HK: 'StockHK',
    # MARKET_TYPE.STOCK_US: 'StockUS',        # 市场包尚未实现
    # MARKET_TYPE.CRYPTOCURRENCY: 'Crypto',   # 市场包尚未实现
}


def resolve_market(market=None):
    """把 `market` 参数解析成市场实例。

    `None` → 当前激活市场（**隐含默认市场**）。
    给了值 → 按 :data:`MARKET_TYPE_TO_MARKET` 找到市场包。

    **未知值明确报错，不静默回落** —— 回落会让你「以为在取美股、实际拿到 A 股」，
    比直接失败危险得多。
    """
    if market is None:
        return get_active_market()
    if isinstance(market, str) and market in ('StockCN', 'StockHK'):
        # 允许直接传市场包名（内部调用与测试方便）
        return get_market(market)
    name = MARKET_TYPE_TO_MARKET.get(market)
    if name is None:
        raise ValueError(
            f'未知的市场类型 {market!r}。已登记: {sorted(MARKET_TYPE_TO_MARKET)}。'
            f'若这是新市场的品种类型，请在 fetch/kline.py 的 '
            f'MARKET_TYPE_TO_MARKET 中登记。')
    return get_market(name)


def get_kline_price_min(symbol, start=None, end=None, verbose=False,
                        realtime=True, market=None):
    """分钟线。`market` 省略则用当前激活市场。"""
    return resolve_market(market).get_kline_price_min(
        symbol, start=start, end=end, verbose=verbose, realtime=realtime)


def get_kline_price_v3(symbol, start=None, end=None, verbose=False,
                       realtime=True, market=None):
    """日线。`market` 省略则用当前激活市场。

    无数据时返回 `None`（不是空对象）—— 调用方 `_daily.py:105` 依赖这一点。
    """
    return resolve_market(market).get_kline_price_v3(
        symbol, start=start, end=end, verbose=verbose, realtime=realtime)
