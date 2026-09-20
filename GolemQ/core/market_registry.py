# coding:utf-8
"""市场注册表与「当前激活市场」。

系统的隐含属性
==============
交易系统天然有一个**默认市场** —— 调用方说「取 600519 的日线」时，并没有指定
是哪个市场；这个信息由系统的当前状态提供。本模块把它显式化。

`fetch/` 下的门面函数**不含任何市场知识**，一律调度到 :func:`get_active_market`：

    # GolemQ/fetch/kline.py
    def get_kline_price_min(symbol, ...):
        return get_active_market().get_kline_price_min(symbol, ...)

好处：

* **加新市场 = 新增一个市场包**，不改任何抽象层
* **切换市场 = 一次 :func:`set_active_market` 调用**，而非改 import
* `features/` `models/` `pipeline/` `portfolio/` 里的代码完全不含市场判断

为什么注册表不放 `GolemQ/__init__.py`
====================================
`__init__.py` 应当只做导入与导出；注册与调度是有逻辑的（惰性发现、冲突检测、
明确报错），放在这里才可测、可读。`GolemQ/__init__.py` 仍**再导出** `GQMARKETS`
与 `GQSUBSCRIBER`，既有 `from GolemQ import GQMARKETS` 的写法不受影响。

⚠️ 惰性注册
===========
市场包在导入时自注册（`markets/StockCN/__init__.py`），但**谁先导入谁决定成败**。
故 :func:`get_active_market` 在注册表为空时先触发一次自动发现，
避免「先 import core 再 import 市场」这种顺序问题。
"""
from __future__ import annotations

#: 所有已注册的市场：`{市场名: 市场实例}`
GQMARKETS: dict = {}

#: 订阅者注册表：`{订阅键: 订阅函数}`
GQSUBSCRIBER: dict = {}

#: 默认市场。系统未显式切换时使用它。
DEFAULT_MARKET = 'StockCN'

#: 当前激活的市场名。私有 —— 只经 set/get 访问，避免被直接改写。
_active_market_name = DEFAULT_MARKET


def register_market(name: str, instance, replace: bool = False) -> bool:
    """把市场实例登记进注册表。

    :param replace: 默认 False —— 重复注册返回 False 而非静默覆盖。
        静默覆盖会让「哪个实例在用」变得不可知，是难查的 bug 来源。
    :returns: 是否真的写入了

    >>> class Fake:
    ...     name = 'FAKE'
    >>> register_market('__doctest__', Fake())
    True

    重复注册**不覆盖**，返回 False（而不是抛错，也不是静默替换）：

    >>> register_market('__doctest__', Fake())
    False

    显式 `replace=True` 才替换：

    >>> register_market('__doctest__', Fake(), replace=True)
    True

    用例结束后清掉，避免污染全局注册表：

    >>> GQMARKETS.pop('__doctest__', None) is not None
    True

    >>> register_market('', Fake())
    Traceback (most recent call last):
        ...
    ValueError: 市场名不能为空
    """
    if not name:
        raise ValueError('市场名不能为空')
    if instance is None:
        raise ValueError(f'市场 {name!r} 的实例不能为 None')
    if name in GQMARKETS and not replace:
        return False
    GQMARKETS[name] = instance
    return True


def register_subscriber(key: str, func, replace: bool = False) -> bool:
    """登记订阅者。`replace` 语义同 :func:`register_market`。"""
    if key in GQSUBSCRIBER and not replace:
        return False
    GQSUBSCRIBER[key] = func
    return True


def active_market_name() -> str:
    """当前激活的市场名。"""
    return _active_market_name


def set_active_market(name: str) -> str:
    """切换激活市场。

    **未注册则明确报错，不静默回落到默认市场** —— 回落的后果是「你以为是美股，
    实际取的是 A 股数据」，这种错误比直接失败危险得多。

    >>> import GolemQ.core.market_registry as reg
    >>> prev = reg._active_market_name          # 先存，用例结束要还原
    >>> class Fake:
    ...     name = 'US'
    >>> register_market('__doctest_us__', Fake())
    True
    >>> set_active_market('__doctest_us__')
    '__doctest_us__'
    >>> active_market_name()
    '__doctest_us__'

    未注册的市场**抛错**，不回落到默认：

    >>> set_active_market('NoSuchMarket')
    Traceback (most recent call last):
        ...
    KeyError: ...'NoSuchMarket' 未注册...

    ⚠️ 本函数改的是模块级全局状态，**doctest 必须自行还原**，
    否则会污染同一次 doctest 运行里的后续用例：

    >>> reg._active_market_name = prev
    >>> GQMARKETS.pop('__doctest_us__', None) is not None
    True
    """
    global _active_market_name
    if name not in GQMARKETS:
        raise KeyError(
            f'市场 {name!r} 未注册，无法激活。已注册: {sorted(GQMARKETS)}。'
            f'（若注册表为空，先 import 对应市场包，或调用 '
            f'GolemQ.cli.tools.auto_register_markets()）')
    _active_market_name = name
    return name


def _ensure_registered():
    """注册表为空时触发一次自动发现。

    **惰性导入 `cli.tools`** —— 它反过来要 import 本模块的 `GQMARKETS`，
    在模块顶层导入会成环。
    """
    if GQMARKETS:
        return
    try:
        from GolemQ.cli.tools import auto_register_markets
        auto_register_markets()
    except Exception:      # noqa: BLE001 发现失败不该伪装成「无市场」
        pass


def get_active_market():
    """返回当前激活的市场实例。

    注册表为空时先尝试自动发现 —— 市场包的自注册依赖导入顺序，
    不补救会让「先 import 了 core」的调用方莫名失败。
    """
    if not GQMARKETS:
        _ensure_registered()
    if _active_market_name not in GQMARKETS:
        raise KeyError(
            f'激活市场 {_active_market_name!r} 不在注册表内。已注册: '
            f'{sorted(GQMARKETS)}。请 import 对应市场包，或用 '
            f'set_active_market() 显式指定。')
    return GQMARKETS[_active_market_name]


def get_market(name: str):
    """按名取市场实例（不影响激活状态）。

    **与 :func:`get_active_market` 一样会触发惰性发现** —— 否则「按名取」与
    「取激活市场」在同一个空注册表下行为不一致：前者报未注册、后者自动发现。
    实测踩过：`get_kline_price_min(..., market=MARKET_TYPE.STOCK_HK)` 在
    全新进程里报「未注册」，而随后一次不传 market 的调用却成功注册了两个市场。
    """
    if name not in GQMARKETS:
        _ensure_registered()
    if name not in GQMARKETS:
        raise KeyError(f'市场 {name!r} 未注册。已注册: {sorted(GQMARKETS)}')
    return GQMARKETS[name]
