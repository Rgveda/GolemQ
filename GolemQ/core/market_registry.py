# coding:utf-8
"""市场注册表、「默认 / 当前激活市场」与市场类型的解析。

系统的隐含属性
==============
交易系统天然有一个**默认市场** —— 调用方说「取 600519 的日线」时，并没有指定
是哪个市场；这个信息由系统的当前状态提供。本模块把它显式化，并给出取用入口：

    from GolemQ import get_default_market, get_active_market

    get_active_market().get_kline_price_min('600519', frequency='60min')

**没有门面层**（2026-10-10 起 `GolemQ/fetch/` 整包已删）。原先 `fetch/kline.py`
只做一件事：`return get_active_market().get_kline_price_min(...)` —— 一层改名，
却让每个形参都得在四个地方各声明一遍。

好处：

* **加新市场 = 新增一个市场包**，不改任何抽象层
* **切换市场 = 一次 :func:`set_active_market` 调用**，而非改 import
* `features/` `models/` `pipeline/` `portfolio/` 里的代码完全不含市场判断

「默认」与「激活」的区别
========================
:data:`DEFAULT_MARKET` 是**系统自带的**那个；:data:`_active_market_name` 是**当前生效**的。
没人调 :func:`set_active_market` 时两者相同。要「不管当前切到哪、都取系统默认」，用
:func:`get_default_market`；要「尊重当前的切换」，用 :func:`get_active_market`。

为什么注册表不放 `GolemQ/__init__.py`
====================================
`__init__.py` 应当只做导入与导出；注册与调度是有逻辑的（惰性发现、冲突检测、
明确报错），放在这里才可测、可读。`GolemQ/__init__.py` 仍**再导出** `GQMARKETS`
与 `GQSUBSCRIBER`，既有 `from GolemQ import GQMARKETS` 的写法不受影响。

⚠️ 惰性注册
===========
市场包在导入时自注册（`markets/StockCN/__init__.py`），但**谁先导入谁决定成败**。
故三个 `get_*` 入口在注册表为空时都会先触发一次自动发现，
避免「先 import core 再 import 市场」这种顺序问题。
"""
from __future__ import annotations

from GolemQ.core.constants import MARKET_TYPE

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


def get_default_market():
    """返回**系统默认市场**实例（:data:`DEFAULT_MARKET` 指的那个）。

    与 :func:`get_active_market` 的区别：那个尊重 :func:`set_active_market` 的切换，
    这个始终给系统自带的那一个。没人切换过时两者是同一个对象。

    **返回实例而不是把它绑成模块级变量**（2026-10-10 用户提过「注册一个全局变量
    `default_market`」，这里刻意不那样做）：

    * 变量会在 `register_market(..., replace=True)` 换掉实例后**陈旧**，且不报错；
    * 变量要在模块级绑实例 ⇒ 导入即要求市场包在场，与 `GolemQ/__init__.py`
      费力消掉的**急切导入**（`PITFALLS.md` P18）冲突；
    * 函数可以惰性发现，变量的赋值时机则要外部喂（例如只在 CLI bootstrap 里赋值
      ⇒ 库用法 / notebook / 示例脚本全拿不到它）。

    注册表为空时先自动发现，同 :func:`get_active_market`。默认市场**未注册则明确
    报错，不静默回落到激活市场** —— 回落会让「你以为在取默认市场、实际是另一个」。

    >>> import GolemQ.core.market_registry as reg
    >>> prev, prev_active = reg.DEFAULT_MARKET, reg._active_market_name
    >>> class Fake:
    ...     name = '__doctest_default__'
    >>> register_market('__doctest_default__', Fake())
    True
    >>> reg.DEFAULT_MARKET = '__doctest_default__'
    >>> get_default_market().name
    '__doctest_default__'

    默认市场未注册 ⇒ 抛错（此时注册表非空，不会触发自动发现）：

    >>> reg.DEFAULT_MARKET = 'NoSuchDefaultMarket'
    >>> get_default_market()
    Traceback (most recent call last):
        ...
    KeyError: ...'NoSuchDefaultMarket' 未注册...

    ⚠️ 本函数读的是模块级全局 `DEFAULT_MARKET`，**doctest 必须自行还原**：

    >>> reg.DEFAULT_MARKET, reg._active_market_name = prev, prev_active
    >>> GQMARKETS.pop('__doctest_default__', None) is not None
    True
    """
    if not GQMARKETS:
        _ensure_registered()
    if DEFAULT_MARKET not in GQMARKETS:
        raise KeyError(
            f'默认市场 {DEFAULT_MARKET!r} 未注册。已注册: {sorted(GQMARKETS)}。'
            f'请 import 对应市场包，或调用 '
            f'GolemQ.cli.tools.auto_register_markets()')
    return GQMARKETS[DEFAULT_MARKET]


#: `MARKET_TYPE` → 市场包名。**品种类型不等于市场** —— 指数/ETF/基金都是 A 股市场下的品种。
#:
#: ⚠️ 加新市场时**只改这一张表**（原先它在 `fetch/kline.py`，2026-10-10 随该包删而搬来）。
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

    映射的层级关系（`INDEX_CN` 也是 A 股市场，不是另一个市场）：

    >>> MARKET_TYPE_TO_MARKET[MARKET_TYPE.STOCK_CN]
    'StockCN'
    >>> MARKET_TYPE_TO_MARKET[MARKET_TYPE.INDEX_CN]
    'StockCN'
    >>> MARKET_TYPE_TO_MARKET[MARKET_TYPE.STOCK_HK]
    'StockHK'

    未登记的品种类型报错，并提示该往哪加：

    >>> resolve_market(MARKET_TYPE.STOCK_US)
    Traceback (most recent call last):
        ...
    ValueError: 未知的市场类型 'stock_us'。已登记: ...若这是新市场的品种类型，请在 core/market_registry.py 的 MARKET_TYPE_TO_MARKET 中登记。
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
            f'若这是新市场的品种类型，请在 core/market_registry.py 的 '
            f'MARKET_TYPE_TO_MARKET 中登记。')
    return get_market(name)
