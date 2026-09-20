# coding:utf-8
"""把 docstring 里的 doctest 纳入 `unittest` 发现。

为什么需要这个文件
==================
`unittest discover` 只找 `test_*.py` 里的测试用例，**不会**去跑 docstring 里的
doctest。没有这个收集器，写在 docstring 里的例子等于写着好看 ——
它们会随代码腐烂而无人察觉，正是我们想避免的。

收集方式用 unittest 的 `load_tests` 协议，所以本文件被 discover 到即可，
不需要在 `run_tests.py` 里额外接线。

⚠️ 什么样的函数**不该**加 doctest
==================================
需要 MongoDB / 网络 / QMT 客户端才能跑的 —— 加了也是永远失败或永远跳过。
本项目的 `kline83` / `refdata*` / `datasource` 下的各适配器都属于这一类，
它们的正确性验证在 `PITFALLS.md` 记录的边界条件里，不在这里。
"""
from __future__ import annotations

import doctest
import importlib
import unittest

#: 含 doctest 的模块。**只收纯函数模块** —— 它们是 hermetic 的，能真跑。
#: 需要 DB/网络的那些（kline83、refdata*、datasource 适配器）故意不在列。
DOCTEST_MODULES = (
    # 仓位优化：数值口径可直接断言，且是「优化仓位」的核心
    'GolemQ.portfolio.sizing',
    'GolemQ.portfolio.costs',
    'GolemQ.portfolio.rules',
    'GolemQ.portfolio.strategy',
    # Active Market：默认市场指针，最容易被用错
    'GolemQ.core.market_registry',
    # 适配层的机制部分（无市场假设，纯逻辑）
    'GolemQ.datasource.base',
    'GolemQ.datasource.proxy',
    'GolemQ.datasource.throttle',
    # 门面的市场映射表
    'GolemQ.fetch.kline',
    # 日期助手：纯函数，且是 QUANTAXIS 解耦时逐个对齐过行为的
    'GolemQ.markets.StockCN.date_utils',
)


def _suite_for(module_name):
    """为一个模块建 doctest suite。导入失败时给**明确**的错误而非静默跳过。"""
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:            # noqa: BLE001
        # 不静默跳过：模块导不进来本身就是需要知道的事实
        raise unittest.SkipTest(f'{module_name} 导入失败: {exc!r}') from exc
    return doctest.DocTestSuite(
        module,
        optionflags=doctest.NORMALIZE_WHITESPACE | doctest.ELLIPSIS,
    )


def load_tests(loader, tests, ignore):
    """unittest 的收集协议：把各模块的 doctest 挂进本次发现。"""
    for name in DOCTEST_MODULES:
        try:
            tests.addTests(_suite_for(name))
        except unittest.SkipTest as skip:
            # 用一条显式的失败用例代替静默跳过 —— 导入不了是要修的，不是要忍的。
            # 用 FunctionTestCase 而非自定义 TestCase 子类：unittest 会按方法名
            # 实例化 TestCase，自定义 __init__ 签名会破坏发现（实测踩过）。
            tests.addTest(unittest.FunctionTestCase(
                _import_failure(name, str(skip)),
                description=f'doctest_import[{name}]'))
    return tests


def _import_failure(module_name, reason):
    """返回一个必然失败的函数，供 FunctionTestCase 包装。"""
    def _fail():
        raise AssertionError(f'doctest 模块不可导入：{reason}')
    return _fail
