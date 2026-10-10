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
本项目的 `refdata*` 与 `datasource` 下的各适配器都属于这一类，
它们的正确性验证在 `PITFALLS.md` 记录的边界条件里，不在这里。

⚠️ 判断依据是**函数**而非**模块**：`kline83` 里既有碰库的读取器，也有纯函数
（如 `market_prefix` / `normalize_frequency`）。给纯函数加 doctest 是合规的，
只要别把碰库的那些也加进来。`markets/StockCN/symbol.py` 同理。
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
    # 市场映射表与解析（原在 fetch/kline.py，2026-10-10 随该包删而搬进注册表）
    # —— 不在这里另列一行：`GolemQ.core.market_registry` 上面已经收过一次了。
    # 日期助手：纯函数，且是 QUANTAXIS 解耦时逐个对齐过行为的
    'GolemQ.markets.StockCN.date_utils',
    # 号段分类器：纯字符串判断（`symbol` 模块本身会 import pymongo，但
    # `is_stock_cn` / `_match_segment` 不碰库 —— 已验证关着 MongoDB 也能导入）。
    # 它驱动集合路由，判错只静默读错集合，所以值得把口径写进 doctest 钉住。
    'GolemQ.markets.StockCN.symbol',
    # 集合路由：纯判断，但**驱动读哪张表**（stock_* / index_* / etf_*）。
    # 判错只会静默读错集合，故把三值口径写进 doctest。本模块的读取器碰库，
    # 但它们没有 doctest，不受影响。
    'GolemQ.markets.StockCN.kline83',
    # 复权的**纯函数核心**：股票与 ETF 共用。刻意不碰数据库（因子取数留在两个
    # 来源模块里），所以能进这个列表。两处调用方都靠它，且各自的数值基准
    # （股票 40,192 行 vs QUANTAXIS、ETF 独立重算）都依赖这些函数。
    'GolemQ.markets.StockCN.fq',
    # 实时/history 落库的**写侧契约**：pytdx bar → 8.3 时序文档的字段映射与分页数学。
    # 纯函数（不碰 DB/网络），判错会静默写坏库里每一根 bar，故把口径写进 doctest 钉住。
    'GolemQ.markets.StockCN.kline_doc',
    # 状态 banner 的**渲染**部分（`render_refdata_banner` / `ansi_enabled`）：
    # 纯函数、不看环境（上不上色由调用方传），故能钉。模块本身不是纯函数模块
    # （还拖 pandas/tqdm/joblib），但按本文件的规矩，**收不收是逐函数的**。
    'GolemQ.core.presentation',
    # CLI 环境自检里的**纯函数**（版本比较）。模块本身会 lazily import
    # `core.presentation` / `core.mongo`，但那些都在函数体内，收集时不会跑。
    'GolemQ.cli.bootstrap',
    # 服务器池的**纯函数**（回环判据）。模块本身 lazily import pytdx，
    # 收集时不会碰网络。
    'GolemQ.markets.StockCN.datasource.tdx_hosts',
    # 缠论中枢的**算法核心** `find_zs`：只用 max/min 与入参 dict，
    # 连 numpy 都不碰，是最纯不过的一段。它的口径错一点，整条中枢链都跟着错。
    'GolemQ.analysis._zs',
    # 中枢层的**纯函数部分**：`classify_pivots`（走势分类）/ `pivots_to_df` /
    # `causal_pivot_series`（逐 bar 因果展开）都不碰 DB 与网络。
    # ⚠️ czsc 是在函数体内 lazily import 的（`bi_confirm_map` / `attach_pivot_features`），
    # 所以收集本模块**不需要** czsc 在场 —— 上面那三条 doctest 也确实是 czsc-free 的。
    'GolemQ.analysis.pivot',
    # `stock_metadata_day` 的**日频元数据契约**：`date_stamp` 口径（墙上时间当 UTC，差 8 小时也查得出来）、单位换算、行构造都是纯函数；`upsert_fields` 的语义也在这里有 doctest（多列共用文档时只 $set 自己的列）。
    'GolemQ.markets.StockCN.metadata_save',
    # 时序累积器与金叉/死叉间隔：纯函数（numpy/pandas），从旧树 `analysis/timeseries.py`
    # 搬回的真实现。⚠️ 其中 `Timeline_duration` 曾是本树 stub 并被当死代码删过 ——
    # 复活的东西必须有测试，否则下次还会被删。见 GLOSSARY「stub vs dummy」。
    'GolemQ.analysis.timing',
    # PEAK_POINT（鲁棒极值识别）：纯函数（numpy/pandas），numba 是**惰性 import**，
    # 缺它时纯实现顶上 —— 故收集本模块不需要 numba 在场。
    'GolemQ.analysis.peak',
    # 持仓浮动收益：显式循环的纯函数（旧树就是为 JIT 写的）。
    'GolemQ.portfolio.returns',
    # 砖块图（RENKO）：`renko_chart` 是**纯 numpy in/out**（无常量、不碰 DB），
    # 也是本模块唯一值得挂 doctest 的入口 —— 它那条 doctest 钉住了一个反直觉的
    # 编码：**价格列把方向编码在符号里**（下跌砖是负数），末行 `-11.0` 是刻意的。
    # ⚠️ 本模块**模块级**导入 numba/scipy/talib（三者都在 MIN_PACKAGES），
    # 所以收集它比别的条目慢一点点 —— 但不需要 DB 与网络，仍是 hermetic 的。
    'GolemQ.analysis.renko',
    # RENKO 的 jit 版：`bricks_directions` 与 `evaluate_renko_jit` 是纯 numpy in/out。
    # ⚠️ numba 是**惰性** import（`_nb()`），但这两条 doctest 真会调它 ——
    # 收集本模块不需要 numba 在场，**跑这两条**需要（与 `regtree_jit` 同理）。
    'GolemQ.analysis.renko_jit',
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
