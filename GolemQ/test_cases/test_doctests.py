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
    # `datetime64` → 秒的**系数查表**（`epoch_divisor`）。判错只会**静默**把
    # `date_stamp` / `time_stamp` 写成另一个数量级（D52 / P35），所以把单位口径钉进 doctest。
    'GolemQ.core.constants',
    # ChartScan 的**换 env 路径推导**（`cv_python`）。纯字符串运算，只看
    # `sys.executable` 与 `$GOLEMQ_CV_PYTHON` —— 不碰 DB / 不起子进程。
    # 判错会**静默用错解释器**（GolemQ env 跑 torch 直接 BLAS 冲突），值得钉住。
    'GolemQ.cli.commands.chartscan',
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
    # `stock_reality_feat_day` 的**日频元数据契约**：`date_stamp` 口径（墙上时间当 UTC，差 8 小时也查得出来）、单位换算、行构造都是纯函数；`upsert_fields` 的语义也在这里有 doctest（多列共用文档时只 $set 自己的列）。
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
    # 资金流/千股千评的行构造与单位换算：`to_code`（int 陷阱）、
    # `parse_cn_amount`（`'3.72亿'`）、`parse_pct`（`'20.02%'`）、
    # `metadata_day_doc_multi`（多列文档）都是纯函数。
    'GolemQ.markets.StockCN.moneyflow_save',
    # 元数据对 K 线的对齐：`align_metadata_frame` 是纯 pandas，无 DB。
    'GolemQ.markets.StockCN.metadata_align',
    # RENKO 的 jit 版：`bricks_directions` 与 `evaluate_renko_jit` 是纯 numpy in/out。
    # ⚠️ numba 是**惰性** import（`_nb()`），但这两条 doctest 真会调它 ——
    # 收集本模块不需要 numba 在场，**跑这两条**需要（与 `regtree_jit` 同理）。
    'GolemQ.analysis.renko_jit',
    # 四量图 + 6 个下游指标的搬运（2026-10-11）。模块**只在模块级 import
    # numpy/pandas/talib**（不碰 DB/网络），对其余全树零引用。只给三个纯助手
    # （`sma`/`ema`/`llv`）挂了 doctest —— `calc_four_quadrant_boost` 那几个
    # 是**原地改 features** 的，不适合 doctest；它们的正确性证据是
    # `test_quadrant.py` 里与老树 `analysis/base.py` 的**逐值对拍**。
    'GolemQ.analysis.quadrant',
    # 上升系数的**自包含**生产路径（对数积水率三件套 + 编排）。
    # 纯函数（prophet/DB 都不碰；`talib` 只在 linearreg 那一支惰性 import）。
    'GolemQ.analysis.quadrant_coef',
    # 指标搬迁（2026-10-16，`DECISIONS.md` D49）新增的三个模块。
    # ⚠️ 三条都只给**纯函数**挂了 doctest：
    #   * `stages` 的 `garident_bands_to_ratio`/`_from_ratio`（比值 ↔ 绝对价互转）；
    #     `sklearn`/`kneed` 已改成**函数内惰性 import**，收集本模块不需要它们在场。
    #   * `timeseries` 的 `rolling_poly9_v2`（递归折半，输出与输入等长）。
    #   * `indicators` 的 `TA_HMA`（暖机 11 根 —— 那个数字与 `period=10` 绑死）。
    'GolemQ.analysis.stages',
    'GolemQ.analysis.timeseries',
    'GolemQ.analysis.indicators',
    # RSRS（阻力支撑相对强度）**指标计算**（2026-10-12 搬）。只搬 `pre_rsrs_data_func`
    # 一条（老树的 8 个择时信号与那个**坏掉的** `pre_rsrs_zscore_np_func` 都没搬）。
    # 它的 doctest 钉两件事：**列序**（5 列，顺序即契约）与**索引对齐**。
    'GolemQ.analysis.rsrs',
    # ChartScan（CV）结果的落点（2026-10-13 搬）。⚠️ 模块里**有碰库的函数**
    # （`save_reality_feat` / `fetch_reality_feat`），但按本文件的规矩**收不收是
    # 逐函数的** —— 只有纯函数 `reality_feat_name` / `reality_feat_rows` 挂了
    # doctest。它们钉的是**字段推导口径**（`time_stamp` 是「北京墙上时间当 UTC」
    # 而**不是** 8.3 的 `ts`；日族不带 `datetime`），判错会静默写坏整张特征表。
    'GolemQ.markets.StockCN.reality_save',
    # scan scope 解析（2026-10-13 从 `markets/StockCN/` **归位到根层**，见 D44）。
    # 模块级**零市场依赖**（数据经 `ScanDataset` 注入），碰库的默认路径
    # `_active_dataset()` 只在真调 `perpar_symbol_range()` 且不传 dataset 时才走。
    # doctest 喂一个**假 dataset**，钉的是「六个 scope 各自怎么切」。
    'GolemQ.analysis.cv.scopes',
    # 那份 `etf` 硬编码码表（506 个）随 D44 从 `scopes.py` 挪到本模块，与 `focus_block`
    # 并列 —— 它是 **A 股名单字面量**。模块级只 import 常量/pandas，不连库。
    'GolemQ.markets.StockCN.symbol',
    # ChartScan 的渲染与权重路径（2026-10-13 搬）。⚠️ 两者**模块级都不 import 重依赖**
    # （mplfinance / matplotlib / ultralytics 都在函数体里）⇒ 收集本模块**不需要 GolemAI env**，
    # 在 GolemQ env 下也进得来。doctest 钉 `calculate_i` 的像素↔bar 标定与权重路径的形状。
    'GolemQ.markets.StockCN.chartscan.render',
    'GolemQ.markets.StockCN.chartscan.model',
    # 客户端的 `market_type_of`（逐码定市场类型，**对旧 CLI 的刻意更正**）。
    # ⚠️ `server.py` / `client.py` **模块级不 import zmq**（只在 GolemAI env 有），
    # 故本模块在 GolemQ env 下也收集得动；`server.py` 没有 doctest，不入列。
    'GolemQ.markets.StockCN.chartscan.client',
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
