#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""`analysis/renko.py` —— 与旧树 `indices/renko.py` 的**逐值对拍**。

为什么这份用例是这次搬运的**主要证据**
======================================
RENKO 那 5 个函数是从 1065 行里挑出来的活链，搬运时做了**五处**改动：
改 import ×2、删 QUANTAXUS 透传、删全部 `print`、补 14 个字段常量。
这些改动**都不该改变数值** —— 但"不该"不是证据，本用例把它钉住。

⚠️ **判据一律是 `array_equal`（逐值相同），不用容差。** 理由见 `PITFALLS.md` P30：
砖块图与缠论同族，是**离散决策**（砖进没进、方向 ±1）；末位浮点差异足以翻转它，
而任何 `assert_allclose` 都会对这种翻转**视而不见**。
（唯一的例外是 S 族被 `float16` cast 过 —— 比之前统一 `astype(float64)`，
那是**无损放大**，不是放容差。）

为什么不连库
============
端到端那一条用**合成行情**（固定种子的随机游走），不读 MongoDB。
理由有二：① 无 DB 的用例才跑得动、跑得稳；② 对拍要的是"两条实现同输入同输出"，
输入从哪来不影响这个判断。

旧模块怎么载入（harness）
========================
旧 `renko.py` 用**绝对 import** `from GolemQ.utils.parameter import …` 与
`from GolemQ.analysis.timeseries import *`，而新树没有这两个模块
（常量并进了 `core.constants`，时序函数搬去了 `analysis.timing`）。所以：

1. 先往 `sys.modules` 注入两片 **shim**；
2. 再把 `GolemQ_old/indices/renko.py` 按**命名空间包**加载，
   绕开 `GolemQ_old/__init__.py` 的一堆绝对导入 —— 与 `analysis/pivot.py`
   搬运时用的手法同一套。

⚠️ 两处**踩过的坑**，改 harness 前先看：

* `GolemQ.analysis.timeseries` 的 shim 必须是**在真模块上补属性**，**不能替换**：
  新树**本来就有** `analysis/timeseries.py`（`GQ_data_min_resample` 在那儿），
  替换掉会让 `StockCN` 注册失败（实测：「cannot import name
  'GQ_data_min_resample'」→ `KeyError: 激活市场 'StockCN' 不在注册表内`）。
* 旧模块用 `ST.VERBOSE` 做调试门控，而新树**有意没补**这个常量（用户
  2026-10-10：「ST.VERBOSE 部分代码去除」）。所以 shim 里单独给一个
  带 `VERBOSE` 的替身；我们的测试帧没有 `verbose` 列，门控恒为假、不触发。

老树不在场时整份**跳过**（同 `test_timing.py` 惯例）。
"""

import os
import sys
import types
import importlib

try:
    import GolemQ  # noqa: F401
except ImportError:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import unittest

import numpy as np
import pandas as pd

from GolemQ.analysis import renko as NEW

_OLD_TREE = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..', 'GolemQ_old'))
_OLD_FILE = os.path.join(_OLD_TREE, 'indices', 'renko.py')


def _load_old_renko():
    """载入旧 `indices/renko.py`（见模块 docstring 的 harness 说明）。不存在则返回 ``None``。

    ⚠️ **注入的 shim 用完必须清干净**。第一版没清，于是把
    `QUANTAXIS*` 留在了 `sys.modules` 里 —— 同一个测试进程后面的
    `test_no_quantaxis.TestQuantaxisNotLoadedAtRuntime` 当场红：
    `['QUANTAXIS', 'QUANTAXIS.QAIndicator', ...] != []`。
    这个测试**跨用例污染全局状态**，所以清理与注入同样承重。
    """
    if not os.path.isfile(_OLD_FILE):
        return None
    from GolemQ.core import constants as _C
    from GolemQ.analysis import timing as _T

    qa_names = ('QUANTAXIS', 'QUANTAXIS.QAIndicator',
                'QUANTAXIS.QAIndicator.talib_numpy')

    # ① 时序：**在真模块上补属性**，绝不替换（替换会打断 StockCN 注册）
    import GolemQ.analysis.timeseries as _ts
    _ts_added = {k: getattr(_ts, k, None) for k in
                 ('Timeline_Integral', 'Timeline_duration',
                  'QA_util_timestamp_to_str')}
    _ts.Timeline_Integral = _T.Timeline_Integral
    _ts.Timeline_duration = _T.Timeline_duration
    _ts.QA_util_timestamp_to_str = lambda *a, **k: ''

    # ② QUANTAXIS：旧模块 `import QUANTAXIS as QA` 是历史残渣（活在 __main__ 里），
    #    给空模块即可 —— 顺带省掉真导入 QUANTAXIS 的几秒
    qa_added = [n for n in qa_names if n not in sys.modules]
    for name in qa_added:
        sys.modules[name] = types.ModuleType(name)

    # ③ 常量：`GolemQ.utils.parameter` 在新树不存在，映射到 `core.constants`
    utils_created = 'GolemQ.utils' not in sys.modules
    if utils_created:
        sys.modules['GolemQ.utils'] = types.ModuleType('GolemQ.utils')
    shim = types.ModuleType('GolemQ.utils.parameter')
    shim.AKA = _C.AKA
    shim.INDICATOR_FIELD = _C.FIELD

    class _ST:                       # 新树有意不补 VERBOSE（见模块 docstring）
        VERBOSE = 'verbose'
    shim.TREND_STATUS = _ST
    sys.modules['GolemQ.utils.parameter'] = shim

    # ④ 命名空间包加载，绕开 GolemQ_old/__init__.py
    pkg = types.ModuleType('_oldrenko')
    pkg.__path__ = [os.path.join(_OLD_TREE, 'indices')]
    sys.modules['_oldrenko'] = pkg
    try:
        return importlib.import_module('_oldrenko.renko')
    finally:
        # ⑤ **清理**：旧模块已经 `import *` 把名字收进自己的 globals，
        #    此后它不再需要 sys.modules 里这些条目 —— 留着就是污染别人。
        for name in qa_added:
            sys.modules.pop(name, None)
        sys.modules.pop('GolemQ.utils.parameter', None)
        if utils_created:
            sys.modules.pop('GolemQ.utils', None)
        for k, old in _ts_added.items():
            if old is None:
                try:
                    delattr(_ts, k)
                except AttributeError:
                    pass
            else:
                setattr(_ts, k, old)


OLD = _load_old_renko()


def _ohlc(n=420, seed=0):
    """合成一份含 hlc 的行情帧（MultiIndex ``(date, code)``）。

    ⚠️ 列名必须是 `high` / `low` / `open` / `close` 四个**小写**名 ——
    `renko_in_cluster_group` 用的是 `data.high` 这种**属性访问**，改名即 AttributeError。
    """
    rng = np.random.RandomState(seed)
    close = 20 + np.cumsum(rng.randn(n)) * 0.3
    idx = pd.MultiIndex.from_arrays(
        [pd.date_range('2024-01-01', periods=n, freq='D'), ['600519'] * n],
        names=['date', 'code'])
    return pd.DataFrame({'open': close + rng.randn(n) * 0.1,
                         'high': close + np.abs(rng.randn(n)) * 0.2,
                         'low': close - np.abs(rng.randn(n)) * 0.2,
                         'close': close,
                         'volume': np.full(n, 1e5)}, index=idx)


@unittest.skipIf(OLD is None, '旧树不在场 —— 对拍跳过')
class TestRenkoChartMatches(unittest.TestCase):
    """`renko_chart` 是纯 numpy in/out、无常量依赖 —— 最干净的一层。"""

    def test_bit_identical_on_random_walks(self):
        rng = np.random.RandomState(0)
        for trial in range(5):
            n = rng.randint(60, 300)
            N = float(rng.uniform(0.3, 2.0))
            px = (100 + np.cumsum(rng.randn(n)) * 0.8).astype(np.float64)
            with self.subTest(trial=trial, n=n, N=N):
                np.testing.assert_array_equal(NEW.renko_chart(px, N),
                                              OLD.renko_chart(px, N))

    def test_negative_price_encodes_direction(self):
        """价格列**把方向编码在符号里** —— 下跌砖是负数，别当 bug 修。"""
        out = NEW.renko_chart(np.array([10., 11., 12., 11.]), 1.0)
        self.assertEqual(out[-1, 0], -11.0)
        self.assertEqual(out[-1, 1], -1.0)


@unittest.skipIf(OLD is None, '旧树不在场 —— 对拍跳过')
class TestBuildHistoryMatches(unittest.TestCase):
    """`class renko` 的整段构建（砖价 / 方向 / 时间轴对齐）逐值相同。"""

    def test_bit_identical(self):
        rng = np.random.RandomState(1)
        hlc = np.c_[100 + np.cumsum(rng.randn(400)) * 0.8,
                    100 + np.cumsum(rng.randn(400)) * 0.8,
                    100 + np.cumsum(rng.randn(400)) * 0.8]
        for brick in (0.5, 1.25, 3.0):
            with self.subTest(brick=brick):
                a, b = OLD.renko(), NEW.renko()
                for obj in (a, b):
                    obj.set_brick_size(auto=False, brick_size=brick)
                    obj.build_history(hlc=hlc)
                np.testing.assert_array_equal(np.asarray(a.get_renko_prices()),
                                              np.asarray(b.get_renko_prices()))
                np.testing.assert_array_equal(np.asarray(a.get_renko_directions()),
                                              np.asarray(b.get_renko_directions()))
                # ⚠️ **从第 1 行比起** —— 第 0 行两边都是 `np.empty` 的**未初始化内存**，
                # 见下面 `test_source_aligned_row0_is_never_written`。
                np.testing.assert_array_equal(a.get_source_aligned()[1:],
                                              b.get_source_aligned()[1:])

    def test_source_aligned_row0_is_never_written(self):
        """⚠️ **旧树既有缺陷**：`source_aligned = np.empty(...)` 起手，而写它的循环
        从 `idx = 1` 开始 ⇒ **第 0 行没有任何人写**，它是未初始化内存。

        ⚠️ **它通常"看起来没事"**：`np.empty` 拿到的多是刚向 OS 申请的新页，
        而那些页是**零页** ⇒ 第 0 行常常正好是 `[0, 0, 0]`。
        这不是契约，是**未定义行为** —— 分配路径一变（例如大数组走了不同的
        分配器，或者内存被复用）它就会露出真实垃圾：本次对拍里真的出现过
        `9.2e-312`，于是**两条本该完全相同的实现**在这一行上不一致。

        本用例钉的是**可观测且稳定**的那部分：第 0 行**不是**按与第 1 行相同的
        规则算出来的（否则它会等于一个真实的砖上下界）。至于它具体是什么值，
        **不可断言** —— 断"必为 0"会变成一条随机红的用例。

        消费方（另案的 `calc_renko_atr_vX` 走 `source_aligned`）据此知道：
        它写出的 `RENKO_TREND_S_LB/UB` **第 0 行作废**。
        这也是上面 `test_bit_identical` 只比 `[1:]` 的原因。
        """
        hlc = np.c_[np.linspace(10, 12, 60), np.linspace(9, 11, 60),
                    np.linspace(9.5, 11.5, 60)]
        a, b = NEW.renko(), NEW.renko()
        for obj in (a, b):
            obj.set_brick_size(auto=False, brick_size=1.0)
            obj.build_history(hlc=hlc)
        sa = a.get_source_aligned()
        self.assertEqual(sa.shape, (60, 3))
        # 第 1 行起是**真值**：不是零，方向列是合法取值
        self.assertFalse(np.array_equal(sa[1], np.zeros(3)))
        self.assertIn(sa[1, 2], (-1.0, 0.0, 1.0))
        # 第 0 行**没被写过** ⇒ 它不等于第 1 行（若它是算出来的，就会是个真砖界）
        self.assertFalse(np.array_equal(sa[0], sa[1]))
        # 第 1 行起两次构建逐值一致
        np.testing.assert_array_equal(sa[1:], b.get_source_aligned()[1:])

    def test_evaluate_matches(self):
        """`evaluate` 的返回字典逐键相同 —— 注意键名 `'sign_changes:'` **带冒号**。"""
        rng = np.random.RandomState(2)
        px = 50 + np.cumsum(rng.randn(300)) * 0.5
        a, b = OLD.renko(), NEW.renko()
        for obj in (a, b):
            obj.set_brick_size(auto=False, brick_size=1.0)
            obj.build_history(prices=px)
        self.assertEqual(set(a.evaluate()), set(b.evaluate()))
        self.assertIn('sign_changes:', a.evaluate())      # 旧树真值，别"修"键名
        for k, v in a.evaluate().items():
            self.assertEqual(v, b.evaluate()[k], '键 {} 不一致'.format(k))


@unittest.skipIf(OLD is None, '旧树不在场 —— 对拍跳过')
class TestTrendCrossMatchesEndToEnd(unittest.TestCase):
    """**端到端**：整条 `renko_trend_cross_func` 的全部列逐值相同。

    这条同时覆盖 `talib.ATR`（定砖高）、`scipy.optimize.fminbound`
    （搜最优砖高）、numba 核、以及四个 `*_TIMING_LAG` 的合成。
    """

    @classmethod
    def setUpClass(cls):
        cls.df = _ohlc(420)

    def _both(self, df):
        return OLD.renko_trend_cross_func(df.copy()), NEW.renko_trend_cross_func(df.copy())

    def test_all_columns_bit_identical(self):
        a, b = self._both(self.df)
        self.assertEqual(list(a.columns), list(b.columns))
        self.assertGreater(len(a.columns), 10, '列数不对 —— 夹具没覆盖到')
        for col in a.columns:
            with self.subTest(col=col):
                # float16 是**无损放大**，不是容差（见模块 docstring）
                np.testing.assert_array_equal(
                    a[col].astype(np.float64).values,
                    b[col].astype(np.float64).values)

    def test_output_is_not_all_empty(self):
        """⚠️ 防"两条实现都静默早退"导致的**假通过** —— 结果必须真有内容。"""
        _, b = self._both(self.df)
        for col in (NEW.FLD.RENKO_TREND_S, NEW.FLD.RENKO_TREND_L,
                    NEW.FLD.RENKO_TREND, NEW.FLD.RENKO_BOOST_S_TIMING_LAG):
            with self.subTest(col=col):
                self.assertGreater(int((b[col].astype(np.float64) != 0).sum()), 0,
                                   '{} 全零 —— 很可能两边都没真算'.format(col))

    def test_s_family_is_float16(self):
        """S 族末尾被 cast 成 float16 —— 对拍要知情，否则容差判断会假。"""
        _, b = self._both(self.df)
        for col in (NEW.FLD.RENKO_TREND_S, NEW.FLD.RENKO_TREND_S_LB,
                    NEW.FLD.RENKO_TREND_S_UB):
            with self.subTest(col=col):
                self.assertEqual(b[col].dtype, np.float16)


@unittest.skipIf(OLD is None, '旧树不在场 —— 对拍跳过')
class TestShortInputContract(unittest.TestCase):
    """`len(data) < 30` 的早退**形状**承重 —— 调用方正是靠这些列名判断要不要重算。"""

    def test_returns_the_same_empty_column_set(self):
        short = _ohlc(20)
        a = OLD.renko_trend_cross_func(short.copy())
        b = NEW.renko_trend_cross_func(short.copy())
        self.assertEqual(list(a.columns), list(b.columns))
        self.assertEqual(len(b), 20)
        for col in b.columns:
            with self.subTest(col=col):
                self.assertTrue(b[col].isna().all(), '{} 应为空列'.format(col))

    def test_short_input_with_indices_concatenates(self):
        short = _ohlc(20)
        idx = pd.DataFrame({'dummy': range(20)}, index=short.index)
        b = NEW.renko_trend_cross_func(short, idx)
        self.assertIn('dummy', b.columns)


class TestNoQuantaxisLeaked(unittest.TestCase):
    """搬运删掉了 QUANTAXIS 透传（`QA_util_timestamp_to_str`）—— 本模块不得依赖它。

    ⚠️ **用 AST 判定，不 grep 源码**：本模块 docstring 里**有意**写了
    「删除了 ``import QUANTAXIS`` / ``QA_util_timestamp_to_str``」这类说明，
    按字符串搜会把**解释本身**当成违规（第一版就是这么写的，自伤）。
    AST 只看**代码**，注释与 docstring 一概不参与。
    """

    @staticmethod
    def _tree():
        import ast
        with open(NEW.__file__, encoding='utf-8') as fh:
            return ast.parse(fh.read())

    def test_no_quantaxis_import(self):
        import ast
        for node in ast.walk(self._tree()):
            if isinstance(node, ast.Import):
                for a in node.names:
                    self.assertFalse(a.name.startswith('QUANTAXIS'), a.name)
            elif isinstance(node, ast.ImportFrom):
                self.assertFalse((node.module or '').startswith('QUANTAXIS'),
                                 node.module)

    def test_no_quantaxis_symbol_in_code(self):
        import ast
        names = {n.id for n in ast.walk(self._tree()) if isinstance(n, ast.Name)}
        attrs = {n.attr for n in ast.walk(self._tree()) if isinstance(n, ast.Attribute)}
        for sym in ('QA_util_timestamp_to_str', 'QA', 'QA_DataStruct_Stock_day'):
            self.assertNotIn(sym, names | attrs, '代码里还引用了 QUANTAXIS 的 {}'.format(sym))

    def test_no_print_call_in_code(self):
        """旧树活链有 6 处 `print`；用户 2026-10-10「不需要打印了」—— 全部已删。"""
        import ast
        calls = [n for n in ast.walk(self._tree())
                 if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                 and n.func.id == 'print']
        self.assertEqual(calls, [], '还有 {} 处 print 调用'.format(len(calls)))


if __name__ == '__main__':
    unittest.main(verbosity=2)
