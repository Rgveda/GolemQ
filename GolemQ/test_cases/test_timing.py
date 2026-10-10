#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""`analysis/timing.py` —— 从旧树搬回的时序累积器与金叉/死叉间隔。

为什么这份用例值得存在
======================
1. **它是「stub 复活」的产物**。`Timeline_duration` 曾以 stub 形式待在
   `analysis/timeseries.py`，因「唯一调用方在 `services/persistence/`」被当死代码删掉；
   2026-10-10 筹码分布模块要它，于是**带着真实现**从旧树搬回来（见 `GLOSSARY.md`
   的「stub vs dummy」一节）。复活的东西必须有测试 —— 否则下一次还会被当死代码删。
2. **口径反直觉的地方要钉住**：`Timeline_Integral` 与 `Timeline_duration` 的**清零条件
   相反**（一个是 `Tm==0` 清、一个是 `Tm==1` 清）；`calc_feature_event_timing_lag`
   的**第一个元素是 -1**（因为 `shift(1)` 给 NaN，比较为 False）。
   这几处**错一个符号不会报错**，只会让下游的 lag 特征整体反号。

对拍用例（`TestMatchesOldTree`）在老树不在场时**跳过**。
"""

import os
import re
import sys

try:
    import GolemQ  # noqa: F401
except ImportError:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import unittest

import numpy as np
import pandas as pd

from GolemQ.analysis.timing import (
    Timeline_Integral, Timeline_duration, calc_energy, calc_energy_f4,
    calc_energy_f8, calc_event_timing_lag, calc_feature_event_timing_lag,
)


class TestAccumulators(unittest.TestCase):
    """两个累积器的**清零条件相反** —— 这是最容易抄错的一处。"""

    def test_integral_clears_on_zero(self):
        """`Timeline_Integral`：`Tm[i]==0` 清零（死叉 1→0）。"""
        self.assertEqual(
            Timeline_Integral(np.array([1, 1, 0, 1, 1], dtype=np.int32)).tolist(),
            [1, 2, 0, 1, 2])

    def test_duration_clears_on_one(self):
        """`Timeline_duration`：`Tm[i]==1` 清零（金叉 0→1）。"""
        self.assertEqual(
            Timeline_duration(np.array([0, 0, 1, 0, 0], dtype=np.int32)).tolist(),
            [1, 2, 0, 1, 2])

    def test_first_element_uses_numpy_negative_index(self):
        """i=0 时 `T[-1]` 取到数组**最后一个**元素（新数组全 0）—— 依赖 numpy 负索引。

        这一条是**刻意的**：改成 `T[max(i-1,0)]` 会改口径，故钉住现状。
        `Timeline_duration` 的 i=0 分支是 `T[-1]+1`（与 Tm[0] 无关）⇒ 恒为 1。
        """
        self.assertEqual(Timeline_duration(np.array([7], dtype=np.int32)).tolist(), [1])

    def test_integral_only_meaningful_for_binary_input(self):
        """⚠️ `Timeline_Integral` **只对 0/1 输入有意义** —— 它是给二值信号的。

        - 输入 1 ⇒ `1*(0+1)` = **1**（正常）
        - 输入 3 ⇒ `3*(0+3)` = **9**（公式产物，不是笔误）

        旧树同公式、同行为（对拍用的是 0/1 的 `jx`，覆盖不到这一支）。
        **别"顺手修"成 `Tm[i] + T[i-1]`** —— 那会改口径；调用方传的就该是 0/1。
        """
        self.assertEqual(Timeline_Integral(np.array([1], dtype=np.int32)).tolist(), [1])
        self.assertEqual(Timeline_Integral(np.array([3], dtype=np.int32)).tolist(), [9])


class TestEventTimingLag(unittest.TestCase):
    def test_sign_convention(self):
        """金叉取正、死叉取负；`<=0` 一律走负分支。"""
        self.assertEqual(calc_event_timing_lag(np.array([1, 1, -1, -1, 1])).tolist(),
                         [1, 2, -1, -2, 1])

    def test_feature_lag_first_element_is_minus_one(self):
        """⚠️ **第一个元素是 -1，不是 +1** —— `shift(1)` 给 NaN ⇒ 两个比较都是 False
        ⇒ 金叉与死叉累积都等于 1 ⇒ `1 < 1` 为假 ⇒ 走 -1 分支。

        （这条我第一版 docstring 猜错成 +1，跑出来才发现。）
        """
        df = pd.DataFrame({'x': [1.0, 2.0, 1.5, 0.5, 1.0]})
        self.assertEqual(calc_feature_event_timing_lag(df, 'x').tolist(),
                         [-1, 1, -1, -2, 1])

    def test_returns_int32(self):
        df = pd.DataFrame({'x': np.arange(20.0) % 3})
        self.assertEqual(calc_feature_event_timing_lag(df, 'x').dtype, np.int32)


class TestEnergy(unittest.TestCase):
    def test_same_sign_accumulates_and_flips_restart(self):
        """同号累加、异号重启。"""
        self.assertEqual(calc_energy_f8(np.array([1.0, 1.0, -1.0, 1.0])).tolist(),
                         [1.0, 2.0, -1.0, 1.0])

    def test_dtype_dispatch(self):
        """float64 → f8 内核；float32 → f4 内核（返回值 dtype 随之不同）。"""
        self.assertEqual(calc_energy(np.array([1.0, 1.0])).dtype, np.float64)
        self.assertEqual(calc_energy(np.array([1.0, 1.0], dtype=np.float32)).dtype,
                         np.float32)

    def test_accepts_series_and_ndarray(self):
        self.assertEqual(calc_energy(pd.Series([1.0, 1.0, -1.0])).tolist(),
                         calc_energy(np.array([1.0, 1.0, -1.0])).tolist())

    def test_other_dtype_falls_back_to_float32(self):
        """int 输入走 `astype(float32)` 那条兜底路（不是 f8）。"""
        self.assertEqual(calc_energy(np.array([1, 1, -1])).dtype, np.float32)

    def test_all_zeros_is_all_zeros(self):
        self.assertEqual(calc_energy_f8(np.zeros(5)).tolist(), [0.0] * 5)

    def test_length_one_and_empty(self):
        self.assertEqual(calc_energy_f8(np.array([2.0])).tolist(), [2.0])
        self.assertEqual(calc_energy_f8(np.array([], dtype=float)).tolist(), [])


class TestResampleIsImportable(unittest.TestCase):
    """对齐函数要能用（细节由筹码分布那条链覆盖；这里只钉「在且可调」）。"""

    def test_importable(self):
        from GolemQ.analysis.timing import resample_multi_frequency_indices_func
        self.assertTrue(callable(resample_multi_frequency_indices_func))


_OLD = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), 'GolemQ_old')


@unittest.skipUnless(os.path.isdir(_OLD), '老树 GolemQ_old/ 不在场 —— 对拍用例跳过')
class TestMatchesOldTree(unittest.TestCase):
    """**对拍**：同一批随机输入，与旧树的同名实现**逐值**比。

    老树不入库（`README.md` 有记），所以本用例会随老树一起消失 —— 但那正是它的用处：
    **搬完当时**证明过「没改口径」，记录在 commit 里；老树还在时就还能量。

    首次实测（2026-10-10）：6 个函数 × 500 个随机样本，**全部逐值一致**。
    """

    @classmethod
    def setUpClass(cls):
        def grab(name, text):
            # 抽到**下一个 def 或装饰器**为止（否则会把下一个函数的 @nb.jit 吞进来）
            m = re.search(r'^def ' + name + r'\(.*?(?=^def |^@)', text, re.M | re.S)
            if not m:
                raise unittest.SkipTest('老树里找不到 {}'.format(name))
            return m.group(0)

        ts = open(os.path.join(_OLD, 'analysis', 'timeseries.py'), encoding='utf-8').read()
        fb = open(os.path.join(_OLD, 'features', 'base.py'), encoding='utf-8').read()
        cls.ns = {'np': np, 'pd': pd}
        for n in ('Timeline_Integral', 'Timeline_duration', 'calc_event_timing_lag',
                  'calc_energy_f8', 'calc_energy_f4', 'calc_energy'):
            exec(grab(n, ts), cls.ns)               # noqa: S102
        exec(grab('calc_feature_event_timing_lag', fb), cls.ns)   # noqa: S102

    def test_all_six_match(self):
        rng = np.random.RandomState(0)
        sig = rng.randn(500)
        jx = np.where(np.diff(np.r_[0, sig]) > 0, 1, 0).astype(np.int32)
        df = pd.DataFrame({'x': sig})
        pairs = [
            ('Timeline_Integral', Timeline_Integral(jx)),
            ('Timeline_duration', Timeline_duration(jx)),
            ('calc_event_timing_lag', calc_event_timing_lag(sig)),
            ('calc_energy_f8', calc_energy_f8(sig)),
            ('calc_energy', calc_energy(sig)),
            ('calc_feature_event_timing_lag', calc_feature_event_timing_lag(df, 'x')),
        ]
        for name, new in pairs:
            with self.subTest(fn=name):
                old = np.asarray(self.ns[name](df, 'x') if name == 'calc_feature_event_timing_lag'
                                 else (self.ns[name](jx) if name in
                                       ('Timeline_Integral', 'Timeline_duration')
                                       else self.ns[name](sig)))
                np.testing.assert_array_equal(np.asarray(new), old)


if __name__ == '__main__':
    unittest.main(verbosity=2)
