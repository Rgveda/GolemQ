#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""`analysis/regtree_jit.py` —— jit 版与 `regtree.py` 的**对拍**。

为什么这份用例是这次搬运的**主证据**
====================================
1. **jit 版「跑起来了」不等于「算对了」**。本模块开发中真踩到过两次：
   * 切分循环照抄旧版的 `range(n-1)`，而这里 y 已被拆出去 ⇒ `range(0)` ⇒
     **一次都不循环、永远返回"不切分"**，且**不报错**；
   * 叶子值返回了标量 `0.0`，而旧版返回的是 **OLS 系数矩阵**
     （`fit_lineareg_slope_of_the_model_leaf`）⇒ 整棵树的叶子全错。
   两次都只有对拍看得出来。
2. **加速数字要留档**：用户口径是「**拿不出数字就不算完成**」。

实测（2026-10-10，350 根 / 16 个特征列 / rate=25 / dur=10）
==========================================================
=============  ==================  ====================  ==========
层               python              jit                   结论
=============  ==================  ====================  ==========
切分搜索         —                   —                     6/6 **逐值一致**
建树（裸）       332–375 ms          3.5 ms                **≈95×**
端到端          **528.9 ms**        **25.1 ms**           **21.1×**
=============  ==================  ====================  ==========

端到端各列的对拍（`bar_limit=350`）：

* `REGTREE_TREND` / `REGTREE_TIMING_LAG` —— **逐值相同**（离散信号）
* `REGTREE_PRICE` 最大相对差 **5.4e-13**；`REGTREE_SLOPE` **4.4e-10**
  （闭式解 vs `xTx.I` 的浮点尾数差，见 `regtree_jit` 模块 docstring 的三条偏离）

⚠️ 端到端只有 21× 而内核 95× —— 因为**编排层成了瓶颈**（特征写入、
`fit_regtree_trend`、pandas 切片）。再往下压要先动那一段。

numba 缺失时整份**跳过**（新树只在本模块用它，且不在 `MIN_PACKAGES`）。
"""

import os
import sys

try:
    import GolemQ  # noqa: F401
except ImportError:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

import time
import unittest

import numpy as np
import pandas as pd

from GolemQ.analysis import regtree_jit as R
from GolemQ.analysis import regtree_jit as J
from GolemQ.core.constants import FIELD as FLD

_HAS_NUMBA = J.available()


def _make(n=350, seed=0):
    """合成一份满足 `calc_regtree_fractal_*` 全部输入列要求的数据。

    ⚠️ 那 16 个输入列是**扫出来的**（`features[FLD.X]` 的读取点），少一个就 KeyError。
    """
    rng = np.random.RandomState(seed)
    idx = pd.MultiIndex.from_arrays(
        [pd.date_range('2024-01-01', periods=n, freq='D'), ['600519'] * n],
        names=['date', 'code'])
    close = 100 + np.cumsum(rng.randn(n))
    s = pd.Series(close, index=idx)
    data = pd.DataFrame({'open': close, 'high': close * 1.01, 'low': close * 0.99,
                         'close': close, 'volume': rng.randint(1e5, 1e6, n)}, index=idx)
    f = pd.DataFrame(index=idx)
    for k in ('HMA5', 'HMA10', 'MA30', 'MA90', 'MAPOWER30', 'MAPOWER120',
              'HMAPOWER120', 'POLYNOMIAL9', 'MACD_DELTA', 'ATR_LB', 'COMBINE_DENSITY'):
        f[getattr(FLD, k)] = s.values + rng.randn(n) * 0.5
    f[FLD.MA90_SLOPE] = np.gradient(close)
    for k in ('DEA_ZERO_TIMING_LAG', 'HMAPOWER120_TIMING_LAG',
              'MAPOWER120_TIMING_LAG', 'MAPOWER30_TIMING_LAG'):
        f[getattr(FLD, k)] = rng.randint(-20, 20, n).astype(np.int32)
    return data, f


@unittest.skipUnless(_HAS_NUMBA, 'numba 不可用 —— jit 对拍跳过')
class TestSplitSearchMatches(unittest.TestCase):
    """**切分搜索必须逐值一致** —— 它是整棵树的骨架。"""

    def test_same_split_on_random_data(self):
        rng = np.random.RandomState(0)
        for trial in range(6):
            m = rng.randint(40, 140)
            price = np.c_[np.arange(m, dtype=float), 100 + np.cumsum(rng.randn(m))]
            a = R.choose_best_split_branch(np.asmatrix(price), 25, 10)
            b = J.choose_best_split_branch_jit(np.asmatrix(price), 25, 10)
            with self.subTest(trial=trial, m=m):
                self.assertEqual(a[0], b[0], '切分特征不一致')
                self.assertAlmostEqual(float(a[1]), float(b[1]), places=9,
                                       msg='切分值不一致')

    def test_no_split_returns_none_feature(self):
        """所有 y 相同 ⇒ 不切分（旧版走 `set(...) == 1` 那条早退）。"""
        price = np.c_[np.arange(30, dtype=float), np.full(30, 5.0)]
        self.assertIsNone(R.choose_best_split_branch(np.asmatrix(price), 25, 10)[0])
        self.assertIsNone(J.choose_best_split_branch_jit(np.asmatrix(price), 25, 10)[0])


# ⚠️ 2026-11-23：`TestEndToEndMatches` / `TestSpeedup` 已删除 —— 它们对拍的是
# `calc_regtree_fractal_func`（纯 Python 编排）vs `calc_regtree_fractal_jit`
# （同一个编排 + jit builder）。用户定：旧树先后改进过多个版本，
# **只保留最先进的 `calc_regtree_fractal_vXIs`**（float32 + `rTrePrcD` 比值 +
# 前置段用 close 填），另两个直接抹掉。
# jit ↔ 纯 Python 的等价性现在由 `test_regtree_builder.py` 钉住
# （`_build_tree` vs `predict_regression_tree_branch`，含 NaN 场景）。


class TestStatusWithoutNumba(unittest.TestCase):
    """`regtree_jit_status` / `available` 不管 numba 在不在都要能答。"""

    def test_status_shape(self):
        st = J.regtree_jit_status()
        self.assertIn('available', st)
        if st['available']:
            self.assertIn('numba', st)
            self.assertIn('disable_jit', st)
        else:
            self.assertIn('reason', st)


if __name__ == '__main__':
    unittest.main(verbosity=2)
