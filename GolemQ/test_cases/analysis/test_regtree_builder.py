#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RegTree 建树：**jit 版必须与纯 Python 版逐值相同**（2026-11-22 优化）。

背景：`regtree_jit.py` 里一直有 jit 建树内核（`choose_best_split` / `_ols_err`），
但 review 的日族走的是 `calc_regtree_fractal_vXIs` → `np.matrix` 的**纯 Python**
建树器 —— 实测 **0.81s/次**（450 根窗口），而 jit 版稳态 **0.007s/次**。
日族每天建一棵树 ⇒ 全量回填 1,700 天白花约 23 分钟/码。

现在 `_build_tree()` 优先 jit、异常回落纯 Python。这里钉住两条：

1. 两条路径建出的**树完全一致**（切分特征/切分点/叶子值逐层）；
2. 端到端 `calc_regtree_fractal_vXIs` 产出的 5 列**逐值相同**。

⚠️ 合成数据，不碰 DB。首次调用含 numba 编译（~1.5s），属环境噪音，不断言耗时。
"""

import os
import sys
import unittest

import numpy as np
import pandas as pd

try:
    import GolemQ  # noqa: F401
except ImportError:
    sys.path.insert(0, os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from GolemQ.analysis import regtree_jit as RT


def _price_frame(n=160, seed=3):
    rs = np.random.RandomState(seed)
    close = 10 + np.cumsum(rs.normal(0, 0.15, n))
    idx = pd.MultiIndex.from_arrays(
        [pd.bdate_range('2024-01-01', periods=n), ['600519'] * n],
        names=['ts', 'code'])
    return pd.DataFrame({'open': close, 'high': close + 0.3, 'low': close - 0.3,
                         'close': close, 'volume': 1e5}, index=idx)


def _same_tree(a, b) -> bool:
    """递归比较两棵树（dict 分裂节点 / ndarray 叶子）。"""
    if isinstance(a, dict) != isinstance(b, dict):
        return False
    if isinstance(a, dict):
        if int(a['spInd']) != int(b['spInd']):
            return False
        if abs(float(a['spVal']) - float(b['spVal'])) > 1e-12:
            return False
        return (_same_tree(a['left'], b['left'])
                and _same_tree(a['right'], b['right']))
    if a is None or b is None:
        return a is b or (np.size(a) == 0 and np.size(b) == 0)
    x = np.ravel(np.asarray(a, dtype=float))
    y = np.ravel(np.asarray(b, dtype=float))
    if x.shape != y.shape:
        return False
    return np.allclose(x, y, rtol=0, atol=1e-12, equal_nan=True)


class TestJitBuilderMatchesPython(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if not RT.available():
            raise unittest.SkipTest('numba 不可用')

    def test_tree_identical(self):
        rs = np.random.RandomState(11)
        n = 140
        price = np.c_[np.arange(n, dtype=float),
                      10 + np.cumsum(rs.normal(0, 0.2, n))]
        old = RT.predict_regression_tree_branch(np.asmatrix(price), 25, 10)
        new = RT.predict_regression_tree_branch_jit(
            np.ascontiguousarray(price, dtype=np.float64), 25, 10)
        self.assertTrue(_same_tree(old, new), 'jit 建树与纯 Python 建树不一致')

    def test_build_tree_helper_prefers_jit(self):
        rs = np.random.RandomState(5)
        n = 120
        price = np.c_[np.arange(n, dtype=float),
                      10 + np.cumsum(rs.normal(0, 0.2, n))]
        a = RT._build_tree(price, 25, 10)
        b = RT.predict_regression_tree_branch(np.asmatrix(price), 25, 10)
        self.assertTrue(_same_tree(a, b))

    def test_end_to_end_columns_identical(self):
        df = _price_frame()
        feats = pd.DataFrame(index=df.index)
        feats['hma10'] = df['close'].rolling(10, min_periods=1).mean()
        new = RT.calc_regtree_fractal_vXIs(
            df, features=feats.copy(), bar_limit=120, rate=25, dur=10)

        # 强制回落纯 Python 路径再算一遍
        real = RT._build_tree
        # ⚠️ 签名要跟 `_build_tree(price, rate, dur, code='')` 对齐
        RT._build_tree = lambda p, r, d, code='': (
            RT.predict_regression_tree_branch(np.asmatrix(p), r, d,
                                              level=0, code=code))
        try:
            old = RT.calc_regtree_fractal_vXIs(
                df, features=feats.copy(), bar_limit=120, rate=25, dur=10)
        finally:
            RT._build_tree = real

        cols = ['regtreePrice', 'regtreeTrd', 'regtreeLag', 'regtreeSlp',
                'rTrePrcD']
        for c in cols:
            x = pd.to_numeric(old[c], errors='coerce').to_numpy(float)
            y = pd.to_numeric(new[c], errors='coerce').to_numpy(float)
            np.testing.assert_allclose(
                y, x, rtol=0, atol=1e-9, equal_nan=True,
                err_msg=f'{c} 在 jit / 纯 Python 两条路径上不一致')


if __name__ == '__main__':
    unittest.main()


class TestNanDoesNotFallBack(unittest.TestCase):
    """**HMA10 带 NaN 时也不能回落纯 Python**（2026-11-23）。

    实测（用户 2026-11-22 的 review）：603983 / 605198 / 605287 / 605289 /
    605296 等码在 `calc_regtree_fractal_vXIs` 上抛
    `LinAlgError: Array must not contain infs or NaNs`（numba 的 OLS 不吃 NaN），
    于是**每码都回落纯 Python 建树**（慢 100 倍），日志刷一片 `[legacy]`。

    修法：三个入口（`_func` / `_vXI` / `_vXIs`）与 `_build_tree` 都按行剔除
    NaN/inf。这里钉住：**带 NaN 的输入不触发回落，且仍产出 5 列**。
    """

    @classmethod
    def setUpClass(cls):
        if not RT.available():
            raise unittest.SkipTest('numba 不可用')

    def test_nan_hma10_does_not_trigger_legacy_fallback(self):
        import io
        from contextlib import redirect_stderr
        df = _price_frame(n=200)
        feats = pd.DataFrame(index=df.index)
        hma = df['close'].rolling(10, min_periods=1).mean()
        hma.iloc[-60:-50] = np.nan          # 模拟停牌/缺数造成的洞
        feats['hma10'] = hma

        buf = io.StringIO()
        with redirect_stderr(buf):
            out = RT.calc_regtree_fractal_vXIs(
                df, features=feats.copy(), bar_limit=120, rate=25, dur=10)

        self.assertNotIn('[legacy]', buf.getvalue(),
                         '带 NaN 的输入不该逼出旧 Python 实现')
        self.assertNotIn('[regtree]', buf.getvalue(),
                         '也不该报 jit 建树失败')
        self.assertGreater(
            int(pd.to_numeric(out['regtreePrice'], errors='coerce').notna().sum()),
            0, '剔除 NaN 行后仍应算出 regtreePrice')

    def test_build_tree_guard_filters_nan(self):
        rs = np.random.RandomState(3)
        price = np.c_[np.arange(80, dtype=float),
                      10 + np.cumsum(rs.normal(0, 0.2, 80))]
        price[40, 1] = np.nan               # 单点 NaN
        tree = RT._build_tree(price, 25, 10, code='TEST')
        self.assertIsNotNone(tree, '_build_tree 应自行过滤 NaN 而不是抛错')

