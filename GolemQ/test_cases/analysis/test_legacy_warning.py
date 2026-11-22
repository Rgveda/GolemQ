#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""旧纯 Python 实现的**不可过滤**提醒（2026-11-22）。

背景：`regtree_jit.py` / `ChipDistribution_jit.py` 里还留着逐行照抄老树的旧实现
（`predict_regression_tree_branch`、`_calcu_cost`、`_calcu_winner1/2`…），
它们比 jit 快路径慢**两个数量级**（实测建树 0.81s vs 0.007s）。用户 2026-11-22：
「直接删也不太好，加不能清除的强制警告，以防误用」。

为什么不用 `warnings.warn`：它会被调用方 / 测试 / `-W ignore` 用
`simplefilter('ignore')` **静默掉**，提醒就白加了。这里直接写 `sys.stderr`。

三条契约：

1. 真的写到 stderr（可被捕获到）；
2. **每个名字每进程只报一次**（不刷屏）；
3. `warnings.simplefilter('ignore')` **压不住**它。
"""

import io
import os
import sys
import unittest
import warnings
from contextlib import redirect_stderr

try:
    import GolemQ  # noqa: F401
except ImportError:
    sys.path.insert(0, os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from GolemQ.core.presentation import _LEGACY_WARNED, legacy_python_warning


class TestLegacyWarning(unittest.TestCase):

    def setUp(self):
        _LEGACY_WARNED.clear()

    def tearDown(self):
        _LEGACY_WARNED.clear()

    def test_writes_to_stderr(self):
        buf = io.StringIO()
        with redirect_stderr(buf):
            legacy_python_warning('demo.fn', 'demo.fn_jit')
        out = buf.getvalue()
        self.assertIn('[GolemQ][legacy]', out)
        self.assertIn('demo.fn', out)
        self.assertIn('demo.fn_jit', out, '应该提示改用哪个快路径')

    def test_only_once_per_process(self):
        buf = io.StringIO()
        with redirect_stderr(buf):
            for _ in range(5):
                legacy_python_warning('demo.fn')
        self.assertEqual(buf.getvalue().count('[GolemQ][legacy]'), 1,
                         '同一名字只该报一次，否则回填时刷屏')

    def test_not_suppressed_by_warnings_filter(self):
        buf = io.StringIO()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            with redirect_stderr(buf):
                legacy_python_warning('demo.suppressed_test')
        self.assertIn('[GolemQ][legacy]', buf.getvalue(),
                      'warnings.simplefilter("ignore") 不能压掉这条提醒')


class TestLegacyFunctionsWarn(unittest.TestCase):
    """真实旧实现确实挂了提醒（而不是只有 helper 能用）。"""

    def setUp(self):
        _LEGACY_WARNED.clear()

    def tearDown(self):
        _LEGACY_WARNED.clear()

    def test_regtree_python_builder_warns(self):
        import numpy as np
        from GolemQ.analysis import regtree_jit as RT
        rs = np.random.RandomState(0)
        price = np.c_[np.arange(80, dtype=float),
                      10 + np.cumsum(rs.normal(0, 0.2, 80))]
        buf = io.StringIO()
        with redirect_stderr(buf):
            RT.predict_regression_tree_branch(np.asmatrix(price), 25, 10)
        self.assertIn('predict_regression_tree_branch', buf.getvalue())

    def test_chip_dict_path_warns(self):
        import numpy as np
        import pandas as pd
        from GolemQ.analysis.ChipDistribution_jit import ChipDistribution
        rs = np.random.RandomState(1)
        n = 60
        low = rs.uniform(8, 12, n)
        df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=n),
            'open': rs.uniform(10, 20, n), 'high': low + rs.uniform(1, 6, n),
            'low': low, 'close': rs.uniform(10, 20, n),
            'volume': rs.uniform(1e5, 5e5, n),
            'TurnoverRate': rs.uniform(0.01, 0.05, n)})
        df['avg'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
        c = ChipDistribution()
        c.data = df
        c.calcuChip(flag=1, AC=1)
        buf = io.StringIO()
        with redirect_stderr(buf):
            c.cost(5)                      # 数组快路径：不该报警
        self.assertNotIn('[GolemQ][legacy]', buf.getvalue(),
                         '生产路径（数组直算）不该触发旧实现提醒')


if __name__ == '__main__':
    unittest.main()
