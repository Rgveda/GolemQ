# coding: utf-8
"""`analysis/cost5_uprising.py`（L2 化简版 `cost5UprBf`）的契约测试。"""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from GolemQ.analysis.cost5_uprising import calc_cost5_uprising_before


def _frame(n=8):
    x = np.arange(n)
    return pd.DataFrame({
        'RetCost5': 0.001 * x,
        'MACD': np.where(x % 4 < 2, 1.0, -1.0),
        'macdMaj': np.where(x % 6 < 3, -1.0, 1.0),
        'stage': np.where(x > n // 2, -1.0, 0.0),
    })


class TestCost5Uprising(unittest.TestCase):
    def test_doctest_case(self):
        f = pd.DataFrame({'RetCost5': [0.0, 0.01, 0.02, 0.02, 0.02],
                          'MACD': [1, 1, -1, -1, -1],
                          'macdMaj': [-1, -1, 1, 1, 1],
                          'stage': [0, 0, 0, -1, -1]})
        self.assertEqual(calc_cost5_uprising_before(f).tolist(),
                         [1, 2, 3, 0, 0])

    def test_missing_columns_is_zero(self):
        f = pd.DataFrame({'close': [1.0, 2.0]})
        out = calc_cost5_uprising_before(f)
        self.assertEqual(out.tolist(), [0, 0])

    def test_causal_prefix(self):
        """截断到 T 的值必须与全量在 T 处相同（无未来函数）。"""
        f = _frame(12)
        full = calc_cost5_uprising_before(f)
        for cut in (5, 8):
            part = calc_cost5_uprising_before(f.iloc[:cut]).iloc[-1]
            self.assertEqual(int(part), int(full.iloc[cut - 1]))


if __name__ == '__main__':
    unittest.main()
