#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""`analysis/zen_ols.py` —— 重新实现的 `ZenOLSLag`（2026-11-22）。

老树那条链依赖一整套聚类/波浪切分（`zen_mapower30_predict` →
`calc_slice_with_wavelet` → `calc_slice_with_cluster_grouped`）。这里只保留
**决定信号的那一步**（段内 LIS/LDS 找趋势主干），分段换成已落库、天然因果的
60min `ZenPivotLag` 符号。

四条要钉住：

1. LIS/LDS 结果是**合法**的最长（递增/递减）子序列；
2. 分段按 `ZenPivotLag` 符号，**NaN ffill 不制造新段**；
3. `in_pivot_marks`：True ⇒ 中枢内也标交叉，False ⇒ 只在中枢外；
4. **因果性**：用 ≤T 的数据算出的 T 处 lag，必须与用全量数据算出的 T 处一致
   —— 老树对整条历史一次算完（段尾信息回流段首），我们刻意不照搬。

⚠️ 合成数据；`peak='scipy'` 避开 CZSC 依赖，纯函数级验证。
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

from GolemQ.analysis.zen_ols import (calc_zen_ols_timing_lag, cross_marks,
                                     lds_positions, lis_positions, segment_ids)


def _ohlc(n=400, seed=7):
    rs = np.random.RandomState(seed)
    close = 10 + np.cumsum(rs.normal(0, 0.2, n))
    high = close + rs.uniform(0.05, 0.3, n)
    low = close - rs.uniform(0.05, 0.3, n)
    idx = pd.MultiIndex.from_arrays(
        [pd.bdate_range('2024-01-01', periods=n), ['600519'] * n],
        names=['ts', 'code'])
    return pd.DataFrame({'open': close, 'high': high, 'low': low,
                         'close': close, 'volume': 1e5}, index=idx)


def _pivot_lag(n=400, block=60):
    """+N / -N 交替的 ZenPivotLag（正值=中枢内）。"""
    vals = []
    for i in range(n):
        k = (i // block) % 2
        vals.append(float(i % block + 1) * (1 if k == 0 else -1))
    return pd.Series(vals, index=_ohlc(n).index)


class TestLisLds(unittest.TestCase):

    def test_lis_is_increasing(self):
        x = [5.0, 1.0, 4.0, 2.0, 3.0]
        pos = lis_positions(x)
        vals = [x[i] for i in pos]
        self.assertEqual(vals, sorted(vals))
        self.assertEqual(len(vals), 3, '最长递增子序列长度应为 3（1,2,3 或 1,4…）')

    def test_lds_is_decreasing(self):
        x = [5.0, 1.0, 4.0, 2.0, 3.0]
        pos = lds_positions(x)
        vals = [x[i] for i in pos]
        self.assertEqual(vals, sorted(vals, reverse=True))

    def test_single_element(self):
        self.assertEqual(lis_positions([1.0]), [0])
        self.assertEqual(lds_positions([1.0]), [0])


class TestSegmentIds(unittest.TestCase):

    def test_sign_runs(self):
        s = pd.Series([5, 3, -2, -1, 4])
        self.assertEqual(segment_ids(s).tolist(), [0, 0, 1, 1, 2])

    def test_nan_ffill_does_not_split(self):
        s = pd.Series([2.0, np.nan, np.nan, -1.0, np.nan])
        self.assertEqual(segment_ids(s).tolist(), [0, 0, 0, 1, 1],
                         'NaN 应延续上一段，不制造新段')

    def test_leading_nan_is_unknown(self):
        """开头 NaN ⇒ 段 -1（那时还不知道符号；bfill 会吃未来值）。"""
        s = pd.Series([np.nan, 2.0, np.nan, -1.0])
        self.assertEqual(segment_ids(s).tolist(), [-1, 0, 0, 1])


class TestCrossMarks(unittest.TestCase):

    def test_in_pivot_marks_controls_coverage(self):
        o = _ohlc()
        pl = _pivot_lag()
        on = cross_marks(o, pivot_lag=pl, peak='scipy', causal=True,
                         in_pivot_marks=True)
        off = cross_marks(o, pivot_lag=pl, peak='scipy', causal=True,
                          in_pivot_marks=False)
        self.assertGreater(int((on != 0).sum()), int((off != 0).sum()),
                           '含中枢应比仅中枢外标得多')

    def test_marks_are_only_plus_or_minus_one(self):
        o = _ohlc()
        cm = cross_marks(o, pivot_lag=_pivot_lag(), peak='scipy')
        self.assertTrue(set(np.unique(cm)).issubset({-1, 0, 1}))


class TestCausality(unittest.TestCase):
    """**硬门槛**：截断到 T 与全量，在 T 处的 lag 必须相同。"""

    def test_lag_matches_truncated_run(self):
        o = _ohlc(n=400)
        pl = _pivot_lag(n=400)
        full = calc_zen_ols_timing_lag(o, pivot_lag=pl, peak='scipy',
                                       causal=True)
        for T in (120, 200, 260, 333, 399):
            cut = o.iloc[:T + 1]
            part = calc_zen_ols_timing_lag(cut, pivot_lag=pl.iloc[:T + 1],
                                           peak='scipy', causal=True)
            self.assertEqual(float(part.iloc[-1]), float(full.iloc[T]),
                             f'T={T} 处出现未来函数：截断 {part.iloc[-1]} vs 全量 {full.iloc[T]}')

    def test_non_causal_differs_somewhere(self):
        """事后口径**允许**与截断不同（它就是偷看未来的那版）。"""
        o = _ohlc(n=400)
        pl = _pivot_lag(n=400)
        full = calc_zen_ols_timing_lag(o, pivot_lag=pl, peak='scipy',
                                       causal=False)
        cut = o.iloc[:301]
        part = calc_zen_ols_timing_lag(cut, pivot_lag=pl.iloc[:301],
                                       peak='scipy', causal=False)
        # 不强制不同（可能恰好相同），但至少要能跑通且长度正确
        self.assertEqual(len(part), 301)
        self.assertEqual(len(full), 400)


if __name__ == '__main__':
    unittest.main()
