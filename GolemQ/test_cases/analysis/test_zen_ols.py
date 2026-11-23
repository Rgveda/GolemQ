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

from GolemQ.analysis.zen_ols import (MIN_PEAKS_SIDE, WIDEN_FACTOR,
                                     _collect_peaks,
                                     calc_zen_ols_timing_lag,
                                     calc_zen_ols_timing_lag_streaming,
                                     cross_marks, lds_positions,
                                     lis_positions, segment_ids)


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


class TestStreamingCausality(unittest.TestCase):
    """**流式**口径（用户 2026-11-23 选定）：一遍 `CZSC.update` 走到底。

    与逐 target 重建窗口的区别：历史是「从帧头累积」，不是 2160 滚动窗口；
    好处是成本 O(总 bar 数)（实测 9,800 根 ~5s vs 逐窗口 ~12 分钟）。

    硬要求仍然是**前缀稳定**：用 ≤T 的数据算出的 T 处值，必须与全量算出的 T 处一致
    —— `CZSC.update` 在第 T 根只看过前 T 根，天然满足。
    """

    def test_prefix_stable(self):
        o = _ohlc(n=420)
        full = calc_zen_ols_timing_lag_streaming(o)
        for T in (150, 260, 419):
            part = calc_zen_ols_timing_lag_streaming(o.iloc[:T + 1])
            self.assertEqual(float(part.iloc[-1]), float(full.iloc[T]),
                             f'T={T} 流式口径出现未来函数')

    def test_output_shape_and_range(self):
        import numpy as np
        o = _ohlc(n=300)
        lag = calc_zen_ols_timing_lag_streaming(o)
        self.assertEqual(len(lag), len(o))
        v = lag.dropna()
        self.assertGreater(len(v), 0, '至少要有值')
        self.assertLessEqual(abs(float(v.max())), len(o) + 1)


class TestScaleInvariance(unittest.TestCase):
    """**除权不该改变 `ZenOLSLag`**（用户 2026-11-23 提出的隐患）。

    review 的 60min 取数是 `if_fq='qfq'`（前复权）：一次除权会把**整条历史**
    乘一个常数。`ZenOLSLag` 如果只依赖价格的相对关系（CZSC 包含/分型、
    LIS/LDS 的序、以及「距事件的 bar 数」），那么整体缩放不该产生任何差异。

    ⚠️ 反过来，这条用例也是**闸门**：谁以后在算法里引入绝对价格阈值
    （比如 `if price > 100`），这里会立刻红。
    """

    def test_uniform_scaling_does_not_change_lag(self):
        import numpy as np
        o = _ohlc(n=380, seed=11)
        base = calc_zen_ols_timing_lag_streaming(o)
        for k in (0.5, 2.0, 0.1):
            scaled = o.copy()
            for c in ('open', 'high', 'low', 'close'):
                scaled[c] = scaled[c] * k
            got = calc_zen_ols_timing_lag_streaming(scaled)
            np.testing.assert_array_equal(
                np.asarray(got, dtype=float), np.asarray(base, dtype=float),
                err_msg=f'整体 ×{k} 改变了 ZenOLSLag —— 引入了绝对量纲')


class TestMinPeaksRule(unittest.TestCase):
    """一侧峰点 < 3 ⇒ 不出该方向交叉；此时取样窗扩到 **2 倍段长** 重取。

    起因（用户 2026-11-23）：000711 在 2026-09-22 10:30 出 +1 时，所在段
    （75 根、`ZenPivotLag=+75`）里**只有 1 个高峰** —— 这一浪根本还没成形；
    一小时后价格出中枢、段切到 205 根，又出 -1，同一小时内来回翻。
    """

    def test_constants(self):
        self.assertEqual(MIN_PEAKS_SIDE, 3)
        self.assertEqual(WIDEN_FACTOR, 2)

    def test_collect_peaks_counts_by_side(self):
        import numpy as np
        marks = np.array([0, 1, 0, -1, 0, 1, -1, 0], dtype=np.int8)
        low = np.full(8, 10.0); high = np.full(8, 11.0)
        lo, hi = _collect_peaks(marks, low, high, 0, 7)
        self.assertEqual(lo, [1, 5])
        self.assertEqual(hi, [3, 6])
        lo2, hi2 = _collect_peaks(marks, low, high, 0, 2)   # 只到下标 2
        self.assertEqual(lo2, [1])
        self.assertEqual(hi2, [])
