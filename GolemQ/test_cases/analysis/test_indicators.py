# coding:utf-8
"""`analysis/indicators.py`（`TA_HMA`）的结构性钉子。

⚠️ **这里没有"与老树对拍"** —— 老树那份 `TA_HMA` 来自 **QUANTAXIS 包**
（`QAIndicator/talib_numpy.py:178`），而 `test_no_quantaxis` 有一条
「**运行时不许加载 QUANTAXIS**」的断言；本文件一 import 它就会把那条例用跑红
（`test_renko` 当年就踩过这个：留下的 `QUANTAXIS*` 让 `test_no_quantaxis` 当场红）。

⇒ 真正的验证分两层：
1. **这里**：公式的可证性质（暖机长度、线性输入的精确性、与 `talib.WMA` 的同构）；
2. **`calc_polynomial9_vXIs` 的端到端对拍**（见 `test_poly9_chain.py`）——
   那里 `HMA10` 是被真正消费的，与老树逐值比才是硬证据。
"""
from __future__ import annotations

import os
import sys

try:
    import GolemQ  # noqa: F401
except ImportError:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

import unittest

import numpy as np

from GolemQ.analysis.indicators import available, TA_HMA

_HAS_TALIB = available()


@unittest.skipUnless(_HAS_TALIB, 'talib 不在场')
class TestHMA(unittest.TestCase):
    """`TA_HMA` —— 赫尔均线。"""

    def test_warmup_is_eleven_for_period_ten(self):
        """⚠️ **`period=10` ⇒ 前 11 个是 `NaN`**。

        这个数字**与 `period=10` 绑死**：老树靠它写 `np.r_[close[:11], HMA10[11:]]`
        去拼 `POLYNOMIAL9` 的输入（`features/base.py:2102`）。
        改 `period` 就得改那个拼接位置 —— 所以在这里钉住。
        """
        y = TA_HMA(np.arange(60, dtype=float), 10)
        self.assertTrue(np.isnan(y[:11]).all(), '前 11 个必须是暖机区')
        self.assertTrue(np.isfinite(y[11:]).all(), '第 11 个起必须全有效')

    def test_linear_series_is_reproduced(self):
        """线性输入 ⇒ 输出也是同斜率的线性（`2*WMA(n/2) − WMA(n)` 消掉了滞后）。"""
        y = TA_HMA(np.arange(80, dtype=float), 10)
        slope = np.diff(y[11:])
        self.assertTrue(np.allclose(slope, 1.0, atol=1e-9), slope[:5])

    def test_is_the_talib_composition(self):
        """就是那条公式本身（`WMA(2·WMA(n/2) − WMA(n), √n)`）—— 自证同构。"""
        import talib
        px = np.cumsum(np.random.default_rng(6).normal(0, 1, 120)) + 100.0
        expect = talib.WMA(2 * talib.WMA(px, 5) - talib.WMA(px, 10), 3)
        self.assertTrue(np.array_equal(TA_HMA(px, 10), expect, equal_nan=True))

    def test_length_always_matches_input(self):
        for n in (20, 60, 200):
            with self.subTest(n=n):
                self.assertEqual(len(TA_HMA(np.arange(n, dtype=float), 10)), n)

    def test_poly9_input_has_no_nan(self):
        """老树拼 `POLYNOMIAL9` 输入的那一招：**拼完不许留 NaN**。

        这就是 `[0:11]` / `[11:]` 那两个数字存在的理由 —— 拼错一位就会给
        `rolling_poly9_v2` 喂进 NaN，而它内部是 Chebyshev 拟合，NaN 会一路扩散。
        """
        close = np.cumsum(np.random.default_rng(8).normal(0, 1, 300)) + 100.0
        hma10 = TA_HMA(close, 10)
        s = np.r_[close[0:11], hma10[11:]]
        self.assertEqual(len(s), len(close))
        self.assertFalse(np.isnan(s).any(), '拼接后不许有 NaN')


class TestAvailability(unittest.TestCase):
    def test_available_is_bool(self):
        self.assertIsInstance(available(), bool)

    def test_missing_talib_raises_import_error(self):
        """缺 talib 时给**人话**，不是裸 `NameError`。"""
        from GolemQ.analysis import indicators
        saved = indicators.talib
        indicators.talib = None
        try:
            with self.assertRaises(ImportError) as ctx:
                indicators.TA_HMA(np.arange(20, dtype=float), 10)
            self.assertIn('talib', str(ctx.exception))
        finally:
            indicators.talib = saved


if __name__ == '__main__':
    unittest.main()
