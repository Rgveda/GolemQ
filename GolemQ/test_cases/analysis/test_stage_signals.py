#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""`stage = -0.5` 回归：000711 实测值（用户 2026-11-16 要求验证）。

数据来源：8.3 `stock_reality_feat_60min`（`stage/garPD` 由 4.4 搬运，见 D81）。

* `garPD` 锚日 **2022-09-21**，锚日 15:00 close = **2.51**（复权后口径）；
* **2022-09-26** 15:00 close = **2.35** → 跌破最底带下沿 **2.39** ⇒ `stage = -0.5`；
* **2022-09-28** close = 2.37 → 同为 `-0.5`（仍在最底带下沿之下）。

⚠️ **还原时必须乘「锚日」的 close**（本例 2.51），不是当根 bar 的 close（2.35）：
用后者会把整组带缩到最底带下沿 ≈2.24，close 2.35 落回带内 ⇒ 错判成 `0`。
`reality_attach._restore_garident_bands` 已按锚日口径实现。
"""
from __future__ import annotations

import os
import sys

try:
    import GolemQ  # noqa: F401
except ImportError:
    sys.path.insert(0, os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', '..', '..')))

import unittest

import numpy as np

from GolemQ.analysis.stages import (calc_garident_stage,
                                    garident_bands_from_ratio)
from GolemQ.analysis.timing import Timeline_duration

#: 000711 在锚日 2022-09-21 写入的 `garPD`（比值带，落库原值）。
GAR_PD_000711 = [[1.0756972342307267, 0.9521912804565009],
                 [1.2111553678908271, 1.0796812941661245],
                 [1.4223107357418137, 1.2350597276227362]]

#: 锚日（2022-09-21 15:00）的 close。
ANCHOR_CLOSE = 2.51


#: 000711 在锚日 **2025-12-18** 写入的 `garPD`（用户 2026-11-16 指定的验证点）：
#: 2026-01-20 收盘 1.68 仍在最底带 [1.67, 1.72] 内（stage 0）；
#: **2026-01-21 15:00 收盘 1.67** 跌破最底带下沿 ⇒ `stage = -0.5`。
GAR_PD_000711_JAN26 = [[1.0238095728273824, 0.994047624575893],
                       [1.0595238256101196, 1.0297619483110132],
                       [1.125000026644346, 1.0654762010937504]]
ANCHOR_CLOSE_JAN26 = 1.68


class TestStageJan2026(unittest.TestCase):
    """000711 在 **2026-01-20 前后** 触发 `stage = -0.5`（用户验证点）。"""

    def _bands(self):
        return garident_bands_from_ratio(GAR_PD_000711_JAN26, ANCHOR_CLOSE_JAN26)

    def test_jan_21_close_breaks_bottom_band(self):
        bands = self._bands()
        self.assertAlmostEqual(bands[0][1], 1.67, places=2)
        self.assertEqual(calc_garident_stage(1.67, bands), -0.5)

    def test_jan_20_close_is_still_inside(self):
        self.assertEqual(calc_garident_stage(1.68, self._bands()), 0.0)

    def test_bst_resets_on_the_trigger(self):
        stage = np.array([0.0, 0.0, -0.5, -0.5, 0.0])   # 01-20 两根 → 01-21/22 → 回落
        self.assertEqual(
            Timeline_duration(np.where(stage < -0.00168, 1, 0)).tolist(),
            [1, 2, 0, 0, 1])


class TestStageMinusHalf(unittest.TestCase):
    def _bands(self):
        return garident_bands_from_ratio(GAR_PD_000711, ANCHOR_CLOSE)

    def test_below_bottom_band_gives_minus_half(self):
        bands = self._bands()
        self.assertAlmostEqual(bands[0][1], 2.39, places=2)
        self.assertEqual(calc_garident_stage(2.35, bands), -0.5)
        self.assertEqual(calc_garident_stage(2.37, bands), -0.5)

    def test_inside_bottom_band_is_zero(self):
        self.assertEqual(calc_garident_stage(2.40, self._bands()), 0)

    def test_bst_resets_when_stage_is_minus_half(self):
        """`bstStgBf = Timeline_duration(stage < -0.00168)` ⇒ -0.5 那根归零。"""
        stage = np.array([1.0, -0.5, 0.5, -0.5, -0.5])
        self.assertEqual(
            Timeline_duration(np.where(stage < -0.00168, 1, 0)).tolist(),
            [1, 0, 1, 0, 0])

    def test_using_bar_close_instead_of_anchor_close_flips_to_zero(self):
        """反例：分母误用当根 close（2.35）⇒ band 缩水，错判成 0（实测踩过）。"""
        wrong = garident_bands_from_ratio(GAR_PD_000711, 2.35)
        self.assertEqual(calc_garident_stage(2.35, wrong), 0)


if __name__ == '__main__':
    unittest.main()
