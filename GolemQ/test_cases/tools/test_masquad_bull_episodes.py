#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""`tools/masquad_bull_episodes.py::build_episodes` 的分段口径。

「MasQuad 版两月牛」= 周线 `MasQuadPwrlnWekMed` 与 `MasQuadPwrlnWekMode`
**同时 >= 0** 的连续区间，中间允许不超过 `gap` 个交易日的负脉冲并入
（用户 2026-11-22：「中间转负之和不到 10 天可视为牛市一直持续」）。

三条要钉住：

1. 负脉冲 ≤ gap ⇒ 并进同一段（**不**把一段牛市切成两段）；
2. 负脉冲 > gap ⇒ 断开（这是真的转熊，不是回踩）；
3. 短于 `min_len` 的碎片丢弃（避免把一两天的抖动当行情）。

⚠️ 纯合成序列，不碰 DB。
"""

import os
import sys
import unittest

import pandas as pd

_REPO = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..'))
for _p in (_REPO, os.path.join(_REPO, 'tools')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import masquad_bull_episodes as M  # noqa: E402


def _series(vals):
    idx = pd.bdate_range('2024-01-01', periods=len(vals))
    return pd.Series(vals, index=idx, dtype=float)


class TestBuildEpisodes(unittest.TestCase):

    def test_short_negative_gap_is_merged(self):
        med = _series([1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0])
        mode = _series([1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0])
        eps, d = M.build_episodes(med, mode, gap=5, min_len=3)
        self.assertEqual(len(eps), 1)
        self.assertEqual((eps[0][0], eps[0][1]), (0, len(d) - 1))

    def test_long_negative_gap_splits(self):
        med = _series([1.0, 1.0, 1.0] + [-1.0] * 6 + [1.0, 1.0, 1.0])
        mode = med.copy()
        eps, d = M.build_episodes(med, mode, gap=5, min_len=3)
        self.assertEqual(len(eps), 2, '6 天负脉冲 > gap=5 必须断成两段')
        self.assertEqual(eps[0], (0, 2))
        self.assertEqual(eps[1], (9, 11))

    def test_min_len_drops_fragments(self):
        med = _series([1.0, 1.0, -1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0])
        mode = med.copy()
        eps, _ = M.build_episodes(med, mode, gap=1, min_len=3)
        # 只有最后那段 3 天够长；开头 2 天与中间 1 天的碎片都丢掉
        self.assertEqual(len(eps), 1)
        self.assertEqual(eps[0], (7, 9))

    def test_mode_negative_breaks_even_if_med_positive(self):
        med = _series([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        mode = _series([1.0, 1.0, -1.0, -1.0, -1.0, 1.0])
        eps, _ = M.build_episodes(med, mode, gap=1, min_len=1)
        self.assertEqual(eps, [(0, 1), (5, 5)],
                         'Mode 转负同样算「非双正」，不是只看 Med')


if __name__ == '__main__':
    unittest.main()


class TestDerivedClearSignals(unittest.TestCase):
    """`derived_clear_signals`：库里的 `PosReClear` 只认 60min 4 根，
    日线上要自己派生。这里钉住「高位跳水」这一条 —— 它才有信息量。"""

    def _daily(self, vals):
        idx = pd.bdate_range('2025-01-02', periods=len(vals))
        return pd.DataFrame({'MasPosOnhold': vals}, index=idx)

    def test_drop_fires_when_high_then_collapses(self):
        d = self._daily([0.60, 0.62, 0.61, 0.55, 0.58, 0.00, 0.10, 0.30])
        sig = M.derived_clear_signals(d)
        self.assertEqual(list(sig['drop']), [False] * 5 + [True, False, False],
                         '前 5 日最高 >=0.5、当日 <0.25 ⇒ 高位跳水；'
                         '只报跌入那天，低仓区里不连报')

    def test_no_drop_in_flat_low_regime(self):
        d = self._daily([0.20, 0.22, 0.10, 0.05, 0.00, 0.18])
        sig = M.derived_clear_signals(d)
        self.assertFalse(sig['drop'].any(),
                         '一直低仓时不算跳水 —— 否则会在低仓区天天触发')

    def test_reclear_needs_quick_reopen(self):
        # 零仓 2 天 → 重开 1 天 → 再零仓 ⇒ 触发
        d = self._daily([0.40, 0.00, 0.00, 0.30, 0.00, 0.30])
        sig = M.derived_clear_signals(d)
        self.assertTrue(sig['reclear_d'].iloc[4])
        # 重开太久（> reopen_days）⇒ 不算
        d2 = self._daily([0.40, 0.00, 0.30, 0.30, 0.30, 0.30, 0.00])
        sig2 = M.derived_clear_signals(d2)
        self.assertFalse(sig2['reclear_d'].any())
