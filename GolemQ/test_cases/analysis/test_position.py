# coding: utf-8
"""「全局智能仓位」纯函数：D-S 合成、日→60min 广播、前奏计数。"""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from GolemQ.analysis.mainstream_timing import (broadcast_daily_to_60min,
                                               calc_prelude_before)
from GolemQ.analysis.position import calc_position_ds, calc_position_series


class TestPositionDS(unittest.TestCase):

    def test_three_way_split(self):
        r = calc_position_ds([10, 1, 5], [1, 10, 5], [-1, 1, 1])
        self.assertAlmostEqual(r['m_bull'], 1 / 3)
        self.assertAlmostEqual(r['m_bear'], 1 / 3)
        self.assertAlmostEqual(r['m_omega'], 1 / 3)
        self.assertAlmostEqual(r['onhold'], 1 / 3)
        self.assertEqual(r['members'], 3)

    def test_forced_clear_at_618(self):
        """做空占比 100% ⇒ 强制清仓；60% 不触发。"""
        self.assertEqual(calc_position_ds([1] * 10, [0] * 10, [-1] * 10)['onhold'], 0.0)
        r = calc_position_ds([1] * 10, [0] * 6 + [1] * 4, [-1] * 6 + [1] * 4)
        self.assertAlmostEqual(r['m_bear'], 0.6)
        self.assertAlmostEqual(r['onhold'], r['m_bull'])

    def test_nan_counts_as_omega(self):
        """NaN 参与比较恒 False ⇒ 落入 Ω（与旧树一致，不提前剔除）。"""
        r = calc_position_ds([1, np.nan], [2, 1], [1, np.nan])
        self.assertEqual(r['members'], 2)
        self.assertAlmostEqual(r['m_bull'], 0.5)
        self.assertAlmostEqual(r['m_omega'], 0.5)

    def test_empty(self):
        r = calc_position_ds([], [], [])
        self.assertEqual(r['members'], 0)
        self.assertTrue(np.isnan(r['m_bull']))

    def test_series_matches_per_bar(self):
        n = np.array([3, 3, 0])
        bull = np.array([1, 0, 0])
        bear = np.array([1, 3, 0])
        s = calc_position_series(n, bull, bear)
        r0 = calc_position_ds([10, 1, 5], [1, 10, 5], [-1, 1, 1])
        self.assertAlmostEqual(s['m_bull'][0], r0['m_bull'])
        self.assertAlmostEqual(s['m_bear'][1], 1.0)
        self.assertEqual(s['onhold'][1], 0.0)
        self.assertTrue(np.isnan(s['onhold'][2]))
        self.assertEqual(s['ds_state'].tolist(), [0, -1, 0])


class TestPositionFlags(unittest.TestCase):

    def test_full_and_onset(self):
        from GolemQ.analysis.position import calc_position_flags
        mb = np.array([0.1, 0.7, 0.7, 0.65, 0.2])
        mbe = np.array([0.8, 0.1, 0.1, 0.7, 0.7])
        mo = np.array([0.1, 0.2, 0.2, 0.1, 0.1])
        oh = np.array([0.0, 0.7, 0.7, 0.0, 0.2])
        f = calc_position_flags(mb, mbe, mo, oh)
        self.assertEqual(f['PosFull'].tolist(), [False, True, True, False, False])
        self.assertEqual(f['PosFullOnset'].tolist(), [False, True, False, False, False])

    def test_re_clear_gap(self):
        from GolemQ.analysis.position import calc_position_flags
        mb = np.zeros(12); mbe = np.ones(12); mo = np.zeros(12)
        oh = np.ones(12)
        oh[[0, 1, 6, 10, 11]] = 0.0      # 段 A(0-1) 段 B(6) 间隔4根 → 暴击；段 C(10-11) 间隔3根 → 暴击
        f = calc_position_flags(mb, mbe, mo, oh, gap_bars=4)
        self.assertEqual(f['PosReClear'].tolist(),
                         [False, False, False, False, False, False,
                          True, False, False, False, True, False])

    def test_re_clear_prev_start(self):
        """暴击点要带回「前一个零仓段的起点」下标（供指数新低条件用）。"""
        from GolemQ.analysis.position import calc_position_flags
        oh = np.ones(12); oh[[0, 1, 6, 10, 11]] = 0.0
        f = calc_position_flags(np.zeros(12), np.ones(12), np.zeros(12), oh,
                                gap_bars=4)
        idx = f['ReClearPrevStart']
        self.assertEqual(idx[6], 0)
        self.assertEqual(idx[10], 6)
        self.assertEqual(idx[0], -1)

    def test_bottom_filter(self):
        """空仓 + STAGE_MODE=-0.5 ⇒ PosBottom；PosProbe 回到原始 m_bull。"""
        from GolemQ.analysis.position import calc_position_flags
        mb = np.array([0.2, 0.2, 0.2, 0.2])
        mbe = np.array([0.7, 0.7, 0.7, 0.7])
        mo = np.array([0.1, 0.1, 0.1, 0.1])
        oh = np.array([0.0, 0.0, 0.0, 0.0])
        f = calc_position_flags(mb, mbe, mo, oh, stage_mode=[-0.5, 0.0, 1.0, np.nan])
        self.assertEqual(f['PosBottom'].tolist(), [True, False, False, False])
        self.assertEqual(f['PosProbe'].tolist(), [0.2, 0.0, 0.0, 0.0])

    def test_deep_bottom(self):
        """雪球底：连续>=3 天 -0.5 且 m_bear>=0.90（2024-02-05/06 唯一命中）。"""
        from GolemQ.analysis.position import calc_position_flags
        n = 6
        f = calc_position_flags(np.zeros(n), np.full(n, 0.95), np.full(n, 0.05),
                                np.zeros(n),
                                stage_mode=[-0.5] * n,
                                stage_streak=[1, 1, 2, 2, 3, 3])
        self.assertEqual(f['PosDeepBottom'].tolist(),
                         [False, False, False, False, True, True])
        # m_bear 只有 0.85 ⇒ 不算雪球底
        f2 = calc_position_flags(np.zeros(n), np.full(n, 0.85), np.full(n, 0.15),
                                 np.zeros(n), stage_mode=[-0.5] * n,
                                 stage_streak=[1, 1, 2, 2, 3, 3])
        self.assertFalse(f2['PosDeepBottom'].any())

    def test_bottom_requires_empty_position(self):
        """仓位没被清零时（有持仓），不触发底部试探。"""
        from GolemQ.analysis.position import calc_position_flags
        f = calc_position_flags(np.array([0.5]), np.array([0.2]), np.array([0.3]),
                                np.array([0.5]), stage_mode=[-0.5])
        self.assertEqual(f['PosBottom'].tolist(), [False])

    def test_first_segment_is_not_re_clear(self):
        """序列第一个清仓段没有「上一段」，不算暴击；之后隔 1 根再清算暴击。"""
        from GolemQ.analysis.position import calc_position_flags
        oh = np.array([0.0, 0.0, 0.5, 0.0])
        f = calc_position_flags(np.zeros(4), np.ones(4), np.zeros(4), oh)
        self.assertEqual(f['PosReClear'].tolist(), [False, False, False, True])


class TestBroadcast(unittest.TestCase):

    def test_multiindex_index60(self):
        idx = pd.MultiIndex.from_tuples(
            [(pd.Timestamp('2024-01-02 10:30'), '600519'),
             (pd.Timestamp('2024-01-02 15:00'), '600519')],
            names=['ts', 'code'])
        daily = pd.Series([1.0, 2.0],
                          index=pd.to_datetime(['2024-01-01', '2024-01-02']))
        out = broadcast_daily_to_60min(daily, idx)
        self.assertEqual(len(out), 2)
        self.assertTrue(np.allclose(out.to_numpy(), [2.0, 2.0]))

    def test_prelude_requires_columns(self):
        with self.assertRaises(KeyError):
            calc_prelude_before(pd.DataFrame({'x': [1]}))

    def test_prelude_masks(self):
        f = pd.DataFrame({
            'QLevMACDLagD': [1, 1, -1, 1],
            'QuadDwnRskLagMaj': [1, 1, 1, -1],
            'QuadDwnRskLag': [1, 1, 1, -1],
            'zDLagD': [1, 1, 1, -1]})
        out = calc_prelude_before(f)
        self.assertEqual(out['QuadPreUprBf'].tolist(), [0, 0, 1, 2])
        self.assertEqual(out['QuadPreEpBf'].tolist(), [1, 2, 3, 0])


if __name__ == '__main__':
    unittest.main()
