# coding:utf-8
"""`analysis/xgboost/markup.py` 的用例。

两块：

* **逐值对拍**：`calc_transcation_stats_np` / `calc_fractal_stats(mode='raw')`
  与老树 `portfolio/utils.py` 在同一组输入上必须**完全相等**；
* **标签规则**：在合成帧上跑 `markup_uprising_labels`，钉住「不修改入参 /
  标签值域 / 缺列拒绝 / 可选输入降级」。
"""
from __future__ import annotations

import os, sys
try:
    import GolemQ
except ImportError:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

import unittest

import numpy as np
import pandas as pd

from GolemQ.analysis.timing import (Timeline_Integral, Timeline_duration,
                                    calc_event_timing_lag)
from GolemQ.analysis.xgboost import markup as M
from GolemQ.core.constants import AKA
from GolemQ.test_cases.old_tree import load_old, old_tree_available

#: 老树 `portfolio/utils.py` 的 `Timeline_duration` 等来自
#: `GolemQ.analysis.timeseries import *`；D49 之后它们住在 `analysis/timing.py`
#: ⇒ 载入老模块前把名字**补到真模块上**（old_tree 的第 2 条纪律：补属性，不替换）。
_PATCH = {
    'GolemQ.analysis.timeseries': {
        'Timeline_duration': Timeline_duration,
        'Timeline_Integral': Timeline_Integral,
        'calc_event_timing_lag': calc_event_timing_lag,
    }
}
_OLD = load_old('portfolio/utils.py', patch_attrs=_PATCH) if old_tree_available() else None


def _ohlc(n=120, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.MultiIndex.from_arrays(
        [pd.date_range('2026-01-01', periods=n), ['600519'] * n],
        names=['date', 'code'])
    close = 10 + np.cumsum(rng.normal(0, 0.2, n))
    return pd.DataFrame({
        AKA.OPEN: close, AKA.CLOSE: close,
        AKA.HIGH: close + 0.1, AKA.LOW: close - 0.1,
        AKA.VOLUME: 1000.0,
    }, index=idx), rng


class TestTransactionStats(unittest.TestCase):

    def test_shape_and_long_column(self):
        rng = np.random.default_rng(1)
        n = 80
        trig = np.where(rng.random(n) > 0.85, 1, -1).astype(np.int16)
        close = (10 + np.cumsum(rng.normal(0, 0.2, n))).astype(np.float32)
        fc = np.where(rng.random(n) > 0.5, 1, -1).astype(np.int32)
        fl = np.where(rng.random(n) > 0.5, 1, -1).astype(np.int32)
        out = M.calc_transcation_stats_np(trig, close, fc, fl)
        self.assertEqual(out.shape[1], 14)
        self.assertTrue((out[:, 12] == 1).all())

    @unittest.skipIf(_OLD is None, '老树 GolemQ_old/ 不在场，对拍整份跳过')
    def test_matches_old_tree_exactly(self):
        rng = np.random.default_rng(2)
        n = 120
        trig = np.where(rng.random(n) > 0.85, 1, -1).astype(np.int16)
        close = (10 + np.cumsum(rng.normal(0, 0.2, n))).astype(np.float32)
        fc = np.where(rng.random(n) > 0.5, 1, -1).astype(np.int32)
        fl = np.where(rng.random(n) > 0.5, 1, -1).astype(np.int32)
        # 老树的 numba 显式签名要求 (int16, f4, int32, int32, float64, int16)
        want = _OLD.calc_transcation_stats_np(
            trig, close, fc, fl, np.float64(0.0003), np.int16(1))
        got = M.calc_transcation_stats_np(trig, close, fc, fl, 0.0003, 1)
        np.testing.assert_array_equal(got, want)


class TestFractalStats(unittest.TestCase):

    @unittest.skipIf(_OLD is None, '老树 GolemQ_old/ 不在场，对拍整份跳过')
    def test_matches_old_tree_exactly(self):
        frame, rng = _ohlc(120, seed=3)
        trig = np.where(rng.random(120) > 0.85, 1, -1).astype(np.int16)
        want = _OLD.calc_fractal_stats(
            '600519', 'x', 'uid', trig, mode='raw',
            ref_features=frame, annual=1008)
        got = M.calc_fractal_stats(
            '600519', 'x', 'uid', trig, mode='raw',
            ref_features=frame, annual=1008)
        pd.testing.assert_frame_equal(got, want)

    def test_only_raw_mode_is_ported(self):
        frame, _ = _ohlc(60)
        with self.assertRaises(NotImplementedError):
            M.calc_fractal_stats('600519', 'x', 'uid',
                                 np.ones(60, dtype=np.int16),
                                 mode='zen', ref_features=frame, annual=1008)


def _label_frame(n=300, seed=5):
    """一帧含全部必需/可选 label 输入（canonical 名优先）。"""
    frame, rng = _ohlc(n, seed=seed)
    vals = rng.normal(0, 0.5, n)
    for col in ('ZenOLSLag', 'POLY9_Donchian_STAR_LAG',
                'MPWR3_MTR_CAN_LAG', 'Score_M15_PUNCH',
                'POLY9_Donchian_STAR', 'POLY9_Donchian_BAND_CAN',
                'MPWR3_MTR_CAN', 'MTM_FLOW_LAG', 'MONEYFLOW_IN_BF',
                'zDLagMajR', 'zPLagMinR', 'zDLagMinR', 'poly9pwr3',
                'mtmNorm', 'ATR_Stopline_LAG', 'ATR_SUPTRD_LAG',
                'DESINTY_CHAIN_LAG', 'MPWR3_RAIL', 'macdOLag',
                'PeakLowLag', 'm9t', 'm9tMaj', 'regtreeSlp',
                'ssm15Norm'):
        frame[col] = vals + rng.normal(0, 0.1, n)
    # 两个分型触发列必须是 ±1（老树同款）
    frame['ZenOLSLag'] = np.where(rng.random(n) > 0.8, 1.0, -1.0)
    frame['POLY9_Donchian_STAR_LAG'] = np.where(rng.random(n) > 0.8, 1.0, -1.0)
    return frame


class TestMarkupLabels(unittest.TestCase):

    def test_labels_are_in_domain_and_input_untouched(self):
        frame = _label_frame()
        before = set(frame.columns)
        out = M.markup_uprising_labels(frame)
        self.assertIn(M.ATTENTION_LABEL, out.columns)
        self.assertIn(M.ATTENTION_NEGATIVE_LABEL, out.columns)
        self.assertEqual(set(frame.columns), before)       # 入参未被修改
        labels = set(out[M.ATTENTION_LABEL].dropna().unique())
        self.assertTrue(labels <= {1, 2, 3, 4, 5, 6, -1, -2, -3, -4}, labels)
        neg = set(out[M.ATTENTION_NEGATIVE_LABEL].dropna().unique())
        self.assertTrue(neg <= {1, 2, 3, 4, 5, 6, -1, -2, -3, -4}, neg)

    def test_missing_required_column_is_rejected(self):
        frame = _label_frame().drop(columns=['macdOLag'])
        with self.assertRaises(ValueError) as ctx:
            M.markup_uprising_labels(frame)
        self.assertIn('MACD_ZERO_TIMING_LAG', str(ctx.exception))

    def test_optional_inputs_can_be_absent(self):
        """`Score_M15_PUNCH` / `MPWR3_MTR_CAN_LAG` / `regtreeSlp` 在旧树里可缺。"""
        frame = _label_frame().drop(columns=[
            'Score_M15_PUNCH', 'MPWR3_MTR_CAN_LAG', 'regtreeSlp'])
        out = M.markup_uprising_labels(frame)
        self.assertIn(M.ATTENTION_LABEL, out.columns)


if __name__ == '__main__':
    unittest.main()
