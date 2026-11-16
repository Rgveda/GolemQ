# coding: utf-8
"""`analysis/feature_flow.py` —— Hamilton DAG 的“各频段只算一次”。"""
from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from GolemQ.analysis import feature_flow as ff
from GolemQ.analysis import quadrant_jit as qj
from GolemQ.analysis.indicators import calc_macd_qa
from GolemQ.analysis.mainstream_timing import (broadcast_daily_to_60min,
                                               calc_prelude_before)
from GolemQ.analysis.quadrant_jit import (calc_four_quadrant_boost,
                                          calc_quadrant_downrisk_major,
                                          calc_quadrant_leverage_macd_timing_lag)
from GolemQ.core.constants import AKA, FIELD as FLD
from GolemQ.models.alias import LTT
from GolemQ.pipeline.mainstream.pipeline import MainstreamPipeline


def _ohlc(index, base=50.0):
    n = len(index)
    close = base + np.arange(n, dtype=float) * 0.01
    return pd.DataFrame({
        'open': close, 'high': close + 0.1, 'low': close - 0.1,
        'close': close, 'volume': 1000.0,
    }, index=index)


def _frames(days=200):
    day_idx = pd.bdate_range('2024-01-02', periods=days)
    stamps = ('10:30', '11:30', '14:00', '15:00')
    idx60 = pd.DatetimeIndex([
        d + pd.Timedelta(hours=int(t[:2]), minutes=int(t[3:]))
        for d in day_idx for t in stamps])
    return _ohlc(idx60), _ohlc(idx60), _ohlc(day_idx)


class TestFeatureFlow(unittest.TestCase):

    def test_graph_exposes_gear_and_sources(self):
        dr = ff.build_driver()
        names = {v.name for v in dr.list_available_variables()}
        self.assertTrue({'QuadGear0', 'QuadGear0Bf', 'q60', 'q15', 'qd',
                         'qlev_macd_lag', 'quad_downrisk_major'} <= names)
        self.assertNotIn('build_driver', names)

    def test_each_quadrant_runs_once(self):
        f60, f15, fd = _frames(40)
        calls = []

        def fake_quad(code, features=None, ohlc_data=None, annual=1008,
                      market_type=None):
            calls.append(annual)
            return pd.DataFrame({
                LTT.QUADRANT_POWERLINE: np.linspace(-3, 3, len(ohlc_data)),
                LTT.QUADRANT_DOWNRISK_TIMING_LAG:
                    np.linspace(3, -3, len(ohlc_data)),
            }, index=ohlc_data.index)

        with patch.object(ff, 'calc_four_quadrant_boost', fake_quad):
            dr = ff.build_driver()
            out = dr.execute(
                ['qlev_macd_lag', 'quad_downrisk_major'],
                inputs={'code': '600519', 'ohlc60': f60, 'ohlc15': f15,
                        'ohlc_day': fd})
        self.assertEqual(sorted(calls), [252, 1008, 4032])
        self.assertEqual(len(out['qlev_macd_lag']), len(f60))
        self.assertEqual(len(out['quad_downrisk_major']), len(f60))

    def test_downrisk_reuses_precomputed_day(self):
        f60, _, fd = _frames(40)
        idx60 = f60.index
        feat_day = calc_four_quadrant_boost(
            code='600519', features=pd.DataFrame(index=fd.index),
            ohlc_data=fd, annual=252)

        base = calc_quadrant_downrisk_major(
            '600519', fd, idx60, annual=252)
        with patch.object(qj, 'calc_four_quadrant_boost',
                          side_effect=AssertionError('不应重算日线四象限')):
            got = calc_quadrant_downrisk_major(
                '600519', None, idx60, feat_day=feat_day)
        pd.testing.assert_series_equal(base, got)

    def test_matches_manual_chain(self):
        """Hamilton 结果必须与 `MainstreamPipeline.code_features` 的手写链逐值一致。"""
        f60, f15, fd = _frames(120)
        q60 = calc_four_quadrant_boost(
            code='600519', features=pd.DataFrame(index=f60.index),
            ohlc_data=f60, annual=1008)
        q15 = calc_four_quadrant_boost(
            code='600519', features=pd.DataFrame(index=f15.index),
            ohlc_data=f15, annual=4032)
        qd = calc_four_quadrant_boost(
            code='600519', features=pd.DataFrame(index=fd.index),
            ohlc_data=fd, annual=252)

        manual = pd.DataFrame(index=f60.index)
        manual[LTT.QUADRANT_POWERLINE] = q60[LTT.QUADRANT_POWERLINE]
        p15 = q15[LTT.QUADRANT_POWERLINE].copy()
        p15.index = pd.DatetimeIndex(p15.index)
        p15 = p15[~p15.index.duplicated(keep='last')]
        manual[LTT.QUADRANT_POWERLINE_MINOR] = p15.reindex(f60.index)
        manual[LTT.QUADRANT_POWERLINE_MAJOR] = broadcast_daily_to_60min(
            qd[LTT.QUADRANT_POWERLINE], f60.index)
        _, _, macd = calc_macd_qa(fd[AKA.CLOSE])
        manual[FLD.MACD_DELTA_MAJOR] = broadcast_daily_to_60min(
            macd.diff(1), f60.index)
        manual = calc_quadrant_leverage_macd_timing_lag(manual)
        manual_qlev = manual[LTT.QUADRANT_LEVERAGE_MACD_TIMING_LAG_DUMMY]
        manual_dwn = calc_quadrant_downrisk_major(
            '600519', fd, f60.index)

        dr = ff.build_driver()
        out = dr.execute(
            ['qlev_macd_lag', 'quad_downrisk_major'],
            inputs={'code': '600519', 'ohlc60': f60, 'ohlc15': f15,
                    'ohlc_day': fd})
        pd.testing.assert_series_equal(
            out['qlev_macd_lag'], manual_qlev, check_names=False)
        np.testing.assert_array_equal(
            out['quad_downrisk_major'].to_numpy(), manual_dwn.to_numpy())

    def test_multiindex_frames(self):
        """真实行情帧是 `(datetime, code)` 两级索引；输出索引必须原样保留。"""
        f60, f15, fd = _frames(120)

        def mi(frame):
            return frame.set_index(pd.MultiIndex.from_arrays(
                [frame.index, ['600519'] * len(frame)],
                names=['datetime', 'code']))

        m60, m15, md = mi(f60), mi(f15), mi(fd)
        out = ff.quadrant_gear_features('600519', m60, m15, md)
        self.assertTrue(out.index.equals(m60.index))

    def test_full_frame_uses_storage_aliases(self):
        f60, f15, fd = _frames(120)
        frame = ff.quadrant_gear_frame('600519', f60, f15, fd)
        expected = set(ff.NODE_ALIASES.values()) | {
            LTT.QUADRANT_GEAR_O, LTT.QUADRANT_GEAR_O_BEFORE}
        self.assertEqual(set(frame.columns), expected)
        self.assertTrue(frame.index.equals(f60.index))
        self.assertTrue(frame[LTT.QUADRANT_GEAR_O_BEFORE].notna().all())

    def test_mainstream_rewind_stoploss_before_major(self):
        idx = pd.bdate_range('2024-01-02', periods=8)
        ms60 = pd.DataFrame({
            LTT.MAINSTREAM_LEADIN_BEFORE: [9, 9, 9, 9, 9, 10, 10, 10],
            LTT.MAINSTREAM_LEADOUT_BEFORE: [9, 9, 9, 9, 9, 10, 10, 10],
        }, index=idx)
        qlev = pd.Series(100.0, index=idx)
        leadin_maj = pd.Series(10.0, index=idx)
        leadout_maj = pd.Series(10.0, index=idx)
        # 4QPwrlnLagMaj：STOPLOSS 要 < -1e-12，REWIND 要 > 1e-12。
        pw_stop = pd.Series([0, 0, 0, 0, -1, 0, 0, 0], index=idx, dtype=float)
        pw_rew = pd.Series([0, 0, 0, 0, 1, 0, 0, 0], index=idx, dtype=float)
        pw_major = pd.Series(1.0, index=idx)  # 0.8 < 1.0 < 1.6
        ohlc = pd.DataFrame(index=idx)

        stop = ff.ms_leadout_stoploss_before_major(
            qlev, leadin_maj, ms60, pw_major, pw_stop, ohlc)
        rew = ff.ms_leadin_rewind_before_major(
            qlev, leadout_maj, ms60, pw_major, pw_rew, ohlc)

        # 只有第 4 行条件成立；第 5~7 行 leadin/leadout=10 不满足 <=9。
        want = [1, 2, 3, 4, 0, 1, 2, 3]
        self.assertEqual(stop.tolist(), want)
        self.assertEqual(rew.tolist(), want)
        self.assertEqual(stop.name,
                         LTT.MAINSTREAM_LEADOUT_STOPLOSS_BEFORE_MAJOR)
        self.assertEqual(rew.name,
                         LTT.MAINSTREAM_LEADIN_REWIND_BEFORE_MAJOR)

        # QLevMACDLagD == -1 的 bar 也强制视为 STOPLOSS 事件。
        qlev_neg = qlev.copy()
        qlev_neg.iloc[4] = -1.0
        stop_neg = ff.ms_leadout_stoploss_before_major(
            qlev_neg, leadin_maj, ms60, pw_major, pw_stop, ohlc)
        self.assertEqual(stop_neg.tolist(), want)

    def test_mainstream_rewind_stoploss_nodes_execute(self):
        f60, f15, fd = _frames(40)
        dr = ff.build_driver()
        out = dr.execute(
            ['pwrln_timing_lag_major',
             'ms_leadout_stoploss_before_major',
             'ms_leadin_rewind_before_major'],
            inputs={'code': '600519', 'ohlc60': f60, 'ohlc15': f15,
                    'ohlc_day': fd})
        for key in ('pwrln_timing_lag_major',
                    'ms_leadout_stoploss_before_major',
                    'ms_leadin_rewind_before_major'):
            self.assertEqual(len(out[key]), len(f60))
            self.assertTrue(out[key].index.equals(f60.index))

    def test_mainstream_code_features_matches_manual(self):
        """`MainstreamPipeline._features_from_frames` 与旧手写链逐值一致。"""
        f60, f15, fd = _frames(300)
        fake_zd = pd.Series(np.ones(len(f60), dtype=np.int32),
                            index=f60.index, name=FLD.ZEN_DASH_TIMING_LAG_DUMMY)
        pipe = MainstreamPipeline()

        with patch.object(ff, 'calc_zen_dash_causal',
                          lambda ohlc, code: fake_zd):
            got = pipe._features_from_frames('600519', f60, f15, fd)

            q60 = calc_four_quadrant_boost(
                code='600519', features=pd.DataFrame(index=f60.index),
                ohlc_data=f60, annual=1008)
            q15 = calc_four_quadrant_boost(
                code='600519', features=pd.DataFrame(index=f15.index),
                ohlc_data=f15, annual=4032)
            qd = calc_four_quadrant_boost(
                code='600519', features=pd.DataFrame(index=fd.index),
                ohlc_data=fd, annual=252)
            idx60 = pd.DatetimeIndex(f60.index)
            manual = pd.DataFrame(index=f60.index)
            manual[LTT.QUADRANT_POWERLINE] = q60[LTT.QUADRANT_POWERLINE]
            p15 = q15[LTT.QUADRANT_POWERLINE].copy()
            p15.index = pd.DatetimeIndex(p15.index)
            p15 = p15[~p15.index.duplicated(keep='last')]
            manual[LTT.QUADRANT_POWERLINE_MINOR] = p15.reindex(idx60)
            manual[LTT.QUADRANT_POWERLINE_MAJOR] = broadcast_daily_to_60min(
                qd[LTT.QUADRANT_POWERLINE], idx60)
            _, _, macd = calc_macd_qa(fd[AKA.CLOSE])
            manual[FLD.MACD_DELTA_MAJOR] = broadcast_daily_to_60min(
                macd.diff(1), idx60)
            manual = calc_quadrant_leverage_macd_timing_lag(manual)
            manual[LTT.QUADRANT_DOWNRISK_TIMING_LAG] = \
                q60[LTT.QUADRANT_DOWNRISK_TIMING_LAG]
            manual[LTT.QUADRANT_DOWNRISK_TIMING_LAG_MAJOR] = \
                calc_quadrant_downrisk_major('600519', fd, idx60).to_numpy()
            manual[FLD.ZEN_DASH_TIMING_LAG_DUMMY] = fake_zd.to_numpy()
            manual = calc_prelude_before(manual, window=pipe.zen_window)

        pd.testing.assert_frame_equal(got, manual, check_dtype=False,
                                      check_like=True)

    def test_gear_end_to_end(self):
        f60, f15, fd = _frames(200)
        out = ff.quadrant_gear_features('600519', f60, f15, fd)
        self.assertEqual(list(out.columns),
                         [LTT.QUADRANT_GEAR_O, LTT.QUADRANT_GEAR_O_BEFORE])
        self.assertTrue(out.index.equals(f60.index))
        self.assertTrue(out[LTT.QUADRANT_GEAR_O_BEFORE].notna().all())
        self.assertTrue(set(out[LTT.QUADRANT_GEAR_O].dropna().unique())
                        <= {1.0})


if __name__ == '__main__':
    unittest.main()
