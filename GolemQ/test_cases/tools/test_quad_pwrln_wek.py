#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""`tools/backfill_quad_pwrln_wek.py` 的三个口径（D86）。

背景：这个工具最初只支持 2024-06 以后的回填，直接把区间拉长会踩三个坑，
而且**每个坑都不报错，只是数变低**：

1. 日线回看不足 ⇒ 周线 MA233 全 NaN ⇒ `4QPwrln` 系统性偏低 1~2 档；
2. `weekly_ohlc` 的周标签是「该周最后一个实际交易日」，区间尾部被截断时
   最后一根会吃到只含两三天的**不完整周**；
3. 面板大小依赖 `end`（`len(day) < 520` 闸 + MA233 需要 233 周），同一个
   历史日期在两份 CSV 里能算出不同的中位数。

这三条都钉在下面。⚠️ 全部用合成序列，**不碰 DB / 网络 / joblib**。
"""

import os
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

_REPO = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..'))
for _p in (_REPO, os.path.join(_REPO, 'tools')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import backfill_quad_pwrln_wek as W  # noqa: E402


def _cal(*dates):
    return pd.DatetimeIndex(pd.to_datetime(list(dates))).normalize()


class TestDayStart(unittest.TestCase):
    """口径①：回看必须够 233 周（≈1,165 根日线），默认给 3,000 自然日。"""

    def test_default_lookback_reaches_2009_for_2018_start(self):
        self.assertEqual(W._day_start('2018-01-01', 3000), '2009-10-15')

    def test_lookback_is_at_least_233_weeks_of_days(self):
        """3,000 自然日 ≈ 2,050 交易日 ≫ 1,165（233 周）。"""
        start, day_start = '2018-01-01', W._day_start('2018-01-01', 3000)
        span = (pd.Timestamp(start) - pd.Timestamp(day_start)).days
        self.assertGreater(span * 5 / 7, 233 * 5)


class TestWeekEndMask(unittest.TestCase):
    """口径②：用延伸日历判「该周最后一个交易日」，节假日/截断都能认出来。"""

    def test_friday_is_week_end(self):
        dates = _cal('2024-07-15', '2024-07-16', '2024-07-17',
                     '2024-07-18', '2024-07-19')
        cal_ext = _cal('2024-07-15', '2024-07-16', '2024-07-17',
                       '2024-07-18', '2024-07-19', '2024-07-22')
        mask = W._week_end_mask(dates, cal_ext)
        self.assertEqual(list(mask), [False, False, False, False, True])

    def test_truncated_tail_is_not_week_end(self):
        """区间止于周三 ⇒ 该周在延伸日历里还没走完，不能当完整周。"""
        dates = _cal('2024-07-29', '2024-07-30', '2024-07-31')
        cal_ext = _cal('2024-07-29', '2024-07-30', '2024-07-31',
                       '2024-08-01', '2024-08-02')
        self.assertEqual(list(W._week_end_mask(dates, cal_ext)),
                         [False, False, False])

    def test_holiday_short_week_end_is_thursday(self):
        """周五休市 ⇒ 该周最后交易日是周四，周四要算完整周。"""
        dates = _cal('2024-09-16', '2024-09-17', '2024-09-18', '2024-09-19')
        cal_ext = _cal('2024-09-16', '2024-09-17', '2024-09-18',
                       '2024-09-19', '2024-09-23')
        self.assertEqual(list(W._week_end_mask(dates, cal_ext)),
                         [False, False, False, True])


class TestOnDates(unittest.TestCase):
    """`_on_dates` 的取值与 `--min-weeks` 过滤。"""

    #: 周一~周五；周标签按 weekly_ohlc 的口径取「该周最后一个交易日」
    DATES = _cal('2024-07-15', '2024-07-16', '2024-07-17',
                 '2024-07-18', '2024-07-19', '2024-07-22')
    WEEK_END = np.array([False, False, False, False, True, False])

    def _series(self, rows):
        idx = pd.DatetimeIndex(pd.to_datetime([d for d, _ in rows]))
        return pd.Series([v for _, v in rows], index=idx, dtype=float)

    def test_midweek_uses_previous_completed_week(self):
        s = self._series([('2024-07-12', 1.0), ('2024-07-19', 2.0)])
        with patch.object(W, '_weekly_series', lambda *a, **k: s):
            out = W._on_dates('600519', self.DATES, '2009-10-15',
                              '2024-07-22', self.WEEK_END)
        # 周一~周四吃 07-12 那一周；周五吃 07-19；下周一回到 07-19
        self.assertEqual(list(out), [1.0, 1.0, 1.0, 1.0, 2.0, 2.0])

    def test_truncated_partial_week_never_leaks_in(self):
        """尾部那根「只含 3 天」的周（标签=周三）不能被当完整周用。"""
        dates = _cal('2024-07-26', '2024-07-29', '2024-07-30', '2024-07-31')
        week_end = np.array([True, False, False, False])   # 07-31 周三被截断
        s = self._series([('2024-07-26', 3.0), ('2024-07-31', 9.0)])
        with patch.object(W, '_weekly_series', lambda *a, **k: s):
            out = W._on_dates('600519', dates, '2009-10-15', '2024-07-31',
                              week_end)
        self.assertEqual(list(out), [3.0, 3.0, 3.0, 3.0])
        self.assertNotIn(9.0, list(out))

    def test_min_weeks_drops_young_dates(self):
        """周线不足 233 根的日期填 NaN（新股 powerline 结构性偏低）。"""
        s = self._series([('2024-07-12', 1.0), ('2024-07-19', 2.0),
                          ('2024-07-22', 3.0)])
        with patch.object(W, '_weekly_series', lambda *a, **k: s):
            out = W._on_dates('301308', self.DATES, '2009-10-15',
                              '2024-07-22', self.WEEK_END, min_weeks=3)
        # 07-22「已见周数」= 3 才放行；前 5 个日期见到的周数 < 3 ⇒ NaN。
        # 放行后仍按周标签规则取值：07-22 是周一，吃上一完整周（07-19）= 2.0。
        self.assertTrue(np.isnan(out[:5]).all())
        self.assertEqual(out[5], 2.0)

    def test_empty_series_returns_none(self):
        with patch.object(W, '_weekly_series',
                          lambda *a, **k: pd.Series(dtype=float)):
            self.assertIsNone(W._on_dates('999999', self.DATES, '2009-10-15',
                                          '2024-07-22', self.WEEK_END))


class TestMatureMask(unittest.TestCase):
    """`_mature_mask`：周线截面与**逐码 `4QPwrlnWek`** 写入口径的唯一实现。

    2026-11-24：主流的 Wek 横截面读逐码列，若逐码列不 gate，截面会混进
    上市不足 233 周的新股（powerline 结构性偏低）—— 两边必须同一把尺子。
    """

    def _s(self, n):
        idx = pd.date_range('2018-01-05', periods=n, freq='7D')
        return pd.Series(np.arange(n, dtype=float), index=idx)

    def test_before_threshold_is_false(self):
        s = self._s(5)
        dates = s.index
        m = W._mature_mask(s, dates, 3)
        self.assertEqual(list(m), [False, False, True, True, True])

    def test_off_means_all_mature(self):
        s = self._s(3)
        self.assertTrue(W._mature_mask(s, s.index, 0).all())

    def test_intraday_dates_ffill_from_week_end(self):
        """60min 时间戳（15:00）按 ffill 取当周已见周数 —— 与 `_write_60min` 同口径。"""
        s = self._s(5)
        ts = pd.DatetimeIndex(['2018-01-05 15:00', '2018-01-08 15:00'])
        # 两条都 ffill 到 2018-01-05（已见 1 根周线）
        self.assertEqual(list(W._mature_mask(s, ts, 1)), [True, True])
        self.assertEqual(list(W._mature_mask(s, ts, 2)), [False, False])


if __name__ == '__main__':
    unittest.main()
