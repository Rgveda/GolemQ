# coding: utf-8
"""`chip_cost_by_day` 缺个别交易日换手率时不能整段跳过（000711 回归）。"""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from GolemQ.pipeline.review.steps.hour_extra import chip_cost_by_day

CODE = '600519'


def _day_frame(periods=40):
    days = pd.bdate_range('2024-01-02', periods=periods)
    idx = pd.MultiIndex.from_arrays([days, [CODE] * periods],
                                    names=['datetime', 'code'])
    close = pd.Series(np.linspace(10.0, 11.0, periods), index=idx)
    return days, pd.DataFrame({
        'open': close, 'high': close + 0.1, 'low': close - 0.1,
        'close': close, 'volume': 1000.0}, index=idx)


def _fake_chip(features=None, ohlc_data=None, annual=1008, verbose=False,
               *args, **kwargs):
    idx = pd.MultiIndex.from_arrays(
        [ohlc_data.index.get_level_values(0), [CODE] * len(ohlc_data)],
        names=['ts', 'code'])
    return pd.DataFrame({
        'cost5Prz': 10.0, 'cost15Prz': 10.5,
        'cost85Prz': 10.8, 'cost95Prz': 11.0}, index=idx)


class TestChipBackfill(unittest.TestCase):

    def test_missing_single_turnover_day_drops_row_not_target(self):
        days, day = _day_frame()
        turnover = pd.Series(0.01, index=day.index)
        turnover.iloc[10] = np.nan                 # 中间缺一天
        targets = pd.DatetimeIndex([days[-1] + pd.Timedelta(hours=15)])

        mock = MagicMock(side_effect=_fake_chip)
        with patch('GolemQ.analysis.ChipDistribution_jit.'
                   'calc_stock_chip_distribution_jit', mock):
            got_days, got = chip_cost_by_day(
                day, targets, CODE, turnover=turnover)

        self.assertEqual(mock.call_count, 1)
        self.assertEqual(len(got_days), 1)
        self.assertEqual(pd.Timestamp(got_days[0]), pd.Timestamp(days[-1]))
        self.assertAlmostEqual(float(got.iloc[0]['cost15Prz']), 10.5)


if __name__ == '__main__':
    unittest.main()
