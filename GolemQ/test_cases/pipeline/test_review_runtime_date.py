# coding: utf-8
"""runtime 的 `latest_closed_bar` 必须落到**最后一个交易日**。"""
from __future__ import annotations

import unittest
from datetime import datetime
from unittest.mock import patch

import pandas as pd

from GolemQ.pipeline.review.runtime import latest_closed_bar


class TestLatestClosedBar(unittest.TestCase):

    def _call(self, now, trade, last_day):
        with patch('GolemQ.markets.StockCN.date_utils.GQ_util_if_trade',
                   return_value=trade), \
                patch('GolemQ.markets.StockCN.date_utils.GQ_util_get_last_day',
                      return_value=datetime.combine(last_day, datetime.min.time())):
            return latest_closed_bar('60min', now=pd.Timestamp(now))

    def test_weekend_falls_back_to_last_trade_close(self):
        got = self._call('2026-11-07 10:00', trade=False,
                         last_day=pd.Timestamp('2026-11-06').date())
        self.assertEqual(got, pd.Timestamp('2026-11-06 15:00'))

    def test_before_open_uses_previous_trade_close(self):
        got = self._call('2026-11-06 08:00', trade=True,
                         last_day=pd.Timestamp('2026-11-05').date())
        self.assertEqual(got, pd.Timestamp('2026-11-05 15:00'))

    def test_after_close_uses_today_close(self):
        got = self._call('2026-11-06 16:00', trade=True,
                         last_day=pd.Timestamp('2026-11-06').date())
        self.assertEqual(got, pd.Timestamp('2026-11-06 15:00'))

    def test_intraday_floors_to_bar(self):
        got = self._call('2026-11-06 10:07', trade=True,
                         last_day=pd.Timestamp('2026-11-06').date())
        self.assertEqual(got, pd.Timestamp('2026-11-06 10:00'))


if __name__ == '__main__':
    unittest.main()
