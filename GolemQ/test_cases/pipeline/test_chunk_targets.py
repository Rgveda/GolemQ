#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""`chunk_targets`：按**日历年**切块，块末落库（D88）。

起因：一个 review 从没跑过的码要一次回填 800 天 × 三族 ≈2.5 小时，
撞单码护栏就全丢。切块后每块 ~30 分钟，块末写库，中断最多丢一块。

三条契约：

1. **按时间切，不按个数切** —— 60min 与 15min 的目标密度差 4 倍，
   按个数切会让某个 60min 目标与它的 15min 依赖落到不同块
   （`_hour_targets` 依赖它们同块）；
2. **并集 = 原集合、块间不重叠**（切块不能丢目标也不能重复算）；
3. **本来就不跨年 ⇒ 只产出一块**（老行为，零风险）。

⚠️ 纯函数，不碰 DB。
"""

import os
import sys
import unittest

import pandas as pd

try:
    import GolemQ  # noqa: F401
except ImportError:
    sys.path.insert(0, os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from GolemQ.pipeline.review.pipeline import ReviewOptions, chunk_targets


def _idx(*dates):
    return pd.DatetimeIndex(pd.to_datetime(list(dates)))


class TestChunkTargets(unittest.TestCase):

    def test_splits_by_calendar_year(self):
        t = _idx('2023-06-01', '2024-03-04', '2024-11-20', '2025-01-06')
        chunks = list(chunk_targets({'day': t}))
        self.assertEqual([len(c['day']) for c in chunks], [1, 2, 1])
        self.assertEqual([[x.year for x in c['day']] for c in chunks],
                         [[2023], [2024, 2024], [2025]])

    def test_union_preserved_and_no_overlap(self):
        day = _idx('2023-06-01', '2024-03-04', '2024-11-20', '2025-01-06')
        m15 = _idx('2023-06-01 10:00', '2024-03-04 10:00', '2025-01-06 10:00')
        chunks = list(chunk_targets({'day': day, '15min': m15}))
        seen_day = [t for c in chunks for t in c['day']]
        seen_15 = [t for c in chunks for t in c['15min']]
        self.assertEqual(sorted(seen_day), sorted(day), '不能丢、不能重复')
        self.assertEqual(sorted(seen_15), sorted(m15))
        self.assertEqual(sum(len(c['day']) for c in chunks), len(day))

    def test_all_families_share_the_same_time_window(self):
        """每块里各族的年份集合必须一致（60min 目标要跟它的 15min 依赖同块）。"""
        day = _idx('2024-02-01', '2025-02-01')
        m15 = _idx('2024-02-01 10:00', '2025-02-01 10:00')
        for c in chunk_targets({'day': day, '15min': m15}):
            years = {int(pd.Timestamp(t).year) for v in c.values() for t in v}
            self.assertEqual(len(years), 1, f'同一块不能跨年：{years}')

    def test_single_year_yields_one_chunk(self):
        t = _idx('2024-01-02', '2024-12-31')
        self.assertEqual(len(list(chunk_targets({'day': t}))), 1)

    def test_max_years_zero_disables_chunking(self):
        t = _idx('2023-06-01', '2025-01-06')
        chunks = list(chunk_targets({'day': t}, max_years=0))
        self.assertEqual(len(chunks), 1)
        self.assertEqual(len(chunks[0]['day']), 2)

    def test_two_year_buckets(self):
        t = _idx('2023-06-01', '2024-03-04', '2025-01-06')
        chunks = list(chunk_targets({'day': t}, max_years=2))
        self.assertEqual([len(c['day']) for c in chunks], [2, 1])

    def test_empty_targets_yield_once(self):
        chunks = list(chunk_targets({'day': pd.DatetimeIndex([])}))
        self.assertEqual(len(chunks), 1)
        self.assertEqual(len(chunks[0]['day']), 0)

    def test_default_option_is_one_year(self):
        self.assertEqual(ReviewOptions().max_years, 1.0)


if __name__ == '__main__':
    unittest.main()
