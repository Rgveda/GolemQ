#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""「这一行算不算已持久化」的判据（D88）。

`persistence.persisted_keys` 的规则是「契约里**任意一列**非空 ⇒ 认为已算完」。
2026-11-21 给 60min 表补了 3,752,516 行**只有 `4QPwrlnWek`** 的 15:00 行
（`tools/backfill_quad_pwrln_wek.py`，D87），于是这些行被 review 当成
"已算完" —— 回填到 2024 年以前时那些 bar 会被**静默跳过**。

修法：判完成时排除「由别的工具写」的列（`contract.PERSIST_EXCLUDE`）。

⚠️ 纯常量/函数级断言，不碰 DB。
"""

import os
import sys
import unittest

try:
    import GolemQ  # noqa: F401
except ImportError:
    sys.path.insert(0, os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from GolemQ.models.alias import LTT
from GolemQ.pipeline.review.contract import (
    FAMILY_COLUMNS, HOUR_WEEKLY_COLUMNS, PERSIST_EXCLUDE,
    persist_check_columns)


class TestPersistCheckColumns(unittest.TestCase):

    def test_weekly_column_excluded_from_60min(self):
        """`4QPwrlnWek` 不在完成判据里 —— 它由回填工具写，不代表 review 跑过。"""
        self.assertIn(LTT.QUADRANT_POWERLINE_WEEKLY, HOUR_WEEKLY_COLUMNS)
        self.assertNotIn(LTT.QUADRANT_POWERLINE_WEEKLY,
                         persist_check_columns('60min'))

    def test_everything_else_kept(self):
        """除被排除的列外，契约列一个不少（别把判据悄悄放宽）。"""
        for fam, cols in FAMILY_COLUMNS.items():
            drop = set(PERSIST_EXCLUDE.get(fam, ()))
            want = tuple(c for c in cols if c not in drop)
            self.assertEqual(persist_check_columns(fam), want)
            self.assertEqual(len(persist_check_columns(fam)), len(cols) - len(drop))

    def test_other_families_untouched(self):
        self.assertEqual(persist_check_columns('day'),
                         tuple(FAMILY_COLUMNS['day']))
        self.assertEqual(persist_check_columns('15min'),
                         tuple(FAMILY_COLUMNS['15min']))

    def test_unknown_family_returns_empty(self):
        self.assertEqual(persist_check_columns('nope'), ())


if __name__ == '__main__':
    unittest.main()
