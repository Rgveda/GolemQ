#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""`tools/plot_mas_quad_pwrln.py`：Maj/Wek 统一输出的落库口径。

2026-11-23 用户要求：Wek、Maj **都落库**，并与 `MasPosOnhold` 输出到同一份
CSV + 同一张 PNG。本文件钉住两件承重的事：

1. `--apply` 必须**四条列一起写**（Wek Med/Mode + Maj Med/Mode），
   不能只写 Maj —— 早先 Maj 只进了 CSV，库里一列都没有；
2. 只写 massive **已存在的 15:00 行**，不插稀疏行（与周线工具同一约定）。
"""

import os
import sys

try:
    import GolemQ  # noqa: F401
except ImportError:
    sys.path.insert(0, os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', '..', '..')))

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..', 'tools')))

import plot_mas_quad_pwrln as PM  # noqa: E402


class _FakeCursor(list):
    def sort(self, *args, **kwargs):
        return self


class _FakeColl:
    def __init__(self, rows):
        self.rows = rows

    def find(self, query, projection):
        return _FakeCursor(self.rows)


class _FakeDb:
    def __init__(self, rows):
        self.stock_massive_reality_60min_models = _FakeColl(rows)


class TestApplyToMassive(unittest.TestCase):

    def _agg(self):
        idx = pd.to_datetime(['2026-11-18', '2026-11-19', '2026-11-20'])
        return pd.DataFrame({
            'MasQuadPwrlnWekMed': [-2.8, -3.8, -3.8],
            'MasQuadPwrlnWekMode': [-3.8, -3.8, -3.8],
            'MasQuadPwrlnMajMed': [-3.3, -3.8, np.nan],
            'MasQuadPwrlnMajMode': [-3.8, -3.8, np.nan],
            'n': [4000, 4000, 4000],
        }, index=idx)

    def test_writes_all_four_columns(self):
        """Wek 两条 + Maj 两条都在同一个 doc 里 —— 不许只写 Maj。"""
        rows = [{'datetime': '2026-11-18 15:00:00', 'time_stamp': 111},
                {'datetime': '2026-11-19 15:00:00', 'time_stamp': 222}]
        with patch('GolemQ.datasource.writer.upsert_fields',
                   return_value={'upserted': 0, 'modified': 2}) as up:
            PM.apply_to_massive(self._agg(), '2026-11-01', '2026-11-20',
                                db=_FakeDb(rows))
        docs = up.call_args[0][1]
        self.assertEqual(len(docs), 2)
        for d in docs:
            for c in PM.MAS_QUAD_COLUMNS:
                self.assertIn(c, d)
        self.assertAlmostEqual(docs[0]['MasQuadPwrlnWekMed'], -2.8)

    def test_only_existing_15min_rows_are_written(self):
        """11-20 在 massive 没有行 ⇒ 不写（不插稀疏行）。"""
        rows = [{'datetime': '2026-11-18 15:00:00', 'time_stamp': 111}]
        with patch('GolemQ.datasource.writer.upsert_fields',
                   return_value={'upserted': 0, 'modified': 1}) as up:
            PM.apply_to_massive(self._agg(), '2026-11-01', '2026-11-20',
                                db=_FakeDb(rows))
        docs = up.call_args[0][1]
        self.assertEqual([d['time_stamp'] for d in docs], [111])

    def test_nan_is_written_as_none(self):
        """NaN 写成 None（不写 float('nan') —— BSON 里那也是 NaN 值）。"""
        rows = [{'datetime': '2026-11-19 15:00:00', 'time_stamp': 222}]
        agg = self._agg().copy()
        agg.loc[pd.Timestamp('2026-11-19'), 'MasQuadPwrlnMajMed'] = np.nan
        with patch('GolemQ.datasource.writer.upsert_fields',
                   return_value={'upserted': 0, 'modified': 1}) as up:
            PM.apply_to_massive(agg, '2026-11-01', '2026-11-20',
                                db=_FakeDb(rows))
        doc = up.call_args[0][1][0]
        self.assertIsNone(doc['MasQuadPwrlnMajMed'])
        self.assertAlmostEqual(doc['MasQuadPwrlnWekMed'], -3.8)


class TestAggregate(unittest.TestCase):
    def test_median_and_mode(self):
        long = pd.DataFrame({'date': pd.to_datetime(['2026-11-19'] * 4),
                             'code': ['a', 'b', 'c', 'd'],
                             'pwrln': [-3.8, -3.8, -2.8, -2.8]})
        agg = PM.aggregate(long)
        self.assertAlmostEqual(float(agg['MasQuadPwrlnMajMed'].iloc[0]), -3.3)
        self.assertAlmostEqual(float(agg['MasQuadPwrlnMajMode'].iloc[0]), -3.8)


if __name__ == '__main__':
    unittest.main()
