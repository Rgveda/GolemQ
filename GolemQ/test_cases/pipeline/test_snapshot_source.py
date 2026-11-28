#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""快照契约的**逐码**源列（`_snapshot_source`）+ 合并后聚合（`snapshot_contracts`）。

背景（用户 2026-11-25）：`M9tTrkLag` 与 reality 列的 attach 必须在 **joblib 的
逐码 worker** 里做（旧树 `workload_p1` 的形状），跑完 `concat` 成
`ret_features_pd` 再一次性聚合 Med/Mode —— 不能放在父进程里全表扫
`stock_reality_feat_60min`（18.2M 行、无索引，实测 135s）。

本文件钉住：
* `SNAP_SRC_COLS` 覆盖 `SNAPSHOT_AGS` 用到的全部 source；
* `_snapshot_source` 的输出列 / `M9tTrkLag` 三列齐全守卫 / 全 NaN 行丢弃 /
  异常与缺窗口 ⇒ 空帧；
* `merge_snapshot_sources` + `snapshot_contracts` 与直接 `snapshot_frame` 等价。
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

from GolemQ.models.alias import LTT
from GolemQ.pipeline.base import CodeResult
from GolemQ.pipeline.mainstream.pipeline import (SNAP_SRC_COLS,
                                                 MainstreamPipeline)
from GolemQ.pipeline.mainstream.snapshot import SNAPSHOT_AGS, snapshot_frame

TS = pd.to_datetime(['2026-11-24 10:30', '2026-11-24 11:30',
                     '2026-11-24 14:00'])


def _reality_frame(code='600519', rows=None):
    """模拟 `fetch_reality_feat(family='60min')` 的返回形状。"""
    rows = rows or [
        # 8 列齐全
        dict(stage=1.0, poly9NorMajR=0.1, cost5PrzD=0.9, cost95PrzD=1.1,
             m9t=1.0, m9tLagMajR=2.0, m9tLagWekR=3.0, **{'4QPwrlnWek': 4.0}),
        # `m9tLagWekR` 缺 ⇒ `M9tTrkLag` 必须是 NaN（三列齐全守卫）
        dict(stage=1.0, poly9NorMajR=0.2, cost5PrzD=0.8, cost95PrzD=1.2,
             m9t=1.0, m9tLagMajR=2.0, m9tLagWekR=np.nan, **{'4QPwrlnWek': 4.2}),
        # 8 列全 NaN ⇒ 整行丢弃
        dict(stage=np.nan, poly9NorMajR=np.nan, cost5PrzD=np.nan,
             cost95PrzD=np.nan, m9t=np.nan, m9tLagMajR=np.nan,
             m9tLagWekR=np.nan, **{'4QPwrlnWek': np.nan}),
    ]
    df = pd.DataFrame(rows)
    df['code'] = code
    df['time_stamp'] = [int(t.timestamp()) for t in TS]
    df['datetime'] = TS.strftime('%Y-%m-%d %H:%M:%S')
    df['date'] = TS.normalize().strftime('%Y-%m-%d')
    return df.set_index([pd.to_datetime(df['datetime']), 'code'], drop=False)


class TestSnapSrcCols(unittest.TestCase):
    def test_derived_from_registry(self):
        """`SNAP_SRC_COLS` = 注册表的真实源列 ∪ M9T 三列（不能多、不能少）。"""
        need = {ag.source for ag in SNAPSHOT_AGS} - {'M9tTrkLag'}
        self.assertEqual(set(SNAP_SRC_COLS),
                         need | {'m9t', 'm9tLagMajR', 'm9tLagWekR'})


class _FakeMarket:
    """只提供 `get_database()`（`fetch_reality_feat` 已 patch 掉，不真读库）。"""

    def get_database(self):
        return None


class TestSnapshotSource(unittest.TestCase):
    def _call(self, fetched, start='2026-11-01', end='2026-11-25'):
        pipe = MainstreamPipeline()
        with patch('GolemQ.get_active_market', return_value=_FakeMarket()), \
                patch('GolemQ.markets.StockCN.reality_save.fetch_reality_feat',
                      return_value=fetched):
            return pipe._snapshot_source('600519', start, end)

    def test_columns_and_combo(self):
        out = self._call(_reality_frame())
        self.assertEqual(list(out.columns),
                         ['ts', *SNAP_SRC_COLS,
                          LTT.MAGIC_NINE_TURNS_TRUNK_TIMING_LAG])
        self.assertEqual(len(out), 2, '8 列全 NaN 的行必须丢弃')
        self.assertTrue(np.isfinite(out['M9tTrkLag'].iloc[0]))
        self.assertTrue(np.isnan(out['M9tTrkLag'].iloc[1]),
                        '三列不齐全 ⇒ M9tTrkLag 必须 NaN（不许当成中性 0）')
        self.assertEqual(pd.Timestamp(out['ts'].iloc[0]), TS[0])

    def test_schema_is_fixed(self):
        """缺某个源列（如 `4QPwrlnWek`）也要补齐成固定模式（NaN），
        否则 `snapshot_frame` 的源列会随码变化。"""
        df = _reality_frame()
        out = self._call(df.drop(columns=['4QPwrlnWek']))
        self.assertEqual(list(out.columns),
                         ['ts', *SNAP_SRC_COLS,
                          LTT.MAGIC_NINE_TURNS_TRUNK_TIMING_LAG])
        self.assertTrue(out['4QPwrlnWek'].isna().all())

    def test_window_slice_keeps_warmup(self):
        """只回 `start..end` 窗口内的 bar（暖机读全历史，但不出窗口）。"""
        out = self._call(_reality_frame(),
                         start='2026-11-24 11:00', end='2026-11-25')
        # 第 3 行是「8 列全 NaN」⇒ 被丢弃，只剩 11:30
        self.assertEqual(list(pd.to_datetime(out['ts'])), [TS[1]])

    def test_none_result_is_empty(self):
        out = self._call(None)
        self.assertTrue(out.empty)
        self.assertIn(LTT.MAGIC_NINE_TURNS_TRUNK_TIMING_LAG, out.columns)

    def test_fetch_error_is_empty(self):
        pipe = MainstreamPipeline()
        with patch('GolemQ.get_active_market', return_value=_FakeMarket()), \
                patch('GolemQ.markets.StockCN.reality_save.fetch_reality_feat',
                      side_effect=RuntimeError('boom')):
            out = pipe._snapshot_source('600519', '2026-11-01', '2026-11-25')
        self.assertTrue(out.empty)

    def test_missing_window_is_empty(self):
        pipe = MainstreamPipeline()
        self.assertTrue(pipe._snapshot_source('600519').empty)


class TestMergeAndAggregate(unittest.TestCase):
    def _src(self, code):
        out = self._fallback_src(code)
        return CodeResult(code=code, status='ok', result={'snap_src': out})

    @staticmethod
    def _fallback_src(code):
        rows = []
        for i, t in enumerate(TS):
            rows.append({'ts': t, 'stage': 1.0 if i < 2 else 2.0,
                         'poly9NorMajR': 0.1 * (i + 1),
                         'cost5PrzD': 1.1, 'cost95PrzD': 0.9,
                         'm9t': 1.0, 'm9tLagMajR': -1.0, 'm9tLagWekR': 2.0,
                         '4QPwrlnWek': 3.0 + i, 'M9tTrkLag': float(i - 1)})
        return pd.DataFrame(rows)

    def test_merge_counts_codes(self):
        merged, n = MainstreamPipeline.merge_snapshot_sources(
            [self._src('600519'), self._src('000711')])
        self.assertEqual(n, 2)
        self.assertEqual(len(merged), 2 * len(TS))
        self.assertNotIn('code', merged.columns,
                         '`snapshot_frame` 只按 ts 分组 —— 逐行 code 字符串白吃内存')

    def test_merge_keeps_no_data_codes(self):
        """`no_data` / 报错的码也可能有 reality 源列（旧路径是全表聚合）。"""
        ok = self._src('600519')
        bad = CodeResult(code='000711', status='no_data',
                         result={'snap_src': self._fallback_src('000711')})
        err = CodeResult(code='603366', status='error', error='boom',
                         result={'snap_src': self._fallback_src('603366')})
        merged, n = MainstreamPipeline.merge_snapshot_sources([ok, bad, err])
        self.assertEqual(n, 3)
        self.assertEqual(len(merged), 3 * len(TS))

    def test_contracts_match_direct_aggregation(self):
        merged, _n = MainstreamPipeline.merge_snapshot_sources(
            [self._src('600519'), self._src('000711')])
        snap = MainstreamPipeline().snapshot_contracts(merged)
        ref = snapshot_frame(merged, ts_col='ts')
        self.assertEqual(len(snap), len(ref))
        extra = set(snap.columns) - set(ref.columns)
        self.assertEqual(extra, {'MAGIC_NINE_TURNS_TIMING_LAG_COMPOUNDED'})
        pd.testing.assert_frame_equal(snap[list(ref.columns)], ref)

    def test_empty_input(self):
        self.assertTrue(
            MainstreamPipeline().snapshot_contracts(pd.DataFrame()).empty)
        self.assertTrue(MainstreamPipeline().snapshot_contracts(None).empty)


if __name__ == '__main__':
    unittest.main()
