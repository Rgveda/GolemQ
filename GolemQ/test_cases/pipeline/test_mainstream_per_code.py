#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Mainstream 并行执行的**契约**：一码一个任务，进度条 step == 代码数。

背景（用户 2026-11-12 明确）：Mainstream 曾经把 5,582 个码切成 56 个
`CHUNK_SIZE=100` 的分片并行 —— 分片会让一个慢码拖住整片，tqdm 也只按
100 码一跳。用户要求**不分片、逐码任务**，并与 compact / review 共用
`BasePipeline.run_codes`（同一套进度条/超时/统计）。

这两条是承重的：

1. `run_market` 对**每个码**各调一次 `code_masks`（不合并成分片）；
2. 进度条的 iterable 就是**代码列表本身** —— `len(iterable) == len(codes)`、
   逐元素是 6 位代码。`tqdm_joblib` 的完成回调按这个 iterable 计步，
   iterable 若变成"分片列表"，进度就按分片跳而不是按码跳。

⚠️ 全部用桩，不碰 DB / 网络 / joblib 真并行。
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

from GolemQ.pipeline.base import BasePipeline, CodeResult
from GolemQ.pipeline.mainstream.pipeline import MainstreamPipeline


class _ProgressRecorder:
    """假的 tqdm：记录它拿到的 iterable 与每一步的 update。"""

    def __init__(self, iterable=None, **kwargs):
        self.iterable = None if iterable is None else list(iterable)
        self.kwargs = kwargs
        self.updates = []
        self.n = 0

    def update(self, n=1):
        self.updates.append(n)
        self.n += n

    def set_description(self, *args, **kwargs):
        pass

    def close(self):
        pass

    def __iter__(self):
        return iter(self.iterable or [])


class _StubMainstream(MainstreamPipeline):
    """替掉所有 I/O：固定时间轴 + 每码一小段掩码。"""

    def __init__(self, timeline, masks):
        super().__init__(lookback_days=1)
        self._timeline = timeline
        self._masks = masks
        self.called = []

    def _setup_cpu_affinity(self):
        pass

    def market_timeline(self, as_of):
        return self._timeline

    def code_masks(self, code, as_of):
        self.called.append(code)                 # ← 记录每个码被单独调用一次
        return self._masks.get(code)

    def snapshot_contracts(self):
        return None

    def _index_close_map(self, code, timeline):
        return np.full(len(timeline), np.nan)


class TestMainstreamPerCode(unittest.TestCase):
    TIMELINE = np.array(pd.to_datetime([
        '2026-11-10 10:30', '2026-11-10 11:30',
        '2026-11-10 14:00', '2026-11-10 15:00',
    ]).values, dtype='datetime64[ns]')

    def _masks(self):
        ts = self.TIMELINE
        bull = np.array([True, False, True, False])
        bear = np.array([False, True, False, True])
        return ts, bull, bear

    def test_code_masks_called_once_per_code(self):
        """**不分片**：每个码各调一次，调用序列与代码列表逐一对应。"""
        codes = ['600000', '600001', '600002', '600003', '600004']
        stub = _StubMainstream(self.TIMELINE, {c: self._masks() for c in codes})

        stub.run_market(codes, pool_size=0)

        self.assertEqual(stub.called, codes,
                         'run_market 必须逐码调用（一个码一个任务），不能合并成分片')

    def test_result_count_matches_code_count(self):
        """结果条数 == 代码数（再次钉住"没有分片聚合"）。"""
        codes = ['000001', '000002', '000003']
        stub = _StubMainstream(self.TIMELINE, {c: self._masks() for c in codes})

        df = stub.run_market(codes, pool_size=0)

        self.assertEqual(stub.total_count, len(codes))
        self.assertEqual(stub.success_count, len(codes))
        self.assertIsNotNone(df)


class _EchoPipeline(BasePipeline):
    """最小的 BasePipeline 子类：`run_code` 立即返回。"""

    def __init__(self):
        super().__init__(name='echo')

    def _setup_cpu_affinity(self):
        pass

    def run_code(self, code, **kwargs):
        return CodeResult(code=code, status='ok', result=1)


class TestProgressStepsEqualCodeCount(unittest.TestCase):
    def test_tqdm_iterable_is_the_code_list(self):
        """进度条的 iterable 就是代码列表：`len(iterable) == len(codes)`。"""
        captured = []

        def _fake_tqdm(iterable=None, **kwargs):
            rec = _ProgressRecorder(iterable, **kwargs)
            captured.append(rec)
            return rec

        codes = ['600000', '600001', '600002', '600003']
        pipe = _EchoPipeline()
        with patch('GolemQ.pipeline.base.tqdm', _fake_tqdm):
            results, stats = pipe.run_codes(codes, pool_size=0, verbose=False)

        self.assertEqual(len(results), len(codes))
        self.assertTrue(captured, 'run_codes 必须给进度条（长任务不能无声跑）')
        rec = captured[0]
        self.assertEqual(rec.iterable, codes,
                         '进度条必须逐码推进；若拿到的是分片列表，step 就不等于代码数')
        self.assertEqual(len(rec.iterable), len(codes))

    def test_no_chunking_left_in_mainstream(self):
        """回归闸：Mainstream 里不允许再出现分片常量/函数。"""
        import inspect
        src = inspect.getsource(sys.modules[MainstreamPipeline.__module__])
        self.assertNotIn('CHUNK_SIZE', src)
        self.assertNotIn('_chunk_counts', src)


if __name__ == '__main__':
    unittest.main()
