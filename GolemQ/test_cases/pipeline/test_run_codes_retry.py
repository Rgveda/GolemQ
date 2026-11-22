#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""`BasePipeline.run_codes`：**不分批、tqdm 逐码**，并行失败只重试一轮（2026-11-22）。

用户硬要求（原话「这点没得商量」）：

1. **不得把 codelist 切批** —— 分片会让进度按片跳，也会让一个慢码拖住整片；
2. **tqdm 必须逐 code** —— `tqdm` 的 iterable 就是完整 codelist，
   且 `Parallel(batch_size=1)`，这样 `tqdm_joblib` 的完成回调**每码**触发一次。

起因：2026-11-22 实测 1,742/2,001 时一个 loky worker 因内存压力被杀，
旧代码 `except Exception` 把这 2,001 个码**降级成串行重跑**（进度条从
8.63 stock/s 重新走一遍），而串行跑在父进程里、numba 编译又失败 ⇒
RegTree 回落纯 Python。现在改成「原样并行重试一次」，仍失败才退串行兜底。

⚠️ 全部用桩（假 `Parallel` / `tqdm`），不真并行、不碰 DB。
"""

import io
import os
import sys
import unittest
from contextlib import contextmanager, redirect_stderr
from unittest.mock import patch

try:
    import GolemQ  # noqa: F401
except ImportError:
    sys.path.insert(0, os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from GolemQ.pipeline.base import BasePipeline, CodeResult


class _Recorder(BasePipeline):
    def __init__(self):
        super().__init__(name='retry')
        self.seen = []

    def _setup_cpu_affinity(self):
        pass

    def run_code(self, code, **kwargs):
        self.seen.append(code)
        return CodeResult(code=code, status='ok', result=1)


@contextmanager
def _noop_tqdm_joblib(pbar, label):
    yield


class _TqdmStub:
    """记录拿到的 iterable 与 n_jobs / batch_size。"""
    inst = []

    def __init__(self, iterable=None, **kwargs):
        self.iterable = None if iterable is None else list(iterable)
        self.kwargs = kwargs
        self.n = 0
        _TqdmStub.inst.append(self)

    def update(self, n=1):
        self.n += n

    def set_description(self, *a, **k):
        pass

    def close(self):
        pass

    def __iter__(self):
        return iter(self.iterable or [])


def _passthrough_tqdm(iterable=None, **kwargs):
    return iterable if iterable is not None else []


def _make_parallel(fail_times=0, calls=None):
    """假 `Parallel`：按顺序执行 delayed 元组；`fail_times` 次调用后先抛异常。"""
    state = {'n': 0, 'failed': 0}

    class _P:
        def __init__(self, n_jobs=None, batch_size=None):
            if calls is not None:
                calls.append({'n_jobs': n_jobs, 'batch_size': batch_size})
            self.batch_size = batch_size

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def __call__(self, gen):
            state['n'] += 1
            items = list(gen)
            if state['failed'] < fail_times:
                state['failed'] += 1
                raise RuntimeError('worker died (stub)')
            return [f(*a, **k) for f, a, k in items]
    return _P


class TestNoChunkingRetry(unittest.TestCase):

    def setUp(self):
        _TqdmStub.inst.clear()

    def _run(self, n_codes, fail_times=0):
        pipe = _Recorder()
        codes = [f'{i:06d}' for i in range(n_codes)]
        calls = []
        par = _make_parallel(fail_times, calls)
        buf = io.StringIO()
        with patch('GolemQ.pipeline.base.Parallel', par), \
                patch('GolemQ.pipeline.base.tqdm', _TqdmStub), \
                patch('GolemQ.pipeline.base.tqdm_joblib', _noop_tqdm_joblib), \
                redirect_stderr(buf):
            results, _ = pipe.run_codes(codes, pool_size=2, verbose=False)
        return pipe, codes, results, calls, buf.getvalue()

    def test_tqdm_iterable_is_the_whole_codelist(self):
        """**不得切批**：进度条拿到的 iterable 必须是完整、按原顺序的码表。"""
        _, codes, _, _, _ = self._run(600)
        self.assertTrue(_TqdmStub.inst)
        self.assertEqual(_TqdmStub.inst[0].iterable, codes)

    def test_parallel_uses_batch_size_one(self):
        """**tqdm 逐码**：`Parallel` 必须 `batch_size=1`（默认 auto 会按批跳）。"""
        _, _, _, calls, _ = self._run(50)
        self.assertTrue(calls)
        self.assertTrue(all(c['batch_size'] == 1 for c in calls),
                        f'必须逐码：{calls}')

    def test_each_code_once_on_success(self):
        pipe, codes, results, calls, _ = self._run(50)
        self.assertEqual(pipe.seen, codes)
        self.assertEqual(len(results), len(codes))
        self.assertEqual(len(calls), 1, '成功时只跑一轮')

    def test_one_parallel_retry_on_worker_death(self):
        """worker 崩掉 ⇒ **原样并行重试一次**（不是降级串行）。"""
        pipe, codes, results, calls, err = self._run(50, fail_times=1)
        self.assertEqual(len(calls), 2, '一次失败 + 一次并行重试')
        self.assertIn('[WARN]', err)
        self.assertIn('并行重试', err)
        self.assertEqual(len(results), len(codes))
        self.assertEqual(sorted(pipe.seen), sorted(codes),
                         '重试不丢码（完成的码会因缺口为空而变快）')

    def test_serial_fallback_only_after_retry_exhausted(self):
        pipe, codes, results, calls, err = self._run(20, fail_times=5)
        self.assertEqual(len(calls), 2, '只并行试 2 次，然后串行兜底')
        self.assertIn('退回**串行**兜底', err)
        self.assertEqual(len(results), len(codes))


if __name__ == '__main__':
    unittest.main()
