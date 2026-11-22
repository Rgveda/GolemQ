#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""`Statistics` 的「Slowest 1%」段（用户 2026-11-21 要求）。

背景：`--strategy review` 在 36 核上跑 1,670 个码，墙钟 3h35m，但绝大多数码
是秒级 —— 时间被极少数码吃掉，光看总时长看不出是谁。要求把**耗时最高的
1%** 列出来，并带上分阶段耗时（`CodeResult.meta`），才能判断该短路还是该查。

这里只钉三件事：

1. 头部那行报的「n/总数 + 占比」用的是 **sum(各码耗时)** 而不是墙钟
   （并行下 sum 远大于墙钟，占比要按 sum 算才对得上「谁吃掉了计算量」）；
2. 慢码逐个列出，非 ok 的带状态标记；
3. `meta` 里的阶段耗时会附在行尾 —— 没有 meta 时不能打空括号。

⚠️ 全部内存构造，不跑 joblib、不碰 DB。
"""

import io
import os
import sys
import unittest
from contextlib import redirect_stdout
from datetime import datetime, timedelta

try:
    import GolemQ  # noqa: F401
except ImportError:
    sys.path.insert(0, os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from GolemQ.pipeline.base import BasePipeline, CodeResult


class _FakePipeline(BasePipeline):
    def __init__(self, results):
        super().__init__(name='fakepipe')
        self.results = results
        self.total_count = len(results)
        self.success_count = sum(1 for r in results if r.ok)
        self.error_count = len(results) - self.success_count
        self.start_time = datetime(2026, 11, 21, 21, 58, 20)
        self.end_time = self.start_time + timedelta(hours=3, minutes=35)

    def run_code(self, code, **kwargs):        # pragma: no cover - 不调用
        raise NotImplementedError


def _capture(pipe) -> str:
    buf = io.StringIO()
    with redirect_stdout(buf):
        pipe._print_statistics()
    return buf.getvalue()


class TestSlowestSection(unittest.TestCase):

    def test_top_one_percent_listed_with_share(self):
        res = [CodeResult(code=f'{i:06d}', duration=1.0) for i in range(199)]
        res.append(CodeResult(code='999999', duration=900.0))
        pipe = _FakePipeline(res)          # 200 码 ⇒ 1% = 2 个

        out = _capture(pipe)

        self.assertIn('Slowest 1%: 2/200 codes', out)
        self.assertIn('999999', out)
        # 前两名 900 + 1 = 901 / 1099 ≈ 82.0%
        self.assertIn('82.0%', out)

    def test_timeout_and_meta_marked(self):
        res = [
            CodeResult(code='600519', duration=820.0,
                       meta={'fetch': 12.3, 'steps': 800.0, 'write': 7.7}),
            CodeResult(code='688520', duration=1800.0, status='timeout'),
        ] + [CodeResult(code=f'{i:06d}', duration=2.0) for i in range(198)]
        pipe = _FakePipeline(res)

        out = _capture(pipe)

        self.assertIn('688520  1800.0s  [timeout]', out)
        self.assertIn('600519  820.0s  (fetch 12.3s / steps 800.0s / write 7.7s)',
                      out)

    def test_no_meta_no_empty_parens(self):
        res = [CodeResult(code='000711', duration=5.0)]
        pipe = _FakePipeline(res)

        out = _capture(pipe)

        self.assertIn('000711  5.0s', out)
        self.assertNotIn('()', out)

    def test_empty_results_does_not_raise(self):
        pipe = _FakePipeline([])
        out = _capture(pipe)
        self.assertIn('Statistics:', out)
        self.assertNotIn('Slowest', out)


if __name__ == '__main__':
    unittest.main()
