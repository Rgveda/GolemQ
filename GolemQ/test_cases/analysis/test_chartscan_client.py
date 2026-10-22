# coding:utf-8
"""`analysis/cv/client.py` 的**休市节流**（`_throttle`）。

2026-10-18 用户实测报：「Stock_CN 的周末休息 4 小时生效，ETF_CN 不生效」。

根因**不是** mode 判定（那 13 只 ETF 近一月一行数据都没有，判 mode=1 是对的），
而是**计数的分母/分子口径**：

| scope | 码数 | `build_jobs` 任务数 | **实际涉及码数** |
|:--|--:|--:|--:|
| `etf` | 506 | **429** | **13** |
| `fast` | 1672 | 142 | 14 |

`build_jobs` 会把 **mode=1 的码拆成 ~34 个月度任务**。按**任务数**算，
13 只被放大成 429 ⇒ `429/506 = 0.848 ≥ 0.832`（ETF 那条门槛）⇒ **不休**；
而 `fast` 的 `142/1672 = 0.085 < 0.168` ⇒ 照常休。

⇒ 分子改成 **`len({job[0] for job in code_job})`（按只去重）**。
旧树（`GolemAI/GolemQ/cli/__main__.py:355`）也是按任务数数的，这个放大同样存在。
"""
from __future__ import annotations

import os, sys
try:
    import GolemQ
except ImportError:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

import unittest
from unittest import mock

from GolemQ.analysis.cv import client


def _hours(jobs, total, scope, weekday=5, hour=10, tradetime=False):
    """跑一次 `_throttle`，返回挂起小时数（0 = 没挂）。"""
    with mock.patch.object(client, '_if_tradetime', return_value=tradetime), \
            mock.patch.object(client, 'dt') as fake, \
            mock.patch('time.sleep') as sleep:
        fake.now.return_value.weekday.return_value = weekday
        fake.now.return_value.hour = hour
        client._throttle(jobs, list(range(total)), scope, verbose=False)
    return sleep.call_args[0][0] / 3600 if sleep.called else 0


def _jobs(n_jobs, n_codes):
    """造 `n_jobs` 条任务，只落在 `n_codes` 只码上（模拟 mode=1 的月度切块）。"""
    return [(f'c{i % n_codes}', 'etf_cn', 1, 0, 0) for i in range(n_jobs)]


class TestThrottleCountsCodesNotJobs(unittest.TestCase):
    def test_etf_rests_when_only_a_few_codes_left(self):
        """⚠️ 核心回归：429 个任务但只有 13 只码 ⇒ 必须按 13/506 算，要休 4 小时。

        按**任务数**算时 429/506=0.848 ≥ etf 的 0.832 门槛 ⇒ 不休（就是用户报的现象）。
        """
        self.assertEqual(_hours(_jobs(429, 13), 506, 'etf'), 4)

    def test_fast_rests_too(self):
        """股票侧本来就休 —— 修完不能把它弄坏。"""
        self.assertEqual(_hours(_jobs(142, 14), 1672, 'fast'), 4)

    def test_nothing_left_no_rest(self):
        """全部码都有任务 ⇒ 不休（阈值 0.618 之上）。"""
        self.assertEqual(_hours(_jobs(506, 506), 506, 'etf'), 0)

    def test_trading_time_never_rests(self):
        self.assertEqual(_hours(_jobs(429, 13), 506, 'etf', tradetime=True), 0)

    def test_night_label_hours(self):
        """夜里（非周末）同一档是 2 小时。"""
        self.assertEqual(_hours(_jobs(429, 13), 506, 'etf', weekday=2, hour=23), 2)

    def test_in_trading_hours_label(self):
        """工作日盘中时段（非交易时段判据为 False）是 1 小时。"""
        self.assertEqual(_hours(_jobs(429, 13), 506, 'etf', weekday=2, hour=10), 1)
