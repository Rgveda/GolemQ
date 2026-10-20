#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys
try:
    import GolemQ
except ImportError:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

import time
import unittest
import numpy as np
import pandas as pd

from GolemQ.test_cases.old_tree import load_old

#: 老树**非 jit** 实现（`GolemQ_old/analysis/ChipDistribution.py`）——
#: 只作对拍/基准，不进生产。
_OLD_CHIP = load_old('analysis/ChipDistribution.py')

try:
    from GolemQ.analysis import ChipDistribution_jit as _MOD
    from GolemQ.analysis.ChipDistribution_jit import ChipDistribution
    _IMPORT_ERROR = None
except ImportError as _import_error:
    _MOD = None
    ChipDistribution = None
    _IMPORT_ERROR = _import_error


@unittest.skipIf(
    ChipDistribution is None,
    "ChipDistribution_jit 无法导入 —— 2026-10-10 搬运后本应能导入。"
    f"若这条又亮起来，先看依赖是否又被删了。（{_IMPORT_ERROR}）",
)
class TestChipDistributionPerformance(unittest.TestCase):
    """性能基准（原为命令行脚本，已纳入 unittest 以便记录依赖缺口）"""

    def test_performance(self):
        """测试优化前后的性能对比"""

        # 创建测试数据
        np.random.seed(42)
        test_data = {
            'date': pd.date_range('2024-01-01', periods=100),
            'open': np.random.uniform(10, 20, 100),
            'high': np.random.uniform(20, 30, 100),
            'low': np.random.uniform(5, 15, 100),
            'close': np.random.uniform(15, 25, 100),
            'volume': np.random.uniform(100000, 500000, 100),
            'TurnoverRate': np.random.uniform(0.01, 0.05, 100),
            'avg': np.random.uniform(15, 25, 100)
        }

        # 创建测试 DataFrame
        test_df = pd.DataFrame(test_data)

        # 创建 ChipDistribution 实例
        chip_dist = ChipDistribution()
        chip_dist.data = test_df

        print("开始性能测试...")

        # 测试 calcuChip 性能
        start_time = time.time()
        chip_dist.calcuChip(flag=1, AC=1)
        end_time = time.time()

        print(f"calcuChip 执行时间: {end_time - start_time:.4f} 秒")

        # ⚠️ 2026-10-10：原来这个用例**一条断言都没有**（它是个打印脚本）。
        # 搬运之后补上**不变量**，让它真的能失败 —— 否则「用例通过」无意义。
        # 首次调用的大头是 **numba 的一次性编译**，故不断言耗时（那是环境噪音）。
        self.assertGreater(len(chip_dist.ChipList), 0, 'calcuChip 应产出筹码分布序列')

        # 测试 winner 性能
        start_time = time.time()
        profit = chip_dist.winner()
        end_time = time.time()

        print(f"winner 执行时间: {end_time - start_time:.4f} 秒")
        print(f"获利盘比例结果长度: {len(profit)}")

        self.assertEqual(len(profit), len(test_df), '获利盘比例应与输入等长')
        profit = np.asarray(profit, dtype=float)
        self.assertTrue(np.all(profit >= 0), '获利盘比例不该为负')
        self.assertTrue(np.all(profit <= 1), '获利盘比例是比例，不该超过 1')

        # 测试 cost 性能
        start_time = time.time()
        cost_result = chip_dist.cost(90)
        end_time = time.time()

        print(f"cost 执行时间: {end_time - start_time:.4f} 秒")
        print(f"成本分布结果长度: {len(cost_result)}")

        self.assertEqual(len(cost_result), len(test_df), '成本分布应与输入等长')

        print("性能测试完成！")


class TestMissingTurnoverRaises(unittest.TestCase):
    """缺换手率 ⇒ **抛**，且点名 `fill_turnover`（`DECISIONS.md` D44）。

    为什么值得钉：这条链原先**自己补算**（`vol / 流通股本`），失败还是静默的
    （最末一支只 print 一行就继续）。而「恰好缺今天这一天」是**日常**情形
    （`--save` 先写 K 线、后写元数据），所以静默兜底等于让「数据还没到」永远不被发现。
    现在补齐挪到 `markets/StockCN/metadata_save.fill_turnover`，这里只负责**不装**。
    """

    N = 300

    def _data(self):
        ts = pd.date_range('2024-01-01', periods=self.N, freq='B')
        idx = pd.MultiIndex.from_arrays([ts, ['600519'] * self.N], names=['ts', 'code'])
        rng = np.random.default_rng(42)
        close = rng.uniform(10, 20, self.N)
        return pd.DataFrame({
            'open': close, 'high': close + 1, 'low': close - 1, 'close': close,
            'volume': rng.uniform(1e4, 5e4, self.N),
            'amount': close * 1e6, 'date': ts.strftime('%Y-%m-%d'),
        }, index=idx)

    def _run(self, meta):
        from unittest import mock
        chip = ChipDistribution()
        with mock.patch.object(_MOD, 'GQ_fetch_stock_reality_feat_day', return_value=meta):
            chip.get_data('600519', data=self._data(), verbose=False)
        return chip

    def _meta(self, days):
        idx = pd.MultiIndex.from_arrays([days, ['600519'] * len(days)],
                                        names=['date', 'code'])
        return pd.DataFrame({'turnover': [0.01] * len(days)}, index=idx)

    def test_missing_turnover_raises_and_names_the_fill_function(self):
        with self.assertRaises(ValueError) as ctx:
            self._run(self._meta([]))
        self.assertIn('fill_turnover', str(ctx.exception))

    def test_full_coverage_runs_to_completion(self):
        """全覆盖时一路跑通，并把 MA250 / MACD 三件套 / 滞回列都算出来。

        ⚠️ 这条用例在 2026-10-13 之前**写不出来** —— `FIELD.DIF` / `FIELD.DEA` /
        `FIELD.MA250_TIMING_LAG_MAJOR` 三个常量没定义（`PITFALLS.md` **P1** 的缺口），
        `get_data` 走到 `talib` 那段必抛 `AttributeError`。常量补齐后才跑得通。
        这也解释了那个缺口为何长期没暴露：进 `get_data` 的唯一路径
        `calc_stock_chip_distribution_jit` **全树无人调用**。
        """
        data = self._data()
        days = pd.to_datetime(data.index.get_level_values(0))
        chip = self._run(self._meta(days))
        for col in ('TurnoverRate', 'MA250', 'DIF', 'DEA', 'MACD', 'ma250LagMaj'):
            self.assertIn(col, chip.data.columns)

    def test_partial_coverage_still_raises(self):
        """只缺一天也抛 —— 不设「差几天才报」的余地。"""
        data = self._data()
        days = pd.to_datetime(data.index.get_level_values(0))[:-1]
        with self.assertRaises(ValueError):
            self._run(self._meta(days))


@unittest.skipIf(_OLD_CHIP is None, '老树 GolemQ_old/ 不在场，对拍跳过')
class TestMatchesOldNonJit(unittest.TestCase):
    """新 `ChipDistribution_jit` vs 老树**非 jit** `ChipDistribution.py`。

    实测（2026-10-21，60 天 / minD=0.01）：筹码字典逐键逐值一致（max diff 0），
    `_calcu_winner1/2` 差 ≤2.6e-15，`_calcu_cost` 完全一致；
    `_calculate_chip_distribution` **37.36 ms → 14.72 ms（2.5×）**。
    """

    def _inputs(self, n=60, seed=0):
        rng = np.random.default_rng(seed)
        dates = pd.date_range('2024-01-01', periods=n, freq='D')
        close = 100 + np.cumsum(rng.normal(0, 0.6, n))
        low, high = close - 0.5, close + 0.5
        avg = (high + low + close) / 2
        vol = rng.uniform(8e4, 2e5, n)
        turn = rng.uniform(0.005, 0.03, n)
        return dates, low, high, avg, vol, turn, close

    def _chips(self):
        dates, low, high, avg, vol, turn, close = self._inputs()
        old_chips = _OLD_CHIP._calculate_chip_distribution(
            dates, low, high, vol, turn, avg)
        new_chips = _MOD._calculate_chip_distribution(
            dates, low, high, vol, turn, avg)
        return dates, close, old_chips, new_chips

    def test_calculate_chip_distribution_matches(self):
        _, _, old_chips, new_chips = self._chips()
        self.assertEqual(list(old_chips[0]), list(new_chips[0]))
        for day in old_chips[0]:
            a, b = old_chips[0][day], new_chips[0][day]
            self.assertEqual(set(a), set(b), day)
            for price in a:
                self.assertAlmostEqual(a[price], b[price], places=12,
                                       msg=f'{day} {price}')

    def test_winner_and_cost_match(self):
        dates, close, old_chips, new_chips = self._chips()
        p = pd.Series(close, index=dates)

        a = _OLD_CHIP._calcu_winner1(p, old_chips[0])
        b = _MOD._calcu_winner1(p, new_chips[0])
        np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12, equal_nan=True)

        a = _OLD_CHIP._calcu_winner2(100.5, old_chips[0])
        b = _MOD._calcu_winner2(100.5, new_chips[0])
        np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12, equal_nan=True)

        a = _OLD_CHIP._calcu_cost(dates, 90, old_chips[0])
        b = _MOD._calcu_cost(dates, 90, new_chips[0])
        np.testing.assert_allclose(a, b, equal_nan=True)


def test_performance():
    """命令行直接运行入口（保持原脚本用法）"""
    TestChipDistributionPerformance("test_performance").test_performance()


if __name__ == "__main__":
    test_performance()
