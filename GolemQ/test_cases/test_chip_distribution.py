#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys
try:
    import GolemQ
except ImportError:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import time
import unittest
import numpy as np
import pandas as pd

try:
    from GolemQ.analysis.ChipDistribution_jit import ChipDistribution
    _IMPORT_ERROR = None
except ImportError as _import_error:
    ChipDistribution = None
    _IMPORT_ERROR = _import_error


@unittest.skipIf(
    ChipDistribution is None,
    "ChipDistribution_jit 无法导入：它依赖已被移除的 GolemQ.features.base "
    "(calc_feature_event_timing_lag)。模块本身存在于 analysis/ChipDistribution_jit.py。"
    f"（{_IMPORT_ERROR}）",
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

        # 测试 winner 性能
        start_time = time.time()
        profit = chip_dist.winner()
        end_time = time.time()

        print(f"winner 执行时间: {end_time - start_time:.4f} 秒")
        print(f"获利盘比例结果长度: {len(profit)}")

        # 测试 cost 性能
        start_time = time.time()
        cost_result = chip_dist.cost(90)
        end_time = time.time()

        print(f"cost 执行时间: {end_time - start_time:.4f} 秒")
        print(f"成本分布结果长度: {len(cost_result)}")

        print("性能测试完成！")


def test_performance():
    """命令行直接运行入口（保持原脚本用法）"""
    TestChipDistributionPerformance("test_performance").test_performance()


if __name__ == "__main__":
    test_performance()
