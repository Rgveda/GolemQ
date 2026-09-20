# coding:utf-8
#
# The MIT License (MIT)
#
# Copyright (c) 2018-2020 azai/Rgveda/GolemQuant
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
Mainstream Benchmark 子类
"""

from .base import BaseBenchmark
import pandas as pd
from typing import Optional

try:
    from GolemQ.models.mainstream import calc_stock_mainstream_analysis
    MAINSTREAM_AVAILABLE = True
except Exception:
    MAINSTREAM_AVAILABLE = False
    print("Warning: Cannot import calc_stock_mainstream_analysis from GolemQ.models.mainstream")


class MainstreamBenchmark(BaseBenchmark):
    """
    Mainstream Benchmark 实现
    """
    
    def __init__(self, verbose: bool = False, market_type=None):
        """
        初始化Mainstream Benchmark
        
        Args:
            verbose: 是否显示详细输出
            market_type: 市场类型
        """
        super().__init__(
            benchmark_name="MainstreamBenchmark",
            verbose=verbose,
            market_type=market_type
        )
        
    def calculate(self, code: str, **kwargs) -> Optional[pd.DataFrame]:
        """
        计算mainstream指标
        
        Args:
            code: 股票代码
            **kwargs: 其他参数，包括kline_data
            
        Returns:
            计算结果DataFrame或None
        """
        if not MAINSTREAM_AVAILABLE:
            print(f"calc_stock_mainstream_analysis not available for {code}")
            return None
            
        kline_data = kwargs.get('kline_data')
        if kline_data is None or len(kline_data) == 0:
            if self.verbose:
                print(f"No kline data for {code}")
            return None
            
        try:
            # 调用mainstream.py中的核心函数
            # 注意：calc_stock_mainstream_analysis可能需要不同的参数
            # 这里需要根据实际情况调整
            result = calc_stock_mainstream_analysis(
                code,
                kline_data,
                **{k: v for k, v in kwargs.items() if k not in ['kline_data', 'code']}
            )
            return result
        except Exception as e:
            self._handle_error(code, e, kline_data, increment_count=False)
            return None


# 兼容性函数
def calc_workload_agent(*args, **kwargs):
    """
    兼容性函数，保持与原有API一致
    
    注意：这个函数创建MainstreamBenchmark实例并调用calc_workload_agent方法
    """
    benchmark = MainstreamBenchmark(
        verbose=kwargs.get('verbose', False),
        market_type=kwargs.get('market_type', None)
    )
    
    # 从kwargs中移除benchmark特定的参数
    benchmark_kwargs = {k: v for k, v in kwargs.items() 
                        if k not in ['verbose', 'market_type']}
    
    return benchmark.calc_workload_agent(*args, **benchmark_kwargs)