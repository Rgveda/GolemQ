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
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
Benchmark 基类
为所有 benchmark 模块提供公共功能
"""

from datetime import datetime as dt, timedelta
import warnings
import pandas as pd
from tqdm import tqdm
import traceback
import func_timeout
from abc import ABC, abstractmethod
from typing import List, Optional, Union, Dict, Any

try:
    from joblib import Parallel, delayed
    JOBLIB_AVAILABLE = True
except Exception:
    print('joblib not installed.')
    JOBLIB_AVAILABLE = False

from GolemQ.core.constants import MARKET_TYPE

from GolemQ.markets.StockCN.date_utils import GQ_util_if_tradetime

from GolemQ.core.base import set_cpu_affinity_even
from GolemQ.core.presentation import tqdm_joblib
from GolemQ.markets.StockCN.realtime import GQ_fetch_stock_min_realtime_adv
from GolemQ.markets.StockCN.fetch import (
    GQ_fetch_stock_min_adv,
    GQ_fetch_index_min_adv,
)
from multiprocessing import shared_memory
from GolemQ.markets.StockCN.date_utils import (
    GQ_util_get_last_day
)
from GolemQ.markets.StockCN.symbol import (
    normalize_code
)


warnings.simplefilter(action='ignore', category=FutureWarning)


class BaseBenchmark(ABC):
    """
    Benchmark 基类
    提供所有 benchmark 模块的公共功能
    """
    
    def __init__(self, 
                 benchmark_name: str = "BaseBenchmark",
                 verbose: bool = False,
                 market_type: Any = None):
        """
        初始化基类
        
        Args:
            benchmark_name: benchmark 名称
            verbose: 是否显示详细输出
            market_type: 市场类型，默认为 `MARKET_TYPE.STOCK_CN`
        """
        self.benchmark_name = benchmark_name
        self.verbose = verbose
        self.market_type = market_type
        
        if self.market_type is None:
            self.market_type = MARKET_TYPE.STOCK_CN
            
        self.error_count = 0
        self.success_count = 0
        self.total_count = 0
        self.start_time = None
        self.end_time = None
        
    def _setup_cpu_affinity(self):
        """设置CPU亲和性"""
        set_cpu_affinity_even()
        
    def _get_date_range(self, offset: str = '', days: int = 960):
        """
        获取日期范围
        
        Args:
            offset: 偏移日期字符串
            days: 回溯天数
            
        Returns:
            (start_date, end_date)
        """
        end_date = pd.to_datetime(GQ_util_get_last_day()) if (len(offset) == 0) else pd.to_datetime(offset)
        start_date = end_date - timedelta(days=days)
        return start_date, end_date
    
    def _fetch_kline_data(self,
                          code: str,
                          start_date: dt,
                          end_date: dt,
                          frequency: str = '60min',
                          fetch_realtime: bool = True) -> Optional[pd.DataFrame]:
        """
        获取K线数据
        
        Args:
            code: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            frequency: 频率，默认60min
            fetch_realtime: 是否获取实盘数据
            
        Returns:
            K线数据DataFrame或None
        """
        try:
            if self.market_type == MARKET_TYPE.STOCK_CN:
                kline_data = GQ_fetch_stock_min_adv(
                    code,
                    start_date.strftime("%Y-%m-%d %H:%M:%S"),
                    (end_date + timedelta(hours=17)).strftime("%Y-%m-%d %H:%M:%S"),
                    frequence=frequency
                )
            elif self.market_type == MARKET_TYPE.INDEX_CN:
                # 指数与 ETF 共用 index_* 集合，本函数按代码自动选容器；
                # 保留独立名字只为让调用点读起来与市场分支对应。
                kline_data = GQ_fetch_index_min_adv(
                    code,
                    start_date.strftime("%Y-%m-%d %H:%M:%S"),
                    (end_date + timedelta(hours=17)).strftime("%Y-%m-%d %H:%M:%S"),
                    frequence=frequency
                )
            else:
                print(f"Unsupported market type: {self.market_type}")
                return None
                
            # 获取实盘数据
            if fetch_realtime and kline_data is not None:
                if (GQ_util_if_tradetime(dt.now()) or
                        ((pd.to_datetime(GQ_util_get_last_day()) - end_date) < timedelta(hours=18))):
                    kline_data = GQ_fetch_stock_min_realtime_adv(
                        normalize_code(code, market_type=self.market_type),
                        kline_data,
                        frequency=frequency,
                    )
                    
            return kline_data.data if kline_data is not None else None
            
        except Exception as e:
            print(f"Error fetching kline data for {code}: {e}")
            return None
    
    def _check_suspension(self, kline_data: pd.DataFrame, code: str) -> bool:
        """
        检查是否停牌
        
        Args:
            kline_data: K线数据
            code: 股票代码
            
        Returns:
            是否停牌
        """
        if kline_data is None or len(kline_data) == 0:
            return True
            
        last_date = kline_data.tail(1).index.get_level_values(level=0)[0]
        current_date = pd.to_datetime(GQ_util_get_last_day())
        
        # 如果最后交易日在84天内，可能是停牌
        if timedelta(days=0) < (current_date - last_date) < timedelta(days=84):
            if self.verbose:
                print(f"Code {code} may be suspended. Last trade: {last_date}")
            return True
            
        return False
    
    def _format_error_message(self,
                              code: str,
                              error_msg: str,
                              kline_data: Optional[pd.DataFrame] = None) -> str:
        """
        格式化错误信息
        
        Args:
            code: 股票代码
            error_msg: 错误信息
            kline_data: K线数据
            
        Returns:
            格式化后的错误信息
        """
        if kline_data is not None:
            return f"{error_msg} Code: {code}, Length: {len(kline_data)}"
        else:
            return f"{error_msg} Code: {code}, No kline data"
    
    def _handle_error(self,
                      code: str,
                      error: Exception,
                      kline_data: Optional[pd.DataFrame] = None,
                      increment_count: bool = True):
        """
        处理错误
        
        Args:
            code: 股票代码
            error: 异常对象
            kline_data: K线数据
            increment_count: 是否增加错误计数
        """
        # 格式化错误信息（用于调试，但当前不直接使用）
        _ = self._format_error_message(code, str(error), kline_data)
        
        # 检查是否为KeyError，KeyError可能不需要特殊处理
        if "KeyError" in str(type(error)):
            if self.verbose:
                print(f"KeyError for {code}: {error}")
        else:
            print(f"Error for {code}: {error}")
            traceback.print_exc()
            
        if increment_count:
            self.error_count += 1
            
    @abstractmethod
    def calculate(self, code: str, **kwargs) -> Optional[pd.DataFrame]:
        """
        计算核心指标（抽象方法，子类必须实现）
        
        Args:
            code: 股票代码
            **kwargs: 其他参数
            
        Returns:
            计算结果DataFrame或None
        """
        pass
    
    def calc_workload_agent(self,
                            codelist: Union[str, List[str]],
                            portfolio_batch: str = '',
                            verbose: bool = False,
                            offset: str = '',
                            massive_trend=None,
                            massive_model_major=None,
                            eval_range: str = 'fast',
                            pool_size: int = 0,
                            markup: bool = False,
                            ret_meta: bool = False,
                            **kwargs) -> Union[None, pd.DataFrame, tuple]:
        """
        计算负载代理，统计计算成功率，统计计算时间等数据
        
        Args:
            codelist: 股票代码列表或单个代码
            portfolio_batch: 投资组合批次
            verbose: 是否显示详细输出
            offset: 偏移日期
            massive_trend: 大规模趋势数据
            massive_model_major: 主要模型
            eval_range: 评估范围
            pool_size: 线程池大小
            markup: 是否标记
            ret_meta: 是否返回元数据
            **kwargs: 其他参数
            
        Returns:
            计算结果
        """
        self._setup_cpu_affinity()
        self.verbose = verbose
        
        # 记录开始时间
        self.start_time = dt.now()
        
        # 处理代码列表
        if isinstance(codelist, str):
            codelist = [codelist]
            
        self.total_count = len(codelist)
        self.error_count = 0
        self.success_count = 0
        
        results = []
        
        # 使用并行处理
        if pool_size > 0 and JOBLIB_AVAILABLE and len(codelist) > 1:
            if self.verbose:
                print(f"Using parallel processing with {pool_size} workers")
                
            try:
                # 使用joblib并行处理
                with tqdm_joblib(tqdm(desc=f"{self.benchmark_name} Processing", total=len(codelist))):
                    results = Parallel(n_jobs=pool_size)(
                        delayed(self._process_single_code)(
                            code, portfolio_batch, offset, massive_trend, 
                            massive_model_major, eval_range, markup, ret_meta, **kwargs
                        ) for code in codelist
                    )
            except Exception as e:
                print(f"Parallel processing error: {e}")
                # 回退到串行处理
                results = []
                for code in tqdm(codelist, desc=f"{self.benchmark_name} Processing"):
                    result = self._process_single_code(
                        code, portfolio_batch, offset, massive_trend,
                        massive_model_major, eval_range, markup, ret_meta, **kwargs
                    )
                    results.append(result)
        else:
            # 串行处理
            for code in tqdm(codelist, desc=f"{self.benchmark_name} Processing"):
                result = self._process_single_code(
                    code, portfolio_batch, offset, massive_trend,
                    massive_model_major, eval_range, markup, ret_meta, **kwargs
                )
                results.append(result)
        
        # 过滤None结果
        valid_results = [r for r in results if r is not None]
        self.success_count = len(valid_results)
        
        # 记录结束时间
        self.end_time = dt.now()
        
        # 打印统计信息
        self._print_statistics()
        
        # 返回结果
        if ret_meta:
            return valid_results, self._get_statistics_dict()
        elif len(valid_results) == 1:
            return valid_results[0]
        else:
            return valid_results
    
    def _process_single_code(self,
                            code: str,
                            portfolio_batch: str,
                            offset: str,
                            massive_trend,
                            massive_model_major,
                            eval_range: str,
                            markup: bool,
                            ret_meta: bool,
                            **kwargs):
        """
        处理单个股票代码
        
        Args:
            code: 股票代码
            **kwargs: 其他参数
            
        Returns:
            处理结果或None
        """
        try:
            # 获取K线数据
            start_date, end_date = self._get_date_range(offset)
            kline_data = self._fetch_kline_data(code, start_date, end_date)
            
            if kline_data is None or len(kline_data) == 0:
                if self.verbose:
                    print(f"No kline data for {code}")
                return None
                
            # 检查是否停牌
            if self._check_suspension(kline_data, code):
                if self.verbose:
                    print(f"Code {code} may be suspended, skipping")
                return None
                
            # 调用子类的calculate方法
            result = self.calculate(code, kline_data=kline_data, **kwargs)
            
            if result is not None:
                self.success_count += 1
                
            return result
            
        except func_timeout.exceptions.FunctionTimedOut:
            print(f"Code {code}: Function timed out")
            self.error_count += 1
            return None
        except Exception as e:
            self._handle_error(code, e)
            return None
    
    def _print_statistics(self):
        """打印统计信息"""
        if self.start_time and self.end_time:
            duration = self.end_time - self.start_time
            print(f"\n{self.benchmark_name} Statistics:")
            print(f"  Total codes: {self.total_count}")
            print(f"  Success: {self.success_count}")
            print(f"  Errors: {self.error_count}")
            if self.total_count > 0:
                success_rate = self.success_count / self.total_count * 100
                print(f"  Success rate: {success_rate:.2f}%")
            print(f"  Duration: {duration}")
    
    def _get_statistics_dict(self) -> Dict[str, Any]:
        """获取统计信息字典"""
        return {
            'benchmark_name': self.benchmark_name,
            'total_count': self.total_count,
            'success_count': self.success_count,
            'error_count': self.error_count,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'duration': self.end_time - self.start_time if self.start_time and self.end_time else None
        }
    
    def get_shared_memory_buffer(self, size: int = 1024):
        """
        获取共享内存缓冲区
        
        Args:
            size: 缓冲区大小
            
        Returns:
            (shm, shm_name, shm_buffer)
        """
        shm = shared_memory.SharedMemory(create=True, size=size)
        return shm, shm.name, shm.buf