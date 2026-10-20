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

"""批处理管线基类。

2026-10-19 起只保留一层：

* :class:`BasePipeline` —— **通用逐码管线**（joblib 并行、逐码超时、统计、
  K 线取数/停牌检查）。review 直接继承它。

⚠️ 原先的 `BaseBenchmark` 兼容层与三个 benchmark 子类
（`mainstream/poolcoef/compact_benchmark`）**已删除** —— 它们的 import 全部悬空、
无任何运行时调用方（`MIGRATION_STATUS.md` 早已记为"未修"）。
将来需要 benchmark 时另写，别把死壳复活。

⚠️ 旧实现的三处问题在这里一并修掉：

1. ``tqdm_joblib`` 需要 ``(tqdm_object, label_template)`` 两个参数，旧调用只传了一个；
2. ``func_timeout`` 只 import/catch、**从未真正包住计算**；
3. 逐码失败被吞成 ``None``，没有状态/耗时，无法区分 ``no_data`` / ``suspended`` /
   ``timeout`` / ``error``。新基类统一返回 :class:`CodeResult`。
"""

from __future__ import annotations

import time
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime as dt, timedelta
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import sys
import traceback
import func_timeout
from tqdm import tqdm

try:
    from joblib import Parallel, delayed
    JOBLIB_AVAILABLE = True
except Exception:                       # pragma: no cover - 环境缺 joblib
    print('joblib not installed.')
    JOBLIB_AVAILABLE = False

from GolemQ.core.constants import MARKET_TYPE
from GolemQ.markets.StockCN.date_utils import GQ_util_if_tradetime, GQ_util_get_last_day
from GolemQ.core.base import set_cpu_affinity_even
from GolemQ.core.presentation import tqdm_joblib
from GolemQ.markets.StockCN.realtime import GQ_fetch_stock_min_realtime_adv
from GolemQ.markets.StockCN.fetch import (
    GQ_fetch_stock_min_adv,
    GQ_fetch_index_min_adv,
)
from multiprocessing import shared_memory
from GolemQ.markets.StockCN.symbol import normalize_code

warnings.simplefilter(action='ignore', category=FutureWarning)


@dataclass
class CodeResult:
    """逐码执行结果 —— 让失败原因可区分，而不是一律 ``None``。"""

    code: str
    status: str = 'ok'          # ok | empty | no_data | suspended | timeout | error
    result: Any = None
    error: Optional[str] = None
    duration: float = 0.0
    market_type: Any = None

    @property
    def ok(self) -> bool:
        return self.status in ('ok', 'empty')


class BasePipeline(ABC):
    """通用逐码批处理管线。"""

    def __init__(self,
                 name: str = 'BasePipeline',
                 verbose: bool = False,
                 market_type: Any = None,
                 timeout: int = 1800):
        self.name = name
        self.pipeline_name = name
        self.verbose = verbose
        self.market_type = market_type if market_type is not None else MARKET_TYPE.STOCK_CN
        self.timeout = timeout
        self.error_count = 0
        self.success_count = 0
        self.total_count = 0
        self.start_time = None
        self.end_time = None
        self.results: List[CodeResult] = []

    # ------------------------------------------------------------------ #
    # 公共工具
    # ------------------------------------------------------------------ #
    def _setup_cpu_affinity(self):
        set_cpu_affinity_even()

    def _get_date_range(self, offset: str = '', days: int = 960):
        end_date = pd.to_datetime(GQ_util_get_last_day()) if len(offset) == 0 else pd.to_datetime(offset)
        return end_date - timedelta(days=days), end_date

    def _fetch_kline_data(self, code: str, start_date: dt, end_date: dt,
                          frequency: str = '60min',
                          fetch_realtime: bool = True) -> Optional[pd.DataFrame]:
        """取分钟 K 线（旧 benchmark 兼容路径）。"""
        try:
            if self.market_type == MARKET_TYPE.STOCK_CN:
                kline_data = GQ_fetch_stock_min_adv(
                    code, start_date.strftime('%Y-%m-%d %H:%M:%S'),
                    (end_date + timedelta(hours=17)).strftime('%Y-%m-%d %H:%M:%S'),
                    frequence=frequency)
            elif self.market_type in (MARKET_TYPE.INDEX_CN, MARKET_TYPE.ETF_CN,
                                      MARKET_TYPE.FUND_CN):
                # ETF 是独立类型/集合；`GQ_fetch_index_min_adv` 内部按
                # `kline83.market_prefix` 落到 `etf_*` / `index_*`。
                kline_data = GQ_fetch_index_min_adv(
                    code, start_date.strftime('%Y-%m-%d %H:%M:%S'),
                    (end_date + timedelta(hours=17)).strftime('%Y-%m-%d %H:%M:%S'),
                    frequence=frequency)
            else:
                print(f'Unsupported market type: {self.market_type}')
                return None

            if fetch_realtime and kline_data is not None:
                if (GQ_util_if_tradetime(dt.now()) or
                        ((pd.to_datetime(GQ_util_get_last_day()) - end_date) < timedelta(hours=18))):
                    kline_data = GQ_fetch_stock_min_realtime_adv(
                        normalize_code(code, market_type=self.market_type),
                        kline_data, frequency=frequency)
            return kline_data.data if kline_data is not None else None
        except Exception as exc:        # noqa: BLE001 取数失败由调用方转状态
            print(f'Error fetching kline data for {code}: {exc}')
            return None

    def _check_suspension(self, kline_data: pd.DataFrame, code: str) -> bool:
        if kline_data is None or len(kline_data) == 0:
            return True
        last_date = kline_data.tail(1).index.get_level_values(0)[0]
        current_date = pd.to_datetime(GQ_util_get_last_day())
        if timedelta(days=0) < (current_date - last_date) < timedelta(days=84):
            if self.verbose:
                print(f'Code {code} may be suspended. Last trade: {last_date}')
            return True
        return False

    def _call_with_timeout(self, fn, *args, **kwargs):
        if self.timeout and self.timeout > 0:
            return func_timeout.func_timeout(self.timeout, fn, args=args, kwargs=kwargs)
        return fn(*args, **kwargs)

    # ------------------------------------------------------------------ #
    # 子类契约
    # ------------------------------------------------------------------ #
    @abstractmethod
    def run_code(self, code: str, **kwargs) -> CodeResult:
        """处理单个 code，返回 :class:`CodeResult`（不得抛未捕获异常）。"""

    # ------------------------------------------------------------------ #
    # 执行器
    # ------------------------------------------------------------------ #
    def _process_single_code(self, code: str, **kwargs) -> CodeResult:
        t0 = time.monotonic()
        try:
            out = self._call_with_timeout(self.run_code, code, **kwargs)
            if not isinstance(out, CodeResult):
                out = CodeResult(code=code, status='ok', result=out,
                                 market_type=self.market_type)
            out.duration = time.monotonic() - t0
            if out.market_type is None:
                out.market_type = self.market_type
            return out
        except func_timeout.exceptions.FunctionTimedOut:
            return CodeResult(code=code, status='timeout',
                              error=f'timeout after {self.timeout}s',
                              duration=time.monotonic() - t0,
                              market_type=self.market_type)
        except Exception as exc:        # noqa: BLE001
            traceback.print_exc()
            return CodeResult(code=code, status='error',
                              error=f'{type(exc).__name__}: {exc}',
                              duration=time.monotonic() - t0,
                              market_type=self.market_type)

    def run_codes(self,
                  codelist: Union[str, List[str]],
                  pool_size: int = 0,
                  verbose: bool = False,
                  **kwargs):
        """逐码执行；返回 ``(results, stats)``。"""
        self._setup_cpu_affinity()
        self.verbose = verbose
        self.start_time = dt.now()
        if isinstance(codelist, str):
            codelist = [codelist]
        codelist = list(codelist)
        self.total_count = len(codelist)
        self.error_count = 0
        self.success_count = 0

        if pool_size > 0 and JOBLIB_AVAILABLE and len(codelist) > 1:
            if verbose:
                print(f'Using parallel processing with {pool_size} workers')
            try:
                with tqdm_joblib(tqdm(desc=f'{self.name} Processing',
                                      total=len(codelist)), '{}'):
                    # ⚠️ **必须用 `with`**：`Parallel.__exit__` 会终止 worker。
                    # 不用上下文管理器时，Windows 上 Ctrl+C 后父进程可能在等
                    # 永远不结束的 loky worker —— 表现为「按键了但停不下来」。
                    with Parallel(n_jobs=pool_size) as parallel:
                        results = parallel(
                            delayed(self._process_single_code)(code, **kwargs)
                            for code in codelist)
            except KeyboardInterrupt:
                # `KeyboardInterrupt` 不是 `Exception`，上面的回退分支**不会**
                # 吞它；这里只补一句提示，然后继续向上抛给 CLI。
                print(f'\n{self.name}: 收到 Ctrl+C，已终止并行 worker',
                      file=sys.stderr)
                raise
            except Exception as exc:    # noqa: BLE001 并行失败回退串行
                print(f'Parallel processing error: {exc}')
                results = [self._process_single_code(code, **kwargs)
                           for code in tqdm(codelist, desc=f'{self.name} Processing')]
        else:
            results = [self._process_single_code(code, **kwargs)
                       for code in tqdm(codelist, desc=f'{self.name} Processing')]

        self.results = results
        self.success_count = sum(1 for r in results if r.ok)
        self.error_count = len(results) - self.success_count
        self.end_time = dt.now()
        self._print_statistics()
        return results, self._get_statistics_dict()

    def _print_statistics(self):
        if self.start_time and self.end_time:
            print(f'\n{self.name} Statistics:')
            print(f'  Total codes: {self.total_count}')
            print(f'  Success: {self.success_count}')
            print(f'  Errors: {self.error_count}')
            if self.total_count:
                print(f'  Success rate: {self.success_count / self.total_count * 100:.2f}%')
            print(f'  Duration: {self.end_time - self.start_time}')

    def _get_statistics_dict(self) -> Dict[str, Any]:
        counts: Dict[str, int] = {}
        for r in self.results:
            counts[r.status] = counts.get(r.status, 0) + 1
        return {
            'pipeline_name': self.name,
            'total_count': self.total_count,
            'success_count': self.success_count,
            'error_count': self.error_count,
            'status_counts': counts,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'duration': self.end_time - self.start_time if self.start_time and self.end_time else None,
        }

    def get_shared_memory_buffer(self, size: int = 1024):
        shm = shared_memory.SharedMemory(create=True, size=size)
        return shm, shm.name, shm.buf


