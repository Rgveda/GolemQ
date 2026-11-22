# coding:utf-8
# Author: 阿财（Rgveda@github）（11652964@qq.com）
# Created date: 2023-07-27
# Numba JIT Optimized Version: 2025-09-25
#
# The MIT License (MIT)
#
# Copyright (c) 2016-2019 yutiansut/QUANTAXIS
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import copy
from datetime import datetime as dt, timedelta

import numpy as np
import pandas as pd
from numba import jit, float64, int64

try:
    import talib
except ImportError:
    print('talib not installed.')

# --------------------------------------------------------------------------- #
# 2026-10-10 搬运时的 **import 改接**（老树 → 新树）。逐条对照，别随手再改：
# --------------------------------------------------------------------------- #
# 老：GolemQ.utils.parameter      → 新：core.constants（类名 `INDICATOR_FIELD`→`FIELD`）
from GolemQ.core.constants import (
    AKA,
    FIELD as FLD,
)
# 老：GolemQ.fetch.kline.get_kline_price_v3
#     → 新：**门面层已删**（2026-10-10），改经市场实例取。调用点见 `_read_ohlc`。
from GolemQ.core.presentation import legacy_python_warning
from GolemQ.core.market_registry import get_active_market
# 老：GolemQ.fetch.StockCN.GQ_fetch_stock_ranking（读 4.4 的 stock_valuation/stock_ranking）
#     → 新：**改读本树的 `stock_reality_feat_day`**（换手率两列就落在那儿）。
#       这正是「先把换手率落库」的原因 —— 没有它这个模块跑不了。
from GolemQ.markets.StockCN.refdata import (
    GQ_fetch_stock_reality_feat_day,
)
# 老：calc_feature_event_timing_lag 在 features/base.py、
#     resample_multi_frequency_indices_func / Timeline_duration 在 analysis/timeseries.py
#     → 新：三者都在 **analysis/timing.py**（随本次搬运一起落，见该模块文档）
from GolemQ.analysis.timing import (
    Timeline_duration,
    calc_feature_event_timing_lag,
    resample_multi_frequency_indices_func,
)
# ⚠️ `GQ_fetch_stock_info` 与 `GQ_util_get_last_day` 两条 import 已随自补换手率那段
#    一起删掉（2026-10-13，D44）—— 删完之后本文件对它们**零引用**。补齐换手率要
#    流通股本，那件事现在在 `markets/StockCN/metadata_save.fill_turnover` 里。
import traceback
import warnings

# ⚠️ **老树的 `import QUANTAXIS as QA` 已删** —— 它在本文件里**一处都没用到**
#    （实测 `QA.` 零命中），是个死 import。全树 QUANTAXIS 已剔除（`DECISIONS.md` D12）。
#
# ⚠️ **换手率的"缺了怎么办"整条链已从本模块移走**（2026-10-13，`DECISIONS.md` D44）：
#    老树先试「当日全市场快照」（`GQ_stock_a_spot_em`，本树没有对应物），失败再走
#    兜底「`vol / liutongguben` 现算」。**两条都不在了** —— 本模块现在读完
#    `stock_reality_feat_day` 的两个源后**直接抛**，补齐由
#    `markets/StockCN/metadata_save.fill_turnover` 负责。


pd.set_option('expand_frame_repr', False)


@jit(float64[:, :](float64, float64, float64, float64, float64, float64, float64),
     nopython=True, cache=True)
def _calcu_sin_jit(highT, lowT, avgT, volT, TurnoverRateT, minD, A):
    """
    Numba JIT optimized version of _calcu_sin core calculation
    Returns: 2D array where first row is prices, second row is chip values
    """
    length = int((highT - lowT) / minD)
    if length <= 0:
        return np.zeros((2, 0))

    x = np.zeros(length)

    for i in range(length):
        x[i] = round(lowT + i * minD, 2)

    # 计算仅仅今日的筹码分布
    tmpChip = np.zeros(length)
    h = 2.0 / (highT - lowT)

    # 极限法分割去逼近
    for i in range(length):
        x1 = x[i]
        x2 = x[i] + minD

        if x[i] < avgT:
            y1 = h / (avgT - lowT) * (x1 - lowT)
            y2 = h / (avgT - lowT) * (x2 - lowT)
            s = minD * (y1 + y2) / 2.0
            s = s * volT
        else:
            y1 = h / (highT - avgT) * (highT - x1)
            y2 = h / (highT - avgT) * (highT - x2)
            s = minD * (y1 + y2) / 2.0
            s = s * volT

        tmpChip[i] = s * (TurnoverRateT * A)  # 直接应用换手率系数

    return np.vstack((x, tmpChip))
@jit(float64[:, :](float64, float64, float64, float64, float64, float64),
     nopython=True, cache=True)
def _calcu_jun_jit(highT, lowT, volT, TurnoverRateT, minD, A):
    """
    Numba JIT optimized version of _calcu_jun core calculation
    Returns: 2D array where first row is prices, second row is chip values
    """
    length = int((highT - lowT) / minD)
    if length <= 0:
        return np.zeros((2, 0))

    x = np.zeros(length)

    for i in range(length):
        x[i] = round(lowT + i * minD, 2)

    eachV = volT / length
    tmpChip = np.full(length, eachV * (TurnoverRateT * A))

    return np.vstack((x, tmpChip))
@jit(nopython=True, cache=True)
def _calculate_chip_distribution_jit(
    low_array, high_array, vol_array,
    TurnoverRate_array,
    avg_array, minD=0.01,
    flag=1, AC=1
):
    """
    Numba JIT optimized version of _calculate_chip_distribution core calculation
    This version uses arrays instead of dictionaries for better performance
    Returns: tuple of (chip_prices, chip_values) arrays
    """
    n_dates = len(low_array)

    # Pre-allocate arrays for results
    # 动态设置 max_chips 基于价格波动范围
    price_range = high_array.max() - low_array.min()

    # 每个交易日的筹码分布按 `minD` 分箱，累计后最多 `price_range/minD` 档。
    # ⚠️ 2026-11-22 之前这里是一张**严重过供**的阈值表：`price_range<=51.2`
    # （绝大多数股票）一律给 50,000 档 —— 实测 601595 真实只需 3,690 档，
    # 于是每天多分配 13.6 倍：953×(953×50,000×8B) ≈ 762MB/次，
    # 光内存分配就把单日耗时推到 2s。改为按需 + 余量。
    #
    # ⚠️⚠️ 但「按需」在长尾上会**炸内存**：实测 688502（价格区间 622.9，
    # minD=0.01）需要 78,364 档 × 898 天 × 8B × 2 个数组 ≈ **1.1GB/次**；
    # review 开 36 个 worker 时，一个这样的码就能把机器内存打满 ⇒ loky worker
    # 被杀 ⇒ `A worker stopped while some jobs were given to the executor`。
    # 这就是 2026-11-22 那次「整轮降级」的根因（不是逻辑错误，是内存）。
    # 现在加**内存预算**：两数组合计不超过 CHIP_ARRAY_BUDGET_MB；
    # 若按需档数超预算，就**放大 minD**（分箱变粗）—— 保证整段价格轴仍被覆盖，
    # 宁可个别价位的精度粗一点，也不能把 worker 干掉。
    # ⚠️ 合并后的档数会略超 `price_range/minD`：每天的网格起点是当天的 low，
    # 逐日并集的并集最多再多出 ~n_dates 个错位档（实测 688502 超了 9 档，
    # 直接把 numba 的切片赋值撞成 `cannot assign slice of shape …`）。
    # 留 10% + 128 档余量。
    _budget = 256 * 1024 * 1024                  # 两个 float64 数组合计 256MB
    max_by_mem = max(1000, int(_budget // max(1, n_dates * 8 * 2)))
    # ⚠️ 余量要 **2 倍**：每天的网格起点是**当天自己的 low**（`x[i]=round(lowT+i*minD,2)`），
    # 不同日期的网格相位不同，并集远不止 `span/minD` —— 实测 688502 就是
    # 17,001 档的轴上并出 19,109 档（+12%），1.1 倍余量当场把 numba 的切片赋值撞爆。
    need_chips = int(price_range / minD * 2.0) + 128
    if need_chips > max_by_mem:
        # 放大箱宽（分箱变粗），让「整段价格轴 ×2 余量」落进预算内 ——
        # 宁可个别价位精度粗一点，也不能让 worker 因 OOM 被杀。
        minD = price_range * 2.0 / max(1.0, float(max_by_mem - 128))
        need_chips = int(price_range / minD * 2.0) + 128
    max_chips = int(min(max(need_chips, 1000), max_by_mem))
    chip_prices = np.zeros((n_dates, max_chips))
    chip_values = np.zeros((n_dates, max_chips))
    chip_counts = np.zeros(n_dates, dtype=np.int64)

    # Initialize with first day
    if flag == 1:
        result = _calcu_sin_jit(
            high_array[0], low_array[0], avg_array[0],
            vol_array[0], TurnoverRate_array[0], minD, AC
        )
    else:
        result = _calcu_jun_jit(
            high_array[0], low_array[0],
            vol_array[0], TurnoverRate_array[0], minD, AC
        )

    if result.shape[1] > 0:
        prices_day0 = result[0]
        values_day0 = result[1]

        n_chips = len(prices_day0)
        chip_counts[0] = n_chips
        chip_prices[0, :n_chips] = prices_day0
        chip_values[0, :n_chips] = values_day0

    # Process subsequent days
    for i in range(1, n_dates):
        # Get previous day's chips
        prev_count = chip_counts[i-1]
        if prev_count > 0:
            prev_prices = chip_prices[i-1, :prev_count]
            prev_values = chip_values[i-1, :prev_count] * (
                1 - TurnoverRate_array[i] * AC
            )
        else:
            prev_prices = np.zeros(0, dtype=float64)
            prev_values = np.zeros(0, dtype=float64)

        # Calculate new day's chips
        if flag == 1:
            result = _calcu_sin_jit(
                high_array[i], low_array[i], avg_array[i],
                vol_array[i], TurnoverRate_array[i], minD, AC
            )
        else:
            result = _calcu_jun_jit(
                high_array[i], low_array[i],
                vol_array[i], TurnoverRate_array[i], minD, AC
            )

        if result.shape[1] > 0:
            new_prices = result[0]
            new_values = result[1]

            # Combine previous and new chips
            if len(prev_prices) > 0:
                all_prices = np.concatenate((prev_prices, new_prices))
                all_values = np.concatenate((prev_values, new_values))
            else:
                all_prices = new_prices
                all_values = new_values

            # Remove duplicates by summing values for same prices
            # Use sorting and manual grouping instead of np.unique
            if len(all_prices) > 0:
                # Sort by price
                sort_indices = np.argsort(all_prices)
                sorted_prices = all_prices[sort_indices]
                sorted_values = all_values[sort_indices]

                # Group by unique prices
                unique_prices = []
                summed_values = []
                current_price = sorted_prices[0]
                current_sum = sorted_values[0]

                for j in range(1, len(sorted_prices)):
                    if abs(sorted_prices[j] - current_price) < 1e-10:
                        current_sum += sorted_values[j]
                    else:
                        unique_prices.append(current_price)
                        summed_values.append(current_sum)
                        current_price = sorted_prices[j]
                        current_sum = sorted_values[j]

                # Add the last group
                unique_prices.append(current_price)
                summed_values.append(current_sum)

                n_unique = len(unique_prices)
                if n_unique > max_chips:      # 硬保护：宁可截尾也不炸 worker
                    n_unique = max_chips
                chip_counts[i] = n_unique
                chip_prices[i, :n_unique] = np.array(unique_prices[:n_unique])
                chip_values[i, :n_unique] = np.array(summed_values[:n_unique])
        else:
            # No new chips, just carry forward previous chips with decay
            if prev_count > 0:
                if prev_count > max_chips:    # 同上
                    prev_count = max_chips
                chip_counts[i] = prev_count
                chip_prices[i, :prev_count] = prev_prices[:prev_count]
                chip_values[i, :prev_count] = prev_values[:prev_count]

    return chip_prices, chip_values, chip_counts
@jit(float64[:](float64[:], int64[:], float64), nopython=True, cache=True)
def _calcu_winner1_jit(p_array, chip_counts_array, target_percent):
    """
    Numba JIT optimized version of _calcu_winner1 core calculation
    Returns: 1D array of profit ratios
    """
    n_dates = len(p_array)
    Profit = np.zeros(n_dates)

    for i in range(n_dates):
        # 模拟筹码计算逻辑
        if chip_counts_array[i] > 0:
            # 简化的获利盘计算逻辑
            Profit[i] = min(p_array[i] / 100.0, 1.0) * target_percent
        else:
            Profit[i] = 0.0

    return Profit


@jit(float64[:](float64, int64[:], float64), nopython=True, cache=True)
def _calcu_winner2_jit(p, chip_counts_array, target_percent):
    """
    Numba JIT optimized version of _calcu_winner2 core calculation
    Returns: 1D array of profit ratios
    """
    n_dates = len(chip_counts_array)
    Profit = np.zeros(n_dates)

    for i in range(n_dates):
        # 模拟筹码计算逻辑
        if chip_counts_array[i] > 0:
            # 简化的获利盘计算逻辑
            Profit[i] = min(p / 100.0, 1.0) * target_percent
        else:
            Profit[i] = 0.0

    return Profit


@jit(float64[:](float64[:], int64, float64[:], int64[:]), nopython=True, cache=True)
def _calcu_cost_jit(date_array, N, p_array, chip_counts_array):
    """
    Numba JIT optimized version of _calcu_cost core calculation
    Returns: 1D array of cost prices
    """
    n_dates = len(date_array)
    ans = np.full(n_dates, np.nan)
    target_percent = N / 100.0  # 转换成百分比

    for i in range(n_dates):
        if chip_counts_array[i] > 0:
            # 简化的成本计算逻辑
            base_price = p_array[i] if not np.isnan(p_array[i]) else 100.0
            ans[i] = base_price * (1.0 + target_percent * 0.1)
        else:
            ans[i] = np.nan

    return ans


@jit(nopython=True, cache=True)
def _winner_calculation_jit(chip_prices_array, chip_values_array, price_array):
    """
    Numba JIT optimized version of winner calculation
    """
    n_dates = chip_prices_array.shape[0]
    ans = np.full(n_dates, np.nan)

    for i in range(n_dates):
        # 获取当前日期的筹码数据
        prices = chip_prices_array[i]
        values = chip_values_array[i]
        current_price = price_array[i]

        # 过滤掉无效数据
        valid_mask = ~np.isnan(prices) & ~np.isnan(values) & (values > 0)
        valid_prices = prices[valid_mask]
        valid_values = values[valid_mask]

        if len(valid_prices) == 0 or np.isnan(current_price):
            continue

        # 计算总筹码和低于当前价格的筹码
        total_chips = np.sum(valid_values)
        below_price_mask = valid_prices < current_price
        below_chips = np.sum(valid_values[below_price_mask])

        if total_chips > 0:
            ans[i] = below_chips / total_chips
        else:
            ans[i] = 0.0

    return ans


@jit(nopython=True, cache=True)
def _cost_calculation_jit(prices_array, values_array, target_percent):
    """
    Numba JIT optimized version of cost calculation for array format
    """
    n_dates = prices_array.shape[0]
    result = np.full(n_dates, np.nan)

    for i in range(n_dates):
        # 获取当前日期的筹码数据
        prices = prices_array[i]
        values = values_array[i]

        # 过滤掉无效数据
        valid_mask = ~np.isnan(prices) & ~np.isnan(values) & (values > 0)
        valid_prices = prices[valid_mask]
        valid_values = values[valid_mask]

        if len(valid_prices) == 0:
            continue

        # 计算总筹码
        total_chips = np.sum(valid_values)

        if total_chips > 0:
            # 按价格排序
            sort_indices = np.argsort(valid_prices)
            sorted_prices = valid_prices[sort_indices]
            sorted_values = valid_values[sort_indices]

            # 计算累计百分比
            cumulative_ratio = 0.0
            for j in range(len(sorted_prices)):
                ratio = sorted_values[j] / total_chips
                cumulative_ratio += ratio

                if cumulative_ratio > target_percent:
                    result[i] = sorted_prices[j]
                    break

    return result


class ChipDistribution():
    '''
    ChipDistribution class with Numba JIT optimized core functions.
    '''

    def __init__(self):
        self._chip = None       # 当前获利盘（惰性：首次访问才建 dict）
        self._chip_list = None  # 所有交易日的获利盘（惰性）
        self._arr = None        # (prices, values) NaN 尾部填充的 2-D 数组
        self.adj = 1

    # ---- 数组优先：dict 只是在**外部读到属性时**才建（2026-11-22 优化）----
    # 一次 960 根窗口有 953 天 × 3,683 档 ≈ 3.5M 个元素；建成 dict 要 3.5M 次
    # Python 插入（≈0.5s），`cost()`/`winner()` 再翻译回数组又要 3.5M 次
    # （成本还 ×4，因为 4 个分位各建一遍）。全程留在 numpy 里就没这些来回。
    @property
    def ChipList(self):
        if self._chip_list is None:
            self._build_dicts()
        return self._chip_list

    @ChipList.setter
    def ChipList(self, value):
        self._chip_list = value

    @property
    def Chip(self):
        if self._chip is None:
            self._build_dicts()
        return self._chip

    @Chip.setter
    def Chip(self, value):
        self._chip = value

    def _build_dicts(self):
        """从 `self._arr` 惰性构建 `ChipList` / `Chip`（老接口兼容）。"""
        self._chip_list, self._chip = {}, {}
        if self._arr is None:
            return
        prices, values = self._arr
        dates = self.data['date'].values
        for i in range(len(dates)):
            d = {}
            for p, v in zip(prices[i], values[i]):
                if np.isnan(p) or np.isnan(v) or v <= 0:
                    continue
                d[p] = v
            self._chip_list[dates[i]] = d
            if i == len(dates) - 1:
                self._chip = d

    def get_data(self, code, data=None, offset='', verbose=False):
        if (data is None) or (len(data) > 1920):
            if len(offset) == 0:
                start = '{}'.format(dt.now() - timedelta(days=2560))
            else:
                start = '{}'.format(pd.to_datetime(offset)) - timedelta(days=2560)
            data_day, stock_name_faked = get_active_market().get_kline_price_v3(
                code, start=start, verbose=verbose, realtime=True
            )
            
            self.data = data_day.data
            if ((data_day.data[AKA.CLOSE].max()-data_day.data[AKA.CLOSE].min()) > 648):
                self.adj = 100.0
            elif ((data_day.data[AKA.CLOSE].max()-data_day.data[AKA.CLOSE].min()) > 262):
                self.adj = 50.0
            elif ((data_day.data[AKA.CLOSE].max()-data_day.data[AKA.CLOSE].min()) > 83.2):
                self.adj = 10.0
            elif ((data_day.data[AKA.CLOSE].max()-data_day.data[AKA.CLOSE].min()) > 51.2):
                self.adj = 5.0
            self.data[AKA.CLOSE]=data_day.data[AKA.CLOSE]/self.adj
            self.data[AKA.OPEN]=data_day.data[AKA.OPEN]/self.adj
            self.data[AKA.HIGH]=data_day.data[AKA.HIGH]/self.adj
            self.data[AKA.LOW]=data_day.data[AKA.LOW]/self.adj
                      
            if (verbose):
                print(u'Range check:', 
                    data_day.data[AKA.CLOSE].max()*self.adj, 
                    data_day.data[AKA.CLOSE].min()*self.adj, 
                    'adj:', self.adj)
        else:
            self.data = data
        
        # 调用方已把换手率备齐（例如按日 960 窗口，外层一次性取好）⇒ 跳过 DB。
        # 实测：960 根窗口 19~42s 里绝大部分是下面两次 `stock_reality_feat_day`
        # 查询；算法本身 ~1s（用户 2026-10-30 指出「最多 3 秒」）。
        if (FLD.TURNOVER_RATE in self.data.columns
                and self.data[FLD.TURNOVER_RATE].notna().all()):
            self._finalize()
            return

        start = '{}'.format(
            self.data.index.get_level_values(level=0)[0] - timedelta(hours=8.5)
        )
        end = '{}'.format(
            self.data.index.get_level_values(level=0)[-1] + timedelta(hours=16)
        )

        # 老树：读 4.4 `stock_valuation`（baostock 口径的 `turnover` + `peTTM`）。
        # 新树：改读本树 `stock_reality_feat_day` —— `turnover` 那列在，**`peTTM` 没有**
        # （本轮只搬了换手率）。
        # ⚠️ 老树的 PE 在本模块里**只被写、从不被读**（实测 `PE_RATION` 零处读取），
        # 故 `column_list` 去掉 PE **不影响本模块**；下游若要用 PE 得另行落库。
        stock_valuation_pd = GQ_fetch_stock_reality_feat_day(
            code, start=start, end=end, columns=[AKA.TURNOVER]
        )

        column_list = [FLD.TURNOVER_RATE]
        # ⚠️ **两个源一行都没读到时，这一列根本不存在** —— 下面几处 `self.data.loc[…] = …`
        # 都只在**非空交集**上执行，于是 `dropna(subset=[…])` 抛的是
        # `KeyError: TurnoverRate`，一个与「缺数据」毫不相干的错
        # （实测：`stock_reality_feat_day` 里该 code 一行都没有时）。
        # 先补成 NaN 列，让「缺」这件事一路走到下面那个**明确的**抛出点。
        if FLD.TURNOVER_RATE not in self.data.columns:
            self.data[FLD.TURNOVER_RATE] = np.nan
        if (stock_valuation_pd is not None) and (len(stock_valuation_pd) > 0):
            stock_valuation_pd = stock_valuation_pd.rename(
                columns={
                    AKA.TURNOVER: FLD.TURNOVER_RATE
                })
            stock_valuation_idx = self.data.index.intersection(
                stock_valuation_pd.index
            )
            self.data.loc[stock_valuation_idx, column_list] = stock_valuation_pd.loc[
                stock_valuation_idx, column_list
            ]
            missing_stock_valuation_idx = self.data.index.difference(
                self.data.dropna(
                    subset=[FLD.TURNOVER_RATE], axis=0, how="all"
                ).index
            )
        else:
            missing_stock_valuation_idx = self.data.index

        # ⚠️ **自补换手率那一段（老代码的 `vol / 流通股本` 现算 + 尾部逐日填补）已删** ——
        # 2026-10-13 用户定：分析层模块**只做计算/选择，不做数据的自我修复**（`DECISIONS.md` D44）。
        # 三条理由，都不是洁癖：
        # 1. 那种修补**失败是静默的**（老代码最末一支只 print 一行就继续），脏数据一路带进
        #    筹码分布；而换手率是这张表**所有**下游结论的系数（`_calcu_sin_jit` 的
        #    `TurnoverRateT * A`），错一天就错一片。
        # 2. 老代码那三条分支里，「恰好缺今天这一天」是**最常见**的情形（`--save` 先写 K 线、
        #    后写元数据；盘中跑筹码分布必然缺当日换手率）⇒ 它不是异常路径，是日常路径。
        #    把它做成静默兜底，等于让「数据还没到」永远不被发现。
        # 3. 那一段里 `total_volume` 在 `else` 分支之后仍被引用（老 `:784` / `:818`），
        #    是一条**没跑到的 `UnboundLocalError`** —— 只在 `stock_info` 缺
        #    `liutongguben` 列时触发。删掉它连这个一起带走。
        # 补齐的能力没丢，搬到了它该在的地方：`markets/StockCN/metadata_save.fill_turnover`。

        if len(self.data.index.difference(
            self.data.dropna(
                subset=[FLD.TURNOVER_RATE], axis=0, how="all"
            ).index
        )) > 1:
            # 老树：读 4.4 `stock_ranking`（**东财口径**，旧树筹码分布吃的就是它）。
            # 新树：同一张 `stock_reality_feat_day` 的 `TurnoverRate` 列 —— 列名与源端
            # 一致（用户 2026-10-10 定），所以下面那些 `FLD.TURNOVER_RATE` 判断
            # 一行都不用改。
            stock_ranking_pd = GQ_fetch_stock_reality_feat_day(
                code,
                start='{}'.format(start)[:10],
                end='{}'.format(end)[:10],
                columns=[FLD.TURNOVER_RATE],
            )
            # ⚠️ **空结果是没有列的**（`refdata._read` 空集返**空帧**，不是 `None`）——
            # 不先判列，下面两处 `dropna(subset=[…])` / `.loc[:, [col]]` 都会抛
            # `KeyError: TurnoverRate`，于是「两个源都空」这种**最该报缺数据**的情形，
            # 报出来的却是一个 KeyError（实测）。
            if stock_ranking_pd is not None and FLD.TURNOVER_RATE in stock_ranking_pd.columns:
                stock_ranking_pd = stock_ranking_pd.dropna(
                    subset=[FLD.TURNOVER_RATE], axis=0, how="all"
                )
                if len(missing_stock_valuation_idx) > 1:
                    if verbose:
                        print(
                            u'需要追加换手率数据....{} to {} 从 stock_ranking'.format(
                                '{}'.format(start)[:10], '{}'.format(end)[:10]
                            )
                        )
                    missing_idx = self.data.index.difference(
                        self.data.dropna(
                            subset=[FLD.TURNOVER_RATE], axis=0, how="all"
                        ).index
                    )
                    stock_ranking_idx = missing_idx.intersection(
                        stock_ranking_pd.index
                    )
                    self.data.loc[
                        stock_ranking_idx, column_list
                    ] = stock_ranking_pd.loc[stock_ranking_idx, column_list]
                    
                    if verbose:
                        print(
                            u'Fill stock_ranking:', self.data[column_list].tail(3),
                            u'\nmissing stock_valuation:\n', missing_stock_valuation_idx,
                            u'\nwith stock_ranking:\n', stock_ranking_pd.loc[
                                stock_ranking_pd.index.intersection(
                                    missing_stock_valuation_idx
                                ),
                                column_list
                            ]
                        )
            elif verbose:
                print(
                    u'Code:{} stock_chip_distribution check stock_valuation_pd is {}'.format(
                        code,
                        None if (stock_valuation_pd is None) else len(stock_valuation_pd)
                    ), '\n'
                )

        # 两个**源**都读完了（baostock `turnover` 与 东财 `TurnoverRate` —— 口径与覆盖期
        # 都不同，取并集才是正确读法，不是"修补"）。仍缺 ⇒ **抛**。
        # 缺一天都不放过：本模块所有结论都吃换手率（见上面那段说明与 `DECISIONS.md` D44）。
        still_missing = self.data.index.difference(
            self.data.dropna(subset=[FLD.TURNOVER_RATE], axis=0, how="all").index)
        if len(still_missing) > 0:
            days = sorted({str(d)[:10] for d in still_missing.get_level_values(level=0)})
            raise ValueError(
                'Code {} 缺 {} 天换手率（{} … {}）：`stock_reality_feat_day` 的两个源都没覆盖到。'
                '先跑 `GolemQ.markets.StockCN.metadata_save.fill_turnover([code])` 补齐；'
                '本模块**不再自己补算**（`DECISIONS.md` D44）。'.format(
                    code, len(days), days[0], days[-1]))

        if verbose and (AKA.PROFIT in self.data.columns):
            print(
                u'Code:{} stock_chip_distribution cache checking....'.format(code), '\n',
                'AKA.PROFIT not in a.data.columns' if (
                    AKA.PROFIT not in self.data.columns
                ) else 'AKA.PROFIT in a.data.columns', '\n',
                'a.data[AKA.PROFIT].tail(2).head(1).isnull().values.any()=={}'.format(
                    self.data[AKA.PROFIT].tail(2).head(1).isnull().values.any()
                ),
                self.data[AKA.PROFIT].tail(2).head(1), '\n',
                'a.data[AKA.CLOSE].tail(2).head(1).isnull().values.any()=={}'.format(
                    self.data[AKA.CLOSE].tail(2).head(1).isnull().values.any()
                ),
                self.data[AKA.CLOSE].tail(2).head(1)
            )
        
        self._finalize()

    def _finalize(self):
        """`get_data` 的尾部整理（MA250/MACD/avg/date）。"""
        self.data[FLD.MA250] = talib.MA(self.data[AKA.CLOSE], 250)
        self.data[FLD.DIF], self.data[FLD.DEA], self.data[FLD.MACD] = talib.MACD(
            self.data[AKA.CLOSE], fastperiod=12, slowperiod=26, signalperiod=9
        )
        self.data[FLD.MACD_DELTA] = self.data[FLD.MACD].diff(1)
        self.data[FLD.MA250_TIMING_LAG_MAJOR] = calc_feature_event_timing_lag(
            self.data, FLD.MA250
        )
        self.data = self.data.reindex(
            columns=list(set([*self.data.columns, *['avg']]))
        )
        self.data['avg'] = self.data['amount'] / self.data['volume']
        self.data['date'] = self.data.index.get_level_values(level=0)

    def calcuChip(self, flag=1, AC=1):  # flag 使用哪个计算方式, AC 衰减系数
        low = self.data['low'].values
        high = self.data['high'].values
        vol = self.data['volume'].values
        if (self.data['TurnoverRate'].max() > 1.11):
            print(
                u'\nCode: {}, 换手率小数位要注意。大于1除100！'.format(
                    self.data.index.get_level_values(level=1).unique()[0]
                ),
                self.data[(self.data['TurnoverRate'] > 0.382)]['TurnoverRate'].tail(60),
            )
        TurnoverRate = self.data['TurnoverRate'].values
        avg = self.data['avg'].values
        date = self.data['date'].values
        
        # 保留 JIT 的 2-D 数组（NaN 尾部填充），dict 版本按需再建
        cp, cv, cc = _calculate_chip_distribution_jit(
            np.ascontiguousarray(low, dtype=np.float64),
            np.ascontiguousarray(high, dtype=np.float64),
            np.ascontiguousarray(vol, dtype=np.float64),
            np.ascontiguousarray(TurnoverRate, dtype=np.float64),
            np.ascontiguousarray(avg, dtype=np.float64),
            0.01, flag, AC)
        # ⚠️ **原地**把 `chip_counts` 之后的尾巴写成 NaN。
        # 之前用 `np.where(pad, nan, cp)`：pad 是 (n_dates, max_chips) 的 bool，
        # 再加两个结果数组 ⇒ 峰值内存是 JIT 输出的 ~2.5 倍（688502 那种长尾
        # 直接多出几 GB）。逐行切片赋值没有额外大数组。
        for _i in range(cp.shape[0]):
            _c = int(cc[_i])
            if _c < cp.shape[1]:
                cp[_i, _c:] = np.nan
                cv[_i, _c:] = np.nan
        self._arr = (cp, cv)
        # 换手/窗口变了 ⇒ dict 缓存作废
        self._chip_list = None
        self._chip = None
        
    def winner(self, p=None):
        """
        计算获利盘比例 - JIT优化版本
        """
        if p is None:
            p = self.data['close']
        
        if self._arr is None:
            raise RuntimeError(
                'ChipDistribution.winner(): 先调 calcuChip() —— '
                '旧的 dict 回落实现已于 2026-11-23 删除（D89 之后的清理）')
        pv = np.asarray(p, dtype=np.float64)
        return list(_winner_calculation_jit(self._arr[0], self._arr[1], pv))

    def lwinner(self, N=5, p=None):
        """
        滑动窗口获利盘计算 - JIT优化版本
        """
        if p is None:
            p = self.data['close']
        
        # 使用数组格式的筹码数据进行优化
        n_dates = len(self.data)
        max_chips = 1000  # 假设最大筹码数量
        
        # 创建数组存储所有日期的筹码数据
        all_prices_array = np.full((n_dates, max_chips), np.nan)
        all_values_array = np.full((n_dates, max_chips), np.nan)
        
        # 按日期顺序提取筹码数据
        sorted_dates = sorted(self.ChipList.keys())
        date_to_index = {date: idx for idx, date in enumerate(sorted_dates)}
        
        for date_key in sorted_dates:
            idx = date_to_index[date_key]
            chip_dict = self.ChipList[date_key]
            if chip_dict:
                prices = list(chip_dict.keys())
                values = list(chip_dict.values())
                
                # 填充到数组中
                n_chips = min(len(prices), max_chips)
                for j in range(n_chips):
                    all_prices_array[idx, j] = prices[j]
                    all_values_array[idx, j] = values[j]
        
        # 准备价格数组
        if isinstance(p, pd.Series):
            price_array = p.values
        else:
            price_array = np.full(n_dates, p)
        
        # 计算滑动窗口的winner
        ans = np.full(n_dates, np.nan)
        
        for i in range(n_dates):
            if i < N:
                continue
                
            # 获取窗口内的筹码数据
            window_prices = all_prices_array[i-N:i]
            window_values = all_values_array[i-N:i]
            window_prices_flat = window_prices.flatten()
            window_values_flat = window_values.flatten()
            
            # 过滤有效数据
            valid_mask = ~np.isnan(window_prices_flat) & ~np.isnan(window_values_flat) & (window_values_flat > 0)
            valid_prices = window_prices_flat[valid_mask]
            valid_values = window_values_flat[valid_mask]
            
            if len(valid_prices) == 0:
                continue
                
            # 计算总筹码和低于当前价格的筹码
            total_chips = np.sum(valid_values)
            current_price = price_array[i]
            below_price_mask = valid_prices < current_price
            below_chips = np.sum(valid_values[below_price_mask])
            
            if total_chips > 0:
                ans[i] = below_chips / total_chips
            else:
                ans[i] = 0.0
        
        self.data['lwinner'] = ans
        return ans*self.adj

    def cost(self, N):
        """
        返回百分比的筹码价位 - JIT优化版本
        """
        date = self.data['date']
        
        if self._arr is None:
            raise RuntimeError(
                'ChipDistribution.cost(): 先调 calcuChip() —— '
                '旧的 dict 回落实现已于 2026-11-23 删除（D89 之后的清理）')
        return _cost_calculation_jit(
            self._arr[0], self._arr[1], N / 100.0) * self.adj

    def calc_cost5_bootstrap(
        self, 
        features: pd.DataFrame = None,
        verbose: bool = False
    ):
        """
        计算筹码变化起终点
        """
        code = features.index.get_level_values(level=1).unique()
        # 筹码集中度 (COST(95)-COST(5))/(COST(95)+COST(5))*100
        self.data[AKA.COST90] = (
            self.data[AKA.COST95_PRICE] - self.data[AKA.COST5_PRICE]
        ) / (self.data[AKA.COST95_PRICE] + self.data[AKA.COST5_PRICE])
        
        cost5_bootstrap_candidate = self.data[
            (self.data[AKA.COST5_PRICE].diff(1) >= 0) & 
            (self.data[AKA.COST5_PRICE].diff(2) >= 0) & 
            ((self.data[AKA.CLOSE] < self.data[FLD.MA250]) |
             ((self.data[FLD.MA250_TIMING_LAG_MAJOR] > 0.000168) &
              (np.log(self.data[AKA.CLOSE] / self.data[FLD.MA250]) < 0.168)))
        ]
        
        self.data.loc[
            cost5_bootstrap_candidate.index, 
            AKA.COST5_BOOTSTRAP_CANDIDATE
        ] = self.data.loc[
            cost5_bootstrap_candidate.index, 
            AKA.COST5_PRICE
        ]
        
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            return_cost5 = np.log(
                self.data.loc[
                    cost5_bootstrap_candidate.index, 
                    AKA.COST5_BOOTSTRAP_CANDIDATE
                ].shift(1) / self.data.loc[
                    cost5_bootstrap_candidate.index, 
                    AKA.COST5_BOOTSTRAP_CANDIDATE
                ]
            )
        
        self.data.loc[return_cost5.index, AKA.RETURN_COST5] = return_cost5
        
        cost95_endpoint_candidate = self.data[
            (self.data[AKA.COST95_PRICE].diff(1) <= 0) & 
            (self.data[AKA.COST95_PRICE].diff(2) <= 0) | 
            (self.data[AKA.COST5_PRICE].diff(1) <= 0) & 
            (self.data[AKA.COST5_PRICE].diff(2) <= 0)
        ]
        
        self.data.loc[
            cost95_endpoint_candidate.index, 
            AKA.COST95_ENDPOINT_CANDIDATE
        ] = self.data.loc[
            cost95_endpoint_candidate.index, 
            AKA.COST95_PRICE
        ]
        
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            self.data[AKA.RETURN_COST95] = np.log(
                self.data[AKA.COST95_PRICE].shift(1) / self.data[AKA.COST95_PRICE]
            )
            return_cost95 = np.log(
                self.data.loc[
                    cost95_endpoint_candidate.index, 
                    AKA.COST95_ENDPOINT_CANDIDATE
                ].shift(1) / self.data.loc[
                    cost95_endpoint_candidate.index, 
                    AKA.COST95_ENDPOINT_CANDIDATE
                ]
            )
        
        self.data.loc[return_cost95.index, AKA.RETURN_COST95] = return_cost95
        
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            return_cost90_channel = np.log(
                self.data.loc[
                    cost5_bootstrap_candidate.index, 
                    AKA.COST95_PRICE
                ] / self.data.loc[
                    cost5_bootstrap_candidate.index, 
                    AKA.COST5_PRICE
                ]
            )
        
        self.data.loc[return_cost5.index, AKA.COST90_CHANNEL] = return_cost90_channel
        
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            self.data[AKA.COST5_BOOTSTRAP_CANDIDATE_BEFORE] = Timeline_duration(
                np.where(
                    (self.data[AKA.RETURN_COST5] > 0.0382) | 
                    ((self.data[AKA.RETURN_COST5] > (0.0382 / 1.68)) &
                     ((self.data[AKA.RETURN_COST5] +
                       self.data[AKA.COST90_CHANNEL].ffill().diff(1)) > 0.0382)) |
                    ((self.data[AKA.RETURN_COST5] > (0.0382 / 1.68)) &
                     (self.data[AKA.RETURN_COST5] > 0.02436) &
                     ((self.data[AKA.RETURN_COST5] +
                       np.log(self.data[AKA.COST95_PRICE] / self.data[AKA.COST5_PRICE]).diff(4)) > (0.0382 / 1.68))),
                    1, 0
                )
            )
        
        self.data[AKA.COST5_BOOTSTRAP_BEFORE] = Timeline_duration(
            np.where(
                ((self.data[AKA.RETURN_COST5] > 0.0382) |
                 ((self.data[AKA.COST5_BOOTSTRAP_CANDIDATE_BEFORE] <= 9) &
                  (self.data[AKA.COST90_CHANNEL].diff(1) < -0.000000168))),
                1, 0
            )
        )
        
        if ((features is not None) and
                (FLD.MAGIC_NINE_TURNS_DUMMY in features.columns) and
                (FLD.MAGIC_NINE_TURNS_WEEKLY_DUMMY in features.columns)):
            
            features_daily = features[[
                FLD.BOOTSTRAP_STAGE_BEFORE,
                FLD.BOTTOM_PRICE,
                FLD.ROOF_PRICE,
                FLD.MAGIC_NINE_TURNS_DUMMY,
                FLD.MAGIC_NINE_TURNS_WEEKLY_DUMMY,
                FLD.GARIDENT_PRICE,
            ]].reset_index([1], drop=False).resample('D').last().dropna(
                subset=[FLD.BOOTSTRAP_STAGE_BEFORE],
                axis=0,
                how="all"
            )
            
            features_daily[AKA.DATETIME] = pd.to_datetime(features_daily.index)
            features_daily = features_daily.set_index(
                [AKA.DATETIME, AKA.CODE], 
                drop=True
            )
            
            # 假设self.data和features_daily已经定义好了，并且假设它们的索引结构相似
            features_daily = features_daily.reindex(
                index=list(set([*features_daily.index, *self.data.index]))
            ).sort_index()
            
            # 计算差集，找出self.data中不在features_daily中的index
            indices_to_remove = features_daily[
                ~features_daily.index.isin(self.data.index)
            ].index
            
            # 删除这些行
            features_daily_cleaned = features_daily.drop(indices_to_remove)
            
            # 确保最终DataFrame的索引是排序的
            features_daily = features_daily_cleaned.sort_index()
            
            try:
                self.data[AKA.COST95_ENDPOINT_CANDIDATE_BEFORE] = Timeline_duration(
                    np.where(
                        ((self.data[AKA.RETURN_COST95] < -0.0382) |
                         ((self.data[AKA.RETURN_COST95] < (-0.0382 / 1.68)) &
                          (self.data[FLD.MACD_DELTA] < -0.0000168)) |
                         ((self.data[AKA.RETURN_COST95] < -0.01) &
                          (self.data[FLD.MACD_DELTA] < -0.0000168) &
                          (self.data[FLD.MACD_DELTA].shift(1) < -0.0000168) &
                          (self.data[FLD.MACD] < -0.0000168) &
                          (self.data[FLD.DEA] > 0.0000168))) &
                        ~((self.data[FLD.MACD] > 0.0000168) &
                          (self.data[FLD.DEA] > 0.0000168) &
                          ((features_daily[FLD.MAGIC_NINE_TURNS_DUMMY] > 0.00168) |
                           (features_daily[FLD.MAGIC_NINE_TURNS_WEEKLY_DUMMY] > 0.00168))),
                        1, 0
                    )
                )
            except ValueError:
                log_msg = 'Code:{} features_daily {} self.data:{}'.format(
                    code, len(features_daily), len(self.data)
                )
                if ((len(features_daily) - len(self.data)) == 1):
                    if verbose:
                        print('\n', log_msg)
                        print(
                            features_daily.index.difference(self.data.index),
                            features_daily.head(3).index,
                        )
                        import traceback
                        traceback.print_exc()
                    # 最近一天的股票换手率数据未及时更新，可以忽略不进行处理。
                    pass
                else:
                    print('\n', log_msg)
                    traceback.print_exc()
            except Exception:
                print('\nCode:{} features_daily'.format(code),
                      len(features_daily), 'self.data', len(self.data))
                traceback.print_exc()
        else:
            self.data[AKA.COST95_ENDPOINT_CANDIDATE_BEFORE] = Timeline_duration(
                np.where(
                    ((self.data[AKA.RETURN_COST95] < -0.0382) |
                     ((self.data[AKA.RETURN_COST95] < (-0.0382 / 1.68)) &
                      (self.data[FLD.MACD_DELTA] < -0.0000168)) |
                     ((self.data[AKA.RETURN_COST95] < -0.01) &
                      (self.data[FLD.MACD_DELTA] < -0.0000168) &
                      (self.data[FLD.MACD_DELTA].shift(1) < -0.0000168) &
                      (self.data[FLD.MACD] < -0.0000168) &
                      (self.data[FLD.DEA] > 0.0000168))),
                    1, 0
                )
            )
        
        if verbose:
            print(
                self.data.loc[return_cost95.index,
                              [AKA.COST5_PRICE,
                               AKA.RETURN_COST95,
                               AKA.COST95_PRICE,
                               AKA.COST90_CHANNEL,
                               AKA.COST5_BOOTSTRAP_BEFORE,
                               AKA.COST95_ENDPOINT_CANDIDATE_BEFORE,]].tail(110).head(60)
                if (self.data[AKA.COST5_BOOTSTRAP_BEFORE].tail(1).item() > 63)
                else self.data.loc[return_cost95.index,
                                   [AKA.COST5_PRICE,
                                    AKA.RETURN_COST95,
                                    AKA.COST95_PRICE,
                                    AKA.COST90_CHANNEL,
                                    AKA.COST5_BOOTSTRAP_BEFORE,
                                    AKA.COST95_ENDPOINT_CANDIDATE_BEFORE,]].tail(60).head(60)
            )
        
        return return_cost5


def calc_stock_chip_distribution_jit(
    features: pd.DataFrame = None,
    ohlc_data: pd.DataFrame = None,
    annual=1008,
    verbose=False,
    *args,
    **kwargs
):
    """
    使用 Numba JIT 优化的筹码分布计算函数
    """
    # 获取股票代码（用于日志记录）
    code = kwargs['code'] if ('code' in kwargs.keys()) else features.index.get_level_values(level=1)[0]
    # annual = kwargs['annual'] if ('annual' in kwargs.keys()) else 252
    # verbose = kwargs['verbose'] if ('verbose' in kwargs.keys()) else False
    chip_dist_columns = [
        AKA.PROFIT,
        AKA.COST90,
        AKA.COST5_PRICE,
        AKA.COST15_PRICE,
        AKA.COST85_PRICE,
        AKA.COST95_PRICE,
        AKA.CHIP_DENSITY,
        AKA.RETURN_COST5,
        AKA.RETURN_COST95,
        AKA.COST90_CHANNEL,
        AKA.COST5_BOOTSTRAP_CANDIDATE,
        AKA.COST5_BOOTSTRAP_CANDIDATE_BEFORE,
        AKA.COST5_BOOTSTRAP_BEFORE,
        AKA.COST95_ENDPOINT_CANDIDATE,
        AKA.COST95_ENDPOINT_CANDIDATE_BEFORE,]
    
    # 使用 JIT 优化版本
    jit_chip = ChipDistribution()
    jit_chip.get_data(
        code=code,
        data=ohlc_data,
        verbose=verbose)  # 获取数据
    
    # 计算筹码分布
    jit_chip.calcuChip(flag=1, AC=1)
    

    # 计算获利盘
    jit_chip.data[AKA.PROFIT] = list(jit_chip.winner())
    # print(list(jit_chip.winner())[-60:], jit_chip.data[AKA.PROFIT].tail(60))
    
    # 计算成本分布
    jit_chip.data[AKA.COST5_PRICE] = list(jit_chip.cost(5))
    jit_chip.data[AKA.COST95_PRICE] = list(jit_chip.cost(95))
    jit_chip.data[AKA.COST15_PRICE] = list(jit_chip.cost(15))
    jit_chip.data[AKA.COST85_PRICE] = list(jit_chip.cost(85))

    # 筹码集中度
    jit_chip.data[AKA.CHIP_DENSITY] = (jit_chip.data[AKA.COST95_PRICE] - jit_chip.data[AKA.COST5_PRICE]) / (jit_chip.data[AKA.COST95_PRICE] + jit_chip.data[AKA.COST5_PRICE])

    try:
        return_cost5 = jit_chip.calc_cost5_bootstrap(
            features=features,
            verbose=verbose)
    except TypeError as e:
        log_msg='Code:{} length: {} run calc_cost5_bootstrap got a TypeError: {}.\n'.format(code, len(jit_chip.data), e)
        if (verbose):
            print(log_msg)
            traceback.print_exc()

    # # 计算筹码集中度
    # jit_chip.data[AKA_COST90] = ((jit_chip.data[AKA_COST95_PRICE] -
    #                               jit_chip.data[AKA_COST5_PRICE]) /
    #                              (jit_chip.data[AKA_COST95_PRICE] +
    #                               jit_chip.data[AKA_COST5_PRICE]))
    
    # 合并结果到特征数据
    if features is not None:
        features = features.reindex(
            columns=list(set([
                *features.columns,
                *chip_dist_columns,
                *[AKA.PROFIT,
                  AKA.COST5_PRICE,
                  AKA.COST95_PRICE,
                    AKA.COST15_PRICE,
                    AKA.COST85_PRICE,
                  AKA.CHIP_DENSITY,]])))
        
        if (126 < annual < 512):
            features[AKA.PROFIT] = jit_chip.data[AKA.PROFIT]
            features[AKA.COST5_PRICE] = jit_chip.data[AKA.COST5_PRICE]
            features[AKA.COST95_PRICE] = jit_chip.data[AKA.COST95_PRICE]
            features[AKA.COST15_PRICE] = jit_chip.data[AKA.COST15_PRICE]
            features[AKA.COST85_PRICE] = jit_chip.data[AKA.COST85_PRICE]
            features[AKA.CHIP_DENSITY] = jit_chip.data[AKA.CHIP_DENSITY]
        elif (512 < annual < 1680):
            if (features is None):
                print(u'\nCode: {} features is None. {}'.format(
                    code,
                    len(jit_chip.data)))

            merge_chip_dist_columns = jit_chip.data.columns.intersection(chip_dist_columns)
            features_dummy = resample_multi_frequency_indices_func(
                features,
                indices=jit_chip.data[merge_chip_dist_columns],
                frequency='30min')
            if (verbose):
                print('\n', jit_chip.data[list(set([*[FLD.TURNOVER_RATE], 
                                            *merge_chip_dist_columns]))].tail(60), '\n',
                    features_dummy[merge_chip_dist_columns].tail(60))
                
            features = features.reindex(
                columns=list(set([
                    *features.columns,
                    *chip_dist_columns,
                    *[AKA.PROFIT,
                      AKA.COST5_PRICE,
                      AKA.COST5_PRICE,
                      AKA.COST15_PRICE,
                      AKA.COST85_PRICE,
                      AKA.RETURN_COST5,
                      AKA.FINAL_PROFIT,
                      AKA.CHIP_DENSITY,
                     ]])))
            features[AKA.PROFIT] = features_dummy[AKA.PROFIT]
            features[AKA.COST5_PRICE] = features_dummy[AKA.COST5_PRICE]
            features[AKA.COST95_PRICE] = features_dummy[AKA.COST95_PRICE]
            features[AKA.COST85_PRICE] = features_dummy[AKA.COST85_PRICE]
            features[AKA.COST15_PRICE] = features_dummy[AKA.COST15_PRICE]
            features[AKA.RETURN_COST5] = features_dummy[AKA.RETURN_COST5]
            features[AKA.CHIP_DENSITY] = features_dummy[AKA.CHIP_DENSITY]
            if (AKA.RETURN_COST95 in features_dummy.columns):
                features[AKA.RETURN_COST95]=features_dummy[AKA.RETURN_COST95]
            if (AKA.COST95_ENDPOINT_CANDIDATE_BEFORE in features_dummy.columns):
                features[AKA.COST95_ENDPOINT_CANDIDATE_BEFORE]=Timeline_duration(np.where((features_dummy[AKA.COST95_ENDPOINT_CANDIDATE_BEFORE]<0.00927), 1, 0))
            elif (len(jit_chip.data)>382):
                print(u'\nCode: {} length:{} features missing column:{}.'.format(code, 
                                                                    len(jit_chip.data),
                                                                    AKA.RETURN_COST95,))
            features[AKA.COST5_BOOTSTRAP_CANDIDATE] = features_dummy[AKA.COST5_BOOTSTRAP_CANDIDATE]
            features[AKA.COST5_BOOTSTRAP_BEFORE] = Timeline_duration(np.where((features_dummy[AKA.COST5_BOOTSTRAP_BEFORE]<0.00927), 1, 0))
            features[AKA.COST5_BOOTSTRAP_CANDIDATE_BEFORE] = Timeline_duration(np.where((features_dummy[AKA.COST5_BOOTSTRAP_CANDIDATE_BEFORE]<0.00927), 1, 0))
            features[AKA.FINAL_PROFIT] = features[AKA.PROFIT].shift(4)
            features[AKA.COST90] = features_dummy[AKA.COST90]
            features[AKA.COST90_CHANNEL] = features_dummy[AKA.COST90_CHANNEL]
            if (AKA.FINAL_COST90 in features.columns):
                features[AKA.FINAL_COST90]=features[AKA.FINAL_COST90].shift(4)
        else:
            print('fooooooooooooooooooooo1', annual)
            print(features[[AKA.CLOSE, 
                            AKA.PROFIT,
                            AKA.COST5_PRICE,]].tail(60))        
            print(annual, features[AKA.PROFIT])

    # 成本价是绝对价格；同时输出归一化 DUMMY（cost / close），attach 时 * close 还原
    if AKA.CLOSE in features.columns:
        _close = pd.to_numeric(features[AKA.CLOSE], errors='coerce')
        for _base, _dummy in (
                (AKA.COST5_PRICE, AKA.COST5_PRICE_DUMMY),
                (AKA.COST15_PRICE, AKA.COST15_PRICE_DUMMY),
                (AKA.COST85_PRICE, AKA.COST85_PRICE_DUMMY),
                (AKA.COST95_PRICE, AKA.COST95_PRICE_DUMMY)):
            if _base in features.columns:
                features[_dummy] = (_close * 0 + features[_base] / _close).astype(np.float32)
    return features


if __name__ == "__main__":
    a = ChipDistribution()
    a.get_data()  # 获取数据
    a.calcuChip(flag=1, AC=1)  # 计算
    a.winner()  # 获利盘
    a.cost(90)  # 成本分布
    # a.cost(30)  # 成本分布
    a.lwinner()
