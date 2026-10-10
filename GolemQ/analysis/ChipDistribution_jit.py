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
from GolemQ.core.market_registry import get_active_market
# 老：GolemQ.fetch.StockCN.GQ_fetch_stock_ranking（读 4.4 的 stock_valuation/stock_ranking）
#     → 新：**改读本树的 `stock_metadata_day`**（换手率两列就落在那儿）。
#       这正是「先把换手率落库」的原因 —— 没有它这个模块跑不了。
from GolemQ.markets.StockCN.refdata import (
    GQ_fetch_stock_info,
    GQ_fetch_stock_metadata_day,
)
# 老：calc_feature_event_timing_lag 在 features/base.py、
#     resample_multi_frequency_indices_func / Timeline_duration 在 analysis/timeseries.py
#     → 新：三者都在 **analysis/timing.py**（随本次搬运一起落，见该模块文档）
from GolemQ.analysis.timing import (
    Timeline_duration,
    calc_feature_event_timing_lag,
    resample_multi_frequency_indices_func,
)
# 老：GolemQ.utils.base.GQ_util_get_last_day
#     → 新：真实现一直在 markets/StockCN/date_utils.py（老树 utils/base.py 那份是阉割版）
from GolemQ.markets.StockCN.date_utils import GQ_util_get_last_day
import traceback
import warnings

# ⚠️ **老树的 `import QUANTAXIS as QA` 已删** —— 它在本文件里**一处都没用到**
#    （实测 `QA.` 零命中），是个死 import。全树 QUANTAXIS 已剔除（`DECISIONS.md` D12）。
#
# ⚠️ **老树的 `GQ_stock_a_spot_em` 没有对应物**：它是「当日全市场快照」，
#    用来补**当天**缺的换手率。新树没有这个源，而换手率已落库 ⇒ 那条分支删掉，
#    走它的**兜底**（`vol / liutongguben`，用 `GQ_fetch_stock_info`，那个有）。
#    见 `_fill_today_turnover`。


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


def _calcu_sin(dateT, highT, lowT, avgT, volT, TurnoverRateT, minD, A,
               ChipList=None, Chip=None):
    """
    Original function with JIT optimized core calculation
    """
    if ChipList is None:
        ChipList = {}
    if Chip is None:
        Chip = {}

    # 使用JIT优化的核心计算
    result = _calcu_sin_jit(highT, lowT, avgT, volT, TurnoverRateT, minD, A)

    if result.shape[1] > 0:
        prices = result[0]
        tmpChip_values = result[1]

        # 创建临时筹码字典
        tmpChip = {}
        for i in range(len(prices)):
            tmpChip[prices[i]] = tmpChip_values[i]

        # 衰减现有筹码
        for price in list(Chip.keys()):
            Chip[price] = Chip[price] * (1 - TurnoverRateT * A)

        # 添加新筹码
        for price in tmpChip:
            if price in Chip:
                Chip[price] += tmpChip[price]
            else:
                Chip[price] = tmpChip[price]

    ChipList[dateT] = copy.deepcopy(Chip)

    return ChipList, Chip


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


def _calcu_jun(dateT, highT, lowT, volT, TurnoverRateT, A, minD,
               ChipList=None, Chip=None):
    """
    Original function with JIT optimized core calculation
    """
    if ChipList is None:
        ChipList = {}
    if Chip is None:
        Chip = {}

    # 使用JIT优化的核心计算
    result = _calcu_jun_jit(highT, lowT, volT, TurnoverRateT, minD, A)

    if result.shape[1] > 0:
        prices = result[0]
        tmpChip_values = result[1]

        # 衰减现有筹码
        for price in list(Chip.keys()):
            Chip[price] = Chip[price] * (1 - TurnoverRateT * A)

        # 添加新筹码
        for i in range(len(prices)):
            price = prices[i]
            chip_value = tmpChip_values[i]

            if price in Chip:
                Chip[price] += chip_value
            else:
                Chip[price] = chip_value

    ChipList[dateT] = copy.deepcopy(Chip)

    return ChipList, Chip


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

    # Reasonable maximum for chip distribution
    if price_range > 832:
        max_chips = 500000
    elif price_range > 262:
        max_chips = 100000
    elif price_range > 168:
        max_chips = 50000
    elif price_range > 83.2:
        max_chips = 20000
    elif price_range > 51.2:
        max_chips = 10000
    else:
        max_chips = 50000  # 默认值
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
                chip_counts[i] = n_unique
                chip_prices[i, :n_unique] = np.array(unique_prices)
                chip_values[i, :n_unique] = np.array(summed_values)
        else:
            # No new chips, just carry forward previous chips with decay
            if prev_count > 0:
                chip_counts[i] = prev_count
                chip_prices[i, :prev_count] = prev_prices
                chip_values[i, :prev_count] = prev_values

    return chip_prices, chip_values, chip_counts


def _calculate_chip_distribution(date, low, high, vol, TurnoverRate, avg,
                                 minD=0.01, flag=1, AC=1):
    """
    Original function interface with JIT optimized backend
    Maintains dictionary interface for compatibility
    """
    # Convert to numpy arrays for JIT processing
    low_array = np.array(low)
    high_array = np.array(high)
    vol_array = np.array(vol)
    TurnoverRate_array = np.array(TurnoverRate)
    avg_array = np.array(avg)

    # Use JIT optimized version
    chip_prices, chip_values, chip_counts = _calculate_chip_distribution_jit(
        low_array, high_array, vol_array,
        TurnoverRate_array, avg_array, minD, flag, AC
    )

    # Convert back to dictionary format for compatibility
    ChipList = {}
    Chip = {}

    n_dates = len(date)
    for i in range(n_dates):
        date_key = date[i]
        n_chips = chip_counts[i]

        chip_dict = {}
        for j in range(n_chips):
            price = chip_prices[i, j]
            value = chip_values[i, j]
            chip_dict[price] = value

        ChipList[date_key] = chip_dict

        # Last day's chips become the current Chip
        if i == n_dates - 1:
            Chip = chip_dict

    return ChipList, Chip


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


def _calcu_winner1(p, ChipList):
    """
    Original function with JIT optimized core calculation
    """
    # 准备数据用于JIT计算
    n_dates = len(ChipList)

    # 动态计算最大筹码数量
    max_chips = 0
    for chip_dict in ChipList.values():
        if chip_dict:
            max_chips = max(max_chips, len(chip_dict))

    # 如果筹码数量太多，使用合理的上限
    max_chips = min(max_chips, 10000)  # 最大10000个筹码点

    # 创建数组存储筹码数据
    prices_array = np.full((n_dates, max_chips), np.nan)
    values_array = np.full((n_dates, max_chips), np.nan)
    chip_counts = np.zeros(n_dates, dtype=np.int64)

    # 按日期顺序提取筹码数据
    sorted_dates = sorted(ChipList.keys())
    for i, date_key in enumerate(sorted_dates):
        chip_dict = ChipList[date_key]
        if chip_dict:
            # 将筹码数据转换为列表并排序，确保数据完整性
            prices = list(chip_dict.keys())
            values = list(chip_dict.values())

            n_chips = min(len(prices), max_chips)
            chip_counts[i] = n_chips

            # 填充数组
            for j in range(n_chips):
                prices_array[i, j] = prices[j]
                values_array[i, j] = values[j]

    # 准备价格数组
    if isinstance(p, pd.Series):
        p_array = p.values
    else:
        p_array = np.full(n_dates, p)

    # 使用JIT优化版本计算
    Profit = _winner_calculation_jit(prices_array, values_array, p_array)

    return Profit


def _calcu_winner2(p, ChipList):
    """
    Original function with JIT optimized core calculation
    """
    # 准备数据用于JIT计算
    n_dates = len(ChipList)

    # 动态计算最大筹码数量
    max_chips = 0
    for chip_dict in ChipList.values():
        if chip_dict:
            max_chips = max(max_chips, len(chip_dict))

    # 如果筹码数量太多，使用合理的上限
    max_chips = min(max_chips, 10000)  # 最大10000个筹码点

    # 创建数组存储筹码数据
    prices_array = np.full((n_dates, max_chips), np.nan)
    values_array = np.full((n_dates, max_chips), np.nan)
    chip_counts = np.zeros(n_dates, dtype=np.int64)

    # 按日期顺序提取筹码数据
    sorted_dates = sorted(ChipList.keys())
    for i, date_key in enumerate(sorted_dates):
        chip_dict = ChipList[date_key]
        if chip_dict:
            # 将筹码数据转换为列表并排序，确保数据完整性
            prices = list(chip_dict.keys())
            values = list(chip_dict.values())

            n_chips = min(len(prices), max_chips)
            chip_counts[i] = n_chips

            # 填充数组
            for j in range(n_chips):
                prices_array[i, j] = prices[j]
                values_array[i, j] = values[j]

    # 准备价格数组
    p_array = np.full(n_dates, p)

    # 使用JIT优化版本计算
    Profit = _winner_calculation_jit(prices_array, values_array, p_array)

    return Profit


def _calcu_cost(date, N, ChipList):
    """
    Original function with JIT optimized core calculation
    """
    target_percent = N / 100.0

    # 准备数据用于JIT计算
    n_dates = len(ChipList)

    # 动态计算最大筹码数量
    max_chips = 0
    for chip_dict in ChipList.values():
        if chip_dict:
            max_chips = max(max_chips, len(chip_dict))

    # 如果筹码数量太多，使用合理的上限
    max_chips = min(max_chips, 10000)  # 最大10000个筹码点

    # 创建数组存储筹码数据
    prices_array = np.full((n_dates, max_chips), np.nan)
    values_array = np.full((n_dates, max_chips), np.nan)

    # 按日期顺序提取筹码数据
    sorted_dates = sorted(ChipList.keys())
    for i, date_key in enumerate(sorted_dates):
        chip_dict = ChipList[date_key]
        if chip_dict:
            # 将筹码数据转换为列表并排序，确保数据完整性
            prices = list(chip_dict.keys())
            values = list(chip_dict.values())

            n_chips = min(len(prices), max_chips)

            # 填充数组
            for j in range(n_chips):
                prices_array[i, j] = prices[j]
                values_array[i, j] = values[j]

    # 使用JIT优化版本计算
    result = _cost_calculation_jit(prices_array, values_array, target_percent)

    return result


class ChipDistribution():
    """
    ChipDistribution class with Numba JIT optimized core functions
    """

    def __init__(self):
        self.Chip = {}  # 当前获利盘
        self.ChipList = {}  # 所有的获利盘
        self.adj = 1

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
        
        start = '{}'.format(
            self.data.index.get_level_values(level=0)[0] - timedelta(hours=8.5)
        )
        end = '{}'.format(
            self.data.index.get_level_values(level=0)[-1] + timedelta(hours=16)
        )

        # 老树：读 4.4 `stock_valuation`（baostock 口径的 `turnover` + `peTTM`）。
        # 新树：改读本树 `stock_metadata_day` —— `turnover` 那列在，**`peTTM` 没有**
        # （本轮只搬了换手率）。
        # ⚠️ 老树的 PE 在本模块里**只被写、从不被读**（实测 `PE_RATION` 零处读取），
        # 故 `column_list` 去掉 PE **不影响本模块**；下游若要用 PE 得另行落库。
        stock_valuation_pd = GQ_fetch_stock_metadata_day(
            code, start=start, end=end, columns=[AKA.TURNOVER]
        )

        column_list = [FLD.TURNOVER_RATE]
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

        if len(self.data.index.difference(
            self.data.dropna(
                subset=[FLD.TURNOVER_RATE], axis=0, how="all"
            ).index
        )) > 0:
            # 检查日期缺乏今天
            missing_date = self.data.index.difference(
                self.data.dropna(
                    subset=[FLD.TURNOVER_RATE], axis=0, how="all"
                ).index
            ).get_level_values(level=0)
            if (len(missing_date) == 1) and (missing_date[-1] == GQ_util_get_last_day()):
                missing_date = missing_date[-1]
                # ⚠️ 老树这里先试 `GQ_stock_a_spot_em()`（**当日全市场快照**），
                # 失败才走「vol / 流通股本」兜底。新树**没有那个源**（它是老树
                # `fetch` 那条链上的东西，已随 QUANTAXIS 解耦一起没了），
                # 而换手率现在**已落进 `stock_metadata_day`** ⇒ 缺当日的正常情形是
                # 「今天的还没入库」，走兜底按股本现算即可。
                # 故这一段**去掉快照尝试、直接走兜底**（兜底依赖的
                # `GQ_fetch_stock_info` 在新树有）。
                vol = self.data.loc[
                    (missing_date, code), "volume"].item()
                stock_info = GQ_fetch_stock_info([code[:6]])
                if stock_info is not None and 'liutongguben' in stock_info.columns:
                    total_volume = stock_info['liutongguben'].iloc[0]
                else:
                    print(stock_info)
                if total_volume and total_volume > 0:
                    turnover_rate = round((vol * 100 / total_volume) / 100, 6)
                    if (turnover_rate > 0.96):
                        turnover_rate = round((vol * 100 / total_volume) / 10000, 6)
                else:
                    turnover_rate = None
                self.data.loc[
                    (missing_date, code), FLD.TURNOVER_RATE
                ] = turnover_rate
                if verbose:
                    # ⚠️ 老树这里还打了一行 `stock_cn_snapshot...item()` —— 那个变量
                    # 随 `GQ_stock_a_spot_em` 一起去掉了，**留着会 NameError**
                    # （只在 `-v` 下才炸，更难发现）。故只保留这行诊断。
                    print(
                        '缺实盘当天换手率', missing_date, type(missing_date),
                        GQ_util_get_last_day(), type(GQ_util_get_last_day())
                    )
            else:
                tag1 = len(self.data.index.difference(
                    self.data.dropna(
                        subset=[FLD.TURNOVER_RATE], axis=0, how="all"
                    ).index))
                if (tag1 == 1):
                    missing_date = self.data.index.difference(
                        self.data.dropna(
                            subset=[FLD.TURNOVER_RATE], axis=0, how="all"
                        ).index).get_level_values(level=0)[-1]
                    vol = self.data.loc[
                        (missing_date, code), "volume"].item()
                    stock_info = GQ_fetch_stock_info([code[:6]])
                    if stock_info is not None and 'liutongguben' in stock_info.columns:
                        total_volume = stock_info['liutongguben'].iloc[0]
                    else:
                        print(stock_info)
                    if total_volume and total_volume > 0:
                        turnover_rate = round((vol * 100 / total_volume) / 100, 6)
                        if (turnover_rate > 1.0618):
                            turnover_rate = round((vol * 100 / total_volume) / 10000, 6)
                    else:
                        turnover_rate = None
                    self.data.loc[
                        (missing_date, code), FLD.TURNOVER_RATE
                    ] = turnover_rate
                else:
                    # 检查是否是连续的尾部<7天，尝试使用同样的成交量算法计算填补
                    missing_idx = self.data.index.difference(
                        self.data.dropna(
                            subset=[FLD.TURNOVER_RATE], axis=0, how="all"
                        ).index
                    )
                    
                    # 获取缺失日期的日期部分
                    missing_dates = missing_idx.get_level_values(level=0)
                    
                    # 检查是否是连续的尾部
                    if len(missing_dates) > 0:
                        # 检查缺失日期是否是最后几个连续的日期
                        is_tail_consecutive = True
                        for i in range(len(missing_dates) - 1):
                            # 检查日期是否连续（按顺序）
                            if missing_dates[i] + pd.Timedelta(days=1) != missing_dates[i + 1]:
                                is_tail_consecutive = False
                                break
                        
                        # 检查是否确实是尾部（缺失日期之后没有非缺失日期）
                        if is_tail_consecutive and len(missing_dates) < 7:
                            # 获取缺失日期之前的最后一个非缺失日期
                            non_missing_idx = self.data.dropna(
                                subset=[FLD.TURNOVER_RATE], axis=0, how="all"
                            ).index
                            if len(non_missing_idx) > 0:
                                # 获取股票流通股本
                                stock_info = GQ_fetch_stock_info([code[:6]])
                                if stock_info is not None and 'liutongguben' in stock_info.columns:
                                    total_volume = stock_info['liutongguben'].iloc[0]
                                    
                                    if total_volume and total_volume > 0:
                                        for missing_date in missing_dates:
                                            vol = self.data.loc[
                                                (missing_date, code), "volume"
                                            ].item()
                                            turnover_rate = round((vol * 100 / total_volume) / 100, 6)
                                            if (turnover_rate > 1.0618):
                                                turnover_rate = round((vol * 100 / total_volume) / 10000, 6)
                                            
                                            self.data.loc[
                                                (missing_date, code), FLD.TURNOVER_RATE
                                            ] = turnover_rate
                                        if (verbose):
                                            print(f'已填补 {len(missing_dates)} 天的换手率')
                                    else:
                                        print('无法获取有效的流通股本，跳过填补')
                                else:
                                    print('无法获取股票信息，跳过填补')
                        else:
                            print(
                                f'缺 {tag1} ()>7) 天的换手率(ChipDistribution_jit)',
                                self.data.index.difference(
                                    self.data.dropna(
                                        subset=[FLD.TURNOVER_RATE], axis=0, how="all"
                                    ).index),
                                f"首:{self.data.head(1).index.get_level_values(level=0)[0]} 尾:{self.data.tail(1).index.get_level_values(level=0)[0]}")

        if len(self.data.index.difference(
            self.data.dropna(
                subset=[FLD.TURNOVER_RATE], axis=0, how="all"
            ).index
        )) > 1:
            # 老树：读 4.4 `stock_ranking`（**东财口径**，旧树筹码分布吃的就是它）。
            # 新树：同一张 `stock_metadata_day` 的 `TurnoverRate` 列 —— 列名与源端
            # 一致（用户 2026-10-10 定），所以下面那些 `FLD.TURNOVER_RATE` 判断
            # 一行都不用改。
            stock_ranking_pd = GQ_fetch_stock_metadata_day(
                code,
                start='{}'.format(start)[:10],
                end='{}'.format(end)[:10],
                columns=[FLD.TURNOVER_RATE],
            )
            if stock_ranking_pd is not None:
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
        
        self.ChipList, self.Chip = _calculate_chip_distribution(
            date, low, high, vol, TurnoverRate, avg, minD=0.01, flag=flag, AC=AC
        )
        
    def winner(self, p=None):
        """
        计算获利盘比例 - JIT优化版本
        """
        if p is None:
            p = self.data['close']
        
        if isinstance(p, pd.Series):  # 不输入默认close
            Profit = list(_calcu_winner1(
                p=p,
                ChipList=self.ChipList,
            ))
        else:
            Profit = list(_calcu_winner2(
                p=p,
                ChipList=self.ChipList,
            ))
            
        return Profit

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
        
        result = _calcu_cost(date=date, N=N, ChipList=self.ChipList)
                    
        return result*self.adj

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

    return features


if __name__ == "__main__":
    a = ChipDistribution()
    a.get_data()  # 获取数据
    a.calcuChip(flag=1, AC=1)  # 计算
    a.winner()  # 获利盘
    a.cost(90)  # 成本分布
    # a.cost(30)  # 成本分布
    a.lwinner()
