# coding:utf-8
# Author: 阿财（Rgveda@github）（4910163#qq.com）
# Created date: 2020-02-27
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
import numpy as np
import pandas as pd
import numba as nb
import scipy.optimize as opt

try:
    import talib
except:
    pass
    #print('PLEASE install TALIB to call these methods')
try:
    import QUANTAXIS as QA
    from QUANTAXIS.QAIndicator.talib_numpy import *
except:
    print('PLEASE run "pip install QUANTAXIS" before call GolemQ.GQIndicator.renko modules')
    pass

from GolemQ.GQUtil.parameter import (
    AKA, 
    INDICATOR_FIELD as FLD, 
    TREND_STATUS as ST
)
from GolemQ.GQAnalysis.timeseries import *

class renko:
    """
    Renko Chart/Renko Brick/Renko Bar
    A renko chart is a type of financial chart of Japanese origin used in 
    technical analysis that measures and plots price changes.
    Renko砖块图是一种由日本人发明的用于衡量和绘制价格变化的财务技術分析图表。Renko图由砖块
    组成，与典型的蜡烛图相比，它可以通过剔除噪音的方式，帮助投资者更好的分析趋势和进行量化研究，
    更清晰地显示市场趋势并提高信噪比。
    """
    def __init__(self, hlc=None):
        """
        Inti data, 
        params
        ---------
        hlc:ndarray | optial, with hlc/ohlc price data

        初始化数据
        可选参数---
        hlc: ndarray | optial, 包含 3/4 列 hlc/ohlc 价格信息
        """
        self.source_prices = []
        self.renko_prices = []
        self.renko_directions = []

        self.source_aligned = []
        self.renko_gaps = []

        # 增加上下影线
        # For upper lower shadow
        self.renko_upper_shadow = []
        self.renko_lower_shadow = []

        if ((hlc is not None) and (len(hlc) > 0)):
            if (len(hlc[0, :]) == 3):
                self.source_hlc = hlc
            elif (len(hlc[0, :]) >= 4):
                self.source_hlc = hlc[:, [1, 2, 3]]
            else:
                print(u'renko(hlc) must has 3~4 columns data with high, low, close price.')
                self.source_hlc = []
        else:
            self.source_hlc = []
        
    def set_brick_size(self, HLC_history=None, auto=True, brick_size=10.0):
        """
        Setting brick size.  Auto mode is preferred, it uses history
        """
        if (isinstance(HLC_history, np.ndarray)):
            HLC_history = HLC_history[:, [0, 1, 2]]
        elif (isinstance(HLC_history, pd.DataFrame)):
            HLC_history = HLC_history.iloc[:, [0, 1, 2]].values

        if auto == True:
            self.brick_size = self.__get_optimal_brick_size(HLC_history)
        else:
            self.brick_size = brick_size
        return self.brick_size
    
    def __renko_rule(self, last_price, aligned_idx=0):
        """
        Renko brick increasing rule
        """
        # Get the gap between two prices
        gap_div = int(float(last_price - self.renko_prices[-1]) / self.brick_size)
        is_new_brick = False
        start_brick = 0
        num_new_bars = 0

        # When we have some gap in prices
        if gap_div != 0:
            # Forward any direction (up or down)
            if (gap_div > 0 and (self.renko_directions[-1] > 0 or self.renko_directions[-1] == 0)) or (gap_div < 0 and (self.renko_directions[-1] < 0 or self.renko_directions[-1] == 0)):
                num_new_bars = gap_div
                is_new_brick = True
                start_brick = 0
            # Backward direction (up -> down or down -> up)
            elif np.abs(gap_div) >= 2: # Should be double gap at least
                num_new_bars = gap_div
                num_new_bars -= np.sign(gap_div)
                start_brick = 2
                is_new_brick = True

                next_price = self.renko_prices[-1] + 2 * self.brick_size * np.sign(gap_div)
                self.renko_prices.append(next_price)
                self.renko_directions.append(np.sign(gap_div))

                # 记录上下影线
                if (len(self.source_hlc) > 0):
                    self.renko_upper_shadow.append(self.source_hlc[aligned_idx, 0])
                    self.renko_lower_shadow.append(self.source_hlc[aligned_idx, 1])
                self.renko_gaps.append(aligned_idx)
            #else:
                #num_new_bars = 0

            if is_new_brick:
                # Add each brick
                for d in range(start_brick, np.abs(gap_div)):
                    next_price = self.renko_prices[-1] + self.brick_size * np.sign(gap_div)
                    self.renko_prices.append(next_price)
                    self.renko_directions.append(np.sign(gap_div))

                    # 记录上下影线
                    if (len(self.source_hlc) > 0):
                        self.renko_upper_shadow.append(self.source_hlc[aligned_idx, 0])
                        self.renko_lower_shadow.append(self.source_hlc[aligned_idx, 1])
                    self.renko_gaps.append(aligned_idx)

        return num_new_bars
    
    def build_history(self, prices=None, hlc=None):
        """
        Getting renko on history
        生成 Renko brick 序列（非时间顺序，按时间顺序的 Renko 序列在）
        """
        if ((hlc is not None) and (len(hlc) > 0)):
            if (len(hlc[0, :]) == 3):
                prices = hlc[:, 2]
                self.source_hlc = hlc
            elif (len(hlc[0, :]) >= 4):
                prices = hlc[:, 3]
                self.source_hlc = hlc[:, [1, 2, 3]]
            else:
                print(u'renko.build_history(hlc) must has 3~4 columns data with high, low, close price.')

        if len(prices) > 0:
            # Init by start values
            self.source_prices = prices
            self.renko_prices.append(prices[0])
            self.renko_directions.append(0)

            # 记录上下影线
            if (len(self.source_hlc) > 0):
                self.renko_upper_shadow.append(self.source_hlc[0, 0])
                self.renko_lower_shadow.append(self.source_hlc[0, 1])

            self.source_aligned = np.empty((len(prices),3))
        
            # For each price in history
            idx = 1
            for p in self.source_prices[1:]:
                ret_new = self.__renko_rule(p, idx)

                # 同步Renko线到真实K线时间
                # Align Renko Price to real ohlc‘s xlim
                if (len(self.renko_prices) <= 1) or \
                    ((len(self.renko_prices) > 1) and (abs(self.renko_prices[-1] - self.renko_prices[-2]) == self.brick_size)):
                    self.source_aligned[idx, 0] = self.renko_prices[-1] - self.brick_size if self.renko_directions[-1] == 1 else self.renko_prices[-1]
                    self.source_aligned[idx, 1] = self.renko_prices[-1] if self.renko_directions[-1] == 1 else self.renko_prices[-1] - self.brick_size
                else:
                    # 跳空
                    self.source_aligned[idx, 0] = self.renko_prices[-1] + self.brick_size if self.renko_directions[-1] == 1 else self.renko_prices[-1]
                    self.source_aligned[idx, 1] = self.renko_prices[-1] if self.renko_directions[-1] == 1 else self.renko_prices[-1] + self.brick_size
                self.source_aligned[idx, 2] = self.renko_directions[-1]

                # 记录上下影线
                if (len(self.source_hlc) > 0):
                    self.renko_upper_shadow[-1] = self.renko_upper_shadow[-1] if (self.renko_upper_shadow[-1] > self.source_hlc[idx, 0]) else self.source_hlc[idx, 0]
                    self.renko_lower_shadow[-1] = self.renko_lower_shadow[-1] if (self.renko_lower_shadow[-1] < self.source_hlc[idx, 1]) else self.source_hlc[idx, 1]

                idx = idx + 1

        return len(self.renko_prices)
    
    # Getting next renko value for last price
    def do_next(self, last_price):
        if len(self.renko_prices) == 0:
            self.source_prices.append(last_price)
            self.renko_prices.append(last_price)
            self.renko_directions.append(0)
            return 1
        else:
            self.source_prices.append(last_price)
            return self.__renko_rule(last_price)
    
    # Simple method to get optimal brick size based on ATR
    def __get_optimal_brick_size(self, HLC_history, atr_timeperiod=14):
        brick_size = 0.0
        
        # If we have enough of data
        if HLC_history.shape[0] > atr_timeperiod:
            brick_size = np.median(talib.ATR(high = np.double(HLC_history[:, 0]), 
                                             low = np.double(HLC_history[:, 1]), 
                                             close = np.double(HLC_history[:, 2]), 
                                             timeperiod = atr_timeperiod)[atr_timeperiod:])
        
        return brick_size

    def evaluate(self, method='simple'):
        balance = 0
        sign_changes = 0
        price_ratio = len(self.source_prices) / len(self.renko_prices)

        if method == 'simple':
            for i in range(2, len(self.renko_directions)):
                if self.renko_directions[i] == self.renko_directions[i - 1]:
                    balance = balance + 1
                else:
                    balance = balance - 2
                    sign_changes = sign_changes + 1

            if sign_changes == 0:
                sign_changes = 1

            score = balance / sign_changes
            if score >= 0 and price_ratio >= 1:
                score = np.log(score + 1) * np.log(price_ratio)
            else:
                score = -1.0

            return {'balance': balance, 'sign_changes:': sign_changes, 
                    'price_ratio': price_ratio, 'score': score}
    
    def get_renko_prices(self):
        """
        返回每个 Renko bar 的价格
        """
        return self.renko_prices
    
    def get_renko_directions(self):
        """
        返回每个 Renko bar 的方向
        """
        return self.renko_directions

    def get_renko_upper_shadow(self):
        """
        返回每个 Renko bar 的上影线
        """
        return self.renko_upper_shadow

    def get_renko_lower_shadow(self):
        """
        返回每个 Renko bar 的下影线
        """
        return self.renko_lower_shadow

    def get_renko_gaps(self):
        """
        返回每个 Renko bar 的原始时间轴坐标起点
        """
        return self.renko_gaps
    
    def get_source_aligned(self):
        """
        返回时间轴对齐原始 OHLC 的 Renko Bars
        """
        return self.source_aligned
 
    def plot_renko(self, ohlc=None, col_up='g', col_down='r'):
        """
        没事画个图
        """
        fig, ax = plt.subplots(1, figsize=(20, 10))
        ax.set_title('Renko chart')
        ax.set_xlabel('Renko bars')
        ax.set_ylabel('Price')

        if (ohlc is not None):
            #ax_ohlc = ax.twinx()
            mpf.candlestick2_ochl(ax,
                                  ohlc.open,
                                  ohlc.close,
                                  ohlc.high,
                                  ohlc.low,
                                  width=0.6, colorup='r', colordown='green',
                                  alpha=0.5)

            # Set datetime as y-label for MultiIndex
            DATETIME_LABEL = ohlc.index.get_level_values(level=0).to_series().apply(lambda x: x.strftime("%Y-%m-%d %H:%M")[2:16])
            # Set datetime as y-label for DatetimIndex
            #DATETIME_LABEL = ohlc.index.to_series().apply(lambda x:
            #x.strftime("%Y-%m-%d %H:%M")[2:16])
            ax.set_xticks(range(0, len(DATETIME_LABEL), round(len(ohlc) / 16)))
            ax.set_xticklabels(DATETIME_LABEL[::round(len(ohlc) / 16)], rotation = 10)
            ax.grid(True)

            ax.plot(DATETIME_LABEL, 
                    self.source_aligned[:, 0] , 
                    lw=0.75, 
                    color='cyan', 
                    alpha=0.6)
            ax.plot(DATETIME_LABEL, 
                    self.source_aligned[:, 1] ,
                    lw=0.75, 
                    color='fuchsia', 
                    alpha=0.6)
            ax.fill_between(DATETIME_LABEL, 
                             np.where(self.source_aligned[:, 2] == 1, 
                                      self.source_aligned[:, 0], 
                                      np.nan), 
                             np.where(self.source_aligned[:, 2] == 1, 
                                      self.source_aligned[:, 1], 
                                      np.nan), 
                             color='lightcoral', alpha=0.15)
            ax.fill_between(DATETIME_LABEL, 
                             np.where(self.source_aligned[:, 2] == -1, 
                                      self.source_aligned[:, 0], 
                                      np.nan), 
                             np.where(self.source_aligned[:, 2] == -1, 
                                      self.source_aligned[:, 1], 
                                      np.nan), 
                             color='limegreen', alpha=0.15)
        else:
            # Calculate the limits of axes
            ax.set_xlim(0.0,
                        len(self.renko_prices) + 1.0)
            ax.set_ylim(np.min(self.renko_prices) - 3.0 * self.brick_size,
                        np.max(self.renko_prices) + 3.0 * self.brick_size)
        
            # Plot each renko bar
            for i in range(1, len(self.renko_prices)):
                # Set basic params for patch rectangle
                col = col_up if self.renko_directions[i] == 1 else col_down
                x = i
                y = self.renko_prices[i] - self.brick_size if self.renko_directions[i] == 1 else self.renko_prices[i]
                height = self.brick_size
                
                # Draw bar with params
                ax.add_patch(patches.Rectangle((x, y), # (x,y)
                        1.0, # width
                        self.brick_size, # height
                        facecolor = col))


# Function for optimization
def evaluate_renko(brick, history, column_name):
    renko_obj = renko()
    renko_obj.set_brick_size(brick_size = brick, auto = False)
    renko_obj.build_history(prices = history)
    return renko_obj.evaluate()[column_name]


@nb.jit(nopython=True)
def renko_chart(price_series, N, condensed=True):
    """
    原版代码来自QUANTAXIS QA.RENKO，经过numba jit优化，最终输出格式统一为
    2维数组，3列数据：Lower Band/Upper Band/Direction
    """
    idx_renko_price = 0
    idx_renko_direction = 1
    idx_renko_lb = 2
    idx_renko_ub = 3
    ret_renko_chart = np.empty((len(price_series), 4,))

    last_price = price_series[0]
    chart = [last_price]
    for i in range(0, len(price_series)):
        price = price_series[i]
        bricks = math.floor(abs(price - last_price) / N)
        if bricks == 0:
            if condensed:
                ret_renko_chart[i, idx_renko_price] = chart[-1]
                chart.append(chart[-1])
            continue
        sign = int(np.sign(price - last_price))
        #print(price_series[i-1], price_series[i], [sign * (last_price + (sign
        #* N * x)) for x in range(1, bricks + 1)])
        chart += [sign * (last_price + (sign * N * x)) for x in range(1, bricks + 1)]
        ret_renko_chart[i, idx_renko_price] = chart[-1]
        last_price = abs(chart[-1])

    ret_renko_chart[:, idx_renko_direction] = np.sign(ret_renko_chart[:, idx_renko_price])
    ret_renko_chart[:, idx_renko_lb] = np.where(ret_renko_chart[:, idx_renko_direction] < 0,
                                                -ret_renko_chart[:, idx_renko_price] - N,
                                                ret_renko_chart[:, idx_renko_price])
    ret_renko_chart[:, idx_renko_ub] = np.where(ret_renko_chart[:, idx_renko_direction] > 0,
                                                ret_renko_chart[:, idx_renko_price] + N,
                                                -ret_renko_chart[:, idx_renko_price])

    return ret_renko_chart


@nb.jit(nopython=True)
def RENKOP(price_series, N, condensed=True):
    last_price = price_series[0]
    chart = [last_price, last_price]
    for price in price_series:
        inc = (price - last_price) / last_price
        #print(inc)
        if abs(inc) < N:
            # if condensed:
            #     chart.append(chart[-1])
            continue

        sign = int(np.sign(price - last_price))
        bricks = math.floor(inc / N)
        #print(bricks)
        #print((N * (price-last_price)) / inc)
        step = math.floor((N * (price - last_price)) / inc)
        print(step)
        #print(sign)
        chart += [sign * (last_price + (sign * step * x))
                  for x in range(1, abs(bricks) + 1)]
        last_price = abs(chart[-1])
    return np.array(chart)


def renko_in_cluster_group(data:pd.DataFrame, 
                           indices:pd.DataFrame=None,) -> np.ndarray:
    """
     (假设)我们对这条行情的走势一无所知，使用机器学习可以快速的识别出走势，
     划分出波浪。renko bricks 将整条行情大致分块，这个随着时间变化会有轻微抖动。
     所以不适合做精确买卖点控制。但是作为趋势判断已经足够了。
    """
    # Get ATR values (it needs to get boundaries)
    # Drop NaNs
    factor_atr = talib.ATR(high = np.double(data.high),
                           low = np.double(data.low),
                           close = np.double(data.close),
                           timeperiod = 14)
    factor_atr = factor_atr[np.isnan(factor_atr) == False]

    sub_offest = 0
    nextsub_first = sub_first = 0
    nextsub_last = sub_last = 1199
    
    ret_cluster_group = np.zeros((len(data.close.values), 5),)
    while (sub_first == 0) or (sub_first < len(data.close.values)):
        # 数量大于1200bar的话无监督聚类效果会变差，应该控制在1000~1500bar之间
        if (len(data.close.values) > 1200):
            highp = np.nan_to_num(data.high.values[sub_first:sub_last], nan=0)
            lowp = np.nan_to_num(data.low.values[sub_first:sub_last], nan=0)
            openp = np.nan_to_num(data.open.values[sub_first:sub_last], nan=0)
            closep = np.nan_to_num(data.close.values[sub_first:sub_last], nan=0)
            subdata = data.iloc[sub_first:sub_last, :]
            atr = factor_atr[sub_first:sub_last]
            if (sub_last + 1200 < len(data.close.values)):
                nextsub_first = sub_first + 1199
                nextsub_last = sub_last + 1200
            else:
                nextsub_first = len(data.close.values) - 1200
                nextsub_first = 0 if (nextsub_first < 0) else nextsub_first
                nextsub_last = len(data.close.values) + 1
        else:
            highp = data.high.values
            lowp = data.low.values
            openp = data.open.values
            closep = data.close.values
            sub_last = nextsub_first = len(data.close.values)
            subdata = data
            atr = factor_atr
            nextsub_last = len(data.close.values) + 1

        if (ST.VERBOSE in data.columns):
            print('slice:', {sub_first, sub_last}, 'total:{}'.format(len(data.close.values)))

        # Get optimal brick size as maximum of score function by Brent's (or
        # similar) method
        # First and Last ATR values are used as the boundaries
        optimal_brick_sfo = opt.fminbound(lambda x: 
                                          -evaluate_renko(brick = x, 
                                                          history = closep, 
                                                          column_name = 'score'), 
                                          np.min(atr), np.max(atr), disp=0)

        # Build Renko chart
        bricks_flex = renko_chart(subdata[AKA.CLOSE].values, optimal_brick_sfo)

        ret_cluster_group[sub_first + 1, 4] = optimal_brick_sfo

        if len(bricks_flex[:, 0]) > 1:
             ret_cluster_group[sub_first:sub_last, 0:4] = bricks_flex
        
        if (sub_last >= len(data.close.values)):
            break
        else:
            sub_first = min(nextsub_first, nextsub_last)
            sub_last = max(nextsub_first, nextsub_last)
        # renko brick 分析完毕

    return ret_cluster_group


def renko_trend_cross_func(data, indices=None):
    """
    使用 Renko brick 砖块图进行趋势判断
    """
    if (ST.VERBOSE in data.columns):
        print('Phase renko_trend_cross_func', QA_util_timestamp_to_str())

    if (len(data) < 30):
        # 数量太少，返回个空值DataFrame
        if (indices is not None):
            ret_indices = pd.concat([indices,
                                    pd.DataFrame(columns=[FLD.RENKO_TREND_S_LB, 
                                                          FLD.RENKO_TREND_S_UB, 
                                                          FLD.RENKO_TREND_S,
                                                          FLD.RENKO_TREND_L_LB, 
                                                          FLD.RENKO_TREND_L_UB, 
                                                          FLD.RENKO_TREND_L,
                                                          FLD.RENKO_TREND,
                                                          FLD.RENKO_TREND_S_BEFORE,
                                                          FLD.RENKO_S_JX_BEFORE,
                                                          FLD.RENKO_TREND_L_BEFORE,
                                                          FLD.RENKO_TREND_L_JX_BEFORE], 
                                                 index=data.index)], axis=1)
        else:
            ret_indices = pd.DataFrame(columns=[FLD.RENKO_TREND_S_LB, 
                                                FLD.RENKO_TREND_S_UB, 
                                                FLD.RENKO_TREND_S,
                                                FLD.RENKO_TREND_L_LB, 
                                                FLD.RENKO_TREND_L_UB, 
                                                FLD.RENKO_TREND_L,
                                                FLD.RENKO_TREND,
                                                FLD.RENKO_TREND_S_BEFORE,
                                                FLD.RENKO_S_JX_BEFORE,
                                                FLD.RENKO_TREND_L_BEFORE,
                                                FLD.RENKO_TREND_L_JX_BEFORE], 
                                       index=data.index)
        return ret_indices

    # Get optimal brick size based
    optimal_brick = renko().set_brick_size(auto = True, 
                                           HLC_history = data[[AKA.HIGH, 
                                                               AKA.LOW, 
                                                               AKA.CLOSE,]])

    # Build Renko chart
    bricks_fixed = renko_chart(data.close.values, optimal_brick)

    if len(bricks_fixed[:, 0]) > 1:
        if (indices is not None):
            ret_indices = pd.concat([indices,
                                    pd.DataFrame(bricks_fixed, 
                                                 columns=[FLD.RENKO_PRICE_S, 
                                                          FLD.RENKO_TREND_S,
                                                          FLD.RENKO_TREND_S_LB, 
                                                          FLD.RENKO_TREND_S_UB], 
                                                 index=data.index)]
                                    , axis=1)
        else:
            ret_indices = pd.DataFrame(bricks_fixed,
                                       columns=[FLD.RENKO_PRICE_S,
                                                FLD.RENKO_TREND_S,
                                                FLD.RENKO_TREND_S_LB, 
                                                FLD.RENKO_TREND_S_UB], 
                                       index=data.index)

    if (ret_indices is not None):
        ret_indices = pd.concat([ret_indices,
                                 pd.DataFrame(renko_in_cluster_group(data, indices),
                                   columns=[FLD.RENKO_PRICE_L,
                                            FLD.RENKO_TREND_L,
                                            FLD.RENKO_TREND_L_LB, 
                                            FLD.RENKO_TREND_L_UB, 
                                            FLD.RENKO_OPTIMAL,], 
                                   index=data.index)], axis=1)
    else:
        ret_indices = pd.DataFrame(renko_in_cluster_group(data, indices),
                                   columns=[FLD.RENKO_PRICE_L,
                                            FLD.RENKO_TREND_L,
                                            FLD.RENKO_TREND_L_LB, 
                                            FLD.RENKO_TREND_L_UB, 
                                            FLD.RENKO_OPTIMAL,], 
                                   index=data.index)
    
    ret_indices[FLD.RENKO_OPTIMAL] = np.where(np.isnan(ret_indices[FLD.RENKO_OPTIMAL]), 
                                              optimal_brick, 
                                              ret_indices[FLD.RENKO_OPTIMAL])

    ret_indices[FLD.RENKO_TREND] = np.where((ret_indices[FLD.RENKO_TREND_L] == 1) & \
                                            (ret_indices[FLD.RENKO_TREND_S] == 1), 1, 
                                            np .where((ret_indices[FLD.RENKO_TREND_L] == -1) & \
                                                     (ret_indices[FLD.RENKO_TREND_S] == -1), -1, 0))
    ret_indices[FLD.RENKO_TREND_S_BEFORE] = Timeline_duration(np.where((ret_indices[FLD.RENKO_TREND_S] == 1) & \
                                                                       (ret_indices[FLD.RENKO_TREND_S].shift(1) != 1), 1, 
                                                                       np.where((ret_indices[FLD.RENKO_TREND_S] == -1) & \
                                                                       (ret_indices[FLD.RENKO_TREND_S].shift(1) != -1), 1, 0)))
    ret_indices[FLD.RENKO_S_JX_BEFORE] = Timeline_duration(np.where((ret_indices[FLD.RENKO_TREND] >= 0) & \
                                                                    (indices[FLD.MAXFACTOR_CROSS] == 1) & \
                                                                    (indices[FLD.MACD_DELTA] > 0) & \
                                                                    (indices[FLD.BOLL_CROSS_JX_BEFORE] < indices[FLD.BOLL_CROSS_SX_BEFORE]) & \
                                                                    (ret_indices[FLD.RENKO_TREND_S_LB] > data[AKA.CLOSE]), 1, 0))
    ret_indices[FLD.RENKO_S_SX_BEFORE] = Timeline_duration(np.where((indices[FLD.MAXFACTOR_CROSS] == -1) & \
                                                                    (indices[FLD.MACD_DELTA] < 0) & \
                                                                    (indices[FLD.BOLL_CROSS_SX_BEFORE] < indices[FLD.BOLL_CROSS_JX_BEFORE]) & \
                                                                    (ret_indices[FLD.RENKO_TREND_S_UB] < data[AKA.CLOSE]), 1, 0))
    ret_indices[FLD.RENKO_TREND_L_BEFORE] = Timeline_duration(np.where((ret_indices[FLD.RENKO_TREND_L] == 1) & \
                                                                       (ret_indices[FLD.RENKO_TREND_L].shift(1) != 1), 1, 
                                                              np.where((ret_indices[FLD.RENKO_TREND_L] == -1) & \
                                                                       (ret_indices[FLD.RENKO_TREND_L].shift(1) != -1), 1, 0)))
    ret_indices[FLD.RENKO_TREND_L_JX_BEFORE] = Timeline_duration(np.where((ret_indices[FLD.RENKO_TREND] >= 0) & \
                                                                          (indices[FLD.MAXFACTOR_CROSS] == 1) & \
                                                                          (indices[FLD.MACD_DELTA] > 0) & \
                                                                          (indices[FLD.BOLL_CROSS_JX_BEFORE] < indices[FLD.BOLL_CROSS_SX_BEFORE]) & \
                                                                          (ret_indices[FLD.RENKO_TREND_L_LB] > data[AKA.CLOSE]), 1, 0))

    with np.errstate(invalid='ignore', divide='ignore'):
        renko_trend_l_jx = Timeline_Integral(np.where(ret_indices[FLD.RENKO_TREND_L] > 0, 1, 0))
        renko_trend_l_sx = np.sign(ret_indices[FLD.RENKO_TREND_L]) * Timeline_Integral(np.where(ret_indices[FLD.RENKO_TREND_L] < 0, 1, 0))
        ret_indices[FLD.RENKO_TREND_L_TIMING_LAG] = renko_trend_l_jx + renko_trend_l_sx

    renko_trend_s_jx = Timeline_Integral(np.where(ret_indices[FLD.RENKO_TREND_S] > 0, 1, 0))
    renko_trend_s_sx = np.sign(ret_indices[FLD.RENKO_TREND_S]) * Timeline_Integral(np.where(ret_indices[FLD.RENKO_TREND_S] < 0, 1, 0))
    ret_indices[FLD.RENKO_TREND_S_TIMING_LAG] = renko_trend_s_jx + renko_trend_s_sx

    renko_boost_s_jx = Timeline_duration(np.where(ret_indices[FLD.RENKO_TREND_S_UB] > ret_indices[FLD.RENKO_TREND_S_UB].shift(), 1, 0)) + 1
    renko_boost_s_sx = Timeline_duration(np.where(ret_indices[FLD.RENKO_TREND_S_LB] < ret_indices[FLD.RENKO_TREND_S_LB].shift(), 1, 0)) + 1
    ret_indices[FLD.RENKO_BOOST_S_TIMING_LAG] = np.where(ret_indices[FLD.RENKO_TREND_S_TIMING_LAG] > 0,
                                                         renko_boost_s_jx,
                                                         np.where(ret_indices[FLD.RENKO_TREND_S_TIMING_LAG] < 0,
                                                         -renko_boost_s_sx, 0))

    renko_boost_l_jx = Timeline_duration(np.where(ret_indices[FLD.RENKO_TREND_L_UB] > ret_indices[FLD.RENKO_TREND_L_UB].shift(), 1, 0)) + 1
    renko_boost_l_sx = Timeline_duration(np.where(ret_indices[FLD.RENKO_TREND_L_LB] < ret_indices[FLD.RENKO_TREND_L_LB].shift(), 1, 0)) + 1
    ret_indices[FLD.RENKO_BOOST_L_TIMING_LAG] = np.where(ret_indices[FLD.RENKO_TREND_L_TIMING_LAG] > 0,
                                                         renko_boost_l_jx,
                                                         np.where(ret_indices[FLD.RENKO_TREND_L_TIMING_LAG] < 0,
                                                         -renko_boost_l_sx, 0))

    if (ST.VERBOSE in data.columns):
        print('Phase renko_trend_cross_func Done', QA_util_timestamp_to_str())

    return ret_indices


def renko_trend_cross_old_func(data, indices=None):
    """
    使用 Renko brick 砖块图进行趋势判断
    """
    if (len(data) < 30):
        # 数量太少，返回个空值DataFrame
        if (indices is not None):
            ret_indices = pd.concat([indices,
                                    pd.DataFrame(columns=[FLD.RENKO_TREND_S_LB, 
                                                          FLD.RENKO_TREND_S_UB, 
                                                          FLD.RENKO_TREND_S,
                                                          FLD.RENKO_TREND_L_LB, 
                                                          FLD.RENKO_TREND_L_UB, 
                                                          FLD.RENKO_TREND_L,
                                                          FLD.RENKO_TREND,
                                                          FLD.RENKO_TREND_S_BEFORE,
                                                          FLD.RENKO_S_JX_BEFORE,
                                                          FLD.RENKO_TREND_L_BEFORE,
                                                          FLD.RENKO_TREND_L_JX_BEFORE], 
                                                 index=data.index)]
                                    , axis=1)
        else:
            ret_indices = pd.DataFrame(columns=[FLD.RENKO_TREND_S_LB, 
                                                FLD.RENKO_TREND_S_UB, 
                                                FLD.RENKO_TREND_S,
                                                FLD.RENKO_TREND_L_LB, 
                                                FLD.RENKO_TREND_L_UB, 
                                                FLD.RENKO_TREND_L,
                                                FLD.RENKO_TREND,
                                                FLD.RENKO_TREND_S_BEFORE,
                                                FLD.RENKO_S_JX_BEFORE,
                                                FLD.RENKO_TREND_L_BEFORE,
                                                FLD.RENKO_TREND_L_JX_BEFORE], 
                                       index=data.index)
        return ret_indices

    # Get optimal brick size based
    optimal_brick = renko().set_brick_size(auto = True, 
                                           HLC_history = data[[AKA.HIGH, 
                                                               AKA.LOW, 
                                                               AKA.CLOSE,]])

    # Build Renko chart
    renko_obj_atr = renko()
    renko_obj_atr.set_brick_size(auto = False, brick_size = optimal_brick)
    renko_obj_atr.build_history(hlc = data[[AKA.HIGH, 
                                            AKA.LOW, 
                                            AKA.CLOSE,]].values)
    #bricks_fixed = QA.RENKO(data.close, optimal_brick)
    #print(bricks_fixed)
    #print(renko_obj_atr.source_aligned)
    if len(renko_obj_atr.get_renko_prices()) > 1:
        if (indices is not None):
            ret_indices = pd.concat([indices,
                                    pd.DataFrame(renko_obj_atr.source_aligned, 
                                                 columns=[FLD.RENKO_TREND_S_LB, 
                                                          FLD.RENKO_TREND_S_UB, 
                                                          FLD.RENKO_TREND_S], 
                                                 index=data.index)]
                                    , axis=1)
        else:
            ret_indices = pd.DataFrame(renko_obj_atr.source_aligned, 
                                       columns=[FLD.RENKO_TREND_S_LB, 
                                                FLD.RENKO_TREND_S_UB, 
                                                FLD.RENKO_TREND_S], 
                                       index=data.index)

    # Get ATR values (it needs to get boundaries)
    # Drop NaNs
    atr = talib.ATR(high = np.double(data.high),
                    low = np.double(data.low),
                    close = np.double(data.close),
                    timeperiod = 14)
    atr = atr[np.isnan(atr) == False]

    # Get optimal brick size as maximum of score function by Brent's (or
    # similar) method
    # First and Last ATR values are used as the boundaries
    optimal_brick_sfo = opt.fminbound(lambda x: 
                                      -evaluate_renko(brick = x, 
                                                      history = data.close, 
                                                      column_name = 'score'), 
                                      np.min(atr), np.max(atr), disp=0)
    # Build Renko chart
    renko_obj_sfo = renko()
    renko_obj_sfo.set_brick_size(auto = False, brick_size = optimal_brick_sfo)
    renko_obj_sfo.build_history(hlc = data[[AKA.HIGH, 
                                            AKA.LOW, 
                                            AKA.CLOSE,]].values)

    if len(renko_obj_sfo.get_renko_prices()) > 1:
        ret_indices = pd.concat([ret_indices,
                                 pd.DataFrame(renko_obj_sfo.source_aligned, 
                                 columns=[FLD.RENKO_TREND_L_LB, 
                                            FLD.RENKO_TREND_L_UB, 
                                            FLD.RENKO_TREND_L], 
                                   index=data.index)], axis=1)
    
    ret_indices[FLD.RENKO_TREND_S_BRICK_SIZE] = renko_obj_atr.brick_size / data[AKA.CLOSE]
    ret_indices[FLD.RENKO_TREND_L_BRICK_SIZE] = renko_obj_sfo.brick_size / data[AKA.CLOSE]
    ret_indices[FLD.RENKO_TREND] = np.where((ret_indices[FLD.RENKO_TREND_L] == 1) & \
                                            (ret_indices[FLD.RENKO_TREND_S] == 1), 1, 
                                            np.where((ret_indices[FLD.RENKO_TREND_L] == -1) & \
                                                     (ret_indices[FLD.RENKO_TREND_S] == -1), -1, 0))
    ret_indices[FLD.RENKO_TREND_S_BEFORE] = Timeline_duration(np.where((ret_indices[FLD.RENKO_TREND_S] == 1) & \
                                                                       (ret_indices[FLD.RENKO_TREND_S].shift(1) != 1), 1, 
                                                                       np.where((ret_indices[FLD.RENKO_TREND_S] == -1) & \
                                                                       (ret_indices[FLD.RENKO_TREND_S].shift(1) != -1), 1, 0)))
    ret_indices[FLD.RENKO_S_JX_BEFORE] = Timeline_duration(np.where((ret_indices[FLD.RENKO_TREND] >= 0) & \
                                                                    (indices[FLD.MAXFACTOR_CROSS] == 1) & \
                                                                    (indices[FLD.MACD_DELTA] > 0) & \
                                                                    (indices[FLD.BOLL_CROSS_JX_BEFORE] < indices[FLD.BOLL_CROSS_SX_BEFORE]) & \
                                                                    (ret_indices[FLD.RENKO_TREND_S_LB] > data[AKA.CLOSE]), 1, 0))
    ret_indices[FLD.RENKO_S_SX_BEFORE] = Timeline_duration(np.where((indices[FLD.MAXFACTOR_CROSS] == -1) & \
                                                                    (indices[FLD.MACD_DELTA] < 0) & \
                                                                    (indices[FLD.BOLL_CROSS_SX_BEFORE] < indices[FLD.BOLL_CROSS_JX_BEFORE]) & \
                                                                    (ret_indices[FLD.RENKO_TREND_S_UB] < data[AKA.CLOSE]), 1, 0))
    ret_indices[FLD.RENKO_TREND_L_BEFORE] = Timeline_duration(np.where((ret_indices[FLD.RENKO_TREND_L] == 1) & \
                                                                       (ret_indices[FLD.RENKO_TREND_L].shift(1) != 1), 1, 
                                                              np.where((ret_indices[FLD.RENKO_TREND_L] == -1) & \
                                                                       (ret_indices[FLD.RENKO_TREND_L].shift(1) != -1), 1, 0)))
    ret_indices[FLD.RENKO_TREND_L_JX_BEFORE] = Timeline_duration(np.where((ret_indices[FLD.RENKO_TREND] >= 0) & \
                                                                          (indices[FLD.MAXFACTOR_CROSS] == 1) & \
                                                                          (indices[FLD.MACD_DELTA] > 0) & \
                                                                          (indices[FLD.BOLL_CROSS_JX_BEFORE] < indices[FLD.BOLL_CROSS_SX_BEFORE]) & \
                                                                          (ret_indices[FLD.RENKO_TREND_L_LB] > data[AKA.CLOSE]), 1, 0))

    ret_indices[FLD.RENKO_OPTIMAL] = optimal_brick
    ret_indices.iat[0, ret_indices.columns.get_loc(FLD.RENKO_OPTIMAL)] = optimal_brick_sfo

    with np.errstate(invalid='ignore', divide='ignore'):
        renko_trend_l_jx = Timeline_Integral(np.where(ret_indices[FLD.RENKO_TREND_L] > 0, 1, 0))
        renko_trend_l_sx = np.sign(ret_indices[FLD.RENKO_TREND_L]) * Timeline_Integral(np.where(ret_indices[FLD.RENKO_TREND_L] < 0, 1, 0))
        ret_indices[FLD.RENKO_TREND_L_TIMING_LAG] = renko_trend_l_jx + renko_trend_l_sx

    renko_trend_s_jx = Timeline_Integral(np.where(ret_indices[FLD.RENKO_TREND_S] > 0, 1, 0))
    renko_trend_s_sx = np.sign(ret_indices[FLD.RENKO_TREND_S]) * Timeline_Integral(np.where(ret_indices[FLD.RENKO_TREND_S] < 0, 1, 0))
    ret_indices[FLD.RENKO_TREND_S_TIMING_LAG] = renko_trend_s_jx + renko_trend_s_sx

    renko_boost_s_jx = Timeline_duration(np.where(ret_indices[FLD.RENKO_TREND_S_UB] > ret_indices[FLD.RENKO_TREND_S_UB].shift(), 1, 0)) + 1
    renko_boost_s_sx = Timeline_duration(np.where(ret_indices[FLD.RENKO_TREND_S_LB] < ret_indices[FLD.RENKO_TREND_S_LB].shift(), 1, 0)) + 1
    ret_indices[FLD.RENKO_BOOST_S_TIMING_LAG] = np.where(ret_indices[FLD.RENKO_TREND_S_TIMING_LAG] > 0,
                                                         renko_boost_s_jx,
                                                         np.where(ret_indices[FLD.RENKO_TREND_S_TIMING_LAG] < 0,
                                                         -renko_boost_s_sx, 0))

    renko_boost_l_jx = Timeline_duration(np.where(ret_indices[FLD.RENKO_TREND_L_UB] > ret_indices[FLD.RENKO_TREND_L_UB].shift(), 1, 0)) + 1
    renko_boost_l_sx = Timeline_duration(np.where(ret_indices[FLD.RENKO_TREND_L_LB] < ret_indices[FLD.RENKO_TREND_L_LB].shift(), 1, 0)) + 1
    ret_indices[FLD.RENKO_BOOST_L_TIMING_LAG] = np.where(ret_indices[FLD.RENKO_TREND_L_TIMING_LAG] > 0,
                                                         renko_boost_l_jx,
                                                         np.where(ret_indices[FLD.RENKO_TREND_L_TIMING_LAG] < 0,
                                                         -renko_boost_l_sx, 0))
    return ret_indices
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                

def renko_border(closep,
                 bricks):

    ymax = max(bricks) 
    ymin = min(np.absolute(bricks))
    width = 1.0
    prev_height = 0

    border = np.zeros((len(bricks), 2),)

    for index, brick in enumerate(bricks):
        facecolor = 'red' if brick > 0 else 'green'
        ypos = abs(brick)
        if index == len(bricks) - 1:
            pass
        elif bricks[index] == bricks[index + 1]:
            height = prev_height
        else:
            height = ypos
            prev_height = height
        border[index, 0] = prev_height
        border[index, 1] = ypos

    return border


def plot_renko(ax, xlabel, bricks):
    from matplotlib.patches import Rectangle

    ymax = max(bricks) 
    ymin = min(np.absolute(bricks))
    width = 1.0
    prev_height = 0
    for index, brick in enumerate(bricks):
        facecolor = 'red' if brick > 0 else 'green'
        ypos = (abs(brick) - ymin) / (ymax - ymin)
        print(brick, ypos)
        if index == len(bricks) - 1:
            pass
        elif bricks[index] == bricks[index + 1]:
            height = prev_height
        else:
            aux1 = (abs(bricks[index + 1]) - ymin) / (ymax - ymin)
            height = abs(aux1 - ypos)
            prev_height = height
        rect = Rectangle((index * width, ypos), width, height,
                         facecolor=facecolor, alpha=0.5)
        ax.add_patch(rect)
    pass


def plot_renko_l(ax, indices, codename=None):
    from matplotlib.patches import Rectangle
    def each_bar(brick):
        facecolor = 'red' if brick.get([FLD.RENKO_TREND_L]).item() > 0 else 'green'
        rect = Rectangle((indices.index.get_loc(brick.name), min(brick.get(FLD.RENKO_TREND_L_LB),
                                                                 brick.get(FLD.RENKO_TREND_L_UB))), 
                         1, abs(brick.get(FLD.RENKO_TREND_L_UB) - brick.get(FLD.RENKO_TREND_L_LB)),
                         facecolor=facecolor, alpha=0.5)
        ax.add_patch(rect)

    indices.apply(lambda x:each_bar(x), axis=1)

    pass


def plot_renko_s(ax, indices, codename=None):
    from matplotlib.patches import Rectangle
    def each_bar(brick):
        facecolor = 'red' if brick.get([FLD.RENKO_TREND_S]).item() > 0 else 'green'
        rect = Rectangle((indices.index.get_loc(brick.name), min(brick.get(FLD.RENKO_TREND_S_LB),
                                                                 brick.get(FLD.RENKO_TREND_S_UB))), 
                         1, abs(brick.get(FLD.RENKO_TREND_S_UB) - brick.get(FLD.RENKO_TREND_S_LB)),
                         facecolor=facecolor, alpha=0.5)
        ax.add_patch(rect)

    indices.apply(lambda x:each_bar(x), axis=1)

    pass


if __name__ == '__main__':

    try:
        import talib
    except:
        print('PLEASE run "pip install TALIB" to call these modules')
        pass
    try:
        import QUANTAXIS as QA
        from QUANTAXIS.QAUtil.QAParameter import ORDER_DIRECTION
        from QUANTAXIS.QAData.QADataStruct import (
            QA_DataStruct_Index_min, 
            QA_DataStruct_Index_day, 
            QA_DataStruct_Stock_day, 
            QA_DataStruct_Stock_min,
            QA_DataStruct_CryptoCurrency_day,
            QA_DataStruct_CryptoCurrency_min,
            )
        from QUANTAXIS.QAIndicator.talib_numpy import *
        from QUANTAXIS.QAUtil.QADate_Adv import (
            QA_util_timestamp_to_str,
            QA_util_datetime_to_Unix_timestamp,
            QA_util_print_timestamp
        )
        from QUANTAXIS.QAUtil.QALogs import (
            QA_util_log_info, 
            QA_util_log_debug,
            QA_util_log_expection)
        from QUANTAXIS.QAFetch.QAhuobi import (
            FIRST_PRIORITY,
        )
    except:
        print('PLEASE run "pip install QUANTAXIS" before call GolemQ.GQIndicator.renko modules')
        pass

    from GolemQ.GQUtil.parameter import (
        AKA, 
        INDICATOR_FIELD as FLD, 
        TREND_STATUS as ST
    )

    from GolemQ.GQIndicator.indices import (
        boll_cross_func,
    )

    import pandas as pd
    import datetime


    # ETF/股票代码，如果选股以后：我们假设有这些代码
    codelist = ['159919', '159908', '159902', '510900', 
                '513100', '512980', '515000', '512800', 
                '512170', '510300', '159941', '512690',
                '159928']
    codelist = ['159919']
    codelist = ['HUOBI.btcusdt']
    codelist = ['600276']
    data_day = QA.QA_fetch_stock_day_adv(codelist,
        start='2014-01-01',
        end='{}'.format(datetime.date.today())).to_qfq()

    ## 获取ETF/股票中文名称，只是为了看得方便，交易策略并不需要ETF/股票中文名称
    #stock_names = QA.QA_fetch_etf_name(codelist)
    #codename = [stock_names.at[code, 'name'] for code in codelist]

    ## 读取 ETF基金 日线，存在index_day中
    #data_day = QA.QA_fetch_index_day_adv(codelist,
    #    start='2014-01-01',
    #    end='{}'.format(datetime.date.today()))

    #frequency = '60min'
    #data_day = QA.QA_fetch_cryptocurrency_min_adv(code=codelist,
    #                start='2019-06-20',
    #                end=QA_util_timestamp_to_str(),
    #                frequence=frequency)
    #data_day = QA.QA_DataStruct_CryptoCurrency_min(data_day.resample('4h'))

    indices = data_day.add_func(boll_cross_func)
    indices = data_day.add_func(renko_trend_cross_func, indices)

    ## Get optimal brick size based
    #optimal_brick = renko().set_brick_size(auto = True, HLC_history =
    #data_day.data[["high", "low", "close"]])

    ## Build Renko chart
    #renko_obj_atr = renko()
    #print('Set brick size to optimal: ', renko_obj_atr.set_brick_size(auto =
    #False, brick_size = optimal_brick))
    #renko_obj_atr.build_history(hlc = data_day.data[["high", "low",
    #"close"]].values)
    ##print('Renko bar prices: ', renko_obj_atr.get_renko_prices())
    ##print('Renko bar directions: ', renko_obj_atr.get_renko_directions())
    #print('Renko bar evaluation: ', renko_obj_atr.evaluate())

    #if len(renko_obj_atr.get_renko_prices()) > 1:
    #    indices = pd.concat([indices,
    #                        pd.DataFrame(renko_obj_atr.source_aligned,
    #                                   columns=[FLD.RENKO_TREND_S_LB,
    #                                            FLD.RENKO_TREND_S_UB,
    #                                            FLD.RENKO_TREND_S],
    #                                   index=data_day.data.index)]
    #                        , axis=1)

    #    #renko_obj_atr.plot_renko(data_day.data)

    ## Function for optimization
    #def evaluate_renko(brick, history, column_name):
    #    renko_obj = renko()
    #    renko_obj.set_brick_size(brick_size = brick, auto = False)
    #    renko_obj.build_history(prices = history)
    #    return renko_obj.evaluate()[column_name]

    ## Get ATR values (it needs to get boundaries)
    ## Drop NaNs
    #atr = talib.ATR(high = np.double(data_day.data.high),
    #                low = np.double(data_day.data.low),
    #                close = np.double(data_day.data.close),
    #                timeperiod = 14)
    #atr = atr[np.isnan(atr) == False]

    ## Get optimal brick size as maximum of score function by Brent's (or
    ## similar) method
    ## First and Last ATR values are used as the boundaries
    #optimal_brick_sfo = opt.fminbound(lambda x: -evaluate_renko(brick = x,
    #                                                            history =
    #                                                            data_day.data.close,
    #                                                            column_name =
    #                                                            'score'),
    #                                  np.min(atr), np.max(atr), disp=0)
    ## Build Renko chart
    #renko_obj_sfo = renko()
    #print('Set brick size to optimal: ', renko_obj_sfo.set_brick_size(auto =
    #False, brick_size = optimal_brick_sfo))
    #renko_obj_sfo.build_history(hlc = data_day.data[["high", "low",
    #"close"]].values)
    ##print('Renko bar prices: ', renko_obj_sfo.get_renko_prices())
    ##print('Renko bar gap: ', renko_obj_sfo.get_renko_gaps())
    ##print('Renko bar upper shadow: ', renko_obj_sfo.get_renko_upper_shadow())
    ##print('Renko bar lower shadow: ', renko_obj_sfo.get_renko_lower_shadow())

    ##print('Renko bar directions: ', renko_obj_sfo.get_renko_directions())
    ##print('Renko bar evaluation: ', renko_obj_sfo.evaluate())

    #if len(renko_obj_sfo.get_renko_prices()) > 1:
    #    indices = pd.concat([indices,
    #                         pd.DataFrame(renko_obj_sfo.source_aligned,
    #                           columns=[FLD.RENKO_TREND_L_LB,
    #                                    FLD.RENKO_TREND_L_UB,
    #                                    FLD.RENKO_TREND_L],
    #                           index=data_day.data.index)], axis=1)
    ##    print(len(renko_obj_sfo.get_renko_prices()))
    ##    print(len(data_day.data))
    #    #renko_obj_sfo.plot_renko(data_day.data)
    
    #indices[FLD.RENKO_TREND] = np.where((indices[FLD.RENKO_TREND_L] == 1) & \
    #                                    (indices[FLD.RENKO_TREND_S] == 1), 1,
    #                                    np.where((indices[FLD.RENKO_TREND_L]
    #                                    == -1) & \
    #                                    (indices[FLD.RENKO_TREND_S] == -1),
    #                                    -1, 0))

    plot_renko(data_day.data, indices)

    plt.show()