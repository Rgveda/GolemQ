# coding:utf-8
"""砖块图（**RENKO**）—— 砖块序列生成与趋势判定。

从哪来
======
自旧树 ``GolemQ_old/indices/renko.py``（1065 行）搬运，**只搬活链**。
它是 `fractal/v0–v7,v9`、`benchmark/*`、`signal/rsrs.py` 共用的趋势特征生产者，
产出 ``RENKO_TREND_S`` / ``RENKO_TREND_L`` 两族列。

搬了什么（5 个，即活链全集）
============================
===============  ====  ==========================================================
名字              旧行  职责
===============  ====  ==========================================================
``renko``          51   砖块构建器：``set_brick_size`` / ``build_history`` /
                        ``do_next`` / ``evaluate`` + 6 个 getter
``renko_chart``   390   ``@nb.jit`` 定砖高压缩 → ``(n,4)`` = 价格/方向/LB/UB
``evaluate_renko`` 382  ``renko().evaluate()[column_name]``，是 ``fminbound`` 的目标函数
``renko_in_cluster_group`` 454
                        按 **1200 bar** 分窗，ATR 定界 + ``fminbound`` 搜最优砖高
``renko_trend_cross_func`` 534
                        **主干**：合成 S/L 两族 + ``RENKO_TREND`` + 三个 ``*_TIMING_LAG``
===============  ====  ==========================================================

**没搬**（零调用者 / 已烂）—— 留痕，免得下一个人以为是疏忽
========================================================
以下每一项都用「全树 grep 函数名」判定过，**不是省略**：

* ``RENKOP``（旧 430）——**全树零调用者**，且 jit 核循环体里有个 ``print(step)``。
* ``renko_border``（旧 822）——零调用者，且 ``closep`` 形参**从未被使用**。
* ``renko_trend_cross_old_func``（旧 667）——零调用者，已被
  ``renko_trend_cross_func`` 取代（前者用 ``renko`` 类逐 bar 慢走，后者用 numba 核）。
* ``plot_renko_l`` / ``plot_renko_s``（旧 873 / 888）——零调用者。
* ``renko.plot_renko``（类方法，旧 305）——零调用者，且**已烂**：
  内部用 ``plt`` / ``mpf`` / ``patches``，而模块**从未 import 它们**，一调即
  ``NameError``。
* 整个 ``__main__`` 演示段（旧 903-1066）——签名对不上（``plot_renko`` 被按 2 参调、
  但签名要 3 参）、且**全是 QUANTAXIS 专属**取数路径。
* 同目录 ``indices/renko02.py``（1065 行的平行旧拷贝，import 早已废弃的
  ``GolemQ.GQUtil.*``）——**全树零 import**，是死文件。

import 改接（老树 → 新树）
=========================
* ``GolemQ.utils.parameter`` → ``GolemQ.core.constants``
  （类名 ``INDICATOR_FIELD`` → ``FIELD``）
* ``from GolemQ.analysis.timeseries import *`` → ``GolemQ.analysis.timing`` 的
  **两个**具名函数：``Timeline_Integral`` / ``Timeline_duration``。
  ⚠️ 这个清单是**扫出来的**不是猜的（比对面模块 43 个函数名 × 本文件的调用点）——
  旧树那个星号导入实际只用到 2 个时序函数 + 1 个 QUANTAXIS 工具（见下）。
* **删除** ``import QUANTAXIS as QA`` 与 ``from QUANTAXIS.QAIndicator.talib_numpy import *``
  —— 旧树靠后者拿到 ``QA_util_timestamp_to_str``，而它**只用在调试打印里**。
* **删除全部 ``print``**（用户 2026-10-10：「不需要打印了，这个已经非常成熟」）。
  含三类：调试进度打印、``except`` 里倾倒整段数组的那两处、以及两处入参列数不符
  的提示。**控制流一律未动** —— 详见下面「已知缺陷」。
* 字段常量补齐 14 个（``FIELD`` 里原有 4 个 ``RENKO_*``），真值取自旧树
  ``GolemQ_old/utils/parameter.py`` L1849-1880，**逐值核对过**。
  ⚠️ 那批值 **L 侧大写 / S 侧小写混用**（``RENKO_lBAR`` vs ``renkosBarBf``）——
  这是旧树真值，**别"统一"**。

依赖
====
``talib.ATR``（定最优砖高）、``scipy.optimize.fminbound``（搜最优砖高）、
``numba``（``renko_chart`` 的 jit 核）—— 三者都在
``cli/bootstrap.py`` 的 ``MIN_PACKAGES`` 里，故这里是**模块级硬导入**。

⚠️ 其中 ``talib`` / ``scipy`` 是**本模块带进新树的第一批使用者**。它们**不能**
换成自实现：``talib.ATR`` 与 ``fminbound`` 的数值就是旧树口径，换成等价实现会
**改变砖高**，与旧树的对拍立刻失真（对拍是本模块唯一的正确性证据）。

写到哪些列
==========
``renko_trend_cross_func`` 向特征帧写：

* **S 族**（``renko_chart``，纯收盘价压缩砖）：``RENKO_PRICE_S`` / ``RENKO_TREND_S``
  / ``RENKO_TREND_S_LB`` / ``RENKO_TREND_S_UB`` —— 末尾三列被 cast 成 **``float16``**。
* **L 族**（``renko_in_cluster_group``，1200 bar 分窗聚类）：``RENKO_PRICE_L`` /
  ``RENKO_TREND_L`` / ``RENKO_TREND_L_LB`` / ``RENKO_TREND_L_UB`` / ``RENKO_OPTIMAL``。
* **合成**：``RENKO_TREND``（S、L 同向才给 ±1）、``RENKO_TREND_L_TIMING_LAG`` /
  ``RENKO_TREND_S_TIMING_LAG`` / ``RENKO_BOOST_S_TIMING_LAG`` /
  ``RENKO_BOOST_L_TIMING_LAG``。

⚠️ **承重的是 ``RENKO_TREND_S`` / ``RENKO_TREND_L``**（旧树 137 处 / 83 处引用）。
``RENKO_OPTIMAL`` / ``RENKO_PRICE_L`` **写后无人读**，``RENKO_PRICE_S`` 被下游统一
drop —— 三个都**照写不删**（``len(data) < 30`` 的早退分支靠这批列名保形状），
只是别指望有人消费。

已知缺陷（**保真保留，未就地修**）
==================================
搬运纪律是先逐字搬、把缺陷记在档上，**不在搬运里顺手改语义** —— 改了就无法对拍。
以下三条旧树原样如此，新树照旧：

1. ``renko_chart`` 在 ``bricks == 0`` 且 ``condensed=False`` 时**不写**该行的
   ``price`` 列，而数组是 ``np.empty`` ⇒ 读到**未初始化内存**。
   现有调用方全走默认 ``condensed=True``，故不触发。
2. ``renko_trend_cross_func`` 合成 L 族那个 ``try/except`` 兜住的其实是
   ``ret_indices`` **可能未定义**（``len(bricks_fixed[:,0]) <= 1`` 时它从没被赋值），
   而兜底分支里**又引用它** ⇒ 二次异常。与 ``PITFALLS.md`` **P28** 同族。
3. ``renko_in_cluster_group`` 的 ``try/except`` 一旦兜住，``optimal_brick_sfo``
   **未定义**，下一行 ``renko_chart`` 直接 ``NameError``。同上。
4. ``renko.build_history`` 里 ``self.source_aligned = np.empty((len(prices), 3))``
   起手，而**写它的循环从 ``idx = 1`` 开始** ⇒ **第 0 行永远是未初始化内存**
   （实测形如 ``9.2e-312``）。**对拍时抓到的**（`test_renko.py`
   `test_source_aligned_row0_is_never_written`）—— 两次独立构建的第 0 行互不相等。

   ⚠️ 这条**有消费方**：``renko_trend_cross_func`` 走的是 ``renko_chart``，
   不碰它；但另案的 ``features/base.py::calc_renko_atr_vX`` 用的正是
   ``source_aligned`` ⇒ 它写出的 ``RENKO_TREND_S_LB/UB`` **第 0 行是垃圾**。

⚠️ 这四处**都不要**"顺手修" —— 修了就得重新论证与旧树的行为差异在哪。
"""
from __future__ import annotations

import math

import numba as nb
import numpy as np
import pandas as pd
import scipy.optimize as opt
import talib

from GolemQ.analysis.timing import Timeline_Integral, Timeline_duration
from GolemQ.core.constants import AKA, FIELD as FLD

__all__ = ['renko', 'renko_chart', 'evaluate_renko', 'renko_in_cluster_group',
           'renko_trend_cross_func']


class renko:
    """
    Renko Chart/Renko Brick/Renko Bar

    Renko砖块图是一种由日本人发明的用于衡量和绘制价格变化的财务技術分析图表。
    Renko图由砖块组成，与典型的蜡烛图相比，它可以通过剔除噪音的方式，
    帮助投资者更好的分析趋势和进行量化研究，更清晰地显示市场趋势并提高信噪比。
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
                # 旧树在这里 print 一行提示后把 source_hlc 置空（已删打印，行为不变）
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
            elif np.abs(gap_div) >= 2:  # Should be double gap at least
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
                # 旧树在这里 print 一行提示（已删打印，行为不变）
                pass

        if len(prices) > 0:
            # Init by start values
            self.source_prices = prices
            self.renko_prices.append(prices[0])
            self.renko_directions.append(0)

            # 记录上下影线
            if (len(self.source_hlc) > 0):
                self.renko_upper_shadow.append(self.source_hlc[0, 0])
                self.renko_lower_shadow.append(self.source_hlc[0, 1])

            self.source_aligned = np.empty((len(prices), 3))

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
            try:
                brick_size = np.median(talib.ATR(high=np.double(HLC_history[:, 0]),
                                                 low=np.double(HLC_history[:, 1]),
                                                 close=np.double(HLC_history[:, 2]),
                                                 timeperiod=atr_timeperiod)[atr_timeperiod:])
            except Exception:
                # ⚠️ 旧树在这里 print 三段整个数组（已删）。异常时 `brick_size`
                # 保持 0.0 —— 与旧树**行为一致**，只是不再往 stdout 倾倒数据。
                # 见模块 docstring「已知缺陷」：这条**不要**顺手改成抛异常。
                pass

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

            # ⚠️ 键名 `'sign_changes:'` 带一个**冒号** —— 旧树如此，照抄。
            # `evaluate_renko` 的调用方传的是 `'score'`，不受影响，但别"修"这个键名。
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


# Function for optimization
def evaluate_renko(brick, history, column_name):
    """用给定砖高跑一遍砖块序列，返回 ``evaluate()`` 里的某一项（调用方传 ``'score'``）。

    它是 :func:`renko_in_cluster_group` 里 ``fminbound`` 的**目标函数** ——
    ⚠️ 每次迭代都**新建一个 ``renko`` 对象并整段重建**，所以那一层很慢。
    旧树原样如此，保真搬运。
    """
    renko_obj = renko()
    renko_obj.set_brick_size(brick_size=brick, auto=False)
    renko_obj.build_history(prices=history)
    return renko_obj.evaluate()[column_name]


@nb.jit(nopython=True)
def renko_chart(price_series, N, condensed=True):
    """定砖高 Renko 压缩：把价格序列压成砖块序列。

    原版代码来自 QUANTAXIS ``QA.RENKO``，经 numba jit 优化。

    :param price_series: 一维价格数组
    :param N: **绝对**砖高
    :param condensed: 砖块未推进时是否把上一砖位重复写出（默认 ``True``）
    :return: ``(len(price_series), 4)`` 的数组，列为
        ``[price, direction, lower_band, upper_band]``
        ⚠️ 旧 docstring 写「3 列数据：Lower Band/Upper Band/Direction」是**错的**
        （实际 4 列，且价格在最前），此处已订正。列序由下方
        ``idx_renko_*`` 常量与调用方的 ``columns=`` 共同钉住，改了就错列。

    ⚠️ ``condensed=False`` 且 ``bricks == 0`` 时该行 ``price`` **不被赋值**，
    而数组是 ``np.empty`` ⇒ 读到未初始化内存（见模块 docstring「已知缺陷」1）。
    现有调用方全走默认值，故不触发。

    空序列会抛 ``IndexError``（``price_series[0]``）—— 旧树如此，未加保护。

    >>> import numpy as np
    >>> renko_chart(np.array([10., 11., 12., 11.]), 1.0).tolist()
    [[10.0, 1.0, 10.0, 11.0], [11.0, 1.0, 11.0, 12.0], [12.0, 1.0, 12.0, 13.0], [-11.0, -1.0, 10.0, 11.0]]

    ⚠️ 末行的价格是 **-11.0** 而不是 11.0 —— 价格列**把方向编码在符号里**
    （`last_price = abs(chart[-1])`，所以下一砖从 12 往下算得 `-(12-1)`）。
    `direction`/`lb`/`ub` 三列全由它推出，**负价是刻意的**，别当 bug 修。

    砖块未推进时重复上一砖位（`condensed=True`）：

    >>> renko_chart(np.array([10., 10.4, 10.1]), 1.0)[:, 0].tolist()
    [10.0, 10.0, 10.0]
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


def renko_in_cluster_group(data: pd.DataFrame,
                           indices: pd.DataFrame = None,
                           maxlength=1200) -> np.ndarray:
    """按 ``maxlength`` 分窗，用 ATR 定界 + 布伦特法搜每窗的**最优砖高**。

    (假设)我们对这条行情的走势一无所知，使用机器学习可以快速的识别出走势，
    划分出波浪。renko bricks 将整条行情大致分块，这个随着时间变化会有轻微抖动。
    所以不适合做精确买卖点控制。但是作为趋势判断已经足够了。

    ⚠️ 名字与旧 docstring 里那句「使用机器学习」都是**误导**：它不读任何外部
    数据、不依赖任何 cluster 分组表 —— 只是把输入 DataFrame 按
    ``maxlength`` 切片后逐窗做一维搜索。那个 ``indices`` 形参**从未被使用**。

    :param data: 含 ``high`` / ``low`` / ``open`` / ``close`` 列的行情帧
    :param indices: **未使用**（保留签名以与旧树一致）
    :param maxlength: 分窗长度（旧注释：>1200 时无监督聚类效果变差，宜 1000~1500）
    :return: ``(len(data), 5)`` = ``[price_l, trend_l, lb_l, ub_l, optimal]``
        ⚠️ ``optimal`` 只在每个窗口的 ``sub_first + 1`` 那一行被写，其余行是 **0**
        （``ret_cluster_group`` 由 ``np.zeros`` 起手）。
        实际使用上这句不能作为对拍失败时的调试打点。
    """
    # Get ATR values (it needs to get boundaries)
    # Drop NaNs
    factor_atr = talib.ATR(high=np.double(data.high),
                           low=np.double(data.low),
                           close=np.double(data.close),
                           timeperiod=14)
    factor_atr = factor_atr[np.isnan(factor_atr) == False]

    sub_offest = 0
    nextsub_first = sub_first = 0
    nextsub_last = sub_last = (maxlength - 1)

    ret_cluster_group = np.zeros((len(data.close.values), 5),)
    while (sub_first == 0) or (sub_first < len(data.close.values)):
        # 数量大于1200bar的话无监督聚类效果会变差，应该控制在1000~1500bar之间
        if (len(data.close.values) > maxlength):
            highp = np.nan_to_num(data.high.values[sub_first:sub_last], nan=0)
            lowp = np.nan_to_num(data.low.values[sub_first:sub_last], nan=0)
            openp = np.nan_to_num(data.open.values[sub_first:sub_last], nan=0)
            closep = np.nan_to_num(data.close.values[sub_first:sub_last], nan=0)
            subdata = data.iloc[sub_first:sub_last, :]
            atr = factor_atr[sub_first:sub_last]
            if (sub_last + maxlength < len(data.close.values)):
                nextsub_first = sub_first + (maxlength - 1)
                nextsub_last = sub_last + maxlength
            else:
                nextsub_first = len(data.close.values) - maxlength
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

        # Get optimal brick size as maximum of score function by Brent's (or
        # similar) method
        # First and Last ATR values are used as the boundaries
        try:
            optimal_brick_sfo = opt.fminbound(lambda x:
                                              -evaluate_renko(brick=x,
                                                              history=closep,
                                                              column_name='score'),
                                              np.min(atr), np.max(atr), disp=0)
        except Exception:
            # ⚠️ 旧树在这里 print 一行（已删）。兜住之后 `optimal_brick_sfo`
            # **未定义** ⇒ 下一行 `renko_chart` 直接 NameError。
            # 见模块 docstring「已知缺陷」3：保真保留，**不要**顺手改成 re-raise。
            pass

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
    """使用 Renko brick 砖块图进行趋势判断 —— 写 ``FLD.RENKO_*`` 一族列。

    输入需要 ``data[[AKA.HIGH, AKA.LOW, AKA.CLOSE]]`` **列**，以及
    ``data.close`` **属性**访问（两者都要，缺一即 KeyError/AttributeError）。

    :param data: 行情帧（MultiIndex ``(时间, code)``）
    :param indices: 已有特征帧；给了就 ``concat`` 在其右侧，否则新建
    :return: 特征帧（``float16`` 的 S 族已在末尾 cast）

    ⚠️ 三处照抄旧树的**非显然**点：

    1. ``len(data) < 30`` 时**不计算**，只返回一批**空的同名列** —— 调用方
       （`fractal/v1-v5`）正是靠「``RENKO_TREND`` 在不在 columns 里」来判断要不要重算，
       所以这个形状**承重**，不能简化成 ``return indices``。
    2. 合成 L 族那个 ``try/except`` 兜住的其实是 ``ret_indices`` **可能未定义**
       （``len(bricks_fixed[:,0]) <= 1`` 时它从没被赋值），而兜底分支**又引用它**
       ⇒ 二次异常。见模块 docstring「已知缺陷」2。
    3. 末尾把 **S 族三列 cast 成 ``float16``**。所以对拍/回归比这些列时要么先
       还原 dtype，要么用 ``float16`` 的精度 —— 用 `assert_allclose` 配宽松容差
       会把真错误一起放过去。
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
    optimal_brick = renko().set_brick_size(auto=True,
                                           HLC_history=data[[AKA.HIGH,
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

    try:
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
    except Exception:
        # ⚠️ 旧树在这里 print 一行 code（已删）。这个裸兜底真正兜住的是
        # `ret_indices` 未定义（上面那个 if 为假时），而兜底里又引用它 ⇒ 二次异常。
        # 见模块 docstring「已知缺陷」2：保真保留。
        ret_indices[FLD.RENKO_TREND_L] = ret_indices[FLD.RENKO_TREND_S]
        ret_indices[FLD.RENKO_TREND_L_LB] = ret_indices[FLD.RENKO_TREND_S_LB]
        ret_indices[FLD.RENKO_TREND_L_UB] = ret_indices[FLD.RENKO_TREND_S_UB]

    ret_indices[FLD.RENKO_TREND] = np.where((ret_indices[FLD.RENKO_TREND_L] == 1) &
                                            (ret_indices[FLD.RENKO_TREND_S] == 1), 1,
                                            np.where((ret_indices[FLD.RENKO_TREND_L] == -1) &
                                                     (ret_indices[FLD.RENKO_TREND_S] == -1), -1, 0))

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

    ret_indices[FLD.RENKO_TREND_S] = ret_indices[FLD.RENKO_TREND_S].astype(np.float16)
    ret_indices[FLD.RENKO_TREND_S_LB] = ret_indices[FLD.RENKO_TREND_S_LB].astype(np.float16)
    ret_indices[FLD.RENKO_TREND_S_UB] = ret_indices[FLD.RENKO_TREND_S_UB].astype(np.float16)

    return ret_indices
