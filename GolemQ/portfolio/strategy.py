# coding:utf-8
"""策略接口 —— **仓位优化工具**与具体策略之间的唯一契约。

策略与引擎的分工
================
**策略只回答两个问题**：

1. 这一刻，哪些标的值得持有？ —— :meth:`Strategy.hold_signal`
2. 同一 bar 上多个候选抢资金时，谁优先？ —— :meth:`Strategy.priority`

**其余全归引擎**：怎么撮合、多少钱、什么成本、T+N、整手、槽位分配、指标、落盘。
换策略不改引擎；改成本模型不改策略。

为什么把「优先级」和「仓位大小」分开
====================================
参照实现（`benchmark/zen_bt.py`）的文档写得很清楚：

    现金耗尽后剩余候选落空；仓位大小仍由梯形/等权槽位决定，**与优先顺序解耦**。

即：**优先级决定「谁能拿到资金」，槽位决定「拿到多少」**。两者混在一起时，
调仓位大小会连带改变选股结果，回测就不可解释了。故此处分作两个方法。

信号口径：一律基于 `features_dummy`
===================================
策略消费的是**特征矩阵** `features_dummy`：两层 MultiIndex ``(date, code)``，
列是特征名（见 `markets/StockCN/MONGODB83.md` 与本模块的 :class:`Strategy` 说明）。
策略**不得自己去取数据** —— 取数走 `fetch/` 门面，策略只做算术。

⚠️ 防未来函数
==============
信号必须在 **bar 收盘**才能算出来，成交在**下一根 bar 开盘**（由引擎保证）。
策略实现**不得**在计算某根 bar 的信号时用到该 bar 之后的任何数据 ——
这是自建回测最容易栽且最难自查的地方；参照实现的文档专门记了这条纪律。
"""
from __future__ import annotations

import abc

import pandas as pd

__all__ = ['Strategy', 'StrategyError']


class StrategyError(RuntimeError):
    """策略实现不满足契约时抛出（如信号形状与 features_dummy 不符）。"""


class Strategy(abc.ABC):
    """仓位优化的策略侧契约。

    子类需要实现 :meth:`hold_signal` 与 :meth:`priority`，并声明 :attr:`name`。

    典型实现（参照 `benchmark/zen_bt.py` 的信号公式）::

        class ZenPivotStrategy(Strategy):
            name = 'zen_pivot'

            def hold_signal(self, features_dummy):
                return ((features_dummy[FTR.XGB_ECHO_TIMING_LAG] > 1e-12) &
                        (features_dummy[LTT.QUADRANT_LEVERAGE_MACD_TIMING_LAG_DUMMY] > 1e-12))

            def priority(self, features_dummy):
                # 大者优先；并列时由引擎按 code 稳定排序，避免结果不可复现
                return features_dummy[TRD.OMNIPATH_UPRISING_I_COEF_DUMMY]
    """

    #: 策略标识，用于落盘文件名与日志。子类必须覆盖。
    name: str = ''

    @abc.abstractmethod
    def hold_signal(self, features_dummy: pd.DataFrame) -> pd.Series:
        """返回「该 bar 该标的是否应持有」的布尔序列。

        :param features_dummy: 两层 MultiIndex ``(date, code)`` 的特征矩阵
        :returns: 与 `features_dummy` **同索引**的 bool Series

        ⚠️ 引擎会断言索引一致（:class:`StrategyError`）—— 索引错位会让回测
        静默按错误标的成交，比报错危险得多。
        """

    @abc.abstractmethod
    def priority(self, features_dummy: pd.DataFrame) -> pd.Series:
        """返回同 bar 候选之间的**抢占资金优先级**，**值大者优先**。

        :returns: 与 `features_dummy` 同索引的数值 Series

        只需保证**相对大小**有意义；绝对值不参与计算。引擎会把非有限值
        （NaN/inf）视为最低优先级，不参与抢占。
        """

    # ---- 供引擎调用的校验入口 -------------------------------------------

    def signals_for(self, features_dummy: pd.DataFrame):
        """校验并返回 `(hold, priority)`。**引擎只应经此调用策略。**

        在这里统一校验，而不是让引擎逐个方法去查 —— 契约检查集中一处，
        新策略接进来时不需要改动引擎。

        >>> import pandas as pd
        >>> idx = pd.MultiIndex.from_tuples(
        ...     [('2024-01-02', '600519'), ('2024-01-02', '000001')],
        ...     names=['date', 'code'])
        >>> fd = pd.DataFrame({'sig': [0.5, -0.5]}, index=idx)
        >>> class S(Strategy):
        ...     name = 's'
        ...     def hold_signal(self, f): return f['sig'] > 0
        ...     def priority(self, f): return f['sig']
        >>> hold, prio = S().signals_for(fd)
        >>> list(hold)
        [True, False]
        >>> list(prio)
        [0.5, -0.5]

        索引错位会被**拦下** —— 否则回测会按错误标的成交且不报错：

        >>> class Bad(Strategy):
        ...     name = 'bad'
        ...     def hold_signal(self, f): return pd.Series([True])
        ...     def priority(self, f): return pd.Series([1.0])
        >>> Bad().signals_for(fd)
        Traceback (most recent call last):
            ...
        GolemQ.portfolio.strategy.StrategyError: hold_signal() 的索引与 features_dummy 不一致 ...

        """
        hold = self.hold_signal(features_dummy)
        prio = self.priority(features_dummy)
        self._check(hold, features_dummy, 'hold_signal', want_bool=True)
        self._check(prio, features_dummy, 'priority', want_bool=False)
        return hold, prio

    @staticmethod
    def _check(series, features_dummy, which, want_bool):
        if not isinstance(series, pd.Series):
            raise StrategyError(f'{which}() 必须返回 pandas.Series，得到 {type(series).__name__}')
        if not series.index.equals(features_dummy.index):
            raise StrategyError(
                f'{which}() 的索引与 features_dummy 不一致 '
                f'（长度 {len(series)} vs {len(features_dummy)}）。'
                f'索引错位会让回测按错误标的成交，且不报错 —— 故在此硬拦。')
        if want_bool and series.dtype != bool:
            raise StrategyError(f'{which}() 必须返回 bool 序列，得到 {series.dtype}')
