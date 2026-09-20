# coding:utf-8
"""仓位优化工具 —— 在已选出的候选里分配资金，并回测。

定位
====
**它不是选股，是分配资金。** 策略给出「谁该持有、谁优先」，本包决定
「同时持几只、每只多少钱」，并把撮合、成本、交易规则、指标、落盘一并做完。

分层
====
==================  ==========================================================
`strategy.py`       策略契约：`hold_signal` + `priority`（策略侧实现）
`sizing.py`         仓位分配：槽位数与单槽金额（**「优化仓位」的核心**）
`costs.py`          成本模型：佣金/印花税/过户费/滑点（中立数据类 + 市场预设）
`rules.py`          交易规则：T+N/整手/涨跌停（中立数据类 + 市场预设）
`engine.py`         回测引擎：撮合/指标/落盘（**撮合实现待移植**）
==================  ==========================================================

**为什么成本与规则不含 A 股默认值**：它们是通用概念、数值因市场而异。
把万0.85、T+1 写成默认值，跨市场使用时会计错一个数量级，而回测里看不出来。
A 股口径以 :data:`costs.ASHARE_COST` / :data:`rules.ASHARE_RULES` /
:data:`sizing.ASHARE_SIZER` **预设**提供，须显式传入。

状态
====
**骨架。** 数据结构与契约完整；`BacktestEngine.run()` 的撮合实现待从
参照实现（`OneWaveQuant/GolemQ/benchmark/zen_bt.py`，821 行）按本包的契约
选择性移植 —— 不整搬，因为它长在 QUANTAXIS 树上。

用法（撮合实现就位后）::

    from GolemQ.portfolio import (
        Strategy, NotionalSlotSizer, ASHARE_COST, ASHARE_RULES, BacktestEngine)

    class MyStrategy(Strategy):
        name = 'my_strategy'
        def hold_signal(self, features_dummy): ...
        def priority(self, features_dummy): ...

    engine = BacktestEngine(MyStrategy(), NotionalSlotSizer(),
                            costs=ASHARE_COST, rules=ASHARE_RULES,
                            principal=3_000_000)
    result = engine.run(features_dummy)
    result.to_csv('...util.csv', '...trades.csv')
"""
from __future__ import annotations

from .costs import ASHARE_COST, CostModel
from .engine import (
    TRADE_COLUMNS,
    UTIL_COLUMNS,
    BacktestEngine,
    BacktestResult,
    make_ashare_engine,
)
from .rules import ASHARE_RULES, TradeRules
from .sizing import ASHARE_SIZER, EvenSizer, NotionalSlotSizer, PositionSizer
from .strategy import Strategy, StrategyError

__all__ = [
    'Strategy',
    'StrategyError',
    'PositionSizer',
    'NotionalSlotSizer',
    'EvenSizer',
    'ASHARE_SIZER',
    'CostModel',
    'ASHARE_COST',
    'TradeRules',
    'ASHARE_RULES',
    'BacktestEngine',
    'BacktestResult',
    'make_ashare_engine',
    'TRADE_COLUMNS',
    'UTIL_COLUMNS',
]
