# coding:utf-8
"""回测引擎 —— **接口与输出契约已定，撮合实现待移植**。

本模块当前是**骨架**：数据结构与契约完整，`BacktestEngine.run()` 尚未实现。
参照实现是 `OneWaveQuant/GolemQ/benchmark/zen_bt.py`（821 行），
**不直接搬** —— 它长在 QUANTAXIS 树上，直接搬会把耦合一起带进来。先定接口，
再照接口选择性移植，可同时完成「解耦」与「可复用」两件事。

> 📍 **参照实现在哪**（2026-09-25 核实）：`OneWaveQuant` 在 **`Y:/代码/OneWaveQuant`**，
> 不在 `Y:/Projects` 下，按后者找会找不到。且树内 **`GolemQ_old/benchmark/zen_bt.py`
> 与它逐字节相同**（md5 `148b63d79e8b19ae0deb767cbd785988`，同为 821 行）——
> 所以不依赖另一个仓库也能读到参照实现。

引擎负责什么（策略**不**负责）
==============================
* **撮合时序**：信号在 bar 收盘得到 → **下一根 bar 开盘**成交
  （参照实现的文档专门记了这条防未来函数的纪律）
* **交易规则**：T+N、整手、涨跌停检查（参数见 `rules.TradeRules`）
* **成本**：佣金/印花税/过户费/滑点（参数见 `costs.CostModel`）
* **仓位**：槽位数与单槽金额（见 `sizing.PositionSizer`）
* **指标**：夏普 / Sortino / 卡玛 / 最大回撤 / 胜率 / 手续费合计 / 资金使用率
* **落盘**：`datastore/export/` 下的 `_util.csv` 与 `_trades.csv`

策略只提供 `hold_signal` 与 `priority`（见 `strategy.Strategy`）。

输出契约（取自参照实现的实际产物）
==================================
``_util.csv`` —— 资金曲线与利用率::

    datetime, equity, util, returns, cum_return, bench_close, bench_cum_return

``_trades.csv`` —— 交割单::

    datetime, code, side, qty, price, amount, fee, pnl

``side`` 取 ``'B'``/``'S'``；``pnl`` 仅卖出行有值（已实现盈亏），买入行为空。
**期末清算也落成 ``side='S'`` 行** —— 否则最后一笔持仓的盈亏不在文件里，
胜率无法由产物复算，对账只能靠信回测器。

⚠️ 落盘绝不覆盖
================
参照实现的文档记了一次真实事故：

    2026-09-16 前用的固定名 `csi300_*` 曾导致两次回测产物被静默覆盖。

故文件名须含运行时间戳，且**目标已存在时自动换名**。这条纪律保留；
`util` 与 `trades` 的后缀必须**成对决定一次**，否则「util 已存在但 trades 不存在」
时会给这一对配上不同前缀，按前缀找同批产物的对账脚本就会错位。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .costs import ASHARE_COST, CostModel
from .rules import ASHARE_RULES, TradeRules
from .sizing import PositionSizer
from .strategy import Strategy

__all__ = ['BacktestResult', 'BacktestEngine', 'TRADE_COLUMNS', 'UTIL_COLUMNS']

#: 交割单列序。**顺序即契约** —— 对账脚本按位置读会依赖它。
TRADE_COLUMNS = ('datetime', 'code', 'side', 'qty', 'price', 'amount', 'fee', 'pnl')

#: 资金曲线列序
UTIL_COLUMNS = ('datetime', 'equity', 'util', 'returns', 'cum_return',
                'bench_close', 'bench_cum_return')


@dataclass
class BacktestResult:
    """一次回测的产物。字段与落盘的两张 CSV 一一对应。"""

    #: 资金曲线与利用率，列见 :data:`UTIL_COLUMNS`
    util: pd.DataFrame = field(default_factory=pd.DataFrame)
    #: 交割单，列见 :data:`TRADE_COLUMNS`
    trades: pd.DataFrame = field(default_factory=pd.DataFrame)
    #: 汇总指标：sharpe / sortino / calmar / max_drawdown / win_rate /
    #: total_fee / utilization / info_ratio 等
    stats: dict = field(default_factory=dict)
    #: 实际落盘路径 `(util_path, trades_path)`；未落盘时为 None
    paths: tuple = None

    def to_csv(self, util_path: str, trades_path: str, encoding: str = 'utf-8-sig'):
        """落盘。**调用方负责保证路径不冲突**（见模块文档的覆盖事故）。

        用 ``utf-8-sig`` 而非 ``utf-8``：参照实现的产物带 BOM，Excel 直接打开
        中文不乱码；改成 utf-8 会让既有的对账脚本读出的首列名多一个不可见字符。
        """
        self.util.to_csv(util_path, index=False, encoding=encoding)
        self.trades.to_csv(trades_path, index=False, encoding=encoding)
        self.paths = (util_path, trades_path)


class BacktestEngine:
    """回测引擎。

    构造时**必须显式给出成本与规则** —— 不提供「不传就用 A 股口径」的默认值。
    默认值会让跨市场使用时静默按 A 股算，而费率差一个数量级在回测里看不出来。

    **当前 `run()` 尚未实现**（见模块文档的「骨架」说明），但本类**可以实例化** ——
    契约与校验是可用的，便于先把配置接好、测试先写起来。刻意不做成 ABC：
    抽象基类的 `__init__` 校验在实际使用中永远走不到，等于放着不生效。

    :param strategy: 策略（只提供信号与优先级）
    :param sizer: 仓位分配
    :param costs: 成本模型；A 股口径可用 `costs.ASHARE_COST`
    :param rules: 交易规则；A 股口径可用 `rules.ASHARE_RULES`
    :param principal: 初始本金
    :param benchmark: 基准代码，用于 ``bench_close`` / ``bench_cum_return``
    """

    def __init__(self, strategy: Strategy, sizer: PositionSizer,
                 costs: CostModel, rules: TradeRules,
                 principal: float = 1_000_000.0,
                 benchmark: str = '000300'):
        if costs is None or rules is None:
            raise ValueError(
                '成本模型与交易规则必须显式传入 —— 不提供 A 股默认值，'
                '因为跨市场误用的差额在回测里不可见。'
                'A 股请传 costs.ASHARE_COST 与 rules.ASHARE_RULES。')
        self.strategy = strategy
        self.sizer = sizer
        self.costs = costs
        self.rules = rules
        self.principal = float(principal)
        self.benchmark = benchmark

    def run(self, features_dummy: pd.DataFrame) -> BacktestResult:
        """跑一次回测。

        :param features_dummy: 两层 MultiIndex ``(date, code)`` 的特征矩阵，
            **已由 pipeline/ 算完所有特征**。
        :returns: :class:`BacktestResult`

        实现要点（照参照实现搬运时逐条对齐）：

        1. 信号用 ``strategy.signals_for()`` 取，**不要绕过它直接调策略方法** ——
           契约校验在那里。
        2. 成交价与成交时点都取自**下一根 bar**；滑点经
           ``costs.fill_price(price, side)``，方向不可反。
        3. T+N 用 ``rules.can_sell(buy_date, current_date)`` 判，
           按**自然持有期**而非 bar 数。
        4. 买入量按 ``rules.lot_size`` 取整；不足一手的资金不建仓。
        5. 涨跌停检查开启时，一字板不成交。
        6. **期末清算是必做项** —— 最后一笔持仓要落成 ``side='S'`` 行，
           否则胜率无法由产物复算。
        """
        raise NotImplementedError(
            '撮合实现待移植。参照 OneWaveQuant/GolemQ/benchmark/zen_bt.py '
            '的 backtest_one / run_portfolio，按本模块的契约选择性搬运。')


def make_ashare_engine(strategy: Strategy, sizer=None, principal: float = 1_000_000.0,
                       benchmark: str = '000300', **kwargs) -> BacktestEngine:
    """便捷构造：用 A 股口径的成本与规则。

    存在的意义是让「A 股口径」成为**一处显式命名**，而不是散落各处的字面量。
    但它仍然要求显式调用 —— 不改变「引擎本身不预设市场」这条。
    """
    from .sizing import ASHARE_SIZER
    return BacktestEngine(strategy, sizer or ASHARE_SIZER, ASHARE_COST, ASHARE_RULES,
                          principal=principal, benchmark=benchmark, **kwargs)
