# coding:utf-8
"""交易规则 —— **中立的参数容器**，不预设任何市场的规则。

同 `costs.py` 的道理：T+N、整手、涨跌停都是**通用概念**，数值因市场而异。
A 股是 T+1、100 股一手、有涨跌停；港股 T+0、每手股数因股而异；美股 T+0、1 股。
把 `T+1` 写死在引擎里，引擎就只服务 A 股了。
"""
from __future__ import annotations

from dataclasses import dataclass

__all__ = ['TradeRules', 'ASHARE_RULES']


@dataclass(frozen=True)
class TradeRules:
    """一次回测的交易规则假设。

    :param t_plus: T+N 的 N。**1 = 当日买入不可卖出**（A 股）；0 = 可当日回转。
        注意本实现按**自然持有期**判断：`t_plus=1` 表示买入次日起才可卖。
    :param lot_size: 一手股数。买入必须为其整数倍；0 或 1 表示不限制。
    :param limit_check: 是否检查涨跌停。开启时「一字板」不会成交 ——
        不做这项检查的回测会凭空得到买在涨停、卖在跌停的收益。
    :param limit_pct: 涨跌停幅度（0.10 = ±10%）。仅当 `limit_check` 为真时使用。
    :param warmup_bars: 显式区间回测时，起点前需要的前置预热 bar 数。
        特征计算需要历史窗口，预热不足会让区间开头的特征全为 NaN ——
        那不是「策略没信号」，是「数据不够」。
    """

    t_plus: int = 1
    lot_size: int = 100
    limit_check: bool = True
    limit_pct: float = 0.10
    warmup_bars: int = 10

    def can_sell(self, buy_date, current_date) -> bool:
        """按 T+N 判断能否卖出。

        **按自然日序号差判断，不按 bar 数** —— 参照实现的口径是「已持仓 ≥ 1 个
        交易日」，用 bar 数会在半日市或停牌时算错。
        """
        if self.t_plus <= 0:
            return True
        return (current_date - buy_date) >= self.t_plus


#: A 股口径预设。来源：`GolemQ_old` 的 `benchmark/zen_bt.py` 文档头
#: （T+1、100 股一手、本金可作参数）。
#:
#: **这是预设不是默认值** —— 引擎要求调用方显式传入。
ASHARE_RULES = TradeRules(t_plus=1, lot_size=100, limit_check=True, limit_pct=0.10)
