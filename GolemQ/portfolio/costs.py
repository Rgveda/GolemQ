# coding:utf-8
"""交易成本模型 —— **中立的参数容器**，不预设任何市场的费率。

为什么不做成常量而做成数据类
============================
成本是**通用概念**（任何市场的回测都要算佣金、滑点、税），但**数值因市场而异**。
把 A 股的万0.85 写成模块常量，等于把市场写死；做成数据类后，
换市场就是换一个 :class:`CostModel` 实例。

A 股口径取自参照实现 `benchmark/zen_bt.py` 的模块常量（那里写死的），
在本模块末尾以 :data:`ASHARE_COST` 预设提供 —— **预设是数据，不是默认值**：
引擎不替你选，由调用方显式传入，避免「不传就默默按 A 股算」。
"""
from __future__ import annotations

from dataclasses import dataclass

__all__ = ['CostModel', 'ASHARE_COST']


@dataclass(frozen=True)
class CostModel:
    """一次回测的成本假设。

    拆分买卖两侧而不是只给一个 ``rate``：A 股的**印花税只在卖出收**，
    把两侧合成一个数会算错，且错得隐蔽（差额随换手率变化，不易察觉）。

    :param commission_rate: 佣金费率，**双边**，按成交额计
    :param commission_min: 单笔最低佣金（A 股 ¥5）。0 表示无下限
    :param stamp_rate: 印花税，**仅卖出**，按成交额计
    :param slippage: 单边滑点，按价格比例计（0.001 = 10bp）
    :param transfer_rate: 过户费，双边，按成交额计。A 股沪深口径不同，默认 0
    """

    commission_rate: float = 0.0
    commission_min: float = 0.0
    stamp_rate: float = 0.0
    slippage: float = 0.0
    transfer_rate: float = 0.0

    def buy_fee(self, turnover: float) -> float:
        """买入侧费用（佣金有下限、无印花税）。"""
        commission = max(turnover * self.commission_rate, self.commission_min) \
            if turnover > 0 else 0.0
        return commission + turnover * self.transfer_rate

    def sell_fee(self, turnover: float) -> float:
        """卖出侧费用（佣金有下限 + 印花税 + 过户费）。"""
        commission = max(turnover * self.commission_rate, self.commission_min) \
            if turnover > 0 else 0.0
        return commission + turnover * self.stamp_rate + turnover * self.transfer_rate

    def fill_price(self, price: float, side: str) -> float:
        """滑点后的成交价。买入抬价、卖出压价 —— **方向不能反**，
        反了会让回测凭空盈利，且参数越极端盈利越好看，非常难自查。"""
        if side.upper().startswith('B'):
            return price * (1.0 + self.slippage)
        return price * (1.0 - self.slippage)


#: A 股口径预设。来源：`GolemQ_old` 的 `benchmark/zen_bt.py` 模块常量
#: （佣金万0.85 双边、单笔最低 ¥5、印花税卖出万5、滑点 10bp）。
#:
#: **这是预设不是默认值** —— 引擎要求调用方显式传入，以免跨市场误用。
ASHARE_COST = CostModel(
    commission_rate=0.000085,
    commission_min=5.0,
    stamp_rate=0.0005,
    slippage=0.001,
    transfer_rate=0.0,
)
