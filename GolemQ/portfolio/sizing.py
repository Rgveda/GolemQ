# coding:utf-8
"""仓位分配 —— 决定「同时持几只」与「每只多少钱」。

这是「**优化仓位工具**」这个名字的由来：不是选股，是**在已选出的候选里分配资金**。

与策略的分工
============
* 策略决定**谁能拿到资金**（`strategy.Strategy.priority`，值大者优先）
* 本模块决定**拿到多少**（槽位数与单槽金额）

参照实现的文档专门强调了这条解耦：

    现金耗尽后剩余候选落空；仓位大小仍由梯形/等权槽位决定，**与优先顺序解耦**。

混在一起时，调仓位大小会连带改变选股结果，回测就不可解释了。

核心设计意图（来自参照实现，值得保留）
======================================
**权益增长时加的是槽位数、不是单仓规模。**

每槽恒定一个金额（A 股口径 ~10 万元），持仓越分散、单仓越小，从而压低冲击成本：

    33 槽满仓时单仓可达权益的 3%，300 槽时降到 0.33%

这是「优化仓位」而非「放大仓位」—— 资金变大时优先降冲击成本，而不是集中押注。
"""
from __future__ import annotations

import abc

__all__ = ['PositionSizer', 'NotionalSlotSizer', 'EvenSizer', 'ASHARE_SIZER']


class PositionSizer(abc.ABC):
    """仓位分配契约。

    子类需实现 :meth:`target_slots` 与 :meth:`slot_notional`；
    :meth:`entry_fractions` 可选（默认一次性建仓）。
    """

    @abc.abstractmethod
    def target_slots(self, equity: float) -> int:
        """当前权益下**最多同时持有几只**。"""

    @abc.abstractmethod
    def slot_notional(self, equity: float, n_slots: int) -> float:
        """**单个槽位的目标金额**。`n_slots` 个槽位合计即为目标总仓位。"""

    def entry_fractions(self) -> tuple:
        """建仓批次。``(1.0,)`` = 一次性；``(0.5, 0.5)`` = 两批各半（梯形建仓）。

        返回的各批次份额之和应为 1.0；引擎会断言这一点 ——
        和不为 1 会让回测凭空多出或少掉仓位，且不报错。
        """
        return (1.0,)


class NotionalSlotSizer(PositionSizer):
    """**每 `notional` 元权益 1 个槽位，下限 `base`。** 参照实现的默认口径。

    实测口径（来自参照实现的文档，2026-09-16）：500万→50、1000万→100、
    2000万→200、3000万→300，即「每 500 万 +50 槽」，等价于每 10 万元 1 槽；
    权益回落时对称减少。
    """

    def __init__(self, notional: float = 100_000.0, base: int = 33,
                 ladder: tuple = (1.0,)):
        if notional <= 0:
            raise ValueError('notional 必须为正')
        self.notional = float(notional)
        self.base = int(base)
        self._ladder = tuple(ladder) or (1.0,)
        if abs(sum(self._ladder) - 1.0) > 1e-9:
            raise ValueError(f'entry_fractions 之和必须为 1.0，得到 {sum(self._ladder)}')

    def target_slots(self, equity: float) -> int:
        """当前权益下的槽位数。

        >>> s = NotionalSlotSizer(notional=100_000, base=33)

        口径来自参照实现（每 500 万 +50 槽，等价于每 10 万元 1 槽）：

        >>> [s.target_slots(e) for e in (5_000_000, 10_000_000, 20_000_000, 30_000_000)]
        [50, 100, 200, 300]

        低于下限时回落到 `base`（不是 0）：

        >>> s.target_slots(1_000_000)      # 1_000_000 // 100_000 = 10 < 33
        33
        >>> s.target_slots(0)
        33
        """
        return max(int(self.base), int(float(equity) // self.notional))

    def slot_notional(self, equity: float, n_slots: int) -> float:
        """单槽金额 = 权益 / 槽位数。

        注意：**不是**固定的 `self.notional`。固定金额会在权益变化时让总仓位
        无法跟随（33 槽 × 10 万 = 330 万，而权益 1000 万时只用了 33%），
        故按权益摊分；`notional` 只用来决定**槽位数**。

        >>> s = NotionalSlotSizer(notional=100_000, base=33)
        >>> round(s.slot_notional(3_300_000, 33), 2)
        100000.0

        槽位数变多则单仓变小 —— 这正是「优化仓位」而非「放大仓位」：

        >>> round(s.slot_notional(3_300_000, 330), 2)
        10000.0

        >>> s.slot_notional(1_000_000, 0)
        Traceback (most recent call last):
            ...
        ValueError: n_slots 必须为正
        """
        if n_slots <= 0:
            raise ValueError('n_slots 必须为正')
        return float(equity) / n_slots

    def entry_fractions(self) -> tuple:
        return self._ladder


class EvenSizer(PositionSizer):
    """**固定持仓只数，等权分配。** 用于「不按权益缩放」的对照实验。"""

    def __init__(self, n_slots: int = 10, ladder: tuple = (1.0,)):
        if n_slots <= 0:
            raise ValueError('n_slots 必须为正')
        self.n_slots = int(n_slots)
        self._ladder = tuple(ladder) or (1.0,)
        if abs(sum(self._ladder) - 1.0) > 1e-9:
            raise ValueError(f'entry_fractions 之和必须为 1.0，得到 {sum(self._ladder)}')

    def target_slots(self, equity: float) -> int:
        return self.n_slots

    def slot_notional(self, equity: float, n_slots: int) -> float:
        return float(equity) / int(n_slots)

    def entry_fractions(self) -> tuple:
        return self._ladder


#: A 股口径预设：每 10 万元 1 槽、下限 33 槽、一次性建仓。
#: 数值取自参照实现 `benchmark/zen_bt.py` 的 `SLOT_NOTIONAL` / `SLOT_MIN`。
#:
#: **这是预设不是默认值** —— 调用方应显式传入，避免跨场景误用。
#: 需要梯形建仓时传 `NotionalSlotSizer(ladder=(0.5, 0.5))`。
ASHARE_SIZER = NotionalSlotSizer(notional=100_000.0, base=33, ladder=(1.0,))
