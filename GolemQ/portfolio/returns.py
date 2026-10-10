# coding:utf-8
"""持仓收益的计算 —— 纯函数，为 JIT / Cython 而写成显式循环。

从旧树 `GolemQ_old/portfolio/utils.py` 搬回（`analysis/regtree.py` 要用）。
新树的 `portfolio/` 原只有 costs / engine / rules / sizing / strategy 五个模块，
**没有** utils —— 所以这一个独立成模块，而不是往回造一个 `utils.py` 杂物间。
"""
from __future__ import annotations

import numpy as np

__all__ = ['calc_onhold_returns_np']


def calc_onhold_returns_np(closep: np.ndarray, daily_position: np.ndarray,
                           long: int = 1) -> np.ndarray:
    """**当前持仓的浮动收益**；一次持仓结束时（仓位归 0）下一根起重新计。

    旧树注释：写成 `np.ndarray` + 显式 `for`，就是为了**支持 JIT / Cython 加速**。

    :param closep: 收盘价序列
    :param daily_position: 每日仓位（``>0`` 视为持仓）；``NaN`` 的 bar **跳过**
        （跳过的是「算收益」这一步，不是「更新持仓状态」）
    :param long: ``1`` 做多 / ``-1`` 做空（其它值直接 `assert` 失败）
    :return: 与 `closep` 等长的浮动收益率

    >>> import numpy as np
    >>> close = np.array([10., 11., 12., 11.])
    >>> pos = np.array([1., 1., 0., 0.])         # 前两根持仓，第三根起空仓
    >>> calc_onhold_returns_np(close, pos).tolist()
    [0.0, 0.1, 0.2, 0.0]

    做空只是整体取负：

    >>> calc_onhold_returns_np(close, pos, long=-1).tolist()
    [-0.0, -0.1, -0.2, -0.0]
    """
    ret_onhold_returns = np.zeros(len(closep),)
    onhold_price = onhold_returns = 0.0     # noqa: F841 与旧树逐字一致（未使用）
    onhold_position = False
    assert long == 1 or long == -1
    for i in range(0, len(daily_position)):
        if (np.isnan(daily_position[i])):
            continue

        if (onhold_position):
            if (daily_position[i - 1] <= 0):
                onhold_price = closep[i]
        else:
            onhold_price = closep[i]
        if (onhold_price > 0.001):
            ret_onhold_returns[i] = (closep[i] - onhold_price) / onhold_price * long

        if (daily_position[i] > 0):
            onhold_position = True

        if (daily_position[i] <= 0):
            onhold_position = False

    return ret_onhold_returns
