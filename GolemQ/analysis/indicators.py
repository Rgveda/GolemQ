# coding:utf-8
"""技术指标的小工具 —— 目前只有 `TA_HMA`。

为什么自实现 `TA_HMA`
====================
它原本取自 **QUANTAXIS**（`QAIndicator/talib_numpy.py:178`），而 QUANTAXIS 已按
`DECISIONS.md` **D12** 从全树剔除。好消息是它**不是 QUANTAXIS 的独有算法** ——
本体只是 **talib 的两行组合**，照抄公式即可，没有移植风险：

    HMA = WMA( 2·WMA(n/2) − WMA(n), sqrt(n) )

⚠️ **只搬了 `TA_HMA`。** 同族的 `CROSS` / `QA_indicator_MACD` **没有搬** ——
实测它们在老树里的调用点**全在 `benchmark/*`**，而唯一会用 MACD 的正式链路
（Magic Nine Turns 的 `_baseline`）**本轮挂账**（见 `docs/decisions/D49.md` 第五节）。
**照 D39/D40「只搬用到的」** ⇒ 没有消费方就不搬；将来真要，再按需补。

⚠️ **对拍怎么做**：老树那份来自 QUANTAXIS 包，而 `test_no_quantaxis` 有一条
「运行时不许加载 QUANTAXIS」的断言 ⇒ **测试里不能 import 它做对拍**。
所以本模块的验证是：① 结构性钉子（暖机长度、线性序列的精确性）；
② **真正的验证放在 Poly9 编排那一步的端到端对拍**（`calc_polynomial9_vXIs` 与
老树逐值比 —— 那里 HMA10 是被真正消费的）。
"""
from __future__ import annotations

import numpy as np

try:
    import talib
    _TALIB_ERROR = None
except ImportError as exc:            # pragma: no cover
    talib = None
    _TALIB_ERROR = exc

__all__ = ['available', 'TA_HMA']


def available() -> bool:
    """`talib` 在不在（`TA_HMA` 全靠它）。"""
    return talib is not None


def TA_HMA(close, period: int):
    """**赫尔移动平均**（Hull Moving Average）。

    逐字复刻 QUANTAXIS `QAIndicator/talib_numpy.py:178`：

        HMA = WMA( 2·WMA(n/2) − WMA(n), sqrt(n) )

    它比同周期的普通均线**更贴价、滞后更小** —— 先做「短均线的两倍减长均线」
    （消滞后），再用 `sqrt(n)` 长的 WMA 把它抹平（消过冲）。

    ⚠️ **暖机长度 = `n/2 + sqrt(n) − 2 + 1`**，实测 `period=10` ⇒ **前 11 个是 `NaN`**。
    老树正是靠这个数字写 `np.r_[close[:11], HMA10[11:]]` 去拼 `POLYNOMIAL9` 的输入
    （见 `features/base.py:2102`）—— **那 11 与 `period=10` 是绑死的**，
    改 `period` 就得改拼接位置。

    >>> import numpy as np
    >>> y = TA_HMA(np.arange(60, dtype=float), 10)
    >>> bool(np.isnan(y[:11]).all())        # 前 11 个是暖机区
    True
    >>> bool(np.isfinite(y[11:]).all())
    True
    >>> round(float(np.diff(y[11:]).mean()), 9)   # 线性输入 ⇒ 输出也线性（斜率恒 1）
    1.0

    :param close: 一维价格序列（`list` / `np.ndarray` 皆可）
    :param period: 周期（`int`）。**传 10 就是 `HMA10`**
    :returns: `np.ndarray`，与输入等长，暖机区为 `NaN`
    """
    if talib is None:                 # pragma: no cover
        raise ImportError(
            'TA_HMA 需要 talib（{}）。本项目把它列在 MIN_PACKAGES 里，'
            '装一下：pip install TA-Lib'.format(_TALIB_ERROR))
    # ⚠️ `int(period/2)` 与 `int(np.sqrt(period))` **都要截断取整** —— 这是老树
    # （也就是 QUANTAXIS）的写法。talib 的 period 要 int，传 float 会抛。
    return talib.WMA(
        2 * talib.WMA(close, int(period / 2)) - talib.WMA(close, period),
        int(np.sqrt(period)))
