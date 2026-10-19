# coding:utf-8
"""技术指标的小工具 —— 目前只有 `TA_HMA`。

为什么自实现 `TA_HMA`
====================
它原本取自 **QUANTAXIS**（`QAIndicator/talib_numpy.py:178`），而 QUANTAXIS 已按
`DECISIONS.md` **D12** 从全树剔除。好消息是它**不是 QUANTAXIS 的独有算法** ——
本体只是 **talib 的两行组合**，照抄公式即可，没有移植风险：

    HMA = WMA( 2·WMA(n/2) − WMA(n), sqrt(n) )

⚠️ `TA_HMA` 是先搬的；`QA_indicator_MACD` 于 **2026-10-19** 补搬
（:func:`calc_macd_qa`）—— 上升系数链的 `MsLeadinx5BfMin` / `MACD_CROSS_SX_BEFORE`
**必须**用 QUANTAXIS 口径，见该函数 docstring。
同族的 `CROSS` 仍未搬（用到的两处一行就能写，`quadrant.py` 内联）。

⚠️ **对拍怎么做**：老树那份来自 QUANTAXIS 包，而 `test_no_quantaxis` 有一条
「运行时不许加载 QUANTAXIS」的断言 ⇒ **测试里不能 import 它做对拍**。
所以本模块的验证是：① 结构性钉子（暖机长度、线性序列的精确性）；
② **真正的验证放在 Poly9 编排那一步的端到端对拍**（`calc_polynomial9_vXIs` 与
老树逐值比 —— 那里 HMA10 是被真正消费的）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import talib
    _TALIB_ERROR = None
except ImportError as exc:            # pragma: no cover
    talib = None
    _TALIB_ERROR = exc

__all__ = ['available', 'TA_HMA', 'calc_macd_qa']


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


def calc_macd_qa(close, short: int = 12, long: int = 26, mid: int = 9):
    """**QUANTAXIS 口径**的 MACD（`QA_indicator_MACD`，逐字等价）。

    老树流水线的 15min / 60min / 日线特征**都用它**（`features/empirical.py:3431`
    `models/mainstream.py:1053`），与 `talib.MACD` **不是同一个东西**：

    * EMA 用 `pandas.ewm(span=N, min_periods=N-1, adjust=True)` —— **不是** talib 的
      递推初始化；
    * 柱 `MACD = (DIF - DEA) * 2` —— talib 的第三项只有 `DIF - DEA`（差 2 倍）。

    ⚠️ 2026-10-19 实测：用 talib 口径喂 `calc_mainstream_feats`，`MsLeadinx5BfMin`
    有 **1364/2075** 根与老树不同（最大 5070）—— 不是符号差，是 EMA 初始化 +
    递推差在阈值附近翻转。**这条链必须用本函数**。

    :returns: `(dif, dea, macd)` 三个 `pd.Series`（与输入同索引）
    """
    s = close.astype(float) if isinstance(close, pd.Series) else pd.Series(
        np.asarray(close, dtype=float))

    def _ema(x, n):
        return x.ewm(span=n, min_periods=n - 1, adjust=True).mean()

    dif = _ema(s, short) - _ema(s, long)
    dea = _ema(dif, mid)
    return dif, dea, (dif - dea) * 2
