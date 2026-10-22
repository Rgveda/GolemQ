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
同族的 `CROSS` 于 **2026-10-21** 补搬为 :func:`cross_above`（markup rank 用）。

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

__all__ = ['available', 'TA_HMA', 'calc_macd_qa', 'cross_above',
           'atr_rsi_stops', 'atr_supertrend', 'lineareg_band']


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


def cross_above(a, b):
    """`A` **上穿** `B` —— 与 QUANTAXIS `CROSS(A, B)` 逐字等价。

    返回 int 序列：上穿那一根为 **1**，其余 **0**。老实现（QUANTAXIS
    `QAIndicator/talib_numpy.py::CROSS`）：

    ``np.where(A < B, 1, 0)`` 再取 ``diff() < 0`` —— 即「上一根 A<B、这一根
    A>=B」。⚠️ 方向容易记反：``CROSS(DEA, DIF) == 1`` 是 **DEA 上穿 DIF**，
    也就是 DIF 下穿 DEA（MACD 死叉）。

    >>> a = pd.Series([1.0, 3.0, 2.0, 4.0])
    >>> b = pd.Series([2.0, 2.0, 3.0, 3.0])
    >>> cross_above(a, b).tolist()
    [0, 1, 0, 1]
    """
    a_s = a if isinstance(a, pd.Series) else pd.Series(np.asarray(a, dtype=float))
    b_s = b if isinstance(b, pd.Series) else pd.Series(np.asarray(b, dtype=float))
    var = np.where(a_s.values < b_s.values, 1, 0)
    return (pd.Series(var, index=a_s.index).diff() < 0).astype(int)


def atr_rsi_stops(ohlc: pd.DataFrame, period: int = 10):
    """ATR RSI Stops（旧 `signal/base.py::ATR_RSI_Stops_v2`，逐字搬）。

    :returns: ``(stop_line, direction)``；``direction`` ∈ {-1, 0, 1}。

    ⚠️ 旧实现的 `period` 形参**未被使用**（内部写死 EMA/ATR 10）；保留签名兼容。
    """
    from GolemQ.analysis.timing import Timeline_duration
    close = ohlc['close'].to_numpy(dtype=float)
    high = ohlc['high'].to_numpy(dtype=float)
    low = ohlc['low'].to_numpy(dtype=float)
    tp = (ohlc['open'].to_numpy(dtype=float) + high + low + close) / 4.0
    rsi_ma = talib.EMA(tp, 10)
    atr = talib.ATR(high, low, close, 10)
    top_line = rsi_ma + atr
    bottom_line = rsi_ma - atr
    jx = Timeline_duration(np.where(close > top_line, 1, 0))
    sx = Timeline_duration(np.where(bottom_line > close, 1, 0))
    direction = np.where(jx < sx, 1, np.where(sx < jx, -1, 0)).astype(np.float32)
    stop_line = (rsi_ma - direction * atr).astype(np.float32)
    return stop_line, direction


def _predict_atr_supertrend(src, up, dn):
    """旧 numba 核 `predict_ATR_SuperTrend_up_down` 的纯 numpy 版。"""
    n = len(src)
    t_up = np.full(n, np.nan)
    t_dn = np.full(n, np.nan)
    trend = np.full(n, np.nan)
    for i in range(1, n):
        t_up[i] = max(up[i], t_up[i - 1]) if src[i - 1] > t_up[i - 1] else up[i]
        t_dn[i] = min(dn[i], t_dn[i - 1]) if src[i - 1] < t_dn[i - 1] else dn[i]
        trend[i] = (1 if src[i] > t_dn[i - 1]
                    else (-1 if src[i] < t_up[i - 1] else trend[i - 1]))
    return t_up, t_dn, trend


def atr_supertrend(ohlc: pd.DataFrame, length: int = 12, factor: float = 3.0):
    """ATR SuperTrend（旧 `signal/base.py::ATR_SuperTrend_cross_v2`，逐字搬）。

    :returns: ``(tsl, trend)``；``trend`` ∈ {-1, 1}。

    ⚠️ 旧实现的 `length` / `factor` 形参**未被使用**（内部写死 ATR 12、Factor 3）。
    """
    close = ohlc['close'].to_numpy(dtype=float)
    high = ohlc['high'].to_numpy(dtype=float)
    low = ohlc['low'].to_numpy(dtype=float)
    atr = talib.ATR(high, low, close, 12)
    up = (high + low) / 2.0 - 3.0 * atr
    dn = (high + low) / 2.0 + 3.0 * atr
    t_up, t_dn, trend = _predict_atr_supertrend(close, up, dn)
    tsl = np.where(trend == 1, t_up, t_dn).astype(np.float32)
    return tsl, trend.astype(np.float32)


def lineareg_band(ohlc: pd.DataFrame, nATR: int = 14, nlookback: int = 20,
                  scale: float = 1.0):
    """线性回归 + ATR 通道（旧 `indices/base.py::lineareg_band`，逐字搬）。

    :returns: ``(lrc, lrc_u, lrc_l, direction)``
    """
    close = ohlc['close']
    lrc = talib.LINEARREG(close, timeperiod=nlookback)
    atr = talib.ATR(ohlc['high'], ohlc['low'], close, timeperiod=nATR)
    lrc_u = lrc + scale * atr
    lrc_l = lrc - scale * atr
    direction = np.where(lrc > lrc.shift(1), 1,
                         np.where(lrc < lrc.shift(1), -1, 0))
    return lrc, lrc_u, lrc_l, direction
