# coding:utf-8
"""上升系数 `QUADRANT_UPRISING_I_COEF_DUMMY`（`QUprICoefD`）—— **解耦后的编排**。

老树对同一列有**三条**写法（2026-10-19 实查后更正）
==================================================
1. `models/fourquad.py:717`（`calc_4Quad_credit_scores`）——
   `DUMMY = each[1]`（对参照序列的稳健回归）。**这条不是老树实际消费的值**。
2. `models/mainstream.py:252-261` —— 只产出 `CANDIDATE`（对数积水率，
   分形长度 = **MACD lag**）与 `COEF`（`each[1]`），**不写 DUMMY**。
3. `analysis/zen_pivot.py:272-285` —— 真正写 `DUMMY` 的地方：
   `DUMMY = CANDIDATE × log10(有效 lag)`。

上一版按 1 号搬，结果与老树"相去甚远"（用户 2026-10-19 肉眼比对）。
现按 **2 + 3** 编排：`CANDIDATE` 的算术仍复用 :func:`calc_wavlet_coef_of_fractal`
（`getPoolRate` 逐字对拍过），`DUMMY` 改为老树的乘积式。

依赖盘点
========
=========================  ===============================================
需要                         状态
=========================  ===============================================
`get_peak_high_price`      本模块（8 行）
`getPoolRate`              本模块（对数积水率）
`getPoolRate_vX`           本模块（稳健版；只服务老树的 `COEF` 列，**不写 DUMMY**）
`LINEAREG_PRICE`           `talib.LINEARREG(close, 20)` —— 一行
`QUADRANT_LEVERAGE_MACD_TIMING_LAG_DUMMY`  `analysis/quadrant.py` 产出
`AKA.CLOSE`                行情自带
`ZEN_DASH_COUNT` / `POWERLINE_UPRISING_LAG`  **可选**（决定 log10 是否夹到 10；
                           缺省时退化为直接用 lag，见 :func:`_effective_lag`）
=========================  ===============================================

公式（**已被老树实际消费的那条**，2026-10-19 二次核对）
=====================================================
⚠️ `fourquad.py:717` 那条（`DUMMY = each[1]`，对参照序列的稳健回归）**不是**
老树流水线最终落库/展示的值。实际链路是：

```python
# models/mainstream.py:252-261 —— 分形长度用 **MACD lag**，不是 energy lag
ret_foo = features.groupby(level=[0]).apply(
    calc_wavlet_coef_of_fractal_vXI(..., column=QUADRANT_LEVERAGE_MACD_TIMING_LAG_DUMMY,
                                    price=AKA.CLOSE))
COEF_CANDIDATE = each[0]                       # 对数积水率（round 4）

# analysis/zen_pivot.py:272-285 —— DUMMY = CANDIDATE × log10(有效 lag)
COEF_DUMMY = COEF_CANDIDATE * np.log10(
    lag > 10 ? (条件成立 ? lag : 10) : 10)
```

条件 = `ZEN_DASH_COUNT > 0.618` 或
(`ZEN_DASH_COUNT < 0.618` 且 `QUADRANT_LEVERAGE_POWERLINE_UPRISING_TIMING_LAG > 1e-12`)。

**实测**（301308，2099 根 60min，老树 2026-10-19 现跑）：
`DUMMY == round4(CANDIDATE) * log10(有效 lag)` **1904/1904 行逐位相等**
（最大差 `8.9e-16`）。旧值 ~1.05 正是 `0.55 × log10(89)` 这类乘积 ——
用 `get_pool_rate_vX` 永远算不出来。

⚠️ 老树 `calc_wavlet_coef_of_fractal_vXI` 的 `price_name` 默认值写成
`FLD.ZEN_WAVELET_TIMING_LAG` 是**笔误**（调用点都显式传 `AKA.CLOSE`）。本模块
**必须**显式传 `price_name`，不给那个默认值。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from GolemQ.core.constants import AKA, FIELD
from GolemQ.models.alias import LTT

__all__ = [
    'get_peak_high_price', 'get_pool_rate', 'get_pool_rate_vX',
    'calc_lineareg_price', 'calc_wavlet_coef_of_fractal', 'calc_uprising_i_coef',
]

#: 参照序列的回看周期 —— 老树 `features/push_credits.py:454`
#: `talib.LINEARREG(close, timeperiod=20)`。
LINEAREG_PERIOD = 20

#: 分形长度取不到时（NaN）用 -1 表示「这次不算」—— 老树同口径。
_NO_LENGTH = -1


def get_peak_high_price(logVs) -> np.ndarray:
    """**历史最高**的累积序列（`logHs[i] = max(logVs[:i+1])`）。纯函数。

    >>> import numpy as np
    >>> get_peak_high_price(np.array([1.0, 3.0, 2.0, 5.0, 4.0])).tolist()
    [1.0, 3.0, 3.0, 5.0, 5.0]
    """
    x = np.asarray(logVs, dtype=np.float64)
    if len(x) == 0:
        return x
    return np.maximum.accumulate(x)


def _ninfp1_to_n1p1(x):
    """`[-inf, +inf) → [-1, 1)` 的压缩（老树那个 `ninfp1_to_n1p1`）。"""
    with np.errstate(invalid='ignore', divide='ignore'):
        return 2 / (2 - x) - 1


def get_pool_rate(logVs) -> float:
    """**对数积水率**（老树 `analysis/pool_coef.py:53`，逐字）。

    `gain / lost_and_gain` → 压缩到 `[-1, 1)`。`gain == 0` ⇒ 0；分母 0 ⇒ `-inf`。

    >>> round(float(get_pool_rate(np.array([1.0, 2.0, 4.0]))), 6)
    1.0
    >>> float(get_pool_rate(np.array([1.0, 1.0, 1.0])))
    0.0
    """
    x = np.asarray(logVs, dtype=np.float64)
    if len(x) == 0:
        return np.nan
    hs = get_peak_high_price(x)
    gain = np.sum(x) - x[0] * len(x)
    lost_and_gain = np.sum(hs) - hs[0] * len(hs)
    if gain == 0:
        rate = 0.0
    elif lost_and_gain == 0:
        rate = -np.inf
    else:
        rate = gain / lost_and_gain
    return float(_ninfp1_to_n1p1(rate))


def get_pool_rate_vX(logVs, regression) -> float:
    """**稳健对数积水率**：把 `gain` 换成参照序列（线性回归价）端的增量。

    `gain = (regression[-1] - regression[0]) * n / 2`；
    `lost_and_gain = |陪跑部分 + gain|`。

    >>> import numpy as np
    >>> x = np.array([1.0, 2.0, 4.0]); r = np.array([1.0, 2.0, 3.0])
    >>> isinstance(get_pool_rate_vX(x, r), float)
    True
    """
    x = np.asarray(logVs, dtype=np.float64)
    reg = np.asarray(regression, dtype=np.float64)
    if len(x) == 0 or len(reg) != len(x):
        return np.nan
    hs = get_peak_high_price(x)
    raw_gain = np.sum(x) - x[0] * len(x)
    lost_and_gain_hist = np.sum(hs) - hs[0] * len(hs)
    lost = lost_and_gain_hist - raw_gain

    gain = (reg[-1] - reg[0]) * len(x) / 2
    denom = np.abs(lost + gain)
    if gain == 0:
        rate = 0.0
    elif denom == 0:
        rate = -np.inf
    else:
        rate = gain / denom
    return float(_ninfp1_to_n1p1(rate))


def calc_lineareg_price(close, timeperiod: int = LINEAREG_PERIOD) -> np.ndarray:
    """`LINEAREG_PRICE` = `talib.LINEARREG(close, timeperiod)`（老树 `push_credits.py:454`）。

    这是 `calc_wavlet_coef_of_fractal` 的**参照序列**（老树在没有
    `POLYNOMIAL9_WEEKLY` 时就用它）。
    """
    import talib
    return talib.LINEARREG(np.asarray(close, dtype=np.float64), timeperiod=timeperiod)


def calc_wavlet_coef_of_fractal(features: pd.DataFrame,
                                column_name: str,
                                price_name: str = AKA.CLOSE,
                                regression=None):
    """逐 bar 算 `(COEF_CANDIDATE, COEF_DUMMY)`。**逐字搬**老树
    `analysis/pool_coef.py:299 calc_wavlet_coef_of_fractal_vXI`。

    每一步：取该 bar 的**分形长度** `L = features[column_name]`，回看
    `[i-L, i]` 这一段价格——

    * `COEF_CANDIDATE = get_pool_rate(该段价格)`
    * `COEF_DUMMY     = get_pool_rate_vX(该段价格, 同段参照序列)`

    :param regression: 参照序列（与 `features` 等长）。``None`` ⇒ 现算
        `calc_lineareg_price(close)`。
    :returns: 两个 `float64` 数组（与 `features` 等长，算不了的 bar 是 `NaN`）。
    """
    if column_name not in features.columns:
        raise KeyError('缺少分形长度列 {}'.format(column_name))
    lens = pd.to_numeric(features[column_name], errors='coerce').to_numpy(np.float64)
    price = pd.to_numeric(features[price_name], errors='coerce').to_numpy(np.float64)
    if regression is not None:
        ref = np.asarray(regression, dtype=np.float64)
    elif FIELD.POLYNOMIAL9_WEEKLY in features.columns:
        # ⚠️ **老树的优先分支**（`pool_coef.py:299` 的 `if FLD.POLYNOMIAL9_WEEKLY in
        # target_features`）：有周线 Poly9 就用它，**没有**才退到 `LINEAREG_PRICE`。
        # 2026-10-19 用户肉眼比对发现两棵树值差很远，根因就是这里 ——
        # 之前只实现了 `LINEAREG` 那一条。
        ref = pd.to_numeric(features[FIELD.POLYNOMIAL9_WEEKLY],
                            errors='coerce').to_numpy(np.float64)
    else:
        ref = calc_lineareg_price(price)

    n = len(features)
    cand = np.full(n, np.nan)
    dummy = np.full(n, np.nan)
    for i in range(n):
        raw = lens[i]
        length = _NO_LENGTH if (raw is None or np.isnan(raw)) else int(raw)
        start = i - length
        if length < 0 or start < 0 or start == i:
            continue
        window = price[start:i + 1]
        cand[i] = round(get_pool_rate(window), 4)
        dummy[i] = round(get_pool_rate_vX(window, ref[start:i + 1]), 4)
    return cand, dummy


def _effective_lag(features: pd.DataFrame, lag: np.ndarray) -> np.ndarray:
    """`log10` 的输入：**截到 10** 的 MACD lag（老树 `zen_pivot.py:276-284`）。

    条件列（`4QuadZenDCnt` / `QLevPwrlnUprLag`）**在场**时按老树原式判；
    不在场时退化为「`lag > 10` 就用 lag」—— 这是**已记录的偏差**：老树 301308
    1904 根里有 37 根（1.9%）因条件不成立被夹到 10（集中在 2024 年早期）。
    2026-10-19 起生产者已按最小表达式补齐（`quadrant.calc_quadrant_zen_dash_count`
    / `calc_quadrant_powerline_uprisal_lag`），示例页与生产链应**显式传这两列**。

    >>> import pandas as pd, numpy as np
    >>> _effective_lag(pd.DataFrame(index=range(3)), np.array([4.0, 12.0, np.nan])).tolist()
    [10.0, 12.0, 10.0]
    """
    base = np.where(lag > 10, lag, 10.0)
    zen_col = LTT.QUADRANT_ZEN_DASH_COUNT
    pwr_col = LTT.QUADRANT_LEVERAGE_POWERLINE_UPRISING_TIMING_LAG
    if zen_col in features.columns and pwr_col in features.columns:
        zen = pd.to_numeric(features[zen_col], errors='coerce').to_numpy(np.float64)
        pwr = pd.to_numeric(features[pwr_col], errors='coerce').to_numpy(np.float64)
        cond = (zen > 0.618) | ((zen < 0.618) & (pwr > 1e-12))
        base = np.where((lag > 10) & cond, lag, 10.0)
    return base


def calc_uprising_i_coef(features: pd.DataFrame,
                         column_name: str = LTT.QUADRANT_LEVERAGE_MACD_TIMING_LAG_DUMMY,
                         price_name: str = AKA.CLOSE) -> pd.DataFrame:
    """在 `features` 上写 **两列**（老树实际消费的那条链，见模块 docstring）：

    * `LTT.QUADRANT_UPRISING_I_COEF_CANDIDATE`（`QUprICoefCan`）
      —— **分形长度取 `column_name`（默认 MACD lag）** 的对数积水率；
    * `LTT.QUADRANT_UPRISING_I_COEF_DUMMY`（`QUprICoefD`）
      —— `CANDIDATE × log10(有效 lag)`，**不是**对参照序列的稳健回归。

    ⚠️ `COEF_DUMMY` 与参照序列**无关** —— 这正是 2026-10-19 上一版把它错接成
    `get_pool_rate_vX`（老树 `fourquad.py:717` 的返回值，实际对应另一列
    `OMNIPATH_UPRISING_I_COEF`）的原因。需要那列时直接调
    :func:`calc_wavlet_coef_of_fractal` 取第二个返回值。
    """
    cand, _coef = calc_wavlet_coef_of_fractal(
        features, column_name=column_name, price_name=price_name)
    lag = pd.to_numeric(features[column_name], errors='coerce').to_numpy(np.float64)
    with np.errstate(invalid='ignore', divide='ignore'):
        features[LTT.QUADRANT_UPRISING_I_COEF_CANDIDATE] = cand
        features[LTT.QUADRANT_UPRISING_I_COEF_DUMMY] = \
            cand * np.log10(_effective_lag(features, lag))
    return features
