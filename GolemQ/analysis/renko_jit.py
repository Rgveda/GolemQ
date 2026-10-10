# coding:utf-8
"""`renko` 的 **numba 加速版**（`analysis/renko.py` 的 jit 对照实现）。

为什么要另写一份而不是给 `renko.py` 挂装饰器
============================================
`renko.py` 是**逐字对拍过**的保真搬运（旧树 `indices/renko.py`），
`class renko` 的状态全在 **Python list** 里（`renko_prices` / `renko_directions` /
上下影 / gaps），`build_history` 又边走边 `append`。给这种结构挂 `@njit` 就是把
「对拍基准」本身改掉 —— 之后**再没有东西可以比对**。

所以这一版换个切法：**只把热点搬进 jit，`renko.py` 一个字不动**。

热点在哪（**实测口径**，2026-10-10，`000711` 60min）
===================================================
==============  ==========  ==========  ============
层               1200 bar    2400 bar    占端到端
==============  ==========  ==========  ============
`renko_chart`       0.03 ms     0.08 ms    **0.3%**
`evaluate_renko`    1.46 ms     4.35 ms    —（被调 17 / 39 次）
聚类那一层         24.2 ms     47.3 ms    **83.6% / 88.3%**
端到端             29.0 ms     53.5 ms    100%
==============  ==========  ==========  ============

⚠️ **要打的不是 `renko_chart`** —— 旧树**早就**给它挂上 `@nb.jit` 了，那是已经
做过的一轮优化，而实测它只占 **0.3%**。真正的大头是 `fminbound` 的**目标函数**：
每个 1200 bar 窗口跑一次 `fminbound`，而**每次迭代**都 `evaluate_renko()`，
在纯 Python 里重建整条砖序列。

`fminbound` 的迭代次数是算法固有的（`xtol≈1.5e-8`，实测 17~39 次），**省不掉**，
只能把每次迭代做快。而它是个「紧标量循环 + 每步 append 一个 Python list」的形状，
正是 numba 的靶心 —— 实测那一段 **654×**。

**端到端 ≈ 6×**（29.0 ms → 4.8 ms）。上限就是 6×：剩下 4.8 ms 是 pandas
特征装配 + `renko_chart`（0.3%），再往下压得动那一层了。

⚠️ 绝对数字要说清楚：单标的 29 ms 对人眼毫无差别，**这条优化是为全市场批处理
准备的**（5579 只 × ~2000 bar，53 ms → ~9 ms，即 ~5 分钟 → ~50 秒）。
在消费方（`fractal/v0–v7,v9`）搬过来之前，它是**前瞻性**的。

⚠️ 三处**刻意的偏离**（逐条记着，别当成"等价"）
=============================================
1. **只算 `directions`，不算价格序列。** `evaluate()` 只需要三样：
   `renko_directions` 全长、`len(source_prices)`、`len(renko_prices)`。
   故 jit 核只产出方向数组，**上下影 / gaps / `source_aligned` 一概不算** ——
   那些是另案 `calc_renko_atr_vX` 那条线要的，与本模块无关。
2. **砖序列走预分配数组 + 溢出检测，绝不静默截断。**
   ⚠️ 砖数**不是** O(n)：一次 `1→100` 的跳空配 `0.01` 的砖高就是一万块砖，而 `n=2`。
   第一版原型按 `n+16` 开缓冲区并 `break` —— 那是**静默算错**，而随机游走的对拍
   恰好没触发它。现在容量按**总变差** `Σ|Δp| / 砖高` 估（每推进一块砖至少要耗掉
   一个砖高的价格行程，这是上界），且核内 `m` **照数不误**、越界只不写，
   由封装层发现后**抛错**。
3. **`brick_size == 0` 显式抛 `ZeroDivisionError`**，与纯 Python 对齐：
   那边 `float / 0.0` 本来就抛；而 numba 里 `int(inf)` 是未定义行为，
   不显式拦会让两种实现行为分叉。⚠️ 这条路径**在旧树里本来就走不通**
   （`__get_optimal_brick_size` 的 `except` 会留 `brick_size=0.0`，随后目标函数抛错
   → 被 `renko_in_cluster_group` 的裸 `except` 兜住 → `optimal_brick_sfo` 未定义
   → `NameError`），见 `renko.py` 模块 docstring 的「已知缺陷」2/3。

⚠️ **本模块有一份无法消除的重复**：`renko_in_cluster_group_jit` 与
`renko_trend_cross_func_jit` 是 `renko.py` 同名函数的**近似拷贝**（只把目标函数
换成 jit 版）。之所以不共用，是因为那两个函数把 `evaluate_renko` /
`renko_in_cluster_group` 当**模块全局**取用，没有注入口。
**唯一的防线是对拍**（`test_renko_jit.py`）—— 所以那 14 列必须**逐值相同**，
不是"差得不多"。这条重复与 `regtree_jit` 同源（那里也照搬了一遍建树递归）。

关于 numba
==========
`numba` **在** `MIN_PACKAGES`（`cli/bootstrap.py`，2026-10-10 起），且 `renko.py`
自己就**模块级** `@nb.jit` 了 `renko_chart`。但本模块的 import 仍是**惰性**的
（`_nb()`）—— 与 `regtree_jit` / `peak` 一致：本模块是**可选加速器**，
`renko.py` 那条保真路径不依赖它。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.optimize as opt
import talib

from GolemQ.analysis.renko import renko, renko_chart
from GolemQ.core.constants import AKA, FIELD as FLD

__all__ = ['available', 'renko_jit_status', 'bricks_directions',
           'evaluate_renko_jit', 'renko_in_cluster_group_jit',
           'renko_trend_cross_func_jit']


def _nb():
    """惰性拿 numba —— 见模块 docstring 的「关于 numba」。"""
    import numba
    return numba


def available() -> bool:
    """numba 在不在。"""
    try:
        _nb()
        return True
    except ImportError:
        return False


def renko_jit_status() -> dict:
    """排障用：numba 版本 + jit 是否真的开了（`DISABLE_JIT` 环境变量会关掉它）。"""
    try:
        nb = _nb()
    except ImportError as exc:
        return {'available': False, 'reason': '{}: {}'.format(type(exc).__name__, exc)}
    cfg = getattr(nb, 'config', None)
    return {'available': True, 'numba': nb.__version__,
            'disable_jit': int(getattr(cfg, 'DISABLE_JIT', 0)) if cfg else None}


# --------------------------------------------------------------------------- #
# jit 核
# --------------------------------------------------------------------------- #
def _kernels():
    """把 njit 的核**惰性**建出来（并缓存）。

    返回 `_bricks_dirs`。放在函数里是因为 numba 的装饰器在 **import 期**就要跑，
    而本模块被间接导入时不该为此付代价（同 `regtree_jit` / `peak`）。
    """
    cached = getattr(_kernels, '_cache', None)
    if cached is not None:
        return cached
    njit = _nb().njit

    @njit(cache=True)
    def _bricks_dirs(prices, brick_size, rd):
        """砖序列扫描 —— `class renko.__renko_rule` 的**逐字**复刻，只产出方向。

        :param prices: 一维价格数组（`build_history(prices=...)` 的那个）
        :param brick_size: 砖高（**必须 > 0**，为 0 时调用方已拦）
        :param rd: **预分配**的输出数组，`rd[0]` 是初始方向 `0`
        :return: 真实砖数 `m`（含起始那颗）。⚠️ `m > len(rd)` 表示**没装下** ——
            此时数组里是**不完整**的，调用方必须抛错，不许拿去算。

        复刻要点（错一条就与纯 Python 分叉）：
        * `gap = int(float(p - last) / brick)` —— **向零截断**，不是 floor；
        * 顺势（`dir >= 0` 且 `gap > 0`，或 `dir <= 0` 且 `gap < 0`）⇒ 推 `|gap|` 块；
        * **反向**且 `|gap| >= 2` ⇒ 先推一块**双倍**高的（`2*brick`），再推 `|gap|-2` 块；
        * 反向且 `|gap| == 1` ⇒ **一块都不推**（旧实现那个 `is_new_brick` 留 False 的分支）。
        """
        n = len(prices)
        cap = len(rd)
        rd[0] = 0.0
        m = 1
        last = prices[0]
        for i in range(1, n):
            p = prices[i]
            gap = int(np.float64(p - last) / brick_size)
            if gap == 0:
                continue
            d_last = rd[m - 1]
            if (gap > 0 and d_last >= 0) or (gap < 0 and d_last <= 0):
                s = 1.0 if gap > 0 else -1.0
                for _ in range(0, abs(gap)):
                    if m < cap:
                        rd[m] = s
                    m += 1
                    last = last + brick_size * s
            elif abs(gap) >= 2:
                s = 1.0 if gap > 0 else -1.0
                if m < cap:
                    rd[m] = s
                m += 1
                last = last + 2.0 * brick_size * s
                for _ in range(2, abs(gap)):
                    if m < cap:
                        rd[m] = s
                    m += 1
                    last = last + brick_size * s
        return m

    _kernels._cache = _bricks_dirs
    return _bricks_dirs


def _capacity(prices, brick_size) -> int:
    """按**总变差**估缓冲容量（见模块 docstring 偏离 2）。

    每推进一块砖，renko 价至少要朝着价格方向走一个砖高 ⇒
    `砖数 <= Σ|Δp| / 砖高 + 常数`。多给 8 块余量，真正的越界由 :func:`bricks_directions` 抛错兜底。
    """
    tv = float(np.abs(np.diff(np.asarray(prices, dtype=np.float64))).sum())
    return int(tv / brick_size) + 8


def bricks_directions(prices, brick_size) -> np.ndarray:
    """`:func:`_kernels` 的公开封装 —— 算容量、跑核、**溢出就抛**。

    :return: 方向数组，`[0]` 是起始的 `0`，其后是 ±1。与
        ``renko().set_brick_size(False, brick_size); build_history(prices=…);
        get_renko_directions()`` 等价（长度与逐值都相同）。

    >>> import numpy as np
    >>> bricks_directions(np.array([100., 100., 95., 95., 99., 99., 103.]), 2.0).tolist()
    [0.0, -1.0, -1.0, 1.0, 1.0]
    """
    if not available():
        raise RuntimeError('numba 不可用 —— renko_jit 需要它（见 renko_jit_status()）')
    prices = np.ascontiguousarray(prices, dtype=np.float64)
    brick_size = float(brick_size)
    if brick_size == 0:
        # 与纯 Python 的 `float / 0.0` 对齐（见模块 docstring 偏离 3）
        raise ZeroDivisionError('float division by zero')
    if len(prices) == 0:
        return np.zeros(0, dtype=np.float64)
    rd = np.zeros(max(_capacity(prices, brick_size), 2), dtype=np.float64)
    m = int(_kernels()(prices, brick_size, rd))
    if m > len(rd):
        raise RuntimeError(
            'renko_jit: 砖数 {0} 超出缓冲区 {1} —— 容量估算失准，**不许**拿不完整的'
            '序列去算（见 renko_jit 模块 docstring 偏离 2）'.format(m, len(rd)))
    return rd[:m]


# --------------------------------------------------------------------------- #
# 目标函数（fminbound 用）—— 这一层是实测的热点
# --------------------------------------------------------------------------- #
def _evaluate_from_dirs(dirs, n_source, column_name):
    """``renko.evaluate()`` 的复刻：由**方向数组** + 原始长度算四个指标。

    ⚠️ `price_ratio` 用 **Python 除法**（`int / int`）算，与旧实现逐位一致；
    `len(dirs) == 0` 时同样抛 `ZeroDivisionError` —— **别改成 numpy 除法**
    （那会返回 inf/nan，两种实现就此分叉）。
    """
    balance = 0
    sign_changes = 0
    price_ratio = n_source / len(dirs)

    for i in range(2, len(dirs)):
        if dirs[i] == dirs[i - 1]:
            balance = balance + 1
        else:
            balance = balance - 2
            sign_changes = sign_changes + 1

    if sign_changes == 0:
        sign_changes = 1

    score = balance / sign_changes
    if score >= 0 and price_ratio >= 1:
        score = np.log(score + 1) * np.log(price_ratio)
    else:
        score = -1.0

    # ⚠️ 键名 `'sign_changes:'` 带冒号 —— 旧树如此，见 renko.evaluate 的注释
    return {'balance': balance, 'sign_changes:': sign_changes,
            'price_ratio': price_ratio, 'score': score}[column_name]


def evaluate_renko_jit(brick, history, column_name):
    """`:func:`renko.evaluate_renko` 的 jit 版 —— 返回值**逐值相同**。

    它是 `fminbound` 的目标函数，**实测 654×**（1.373 ms → 0.0021 ms，1200 bar）。

    >>> import numpy as np
    >>> round(float(evaluate_renko_jit(1.0, np.arange(10., 20., 0.6), 'score')), 12)
    1.165909434663

    ⚠️ 上面这组 **`price_ratio > 1` 且 `score > 0`**，才走得到 `log` 那条分支。
    砖数多于 bar 数时 `price_ratio < 1` ⇒ 恒返回 `-1.0`（与纯 Python 一致）：

    >>> evaluate_renko_jit(2.0, np.array([100., 95., 99., 103.]), 'score')
    -1.0
    """
    dirs = bricks_directions(history, brick)
    return _evaluate_from_dirs(dirs, len(history), column_name)


# --------------------------------------------------------------------------- #
# 端到端（与 renko.py 同名函数的近似拷贝 —— 见模块 docstring 末尾那条说明）
# --------------------------------------------------------------------------- #
def renko_in_cluster_group_jit(data: pd.DataFrame,
                               indices: pd.DataFrame = None,
                               maxlength=1200) -> np.ndarray:
    """`:func:`renko.renko_in_cluster_group` 的 jit 版 —— **只有目标函数不同**。

    其余（分窗、ATR 定界、`fminbound`、`ret_cluster_group` 的形状与写位）逐行照搬。
    `indices` 形参**同旧树一样未被使用**（保留签名）。
    """
    factor_atr = talib.ATR(high=np.double(data.high),
                           low=np.double(data.low),
                           close=np.double(data.close),
                           timeperiod=14)
    factor_atr = factor_atr[np.isnan(factor_atr) == False]

    nextsub_first = sub_first = 0
    nextsub_last = sub_last = (maxlength - 1)

    ret_cluster_group = np.zeros((len(data.close.values), 5),)
    while (sub_first == 0) or (sub_first < len(data.close.values)):
        if (len(data.close.values) > maxlength):
            closep = np.nan_to_num(data.close.values[sub_first:sub_last], nan=0)
            subdata = data.iloc[sub_first:sub_last, :]
            atr = factor_atr[sub_first:sub_last]
            if (sub_last + maxlength < len(data.close.values)):
                nextsub_first = sub_first + (maxlength - 1)
                nextsub_last = sub_last + maxlength
            else:
                nextsub_first = len(data.close.values) - maxlength
                nextsub_first = 0 if (nextsub_first < 0) else nextsub_first
                nextsub_last = len(data.close.values) + 1
        else:
            closep = data.close.values
            sub_last = nextsub_first = len(data.close.values)
            subdata = data
            atr = factor_atr
            nextsub_last = len(data.close.values) + 1

        try:
            optimal_brick_sfo = opt.fminbound(lambda x:
                                              -evaluate_renko_jit(brick=x,
                                                                  history=closep,
                                                                  column_name='score'),
                                              np.min(atr), np.max(atr), disp=0)
        except Exception:
            # ⚠️ 与 renko.py 一样保留裸兜底（旧树口径）：兜住后 `optimal_brick_sfo`
            # 未定义，下一行直接 NameError。**别顺手修**，否则两种实现的行为分叉。
            pass

        bricks_flex = renko_chart(subdata[AKA.CLOSE].values, optimal_brick_sfo)

        ret_cluster_group[sub_first + 1, 4] = optimal_brick_sfo

        if len(bricks_flex[:, 0]) > 1:
            ret_cluster_group[sub_first:sub_last, 0:4] = bricks_flex

        if (sub_last >= len(data.close.values)):
            break
        else:
            sub_first = min(nextsub_first, nextsub_last)
            sub_last = max(nextsub_first, nextsub_last)

    return ret_cluster_group


def renko_trend_cross_func_jit(data, indices=None):
    """`:func:`renko.renko_trend_cross_func` 的 jit 版。

    与纯版**只差一处**：L 族走 :func:`renko_in_cluster_group_jit`。
    S 族（`renko_chart`，本就是 jit）与四个 `*_TIMING_LAG` 的合成逐行照搬 ——
    所以出来的 14 列应当**逐值相同**，由 `test_renko_jit.py` 钉住。
    """
    from GolemQ.analysis.timing import Timeline_Integral, Timeline_duration

    if (len(data) < 30):
        if (indices is not None):
            ret_indices = pd.concat([indices,
                                     pd.DataFrame(columns=[FLD.RENKO_TREND_S_LB,
                                                           FLD.RENKO_TREND_S_UB,
                                                           FLD.RENKO_TREND_S,
                                                           FLD.RENKO_TREND_L_LB,
                                                           FLD.RENKO_TREND_L_UB,
                                                           FLD.RENKO_TREND_L,
                                                           FLD.RENKO_TREND,
                                                           FLD.RENKO_TREND_S_BEFORE,
                                                           FLD.RENKO_S_JX_BEFORE,
                                                           FLD.RENKO_TREND_L_BEFORE,
                                                           FLD.RENKO_TREND_L_JX_BEFORE],
                                                  index=data.index)], axis=1)
        else:
            ret_indices = pd.DataFrame(columns=[FLD.RENKO_TREND_S_LB,
                                                FLD.RENKO_TREND_S_UB,
                                                FLD.RENKO_TREND_S,
                                                FLD.RENKO_TREND_L_LB,
                                                FLD.RENKO_TREND_L_UB,
                                                FLD.RENKO_TREND_L,
                                                FLD.RENKO_TREND,
                                                FLD.RENKO_TREND_S_BEFORE,
                                                FLD.RENKO_S_JX_BEFORE,
                                                FLD.RENKO_TREND_L_BEFORE,
                                                FLD.RENKO_TREND_L_JX_BEFORE],
                                       index=data.index)
        return ret_indices

    # ⚠️ S 族的砖高仍需 `class renko` 的 ATR 中位数 —— 那是一次 `talib.ATR` 调用，
    # 与 jit 无关，故复用纯版；`bricks_directions` 只服务 fminbound 的目标函数。
    optimal_brick = renko().set_brick_size(auto=True,
                                           HLC_history=data[[AKA.HIGH,
                                                             AKA.LOW,
                                                             AKA.CLOSE,]])

    bricks_fixed = renko_chart(data.close.values, optimal_brick)

    if len(bricks_fixed[:, 0]) > 1:
        if (indices is not None):
            ret_indices = pd.concat([indices,
                                     pd.DataFrame(bricks_fixed,
                                                  columns=[FLD.RENKO_PRICE_S,
                                                           FLD.RENKO_TREND_S,
                                                           FLD.RENKO_TREND_S_LB,
                                                           FLD.RENKO_TREND_S_UB],
                                                  index=data.index)]
                                    , axis=1)
        else:
            ret_indices = pd.DataFrame(bricks_fixed,
                                       columns=[FLD.RENKO_PRICE_S,
                                                FLD.RENKO_TREND_S,
                                                FLD.RENKO_TREND_S_LB,
                                                FLD.RENKO_TREND_S_UB],
                                       index=data.index)

    try:
        if (ret_indices is not None):
            ret_indices = pd.concat([ret_indices,
                                     pd.DataFrame(renko_in_cluster_group_jit(data, indices),
                                                  columns=[FLD.RENKO_PRICE_L,
                                                           FLD.RENKO_TREND_L,
                                                           FLD.RENKO_TREND_L_LB,
                                                           FLD.RENKO_TREND_L_UB,
                                                           FLD.RENKO_OPTIMAL,],
                                                  index=data.index)], axis=1)
        else:
            ret_indices = pd.DataFrame(renko_in_cluster_group_jit(data, indices),
                                       columns=[FLD.RENKO_PRICE_L,
                                                FLD.RENKO_TREND_L,
                                                FLD.RENKO_TREND_L_LB,
                                                FLD.RENKO_TREND_L_UB,
                                                FLD.RENKO_OPTIMAL,],
                                       index=data.index)
    except Exception:
        # 与 renko.py 同：保真保留的裸兜底（见 renko.py「已知缺陷」2）
        ret_indices[FLD.RENKO_TREND_L] = ret_indices[FLD.RENKO_TREND_S]
        ret_indices[FLD.RENKO_TREND_L_LB] = ret_indices[FLD.RENKO_TREND_S_LB]
        ret_indices[FLD.RENKO_TREND_L_UB] = ret_indices[FLD.RENKO_TREND_S_UB]

    ret_indices[FLD.RENKO_TREND] = np.where((ret_indices[FLD.RENKO_TREND_L] == 1) &
                                            (ret_indices[FLD.RENKO_TREND_S] == 1), 1,
                                            np.where((ret_indices[FLD.RENKO_TREND_L] == -1) &
                                                     (ret_indices[FLD.RENKO_TREND_S] == -1), -1, 0))

    with np.errstate(invalid='ignore', divide='ignore'):
        renko_trend_l_jx = Timeline_Integral(np.where(ret_indices[FLD.RENKO_TREND_L] > 0, 1, 0))
        renko_trend_l_sx = np.sign(ret_indices[FLD.RENKO_TREND_L]) * Timeline_Integral(np.where(ret_indices[FLD.RENKO_TREND_L] < 0, 1, 0))
        ret_indices[FLD.RENKO_TREND_L_TIMING_LAG] = renko_trend_l_jx + renko_trend_l_sx

    renko_trend_s_jx = Timeline_Integral(np.where(ret_indices[FLD.RENKO_TREND_S] > 0, 1, 0))
    renko_trend_s_sx = np.sign(ret_indices[FLD.RENKO_TREND_S]) * Timeline_Integral(np.where(ret_indices[FLD.RENKO_TREND_S] < 0, 1, 0))
    ret_indices[FLD.RENKO_TREND_S_TIMING_LAG] = renko_trend_s_jx + renko_trend_s_sx

    renko_boost_s_jx = Timeline_duration(np.where(ret_indices[FLD.RENKO_TREND_S_UB] > ret_indices[FLD.RENKO_TREND_S_UB].shift(), 1, 0)) + 1
    renko_boost_s_sx = Timeline_duration(np.where(ret_indices[FLD.RENKO_TREND_S_LB] < ret_indices[FLD.RENKO_TREND_S_LB].shift(), 1, 0)) + 1
    ret_indices[FLD.RENKO_BOOST_S_TIMING_LAG] = np.where(ret_indices[FLD.RENKO_TREND_S_TIMING_LAG] > 0,
                                                         renko_boost_s_jx,
                                                         np.where(ret_indices[FLD.RENKO_TREND_S_TIMING_LAG] < 0,
                                                                  -renko_boost_s_sx, 0))

    renko_boost_l_jx = Timeline_duration(np.where(ret_indices[FLD.RENKO_TREND_L_UB] > ret_indices[FLD.RENKO_TREND_L_UB].shift(), 1, 0)) + 1
    renko_boost_l_sx = Timeline_duration(np.where(ret_indices[FLD.RENKO_TREND_L_LB] < ret_indices[FLD.RENKO_TREND_L_LB].shift(), 1, 0)) + 1
    ret_indices[FLD.RENKO_BOOST_L_TIMING_LAG] = np.where(ret_indices[FLD.RENKO_TREND_L_TIMING_LAG] > 0,
                                                         renko_boost_l_jx,
                                                         np.where(ret_indices[FLD.RENKO_TREND_L_TIMING_LAG] < 0,
                                                                  -renko_boost_l_sx, 0))

    ret_indices[FLD.RENKO_TREND_S] = ret_indices[FLD.RENKO_TREND_S].astype(np.float16)
    ret_indices[FLD.RENKO_TREND_S_LB] = ret_indices[FLD.RENKO_TREND_S_LB].astype(np.float16)
    ret_indices[FLD.RENKO_TREND_S_UB] = ret_indices[FLD.RENKO_TREND_S_UB].astype(np.float16)

    return ret_indices
