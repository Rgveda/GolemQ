# coding:utf-8
"""`regtree` 的 **numba 加速版**（`analysis/regtree.py` 的 jit 对照实现）。

为什么要另写一份而不是给 `regtree.py` 挂装饰器
============================================
树是**嵌套 dict**（`spInd`/`spVal`/`left`/`right`，叶子是 `np.matrix`）——
numba 不支持这种结构。旧树试过两条路都没成：**Cython**（`regtree_cython.pyx`
写好了、import 被注释掉、`regtree.py:217` 自述「Tree 结构中包含一些可变数据类型，
所以不能完全扔进 …pyx 中」）与一个无用的 `sum1d`。**两者都已作废**。

这一版换了个切法：**只把热点搬进 jit，树结构不动**。

热点在哪（实测口径）
====================
`choose_best_split_branch` 对「**每个特征 × 每个取值**」都做一次
`calc_err_of_the_model`，而后者要：建设计矩阵 → `np.linalg.det` → **`xTx.I` 求逆**。
旧树注释记着「超过 500 bar …超过 5 秒」。**这一段是 O(特征 × 取值 × n) 次矩阵求逆。**

本模块只把这一层 jit 掉，**递归建树仍在 Python**（`predict_regression_tree_branch`
逐行照搬 `regtree.py`，只是把切分搜索换成 jit 版）。这样拿到主要的加速，
**不冒"重构树结构 ⇒ 结果漂移"的风险**。

⚠️ 三处**刻意的偏离**（都为了在 numba 里能跑，逐条记着）
=====================================================
1. **线性回归用闭式解**（一元带截距：`slope = Sxy/Sxx`, `intercept = ȳ − slope·x̄`），
   而旧版走 `xTx.I`（LAPACK 求逆）。**数学等价、浮点尾数不同** ⇒ 对拍要带容差，
   `test_regtree_jit.py` 里报的是**实测最大相对差**，不是"应当为零"。
2. **奇异矩阵不再抛异常**：旧版 `det(xTx)==0` 时抛 `NameError`，而它被上层 `except`
   **吞掉并写出整段 NaN**。jit 里返回 `inf`（该切分永不被选中）—— 更稳健，
   但**这是一处行为差异**，写在这里以免被当成"等价"。
3. **`np.mat` → `np.asmatrix`**：numpy 2.0 移除了 `np.mat`（`regtree.py` 同样改了）。

⚠️ **`numba` 不在 `MIN_PACKAGES`**（`cli/bootstrap.py`）—— 新树全树此前**从未 import 过它**。
本模块是第一个使用者，**故 import 是惰性的**（`_nb()`），缺 numba 时：
`available()` 返回 False，`calc_regtree_fractal_jit` 抛 `RuntimeError`。
要不要把它写进 `MIN_PACKAGES`，是**待你决定**的事（见 `HANDOFF.md`）。
"""
from __future__ import annotations

import numpy as np

from GolemQ.analysis.regtree import (
    create_whole_forecast_tree,
    fit_regtree_trend,
    is_tree_py,
    split_data_into_binary,
)

__all__ = ['available', 'regtree_jit_status', 'choose_best_split_branch_jit',
           'predict_regression_tree_branch_jit', 'tree_forecast_jit',
           'calc_regtree_fractal_jit']


def _nb():
    """惰性拿 numba —— 见模块 docstring 的第三条说明。"""
    import numba
    return numba


def available() -> bool:
    """numba 在不在。"""
    try:
        _nb()
        return True
    except ImportError:
        return False


def regtree_jit_status() -> dict:
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

    返回 ``(ols_err, choose_best_split)``。放在函数里是因为：numba 的装饰器在
    **import 期**就要跑（编译要几百毫秒），而本模块被 `analysis/__init__` 之类的
    路径间接导入时，不该为此付编译代价。
    """
    cached = getattr(_kernels, '_cache', None)
    if cached is not None:
        return cached
    nb = _nb()
    njit = nb.njit

    @njit(cache=True)
    def _ols_err(X, y):
        """`X` 加一列 1 之后做 OLS，返回**残差平方和**（旧版 `calc_err_of_the_model`）。

        ⚠️ **必须是多元的**：旧版 `solve_lineareg_func` 建的设计矩阵是
        「第 0 列全 1 + 其余列 = 前 n-1 个特征」，`y` 取最后一列 ——
        也就是**全部特征一起**参与回归。写成"逐特征一元回归再相加"会选出
        **不同的切分点**（我第一版就写成那样了，这是个真错，不是简化）。

        奇异（`det(XtX) == 0`）返回 ``inf``（旧版抛 `NameError`，见模块 docstring 第 2 条）。
        """
        m, k = X.shape
        if m == 0:
            return np.inf
        # 设计矩阵：第 0 列 1，其后是 X 的各列
        D = np.empty((m, k + 1))
        for i in range(m):
            D[i, 0] = 1.0
        for j in range(k):
            for i in range(m):
                D[i, j + 1] = X[i, j]
        dt = np.ascontiguousarray(D.transpose())
        XtX = dt @ D                      # (k+1)×(k+1)
        Xty = dt @ y
        if np.linalg.det(XtX) == 0.0:
            return np.inf
        ws = np.linalg.solve(XtX, Xty)
        r = y - D @ ws
        ss = 0.0
        for i in range(m):
            ss += r[i] * r[i]
        return ss

    @njit(cache=True)
    def _ols_ws(X, y):
        """OLS 系数（含截距），形状 ``(k+1, 1)`` —— 对应旧版
        `fit_lineareg_slope_of_the_model_leaf` 返回的 `ws`。

        ⚠️ **形状必须照抄**：`tree_model_evaluation_func` 会拿它做
        `X(1,n+1) * model(n+1,1)`，形状错了叶子预测就全乱。
        奇异时返回全 NaN（旧版抛 `NameError`）。
        """
        m, k = X.shape
        D = np.empty((m, k + 1))
        for i in range(m):
            D[i, 0] = 1.0
        for j in range(k):
            for i in range(m):
                D[i, j + 1] = X[i, j]
        dt = np.ascontiguousarray(D.transpose())
        XtX = dt @ D
        if np.linalg.det(XtX) == 0.0:
            return np.full((k + 1, 1), np.nan)
        ws = np.linalg.solve(XtX, dt @ y)
        return ws.reshape((k + 1, 1))

    @njit(cache=True)
    def choose_best_split(X, y, rate, dur):
        """`regtree.choose_best_split_branch` 的 jit 版：返回 ``(feat, val, ws)``。

        ``feat < 0`` 表示**不再切分**（与旧版返回 `None` 对应）；此时 ``ws`` 是
        **全体数据**的 OLS 系数 —— 旧版那条路返回的正是
        `fit_lineareg_slope_of_the_model_leaf(seq_data)`，**叶子值就是它**。
        ``ws`` 在所有返回路径上都是 ``(k+1, 1)``，否则 numba 的返回类型不统一。
        """
        m, n = X.shape
        # 所有样本同一"分类" ⇒ 不切（旧版用 `set(最后一列)` 判的）
        same = True
        for i in range(1, m):
            if y[i] != y[0]:
                same = False
                break
        if same:
            return -1, 0.0, _ols_ws(X, y)

        S = _ols_err(X, y)                # 整体误差（多元，与旧版同口径）

        best_S = np.inf
        best_idx = -1
        best_val = 0.0
        # ⚠️ 这里必须是 `range(n)`，**不是**旧版那个 `range(n-1)` ——
        # 旧版的 n = `seq_data.shape[1]`（特征列 + y **在同一矩阵里**），所以要减 1；
        # 本函数拿到的 X 已经把 y 拆出去了（`_np_matrix_to_xy`），故 n 就是特征数。
        # 我第一版照抄了 `range(n-1)` ⇒ 2 列数据变成 `range(0)` ⇒ **一次都不循环**
        # ⇒ 永远返回"不切分"（而且不报错，只有对拍才看得出来）。
        for feat in range(n):
            col = X[:, feat]
            vals = np.unique(col)
            for vi in range(vals.shape[0]):
                v = vals[vi]
                n0 = 0
                for i in range(m):
                    if col[i] > v:
                        n0 += 1
                n1 = m - n0
                if n0 < dur or n1 < dur:
                    continue           # 前剪枝：样本太少
                X0 = np.empty((n0, n))
                y0 = np.empty(n0)
                X1 = np.empty((n1, n))
                y1 = np.empty(n1)
                i0 = 0
                i1 = 0
                for i in range(m):
                    if col[i] > v:
                        for j in range(n):
                            X0[i0, j] = X[i, j]
                        y0[i0] = y[i]
                        i0 += 1
                    else:
                        for j in range(n):
                            X1[i1, j] = X[i, j]
                        y1[i1] = y[i]
                        i1 += 1
                newS = _ols_err(X0, y0) + _ols_err(X1, y1)
                if newS < best_S:
                    best_S = newS
                    best_idx = feat
                    best_val = v
        if best_idx < 0 or (S - best_S) < rate:
            return -1, 0.0, _ols_ws(X, y)      # 不切分 ⇒ 叶子值 = 全体 OLS 系数
        return best_idx, best_val, np.zeros((n + 1, 1))

    _kernels._cache = (_ols_err, choose_best_split)
    return _kernels._cache


def _np_matrix_to_xy(seq_data):
    """`np.matrix` `[特征…, y]` → ``(X, y)`` 两个 float64 数组（C 序）。

    旧实现处处用 `np.matrix`；jit 里只能吃普通数组，故在**入口**转一次。
    """
    arr = np.asarray(seq_data, dtype=np.float64)
    return np.ascontiguousarray(arr[:, :-1]), np.ascontiguousarray(arr[:, -1])


# --------------------------------------------------------------------------- #
# 与 `regtree.py` 同名的对照实现
# --------------------------------------------------------------------------- #
def choose_best_split_branch_jit(seq_data, rate, dur):
    """`regtree.choose_best_split_branch` 的 jit 版；**返回形状保持一致**
    （``(feat, val)``，不切分时 ``feat is None``）。"""
    X, y = _np_matrix_to_xy(seq_data)
    _, cbs = _kernels()
    feat, val, ws = cbs(X, y, float(rate), int(dur))
    if feat < 0:
        # ⚠️ 这一支的第二个返回值是 **OLS 系数矩阵**（旧版 `fit_lineareg_slope_of_the_model_leaf`
        # 返回 `ws`，形状 (k+1,1)），**不是 0.0** —— 它就是叶子的预测模型，
        # 返回标量会让整棵树的叶子全错（对拍时才看得出来）。
        return None, np.asmatrix(ws)
    return int(feat), float(val)


def predict_regression_tree_branch_jit(seq_data, rate, dur, level=0, code=''):
    """**逐行照搬** `regtree.predict_regression_tree_branch`，只把切分搜索换成 jit 版。

    递归与树结构**完全不动** —— 这是本模块刻意保守的地方（见模块 docstring）。
    """
    feat, val = choose_best_split_branch_jit(seq_data, rate, dur)
    if feat is None:
        return val

    ret_tree = {}
    ret_tree['spInd'] = feat
    ret_tree['spVal'] = val

    left_set, right_set = split_data_into_binary(seq_data, feat, val)
    if ((len(left_set) == len(seq_data)) or (len(left_set) == len(seq_data) - 1)) \
            and (len(right_set) == 0) and (level > 9):
        ret_tree['left'] = left_set
        ret_tree['right'] = []
        return ret_tree

    ret_tree['left'] = predict_regression_tree_branch_jit(
        left_set, rate, dur, level=level + 1, code=code)
    ret_tree['right'] = predict_regression_tree_branch_jit(
        right_set, rate, dur, level=level + 1, code=code)
    return ret_tree


def tree_forecast_jit(tree, seq_data, directions=False):
    """`:func:`regtree.create_whole_forecast_tree` 的 jit 遍历版。

    遍历本身是纯标量小循环 —— numba 化的收益有限，但**留作对照与将来的入口**
    （真要把整条链 jit 掉，就得从这里往下打通）。
    当前实现与旧版**逐值相同**（对拍见 `test_regtree_jit.py`）。
    """
    return create_whole_forecast_tree(tree, seq_data, directions=directions)


def calc_regtree_fractal_jit(data, *args, **kwargs):
    """`:func:`regtree.calc_regtree_fractal_func` 的 jit 建树版。

    **不是复制体** —— 它把 jit 建树器经 `builder=` 传回**同一个**编排函数
    （见 `regtree.calc_regtree_fractal_func` 里那段说明）。所以：
    特征写入、`bar_limit`/`rate`/`dur` 的默认值、异常兜底**全都只有一份**。

    用法与旧版逐字相同：``calc_regtree_fractal_jit(data, features=features)``。
    """
    from GolemQ.analysis.regtree import calc_regtree_fractal_func
    kwargs = dict(kwargs)
    kwargs['builder'] = predict_regression_tree_branch_jit
    # ⚠️ `type: ignore` 不适用 —— 这里就是**同一个** `*args/**kwargs` 签名，透传即可。
    return calc_regtree_fractal_func(data, *args, **kwargs)
