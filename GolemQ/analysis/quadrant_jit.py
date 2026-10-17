# coding:utf-8
"""`quadrant` 里 `mapower` 那一族的 **numba 加速版**（D36 形态的 jit 对照实现）。

范围
====
只覆盖 `inv_nums` / `mapower_np_vXIIs`（`analysis/quadrant.py`）。
**不是** `quadrant.py` 全文的 jit 版 —— 那里面其余函数（四量图等）是向量化的
numpy/pandas，没有可 JIT 的逐 bar 循环。

为什么这两个值得 JIT
==================
`mapower_np_vXIIs` 有一个**逐 bar** 循环（`for i in range(ma_max, len(price))`），
里面每次调 `inv_nums` —— 那是**双重 Python 循环**。老树本来就给这两个挂了
`@nb.jit`，这里只是把装饰器接回来。

⚠️ **量级别高估**：`inv_nums` 收到的是「25 条均线」那条切片，所以是 25² 量级，
**不是**价格序列长度的平方。实测纯 Python 版跑 5000 根也在秒级 —— JIT 是**锦上添花**，
不是"不 JIT 就跑不动"。这一点与 `renko_jit`（那里实测 **654×**）不同，别照抄结论。

关于 numba
=========
惰性 import（同 `renko_jit` / `regtree_jit` / `magic_nine_jit`）—— 装饰器在
**import 期**就要跑，本模块被间接导入时不该为此付代价。
`DISABLE_JIT` 环境变量由 numba 自己处理（`numba.config.DISABLE_JIT`）。

⚠️ 核是**逐字复刻**：连 `np.convolve(w, price)[r-1:-r+1]` 那个滑窗取法都照抄 ——
换一种写法就会动最低位，破坏 `array_equal` 对拍。
"""
from __future__ import annotations

import numpy as np

__all__ = ['available', 'quadrant_jit_status',
           'inv_nums_jit', 'mapower_np_vXIIs_jit']


def _nb():
    """惰性拿 numba。"""
    import numba
    return numba


def available() -> bool:
    """numba 在不在。"""
    try:
        _nb()
        return True
    except ImportError:
        return False


def quadrant_jit_status() -> dict:
    """排障用：numba 版本 + jit 是否真的开了（`DISABLE_JIT` 环境变量会关掉它）。"""
    try:
        nb = _nb()
    except ImportError as exc:
        return {'available': False, 'reason': '{}: {}'.format(type(exc).__name__, exc)}
    cfg = getattr(nb, 'config', None)
    return {'available': True, 'numba': nb.__version__,
            'disable_jit': int(getattr(cfg, 'DISABLE_JIT', 0)) if cfg else None}


def _kernels():
    """把两个 njit 核**惰性**建出来（并缓存）。见模块 docstring 的「关于 numba」。"""
    cached = getattr(_kernels, '_cache', None)
    if cached is not None:
        return cached
    njit = _nb().njit

    @njit('int16(f4[:])', cache=True)
    def _inv_nums(series):
        """`quadrant.inv_nums` 的逐字复刻（老树的签名原样：返回 `int16`）。

        ⚠️ 计数上界是 `C(25,2)=300`，`int16` 装得下 —— 但**别把 range_list 撑到 183 条以上**，
        那会溢出成负数（老树也有这个限制，不是本次引入的）。
        """
        cnt = 0
        for i in range(len(series)):
            val = series[i]
            for j in range(i):
                if val < series[j]:
                    cnt += 1
        return cnt

    @njit('f4[:](f4[:], int16[:])', cache=True)
    def _mapower(price, range_list):
        """`quadrant.mapower_np_vXIIs` 的逐字复刻（只加 jit）。"""
        ma_np = np.empty((len(price), len(range_list)), dtype=np.float32)
        ma_count = 0
        for r in range_list:
            n = np.ones(r)
            weights = n / r
            ma_np[r - 1:, ma_count] = np.convolve(weights, price)[r - 1:-r + 1]
            ma_count = ma_count + 1

        ma_max = max(range_list)
        len_range_list = len(range_list)
        num = np.zeros(len(price), dtype=np.float32)
        ratio = np.zeros(len(price), dtype=np.float32)
        for i in range(ma_max, len(price)):
            num[i] = _inv_nums(ma_np[i, :])
            ratio[i] = num[i] / (len_range_list * (len_range_list - 1)) * 2

        return ratio

    _kernels._cache = (_inv_nums, _mapower)
    return _kernels._cache


def inv_nums_jit(series):
    """`quadrant.inv_nums` 的 jit 版。numba 不在场时**回落到纯 Python 版**（宁可慢，不要抛）。"""
    if not available():
        from .quadrant import inv_nums
        return inv_nums(series)
    return _kernels()[0](np.asarray(series, dtype=np.float32))


def mapower_np_vXIIs_jit(price, range_list=None):
    """`quadrant.mapower_np_vXIIs` 的 jit 版 —— 返回值**逐值相同**。

    ⚠️ 输入**先转 float32**、`range_list` **先转 int16**（jit 签名强制了这点，
    与纯 Python 版同一处理）—— 不统一的 dtype 会在最低位上分叉。
    """
    arr = np.asarray(price, dtype=np.float32)
    if range_list is None:
        rl = np.array(range(5, 30)).astype(np.int16)
    else:
        rl = np.asarray(range_list, dtype=np.int16)
    if not available():
        from .quadrant import mapower_np_vXIIs
        return mapower_np_vXIIs(arr, rl)
    return _kernels()[1](arr, rl)
