# coding:utf-8
"""`magic_nine` 的 **numba 加速版**（`analysis/magic_nine.py` 的 jit 对照实现）。

形态照 `DECISIONS.md` **D36**（`renko_jit` / `regtree_jit` 同一套）：
**另写一份，`magic_nine.py` 一个字不动**；两版必须**逐值一致**，各有用例钉着。

为什么这个值得 JIT
=================
`calc_magic_nine_turns_np_vXs` 是**逐 bar 循环**（`for i in truthPrice`），
内层还有两个有界小循环（补 `NaN` / 落 9 个标记）。老树本来也给它挂了
`@nb.jit('f4[:](f4[:])', nopython=True)` —— 这里只是把那个装饰器接回来。

⚠️ 与本族另外几个不同：`POLYNOMIAL9` 的 `rolling_poly9_v2` **不该** JIT ——
它的热循环里调的是 `np.polynomial.Chebyshev.fit`（numpy 的 C 实现），
numba 编译不了，硬搓最小二乘会**动最低位、破坏 `array_equal` 对拍**。见 D49。

关于 numba
=========
`numba` 在 `MIN_PACKAGES`（`cli/bootstrap.py`），且老树这个函数本来就 jit 过。
但本模块的 import 仍是**惰性**的（同 `renko_jit` / `regtree_jit`）：
装饰器在 **import 期**就要跑，而本模块被间接导入时不该为此付代价。
`DISABLE_JIT` 环境变量会关掉它（numba 自带的开关，`numba.config.DISABLE_JIT`）。

⚠️ **核是逐字复刻**：连 `_list_4` / `_list_t_up` / `_list_t_down` 的那套
「满 9 落标记、反向补 NaN、收尾补 ±2」顺序都照抄 —— 顺序错一步就会与纯 Python 版分叉。
已实测 numba 的 `nopython` **支持**这种 `list.append` + `np.array(list)` 写法
（2026-10-16 验；新版 numba 对反射列表的支持收窄过，所以这一条要跟着测）。
"""
from __future__ import annotations

import numpy as np

__all__ = ['available', 'magic_nine_jit_status', 'calc_magic_nine_turns_np_vXs_jit']


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


def magic_nine_jit_status() -> dict:
    """排障用：numba 版本 + jit 是否真的开了（`DISABLE_JIT` 环境变量会关掉它）。"""
    try:
        nb = _nb()
    except ImportError as exc:
        return {'available': False, 'reason': '{}: {}'.format(type(exc).__name__, exc)}
    cfg = getattr(nb, 'config', None)
    return {'available': True, 'numba': nb.__version__,
            'disable_jit': int(getattr(cfg, 'DISABLE_JIT', 0)) if cfg else None}


def _kernel():
    """把 njit 的核**惰性**建出来（并缓存）。见模块 docstring 的「关于 numba」。"""
    cached = getattr(_kernel, '_cache', None)
    if cached is not None:
        return cached
    njit = _nb().njit

    # ⚠️ **不写 `nopython=True`** —— `njit` 已经隐含它，写了 numba 会打一行
    # `RuntimeWarning: nopython is set for njit and is ignored`（实测）。
    @njit('f4[:](f4[:])', cache=True)
    def _vXs(truthPrice):
        """`magic_nine.calc_magic_nine_turns_np_vXs` 的**逐字**复刻（只加 jit）。"""
        ret_magic_nine_turns = []
        _list_4 = []
        _list_t_up = []
        _list_t_down = []
        for i in truthPrice:
            _list_4.append(i)

            if len(_list_t_up) == 9:
                for j in _list_t_up:
                    ret_magic_nine_turns.append(1)
                _list_t_up.clear()

            if len(_list_t_down) == 9:
                for j in _list_t_down:
                    ret_magic_nine_turns.append(-1)
                _list_t_down.clear()

            if len(_list_4) == 4:
                _list_t = _list_4.pop(0)
                if i > _list_t:
                    _list_t_up.append(i)
                    for j in range(0, len(_list_t_down)):
                        ret_magic_nine_turns.append(np.nan)
                    _list_t_down.clear()

                if i < _list_t:
                    _list_t_down.append(i)
                    for j in range(0, len(_list_t_up)):
                        ret_magic_nine_turns.append(np.nan)
                    _list_t_up.clear()

                if i == _list_t:
                    ret_magic_nine_turns.append(-1)
                    for j in range(0, len(_list_t_down)):
                        ret_magic_nine_turns.append(np.nan)
                    for j in range(0, len(_list_t_up)):
                        ret_magic_nine_turns.append(np.nan)
                    _list_t_down.clear()
                    _list_t_up.clear()
            else:
                ret_magic_nine_turns.append(-1)

        if len(_list_t_up) != 0:
            for j in _list_t_up:
                ret_magic_nine_turns.append(2)
            _list_t_up.clear()
        if len(_list_t_down) != 0:
            for j in _list_t_down:
                ret_magic_nine_turns.append(-2)
            _list_t_down.clear()

        if len(truthPrice) == len(np.array(ret_magic_nine_turns)):
            return np.array(ret_magic_nine_turns).astype(np.float32)
        elif len(truthPrice) < len(np.array(ret_magic_nine_turns)):
            return np.array(ret_magic_nine_turns).astype(np.float32)[-len(truthPrice):]
        ret_magic_nine_turns2 = np.zeros(len(truthPrice))
        ret_magic_nine_turns2[-len(np.array(ret_magic_nine_turns)):] = ret_magic_nine_turns
        return ret_magic_nine_turns2.astype(np.float32)

    _kernel._cache = _vXs
    return _vXs


def calc_magic_nine_turns_np_vXs_jit(truthPrice) -> np.ndarray:
    """`magic_nine.calc_magic_nine_turns_np_vXs` 的 jit 版 —— 返回值**逐值相同**。

    ⚠️ 输入**先转 float32**（jit 签名是 `f4[:](f4[:])`，与纯 Python 版同一处理）。

    numba 不可用时**回落到纯 Python 版** —— 宁可慢，不要抛。
    """
    arr = np.asarray(truthPrice, dtype=np.float32)
    if not available():
        from .magic_nine import calc_magic_nine_turns_np_vXs
        return calc_magic_nine_turns_np_vXs(arr)
    return _kernel()(arr)
