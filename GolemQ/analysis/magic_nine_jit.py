# coding:utf-8
"""神奇九转 —— **numba.jit 版（唯一实现）**。

原 `analysis/magic_nine.py` 的纯 Python 版已删除：本模块的公开接口
`calc_magic_nine_turns_np_vXs_jit` 直接就是 ``numba.jit`` 函数，签名显式声明为
``float32[:](float32[:])``。

算法与老树 `features/base.py:268-334` 逐字一致（float32 版）；返回与输入等长。
`DECISIONS.md` D36 的「另写一份对照」形态到此结束 —— 只保留 jit 这一份。
"""
from __future__ import annotations

import numba
import numpy as np

__all__ = ['available', 'magic_nine_jit_status',
           'calc_magic_nine_turns_np_vXs_jit']


def available() -> bool:
    """numba 在不在。"""
    try:
        import numba as _n  # noqa: F401
        return True
    except ImportError:
        return False


def magic_nine_jit_status() -> dict:
    """排障用：numba 版本 + jit 是否真的开了（`DISABLE_JIT` 环境变量会关掉它）。"""
    try:
        import numba as _n
    except ImportError as exc:
        return {'available': False, 'reason': '{}: {}'.format(type(exc).__name__, exc)}
    cfg = getattr(_n, 'config', None)
    return {'available': True, 'numba': _n.__version__,
            'disable_jit': int(getattr(cfg, 'DISABLE_JIT', 0)) if cfg else None}


@numba.jit('float32[:](float32[:])', cache=True)
def calc_magic_nine_turns_np_vXs_jit(truthPrice):
    """收盘价序列 → 神奇九转标记（`float32`，与输入等长）。

    显式签名要求输入是 ``float32[:]``；调用方先 ``astype(np.float32)``。
    """
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
