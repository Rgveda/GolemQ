# coding:utf-8
"""神奇九转（Magic Nine Turns，TD-Sequential 风格）—— **纯算法核**。

它是什么
========
逐 bar 把当日收盘价与**前第 4 日**的收盘价比（老树注释的说法）：

* 连续 9 天**高于**前第 4 日 ⇒ 判为**顶部**，输出 9 个 `1`；
* 连续 9 天**低于**前第 4 日 ⇒ 判为**底部**，输出 9 个 `-1`；
* 中途被打断 ⇒ 给被打断的那一串补 `NaN`，当前 bar 记 `-1`；
* 收尾时**没走满 9 天**的串 ⇒ 补 `2`（向上未完）/ `-2`（向下未完）。

返回值与输入**等长**（老树末尾有三条等长/截断/补零的分支）。

来源与范围
==========
逐字搬自 `GolemQ_old/features/base.py:268-334` 的 `calc_magic_nine_turns_np_vXs`
（float32 版）。

⚠️ **老树有两份**：`_np_vXs`（float32，**主路径在用**）与 `_np_vX`（float64，只在
benchmark/cli/sensor_old 等旁路用，且 `features/base.py` 与 `fractal/vXI.py` 各有一份
重复副本）。按 `DECISIONS.md` D39/D40「多副本只搬**规范版**」⇒ **只搬 `_np_vXs`**。
两者算法体逐字相同，只差 dtype：`_np_vX` 返回 float64、不 `.astype(np.float32)`。

⚠️ **本模块只搬「核」，不搬编排**（用户 2026-10-16 定）：
老树的 `calc_magic_nine_turns_dummy` / `_baseline` / `_baselines` **没有一起搬** ——
它们的输入列（`MACD_ZERO_TIMING_LAG` / `DEA_NORM_MAJOR` / `DIF_NORM_MAJOR` /
`MTM_NORM_MAJOR` / `MAPOWER30_MAJOR` / `MAGIC_NINE_TURNS_MAJOR` / `_WEEKLY`…）
**在新树里一个都没有生产者**（常量在 ≠ 有人算它），另有
`calc_zen_chopstickss`（走 CZSC）与 `calc_renko_atr_vX` **两个函数整个不存在**。
搬了它们只会算出**静默的错值**（`NaN` 参与比较）。挂账见 `HANDOFF.md`。

关于 numba
=========
老树给这个函数挂了 `@nb.jit('f4[:](f4[:])', nopython=True)`，它是**逐 bar 循环**，
正是 JIT 的靶子。本模块**不带 numba**（与 `analysis/` 的既有约定一致 ——
`timing.py` 里写着「新树 `analysis/` 全程不用 numba」）；
**加速版另见 `analysis/magic_nine_jit.py`**（`DECISIONS.md` D36 的形态：
另写一份对照实现，本文件一个字不动；两版必须逐值一致）。

⚠️ 输入**一律先转 float32**：老树的 jit 签名 `f4[:](f4[:])` 就强制了这一点，
不转的话纯 Python 版与 jit 版可能在**最低位**上分叉（比较运算符会跨过边界）。
"""
from __future__ import annotations

import numpy as np

__all__ = ['calc_magic_nine_turns_np_vXs']


def calc_magic_nine_turns_np_vXs(truthPrice) -> np.ndarray:
    """收盘价序列 → 神奇九转标记（`float32`，与输入等长）。

    :param truthPrice: 一维价格序列（array-like）。**会先转 float32** —— 见模块 docstring。
    :returns: `float32` 数组，取值 `{1, -1, 2, -2, NaN}`：
        `1` / `-1` = 走满 9 天的顶 / 底那 9 根；
        `2` / `-2` = 收尾时**没走满**的多 / 空串；
        `NaN` = 被反向串打断的那些 bar；`-1` = 尚未进入比较期的前几根。

    ⚠️ **不要把 `-1` 读成"底"** —— 它同时用于「走满 9 天的顶部那 9 根」（值 `1`）
    之外的**默认填充**。老树的约定如此，下游按 `>0.000168` / `<-0.000168` 判正负。
    """
    truthPrice = np.asarray(truthPrice, dtype=np.float32)

    ret_magic_nine_turns = []
    _list_4 = []            # 最近 4 根收盘价；满 4 就弹出最老那根做比较
    _list_t_up = []         # 当前「高于前第 4 日」的连续串
    _list_t_down = []       # 当前「低于前第 4 日」的连续串

    for i in truthPrice:
        _list_4.append(i)

        if len(_list_t_up) == 9:             # 向上的串走满 9 天 → 落 9 个 1
            for _ in _list_t_up:
                ret_magic_nine_turns.append(1)
            _list_t_up.clear()

        if len(_list_t_down) == 9:           # 向下的串走满 9 天 → 落 9 个 -1
            for _ in _list_t_down:
                ret_magic_nine_turns.append(-1)
            _list_t_down.clear()

        if len(_list_4) == 4:
            _list_t = _list_4.pop(0)
            if i > _list_t:
                _list_t_up.append(i)
                for _ in range(0, len(_list_t_down)):
                    ret_magic_nine_turns.append(np.nan)   # 被反向打断的串 → NaN
                _list_t_down.clear()

            if i < _list_t:
                _list_t_down.append(i)
                for _ in range(0, len(_list_t_up)):
                    ret_magic_nine_turns.append(np.nan)
                _list_t_up.clear()

            if i == _list_t:
                ret_magic_nine_turns.append(-1)
                for _ in range(0, len(_list_t_down)):
                    ret_magic_nine_turns.append(np.nan)
                for _ in range(0, len(_list_t_up)):
                    ret_magic_nine_turns.append(np.nan)
                _list_t_down.clear()
                _list_t_up.clear()
        else:
            ret_magic_nine_turns.append(-1)

    # 收尾：没走满 9 天的串 → 2（多）/ -2（空）
    if len(_list_t_up) != 0:
        for _ in _list_t_up:
            ret_magic_nine_turns.append(2)
        _list_t_up.clear()
    if len(_list_t_down) != 0:
        for _ in _list_t_down:
            ret_magic_nine_turns.append(-2)
        _list_t_down.clear()

    out = np.array(ret_magic_nine_turns).astype(np.float32)
    if len(truthPrice) == len(out):
        return out
    elif len(truthPrice) < len(out):
        return out[-len(truthPrice):]
    # 输出比输入短（老树认为不该发生）→ 左补 0
    pad = np.zeros(len(truthPrice))
    pad[-len(out):] = out
    return pad.astype(np.float32)
