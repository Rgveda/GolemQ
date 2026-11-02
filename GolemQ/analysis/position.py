# coding: utf-8
"""「策略仓位控制」的 Dempster–Shafer 合成（纯计算，不连库）。

逐字搬自旧树 `GolemQ_old/scribe/empirical.py:507-606`
（`calc_massive_ChartsScanAI_model` 的 D-S 段）。

语义
====
对**同一根 60min bar 的全市场横截面**，每只股票投一票：

* **Bull**：`QuadPreUprBf < QuadPreEpBf`（上涨前奏比见顶前奏更近）
* **Bear**：`QuadPreEpBf < QuadPreUprBf` 且 `zDLagD < 0`（且缠论笔向下）
* **Ω（摇摆）**：其余

`m_bull / m_bear / m_omega` 为成员占比。仓位 = `m_bull`
（代数上等于 `ds_confidence × certainty`），**`m_bear > 0.618` 时强制清仓**。

⚠️ 与旧树的差异（刻意）：旧 `ds_state == 2` 分支不可达
（`Bel_bull + Bel_bear ≤ 1`，不可能同时 > 0.5），这里**删掉**并注释；
`DS_CONFLICT_K` 旧树恒为 0.0，这里不再输出。
"""
from __future__ import annotations

import numpy as np

__all__ = ['calc_position_ds', 'calc_position_series', 'calc_position_flags',
           'BEAR_CLEAR_RATIO']

#: 黄金分割清仓闸：做空占比超过它且占优 ⇒ 仓位清零。
BEAR_CLEAR_RATIO = 0.618


def calc_position_series(n, bull, bear, eps: float = 1e-12) -> dict:
    """逐 bar 的 D-S 仓位（向量化版 `calc_position_ds`）。

    :param n:    每 bar 参与横截面的成员数
    :param bull: 每 bar 投 Bull 的成员数
    :param bear: 每 bar 投 Bear 的成员数
    :returns: ``{m_bull, m_bear, m_omega, onhold, bel_bull, bel_bear,
               pl_bull, pl_bear, uncertainty, ds_state}``，均为 ndarray

    >>> r = calc_position_series([3, 3], [1, 0], [1, 3])
    >>> (round(float(r['m_bull'][0]), 4), round(float(r['m_bear'][1]), 4),
    ...  [round(float(x), 4) for x in r['onhold']])
    (0.3333, 1.0, [0.3333, 0.0])
    """
    n = np.asarray(n, dtype=float)
    bull = np.asarray(bull, dtype=float)
    bear = np.asarray(bear, dtype=float)
    safe = np.where(n > 0, n, 1.0)
    m_bull = np.where(n > 0, bull / safe, np.nan)
    m_bear = np.where(n > 0, bear / safe, np.nan)
    m_omega = np.where(n > 0, 1.0 - m_bull - m_bear, np.nan)

    bel_bull = m_bull
    bel_bear = m_bear
    pl_bull = 1.0 - m_bear
    pl_bear = 1.0 - m_bull
    uncertainty = np.where(n > 0, pl_bull - bel_bull, np.nan)

    total = np.where(n > 0, bel_bull + bel_bear, 0.0)
    ds_conf = np.where(total > eps, bel_bull / np.where(total > eps, total, 1.0),
                       np.where(n > 0, 0.5, np.nan))
    onhold = ds_conf * (1.0 - uncertainty)
    force = (m_bear > BEAR_CLEAR_RATIO) & (m_bear > m_bull) & (m_bear > m_omega)
    onhold = np.where(force, 0.0, onhold)

    ds_state = np.zeros(len(n), dtype=np.int16)
    live = n > 0
    s1 = live & (bel_bull > 0.5) & (uncertainty < 0.1)
    s_1 = live & (bel_bear > 0.5) & (uncertainty < 0.1) & ~s1
    s0 = live & (bel_bull < 0.5) & (bel_bear < 0.5) & ~s1 & ~s_1
    rest = live & ~s1 & ~s_1 & ~s0
    ds_state[s1] = 1
    ds_state[s_1] = -1
    ds_state[s0] = 0
    ds_state[rest] = np.sign(bel_bull[rest] - bel_bear[rest]).astype(np.int16)

    return {'m_bull': m_bull, 'm_bear': m_bear, 'm_omega': m_omega,
            'onhold': onhold, 'bel_bull': bel_bull, 'bel_bear': bel_bear,
            'pl_bull': pl_bull, 'pl_bear': pl_bear,
            'uncertainty': uncertainty, 'ds_state': ds_state}


def calc_position_flags(m_bull, m_bear, m_omega, onhold,
                        ratio: float = 0.618, gap_bars: int = 4,
                        stage_mode=None, stage_streak=None) -> dict:
    """仓位序列的三个**事件标记**（纯派生，不重算）。

    * ``PosFull``：满仓信号 —— `m_bull > ratio` 且 `m_bull` 同时大于
      `m_bear` / `m_omega`（清仓规则的**反函数**：清仓是 `m_bear > 0.618` 且占优）；
    * ``PosFullOnset``：满仓段的**第一根**（回测取样点，避免整段重复计数）；
    * ``PosReClear``：**清仓暴击** —— 本清仓段与上一段之间只隔 `gap_bars` 根以内
      （「清仓→短暂重开→立刻再清仓」）；
    * ``PosBottom``：**空仓 + 见底** —— `onhold <= 0` 且 `STAGE_MODE <= -0.5`
      （市场横截面 STAGE 众数跌破最底带；4.4 实测该组合的出现点全部落在底部区）；
    * ``PosProbe``：底部信号时**撤销强制清仓**、回到原始 `m_bull` 作为试探仓，
      否则等于 `onhold`。

    >>> m_bull = np.array([0.1, 0.7, 0.7, 0.2, 0.0, 0.0, 0.0])
    >>> m_bear = np.array([0.8, 0.1, 0.1, 0.7, 0.7, 0.7, 0.7])
    >>> m_pad = np.array([0.1, 0.2, 0.2, 0.1, 0.3, 0.3, 0.3])
    >>> onhold = np.array([0.0, 0.7, 0.7, 0.0, 0.2, 0.0, 0.0])
    >>> f = calc_position_flags(m_bull, m_bear, m_pad, onhold)
    >>> f['PosFull'].tolist(), f['PosFullOnset'].tolist(), f['PosReClear'].tolist()
    ([False, True, True, False, False, False, False], [False, True, False, False, False, False, False], [False, False, False, True, False, True, False])
    """
    m_bull = np.asarray(m_bull, dtype=float)
    m_bear = np.asarray(m_bear, dtype=float)
    m_omega = np.asarray(m_omega, dtype=float)
    onhold = np.asarray(onhold, dtype=float)
    n = len(m_bull)
    full = (m_bull > ratio) & (m_bull > m_bear) & (m_bull > m_omega)
    onset = np.zeros(n, dtype=bool)
    if n:
        onset[0] = bool(full[0])
        onset[1:] = full[1:] & ~full[:-1]

    zero = onhold <= 0
    re_clear = np.zeros(n, dtype=bool)
    re_clear_prev = np.full(n, -1, dtype=np.int64)   # 前一个零仓段的**起点**下标
    prev_end = None
    prev_start = None
    i = 0
    while i < n:
        if not zero[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and zero[j + 1]:
            j += 1
        if prev_end is not None and (i - prev_end - 1) <= gap_bars:
            re_clear[i] = True
            re_clear_prev[i] = prev_start
        prev_end = j
        prev_start = i
        i = j + 1

    if stage_mode is None:
        bottom = np.zeros(n, dtype=bool)
        deep = np.zeros(n, dtype=bool)
    else:
        sm = np.asarray(stage_mode, dtype=float)
        bottom = zero & np.isfinite(sm) & (sm <= -0.5 + 1e-9)
        if stage_streak is None:
            deep = np.zeros(n, dtype=bool)
        else:
            sk = np.asarray(stage_streak, dtype=float)
            deep = (bottom & np.isfinite(sk) & (sk >= 3)
                    & np.isfinite(m_bear) & (m_bear >= 0.90))
    probe = np.where(bottom & np.isfinite(m_bull), m_bull, onhold)
    return {'PosFull': full, 'PosFullOnset': onset, 'PosReClear': re_clear,
            'PosBottom': bottom, 'PosProbe': probe, 'PosDeepBottom': deep,
            'ReClearPrevStart': re_clear_prev}


def calc_position_ds(upr, ep, zd, eps: float = 1e-12) -> dict:
    """一个 bar 的横截面 → D-S 仓位字典。

    :param upr: `QuadPreUprBf`（上涨前奏持续根数）
    :param ep:  `QuadPreEpBf`（见顶前奏持续根数）
    :param zd:  `zDLagD`（缠论笔方向 event lag，负=向下）
    :returns: ``{members, m_bull, m_bear, m_omega, onhold, bel_bull,
               bel_bear, pl_bull, pl_bear, uncertainty, ds_state}``

    >>> r = calc_position_ds([10, 1, 5], [1, 10, 5], [-1, 1, 1])
    >>> round(r['m_bull'], 4), round(r['m_bear'], 4), round(r['onhold'], 4)
    (0.3333, 0.3333, 0.3333)

    做空占比超过 61.8% ⇒ 强制清仓：

    >>> calc_position_ds([1, 1, 1], [0, 0, 0], [-1, -1, -1])['onhold']
    0.0
    """
    upr = np.asarray(upr, dtype=float)
    ep = np.asarray(ep, dtype=float)
    zd = np.asarray(zd, dtype=float)
    n = int(len(upr))
    if n == 0:
        nan = float('nan')
        return {'members': 0, 'm_bull': nan, 'm_bear': nan, 'm_omega': nan,
                'onhold': nan, 'bel_bull': nan, 'bel_bear': nan,
                'pl_bull': nan, 'pl_bear': nan, 'uncertainty': nan,
                'ds_state': 0}

    # NaN 参与比较一律为 False ⇒ 落入 Ω（旧树同：不预先剔除成员）。
    bull = upr < ep
    bear = (ep < upr) & (zd < -eps)
    m_bull = float(bull.sum()) / n
    m_bear = float(bear.sum()) / n
    m_omega = 1.0 - m_bull - m_bear

    bel_bull = m_bull
    pl_bull = 1.0 - m_bear
    bel_bear = m_bear
    pl_bear = 1.0 - m_bull
    uncertainty = pl_bull - bel_bull          # = m_omega

    total = bel_bull + bel_bear
    ds_conf = (bel_bull / total) if total > 1e-12 else 0.5
    onhold = ds_conf * (1.0 - uncertainty)    # 恒等于 m_bull
    if (m_bear > BEAR_CLEAR_RATIO) and (m_bear > m_bull) and (m_bear > m_omega):
        onhold = 0.0

    if bel_bull > 0.5 and uncertainty < 0.1:
        ds_state = 1
    elif bel_bear > 0.5 and uncertainty < 0.1:
        ds_state = -1
    elif bel_bull < 0.5 and bel_bear < 0.5:
        ds_state = 0
    else:
        ds_state = int(np.sign(bel_bull - bel_bear))

    return {'members': n, 'm_bull': m_bull, 'm_bear': m_bear,
            'm_omega': m_omega, 'onhold': float(onhold),
            'bel_bull': bel_bull, 'bel_bear': bel_bear,
            'pl_bull': pl_bull, 'pl_bear': pl_bear,
            'uncertainty': uncertainty, 'ds_state': ds_state}
