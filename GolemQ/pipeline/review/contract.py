# coding: utf-8
"""review 管线的**列契约**（唯一真相源）。

原则：只保留新树消费方实际读取的列 + zen_minor 中间列；老树 136 列宽表里
没人读的列**不进库**。改这里之前先 `grep` 消费方（见 `API_INDEX.md`）。

⚠️ 老表 `stock_reality_features` / `index_reality_features` **尺度是 60min**
（实测 10:30/11:30/14:00/15:00），但命名缺频率后缀。新树统一并入
`*_reality_feat_60min`；其列做以下**重命名**（不保留旧名）：

===========================  ===========================
老 `reality_features` 列      新 60min 列
===========================  ===========================
`ZEN_DASH_LAG`               `zDLagR`
`ZEN_PEAK_LAG`               `zPLagR`
`ZEN_D_LAG_MAJ`              `zDLagMajR`
`ZEN_PEAK_LAG_MAJ`           `ZPLagMajR`
`M9T` / `M9T_LAG`            `m9t` / `m9tLag`
===========================  ===========================

`GAR_*` / `EP_STG_BF` **2026-10-20 起纳入**（用户明确要用）：由 coldstart 输出到
60min 族。⚠️ `garP` 落的是**相对锚 bar close 的比值带**，必须配 `garPAnchor`
一起读（D49）；不是老树的绝对价位带。
"""
from __future__ import annotations

from typing import Dict, Tuple

#: 日族 14 列（老树 `--daily` 的 `daily_columns`）。
#: ⚠️ `poly9MajR` / `poly9WekR` / `rTrePrcMajR` 存的是**除以同一行 close 的比值**，
#: 不是绝对价位 —— 复权换挡时绝对价会断层（D49 通则）。`*Nor*` 本来就是 0..1。
DAY_COLUMNS: Tuple[str, ...] = (
    'zDLagMajR', 'ZPLagMajR',
    'zDLagWekR',
    'poly9MajR', 'poly9NorMajR',
    'poly9WekR', 'poly9NorWekR',
    'm9tMajR', 'm9tLagMajR',
    'm9tWekR', 'm9tLagWekR',
    'rTrePrcMajR', 'rTreLagMajR', 'rTreSlopMajR',
)

#: 15min minor 信号（老树 `calc_zen_wavelet_reality_minor` 的 rename 表）。
MINOR_COLUMNS: Tuple[str, ...] = (
    'AroUprsgLagMin',
    '1QUprLagMin', '2QUprLagMin', '3QUprLagMin', '4QUprLagMin',
    '4QPwrlnMin',
    '4QPwrlnLedInBfMin', '4QPwrlnLedOutBfMin',
    'BSBuBfMin', 'BSSelBfMin', 'BSStarUpBfMin', 'BSStarDwnBfMin',
    'MsLeadinVarMin', 'MsLeadinEntryMin', 'MsLeadinBfMin',
    'MsLeadinx3BfMin', 'MsLeadinx5BfMin',
)

#: 60min 族：4 个 zen + 17 个 minor + 并入的前 `reality_features` 列
#: + coldstart 子集。**不再单独写 `*_coldstart_reviews`**（用户 2026-10-19 定）：
#: 冷启动列的唯一下游仍是 60min 快照，合并进同一张表，避免两张表两套真相。
HOUR_ZEN_COLUMNS: Tuple[str, ...] = (
    'zDLagR', 'zPLagR', 'zDLagMinR', 'zPLagMinR',
)
#: coldstart 子集（模型层实际读取、纯 K 线可算；不含东财/同花顺/massive）。
#: 合并了旧 `reality_features` 的日尺度列（`STAGE/BST/m9t/m9tLag`）。
COLDSTART_COLUMNS: Tuple[str, ...] = (
    'STAGE', 'BST_STG_BF', 'EP_STG_BF',
    'garP', 'garPCount', 'garPClr', 'garPAnchor', 'garSLag',
    'm9t', 'm9tLag', 'm9tMaj',
    'poly9_MAJ', 'poly9pwr3_MAJ',
    'PROPHET_LB', 'PROPHET_UB', 'PROPHET_CLR_NORM',
    'PROPHET_LB_MAJ', 'PROPHET_UB_MAJ', 'PROPHET_CLR_NORM_MAJ',
)
#: ⚠️ `m9tBsl` / `m9tBslLag` 暂不纳入：它们需要 compact 特征
#: （MACD_ZERO_TIMING_LAG / DEA/DIF/POLY9/MAPOWER30 的 major norm）那一整条链，
#: 尚未搬。用户 2026-10-19 最终口径列的是「M9T」，先落 raw/major。
HOUR_COLUMNS: Tuple[str, ...] = (HOUR_ZEN_COLUMNS + MINOR_COLUMNS +
                                 COLDSTART_COLUMNS)

#: 15min minor 的**完整**输出（老树 `calc_zen_wavelet_reality_minor` 的 19 列）：
#: 2 个 zen + 17 个 minor。
#: ⚠️ 60min 的 `zDLagMinR/zPLagMinR` 由 `zen_snapshots` 产出；`hour_minor_sample`
#: 仍只采 17 个 minor 列，**不要**把 zen 两列再采一遍（否则 60min 族列名冲突）。
MINOR_15MIN_COLUMNS: Tuple[str, ...] = ('zDLagMinR', 'zPLagMinR') + MINOR_COLUMNS

#: 15min 族：minor 信号的原生频率版本（CV 列归 ChartScan，不在契约内）。
MIN15_COLUMNS: Tuple[str, ...] = MINOR_15MIN_COLUMNS

FAMILY_COLUMNS: Dict[str, Tuple[str, ...]] = {
    'day': DAY_COLUMNS,
    '60min': HOUR_COLUMNS,
    '15min': MIN15_COLUMNS,
}

#: 老 `reality_features` → 新列名（落库前 rename）。
LEGACY_RENAME: Dict[str, str] = {
    'ZEN_DASH_LAG': 'zDLagR',
    'ZEN_PEAK_LAG': 'zPLagR',
    'ZEN_D_LAG_MAJ': 'zDLagMajR',
    'ZEN_PEAK_LAG_MAJ': 'ZPLagMajR',
    'M9T': 'm9t',
    'M9T_LAG': 'm9tLag',
}

#: 老 `metadata(revision='zw_reality')` 的 19 列 → 60min 契约列。
#: 这 19 列与 `MINOR_COLUMNS` + `zDLagMinR/zPLagMinR` 一一对应（无重命名）。
MINOR_PASSTHROUGH: Tuple[str, ...] = (
    'zDLagMinR', 'zPLagMinR',
) + MINOR_COLUMNS
