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
`ZEN_PEAK_LAG_MAJ`           `zPLagMajR`
`M9T` / `M9T_LAG`            `m9t` / **已删除**（D67）
===========================  ===========================

`GAR_*` / `epStgBf` **2026-10-20 起纳入**（用户明确要用）：由 coldstart 输出到
60min 族。⚠️ `garP` 落的是**相对锚 bar close 的比值带**，必须配 `garPAnchor`
一起读（D49）；不是老树的绝对价位带。
"""
from __future__ import annotations

from typing import Dict, Tuple

from GolemQ.core.constants import AKA, FIELD as FLD
from GolemQ.models.alias import LTT, ZEN


#: Renko 价格量纲列 → 落库比值列（`÷close`，D70）。生产端与回填工具共用这一份，
#: 别再各写一遍（平行实现只会分叉）。
RENKO_PRICE_DUMMIES: Tuple[Tuple[str, str], ...] = (
    (FLD.RENKO_PRICE_S, FLD.RENKO_PRICE_S_DUMMY),
    (FLD.RENKO_PRICE_L, FLD.RENKO_PRICE_L_DUMMY),
    (FLD.RENKO_TREND_S_LB, FLD.RENKO_TREND_S_LB_DUMMY),
    (FLD.RENKO_TREND_S_UB, FLD.RENKO_TREND_S_UB_DUMMY),
    (FLD.RENKO_TREND_L_LB, FLD.RENKO_TREND_L_LB_DUMMY),
    (FLD.RENKO_TREND_L_UB, FLD.RENKO_TREND_L_UB_DUMMY),
)

#: review 日族的 Renko 14 列（方向/状态 + 时序原值，价格 6 列落 `*D` 比值）。
RENKO_DAY_COLUMNS: Tuple[str, ...] = (
    *(dummy for _base, dummy in RENKO_PRICE_DUMMIES),
    FLD.RENKO_TREND_S, FLD.RENKO_TREND_L, FLD.RENKO_TREND, FLD.RENKO_OPTIMAL,
    FLD.RENKO_TREND_S_TIMING_LAG, FLD.RENKO_TREND_L_TIMING_LAG,
    FLD.RENKO_BOOST_S_TIMING_LAG, FLD.RENKO_BOOST_L_TIMING_LAG,
)

#: 日族：老树 `--daily` 的 14 列（**Renko 2026-10-30 起归 60min 族**，D78）。
#: ⚠️ `poly9MajD` / `poly9WekD` / `rTrePrcMajD` 存的是**除以该行 as-of 收盘的比值**
#: （日行 = 自身 close；60min 采样行 = 快照日 close，见 D82），
#: 不是绝对价位 —— 复权换挡时绝对价会断层（D49 通则）。`*Nor*` 本来就是 0..1。
#: ⚠️ `se` / `ZenPivotCnt` **不在日族**（D78 撤销 D74）：60min 族承载，
#: 见 `HOUR_EXTRA_COLUMNS`。
DAY_COLUMNS: Tuple[str, ...] = (
    'zDLagMajR', 'zPLagMajR',
    # 日线版中枢 lag（用户 2026-11-20）；60min 的 `ZenPivotLag` 在 HOUR_ZEN_COLUMNS
    ZEN.PIVOT_TIMING_LAG_MAJOR,
    'zDLagWekR',
    'poly9MajD', 'poly9NorMajR',
    'poly9WekD', 'poly9NorWekR',
    'm9tMajR', 'm9tLagMajR',
    'm9tWekR', 'm9tLagWekR',
    'rTrePrcMajD', 'rTreLagMajR', 'rTreSlopMajD',
    # ChipDist（日频，960 根日线窗口；D78）：÷ 当日 close 的比值
    AKA.COST5_PRICE_DUMMY, AKA.COST15_PRICE_DUMMY,
    AKA.COST85_PRICE_DUMMY, AKA.COST95_PRICE_DUMMY,
)

#: 60min 族里**复用日行计算**的列（日族全部 14 列，用户 2026-10-30）：
#: 它们在 `day_core` 已算过，60min 行按「盘中=上一交易日、15:00=当日」采样，
#: 不重复算一遍。名字保留独立元组，避免以后日族加减列时静默改变 60min 契约。
DAY_SAMPLE_COLUMNS: Tuple[str, ...] = DAY_COLUMNS

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
    ZEN.PIVOT_TIMING_LAG,          # ZenPivotLag：距离开中枢的 lag（CZSC）
    # ZenOLSLag：段内 LIS/LDS 的「波浪趋势」时序（2026-11-22 重新实现，D89）
    ZEN.OLS_TIMING_LAG,
)
#: coldstart 子集（模型层实际读取、纯 K 线可算；不含东财/同花顺/massive）。
#: 合并了旧 `reality_features` 的日尺度列（`STAGE/BST/m9t`）。
#: ⚠️ 旧 `M9T_LAG` / `m9tLag` **已删除**：库里全为 None、review 无生产者；
#: 老树 REAL 常量 `m9tLagR` 仅保留定义，不落库（D67）。
#: **stage 族**（用户 2026-11-18 更正 D81）：**搬 4.4 的算法**，不是搬它的行。
#: `stage` 由 review 的 `analysis/stages.py::calc_stage_frame`（4.4
#: `calc_blh_ground_zero_anchor` + `calc_cluster_garident_stages`）重算；
#: 历史存量仍是 2026-11-16 一次性对拍时落的值（同源算法，可用于校验）。
#: `garPD/garPAnchor/garPCount/garPClr` 目前仍以历史搬运值为准，后续再从同一份
#: Garident 带换算；`bstStgBf/epStgBf` 由 `reality_attach` 用 `garPD → garP` 现算、**不落库**。
STAGE_COLUMNS: Tuple[str, ...] = (
    'stage',
    'bstStgBf', 'epStgBf',
    FLD.GARIDENT_PRICE_DUMMY, 'garPCount', 'garPClr', 'garPAnchor', 'garSLag',
)

#: 日线 coldstart 的 **Maj 列**（与 stage 无关的部分）。
COLDSTART_COLUMNS: Tuple[str, ...] = (
    'm9tMaj',
    'poly9pwr3Maj',
    FLD.FBPROPHET_LB_MAJOR_DUMMY, FLD.FBPROPHET_UB_MAJOR_DUMMY, 'prophetClrNormMaj',
)
#: 60min 额外两列（D78 从日族**改回 60min**，撤销 D74）：
#: `se` = 60min 收盘价的样本熵（15:00 日频、最后 ≤2160 根、round 3，盘内因果 ffill）；
#: `ZenPivotCnt` = 截至当前 bar 已确认的缠论中枢个数（`causal_pivot_count`）。
HOUR_EXTRA_COLUMNS: Tuple[str, ...] = (AKA.SAMPLE_ENTROPY, ZEN.PIVOT_COUNT)

#: `stage`：**4.4 算法移植**后由 review 重算（2026-11-18 修正 D81 ——
#: 用户要的是搬 4.4 旧树的算法，不是搬它落库的 STAGE 行）。
#: 算法入口 `analysis/stages.py::calc_stage_frame`（`CLT_GRP_CHK` + Garident 带）。
HOUR_STAGE_COLUMNS: Tuple[str, ...] = ('stage',)

#: ChipDist 的 `RETURN_COST5` + 化简版 `COST5_UPRISING_BEFORE`（D85）。
#: `RetCost5` 来自 **60min 全帧一次** ChipDist；`cost5UprBf` 是 `Timeline_duration`。
HOUR_COST5_UPRISING_COLUMNS: Tuple[str, ...] = (
    AKA.RETURN_COST5, AKA.COST5_UPRISING_BEFORE)

#: 周线派生列（用户 2026-11-21）：`4QPwrlnWek` 由周线 OHLC 算、映射到 60min。
HOUR_WEEKLY_COLUMNS: Tuple[str, ...] = (LTT.QUADRANT_POWERLINE_WEEKLY,)

#: **判「这一行算不算已持久化」时要排除的列**：它们由别的工具单独写，
#: 不代表 review 真跑过。
#:
#: ⚠️ 2026-11-22 实测：`tools/backfill_quad_pwrln_wek.py` 给 2018-01~2024-07
#: 补了 3,752,516 行**只有 `4QPwrlnWek`** 的 15:00 行；而
#: `persistence.persisted_keys` 的判据是「契约里**任意一列**非空」⇒ 这些行
#: 被当成"已算完"，**回填到 2024 年以前时那些 bar 会被静默跳过**。
#: 当前 800 天窗口（2024-09 起）还没踩到，属于等着爆炸的地雷。
PERSIST_EXCLUDE: Dict[str, Tuple[str, ...]] = {
    'day': (),
    '60min': HOUR_WEEKLY_COLUMNS,
    '15min': (),
}


def persist_check_columns(family: str) -> Tuple[str, ...]:
    """该族参与「是否已持久化」判定的列（已剔除 `PERSIST_EXCLUDE`）。

    >>> '4QPwrlnWek' in persist_check_columns('60min')
    False
    >>> persist_check_columns('60min') == tuple(
    ...     c for c in FAMILY_COLUMNS['60min'] if c != '4QPwrlnWek')
    True
    """
    drop = set(PERSIST_EXCLUDE.get(family, ()))
    return tuple(c for c in FAMILY_COLUMNS.get(family, ()) if c not in drop)

#: ChipDist 的筹码成本列（D78 补全）：只落 **÷close 的比值**（D49）；
#: 绝对值由 attach 的 `restore_price_columns` 乘回。ChipDist **前缀稳定**，
#: 每码整条 60min 序列算一次即可（实测 ≈2s/码）。
HOUR_CHIP_COLUMNS: Tuple[str, ...] = (
    AKA.COST5_PRICE_DUMMY, AKA.COST15_PRICE_DUMMY,
    AKA.COST85_PRICE_DUMMY, AKA.COST95_PRICE_DUMMY,
)

#: 60min **原生**特征组（D78）：算法不变，输入换成 60min 的 2160 根窗口，
#: 每个交易日 15:00 算一次（用户 2026-10-30：交易日尾部按 60min 实际时间算）。
#: 与日频版（`m9tMaj` / `poly9MajD` / `rTrePrcMajD` / `prophet*Maj*`）区分。
HOUR_NATIVE_COLUMNS: Tuple[str, ...] = (
    FLD.MAGIC_NINE_TURNS, FLD.MAGIC_NINE_TURNS_TIMING_LAG,   # m9t / m9tLag
    'poly9D', 'poly9NorR',                                   # Poly9 原生
    'rTrePrcD', 'rTreLagR', 'rTreSlopD',                     # RegTree 原生
    FLD.FBPROPHET_LB_DUMMY, FLD.FBPROPHET_UB_DUMMY,          # Prophet 原生
    'prophetClrNorm',
)

#: ⚠️ `m9tBsl` / `m9tBslLag` 暂不纳入：它们需要 compact 特征
#: （MACD_ZERO_TIMING_LAG / DEA/DIF/POLY9/MAPOWER30 的 major norm）那一整条链，
#: 尚未搬。用户 2026-10-19 最终口径列的是「M9T」，先落 raw/major。
#: 60min 行是**一体化特征行**（用户 2026-10-30）：zen/minor + coldstart +
#: `se`/`ZenPivotCnt` + **Renko 全族（60min K 线上 as-of 计算）**
#: + Poly9/RegTree（复用日行，按日采样）。
HOUR_COLUMNS: Tuple[str, ...] = (HOUR_ZEN_COLUMNS + MINOR_COLUMNS +
                                 COLDSTART_COLUMNS + HOUR_EXTRA_COLUMNS +
                                 HOUR_STAGE_COLUMNS +
                                 HOUR_COST5_UPRISING_COLUMNS +
                                 HOUR_WEEKLY_COLUMNS +
                                 RENKO_DAY_COLUMNS + DAY_SAMPLE_COLUMNS +
                                 HOUR_NATIVE_COLUMNS)

#: 15min minor 的**完整**输出（老树 `calc_zen_wavelet_reality_minor` 的 19 列）：
#: 2 个 zen + 17 个 minor。
#: ⚠️ 60min 的 `zDLagMinR/zPLagMinR` 由 `zen_snapshots` 产出；15min 表由
#: `minute15_aligned` 从同一份 60min 对齐结果取样，**不再单独重算**。
MINOR_15MIN_COLUMNS: Tuple[str, ...] = (
    'zDLagMinR', 'zPLagMinR', ZEN.PIVOT_TIMING_LAG_MINOR) + MINOR_COLUMNS

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
    'ZEN_PEAK_LAG_MAJ': 'zPLagMajR',
    'M9T': 'm9t',
    # `M9T_LAG` 已删除（D67）：库里全 None、review 无生产者；不再 rename。
    'PROPHET_LB': FLD.FBPROPHET_LB_DUMMY,
    'PROPHET_UB': FLD.FBPROPHET_UB_DUMMY,
    'PROPHET_LB_MAJ': FLD.FBPROPHET_LB_MAJOR_DUMMY,
    'PROPHET_UB_MAJ': FLD.FBPROPHET_UB_MAJOR_DUMMY,
}

#: 老 `metadata(revision='zw_reality')` 的 19 列 → 60min 契约列。
#: 这 19 列与 `MINOR_COLUMNS` + `zDLagMinR/zPLagMinR` 一一对应（无重命名）。
MINOR_PASSTHROUGH: Tuple[str, ...] = (
    'zDLagMinR', 'zPLagMinR',
) + MINOR_COLUMNS
