# coding:utf-8
"""
Model alias constants (stub — to be populated).
"""


class LTT:
    """LTT alias constants (stub)."""
    pass
    QUADRANT_PUSH_CREDIT = 'QuadPushCred'
    MAINSTREAM_LEADIN_VAR5 = 'MsLeadinVar5'
    MAINSTREAM_LEADIN_ENTRY = 'MsLeadinEntry'
    QUADRANT_ZEN_DASH_COUNT = '4QuadZenDCnt'
    ONE_QUADRANT_UPRISING_TIMING_LAG_MAJOR = '1QUprLagMaj'
    QUADRANT_LEVERAGE_MACD_TIMING_LAG_DUMMY = 'QLevMACDLagD'
    QUADRANT_M9T_DEADPOOL_COUNT = '4QuadM9TDpCnt'
    QUADRANT_LEVERAGE_POWERLINE_UPRISING_TIMING_LAG = 'QLevPwrlnUprLag'
    TWO_QUADRANT_UPRISING_TIMING_LAG_MAJOR = '2QUprLagMaj'
    MAINSTREAM_LEADIN_BEFORE_MINOR = 'MsLeadinBfMin'
    #: 15min 的 `MsLeadin*` 重命名（老树 `reviews.py:1833` 的 rename 表）。
    MAINSTREAM_LEADIN_VAR5_MINOR = 'MsLeadinVarMin'
    MAINSTREAM_LEADIN_ENTRY_MINOR = 'MsLeadinEntryMin'
    MAINSTREAM_LEADINx3_BEFORE_MINOR = 'MsLeadinx3BfMin'
    MAINSTREAM_LEADINx5_BEFORE_MINOR = 'MsLeadinx5BfMin'
    BS_STAR_UP_BEFORE_MAJOR = 'BSStarUpBfMaj'
    BS_STAR_DOWN_BEFORE_MAJOR = 'BSStarDwnBfMaj'
    MAINSTREAM_LEADIN_BEFORE = 'MsLeadinBf'
    QUADRANT_TRIGGER_CREDIT = 'QuadTrgCred'
    QUADRANT_LEVERAGE_MACD_LEADIN_UB = 'QLevMACDLinUb'
    # ── 以下 2026-10-11 随「四量图」搬运补（`analysis/quadrant.py` 要用）────────
    # 真值**逐字**取自老树 `GolemQ_old/models/alias.py` 的 `LTT` 类 —— 全是 MongoDB
    # 实际存储名，改一个字母就是另一个字段（PITFALLS P1：只填真值，不许造）。
    QUADRANT_1 = 'Q1'
    QUADRANT_2 = 'Q2'
    QUADRANT_3 = 'Q3'
    QUADRANT_4 = 'Q4'
    QUADRANT_POWERLINE = '4QPwrln'
    QUADRANT_POWERLINE_MINOR = '4QPwrlnMin'
    QUADRANT_POWERLINE_MAJOR = '4QPwrlnMaj'
    #: 15min 重命名族（老树 `reviews.py:1815-1835`）。
    ONE_QUADRANT_UPRISING_TIMING_LAG_MINOR = '1QUprLagMin'
    TWO_QUADRANT_UPRISING_TIMING_LAG_MINOR = '2QUprLagMin'
    THREE_QUADRANT_UPRISING_TIMING_LAG_MINOR = '3QUprLagMin'
    FOUR_QUADRANT_UPRISING_TIMING_LAG_MINOR = '4QUprLagMin'
    QUADRANT_POWERLINE_LEADIN_BEFORE_MINOR = '4QPwrlnLedInBfMin'
    QUADRANT_POWERLINE_LEADOUT_BEFORE_MINOR = '4QPwrlnLedOutBfMin'
    QUADRANT_LEVERAGE_ENERGY_MINOR = 'QLevEngMin'
    QUADRANT_LEVERAGE_ENERGY = 'QLevEng'
    QUADRANT_LEVERAGE_ENERGY_MAJOR = 'QLevEngMaj'
    QUADRANT_LEVERAGE_ENERGY_DUMMY = 'QLevEngD'
    QUADRANT_LEVERAGE_ENERGY_TIMING_LAG_DUMMY = 'QLevEngLagD'
    #: 星线死叉计数（老树 `models/alias.py:1168`；`models/mainstream.py:515` 产出）。
    QUADRANT_STARDOWN_COUNT = '4QuadStDwnCnt'
    QUADRANT_POWERLINE_LEADIN = '4QPwrlnLeIn'
    QUADRANT_POWERLINE_LEADIN_BEFORE = '4QPwrlnLedInBf'
    QUADRANT_POWERLINE_LEADOUT = '4QPwrlnLedOut'
    QUADRANT_POWERLINE_LEADOUT_BEFORE = '4QPwrlnLedOutBf'
    ONE_QUADRANT_UPRISING_TIMING_LAG = '1QUprLag'
    TWO_QUADRANT_UPRISING_TIMING_LAG = '2QUprLag'
    TWO_QUADRANT_DOWNRISK_TIMING_LAG = '2QDrLag'
    THREE_QUADRANT_UPRISING_TIMING_LAG = '3QUprLag'
    THREE_QUADRANT_DOWNRISK_TIMING_LAG = '3QDrLag'
    FOUR_QUADRANT_UPRISING_TIMING_LAG = '4QUprLag'
    FOUR_QUADRANT_PRE_PRELUDE_BEFORE = '4QPreludePreBf'
    QUADRANT_DOWNRISK_TIMING_LAG = 'QuadDwnRskLag'
    QUADRANT_STATUS = 'QuadSt'
    CHOBITS = 'ChoBits'
    QUADRANT_CODEX = 'QuadCodex'
    # 四量图 ↔ BS/主力资金 链（`calc_mainstream_feats` / `calc_bs_before`）
    #: 2026-10-19：由 `TRD.OMNIPATH_UPRISING_I_COEF_DUMMY` **改名并归入四量图家族**
    #: （用户定）。旧值 `'OmniPathUpR1CoefDmy'`（老树 `models/alias.py:457`）。
    #: 实测**4.4 与 8.3 都没有任何一行存过这一列**（各 0 行）⇒ 没有数据契约在，
    #: 值随之改成四量图风格 `'QUprICoefD'`；若将来有旧列进来，在这里加映射。
    QUADRANT_UPRISING_I_COEF_CANDIDATE = 'QUprICoefCan'
    QUADRANT_UPRISING_I_COEF_DUMMY = 'QUprICoefD'
    QUADRANT_BS_TRUNK = 'QuadBsTuk'
    QUADRANT_BS_TRUNK_TIMING_LAG = 'QuadBsTukLag'
    MAINSTREAM_LEADIN_SHAKEOUT = 'MsLeadinSko'
    MAINSTREAM_LEADINx3_BEFORE = 'MsLeadinx3Bf'
    MAINSTREAM_LEADINx5_BEFORE = 'MsLeadinx5Bf'
    BS_REF_LINE = 'BSRefLn'
    BS_BORDER_LINE = 'BSBrdLn'
    BS_BUY = 'BSBu'
    BS_SELL = 'BSSel'
    BS_STAR_UP = 'BSStarUp'
    BS_STAR_DOWN = 'BSStarDwn'
    BS_BUY_MAJOR = 'BSBuMaj'
    BS_SELL_MAJOR = 'BSSelMaj'
    BS_STAR_UP_MAJOR = 'BSStarUpMaj'
    BS_STAR_DOWN_MAJOR = 'BSStarDwnMaj'
    BS_BUY_BEFORE = 'BSBuBf'
    BS_SELL_BEFORE = 'BSSelBf'
    BS_STAR_UP_BEFORE = 'BSStarUpBf'
    BS_STAR_DOWN_BEFORE = 'BSStarDwnBf'
    #: 星线/买卖线的 15min 重命名（老树 `reviews.py:1826-1829`）。
    BS_BUY_BEFORE_MINOR = 'BSBuBfMin'
    BS_SELL_BEFORE_MINOR = 'BSSelBfMin'
    BS_STAR_UP_BEFORE_MINOR = 'BSStarUpBfMin'
    BS_STAR_DOWN_BEFORE_MINOR = 'BSStarDwnBfMin'
    BS_BUY_BEFORE_MAJOR = 'BSBuBfMaj'
    BS_SELL_BEFORE_MAJOR = 'BSSelBfMaj'


class TRD:
    """交易信号字段名 —— **只补了四量图链用到的那两条**。

    ⚠️ 新树此前**没有这个类**（老树 `models/alias.py` 的 `TRD` 有 380+ 个常量）。
    2026-10-11 随四量图搬运只补了 `AROON_UPRISING_TIMING_LAG(_MINOR)`（`calc_four_quadrant_boost`
    的 `virtual_pressure_line` 分支要用）。**别把这里当成「TRD 搬完了」** ——
    其余要用到的，仍须从老树取真值再补。
    """
    AROON_UPRISING_TIMING_LAG = 'AroUprsgLag'
    AROON_UPRISING_TIMING_LAG_MINOR = 'AroUprsgLagMin'


class ZEN:
    """走势中枢（盘整箱体 / pivot）的字段名。

    值与老树 ``GolemQ_old/models/alias.py`` 的 ``ZEN`` **逐字相同** ——
    它们是数据在 MongoDB 里的实际存储名，改一个字母就是另一个字段。
    消费方：``GolemQ/analysis/pivot.py`` 的 ``attach_pivot_features``。
    """
    PIVOT_ZD = 'ZenPivotZD'              # 中枢下沿（箱底）
    PIVOT_ZG = 'ZenPivotZG'              # 中枢上沿（箱顶）
    PIVOT_GG = 'ZenPivotGG'              # 中枢最高点（延伸上界）
    PIVOT_DD = 'ZenPivotDD'              # 中枢最低点（延伸下界）
    PIVOT_DIRECTION = 'ZenPivotDir'      # 中枢方向 up/down
    PIVOT_TIMING_LAG = 'ZenPivotLag'     # 距离开中枢（PIVOT_ZD NaN）以来的 lag
    PIVOT_COUNT = 'ZenPivotCnt'          # 截至当前 bar 已确认的中枢个数（因果）
