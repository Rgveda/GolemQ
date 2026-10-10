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
    BS_STAR_UP_BEFORE_MAJOR = 'BSStarUpBfMaj'
    BS_STAR_DOWN_BEFORE_MAJOR = 'BSStarDwnBfMaj'
    MAINSTREAM_LEADIN_BEFORE = 'MsLeadinBf'
    QUADRANT_TRIGGER_CREDIT = 'QuadTrgCred'
    QUADRANT_LEVERAGE_MACD_LEADIN_UB = 'QLevMACDLinUb'


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
