# coding:utf-8
"""
Massive model constants (stub — to be populated).

Used by services/persistence/ for concept and massive model column references.
"""


class _StubMeta(type):
    """未定义的常量一律 **抛 AttributeError**。

    此前它返回 ``'mas_' + name.lower()`` —— 一个「看起来合理」的假名。
    后果见 ``PITFALLS.md`` P1：假名在数据里从不存在，取值静默失败，
    回测「跑通了」却零成交，**没有任何报错**。

    ⚠️ **不要改回返回值。** 若某常量抛 AttributeError，正解是从老树取
    真实值补上 —— 那才是该字段在 MongoDB 里的实际存储名。
    """
    def __getattr__(cls, name):
        raise AttributeError(
            f'{cls.__name__}.{name} 未定义。本类不再伪造常量名'
            f'（见 PITFALLS.md P1）。请从老树取真实值补上 ——'
            f'那是数据在 MongoDB 里的实际字段名。')


class MAS(metaclass=_StubMeta):
    """Massive model feature constants (stub)."""
    _instance = None

    # Concept combo features
    CONCEPT_ZEN_DASH_TIMING_LAG_COMBO = 'cmZdLagCmb'
    CONCEPT_XGB_ECHO_TIMING_LAG_COMBO = 'cmXgbLagCmb'
    CONCEPT_STOCK_SCORE_M15_TIMING_LAG_COMBO = 'cmSSM15LagCmb'
    CONCEPT_STOCK_SCORE_M15_NORM_TIMING_LAG_COMBO = 'cmSSM15NorLagCmb'
    CONCEPT_DIF_WAVELET_TIMING_LAG_COMBO = 'cmDifWavLagCmb'
    CONCEPT_MTM_NORM_TIMING_LAG_MAJOR_COMBO = 'C_MTM_NOR_LAG_MAJ_CMB'
    CONCEPT_DIF_NORM_TIMING_LAG_MAJOR_COMBO = 'C_DIF_NOR_LAG_MAJ_CMB'
    CONCEPT_DEA_NORM_TIMING_LAG_MAJOR_COMBO = 'C_DEA_NOR_LAG_MAJ_CMB'
    CONCEPT_STAGE = 'mCeptStg'
    CONCEPT_STAGE_DUMMY = 'mCeptStgDmy'
    CONCEPT_BOOTSTRAP_BEFORE = 'mCeptBstBf'
    CONCEPT_ENDPOINT_BEFORE = 'mCeptEpBf'
    CONCEPT_TREND_TIMING_LAG = 'mCeptTrdLag'
    CONCEPT_MACD_COMPOUDED_RATIO_MEDIAN = 'mCeptMacdCpdRtoMed'

    # Massive model features
    MACD_COMPOUDED_BAND_RATIO_MEDIAN = 'mMacdCpdBandRtoMed'
    STAGE_MODE = 'STAGE_MOD'
    BOOTSTRAP_STAGE_MODE_BEFORE = 'BST_STG_MOD_BF'

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MAS, cls).__new__(cls)
        return cls._instance
