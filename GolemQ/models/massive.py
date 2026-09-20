# coding:utf-8
"""
Massive model constants (stub — to be populated).

Used by services/persistence/ for concept and massive model column references.
"""


class _StubMeta(type):
    def __getattr__(cls, name):
        return f'mas_{name.lower()}'


class MAS(metaclass=_StubMeta):
    """Massive model feature constants (stub)."""
    _instance = None

    # Concept combo features
    CONCEPT_ZEN_DASH_TIMING_LAG_COMBO = 'concept_zen_dash_timing_lag_combo'
    CONCEPT_XGB_ECHO_TIMING_LAG_COMBO = 'concept_xgb_echo_timing_lag_combo'
    CONCEPT_STOCK_SCORE_M15_TIMING_LAG_COMBO = 'concept_stock_score_m15_timing_lag_combo'
    CONCEPT_STOCK_SCORE_M15_NORM_TIMING_LAG_COMBO = 'concept_stock_score_m15_norm_timing_lag_combo'
    CONCEPT_DIF_WAVELET_TIMING_LAG_COMBO = 'concept_dif_wavelet_timing_lag_combo'
    CONCEPT_MTM_NORM_TIMING_LAG_MAJOR_COMBO = 'concept_mtm_norm_timing_lag_major_combo'
    CONCEPT_DIF_NORM_TIMING_LAG_MAJOR_COMBO = 'concept_dif_norm_timing_lag_major_combo'
    CONCEPT_DEA_NORM_TIMING_LAG_MAJOR_COMBO = 'concept_dea_norm_timing_lag_major_combo'
    CONCEPT_STAGE = 'concept_stage'
    CONCEPT_STAGE_DUMMY = 'concept_stage_dummy'
    CONCEPT_BOOTSTRAP_BEFORE = 'concept_bootstrap_before'
    CONCEPT_ENDPOINT_BEFORE = 'concept_endpoint_before'
    CONCEPT_TREND_TIMING_LAG = 'concept_trend_timing_lag'
    CONCEPT_MACD_COMPOUDED_RATIO_MEDIAN = 'concept_macd_compouded_ratio_median'

    # Massive model features
    MACD_COMPOUDED_BAND_RATIO_MEDIAN = 'macd_compouded_band_ratio_median'
    STAGE_MODE = 'stage_mode'
    BOOTSTRAP_STAGE_MODE_BEFORE = 'bootstrap_stage_mode_before'

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MAS, cls).__new__(cls)
        return cls._instance
