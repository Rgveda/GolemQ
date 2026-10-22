# coding:utf-8
"""XGBoost 模型层（**跨市场**）—— `xgb_scan` 的 markup / 训练 / 预测。

搬自旧树（4.4）：

* ``markup/mark_uprising.py``      → :mod:`GolemQ.analysis.xgboost.markup`
* ``portfolio/base.py`` 的 -9..9 评分 rank → :mod:`GolemQ.analysis.xgboost.rank`
* ``benchmark/train_xgboost.py``   → :mod:`GolemQ.analysis.xgboost.train`
* ``fractal/vXI.py`` 的预测段       → :mod:`GolemQ.analysis.xgboost.predict`
* ``models/preprocessing.py`` 的特征表 → :mod:`GolemQ.analysis.xgboost.features`

⚠️ 本包**只做 DataFrame 进/出**：

* 不读 8.3、不写库（落库/特征生产是后续轮次）；
* 模块级不 import xgboost / sklearn / matplotlib（见各模块的函数内 import），
  这样 GolemQ 主环境 import 本包不会付重型依赖的代价。
"""
from .features import (DEPRECATED_FEATURES, LEGACY_TO_CANONICAL, FeatureSpec,
                       XGB_SCAN_FEATURES, feature_names, feature_specs,
                       feature_status_report, missing_columns,
                       validate_features)
from .markup import (calc_fractal_stats, calc_transcation_stats_np,
                     markup_required_columns, markup_uprising_labels,
                     validate_markup_inputs)
from .model import (load_booster, model_name, resolve_model, save_booster,
                    save_model_path)
from .predict import (PREDICT_LONG, PREDICT_PROB_LONG,
                      calc_xgbclf_pred_prob, predict_xgb_scan)
from .rank import (ZEN_SHAPELET_TIMING_LAG, ZEN_SHAPELET_TIMING_LAG_MAJOR,
                   calc_portfolio_metaphase, calc_portfolio_rank,
                   get_wait_all_greedy_neg, get_wait_all_greedy_opt,
                   macd_cross_features, rank_from_83)
from .train import Main, PrepareData, default_params, train_xgb_scan

__all__ = [
    'FeatureSpec', 'XGB_SCAN_FEATURES', 'LEGACY_TO_CANONICAL',
    'DEPRECATED_FEATURES',
    'feature_specs', 'feature_names', 'missing_columns', 'validate_features',
    'feature_status_report',
    'markup_required_columns', 'validate_markup_inputs',
    'calc_transcation_stats_np', 'calc_fractal_stats',
    'markup_uprising_labels',
    'model_name', 'resolve_model', 'save_model_path', 'load_booster',
    'save_booster',
    'PREDICT_LONG', 'PREDICT_PROB_LONG',
    'calc_xgbclf_pred_prob', 'predict_xgb_scan',
    'ZEN_SHAPELET_TIMING_LAG', 'ZEN_SHAPELET_TIMING_LAG_MAJOR',
    'get_wait_all_greedy_opt', 'get_wait_all_greedy_neg',
    'macd_cross_features', 'calc_portfolio_metaphase', 'calc_portfolio_rank',
    'rank_from_83',
    'PrepareData', 'Main', 'default_params', 'train_xgb_scan',
]
