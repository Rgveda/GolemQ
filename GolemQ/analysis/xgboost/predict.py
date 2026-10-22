# coding:utf-8
"""`xgb_scan` 的**推理**（旧树 `fractal/vXI.py:5141-5235`）。

逐字移植 `calc_xgbclf_pred_prob`：

1. 按 :func:`features.feature_names` 取列（**顺序即训练时的列顺序**）；
2. ``xgb.Booster().load_model(path)`` → ``xgb.DMatrix`` → ``booster.predict``；
3. ``argmax`` 得到分类，``prob_1`` 作为多头概率；
4. 把两列写回特征帧（列名照旧树 `TREND_STATUS.PREDICT_LONG` /
   `FIELD.PREDICT_PROB_LONG`）。

⚠️ **本轮只返回 DataFrame，不落库**（用户 2026-10-21 定）。落库路径等特征链
与 CLI 稳定后再议；不要在这个模块里 import 市场包或 Mongo。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .features import feature_names, validate_features
from .model import DEFAULT_EVAL_RANGE, load_booster, resolve_model

__all__ = [
    'PREDICT_LONG', 'PREDICT_PROB_LONG',
    'calc_xgbclf_pred_prob', 'predict_xgb_scan',
]

#: 旧树 `TREND_STATUS.PREDICT_LONG` 的存储值（新树常量类尚未收录）。
PREDICT_LONG = 'predict_long'
#: 旧树 `FIELD.PREDICT_PROB_LONG` 的存储值（新树常量类尚未收录）。
PREDICT_PROB_LONG = 'PRED_PROB_LONG'


def calc_xgbclf_pred_prob(features: pd.DataFrame,
                          clf_filename: str = None,
                          features_subset: str = 'xgb_scan',
                          columns=(PREDICT_LONG, PREDICT_PROB_LONG),
                          verbose: bool = False) -> pd.DataFrame:
    """对 `features` 的每一行跑一次分类，返回**新帧**（不修改入参）。

    :param clf_filename: 模型路径；``None`` ⇒ 用 :func:`model.resolve_model`
        （默认 `xgb_scan_hs300.pickle`，先程序目录、再用户目录）。
    :raises ValueError: 特征缺列（见 :func:`features.validate_features`）。
    :raises FileNotFoundError: 模型文件不存在。
    """
    validate_features(features, model=features_subset)
    path = clf_filename or resolve_model(eval_range=DEFAULT_EVAL_RANGE)

    import xgboost as xgb                    # noqa: PLC0415 惰性：推理才需要

    booster = load_booster(path)
    x_samples = features.loc[:, list(feature_names(features_subset))]
    # `DMatrix` 不接受 object/字符串列；契约列理论上都是数值，这里显式转 float32
    # （与旧树 `features.values` 的行为一致，但把 NaN → NaN 保留给 xgboost 处理）。
    dmatrix = xgb.DMatrix(x_samples.to_numpy(dtype=np.float32))
    proba = np.asarray(booster.predict(dmatrix))

    if proba.ndim == 1:                      # 二分类 booster 可能返回 1-D
        proba = np.c_[1.0 - proba, proba]
    y_pred = np.argmax(proba, axis=1)

    out = features.copy()
    out[columns[0]] = y_pred
    out[columns[1]] = proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]

    if verbose:
        print('[xgb] {} 行 → {}'.format(len(out), path))
    return out


def predict_xgb_scan(features: pd.DataFrame,
                     eval_range: str = DEFAULT_EVAL_RANGE,
                     short: bool = False,
                     verbose: bool = False) -> pd.DataFrame:
    """`xgb_scan` 的便捷入口：默认多头模型，``short=True`` 用空头模型。"""
    path = resolve_model(eval_range=eval_range, short=short)
    return calc_xgbclf_pred_prob(features, clf_filename=path, verbose=verbose)
