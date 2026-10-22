# coding:utf-8
"""XGBoost 模型文件的**路径解析与 Booster 存取**。

路径口径照旧树 `fractal/vXI.py:5091-5137`：先找**程序目录**
（``<cwd>/datastore/cache/model``），再找**用户目录**
（``~/.GolemQ/datastore/cache/model``）；都没有就明确报错，
不静默返回空预测。

⚠️ **本模块模块级不 import xgboost** —— 它只在 `load_booster` /
`save_booster` 里惰性导入。理由同 `analysis/fbprophet.py` 的 prophet：
`analysis/` 在 GolemQ 主环境里要能被 import，而重型模型库可能只想在
需要训练/推理时才付代价。
"""
from __future__ import annotations

import os
from typing import Optional

from GolemQ.core.path import basepath, cache_path, user_path

__all__ = [
    'DEFAULT_EVAL_RANGE', 'model_name', 'portable_model_dir', 'user_model_dir',
    'resolve_model', 'save_model_path', 'load_booster', 'save_booster',
]

#: 旧树默认的 eval 后缀（`xgb_scan_hs300.pickle`）。
DEFAULT_EVAL_RANGE = 'hs300'


def model_name(eval_range: str = DEFAULT_EVAL_RANGE, short: bool = False) -> str:
    """模型文件名。

    >>> model_name('fast'), model_name('fast', short=True)
    ('xgb_scan_fast.pickle', 'xgb_scan_neg_fast.pickle')
    """
    if short:
        return 'xgb_scan_neg_{}.pickle'.format(eval_range)
    return 'xgb_scan_{}.pickle'.format(eval_range)


def portable_model_dir() -> str:
    """程序目录下的模型目录（**不创建**，供解析用）。

    口径与 `core.path.cache_path(..., portable=True)` 一致：都基于模块导入时
    的 `basepath`（= 启动目录），不是调用时的 `os.getcwd()`。
    """
    return os.path.join(basepath, 'datastore', 'cache', 'model')


def user_model_dir() -> str:
    """用户目录下的模型目录（**不创建**，供解析用）。"""
    return os.path.join(user_path, 'datastore', 'cache', 'model')


def resolve_model(clf_name: str = None, eval_range: str = DEFAULT_EVAL_RANGE,
                  short: bool = False) -> str:
    """解析模型文件的绝对路径；先 portable、再用户目录。

    :raises FileNotFoundError: 两处都不存在（**不要**退化成空预测）。
    """
    name = clf_name or model_name(eval_range, short=short)
    for root in (portable_model_dir(), user_model_dir()):
        path = os.path.join(root, name)
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(
        '未找到 XGBoost 模型 {!r}；已查找 {} 与 {}'.format(
            name, portable_model_dir(), user_model_dir()))


def save_model_path(clf_name: str = None,
                    eval_range: str = DEFAULT_EVAL_RANGE,
                    short: bool = False) -> str:
    """训练输出的落点（程序目录；目录不存在会被创建）。"""
    name = clf_name or model_name(eval_range, short=short)
    return os.path.join(cache_path('model', portable=True), name)


def load_booster(path: str):
    """加载原生 XGBoost booster。

    ⚠️ 与训练端 `XGBClassifier.save_model()` 保存的文件兼容（同一个原生格式）。
    """
    import xgboost as xgb                    # noqa: PLC0415 惰性：训练/推理才需要

    booster = xgb.Booster()
    booster.load_model(path)
    return booster


def save_booster(model, path: str) -> str:
    """把 `XGBClassifier` / `Booster` 存成原生格式，返回路径。"""
    model.save_model(path)
    return path
