# coding:utf-8
"""`xgb_scan` 的**训练**（旧树 `benchmark/train_xgboost.py:435-540`）。

与旧实现的差别（其余逐条保留）：

* **输入是 DataFrame**，不再读写 `*_current.pickle` 快照 —— 缓存/落库属于
  后续轮次，本模块只做纯训练。
* **xgboost 3.x API**：旧代码 `xgb.XGBClassifier(params)`（位置参数 dict）、
  `silent=0`、`nthread=10` 在新版不可用；改为关键字参数 + `n_jobs`。
* 绘图是**可选**步骤：`plot_feature_imp` 惰性 import matplotlib/seaborn，
  失败只提示（GolemQ env 是 NoMKL/headless）。
* 训练前先跑 :func:`features.validate_features` —— **缺列直接拒绝**，
  不把缺失指标静默当 NaN 喂给模型。

标签口径照旧树 `Prepare_data.prepare_X_y`：单列标签映射成
``<0 → 2`` / ``>0 → 1`` / ``==0 → 0``（3 类），并丢掉**最后一行**
（旧树原文 `y = y[:-1]; X = X[:-1]`）。
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from .features import feature_names, validate_features
from .model import DEFAULT_EVAL_RANGE, save_model_path

__all__ = ['PrepareData', 'Main', 'train_xgb_scan', 'default_params']

#: 旧树 `train_model` 的参数（去掉新版不接受的 `silent`/`nthread`/位置 dict）。
DEFAULT_PARAMS = {
    'objective': 'multi:softprob',
    'n_estimators': 300,
    'max_depth': 6,
    'learning_rate': 0.1,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'min_child_weight': 3,
    'gamma': 0.1,
    'reg_lambda': 2.0,
    'random_state': 1000,
}


def default_params(**overrides) -> Dict:
    """返回一份可传给 `XGBClassifier` 的参数副本。

    >>> default_params()['objective']
    'multi:softprob'
    """
    params = dict(DEFAULT_PARAMS)
    params.update(overrides)
    return params


class PrepareData:
    """按契约列切出训练矩阵（旧树 `Prepare_data` 的 DataFrame 版）。"""

    def __init__(self, features: pd.DataFrame, labels, columns=None,
                 verbose: bool = True):
        self.columns = list(columns or feature_names('xgb_scan'))
        self.data = features.loc[:, self.columns].copy()
        self.labels = labels
        self.verbose = verbose
        if verbose:
            print('[xgb] 训练特征 {} 列 / {} 行'.format(
                self.data.shape[1], self.data.shape[0]))

    def rescale_data(self, data: pd.DataFrame = None) -> pd.DataFrame:
        """旧树的标准化步骤（`core.preprocessing.normalize`）。

        ⚠️ 旧 `train_xgboost` 主路径**没有调用**它；保留为可选工具。
        """
        from GolemQ.core.preprocessing import normalize
        target = self.data if data is None else data
        target = target.bfill()
        return pd.DataFrame(normalize(target.values), columns=target.columns,
                            index=target.index)


class Main:
    """训练/评估的最小封装（旧树 `benchmark/train_xgboost.py:110-430`）。"""

    def __init__(self):
        self.data = None
        self.X_train = self.X_test = None
        self.y_train = self.y_test = None
        self.model = None

    def prepare_X_y(self, data: pd.DataFrame, labels):
        """标签 → 3 类整数；最后一行丢弃（旧树原样）。

        >>> md = Main()
        >>> frame = pd.DataFrame({'a': [1.0, 2.0, 3.0]})
        >>> labels = pd.DataFrame({'lab': [1.0, -1.0, 0.0]})
        >>> X, y = md.prepare_X_y(frame, labels)
        >>> X.shape, y.tolist()
        ((2, 1), [1, 2])
        """
        X = data.values
        if isinstance(labels, pd.Series):
            labels = labels.to_frame()
        if labels.shape[1] != 1:
            y = labels
        else:
            col = labels.iloc[:, 0]
            y = pd.DataFrame(np.where(col < 0, 2, np.where(col > 0, 1, 0)),
                             columns=labels.columns, index=labels.index)
        y = y.iloc[:-1]
        X = X[:-1]
        return X, np.asarray(y).ravel()

    def split_train_test(self, X, y, split_ratio: float = 0.9):
        """时序切分：前 90% 训练、后 10% 测试（旧树同款，**不 shuffle**）。"""
        train_size = int(round(split_ratio * X.shape[0]))
        return (X[:train_size], X[train_size:],
                y[:train_size], y[train_size:])

    def train_model(self, train_x, train_y, **params):
        """训练 XGBClassifier；参数默认见 :func:`default_params`。"""
        import xgboost as xgb                # noqa: PLC0415 惰性

        clf = xgb.XGBClassifier(**default_params(**params))
        clf.fit(train_x, train_y)
        return clf

    def predict(self, model=None, X_test=None, y_test=None):
        """返回预测数组；给了 `y_test` 时打印一份分类报告。"""
        model = model or self.model
        X_test = self.X_test if X_test is None else X_test
        y_pred = model.predict(X_test)
        if y_test is not None:
            from sklearn.metrics import classification_report
            print(classification_report(y_test, y_pred, zero_division=0))
        return y_pred

    def predict_proba(self, model=None, X_test=None, y_test=None):
        """返回概率矩阵（列顺序 = `model.classes_`）。"""
        model = model or self.model
        X_test = self.X_test if X_test is None else X_test
        return model.predict_proba(X_test)

    def plot_feature_imp(self, data=None, model=None):
        """特征重要性图（**可选**；无 matplotlib/seaborn 时只打印前 20）。"""
        data = self.data if data is None else data
        model = model or self.model
        try:
            import matplotlib.pyplot as plt      # noqa: PLC0415
            import seaborn as sns                # noqa: PLC0415
        except Exception as exc:                 # noqa: BLE001 只提示，不毁训练
            print('[xgb] 跳过特征重要性绘图（{}）'.format(exc))
            return None
        importances = pd.Series(model.feature_importances_,
                                index=data.columns).sort_values(ascending=False)
        print(importances.head(20))
        sns.barplot(x=importances.values, y=importances.index)
        plt.tight_layout()
        plt.show()
        return importances


def train_xgb_scan(features: pd.DataFrame, labels,
                   neg_labels=None,
                   eval_range: str = DEFAULT_EVAL_RANGE,
                   columns=None,
                   params: Optional[dict] = None,
                   verbose: bool = False) -> Dict[str, object]:
    """训练 `xgb_scan` 多头 + （可选）空头模型，返回模型与落点。

    :param features: 含契约列的 DataFrame（索引任意；行序即时间序）。
    :param labels: 多头标签（单列；旧树 `ATTEN_LABEL`）。
    :param neg_labels: 空头标签（单列；旧树 `ATTEN_NEG_LABEL`）；``None`` 只训多头。
    :param eval_range: 模型文件名后缀（`xgb_scan_{eval_range}.pickle`）。
    :raises ValueError: 契约列缺失（见 :func:`features.validate_features`）。
    :returns: ``{'long': model, 'short': model|None, 'path': ..., 'path_short': ...}``
    """
    columns = list(columns or feature_names('xgb_scan'))
    validate_features(features, model='xgb_scan')

    prepared = PrepareData(features, labels, columns, verbose=verbose)
    manifest = Main()
    X, y = manifest.prepare_X_y(prepared.data, prepared.labels)
    manifest.data = prepared.data
    manifest.X_train, manifest.X_test, manifest.y_train, manifest.y_test = \
        manifest.split_train_test(X, y)
    manifest.model = manifest.train_model(manifest.X_train, manifest.y_train,
                                          **(params or {}))
    long_path = save_model_path(eval_range=eval_range, short=False)
    manifest.model.save_model(long_path)
    if verbose:
        print('[xgb] 多头模型 → {}'.format(long_path))

    out = {'long': manifest.model, 'short': None,
           'path': long_path, 'path_short': None}

    if neg_labels is not None:
        prepared_neg = PrepareData(features, neg_labels, columns, verbose=verbose)
        manifest_neg = Main()
        Xn, yn = manifest_neg.prepare_X_y(prepared_neg.data, prepared_neg.labels)
        manifest_neg.data = prepared_neg.data
        (manifest_neg.X_train, manifest_neg.X_test,
         manifest_neg.y_train, manifest_neg.y_test) = \
            manifest_neg.split_train_test(Xn, yn)
        manifest_neg.model = manifest_neg.train_model(
            manifest_neg.X_train, manifest_neg.y_train, **(params or {}))
        short_path = save_model_path(eval_range=eval_range, short=True)
        manifest_neg.model.save_model(short_path)
        if verbose:
            print('[xgb] 空头模型 → {}'.format(short_path))
        out.update(short=manifest_neg.model, path_short=short_path)

    return out
