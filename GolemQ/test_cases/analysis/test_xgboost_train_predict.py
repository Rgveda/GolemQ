# coding:utf-8
"""`analysis/xgboost/{train,predict,model}.py` 的端到端用例（合成数据，真 xgboost）。

不读 8.3、不写库；模型落在 `tempfile.TemporaryDirectory()` 里，用完即删。
"""
from __future__ import annotations

import os, sys
try:
    import GolemQ
except ImportError:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd

from GolemQ.analysis.xgboost import features as F
from GolemQ.analysis.xgboost import model as MODEL
from GolemQ.analysis.xgboost import predict as P
from GolemQ.analysis.xgboost import train as T


def _frame(n=150, seed=7):
    rng = np.random.default_rng(seed)
    data = {c: rng.normal(0, 1, n) for c in F.feature_names()}
    return pd.DataFrame(data)


def _labels(n=150):
    # 0 / 1 / -1 → prepare_X_y 映射成 0 / 1 / 2
    return pd.DataFrame({'ATTEN_LABEL': np.array([-1.0, 0.0, 1.0] * (n // 3) + [-1.0] * (n % 3))})


class TestTrainPredictRoundtrip(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

        def fake_save_path(clf_name=None, eval_range=MODEL.DEFAULT_EVAL_RANGE,
                           short=False):
            return os.path.join(self.tmp.name, MODEL.model_name(eval_range, short))

        patcher = mock.patch.object(T, 'save_model_path', fake_save_path)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_train_save_load_predict(self):
        frame = _frame()
        labels = _labels()
        out = T.train_xgb_scan(frame, labels, eval_range='unittest',
                               params={'n_estimators': 10})
        self.assertTrue(os.path.isfile(out['path']))
        self.assertIsNotNone(out['long'])

        booster = MODEL.load_booster(out['path'])
        self.assertIsNotNone(booster)

        pred = P.calc_xgbclf_pred_prob(frame, clf_filename=out['path'])
        self.assertIn(P.PREDICT_LONG, pred.columns)
        self.assertIn(P.PREDICT_PROB_LONG, pred.columns)
        self.assertEqual(len(pred), len(frame))
        self.assertTrue(pred[P.PREDICT_PROB_LONG].between(0, 1).all())

        # 便捷入口：走 resolve_model
        with mock.patch.object(P, 'resolve_model', return_value=out['path']):
            pred2 = P.predict_xgb_scan(frame, eval_range='unittest')
        self.assertEqual(len(pred2), len(frame))

    def test_short_model_is_trained_when_labels_given(self):
        frame = _frame()
        out = T.train_xgb_scan(frame, _labels(), neg_labels=_labels(),
                               eval_range='unittest_neg',
                               params={'n_estimators': 5})
        self.assertIsNotNone(out['short'])
        self.assertTrue(os.path.isfile(out['path_short']))

    def test_train_refuses_missing_feature_columns(self):
        frame = _frame().drop(columns=['CCI_NORM'])
        with self.assertRaises(ValueError) as ctx:
            T.train_xgb_scan(frame, _labels())
        self.assertIn('CCI_NORM', str(ctx.exception))

    def test_predict_refuses_missing_feature_columns(self):
        with self.assertRaises(ValueError):
            P.calc_xgbclf_pred_prob(_frame().drop(columns=['RSI']),
                                    clf_filename='unused.pickle')


class TestModelResolution(unittest.TestCase):

    def test_missing_model_raises(self):
        with self.assertRaises(FileNotFoundError):
            MODEL.resolve_model('definitely_not_a_model.pickle')

    def test_model_name(self):
        self.assertEqual(MODEL.model_name('fast'), 'xgb_scan_fast.pickle')
        self.assertEqual(MODEL.model_name('fast', short=True),
                         'xgb_scan_neg_fast.pickle')


if __name__ == '__main__':
    unittest.main()
