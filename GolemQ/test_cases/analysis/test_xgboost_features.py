# coding:utf-8
"""`analysis/xgboost/features.py` 的契约用例。

钉住三件事：

1. 旧树 58 个条目 → **55 个唯一列**（3 组 `MTMPOWER30*` / `MTM_NORM*` 重复）；
2. 用户 2026-10-21 定的 **8 个废弃列**不被默认 `feature_names()` 带出；
3. 缺列时 `validate_features` **直接抛错**，不静默 NaN。
"""
from __future__ import annotations

import os, sys
try:
    import GolemQ
except ImportError:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

import unittest

import pandas as pd

from GolemQ.analysis.xgboost import features as F


class TestFeatureContract(unittest.TestCase):

    def test_58_entries_collapse_to_55_unique_columns(self):
        self.assertEqual(len(F.XGB_SCAN_FEATURES), 55)
        self.assertEqual(len({s.name for s in F.XGB_SCAN_FEATURES}), 55)
        self.assertEqual(len({s.legacy for s in F.XGB_SCAN_FEATURES}), 55)

    def test_default_feature_names_exclude_deprecated(self):
        self.assertEqual(len(F.feature_names()), 41)
        self.assertEqual(len(F.feature_names(include_deprecated=True)), 55)
        self.assertEqual(len(F.DEPRECATED_FEATURES), 14)
        for name in F.DEPRECATED_FEATURES:
            self.assertNotIn(name, F.feature_names())

    def test_deprecated_set_is_the_user_chosen_group(self):
        """2026-10-21 用户定：14 列放弃（资金流 / MASSIVE / 两个 GAR / 旧 4 列）。"""
        self.assertEqual(
            F.DEPRECATED_FEATURES,
            frozenset({'ZW_COEF_SN', 'ZW_COEF', 'DESINTY_CHAIN_LAG',
                       'MPWR3_MTR', 'MASS_TRD_JX_BF', 'MASS_SEMI_TRD_JX_BF',
                       'ZEN_PEAK_RET_GAR', 'ZEN_PEAK_RET_GAR_MAJOR',
                       'ma90ClrRetGar', 'moneyflowIn', 'MONEYFLOW_IN_BF',
                       'moneyflowVolume', 'moneyflowPct', 'MTM_FLOW_LAG'}))

    def test_replacements_are_produced_columns(self):
        self.assertEqual(set(F.REPLACEMENTS), F.DEPRECATED_FEATURES)
        produced = set(F.feature_names())
        for old, repl in F.REPLACEMENTS.items():
            for name in repl:
                self.assertIn(name, produced, (old, name))

    def test_status_report_totals(self):
        report = F.feature_status_report()
        self.assertEqual(set(report), set(F.STATUSES))
        self.assertEqual(sum(len(v) for v in report.values()), 55)
        self.assertEqual(len(report[F.STATUS_DEPRECATED]), 14)
        # 废弃 14 列后，剩下的 41 列全部有生产者。
        self.assertEqual(len(report[F.STATUS_PRODUCED]), 41)
        self.assertEqual(len(report[F.STATUS_DERIVABLE]), 0)
        self.assertEqual(len(report[F.STATUS_MISSING_SOURCE]), 0)
        self.assertEqual(len(report[F.STATUS_MISSING_PRODUCER]), 0)

    def test_validate_features_reports_every_missing_column(self):
        with self.assertRaises(ValueError) as ctx:
            F.validate_features(pd.DataFrame({'RSI': [1.0]}))
        msg = str(ctx.exception)
        self.assertIn('缺 40 列', msg)
        self.assertIn('cciNorm', msg)

    def test_validate_features_passes_with_full_contract(self):
        frame = pd.DataFrame({c: [0.0] for c in F.feature_names()})
        F.validate_features(frame)      # 不抛即通过

    def test_legacy_alias_duplicates_resolve_to_one_canonical(self):
        # 旧树三个 MTMPOWER30 常量与 MTM_NORM 系列指向同一存储列；
        # 契约按存储名去重，只保留一个 legacy 键。
        self.assertEqual(F.LEGACY_TO_CANONICAL['MTM_NORM'], 'mtmNorm')
        self.assertEqual(F.LEGACY_TO_CANONICAL['MTM_NORM_MAJ'], 'mtmNormMaj')
        self.assertEqual(F.LEGACY_TO_CANONICAL['MTM_NORM_WEK'], 'mtmNormWek')

    def test_feature_names_order_is_legacy_order(self):
        names = F.feature_names(include_deprecated=True)
        self.assertEqual(names[0], 'cciNorm')
        self.assertEqual(names[-1], 'mtmmaNormMaj')


if __name__ == '__main__':
    unittest.main()
