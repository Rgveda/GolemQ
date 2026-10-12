# coding:utf-8
"""`analysis/rsrs.py`（RSRS 指标计算）的正确性测试。

对拍是唯一正确性证据（`DECISIONS.md` D35/D36）
==========================================
`pre_rsrs_data_func` 是**逐字搬**的，判据 `np.array_equal` 语义、**不容差** ——
尤其这里：回归是逐 bar 一个 `statsmodels.OLS`，向量化会动最低位。老树不在场时整份跳过。

旧模块怎么载入（harness）
======================
老树 `signal/rsrs.py` 用绝对 import `from GolemQ.utils.parameter import (AKA,
INDICATOR_FIELD as FLD, TREND_STATUS as ST)` 与 `from GolemQ.analysis.timeseries import *`
（`rolling_zscore` 就在那），新树都没有。故照 `test_quadrant.py` / `test_renko.py` 同一套：
注入 shim → 命名空间包加载 → **用完清干净**（不清理会污染 `test_no_quantaxis`）。
⚠️ `GolemQ.analysis.timeseries` 是**补属性、不替换**。
"""
from __future__ import annotations

import importlib
import os
import sys
import types

try:
    import GolemQ  # noqa: F401
except ImportError:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import unittest

import numpy as np
import pandas as pd

from GolemQ.analysis import rsrs as NEW
from GolemQ.analysis.timing import rolling_zscore, strided_app
from GolemQ.core import constants as _C
from GolemQ.core.constants import FIELD as FLD

_OLD_TREE = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..', 'GolemQ_old'))
_OLD_FILE = os.path.join(_OLD_TREE, 'signal', 'rsrs.py')

_SHIM_MODULES = ('GolemQ.utils', 'QUANTAXIS', 'QUANTAXIS.QAUtil', 'QUANTAXIS.QAUtil.QADate_Adv')


def _auto_shim(name: str) -> types.ModuleType:
    m = types.ModuleType(name)

    class _Shim(types.ModuleType):
        def __getattr__(self, k):
            if k.startswith('__'):
                raise AttributeError(k)
            setattr(self, k, (lambda *a, **kw: None))
            return getattr(self, k)

    m.__class__ = _Shim
    return m


def _load_old_rsrs():
    if not os.path.isfile(_OLD_FILE):
        return None
    saved = {n: sys.modules.get(n) for n in _SHIM_MODULES}
    saved['GolemQ.utils.parameter'] = sys.modules.get('GolemQ.utils.parameter')
    saved['_oldrsrs'] = sys.modules.get('_oldrsrs')

    import GolemQ.analysis.timeseries as _ts
    ts_added = {k: getattr(_ts, k, None)
                for k in ('rolling_zscore', 'strided_app', 'Timeline_duration',
                          'calc_event_timing_lag', 'calc_energy')}

    try:
        sys.modules['GolemQ.utils'] = types.ModuleType('GolemQ.utils')
        param = types.ModuleType('GolemQ.utils.parameter')
        param.AKA, param.INDICATOR_FIELD = _C.AKA, _C.FIELD

        class _ST:                       # 新树有意不补 VERBOSE（见 test_renko 的说明）
            VERBOSE = 'verbose'
        param.TREND_STATUS = _ST
        sys.modules['GolemQ.utils.parameter'] = param
        for name in ('QUANTAXIS', 'QUANTAXIS.QAUtil', 'QUANTAXIS.QAUtil.QADate_Adv'):
            sys.modules[name] = _auto_shim(name)

        # timeseries：**补属性、不替换** —— 旧模块 `import *` 要在这里取到这些名字
        from GolemQ.analysis import timing as _T
        for k, v in (('rolling_zscore', rolling_zscore), ('strided_app', strided_app),
                     ('Timeline_duration', _T.Timeline_duration),
                     ('calc_event_timing_lag', _T.calc_event_timing_lag),
                     ('calc_energy', _T.calc_energy)):
            setattr(_ts, k, v)

        pkg = types.ModuleType('_oldrsrs')
        pkg.__path__ = [os.path.join(_OLD_TREE, 'signal')]
        sys.modules['_oldrsrs'] = pkg
        return importlib.import_module('_oldrsrs.rsrs')
    finally:
        for name, old in saved.items():
            if old is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old
        for k, old in ts_added.items():
            if old is None:
                try:
                    delattr(_ts, k)
                except AttributeError:
                    pass
            else:
                setattr(_ts, k, old)
        sys.modules.pop('_oldrsrs', None)


def _make_ohlc(n=400, seed=7, code='600519'):
    rng = np.random.default_rng(seed)
    idx = pd.MultiIndex.from_product(
        [pd.date_range('2020-01-01', periods=n, freq='D'), [code]],
        names=['datetime', 'code'])
    close = 100 + np.cumsum(rng.normal(size=n))
    return pd.DataFrame({'high': close + rng.uniform(0, 1, n),
                         'low': close - rng.uniform(0, 1, n)}, index=idx)


@unittest.skipUnless(os.path.isfile(_OLD_FILE), '老树 GolemQ_old/ 不在场 —— 对拍用例跳过')
class TestMatchesOldTree(unittest.TestCase):
    """与老树 `signal/rsrs.py::pre_rsrs_data_func` 逐值对拍。"""

    @classmethod
    def setUpClass(cls):
        cls.old = _load_old_rsrs()

    def test_pre_rsrs_data_func(self):
        ohlc = _make_ohlc()
        old_df = self.old.pre_rsrs_data_func(ohlc)
        new_df = NEW.pre_rsrs_data_func(ohlc)
        self.assertEqual(list(new_df.columns), list(old_df.columns))
        for col in old_df.columns:
            self.assertTrue(
                np.array_equal(old_df[col].values, new_df[col].values),
                '%s 与老树不一致' % col)
        # 空心探针：确认真算出了非零 beta（别让「两边都全 0」混过去）
        self.assertTrue(new_df[FLD.RSRS_BETA].abs().sum() > 0)

    def test_pre_rsrs_data_func_with_indices(self):
        """给 `indices=` 时 5 列**写回**那个帧（而不是新建）。"""
        ohlc = _make_ohlc(seed=13)
        ind_old = pd.DataFrame(index=ohlc.index)
        ind_new = pd.DataFrame(index=ohlc.index)
        r_old = self.old.pre_rsrs_data_func(ohlc, indices=ind_old)
        r_new = NEW.pre_rsrs_data_func(ohlc, indices=ind_new)
        for col in (FLD.RSRS_BETA, FLD.RSRS_RIGHT):
            self.assertTrue(np.array_equal(r_old[col].values, r_new[col].values),
                            '%s 与老树不一致' % col)

    def test_pre_rsrs_data_func_short_series(self):
        """`len <= M` ⇒ `rolling_zscore` 走**整段 zscore** 那支。"""
        ohlc = _make_ohlc(n=60, seed=29)
        old_df = self.old.pre_rsrs_data_func(ohlc)
        new_df = NEW.pre_rsrs_data_func(ohlc)
        for col in old_df.columns:
            self.assertTrue(np.array_equal(old_df[col].values, new_df[col].values),
                            '%s 与老树不一致（短序列支线）' % col)


class TestShape(unittest.TestCase):
    """老树不在场也能跑：列序、索引、dtype。"""

    def test_columns_and_index(self):
        ohlc = _make_ohlc(n=60, seed=3)
        out = NEW.pre_rsrs_data_func(ohlc)
        self.assertEqual(list(out.columns),
                         [FLD.RSRS_BETA, FLD.RSRS_BETA_R2, FLD.RSRS_ZSCORE,
                          FLD.RSRS_ZSCORE_R2, FLD.RSRS_RIGHT])
        self.assertTrue(out.index.equals(ohlc.index))
        self.assertEqual(out.shape, (60, 5))

    def test_missing_high_low_raises(self):
        """入参不再支持 QUANTAXIS DataStruct —— 缺 `high`/`low` 列要**报错**，不静默。"""
        idx = pd.MultiIndex.from_product(
            [pd.date_range('2020-01-01', periods=30, freq='D'), ['600519']],
            names=['datetime', 'code'])
        df = pd.DataFrame({'close': np.arange(30.0)}, index=idx)
        with self.assertRaises(AttributeError):
            NEW.pre_rsrs_data_func(df)


if __name__ == '__main__':
    unittest.main()
