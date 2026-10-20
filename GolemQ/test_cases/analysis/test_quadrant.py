# coding:utf-8
"""`analysis/quadrant.py`（四量图 + 6 指标）的正确性测试。

对拍是唯一正确性证据（`DECISIONS.md` D35/D36）
==========================================
`calc_four_quadrant_boost` 与 `calc_mainstream_feats` 是**逐字搬**的，判据一律
`np.array_equal(..., equal_nan=True)` —— **不用容差**（D36：判据是 `array_equal`）。
老树 `GolemQ_old/` 不在场时，对拍用例**整份跳过**（同 `test_timing.py` 惯例）。

旧模块怎么载入（harness）
======================
老树 `analysis/base.py` 用一大串**绝对 import**（`GolemQ.utils.parameter` /
`GolemQ.models.alias` / `GolemQ.fetch.kline` / `GolemQ.indices.indices` …），新树多不存在。
所以照 `test_renko.py` 的同一套手法：

1. 往 `sys.modules` 注入 shim（常量映射到 `core.constants`；`LTT`/`TRD` 用**新树真对象**，
   否则对拍没有意义；其余名字给**惰性 no-op**，它们在函数体里才被调用）；
2. 把 `GolemQ_old/analysis/base.py` 按**命名空间包**加载，绕开 `GolemQ_old/__init__.py`；
3. **用完把注入的条目清干净** —— 不清会污染同进程后续用例（`test_renko.py` 踩过：
   留下的 `QUANTAXIS*` 让 `test_no_quantaxis` 当场红）。

⚠️ `GolemQ.analysis.timeseries` 是**在真模块上补属性、绝不替换** —— 替换会打断
`StockCN` 注册（`test_renko.py` 实测过）。
"""
from __future__ import annotations

import importlib
import os
import sys
import types

try:
    import GolemQ  # noqa: F401
except ImportError:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

import unittest

import numpy as np
import pandas as pd

from GolemQ.analysis import quadrant_jit as NEW
from GolemQ.analysis.timing import Timeline_duration, calc_event_timing_lag
from GolemQ.core import constants as _C
from GolemQ.models.alias import LTT, TRD, ZEN

_OLD_TREE = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..', '..', 'GolemQ_old'))
_OLD_FILE = os.path.join(_OLD_TREE, 'analysis', 'base.py')

#: 老树 `analysis/base.py` 顶层要的 `GolemQ.*` 子模块 —— 缺一个就 ImportError。
_SHIM_MODULES = (
    'GolemQ.utils', 'GolemQ.utils.settings', 'GolemQ.fetch', 'GolemQ.fetch.kline',
    'GolemQ.fetch.StockCN', 'GolemQ.models.zen', 'GolemQ.models.fbprophet',
    'GolemQ.analysis.pool_coef', 'GolemQ.fractal', 'GolemQ.fractal.v0',
    'GolemQ.indices', 'GolemQ.indices.indices', 'GolemQ.features',
    'QUANTAXIS', 'QUANTAXIS.QAUtil',
)


def _auto_shim(name: str) -> types.ModuleType:
    """惰性 no-op 模块：`from X import Y` 拿到一个什么都不做的函数。

    这些名字（`GQ_fetch_stock_ranking` / `calc_wavlet_coef_vXIIs` …）只在我们**不调用**
    的函数体里出现，所以替身够了。
    """
    m = types.ModuleType(name)

    class _Shim(types.ModuleType):
        def __getattr__(self, k):
            if k.startswith('__'):
                raise AttributeError(k)
            setattr(self, k, (lambda *a, **kw: None))
            return getattr(self, k)

    m.__class__ = _Shim
    return m


def _load_old_base():
    """载入旧 `analysis/base.py`。老树不在场返回 ``None``。"""
    if not os.path.isfile(_OLD_FILE):
        return None
    from GolemQ.models import alias as _alias

    saved = {n: sys.modules.get(n) for n in _SHIM_MODULES}
    saved['GolemQ.utils.parameter'] = sys.modules.get('GolemQ.utils.parameter')
    saved['GolemQ.models.alias'] = sys.modules.get('GolemQ.models.alias')
    saved['GolemQ.features.base'] = sys.modules.get('GolemQ.features.base')
    saved['_oldbase'] = sys.modules.get('_oldbase')

    import GolemQ.analysis.timeseries as _ts
    ts_added = {k: getattr(_ts, k, None)
                for k in ('Timeline_duration', 'calc_energy', 'calc_event_timing_lag')}

    try:
        sys.modules['GolemQ.utils'] = types.ModuleType('GolemQ.utils')
        param = types.ModuleType('GolemQ.utils.parameter')
        param.AKA, param.STATE = _C.AKA, _C.STATE
        param.INDICATOR_FIELD, param.FEATURES = _C.FIELD, _C.FEATURES
        sys.modules['GolemQ.utils.parameter'] = param

        for name in _SHIM_MODULES:
            if name not in ('GolemQ.utils', 'QUANTAXIS'):
                sys.modules[name] = _auto_shim(name)
        qa = types.ModuleType('QUANTAXIS')
        qa.MARKET_TYPE = _C.MARKET_TYPE
        sys.modules['QUANTAXIS'] = qa

        # alias：用**新树真 LTT/TRD**（否则对拍无意义），RSK/FBP/RAIL 走 no-op
        al = _auto_shim('GolemQ.models.alias')
        al.LTT, al.TRD = _alias.LTT, _alias.TRD
        sys.modules['GolemQ.models.alias'] = al

        # features.base：clearup 用新树的**真实现**
        fb = _auto_shim('GolemQ.features.base')
        fb.clearup_compact_dataframe_int16 = NEW.clearup_compact_dataframe_int16
        fb.clearup_compact_dataframe_float32 = NEW.clearup_compact_dataframe_float32
        sys.modules['GolemQ.features.base'] = fb

        # timeseries：**补属性，不替换**（替换会打断 StockCN 注册）
        for k, v in (('Timeline_duration', Timeline_duration),
                     ('calc_energy', NEW.calc_energy),
                     ('calc_event_timing_lag', calc_event_timing_lag)):
            setattr(_ts, k, v)

        pkg = types.ModuleType('_oldbase')
        pkg.__path__ = [os.path.join(_OLD_TREE, 'analysis')]
        sys.modules['_oldbase'] = pkg
        return importlib.import_module('_oldbase.base')
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
        # 旧 base.py 已经把这些名字收进自己的 globals，此后不再需要 sys.modules 条目
        for name in ('_oldbase', 'GolemQ.indices', 'GolemQ.indices.indices'):
            sys.modules.pop(name, None)


def _make_ohlc(n=400, seed=7, code='600519'):
    np.random.seed(seed)
    idx = pd.MultiIndex.from_product(
        [pd.date_range('2020-01-01', periods=n, freq='D'), [code]],
        names=['datetime', 'code'])
    close = pd.Series(100 + np.cumsum(np.random.randn(n)), index=idx)
    return pd.DataFrame({
        'open': close.shift(1).fillna(close.iloc[0]),
        'high': close + 1.0,
        'low': close - 1.0,
        'close': close,
        'volume': np.random.randint(1000, 9999, n),
    }, index=idx)


def _same(a, b) -> bool:
    """逐值相同（含 NaN）—— 判据不容差（D36）。

    ⚠️ 不能直接 `np.array_equal(..., equal_nan=True)`：列里混着 float 与 object
    （如 `ChoBits` 是字符串）时 `isnan` 会抛 `TypeError`。用 pandas 逐元素比。
    """
    a = np.asarray(a, dtype=object)
    b = np.asarray(b, dtype=object)
    if a.shape != b.shape:
        return False
    sa = pd.Series(a.ravel())
    sb = pd.Series(b.ravel())
    return bool(((sa == sb) | (sa.isna() & sb.isna())).all())


@unittest.skipUnless(os.path.isfile(_OLD_FILE), '老树 GolemQ_old/ 不在场 —— 对拍用例跳过')
class TestMatchesOldTree(unittest.TestCase):
    """与老树 `analysis/base.py` 逐值对拍。"""

    @classmethod
    def setUpClass(cls):
        cls.old = _load_old_base()

    def _assert_same_columns(self, old_df, new_df):
        cols = sorted(set(old_df.columns) | set(new_df.columns))
        diffs = [c for c in cols
                 if c not in old_df.columns or c not in new_df.columns
                 or not _same(old_df[c], new_df[c])]
        self.assertEqual(diffs, [], '以下列与老树不一致: %s' % diffs)

    def test_four_quadrant_boost(self):
        ohlc = _make_ohlc()
        old_df = self.old.calc_four_quadrant_boost('600519', pd.DataFrame(index=ohlc.index), ohlc)
        new_df = NEW.calc_four_quadrant_boost('600519', pd.DataFrame(index=ohlc.index), ohlc)
        self._assert_same_columns(old_df, new_df)
        # 空心探针：确认真算出了东西（别让「两边都空」混过去）
        self.assertIn(LTT.QUADRANT_1, new_df.columns)
        self.assertEqual(new_df[LTT.QUADRANT_1].notna().sum(), len(ohlc))

    def test_four_quadrant_boost_with_virtual_pressure_line(self):
        """带 `TRD.AROON_UPRISING_TIMING_LAG` ⇒ 走 `annual>512` 的 virtual_pressure_line 分支。"""
        ohlc = _make_ohlc(seed=11)
        base = pd.DataFrame(index=ohlc.index)
        np.random.seed(3)
        for col in (TRD.AROON_UPRISING_TIMING_LAG, TRD.AROON_UPRISING_TIMING_LAG_MINOR):
            base[col] = np.random.randn(len(ohlc))
        old_df = self.old.calc_four_quadrant_boost('600519', base.copy(), ohlc)
        new_df = NEW.calc_four_quadrant_boost('600519', base.copy(), ohlc)
        self._assert_same_columns(old_df, new_df)
        self.assertIn(LTT.QUADRANT_POWERLINE_LEADIN, new_df.columns)

    def test_mainstream_feats(self):
        """#4/#5/#6 对拍 —— 预置 `BS_BUY_BEFORE` 让老树也不去调 `calc_bs_signals`。"""
        ohlc = _make_ohlc(seed=23)
        base = pd.DataFrame(index=ohlc.index)
        base[LTT.BS_BUY_BEFORE] = 0            # 老树的 guard：有它就跳过 calc_bs_signals
        np.random.seed(5)
        base[_C.FIELD.MACD] = np.random.randn(len(ohlc))   # x3/x5 分支要它
        new_df = NEW.calc_mainstream_feats(ohlc, base.copy())
        old_df = self.old.calc_mainstream_feats(ohlc, base.copy())
        for col in (LTT.MAINSTREAM_LEADIN_BEFORE,
                    LTT.MAINSTREAM_LEADINx3_BEFORE,
                    LTT.MAINSTREAM_LEADINx5_BEFORE):
            self.assertIn(col, new_df.columns)
            self.assertTrue(_same(old_df[col], new_df[col]), '%s 与老树不一致' % col)


class TestFormulaWiring(unittest.TestCase):
    """老树不在场也能跑：把每个函数的**接线**与它声明的公式对一遍。"""

    def test_pivot_timing_lag(self):
        idx = pd.MultiIndex.from_product(
            [pd.date_range('2020-01-01', periods=6, freq='D'), ['600519']],
            names=['datetime', 'code'])
        f = pd.DataFrame({ZEN.PIVOT_ZD: [1.0, 1.0, np.nan, np.nan, np.nan, 1.0]}, index=idx)
        out = NEW.calc_pivot_timing_lag(f)
        expect = calc_event_timing_lag(np.where(~f[ZEN.PIVOT_ZD].isna(), 1, -1))
        self.assertTrue(_same(out[ZEN.PIVOT_TIMING_LAG], expect))

    def test_bs_before(self):
        idx = pd.MultiIndex.from_product(
            [pd.date_range('2020-01-01', periods=5, freq='D'), ['600519']],
            names=['datetime', 'code'])
        f = pd.DataFrame({LTT.BS_BUY: [0.0, 1.0, 0.0, 1.0, 1.0],
                          LTT.BS_SELL: [1.0, 0.0, 0.0, 0.0, 0.0],
                          LTT.BS_STAR_UP: [0.0, 0.0, 1.0, 0.0, 0.0],
                          LTT.BS_STAR_DOWN: [0.0, 0.0, 0.0, 0.0, 0.0],
                          LTT.BS_BUY_MAJOR: [1.0, 1.0, 0.0, 0.0, 0.0],
                          LTT.BS_SELL_MAJOR: [0.0, 0.0, 0.0, 1.0, 1.0],
                          LTT.BS_STAR_UP_MAJOR: [0.0, 0.0, 0.0, 0.0, 0.0],
                          LTT.BS_STAR_DOWN_MAJOR: [0.0, 1.0, 1.0, 1.0, 0.0]}, index=idx)
        out = NEW.calc_bs_before(f)
        self.assertTrue(_same(out[LTT.BS_BUY_BEFORE],
                              Timeline_duration(np.where(f[LTT.BS_BUY] > 1e-19, 1, 0))))
        self.assertTrue(_same(out[LTT.BS_STAR_DOWN_BEFORE_MAJOR],
                              Timeline_duration(np.where(f[LTT.BS_STAR_DOWN_MAJOR] > 1e-19, 1, 0))))

    def test_quadrant_bs_trunk_timing_lag_matches_its_condition(self):
        idx = pd.MultiIndex.from_product(
            [pd.date_range('2020-01-01', periods=5, freq='D'), ['600519']],
            names=['datetime', 'code'])
        f = pd.DataFrame({LTT.QUADRANT_LEVERAGE_MACD_TIMING_LAG_DUMMY: [1.0, 1.0, -1.0, 1.0, 1.0],
                          LTT.BS_STAR_UP_BEFORE_MAJOR: [1.0, 5.0, 1.0, 1.0, 1.0],
                          LTT.BS_STAR_DOWN_BEFORE_MAJOR: [2.0, 2.0, 2.0, 2.0, 2.0],
                          LTT.BS_STAR_UP_BEFORE: [1.0, 1.0, 1.0, 1.0, 1.0],
                          LTT.BS_STAR_DOWN_BEFORE: [2.0, 2.0, 2.0, 2.0, 2.0],
                          LTT.BS_BUY_BEFORE_MAJOR: [1.0, 1.0, 1.0, 1.0, 1.0],
                          LTT.BS_SELL_BEFORE_MAJOR: [2.0, 2.0, 2.0, 2.0, 2.0]}, index=idx)
        cond = ((f[LTT.QUADRANT_LEVERAGE_MACD_TIMING_LAG_DUMMY] > 1e-12) &
                ((f[LTT.BS_STAR_UP_BEFORE_MAJOR] < f[LTT.BS_STAR_DOWN_BEFORE_MAJOR]) |
                 (f[LTT.BS_STAR_UP_BEFORE] < f[LTT.BS_STAR_DOWN_BEFORE]) |
                 ((f[LTT.BS_STAR_UP_BEFORE_MAJOR] > f[LTT.BS_STAR_DOWN_BEFORE_MAJOR]) &
                  (f[LTT.BS_BUY_BEFORE_MAJOR] < f[LTT.BS_SELL_BEFORE_MAJOR]))))
        out = NEW.calc_quadrant_bs_trunk_timing_lag(f)
        self.assertTrue(_same(out[LTT.QUADRANT_BS_TRUNK_TIMING_LAG],
                              calc_event_timing_lag(np.where(cond, 1, -1))))


class TestDecoupled(unittest.TestCase):
    """用户 2026-10-11：「尽可能解耦，越少增加引用和管线越好」。"""

    def test_module_source_has_no_quantaxis_or_db(self):
        """按 **AST** 查，不按文本 —— docstring 里**故意写着**这些名字（解释丢掉了什么）。"""
        import ast
        with open(NEW.__file__, encoding='utf-8') as fh:
            tree = ast.parse(fh.read())
        imported, used = set(), set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(a.name.split('.')[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imported.add(node.module.split('.')[0])
            elif isinstance(node, ast.Name):
                used.add(node.id)
            elif isinstance(node, ast.Attribute):
                used.add(node.attr)
        for bad in ('QUANTAXIS', 'pymongo'):
            self.assertNotIn(bad, imported, '解耦验收：不该 import %s' % bad)
        for bad in ('DATABASE', 'GQ_SU_remove_stock_min_adv', 'get_kline_price_min'):
            self.assertNotIn(bad, used, '解耦验收：代码里不该出现 %s' % bad)

    def test_import_does_not_pull_quantaxis(self):
        import subprocess
        code = ('import sys, GolemQ.analysis.quadrant_jit;'
                'print("QUANTAXIS" in sys.modules)')
        out = subprocess.run([sys.executable, '-c', code],
                             capture_output=True, text=True, encoding='utf-8')
        self.assertEqual(out.stdout.strip(), 'False')


if __name__ == '__main__':
    unittest.main()
