# coding:utf-8
"""`analysis/poly9.py`（Poly9 **编排层**）与老树的**逐值对拍**。

判据 `np.array_equal`，**不用容差**（`DECISIONS.md` D35/D36）。
老树 `GolemQ_old/` 不在场时对拍整份跳过（harness 见 `old_tree.py`）。

这一份对拍同时是 `TA_HMA` 的**真验证**
=====================================
`HMA10` 在这条链上是被**真正消费**的（拼 `s` 的 `[0:11]` / `[11:]`），
而 `analysis/indicators.TA_HMA` 没法单独与老树对拍 —— 老树那份来自 **QUANTAXIS 包**，
一 import 就会踩 `test_no_quantaxis` 的「运行时不许加载 QUANTAXIS」。
所以这里用**同一条链**去比：只要 `s` 拼对、`POLYNOMIAL9` 逐值相同，
就说明 `TA_HMA` 与 QUANTAXIS 那份在这个用法上等价。

⚠️ 给老模块补的是**老树自己那份** `rolling_poly9_v2` / `calc_event_timing_lag`
=========================================================================
老 `features/base.py` 从 `GolemQ.analysis.timeseries` import 这两个。
若补成**新树**的实现，对拍就退化成"自己跟自己比"（编排层之外全是我自己的代码）。
故先从老树载入它自己的 `analysis/timeseries.py`，再把那两个函数**补到真模块上**
（`patch_attrs`，不是用替身顶替 —— 见 `old_tree.load_old` 的说明）。
"""
from __future__ import annotations

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

from GolemQ.analysis import poly9 as NEW
from GolemQ.analysis.indicators import TA_HMA
from GolemQ.core import constants as _C
from GolemQ.test_cases.old_tree import field_compat, load_old

#: 老树**自己**的 timeseries —— 它提供 `rolling_poly9_v2` / `calc_event_timing_lag`
#: 的真实现，用来喂老 `features/base.py`（避免"自己跟自己比"）。
_OLD_TS = load_old('analysis/timeseries.py')


def _stub(*_a, **_kw):
    """给老模块**用不到**的那 5 个 import 名一个 no-op（只求 `from ... import` 能过）。"""
    return None


def _any_stub(_name):
    """模块级 `__getattr__` 的兜底：**任何**名字都给 no-op。"""
    return _stub


def _load_old_base():
    """载入老 `features/base.py`。

    ⚠️ **不能逐个枚举名字**：老 `features/base.py` 有**两处**
    `from GolemQ.analysis.timeseries import …`（`:70` 与 `:2927`），
    加起来十几个名字，而且第二处还夹着新树没有的（`resample_multi_frequency_indices_func`）。
    漏一个就 `ImportError`（**实测踩过**）。

    故给**真模块**挂一个兜底的模块级 `__getattr__`（PEP 562，Python 3.7+）——
    `from X import Y` 走的就是 `getattr(X, 'Y')`，所以兜得住。
    再显式覆盖两个**老函数真正在用**的名字，让它们取到**老树自己那份**：

    * `rolling_poly9_v2` / `calc_event_timing_lag` ← `_OLD_TS`（**不是**新树的实现，
      否则这份对拍就退化成"自己跟自己比"，见模块 docstring）

    这两条在 `patch_attrs` 里的顺序无关（字典里都设成属性），
    且 `load_old` 退出时会**原样还原**（连同 `__getattr__` 一起删掉）。
    """
    if _OLD_TS is None:
        return None
    attrs = {
        '__getattr__': _any_stub,
        'rolling_poly9_v2': _OLD_TS.rolling_poly9_v2,
        'calc_event_timing_lag': _OLD_TS.calc_event_timing_lag,
    }
    return load_old('features/base.py',
                    patch_attrs={'GolemQ.analysis.timeseries': attrs})


_OLD = _load_old_base()

#: ⚠️ 老模块的 `FLD` 要换成**兼容代理** —— 它 import 进来的是**老树那张表**
#: （`old_tree` 挂的），于是老函数写 `'poly9_delta'` 而新树读 `'poly9Delta'`
#: ⇒ `KeyError`（**实测踩过**）。代理是「新树优先、老表兜底」，两边就同名了。
if _OLD is not None:
    _OLD.FLD = field_compat()


def _frame(n=300, seed=7):
    """造一份 `(datetime, code)` 双层索引的帧，含 `close` 与 `HMA10`。

    ⚠️ `HMA10` 用**新树**的 `TA_HMA` 造 —— 两边喂的是**同一份 `s` 输入**，
    所以 `TA_HMA` 本身的差异会**原样传下去**，对拍不会把它掩盖掉。
    """
    idx = pd.MultiIndex.from_product(
        [pd.date_range('2024-01-01', periods=n, freq='D'), ['600519']],
        names=['datetime', 'code'])
    rng = np.random.default_rng(seed)
    close = np.cumsum(rng.normal(0, 0.8, n)) + 1500.0
    ohlc = pd.DataFrame({'open': close, 'high': close * 1.01,
                         'low': close * 0.99, 'close': close}, index=idx)
    feats = ohlc.copy()
    feats[_C.FIELD.HMA10] = TA_HMA(
        (ohlc['open'].values + ohlc['high'].values
         + ohlc['low'].values + ohlc['close'].values) / 4, 10)
    return ohlc, feats


@unittest.skipIf(_OLD is None, '老树 GolemQ_old/ 不在场，对拍整份跳过')
class TestMatchesOld(unittest.TestCase):
    """与老树 `calc_polynomial9_vXIs` 逐值相同。"""

    COLS = (_C.FIELD.POLYNOMIAL9, _C.FIELD.POLYNOMIAL9_DELTA,
            _C.FIELD.POLYNOMIAL9_TIMING_LAG)

    def _cmp(self, n, seed):
        ohlc_o, feats_o = _frame(n, seed)
        ohlc_n, feats_n = _frame(n, seed)
        old = _OLD.calc_polynomial9_vXIs(ohlc_o, feats_o)
        new = NEW.calc_polynomial9_vXIs(ohlc_n, feats_n)
        for col in self.COLS:
            with self.subTest(column=col, n=n, seed=seed):
                o = old[col].to_numpy(dtype=float)
                x = new[col].to_numpy(dtype=float)
                self.assertEqual(len(x), len(o))
                self.assertTrue(np.array_equal(x, o, equal_nan=True),
                                f'{col} 与老树不同（n={n} seed={seed}）')

    def test_lengths_and_seeds(self):
        for seed in (1, 5):
            for n in (60, 300, 900):
                self._cmp(n, seed)

    def test_short_series_falls_back_to_close(self):
        """`HMA10` 长度 ≤ 12 时退化成纯收盘价 —— 那条分支也要覆盖。"""
        ohlc_o, feats_o = _frame(12, 3)
        ohlc_n, feats_n = _frame(12, 3)
        old = _OLD.calc_polynomial9_vXIs(ohlc_o, feats_o)
        new = NEW.calc_polynomial9_vXIs(ohlc_n, feats_n)
        self.assertTrue(np.array_equal(new[_C.FIELD.POLYNOMIAL9].to_numpy(dtype=float),
                                       old[_C.FIELD.POLYNOMIAL9].to_numpy(dtype=float),
                                       equal_nan=True))

    def test_none_features_returns_none_on_both(self):
        ohlc, _ = _frame(30, 1)
        self.assertIsNone(_OLD.calc_polynomial9_vXIs(ohlc, None))
        self.assertIsNone(NEW.calc_polynomial9_vXIs(ohlc, None))


class TestContract(unittest.TestCase):
    """不依赖老树的结构性钉子。"""

    def test_writes_three_columns(self):
        ohlc, feats = _frame(300, 9)
        out = NEW.calc_polynomial9_vXIs(ohlc, feats)
        for col in (_C.FIELD.POLYNOMIAL9, _C.FIELD.POLYNOMIAL9_DELTA,
                    _C.FIELD.POLYNOMIAL9_TIMING_LAG):
            with self.subTest(column=col):
                self.assertIn(col, out.columns)

    def test_is_idempotent(self):
        """列已在 ⇒ 整段跳过（老树语义）。第二次调用不得改动已有值。"""
        ohlc, feats = _frame(300, 9)
        first = NEW.calc_polynomial9_vXIs(ohlc, feats)[_C.FIELD.POLYNOMIAL9].copy()
        # 把 close 换成别的东西 —— 若没有幂等闸，值会变
        ohlc2 = ohlc.copy()
        ohlc2['close'] = ohlc2['close'] * 3.0
        second = NEW.calc_polynomial9_vXIs(ohlc2, first.to_frame())
        self.assertTrue(np.array_equal(first.to_numpy(),
                                       second[_C.FIELD.POLYNOMIAL9].to_numpy(),
                                       equal_nan=True))

    def test_spliced_input_has_no_nan(self):
        """`s = [close[:11], HMA10[11:]]` 拼完**不许有 NaN** —— 否则 Chebyshev 会扩散。"""
        n = 300
        ohlc, feats = _frame(n, 11)
        s = np.r_[ohlc[_C.AKA.CLOSE].values[0:NEW.HMA_WARMUP],
                  feats[_C.FIELD.HMA10].values[NEW.HMA_WARMUP:]]
        self.assertEqual(len(s), n)
        self.assertFalse(np.isnan(s).any())


class TestNormGate(unittest.TestCase):
    """`POLYNOMIAL9_NORM` 的 `annual > 200` 闸。"""

    def _feats(self):
        ohlc, feats = _frame(300, 13)
        return NEW.calc_polynomial9_vXIs(ohlc, feats)

    def test_written_when_annual_above_gate(self):
        out = NEW.calc_polynomial9_norm_vXIs(self._feats(), annual=252)
        self.assertIn(_C.FIELD.POLYNOMIAL9_NORM, out.columns)
        y = out[_C.FIELD.POLYNOMIAL9_NORM].to_numpy(dtype=float)
        self.assertFalse(np.isnan(y).any())
        self.assertGreaterEqual(y.min(), 0.0)
        self.assertLessEqual(y.max(), 1.0)

    def test_not_written_below_gate(self):
        """闸不满足 ⇒ **不写该列**（不是写 0）。"""
        out = NEW.calc_polynomial9_norm_vXIs(self._feats(), annual=100)
        self.assertNotIn(_C.FIELD.POLYNOMIAL9_NORM, out.columns)

    def test_gate_boundary_is_strict(self):
        """判据是 `>` 不是 `>=`：`annual == 200` 不算。"""
        out = NEW.calc_polynomial9_norm_vXIs(self._feats(), annual=200)
        self.assertNotIn(_C.FIELD.POLYNOMIAL9_NORM, out.columns)


if __name__ == '__main__':
    unittest.main()
