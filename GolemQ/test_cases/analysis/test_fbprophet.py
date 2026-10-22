# coding:utf-8
"""`analysis/fbprophet.py` —— Prophet 区间（**存比值**）的用例（`DECISIONS.md` D53）。

只钉四件事：

1. 区间对齐：输出索引与输入**逐行相同**，`lower ≤ upper`，无 NaN；
2. 比值 ↔ 绝对价互逆；
3. `_MAJOR` 按**日期**对齐到任意频率（分钟帧上要能取到当天的日线值）；
4. **尺度行为**：整窗 ×k 后，比值只差 ~0.8%（Prophet 先验是绝对量纲，**不是**
   尺度等变）—— 这条写成**上界断言**，防止有人把「存比值」吹成"数学不变"。

⚠️ 这些用例会**真的跑 Prophet**（约 1–2 秒/次），但**不碰 DB / 网络**。
"""
from __future__ import annotations

import importlib.util
import unittest

import numpy as np
import pandas as pd

from GolemQ.analysis.fbprophet import (
    _reindex_by_date, calc_fbprophet_bands, fbp_bands_from_ratio,
    fbp_bands_to_ratio, prophet_band,
)
from GolemQ.core.constants import FIELD

HAS_PROPHET = importlib.util.find_spec('prophet') is not None


def _ohlc(n=120, base=10.0, code='600519', start='2025-01-06'):
    ts = pd.bdate_range(start, periods=n)
    idx = pd.MultiIndex.from_arrays([ts, [code] * n], names=['ts', 'code'])
    close = base + np.sin(np.arange(n) / 9.0) * 2.0 + np.arange(n) * 0.05
    return pd.DataFrame({'open': close, 'high': close * 1.01, 'low': close * 0.99,
                         'close': close, 'volume': np.full(n, 1000.0)}, index=idx)


class TestRatioHelpers(unittest.TestCase):
    def test_roundtrip(self):
        ohlc = _ohlc(5)
        bands = pd.DataFrame({'yhat': [10.5] * 5, 'yhat_lower': [9.0] * 5,
                              'yhat_upper': [11.0] * 5}, index=ohlc.index)
        r = fbp_bands_to_ratio(bands, ohlc['close'])
        back = fbp_bands_from_ratio(r, ohlc['close'])
        np.testing.assert_allclose(back.to_numpy(), bands.to_numpy(), rtol=1e-12)

    def test_ratio_is_one_at_close(self):
        """区间上下界都等于 close ⇒ 比值恒 1（口径的锚点）。"""
        ohlc = _ohlc(3)
        bands = pd.DataFrame({'yhat_lower': ohlc['close'].to_numpy(),
                              'yhat_upper': ohlc['close'].to_numpy()}, index=ohlc.index)
        r = fbp_bands_to_ratio(bands, ohlc['close'])
        np.testing.assert_allclose(r.to_numpy(), 1.0, rtol=1e-12)


class TestReindexByDate(unittest.TestCase):
    def test_daily_values_map_onto_intraday_bars(self):
        daily = _ohlc(3)
        d0, d1 = daily.index.get_level_values(0)[0], daily.index.get_level_values(0)[1]
        intra = pd.MultiIndex.from_tuples(
            [(d0 + pd.Timedelta(hours=10), '600519'),
             (d0 + pd.Timedelta(hours=14), '600519'),
             (d1 + pd.Timedelta(hours=10), '600519')], names=['ts', 'code'])
        s = pd.Series([0.90, 0.95, 1.00], index=daily.index)
        out = _reindex_by_date(s, intra)
        np.testing.assert_allclose(out.to_numpy(), [0.90, 0.90, 0.95])


@unittest.skipUnless(HAS_PROPHET, 'prophet 未安装')
class TestProphetBand(unittest.TestCase):

    def test_cmdstanpy_logging_is_suppressed(self):
        """`Chain [1] start processing` 不许刷屏，且退出后 logger 要还原。"""
        import logging
        before = logging.getLogger('cmdstanpy').level
        with self.assertNoLogs('cmdstanpy', level='INFO'), \
                self.assertNoLogs('prophet', level='INFO'):
            prophet_band(self.ohlc)
        self.assertEqual(logging.getLogger('cmdstanpy').level, before)

    @classmethod
    def setUpClass(cls):
        cls.ohlc = _ohlc(120)
        cls.band = prophet_band(cls.ohlc)

    def test_columns_and_index(self):
        self.assertEqual(list(self.band.columns),
                         ['yhat', 'yhat_lower', 'yhat_upper'])
        self.assertTrue(self.band.index.equals(self.ohlc.index),
                        '输出必须与输入逐行对齐（错了会静默错位）')

    def test_ordering_and_no_nan(self):
        self.assertFalse(self.band.isna().any().any())
        self.assertTrue((self.band['yhat_lower'] <= self.band['yhat']).all())
        self.assertTrue((self.band['yhat'] <= self.band['yhat_upper']).all())

    def test_multi_code_raises(self):
        ohlc = _ohlc(30)
        ohlc.index = pd.MultiIndex.from_arrays(
            [ohlc.index.get_level_values(0), ['600519'] * 15 + ['000001'] * 15],
            names=['ts', 'code'])
        with self.assertRaises(ValueError):
            prophet_band(ohlc)


@unittest.skipUnless(HAS_PROPHET, 'prophet 未安装')
class TestCalcBands(unittest.TestCase):
    def test_writes_four_ratio_columns(self):
        ohlc = _ohlc(120)
        feat = calc_fbprophet_bands(ohlc)
        for col in (FIELD.FBPROPHET_LB, FIELD.FBPROPHET_UB,
                    FIELD.FBPROPHET_LB_MAJOR, FIELD.FBPROPHET_UB_MAJOR):
            self.assertIn(col, feat.columns)
            self.assertFalse(feat[col].isna().any(), col)
        self.assertTrue((feat[FIELD.FBPROPHET_LB] > 0).all())
        self.assertLess(float((feat[FIELD.FBPROPHET_LB]
                               / feat[FIELD.FBPROPHET_UB]).max()), 1.0)

    def test_major_equals_own_when_no_daily_frame(self):
        """`major_ohlc=None` ⇒ 认为 `ohlc` 就是日线，`_MAJOR` 直接沿用（旧树同做法）。"""
        feat = calc_fbprophet_bands(_ohlc(120))
        np.testing.assert_array_equal(feat[FIELD.FBPROPHET_LB_MAJOR].to_numpy(),
                                      feat[FIELD.FBPROPHET_LB].to_numpy())


@unittest.skipUnless(HAS_PROPHET, 'prophet 未安装')
class TestMatchesOldTree(unittest.TestCase):
    """与老树 `fractal/v9.py:calc_fbprophet_band` **逐值对拍**（`DECISIONS.md` D53）。

    ⚠️ 两个前置条件，缺一条这条用例就是"碰巧过"：

    1. **必须固定种子** —— `yhat_lower/upper` 是**抽样**出来的（见
       `fbprophet.UNCERTAINTY_SEED`），不固定的话同一份数据两次跑就能差 0.22。
       老树那侧也要用**同一个** `np.random.seed()` 再调用。
    2. **老模块需要一个补丁**：`GolemQ_old/fractal/v9.py` 模块级**没有
       `import pandas as pd`**（老树靠包内 `import *` 注入），独立载入会 NameError。
       已在本地参考树补了那一行（`GolemQ_old/` 在 `.gitignore` 里，不影响仓库）。
    3. 另有两个**与 LB/UB 无关**的帮手在新树里是替身（`calc_feature_event_timing_lag`
       / `calc_energy` / `calc_event_timing_lag`，只喂 `CHANGE_POINT*`）——
       用零值顶替，否则老函数在赋值 LB/UB **之前**就抛。

    比对口径：老树 `astype(np.float32)` 落库，我们算 float64 ⇒ **转 float32 后逐位相同**。
    """

    @classmethod
    def setUpClass(cls):
        from GolemQ.test_cases.old_tree import load_old, old_tree_available
        if not old_tree_available():
            raise unittest.SkipTest('GolemQ_old/ 不在场')
        import pandas as pd

        # ⚠️ `load_old` 只有在 `GolemQ.analysis.timeseries` **尚未被导入**时才顺利
        # （否则走真模块，而老 v9 从它拿 7 个新树没有的名字 ⇒ ImportError）。
        # 照 `test_quadrant` / `test_renko` 的做法：给**真模块**临时补一个兜底
        # `__getattr__`（绝不替换模块本身），用完删掉。
        import GolemQ.analysis.timeseries as _ts
        cls._ts_patched = not hasattr(_ts, '__getattr__')

        def _fallback(name):
            if name.startswith('__'):
                raise AttributeError(name)
            return lambda *a, **k: None          # 只被 import，不会被真调用

        if cls._ts_patched:
            _ts.__getattr__ = _fallback

        cls.v9 = load_old('fractal/v9.py')
        from GolemQ.core.presentation import suppress_stdout_stderr
        cls.v9.suppress_stdout_stderr = suppress_stdout_stderr
        zero = lambda x, *a, **k: pd.Series(0.0, index=pd.RangeIndex(len(x)))  # noqa: E731
        cls.v9.calc_feature_event_timing_lag = \
            lambda f, c: pd.Series(0.0, index=f.index)                        # noqa: E731
        cls.v9.calc_energy = zero
        cls.v9.calc_event_timing_lag = zero

    @classmethod
    def tearDownClass(cls):
        if getattr(cls, '_ts_patched', False):
            import GolemQ.analysis.timeseries as _ts
            try:
                delattr(_ts, '__getattr__')
            except AttributeError:
                pass

    def test_bands_match_old_implementation(self):
        import numpy as np
        from GolemQ.analysis.fbprophet import UNCERTAINTY_SEED, prophet_band

        n = 150
        ts = pd.bdate_range('2025-01-06', periods=n)
        idx = pd.MultiIndex.from_arrays([ts, ['600519'] * n], names=['date', 'code'])
        close = 10 + np.sin(np.arange(n) / 9.0) * 2 + np.arange(n) * 0.05
        ohlc = pd.DataFrame({'open': close, 'high': close * 1.01, 'low': close * 0.99,
                             'close': close, 'volume': np.full(n, 1000.0)}, index=idx)
        feat = pd.DataFrame(index=idx)
        feat[self.v9.FLD.LINEAREG_PRICE] = close     # 跳过 lineareg 计算

        np.random.seed(UNCERTAINTY_SEED)             # 老树的 predict 也用这一串
        old = self.v9.calc_fbprophet_band(ohlc.copy(), features=feat.copy(), annual=252)
        new = prophet_band(ohlc)

        for col, key in ((self.v9.FLD.FBPROPHET_LB, 'yhat_lower'),
                         (self.v9.FLD.FBPROPHET_UB, 'yhat_upper')):
            with self.subTest(column=key):
                a = old[col].to_numpy()
                b = new[key].to_numpy().astype(np.float32)   # 老树落库就是 float32
                np.testing.assert_array_equal(a, b)


@unittest.skipUnless(HAS_PROPHET, 'prophet 未安装')
class TestScaleBehaviour(unittest.TestCase):
    """⚠️ **比值不是"数学不变"** —— 把上界钉住，防吹。

    2026-10-18 实测（600519 日线，整窗 ×0.5）：比值与绝对价的相对差都是 **0.82%**。
    差的是「除权后要不要重算」，不是存比值的缺陷（不重算 ⇒ 完全跟随）。
    """

    def test_rescale_deviation_is_small_but_not_zero(self):
        ohlc = _ohlc(120)
        r1 = fbp_bands_to_ratio(prophet_band(ohlc), ohlc['close'])
        ohlc2 = ohlc.copy()
        ohlc2['close'] = ohlc2['close'] * 0.5
        r2 = fbp_bands_to_ratio(prophet_band(ohlc2), ohlc2['close'])
        dev = float(np.nanmax(np.abs(r2.to_numpy() - r1.to_numpy())
                              / np.abs(r1.to_numpy())))
        self.assertLess(dev, 0.05, '比值偏差超过 5% —— 口径或数据有问题')
