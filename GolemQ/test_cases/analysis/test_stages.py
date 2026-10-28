# coding:utf-8
"""`analysis/stages.py`（Garident STAGE）与老树的**逐值对拍**。

判据：`np.array_equal(..., equal_nan=True)` —— **不用容差**（`DECISIONS.md` D35/D36）。
老树 `GolemQ_old/` 不在场时，对拍用例**整份跳过**（同 `test_timing.py` 惯例）。

⚠️ 这一族有个**会让朴素对拍假红**的性质：`calc_garident_price` 里的
`KMeans` **没设 `random_state`**（老树如此，本树照旧 —— 设了会改变结果、破坏对拍）。
`random_state=None` 时 sklearn 走的是 **numpy 全局 RNG**，所以：

    对拍前必须把两边的全局种子**对齐**（`np.random.seed(N)` → 调一遍 → 再来一遍）。

不这么做的话，同一个输入两次调用本来就会得到不同的簇 ⇒ 用例会**随机红**，
而那种红**看起来像算法搬错了**。老树就有的性质，不是本次引入的。

载入老模块的 harness 比 `test_quadrant.py` 那套轻得多 —— 老 `analysis/stages.py`
只 import 三个 `GolemQ.*` 名字。但它有**一处自己的坑**（见 `_load_old_stages` 的说明）。
"""
from __future__ import annotations

import os
import sys

try:
    import GolemQ  # noqa: F401
except ImportError:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

import unittest

import numpy as np
import pandas as pd

from GolemQ.analysis import stages as NEW
from GolemQ.analysis.stages import (
    calc_garident_price, calc_garident_stage,
    garident_bands_from_ratio, garident_bands_to_ratio,
)
from GolemQ.analysis.timing import Timeline_duration
from GolemQ.core import constants as _C
from GolemQ.test_cases.old_tree import field_compat, load_old



#: 老模块的 harness 抽在 `old_tree.py`（三处对拍共用，别在这里再写一遍）。
#: ⚠️ `GolemQ.analysis.timeseries` 用 **`patch_attrs` 而不是 `override`**：
#: 老 `stages.py` 从它 import `Timeline_duration`，那个名字新树里**有真实现**
#: （在 `timing.py`）—— 顶替成 no-op 的话老代码 `.astype(None)` 当场炸，
#: 报错点还离原因很远（本用例踩过）。补属性才是对的。
#: 顺带这也**顺带验证了**新树 `Timeline_duration` 与老树那份在这些输入上一致。
_OLD = load_old('analysis/stages.py', patch_attrs={
    'GolemQ.analysis.timeseries': {'Timeline_duration': Timeline_duration},
})

#: ⚠️ 老 `stages.py` 的 `try` 分支 `from GolemQ.utils.parameter import (AKA, STATE,
#: TREND_STATUS)` —— **漏了 `INDICATOR_FIELD as FLD`**（只有 `except` 兜底那支才
#: import 它）。于是 import 顺利通过，但一调 `calc_garident_stages` 就 `NameError: FLD`。
#: 这正解释了它为什么在老树里**全树无人 import**（死文件）—— 真正被调用的是
#: `GolemQ_old/fractal/vXII.py` 里那份（算法逐字相同，见 NEW 的模块 docstring）。
if _OLD is not None:
    # 老模块的 `FLD` 换成**兼容代理**（新树优先、老表兜底）——
    # 只用新表会在老模块**模块体**里 AttributeError；只用老表则老函数写出老列名
    # 而用例读新列名 ⇒ KeyError。两个方向都实测踩过。
    _OLD.FLD = field_compat()


def _prices(n=560, seed=11):
    rng = np.random.default_rng(seed)
    return np.cumsum(rng.normal(0, 0.6, n)) + 100.0


def _with_seed(seed, fn):
    """把 numpy 全局种子对齐后调用 —— 见模块 docstring 的 ⚠️。"""
    np.random.seed(seed)
    return fn()


@unittest.skipIf(_OLD is None, '老树 GolemQ_old/ 不在场，对拍整份跳过')
class TestGaridentPriceMatchesOld(unittest.TestCase):
    """K-Means 定带：带数与每一对 `[上沿, 下沿]` 都要逐值相同。"""

    def test_same_seed_same_bands(self):
        px = _prices()
        for seed in (0, 7, 20261016):
            with self.subTest(seed=seed):
                old = _with_seed(seed, lambda: _OLD.calc_garident_price(px))
                new = _with_seed(seed, lambda: NEW.calc_garident_price(px))
                self.assertEqual(len(new), len(old))
                self.assertTrue(np.array_equal(np.array(new, dtype=float),
                                               np.array(old, dtype=float),
                                               equal_nan=True))

    def test_short_series(self):
        px = _prices(40, seed=3)
        with self.subTest('短序列'):
            old = _with_seed(5, lambda: _OLD.calc_garident_price(px))
            new = _with_seed(5, lambda: NEW.calc_garident_price(px))
            self.assertTrue(np.array_equal(np.array(new, dtype=float),
                                           np.array(old, dtype=float),
                                           equal_nan=True))


@unittest.skipIf(_OLD is None, '老树 GolemQ_old/ 不在场，对拍整份跳过')
class TestGaridentStageMatchesOld(unittest.TestCase):
    """落带判定是**纯函数**（无随机），可以逐值硬比。"""

    def test_matrix_of_positions(self):
        bands = [[100.0, 90.0], [110.0, 101.0], [125.0, 111.0]]
        probes = [89.0, 90.0, 95.0, 100.0, 101.0, 110.0, 111.0, 125.0, 130.0]
        for p in probes:
            with self.subTest(close=p):
                self.assertEqual(NEW.calc_garident_stage(p, bands),
                                 _OLD.calc_garident_stage(p, bands))
        # ⚠️ `NaN` 单独比 —— `assertEqual(nan, nan)` 会失败（`nan != nan`）
        with self.subTest(close='nan'):
            self.assertTrue(np.isnan(NEW.calc_garident_stage(np.nan, bands)))
            self.assertTrue(np.isnan(_OLD.calc_garident_stage(np.nan, bands)))

    def test_tuple_input_is_accepted(self):
        bands = ((100.0, 90.0), (110.0, 101.0))
        self.assertEqual(NEW.calc_garident_stage(95.0, bands),
                         _OLD.calc_garident_stage(95.0, bands))

    def test_float_input_returns_nan(self):
        """老树把某一根的 `GAR_P` 存成了**标量** —— 这条分支必须一致（返回 `NaN`）。"""
        self.assertTrue(np.isnan(NEW.calc_garident_stage(95.0, float('nan'))))
        self.assertTrue(np.isnan(_OLD.calc_garident_stage(95.0, float('nan'))))


@unittest.skipIf(_OLD is None, '老树 GolemQ_old/ 不在场，对拍整份跳过')
class TestGaridentStagesMatchesOld(unittest.TestCase):
    """整条编排：8 列逐值相同。"""

    def _frames(self):
        n = 120
        idx = pd.MultiIndex.from_product(
            [pd.date_range('2024-01-01', periods=n, freq='D'), ['600519']],
            names=['datetime', 'code'])
        rng = np.random.default_rng(42)
        close = np.cumsum(rng.normal(0, 0.8, n)) + 1500.0
        feats = pd.DataFrame({'close': close,
                              _C.TREND_STATUS.CLUSTER_GROUP_CHECKPOINTS:
                                  rng.normal(0, 1, n)}, index=idx)
        # 每 10 根一个 checkpoint（老树由 `CLUSTER_GROUP_CHECKPOINTS` 决定，这里造样例）
        checkpoints = feats.iloc[::10]
        return feats, checkpoints

    #: 数值列 —— 逐值硬比。
    #: ⚠️ 老树还有 `STOCK_TREND_CHECKOUT_CLOSE` / `RETURNS` / `BEFORE` /
    #: `RETURNS_GARIDENT` 四列，**本树整条删掉了**（老树从没落库、新树零消费，
    #: 见 `docs/decisions/D49.md` 与 `calc_garident_stages` 的 docstring）——
    #: 所以对拍只覆盖**活下来的**这几列。
    NUMERIC_COLS = (_C.AKA.STAGE,
                    _C.FIELD.GARIDENT_PRICE_COUNT,
                    _C.FIELD.GARIDENT_PRICE_CLEARANCE)

    def test_all_columns_equal(self):
        feats_o, ck_o = self._frames()
        feats_n, ck_n = self._frames()
        old = _with_seed(13, lambda: _OLD.calc_garident_stages(ck_o, feats_o))
        new = _with_seed(13, lambda: NEW.calc_garident_stages(ck_n, feats_n))

        # 老树常量表里 `AKA.STAGE='STAGE'`，新树已改为伪驼峰 `stage` ——
        # 对拍时按老树那一侧的实际列名取。
        old_names = {'stage': 'STAGE'}
        for col in self.NUMERIC_COLS:
            with self.subTest(column=col):
                o = pd.to_numeric(old[old_names.get(col, col)],
                                  errors='coerce').to_numpy(dtype=float)
                n = pd.to_numeric(new[col], errors='coerce').to_numpy(dtype=float)
                self.assertEqual(len(n), len(o))
                self.assertTrue(np.array_equal(n, o, equal_nan=True),
                                f'{col} 与老树不同')

    def test_garident_price_column_equal(self):
        """`GAR_P` 那列装的是**带本身**（tuple 或 NaN），不能走 `to_numeric` —— 单独比。"""
        feats_o, ck_o = self._frames()
        feats_n, ck_n = self._frames()
        old = _with_seed(13, lambda: _OLD.calc_garident_stages(ck_o, feats_o))
        new = _with_seed(13, lambda: NEW.calc_garident_stages(ck_n, feats_n))

        col = _C.FIELD.GARIDENT_PRICE
        for i, (o, n) in enumerate(zip(old[col], new[col])):
            with self.subTest(row=i):
                if isinstance(o, float) and np.isnan(o):
                    self.assertTrue(isinstance(n, float) and np.isnan(n))
                else:
                    self.assertTrue(np.array_equal(np.array(n, dtype=float),
                                                   np.array(o, dtype=float),
                                                   equal_nan=True))


class TestStructure(unittest.TestCase):
    """不依赖老树的结构性钉子。"""

    @staticmethod
    def _frame(n=60, code='000001'):
        idx = pd.MultiIndex.from_product(
            [pd.date_range('2024-01-01', periods=n, freq='D'), [code]],
            names=['datetime', 'code'])
        rng = np.random.default_rng(1)
        return pd.DataFrame(
            {'close': np.cumsum(rng.normal(0, 0.5, n)) + 20.0,
             _C.TREND_STATUS.CLUSTER_GROUP_CHECKPOINTS: rng.normal(0, 1, n)},
            index=idx)

    def test_existing_stage_is_not_overwritten(self):
        """`STAGE` **只在原本是 `NaN` 时才写**（`np.where(np.isnan(...))`）。

        ⚠️ 别把这条改成"重复调用结果相同" —— 那个断言**测不出**这里的语义：
        `calc_garident_stages` **就地改入参**，第二次调用拿到的是已被填好的帧。
        """
        feats = self._frame()
        feats[_C.AKA.STAGE] = np.arange(len(feats), dtype=float)   # 预置非 NaN
        before = feats[_C.AKA.STAGE].to_numpy(dtype=float)
        NEW.calc_garident_stages(feats.iloc[::10], feats)
        after = feats[_C.AKA.STAGE].to_numpy(dtype=float)
        self.assertTrue(np.array_equal(before, after, equal_nan=True))

    def test_skipped_checkpoints_do_not_raise(self):
        """定带失败 ⇒ 跳过该 checkpoint，**不抛**（老树行为）。"""
        n = 30
        idx = pd.MultiIndex.from_product(
            [pd.date_range('2024-01-01', periods=n, freq='D'), ['000001']],
            names=['datetime', 'code'])
        feats = pd.DataFrame({'close': np.full(n, np.nan),
                              _C.TREND_STATUS.CLUSTER_GROUP_CHECKPOINTS: np.zeros(n)},
                             index=idx)
        ck = feats.iloc[::10]
        lines = []
        NEW.calc_garident_stages(ck, feats, verbose=True, echo=lines.append)
        self.assertTrue(any('定带失败' in ln for ln in lines), lines)


class TestDroppedColumnsStayDropped(unittest.TestCase):
    """老树那条 `STOCK_TREND_*` 链**必须是删掉的**（2026-10-16 用户定）。

    钉子在这里的理由：这四列在老树里看起来"像回事"，很容易被下一个接手的人
    照着老树加回来 —— 而它们**老树从没落过库、新树零消费**。
    """

    DROPPED = ('STOCK_TREND_CHECKOUT_CLOSE', 'STOCK_TREND_RETURNS',
               'STOCK_TREND_BEFORE', 'STOCK_TREND_RETURNS_GARIDENT')

    def test_constants_are_gone(self):
        for name in self.DROPPED:
            with self.subTest(constant=name):
                with self.assertRaises(AttributeError):
                    getattr(_C.FIELD, name)

    def test_not_in_output(self):
        n = 60
        idx = pd.MultiIndex.from_product(
            [pd.date_range('2024-01-01', periods=n, freq='D'), ['600519']],
            names=['datetime', 'code'])
        rng = np.random.default_rng(5)
        feats = pd.DataFrame(
            {'close': np.cumsum(rng.normal(0, 0.5, n)) + 100.0,
             _C.TREND_STATUS.CLUSTER_GROUP_CHECKPOINTS: rng.normal(0, 1, n)},
            index=idx)
        out = NEW.calc_garident_stages(feats.iloc[::10], feats)
        for name in self.DROPPED:
            with self.subTest(column=name):
                self.assertNotIn(getattr(_C.FIELD, name, name), out.columns)

    def test_surviving_columns_are_present(self):
        """正面钉子：活下来的几列一个都不能少（尤其新加的锚日期）。"""
        n = 60
        idx = pd.MultiIndex.from_product(
            [pd.date_range('2024-01-01', periods=n, freq='D'), ['600519']],
            names=['datetime', 'code'])
        rng = np.random.default_rng(5)
        feats = pd.DataFrame(
            {'close': np.cumsum(rng.normal(0, 0.5, n)) + 100.0,
             _C.TREND_STATUS.CLUSTER_GROUP_CHECKPOINTS: rng.normal(0, 1, n)},
            index=idx)
        out = NEW.calc_garident_stages(feats.iloc[::10], feats)
        for col in (_C.AKA.STAGE, _C.FIELD.GARIDENT_PRICE,
                    _C.FIELD.GARIDENT_PRICE_COUNT, _C.FIELD.GARIDENT_PRICE_CLEARANCE,
                    _C.FIELD.GARIDENT_PRICE_ANCHOR):
            with self.subTest(column=col):
                self.assertIn(col, out.columns)


class TestRatioForm(unittest.TestCase):
    """`GAR_P` 落**比值**（`docs/decisions/D49.md`）—— 不变量与陷阱都钉住。

    背景：绝对价位一旦落库就**跟着复权过期**（前复权因子每次除权都重标整条历史，
    而库里那串绝对价不动）。比值不会过期，因为算法**尺度等变**、
    而未来除权对锚 bar 及整窗是**均匀**缩放。
    """

    @staticmethod
    def _px(n=520, seed=7):
        rng = np.random.default_rng(seed)
        return np.cumsum(rng.normal(0, 0.5, n)) + 100.0

    def test_alg_is_scale_equivariant(self):
        """**核心不变量**：整窗乘常数 k ⇒ 带等比例缩放（实测精确，不是近似）。"""
        px = self._px()
        np.random.seed(2026)
        base = np.array(calc_garident_price(px), dtype=float)   # noqa: F821
        for k in (0.37, 2.5, 100.0):
            with self.subTest(k=k):
                np.random.seed(2026)
                got = np.array(calc_garident_price(px * k), dtype=float)  # noqa: F821
                self.assertEqual(len(got), len(base), '带数必须相同')
                self.assertTrue(np.array_equal(got, k * base))

    def test_ratio_is_invariant_to_future_event(self):
        """未来除权 = 锚及整窗**均匀**缩放 ⇒ 比值不变（**到机器精度**，非逐位）。

        ⚠️ 别把判据写成 `array_equal` —— `(b*k)/(p*k)` 与 `b/p` 在浮点下**不逐位相等**
        （实测最大相对差 ~1e-16）。这里要的是"**不存在系统性漂移**"，不是位相等。
        """
        px = self._px()
        np.random.seed(5)
        b_pre = np.array(calc_garident_price(px), dtype=float)          # noqa: F821
        np.random.seed(5)
        b_post = np.array(calc_garident_price(px * 0.91), dtype=float)  # noqa: F821
        r_pre = garident_bands_to_ratio(b_pre, px[-1])                  # noqa: F821
        r_post = garident_bands_to_ratio(b_post, px[-1] * 0.91)         # noqa: F821
        self.assertTrue(np.allclose(np.array(r_pre, dtype=float),
                                    np.array(r_post, dtype=float),
                                    rtol=1e-12, atol=0))

    def test_round_trip_is_lossless_enough(self):
        """绝对 ↔ 比值往返**无系统性损失**（乘回**锚** close）。

        ⚠️ 断言**不能**写成 `assertEqual`：`x/(p)*p` 在浮点下未必退回 `x`
        （如 `110.0/108.25*108.25 = 110.00000000000001`）。判据是相对误差 ~1e-16。
        """
        bands = [(110.0, 100.0), (125.0, 111.0), (140.5, 126.0)]
        r = garident_bands_to_ratio(bands, 108.25)                      # noqa: F821
        back = garident_bands_from_ratio(r, 108.25)                     # noqa: F821
        self.assertTrue(np.allclose(np.array(back, dtype=float),
                                    np.array(bands, dtype=float),
                                    rtol=1e-12, atol=0))

    def test_missing_bands_stay_missing(self):
        for bad in (None, float('nan'), []):
            with self.subTest(bad=bad):
                self.assertIsNone(garident_bands_to_ratio(bad, 100.0))  # noqa: F821
                self.assertIsNone(garident_bands_from_ratio(bad, 100.0))  # noqa: F821

    def test_stage_survives_rebasing_via_anchor(self):
        """复权后 STAGE **不变** —— 前提是乘**锚** close 还原。"""
        px = self._px()
        np.random.seed(9)
        bands = np.array(calc_garident_price(px), dtype=float)          # noqa: F821
        np.random.seed(9)
        bands_rebased = np.array(calc_garident_price(px * 0.83), dtype=float)  # noqa: F821

        ratio = garident_bands_to_ratio(bands, px[-1])                  # noqa: F821
        # 未来事件 ⇒ 锚 close 也随之变成 px[-1]*0.83；还原出的带应与重算的**等价**
        restored = garident_bands_from_ratio(ratio, px[-1] * 0.83)      # noqa: F821
        self.assertTrue(np.allclose(np.array(restored, dtype=float),
                                    bands_rebased, rtol=1e-12, atol=0))

    def test_multiplying_by_current_close_is_the_trap(self):
        """⚠️ **反面钉子**：乘**当前** close 还原 ⇒ `STAGE` 塌成常数。

        这条不是"另一种实现"，是**错的** —— `GAR_P` 是「最近 checkpoint 的绝对
        价位水平」，被 ffill 往后带；用当前价缩放等于让带跟着今天的价走。
        """
        px = self._px()
        np.random.seed(11)
        bands = np.array(calc_garident_price(px), dtype=float)          # noqa: F821
        ratio = np.array(garident_bands_to_ratio(bands, px[-1]), dtype=float)  # noqa: F821

        right = [calc_garident_stage(px[t], bands) for t in (-1, -60, -300)]  # noqa: F821
        wrong = [calc_garident_stage(px[t], ratio * px[t])              # noqa: F821
                 for t in (-1, -60, -300)]
        self.assertEqual(len(set(wrong)), 1, '乘当前 close ⇒ 全部塌成同一个值')
        self.assertGreater(len(set(right)), 1, '正确做法下 STAGE 本来就该变化')


if __name__ == '__main__':
    unittest.main()
