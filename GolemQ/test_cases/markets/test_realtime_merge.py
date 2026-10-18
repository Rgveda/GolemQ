# coding:utf-8
"""REALTIME 合成读路径（`get_kline_price_*` 的 `realtime=True`）的回归钉子。

为什么单独一个文件
================
2026-10-12 用户实盘验「`get_kline_price/_min` 有没有 realtime 数据」时，
一条链上**连着三个潜伏 bug**，每一个都被某层的 `except` 吞成静默，现象都是
「`realtime=True` 与 `False` 结果一模一样」：

1. `realtime.py` 的 `import time`（模块）把 `datetime.time` **遮住**了 ——
   而 `GQ_data_tick_resample_1min` 里有 **56 处** `time(9, 25)` 切片 ⇒ 一调就
   `TypeError: 'module' object is not callable`，被它那层 `except: pass` 吞掉。
2. `analysis/timeseries.py` 的 `_MIN_CONVERSION_VOL` / `_MIN_CONVERSION_VOLUME`
   **被引用却从未定义**（回迁时漏抄了定义）⇒ `GQ_data_min_resample` 一调就
   `NameError`。
3. 合并代码用了**老树 QUANTAXIS DataStruct 的 `select_code`**，而新树读口返回的
   `KlineResult` 只有 `.data` ⇒ `AttributeError`。

三条都修了；本文件钉住前两条（第 3 条靠 `_select_code` 的单测）。
"""
from __future__ import annotations

import contextlib
import datetime as _dt
import io
import os
import sys

try:
    import GolemQ  # noqa: F401
except ImportError:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

import unittest
import unittest.mock          # `unittest.mock` 不随 `import unittest` 自动可用

import numpy as np
import pandas as pd

from GolemQ.analysis import timeseries as TS
from GolemQ.markets.StockCN import realtime as RT
from GolemQ.markets.StockCN.kline83 import KlineResult


class TestTimeNotShadowed(unittest.TestCase):
    """bug ①：`realtime.py` 里 `time` 必须是 `datetime.time`，不是 time 模块。"""

    def test_time_is_datetime_time(self):
        self.assertIs(RT.time, _dt.time,
                      'realtime.py 的 `time` 被 time 模块遮住了 —— '
                      '`time(9, 25)` 那 56 处切片会全炸（且被 except 吞掉）')

    def test_the_shadowing_module_is_still_reachable_under_its_aliases(self):
        """遮住的那个 time **模块**仍要能拿到（`timer.sleep` / `ts.time` 在用）。"""
        self.assertTrue(hasattr(RT.timer, 'sleep'))
        self.assertTrue(hasattr(RT.ts, 'time'))


class TestMinResampleConstants(unittest.TestCase):
    """bug ②：`_MIN_CONVERSION_*` 必须存在，否则两个重采样器全是死的。"""

    def test_constants_defined_and_differ_only_in_volume_column(self):
        self.assertIn('vol', TS._MIN_CONVERSION_VOL)
        self.assertIn('volume', TS._MIN_CONVERSION_VOLUME)
        self.assertNotIn('volume', TS._MIN_CONVERSION_VOL)
        # 键序即输出的列集合（`list(CONVERSION.keys())` 去裁剪），故逐字比
        self.assertEqual(list(TS._MIN_CONVERSION_VOL),
                         ['code', 'open', 'high', 'low', 'close', 'vol', 'amount'])

    def test_min_resample_runs(self):
        """真跑一次 —— 修之前这里 `NameError`。

        ⚠️ 入参形状要照**真实调用方**（`realtime.py`）：`index` 是 **DatetimeIndex**
        （名字 `datetime`），`code` 落成**列** —— 函数内部用
        `idx.indexer_between_time(...)`，那只有 DatetimeIndex 才有。
        """
        idx = pd.date_range('2026-10-09 09:31', periods=60, freq='1min')
        idx = idx[(idx.time >= _dt.time(9, 31)) & (idx.time <= _dt.time(11, 30))]
        idx = idx.rename('datetime')
        n = len(idx)
        df = pd.DataFrame({
            'code': ['600519'] * n,
            'open': np.arange(n, dtype=float),
            'high': np.arange(n, dtype=float) + 1,
            'low': np.arange(n, dtype=float) - 1,
            'close': np.arange(n, dtype=float),
            'vol': np.ones(n),
            'amount': np.ones(n),
        }, index=idx)
        out = TS.GQ_data_min_resample(df, type_='5min')
        self.assertGreater(len(out), 0)
        self.assertEqual(list(out.index.names), ['datetime', 'code'])


class TestSelectCodeHelper(unittest.TestCase):
    """bug ③：`_select_code` 只认 `.data`（`KlineResult` 没有 `select_code`）。"""

    def _frame(self):
        idx = pd.MultiIndex.from_product(
            [pd.date_range('2026-10-09', periods=3, freq='D'), ['600519', '000001']],
            names=['datetime', 'code'])
        return pd.DataFrame({'close': np.arange(6.0)}, index=idx)

    def test_selects_by_code_on_a_plain_dataframe(self):
        sel = RT._select_code(self._frame(), '600519')
        self.assertEqual(len(sel), 3)
        self.assertTrue((sel.index.get_level_values('code') == '600519').all())

    def test_accepts_an_object_with_data_attr(self):
        """`KlineResult` 只有 `.data` —— 这正是老代码 `AttributeError` 的来由。"""
        class _R:
            def __init__(self, d):
                self.data = d
        sel = RT._select_code(_R(self._frame()), '000001')
        self.assertEqual(len(sel), 3)

    def test_single_level_index_is_returned_as_is(self):
        df = pd.DataFrame({'close': [1.0, 2.0]},
                          index=pd.date_range('2026-10-09', periods=2))
        self.assertIs(RT._select_code(df, '600519'), df)


class _FrozenDT(_dt.datetime):
    """把 `realtime.py` 里的 `dt.now()` 钉死 —— 否则本类**只在开盘时段能过**。

    这是 ② 那条教训的同一个病根（`test_kline_save` 曾拿真实时钟判「盘中×分钟」）。
    用法：`unittest.mock.patch.object(RT, 'dt', _FrozenDT)` —— `dt.strptime` 等
    类方法照常（继承自 `datetime`）。
    """
    NOW = _dt.datetime(2026, 10, 12, 10, 30)      # 盘中

    @classmethod
    def now(cls, tz=None):
        return cls.NOW


class TestMinRealtimeMergeBody(unittest.TestCase):
    """`GQ_fetch_stock_min_realtime_adv` 的**函数体**：tick → 1min → 合并。

    为什么必须有这一类
    ================
    `test_realtime_read.py` 只断言「合并器**被调用**」—— 它把 `GQ_fetch_stock_min_
    realtime_adv` **整个 mock 掉了**；于是下面这条链的**体内**一行都没测过：

        tick 帧 → GQ_data_tick_resample_1min → GQ_data_min_resample → concat 进历史

    而链上有**三层 `except`** 层层吞异常 ⇒ 三处 bug（见模块 docstring）能藏很久，
    只有**开盘时段手工验**才暴露。本类把 **tick 与「现在」都注入**，于是：

    * 任何时刻都能跑（不依赖开不开盘）；
    * 链条哪一环断了会**当场红**，而不是静默返回纯历史。

    ⚠️ 断言要**正向**：「历史里多出了注入 tick 合成的那些 bar」，
    不能只断言「没报错」—— 返回纯历史时也不会报错。
    """

    def _history(self):
        """合并前的历史：止于 **2026-10-09 15:00**（比注入的「最近交易日」落后 >10h）。"""
        dts = pd.date_range('2026-10-09 09:31', '2026-10-09 15:00', freq='1min')
        n = len(dts)
        idx = pd.MultiIndex.from_arrays([dts, ['600519'] * n], names=['datetime', 'code'])
        return KlineResult(pd.DataFrame({
            'code': '600519',
            'open': np.full(n, 1260.0), 'high': np.full(n, 1261.0),
            'low': np.full(n, 1259.0), 'close': np.full(n, 1260.5),
            'vol': np.full(n, 1.0), 'amount': np.full(n, 1000.0),
        }, index=idx))

    def _ticks(self):
        """合成的 tick：2026-10-12 的 09:31–09:40，每分钟一笔（**形态照真 L1 行**）。"""
        dts = pd.date_range('2026-10-12 09:31', periods=10, freq='1min')
        return pd.DataFrame({
            'datetime': dts, 'code': '600519',
            'price': 1270.0 + np.arange(10, dtype=float),
            'vol': np.full(10, 100.0), 'amount': np.full(10, 127000.0),
        })

    def _merge(self, history, ticks):
        with unittest.mock.patch.object(RT, 'GQ_fetch_stock_realtime_adv',
                                        return_value=ticks), \
                unittest.mock.patch.object(RT, 'GQ_util_get_last_day',
                                           return_value=_FrozenDT.NOW.replace(hour=0,
                                                                              minute=0,
                                                                              second=0)), \
                unittest.mock.patch.object(RT, 'dt', _FrozenDT), \
                contextlib.redirect_stdout(io.StringIO()):
            return RT.GQ_fetch_stock_min_realtime_adv(['600519'], history, '1min',
                                                      verbose=False)

    def test_injected_ticks_land_in_the_result(self):
        hist = self._history()
        before = len(hist.data)
        out = self._merge(hist, self._ticks())
        after = len(out.data)
        self.assertGreater(after, before,
                           '合并没生效（返回了纯历史）—— 链条某环又断了')
        days = set(pd.to_datetime(out.data.index.get_level_values(0)).date)
        self.assertIn(_dt.date(2026, 10, 12), days,
                      '注入 tick 的那一天没进结果（末根 %s）'
                      % out.data.index.get_level_values(0).max())

    def test_history_rows_are_preserved(self):
        """合并是**追加**，不是替换 —— 老 bar 一根都不许丢。"""
        hist = self._history()
        out = self._merge(hist, self._ticks())
        old = out.data.loc[(slice(None), '600519'), :]
        old = old[pd.to_datetime(old.index.get_level_values(0)).date == _dt.date(2026, 10, 9)]
        self.assertGreater(len(old), 0, '合并把历史那天的 bar 弄丢了')

    def test_no_ticks_means_pure_history(self):
        """tick 为空 ⇒ 返回原历史（这条**反向**钉住「不是无脑 append」）。"""
        hist = self._history()
        before = len(hist.data)
        out = self._merge(hist, self._ticks().iloc[0:0])
        self.assertEqual(len(out.data), before)

    def test_frozen_clock_really_freezes(self):
        """自检：`_FrozenDT` 若失效，上面几条会退回「只在开盘时段过」—— 钉住它。"""
        with unittest.mock.patch.object(RT, 'dt', _FrozenDT):
            self.assertEqual(RT.dt.now(), _FrozenDT.NOW)


if __name__ == '__main__':
    unittest.main()
