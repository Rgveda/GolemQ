#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""缠论中枢（盘整箱体）的回归用例。

覆盖四件事：

1. **识别** —— 合成「上涨 → 箱体震荡 → 下跌」序列，应识别出单一中枢，且箱体
   落在震荡区；
2. **分类** —— 手工构造中枢序列，逐条钉住 盘整 / 上涨趋势 / 下跌趋势 / 无中枢；
3. **接口** —— `bi_list` 与 `points` 两种输入必须给出相同结果；
4. **无未来函数** —— 把数据截断到历史某点重算，**已经走完**的中枢其
   `ZG/ZD/direction` 不得改变。

第 4 条是这份用例存在的主要理由：未来函数不会报错，只会让回测「好看」。
口径的实现见 `GolemQ/analysis/pivot.py` 的 `causal_pivot_series` 与
`calc_pivots` 里的窗口修正。

真数据用例（8.3）在没有库时**跳过而不是失败** —— 它是端到端确认，不是单测前提。
"""

import os
import sys

try:
    import GolemQ  # noqa: F401
except ImportError:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import unittest
from datetime import datetime, timedelta

import numpy as np

from czsc import CZSC
from czsc.objects import Freq, RawBar

from GolemQ.analysis.pivot import (
    _mark_to_str, bi_list_to_points, calc_pivots, classify_pivots, pivots_to_df,
)


# --------------------------------------------------------------------------- #
# 夹具
# --------------------------------------------------------------------------- #
def _make_bars(price):
    """由收盘价序列合成 60 分钟 RawBar（简单 OHLC）。"""
    base = datetime(2024, 1, 2, 9, 30)
    bars = []
    for i, c in enumerate(price):
        dt = base + timedelta(hours=i)
        o = price[i - 1] if i > 0 else c
        h = max(o, c) + 0.05
        low = min(o, c) - 0.05
        bars.append(RawBar(symbol='test', dt=dt, id=i, freq=Freq.F60,
                           open=o, close=c, high=h, low=low, vol=1000, amount=10000))
    return bars


def _synthetic_consolidation(seed=0):
    """上升 → 箱体震荡 → 下降（盘整场景，单一中枢）。"""
    rng = np.random.RandomState(seed)
    p, price = 10.0, []
    for _ in range(60):
        p += 0.1 + rng.randn() * 0.02
        price.append(p)
    for i in range(80):
        p = 16.5 + np.sin(i / 6) * 0.3 + rng.randn() * 0.05
        price.append(p)
    p = 16.5
    for _ in range(60):
        p -= 0.1 + rng.randn() * 0.02
        price.append(p)
    return price


def _zs(direction, ZD, ZG, GG, DD):
    """手工构造一个中枢 dict（分类逻辑只读这几个字段）。"""
    return {'ZD': ZD, 'ZG': ZG, 'GG': GG, 'DD': DD, 'direction': direction,
            'start_dt': None, 'end_dt': None, 'points': [], 'zn': []}


# --------------------------------------------------------------------------- #
# 1. 识别
# --------------------------------------------------------------------------- #
class TestPivotConsolidation(unittest.TestCase):
    def test_single_consolidation_is_found(self):
        bars = _make_bars(_synthetic_consolidation())
        czsc = CZSC(bars, max_bi_count=126, verbose=False)
        pivots = calc_pivots(bi_list=czsc.bi_list)

        self.assertGreaterEqual(len(pivots), 1, '盘整场景应至少识别出一个中枢')
        z = pivots[0]
        # 箱体应大致落在震荡中心 16.5 附近
        self.assertTrue(15.5 < z['ZD'] < 17.0, f'箱底异常: {z["ZD"]}')
        self.assertTrue(15.5 < z['ZG'] < 17.0, f'箱顶异常: {z["ZG"]}')
        # 边界关系：ZD < ZG <= GG 且 DD <= ZD
        self.assertLess(z['ZD'], z['ZG'])
        self.assertLessEqual(z['ZG'], z['GG'])
        self.assertLessEqual(z['DD'], z['ZD'])
        self.assertEqual(classify_pivots(pivots)['kind'], '盘整')

    def test_empty_and_tiny_input_do_not_raise(self):
        """笔数不足成枢时返回空列表 —— 空结果不是错误。"""
        self.assertEqual(calc_pivots(points=[]), [])
        self.assertEqual(calc_pivots(points=[{'dt': 0, 'fx_mark': 'd', 'bi': 1.0}]), [])

    def test_requires_one_of_the_two_inputs(self):
        with self.assertRaises(ValueError):
            calc_pivots()


# --------------------------------------------------------------------------- #
# 2. 分类
# --------------------------------------------------------------------------- #
class TestPivotClassify(unittest.TestCase):
    CASES = (
        ('上涨趋势', [_zs('up', 10, 11, 11.5, 9.5), _zs('up', 14, 15, 15.5, 13.5)]),
        ('下跌趋势', [_zs('down', 14, 15, 15.5, 13.5), _zs('down', 10, 11, 11.5, 9.5)]),
        ('盘整', [_zs('up', 10, 12, 12.5, 9.5), _zs('up', 11, 13, 13.5, 10.5)]),
        ('盘整', [_zs('up', 10, 11, 11.5, 9.5)]),
        ('无中枢', []),
    )

    def test_kind_per_case(self):
        for expect, pivots in self.CASES:
            with self.subTest(expect=expect, n=len(pivots)):
                self.assertEqual(classify_pivots(pivots)['kind'], expect)

    def test_per_pivot_kinds_are_one_way(self):
        """逐中枢标签只能 盘整 → 趋势，不来回翻转。"""
        pivots = [_zs('up', 10, 11, 11.5, 9.5), _zs('up', 14, 15, 15.5, 13.5)]
        kinds = classify_pivots(pivots)['kinds']
        self.assertEqual(len(kinds), len(pivots))
        self.assertEqual(kinds[-1], '上涨趋势')
        self.assertNotEqual(kinds[0], '下跌趋势')

    def test_pivots_to_df_takes_per_row_labels(self):
        pivots = [_zs('up', 10, 11, 11.5, 9.5), _zs('down', 13, 14, 14.5, 12.5)]
        df = pivots_to_df(pivots, kind=classify_pivots(pivots)['kinds'])
        self.assertEqual(len(df), 2)
        self.assertEqual(list(df['kind']), classify_pivots(pivots)['kinds'])
        self.assertAlmostEqual(df['amplitude'].iloc[0], 1.0)
        # 传字符串 ⇒ 全行同一标签（兼容旧行为）
        self.assertEqual(set(pivots_to_df(pivots, kind='盘整')['kind']), {'盘整'})


# --------------------------------------------------------------------------- #
# 3. 接口
# --------------------------------------------------------------------------- #
class TestPivotInterface(unittest.TestCase):
    def setUp(self):
        self.bars = _make_bars(_synthetic_consolidation())
        self.czsc = CZSC(self.bars, max_bi_count=126, verbose=False)

    def test_points_count_is_bi_count_plus_one(self):
        pts = bi_list_to_points(self.czsc.bi_list)
        self.assertEqual(len(pts), len(self.czsc.bi_list) + 1)
        self.assertTrue(all(p['fx_mark'] in ('d', 'g') for p in pts))

    def test_bi_list_and_points_agree(self):
        pts = bi_list_to_points(self.czsc.bi_list)
        p1 = calc_pivots(bi_list=self.czsc.bi_list)
        p2 = calc_pivots(points=pts)
        self.assertEqual([z['ZD'] for z in p1], [z['ZD'] for z in p2])
        self.assertEqual([z['ZG'] for z in p1], [z['ZG'] for z in p2])
        self.assertEqual([z['direction'] for z in p1], [z['direction'] for z in p2])

    def test_mark_normalisation_covers_czsc_enum(self):
        """czsc 给的是 `Mark.G` / `Mark.D`，归一化后必须是 'g' / 'd'。"""
        for bi in self.czsc.bi_list:
            self.assertIn(_mark_to_str(bi.fx_a.mark), ('d', 'g'))
            self.assertIn(_mark_to_str(bi.fx_b.mark), ('d', 'g'))


# --------------------------------------------------------------------------- #
# 4. 无未来函数
# --------------------------------------------------------------------------- #
class TestPivotNoLookahead(unittest.TestCase):
    """截断历史重算，**已走完**的中枢不得改变。

    不变量：`ZG` / `ZD` / `direction` / `start_dt` 由已完成的笔决定 ——
    未来数据不能改写它们。会随未来变化的（`GG` / `DD` / `end_dt` / 走势分类）
    刻意**不**断言：它们本来就该继续延伸。
    """

    @classmethod
    def setUpClass(cls):
        cls.bars = _make_bars(_synthetic_consolidation(seed=7))
        cls.full = calc_pivots(bi_list=CZSC(cls.bars, max_bi_count=126, verbose=False).bi_list)

    def test_closed_pivots_survive_truncation(self):
        self.assertGreaterEqual(len(self.full), 1, '夹具本身要能识别出中枢')
        checked = 0
        for frac in (0.75, 0.85, 0.95):
            cut = self.bars[int(len(self.bars) * frac)].dt
            cut_bars = [b for b in self.bars if b.dt <= cut]
            got = calc_pivots(
                bi_list=CZSC(cut_bars, max_bi_count=126, verbose=False).bi_list)
            for want in self.full:
                # 只查**在截断点之前就已结束**的中枢
                if want['end_dt'] is None or want['end_dt'] > cut:
                    continue
                same = [g for g in got if g['start_dt'] == want['start_dt']]
                self.assertTrue(same, f'截断到 {cut:%Y-%m-%d} 后中枢消失了: {want["start_dt"]}')
                for g in same:
                    self.assertEqual(g['ZG'], want['ZG'], f'截断改写了 ZG @ {cut:%Y-%m-%d}')
                    self.assertEqual(g['ZD'], want['ZD'], f'截断改写了 ZD @ {cut:%Y-%m-%d}')
                    self.assertEqual(g['direction'], want['direction'],
                                     f'截断改写了 direction @ {cut:%Y-%m-%d}')
                    checked += 1
        self.assertGreater(checked, 0, '没有查到一个已闭合中枢 —— 夹具没有覆盖到该不变量')


# --------------------------------------------------------------------------- #
# 5. 真数据端到端（无库则跳过）
# --------------------------------------------------------------------------- #
def _has_83():
    """能连上 MongoDB 8.3 吗？连不上就跳过真数据用例。

    ⚠️ 必须**显式给短超时**：`GQ_util_mongodb_client` 的默认
    `serverSelectionTimeoutMS` 是 30s（见其 docstring），服务器不可达时
    这一句会把整个测试套件拖住半分钟 —— 而「没库就跳过」应当是**瞬间**的。
    """
    try:
        from GolemQ.core.mongo import GQ_util_mongodb_client
        from GolemQ.core.settings import GQSETTING
        client = GQ_util_mongodb_client(
            GQSETTING.get_config('MONGODB', 'uri'), serverSelectionTimeoutMS=1500)
        client.admin.command('ping')
        return True
    except Exception:
        return False


@unittest.skipUnless(_has_83(), 'MongoDB 8.3 不可用 —— 真数据用例跳过')
class TestPivotRealData(unittest.TestCase):
    def test_000711_60min_end_to_end(self):
        from GolemQ import get_active_market

        d, name = get_active_market().get_kline_price_min(
            '000711', frequency='60min', realtime=False, verbose=False)
        bars = []
        for i, (idx, row) in enumerate(d.data.iterrows()):
            dt = idx[0] if isinstance(idx, tuple) else idx
            bars.append(RawBar(symbol='000711', dt=dt, id=i, freq=Freq.F60,
                               open=float(row['open']), close=float(row['close']),
                               high=float(row['high']), low=float(row['low']),
                               vol=float(row['volume']),
                               amount=float(row.get('amount', 0) or 0)))

        czsc = CZSC(bars, max_bi_count=126, verbose=False)
        pivots = calc_pivots(bi_list=czsc.bi_list)
        cls = classify_pivots(pivots)

        self.assertGreater(len(pivots), 0, '真实数据应至少识别出一个中枢')
        self.assertIn(cls['kind'], ('盘整', '上涨趋势', '下跌趋势'))
        df = pivots_to_df(pivots, kind=cls['kind'])
        self.assertEqual(len(df), len(pivots))


if __name__ == '__main__':
    unittest.main(verbosity=2)
