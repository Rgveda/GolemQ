#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""`markets/StockCN/crawler/moneyflow.py` —— 两个源的行构造与单位换算。

为什么这份用例值得存在
======================
1. **单位的坑是静默的**。千股千评的「换手率」是**百分数**（`0.56` = 0.56%），
   要 ÷100 成小数才与 `stock_reality_feat_day` 里那列**同口径**（实测 `600519`
   `0.28→0.0028` 与库内值逐位相同）。少除或多除一次都不报错，只是数全错。
2. **`股票代码` 是 int**（同花顺那列给 `300530` 而不是 `'300530'`）——
   直接 `str()` 会把 `000001` 变成 **`'1'`**，静默写错行。
3. **两个源的载荷列必须互不重叠**（`:func:`…PAYLOAD` 是 `upsert_fields` 的白名单）。
   一旦重叠，后跑的源会覆盖前一个源的列 —— 这正是那张表的头号禁忌。
"""

import os
import sys

try:
    import GolemQ  # noqa: F401
except ImportError:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

import unittest
from unittest.mock import patch

import pandas as pd

from GolemQ.core.constants import FIELD as FLD
from GolemQ.markets.StockCN.crawler import moneyflow as M


class TestScalarParsers(unittest.TestCase):
    def test_to_code_keeps_six_digits(self):
        """⚠️ **int 陷阱**：`000001` 必须还是 `'000001'`，不能塌成 `'1'`。"""
        self.assertEqual(M.to_code(1), '000001')
        self.assertEqual(M.to_code(300530), '300530')
        self.assertEqual(M.to_code('600519'), '600519')
        self.assertEqual(M.to_code('600519.SH'), '600519')

    def test_parse_cn_amount_units(self):
        self.assertEqual(round(M.parse_cn_amount('3.72亿')), 372000000)
        self.assertEqual(round(M.parse_cn_amount('2098.28万')), 20982800)
        self.assertIsNone(M.parse_cn_amount('-'))
        self.assertIsNone(M.parse_cn_amount(None))

    def test_parse_pct_is_divide_by_100(self):
        self.assertEqual(M.parse_pct('20.02%'), 0.2002)
        self.assertEqual(M.parse_pct('-'), None)


def _em_frame():
    """造一份千股千评形状的帧（只留本用例关心的列）。"""
    return pd.DataFrame({
        '代码': ['600519', '000001', '300530'],
        '名称': ['贵州茅台', '平安银行', '领湃科技'],
        '交易日': [pd.Timestamp('2026-10-09').date()] * 3,
        '换手率': [0.28, 0.56, 3.61],
        '市盈率': [17.73, 4.38, 100.0],
        '主力成本': [1268.34, 11.6845, 28.0],
        '机构参与度': [0.5708, 0.3956, 0.1],
        '综合得分': [76.0, 66.57, 50.0],
        '上升': [244.0, -984.0, 1.0],
        '目前排名': [73.0, 1302.0, 900.0],
        '关注指数': [95.2, 92.8, 10.0],
        '涨跌幅': [0.57, -1.61, 20.02],
    })


def _ths_frame():
    return pd.DataFrame({
        '股票代码': [600519, 1, 300530],          # ⚠️ int
        '股票简称': ['贵州茅台', '平安银行', '领湃科技'],
        '换手率': ['0.28%', '0.56%', '15.04%'],   # ⚠️ 带 % 的字符串
        '流入资金': ['25.29亿', '4.71亿', '3.72亿'],
        '流出资金': ['19.25亿', '6.81亿', '3.51亿'],
        '净额': ['6.04亿', '-2.10亿', '2098.28万'],
        '成交额': ['44.53亿', '11.52亿', '7.23亿'],
    })


class TestShortCircuit(unittest.TestCase):
    """同一交易日已覆盖 >=90% 时不再抓取/落库（用户 2026-10-22 要求）。"""

    def test_skips_when_day_covered(self):
        with patch.object(M, 'last_trade_day', return_value='2026-10-09'), \
                patch.object(M, '_stock_universe_size', return_value=5000), \
                patch.object(M, 'metadata_covered', return_value={'covered': 5000}), \
                patch.object(M, 'fetch_eastmoney_comment') as fetch:
            out = M.save_moneyflow(source='eastmoney', verbose=False)
        fetch.assert_not_called()
        self.assertTrue(out['skipped'])
        self.assertEqual(out['written'], 0)

    def test_partial_coverage_still_fetches(self):
        """上一次半途失败只写了 100 只 —— 不到阈值，必须继续抓（不能留洞）。"""
        with patch.object(M, 'last_trade_day', return_value='2026-10-09'), \
                patch.object(M, '_stock_universe_size', return_value=5000), \
                patch.object(M, 'metadata_covered', return_value={'covered': 100}), \
                patch.object(M, 'fetch_eastmoney_comment',
                             return_value=_em_frame()) as fetch, \
                patch.object(M, '_db', return_value=object()), \
                patch('GolemQ.markets.StockCN.reality_save.save_reality_feat',
                      return_value={'written': 3, 'upserted': 3, 'modified': 0}):
            out = M.save_moneyflow(source='eastmoney', verbose=False)
        fetch.assert_called_once()
        self.assertFalse(out['skipped'])
        self.assertEqual(out['written'], 3)


class TestEastmoneyRows(unittest.TestCase):
    def test_columns_and_units(self):
        docs = M.rows_eastmoney(_em_frame(), created_at=0)
        self.assertEqual(len(docs), 3)
        d = {x['code']: x for x in docs}['600519']
        self.assertEqual(d['date'], '2026-10-09')
        # ⚠️ 这里**该**用容差：`0.28 / 100` 是一次**真实的除法**，二进制里没有
        # 精确表示（实测 = 0.0028000000000000004，**库里存的就是这个值**）。
        # 与 `PITFALLS.md` P30 那类「本该是精确值却被容差放过」**不是一回事** ——
        # 那边要求的是恒等变换（无事日比率），这边本来就是个算出来的数。
        self.assertAlmostEqual(d['TurnoverRate'], 0.0028, places=12)     # ÷100
        self.assertAlmostEqual(d['ChangePercent'], 0.0057, places=12)    # ÷100
        self.assertEqual(d[FLD.PE_RATION], 17.73)           # 原值，不经换算 ⇒ 精确
        self.assertEqual(d[FLD.ZLCB], 1268.34)
        self.assertEqual(d[FLD.RANKING], 73.0)

    def test_date_stamp_is_recomputed_not_copied(self):
        """`date_stamp` 由 `day_stamp` 重算（不取源端）—— 与 8.3 口径一致。"""
        from GolemQ.markets.StockCN.metadata_save import day_stamp
        docs = M.rows_eastmoney(_em_frame(), created_at=0)
        for d in docs:
            self.assertEqual(d['date_stamp'], day_stamp(d['date']))

    def test_does_not_carry_price_or_name(self):
        """`最新价`/`名称` **刻意不落**（价在 K 线、名在 stock_list）。"""
        d = M.rows_eastmoney(_em_frame(), created_at=0)[0]
        self.assertNotIn('New', d)
        self.assertNotIn('name', d)

    def test_bad_rows_are_skipped(self):
        df = pd.DataFrame({'代码': [None, '600519'], '交易日': [None, None],
                           '换手率': [1.0, 1.0]})
        self.assertEqual(len(M.rows_eastmoney(df, created_at=0)), 0)


class TestThsRows(unittest.TestCase):
    def test_code_does_not_collapse(self):
        """⚠️ 同花顺的代码是 int —— `000001` 必须是 `'000001'`。"""
        codes = {d['code'] for d in M.rows_ths(_ths_frame(), '2026-10-09', 0)}
        self.assertEqual(codes, {'600519', '000001', '300530'})

    def test_amounts_are_parsed(self):
        d = {x['code']: x for x in M.rows_ths(_ths_frame(), '2026-10-09', 0)}['600519']
        self.assertEqual(round(d[FLD.THS_INFLOW]), 2529000000)
        self.assertEqual(round(d[FLD.THS_OUTFLOW]), 1925000000)
        self.assertEqual(round(d[FLD.THS_NET]), 604000000)

    def test_turnover_and_change_are_dropped(self):
        """⚠️ 同花顺的换手率是**第三个口径** —— 丢，不落（否则就是第三列死列）。"""
        d = M.rows_ths(_ths_frame(), '2026-10-09', 0)[0]
        self.assertNotIn(FLD.TURNOVER_RATE, d)
        self.assertNotIn(FLD.CHANGE_PERCENT, d)
        self.assertNotIn('股票简称', d)

    def test_day_comes_from_the_caller(self):
        """同花顺不带交易日字段 ⇒ `day` 由调用方给（实测：周六给的是最后交易日）。"""
        for d in M.rows_ths(_ths_frame(), '2026-10-09', 0):
            self.assertEqual(d['date'], '2026-10-09')


class TestPayloadWhitelists(unittest.TestCase):
    """白名单是 `upsert_fields` 的**防线** —— 它必须覆盖本模块真写的每一列。"""

    def test_payloads_cover_what_rows_write(self):
        for docs, payload in ((M.rows_eastmoney(_em_frame(), 0), M.EM_PAYLOAD),
                              (M.rows_ths(_ths_frame(), '2026-10-09', 0), M.THS_PAYLOAD)):
            for d in docs:
                missing = set(d) - set(payload)
                self.assertEqual(missing, set(),
                                 '这些列会写库但不在白名单里：{}'.format(missing))

    def test_the_two_sources_do_not_share_payload_columns(self):
        """⚠️ **两个源不得碰同一列** —— 否则后跑的会覆盖前一个的成果。"""
        shared = (set(M.EM_PAYLOAD) & set(M.THS_PAYLOAD)) - {
            'code', 'date', 'date_stamp', 'datetime', 'ts', 'created_at'}
        self.assertEqual(shared, set(), '两个源共写了这些列：{}'.format(shared))


class TestSaveFrame(unittest.TestCase):
    """落库前的帧形状 —— 钉住 b5e95f3 引入的重复列回归。

    `keep` 曾把 `payload` 里已有的 `code`/`date` 再拼一次 ⇒
    `frame.loc[:, keep]` 出现重复列 ⇒ `set_index(['date','code'])`
    报 `Length mismatch: Expected axis has N elements, new values have 2`。
    """

    def setUp(self):
        # 本类用 `source='ths'`，本不走短路闸；但模块导入面保持 hermetic：
        # 万一将来 ths 也接闸，测试不能突然去读真库。
        for target, ret in (('metadata_covered', {'covered': 0}),
                            ('_stock_universe_size', 0)):
            p = patch.object(M, target, return_value=ret)
            p.start()
            self.addCleanup(p.stop)

    def test_frame_has_unique_index_and_columns(self):
        with patch.object(M, 'fetch_ths_fundflow', return_value=_ths_frame()), \
                patch.object(M, 'last_trade_day', return_value='2026-10-09'), \
                patch.object(M, '_db', return_value=object()), \
                patch('GolemQ.markets.StockCN.reality_save.save_reality_feat',
                      return_value={'written': 3, 'upserted': 3,
                                    'modified': 0}) as save:
            out = M.save_moneyflow(source='ths', verbose=False)

        frame = save.call_args.args[0]
        self.assertEqual(frame.index.names, ['date', 'code'])
        self.assertFalse(frame.index.has_duplicates)
        self.assertFalse(frame.columns.has_duplicates)
        self.assertEqual(len(frame), 3)
        self.assertEqual(out['written'], 3)

    def test_duplicate_source_rows_collapse(self):
        """同花顺分页会重叠（实测 3,300 行 / 2,237 码）—— 落库前按 `(date, code)` 去重。"""
        df = pd.concat([_ths_frame(), _ths_frame().iloc[[0]]], ignore_index=True)
        with patch.object(M, 'fetch_ths_fundflow', return_value=df), \
                patch.object(M, 'last_trade_day', return_value='2026-10-09'), \
                patch.object(M, '_db', return_value=object()), \
                patch('GolemQ.markets.StockCN.reality_save.save_reality_feat',
                      return_value={'written': 3, 'upserted': 3,
                                    'modified': 0}) as save:
            out = M.save_moneyflow(source='ths', verbose=False)

        frame = save.call_args.args[0]
        self.assertEqual(len(frame), 3)
        self.assertFalse(frame.index.has_duplicates)
        self.assertEqual(out['written'], 3)


if __name__ == '__main__':
    unittest.main(verbosity=2)
