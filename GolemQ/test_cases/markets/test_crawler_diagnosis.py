#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""`markets/StockCN/crawler/diagnosis.py` —— 牛叉诊股页解析与落库形状。

钉住三件事：

1. **页面改版后的口径**（2026-10-22）：诊股页在 `doctor.10jqka.com.cn`，
   综合诊断 + 五维评分；旧的 `stockpage` 已经没有那些 class。
2. **日期取页面诊断日期**，不取 `today()` —— 周末抓到的应是上一个交易日。
3. **落库只写本模块的列**（`DIAG_PAYLOAD` 白名单），不碰同一文档里别人的列。
"""

import os
import sys

try:
    import GolemQ  # noqa: F401
except ImportError:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

import unittest
from unittest.mock import patch

from GolemQ.core.constants import AKA, FIELD as FLD
from GolemQ.markets.StockCN.crawler import diagnosis as D

_FIXTURE = '''
<div class="stockvalue"><span class="bignum">6</span><span class="smallnum">.7</span></div>
<div class="stocktotal">综合诊断：6.7分 打败了98%的股票！</div>
<div class="box2 indexStat" id="nav_technical">
  <h2 class="m_hd hd1"><a>技术面诊股</a></h2>
  <p><span class="title">诊断结果：<em>7.5</em></span>
     <span class="gray">技术面打败了93%的股票</span></p>
  <p class="content">近期的平均成本为<strong>1314.02元</strong>。多头行情中，并且有加速上涨趋势。
     <span class='date'>[诊断日期:2026年10月22日 12:46]</span></p>
</div>
<div class="box2 indexStat" id="nav_funds">
  <h2><a>资金面诊股</a></h2>
  <p><span class="title">诊断结果：<em>6.8</em></span>
     <span class="gray">资金面打败了92%的股票</span></p>
</div>
<div class="box2 indexStat" id="nav_message">
  <h2><a>消息面诊股</a></h2>
  <p><span class="title">诊断结果：<em>6.6</em></span>
     <span class="gray">消息面打败了87%的股票</span></p>
</div>
<div class="box2 indexStat" id="nav_trade">
  <h2><a>行业面诊股</a></h2>
  <p><span class="title">诊断结果：<em>3.5</em></span>
     <span class="gray">行业面打败了63%的股票</span></p>
</div>
<div class="box2 indexStat" id="nav_basic">
  <h2><a>基本面诊股</a></h2>
  <p><span class="title">诊断结果：<em>6.8</em></span>
     <span class="gray">基本面打败了86%的股票</span></p>
</div>
'''


class TestParseDiagnosis(unittest.TestCase):

    def test_extracts_overall_and_five_scores(self):
        out = D.parse_diagnosis(_FIXTURE)
        self.assertEqual(out['overall'], 6.7)
        self.assertEqual(out['pct_rank'], 0.98)
        self.assertEqual(out['avg_cost'], 1314.02)
        self.assertEqual(out['day'], '2026-10-22')
        self.assertEqual(out['scores'], {'技术面': 7.5, '资金面': 6.8,
                                         '消息面': 6.6, '行业面': 3.5,
                                         '基本面': 6.8})

    def test_empty_or_unrelated_page_returns_none(self):
        self.assertIsNone(D.parse_diagnosis(''))
        self.assertIsNone(D.parse_diagnosis('<html><body>404</body></html>'))

    def test_partial_page_still_keeps_dimensions(self):
        """没有综合诊断、只有维度评分时也收（单维页面/改版兜底）。"""
        out = D.parse_diagnosis(
            '<div id="nav_technical"><h2>技术面诊股</h2>'
            '诊断结果：<em>8.1</em></div>')
        self.assertEqual(out['scores'], {'技术面': 8.1})
        self.assertNotIn('overall', out)


class TestRowsDiagnosis(unittest.TestCase):

    def test_maps_to_reality_feat_columns(self):
        item = dict(D.parse_diagnosis(_FIXTURE), code='600519')
        doc = D.rows_diagnosis([item], created_at=0)[0]
        self.assertEqual(doc['code'], '600519')
        self.assertEqual(doc['date'], '2026-10-22')
        self.assertEqual(doc[AKA.OVERALL], 6.7)
        self.assertEqual(doc[AKA.PCT_RANKING], 0.98)
        self.assertEqual(doc[AKA.DIAG_AVG_COST], 1314.02)
        self.assertEqual(doc[AKA.TECH_SCORE], 7.5)
        self.assertEqual(doc[FLD.MONEYFLOW_SCORE], 6.8)
        self.assertEqual(doc[AKA.NEWS_SCORE], 6.6)
        self.assertEqual(doc[AKA.INDUSTRY_SCORE], 3.5)
        self.assertEqual(doc[AKA.BASIC_SCORE], 6.8)
        self.assertEqual(doc['created_at'], 0)

    def test_skips_item_without_code_or_day(self):
        self.assertEqual(D.rows_diagnosis([{'code': '', 'day': '2026-10-22'}]), [])
        self.assertEqual(D.rows_diagnosis([{'code': '600519'}]), [])


class TestShortCircuit(unittest.TestCase):
    """同一交易日已覆盖 >=90% 时不再逐只抓（用户 2026-10-22 要求）。"""

    def test_skips_when_day_covered(self):
        with patch.object(D, '_resolve_codes',
                          return_value=['600519', '000001']), \
                patch.object(D, 'metadata_covered',
                             return_value={'covered': 2}), \
                patch.object(D, 'fetch_diagnosis') as fetch:
            out = D.save_diagnosis(codes=['600519', '000001'],
                                   delay=0, verbose=False)
        fetch.assert_not_called()
        self.assertTrue(out['skipped'])
        self.assertEqual(out['written'], 0)

    def test_partial_coverage_still_fetches(self):
        item = dict(D.parse_diagnosis(_FIXTURE), code='600519')
        with patch.object(D, '_resolve_codes',
                          return_value=['600519', '000001']), \
                patch.object(D, 'metadata_covered',
                             return_value={'covered': 1}), \
                patch.object(D, 'fetch_diagnosis',
                             side_effect=[item, None]) as fetch, \
                patch.object(D, '_db', return_value=object()), \
                patch('GolemQ.markets.StockCN.reality_save.save_reality_feat',
                      return_value={'written': 1, 'upserted': 1, 'modified': 0}):
            out = D.save_diagnosis(codes=['600519', '000001'],
                                   delay=0, verbose=False)
        self.assertEqual(fetch.call_count, 2)
        self.assertFalse(out['skipped'])
        self.assertEqual(out['failed'], 1)


class TestSaveDiagnosis(unittest.TestCase):

    def setUp(self):
        # 短路闸默认放行（covered=0），本类测的是落库形状，不测闸本身。
        p = patch.object(D, 'metadata_covered', return_value={'covered': 0})
        p.start()
        self.addCleanup(p.stop)

    def test_writes_only_own_payload_columns(self):
        item = dict(D.parse_diagnosis(_FIXTURE), code='600519')
        with patch.object(D, '_resolve_codes', return_value=['600519']), \
                patch.object(D, 'fetch_diagnosis', return_value=item), \
                patch.object(D, '_db', return_value=object()), \
                patch('GolemQ.markets.StockCN.reality_save.save_reality_feat',
                      return_value={'written': 1, 'upserted': 1, 'modified': 0}) as save:
            out = D.save_diagnosis(codes=['600519'], delay=0, verbose=False)

        frame = save.call_args.args[0]
        self.assertEqual(frame.index.names, ['date', 'code'])
        self.assertEqual(len(frame), 1)
        # 只带本模块 payload + 索引键；不许把别人写的列搅进来
        self.assertEqual(set(frame.columns), set(D.DIAG_PAYLOAD))
        self.assertEqual(out['written'], 1)
        self.assertEqual(out['failed'], 0)
        self.assertFalse(out['dry_run'])

    def test_dry_run_does_not_write(self):
        item = dict(D.parse_diagnosis(_FIXTURE), code='600519')
        with patch.object(D, '_resolve_codes', return_value=['600519']), \
                patch.object(D, 'fetch_diagnosis', return_value=item), \
                patch('GolemQ.markets.StockCN.reality_save.save_reality_feat') as save:
            out = D.save_diagnosis(codes=['600519'], delay=0, verbose=False,
                                   dry_run=True)
        save.assert_not_called()
        self.assertTrue(out['dry_run'])
        self.assertEqual(out['written'], 1)

    def test_progress_lines_only_under_verbose(self):
        """文字进度锚点只在 `--verbose` 下打；默认输出留给 tqdm 进度条。"""
        codes = ['{:06d}'.format(i) for i in range(100)]

        def _run(verbose):
            lines = []
            with patch.object(D, '_resolve_codes', return_value=codes), \
                    patch.object(D, 'fetch_diagnosis', return_value=None):
                D.save_diagnosis(codes=['x'], delay=0, verbose=verbose,
                                 echo=lines.append)
            return lines

        self.assertFalse(any('100/100' in line for line in _run(False)))
        self.assertTrue(any('100/100' in line for line in _run(True)))

    def test_all_failures_return_error_not_raise(self):
        with patch.object(D, '_resolve_codes', return_value=['600519']), \
                patch.object(D, 'fetch_diagnosis',
                             side_effect=RuntimeError('boom')):
            out = D.save_diagnosis(codes=['600519'], delay=0, verbose=False)
        self.assertEqual(out['written'], 0)
        self.assertEqual(out['failed'], 1)
        self.assertIn('boom', out['error'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
