#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""`pytdx` 常驻连接被毒死时的**不崩**与**自愈**（`PITFALLS.md` P3b）。

2026-11-12 `--save tdx` 实报：

    index_list  failed  0  pytdx  TypeError: 'NoneType' object cannot be
                                   interpreted as an integer

前四张表（stock_list / stock_info / stock_block / etf_list）全部 `ok`，只有
`index_list` 炸。根因不是「没有指数」，而是 `stock_info` 在**同一条常驻连接**上
打了 5,582 次 `get_finance_info` 之后连接被打歪 —— 之后 `get_security_count`
静默返回 `None`，而它被直接喂给了 `range(0, total, 1000)`。

⚠️ 这几条**不碰真网络**：假的 `_api` 复现「返回 None 而不抛异常」，正是 P3b
描述的那种最难发现的失败形态。真连接的行为在 `test_tdx_hosts` 里另测。
"""

import os
import sys

try:
    import GolemQ  # noqa: F401
except ImportError:
    sys.path.insert(0, os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', '..', '..')))

import unittest
from unittest.mock import patch

from GolemQ.datasource.base import DataSourceNotAvailable
from GolemQ.markets.StockCN.datasource.pytdx_source import TdxSource


class _FakeApi:
    """按顺序吐 `get_security_count` 的返回值；吐完就返回 None（= 连接已死）。"""

    def __init__(self, counts, rows=None):
        self._counts = list(counts)
        self._rows = rows or {}
        self.count_calls = 0
        self.disconnected = False

    def get_security_count(self, market):
        self.count_calls += 1
        return self._counts.pop(0) if self._counts else None

    def get_security_list(self, market, start):
        return self._rows.get(market, [])

    def disconnect(self):
        self.disconnected = True


class TestPoisonedConnection(unittest.TestCase):
    def _source(self):
        # 传死主机表：构造期不去读 tdx_hosts 缓存，也不会真连。
        return TdxSource(hosts=[('127.0.0.1', 7709)])

    def test_index_list_reconnects_when_count_is_none(self):
        """P3b：None ⇒ 换新连接重试，取数照常完成（而不是 range(None) 崩）。"""
        dead = _FakeApi([None])
        good = _FakeApi([1000, 1000],
                        {0: [{'code': '399001', 'name': '深证成指'}]})
        src = self._source()
        src._api = dead

        with patch.object(TdxSource, 'new_api', return_value=good):
            rows = src.fetch_index_list()

        self.assertEqual([r['code'] for r in rows], ['399001'])
        self.assertTrue(dead.disconnected, '毒死的连接必须被关掉，不能继续复用')

    def test_second_attempt_none_raises_instead_of_partial_list(self):
        """重连后仍是 None ⇒ **抛**，不能返回部分清单。

        `stock_list` / `stock_info` 带差量删除：少一个市场 = 把该市场全部标的
        静默删掉（`PITFALLS.md` P19）。所以失败必须显式报出来，让
        `save_refdata` 记 failed、不落库。
        """
        src = self._source()
        src._api = _FakeApi([None])

        with patch.object(TdxSource, 'new_api', return_value=_FakeApi([None])):
            with self.assertRaises(DataSourceNotAvailable):
                src.fetch_index_list()

    def test_unsupported_market_still_skips(self):
        """市场本身不可用（抛异常，如北交所）仍是「跳过」，不拖垮整批。"""

        class _Api(_FakeApi):
            def get_security_count(self, market):
                if market == 0:
                    raise RuntimeError('market not supported')
                return 1000

            def get_security_list(self, market, start):
                return [{'code': '000001', 'name': '上证指数'}]

        src = self._source()
        src._api = _Api([])

        rows = src.fetch_index_list()
        self.assertEqual([r['code'] for r in rows], ['000001'])

    def test_new_api_rejects_connection_whose_probe_returns_none(self):
        """探活返回 None 的服务器不能算「可用」—— 否则静默取空会扩散到整批。"""

        class _Api:
            def __init__(self, *args, **kwargs):
                pass

            def connect(self, host, port, time_out=10):
                return True

            def get_security_count(self, market):
                return None

        with patch('pytdx.hq.TdxHq_API', _Api):
            with self.assertRaises(DataSourceNotAvailable):
                self._source().new_api()


if __name__ == '__main__':
    unittest.main()
