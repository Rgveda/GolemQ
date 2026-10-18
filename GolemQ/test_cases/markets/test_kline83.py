#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""`markets/StockCN/kline83.py` 的**展示名解析**（`_display_name`）。

为什么值得测
============
读取器返回的第二个值（`codename`）**曾经就是代码本身** —— 旧树这里调的是
`GQ_fetch_stock_name`（真名字），重构时被简化成 `codelist[0]`，于是三个 streamlit
demo 全印成 `301308（301308）`（用户 2026-10-11 实报）。
🔎 旁证：`analysis/ChipDistribution_jit.py:695` 的接收变量名当时就叫
`stock_name_faked`。

补回来之后，这个函数有**三条回落**，而回落路径**在真实数据上大多触发不到**
（实测：`stock_list` 那一跳今天 0 个代码命中）—— 所以必须用 **monkeypatch**
把每条路径都走一遍，否则它长期是一段没人验证过的代码。

⚠️ 本文件的 monkeypatch **不是**为了省掉 DB，而是因为**链路的顺序与回落**
本身就是被测对象：真实数据里 `stock_info` 与 `stock_list` 的覆盖完全一致，
不 mock 就永远碰不到第二条。
"""

import os
import sys

try:
    import GolemQ  # noqa: F401
except ImportError:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

import unittest
from unittest import mock

import pandas as pd

from GolemQ.markets.StockCN import refdata
from GolemQ.markets.StockCN.kline83 import _display_name

_CODE = '301308'


def _df(code, name):
    """造一个与 `refdata._read` 同形的返回（索引是 `code`，列含 `name`）。"""
    return pd.DataFrame({'code': [code], 'name': [name]}).set_index('code', drop=False)


class TestDisplayNameChain(unittest.TestCase):
    """**不带数据**地验证回落链 —— 见模块 docstring。"""

    def test_stock_prefers_stock_info(self):
        with mock.patch.object(refdata, 'GQ_fetch_stock_info', return_value=_df(_CODE, '江波龙')) as m1, \
                mock.patch.object(refdata, 'GQ_fetch_stock_list', return_value=_df(_CODE, '不该被用到')) as m2:
            self.assertEqual(_display_name(_CODE, 'stock'), '江波龙')
            m1.assert_called_once()
            m2.assert_not_called()          # 命中即停，不查第二跳

    def test_stock_falls_back_to_stock_list(self):
        """`stock_info` 空 ⇒ 查 `stock_list`（用户 2026-10-11 明确要求的那一跳）。

        ⚠️ 实测这一步**今天一次都不会触发**（两张表 5579 行、351 个空名、
        覆盖完全一致）—— 留着是防两张表将来分叉。所以它的正确性**只能靠这条用例**。
        """
        with mock.patch.object(refdata, 'GQ_fetch_stock_info', return_value=pd.DataFrame()), \
                mock.patch.object(refdata, 'GQ_fetch_stock_list', return_value=_df(_CODE, '江波龙')):
            self.assertEqual(_display_name(_CODE, 'stock'), '江波龙')

    def test_null_or_blank_name_falls_through(self):
        """`name` 为 `None` / 空串 / 纯空白 —— 都算"取不到"，继续往下走。"""
        for bad in (None, '', '   '):
            with self.subTest(name=bad):
                with mock.patch.object(refdata, 'GQ_fetch_stock_info',
                                       return_value=_df(_CODE, bad)), \
                        mock.patch.object(refdata, 'GQ_fetch_stock_list',
                                          return_value=_df(_CODE, '江波龙')):
                    self.assertEqual(_display_name(_CODE, 'stock'), '江波龙')

    def test_name_is_stripped(self):
        with mock.patch.object(refdata, 'GQ_fetch_stock_info',
                               return_value=_df(_CODE, '  江波龙  ')), \
                mock.patch.object(refdata, 'GQ_fetch_stock_list', return_value=pd.DataFrame()):
            self.assertEqual(_display_name(_CODE, 'stock'), '江波龙')

    def test_unknown_code_returns_the_code(self):
        """两条链都取不到 ⇒ **回落成代码本身**（真指数、未知代码都走这条）。"""
        with mock.patch.object(refdata, 'GQ_fetch_stock_info', return_value=pd.DataFrame()), \
                mock.patch.object(refdata, 'GQ_fetch_stock_list', return_value=pd.DataFrame()):
            self.assertEqual(_display_name('999999', 'stock'), '999999')

    def test_null_name_column_absent_is_tolerated(self):
        """返回帧里**没有 `name` 列**（读取器改过列）时不能崩。"""
        with mock.patch.object(refdata, 'GQ_fetch_stock_info',
                               return_value=pd.DataFrame({'code': [_CODE]})), \
                mock.patch.object(refdata, 'GQ_fetch_stock_list', return_value=pd.DataFrame()):
            self.assertEqual(_display_name(_CODE, 'stock'), _CODE)

    def test_never_raises_when_the_lookup_blows_up(self):
        """⚠️ 名字是**展示用品** —— 查询抛异常绝不能让取数失败。"""
        with mock.patch.object(refdata, 'GQ_fetch_stock_info',
                               side_effect=RuntimeError('db down')), \
                mock.patch.object(refdata, 'GQ_fetch_stock_list',
                                  side_effect=RuntimeError('db down')):
            self.assertEqual(_display_name(_CODE, 'stock'), _CODE)

    def test_index_prefers_index_list(self):
        """`index` 分支**先查 `index_list`**（真指数只在那里）。

        这张表是 2026-10-11 新建的：4.4 的 1291 行迁移时整张漏了，8.3 此前没有，
        于是真指数一直回落成代码（用户实报 `301308（301308）` 时顺带发现）。
        """
        with mock.patch.object(refdata, 'GQ_fetch_index_list',
                               return_value=_df('399001', '深证成指')) as m1, \
                mock.patch.object(refdata, 'GQ_fetch_etf_list',
                                  return_value=_df('399001', '不该被用到')) as m2:
            self.assertEqual(_display_name('399001', 'index'), '深证成指')
            m1.assert_called_once()
            m2.assert_not_called()          # 命中即停

    def test_index_falls_back_to_etf_list(self):
        """`index_list` 里没有 ⇒ 查 `etf_list`（ETF 只在那里）。

        ⚠️ 两张表**码段不重叠**（实测 `510300` 在 `etf_list`、不在 `index_list`），
        所以这两跳是确定的、不是碰运气。ETF 与真指数**共用** `index_*` 行情集合
        正是容易让人误以为名单也共用 —— 它们不共用。
        """
        with mock.patch.object(refdata, 'GQ_fetch_index_list', return_value=pd.DataFrame()), \
                mock.patch.object(refdata, 'GQ_fetch_etf_list',
                                  return_value=_df('510300', '沪深300ETF华泰柏瑞')) as m:
            self.assertEqual(_display_name('510300', 'index'), '沪深300ETF华泰柏瑞')
            m.assert_called_once()

    def test_index_without_name_returns_the_code(self):
        with mock.patch.object(refdata, 'GQ_fetch_index_list', return_value=pd.DataFrame()), \
                mock.patch.object(refdata, 'GQ_fetch_etf_list', return_value=pd.DataFrame()):
            self.assertEqual(_display_name('999999', 'index'), '999999')


class TestDisplayNameOnRealData(unittest.TestCase):
    """真实数据上的**冒烟** —— 链路逻辑已由上面覆盖，这里只确认接得上 8.3。"""

    @classmethod
    def setUpClass(cls):
        try:
            refdata.GQ_fetch_stock_info(codes=['600519'])
        except Exception as exc:                        # noqa: BLE001
            raise unittest.SkipTest('8.3 不可达 —— 跳过：{}'.format(exc))

    def test_known_stock(self):
        self.assertEqual(_display_name('600519', 'stock'), '贵州茅台')

    def test_known_real_index(self):
        """真指数走 `index_list`（2026-10-11 新建的表）—— 这条钉住"表还在"。

        ⚠️ 表在不在**必须钉**：它没有别的使用者，哪天被当成"没人读的集合"清掉，
        真指数就会静默回落成代码（正是 2026-10-11 之前的状态）。
        """
        self.assertEqual(_display_name('399001', 'index'), '深证成指')

    def test_known_etf(self):
        self.assertTrue(_display_name('510300', 'index'))       # 非空即可，名字可能变

    def test_unknown_code_falls_back(self):
        self.assertEqual(_display_name('999999', 'stock'), '999999')


if __name__ == '__main__':
    unittest.main(verbosity=2)
