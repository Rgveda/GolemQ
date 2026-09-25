# coding:utf-8
"""`symbol.is_stock_cn` 的号段分类回归测试。

为什么值得单独一个文件
====================
`is_stock_cn` 是**纯函数**（只做字符串号段判断），但它的返回值被全树十余处
按位置解包，且**驱动集合路由**（`kline83.market_prefix` → 读 `stock_*` /
`index_*` / `etf_*`）。判错不会有异常，只会**静默读错集合**。

2026-09 重编排它以让 ETF 独立成 `ETF_CN` 时，发现原实现**深市 5 段里错了 4 段**
（把 `150` 分级 / `16x` LOF / `180` REITs / `20` B股 都判成「深交所ETF基金」），
且 `200`（B股）分支被 `20` 抢先命中而**从未生效**。本文件把这些号段钉死。

号段口径来源（交易所规则，非"看起来像"）见 `MIGRATION_STATUS.md` 的
「ETF 独立成 ETF_CN」记录；重编排的**全量**回归依据另有
`tools/dump_is_stock_cn_baseline.py`（扫全部 100 万代码与基线逐条比对）。
"""
import os
import sys
import unittest

try:
    import GolemQ  # noqa: F401
except ImportError:                     # pragma: no cover
    sys.path.insert(0, os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', '..')))

from GolemQ.core.constants import MARKET_TYPE          # noqa: E402
from GolemQ.markets.StockCN.symbol import is_stock_cn  # noqa: E402

STOCK, INDEX, ETF, FUND = (MARKET_TYPE.STOCK_CN, MARKET_TYPE.INDEX_CN,
                           MARKET_TYPE.ETF_CN, MARKET_TYPE.FUND_CN)


class TestIsStockCnSegments(unittest.TestCase):
    """逐号段的 `(code, 期望 market_type, 期望交易所)`。"""

    CASES = [
        # —— 沪市股票 ——
        ('600519', STOCK, 'SH'),        # 主板
        ('601398', STOCK, 'SH'),
        ('603259', STOCK, 'SH'),
        ('688981', STOCK, 'SH'),        # 科创板
        ('689009', STOCK, 'SH'),        # 科创板存托凭证
        ('900901', STOCK, 'SH'),        # B股
        # —— 沪市基金：50x 是基金，51x–58x 才是 ETF ——
        ('500001', FUND, 'SH'),         # 传统封闭式基金
        ('501050', FUND, 'SH'),         # LOF
        ('502001', FUND, 'SH'),         # 分级基金
        ('505001', FUND, 'SH'),         # 创新型封闭式基金
        ('506001', FUND, 'SH'),         # 科创板 LOF
        ('510300', ETF, 'SH'),          # 沪深300ETF
        ('513050', ETF, 'SH'),          # 跨境ETF
        ('515180', ETF, 'SH'),          # 跨市场股票ETF
        ('560050', ETF, 'SH'),          # 跨市场股票ETF（560–563 段）
        ('588000', ETF, 'SH'),          # 科创板ETF
        ('589000', ETF, 'SH'),          # 单市场科创板ETF（2025 启用）
        # —— 深市股票 ——
        ('000001', STOCK, 'SZ'),        # 主板（平安银行）
        ('001979', STOCK, 'SZ'),
        ('002594', STOCK, 'SZ'),        # 中小板
        ('300750', STOCK, 'SZ'),        # 创业板
        ('301269', STOCK, 'SZ'),
        ('302132', STOCK, 'SZ'),
        ('303001', STOCK, 'SZ'),
        ('200037', STOCK, 'SZ'),        # B股 ← 曾被 '20' 误判成 ETF
        ('205001', STOCK, 'SZ'),        # B股是整个 20 段
        ('280001', STOCK, 'SZ'),        # B股配股权证
        # —— 深市基金：ETF 只有 158/159 ——
        ('159915', ETF, 'SZ'),          # 创业板ETF
        ('158001', ETF, 'SZ'),
        ('150001', FUND, 'SZ'),         # 分级基金子份额 ← 原误判为 ETF
        ('160105', FUND, 'SZ'),         # LOF ← 原误判为 ETF
        ('161725', FUND, 'SZ'),         # LOF（原实现落到「未知」）
        ('169999', FUND, 'SZ'),
        ('180101', FUND, 'SZ'),         # 基础设施基金(REITs) ← 原误判为 ETF
        ('184001', FUND, 'SZ'),         # 封闭式基金（原实现落到「未知」）
        # —— 指数 ——
        ('399001', INDEX, 'SZ'),        # 深证成指
        ('000300', INDEX, 'SH'),        # 沪深300（走 SH 别名，沿袭原行为）
        ('000003', INDEX, 'SH'),
        # —— 北交所 / 新三板 ——
        ('920819', STOCK, 'BJ'),        # 北交所（2025-10 起存量全部切到 920 段）
        ('430489', STOCK, 'BJ'),        # 原三板
        ('830799', STOCK, 'BJ'),
        ('870436', STOCK, 'BJ'),
        ('889999', STOCK, 'BJ'),
        ('820001', STOCK, 'BJ'),        # 优先股 ← 原写成「北证A股」
    ]

    def test_segments(self):
        for code, market_type, exchange in self.CASES:
            with self.subTest(code=code):
                ok, mt, ex, desc = is_stock_cn(code)
                self.assertTrue(ok, f'{code} 应判为 A 股')
                self.assertEqual(mt, market_type, f'{code} 的 market_type')
                self.assertEqual(ex, exchange, f'{code} 的交易所别名')
                self.assertIsInstance(desc, str)
                self.assertTrue(desc, f'{code} 的描述不应为空')

    def test_etf_is_not_index(self):
        """本次改动的**核心断言**：ETF 与真指数必须是两个类型。

        这是「ETF 与真指数共用 index_*」那个历史约定的反面 ——
        两者一旦重新合并，ETF 的复权与容器选择会一起退化。
        """
        etf_codes = ['510300', '513050', '588000', '159915', '158001']
        index_codes = ['399001', '000300', '000003']
        for code in etf_codes:
            self.assertEqual(is_stock_cn(code)[1], MARKET_TYPE.ETF_CN, code)
        for code in index_codes:
            self.assertEqual(is_stock_cn(code)[1], MARKET_TYPE.INDEX_CN, code)


class TestExchangeTaggedForms(unittest.TestCase):
    """长度 >6 的带交易所标记写法。

    这类写法在全树有实际用处：`qmt_source.py` 解析 `'sh.600000'`，
    QUANTAXIS 口径用 `'600519.XSHG'`。**显式标记优先于号段推断** ——
    它本来就是用来消解 `000xxx` 这类固有歧义的。

    原实现的两个缺陷在此钉住：
    ① 长度判断**只认自己的交易所**，于是 `bj430489` 谁都进不去 → 返回 `False`；
    ② 深市分支的 `elif code.startswith('XSHE'): pass` 会**掉出函数**，
       使 `is_stock_cn('XSHE004000')` 返回 `None` —— 调用方解包即 `TypeError`。
    """

    def test_exchange_tokens(self):
        cases = [
            ('sh600519', 'SH', STOCK), ('sz000001', 'SZ', STOCK),
            ('bj430489', 'BJ', STOCK),
            ('XSHG600519', 'SH', STOCK), ('XSHE000001', 'SZ', STOCK),
            ('sh.600000', 'SH', STOCK), ('sz.000001', 'SZ', STOCK),
            ('600519.XSHG', 'SH', STOCK), ('000001.XSHE', 'SZ', STOCK),
            ('510300.XSHG', 'SH', ETF), ('510300.SH', 'SH', ETF),
            ('sh.510300', 'SH', ETF), ('XSHG510300', 'SH', ETF),
            ('159915.XSHE', 'SZ', ETF), ('sz.159915', 'SZ', ETF),
        ]
        for code, exchange, market_type in cases:
            with self.subTest(code=code):
                ok, mt, ex, desc = is_stock_cn(code)
                self.assertTrue(ok, f'{code} 应判为 A 股')
                self.assertEqual(ex, exchange, f'{code} 的交易所')
                self.assertEqual(mt, market_type, f'{code} 的 market_type')

    def test_owner_named_forms(self):
        """所有者点名的四种写法 —— **点号/紧贴 × 前缀/后缀**，四种都要认。

        这些写法在树里都有出处：`qmt_source.py:99` 解析 `'sh.600000'`，
        QUANTAXIS 口径用 `'600519.XSHG'`，QMT 用 `'600000.SH'`。
        """
        cases = [
            ('XSHE.004000', 'SZ'), ('004000.XSHE', 'SZ'),   # 点号：前缀式 / 后缀式
            ('sh600157', 'SH'), ('600157.sh', 'SH'),        # 紧贴 + 小写后缀
        ]
        for code, exchange in cases:
            with self.subTest(code=code):
                ok, mt, ex, desc = is_stock_cn(code)
                self.assertTrue(ok, f'{code} 应判为 A 股')
                self.assertEqual(ex, exchange, f'{code} 的交易所')

        # 有明确号段的两个还要能定出品种（不只是交易所）
        self.assertEqual(is_stock_cn('sh600157')[1], MARKET_TYPE.STOCK_CN)
        self.assertEqual(is_stock_cn('600157.sh')[1], MARKET_TYPE.STOCK_CN)
        # `004xxx` 是深市**未分配段** —— 交易所认得、品种没有。这不是缺陷：
        # 断言它**不返回 None**（老实现在这里返回 None，调用方解包即 TypeError）。
        self.assertIsNotNone(is_stock_cn('XSHE.004000'))
        self.assertIsNone(is_stock_cn('XSHE.004000')[1])

    def test_dotted_matrix(self):
        """点号写法的完整矩阵 —— 前后缀 × 三个交易所，大小写都收。"""
        for code, exchange in [
            ('XSHG.600157', 'SH'), ('600157.XSHG', 'SH'),
            ('sh.600157', 'SH'), ('600157.SH', 'SH'), ('600157.sh', 'SH'),
            ('SZ.000001', 'SZ'), ('000001.SZ', 'SZ'), ('sz.000001', 'SZ'),
            ('bj.430489', 'BJ'), ('430489.BJ', 'BJ'), ('430489.bj', 'BJ'),
        ]:
            with self.subTest(code=code):
                ok, _mt, ex, _desc = is_stock_cn(code)
                self.assertTrue(ok, code)
                self.assertEqual(ex, exchange, code)

    def test_bj_prefix_no_longer_false(self):
        """`bj430489` 曾被判成"不是 A 股"（返回 False），并连带丢掉交易所。"""
        ok, mt, ex, _ = is_stock_cn('bj430489')
        self.assertTrue(ok)
        self.assertEqual(ex, 'BJ')
        self.assertEqual(mt, MARKET_TYPE.STOCK_CN)

    def test_never_returns_none(self):
        """**绝不能返回 `None`** —— 全树按位置解包，`None[1]` 会 TypeError。

        原实现在深市分支的 `elif ...: pass` 处会掉出函数（实测 `XSHE004000`）。
        """
        for code in ('XSHE004000', 'XSHE200000', 'sh.600000', 'bj430489',
                     'XSHG510300', '510300.XSHE', '009000.XSHE'):
            with self.subTest(code=code):
                self.assertIsNotNone(is_stock_cn(code), f'{code} 返回了 None')

    def test_unknown_token_is_not_guessed(self):
        """**不认识的 token 不猜。**

        曾短暂把 `cz` 当成深交所别名（源自一次口头笔误），查证后确认两棵树里
        零出现，已撤掉 —— 凭空容忍一个 token 会让**打错的代码被静默当成深交所**。
        正确行为：判成不认识（`False`）。
        """
        for code in ('cz000001', 'zz600519', 'xx000001'):
            with self.subTest(code=code):
                ok, mt, _ex, _desc = is_stock_cn(code)
                self.assertFalse(ok, f'{code} 不该被当成 A 股')

    def test_explicit_token_beats_segment(self):
        """显式交易所优先 —— 即使号段暗示另一个交易所。

        ⚠️ 这是**行为约定**：`430000.XSHG` 矛盾输入下取 SH（显式）而**不是** BJ
        （号段）。原实现是 BJ（号段判断在前），本次改为显式优先。
        """
        self.assertEqual(is_stock_cn('430000.XSHG')[2], 'SH')
        self.assertEqual(is_stock_cn('430000')[2], 'BJ')      # 无标记时仍按号段
        self.assertEqual(is_stock_cn('510300.XSHE')[2], 'SZ')  # 矛盾输入取显式


class TestIsStockCnContract(unittest.TestCase):
    """返回契约与输入容忍度 —— 全树十余处按位置解包，形状不能变。"""

    def test_returns_four_tuple(self):
        for code in ('600519', '399001', '510300', '111111'):
            self.assertEqual(len(is_stock_cn(code)), 4, code)

    def test_list_takes_first_element(self):
        """列表输入只看第一个 —— 沿袭原行为，勿改（调用方依赖它）。"""
        self.assertEqual(is_stock_cn(['600519', '000001']),
                         is_stock_cn('600519'))

    def test_unknown_code(self):
        for code in ('111111', '444444', '777777'):
            ok, mt, ex, desc = is_stock_cn(code)
            self.assertFalse(ok, code)
            self.assertIsNone(mt, code)

    def test_empty_code(self):
        ok, mt, ex, desc = is_stock_cn('')
        self.assertFalse(ok)
        self.assertIsNone(mt)

    def test_suffix_forms(self):
        """带交易所后缀的写法要归一化到同一结论。"""
        self.assertEqual(is_stock_cn('600519.XSHG')[1], MARKET_TYPE.STOCK_CN)
        self.assertEqual(is_stock_cn('000001.XSHE')[1], MARKET_TYPE.STOCK_CN)
        self.assertEqual(is_stock_cn('510300.XSHG')[1], MARKET_TYPE.ETF_CN)

    def test_longest_prefix_wins(self):
        """包含关系必须由**最长前缀**解决，而不是靠书写顺序。

        原实现是深嵌套 `elif`，靠顺序保证优先级 —— `20`（B股）写在
        `200`（B股）之前，于是后者成了死代码。这里把那个 bug 钉住。
        """
        # 20 与 200 是同一类，但都**不是** ETF
        self.assertNotEqual(is_stock_cn('200037')[1], MARKET_TYPE.ETF_CN)
        self.assertNotEqual(is_stock_cn('205001')[1], MARKET_TYPE.ETF_CN)
        # 399（指数）不能被 39 之类的短前缀抢走；588/589（ETF）不能被 58 之外抢走
        self.assertEqual(is_stock_cn('399001')[1], MARKET_TYPE.INDEX_CN)
        self.assertEqual(is_stock_cn('588000')[1], MARKET_TYPE.ETF_CN)


if __name__ == '__main__':
    unittest.main()
