import os, sys
try:
    import GolemQ
except ImportError:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

import pathlib
import re
import unittest

#: 项目根（`GolemQ/` 的上一级）
_ROOT = pathlib.Path(__file__).resolve().parents[3]
_PKG = _ROOT / 'GolemQ'

#: 允许出现库名/句柄的**唯一**目录（市场包自己）。
_ALLOWED = _PKG / 'markets' / 'StockCN'

#: 允许例外：测试要构造假库、要 patch 市场包内部模块的属性。
_SKIP_DIRS = (_PKG / 'test_cases',)

#: 库名（含实时库）与句柄名 —— 「库名不外泄」的扫描 token。
#: ⚠️ 只匹配 `golemq_stock_cn`，**不要**误伤框架运维库 `golemq`。
_PATTERNS = (
    re.compile(r'GOLEMQ_STOCK_CN'),
    re.compile(r'golemq_stock_cn\b'),
)


def _iter_sources():
    for f in _PKG.rglob('*.py'):
        if '__pycache__' in f.parts:
            continue
        if _ALLOWED in f.parents:
            continue
        if any(d in f.parents for d in _SKIP_DIRS):
            continue
        yield f


class TestNoDbNameLeak(unittest.TestCase):
    """「库名不外泄」守卫（`CLAUDE.md` 硬规定 / D60）。

    行情库的库名与句柄只允许出现在 `markets/StockCN/` 之内；别处一律经
    ``get_active_market().get_database()`` 取实例。判错**不会报错**，只会静默
    读到另一个库/空集 —— 所以把它从人工 grep 升级成会红的测试。
    """

    def test_no_db_name_or_handle_outside_market_package(self):
        hits = []
        for f in _iter_sources():
            text = f.read_text(encoding='utf-8')
            for pat in _PATTERNS:
                for m in pat.finditer(text):
                    line = text.count('\n', 0, m.start()) + 1
                    hits.append('{}:{}: {}'.format(f.relative_to(_ROOT), line, m.group(0)))
        self.assertEqual(hits, [],
                         '库名/句柄只允许出现在 markets/StockCN/ 内（D60）；'
                         '外部请走 get_database()。命中：%s' % hits)

    def test_database_accessor_is_the_same_handle(self):
        """中性访问口必须就是市场自己的那个库实例（不是另开连接）。"""
        from GolemQ.markets.StockCN import StockCN, GOLEMQ_STOCK_CN
        self.assertIs(StockCN().get_database(), GOLEMQ_STOCK_CN)

    def test_handles_are_not_in_public_all(self):
        """`__all__` 是公开 API 声明 —— 句柄/库名不该出现在里面。"""
        from GolemQ.markets.StockCN import __all__
        self.assertNotIn('GOLEMQ_STOCK_CN', __all__)
        self.assertNotIn('GOLEMQ_STOCK_CN_NAME', __all__)


if __name__ == '__main__':
    unittest.main()
