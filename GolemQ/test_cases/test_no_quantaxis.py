import os, sys
try:
    import GolemQ
except ImportError:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import pathlib
import re
import unittest

#: 项目根（`GolemQ/` 的上一级）
_ROOT = pathlib.Path(__file__).resolve().parents[2]
_PKG = _ROOT / 'GolemQ'

#: 已随 D12 删除的 QUANTAXIS 句柄
_DEAD_SYMBOLS = ('DATABASE', 'DATABASE_QA', 'QAREALTIME', 'QASETTING',
                 'DATABASE_ASYNC')


class TestNoQuantaxis(unittest.TestCase):
    """把「QUANTAXIS 已从全树剔除」从人工 grep 升级成**会红的测试**。

    背景（`DECISIONS.md` D12）：D9 当年定「只解耦不搬数据」，把
    `core/settings.py` 那一处 `import QUANTAXIS` 留成了"暂留"，
    于是三周的会话都当定论照做。这次把它拔干净，并用三条断言钉住：
    ① 源码里不许再有 import；② 运行时不许被加载；③ 那 5 个句柄不许复活。
    """

    def test_no_source_import_of_quantaxis(self):
        """① 源码守卫：`GolemQ/**/*.py` 里不许出现 `import QUANTAXIS`。"""
        pat = re.compile(r'^\s*(?:from|import)\s+QUANTAXIS\b', re.M)
        hits = []
        for f in _PKG.rglob('*.py'):
            if '__pycache__' in f.parts:
                continue
            m = pat.search(f.read_text(encoding='utf-8'))
            if m:
                hits.append('{}:{}'.format(f.relative_to(_ROOT), m.group(0).strip()))
        self.assertEqual(hits, [], 'QUANTAXIS import 必须为 0（D12）；命中：%s' % hits)

    def test_quantaxis_not_loaded_at_runtime(self):
        """② 运行时守卫：把入口与市场包导入一遍，`sys.modules` 里不许有它。

        ⚠️ 这条同时是 `PITFALLS.md` P18 的判据 —— 模块级的跨库句柄/数据源包
        会把整棵树拖进来（实测 3,564 → 1,510 个模块的差别）。
        """
        import importlib
        for m in ('GolemQ.cli.__main__', 'GolemQ.markets.StockCN',
                  'GolemQ.supervisor.heartbeat', 'GolemQ.cli.watchdog_manager'):
            importlib.import_module(m)
        for root in ('QUANTAXIS', 'xtquant', 'akshare'):
            loaded = [m for m in sys.modules if m == root or m.startswith(root + '.')]
            self.assertEqual(loaded, [], '%s 不该被加载（%d 个模块）' % (root, len(loaded)))

    def test_dead_handles_stay_dead(self):
        """③ 符号守卫：那 5 个句柄不许在 settings / core / GolemQ 上复活。"""
        import GolemQ
        from GolemQ import core
        from GolemQ.core import settings
        for mod in (settings, core, GolemQ):
            for name in _DEAD_SYMBOLS:
                with self.subTest(module=mod.__name__, symbol=name):
                    with self.assertRaises(AttributeError):
                        getattr(mod, name)


if __name__ == '__main__':
    unittest.main()
