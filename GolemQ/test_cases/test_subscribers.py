import os, sys
try:
    import GolemQ
except ImportError:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import inspect
import unittest

from GolemQ import GQSUBSCRIBER
import GolemQ.markets.StockCN      # noqa: F401 —— 导入即注册（StockCN 是单例）


class TestSubscriberRegistry(unittest.TestCase):
    """`GQSUBSCRIBER` 的两条契约（`cli/__main__.py` 的 `--sub` 依赖它们）。

    ① **键名**：`--sub <key>` 直接查这张表，键名就是用户输入的字符串；
    ② **可零参调用**：分发处是 `subscriber_func()`，不带任何实参
       （见 `PITFALLS.md` P10 —— 这条曾让 `--sub l1_tencent` 必然 TypeError）。
    """

    def test_legacy_tencent_key_points_to_l1(self):
        """老树的 `--sub tencent` 必须仍可用，且与 `l1_tencent` 是同一个函数。

        老树 `GolemQ_old/cli/__main__.py` 里 `sub == 'tencent'` 调的是
        `sub_l1_from_tencent(database_realtime=QAREALTIME)`；新树把库换成了 8.3 的
        `golemq_stock_cn_realtime`（按日时间序列集合），函数体已就地迁移 ——
        故这里只需把**旧键名**接回来，不新增第二个实现。
        """
        self.assertIn('tencent', GQSUBSCRIBER)
        self.assertIs(GQSUBSCRIBER['tencent'], GQSUBSCRIBER['l1_tencent'])

    def test_every_subscriber_is_callable_without_args(self):
        """表里每个订阅函数都必须能零参调用 —— 否则 CLI 一跑就 TypeError。"""
        self.assertTrue(GQSUBSCRIBER, '注册表为空：market 注册没生效？')
        for key, func in GQSUBSCRIBER.items():
            with self.subTest(key=key):
                self.assertTrue(callable(func), f'{key} 不是可调用对象')
                sig = inspect.signature(func)
                required = [p.name for p in sig.parameters.values()
                            if p.default is inspect.Parameter.empty
                            and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD,
                                           p.KEYWORD_ONLY)]
                self.assertEqual(required, [],
                                 f'{key} 有必填参数 {required}，CLI 是无参调用')

    def test_expected_keys_are_present(self):
        """当前应支持的三条实时订阅键（新增/改名时这个用例会红 —— 那是有意的）。"""
        self.assertEqual(sorted(GQSUBSCRIBER), ['l1_tencent', 'l2_tencent', 'tencent'])


if __name__ == '__main__':
    unittest.main()
