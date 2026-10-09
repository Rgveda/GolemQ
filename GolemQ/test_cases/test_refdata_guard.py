import os, sys
try:
    import GolemQ
except ImportError:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import unittest
from unittest.mock import MagicMock, patch


class _FakeSource:
    name = 'fake'

    def supports(self, collection):
        return True

    def available(self):
        return True

    def fetch(self, collection, **kwargs):
        return [{'code': '600519', 'name': 'x'}]


class TestPartialScopeNeverDeletes(unittest.TestCase):
    """**部分取数 × 差量删除 = 静默删掉其余全部**（`PITFALLS.md` P19）。

    实测踩过：`--save-codes` 跑一次把 `stock_info` 从 5,574 行删到 250 行 ——
    因为 `delete_delta_key` 的语义是「本次取到的就是全部」，而 `codelist`
    恰恰是「本次故意只取一部分」。护栏在 `save_refdata` 里。
    """

    def _run(self, codelist):
        from GolemQ.markets.StockCN import refdata_save as rs

        coll = MagicMock()
        db = MagicMock()
        db.__getitem__.return_value = coll
        db.__getitem__.return_value.distinct.return_value = ['600519', '000002']

        with patch.object(rs, '_pick_source', return_value=_FakeSource()), \
                patch.object(rs, 'GOLEMQ_STOCK_CN', db), \
                patch.object(rs, 'mark_refdata_success'):
            report = rs.save_refdata(collections=['stock_info'], codelist=codelist,
                                     verbose=False)
        return coll, report

    def test_partial_scope_skips_delta_delete(self):
        coll, report = self._run(['600519'])

        self.assertEqual(report['stock_info']['status'], 'ok')
        coll.delete_many.assert_not_called()          # ← 本次没取到的标的一个都不许删

    def test_full_scope_still_deletes_delta(self):
        """不给 codelist 时语义仍是「全量」，差量删除照旧（别把护栏扩大成不删）。"""
        coll, _ = self._run(None)

        self.assertTrue(coll.delete_many.called)


class TestOnProgressCallback(unittest.TestCase):
    """`--save` 的 banner 靠 `on_progress` 点亮节点，**每条出口都得调到**。

    `save_refdata` 里有 4 条 `continue` 出口（跳过/无源/取数失败/落库失败），
    回调要是写在循环末尾，这些出口就漏掉了 —— 屏上于是留下**永远点不亮**的节点。
    故整个循环体包在 `try/finally` 里，这个测试就是钉住那件事。
    """

    def _run(self, collections, pick_source):
        from GolemQ.markets.StockCN import refdata_save as rs

        coll = MagicMock()
        db = MagicMock()
        db.__getitem__.return_value = coll
        db.__getitem__.return_value.distinct.return_value = ['600519']

        seen = []
        with patch.object(rs, '_pick_source', side_effect=pick_source), \
                patch.object(rs, 'GOLEMQ_STOCK_CN', db), \
                patch.object(rs, 'mark_refdata_success'):
            report = rs.save_refdata(collections=collections, verbose=False,
                                     on_progress=lambda name, phase, entry: (
                                         seen.append((name, phase, entry['status'],
                                                      entry['rows']))))
        return seen, report

    def test_fires_once_per_requested_collection(self):
        seen, _ = self._run(['stock_info', 'stock_block'],
                            lambda *a, **k: _FakeSource())
        done = [x for x in seen if x[1] == 'done']
        self.assertEqual([x[0] for x in done], ['stock_info', 'stock_block'])

    def test_fires_on_every_exit_including_failure(self):
        """源不可用（`--save qmt` 的常态）时也要回调，否则节点永远不亮。"""
        from GolemQ.datasource.base import DataSourceNotAvailable

        def boom(*a, **k):
            raise DataSourceNotAvailable('qmt 当前不可用：测试')

        seen, _ = self._run(['stock_info', 'stock_block'], boom)
        done = [x for x in seen if x[1] == 'done']
        self.assertEqual([x[0] for x in done], ['stock_info', 'stock_block'])
        done = [x for x in seen if x[1] == 'done']
        self.assertEqual([x[2] for x in done], ['skipped', 'skipped'])
        # rows 为 0 → banner 据此退回 `·`（队列中），不谎报「读取完成」
        self.assertEqual([x[3] for x in done], [0, 0])

    def test_absent_callback_is_harmless(self):
        """默认 `on_progress=None`：除 `--save` 外的调用方不受任何影响。"""
        from GolemQ.markets.StockCN import refdata_save as rs

        coll = MagicMock()
        db = MagicMock()
        db.__getitem__.return_value = coll
        db.__getitem__.return_value.distinct.return_value = ['600519']
        with patch.object(rs, '_pick_source', return_value=_FakeSource()), \
                patch.object(rs, 'GOLEMQ_STOCK_CN', db), \
                patch.object(rs, 'mark_refdata_success'):
            report = rs.save_refdata(collections=['stock_info'], verbose=False)
        self.assertEqual(report['stock_info']['status'], 'ok')


class TestEtfListReachableFromSaveTdx(unittest.TestCase):
    """`--save tdx` 现在也做 `etf_list`。⚠️ 它**只能由 akshare 供**。

    实测（2026-10-09）：`tdxaidata` 的 `get_trackzs_etf_info` 返回 **0 行 +
    `[错误码 2] 股票代码错误`**，那条路是坏的；akshare 一次调用给 1694 行 / 1.4 秒。
    所以 CLI 侧**去掉了** `exclude_sources=('akshare',)` —— 那个排除对另外三个集合
    本就是空操作（akshare 只出现在 etf_list / financial 的优先级里），却恰好会把
    etf_list 推给坏掉的那条路。

    本类**不联网**：只钉「表怎么配的」，真取数已在上面那条实测里验过。
    """

    def test_tdx_source_can_supply_etf_list(self):
        from GolemQ.markets.StockCN import refdata_save as rs
        self.assertIn('etf_list', rs.REFDATA_BY_SOURCE['pytdx'])

    def test_qmt_source_does_not_claim_etf_list(self):
        """qmt 适配器没声明 etf_list —— 列进去只会每次报一遍 UnsupportedCollection。"""
        from GolemQ.markets.StockCN import refdata_save as rs
        self.assertNotIn('etf_list', rs.REFDATA_BY_SOURCE['qmt'])

    def test_akshare_is_first_for_etf_list(self):
        from GolemQ.markets.StockCN.datasource import COLLECTION_SOURCE_PRIORITY
        self.assertEqual(COLLECTION_SOURCE_PRIORITY['etf_list'][0], 'akshare')

    def test_neither_tdx_side_source_supplies_etf_list(self):
        """pytdx / qmt 都不提供 etf_list → akshare 是**唯一**可能，不能排除它。

        这条是给「将来有人想恢复 `exclude_sources=('akshare',)`」留的钉子：
        一恢复，etf_list 就只剩 tdxaidata 那条返回 0 行的坏路。
        """
        from GolemQ.markets.StockCN.datasource import COLLECTION_SOURCE_PRIORITY
        prio = COLLECTION_SOURCE_PRIORITY['etf_list']
        self.assertNotIn('pytdx', prio)
        self.assertNotIn('qmt', prio)


class TestRefdataTtlGate(unittest.TestCase):
    """刷新闸：距上次**成功完成**不足 `ttl_hours` 就跳过、**连源都不选**。

    起算点是「上次**成功完成**」而不是「上次启动」—— 否则跑挂了 / 跑一半会被当成
    刚刷新过。阈值由 `refdata_ttl_hours()` 给（盘中 5h / 盘后 24h）。
    """

    @staticmethod
    def _run(age_hours, ttl_hours):
        from GolemQ.markets.StockCN import refdata_save as rs

        db = MagicMock()
        db.__getitem__.return_value = MagicMock()
        db.__getitem__.return_value.distinct.return_value = ['600519']
        pick = MagicMock(return_value=_FakeSource())
        marked = []
        # ⚠️ `side_effect` 要收 `**kw`：`mark_refdata_success` 现在多一个 `echo=`
        # （走 banner 的输出汇，不写死 `print` —— 见 PITFALLS P22）。
        # 本用例只关心**哪个集合被记账**。
        # （注：这条注释**不能**写进下面的 `\` 续行中间 —— 注释会终止逻辑行，
        #   会让整个 `with` 语句截断成 SyntaxError。实测踩过。）
        with patch.object(rs, '_pick_source', pick), \
                patch.object(rs, 'GOLEMQ_STOCK_CN', db), \
                patch.object(rs, 'refdata_age_hours', return_value=age_hours), \
                patch.object(rs, 'mark_refdata_success',
                             side_effect=lambda n, **kw: marked.append(n)):
            report = rs.save_refdata(collections=['stock_info'], verbose=False,
                                     ttl_hours=ttl_hours)
        return report['stock_info'], pick, marked

    def test_fresh_is_skipped_and_source_untouched(self):
        entry, pick, _ = self._run(age_hours=1.0, ttl_hours=5)

        self.assertTrue(entry['cached'])          # ← 调用方据此点亮节点（数据是好的）
        self.assertEqual(entry['rows'], 0)
        self.assertIn('未过期', entry['detail'])
        pick.assert_not_called()                  # ← 不碰源 = 省下那次 HTTP 查询

    def test_stale_is_fetched(self):
        entry, pick, marked = self._run(age_hours=6.0, ttl_hours=5)

        self.assertFalse(entry['cached'])
        pick.assert_called_once()
        self.assertEqual(marked, ['stock_info'])       # 记的是「成功过」这件事

    def test_never_succeeded_is_fetched(self):
        """从没成功过（age=None）按「该取」处理 —— 别把新库冻住。"""
        _, pick, _ = self._run(age_hours=None, ttl_hours=5)
        pick.assert_called_once()

    def test_no_ttl_means_no_gate(self):
        """默认 `ttl_hours=None` = 不做闸，行为与从前逐字相同。"""
        _, pick, _ = self._run(age_hours=0.1, ttl_hours=None)
        pick.assert_called_once()


if __name__ == '__main__':
    unittest.main()
