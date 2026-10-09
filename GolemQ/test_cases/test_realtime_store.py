import os, sys
try:
    import GolemQ
except ImportError:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import io
import unittest
from contextlib import redirect_stdout
from datetime import date, datetime as dt, timedelta, timezone
from unittest.mock import MagicMock

from GolemQ.markets.StockCN.realtime import (
    REALTIME_SOURCE_TENCENT_L1,
    REALTIME_SOURCE_TENCENT_L2,
    _write_ts_rows,
    realtime_collection_name,
)
from GolemQ.markets.StockCN.tools import purge_historical_collections


class TestRealtimeCollectionName(unittest.TestCase):
    """集合名是**跨模块契约**（写入端 ↔ 读取端 ↔ purge），故逐字钉住。"""

    def test_none_means_today(self):
        self.assertEqual(realtime_collection_name(),
                         'realtime_{}'.format(dt.now().strftime('%Y-%m-%d')))

    def test_accepts_four_input_types(self):
        for arg in (date(2026, 10, 8), dt(2026, 10, 8, 1, 2, 3), '2026-10-08'):
            with self.subTest(arg=arg):
                self.assertEqual(realtime_collection_name(arg),
                                 'realtime_2026-10-08')

    def test_never_contains_time_of_day(self):
        """``'realtime_{}'.format(dt.today())`` 会拼出带时分秒的名字。

        本模块的 ``dt`` 是 ``datetime`` **类**（不是 ``date``），老树那里用的是
        ``date.today()``。读路径因此拼出一个**永远不存在**的集合名，
        症状是**静默读空**而非报错。这个断言就是那颗钉子。
        """
        name = realtime_collection_name(dt(2026, 10, 8, 1, 2, 3, 123456))
        self.assertEqual(name, 'realtime_2026-10-08')
        self.assertNotIn(' ', name)
        self.assertNotIn(':', name)

    def test_name_matches_what_purge_looks_for(self):
        """写入端产出的名字，purge 必须认（否则保留策略静默失效）。

        这是**跨模块**断言：名字由 `realtime_collection_name` 生成，删它的是
        `tools.purge_historical_collections`（它自己按 ``%Y-%m-%d`` 拼名字）。
        """
        old = dt.now() - timedelta(days=14)
        client = MagicMock()
        client.list_collection_names.return_value = [realtime_collection_name(old)]

        with redirect_stdout(io.StringIO()):
            purge_historical_collections(client)

        self.assertEqual([c.args[0] for c in client.__getitem__.call_args_list],
                         [realtime_collection_name(old)])
        client.__getitem__.return_value.drop.assert_called_once()


class TestWriteTsRows(unittest.TestCase):
    """``_write_ts_rows`` 是唯一的写口，幂等性全靠它（时间序列不能 upsert）。"""

    TS = dt(2026, 10, 8, 1, 30, tzinfo=timezone.utc)

    @classmethod
    def _rows(cls, *offsets, code='600519.XSHG'):
        return [{'code': code, 'ts': cls.TS + timedelta(seconds=o),
                 'datetime': 'x', 'price': 1.0} for o in offsets]

    def setUp(self):
        self.collection = MagicMock()
        self.last_ts = {}

    def test_delete_carries_source_and_runs_before_insert(self):
        # 两个 code 各一行 —— 都是本进程首次见到的，故都进删除集
        rows = self._rows(0) + self._rows(2, code='000001.XSHE')
        n = _write_ts_rows(self.collection, rows, self.last_ts,
                           REALTIME_SOURCE_TENCENT_L1)

        self.assertEqual(n, 2)
        cond = self.collection.delete_many.call_args.args[0]
        self.assertEqual(cond['source'], REALTIME_SOURCE_TENCENT_L1)
        self.assertEqual(cond['code'], {'$in': ['000001.XSHE', '600519.XSHG']})
        self.assertEqual(cond['ts'], {'$in': [self.TS, self.TS + timedelta(seconds=2)]})
        # 顺序：先删后插 —— 反了就不幂等（重复行会留下）
        self.assertLess(self.collection.mock_calls.index(
                            next(c for c in self.collection.mock_calls
                                 if c[0] == 'delete_many')),
                        self.collection.mock_calls.index(
                            next(c for c in self.collection.mock_calls
                                 if c[0] == 'insert_many')))
        for r in rows:
            self.assertEqual(r['source'], REALTIME_SOURCE_TENCENT_L1)

    def test_first_seen_covers_every_ts_of_that_code_in_the_batch(self):
        """重启后第一批：同一 code 的**每个** ts 都要进删除条件。

        这条是回归钉子。收窄 delete（只在首次见到该 key 时删）的第一版是**按行**
        判的：同一个 code 的第二行起就不进删除集了 → 幂等被破坏，
        端到端实测"模拟重启后再写同一批"库内从 5 行涨到 **9 行**。
        判据必须是**按 key**，然后删该 key 在本批次里的全部 ts。
        """
        n = _write_ts_rows(self.collection, self._rows(0, 2), self.last_ts,
                           REALTIME_SOURCE_TENCENT_L1)

        self.assertEqual(n, 2)
        cond = self.collection.delete_many.call_args.args[0]
        self.assertEqual(cond['ts'],
                         {'$in': [self.TS, self.TS + timedelta(seconds=2)]})

    def test_delete_only_for_first_seen_codes(self):
        """稳态下**不删** —— 每轮 4900 个值的 delete 实测约 3.7 秒，跑不动。

        收窄的依据：进程内重复由 `last_ts` 挡，跨进程重复只可能出现在重启后的
        第一批（那时 `last_ts` 是空的，所有行都是 first_seen，照样会删）。
        """
        _write_ts_rows(self.collection, self._rows(0), self.last_ts,
                       REALTIME_SOURCE_TENCENT_L1)
        self.collection.reset_mock()

        n = _write_ts_rows(self.collection, self._rows(2), self.last_ts,
                           REALTIME_SOURCE_TENCENT_L1)
        self.assertEqual(n, 1)
        self.collection.delete_many.assert_not_called()
        self.collection.insert_many.assert_called_once()

    def test_row_not_newer_than_last_ts_is_skipped(self):
        _write_ts_rows(self.collection, self._rows(0), self.last_ts,
                       REALTIME_SOURCE_TENCENT_L1)
        self.collection.reset_mock()

        n = _write_ts_rows(self.collection, self._rows(0, -2), self.last_ts,
                           REALTIME_SOURCE_TENCENT_L1)
        self.assertEqual(n, 0)
        self.collection.delete_many.assert_not_called()
        self.collection.insert_many.assert_not_called()

    def test_last_ts_is_keyed_by_source_too(self):
        """同一个 code、同一个 ts，两条流互不遮挡。

        L1 与 L2 落在同一集合、同一 `(code, ts)`，若 `last_ts` 只按 code 记，
        第二条流的第一行会被当成"重复"丢掉。
        """
        ts = self._rows(0)
        self.assertEqual(_write_ts_rows(self.collection, ts, self.last_ts,
                                        REALTIME_SOURCE_TENCENT_L1), 1)
        self.assertEqual(_write_ts_rows(self.collection, self._rows(0),
                                        self.last_ts,
                                        REALTIME_SOURCE_TENCENT_L2), 1)

    def test_rows_missing_code_or_ts_are_dropped(self):
        rows = [{'code': '600519.XSHG'},                 # 缺 ts
                {'ts': self.TS},                          # 缺 code
                {'code': '600519.XSHG', 'ts': None}]      # ts 为空
        n = _write_ts_rows(self.collection, rows, self.last_ts,
                           REALTIME_SOURCE_TENCENT_L1)
        self.assertEqual(n, 0)
        self.collection.insert_many.assert_not_called()


if __name__ == '__main__':
    unittest.main()
