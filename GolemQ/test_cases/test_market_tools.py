import os, sys
try:
    import GolemQ
except ImportError:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import unittest
from datetime import datetime as dt, timedelta
from unittest.mock import MagicMock
from GolemQ.markets.StockCN.tools import purge_historical_collections


class TestPurgeHistoricalCollections(unittest.TestCase):
    """``purge_historical_collections`` 的契约有两半：**集合名格式** + **返回删掉的名单**。

    它按名字（``realtime_YYYY-MM-DD``）在 ``list_collection_names()`` 里找集合，
    找不到就 ``consecutive_misses += 1``，连续 14 次后**安静退出**。
    所以名字格式一旦分叉（例如写成 ``realtime_20261008`` 或把 ``datetime``
    直接 format 进去），症状不是报错，而是**保留策略静默失效、磁盘无限涨**。
    下面后两个用例就是钉这个格式的钉子。

    ⚠️ 本函数**不打印**（唯一的打印处在 `cli/tools.py::purge_mongodb_database`），
    所以这里断言**返回值**，不再需要重定向 stdout。
    """

    @staticmethod
    def _client(names):
        client = MagicMock()
        client.list_collection_names.return_value = names
        return client

    @staticmethod
    def _name(days_ago):
        return 'realtime_{}'.format(
            (dt.now() - timedelta(days=days_ago)).strftime('%Y-%m-%d'))

    @staticmethod
    def _run(client):
        """跑一遍 purge，返回它删掉的集合名。"""
        return purge_historical_collections(client)

    def test_drops_collection_from_14_days_ago(self):
        """14 天前的那一个被 drop，14 天内的不动。"""
        target = self._name(14)
        client = self._client([target, self._name(3), 'other_collection'])

        dropped = self._run(client)

        self.assertEqual(dropped, [target])
        self.assertEqual([c.args[0] for c in client.__getitem__.call_args_list],
                         [target])
        client.__getitem__.return_value.drop.assert_called_once()

    def test_walks_back_until_14_consecutive_misses(self):
        """命中就继续往回走：14/15/16 天前三个都该删。"""
        targets = [self._name(d) for d in (14, 15, 16)]
        client = self._client(targets)

        dropped = self._run(client)

        self.assertEqual(dropped, targets)
        self.assertEqual([c.args[0] for c in client.__getitem__.call_args_list],
                         targets)
        self.assertEqual(client.__getitem__.return_value.drop.call_count, 3)

    def test_no_match_means_no_drop(self):
        client = self._client(['other_collection', 'realtime'])

        self.assertEqual(self._run(client), [])

        self.assertEqual(client.__getitem__.call_count, 0)

    def test_name_without_hyphens_is_not_matched(self):
        """``realtime_20261008``（无连字符）不认 —— 格式是契约的一部分。"""
        client = self._client(['realtime_{}'.format(dt.now().strftime('%Y%m%d'))])

        self.assertEqual(self._run(client), [])

        self.assertEqual(client.__getitem__.call_count, 0)

    def test_name_containing_time_is_not_matched(self):
        """``realtime_2026-10-08 01:00:27.222907`` 不认。

        这是 ``'realtime_{}'.format(dt.today())`` 的产物（本模块 ``dt`` 是
        ``datetime`` 类，不是 ``date``）—— 老树用的是 ``date.today()``，
        重构时被换掉，导致读取器拼出的名字**永远不存在**。见 PITFALLS。
        """
        client = self._client(['realtime_{}'.format(dt.now())])

        self.assertEqual(self._run(client), [])

        self.assertEqual(client.__getitem__.call_count, 0)

    def test_collection_list_is_fetched_once(self):
        """集合清单只取一次 —— 原来在循环里逐日问，一轮跑 28 次。"""
        client = self._client(['other_collection'])

        self._run(client)

        self.assertEqual(client.list_collection_names.call_count, 1)


if __name__ == '__main__':
    unittest.main()
