#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""`markets/StockCN/datasource/tdx_hosts.py` —— 服务器池的候选 / 探活 / 缓存。

⚠️ **不碰真网络的那几条**是主体；真探活只留一条「连不上要判成不可用」。
缓存一律打到临时文件 —— 绝不能覆盖用户 `~/.GolemQ/settings/tdx_hosts.json` 的真身。

为什么值得测：这份缓存**决定了每一次取数走哪台服务器**。它坏了（或写成空的）
不会报错，只会让 `TdxSource` 静默退回另一级，很难发现。
"""

import datetime as dt
import json
import os
import sys
import tempfile

try:
    import GolemQ  # noqa: F401
except ImportError:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import unittest
from unittest.mock import patch

from GolemQ.markets.StockCN.datasource import tdx_hosts


class TestCandidates(unittest.TestCase):
    def test_returns_deduped_ip_port_pairs(self):
        pool = tdx_hosts.candidates()
        self.assertGreaterEqual(len(pool), 60, 'pytdx 包内池应有 60+ 台')
        for ip, port in pool:
            self.assertIsInstance(ip, str)
            self.assertIsInstance(port, int)
        self.assertEqual(len(pool), len(set(pool)), '候选池里有重复')

    def test_never_empty_without_builtin_pool(self):
        """拿不到 pytdx 包内池时，也要有东西可试 —— 否则源直接不可用。"""
        pool = tdx_hosts.candidates(include_builtin_pool=False)
        self.assertTrue(pool, '没有包内池时候选为空 —— 兜底链断了')


class TestCacheRoundTrip(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.mkdtemp(prefix='gq_tdxhosts_')
        self.path = os.path.join(tmp, 'tdx_hosts.json')
        patcher = patch.object(tdx_hosts, 'cache_path', return_value=self.path)
        patcher.start()
        self.addCleanup(patcher.stop)

    @staticmethod
    def _alive():
        return [{'ip': '1.1.1.1', 'port': 7709, 'median': 0.08,
                 'runs': 3, 'fail_rate': 0.0},
                {'ip': '2.2.2.2', 'port': 7709, 'median': 0.15,
                 'runs': 2, 'fail_rate': 0.34}]

    def test_save_then_load_keeps_order(self):
        tdx_hosts.save(self._alive())
        self.assertEqual(tdx_hosts.load(), [('1.1.1.1', 7709), ('2.2.2.2', 7709)])

    def test_saved_file_records_when_and_why(self):
        """缓存要能自证「什么时候探的、当时多快」—— 排障全靠它。"""
        tdx_hosts.save(self._alive())
        with open(self.path, encoding='utf-8') as fh:
            doc = json.load(fh)
        self.assertIn('probed_at', doc)
        self.assertEqual(doc['hosts'][0]['median_ms'], 80.0)

    def test_missing_cache_is_none(self):
        self.assertIsNone(tdx_hosts.load())

    def test_stale_cache_is_none(self):
        """超过 `REFRESH_DAYS` 就当没有 —— 这是「每周一次」的实现方式（靠 mtime）。"""
        tdx_hosts.save(self._alive())
        old = (dt.datetime.now()
               - dt.timedelta(days=tdx_hosts.REFRESH_DAYS + 1)).timestamp()
        os.utime(self.path, (old, old))
        self.assertIsNone(tdx_hosts.load())

    def test_corrupt_cache_is_none_not_crash(self):
        """缓存坏了必须**静默退回**，不能让取数链跑不起来。"""
        with open(self.path, 'w', encoding='utf-8') as fh:
            fh.write('{ this is not json')
        self.assertIsNone(tdx_hosts.load())

    def test_refresh_skips_probing_when_cache_is_fresh(self):
        tdx_hosts.save(self._alive())
        with patch.object(tdx_hosts, 'probe_pool') as fake:
            hosts, did_probe = tdx_hosts.refresh()
        self.assertFalse(did_probe)
        fake.assert_not_called()
        self.assertEqual(hosts, [('1.1.1.1', 7709), ('2.2.2.2', 7709)])

    def test_refresh_keeps_old_cache_when_nothing_probed(self):
        """探测全灭（网络抖动）时**保留旧缓存**，别把可用列表清空 —— 那是自毁。"""
        tdx_hosts.save(self._alive())
        with patch.object(tdx_hosts, 'probe_pool', return_value=[]):
            hosts, did_probe = tdx_hosts.refresh(force=True)
        self.assertTrue(did_probe)
        self.assertEqual(hosts, [('1.1.1.1', 7709), ('2.2.2.2', 7709)],
                         '探不到时把旧缓存弄丢了')


class TestLoopbackIsNeverATdxServer(unittest.TestCase):
    """⚠️ **`127.0.0.1` 绝不能被判成一台 TDX 服务器**（用户 2026-10-09 明确）。

    为什么单独立一类：自然状态下这个用例会**碰巧通过** —— 本机没东西监听 7709，
    探针连不上，于是判成不可用。但那是**巧合，不是规则**：只要本机有任何一个进程
    监听 7709（开发用的假服务器、代理、别的行情客户端），探针就会把它收进来，
    而**由此产生的行情数据是假的**。

    更要命的是我第一版探针连"碰巧"都靠不住 —— 它只判「有没有抛异常」，而
    pytdx 连不上时**不抛异常**（`connect()` 重试约 7s 后返回，`get_security_count()`
    返回 `None`）→ 死服被判成 **7.03s「成功」**。**下面三条一起钉住这件事。**
    """

    def test_loopback_is_rejected_without_probing(self):
        """连探都不探 —— 判据是**地址本身**，不是"连不上"。"""
        with patch.object(tdx_hosts, '_probe_once') as fake:
            row = tdx_hosts.probe('127.0.0.1', 7709)
        fake.assert_not_called()
        self.assertFalse(row['ok'])
        self.assertIsNone(row['median'])

    def test_all_loopback_and_unspecified_forms(self):
        for ip in ('127.0.0.1', '127.1.2.3', '0.0.0.0', '::1'):
            with self.subTest(ip=ip):
                self.assertTrue(tdx_hosts.is_loopback(ip))
                self.assertFalse(tdx_hosts.probe(ip, 7709)['ok'])

    def test_pool_never_returns_loopback(self):
        """整池探活也不许把回环混进结果 —— 它会被写进缓存、被 `TdxSource` 选中。"""
        alive = tdx_hosts.probe_pool(pool=[('127.0.0.1', 7709), ('::1', 7709)],
                                     attempts=1, workers=1)
        self.assertEqual(alive, [])

    def test_domain_name_is_not_loopback(self):
        """`is_loopback` 只认 IP 字面量；域名（`shtdx.gtjas.com`）不是回环。"""
        self.assertFalse(tdx_hosts.is_loopback('shtdx.gtjas.com'))


class TestProbe(unittest.TestCase):
    def test_unreachable_host_is_not_ok(self):
        """连不上 → `ok=False`。**不能只看"没抛异常"** —— pytdx 连不上时返回 `None`。

        这条用的地址刻意**不是**回环（回环被上面那一类挡掉了），所以它测的是
        「返回值判据」本身：`get_security_count()` 返回 `None` 也算失败。
        """
        with patch.object(tdx_hosts, '_probe_once', return_value=None) as fake:
            row = tdx_hosts.probe('198.51.100.7', 7709, attempts=2)
        self.assertEqual(fake.call_count, 2)
        self.assertFalse(row['ok'])
        self.assertIsNone(row['median'])
        self.assertEqual(row['fail_rate'], 1.0)

    def test_none_return_from_the_wire_counts_as_failure(self):
        """真连一个**连不上但会返回 None** 的地址 —— 端口 1，pytdx 约 7s 后返回 None。"""
        self.assertIsNone(tdx_hosts._probe_once('198.51.100.7', 1, timeout=2))

    def test_probe_pool_sorts_by_median(self):
        fake = [{'ip': 'slow', 'port': 1, 'ok': True, 'runs': 3,
                 'median': 0.5, 'fail_rate': 0.0},
                {'ip': 'fast', 'port': 1, 'ok': True, 'runs': 3,
                 'median': 0.05, 'fail_rate': 0.0},
                {'ip': 'bad', 'port': 1, 'ok': False, 'runs': 0,
                 'median': None, 'fail_rate': 1.0}]
        with patch.object(tdx_hosts, 'probe', side_effect=fake):
            alive = tdx_hosts.probe_pool(pool=[('x', 1), ('y', 1), ('z', 1)],
                                         workers=1)
        self.assertEqual([r['ip'] for r in alive], ['fast', 'slow'],
                         '不可用的必须被剔除，且按中位延迟升序')


class TestHostOrderPerThread(unittest.TestCase):
    """服务器选择的顺序：**每个 worker 线程各粘各的**，首次用随机起点。

    ⚠️ 这条是「分布式读取」的落点。两个走极端都不行：

    * **全局粘一台**：所有 worker 挤到同一台（实测 60 次连接全打 `124.71.9.153`），
      那台负载陡增、更容易进入「不应答」的相位，且单台抖动拖慢全体。
    * **纯随机**：每次都有 `1/台数` 概率踩坏服、白等一个超时（D21 实测 10s；
      66 台里坏 1 台 × 33,474 次 ≈ 500 次 × 10s ≈ 83 分钟）。

    正解是**每线程随机起点 + 各自粘住自己那台**。
    """

    def _src(self, hosts):
        from GolemQ.markets.StockCN.datasource.pytdx_source import TdxSource
        return TdxSource(hosts=hosts)

    def test_order_is_a_permutation_of_hosts(self):
        hosts = (('a', 1), ('b', 1), ('c', 1), ('d', 1))
        order = self._src(hosts)._host_order()
        self.assertEqual(sorted(order), sorted(hosts), '顺序里漏台或多台了')
        self.assertEqual(len(order), len(set(order)), '顺序里有重复')

    def test_sticks_to_this_thread_s_last_good_host(self):
        hosts = (('a', 1), ('b', 1), ('c', 1))
        src = self._src(hosts)
        src._local.host = ('c', 1)          # 本线程上次成功的那台
        self.assertEqual(src._host_order()[0], ('c', 1), '没粘住自己那台')

    def test_two_threads_are_independent(self):
        """**两个线程互不影响** —— 这正是「分布式」：各粘各的，不会挤一起。"""
        import threading
        hosts = (('a', 1), ('b', 1), ('c', 1), ('d', 1))
        src = self._src(hosts)
        seen = {}

        def work(tag, preferred):
            src._local.host = preferred
            seen[tag] = src._host_order()[0]

        t1 = threading.Thread(target=work, args=('w1', ('a', 1)))
        t2 = threading.Thread(target=work, args=('w2', ('b', 1)))
        t1.start(); t2.start(); t1.join(); t2.join()

        self.assertEqual(seen, {'w1': ('a', 1), 'w2': ('b', 1)})

    def test_fresh_thread_shuffles(self):
        """新线程没有粘性 → 起点是整个列表的一个排列（随机打散）。

        不断言"一定不同"（那会 flaky），只断言**首项是池子里的合法成员**，
        且**多次调用首项会变**（随机性存在的证据）。
        """
        hosts = tuple((str(i), 1) for i in range(8))
        src = self._src(hosts)
        firsts = {src._host_order()[0] for _ in range(20)}
        self.assertTrue(firsts <= set(hosts))
        self.assertGreater(len(firsts), 1, '首项恒定 → 随机打散没生效')


if __name__ == '__main__':
    unittest.main()
