#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""`~/.GolemQ/datastore/cache/stock_cn/stock_codes.conf` 的刷新契约。

为什么这份用例值得存在
======================
这份码表**不是白名单**，而是 `easyquotation` 全市场快照（`--sub l1_tencent`）
的**枚举源**。它此前全树无人写，于是新股永远收不到、退市股永远在收
—— 而这两件事**都不报错**：订阅器照样在跑、照样签到，只是实时集合里少了那几只。

三条钉子：
* **空表是灾难**（`TestRefusesToClobberOnEmpty`）—— 表空了就一台标的都不取，
  所以低于下限**宁可留旧表**，绝不用一个可疑的小列表覆盖好的；
* **无增减不写盘**（`TestOnlyWritesOnDelta`）—— 每日跑的大多数天码表不变，
  用「写盘前后字节相同」来钉，而不是拿 mtime 比（粒度太粗）；
* **只有一个写口**（`TestSingleWritePath`）—— 该文件曾有**两份写法**
  （`update_stock_codes` 自己 `open(...,'w')`），这一份既没建父目录也没写编码。
  平行实现不会报错，只会分叉。

本用例**不碰真库、不碰网络**：`_authoritative_codes` 与缓存路径都是替换掉的。
"""

import json
import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

try:
    import GolemQ  # noqa: F401
except ImportError:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from GolemQ.markets.StockCN import refdata_save as rs
from GolemQ.markets.StockCN.easyquotation import helpers


def _pad(n=None):
    """`MIN_STOCK_CODES` 之上的**填充码**，`X` 开头（**不可能是真码**）。

    ⚠️ 夹具**必须够长**：低于下限会被 `refuses_below_floor` 那条护栏挡下，
    于是"没写盘"变成**因为拒写**而不是因为没增量 —— 用例会**假绿**。

    ⚠️ 前缀不能用 `9xxxxx`：**沪市 B 股就是 900xxx**，实测与仓库内那份
    `stock_codes.conf` 撞了 53 个 —— 于是"缓存不存在"那条用例里
    `added` 会莫名少 53 个（`get_stock_codes` 读的是**仓库内兜底那份**）。
    """
    n = rs.MIN_STOCK_CODES + 5 if n is None else n
    return ['X%05d' % i for i in range(n)]


def _write_raw(path, payload):
    """按**给定字节**写缓存文件（绕开 save_stock_codes，便于钉"没被动过"）。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write(payload)


class _CacheRun:
    """把缓存路径挪进临时目录 + 给定「库里的码表」，跑一次 refresh。

    两侧都自动补 `_pad()`，于是**差值只反映用例真正关心的那几个码**。
    """

    def __init__(self, db_codes, cache_codes=None, cache_raw=None, pad=True):
        tail = _pad() if pad else []
        self.db = sorted(tail + list(db_codes))
        if cache_raw is not None:
            self.cache_raw = cache_raw
        elif cache_codes is not None:
            self.cache_raw = json.dumps({'stock': tail + sorted(cache_codes)})
        else:
            self.cache_raw = None                      # 缓存文件不存在
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, 'datastore', 'cache',
                                 'stock_cn', 'stock_codes.conf')

    def __enter__(self):
        if self.cache_raw is not None:
            _write_raw(self.path, self.cache_raw)
        self._p = patch.object(helpers, 'STOCK_CODE_PATH', self.path)
        self._p.start()
        self._a = patch.object(rs, 'snapshot_universe', return_value=self.db)
        self._a.start()
        return self

    def __exit__(self, *exc):
        self._a.stop()
        self._p.stop()
        self.tmp.cleanup()

    def run(self):
        return rs.refresh_stock_codes_cache(verbose=False, echo=lambda *a: None)

    def read(self):
        with open(self.path, encoding='utf-8') as fh:
            return json.load(fh)['stock']

    def raw(self):
        with open(self.path, encoding='utf-8') as fh:
            return fh.read()


class TestOnlyWritesOnDelta(unittest.TestCase):
    """**码表没有增减就不写盘** —— 每日 `--save tdx` 大多数天是这种。"""

    def test_no_delta_leaves_file_byte_identical(self):
        # 故意**乱序 + 无空白**写入：如果实现无脑重写，字节会变成排序后的版本
        payload = json.dumps({'stock': _pad() + ['600519', '000001', '000002']})
        with _CacheRun(['000001', '000002', '600519'], cache_raw=payload) as run:
            out = run.run()
            self.assertFalse(out['changed'])
            self.assertIsNone(out['refused'])              # ← 不是因为低于下限才没写
            self.assertEqual(run.raw(), payload)           # ← 一个字节都没动
        self.assertEqual(out['added'], [])
        self.assertEqual(out['removed'], [])

    def test_delta_is_reported_and_written(self):
        with _CacheRun(['000002', '000003', '600519'],
                       cache_codes=['000001', '000002', '600519']) as run:
            out = run.run()
            self.assertTrue(out['changed'])
            self.assertEqual(out['added'], ['000003'])     # 新上市
            self.assertEqual(out['removed'], ['000001'])   # 退市/换码
            self.assertEqual(run.read(), sorted(run.db))

    def test_missing_cache_is_created(self):
        """缓存文件不存在（全新机器）时，是**建**而不是"无增减"。

        ⚠️ 这时 `added` **不等于**全部库内码 —— `get_stock_codes()` 在缓存缺失时
        回落读**仓库内兜底那份**（6,000+ 条），真码基本都在里面。断言写「等于全部」
        会把兜底那份的行为焊进用例里，故只钉**真正的契约**：文件被建出来、内容 = 库。
        """
        with _CacheRun(['000001', '000002']) as run:
            out = run.run()
            self.assertTrue(out['changed'])
            self.assertTrue(out['added'])                  # 至少"建起来了"
            self.assertTrue(os.path.exists(run.path))
            self.assertEqual(run.read(), sorted(set(run.db)))


class TestRefusesToClobberOnEmpty(unittest.TestCase):
    """**空表是大灾难**：订阅器会静默地一台标的都取不到。低于下限就拒写。"""

    def test_refuses_below_floor(self):
        good = json.dumps({'stock': _pad()})
        with _CacheRun(['000001'], cache_raw=good, pad=False) as run:   # 库里只剩 1 只
            out = run.run()
            self.assertFalse(out['changed'])
            self.assertIsNotNone(out['refused'])
            self.assertEqual(out['total'], 1)
            self.assertEqual(run.raw(), good)                  # ← 好表原样留着

    def test_floor_is_above_any_plausible_accident(self):
        """下限得高于"一次抽风返回几十条"这种量级。"""
        self.assertGreaterEqual(rs.MIN_STOCK_CODES, 1000)

    def test_exactly_at_floor_is_allowed(self):
        """边界：**等于**下限允许写（判据是 `<`，不是 `<=`）。"""
        codes = ['%06d' % i for i in range(rs.MIN_STOCK_CODES)]
        with _CacheRun(codes, pad=False) as run:
            out = run.run()
            self.assertTrue(out['changed'])
            self.assertIsNone(out['refused'])


class TestWriteShape(unittest.TestCase):
    """落盘形态：`{"stock": [...]}`、去重有序、**原子**。"""

    def test_sorted_deduped_codes(self):
        with _CacheRun(['sh600519', 'sz000001', 'sh600519']) as run:
            run.run()
            self.assertEqual(run.read(), sorted(set(run.db)))

    def test_no_tmp_file_left_behind(self):
        """原子写的中间文件不许留在缓存目录里（留着会被当成第二张码表读到）。"""
        with _CacheRun(['600519']) as run:
            run.run()
            self.assertFalse(os.path.exists(run.path + '.tmp'))

    def test_creates_missing_parent_dirs(self):
        """`~/.GolemQ/datastore/cache/stock_cn/` 在全新机器上**不存在**。"""
        with _CacheRun(['600519']) as run:
            self.assertFalse(os.path.exists(os.path.dirname(run.path)))
            run.run()
            self.assertTrue(os.path.isdir(os.path.dirname(run.path)))


class TestSingleWritePath(unittest.TestCase):
    """该文件**只有一个写口** —— 两份写法必然分叉。"""

    def test_update_stock_codes_delegates(self):
        """上游那个 `shdjt.com` 版本必须走同一个写口（并因此拿到原子写 + 建目录）。"""
        html = 'var x = "~sh600519`~sz000001`";'
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'deep', 'stock_codes.conf')
            with patch.object(helpers, 'STOCK_CODE_PATH', path), \
                    patch.object(helpers.requests, 'get') as get:
                get.return_value.text = html
                got = helpers.update_stock_codes()
                # 上游那份码表是**带前缀**的（`sh600519`），`save_stock_codes` 原样保留
                self.assertEqual(got, ['sh600519', 'sz000001'])
                # ⚠️ 这两个断言必须在 `patch` 里 —— 出了 with 读的就是**真用户缓存**
                self.assertEqual(helpers.get_stock_codes(), ['sh600519', 'sz000001'])

    def test_get_stock_codes_falls_back_to_vendored(self):
        """缓存不存在时读仓库内那份（行为不变，`basequotation` 依赖它）。"""
        with patch.object(helpers, 'STOCK_CODE_PATH',
                          os.path.join(tempfile.gettempdir(), 'gq_no_such_conf')):
            codes = helpers.get_stock_codes()
        self.assertGreater(len(codes), 1000)                   # 仓库那份是 6000+


class _FakeDB:
    """`{集合名: [文档]}` → 一个只支持 `db[name].find(...)` 的假库。"""

    def __init__(self, rows):
        self._rows = rows

    def __getitem__(self, name):
        m = MagicMock()
        m.find.return_value = list(self._rows.get(name, []))
        return m


def _universe(rows):
    with patch.object(rs, 'GOLEMQ_STOCK_CN', _FakeDB(rows)):
        return rs.snapshot_universe()


class TestSnapshotUniverse(unittest.TestCase):
    """码表宇宙 = `stock_list` ∪ `etf_list` ∪ `index_list`，一律**带交易所前缀**。

    ⚠️ 前缀不是风格问题，是**正确性**问题 —— 见 `snapshot_universe` 的 docstring：
    不带前缀时 `000001` 会被 `get_stock_type` 判成深市，于是拿回**平安银行**
    而不是**上证指数**（同一个号，两个集合共用）。
    """

    def test_prefix_comes_from_the_sse_column(self):
        got = _universe({
            'stock_list': [{'code': '600519', 'sse': 'sh'}],
            'etf_list': [{'code': '159998', 'sse': 'sz'}],
            'index_list': [{'code': '399001', 'sse': 'sz'}],
        })
        self.assertEqual(got, ['sh600519', 'sz159998', 'sz399001'])

    def test_same_number_two_markets_stay_distinct(self):
        """**这条是同码陷阱的钉子**：`000001` 在 stock_list 是平安银行（sz）、
        在 index_list 是上证指数（sh）—— 两个都要在，且**不能并成一个**。"""
        got = _universe({
            'stock_list': [{'code': '000001', 'sse': 'sz'}],
            'index_list': [{'code': '000001', 'sse': 'sh'}],
        })
        self.assertEqual(got, ['sh000001', 'sz000001'])

    def test_blocked_prefix_is_dropped(self):
        """`880xxx`（通达信自编指数）腾讯**一只都不供**，不该占名额。"""
        got = _universe({
            'index_list': [{'code': '880001', 'sse': 'sh'},
                           {'code': '399001', 'sse': 'sz'}],
        })
        self.assertEqual(got, ['sz399001'])

    def test_rows_without_sse_are_skipped_not_guessed(self):
        """`sse` 缺失时**跳过**，绝不猜 —— 猜错就是静默取错标的。"""
        got = _universe({'index_list': [{'code': '000001'},
                                        {'code': '399001', 'sse': 'sz'}]})
        self.assertEqual(got, ['sz399001'])

    def test_dedupes_across_collections(self):
        """同一只标的出现在两个集合（如 ETF 也可能在 index_list）只算一次。"""
        got = _universe({
            'etf_list': [{'code': '510300', 'sse': 'sh'}],
            'index_list': [{'code': '510300', 'sse': 'sh'}],
        })
        self.assertEqual(got, ['sh510300'])

    def test_covers_all_three_collections_by_default(self):
        """默认口径就是三集合 —— 少一个都意味着「ETF 或指数又掉出流了」。"""
        self.assertEqual(rs.SNAPSHOT_COLLECTIONS,
                         ('stock_list', 'etf_list', 'index_list'))


class TestQuietUnlessVerbose(unittest.TestCase):
    """成功**静默**，「更新了几只」只在 `-v` 下打（用户 2026-10-15 定）。

    口径与参考数据的**刷新闸**那句、逐集合状态表一致 —— 都是实现细节。
    但**拒写**与**失败**不跟 `-v` 走：那意味着「这次没写 / 写坏了」，
    静默会让人以为写成功了。
    """

    def _run(self, db_codes, verbose, **kw):
        lines = []
        with _CacheRun(db_codes, **kw) as run:
            out = rs.refresh_stock_codes_cache(verbose=verbose, echo=lines.append)
        return out, lines

    def test_success_is_silent_without_verbose(self):
        out, lines = self._run(['sh600519'], verbose=False)
        self.assertTrue(out['changed'])
        self.assertEqual(lines, [])

    def test_verbose_reports_what_changed(self):
        out, lines = self._run(['sh600519'], verbose=True)
        self.assertTrue(out['changed'])
        self.assertTrue(any('码表缓存已更新' in l for l in lines), lines)

    def test_no_delta_is_silent_even_with_verbose_off(self):
        out, lines = self._run(['sh600519'], verbose=False,
                               cache_codes=['sh600519'])
        self.assertFalse(out['changed'])
        self.assertEqual(lines, [])

    def test_refusal_speaks_regardless_of_verbose(self):
        """**拒写照打** —— 这次真的没写，静默会被读成"写成功了"。"""
        good = json.dumps({'stock': _pad()})
        _out, lines = self._run(['000001'], verbose=False,
                                cache_raw=good, pad=False)
        self.assertTrue(any('拒绝覆盖' in l for l in lines), lines)


if __name__ == '__main__':
    unittest.main()
