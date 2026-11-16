#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""参考名单的**退市归档**：名单差集 × 行情确认，缺一不搬。

背景（用户 2026-11-12）：原设计里 `stock_info` 会按差集硬删，而名单源历史上
**枚举不全过**（pytdx 拿不到北交所、akshare 的 `etf_list` 只到 927/1,737）——
一次网络错误就可能删掉大量仍然有效的代码。现在改成：

1. 名单源本次没出现的码只是**候选**；
2. 再用**行情库**确认它最近 `DELIST_QUIET_DAYS` 天确实没有日线；
3. 两条都成立才搬进 `*_delisted`，否则进 `pending` 只报告、不动数据。

基准日取**行情库自己的最新交易日**，所以「行情整体断更」不会变成「批量归档」。

⚠️ 全部用假集合，不碰 DB / 网络。
"""

import os
import sys
import time

try:
    import GolemQ  # noqa: F401
except ImportError:
    sys.path.insert(0, os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', '..', '..')))

import unittest
from unittest.mock import patch

from GolemQ.markets.StockCN import refdata_save as rs


class _Res:
    def __init__(self, upserted=0, modified=0, deleted=0):
        self.upserted_count = upserted
        self.modified_count = modified
        self.deleted_count = deleted


class _Coll:
    """够 `archive_delisted` / `confirm_delisted` 用的极简集合。"""

    def __init__(self, name, docs=None):
        self.name = name
        self.docs = [dict(d) for d in (docs or [])]

    # ---- 读 ----
    def find(self, q=None, proj=None, **kw):
        code = (q or {}).get('code')
        if isinstance(code, dict) and '$in' in code:
            allowed = set(code['$in'])
            return [dict(d) for d in self.docs if d['code'] in allowed]
        return [dict(d) for d in self.docs]

    def find_one(self, q=None, proj=None, **kw):
        if not self.docs:
            return None
        if kw.get('sort'):
            return dict(max(self.docs, key=lambda d: d.get('date_stamp') or 0))
        return dict(self.docs[0])

    def aggregate(self, pipeline):
        """只支持 `$match code $in` + `$group $max date_stamp` 这一种管道。"""
        allowed = set(pipeline[0]['$match']['code']['$in'])
        last = {}
        for d in self.docs:
            if d['code'] in allowed:
                last[d['code']] = max(last.get(d['code'], 0), d.get('date_stamp') or 0)
        return [{'_id': c, 'last': v} for c, v in last.items()]

    # ---- 写 ----
    def bulk_write(self, ops, ordered=False):
        up = mod = 0
        for op in ops:
            code = (getattr(op, '_filter', None) or {}).get('code')
            doc = dict(getattr(op, '_doc', None) or {})
            for i, d in enumerate(self.docs):
                if d.get('code') == code:
                    self.docs[i] = doc
                    mod += 1
                    break
            else:
                self.docs.append(doc)
                up += 1
        return _Res(up, mod)

    def delete_many(self, q):
        codes = set(q['code']['$in'])
        hit = [d for d in self.docs if d['code'] in codes]
        self.docs = [d for d in self.docs if d['code'] not in codes]
        return _Res(deleted=len(hit))


class _DB(dict):
    def __getitem__(self, name):
        return dict.__getitem__(self, name)


class TestConfirmDelisted(unittest.TestCase):
    DAY = 86400

    def _patch_db(self, colls):
        db = _DB(colls)
        patcher = patch.object(rs, 'GOLEMQ_STOCK_CN', db)
        patcher.start()
        self.addCleanup(patcher.stop)
        return db

    def test_recent_bar_means_still_trading(self):
        """名单缺失 + 刚刚还有日线 ⇒ 不是退市（pending），绝不能搬。"""
        now = int(time.time())
        self._patch_db({
            'index_list': _Coll('index_list'),
            'index_day': _Coll('index_day', [
                {'code': '000001', 'date_stamp': now},
                {'code': '000013', 'date_stamp': now - 100 * self.DAY},
            ]),
        })
        confirmed, pending = rs.confirm_delisted('index_list',
                                                 {'000001', '000013'})
        self.assertEqual(confirmed, {'000013'})
        self.assertEqual(pending, {'000001'})

    def test_baseline_is_kline_latest_not_wall_clock(self):
        """行情库整体停更时，基准日跟着退 —— 不会把整批判成「安静」。"""
        old = int(time.time()) - 400 * self.DAY
        self._patch_db({
            'index_list': _Coll('index_list'),
            'index_day': _Coll('index_day', [
                {'code': '000001', 'date_stamp': old},
                {'code': '000013', 'date_stamp': old - 400 * self.DAY},
            ]),
        })
        confirmed, pending = rs.confirm_delisted('index_list',
                                                 {'000001', '000013'})
        self.assertEqual(pending, {'000001'}, '它就是在库里最新那天还有行情的码')
        self.assertEqual(confirmed, {'000013'})

    def test_empty_kline_store_archives_nothing(self):
        self._patch_db({
            'index_list': _Coll('index_list'),
            'index_day': _Coll('index_day', []),
        })
        confirmed, pending = rs.confirm_delisted('index_list', {'000001'})
        self.assertEqual(confirmed, set())
        self.assertEqual(pending, {'000001'})


class TestArchiveDelisted(unittest.TestCase):
    DAY = 86400

    def _patch_db(self, colls):
        db = _DB(colls)
        patcher = patch.object(rs, 'GOLEMQ_STOCK_CN', db)
        patcher.start()
        self.addCleanup(patcher.stop)
        return db

    def _patch_veto(self, codes=None, exc=None):
        """中证指数官方清单（第三信源）。默认返回空 = 谁都不保护。"""
        target = ('GolemQ.markets.StockCN.datasource.akshare_source'
                  '.index_roster_codes')
        kwargs = {'side_effect': exc} if exc else {'return_value': set(codes or ())}
        patcher = patch(target, **kwargs)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _index_fixture(self, now):
        main = _Coll('index_list', [
            {'code': '000001', 'name': '上证指数'},
            {'code': '000013', 'name': '企债指数'},
            {'code': '000805', 'name': 'A股资源'},
        ])
        archived = _Coll('index_delisted')
        self._patch_db({
            'index_list': main,
            'index_delisted': archived,
            'index_day': _Coll('index_day', [
                {'code': '000001', 'date_stamp': now},
                {'code': '000013', 'date_stamp': now - 60 * self.DAY},
                {'code': '000805', 'date_stamp': now - 90 * self.DAY},
            ]),
        })
        return main, archived

    def test_moves_only_confirmed_dead_codes(self):
        now = int(time.time())
        main, archived = self._index_fixture(now)
        self._patch_veto()

        out = rs.archive_delisted('index_list', [{'code': '000001'}],
                                  echo=lambda *a: None)

        self.assertEqual(out['archived'], 2)
        self.assertEqual(out['pending'], [])
        self.assertEqual({d['code'] for d in main.docs}, {'000001'})
        moved = {d['code'] for d in archived.docs}
        self.assertEqual(moved, {'000013', '000805'})
        for d in archived.docs:
            self.assertEqual(d['delisted_from'], 'index_list')
            self.assertEqual(d['delisted_reason'], 'no_feed')
            self.assertIn('delisted_at', d)
        self.assertEqual(
            {d['code']: d['name'] for d in archived.docs},
            {'000013': '企债指数', '000805': 'A股资源'},
            '归档要保留原文档字段（名字还要能查）')

    def test_official_list_vetoes_alive_index(self):
        """中证官方清单里还有 ⇒ 指数只是 TDX 停推，不是退市 —— 不搬。"""
        now = int(time.time())
        main, archived = self._index_fixture(now)
        self._patch_veto({'000013'})          # 000013 仍在中证官方清单

        out = rs.archive_delisted('index_list', [{'code': '000001'}],
                                  echo=lambda *a: None)

        self.assertEqual({d['code'] for d in archived.docs}, {'000805'})
        self.assertEqual(out['archived'], 1)
        self.assertEqual(out['pending'], ['000013'])
        self.assertIn('000013', {d['code'] for d in main.docs})

    def test_veto_source_down_means_no_archive(self):
        """拿不到第三信源就不拍板 —— 两源皆无不足以判退市。"""
        now = int(time.time())
        main, archived = self._index_fixture(now)
        self._patch_veto(exc=ConnectionError('akshare timeout'))

        out = rs.archive_delisted('index_list', [{'code': '000001'}],
                                  echo=lambda *a: None)

        self.assertEqual(out['archived'], 0)
        self.assertEqual(len(main.docs), 3)
        self.assertEqual(archived.docs, [])
        self.assertIn('第三信源', out['skipped'])

    def test_still_trading_codes_go_to_pending_not_archive(self):
        """akshare 只枚举到 927/1,737 只 ETF —— 那 810 只还在交易，一只都不能动。"""
        now = int(time.time())
        main = _Coll('etf_list', [{'code': '158000'}, {'code': '510300'}])
        archived = _Coll('etf_delisted')
        self._patch_db({
            'etf_list': main,
            'etf_delisted': archived,
            'etf_day': _Coll('etf_day', [
                {'code': '158000', 'date_stamp': now},
                {'code': '510300', 'date_stamp': now},
            ]),
        })

        out = rs.archive_delisted('etf_list', [{'code': '510300'}],
                                  echo=lambda *a: None)

        self.assertEqual(out['archived'], 0)
        self.assertEqual(out['pending'], ['158000'])
        self.assertEqual(len(main.docs), 2, '主表一行都不能少')
        self.assertEqual(archived.docs, [])

    def test_empty_fetch_never_archives(self):
        self._patch_db({
            'index_list': _Coll('index_list', [{'code': '000001'}]),
            'index_delisted': _Coll('index_delisted'),
            'index_day': _Coll('index_day', []),
        })
        out = rs.archive_delisted('index_list', [], echo=lambda *a: None)
        self.assertEqual(out['archived'], 0)
        self.assertIsNotNone(out['skipped'])

    def test_stock_info_is_never_archived(self):
        """用户 2026-11-12 定：stock_info 不删不归档。"""
        self._patch_db({
            'stock_info': _Coll('stock_info', [{'code': '000001'}]),
        })
        out = rs.archive_delisted('stock_info', [{'code': '600519'}],
                                  echo=lambda *a: None)
        self.assertEqual(out['archived'], 0)
        self.assertEqual(out['skipped'], 'stock_info 不归档')


if __name__ == '__main__':
    unittest.main()
