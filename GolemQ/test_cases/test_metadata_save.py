#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""`stock_metadata_day` 的落库契约。

为什么这份用例值得存在
======================
这张表的**唯一键是 `(code, date_stamp)`**，而 `date_stamp` 的口径是
「把北京日历日的零点当作 UTC」—— 不是真实时刻。**差 8 小时也查得出来、
也不会报错**，只会静默命中 0 行或插出重复行。所以口径必须被钉死。

第二条要钉的是**「只 `$set` 自己的列」**：这张表是**多列共用一个文档**的
（东财换手率一列、baostock 一列，以后还有别的任务）。一旦哪个写路径用了
`save_collection`（`ReplaceOne` 整文档替换），别人的列会被**静默抹掉**。
`TestUpsertFieldsDoesNotClobber` 就是这条的钉子 —— 它同时验证
`save_collection` **确实会**抹（反证危害真实存在）。

真库用例在没有 MongoDB 时**跳过而不是失败**。
"""

import os
import sys

try:
    import GolemQ  # noqa: F401
except ImportError:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import unittest

from GolemQ.datasource.writer import upsert_fields
from GolemQ.markets.StockCN.metadata_save import (
    MAX_TURNOVER_RATIO, METADATA_DAY_COLL, SOURCES, TURNOVER_BS, TURNOVER_DC,
    UNIQUE_KEYS, _day_stamp, day_date, metadata_day_doc, turnover_rows,
)


class TestDayStampConvention(unittest.TestCase):
    """`date_stamp` 的口径 —— 差 8 小时就静默查空，所以拿**库里实测值**钉住。"""

    #: 实测自 4.4：`{'date': '2018-12-20 00:00:00', 'date_stamp': 1545264000}`
    STORED = (('2018-12-20', 1545264000), ('2026-09-01', 1788220800))

    def test_matches_stored_values(self):
        for day, stamp in self.STORED:
            with self.subTest(day=day):
                self.assertEqual(_day_stamp(day), stamp)

    def test_is_not_the_real_instant(self):
        """反证：**不能**用 `bj_date().timestamp()` —— 那差 8 小时。"""
        from GolemQ.markets.StockCN.kline83 import bj_date
        real = int(bj_date('2018-12-20 00:00:00').timestamp())
        self.assertEqual(real, 1545264000 - 8 * 3600)
        self.assertNotEqual(real, _day_stamp('2018-12-20'))

    def test_accepts_datetime_string(self):
        """源端两种格式都收（`stock_ranking` 带时分秒、`stock_valuation` 不带）。"""
        self.assertEqual(_day_stamp('2018-12-20 00:00:00'), _day_stamp('2018-12-20'))
        self.assertEqual(day_date('2018-12-20 00:00:00'), '2018-12-20')


class TestMetadataDayDoc(unittest.TestCase):
    def test_field_set_and_types(self):
        d = metadata_day_doc('600519', 1791388800, '2026-10-09', TURNOVER_DC, 0.0123,
                             created_at=1791400000)
        # ⚠️ sorted() 按码点：大写 T 排在小写前 ⇒ 换手率那列在**最前**
        self.assertEqual(sorted(d), [TURNOVER_DC, 'code', 'created_at', 'date',
                                     'date_stamp', 'datetime', 'ts'])
        self.assertEqual(d['date_stamp'], 1791388800)
        self.assertIsInstance(d['date_stamp'], int)
        self.assertEqual(d['date'], '2026-10-09')
        self.assertEqual(d['datetime'], '2026-10-09 00:00:00')
        self.assertEqual(d[TURNOVER_DC], 0.0123)
        # `ts` 是**真实时刻**且 UTC-aware：北京 2026-10-09 00:00 == UTC 2026-10-08 16:00
        # （⚠️ 这与 `date_stamp` 的「墙上时间当 UTC」是**两套口径**，别混）
        self.assertIsNotNone(d['ts'].tzinfo)
        self.assertEqual(d['ts'].isoformat(), '2026-10-08T16:00:00+00:00')

    def test_float_stamp_is_coerced_to_int(self):
        """源端 float 的 stamp 会被 `int()` —— 否则与存量的 int32 不相等，唯一键插重复。"""
        d = metadata_day_doc('600519', 1791388800.0, '2026-10-09', TURNOVER_DC, 0.1, 0)
        self.assertIsInstance(d['date_stamp'], int)

    def test_payload_column_is_parameterised(self):
        """同一行可以带东财列、也可以带 baostock 列 —— 载荷字段名由调用方给。"""
        a = metadata_day_doc('x', 1, '2026-10-09', TURNOVER_DC, 0.5, 0)
        b = metadata_day_doc('x', 1, '2026-10-09', TURNOVER_BS, 0.5, 0)
        self.assertIn(TURNOVER_DC, a)
        self.assertNotIn(TURNOVER_BS, a)
        self.assertIn(TURNOVER_BS, b)


class TestTurnoverRows(unittest.TestCase):
    def _row(self, **kw):
        base = {'code': '600519', 'date': '2026-10-09 00:00:00',
                'date_stamp': 1791388800, 'TurnoverRate': 0.0123}
        base.update(kw)
        return base

    def test_missing_values_are_skipped_not_zeroed(self):
        """缺值**丢掉**，不写 0 —— 0 是「当天真的零换手」的意思。"""
        rows = [self._row(TurnoverRate=None), {'code': 'x', 'date_stamp': 1}]
        docs, skipped = turnover_rows(rows, 'TurnoverRate', TURNOVER_DC, created_at=0)
        self.assertEqual(docs, [])
        self.assertEqual(skipped, 2)

    def test_above_max_ratio_is_rejected(self):
        """> 1.08 判为「百分比没换算就落库」，丢并计数（用户 2026-10-10 的判据）。"""
        rows = [self._row(TurnoverRate=2.68), self._row(TurnoverRate=MAX_TURNOVER_RATIO + 1e-9)]
        docs, skipped = turnover_rows(rows, 'TurnoverRate', TURNOVER_DC, created_at=0)
        self.assertEqual(docs, [])
        self.assertEqual(skipped, 2)

    def test_boundary_is_inclusive(self):
        docs, skipped = turnover_rows([self._row(TurnoverRate=MAX_TURNOVER_RATIO)],
                                      'TurnoverRate', TURNOVER_DC, created_at=0)
        self.assertEqual(len(docs), 1)
        self.assertEqual(skipped, 0)

    def test_negative_and_non_numeric_are_skipped(self):
        rows = [self._row(TurnoverRate=-0.01), self._row(TurnoverRate='x')]
        _, skipped = turnover_rows(rows, 'TurnoverRate', TURNOVER_DC, created_at=0)
        self.assertEqual(skipped, 2)

    def test_writes_only_the_named_source_field(self):
        """两个源各自只认自己的字段 —— 免得 `stock_valuation` 的行被当东财的写进去。"""
        rows = [{'code': 'x', 'date': '2026-10-09', 'date_stamp': 1,
                 'turnover': 0.02, 'TurnoverRate': 0.9}]
        docs, _ = turnover_rows(rows, 'turnover', TURNOVER_BS, created_at=0)
        self.assertEqual(docs[0][TURNOVER_BS], 0.02)


class TestSourceTable(unittest.TestCase):
    """两个源的声明本身是契约：集合名 / 源字段 / 目标列。"""

    def test_two_sources_two_columns(self):
        self.assertEqual(len(SOURCES), 2)
        dst = [s[2] for s in SOURCES]
        self.assertEqual(sorted(dst), sorted([TURNOVER_DC, TURNOVER_BS]))
        self.assertEqual(len(set(dst)), 2, '两列必须不同，否则互相覆盖')

    def test_unique_key_has_no_revision(self):
        """`revision` 是旧库的历史包袱，明确丢弃（用户 2026-10-10）。"""
        self.assertEqual(tuple(UNIQUE_KEYS), ('code', 'date_stamp'))
        self.assertNotIn('revision', UNIQUE_KEYS)

    def test_collection_name(self):
        self.assertEqual(METADATA_DAY_COLL, 'stock_metadata_day')


def _has_83():
    """能连上 8.3 吗？连不上就跳过真库用例（短超时，别让套件卡 30 秒）。"""
    try:
        from GolemQ.core.mongo import GQ_util_mongodb_client
        from GolemQ.core.settings import GQSETTING
        client = GQ_util_mongodb_client(
            GQSETTING.get_config('MONGODB', 'uri'), serverSelectionTimeoutMS=1500)
        client.admin.command('ping')
        return True
    except Exception:            # noqa: BLE001
        return False


@unittest.skipUnless(_has_83(), 'MongoDB 8.3 不可用 —— 真库用例跳过')
class TestUpsertFieldsDoesNotClobber(unittest.TestCase):
    """**多列共存**：这是「用 `$set` 而不是 `ReplaceOne`」的唯一证明。

    用一次性集合，跑完 drop。⚠️ 绝不碰真集合 —— 这张表的用法就是多任务共写，
    拿真集合当草稿纸会把别人的列抹掉（正是本用例要防的事）。
    """

    COLL = '__tmp_metadata_save_probe__'

    def setUp(self):
        from GolemQ.core.mongo import GQ_util_mongodb_client
        from GolemQ.core.settings import GQSETTING
        self.client = GQ_util_mongodb_client(
            GQSETTING.get_config('MONGODB', 'uri'), serverSelectionTimeoutMS=8000)
        self.coll = self.client['golemq_stock_cn'][self.COLL]
        self.coll.drop()
        self.addCleanup(self.coll.drop)

    def test_second_column_does_not_wipe_the_first(self):
        key = list(UNIQUE_KEYS)
        upsert_fields(self.coll, [{'code': '600519', 'date_stamp': 1791388800,
                                   TURNOVER_DC: 0.0123}], key)
        upsert_fields(self.coll, [{'code': '600519', 'date_stamp': 1791388800,
                                   TURNOVER_BS: 0.0228}], key)
        doc = self.coll.find_one({'code': '600519'}, {'_id': 0})
        self.assertEqual(doc[TURNOVER_DC], 0.0123)
        self.assertEqual(doc[TURNOVER_BS], 0.0228)
        self.assertEqual(self.coll.count_documents({}), 1, '同一键不该插出第二行')

    def test_unique_index_is_created(self):
        upsert_fields(self.coll, [{'code': 'x', 'date_stamp': 1,
                                   TURNOVER_DC: 0.01}], list(UNIQUE_KEYS))
        idx = {i['name']: i for i in self.coll.list_indexes()}
        self.assertTrue(idx['code_1_date_stamp_1'].get('unique'))

    def test_save_collection_would_wipe_it(self):
        """**反证**：证明「用 `$set`」不是洁癖 —— `ReplaceOne` 真会抹掉别列。"""
        from GolemQ.datasource.writer import save_collection
        key = list(UNIQUE_KEYS)
        upsert_fields(self.coll, [{'code': 'x', 'date_stamp': 1,
                                   TURNOVER_DC: 0.01, TURNOVER_BS: 0.02}], key)
        save_collection(self.coll, [{'code': 'x', 'date_stamp': 1,
                                     TURNOVER_DC: 0.01}], key)
        doc = self.coll.find_one({'code': 'x'}, {'_id': 0})
        self.assertNotIn(TURNOVER_BS, doc, '整文档替换应当抹掉未出现在本次行里的列')


if __name__ == '__main__':
    unittest.main(verbosity=2)
