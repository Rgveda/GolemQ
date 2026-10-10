#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""`stock_metadata_day` 的落库契约。

为什么这份用例值得存在
======================
**① `date_stamp` 的口径**。这张表住在 8.3，而实测**三种口径并存**
（2026-10-10）：8.3 自己的表是「北京零点的真实 UTC」、4.4 `stock_ranking` 是
「墙上时间当 UTC」（差 +8h）、4.4 `stock_valuation` **两套混着**（2026-09-01：
5131 行真实 UTC / 61 行墙上）。**差 8 小时不会报错**，只会静默查空或插重复行。
本表按 8.3 口径重算，锚点用 `stock_day` 的实测值钉住。

**② 「只 `$set` 自己的列」**。这张表是**多列共用一个文档**的（东财一列、
baostock 一列，以后还有别的任务）。一旦哪个写路径用了 `save_collection`
（`ReplaceOne` 整文档替换），别人的列会被**静默抹掉**。
`TestUpsertFieldsDoesNotClobber` 是这条的钉子 —— 它同时验证 `save_collection`
**确实会**抹（反证危害真实存在）。

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
    UNIQUE_KEYS, _date_range, day_date, day_stamp, metadata_day_doc, turnover_rows,
)


class TestDayStampConvention(unittest.TestCase):
    """`date_stamp` 必须与 **8.3 的邻居**一致（差 8h 就静默 join 错）。"""

    def test_matches_stock_day(self):
        """锚点取自 8.3 实测：`stock_day`/`stock_adj` 里 `date='2026-10-08'`
        的 `date_stamp` = 1791388800。"""
        self.assertEqual(day_stamp('2026-10-08'), 1791388800)

    def test_is_not_the_wall_clock_convention(self):
        """反证：**不是**「墙上时间当 UTC」那套（4.4 `stock_ranking` 用的是它）。

        `datetime.strptime('2026-10-08').replace(tzinfo=utc).timestamp()` = 1791417600，
        比正确值**大 8 小时**；照搬源端就会落成这个。
        """
        from datetime import datetime, timezone
        wall = int(datetime.strptime('2026-10-08', '%Y-%m-%d')
                   .replace(tzinfo=timezone.utc).timestamp())
        self.assertEqual(wall - day_stamp('2026-10-08'), 8 * 3600)
        self.assertNotEqual(day_stamp('2026-10-08'), wall)

    def test_accepts_both_source_date_formats(self):
        """源端两种格式都收（`stock_ranking` 带时分秒、`stock_valuation` 裸日期）。"""
        self.assertEqual(day_stamp('2026-10-08 15:00:00'), day_stamp('2026-10-08'))
        self.assertEqual(day_date('2026-10-08 15:00:00'), '2026-10-08')

    def test_ts_is_the_real_instant(self):
        """`ts` 是 UTC-aware 的真实时刻；与同为真实时刻的 `date_stamp` 差 8h
        （北京零点 vs UTC 零点）—— 两个字段各司其职，别当同一个值。"""
        d = metadata_day_doc('600519', '2026-10-09', TURNOVER_DC, 0.0123, created_at=0)
        self.assertEqual(d['ts'].isoformat(), '2026-10-08T16:00:00+00:00')
        self.assertEqual(d['date_stamp'], day_stamp('2026-10-09'))


class TestDateRange(unittest.TestCase):
    """范围过滤**按源端 date 格式**造 —— 不能一刀切补 ' 00:00:00'。"""

    def test_datetime_format_gets_time_part(self):
        self.assertEqual(_date_range('datetime', '2026-09-01', '2026-09-30'),
                         {'$gte': '2026-09-01 00:00:00', '$lte': '2026-09-30 23:59:59'})

    def test_bare_date_format_gets_none(self):
        """⚠️ 裸日期源**不能**补时分秒：字符串比较下 `'2026-09-30'` **小于**
        `'2026-09-30 00:00:00'`，补了就把当天整段静默排掉。"""
        self.assertEqual(_date_range('date', '2026-09-01', '2026-09-30'),
                         {'$gte': '2026-09-01', '$lte': '2026-09-30'})

    def test_no_bounds_is_empty(self):
        self.assertEqual(_date_range('date'), {})

    def test_each_source_declares_its_date_format(self):
        self.assertEqual({s[0]: s[3] for s in SOURCES},
                         {'stock_ranking': 'datetime', 'stock_valuation': 'date'})


class TestMetadataDayDoc(unittest.TestCase):
    def test_field_set_and_types(self):
        d = metadata_day_doc('600519', '2026-10-09', TURNOVER_DC, 0.0123,
                             created_at=1791400000)
        # ⚠️ sorted() 按码点：大写 T 排在小写前 ⇒ 换手率那列在**最前**
        self.assertEqual(sorted(d), [TURNOVER_DC, 'code', 'created_at', 'date',
                                     'date_stamp', 'datetime', 'ts'])
        self.assertIsInstance(d['date_stamp'], int)
        self.assertEqual(d['date'], '2026-10-09')
        self.assertEqual(d['datetime'], '2026-10-09 00:00:00')
        self.assertEqual(d[TURNOVER_DC], 0.0123)

    def test_payload_column_is_parameterised(self):
        """同一行可以带东财列、也可以带 baostock 列 —— 载荷字段名由调用方给。"""
        a = metadata_day_doc('x', '2026-10-09', TURNOVER_DC, 0.5, 0)
        b = metadata_day_doc('x', '2026-10-09', TURNOVER_BS, 0.5, 0)
        self.assertIn(TURNOVER_DC, a)
        self.assertNotIn(TURNOVER_BS, a)
        self.assertIn(TURNOVER_BS, b)


class TestTurnoverRows(unittest.TestCase):
    def _row(self, **kw):
        base = {'code': '600519', 'date': '2026-10-09 00:00:00', 'TurnoverRate': 0.0123}
        base.update(kw)
        return base

    def test_missing_values_are_skipped_not_zeroed(self):
        """缺值**丢掉**，不写 0 —— 0 是「当天真的零换手」的意思。"""
        rows = [self._row(TurnoverRate=None), {'code': 'x', 'date': '2026-10-09'}]
        docs, skipped = turnover_rows(rows, 'TurnoverRate', TURNOVER_DC, created_at=0)
        self.assertEqual(docs, [])
        self.assertEqual(skipped, 2)

    def test_row_without_date_is_skipped(self):
        """没有 `date` 就推不出 `date_stamp` ⇒ 丢（源端 stamp 不再被信任）。"""
        _, skipped = turnover_rows([{'code': '1', 'TurnoverRate': 0.01}],
                                   'TurnoverRate', TURNOVER_DC)
        self.assertEqual(skipped, 1)

    def test_above_max_ratio_is_rejected(self):
        """> 1.08 判为「百分比没换算就落库」，丢并计数（用户 2026-10-10 的判据）。"""
        rows = [self._row(TurnoverRate=2.68),
                self._row(TurnoverRate=MAX_TURNOVER_RATIO + 1e-9)]
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
        rows = [{'code': 'x', 'date': '2026-10-09', 'turnover': 0.02, 'TurnoverRate': 0.9}]
        docs, _ = turnover_rows(rows, 'turnover', TURNOVER_BS, created_at=0)
        self.assertEqual(docs[0][TURNOVER_BS], 0.02)


class TestSourceTable(unittest.TestCase):
    """源的声明本身是契约：集合名 / 源字段 / 目标列 / date 格式。"""

    def test_two_sources_two_columns(self):
        self.assertEqual(len(SOURCES), 2)
        dst = [s[2] for s in SOURCES]
        self.assertEqual(sorted(dst), sorted([TURNOVER_DC, TURNOVER_BS]))
        self.assertEqual(len(set(dst)), 2, '两列必须不同，否则互相覆盖')

    def test_columns_keep_the_source_names(self):
        """列名**保留源端原名**（用户 2026-10-10）—— 好让 `calcuChip` 读
        `FIELD.TURNOVER_RATE` 那侧一行都不用改。"""
        for coll, src_field, dst_field, _ in SOURCES:
            with self.subTest(coll=coll):
                self.assertEqual(src_field, dst_field)

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
        stamp = day_stamp('2026-10-09')
        upsert_fields(self.coll, [{'code': '600519', 'date_stamp': stamp,
                                   TURNOVER_DC: 0.0123}], key)
        upsert_fields(self.coll, [{'code': '600519', 'date_stamp': stamp,
                                   TURNOVER_BS: 0.0228}], key)
        doc = self.coll.find_one({'code': '600519'}, {'_id': 0})
        self.assertEqual(doc[TURNOVER_DC], 0.0123)
        self.assertEqual(doc[TURNOVER_BS], 0.0228)
        self.assertEqual(self.coll.count_documents({}), 1, '同一键不该插出第二行')

    def test_unique_index_is_created(self):
        upsert_fields(self.coll, [{'code': 'x', 'date_stamp': day_stamp('2026-10-09'),
                                   TURNOVER_DC: 0.01}], list(UNIQUE_KEYS))
        idx = {i['name']: i for i in self.coll.list_indexes()}
        self.assertTrue(idx['code_1_date_stamp_1'].get('unique'))

    def test_save_collection_would_wipe_it(self):
        """**反证**：证明「用 `$set`」不是洁癖 —— `ReplaceOne` 真会抹掉别列。"""
        from GolemQ.datasource.writer import save_collection
        key = list(UNIQUE_KEYS)
        stamp = day_stamp('2026-10-09')
        upsert_fields(self.coll, [{'code': 'x', 'date_stamp': stamp,
                                   TURNOVER_DC: 0.01, TURNOVER_BS: 0.02}], key)
        save_collection(self.coll, [{'code': 'x', 'date_stamp': stamp,
                                     TURNOVER_DC: 0.01}], key)
        doc = self.coll.find_one({'code': 'x'}, {'_id': 0})
        self.assertNotIn(TURNOVER_BS, doc, '整文档替换应当抹掉未出现在本次行里的列')


if __name__ == '__main__':
    unittest.main(verbosity=2)
