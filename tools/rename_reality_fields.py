#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 `*_reality_feat_*` 里的**遗留大写列名**原地改成伪驼峰缩写。

为什么是 `$rename` 而不是重算/重落库
====================================
这些列是 review/coldstart 已经算好的值，改名不改变数值 —— Mongo 的
`$rename` 是**原地重命名**，不读不写业务值；重算反而会引入新数据/时间成本。

映射（2026-10-24）
==================
60min 族：

    STAGE            -> stage
    BST_STG_BF       -> bstStgBf
    EP_STG_BF        -> epStgBf
    PROPHET_LB       -> prophetLb
    PROPHET_UB       -> prophetUb
    PROPHET_CLR_NORM -> prophetClrNorm
    PROPHET_LB_MAJ   -> prophetLbMaj
    PROPHET_UB_MAJ   -> prophetUbMaj
    PROPHET_CLR_NORM_MAJ -> prophetClrNormMaj
    poly9_MAJ        -> poly9Maj
    poly9pwr3_MAJ    -> poly9pwr3Maj
    SE               -> se

day 族：

    ZPLagMajR        -> zPLagMajR

⚠️ 只改**新树 review/coldstart 产出的列**。ChartScan 的 `ClsMaj` / `Class` /
`ClsMin` 等是 4.4 存量契约，不在本脚本范围。

用法：

    python tools/rename_reality_fields.py --dry-run   # 只看命中数（默认）
    python tools/rename_reality_fields.py --apply     # 真正改名
"""
from __future__ import annotations

import argparse
import os
import sys

# `python tools/xxx.py` 的 sys.path[0] 是 tools/，不是仓库根。
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from GolemQ.core.constants import MARKET_TYPE
from GolemQ.markets.StockCN.reality_save import reality_feat_name

#: 族 → {旧名: 新名}
RENAMES = {
    '60min': {
        'STAGE': 'stage',
        'BST_STG_BF': 'bstStgBf',
        'EP_STG_BF': 'epStgBf',
        'PROPHET_CLR_NORM': 'prophetClrNorm',
        'PROPHET_CLR_NORM_MAJ': 'prophetClrNormMaj',
        'poly9_MAJ': 'poly9Maj',
        'poly9pwr3_MAJ': 'poly9pwr3Maj',
        'SE': 'se',
        'prophetLb': 'prophetLbD',
        'prophetUb': 'prophetUbD',
        'prophetLbMaj': 'prophetLbMajD',
        'prophetUbMaj': 'prophetUbMajD',
        'garP': 'garPD',
    },
    'day': {
        'ZPLagMajR': 'zPLagMajR',
        'poly9MajR': 'poly9MajD',
        'poly9WekR': 'poly9WekD',
        'rTrePrcMajR': 'rTrePrcMajD',
        'rTreSlopMajR': 'rTreSlopMajD',
    },
}

MARKET_TYPES = (MARKET_TYPE.STOCK_CN, MARKET_TYPE.ETF_CN,
                MARKET_TYPE.INDEX_CN, MARKET_TYPE.FUND_CN)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--apply', action='store_true',
                    help='真正执行 $rename；不加则只统计（dry-run）')
    ap.add_argument('--dry-run', action='store_true',
                    help='只统计（默认行为；显式写出来更清楚）')
    ap.add_argument('--unset-60min-poly9maj', action='store_true',
                    help='删除 60min 的 poly9Maj（day 的 poly9MajD 已覆盖同一值）')
    args = ap.parse_args()
    if args.apply and args.dry_run:
        ap.error('--apply 与 --dry-run 不能同时给')

    from GolemQ import get_active_market
    db = get_active_market().get_database()
    existing = set(db.list_collection_names())
    total_hit = total_done = 0
    conflicts = []

    for family, mapping in RENAMES.items():
        for mt in MARKET_TYPES:
            name = reality_feat_name(family, mt)
            if name not in existing:
                continue
            coll = db[name]
            old_filter = {'$or': [{old: {'$exists': True}} for old in mapping]}
            hits = coll.count_documents(old_filter)
            if not hits:
                continue
            # 目标列已存在 = 会覆盖别人的值 -> 先报冲突、不执行。
            bad = {old: coll.count_documents(
                       {old: {'$exists': True}, new: {'$exists': True}})
                   for old, new in mapping.items()}
            bad = {k: v for k, v in bad.items() if v}
            print('{:<28s} 命中 {:>8d} 行{}'.format(
                name, hits,
                '  冲突: {}'.format(bad) if bad else ''))
            if bad:
                conflicts.append((name, bad))
                continue
            total_hit += hits
            if args.apply:
                res = coll.update_many(old_filter, {'$rename': mapping})
                total_done += res.modified_count
                print('  -> renamed {} docs'.format(res.modified_count))

    if args.unset_60min_poly9maj:
        for mt in MARKET_TYPES:
            name = reality_feat_name('60min', mt)
            if name not in existing:
                continue
            coll = db[name]
            hits = coll.count_documents({'poly9Maj': {'$exists': True}})
            if hits:
                print('{:<28s} poly9Maj 命中 {:>8d} 行{}'.format(
                    name, hits, '' if args.apply else '（dry-run）'))
                if args.apply:
                    res = coll.update_many({'poly9Maj': {'$exists': True}},
                                           {'$unset': {'poly9Maj': ''}})
                    print('  -> unset {} docs'.format(res.modified_count))
                total_hit += hits

    if conflicts:
        print('\n⚠️ 有目标列已存在，未改：')
        for name, bad in conflicts:
            print('  {}: {}'.format(name, bad))
        return 1
    if args.apply:
        print('\n完成：改 {} 行（命中 {} 行）'.format(total_done, total_hit))
    else:
        print('\n[dry-run] 命中 {} 行；加 --apply 执行'.format(total_hit))
    return 0


if __name__ == '__main__':
    sys.exit(main())
