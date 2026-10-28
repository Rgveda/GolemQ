#!/usr/bin/env python3
# coding: utf-8
"""把 8.3 库里的老列名同步改成伪驼峰（`$rename`，不重算）。

背景：D77 把 39 个老树真值常量统一成伪驼峰。4.4 是**只读源**（D75），
本工具**只动 8.3**。默认 dry-run；`--apply` 才写。

做法：逐集合 `$sample` 一批文档，收集实际存在的键；命中映射的键就
`update_many({old: {'$exists': True}}, {'$rename': {old: new}})`。

```bash
python tools/rename_legacy_fields.py            # 只报告
python tools/rename_legacy_fields.py --apply    # 真正改名
```
"""
from __future__ import annotations

import argparse
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

#: 老树值 → 伪驼峰（D77 统一；只用于 8.3）。
MAPPING = {
    'ATR_SUPTRD': 'atrSuptrd',
    'ATR_SUPTRD_LAG_WEK': 'atrSuptrdLagWek',
    'ATR_Stopline': 'atrStopline',
    'ATR_Stopline_LAG_WEK': 'atrStoplineLagWek',
    'BBW_diff_1T': 'bbwDiff1T',
    'CCI_NORM': 'cciNorm',
    'DEA_MAJ': 'deaMaj',
    'DRDN_RATIO': 'drdnRatio',
    'FIN_COST9': 'finCost9',
    'FIN_PROF': 'finProf',
    'HMPWR12': 'hmpwr12',
    'LRC_PRICE': 'lrgPrice',
    'LRG_BAND_LAG': 'lrgBandLag',
    'LRG_TRD_LAG': 'lrgTrdLag',
    'MA90_CLR': 'ma90Clr',
    'MA90_CLR_LAG': 'ma90ClrLag',
    'MA90_CLR_RET_GAR': 'ma90ClrRetGar',
    'MACD_SX_BF': 'macdSxBf',
    'MAXFACTOR': 'maxFactor',
    'MPWR12': 'mpwr12',
    'MPWR3': 'mpwr3',
    'MPWR3_LAG_MAJ': 'mpwr3LagMaj',
    'MPWR3_MAJ_MED': 'mpwr3MajMed',
    'MPWR3_MED': 'mpwr3Med',
    'MPWR3_PEAK_HI_BF': 'mpwr3PeakHiBf',
    'MPWR3_PEAK_LO_BF': 'mpwr3PeakLoBf',
    'MTMMA_NORM': 'mtmmaNorm',
    'MTMMA_NORM_MAJ': 'mtmmaNormMaj',
    'MTMMA_NORM_WEK': 'mtmmaNormWek',
    'MTM_NORM_WEK': 'mtmNormWek',
    'PEAK_HIGH_LAG': 'peakHighLag',
    'PEAK_HI_BKWRD': 'peakHiBkwrd',
    'PEAK_LO_BKWRD': 'peakLoBkwrd',
    'PEAK_TOWARDS_BF': 'peakTowardsBf',
    'PEAK_TOWARDS_LAG': 'peakTowardsLag',
    'VOL_NORM': 'volNorm',
    'VOL_NORM_LAG': 'volNormLag',
    'last_close': 'lastClose',
    'poly9_LAG_MAJ': 'poly9LagMaj',
}


def keys_of(coll, n=300):
    try:
        cur = coll.aggregate([
            {'$sample': {'size': n}},
            {'$project': {'k': {'$objectToArray': '$$ROOT'}}},
            {'$unwind': '$k'},
            {'$group': {'_id': '$k.k'}},
        ], allowDiskUse=True)
        return {d['_id'] for d in cur}
    except Exception:               # noqa: BLE001 空集合/权限问题 → 跳过
        return set()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--apply', action='store_true', help='真正写库（默认 dry-run）')
    ap.add_argument('--sample', type=int, default=200, help='每集合抽样文档数')
    ap.add_argument('--all', action='store_true',
                    help='扫描全部集合（默认只扫 reality_feat / metadata_day / snapshot）')
    args = ap.parse_args()

    from GolemQ import get_active_market
    db = get_active_market().get_database()

    hits = {}
    for name in sorted(db.list_collection_names()):
        if name.startswith('system.'):
            continue
        if not args.all and not any(k in name for k in
                                    ('reality_feat', 'metadata_day', 'snapshot')):
            continue
        keys = keys_of(db[name], args.sample)
        for old, new in MAPPING.items():
            if old in keys:
                hits.setdefault(name, []).append((old, new))

    if not hits:
        print('8.3 目标集合（每集抽样 %d 文档）：没有任何老列名命中（无需迁移）'
              % args.sample)
        return 0

    total = 0
    for name, pairs in hits.items():
        coll = db[name]
        for old, new in pairs:
            q = {old: {'$exists': True}}
            n = coll.count_documents(q, limit=1) and 1 or 0
            print('  %-32s %-20s -> %-20s 存在=%s' % (name, old, new, bool(n)))
            total += 1
            if args.apply:
                coll.update_many(q, {'$rename': {old: new}})
    print('%s %d 个字段（%d 个集合）' % ('已改名' if args.apply else '将改名',
                                       total, len(hits)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
