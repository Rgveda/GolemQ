#!/usr/bin/env python3
# coding: utf-8
"""**只读**导出 4.4 的 review 金标准 + 8.3 的输入 K 线，供搬迁对拍。

为什么要它
==========
旧树手写代码没有测试，搬迁时唯一可靠的裁判是「老树当时算出来的行」。
本工具从 4.4 **只读**导出样本，同时导出 8.3 对应区间的 K 线作为**确定性输入**，
这样新实现可以在不连 4.4、不重跑老管线的前提下离线复算。

    python tools/dump_review_golden.py
    python tools/dump_review_golden.py --codes 600519,510300 --start 2024-01-01 --end 2026-09-30
    python tools/dump_review_golden.py --out GolemQ/test_cases/pipeline/fixtures

4.4 通道只用于一次性搬运（`DECISIONS.md` D33 修正条 / D34）；
**运行时代码禁止 import `core.migrate44`**，本工具也不被任何运行时路径引用。

输出
====
``<out>/<family>/<code>.parquet`` + ``<out>/inputs/<code>_<freq>.parquet`` +
``<out>/manifest.json``。manifest 里记每个文件的来源集合、过滤条件、行数与 sha256。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import pandas as pd

# 允许 `python tools/dump_review_golden.py` 直接跑（sys.path[0] 是 tools/）
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

DEFAULT_CODES = ['600519', '301308', '510300', '159915', '399001', '000001.XSHG']
OLD_DB = 'golemq'

#: 老树把 ETF 归在 `index_*`（D51 §一#5 实测 94–98%）。这里按市场类型选旧前缀。
OLD_PREFIX = {'stock_cn': 'stock', 'index_cn': 'index', 'etf_cn': 'index',
              'fund_cn': 'index'}

#: (family, 旧集合后缀, 日期字段, 备注, 新树目标后缀)
#:
#: ⚠️ `stock_reality_features` / `index_reality_features` 的**尺度是 60min**
#: （实测时间戳 10:30 / 11:30 / 14:00 / 15:00），但名字没有频率后缀。
#: 新树统一并入 `*_reality_feat_60min`，不再保留无后缀旧表。
OLD_FAMILIES = [
    ('day', 'reality_feat_day', 'date', '--review --daily 日族', 'reality_feat_day'),
    ('reality60', 'reality_features', 'date', '--review --reality 的 60min 小表',
     'reality_feat_60min'),
    ('minor60', 'metadata', 'date', 'metadata revision=zw_reality（60min 快照）',
     'reality_feat_60min'),
    ('cv60', 'reality_feat_60min', 'date', '模型层 60min 大表（结构性核对）',
     'reality_feat_60min'),
    ('cv15', 'reality_feat_15min', 'date', '仅 CV 三列（结构性核对）',
     'reality_feat_15min'),
    ('coldstart', 'coldstart_reviews', 'date', 'coldstart 子集',
     'coldstart_reviews'),
]

#: coldstart 只导出用户点名的列（其余列不搬，见计划）。
COLDSTART_COLUMNS = [
    'STAGE', 'BST_STG_BF', 'm9tBsl', 'M9T', 'M9T_LAG', 'M9T_MAJ',
    'poly9_MAJ', 'poly9pwr3_MAJ',
    'PROPHET_LB', 'PROPHET_UB', 'PROPHET_CLR_NORM',
    'PROPHET_LB_MAJ', 'PROPHET_UB_MAJ', 'PROPHET_CLR_NORM_MAJ',
]


def _coerce_parquet(df: pd.DataFrame) -> pd.DataFrame:
    """把混合类型（如 bool/float 共存的 object 列）转成可写 parquet 的类型。"""
    out = df.copy()
    for c in out.columns:
        if out[c].dtype != object:
            continue
        try:
            out[c] = pd.to_numeric(out[c])
        except (TypeError, ValueError):
            out[c] = out[c].astype(str)
    return out


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _frame_from_cursor(cursor) -> pd.DataFrame:
    df = pd.DataFrame(list(cursor))
    if df.empty:
        return df
    df = df.drop(columns=[c for c in ('_id',) if c in df.columns])
    return df


def _query(db, coll: str, code: str, start: str, end: str,
           date_field: str, extra: dict | None = None):
    q = {'code': code}
    if start:
        q[date_field] = {'$gte': start}
    if end:
        q.setdefault(date_field, {})
        if isinstance(q[date_field], dict):
            q[date_field]['$lte'] = end
        else:
            q[date_field] = {'$gte': start, '$lte': end}
    if extra:
        q.update(extra)
    return _frame_from_cursor(db[coll].find(q).sort(date_field, 1))


def _dump_family(db, old_prefix, suffix, family, code, start, end,
                 out: Path, date_field='date', extra=None):
    coll = f'{old_prefix}_{suffix}'
    if coll not in db.list_collection_names():
        return None
    df = _query(db, coll, code, start, end, date_field, extra=extra)
    if df.empty:
        return None
    if family == 'coldstart':
        keep = [c for c in COLDSTART_COLUMNS if c in df.columns]
        meta = [c for c in ('code', 'date', 'datetime', 'time_stamp',
                            'date_stamp', 'created_at') if c in df.columns]
        df = df[meta + keep]
    path = out / family / f'{code}.parquet'
    path.parent.mkdir(parents=True, exist_ok=True)
    _coerce_parquet(df).to_parquet(path, index=False)
    return {'collection': coll, 'rows': len(df), 'file': str(path.relative_to(out)),
            'sha256': _sha256(path)}


def _dump_inputs(market, code, start, end, out: Path):
    records = {}
    for freq in ('day', '60min', '15min'):
        if freq == 'day':
            res, _ = market.get_kline_price_v3(code, start=start, end=end,
                                               realtime=False, verbose=False)
        else:
            res, _ = market.get_kline_price_min(code, start=start, end=end,
                                                frequency=freq, realtime=False,
                                                verbose=False)
        data = None if res is None else getattr(res, 'data', None)
        if data is None or len(data) == 0:
            continue
        df = data.reset_index()
        path = out / 'inputs' / f'{code}_{freq}.parquet'
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
        records[freq] = {'rows': len(df), 'file': str(path.relative_to(out)),
                         'sha256': _sha256(path)}
    return records


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--codes', default=','.join(DEFAULT_CODES))
    ap.add_argument('--start', default='2024-01-01')
    ap.add_argument('--end', default='2026-09-30')
    ap.add_argument('--out', default='GolemQ/test_cases/pipeline/fixtures')
    args = ap.parse_args(argv)

    from GolemQ.core.migrate44 import db44
    from GolemQ.markets.StockCN.symbol import is_stock_cn
    from GolemQ import get_active_market

    db = db44(OLD_DB)
    market = get_active_market()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    manifest = {'old_db': OLD_DB, 'start': args.start, 'end': args.end,
                'codes': {}, 'generated_at': pd.Timestamp.now().isoformat()}
    for code in [c.strip() for c in args.codes.split(',') if c.strip()]:
        market_type = is_stock_cn(code)[1]
        old_prefix = OLD_PREFIX.get(market_type, 'index')
        # ⚠️ 4.4 的 index 集合用**裸 6 位**（`000001` 既可能是上证指数、也可能是
        # 平安银行 —— D51 实测过的歧义）。金标准查询用裸码，8.3 输入用完整码。
        old_code = code.split('.')[0] if '.' in code else code
        rec = {'market_type': market_type, 'old_prefix': old_prefix,
               'old_code': old_code, 'old': {}, 'inputs': {}}
        for family, suffix, date_field, _note, target in OLD_FAMILIES:
            extra = {'revision': 'zw_reality'} if family == 'minor60' else None
            try:
                got = _dump_family(db, old_prefix, suffix, family, old_code,
                                   args.start, args.end, out,
                                   date_field=date_field, extra=extra)
            except Exception as exc:            # noqa: BLE001 单个集合失败不毁整批
                got = {'error': f'{type(exc).__name__}: {exc}'}
            if got is not None:
                got['target'] = target
                rec['old'][family] = got
        try:
            rec['inputs'] = _dump_inputs(market, code, args.start, args.end, out)
        except Exception as exc:                # noqa: BLE001
            rec['inputs'] = {'error': f'{type(exc).__name__}: {exc}'}
        manifest['codes'][code] = rec
        print(f'[golden] {code} ({market_type} -> {old_prefix}_*) '
              f'families={sorted(rec["old"])} inputs={sorted(rec["inputs"])}')

    mf = out / 'manifest.json'
    mf.write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                  encoding='utf-8')
    print(f'[golden] manifest -> {mf}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
