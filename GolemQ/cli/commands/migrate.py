# coding:utf-8
"""**8.3 内部**一次性合并入口（**不连 4.4**）。

⚠️ 2026-11-18 起（`DECISIONS.md` **D83**）：所有 4.4 访问只允许留在 `tools/`。
原先的 `--migrate-turnover` / `--migrate-index-list` / `--migrate-reality-feat`
（都要连 4.4）已从运行时删除；本模块只保留 **8.3 → 8.3** 的元数据合并。

保留的一条：`--migrate-metadata-merge` / `--migrate-metadata-verify`
（`stock_metadata_day` → `stock_reality_feat_day`，D50）。
"""
from __future__ import annotations

import sys

from ._registry import Command


def add_metadata_merge_arguments(parser) -> None:
    parser.add_argument('--migrate-metadata-merge',
                        help="把 8.3 的 stock_metadata_day **原样**搬进 "
                             "stock_reality_feat_day（**8.3 内部合并**，不连 4.4）。"
                             "幂等（只 $set、键 (code, date_stamp)），中途挂了重跑即可。"
                             "",
                        action='store_true',
                        default=False)
    parser.add_argument('--migrate-metadata-verify',
                        help="**只读**核对：抽样比对两表在 (code, date_stamp) 上"
                             "是否逐字段相同。⚠️ 合并跑完必须先跑它再谈 drop。",
                        action='store_true',
                        default=False)
    # NOTE: `--migrate-limit` 是**内部/调试**旗子，只在
    # `python -m GolemQ.cli.internal` 里注册（见 `commands/_internal.py`）。


def run_migrate_metadata_merge(args) -> None:
    """`--migrate-metadata-merge` / `--migrate-metadata-verify` 的处理体。"""
    from GolemQ.markets.StockCN.metadata_save import (
        merge_metadata_into_feat, verify_metadata_merge)

    if args.migrate_metadata_verify:
        out = verify_metadata_merge()
        # **不符就必须非零退出** —— 否则「核对过了」会被读成「核对通过」。
        if out['missing'] or out['mismatch']:
            print('[migrate] ❌ 核对不通过：缺 {} / 不符 {}'.format(
                out['missing'], out['mismatch']), file=sys.stderr)
            sys.exit(1)
        return

    stats = merge_metadata_into_feat(dry_run=getattr(args, 'migrate_dry_run', False),
                                     limit=getattr(args, 'migrate_limit', None))
    if stats['skipped']:
        print('[migrate] ⚠️ 有丢弃行 —— 见上面的说明', file=sys.stderr)


MIGRATE_METADATA_MERGE = Command(
    name='migrate-metadata-merge',
    flags=('migrate_metadata_merge', 'migrate_metadata_verify'),
    add_arguments=add_metadata_merge_arguments,
    run=run_migrate_metadata_merge,
    internal_only=True,
)
