# coding:utf-8
"""**一次性搬运**入口：从 4.4 取材落进 8.3。

⚠️ 本模块是 `core/migrate44.py` 那条只读通道的**唯一**消费者。
`D33` 曾在 2026-10-10 把整条通道删掉（理由「搬运完成」），**那个前提是错的** ——
元数据类集合（`stock_ranking` / `stock_valuation`）从没搬过。现在按需恢复，
用途限定为「一次性搬运」，见 `DECISIONS.md` 的 D33 修正条。

两条：`--migrate-turnover`（换手率）与 `--migrate-index-list`（指数名单）。
"""
from __future__ import annotations

import sys

from ._registry import Command


def add_turnover_arguments(parser) -> None:
    parser.add_argument('--migrate-turnover',
                        help="把 4.4 的换手率搬进 8.3 stock_reality_feat_day"
                             "（两个源各写一列，**保留源端原字段名**："
                             "东财 TurnoverRate / baostock turnover）",
                        action='store_true',
                        default=False)
    parser.add_argument('--migrate-since',
                        help="只搬该日之后的（YYYY-MM-DD）。不传 = 全量",
                        type=str, default=None)
    parser.add_argument('--migrate-until',
                        help="只搬该日之前的（YYYY-MM-DD）",
                        type=str, default=None)
    parser.add_argument('--migrate-dry-run',
                        help="只读 4.4 并统计，**不写 8.3**（先量规模与丢弃量用）",
                        action='store_true',
                        default=False)


def run_migrate_turnover(args) -> None:
    """`--migrate-turnover` 的处理体：只做参数校验，干活在 `metadata_save`。"""
    from GolemQ.markets.StockCN.metadata_save import REALITY_FEAT_DAY_COLL, save_turnover

    try:
        out = save_turnover(since=args.migrate_since,
                            until=args.migrate_until,
                            dry_run=args.migrate_dry_run)
    except Exception as exc:                     # noqa: BLE001
        # 连不上 4.4 是最可能的失败。**要说清它推的是哪个地址** ——
        # 地址是从 8.3 的 uri 推出来的（同机换端口），推错了会指到别的主机。
        from GolemQ.core.migrate44 import mongo44_uri
        print('[migrate] 失败：{}: {}'.format(type(exc).__name__, exc), file=sys.stderr)
        print('[migrate] 4.4 地址由 8.3 的 uri 推导得到：{}'.format(mongo44_uri()),
              file=sys.stderr)
        print('[migrate] 若 4.4 不在同一台机，请改 ~/.GolemQ/settings/config.ini',
              file=sys.stderr)
        sys.exit(1)

    total_read = sum(s['read'] for s in out.values())
    total_written = sum(s['written'] for s in out.values())
    total_skipped = sum(s['skipped'] for s in out.values())
    if args.migrate_dry_run:
        print('[migrate] 干跑：读 {} / 可写 {} / 丢弃 {}（【没有写库】）'.format(
            total_read, total_written, total_skipped))
    else:
        print('[migrate] 完成 → {}：读 {} / 写 {} / 丢弃 {}'.format(
            REALITY_FEAT_DAY_COLL, total_read, total_written, total_skipped))
    if total_skipped:
        # 丢行**必须显眼** —— 静默丢行会让「搬完了」这句话没有依据
        print('[migrate] ⚠️ 丢弃 {} 行（缺值 / 非数字 / 换手率 > 1.08 判为未换算）'.format(
            total_skipped))


MIGRATE_TURNOVER = Command(
    name='migrate-turnover',
    flags=('migrate_turnover',),
    add_arguments=add_turnover_arguments,
    run=run_migrate_turnover,
)


def add_index_list_arguments(parser) -> None:
    parser.add_argument('--migrate-index-list',
                        help="把 4.4 的 quantaxis.index_list **补**进 8.3 的 index_list"
                             "（**只补缺、不覆盖** —— 4.4 的 pre_close 早已过期，"
                             "盖掉 pytdx 刚取的新值是净损失）。"
                             "配合 --migrate-dry-run 干跑。",
                        action='store_true',
                        default=False)


def run_migrate_index_list(args) -> None:
    """`--migrate-index-list` 的处理体：只做参数校验，干活在 `refdata_save`。"""
    from GolemQ.markets.StockCN.refdata_save import save_index_list_from44

    try:
        save_index_list_from44(dry_run=args.migrate_dry_run)
    except Exception as exc:                     # noqa: BLE001
        from GolemQ.core.migrate44 import mongo44_uri
        print('[migrate] 失败：{}: {}'.format(type(exc).__name__, exc), file=sys.stderr)
        print('[migrate] 4.4 地址由 8.3 的 uri 推导得到：{}'.format(mongo44_uri()),
              file=sys.stderr)
        sys.exit(1)


MIGRATE_INDEX_LIST = Command(
    name='migrate-index-list',
    flags=('migrate_index_list',),
    add_arguments=add_index_list_arguments,
    run=run_migrate_index_list,
)


def add_metadata_merge_arguments(parser) -> None:
    parser.add_argument('--migrate-metadata-merge',
                        help="把 8.3 的 stock_metadata_day **原样**搬进 "
                             "stock_reality_feat_day（**8.3 内部合并**，不连 4.4）。"
                             "幂等（只 $set、键 (code, date_stamp)），中途挂了重跑即可。"
                             "配合 --migrate-dry-run 先量规模、--migrate-limit 试跑。",
                        action='store_true',
                        default=False)
    parser.add_argument('--migrate-metadata-verify',
                        help="**只读**核对：抽样比对两表在 (code, date_stamp) 上"
                             "是否逐字段相同。⚠️ 合并跑完必须先跑它再谈 drop。",
                        action='store_true',
                        default=False)
    parser.add_argument('--migrate-limit',
                        help="只搬前 N 行（试跑用）。不传 = 全量",
                        type=int, default=None)


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

    stats = merge_metadata_into_feat(dry_run=args.migrate_dry_run,
                                     limit=args.migrate_limit)
    if stats['skipped']:
        print('[migrate] ⚠️ 有丢弃行 —— 见上面的说明', file=sys.stderr)


MIGRATE_METADATA_MERGE = Command(
    name='migrate-metadata-merge',
    flags=('migrate_metadata_merge', 'migrate_metadata_verify'),
    add_arguments=add_metadata_merge_arguments,
    run=run_migrate_metadata_merge,
)


def add_reality_feat_arguments(parser) -> None:
    parser.add_argument('--migrate-reality-feat',
                        help="把 4.4 的 **ChartScan CV 检出结果**搬进 8.3 的 "
                             "*_reality_feat_*（**只搬 CV 列**，不搬那 136 个指标列）。"
                             "逐码路由：etf_cn→etf_*、index_cn/fund_cn→index_*、"
                             "stock_cn→stock_*。幂等（只 $set）。"
                             "配合 --migrate-dry-run 先量规模、--migrate-limit 试跑。",
                        action='store_true',
                        default=False)
    parser.add_argument('--migrate-reality-verify',
                        help="**只读**核对：4.4 的 CV 行是否都在 8.3 对应表里、逐值相同。"
                             "⚠️ 搬完必须先跑它才算「库齐全」。",
                        action='store_true',
                        default=False)


def run_migrate_reality_feat(args) -> None:
    """`--migrate-reality-feat` 的处理体：只做参数校验，干活在 `reality_save`。"""
    from GolemQ.markets.StockCN.reality_save import (
        migrate_reality_feat_from44, verify_reality_feat_migration)

    if args.migrate_reality_verify:
        v = verify_reality_feat_migration()
        if v['missing'] or v['mismatch']:
            print('[migrate] ❌ 核对不通过：缺 {} / 不符 {}'.format(
                v['missing'], v['mismatch']), file=sys.stderr)
            sys.exit(1)
        return

    try:
        out = migrate_reality_feat_from44(limit=args.migrate_limit,
                                          dry_run=args.migrate_dry_run)
    except Exception as exc:                     # noqa: BLE001
        from GolemQ.core.migrate44 import mongo44_uri
        print('[migrate] 失败：{}: {}'.format(type(exc).__name__, exc), file=sys.stderr)
        print('[migrate] 4.4 地址由 8.3 的 uri 推导得到：{}'.format(mongo44_uri()),
              file=sys.stderr)
        sys.exit(1)
    if not args.migrate_dry_run and not out:
        print('[migrate] ⚠️ 一行都没搬到 —— 检查 4.4 是否可达', file=sys.stderr)


MIGRATE_REALITY_FEAT = Command(
    name='migrate-reality-feat',
    flags=('migrate_reality_feat', 'migrate_reality_verify'),
    add_arguments=add_reality_feat_arguments,
    run=run_migrate_reality_feat,
)
