# coding:utf-8
"""**一次性搬运**入口：从 4.4 取材落进 8.3。

⚠️ 本模块是 `core/migrate44.py` 那条只读通道的**唯一**消费者。
`D33` 曾在 2026-10-10 把整条通道删掉（理由「搬运完成」），**那个前提是错的** ——
元数据类集合（`stock_ranking` / `stock_valuation`）从没搬过。现在按需恢复，
用途限定为「一次性搬运」，见 `DECISIONS.md` 的 D33 修正条。

目前只有一条：`--migrate-turnover`。
"""
from __future__ import annotations

import sys

from ._registry import Command


def add_turnover_arguments(parser) -> None:
    parser.add_argument('--migrate-turnover',
                        help="把 4.4 的换手率搬进 8.3 stock_metadata_day"
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
    from GolemQ.markets.StockCN.metadata_save import METADATA_DAY_COLL, save_turnover

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
            METADATA_DAY_COLL, total_read, total_written, total_skipped))
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
