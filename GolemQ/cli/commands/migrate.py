# coding:utf-8
"""4.4 → 8.3 的**一次性搬运**入口（`--migrate-eneloop` / `--migrate-financial`）。

两者都**不联网取数**，只是把 4.4 库里已有的东西搬过来；跑完即可弃
（`DECISIONS.md` D12：运行时代码不连 4.4，只有这两条一次性通道）。
"""
from __future__ import annotations

import sys

from ._registry import Command


def add_migrate_eneloop_arguments(parser) -> None:
    parser.add_argument('--migrate-eneloop', '--migrate_eneloop',
                        help="把 4.4 golemq 的关注列表（StockCN_watchdog_eneloop*）搬到 8.3 "
                             "的 golemq（一次性；不搬则 --eneloop-list 会空）",
                        action="store_true",
                        default=False)

    parser.add_argument('--migrate-source-uri',
                        help="4.4 地址（默认取环境变量 GQ_MIGRATE44_URI 或内置常量），"
                             "供 --migrate-* 用",
                        type=str,
                        default=None)


def add_migrate_financial_arguments(parser) -> None:
    parser.add_argument('--migrate-financial', '--migrate_financial',
                        help="把 4.4 quantaxis.financial 搬到 8.3 golemq_stock_cn.financial"
                             "（不联网取数；可反复跑，按 (code,report_date) upsert）",
                        action="store_true",
                        default=False)


def run_migrate_eneloop(args) -> None:
    from GolemQ.cli.watchdog_manager import migrate_eneloop_watchlist
    print("把 4.4 的关注列表搬到 8.3 …")
    stats = migrate_eneloop_watchlist(verbose=True,
                                      source_uri=args.migrate_source_uri)
    if any(v['dst'] == 0 for v in stats.values()):
        print("警告: 有集合搬到 0 行 —— 请核对源集合是否为空")
        sys.exit(1)


def run_migrate_financial(args) -> None:
    # 搬运而非取数：4.4 库里已有这份数据，而 akshare 逐只取全量要 46.5 小时。
    # 详见 GQ_migrate_financial 的 docstring。
    from GolemQ.markets.StockCN.refdata_save import (
        GQ_migrate_financial,
        format_status,
    )
    stats = GQ_migrate_financial(verbose=args.verbose)
    print()
    print(format_status())
    if stats.get('rows', 0) == 0:
        print('警告: 源集合没有搬到任何行，请确认 4.4 的 quantaxis.financial 是否存在。')
        sys.exit(1)


MIGRATE_ENELOOP = Command('migrate-eneloop', ('migrate_eneloop',),
                          add_migrate_eneloop_arguments, run_migrate_eneloop)
MIGRATE_FINANCIAL = Command('migrate-financial', ('migrate_financial',),
                            add_migrate_financial_arguments, run_migrate_financial)
