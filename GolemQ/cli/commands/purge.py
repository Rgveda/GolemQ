# coding:utf-8
"""`--purge-l1` / `--purge`：清理 MongoDB 数据库。**破坏性**，故有确认闸。"""
from __future__ import annotations

import sys

from GolemQ.cli.tools import purge_mongodb_database

from ._registry import Command, usage_error


def add_purge_arguments(parser) -> None:
    parser.add_argument('-a', '--purge-l1', '--purge',
                        dest='purge_l1',
                        help="清理 MongoDB 数据库（老树的写法是 --purge <mgdb|mongodb>，"
                             "两处短标志同为 -a，见 GolemQ_old/cli/__main__.py:286）",
                        nargs='?',
                        const=True,
                        default=False)


def run_purge(args) -> None:
    # `--purge` 是老树的写法，那边带一个「数据源」取值（`mgdb`/`MongoDB`/`mongodb`）。
    # 先把它接住：老脚本原样照跑，取值不对就报错 —— **不要静默忽略**，
    # 否则 `--purge tdx` 这种会被当成"清理成功"，而用户以为清的是别的库。
    if isinstance(args.purge_l1, str) and \
            args.purge_l1.lower() not in ('mgdb', 'mongodb'):
        # **用法错**（取值非法）→ `usage_error`：与 argparse 同形（usage 行）同码（2）
        usage_error("argument --purge: 只接受 mgdb / mongodb（老树的取值），"
                    "收到 '{}'".format(args.purge_l1),
                    hint='提示: 老脚本写 --purge <数据源>；本树只看这个取值。')

    # 确认操作
    if not args.verbose:
        confirm = input("警告: 这将删除所有 L1 数据! 确认操作? (y/N): ").strip().lower()
        if confirm != 'y':
            print("操作已取消")
            return

    purge_mongodb_database(args.verbose)


PURGE = Command('purge', ('purge_l1',), add_purge_arguments, run_purge)
