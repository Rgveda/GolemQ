# coding:utf-8
"""关注列表：`--eneloop-add` / `--eneloop-remove` / `--eneloop-list`（+ `--symbols`）。

⚠️ 三条独立命令：老链里是三条 `elif`，同时给两个开关只有先命中的会跑。
`--symbols` 归**这里**注册（add 与 remove 共用它），但**一次性**注册 ——
见 `add_add_arguments` 的说明。
"""
from __future__ import annotations

import sys

from GolemQ.cli.watchdog_manager import (
    add_symbols_to_watchlist,
    list_watchlist_symbols,
    parse_symbols,
    remove_symbols_from_watchlist,
)

from ._registry import Command, usage_error


def add_add_arguments(parser) -> None:
    """`--eneloop-add` 与 `--symbols` 的注册点。

    `--symbols` 被 add 与 remove 共用，但只在**这一处**注册 —— 两处都注册
    argparse 会报 `conflicting option string`。remove 那边用 `no_arguments`。
    """
    parser.add_argument('--eneloop-add',
                        help="添加股票代码到关注列表",
                        action="store_true",
                        default=False)

    parser.add_argument('--symbols',
                        help="股票代码列表，以逗号或换行分割",
                        type=str,
                        default="")


def add_remove_arguments(parser) -> None:
    parser.add_argument('--eneloop-remove',
                        help="从关注列表删除股票代码并归档",
                        action="store_true",
                        default=False)


def add_list_arguments(parser) -> None:
    parser.add_argument('--eneloop-list',
                        help="列出当前关注列表中的股票代码",
                        action="store_true",
                        default=False)


def _parse_or_exit(args, action: str):
    """`--symbols` 的共用前戏：没给 / 解析为空都**报错退出**，不静默当成功。"""
    if not args.symbols:
        # **用法错**（必填参数缺失）→ `usage_error`，与 argparse 同形同码
        usage_error("argument --symbols: 必填（--eneloop-{} 需要它）".format(
            'add' if action == '添加' else 'remove'))
    symbols = parse_symbols(args.symbols)
    if not symbols:
        # **用法错**（给了值但解析不出代码）
        usage_error("argument --symbols: 没有解析出任何有效代码，"
                    "收到 {!r}".format(args.symbols))
    return symbols


def run_add(args) -> None:
    symbols = _parse_or_exit(args, '添加')

    print("准备添加 {} 个股票代码到关注列表:".format(len(symbols)))
    for symbol in symbols:
        print("  {}".format(symbol))

    if not args.verbose:
        confirm = input("确认添加? (y/N): ").strip().lower()
        if confirm != 'y':
            print("操作已取消")
            return

    success_count = add_symbols_to_watchlist(symbols, args.verbose)
    if success_count > 0:
        print("操作完成!")
    else:
        print("没有股票代码被添加")


def run_remove(args) -> None:
    symbols = _parse_or_exit(args, '删除')

    print("准备从关注列表删除并归档 {} 个股票代码:".format(len(symbols)))
    for symbol in symbols:
        print("  {}".format(symbol))

    if not args.verbose:
        confirm = input("确认删除并归档? (y/N): ").strip().lower()
        if confirm != 'y':
            print("操作已取消")
            return

    success_count = remove_symbols_from_watchlist(symbols, args.verbose)
    if success_count > 0:
        print("操作完成!")
    else:
        print("没有股票代码被删除")


def run_list(args) -> None:
    list_watchlist_symbols(args.verbose)


ENELOOP_ADD = Command('eneloop-add', ('eneloop_add',), add_add_arguments, run_add)
ENELOOP_REMOVE = Command('eneloop-remove', ('eneloop_remove',),
                         add_remove_arguments, run_remove)
ENELOOP_LIST = Command('eneloop-list', ('eneloop_list',), add_list_arguments,
                       run_list)
