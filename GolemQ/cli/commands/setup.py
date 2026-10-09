# coding:utf-8
"""`--setup` / `--init` 与三个单项初始化开关。

⚠️ 四个命令**各自登记、各自注册参数**（不共用一个 `add_arguments`）——
共用会让同一个开关被注册多次（argparse 直接报错），而且「谁拥有哪个开关」
就不再是一对一的。
⚠️ 它们是**四条独立命令**而不是一个 `run` 里自己判断：老链里就是四条独立
`elif`，同时给两个开关时**只有先命中的那条会跑**，合并会改掉语义。
"""
from __future__ import annotations

from GolemQ.agents.messenger import (
    setup_dingtalk_config_interactive,
    setup_serverchan_config_interactive,
)
from GolemQ.core.settings import setup_mongodb_config

from ._registry import Command


def add_setup_arguments(parser) -> None:
    parser.add_argument('-s', '--setup',
                        help="初始化配置 (MongoDB和钉钉)",
                        action="store_true",
                        default=False)

    parser.add_argument('-i', '--init',
                        help="初始化配置 (MongoDB和钉钉) - 与 --setup 相同",
                        action="store_true",
                        default=False)


def add_mongodb_arguments(parser) -> None:
    parser.add_argument('--mongodb-init',
                        help="初始化MongoDB配置",
                        action="store_true",
                        default=False)


def add_dingtalk_arguments(parser) -> None:
    parser.add_argument('--dingtalk-init',
                        help="初始化钉钉配置",
                        action="store_true",
                        default=False)


def add_serverchan_arguments(parser) -> None:
    parser.add_argument('--serverchan-init',
                        help="初始化Server酱配置",
                        action="store_true",
                        default=False)


def run_setup(args) -> None:
    if args.verbose:
        print("开始初始化配置...")
    setup_mongodb_config()
    setup_dingtalk_config_interactive()
    if args.verbose:
        print("配置初始化完成!")


def run_mongodb_init(args) -> None:      # noqa: ARG001 约定签名
    setup_mongodb_config()


def run_dingtalk_init(args) -> None:     # noqa: ARG001 约定签名
    setup_dingtalk_config_interactive()


def run_serverchan_init(args) -> None:   # noqa: ARG001 约定签名
    setup_serverchan_config_interactive()


# ⚠️ 四条**都不声明 `needs_db`**（即 needs_db=False）：它们的作用就是**修配置** ——
# 要求先连上库，等于「配置连错了就永远修不回来」。环境自检因此跳过它们。
SETUP = Command('setup', ('setup', 'init'), add_setup_arguments, run_setup,
                needs_db=False)
MONGODB_INIT = Command('mongodb-init', ('mongodb_init',), add_mongodb_arguments,
                       run_mongodb_init, needs_db=False)
DINGTALK_INIT = Command('dingtalk-init', ('dingtalk_init',),
                        add_dingtalk_arguments, run_dingtalk_init, needs_db=False)
SERVERCHAN_INIT = Command('serverchan-init', ('serverchan_init',),
                          add_serverchan_arguments, run_serverchan_init,
                          needs_db=False)
