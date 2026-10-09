# coding:utf-8
"""CLI 命令注册表 —— **机制层**，不认识任何具体命令。

为什么要有它
============
`cli/__main__.py` 曾是一条 16 个 `elif`、742 行的链：参数注册与分支实现都堆在
一个函数里。命令一多就没法维护（用户 2026-10-09：「未来可能会有上千条不同
命令组合，都写在 `__main__.py`？这显然不合适」）。

现在每个命令一个模块（`cli/commands/<name>.py`），只暴露三件东西：

    FLAGS          命中哪几个 args 属性就归它（见下）
    add_arguments  `f(parser)`，把这个命令的参数注册上去
    run            `f(args) -> int | None`，**校验 + 委托**，返回值当退出码

`cli/commands/__init__.py` 按**固定顺序**把它们登记进来。

⚠️ **顺序就是语义**
====================
老 `if/elif` 链里哪条先写，决定两个开关同时给时谁赢（例如
`--save-coverage` 必须排在 `--save` 之前）。那个顺序**藏在书写顺序里**，
正是 `PITFALLS.md` P12 记过的坑。现在它变成 `__init__.py` 里那张显式列表。
**加命令时把它插进列表的正确位置**，别只 append。

⚠️ **`FLAGS` 的命中判据**（:attr:`Command.is_set`，默认**真值判据**）
====================================================================
默认与原链的 `elif args.X:` **逐字等价**（`bool(getattr(args, f, None))`）。
需要别的判据（例如 `is not None`）时传 `is_set=`，但目前**没有命令用到** ——
唯一曾经需要的 `--save` 改用了 argparse 的 `choices=`，取值只可能是
`None` 或三个非空值之一，真值判据就够。
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

#: 退出码口径（用户 2026-10-09 定）：
#:   * **用法/参数错 → 2**（argparse 的约定，`usage:` 输出就是这个语义）；
#:   * **运行期失败 → 1**（取数取不到、落库失败、搬过来 0 行…）。
#: ⚠️ 分界线是「**调用错了**」还是「**跑起来没成**」——不是「严重程度」。
EXIT_USAGE = 2
EXIT_FAILURE = 1

#: `build_parser` 建好 parser 后钉在这里 —— 只为了让 :func:`usage_error`
#: 能打出与 argparse **完全同形**的 `usage:` 行。
_PARSER = None


def bind_parser(parser) -> None:
    """`build_parser` 调一次。"""
    global _PARSER
    _PARSER = parser


def usage_error(message: str, hint: str = None) -> None:
    """按 **argparse 的形状**报一个用法错误并退出 :data:`EXIT_USAGE`。

    为什么需要它：有一类取值合法性 argparse **表达不了** —— 合法集来自**运行期
    注册表**（`--sub` 的订阅器键、`--save-collections` 的集合名）。那几处只能自己报，
    但要报成**同一个形状、同一个退出码**，否则同一类错误在 CLI 上有两种表现
    （实测过：`--save tdxx` 给 2，`--purge-l1 tdx` 给 1）。

    输出与 ``argparse.ArgumentParser.error`` 同构：usage 进 stderr，
    再一行 ``<prog>: error: <message>``；``hint`` 是可选的中文补充行。
    """
    prog = getattr(_PARSER, 'prog', None) or 'GolemQ'
    if _PARSER is not None:
        _PARSER.print_usage(sys.stderr)
    print('{}: error: {}'.format(prog, message), file=sys.stderr)
    if hint:
        print(hint, file=sys.stderr)
    sys.exit(EXIT_USAGE)


@dataclass(frozen=True)
class Command:
    """一个 CLI 命令。字段含义见模块 docstring。"""

    name: str
    flags: Sequence[str]
    add_arguments: Callable
    run: Callable
    #: 这条命令**要不要连库**。默认要 —— 忘了声明只会多连一次，不会静默漏检。
    #:
    #: ⚠️ **配置类命令必须显式 `needs_db=False`**（`--setup` / `--mongodb-init` /
    #: `--dingtalk-init` / `--serverchan-init`）：它们的作用就是**修配置** ——
    #: 要求先连上库，等于配置连错了就永远修不回来。`--sub` 同理（订阅器自己要什么
    #: 自己取，别在入口处先卡一道）。
    needs_db: bool = True
    #: 命中判据，逐个 flag 调。默认 `bool` = 真值判据（等价老链 `elif args.X:`）。
    is_set: Callable = bool

    def matches(self, args) -> bool:
        return any(self.is_set(getattr(args, f, None)) for f in self.flags)


#: 登记表。**顺序即分发顺序** —— 由 `commands/__init__.py` 决定。
COMMANDS: list = []


def register(commands: Sequence[Command]) -> None:
    """按给定顺序登记。**不要**在别处 append `COMMANDS`。"""
    for cmd in commands:
        if not isinstance(cmd, Command):
            raise TypeError('只登记 Command，收到 {!r}'.format(type(cmd).__name__))
        COMMANDS.append(cmd)


def no_arguments(parser) -> None:      # noqa: ARG001 约定签名
    """给「不拥有任何参数」的命令用（参数归别的模块注册，见 `__init__.py`）。"""
    return None


def pick(args) -> Optional[Command]:
    """按登记顺序找第一个命中的命令；都不命中 → ``None``。

    **与 :func:`dispatch` 分开**是为了让调用方**先拿到命令、再决定要不要连库自检**
    （`Command.needs_db`），最后才 :meth:`Command.run`。合成一步就没这个机会了。
    """
    for cmd in COMMANDS:
        if cmd.matches(args):
            return cmd
    return None


def dispatch(args) -> tuple:
    """``pick`` + ``run`` 的便捷版。

    :returns: ``(是否命中, 退出码)`` —— 都不命中时 ``(False, None)``，
        调用方据此去打 help（老链的 `else:` 分支）。
    """
    cmd = pick(args)
    if cmd is None:
        return False, None
    return True, cmd.run(args)
