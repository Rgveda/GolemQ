# coding:utf-8
#
# The MIT License (MIT)
#
# Copyright (c) 2020-2025 azai/GolemQ
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""GolemQ 命令行入口 —— **只剩「建 parser → 分发 → help」**。

本文件曾是一条 742 行、16 个 `elif` 的链：参数注册与分支实现全堆在一个
`main()` 里。用户 2026-10-09 的原话：「未来可能会有上千条不同命令组合，
都写在 `__main__.py`？这显然不合适」。

现在：

* **每条命令一个模块** —— `cli/commands/<name>.py`，只暴露
  `FLAGS` / `add_arguments(parser)` / `run(args)`；
* **登记顺序即分发顺序** —— 见 `cli/commands/__init__.py` 里那张显式列表；
* 这里只负责：建 parser、注册全局开关（`-v`）、按序分发、没命中就打 help。

加一条命令：
1. 写 `cli/commands/<name>.py`（照同目录里的先例）；
2. 在 `cli/commands/__init__.py` 的登记列表里**按正确位置**加一行。

⚠️ **唯一刻意的输出变化**：`--help` 里选项的排列顺序。以前按老链里
`add_argument` 的书写顺序（各命令的开关是交错的），现在按**命令分组**。
开关心智与参数名逐字未变。
"""
import argparse
import sys

from GolemQ.cli import bootstrap
from GolemQ.cli.commands import COMMANDS, pick
from GolemQ.cli.commands._registry import bind_parser

#: 版权信息 —— 真身在 `bootstrap.copyright_infos`（自检那一套都在那儿）。
#: 这里转出一次，`from GolemQ.cli.__main__ import copyright_infos` 的写法照旧能用。
copyright_infos = bootstrap.copyright_infos


def build_parser() -> argparse.ArgumentParser:
    """建 parser：全局开关 + 每条命令自己的参数。"""
    parser = argparse.ArgumentParser(
        description="GolemQ 命令行工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python -m GolemQ.cli --setup           # 初始化配置
  python -m GolemQ.cli --mongodb-init    # 初始化MongoDB配置
  python -m GolemQ.cli --dingtalk-init   # 初始化钉钉配置
  python -m GolemQ.cli --serverchan-init # 初始化Server酱配置
  python -m GolemQ.cli --purge-l1           # 清理数据库
  python -m GolemQ.cli --purge-l1 --verbose # 详细模式清理数据库
  python -m GolemQ.cli --eneloop-add --symbols "000001,000002" # 添加个股到关注列表
  python -m GolemQ.cli --eneloop-remove --symbols "000001,000002" # 删除个股并归档
  python -m GolemQ.cli --eneloop-list    # 查看当前关注列表
  python -m GolemQ.cli --sub l1_tencent  # 执行L1数据订阅功能
        """
    )

    # 全局开关：**归 `__main__` 所有**（几乎每条命令都会读 `args.verbose`），
    # 所以不进任何 `commands/*.py` 的 `add_arguments`。
    parser.add_argument("-v", "--verbose",
                        help="增加输出详细程度",
                        action="store_true",
                        default=False)

    for cmd in COMMANDS:
        cmd.add_arguments(parser)

    # 钉住 —— 给 `usage_error()` 用（它要打与 argparse 同形的 usage 行）
    bind_parser(parser)
    return parser


def main() -> None:
    """主函数：处理命令行参数"""
    # ① **身份块**（产品名 + 本次运行的时刻戳 + 版权行 + 空行）：**在 `parse_args()`
    #    之前** —— 所以 `--help` 与用法错误上也看得见。
    #    `flush=True` 是必须的：stdout 被重定向时是块缓冲，而 argparse 的 usage 走
    #    stderr（无缓冲）—— 不 flush 会排到 error 行后面（实测踩过）。
    #    时刻戳**只在这里出现一次**：两个 banner 都用 `header=False`，阶段行
    #    （环境自检 / 数据源 / 参考数据 / K线 / 复权）跨它们对齐成一列。
    bootstrap.print_identity()

    parser = build_parser()
    args = parser.parse_args()

    # 先认得是哪条命令 —— 自检的**严格度**取决于它：`--setup` / `--mongodb-init`
    # 这类「修配置」的入口必须放行，否则环境不对就永远修不回来
    # （与 `require_mongodb` 不为配置类命令连库同一个道理）。
    cmd = pick(args)

    # ② 本地自检 —— 解析之后，因为详略要由 `--verbose` 定；画一块 banner，
    #    自开自闭（`PITFALLS.md` P22：下一块 banner 开之前，前一块必须已 close）。
    bootstrap.check_environment(
        verbose=args.verbose,
        strict=cmd is not None and cmd.name not in bootstrap.REPAIR_COMMANDS)

    if cmd is None:
        # 没有指定任何参数（或没有命令认领）→ 打帮助
        parser.print_help()
        sys.exit(0)

    # ③ 服务自检（MongoDB 连接 + 8.3）—— 只对**要用库**的命令；连不上硬失败。
    #    放在 `run()` 之前，为的是「环境不到位」报得清楚，而不是让它死在命令内部。
    if cmd.needs_db:
        bootstrap.require_mongodb(verbose=args.verbose)

    sys.exit(cmd.run(args))


if __name__ == "__main__":
    main()
