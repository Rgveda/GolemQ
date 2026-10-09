# coding:utf-8
"""`--update-tdx-hosts`：**每周一次**的通达信服务器池刷新。

服务器会时好时坏（实测同一台能在 0.2s 与 >10s 之间跳），手写的 4 台既漏又旧。
本命令用**真协议探针**把 pytdx 包内那 64 台池子筛一遍、按中位延迟排序、落缓存；
`TdxSource` 下次就优先用缓存里的顺序。

⚠️ **`needs_db=False`**：它只做网络探活 + 写一个 JSON 文件，不碰 Mongo ——
每个 source 的**单一职责**。这也让它可以在一台还没配好库的机器上跑。
"""
from __future__ import annotations

from ._registry import Command, EXIT_FAILURE


def add_update_arguments(parser) -> None:
    parser.add_argument('--update-tdx-hosts', '--update_tdx_hosts',
                        dest='update_tdx_hosts',
                        action='store_true', default=False,
                        help="刷新通达信行情服务器池（真协议探活 + 按中位延迟排序 + 落缓存）。"
                             "默认**每周一次**：不到期的直接报缓存状态；"
                             "配合 --force 强制重探")

    # ⚠️ `--force` **在这里注册、但不当独立命令** —— 它是修饰符。
    # 单独立一条命令会让「只打 --force」命中一个空命令（打了没反应）。
    parser.add_argument('--force',
                        help="配合 --update-tdx-hosts：忽略每周周期，立刻重探",
                        action='store_true', default=False)


def run_update(args) -> None:
    from GolemQ.markets.StockCN.datasource import tdx_hosts

    cached = tdx_hosts.load()
    print('[tdx_hosts] 缓存: {}'.format(
        '{} 台（未过期）'.format(len(cached)) if cached else '无 / 已过期'))
    print('[tdx_hosts] 候选池: {} 台（pytdx 包内池 ∪ DEFAULT_HOSTS），每台测 {} 次'.format(
        len(tdx_hosts.candidates()), tdx_hosts.ATTEMPTS))

    if not cached and not args.force:
        print('[tdx_hosts] 正在探活（可能要一两分钟）…')

    hosts, did_probe = tdx_hosts.refresh(force=args.force, verbose=True)

    if not hosts:
        # 探不到时 refresh() **保留旧缓存**、不清空；连旧缓存也没有才是真失败
        print('✗ 没探到任何可用服务器，且没有可退回的缓存 —— '
              '`TdxSource` 仍会用 `DEFAULT_HOSTS` 那几台兜底')
        return

    if did_probe:
        print('[tdx_hosts] 已写入 {}（{} 台）'.format(tdx_hosts.cache_path(), len(hosts)))
        print('[tdx_hosts] `TdxSource` 下次构造时会优先用这份顺序')
    elif args.verbose:
        print('[tdx_hosts] 未到期，本次没重探（要强制加 --force）')


UPDATE_TDX_HOSTS = Command('update-tdx-hosts', ('update_tdx_hosts',),
                           add_update_arguments, run_update, needs_db=False)
