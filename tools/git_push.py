#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""推送当前分支到 origin —— **两条路都试 + 重试 + `ls-remote` 复核**。

为什么需要它（`HANDOFF.md` 的「三条教训」，逐条对应这里的做法）
==============================================================
1. **方向会翻。** 本机到 GitHub 一天之内见过「直连通 / 代理不通」与「代理通 / 直连不通」
   两种。手敲 `git -c http.proxy=… push` 就是**假定代理通**的写法 —— 代理一关就白等。
2. **`git push` 的输出会骗人。** 直连失败时它可能打 `Everything up-to-date`，
   而远端其实停在上一笔（本地 `origin/main` 引用只有 fetch 才更新）。
   ⇒ 本脚本**只认 `ls-remote` 的结果**，不认 push 的回显。
3. ⚠️ **直连随机抽风**（用户 2026-10-11 明确）。
   ⇒ **一次失败不下结论**：退避重试，再换另一条路 —— 否则会把「随机抽风」
   误判成「这条路死了」，而那种误判会被当成事实写进文档。

用法：

    python tools/git_push.py                  # 推当前分支 → origin
    python tools/git_push.py --branch main
    python tools/git_push.py --attempts 6 --proxy http://127.0.0.1:7897

退出码：0 = 远端**已确认**与本地相同；1 = 试完所有组合仍未确认（**此时别当已推**）。
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time

#: 两条路的定义：``(名字, 额外 git 参数)``。
#: ⚠️ **顺序不代表优先级** —— 两个都会被试到（见模块 docstring 第 1 条）。
ROUTES = (
    ('直连', ()),
    ('代理', ('-c', 'http.proxy={proxy}')),        # {proxy} 由 --proxy 填
)


def _git(args, timeout=180):
    """跑一条 git 命令 → ``(returncode, stdout+stderr)``。失败不抛（调用方要读到输出）。"""
    try:
        p = subprocess.run(['git', *args], capture_output=True, text=True,
                           timeout=timeout, encoding='utf-8', errors='replace')
        return p.returncode, (p.stdout or '') + (p.stderr or '')
    except subprocess.TimeoutExpired:
        return 124, 'timeout after {}s'.format(timeout)


def local_head(branch) -> str:
    rc, out = _git(['rev-parse', branch])
    return out.strip() if rc == 0 else ''


def remote_head(remote, branch, extra=()) -> str:
    """远端分支的哈希 —— **唯一的判据**（见模块 docstring 第 2 条）。

    空字符串 = 取不到（网络问题），**不是**"远端没有这个分支"。
    """
    rc, out = _git([*extra, 'ls-remote', remote, 'refs/heads/{}'.format(branch)],
                   timeout=60)
    if rc != 0 or not out.strip():
        return ''
    return out.split()[0][:12]


def _probe(proxy: str) -> str:
    """两条路的**粗探**（只打印，不做决定）—— 让人一眼看出当前哪条通。"""
    import socket
    import urllib.request
    host, _, port = proxy.replace('http://', '').partition(':')
    try:
        with socket.create_connection((host, int(port or 80)), timeout=3):
            proxied = '在听'
    except OSError:
        proxied = '不通'
    try:
        req = urllib.request.Request('https://github.com',
                                     headers={'User-Agent': 'git_push.py'})
        urllib.request.urlopen(req, timeout=12).read(1)
        direct = '200'
    except Exception:                      # noqa: BLE001 探活失败不值得抛
        direct = '不通'
    return '代理 {}: {}   直连 github: {}'.format(proxy, proxied, direct)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description='两条路都试 + 重试 + ls-remote 复核')
    ap.add_argument('--remote', default='origin')
    ap.add_argument('--branch', default=None, help='默认取当前分支')
    ap.add_argument('--proxy', default='http://127.0.0.1:7897')
    ap.add_argument('--attempts', type=int, default=4,
                    help='**每条路**的重试次数（默认 4）')
    args = ap.parse_args(argv)

    branch = args.branch
    if not branch:
        rc, out = _git(['rev-parse', '--abbrev-ref', 'HEAD'])
        if rc != 0:
            print('取不到当前分支：' + out.strip(), file=sys.stderr)
            return 1
        branch = out.strip()
    want = local_head(branch)
    print('[push] 分支 {} → {}/{}    本地 {}'.format(branch, args.remote, branch,
                                                   want), flush=True)
    print('[push] 探活：' + _probe(args.proxy), flush=True)

    routes = [(name, tuple(a.format(proxy=args.proxy) for a in extra))
              for name, extra in ROUTES]
    for round_no in range(args.attempts):
        for name, extra in routes:
            rc, out = _git([*extra, 'push', args.remote,
                            '{}:{}'.format(branch, branch)])
            tail = [ln for ln in out.splitlines() if ln.strip()][-1:] or ['(无输出)']
            got = remote_head(args.remote, branch, extra)
            ok = (got == want[:len(got)]) and got != ''
            print('[push] 第 {} 轮 / {:<4s} rc={} | {} | 远端 {}'.format(
                round_no + 1, name, rc, tail[0][:70], got or '取不到'), flush=True)
            if ok:
                print('[push] ✓ 远端**已确认**同步：{}'.format(got), flush=True)
                return 0
        time.sleep(3 + 2 * round_no)        # 退避：直连会随机抽风，别打太密
    # ⚠️ 走 **stdout 且 flush**：与上面那些行同一条流，顺序才不会被缓冲打乱。
    # （写 stderr 时它与 block-buffered 的 stdout 不同步 —— 实测那行会跑到标题前面。）
    print('[push] ✗ 试完 {} 轮 × {} 条路仍未确认同步 —— **别当已推**；'
          '复核：git ls-remote {} refs/heads/{}'.format(
              args.attempts, len(routes), args.remote, branch), flush=True)
    return 1


if __name__ == '__main__':
    sys.exit(main())
