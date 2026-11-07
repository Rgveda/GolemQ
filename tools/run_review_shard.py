# coding: utf-8
"""按**代码清单文件**跑 `--review`（多机**冷启动缺码**回填的短命令行启动器）。

⚠️ **补已有码的 review 尾巴不要用本工具。** 代码清单只含「完全没有 review
行」的冷启动码；已 review 过的码（如 000711）不在清单里，会被漏掉。补尾巴
直接跑 scope 级 review（`persistence.missing_targets` 会自动只算缺的 bar）：

    python -m GolemQ.cli --review --eval all
    python -m GolemQ.cli --review --eval 000711

为什么不直接把代码塞进 `--eval`：完整 `etc+fast` 缺口有 3,140 个 6 位代码，
拼成一行约 22 KB，超过 `cmd.exe` 的 8,191 字符上限；分 3 片后每片仍有 7 KB
左右，贴着上限跑很脆。本脚本用 UTF-8 读文件，再直接调
`pipeline.review.run_review` —— 与 `python -m GolemQ.cli.internal --review`
走**同一条** `run_review` 路径。

用法（在 `Y:\\Projects\\GolemQ` 下）::

    python tools/run_review_shard.py _review_shard1.txt --pct-of-cores 50
    python tools/run_review_shard.py _review_shard1.txt --dry-run --limit 1 \
        --from-date 2026-10-20          # 冒烟

⚠️ 多机时每台设**本机**缓存目录（不要写 Y: 共享盘）：

    set NUMBA_CACHE_DIR=%LOCALAPPDATA%\\GolemQ\\numba
    set PYTHONPYCACHEPREFIX=%LOCALAPPDATA%\\GolemQ\\pycache
"""
from __future__ import annotations

import argparse
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description='按代码文件跑 review（多机回填）')
    p.add_argument('file', help='每行一个 6 位代码的 UTF-8 文件（仅冷启动缺码；'
                                '补尾巴请用 --review --eval <scope>）')
    p.add_argument('--pct-of-cores', type=float, default=0.5,
                   help='CPU 百分比（>1）或比例（0~1）；默认 0.5')
    p.add_argument('--review-timeout', type=int, default=4 * 60 * 60,
                   help='单码超时秒数（默认 4h）')
    p.add_argument('--from-date', default='', help='只算该日期之后（调试用）')
    p.add_argument('--limit', type=int, default=0, help='只跑前 N 个（调试用）')
    p.add_argument('--dry-run', action='store_true', help='只算不写')
    p.add_argument('--verbose', action='store_true')
    a = p.parse_args(argv)

    with open(a.file, encoding='utf-8') as fh:
        codes = [ln.strip() for ln in fh if ln.strip()]
    if not codes:
        print(f'[review-shard] {a.file} 里没有代码', file=sys.stderr)
        return 2

    from GolemQ.pipeline.review import run_review
    print(f'[review-shard] {os.path.basename(a.file)}: {len(codes)} 码 '
          f'pct={a.pct_of_cores} timeout={a.review_timeout}s '
          f'from={a.from_date or "-"} dry_run={a.dry_run}', flush=True)
    results, stats = run_review(
        codes=codes, pct_of_cores=a.pct_of_cores,
        dry_run=a.dry_run, verbose=a.verbose,
        from_date=a.from_date, limit=a.limit,
        code_timeout=a.review_timeout)
    print('[review-shard] total={} success={} errors={} status={}'.format(
        stats['total_count'], stats['success_count'], stats['error_count'],
        stats['status_counts']), flush=True)
    return 0 if stats['error_count'] == 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())
