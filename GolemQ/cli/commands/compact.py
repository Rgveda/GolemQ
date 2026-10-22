# coding:utf-8
"""`--strategy compact` —— 逐码计算 compact 特征（**只读，不落库**）。

命令形状：

    --strategy compact
    --strategy compact --eval 600519,000001

`--eval` 是全局第二参数（scope 或逗号分隔代码）；缺省 **fast**。
⚠️ compact **没有命令专属开关**（用户 2026-10-21 明确）——
固定 60min + day + rank，不要加 `--compact-*`。
"""
from __future__ import annotations

import sys

from ._registry import Command, no_arguments
from .review import _codes_from_eval, _eval_value


def add_compact_arguments(parser) -> None:
    """compact 没有自己的参数：scope/代码走全局 `--eval`。

    ⚠️ 不要加 `--compact-limit` / `--compact-no-day` 这类前缀变体 ——
    用户 2026-10-21 明确：不需要这些开关。
    """
    return None


def run_compact_cli(args) -> int:
    from GolemQ.pipeline.compact import run_compact

    eval_value = _eval_value(args)
    if not eval_value:
        # 用户 2026-10-21：只需要 `--strategy compact`；不给 --eval 就默认 fast。
        eval_value = 'fast'
        print('[compact] 未给 --eval，默认 scope=fast', file=sys.stderr)

    from GolemQ.analysis.cv.scopes import SCOPES
    if eval_value.lower() in SCOPES:
        eval_range = eval_value.lower()
        codes = None
    else:
        codes = _codes_from_eval(eval_value)
        if not codes:
            print('用法错误: --eval 既不是 scope 也不含 6 位代码：{!r}'.format(
                eval_value[:80]), file=sys.stderr)
            return 2
        eval_range = 'fast'

    pct = getattr(args, 'pct_of_cores', None)
    if pct is None:
        pct = 0.32
        print('[compact] 未指定 --pct-of-cores，使用默认 0.32（32% 核）',
              file=sys.stderr)
    try:
        results, stats = run_compact(
            eval_range=eval_range, codes=codes, pct_of_cores=pct,
            offset=getattr(args, 'offset', '') or '',
            verbose=args.verbose)
    except KeyboardInterrupt:
        print('\n[compact] 已中断（Ctrl+C）', file=sys.stderr)
        return 130
    except ValueError as exc:
        print('用法错误: {}'.format(exc), file=sys.stderr)
        return 2

    ok = [r for r in results if r.ok and r.result]
    if ok:
        sample = ok[-1]
        print('[compact] sample {}: rows={} cols={} rank={} last={}'.format(
            sample.code, sample.result['rows'], sample.result['cols'],
            sample.result['rank_rows'], sample.result['last']))
    print('[compact] total={} success={} errors={} status={}'.format(
        stats['total_count'], stats['success_count'], stats['error_count'],
        stats['status_counts']))
    return 0 if stats['error_count'] == 0 else 1


COMPACT = Command(
    name='compact',
    flags=('strategy',),
    add_arguments=no_arguments,
    run=run_compact_cli,
    is_set=lambda value: value == 'compact',
)
