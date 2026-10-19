# coding: utf-8
"""`--review --daily` / `--review-runtime` —— 统一因果 review 管线。

与老树的对应
============
老树 ``python -m GolemQ.cli --review --daily --pct_of_cores 32 --eval fast``
只写日族；本命令把 daily + zen_minor + coldstart 子集合成一条管线，
按 Code 类别写 ``stock_/etf_/index_reality_feat_{day,60min,15min}``。

⚠️ ``--eval`` 这个**选项串已被 ChartScan 占用**（`-e/--eval`, dest=`scan_eval`）。
本命令因此提供 ``--review-eval``；为了兼容老命令形状，**也接受** ChartScan 注册的
``--eval``（读 `args.scan_eval`），两者等价。
"""
from __future__ import annotations

import sys

from ._registry import Command


def add_review_daily_arguments(parser) -> None:
    parser.add_argument('--review', action='store_true', default=False,
                        help='复盘（统一管线）。与 --daily 同义，保留老命令形状。')
    parser.add_argument('--daily', action='store_true', default=False,
                        help='批量增量复盘：所有 tier，落 reality_feat_{day,60min,15min}。')
    parser.add_argument('--review-eval', dest='review_eval', default='',
                        help='scope：fast/block/etf/index/all/etc/etc_1/etc_2，或逗号分隔代码。'
                             '也可直接用 ChartScan 注册的 --eval。')
    parser.add_argument('--pct-of-cores', '--pct_of_cores', dest='pct_of_cores',
                        type=float, default=0.32,
                        help='并行核数占比：0=串行；0~1=比例；1~100=百分数（老树 32=32%%）。')
    parser.add_argument('--review-codes', dest='review_codes', default=None,
                        help='显式代码清单（逗号分隔）；给了就跳过 scope 解析。')
    parser.add_argument('--families', default='day,60min,15min',
                        help='要写的族，逗号分隔（day,60min,15min）。'
                             '冷启动列已并入 60min。')
    parser.add_argument('--review-limit', dest='review_limit', type=int, default=0,
                        help='只跑前 N 只（调试/分批）。')
    # ⚠️ `--force` 已被 `--update-tdx-hosts` 占用；本命令用 `--review-force`。
    parser.add_argument('--review-force', dest='review_force',
                        action='store_true', default=False,
                        help='忽略水位，重算全部可见日期。')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='只算不写（不写行、不写水位）。')
    parser.add_argument('--offset', default='', help='截止日期 YYYY-MM-DD。')
    parser.add_argument('--review-from', dest='review_from', default='',
                        help='起始日期 YYYY-MM-DD（覆盖水位下界）。')
    parser.add_argument('--recheck-days', dest='recheck_days', type=int, default=0,
                        help='水位前溯 N 天，修补迟到/回填的源行。')


def add_review_runtime_arguments(parser) -> None:
    parser.add_argument('--review-runtime', action='store_true', default=False,
                        help='runtime 计算（只读内存，不落库）。')
    parser.add_argument('--runtime-code', action='append', default=None,
                        help='runtime 模式要算的代码（可重复）。')
    parser.add_argument('--runtime-as-of', default='',
                        help='runtime 的 as-of（YYYY-MM-DD[ HH:MM]）；默认最近已收盘 bar。')
    parser.add_argument('--runtime-heavy', action='store_true', default=False,
                        help='runtime 允许现算重档（仅建议 watchlist 小范围）。')


def _eval_value(args) -> str:
    """`--review-eval` 优先；否则复用 ChartScan 注册的 `--eval`（dest=scan_eval）。"""
    return (getattr(args, 'review_eval', '') or
            getattr(args, 'scan_eval', '') or 'fast')


def _codes(args):
    raw = getattr(args, 'review_codes', None)
    if raw:
        return [c.strip() for c in str(raw).split(',') if c.strip()]
    return None


def run_review_daily(args) -> int:
    from GolemQ.pipeline.review import run_review

    eval_range = _eval_value(args)
    codes = _codes(args)
    if not codes and (',' in eval_range or len(eval_range) > 12 or
                      (eval_range.isdigit() and len(eval_range) == 6) or
                      ('.' in eval_range and len(eval_range) <= 12)):
        codes = [c.strip() for c in eval_range.split(',') if c.strip()]
        eval_range = 'fast'
    families = tuple(f.strip() for f in str(args.families).split(',') if f.strip())
    try:
        results, stats = run_review(
            eval_range=eval_range, codes=codes,
            pct_of_cores=args.pct_of_cores, families=families,
            force=args.review_force, dry_run=args.dry_run, verbose=args.verbose,
            offset=args.offset, from_date=args.review_from,
            recheck_days=args.recheck_days, limit=args.review_limit)
    except ValueError as exc:
        print(f'用法错误: {exc}', file=sys.stderr)
        return 2
    print('[review] total={} success={} errors={} status={}'.format(
        stats['total_count'], stats['success_count'], stats['error_count'],
        stats['status_counts']))
    return 0 if stats['error_count'] == 0 else 1


def run_review_runtime(args) -> int:
    from GolemQ.pipeline.review import compute_runtime

    codes = args.runtime_code or _codes(args)
    if not codes:
        print('用法错误: --review-runtime 需要 --runtime-code（可重复）', file=sys.stderr)
        return 2
    families = tuple(f.strip() for f in str(args.families).split(',') if f.strip())
    as_of = args.runtime_as_of or None
    for code in codes:
        out = compute_runtime(code, as_of=as_of, families=families,
                              include_heavy=args.runtime_heavy,
                              verbose=args.verbose)
        if not out:
            print(f'[review-runtime] {code}: 无数据')
            continue
        for fam, frame in out.items():
            last = None if frame is None or len(frame) == 0 else frame.tail(1)
            print(f'[review-runtime] {code} {fam}: rows={0 if frame is None else len(frame)} '
                  f'last={None if last is None else last.index[0]}')
    return 0


REVIEW_DAILY = Command(
    name='review-daily',
    flags=('daily', 'review'),
    add_arguments=add_review_daily_arguments,
    run=run_review_daily,
)

REVIEW_RUNTIME = Command(
    name='review-runtime',
    flags=('review_runtime',),
    add_arguments=add_review_runtime_arguments,
    run=run_review_runtime,
)
