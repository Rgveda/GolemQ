# coding: utf-8
"""`--strategy review`（别名 `--review`）/ `--review-runtime` —— 统一因果 review 管线。

与老树的对应
============
老树 ``python -m GolemQ.cli --review --daily --pct_of_cores 32 --eval fast``
只写日族；本命令把 daily + zen_minor + coldstart 子集合成一条管线，
按 Code 类别写 ``stock_/etf_/index_reality_feat_{day,60min,15min}``。
⚠️ ``--daily`` 已**删除**（语义指向不明，用户 2026-10-20）—— 不要再加回来。

命令形状（用户 2026-10-20 明确）：

    --review  ==  --strategy review
    --eval    =  --strategy 的**第二参数**（scope 或逗号分隔代码；全局选项）

**不要**再给单条命令造 ``--review-eval`` / ``--review-codes`` 这类前缀变体
—— 第二参数只有 ``--eval`` 一个入口。
"""
from __future__ import annotations

import re
import sys

from ._registry import Command


def add_review_daily_arguments(parser) -> None:
    parser.add_argument('--review', action='store_true', default=False,
                        help='复盘（统一管线）—— `--strategy review` 的别名。')
    # NOTE: scope 用**全局 `--eval`**（dest=`eval_scope`），这里不注册第二份。
    parser.add_argument('--pct-of-cores', '--pct_of_cores', dest='pct_of_cores',
                        type=float, default=None,
                        help='并行核数占比：0=串行；0~1=比例；1~100=百分数（老树 32=32%%）。'
                             '不给则用默认 0.32，并在 stderr 明示。')
    parser.add_argument('--families', default='day,60min,15min',
                        help='要写的族，逗号分隔（day,60min,15min）。'
                             '冷启动列已并入 60min。')
    parser.add_argument('--review-limit', dest='review_limit', type=int, default=0,
                        help='只跑前 N 只（调试/分批）。')
    # NOTE: `--force` / `--from` 是**全局修饰符**（`cli/__main__.py`），
    # 这里不再注册 `--review-force` / `--review-from`。
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='只算不写（不写行、不写水位）。')
    parser.add_argument('--offset', default='', help='截止日期 YYYY-MM-DD。')
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


#: `--strategy review` 未给第二参数时**不猜**：直接退出 2。
#: 这条是用户 2026-10-20 明确要求 —— 静默回落 `fast` 会让人以为「参数没生效」，
#: 而实际已经跑了一段全市场复盘。
#:
#: ⚠️ **第二参数统一走 `--eval`**：scope（fast/block/…）或逗号分隔代码；
#: **没有独立的 `--review-codes`** —— 用户 2026-10-20 明确它多余。
_SCOPE_HELP = '--eval fast|block|etf|index|all|etc|etc_1|etc_2|600519,000001'


def _review_selector(value) -> bool:
    """命中判据：`--review`（True）或 `--strategy review`（'review'）。"""
    return value is True or value == 'review'


def _eval_value(args) -> str:
    """全局 `--eval`（`eval_scope`）；空串表示**调用方没给**。"""
    return str(getattr(args, 'eval_scope', '') or '').strip()


def _codes_from_eval(text) -> list:
    """`--eval` 的代码形态：逗号 / 空白 / 换行分隔，每项取第一个 6 位代码。

    兼容 `300641 正丹股份`、`sh600519`、`600519.SH`、`000001,000002` 等写法；
    名称里的数字不会被误取（只认 **恰好 6 位** 的数字段）。保持首次出现顺序、去重。
    """
    out = []
    for token in re.split(r'[,;\s]+', str(text)):
        m = re.search(r'(?<!\d)(\d{6})(?!\d)', token)
        if m:
            code = m.group(1)
            if code not in out:
                out.append(code)
    return out


def run_review_daily(args) -> int:
    from GolemQ.pipeline.review import run_review

    eval_range = _eval_value(args)
    if not eval_range:
        print('用法错误: --strategy review（或 --review）需要 {}'.format(
            _SCOPE_HELP), file=sys.stderr)
        return 2
    # `--eval` 的第二参数有两种形态：scope（`SCOPES`）或代码清单。
    from GolemQ.analysis.cv.scopes import SCOPES
    if eval_range.lower() in SCOPES:
        eval_range = eval_range.lower()
        codes = None
    else:
        codes = _codes_from_eval(eval_range)
        if not codes:
            print('用法错误: --eval 既不是 scope 也不含 6 位代码：{!r}'.format(
                eval_range[:80]), file=sys.stderr)
            return 2
        eval_range = 'fast'
    pct_of_cores = args.pct_of_cores
    if pct_of_cores is None:
        pct_of_cores = 0.32
        print('[review] 未指定 --pct-of-cores，使用默认 0.32（32% 核）',
              file=sys.stderr)
    families = tuple(f.strip() for f in str(args.families).split(',') if f.strip())
    try:
        results, stats = run_review(
            eval_range=eval_range, codes=codes,
            pct_of_cores=pct_of_cores, families=families,
            force=args.force, dry_run=args.dry_run, verbose=args.verbose,
            offset=args.offset, from_date=args.from_date,
            recheck_days=args.recheck_days, limit=args.review_limit)
    except KeyboardInterrupt:
        # 128 + SIGINT 的惯例退出码；worker 的终止由 `BasePipeline` 的
        # `with Parallel(...)` 负责（见该处注释）。
        print('\n[review] 已中断（Ctrl+C）', file=sys.stderr)
        return 130
    except ValueError as exc:
        print(f'用法错误: {exc}', file=sys.stderr)
        return 2
    print('[review] total={} success={} errors={} status={}'.format(
        stats['total_count'], stats['success_count'], stats['error_count'],
        stats['status_counts']))
    return 0 if stats['error_count'] == 0 else 1


def run_review_runtime(args) -> int:
    from GolemQ.pipeline.review import compute_runtime

    codes = args.runtime_code
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
    flags=('review', 'strategy'),
    add_arguments=add_review_daily_arguments,
    run=run_review_daily,
    is_set=_review_selector,
)

REVIEW_RUNTIME = Command(
    name='review-runtime',
    flags=('review_runtime',),
    add_arguments=add_review_runtime_arguments,
    run=run_review_runtime,
)
