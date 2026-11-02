# coding: utf-8
"""`--strategy mainstream` —— 全市场「全局智能仓位」（Dempster–Shafer）。

命令形状：

    --strategy mainstream --eval fast
    --strategy mainstream --eval etc_1 --offset 2026-10-30
    --strategy mainstream --eval 600519,000001

产物：`stock_massive_position_60min`（每根 60min bar 一行：
`PosBsln / PosPad / PosNegBsln / MasPosOnhold / mberCnt / DS_*`）。

⚠️ **没有命令专属开关**：scope 走全局 `--eval`，并行走 `--pct-of-cores`，
截止日走 `--offset` —— 这三个都由 `review` 命令注册在同一个 parser 上，
这里**不再注册第二份**（重复 dest 会让 argparse 直接冲突）。
"""
from __future__ import annotations

import sys

from ._registry import Command, no_arguments
from .review import _codes_from_eval, _eval_value


def run_mainstream_cli(args) -> int:
    from GolemQ import get_active_market
    from GolemQ.core.constants import MARKET_TYPE
    from GolemQ.markets.StockCN.reality_save import save_position_series
    from GolemQ.pipeline.mainstream import run_mainstream

    eval_value = _eval_value(args)
    if not eval_value:
        print('用法错误: --strategy mainstream 需要 --eval '
              'fast|block|etf|index|all|etc|etc_1|etc_2 或逗号分隔代码',
              file=sys.stderr)
        return 2

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
        print('[mainstream] 未指定 --pct-of-cores，使用默认 0.32（32% 核）',
              file=sys.stderr)

    try:
        df, stats = run_mainstream(
            eval_range=eval_range, codes=codes,
            as_of=getattr(args, 'offset', '') or '',
            pct_of_cores=pct, verbose=args.verbose)
    except KeyboardInterrupt:
        print('\n[mainstream] 已中断（Ctrl+C）', file=sys.stderr)
        return 130
    except ValueError as exc:
        print('用法错误: {}'.format(exc), file=sys.stderr)
        return 2

    if df is None or len(df) == 0:
        print('[mainstream] 无仓位行（数据不足或 scope 为空）', file=sys.stderr)
        return 1

    market_type = (MARKET_TYPE.ETF_CN if eval_range == 'etf'
                   else MARKET_TYPE.STOCK_CN)
    out = save_position_series(df, market_type=market_type,
                               db=get_active_market().get_database(),
                               verbose=args.verbose)
    last = df.iloc[-1]
    print('[mainstream] bars={} members={} last={} 仓位={:.2%} '
          '做空比={:.2%} | {} 写 {} 行'.format(
              len(df), int(last['mberCnt']),
              str(last['datetime']), float(last['MasPosOnhold']),
              float(last['PosNegBsln']), out['name'], out['rows']))
    return 0


MAINSTREAM = Command(
    name='mainstream',
    flags=('strategy',),
    add_arguments=no_arguments,
    run=run_mainstream_cli,
    is_set=lambda value: value == 'mainstream',
)
