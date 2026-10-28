# coding:utf-8
"""`--fill-turnover` —— 用「成交量 ÷ 流通股本」**现算**补齐 `stock_reality_feat_day` 缺的换手率。

与邻居的分工（都在写/看同一张 `stock_reality_feat_day`）：

===================  ==========================================================
`--moneyflow`        抓**当日**资金流/千股千评（外部 HTTP）
`--migrate-turnover` 从 **4.4** 搬换手率（internal；一次性搬运，连 4.4）
`--fill-turnover`    **本命令**：两个源都没覆盖到的日子，按流通股本**现算**
`--metadata-align`   **只读**覆盖率报告（internal；放在最后，才看得出刚补上的列）
===================  ==========================================================

**它补的是谁**：`analysis/ChipDistribution_jit.py` 现在读完 `stock_reality_feat_day` 的两个源
仍缺就**抛**（`DECISIONS.md` D44），补齐归本命令。最典型的场景是「`--save` 先写 K 线、
后写元数据」之间那段窗口 —— 盘中跑筹码分布必然缺当日换手率。

⚠️ **默认是全市场**，会逐只取日线 + 流通股本，跑得慢（几千只）。**只补真的缺的日子**，
所以重复跑没有代价；要缩范围走 internal 入口的 `--fill-turnover-codes`。

⚠️ **补算值与东财那列不是一个量**：本命令按**流通**股本，写进 `turnover` 那列
（与 baostock 同源）；东财 `TurnoverRate` 按**自由**流通股本。
"""
from __future__ import annotations

import sys

from ._registry import Command


def add_fill_turnover_arguments(parser) -> None:
    parser.add_argument('--fill-turnover',
                        help="用「成交量 ÷ 流通股本」现算补齐 stock_reality_feat_day 缺的换手率。"
                             "只补两个源都没覆盖到的日子，重复跑无代价。"
                             "⚠️ 默认全市场，跑得慢。",
                        action='store_true', default=False)
    # NOTE: `--fill-turnover-codes/start/end` 是内部/调试旗子，
    # 只在 `python -m GolemQ.cli.internal` 里注册（见 `commands/_internal.py`）。


def run_fill_turnover(args) -> None:
    """`--fill-turnover` 的处理体：只做参数校验，干活在 `metadata_save.fill_turnover`。"""
    from datetime import date, timedelta

    from GolemQ.markets.StockCN.metadata_save import fill_turnover

    codes = None
    if getattr(args, 'fill_turnover_codes', None):
        codes = [c.strip() for c in str(getattr(args, 'fill_turnover_codes', None)).split(',') if c.strip()]
        if not codes:
            print('用法错误: --fill-turnover-codes 只给了分隔符，解析不出代码',
                  file=sys.stderr)
            sys.exit(2)

    start = getattr(args, 'fill_turnover_start', None) or (date.today() - timedelta(days=7)).isoformat()
    end = getattr(args, 'fill_turnover_end', None) or date.today().isoformat()
    if start > end:
        print('用法错误: --fill-turnover-start {} 晚于 --fill-turnover-end {}'.format(
            start, end), file=sys.stderr)
        sys.exit(2)

    out = fill_turnover(codes, start=start, end=end)
    print('[fill_turnover] 扫 {} 只 / 补 {} 天 / 跳过 {} 天 / upserted {} / modified {}'.format(
        out['codes'], out['filled'], out['skipped'], out['upserted'], out['modified']))


FILL_TURNOVER = Command(
    name='fill-turnover',
    flags=('fill_turnover',),
    add_arguments=add_fill_turnover_arguments,
    run=run_fill_turnover,
)
