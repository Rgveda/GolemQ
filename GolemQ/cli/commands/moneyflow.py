# coding:utf-8
"""`--moneyflow` —— 抓资金流 / 千股千评并落 `stock_reality_feat_day`。

旧树是 `cli/__main__.py:514` 那个分支（`-f/--moneyflow`）。⚠️ 旧分支**名字骗人**：
取值列了 `eastmoney` / `ths`，分支体里**只有东财**。**这次两个值都是真的** ——
但 `ths` 的来源换了（见下）。

⚠️ 两个源的可用性是**实测**的，不是照旧树采信（2026-10-11）：
东财的 `push2his` / `push2` **两个 host 全不通**（http 502 / https 代理拒），
所以旧树那个**逐只**资金流爬虫搬不了；同花顺的当日全市场快照**是通的**。
详见 `markets/StockCN/crawler/moneyflow.py` 的模块 docstring。

⚠️ **本命令不回填历史** —— 两个源给的都是**当日**快照。
"""
from __future__ import annotations

import sys

from ._registry import Command


def add_moneyflow_arguments(parser) -> None:
    parser.add_argument('--moneyflow',
                        help="抓资金流/千股千评并落 stock_reality_feat_day。"
                             "eastmoney=千股千评（东财，5200 行/11 次请求）；"
                             "ths=当日全市场资金流（同花顺，5210 行/105 次请求）；"
                             "all=两个都跑。"
                             "⚠️ 只有当日快照，不回填历史。",
                        choices=('eastmoney', 'ths', 'all'), default=None)
    # NOTE: `--moneyflow-dry-run` 是内部/调试旗子，只在
    # `python -m GolemQ.cli.internal` 里注册（见 `commands/_internal.py`）。


def run_moneyflow(args) -> None:
    """`--moneyflow` 的处理体：只做参数校验，干活在 `crawler.moneyflow`。"""
    from GolemQ.markets.StockCN.crawler.moneyflow import (MONEYFLOW_SOURCES,
                                                          save_moneyflow)
    sources = MONEYFLOW_SOURCES if args.moneyflow == 'all' else (args.moneyflow,)
    failed = []
    for src in sources:
        try:
            out = save_moneyflow(source=src, dry_run=getattr(args, 'moneyflow_dry_run', False))
        except Exception as exc:                  # noqa: BLE001 逐源隔离
            # 失败**不退出**：一个源挂了不该拖垮另一个。
            print('[moneyflow] {} 失败：{}: {}'.format(src, type(exc).__name__, exc),
                  file=sys.stderr)
            failed.append(src)
            continue
        if not getattr(args, 'moneyflow_dry_run', False) and out['written']:
            print('[moneyflow] ⚠️ 本命令只抓**当日**，不回填历史'
                  '（同花顺无历史接口；东财历史接口实测已停）。')
    if failed and len(failed) == len(sources):
        sys.exit(1)


MONEYFLOW = Command(
    name='moneyflow',
    flags=('moneyflow',),
    add_arguments=add_moneyflow_arguments,
    run=run_moneyflow,
)
