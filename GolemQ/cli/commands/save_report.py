# coding:utf-8
"""`--save` 的**只读**报告：`--save-coverage`（K 线覆盖缺口）与 `--save-status`（库存量/源可用性）。

两条都是**独立命令**（老链里也是独立 `elif`），都**不写库**。
⚠️ `--save-coverage` 在分发顺序里**必须排在 `--save` 之前**（两个都给时老链选
coverage）—— 见 `commands/__init__.py` 的登记列表。
"""
from __future__ import annotations

from ._registry import Command


def add_coverage_arguments(parser) -> None:
    parser.add_argument('--save-coverage', action='store_true', default=False,
                        help="只读：出 K 线覆盖缺口报告（含 index/etf），不写库")


def add_status_arguments(parser) -> None:
    parser.add_argument('--save-status',
                        help="查看参考集合的库存量与各数据源可用性，不写库",
                        action="store_true",
                        default=False)


def run_coverage(args) -> None:
    # 只读：覆盖缺口报告（含「存在但缺根」的抽查）。**不写库**。
    from GolemQ.markets.StockCN.kline_status import (
        format_kline_status,
        kline_status,
        roots_check,
    )
    # ⚠️ `--save-frequencies` / `--save-targets` / `--save-no-adj` 由 `save.py` 注册
    # （它们是 `--save` 家族的共用开关），这里只读不注册。
    cov_freqs = ([x.strip() for x in args.save_frequencies.split(',') if x.strip()]
                 if args.save_frequencies else ('day',))
    if any(f != 'day' for f in cov_freqs):
        print('[coverage] 含分钟线核对，代价高一个量级（每个 code 要 distinct 一遍）——'
              ' 可以再用 --save-frequencies 缩小范围')
    cov_targets = ([x.strip() for x in args.save_targets.split(',') if x.strip()]
                   if args.save_targets else None)
    print(format_kline_status(kline_status(
        targets=cov_targets, frequencies=cov_freqs,
        with_adj=(not args.save_no_adj) and (cov_targets is None or 'stock' in cov_targets),
        verbose=args.verbose)))
    if '1min' in cov_freqs:
        print(roots_check('stock_1min', verbose=False))


def run_status(args) -> None:            # noqa: ARG001 约定签名
    # 只读：报告库存量与各源可用性，不写库
    from GolemQ.markets.StockCN.refdata_save import format_status
    print(format_status())


SAVE_COVERAGE = Command('save-coverage', ('save_coverage',),
                        add_coverage_arguments, run_coverage)
SAVE_STATUS = Command('save-status', ('save_status',),
                      add_status_arguments, run_status)
