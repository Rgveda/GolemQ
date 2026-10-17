# coding:utf-8
"""`--metadata-align` —— `stock_reality_feat_day` 对 K 线的**覆盖率报告**（只读）。

旧树是 `cli/__main__.py:426` 的 `--stock-min-aligned` 分支（调
`markets/StockCN/align.py::stock_min_aligned`）。⚠️ **不是照搬** ——
旧实现名不副实（全文没有分钟数据）、且依赖 QUANTAXIS + xtquant（已停服）
+ 4.4 的 `stock_valuation`。用户 2026-10-11 定：**保留「对齐」这个能力，
源换成 8.3 自己的 `stock_reality_feat_day`**。理由与新语义见
`markets/StockCN/metadata_align.py` 的模块 docstring。

⚠️ **只读**：报告「哪些交易日缺哪些列」，**不自动补**。
补数据要联网、要限频，不该藏在一条报告命令里 —— 该跑 `--moneyflow` 或 `--save`。
"""
from __future__ import annotations

from ._registry import Command

#: 不给 `--metadata-align-codes` 时的默认样本（跨市场、含ETF，便于看出覆盖差异）
DEFAULT_CODES = ('600519', '000001', '301308', '510300')


def add_metadata_align_arguments(parser) -> None:
    parser.add_argument('--metadata-align',
                        help="只读：报告 stock_reality_feat_day 对 K 线的覆盖率"
                             "（每个标的、每一列各有多少交易日有值）",
                        action='store_true', default=False)
    parser.add_argument('--metadata-align-codes',
                        help="逗号分隔的标的；不传用一个默认样本（{}）"
                             .format(','.join(DEFAULT_CODES)),
                        type=str, default=None)
    parser.add_argument('--metadata-align-frequency',
                        help="用哪个频率的 K 线当轴（默认 day）", type=str,
                        default='day')


def run_metadata_align(args) -> None:
    """`--metadata-align` 的处理体：取 K 线 → 对齐 → 打覆盖率表。"""
    from GolemQ import get_active_market
    from GolemQ.markets.StockCN.metadata_align import align_metadata_to_kline

    codes = ([c.strip() for c in args.metadata_align_codes.split(',') if c.strip()]
             if args.metadata_align_codes else list(DEFAULT_CODES))
    market = get_active_market()
    freq = args.metadata_align_frequency

    print('[metadata-align] 频率 {}，{} 只标的'.format(freq, len(codes)))
    print('{:<10s} {:>6s}  {}'.format('标的', 'K线', '各列有值的行数 / 占比'))
    for code in codes:
        try:
            if freq == 'day':
                d, _ = market.get_kline_price_v3(code, realtime=False, verbose=False)
            else:
                d, _ = market.get_kline_price_min(code, frequency=freq,
                                                  realtime=False, verbose=False)
        except Exception as exc:                  # noqa: BLE001 逐个隔离
            print('{:<10s}   取数失败：{}: {}'.format(code, type(exc).__name__, exc))
            continue
        if d is None or len(getattr(d, 'data', ())) == 0:
            print('{:<10s}   **无 K 线**（该频率下取不到）'.format(code))
            continue
        kline = d.data
        aligned = align_metadata_to_kline(kline)
        if aligned.empty:
            print('{:<10s} {:>6d}  **元数据表里一行都没有**'.format(code, len(kline)))
            continue
        n = len(kline)
        parts = ['{}={}/{}'.format(c, int(aligned[c].notna().sum()), n)
                 for c in aligned.columns]
        print('{:<10s} {:>6d}  {}'.format(code, n, '  '.join(parts)))
    print('[metadata-align] ⚠️ 只读报告，**没有补任何数据**；'
          '要补请跑 --moneyflow（当日）或 --save（换手率）。')


METADATA_ALIGN = Command(
    name='metadata-align',
    flags=('metadata_align',),
    add_arguments=add_metadata_align_arguments,
    run=run_metadata_align,
)
