# coding:utf-8
"""scan scope 解析 —— 从旧树 `perpar_symbol_range` 搬来的 **6 个**分支。

⚠️ 本模块是**跨市场**的：它**只认 `dataset` 参数，不 import 任何市场包**。
A 股的数据落点由 `markets/StockCN/` 提供（`StockCN.scan_dataset()`）。

它为什么不在 `markets/StockCN/` 下（2026-10-13 用户点的）
====================================================
原先它就在那儿（`markets/StockCN/scopes.py`），并且直接 `from .refdata import …`。
「scope → 清单」这件事本身与市场无关，放市场包里等于**根层模块直连市场包** ——
`CLAUDE.md` 的「取数的唯一入口是市场实例」与「库名不外泄」两条都违。
搬上来之后，A 股那三样数据（名单 / 板块 / ETF 表）改由 :class:`ScanDataset` 注入。

**留在本模块的**是「**怎么切**」：六个 scope 的名字与含义、`etc = 全市场 − focus_block`、
`etc_1/etc_2` 的奇偶二分与 `>1700` 阈值、空清单护栏、未知名报错。
**注入的**是「**拿什么切**」：码表 / 板块名 / 名单。

搬了哪 6 个 / 丢了哪 14 个
========================
用户 2026-10-13 定：**只搬实际在用的**。旧函数是 358 行 `if/elif`、约 20 个分支，
其中一堆是硬编码垃圾（`hs300` / `zz500` / `zz100` / `csindex` / `sector` / `industry` /
`gcd` / `test` / `all` …，且**多个已实测有缺陷**）—— **都不搬**。

============  ==================================================================
`fast`        旧函数里 **没有** `fast` 分支 —— 它落 `else` 兜底 = `block` 的语义。
`block`       同上（旧 help 写「重要指标股(默认)」）。
`etf`         **8.3 的 `etf_list`**（A 股侧 `refdata.GQ_fetch_etf_list`，经
              `dataset.etf_codes` 注入）。⚠️ 2026-10-18 起**不再用**旧树那张 506 码
              硬编码表（已删 —— 实测它比库里少 **1244 只**，还留着 **44 只已退市/换码**
              的码，其中 13 只近一月无数据 ⇒ 判 mode=1 ⇒ 把休市节流的计数撑爆）。
`etc`         全市场 **减去** `focus_block`。
`etc_1`/`etc_2` `etc` 再按 `(int(code[4])+int(code[5])) % 2` **奇偶二分**（保留旧的 `>1700` 条件）。
============  ==================================================================

⚠️ 三条**搬运时的更正**（都随 commit 记，免得被当成漏搬）
=====================================================
1. **旧 `_fetch_stock_blocks` 自己递归自己**（旧 `GolemQ/utils/symbol.py:631` 第一支调的是
   它自己）⇒ 必然 `RecursionError`、被 `except` 吞掉、再落回 QUANTAXIS。现在读
   `dataset.stock_block(...)`（A 股侧是 `refdata.GQ_fetch_stock_block`，读
   8.3 行情库里的 `stock_block` 集合，**`--save` 写的那个**）。旧代码那段注释说
   「`golemq` 里没有 stock_block」是 **4.4** 的事实，**不是 8.3 的**。
2. **旧 `etc_2` 比 `etc_1` 多一步 7→6 位截断**（`:717` vs `:693` 的不对称）——
   8.3 的板块写入端**源头就截到 6 位**，这里两边都去掉。
3. **空清单加护栏**：旧代码解析出空清单会让 scan **静默什么都不做**（CLI 只判 `is False`）
   ⇒ 这里返回 `False` 并打印原因。

⚠️ **`etf` 清单现在不混族了**（2026-10-18 起）：取自 8.3 的 `etf_list`，
实测 **1706 个码全部是 `etf_cn`**（旧树那张 506 码表混着 30 个真指数，已删）。

**但「调用方不能给它传单一 `market_type`」这条仍然成立** —— 现在它没有额外成本，
也防着名单将来再混；传 `INDEX_CN` 会让 ETF 去读 `index_60min` 而**静默读空**。
传 `None` 让 `is_stock_cn` **逐码**分类：它同时决定**读哪张表**与**写哪个族**。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

#: 支持的 scope。**顺序即 `--help` 的顺序**。
SCOPES = ('fast', 'block', 'etf', 'index', 'all', 'etc', 'etc_1', 'etc_2')


@dataclass(frozen=True)
class ScanDataset:
    """scope 解析所需的**数据落点**。跨市场：换市场 = 换一个实例。

    四个成员**照着 A 股侧现有读口的返回形状定**（`markets/StockCN/refdata.py`），
    不另造形状 —— 否则市场侧还要再包一层「翻译」。

    :param stock_list: ``() -> DataFrame``，含 ``code`` 列（6 位）。
    :param stock_block: ``(blocknames) -> DataFrame``，索引 **`(blockname, code)` 两层**
        （一只股票属于多个板块、一个板块含多只股票，`refdata.GQ_fetch_stock_block` 原样）。
        空集返空帧。
    :param focus_blocks: 关注板块的**名字清单**（A 股侧是 `symbol.focus_block`）。
    :param etf_codes: 「``etf``」这个 scope 的码表。A 股侧取自 **8.3 的 `etf_list`**。
    :param index_codes: 「``index``」这个 scope 的码表（8.3 `index_list`）。
        ``all`` = stock ∪ etf ∪ index。
    """

    stock_list: Callable[..., 'object']
    stock_block: Callable[..., 'object']
    focus_blocks: Sequence[str]
    etf_codes: Sequence[str]
    index_codes: Sequence[str] = ()


def _active_dataset() -> ScanDataset:
    """不传 `dataset` 时取**当前激活市场**的数据落点。

    「取数的唯一入口是市场实例」（`CLAUDE.md`）—— 市场在导入时自注册进 `GQMARKETS`
    （`markets/StockCN/__init__.py`），所以这里不需要任何额外的注册机制。
    """
    from GolemQ.core.market_registry import get_active_market
    market = get_active_market()
    dataset = getattr(market, 'scan_dataset', None)
    if dataset is None:
        raise TypeError(
            f'市场 {market!r} 没有 `scan_dataset()`，无法解析 scan scope。'
            f'若该市场要跑 scan，请实现它（契约见 ScanDataset）。')
    return dataset()


def focus_block_codes(dataset: ScanDataset, verbose: bool = True, echo=None) -> list:
    """`focus_block` 的成分股（**去重、6 位、排序**）。

    点名却不在板块表里的板块**打印一行**（照旧树），因为它们「在通达信板块分类中已被移除」。
    """
    df = dataset.stock_block(dataset.focus_blocks)
    if df is None or len(df) == 0:
        return []
    have = set(df.index.get_level_values('blockname'))
    missing = sorted(set(dataset.focus_blocks) - have)
    if missing and verbose:
        (echo or print)(f'系统预设的A股板块: {missing} 在通达信板块分类中已被移除')
    return sorted({str(c)[:6] for c in df.index.get_level_values('code')})


def perpar_symbol_range(eval_range='', dataset: ScanDataset = None,
                        verbose: bool = True, echo=None):
    """`--eval <scope>` → 代码列表；**解析不出来返回 `False`**（沿用旧树约定）。

    :param eval_range: :data:`SCOPES` 之一（大小写不敏感）。
    :param dataset: 数据落点（见 :class:`ScanDataset`）；``None`` ⇒ 当前激活市场的。
    :returns: ``list[str]``（升序、去重、6 位）或 ``False``。

    >>> import pandas as pd
    >>> idx = pd.MultiIndex.from_tuples(
    ...     [('沪深300', '600519'), ('沪深300', '000001'), ('中证500', '601318')],
    ...     names=['blockname', 'code'])
    >>> ds = ScanDataset(
    ...     stock_list=lambda: pd.DataFrame({'code': ['600519', '000001', '601318', '300750']}),
    ...     stock_block=lambda blocknames: idx.to_frame(index=False).set_index(
    ...         ['blockname', 'code']),
    ...     focus_blocks=['沪深300', '中证500'],
    ...     etf_codes=['510300', '159915'])

    `fast` / `block` = `focus_block` 的成分（去重、排序）：

    >>> perpar_symbol_range('fast', dataset=ds, verbose=False)
    ['000001', '600519', '601318']

    `etc` = 全市场 − 成分：

    >>> perpar_symbol_range('etc', dataset=ds, verbose=False)
    ['300750']

    `etf` 直接用注入的码表；未知 scope 返 `False` 而不是抛：

    >>> perpar_symbol_range('etf', dataset=ds, verbose=False)
    ['159915', '510300']

    未知 scope 返 `False` 而**不是抛**（沿用旧树约定）；诊断行与 `verbose` 无关，
    它是要给人看的错，所以这里用 `echo` 收走：

    >>> perpar_symbol_range('nosuch', dataset=ds, verbose=False,
    ...                     echo=lambda *a: None)
    False
    """
    say = echo or print
    rng = str(eval_range or '').strip().lower()
    if rng not in SCOPES:
        say(f'[scopes] 未知 scope {eval_range!r}；可用: {SCOPES}')
        return False

    ds = _active_dataset() if dataset is None else dataset

    if rng == 'etf':
        codes = list(ds.etf_codes)
    elif rng == 'index':
        codes = list(ds.index_codes)
    elif rng == 'all':
        df = ds.stock_list()
        stocks = [] if df is None or len(df) == 0 else [str(c)[:6] for c in df['code']]
        codes = sorted(set(stocks) | set(ds.etf_codes) | set(ds.index_codes))
    elif rng in ('fast', 'block'):
        codes = focus_block_codes(ds, verbose=verbose, echo=say)
    else:                                   # etc / etc_1 / etc_2
        df = ds.stock_list()
        all_codes = [] if df is None or len(df) == 0 else [str(c)[:6] for c in df['code']]
        if not all_codes:
            say('[scopes] stock_list 为空 —— 先跑一次 `--save tdx` 灌参考数据')
            return False
        codes = sorted(set(all_codes) - set(focus_block_codes(ds, verbose=verbose, echo=say)))
        if rng in ('etc_1', 'etc_2') and len(codes) > 1700:
            want = 1 if rng == 'etc_1' else 0
            codes = [c for c in codes if (int(c[4]) + int(c[5])) % 2 == want]

    codes = sorted(set(codes))
    if not codes:
        # ⚠️ 旧代码在这里会让 scan **静默什么都不做**（CLI 只判 `is False`）。
        say(f'[scopes] scope {rng!r} 解析出**空清单** —— 一个都不会扫。'
            f'（板块名对不上时，先看 8.3 的 stock_block 是怎么命名的）')
        return False
    if verbose:
        say(f'[scopes] {rng}: {len(codes)} 只')
    return codes
