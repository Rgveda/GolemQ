# coding:utf-8
"""同花顺「牛叉诊股」→ **`stock_reality_feat_day`**（`--save tdx` 的元数据段之一）。

从哪来
======
旧树 `fetch/crawler.py` 的两个函数：

* :func:`get_stock_diagnosis_details` —— 逐只抓 `10jqka` 的个股页面；
* :func:`GQ_SU_crawl_stock_diagnosis` / `_v2` / `_v3` —— 把结果写进
  `golemq.stock_diagnosis*` 三张 4.4 集合。

新树没有 4.4 那三张表，用户 2026-10-11 定：日频元数据**统一落**
`stock_reality_feat_day`（`DECISIONS.md` D50）。本模块因此只保留
**取数 + 解析 + 写这一张表**，不搬 4.4 的三份落库副本。

⚠️ 旧抓取地址 **已改版**（2026-10-22 实测）
==========================================
* `stockpage.10jqka.com.cn/<code>/` 现在的标题是「个股资金流向查询」，
  旧的 `analyze-tips` / `analyze-txt` / `analyze-stars` **一个都没有**；
* 诊股页搬到了 **`doctor.10jqka.com.cn/<code>/`**，标题「牛叉诊股」，
  页面结构也换了 —— 现在是「综合诊断 + 五个维度」：

  ==================  ==========================
  页面                    落库列
  ==================  ==========================
  综合诊断 6.7 分          ``overall``
  打败了 98% 的股票         ``pctRank``（小数 0.98）
  技术面 7.5              ``TechScr``
  资金面 6.8              ``MoneyflowScr``
  消息面 6.6              ``NewsScr``
  行业面 3.5              ``IndustryScr``
  基本面 6.8              ``BasicScr``
  平均成本 1314.02 元       ``DiagAvgCost``
  ==================  ==========================

日期取页面上的**诊断日期**（``[诊断日期:2026年10月22日 12:46]``）——
周末抓到的就是上一个交易日的诊断，不能拿 `today()` 顶（那会写出一个
当天并不存在的行）。

⚠️ 每只一次 HTTP（实测 ~0.2s），全市场 5000+ 只约 20–40 分钟。
`--save tdx` 里排在最后，与它「收盘后那一步」的定位一致。
"""
from __future__ import annotations

import re
import math
import time
from typing import Iterable, Optional

import pandas as pd

from GolemQ.core.constants import AKA, FIELD as FLD

from ..metadata_save import (COVERAGE_SKIP_RATIO, REALITY_FEAT_DAY_COLL, _db,
                             metadata_covered, metadata_day_doc_multi)

__all__ = ['parse_diagnosis', 'fetch_diagnosis', 'rows_diagnosis',
           'save_diagnosis', 'DIAG_DIMENSIONS', 'DIAG_PAYLOAD']

DIAG_URL = 'http://doctor.10jqka.com.cn/{}/'
DIAG_HEADERS = {
    'User-Agent': ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                   'AppleWebKit/537.36 (KHTML, like Gecko) '
                   'Chrome/120.0 Safari/537.36'),
}
#: 逐只抓取的间隔（秒）。同花顺对高频抓取会返回空页/验证码；
#: 旧树默认 0.8–1.2s，这里压到 0.2s —— 实测 600519 连续 20 次无异常。
DIAG_DELAY = 0.2

#: 页面维度名 → 页面段落 id → 落库列。顺序即页面顺序。
#: ⚠️ **必须按段落 id 定位**：页面顶部导航里五个「…面诊股」全都在正文之前，
#: 只按文字找 `诊断结果` 会让每一段都命中**技术面**那一个（实测五维全等于 7.5）。
DIAG_DIMENSIONS = (
    ('技术面', 'nav_technical', AKA.TECH_SCORE),
    ('资金面', 'nav_funds', FLD.MONEYFLOW_SCORE),
    ('消息面', 'nav_message', AKA.NEWS_SCORE),
    ('行业面', 'nav_trade', AKA.INDUSTRY_SCORE),
    ('基本面', 'nav_basic', AKA.BASIC_SCORE),
)

#: 本模块负责的列（**只写这些**）—— `stock_reality_feat_day` 是多列共用的表。
DIAG_PAYLOAD = (AKA.OVERALL, AKA.PCT_RANKING, AKA.DIAG_AVG_COST,
                AKA.TECH_SCORE, FLD.MONEYFLOW_SCORE, AKA.NEWS_SCORE,
                AKA.INDUSTRY_SCORE, AKA.BASIC_SCORE)

_SECTION_RE = {label: re.compile(
    r'id=["\']' + section + r'["\'].*?诊断结果[：:]\s*(?:<em>)?\s*'
    r'([0-9]+(?:\.[0-9]+)?)', re.S)
    for label, section, _ in DIAG_DIMENSIONS}
_OVERALL_RE = re.compile(r'综合诊断[：:]\s*([0-9]+(?:\.[0-9]+)?)\s*分')
_PCT_RE = re.compile(r'打败了\s*([0-9]+(?:\.[0-9]+)?)\s*%')
_DATE_RE = re.compile(r'诊断日期[：:]\s*(\d{4})年(\d{1,2})月(\d{1,2})日')
_COST_RE = re.compile(r'平均成本为\s*(?:<strong>)?\s*([0-9]+(?:\.[0-9]+)?)\s*元')


def parse_diagnosis(html_text: str) -> Optional[dict]:
    """牛叉诊股页 HTML → 数值字典；取不到任何评分时返回 ``None``。**纯函数**。

    >>> h = ('<div>综合诊断：6.7分 打败了98%的股票！</div>'
    ...      '<div id="nav_technical"><h2>技术面诊股</h2>'
    ...      '<span>诊断结果：<em>7.5</em></span>'
    ...      '...平均成本为<strong>1314.02元</strong>...'
    ...      '[诊断日期:2026年10月22日 12:46]</div>')
    >>> d = parse_diagnosis(h)
    >>> d['overall'], d['pct_rank'], d['avg_cost'], d['scores'], d['day']
    (6.7, 0.98, 1314.02, {'技术面': 7.5}, '2026-10-22')
    >>> parse_diagnosis('') is None
    True
    """
    if not html_text:
        return None
    overall = _OVERALL_RE.search(html_text)
    scores = {}
    for label, _section, _ in DIAG_DIMENSIONS:
        m = _SECTION_RE[label].search(html_text)
        if m:
            scores[label] = float(m.group(1))
    if overall is None and not scores:
        return None

    out = {'scores': scores}
    if overall is not None:
        out['overall'] = float(overall.group(1))
    pct = _PCT_RE.search(html_text)
    if pct is not None:
        out['pct_rank'] = float(pct.group(1)) / 100.0
    cost = _COST_RE.search(html_text)
    if cost is not None:
        out['avg_cost'] = float(cost.group(1))
    date = _DATE_RE.search(html_text)
    if date is not None:
        out['day'] = '{:04d}-{:02d}-{:02d}'.format(
            int(date.group(1)), int(date.group(2)), int(date.group(3)))
    return out


def fetch_diagnosis(code: str, timeout: int = 15) -> Optional[dict]:
    """抓单只股票 → :func:`parse_diagnosis` 的字典（附 ``code``）。网络失败抛异常。

    ⚠️ 页面声明 ``charset=GBK``，`requests` 猜编码会得到乱码 —— 必须显式设。
    """
    import requests

    code = str(code).strip()[:6]
    resp = requests.get(DIAG_URL.format(code), headers=DIAG_HEADERS,
                        timeout=timeout)
    resp.raise_for_status()
    resp.encoding = 'gbk'
    out = parse_diagnosis(resp.text)
    if out is not None:
        out['code'] = code
    return out


def rows_diagnosis(items: Iterable[dict], created_at=None) -> list:
    """解析结果 → `stock_reality_feat_day` 文档列表（``(code, date)`` 唯一）。

    与 :func:`moneyflow.rows_eastmoney` 同一套文档构造
    （``metadata_day_doc_multi``）：值为 ``None``/NaN 的列**不写**。
    """
    stamp = int(created_at if created_at is not None else time.time())
    docs = []
    for item in items:
        code = str(item.get('code') or '').strip()[:6]
        day = item.get('day')
        if not code or not day:
            continue
        fields = {
            AKA.OVERALL: item.get('overall'),
            AKA.PCT_RANKING: item.get('pct_rank'),
            AKA.DIAG_AVG_COST: item.get('avg_cost'),
        }
        for label, _section, col in DIAG_DIMENSIONS:
            fields[col] = item.get('scores', {}).get(label)
        docs.append(metadata_day_doc_multi(code, day, fields, stamp))
    return docs


def _resolve_codes(codes=None) -> list:
    """``--save-codes`` 优先；否则读 `stock_list` 全市场。"""
    if codes:
        return [str(c).strip()[:6].zfill(6) for c in codes if str(c).strip()]
    from ..refdata import GQ_fetch_stock_list
    df = GQ_fetch_stock_list()
    if df is None or len(df) == 0 or 'code' not in df.columns:
        return []
    return sorted({str(c).strip()[:6].zfill(6) for c in df['code']})


def save_diagnosis(codes=None, delay: float = DIAG_DELAY, verbose: bool = True,
                   echo=None, dry_run: bool = False, timeout: int = 15) -> dict:
    """逐只抓牛叉诊股 → `stock_reality_feat_day`。

    :param codes: 限定代码；``None`` ⇒ `stock_list` 全市场。
    :param delay: 每只之间的间隔（秒），见 :data:`DIAG_DELAY`。
    :param dry_run: 只取数并统计，**不写库**。
    :returns: ``{'source','day','read','written','upserted','modified',
        'failed','error','dry_run'}``
    """
    say = echo or print
    codelist = _resolve_codes(codes)
    out = {'source': 'diagnosis', 'day': None, 'read': 0, 'written': 0,
           'upserted': 0, 'modified': 0, 'failed': 0, 'error': None,
           'skipped': False, 'dry_run': bool(dry_run)}
    if not codelist:
        out['error'] = 'no codes（stock_list 为空）'
        say('[diagnosis] 没有可抓取的代码：{}'.format(out['error']))
        return out

    # 短路：同一交易日已覆盖 >=90% 的目标代码，就不逐只抓（全市场 20–40 分钟）。
    # 页面的「诊断日期」在交易日就是今天、周末是上一交易日，与 `last_trade_day()` 同口径。
    from ..date_utils import GQ_util_get_last_day
    day = str(GQ_util_get_last_day())[:10]
    cov = metadata_covered(AKA.TECH_SCORE, day, codes=codelist)
    threshold = max(1, math.ceil(COVERAGE_SKIP_RATIO * len(codelist)))
    if not dry_run and cov['covered'] >= threshold:
        out.update(day=day, skipped=True)
        say('[diagnosis] 短路：{} 已覆盖 {}/{} 只，跳过抓取'
            .format(day, cov['covered'], len(codelist)))
        return out

    # 进度条：全市场 20–40 分钟，没有它屏上只有一个绿点。
    # `disable=None` = 非 TTY（日志/管道）自动关，不会往日志里灌控制字符；
    # banner 活跃时也安全 —— 它的 `_paint` 每行都带 `\033[K` 并给 tqdm 留了一行
    # （与 `kline_save` 的 tqdm 同一套做法）。
    from tqdm import tqdm
    bar = tqdm(codelist, unit='stock', desc='diagnosis', disable=None, leave=False)
    items, failed = [], []
    try:
        for i, code in enumerate(bar, 1):
            try:
                item = fetch_diagnosis(code, timeout=timeout)
            except Exception as exc:             # noqa: BLE001 逐只隔离
                item = None
                out['error'] = '{}: {}'.format(type(exc).__name__, exc)
            if item is None:
                failed.append(code)
            else:
                items.append(item)
            bar.set_postfix(ok=len(items), fail=len(failed))
            # 文字进度锚点**只在 `--verbose` 下打**（用户 2026-10-23：默认输出要干净）。
            # 默认仍有 tqdm 进度条；非 TTY 下进度条自动关掉，那是日志场景，
            # 本来就不该刷 100 行进度。
            if verbose and (i % 100 == 0 or i == len(codelist)):
                say('[diagnosis] {}/{}：已取 {} 只 / 失败 {} 只'
                    .format(i, len(codelist), len(items), len(failed)))
            if delay and i < len(codelist):
                time.sleep(delay)
    finally:
        bar.close()

    docs = rows_diagnosis(items)
    out.update(read=len(items), failed=len(failed), written=len(docs))
    if docs:
        out['day'] = docs[0]['date']
    if dry_run:
        say('[diagnosis] 干跑：{} / {} 只可写（【没有写库】）'
            .format(len(docs), len(codelist)))
        return out
    if not docs:
        if out['error'] is None:
            out['error'] = 'no diagnosis rows'
        say('[diagnosis] **没取到任何可落库的行**：{}'.format(out['error']))
        return out

    from GolemQ.core.constants import MARKET_TYPE
    from ..reality_save import save_reality_feat

    frame = pd.DataFrame(docs)
    keep = [c for c in dict.fromkeys(('code', 'date', *DIAG_PAYLOAD))
            if c in frame.columns]
    frame = frame.loc[:, keep].set_index(['date', 'code'])
    frame = frame[~frame.index.duplicated(keep='last')]
    stats = save_reality_feat(frame, 'day', MARKET_TYPE.STOCK_CN,
                              verbose=False, db=_db())
    out.update({k: stats.get(k, 0)
                for k in ('written', 'upserted', 'modified')})
    if verbose:
        say('[diagnosis] → {}：{} / {} 只可写，写 {} 文档 / 新增 {} / 更新 {}'
            .format(REALITY_FEAT_DAY_COLL, len(docs), len(codelist),
                    out['written'], out['upserted'], out['modified']))
    return out
