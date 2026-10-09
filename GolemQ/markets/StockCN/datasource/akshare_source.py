# coding:utf-8
"""akshare 数据源适配器。供 `etf_list` 与 `financial`。

为什么是 akshare 而不是 baostock 做 financial
=============================================
原计划 `financial` 用 baostock。**实测 baostock 从本机连不上**：
``login`` 返回 ``10002007 网络接收错误``、``[WinError 10057]``，
`query_*` 全部拿不到数据。写了也是无法验证的代码。

改测 akshare，三个财务接口**全部可用**（600519 实测）：

=========================================  ==================  ===============
接口                                        返回                 适配度
=========================================  ==================  ===============
``stock_financial_analysis_indicator``      (10 期 × 86 指标)    **最佳**，一行一报告期
``stock_financial_abstract``                (80, 105)           报告期为列，需转置
``stock_financial_report_sina``             (103, 147)          单张报表，需拼装
=========================================  ==================  ===============

故选用 ``stock_financial_analysis_indicator``。

⚠️ 指标名保留中文
=================
该接口的 86 个指标名是中文（如「摊薄每股收益(元)」）。**原样存中文键** ——
映射成英文意味着凭空造 86 条对照表，且多数指标并无既有消费方，造了也无从验证。
MongoDB 对 UTF-8 键名无障碍。若日后确需英文键，再按实际用到的指标逐步映射，
而不是一次造满。

⚠️ 限频：akshare 确有访问频率控制
=================================
走本包 `throttle.py`，默认 30s。**不套用** `scribe.GQ_get_etf_list` 的
`checkin_function` 闸门（15 分钟「到期前拒绝」）—— 两者语义不同，见 `throttle.py`。

⚠️ 逐只查询，全量很慢
=====================
`financial` 与 `stock_info` 都是**按代码逐只查**。5000+ 只 × 30s 间隔 = 数十小时。
故 `fetch_financial` 支持 `codelist` 限定范围，全量回填应作为长任务分批跑。
"""
from __future__ import annotations

from GolemQ.core.presentation import suppress_stdout_stderr
from GolemQ.datasource.base import (
    ETF_LIST,
    FINANCIAL,
    DataSource,
    DataSourceNotAvailable,
    UnsupportedCollection,
    register,
)

#: 与 `scribe.GQ_get_etf_list` 保持一致的列映射。改动需同步。
ETF_COLUMN_MAP = {
    '代码': 'code',
    '名称': 'name',
    '昨收': 'last_close',
    '成交量': 'volume',
    '成交额': 'amount',
    '换手率': 'turnover_rate',
    '最新价': 'price',
    '涨跌幅': 'pct_change',
    '最高': 'high',
    '最低': 'low',
    '今开': 'open',
    '总市值': 'capitalization',
}

#: etf_list 落库时要保留的列 —— 只留映射过的 + 派生列。
#: akshare 原表还带「涨跌额/买入/卖出」等未映射列，不过滤会污染 schema。
ETF_KEEP = tuple(ETF_COLUMN_MAP.values()) + ('sse', 'sec', 'decimal_point', 'volunit', 'source')

#: financial 落库键（其余列即指标，原样保留）
FINANCIAL_KEYS = ('code', 'report_date', 'year', 'quarter', 'source')


def _to_quarter(report_date: str):
    """``'2026-06-30'`` → ``(2026, 2)``。非季末返回 (year, None)。"""
    try:
        y, m, _ = str(report_date).split('-')
        return int(y), (int(m) // 3 if int(m) in (3, 6, 9, 12) else None)
    except Exception:  # noqa: BLE001
        return None, None


@register
class AkshareSource(DataSource):
    name = 'akshare'
    collections = (ETF_LIST, FINANCIAL)
    #: akshare 有频率控制 —— 沿用 30s 默认间隔。
    default_interval = 30.0

    def available(self) -> bool:
        try:
            import akshare  # noqa: F401
        except ImportError:
            return False
        return True

    def fetch(self, collection: str, **kwargs) -> list:
        if not self.supports(collection):
            raise UnsupportedCollection(
                f'{self.name} 不提供 {collection}（可提供 {self.collections}）'
            )
        if not self.available():
            raise DataSourceNotAvailable('akshare 未安装')
        if collection == ETF_LIST:
            self.gate()
            return self.fetch_etf_list(**kwargs)
        return self.fetch_financial(**kwargs)

    # ---- etf_list ------------------------------------------------------

    def fetch_etf_list(self, verbose: bool = False) -> list:
        """新浪 ETF 分类快照 → GolemQ 口径的行。**不落库**。

        写库键为 `code`，**无 delta 删除** —— 这是快照不是权威名单，
        用快照删差量会把停牌或新上市未收录的 ETF 误删。
        """
        import akshare as ak

        df = ak.fund_etf_category_sina(symbol='ETF基金')
        if df is None or len(df) == 0:
            raise DataSourceNotAvailable('akshare fund_etf_category_sina 返回空')

        df = df.rename(columns=ETF_COLUMN_MAP)

        # 新浪返回的代码带交易所前缀（'sh510300'）：前两位是交易所、其余是代码。
        df['sse'] = df['code'].str[:2].str.lower()
        df['code'] = df['code'].str[2:]
        df['sec'] = 'etf_cn'
        df['decimal_point'] = 3
        df['volunit'] = 100

        # 只保留映射过的 + 派生列，剔除 akshare 原表的未映射列
        keep = [c for c in ETF_KEEP if c in df.columns]
        rows = df[keep].to_dict('records')
        for r in rows:
            r.setdefault('source', self.name)
        if verbose:
            print(f'[akshare:etf_list] 取到 {len(rows)} 条')
        return rows

    # ---- financial -----------------------------------------------------

    def fetch_financial(self, codelist=None, start_year: str = None,
                        verbose: bool = False) -> list:
        """季频财务指标（`stock_financial_analysis_indicator`）→ 宽表行。

        一行 = 一个 `(code, report_date)`。指标名保留中文，见模块文档。
        `codelist` 省略则取全市场 —— **会很慢**，见模块文档的耗时说明。
        """
        import akshare as ak
        import datetime as _dt

        if codelist is None:
            from . import get_source
            codelist = [r['code'] for r in get_source('pytdx').fetch('stock_list')]
        elif isinstance(codelist, str):
            codelist = [codelist]
        start_year = start_year or str(_dt.date.today().year - 1)

        rows: list = []
        # `desc` 报**当前 code**，不报固定的 `[akshare:financial]` —— 跑的是哪一只
        # 比跑的是哪个集合有用（与 pytdx 的 `fetch_stock_info` 同一口径）。
        from tqdm import tqdm
        bar = tqdm(codelist, unit='stock', disable=None, leave=False)
        for code in bar:
            code = str(code).split('.')[0][-6:]
            bar.set_description(code)
            self.gate()
            try:
                # akshare 内部有 tqdm（见上），逐只调用会闪 —— 压掉它
                with suppress_stdout_stderr():
                    df = ak.stock_financial_analysis_indicator(symbol=code,
                                                               start_year=start_year)
            except Exception as exc:      # noqa: BLE001 单只失败不中断全批
                if verbose:
                    print(f'[akshare:financial] {code} 失败: {exc!r}')
                continue
            if df is None or len(df) == 0:
                continue
            for rec in df.to_dict('records'):
                report_date = str(rec.pop('日期', '') or '')
                if not report_date:
                    continue
                year, quarter = _to_quarter(report_date)
                row = {
                    'code': code,
                    'report_date': report_date,
                    'year': year,
                    'quarter': quarter,
                    'source': self.name,
                }
                # 其余列即指标，原样保留（中文键）
                for k, v in rec.items():
                    if k in FINANCIAL_KEYS:
                        continue
                    row[k] = None if v is None or v != v else v   # NaN → None
                rows.append(row)
        if verbose:
            print(f'[akshare:financial] 取到 {len(rows)} 行，覆盖 {len(codelist)} 只')
        return rows
