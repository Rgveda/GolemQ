# coding:utf-8
"""pytdx 数据源适配器 —— 通达信协议的社区实现。

为什么这个源优先
================
重构前，`stock_list` / `stock_block` 是经 QUANTAXIS 的 `QA_SU_save_*('tdx')`
拉取的 —— 也就是说 **QUANTAXIS 一直在替 GolemQ 干这活**。把这两个集合改为
直接经 pytdx 取，是解耦中最实质的一步：它把「谁负责取数」从 QUANTAXIS 手里
拿回来。

能供什么、不能供什么（已实测）
==============================
================  ==========================================================
`stock_list`      ✅ 完整。返回字段 ``code/volunit/decimal_point/name/pre_close``
                    与目标 schema 几乎逐一对应，只需补 ``sse`` 与 ``sec``
`stock_block`     ✅ 完整。四个板块文件：概念(gn)/行业(yb)/指数(zs)/风格(fg)
`stock_info`      ⚠️ 部分。``get_company_info_*`` 只给 ipo_date/industry/province，
                    **拿不到 ``liutongguben``/``zongguben``** —— 而那是唯一被真实消费的字段
`etf_list`        ❌ 无 ETF 分类清单
`financial`       ❌ ``get_finance_info`` 只返回当期几个比率，不是报告期归档
================  ==========================================================

故本适配器只声明 ``collections = (STOCK_LIST, STOCK_BLOCK)``。
不支持的集合走 :class:`UnsupportedCollection` 明确报错，**不返回空列表**。

⚠️ 已知缺口：**北交所取不到**（已实测）
=====================================
``get_security_count(2)`` 会报 383 只，但 ``get_security_list(2, 0)`` **恒返回 0 行**。
已在 4 台服务器（123.125.108.14 / 180.153.18.170 / 115.238.90.165 / 124.71.187.122）
上复现，且 ``get_security_list2`` / ``get_security_list3`` 同样为 0 ——
是 **pytdx 库对 market 2 的能力缺口**，换服务器无用。

后果：``stock_list`` 实收约 **5426** 条，对照 4.4 的 5591 条**少约 369 只北交所**
（``82``/``92`` 前缀）。本轮接受该缺口并在此记录；若要补齐，需另找源
（MiniQMT 的 ``get_stock_list_in_sector`` 或 akshare 的北交所清单）。

⚠️ 限频：pytdx 不是 HTTP
========================
它走通达信私有 TCP 协议，**不受 HTTP 频率限制，`requests` 的 proxies 对它无效**
（见 `proxy.py` 的说明）。因此本源的 ``default_interval`` 是 **0** ——
把 30s 默认值套上来会让一次全量拉取（SZ+SH 分页各约 25~28 次调用）平白多花
十几分钟。30s 那个默认值是为 baostock/akshare/eastmoney 这些有频率控制准备
的，不要在这里沿用。

⚠️ 服务器列表是配置项，不是代理
==============================
``hosts`` 是**通达信行情服务器地址**（轮换/中继），与 HTTP 代理是两回事，
不要混用（见 `proxy.py`）。默认服务器已实测可用，但机器在不同网络环境下未必。
"""
from __future__ import annotations

from .base import (
    STOCK_BLOCK,
    STOCK_LIST,
    DataSource,
    DataSourceNotAvailable,
    UnsupportedCollection,
    register,
)

#: 2026-09-20 实测可用。仅作为兜底，正常应由配置提供。
DEFAULT_HOSTS = (('123.125.108.14', 7709),)

#: 4.4 `quantaxis.stock_list` 实测的代码前缀分布（合计 5591）。
#: pytdx 的证券列表混有基金/债券/指数，必须按此前缀过滤才能得到与既有数据一致的集合。
STOCK_CODE_PREFIXES = ('00', '30', '60', '68', '82', '92')

#: 板块文件 → GolemQ 的 type 取值
BLOCK_FILES = {
    'block_gn.dat': 'gn',    # 概念
    'block.dat': 'yb',       # 行业
    'block_zs.dat': 'zs',    # 指数
    'block_fg.dat': 'fg',    # 风格
}


def _market_of(code: str) -> str:
    """6 位代码 → 交易所。与 4.4 既有数据的 sse 口径对齐（sz/sh/bj）。"""
    if code.startswith(('60', '68')):
        return 'sh'
    if code.startswith(('00', '30')):
        return 'sz'
    # 82/92 为北交所
    return 'bj'


@register
class TdxSource(DataSource):
    name = 'pytdx'
    collections = (STOCK_LIST, STOCK_BLOCK)
    #: TCP 协议，无 HTTP 频率限制 —— 显式覆盖 30s 默认值，理由见模块文档。
    default_interval = 0.0

    def __init__(self, throttle=None, proxy=None, hosts=None):
        super().__init__(throttle=throttle, proxy=proxy)
        self.hosts = tuple(hosts) if hosts else DEFAULT_HOSTS
        self._api = None
        self._host = None

    # ---- 可用性 --------------------------------------------------------

    def available(self) -> bool:
        """pytdx 可导入且有候选服务器即算可用。**不在此处做网络探测** ——
        排障路径本身不该成为故障点，真正的连通性失败由 `fetch()` 抛出。"""
        try:
            import pytdx.hq  # noqa: F401
        except ImportError:
            return False
        return bool(self.hosts)

    # ---- 连接 ----------------------------------------------------------

    def _connect(self):
        """连上第一个可用的服务器并保持。失败时逐个换，全失败才报不可用。"""
        if self._api is not None:
            return self._api
        try:
            from pytdx.hq import TdxHq_API
        except ImportError as exc:
            raise DataSourceNotAvailable('pytdx 未安装') from exc

        last_err = None
        for host, port in self.hosts:
            try:
                api = TdxHq_API(heartbeat=False)
                api.connect(host, port, time_out=10)
                # 探一次真实取数，确认连接可用（connect 本身不验证协议层）
                api.get_security_count(0)
                self._api, self._host = api, (host, port)
                return api
            except Exception as exc:      # noqa: BLE001 - 逐个换服，记录最后错误
                last_err = exc
        raise DataSourceNotAvailable(
            f'pytdx 无可用行情服务器（试过 {len(self.hosts)} 个）：{last_err}'
        )

    @property
    def host(self):
        return self._host

    def close(self):
        if self._api is not None:
            try:
                self._api.disconnect()
            finally:
                self._api = None

    # ---- 取数 ----------------------------------------------------------

    def fetch(self, collection: str, **kwargs) -> list:
        if not self.supports(collection):
            raise UnsupportedCollection(
                f'{self.name} 不提供 {collection}（可提供 {self.collections}）'
            )
        self.gate()
        if collection == STOCK_LIST:
            return self.fetch_stock_list(**kwargs)
        return self.fetch_stock_block(**kwargs)

    # ---- stock_list ----------------------------------------------------

    def fetch_stock_list(self, markets=None, verbose=False) -> list:
        """全市场股票列表，已过滤为真实股票（排除基金/债券/指数）。

        过滤按 4.4 `quantaxis.stock_list` 实测的前缀分布反推，
        保证新库的标的集合与既有数据一致 —— 不是拍脑袋定的范围。
        """
        api = self._connect()
        from pytdx.params import TDXParams

        # 0=SZ, 1=SH, 2=BJ。
        # 北交所（2）见模块文档的已知缺口：count 有值但 list 恒空，
        # 多服务器多方法均已实测无效。留着尝试是为了「哪天上游修了能自动接上」——
        # 它失败时只跳过，不会让整批取数失败。
        market_enum = markets or (
            (TDXParams.MARKET_SZ, 'sz'),
            (TDXParams.MARKET_SH, 'sh'),
            (2, 'bj'),
        )

        rows: list = []
        for market, sse in market_enum:
            try:
                total = api.get_security_count(market)
            except Exception:
                # 北交所在部分服务器上不可用 —— 跳过而不是整批失败
                if verbose:
                    print(f'[pytdx] market={market} 不可用，跳过')
                continue
            for start in range(0, total, 1000):
                self.gate()
                try:
                    batch = api.get_security_list(market, start) or []
                except Exception:
                    break
                for r in batch:
                    code = str(r.get('code', ''))
                    if not code.startswith(STOCK_CODE_PREFIXES):
                        continue
                    rows.append({
                        'code': code,
                        'volunit': r.get('volunit'),
                        'decimal_point': r.get('decimal_point'),
                        'name': r.get('name'),
                        'pre_close': r.get('pre_close'),
                        'sse': sse,
                        'sec': 'stock_cn',
                    })
        return rows

    # ---- stock_block ---------------------------------------------------

    def fetch_stock_block(self, verbose=False) -> list:
        """板块成分。四个文件分别对应概念/行业/指数/风格。

        代码统一截成 6 位 —— 通达信原始数据带市场前缀（``S600000`` 型 7 位），
        而 `fetch.py` 的调用方做的是 ``code[1:7] if len(code)==7 else code``
        的防御式截断；在源头就截断更干净。
        """
        api = self._connect()
        rows: list = []
        for fname, btype in BLOCK_FILES.items():
            self.gate()
            try:
                info = api.get_and_parse_block_info(fname) or []
            except Exception as exc:      # noqa: BLE001
                if verbose:
                    print(f'[pytdx] 板块文件 {fname} 解析失败: {exc}')
                continue
            for item in info:
                raw = str(item.get('code', ''))
                code = raw[1:7] if len(raw) == 7 else raw
                if len(code) != 6:
                    continue
                rows.append({
                    'blockname': item.get('blockname'),
                    'code': code,
                    'type': btype,
                    'source': self.name,
                })
        return rows
