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
`stock_list`      ✅ 完整。字段 ``code/volunit/decimal_point/name/pre_close``
                    与目标 schema 几乎逐一对应，只需补 ``sse`` 与 ``sec``
`stock_block`     ✅ 完整。四个板块文件：概念(gn)/行业(yb)/指数(zs)/风格(fg)
`stock_info`      ✅ 完整。``get_finance_info`` 给的字段名与目标 schema **逐字相同**
`etf_list`        ❌ 无 ETF 分类清单
`financial`       ❌ ``get_finance_info`` 是**当期快照**，不是报告期归档
================  ==========================================================

故本适配器声明 ``collections = (STOCK_LIST, STOCK_BLOCK, STOCK_INFO)``。
不支持的集合走 :class:`UnsupportedCollection` 明确报错，**不返回空列表**。

⚠️ 更正：曾误判 `stock_info` 拿不到股本
=======================================
本模块早期版本（及其上一级的方案）声称 ``get_finance_info``
「拿不到 ``liutongguben``/``zongguben``」—— **这是错的，且当时未经实测**。
实测 ``get_finance_info`` 返回的正是这些字段，**且字段名与目标 schema 逐字相同**：

========================  ==============================
实测字段                   平安银行 000001
========================  ==============================
``liutongguben``           19,405,685,000
``zongguben``              19,405,918,750
``gudongrenshu``           450,712
``ipo_date``               19910403
``province`` / ``industry`` 18 / 1（**整型编码**，非名称）
``updated_date``           20260815
========================  ==============================

抽查 20 只深市股票，**20/20** 取到 ``zongguben``。

因此 ``stock_info`` 的主源改为 **pytdx**（无需 QMT 在线），QMT 降为备选。

⚠️ 另一处对 ``market=2`` 的更正
===============================
早前判断「北交所取不到」**只对 ``get_security_list`` 成立**：
``get_security_list(2, 0)`` 恒返回空，但 **``get_finance_info(2, code)`` 是通的**
（实测 `920808` 取到完整股本）。所以：
  * ``stock_list`` 的北交所缺口**依然存在**（拿不到代码清单）
  * ``stock_info`` 的北交所**没有缺口**（只要知道代码就能取）

``province`` / ``industry`` 为整型编码，无码表可映射，故**原样存整型** ——
比旧 QMT writer 写 ``None`` 是改进；但若日后要展示，需要另建码表。

⚠️ 北交所（market=2）**不只是取不到，还会毒死连接**
=================================================
``get_security_count(2)`` 会报 383 只，但 ``get_security_list(2, 0)`` 返回 **None**
（已在 4 台服务器上复现，``get_security_list2`` / ``get_security_list3`` 同样为空）。

**真正的危险在于它的副作用**：那一次 None 之后，**同一连接上的所有调用都失效** ——
``get_finance_info`` 返回 None、``get_security_list`` 返回空，**且不抛任何异常**。
pytdx 是请求/响应式 socket，一个畸形响应让字节流错位，此后每次都读错位置。

症状极具误导性：``fetch_stock_info`` 静默拿到 0 行并报 ``skipped``，
看起来像「没有财务数据」，实际是连接被这一句试调用打死了。
**本模块曾为此真实受害** —— 为了「哪天上游修了能自动接上」而保留 market=2 的
尝试，结果把 ``stock_info`` 整条链路变成静默空结果。

故 ``market_enum`` **默认不含 market 2**。若日后确要重试，
**必须用独立连接，用完即弃**。

后果：``stock_list`` 实收 **5226** 条（深 2906 + 沪 2320），
对照 4.4 的 5591 条**少约 369 只北交所**（``82``/``92`` 前缀）。
若要补齐，需另找源 —— ``tdxaidata`` 的 ``get_stock_list(market='北交所')``
实测可取 348 只，已在优先级表里作为 ``stock_list`` 的主源。

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

from GolemQ.datasource.base import (
    STOCK_BLOCK,
    STOCK_INFO,
    STOCK_LIST,
    DataSource,
    DataSourceNotAvailable,
    UnsupportedCollection,
    register,
)

#: 2026-09-20 实测可用。仅作为兜底，正常应由配置提供。
DEFAULT_HOSTS = (('123.125.108.14', 7709),)

#: **按市场各自的**股票代码前缀。
#:
#: 早先用一张统一的前缀表 ``('00','30','60','68','82','92')`` 过滤所有市场 ——
#: **那是错的**：沪市列表里的 ``00xxxx`` 全是**指数**（``000001`` 上证指数、
#: ``000300`` 沪深300），与深市股票**同码**。统一前缀会把它们当股票收进来，
#: 于是 ``000001`` 同时出现两次 —— 一次是上证指数(sh)、一次是平安银行(sz)，
#: 下游按裸 code 建字典时静默丢掉一条，把指数的名字填进股票记录
#: （实测症状：``000001`` 显示 name='上证指数' 却挂 sse='sz'）。
#:
#: 老代码的 docstring 记过同一个坑（「指数池与股票池交集 216 个 000xxx」），
#: 解法同样是**别按裸代码判断市场**。
MARKET_CODE_PREFIXES = {
    0: ('00', '30'),   # 深市：主板 + 创业板
    1: ('60', '68'),   # 沪市：主板 + 科创板（**不含 00xxxx，那些是指数**）
    2: ('82', '92'),   # 北交所
}

#: 兼容旧名：全部市场前缀的并集（仅用于「像不像股票代码」的粗判）
STOCK_CODE_PREFIXES = ('00', '30', '60', '68', '82', '92')

#: 板块文件 → GolemQ 的 type 取值
BLOCK_FILES = {
    'block_gn.dat': 'gn',    # 概念
    'block.dat': 'yb',       # 行业
    'block_zs.dat': 'zs',    # 指数
    'block_fg.dat': 'fg',    # 风格
}


def _tdx_market_of(code: str):
    """6 位代码 → 通达信 market 号（0=深 1=沪 2=北）。非股票返回 None。"""
    if code.startswith(('60', '68')):
        return 1
    if code.startswith(('00', '30')):
        return 0
    if code.startswith(('82', '92')):
        # 北交所。注意 get_security_list(2,0) 取不到清单，但 get_finance_info(2,c)
        # 是可用的（实测 920808）—— 只要知道代码就能取。
        return 2
    return None


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
    collections = (STOCK_LIST, STOCK_BLOCK, STOCK_INFO)
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
        if collection == STOCK_INFO:
            return self.fetch_stock_info(**kwargs)
        return self.fetch_stock_block(**kwargs)

    # ---- stock_list ----------------------------------------------------

    def fetch_stock_list(self, markets=None, verbose=False) -> list:
        """全市场股票列表，已过滤为真实股票（排除基金/债券/指数）。

        过滤按 4.4 `quantaxis.stock_list` 实测的前缀分布反推，
        保证新库的标的集合与既有数据一致 —— 不是拍脑袋定的范围。
        """
        api = self._connect()
        from pytdx.params import TDXParams

        # 0=SZ, 1=SH。
        #
        # ⚠️ **不要把北交所（market=2）加回来。** 它不只是「取不到」——
        # 实测 `get_security_list(2, 0)` 返回 None 后，**整条连接被永久污染**：
        # 之后同一连接上的 `get_finance_info` 与 `get_security_list` 全部返回
        # None/空，而**不抛任何异常**。pytdx 走的是请求/响应式 socket，
        # 一个畸形响应会让字节流错位，此后每次调用都读错位置。
        #
        # 症状极具误导性：`fetch_stock_info` 会静默拿到 0 行、报 skipped，
        # 看起来像「没有财务数据」，实际是连接被这一句试调用打死了。
        # 这个 bug 曾经真实存在 —— 本模块为了「哪天上游修了能自动接上」而保留
        # 了 market=2 的尝试，结果把 `stock_info` 整条链路打成静默空结果。
        #
        # 若日后确要重试北交所，**必须用独立连接**，用完即弃，不能共用。
        market_enum = markets or (
            (TDXParams.MARKET_SZ, 'sz'),
            (TDXParams.MARKET_SH, 'sh'),
        )

        rows: list = []
        seen: set = set()
        for market, sse in market_enum:
            allowed = MARKET_CODE_PREFIXES.get(market, ())
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
                    # **按市场各自的前缀**过滤，不是统一前缀 —— 见
                    # MARKET_CODE_PREFIXES 的说明：沪市 00xxxx 是指数，
                    # 与深市股票同码，统一前缀会制造重复与张冠李戴。
                    if not code.startswith(allowed):
                        continue
                    if code in seen:
                        # 同码跨市场（理论上修好前缀后不该出现）—— 保留先到的
                        # 并留痕，而不是静默覆盖。
                        if verbose:
                            print(f'[pytdx] 重复 code {code}（{sse}）已跳过')
                        continue
                    seen.add(code)
                    rows.append({
                        'code': code,
                        'volunit': r.get('volunit'),
                        'decimal_point': r.get('decimal_point'),
                        'name': r.get('name'),
                        'pre_close': r.get('pre_close'),
                        'sse': sse,
                        'sec': 'stock_cn',
                        'source': self.name,
                    })
        return rows

    # ---- stock_info ----------------------------------------------------

    def fetch_stock_info(self, codelist=None, verbose: bool = False) -> list:
        """股本与上市信息。字段名与目标 schema 逐字对齐。

        ``codelist`` 省略时用 :meth:`fetch_stock_list` 的全市场名单。
        **注意会逐只调用 ``get_finance_info``** —— 5000+ 只就是 5000+ 次请求。
        pytdx 走 TCP 无 HTTP 限频，实测 20 只毫秒级；但全量仍应放在带心跳的
        长任务里跑，不要在交互路径上调用。

        ``province``/``industry`` 是通达信的**整型编码**，无码表可映射，故原样存
        —— 旧 QMT writer 在此写 ``None``，存整型是改进，但展示前需另建码表。
        """
        api = self._connect()
        if codelist is None:
            codelist = [r['code'] for r in self.fetch_stock_list(verbose=verbose)]
        elif isinstance(codelist, str):
            codelist = [codelist]

        rows = []
        for code in codelist:
            code = str(code).split('.')[0][-6:]
            market = _tdx_market_of(code)
            if market is None:
                continue
            self.gate()
            try:
                fi = api.get_finance_info(market, code)
            except Exception as exc:      # noqa: BLE001 单只失败不该中断全批
                if verbose:
                    print(f'[pytdx] get_finance_info({code}) 失败: {exc!r}')
                continue
            if not fi:
                continue
            rows.append({
                'code': code,
                'name': None,                       # 名称走 stock_list；此处不重复取
                'market': 1 if market == 1 else 0,
                'liutongguben': fi.get('liutongguben'),
                'zongguben': fi.get('zongguben'),
                'ipo_date': fi.get('ipo_date'),
                'IPODate': fi.get('ipo_date'),      # AKA.IPO_DATE 的拼写
                'updated_date': fi.get('updated_date'),
                'province': fi.get('province'),
                'industry': fi.get('industry'),
                'gudongrenshu': fi.get('gudongrenshu'),
                'source': self.name,
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
