# coding:utf-8
"""MiniQMT（xtquant）数据源适配器。

从 `GolemQ_old/gateway/xtquant/data_source.py` 移植**子集** —— 只搬那 5 个参考集合
需要的助手，不搬整个 370 行模块。被舍掉的部分（`GQ_qmt_normalize_day/min`、
`GQ_qmt_now_time`）是 K 线规范化，依赖 QUANTAXIS 的 `QA_util_date_stamp` /
`QA_util_time_stamp` / `QA_util_if_trade`，与本包无关。

移植时唯一改的是 import 路径：`GolemQ.utils.constants` → `GolemQ.core.constants`。
其余逐字保留，因为其中的边界条件都是实盘踩出来的（见下）。

能供什么
========
================  ============================================================
`stock_list`      ✅ 字段最全（`volunit`/`decimal_point`/`name`/`pre_close`/`sse`）
`stock_info`      ✅ 唯一能给出 `liutongguben`/`zongguben` 的源
`stock_block`     ✅ QMT 板块（含 QMT 特有的板块名空间）
`etf_list`        ❌ 无 ETF 分类清单（走 akshare）
`financial`       ❌ 无报告期归档（走 baostock）
================  ============================================================

⚠️ 运行前提：MiniQMT 客户端必须在线
====================================
xtquant 是**本地客户端 SDK**，不是网络 API —— 没有 QMT/MiniQMT 进程在跑时，
`xtdata` 的调用会返回空或抛异常。因此 `available()` 只检查包是否可导入，
真正的在线判断放在 `fetch()` 里经 `_ensure_ready()` 抛出**明确**错误
（「请先启动 QMT/MiniQMT 客户端」），而不是让调用方拿到一片空数据去猜。

⚠️ 保留的两处实战教训（不要「简化」掉）
=======================================
1. **`prefer_index` 不能删。** 裸 6 位代码存在「上交所指数 ↔ 深市股票**同码**」的
   固有歧义（实测指数池与股票池交集 **216 个** `000xxx`）。`is_stock_cn` 对
   `000xxx` 一律判为深市主板股票，于是按它路由会把 `000905.SZ`（厦门港务）的
   数据写进指数路径，覆盖 `000905.SH`（中证500）的历史。故指数路径按号段显式
   定交易所，不走分类器。
2. **`SECTOR_SKIP` 不能删。** QMT 会把指数与基金混进容器板块（`京市A股` 含北证
   指数 `899050`/`899601`/`810011`，`沪深京A股` 另含 `910000`-`910005` 上证证券
   投资基金）。不排除会让「按板块取成分」的下游把指数当股票。
"""
from __future__ import annotations

import math

from .base import (
    STOCK_BLOCK,
    STOCK_INFO,
    STOCK_LIST,
    DataSource,
    DataSourceNotAvailable,
    UnsupportedCollection,
    register,
)

#: 股票范围板块。QMT 没有独立的「京市指数」板块，北证指数被列在 `京市A股` 里。
QMT_STOCK_SECTORS = ['沪深A股', '京市A股']
#: 指数范围板块
QMT_INDEX_SECTORS = ['沪深指数']
#: 北证指数寄居地（`899050` 北证50、`899601`、`810011`）
QMT_BJ_INDEX_SECTORS = ['京市A股']
#: ETF 范围板块
QMT_ETF_SECTORS = ['沪深ETF', '沪市ETF', '深市ETF']

#: 市场级容器板块 —— 一律不落库。理由见模块文档第 2 条。
SECTOR_SKIP = {
    '沪深A股', '沪深B股', '沪深基金', '沪深债券', '沪深转债', '沪深指数', '沪深ETF',
    '上证A股', '上证B股', '上证基金', '上证债券', '上证转债', '上证指数', '上证ETF',
    '深证A股', '深证B股', '深证基金', '深证债券', '深证转债', '深证指数', '深证ETF',
    '京市A股', '京市B股', '京市指数', '京市ETF', '北交所', '沪深京A股',
    '沪市A股', '沪市B股', '沪市基金', '沪市债券', '沪市转债', '沪市ETF', '沪市指数',
    '深市A股', '深市B股', '深市基金', '深市债券', '深市转债', '深市ETF', '深市指数',
    '创业板', '科创板', '我的自选', '中金所', '上期所', '大商所', '郑商所', '广期所',
}


def _sector_type(name: str) -> str:
    """板块类型。读取侧不消费，仅作元数据；拿不到官方 category 时按名称启发。"""
    if '指数' in name:
        return 'zs'
    if 'ETF' in name or '基金' in name:
        return 'etf'
    if '概念' in name or '板块' in name:
        return 'gn'
    return 'yb'


# --------------------------------------------------------------------------- #
# 代码与基础助手（逐字移植，只改 import 路径）
# --------------------------------------------------------------------------- #

def GQ_qmt_resolve_xt_code(code, prefer_index: bool = False) -> str:
    """``'600000'`` → ``'600000.SH'``；``'sh.600000'`` / 已带后缀则归一。"""
    if isinstance(code, (list, tuple)):
        return [GQ_qmt_resolve_xt_code(c, prefer_index=prefer_index) for c in code]
    s = str(code).strip()
    if '.' in s:
        left, right = s.rsplit('.', 1)
        if right.upper() in ('SH', 'SZ', 'BJ'):
            return '%s.%s' % (left[-6:], right.upper())
        if left.lower() in ('sh', 'sz', 'bj'):
            return '%s.%s' % (right[-6:], left.upper())
        return s
    if prefer_index:
        # 指数号段：399xxx=深证、899xxx/81xxxx/89xxxx=北证、其余(000xxx/98xxxx)=上交所。
        # 保留深市 ETF/基金号段以防显式传入时被误判成沪市；指数池实测不含这些号段。
        if s[:2] in ('39', '15', '16', '18', '20'):
            return '%s.SZ' % s[-6:]
        if s[:3] == '899' or s[:2] in ('81', '89'):
            return '%s.BJ' % s[-6:]
        return '%s.SH' % s[-6:]
    try:
        from GolemQ.markets.StockCN.symbol import is_stock_cn
        alias = is_stock_cn(s)[2]
    except Exception:  # noqa: BLE001
        alias = None
    if not alias:
        alias = 'SH' if s[0] in ('6', '5', '9') else 'SZ'
    return '%s.%s' % (s[-6:], str(alias).upper())


def GQ_qmt_to_qa_code(xt_code: str) -> str:
    """``'600000.SH'`` → ``'600000'``（GolemQ 的 6 位 code）。"""
    return str(xt_code).split('.')[0][-6:]


def GQ_qmt_sse(code: str) -> str:
    """小写交易所后缀 ``'sh'``/``'sz'``/``'bj'``。"""
    s = str(code)
    if '.' in s:
        suffix = s.rsplit('.', 1)[1]
        if suffix.upper() in ('SH', 'SZ', 'BJ'):
            return suffix.lower()
    try:
        from GolemQ.markets.StockCN.symbol import is_stock_cn
        alias = is_stock_cn(s[-6:])[2]
        if alias:
            return str(alias).lower()
    except Exception:  # noqa: BLE001
        pass
    return 'sh' if s[-6:][0] in ('6', '5', '9') else 'sz'


def GQ_qmt_decimal_point(detail: dict) -> int:
    """由 ``PriceTick`` 推小数位（A 股 = 2）。"""
    tick = detail.get('PriceTick')
    try:
        tick = float(tick)
    except (TypeError, ValueError):
        return 2
    if tick <= 0:
        return 2
    return max(0, min(6, -int(round(math.log10(tick)))))


def GQ_qmt_sector_members(sector_names) -> list:
    """板块成分并集 → 6 位 code 列表（需已 ``download_sector_data``）。"""
    from xtquant import xtdata

    out: list = []
    for name in sector_names:
        try:
            out.extend(xtdata.get_stock_list_in_sector(name) or [])
        except Exception as e:  # noqa: BLE001
            print(f'[qmt] get_stock_list_in_sector({name}) 失败: {e!r}')
    return sorted({GQ_qmt_to_qa_code(x) for x in out})


def GQ_qmt_stock_codes() -> list:
    """``沪深A股``+``京市A股`` 里**确实是 A 股**的 code。

    这两个板块混入了非股票（实测北证指数 ``899050``/``899601``/``810011`` 被列在
    ``京市A股``；另有 ``910000``-``910005`` 上证证券投资基金）。不按市场类型过滤，
    会把指数与基金当股票存进 ``stock_list``/``stock_day``。实测 5596 → 5587。
    """
    from GolemQ.core.constants import MARKET_TYPE

    out: list = []
    for code in GQ_qmt_sector_members(QMT_STOCK_SECTORS):
        try:
            if _is_stock_cn(code)[1] == MARKET_TYPE.STOCK_CN:
                out.append(code)
        except Exception as e:  # noqa: BLE001  分类器异常不应中断整个保存任务
            print(f'[qmt] is_stock_cn({code}) 失败，已从股票范围剔除: {e!r}')
    return out


def _is_stock_cn(code):
    from GolemQ.markets.StockCN.symbol import is_stock_cn
    return is_stock_cn(code)


def _ensure_ready(verbose: bool = True) -> bool:
    """MiniQMT 在线且板块数据已下载；不可用抛 RuntimeError。"""
    from xtquant import xtdata

    try:
        detail = xtdata.get_instrument_detail('000001.SZ')
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f'MiniQMT/xtquant 不可用: {e!r}') from e
    if not detail:
        raise RuntimeError('MiniQMT/xtquant 行情服务未在线（请先启动 QMT/MiniQMT 客户端）')
    try:
        xtdata.download_sector_data()
    except Exception as e:  # noqa: BLE001
        if verbose:
            print(f'[qmt] download_sector_data 警告: {e!r}')
    if verbose:
        print('[qmt] MiniQMT 已连接: %s' % detail.get('InstrumentName'))
    return True


@register
class QmtSource(DataSource):
    name = 'qmt'
    collections = (STOCK_LIST, STOCK_INFO, STOCK_BLOCK)
    #: 本地客户端 SDK，不经网络限频 —— 覆盖 30s 默认值，理由同 pytdx。
    default_interval = 0.0

    def available(self) -> bool:
        """只判断包是否可导入。**不做在线探测** —— 那要连客户端，成本高且
        属于 fetch 的职责；排障路径不该成为故障点。"""
        try:
            import xtquant  # noqa: F401
        except ImportError:
            return False
        return True

    def fetch(self, collection: str, **kwargs) -> list:
        if not self.supports(collection):
            raise UnsupportedCollection(
                f'{self.name} 不提供 {collection}（可提供 {self.collections}）'
            )
        if not self.available():
            raise DataSourceNotAvailable('xtquant 未安装')
        self.gate()
        if collection == STOCK_LIST:
            return self.fetch_stock_list(**kwargs)
        if collection == STOCK_INFO:
            return self.fetch_stock_info(**kwargs)
        return self.fetch_stock_block(**kwargs)

    # ---- stock_list ----------------------------------------------------

    def fetch_stock_list(self, verbose: bool = False) -> list:
        """全市场 A 股列表。字段与 4.4 `quantaxis.stock_list` 逐一对齐。"""
        _ensure_ready(verbose=verbose)
        from xtquant import xtdata

        codes = GQ_qmt_stock_codes()
        if not codes:
            raise DataSourceNotAvailable(
                '[qmt:stock_list] 名单为空：请确认 MiniQMT 在线且已 download_sector_data()'
            )
        rows = []
        for code in codes:
            detail = xtdata.get_instrument_detail(
                GQ_qmt_resolve_xt_code(code)) or {}
            name = detail.get('InstrumentName') or None
            try:
                pre_close = float(detail.get('PreClose'))
            except (TypeError, ValueError):
                pre_close = 0.0
            if pre_close != pre_close or pre_close >= 1e30:   # NaN / 客户端无数据哨兵
                pre_close = 0.0
            if (not name) and pre_close <= 0:
                # 完全没有信息的占位代码跳过；但 920808.BJ 这类真实北交所股票
                # name 可能为空串而 PreClose 有效 → 保留
                continue
            rows.append({
                'code': GQ_qmt_to_qa_code(code),
                'volunit': 100,                       # A 股每手 100 股
                'decimal_point': GQ_qmt_decimal_point(detail),
                'name': name,
                'pre_close': pre_close,
                'sse': GQ_qmt_sse(GQ_qmt_resolve_xt_code(code)),
                'sec': 'stock_cn',
                'source': self.name,
            })
        return rows

    # ---- stock_info ----------------------------------------------------

    def fetch_stock_info(self, verbose: bool = False) -> list:
        """股本与上市信息。``liutongguben``/``zongguben`` 是唯一被真实消费的字段，
        而**只有 QMT 能给**（pytdx 与 akshare 都拿不到）。"""
        _ensure_ready(verbose=verbose)
        from xtquant import xtdata

        codes = GQ_qmt_stock_codes()
        if not codes:
            raise DataSourceNotAvailable('[qmt:stock_info] 名单为空')
        rows = []
        for code in codes:
            xt_code = GQ_qmt_resolve_xt_code(code)
            detail = xtdata.get_instrument_detail(xt_code) or {}
            if not detail:
                continue
            ipo = detail.get('OpenDate')
            rows.append({
                'code': GQ_qmt_to_qa_code(code),
                'name': detail.get('InstrumentName'),
                'market': 1 if GQ_qmt_sse(xt_code) == 'sh' else 0,
                'liutongguben': detail.get('FloatVolume'),
                'zongguben': detail.get('TotalVolume'),
                'ipo_date': ipo,
                'IPODate': ipo,          # AKA.IPO_DATE 的拼写，消费方按此读
                'updated_date': None,
                'province': None,
                'industry': None,
                'gudongrenshu': None,
                'source': self.name,
            })
        return rows

    # ---- stock_block ---------------------------------------------------

    def fetch_stock_block(self, include_market_sectors: bool = False,
                          verbose: bool = False) -> list:
        """QMT 板块成分。容器板块按 `SECTOR_SKIP` 排除。

        **不做逐 code 的股票类型过滤** —— 板块里存在跨市场成分（沪港通/AH 股含港股），
        按 `is_stock_cn` 过滤会连港股一起削掉。容器板块的排除已足够精确：
        实测那 9 个代码每个都只在 `沪深京A股` 里出现过一行。
        """
        _ensure_ready(verbose=verbose)
        from xtquant import xtdata

        sectors = xtdata.get_sector_list() or []
        if not sectors:
            raise DataSourceNotAvailable(
                '[qmt:stock_block] 板块列表为空：请确认 MiniQMT 在线且已 download_sector_data()'
            )
        rows = []
        for name in sectors:
            if (not include_market_sectors) and name in SECTOR_SKIP:
                continue
            try:
                members = xtdata.get_stock_list_in_sector(name) or []
            except Exception as e:  # noqa: BLE001
                if verbose:
                    print(f'[qmt:stock_block] {name} 取成分失败: {e!r}')
                continue
            btype = _sector_type(name)
            for xt_code in members:
                rows.append({
                    'blockname': name,
                    'code': GQ_qmt_to_qa_code(xt_code),
                    'type': btype,
                    'source': self.name,
                })
        return rows
