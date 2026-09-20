# coding:utf-8
r"""tdxaidata（通达系官方数据源）适配器 —— 已打通，可提供全部 5 个集合。

配置与"桥"
==========
token 存在 **GolemQ 自己的配置**里（`~/.GolemQ/settings/config.ini`）：

    [TDXAIDATA]
    token = TDX-xxxxxxxx

但库**不读** GolemQ 的配置 —— 它由原生 DLL 从自己所在目录读 `TdxAiData.ini`
（`tdxaidata.py:195-197` 在启动 DLL 前 `os.chdir(os.path.dirname(self._dll_path))`）。
而那个目录在 `site-packages` 下，**当前用户无写权限**（实测 `PermissionError`，
需管理员，且升级 tdxaidata 会被覆盖）。

**解法（不需要管理员）**：把库目录复制到可写位置，把 token 写进副本的 INI，
再用 `TDX_AI_DATA_LIB` 把库指向副本里的 DLL —— CWD 随之落在副本目录，
DLL 就从副本读 INI。整条路由 `_ensure_lib()` 自动完成，实测可用。

实测接口（2026-09-20）
======================
==========================  ==============================================
集合                          接口与实测结果
==========================  ==============================================
`stock_list`                ``get_stock_list(market=...)``。market 是**中文市场名**，
                            不是编码 —— 默认 ``"上证主板"``。实测：
                            ``沪深A股`` 5226 + ``北交所`` 348 = **5574**
`stock_info`                ``get_gb_info`` → ``Ltgb``/``Zgb``（流通/总股本）；
                            ``get_stock_info`` → ``ActiveCapital`` 等字段
`financial`                 ``get_financial_data``（**须传 field_list，空则报错**）
`stock_block`               ``get_sector_list`` 560 个板块 + ``get_stock_list_in_sector``
`etf_list`                  ``get_trackzs_etf_info``
==========================  ==============================================

另有 `get_trading_dates`（交易日历）、`get_divid_factors`（复权因子）。

⚠️ 与 pytdx 的区别
==================
* **pytdx** —— 社区爬虫接口，免费无凭证，但北交所行情列表取不到。
* **tdxaidata** —— 通达系**官方**源，**积分收费**，但覆盖完整含北交所。

两者互补而非替代。tdxaidata 是唯一**单源覆盖全部 5 个集合**的源。
"""
from __future__ import annotations

import configparser
import os
import shutil

from GolemQ.datasource.base import (
    ALL_COLLECTIONS,
    DataSource,
    DataSourceNotAvailable,
    UnsupportedCollection,
    register,
)

#: 库自带的配置文件名（原生 DLL 从 DLL 所在目录读它）
LIB_INI_NAME = 'TdxAiData.ini'
#: 官方 DLL 名
WIN_DLL = 'TdxAiData.dll'

#: 可写的库副本位置 —— 放 GolemQ 自己的目录下，避免动 site-packages
LIB_MIRROR_DIR = os.path.join(os.path.expanduser('~'), '.GolemQ', 'tdxaidata_lib')

#: 市场名（**中文**，不是编码）。实测组合见模块文档。
STOCK_MARKETS = ('沪深A股', '北交所')


def _pkg_lib_dir():
    try:
        import tdxaidata
        return os.path.join(os.path.dirname(os.path.abspath(tdxaidata.__file__)), 'lib')
    except Exception:  # noqa: BLE001
        return None


def _token_from_settings():
    try:
        from GolemQ.core.settings import GQSETTING
        return (GQSETTING.get_config('TDXAIDATA', 'token', '') or '').strip()
    except Exception:  # noqa: BLE001
        return ''


def _write_token(ini_path, token) -> bool:
    cp = configparser.ConfigParser()
    try:
        cp.read(ini_path, encoding='utf-8')
        if not cp.has_section('Token'):
            cp.add_section('Token')
        cp.set('Token', 'token', token)
        with open(ini_path, 'w', encoding='utf-8') as fh:
            cp.write(fh)
        return True
    except Exception:  # noqa: BLE001
        return False


def _ensure_lib(verbose: bool = False):
    """把库目录镜像到可写位置并注入 token，返回 DLL 路径。

    **为什么必须这么做**：库从 DLL 所在目录读 INI，而那是 site-packages，
    当前用户不可写。镜像 + `TDX_AI_DATA_LIB` 是唯一不需要管理员的路子。

    已镜像过则只核对 token 是否需要更新 —— 每次调用都重写 INI 没必要，
    且会在只读介质上失败。
    """
    token = _token_from_settings()
    if not token:
        raise DataSourceNotAvailable(
            '未配置 tdxaidata token。请写入 ~/.GolemQ/settings/config.ini 的 '
            '[TDXAIDATA] 段 token = TDX-...')

    src = _pkg_lib_dir()
    if not src or not os.path.isdir(src):
        raise DataSourceNotAvailable('找不到 tdxaidata 的库目录')

    dll = os.path.join(LIB_MIRROR_DIR, WIN_DLL)
    if not os.path.isfile(dll):
        os.makedirs(os.path.dirname(LIB_MIRROR_DIR), exist_ok=True)
        shutil.copytree(src, LIB_MIRROR_DIR, dirs_exist_ok=True)
        if verbose:
            print(f'[tdxaidata] 已镜像库目录 -> {LIB_MIRROR_DIR}')

    ini = os.path.join(LIB_MIRROR_DIR, LIB_INI_NAME)
    # 只在 token 不一致时写，避免每次调用都改文件
    cp = configparser.ConfigParser()
    if os.path.isfile(ini):
        try:
            cp.read(ini, encoding='utf-8')
        except Exception:  # noqa: BLE001
            pass
    if not cp.has_section('Token') or cp.get('Token', 'token', fallback='') != token:
        if not _write_token(ini, token):
            raise DataSourceNotAvailable(f'无法写入镜像 INI: {ini}')
        if verbose:
            print('[tdxaidata] token 已注入镜像 INI')

    os.environ['TDX_AI_DATA_LIB'] = dll
    return dll


def _tqs(verbose: bool = False):
    """确保库就绪并返回 `tqs`。"""
    _ensure_lib(verbose=verbose)
    from tdxaidata import tqs
    return tqs


def _split(code: str):
    """``'600000.SH'`` → ``('600000', 'SH')``。"""
    s = str(code)
    if '.' in s:
        a, b = s.rsplit('.', 1)
        return a[-6:], b.upper()
    return s[-6:], ''


def _sse_of(code: str) -> str:
    _, suf = _split(code)
    if suf:
        return suf.lower()
    if code.startswith(('60', '68')):
        return 'sh'
    if code.startswith(('00', '30')):
        return 'sz'
    return 'bj'


@register
class TdxAiDataSource(DataSource):
    name = 'tdxaidata'
    collections = ALL_COLLECTIONS
    #: 官方源，按积分限频 —— 实际速率待观察，先沿用 30s
    default_interval = 30.0

    def available(self) -> bool:
        """包在 + 配了 token。**不在此处做镜像或联网** —— 那是 fetch 的职责。"""
        try:
            import tdxaidata  # noqa: F401
        except ImportError:
            return False
        return bool(_token_from_settings())

    def unavailable_reason(self) -> str:
        try:
            import tdxaidata  # noqa: F401
        except ImportError:
            return 'tdxaidata 未安装'
        if not _token_from_settings():
            return ('未配置 token。请写入 ~/.GolemQ/settings/config.ini 的 '
                    '[TDXAIDATA] 段：token = TDX-...')
        return '未知原因'

    def fetch(self, collection: str, **kwargs) -> list:
        if not self.supports(collection):
            raise UnsupportedCollection(
                f'{self.name} 不提供 {collection}（可提供 {self.collections}）')
        if not self.available():
            raise DataSourceNotAvailable(self.unavailable_reason())
        self.gate()
        handler = getattr(self, f'fetch_{collection}')
        return handler(**kwargs)

    # ---- stock_list ----------------------------------------------------

    def fetch_stock_list(self, verbose: bool = False) -> list:
        """全市场 A 股。`沪深A股`(5226) + `北交所`(348)，实测合计 5574。

        返回的是**带后缀**的 `'600000.SH'` 形式；落库用 6 位 code + `sse`。
        名称与昨收不在此接口 —— `get_stock_list` 只给代码。故 `name`/`pre_close`
        留空，由 `stock_info` 或其他源补（这是接口本身的边界，不是遗漏）。
        """
        tqs = _tqs(verbose=verbose)
        rows, seen = [], set()
        for market in STOCK_MARKETS:
            try:
                codes = tqs.get_stock_list(market=market) or []
            except Exception as exc:      # noqa: BLE001
                if verbose:
                    print(f'[tdxaidata:stock_list] market={market} 失败: {exc!r}')
                continue
            for xt in codes:
                code, suf = _split(xt)
                if code in seen:
                    continue
                seen.add(code)
                rows.append({
                    'code': code,
                    'volunit': 100,
                    'decimal_point': 2,
                    'name': None,           # 本接口不给名称
                    'pre_close': None,      # 本接口不给昨收
                    'sse': (suf.lower() if suf else _sse_of(code)),
                    'sec': 'stock_cn',
                    'source': self.name,
                })
        if verbose:
            print(f'[tdxaidata:stock_list] 取到 {len(rows)} 只')
        return rows

    # ---- stock_info ----------------------------------------------------

    def fetch_stock_info(self, codelist=None, verbose: bool = False) -> list:
        """股本与基础信息。`get_gb_info` 给 `Ltgb`(流通)/`Zgb`(总股本)。

        `codelist` 省略则取全市场 —— 逐只调用，**很慢**，见模块文档。
        """
        tqs = _tqs(verbose=verbose)
        if codelist is None:
            codelist = [r['code'] for r in self.fetch_stock_list(verbose=verbose)]
        elif isinstance(codelist, str):
            codelist = [codelist]

        rows = []
        for code in codelist:
            code = str(code).split('.')[0][-6:]
            self.gate()
            xt = self._xt_code(code)
            gb = {}
            try:
                got = tqs.get_gb_info(stock_code=xt) or []
                gb = got[0] if got else {}
            except Exception as exc:      # noqa: BLE001
                if verbose:
                    print(f'[tdxaidata:stock_info] gb_info({code}) 失败: {exc!r}')
            rows.append({
                'code': code,
                'name': None,
                'market': 1 if _sse_of(code) == 'sh' else 0,
                'liutongguben': gb.get('Ltgb'),
                'zongguben': gb.get('Zgb'),
                'ipo_date': None,
                'IPODate': None,
                'updated_date': None,
                'province': None,
                'industry': None,
                'gudongrenshu': None,
                'source': self.name,
            })
        return rows

    # ---- stock_block ---------------------------------------------------

    def fetch_stock_block(self, verbose: bool = False) -> list:
        """板块成分。`get_sector_list` 给 560 个板块，逐个取成分。

        ⚠️ 560 次调用 × 30s 间隔 = 约 4.7 小时。批量回填应放宽 interval 或
        作为长任务跑。
        """
        tqs = _tqs(verbose=verbose)
        sectors = tqs.get_sector_list() or []
        rows = []
        for sec in sectors:
            self.gate()
            try:
                members = tqs.get_stock_list_in_sector(block_code=sec) or []
            except Exception as exc:      # noqa: BLE001
                if verbose:
                    print(f'[tdxaidata:stock_block] {sec} 失败: {exc!r}')
                continue
            for xt in members:
                code, _ = _split(xt)
                rows.append({
                    'blockname': str(sec),
                    'code': code,
                    'type': 'tdx',
                    'source': self.name,
                })
        if verbose:
            print(f'[tdxaidata:stock_block] {len(rows)} 条 / {len(sectors)} 板块')
        return rows

    # ---- financial -----------------------------------------------------

    #: `get_financial_data` 要求非空 field_list —— 空列表直接报错（实测）。
    #: 这里用一批通用字段；字段名是英文，比 akshare 的中文键更规整。
    FINANCIAL_FIELDS = (
        'ReportDate', 'TotalAssets', 'TotalLiability', 'TotalEquity',
        'Revenue', 'NetProfit', 'BasicEPS', 'NetAssetPS',
        'ROE', 'GrossMargin', 'NetMargin', 'DebtToAssetRatio',
    )

    def fetch_financial(self, codelist=None, start_time: str = '',
                        end_time: str = '', verbose: bool = False) -> list:
        """季频财务。**必须传 field_list**，否则库报 `[错误] field_list 不能为空`。"""
        tqs = _tqs(verbose=verbose)
        if codelist is None:
            codelist = [r['code'] for r in self.fetch_stock_list(verbose=verbose)]
        elif isinstance(codelist, str):
            codelist = [codelist]

        xt_codes = [self._xt_code(str(c).split('.')[0][-6:]) for c in codelist]
        self.gate()
        try:
            payload = tqs.get_financial_data(
                stock_list=xt_codes,
                field_list=list(self.FINANCIAL_FIELDS),
                start_time=start_time, end_time=end_time)
        except Exception as exc:          # noqa: BLE001
            raise DataSourceNotAvailable(f'tdxaidata get_financial_data 失败: {exc!r}') from exc

        rows = []
        for xt, blob in (payload or {}).items():
            code, _ = _split(xt)
            records = blob if isinstance(blob, list) else [blob]
            for rec in records:
                if not isinstance(rec, dict):
                    continue
                report_date = rec.get('ReportDate')
                if not report_date:
                    continue
                row = {'code': code, 'report_date': str(report_date),
                       'source': self.name}
                row.update(rec)
                rows.append(row)
        if verbose:
            print(f'[tdxaidata:financial] {len(rows)} 行 / {len(codelist)} 只')
        return rows

    # ---- etf_list ------------------------------------------------------

    def fetch_etf_list(self, verbose: bool = False) -> list:
        """跟踪指数的 ETF 列表（`get_trackzs_etf_info`）。

        注意这是**跟踪指数**口径，不是全量 ETF 清单 —— 与 akshare 的
        `fund_etf_category_sina` 覆盖不同。作为补充源而非替代。
        """
        tqs = _tqs(verbose=verbose)
        self.gate()
        try:
            payload = tqs.get_trackzs_etf_info() or {}
        except Exception as exc:          # noqa: BLE001
            raise DataSourceNotAvailable(f'tdxaidata get_trackzs_etf_info 失败: {exc!r}') from exc
        rows = []
        items = payload.items() if isinstance(payload, dict) else []
        for zs, etfs in items:
            for xt in (etfs if isinstance(etfs, (list, tuple)) else [etfs]):
                code, _ = _split(xt)
                rows.append({
                    'code': code, 'name': None, 'track_index': str(zs),
                    'sec': 'etf_cn', 'sse': _sse_of(code),
                    'volunit': 100, 'decimal_point': 3,
                    'source': self.name,
                })
        if verbose:
            print(f'[tdxaidata:etf_list] {len(rows)} 条')
        return rows

    # ---- 助手 ----------------------------------------------------------

    @staticmethod
    def _xt_code(code: str) -> str:
        """6 位 code → 带后缀的 `'600519.SH'`。"""
        code = str(code).split('.')[0][-6:]
        return f'{code}.{_sse_of(code).upper()}'
