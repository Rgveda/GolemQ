# coding:utf-8
r"""tdxaidata（通达系官方数据源）适配器 —— **骨架，凭证位置不对未启用**。

⚠️ 与 pytdx 的区别，别混为一谈
==============================
* **pytdx** —— 社区维护的**爬虫接口**，走通达信公开行情协议，免费、无需凭证。
* **tdxaidata** —— 通达系**官方**数据源，**积分收费**，需要 token。

两者是不同性质的源，不能互相替代：pytdx 拿不到的（如成套财务归档），
tdxaidata 可能拿得到，但要用积分换。

⚠️ 状态：无 token 时 `available()` 返回 False
==============================================
tdxaidata 1.0.2 已安装，但其配置**不在 GolemQ 的配置体系里** —— 它有独立的 INI：

    [TcHqHost] aihs.tdx.com.cn:7709
    [TcDsHost] aids.tdx.com.cn:7727
    [Token]    token=<积分 token>

位于 `site-packages/tdxaidata/lib/TdxAiData.ini`，或在
`%USERPROFILE%/.tdxaidata/lib/TdxAiData.ini` 放一份可写副本（后者无需管理员权限）。
出处仅见于老树的 `docs/claude_change_log.md`，**两棵树都没有可运行代码**。

⚠️ 实测：token 放错了文件，改对需要管理员（2026-09-20）
=====================================================
用户**已有 token**，但它在库读不到的位置：

==========================================  ==========================
文件                                          `[Token] token`
==========================================  ==========================
`~/.tdxaidata/lib/TdxAiData.ini`（用户副本）  **非空 len=36** ← 你放的
`site-packages/tdxaidata/lib/TdxAiData.ini`  **空**       ← 库读的
==========================================  ==========================

**为什么库只读包内那份** —— `tdxaidata.py:195-197`：

    # 动态库后续数据请求依赖同目录中的 NewTc.dat，不能在启动后切回调用方目录。
    os.chdir(os.path.dirname(self._dll_path))
    result = self._dll.start(b"")

原生 DLL 启动时 CWD 已被切到 **DLL 所在目录**，它从那里读 `TdxAiData.ini`。
`~/.tdxaidata/` 那份**从未被读到** —— 老树文档说「用户副本可行」，实测不成立。

故当前调用报：`[错误码 13] 未知错误` / `[错误] Need TokenKey, Check TdxAiData.ini`。

**要启用，需把 token 写进包内那份**，而该路径不可写（实测 `PermissionError`）：

    C:\ProgramData\miniconda3\envs\GolemQ\Lib\site-packages\tdxaidata\lib\TdxAiData.ini

这需要管理员权限，且**升级/重装 tdxaidata 会被覆盖**。此事涉及凭证写入，
本模块不代做 —— 由用户决定是否写入，以及是否改用 `TDX_AI_DATA_LIB` 环境变量
把库指向一个可写的自定义目录（若该变量也影响 INI 解析，则是更干净的解法，
**待验证**）。

⚠️ 接口能力（据方法名，未实测）
===============================
`tqs` 暴露的接口**覆盖全部 5 个集合**，且比 pytdx 更全：

================  ==========================================================
`stock_list`      ``get_stock_list``
`stock_info`      ``get_stock_info`` / ``get_gb_info``（**股本**）/ ``get_ipo_info``
`financial`       ``get_financial_data`` / ``get_financial_data_by_date``
`stock_block`     ``get_sector_list`` / ``get_stock_list_in_sector``
`etf_list`        ``get_trackzs_etf_info``
================  ==========================================================

另有 `get_trading_dates`（交易日历）、`get_divid_factors`（复权因子）、
`get_pricevol`、`get_market_data`、`get_minute_data` 等。

**但未实测，故不声明能力。** 老树文档另记录：分钟线取数的 `period`/`start_time`
格式从未验证，60 分钟线曾返回错误码 3。token 就位后需先做接口踏勘，
不能照文档直接写。
"""
from __future__ import annotations

import os

from .base import DataSource, DataSourceNotAvailable, register

def _ini_read_by_library():
    """库**实际读取**的 INI 路径：DLL 所在目录下的那份。

    依据 ``tdxaidata.py:195-197`` —— 原生 DLL 启动前 CWD 被切到 DLL 目录，
    它从那里读 `TdxAiData.ini`。故 `~/.tdxaidata/lib/` 的用户副本**从未被读到**，
    判断可用性必须查包内这份，否则会把「token 放错位置」误判成「已配置」。
    """
    try:
        import tdxaidata
        pkg = os.path.dirname(os.path.abspath(tdxaidata.__file__))
    except Exception:  # noqa: BLE001
        return None
    for sub in ('lib', ''):
        cand = os.path.join(pkg, sub, 'TdxAiData.ini')
        if os.path.isfile(cand):
            return cand
    return None


def _token_in(path) -> bool:
    if not path or not os.path.isfile(path):
        return False
    try:
        import configparser
        cp = configparser.ConfigParser()
        cp.read(path, encoding='utf-8')
        for section in cp.sections():
            for key, val in cp.items(section):
                if 'token' in key.lower() and (val or '').strip():
                    return True
    except Exception:  # noqa: BLE001
        return False
    return False


@register
class TdxAiDataSource(DataSource):
    name = 'tdxaidata'
    #: **不声明任何集合** —— 接口未踏勘，声明了就是承诺。
    #: 踏勘后再填，届时自动获得注册/限频/代理/降级的整套能力。
    collections = ()
    #: 官方源，按积分限频；实际数值待踏勘
    default_interval = 30.0

    def available(self) -> bool:
        """包在 + **库实际读取的那份** INI 里有非空 token。缺任一项返回 False。

        注意查的是 `_ini_read_by_library()` 的结果，不是用户副本 —— 用户副本
        放对了 token 也不算数，因为库根本不读它（这是实测踩到的坑）。
        """
        try:
            import tdxaidata  # noqa: F401
        except ImportError:
            return False
        return _token_in(_ini_read_by_library())

    def unavailable_reason(self) -> str:
        try:
            import tdxaidata  # noqa: F401
        except ImportError:
            return 'tdxaidata 未安装'
        lib_ini = _ini_read_by_library()
        user_ini = os.path.join(os.path.expanduser('~'), '.tdxaidata', 'lib', 'TdxAiData.ini')
        if _token_in(user_ini) and not _token_in(lib_ini):
            return (f'token 放错了位置：用户副本 {user_ini} 里有 token，'
                    f'但库读的是 {lib_ini}（原生 DLL 启动时 CWD 已切到该目录），'
                    f'后者为空。需把 token 写进前者所在的包内文件 —— 该路径'
                    f'需要管理员权限，且升级 tdxaidata 会被覆盖。')
        return (f'未在 {lib_ini} 找到 token（积分收费的官方源）。'
                f'注意其接口从未被验证过，token 就位后需先做接口踏勘。')

    def fetch(self, collection: str, **kwargs) -> list:
        raise DataSourceNotAvailable(self.unavailable_reason())
