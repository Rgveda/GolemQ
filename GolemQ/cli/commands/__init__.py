# coding:utf-8
"""CLI 命令的**登记入口** —— 导入即注册。

⚠️ 下面那张列表的**顺序就是分发顺序**，与老 `if/elif` 链**逐条对应**。
**加新命令时把它插进列表的正确位置，别只往末尾 append** ——
哪条先判决定了「两个开关同时给时谁赢」，老链里那个顺序是藏在书写顺序里的
（`PITFALLS.md` P12 记过这个坑），现在它是显式的。

最要命的一条：`save_report.SAVE_COVERAGE` **必须**在 `save.SAVE` **之前**
—— `--save tdx --save-coverage` 同时给时，老链选 coverage。
"""
from ._registry import (      # noqa: F401 再导出，供 __main__ 与测试用
    COMMANDS,
    Command,
    dispatch,
    no_arguments,
    pick,
    register,
)
from . import (      # noqa: F401 导入即注册（见上）
    heartbeat,
    migrate,
    purge,
    save,
    save_report,
    setup,
    subscribe,
    tdx_hosts,
    watchlist,
)

register([
    # —— 初始化 ——
    setup.SETUP,                 # --setup / -i/--init
    setup.MONGODB_INIT,          # --mongodb-init
    setup.DINGTALK_INIT,         # --dingtalk-init
    setup.SERVERCHAN_INIT,       # --serverchan-init
    # —— 关注列表 ——
    watchlist.ENELOOP_ADD,
    watchlist.ENELOOP_REMOVE,
    watchlist.ENELOOP_LIST,
    # —— 运维 ——
    purge.PURGE,                 # --purge-l1 / --purge
    subscribe.SUBSCRIBE,         # --sub
    heartbeat.HEARTBEAT_WATCHDOG,
    heartbeat.STOP_HEARTBEAT,
    tdx_hosts.UPDATE_TDX_HOSTS,      # 服务器池刷新（只做网络探活，不碰库）
    # —— 数据 ——
    save_report.SAVE_COVERAGE,   # ⚠️ 必须在 SAVE 之前
    save.SAVE,
    save_report.SAVE_STATUS,
    migrate.MIGRATE_ENELOOP,
    migrate.MIGRATE_FINANCIAL,
])

__all__ = ['COMMANDS', 'Command', 'dispatch', 'no_arguments', 'pick',
           'register']
