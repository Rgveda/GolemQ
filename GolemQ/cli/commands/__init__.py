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
    chartscan,
    compact,
    fill_turnover,
    heartbeat,
    metadata_align,
    migrate,
    moneyflow,
    purge,
    review,
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
    # —— CV / 图形识别（⚠️ 跑在 **GolemAI conda env**，见 `bootstrap.CROSS_ENV_COMMANDS`）——
    chartscan.CHARTSCAN_SCAN,    # --scan ChartScan --eval <scope> --port <port>
    chartscan.CHARTSCAN_SERVER,  # --server ChartScan --port <port[,port...]>
    heartbeat.HEARTBEAT_WATCHDOG,
    heartbeat.STOP_HEARTBEAT,
    tdx_hosts.UPDATE_TDX_HOSTS,      # 服务器池刷新（只做网络探活，不碰库）
    # —— 数据 ——
    save_report.SAVE_COVERAGE,   # ⚠️ 必须在 SAVE 之前
    save.SAVE,
    save_report.SAVE_STATUS,
    # —— 复盘（统一因果管线；写 reality_feat_{day,60min,15min}）——
    # `--review` == `--strategy review`；`--daily` 已删除（语义不明）；
    # `--review-runtime` 是只读内存 API。scope 用全局 `--eval`。
    review.REVIEW_DAILY,
    review.REVIEW_RUNTIME,
    # compact（只读：8.3 60min+day → calc_compact_features）—— `--strategy compact`
    compact.COMPACT,
    # —— 外部抓取（HTTP，落 stock_reality_feat_day）——
    moneyflow.MONEYFLOW,             # --moneyflow {eastmoney|ths|all}
    # 现算补齐（不连 4.4、不抓外网）—— 与 MONEYFLOW 同层，都落 stock_reality_feat_day
    fill_turnover.FILL_TURNOVER,     # --fill-turnover（缺换手率就现算补上）
    # ⚠️ 放在 MONEYFLOW / FILL_TURNOVER **之后**：它是**只读报告**，
    # 跑在写路径后面才看得出刚补上的列
    metadata_align.METADATA_ALIGN,   # --metadata-align（只读覆盖率）
    # —— 一次性搬运（连 4.4；`core/migrate44.py` 的唯一消费者）——
    migrate.MIGRATE_TURNOVER,
    migrate.MIGRATE_INDEX_LIST,
    migrate.MIGRATE_METADATA_MERGE,
    migrate.MIGRATE_REALITY_FEAT,
])

__all__ = ['COMMANDS', 'Command', 'dispatch', 'no_arguments', 'pick',
           'register']
