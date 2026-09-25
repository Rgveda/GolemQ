# coding:utf-8
"""对齐与完整性检查 —— ``services/align`` 包。

布局
====
按职责分册，每个文件都在 300 行以内（项目约定：`services/` 单文件不得超 300 行）：

============================ ====================================================
``_checkpoint.py``           检查点日志的写入与查询
                             ``save_symbol_checkpoint_log``（写）
                             ``symbol_checkpoint_log``（组装后写，**唯一有外部调用者的**）
                             ``GQ_fetch_checkpoint_symbols``（按 FrozenExpired 区间查回）
``_missing.py``              缺失数据检测
                             ``calc_stock_hourly_kline_align``（⚠️ 已知坏损，见其模块说明）
                             ``calc_stock_metadata_missing_queries``
                             ``kline_missing_checkpoints``
============================ ====================================================

本文件把上面 6 个函数**原样再导出**，使消费方一行都不用改 ——
``services/persistence/_daily.py`` 与 ``_stock.py`` 都是
``from GolemQ.services.align import symbol_checkpoint_log``。

⚠️ 为什么必须有这个 ``__init__.py``（不是形式主义）
====================================================
同类事故本项目**已经发生过一次**：``services/features/`` 曾拆好 6 个分册却**从未建
``__init__.py``**，同层还留着 956 行的旧巨石 ``features.py``，于是 Python 解析时
**落到旧模块**上，6 个分册**一个都没生效过**（见该包的 docstring）。

所以拆分**必须**分两步且**同时**完成：① 建 ``__init__.py`` 并再导出；
② **删掉同名的 ``align.py``**。少了第 ② 步，普通模块会压住命名空间包。

⚠️ 注意与 ``markets/StockCN/align.py`` 区分 —— 那是另一个模块
（A 股市场层的对齐实现），与本案无关，**不要**合并。
"""

from ._checkpoint import (
    GQ_fetch_checkpoint_symbols,
    save_symbol_checkpoint_log,
    symbol_checkpoint_log,
)
from ._missing import (
    calc_stock_hourly_kline_align,
    calc_stock_metadata_missing_queries,
    kline_missing_checkpoints,
)

__all__ = [
    # _checkpoint
    'save_symbol_checkpoint_log',
    'symbol_checkpoint_log',
    'GQ_fetch_checkpoint_symbols',
    # _missing
    'calc_stock_hourly_kline_align',
    'calc_stock_metadata_missing_queries',
    'kline_missing_checkpoints',
]
