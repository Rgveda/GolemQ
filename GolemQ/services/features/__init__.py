# coding:utf-8
"""特征元数据的读写 —— ``services/features`` 包。

布局
====
按「粒度 + 职责」分册，每个文件都在 300 行以内（项目约定：`services/` 单文件
不得超 300 行）：

======================== ====================================================
``_reality_save.py``     ``GQ_save_metadata_reality``
``_daily_save.py``       ``GQ_save_daily_metadata_reality``
``_daily_fetch.py``      ``GQ_fetch_daily_metadata_reality``
``_hourly_fetch.py``     ``GQ_fetch_hourly_metadata_reality``
``_daily_crud.py``       ``GQ_fix/update/remove_daily_metadata``
``_hourly_crud.py``      ``GQ_update/remove/move_hourly_metadata``
``_valuation.py``        ``GQ_save_stock_valuation``
======================== ====================================================

本文件把上面全部 11 个函数**原样再导出**，使消费方一行都不用改 ——
``markets/StockCN/align.py``、``markets/StockCN/crawler.py``、
``services/align.py`` 都是 ``from GolemQ.services.features import (...)`。

⚠️ 为什么这个 ``__init__.py`` 是必需的（不是形式主义）
======================================================
拆分其实早就做完了 6 个分册，但**从来没建过 ``__init__.py``**，而同层还留着
956 行的旧巨石 ``features.py``。于是 Python 解析 ``GolemQ.services.features`` 时
**普通模块优先于命名空间包** —— 落到旧模块上，那 6 个分册**一个都加载不到**，
且全树零引用。也就是说：拆分做完了、但**一个字节都没生效**。

2026-09-21 补齐：新增 ``_hourly_crud.py``（原巨石里最后 3 个未拆的函数）、
建本文件、删除旧巨石。搬运前逐函数比对过，8 个已拆函数与巨石**逐行相同**
（无漂移），故这次收敛不改变任何行为。
"""
from ._reality_save import GQ_save_metadata_reality
from ._daily_save import GQ_save_daily_metadata_reality
from ._daily_fetch import GQ_fetch_daily_metadata_reality
from ._hourly_fetch import GQ_fetch_hourly_metadata_reality
from ._daily_crud import (
    GQ_fix_daily_metadata,
    GQ_update_daily_metadata,
    GQ_remove_daily_metadata,
)
from ._hourly_crud import (
    GQ_update_hourly_metadata,
    GQ_remove_hourly_metadata,
    GQ_move_hourly_metadata,
)
from ._valuation import GQ_save_stock_valuation

__all__ = [
    'GQ_save_metadata_reality',
    'GQ_save_daily_metadata_reality',
    'GQ_fetch_daily_metadata_reality',
    'GQ_fetch_hourly_metadata_reality',
    'GQ_fix_daily_metadata',
    'GQ_update_daily_metadata',
    'GQ_remove_daily_metadata',
    'GQ_update_hourly_metadata',
    'GQ_remove_hourly_metadata',
    'GQ_move_hourly_metadata',
    'GQ_save_stock_valuation',
]
