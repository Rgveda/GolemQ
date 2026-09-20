# coding:utf-8
"""日志接口 —— 替代 QUANTAXIS 的 `QA_util_log_info`。

为什么需要这个模块
==================
`services/features/` 包里曾（连同当时的 `services/features.py`）有 7 处
``from QUANTAXIS.QAUtil import QA_util_log_info, ...``，而其中
**只有 `QA_util_log_info` 在新树没有等价物** —— 其余四个（`QA_util_code_tolist`、
`QA_util_date_valid`、`QA_util_date_stamp`、`QA_util_time_stamp`）在
`core/symbol.py` 与 `markets/StockCN/date_utils.py` 里都有，
且**已实测行为逐位一致**。

那 7 处后来都改成了本模块与上述等价物（2026-09-21 前完成），现在全树的
QUANTAXIS 依赖只剩 `core/settings.py` 一处（D9 定的暂留）。`services/features.py`
那个巨石模块也已退休 —— 拆分后的 `services/features/` 包现在真正生效，
见该包 `__init__.py` 的说明。

⚠️ 为什么用 `logging.warning` 而不是 `logging.info`
==================================================
**对齐原实现。** QUANTAXIS 的 `QA_util_log_info` 函数体里写的就是
``logging.warning(logs)`` —— 名字叫 log_info，落到的却是 warning 级。

改成 `info` 会让既有的调用方**日志级别静默变化**：那些消息在默认配置下
本会打到 stderr，改成 info 后可能被默认的 WARNING 阈值拦掉，
**变成看不到的错误信息**。所以这里照搬，不"顺手改正"。
"""
from __future__ import annotations

import logging

__all__ = ['GQ_util_log_info']


def GQ_util_log_info(logs, ui_log=None, ui_progress=None,
                     ui_progress_int_value=None):
    """INFO 级日志接口 —— 行为对齐 `QUANTAXIS.QAUtil.QA_util_log_info`。

    :param logs: 日志内容，字符串或字符串列表
    :param ui_log: 可选。GUI 的日志槽（有 ``emit`` 方法）；给了就同时 emit
    :param ui_progress: 可选。GUI 的进度槽
    :param ui_progress_int_value: 可选。进度值

    >>> import logging
    >>> with_log = logging.getLogger()          # doctest 里只想确认不抛异常
    >>> GQ_util_log_info('一条日志')
    >>> GQ_util_log_info(['a', 'b'])
    >>> class _Sink:
    ...     def __init__(self): self.got = []
    ...     def emit(self, x): self.got.append(x)
    >>> s = _Sink()
    >>> GQ_util_log_info('hello', ui_log=s)
    >>> s.got
    ['hello']
    >>> s2 = _Sink()
    >>> GQ_util_log_info(['a', 'b'], ui_log=s2)
    >>> s2.got
    ['a', 'b']
    """
    # 与 QUANTAXIS 一致：核心落到 warning 级，不是 info
    logging.warning(logs)

    # 给 GUI 使用，更新当前任务到日志与进度
    if ui_log is not None:
        if isinstance(logs, str):
            ui_log.emit(logs)
        elif isinstance(logs, list):
            for item in logs:
                ui_log.emit(item)

    if ui_progress is not None and ui_progress_int_value is not None:
        ui_progress.emit(ui_progress_int_value)
