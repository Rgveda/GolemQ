# coding:utf-8
"""根层的基础工具（**与市场无关**的那些）。

⚠️ `GQ_util_get_last_day` 曾经住在这里，**2026-09-25 已移除**。
============================================================
它是重构时从老树 `utils/base.py:156` 搬过来的，但**被阉割成只剩周末判断**：

    老树：`(ts - QA.trade_date_sse) > timedelta(hours=9.5)` → 真·交易日历 + 09:30 切点
    新树：`today.weekday() >= 5` → 只判周末，法定节假日算错，且**没有 09:30 切点**

真实现一直在同树：`markets/StockCN/date_utils.py:52`（用 `TRADE_DATE_SSE`），
老调用方 `scribe/persistence.py:80` 当年用的就是**等价**的那一版。
**5 处调用方已被接过去**（`services/align.py`×2、`services/iwencai.py`、
`persistence/_daily.py`、`persistence/_stock.py`）。

**为什么是删掉而不是留个转发**：这个函数需要 A 股交易日历（`TRADE_DATE_SSE`），
属于**市场知识**，放根层会让 `core/` 反向依赖 `markets/StockCN/` —— 违背两层模型。
而留个「转发到市场层的同名函数」等于把依赖藏起来，正是本树反复踩的坑。

**将来若别的市场也要它**：让那个市场包自己实现一份（各市场的交易日历本就不同），
**不要**提到根层来「统一」。
"""


def set_cpu_affinity_even():
    """Set CPU affinity to even-numbered cores (stub).

    ⚠️ **这是同一类「阉割移植」**（与上面 `GQ_util_get_last_day` 同源）：
    老树 `GolemQ_old/utils/base.py:190` 有真实实现（用 `psutil.cpu_count`
    判物理/逻辑核与超线程，再按平台设亲和性），新树只剩 `pass` —— 即
    **CPU 亲和性从未被设置过**，静默无效果。

    **本次未修**：移植它会**真的改变运行时行为**（多进程管线的 CPU 绑定），
    属行为变更，需所有者点头。唯一调用方是 `pipeline/base.py:50`。

    要移植的话，**物理核/超线程怎么判**已经有现成的、实测过的实现：
    :func:`GolemQ.cli.bootstrap.check_cpu` / `_cpu_topology`（2026-10-09）。
    两个坑写在它的 docstring 里：`GetSystemCpuSetInformation` 的**条目步长要读
    `Size` 字段**，超线程**别读 `AllFlags` 的 `Smt` 位**（本机实测有假阳性）。
    """
    pass
