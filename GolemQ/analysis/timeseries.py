# coding:utf-8
"""时间序列工具 —— 多频重采样与时间轴对齐。

**本模块的定位（两层模型）**：重采样是「所有交易系统共性」的能力，属**第一层**，
放根目录下的 `analysis/`，而不是 `markets/<Market>/`。

本文件只保留两个重采样器（`GQ_data_min_resample` / `GQ_data_min_to_day`）。

⚠️ **2026-10-10 更正**：原先这里还有两个 stub（`Timeline_duration` /
`align_kline_timeline`），因「唯一调用方在 `services/persistence/`」被当死代码删掉。
**其中 `Timeline_duration` 已经回来了** —— 筹码分布模块要用，真消费方到了。
但它**不在本文件**，而在同级的 `analysis/timing.py`（连同 `Timeline_Integral` /
`calc_event_timing_lag` / `calc_energy` 那一族，都是旧树 `analysis/timeseries.py`
搬回来的**真实现**，非 stub）。分开是因为那边做「逐 bar 累积」、这边做「重采样」，
两个关注点；并进来会立刻超过 300 行。
`align_kline_timeline` **仍然不存在**（它的调用方随 `services/persistence/` 一起没了，
也没有新消费方）—— 要用就得先有调用方。
本文件新增的两个重采样器是从 QUANTAXIS **逐行回迁**的（原先
`markets/StockCN/realtime.py` 直接调 QUANTAXIS 的 `QA_data_min_resample` /
`QA_data_min_to_day`）。回迁而非重写是为了保行为：它们内含 A 股**分段交易时段**
的处理（上午 9:30-11:30、下午 13:00-15:00 分别 resample 再拼接），自成一套口径。

回迁后实测与 QUANTAXIS **输出逐值一致**（2026-09-21，600519 的 960 根 1min：
5min→192 / 15min→64 / 30min→32 / 60min→16 / 1D→8 行，形状与数值最大差 0）。
"""
import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset

# ⚠️ **2026-10-12 补**：这两个常量原先**被引用却从未定义** —— `_min_conversion()` 一调就
# `NameError`，于是 `GQ_data_min_resample` / `GQ_data_min_to_day` **两个重采样器一直是死的**
# （回迁时只抄了 `CONVERSION` 的**用法**，漏抄了它的**定义**）。
# 真值逐字取自 QUANTAXIS `QAData/data_resample.py::QA_data_min_resample` 里那个
# 三元表达式（`'vol' in min_data.columns` 决定用哪一支）—— 键序也是原样，因为
# `list(CONVERSION.keys())` 同时是**输出的列集合**。
_MIN_CONVERSION_VOL = {
    'code': 'first', 'open': 'first', 'high': 'max', 'low': 'min',
    'close': 'last', 'vol': 'sum', 'amount': 'sum',
}
_MIN_CONVERSION_VOLUME = {
    'code': 'first', 'open': 'first', 'high': 'max', 'low': 'min',
    'close': 'last', 'volume': 'sum', 'amount': 'sum',
}


def _min_conversion(min_data):
    """按帧里是 `vol` 还是 `volume` 选口径。

    注意 `CONVERSION` 的键同时是**输出的列集合** —— QUANTAXIS 用
    `min_data.loc[:, list(CONVERSION.keys())]` 裁剪，缺列会抛 `KeyError`。
    这是刻意的：宁可报错，也不要静默少一列。
    """
    return _MIN_CONVERSION_VOL if 'vol' in min_data.columns else _MIN_CONVERSION_VOLUME


def GQ_data_min_resample(min_data, type_='5min'):
    """分钟线 → 更大周期的分钟线（`5min` / `15min` / `30min` / `60min` / `1D`）。

    QUANTAXIS `QA_data_min_resample` 的忠实回迁（逐行，含其口径选择）。

    **A 股分段交易时段**：上午段 `9:30-11:30` 用 `offset="30min"` 且
    `closed='right'`，下午段 `13:00-15:00` 用 `offset="0min"`，两段各 resample
    后拼接并按索引排序。这个不对称的 offset 是原实现的形状，**不要「统一」它** ——
    改了会让 11:30 与 13:00 那两根落进错误的桶。

    返回帧的索引是 `['datetime', 'code']`，并带一列 `type`（`type_`；`'1D'` 时写成
    `'day'`）。空结果返回空帧。

    :param min_data: 分钟线帧，索引含 `datetime` 与 `code` 两层。
    :param type_: 目标周期，pandas 频率串（`'5min'` / `'60min'` / `'1D'` …）。
    """
    conversion = _min_conversion(min_data)
    min_data = min_data.loc[:, list(conversion.keys())]
    idx = min_data.index
    part_1 = min_data.iloc[idx.indexer_between_time('9:30', '11:30')]
    part_1_res = part_1.resample(
        type_,
        offset="30min",
        closed='right',
    ).apply(conversion)
    part_1_res.index = part_1_res.index + to_offset(type_)
    part_2 = min_data.iloc[idx.indexer_between_time('13:00', '15:00')]
    part_2_res = part_2.resample(
        type_,
        offset="0min",
        closed='right',
    ).agg(conversion)
    part_2_res.index = part_2_res.index + to_offset(type_)
    part_1_res['type'] = part_2_res['type'] = type_ if (type_ != '1D') else 'day'
    return pd.concat(
        [part_1_res, part_2_res]
    ).dropna().sort_index().reset_index().set_index(['datetime', 'code'])


def GQ_data_min_to_day(min_data, type_='1D'):
    """分钟线 → 日线。QUANTAXIS `QA_data_min_to_day` 的忠实回迁。

    与 :func:`GQ_data_min_resample` 不同，本函数**不分交易时段**（日线不需要分段），
    直接对 `datetime` 重采样。`min_data` 的索引需含 `code` 层（第 1 层），它会先
    `reset_index(1)` 把 `code` 落到列上，因为聚合口径里 `code` 取 `first`。

    返回帧的索引是 `datetime`（未命名轴由调用方决定，原实现的调用点会
    `rename_axis('date')`）。
    """
    conversion = _min_conversion(min_data)
    return min_data.reset_index(1).resample(
        type_,
        offset="0min",
        closed='right'
    ).agg(conversion).dropna()


def rolling_poly9_v2(s, w: int = 252):
    """滚动**九次多项式**拟合，取每个窗口的**端点值**作输出（第二版）。

    逐字搬自老树 `analysis/timeseries.py:1014`（`POLYNOMIAL9` 的生产核，
    `features/base.py:2110` 用 `rolling_poly9_v2(s, 252)` 调它）。

    它在算一条**平滑的趋势基线**：在收盘价序列上开 `w` 长的滑动窗，
    每窗用 Chebyshev 九次多项式拟合，只要**窗口末端那一根**的拟合值 ——
    于是得到与原序列等长的一条"去噪趋势线"。`POLYNOMIAL9_DELTA`（它的一阶差分）
    与 `POLYNOMIAL9_TIMING_LAG`（由差分符号算的金叉/死叉时序）都建在它上面。

    递归折半（**这是 `_v2` 与 `_v1` 的唯一区别**）
    ============================================
    `w` 大而序列不够长时，若直接开窗会得到**一整段 NaN**。`_v2` 的做法是把 `w` 折半
    **递归**，并把前缀段单独拿去递归补齐：

    * `len(s) > w 且 w > 122` → 开窗扫一遍（每窗出 1 个点），**前缀 `s[:w-1]` 用 `w/2` 递归补**；
    * `len(s) > w/2 且 w > 64` → 只做递归，不扫窗；
    * 否则 → 整段一次拟合（度数 9）。

    实测 `w=252`、`len=900` 的实际路径是 `252 → 126 → 63`（最后落 else），
    三段拼起来**恰好 900 个数**。

    ⚠️ **不设 `random_state` 之类**：这里没有随机性，但**也没有容差** ——
    搬运判据是 `np.array_equal`（`DECISIONS.md` D35/D36）。

    ⚠️ **本函数刻意不做 JIT**（用户 2026-10-16 问过）：它的热循环是
    `map(last_poly9, windows)`，而每次迭代调的是 `np.polynomial.Chebyshev.fit`
    —— **numpy 的 C 实现**，numba 编译不了。硬写成手搓最小二乘会**动最低位**，
    直接破坏上面的 `array_equal` 对拍。这一族真正该 JIT 的是
    `mapower_np_vXIIs` / `inv_nums`（O(n²) 双重 Python 循环）与
    `magic_nine`（逐 bar 循环）—— 那两处已各有实现。

    >>> import numpy as np
    >>> y = rolling_poly9_v2(np.arange(30, dtype=float), 20)
    >>> len(y)                      # 输出与输入**等长**
    30
    >>> round(float(y[-1]), 6)      # 线性序列上九次拟合能精确落到端点
    29.0

    :param s: 价格序列（1-D array-like）
    :param w: 窗口长；`>122` 才会真的滑动开窗，否则退化成整段拟合
    :returns: 与 `s` 等长的趋势线（`np.ndarray`）

    ⚠️ **长度对不上时它是补零而不是报错**（老树行为，见下面的分支）——
    正常输入不会走到那里；走到就说明上游给的序列有问题。
    """
    s = np.asarray(s)
    x_index = range(w)

    def last_poly9(sx):
        p = np.polynomial.Chebyshev.fit(x_index, sx, 9)
        return p(x_index)[-1]

    if (len(s) > w) and (w > 122):
        # 惰性 import：**不在模块顶层** —— `timing` 是本模块的同级，顶层导入没问题，
        # 但这里放在函数内是为了让「谁真的用到 sliding window」一眼可见。
        from .timing import strided_app
        x = strided_app(s, w, 1)
        return np.r_[rolling_poly9_v2(s[0: w - 1], w=int(w / 2)),
                     np.array(list(map(last_poly9, x)))]
    elif (len(s) > int(w / 2)) and (w > 64):
        return rolling_poly9_v2(s, w=int(w / 2))
    else:
        x_index = range(len(s))
        p = np.polynomial.Chebyshev.fit(x_index, s, 9)
        return p(x_index)
