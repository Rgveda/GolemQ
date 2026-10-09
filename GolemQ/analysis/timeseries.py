# coding:utf-8
"""时间序列工具 —— 多频重采样与时间轴对齐。

**本模块的定位（两层模型）**：重采样是「所有交易系统共性」的能力，属**第一层**，
放根目录下的 `analysis/`，而不是 `markets/<Market>/`。

本文件只保留两个重采样器（`GQ_data_min_resample` / `GQ_data_min_to_day`）——
原先的两个 stub（`Timeline_duration` / `align_kline_timeline`）已随
`services/persistence/` 的删除一并移除（它们的唯一调用方在那个包里）。
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
