# coding:utf-8
"""ChartScan 的 **K 线渲染**（纯出图，不碰 DB、不做推理）。

搬自旧树 `GolemAI/GolemQ/CV/charts.py:71-121`，**行为逐字保留** —— 只把重依赖
（mplfinance / matplotlib / PIL）挪进函数体（GolemQ env 里没有它们；见 `model.py`）。
"""
from __future__ import annotations

from io import BytesIO

__all__ = [
    'CHUNK_SIZE', 'FIGSIZE', 'DPI', 'CLOSE_TIMES',
    'GQ_generate_chart_stock', 'calculate_i',
]

#: ⚠️ **这三个 + :func:`calculate_i` 的像素映射是**一套**标定，不许各自顺手调**：
#: `calculate_i` 把 YOLO 检出的像素 x 换算成第几根 bar，公式里写死了 `386` 与 `6.5`
#: —— 那两个常数只在 `figsize=(18, 6.5)` + `dpi=100` + `chunk_size=180` 下成立。
#: 动其中任何一个，**所有 x 坐标会静默挪位**，而模型照样给出「可信」的框。
CHUNK_SIZE = 180
FIGSIZE = (18, 6.5)
DPI = 100

#: 60min 的「收盘那几根」—— 只有落在这些时刻的 bar 才算「已收盘」，
#: 才允许把 60min 的检出写进 `*_reality_feat_60min`（旧树 `charts.py:369/379/419/433`）。
CLOSE_TIMES = ('10:30', '11:30', '14:00', '15:00')


def GQ_generate_chart_stock(kline, ticker='', chunk_size=CHUNK_SIZE,
                            figsize=FIGSIZE, dpi=DPI, freq='latest'):
    """把 `kline` 的**最后 `chunk_size` 根**画成蜡烛图，返回内存里的 PNG `BytesIO`。

    ⚠️ **只进内存、不落盘**（旧树也如此 —— `datastore/ChartScanAI/` 是空的）。
    无数据返回 `None`（并打印一行，照旧）。
    """
    import matplotlib.font_manager as fm      # noqa: PLC0415 惰性：GolemQ env 没装
    import matplotlib.pyplot as plt            # noqa: PLC0415
    import mplfinance as mpf                   # noqa: PLC0415

    data = kline
    zh_font = fm.FontProperties(fname='C:/Windows/Fonts/msyh.ttc')
    custom_style = mpf.make_mpf_style(base_mpf_style='yahoo',
                                      rc={'font.family': zh_font.get_name()})
    if len(data) == 0:
        print('No data found for the specified ticker and interval.')
        return None
    import pandas as pd                        # noqa: PLC0415
    data = data.copy()
    data.index = pd.to_datetime(data.index.get_level_values(level=0))
    data = data.iloc[-chunk_size:]
    fig, ax = mpf.plot(data, style=custom_style, type='candle',
                       title=f'{ticker} {freq} {chunk_size} Candles',
                       axisoff=True, ylabel='', ylabel_lower='',
                       volume=False, figsize=figsize, returnfig=True)
    buffer = BytesIO()
    fig.savefig(buffer, format='png', dpi=dpi)
    buffer.seek(0)
    plt.close()
    return buffer


def calculate_i(x, chunk_size=CHUNK_SIZE):
    """YOLO 框的像素 x → **第几根 bar**（相对 `kline.tail(chunk_size)`）。

    逐字照搬旧树 `charts.py:121`：`(x - 386) / 6.5`，下取整再 −1，并夹在
    `[.., chunk_size - 1]`。⚠️ 那两个魔数**与 :data:`FIGSIZE` / :data:`DPI` /
    :data:`CHUNK_SIZE` 是一套标定** —— 见上面的告示。

    >>> calculate_i(386 + 6.5 * 5)      # 第 5 格 → 下标 4
    4
    >>> calculate_i(10 ** 9)            # 越界夹到最后一根
    179
    """
    i = int((x - 386) / 6.5) - 1
    return i if i < chunk_size else (chunk_size - 1)
