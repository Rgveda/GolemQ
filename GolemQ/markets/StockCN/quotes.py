# coding:utf-8
#
# The MIT License (MIT)
#
# Copyright (c) 2020-2025 azai/GolemQ(uant)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
import warnings
import pandas as pd

from GolemQ.core.constants import AKA
from .symbol import normalize_code
# 日线读取改走本树自己的 8.3 路径（`kline83` + `datastruct`），不再依赖
# QUANTAXIS。返回类型 `GQ_DataStruct_Stock_day` 与 `QA_DataStruct_Stock_day`
# 同接口：`.data` / `.to_qfq()`。
from .kline83 import GQ_fetch_stock_day_adv, normalize_frequency
from .etf_fq import GQ_apply_etf_qfq, GQ_is_etf
from .fetch import GQ_fetch_stock_min_adv


def _apply_fq(data, code, fq):
    """按标的种类选复权方式。三种情形，**互斥且都有明确归属**：

    - **ETF** → `GQ_apply_etf_qfq`（因子表 `etf_adj`）。ETF 与真指数共用
      `index_*` 集合，拿到的容器**没有** `to_qfq` —— QUANTAXIS 的 Index
      datastruct 本就没有（见 `datastruct.py`），ETF 复权一直是
      `etf_fq.py` 单独负责的。
    - **股票** → 容器的 `.to_qfq()`（因子表 `stock_adj`）。
    - **真指数** → 不复权，原样返回。真指数在 `etf_adj` 里没有行，
      `GQ_apply_etf_qfq` 也不会为它查库。

    判据用 `hasattr(..., 'to_qfq')` 而不是 `isinstance`：问的是「这个容器提不提供
    股票式复权」，而 `to_qfq` 只挂在 Stock 类上正是 `datastruct.py` 刻意的层级设计。
    """
    if not fq:
        return data
    if GQ_is_etf(code):
        return GQ_apply_etf_qfq(data, codelist=code)
    if hasattr(data, 'to_qfq'):
        return data.to_qfq()
    return data


class StockCNQuotes:
    """A股市场行情数据获取"""

    def get_kline_quotes(
        self,
        code: str,
        start: str = '2025-01-01',
        end: str = '2026-01-01',
        fq: int = 1,
    ) -> pd.DataFrame:
        """获取A股单只股票日线历史行情（前复权）

        Args:
            code: 股票代码，如 '000001' 或 '000001.XSHE'
            start: 开始日期 YYYY-MM-DD
            end: 结束日期 YYYY-MM-DD
            fq: 复权方式，1=前复权，0=不复权

        Returns:
            pd.DataFrame with columns: open, high, low, close, volume, amount, date, code
            返回空DataFrame如果无数据
        """
        short_code = normalize_code(code)[:6]
        data_day = GQ_fetch_stock_day_adv(
            short_code,
            start=str(start)[:10],
            end=str(end)[:10],
        )

        if data_day is None:
            return pd.DataFrame()

        # 前复权（股票走 stock_adj，ETF 走 etf_adj，真指数不复权）
        data_day = _apply_fq(data_day, short_code, fq)

        data_day.data[AKA.FULL_SYMBOL] = normalize_code(code)
        data_day.data[AKA.MARKET_TYPE] = 'stock_cn'

        # 修复复权后尾部数据缺失
        try:
            if data_day.data[[AKA.OPEN, AKA.CLOSE]].tail(10).isnull().values.any():
                predict_null = pd.isnull(data_day.data[AKA.CLOSE])
                data_null = data_day.data[predict_null == True]  # noqa: E712
                data_day.data.loc[data_null.index, :] = GQ_fetch_stock_day_adv(
                    short_code,
                    '{}'.format(data_null.index.get_level_values(level=0).values[0]),
                    '{}'.format(data_null.index.get_level_values(level=0).values[-1]),
                ).data
        except Exception:
            pass

        return data_day.data

    def get_kline_quotes_min(
        self,
        code: str,
        start: str = '2025-01-01',
        end: str = '2026-01-01',
        frequency: str = '60min',
        fq: int = 1,
    ) -> pd.DataFrame:
        """获取A股单只股票分钟线历史行情（前复权）

        Args:
            code: 股票代码，如 '000001' 或 '000001.XSHE'
            start: 开始日期时间
            end: 结束日期时间
            frequency: K线周期 '1min', '5min', '15min', '30min', '60min'
            fq: 复权方式，1=前复权，0=不复权

        Returns:
            pd.DataFrame with columns: open, high, low, close, volume, datetime, code
            返回空DataFrame如果无数据
        """
        # 别名归一化收敛到 kline83 一处。未知频率**抛 ValueError**（原来三份
        # 抄本各自静默沿用原值，会拼出不存在的集合名 → 空结果）。
        frequency = normalize_frequency(frequency)

        # 补全时间部分
        if len(str(start)) == 10:
            start = '{} 09:30:00'.format(str(start)[:10])
        if len(str(end)) == 10:
            end = '{} 15:00:00'.format(str(end)[:10])

        data_min = GQ_fetch_stock_min_adv(
            normalize_code(code),
            start=str(start)[:19],
            end=str(end)[:19],
            frequence=frequency,
        )

        if data_min is None:
            return pd.DataFrame()

        # 修复午休开盘(13:00)导致的NaN数据
        nan_sum = data_min.data[['close', 'open', 'high', 'low']].isna().sum(axis=1)
        nan_bars = data_min.data[(nan_sum > 2)]

        if len(nan_bars) == 0:
            nan_bars = data_min.data.query("volume < 1").copy()
            nan_bars['_date'] = nan_bars.index.get_level_values(level=0).date
            date_counts = nan_bars['_date'].value_counts()
            valid_dates = date_counts[date_counts > 1].index
            nan_bars = nan_bars[nan_bars['_date'].isin(valid_dates)]

        if len(nan_bars) > 0:
            # 条数为4的整倍数（15min频率午休1小时=4条），直接删除
            if len(nan_bars) % 4 == 0:
                data_min.data = data_min.data.drop(nan_bars.index)
            elif any(
                t.time() == pd.Timestamp("13:00:00").time()
                for t in nan_bars.index.get_level_values(level=0)
            ):
                # 按日期分组处理13:00问题
                dates_with_13 = set(
                    t.date() for t in nan_bars.index.get_level_values(level=0)
                    if t.time() == pd.Timestamp("13:00:00").time()
                )
                drop_indices = []
                for d in dates_with_13:
                    day_bars = data_min.data[
                        data_min.data.index.get_level_values(level=0).date == d
                    ]
                    day_volume = day_bars['volume'].sum()
                    has_1130 = any(
                        t.time() == pd.Timestamp("11:30:00").time()
                        for t in day_bars.index.get_level_values(level=0)
                    )
                    if day_volume < 0.168:
                        # 当日全部是零交易，删除该日期全部记录
                        drop_indices.extend(day_bars.index.tolist())
                    elif not has_1130:
                        # 缺少11:30数据，13:00是时间戳错误，修正为11:30
                        for idx in day_bars.index:
                            if idx[0].time() == pd.Timestamp("13:00:00").time():
                                new_time = pd.Timestamp(
                                    f"{d:%Y-%m-%d} 11:30:00"
                                )
                                data_min.data.rename(index={idx: (new_time, idx[1])}, inplace=True)
                    else:
                        # 有11:30且非全零量，仅删除NaN行
                        day_nan = nan_bars[
                            nan_bars.index.get_level_values(level=0).date == d
                        ]
                        drop_indices.extend(day_nan.index.tolist())
                if drop_indices:
                    data_min.data = data_min.data.drop(drop_indices)

        # 前复权（股票走 stock_adj，ETF 走 etf_adj，真指数不复权）
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=FutureWarning)
            data_min = _apply_fq(data_min, normalize_code(code)[:6], fq)

        data_min.data[AKA.FULL_SYMBOL] = normalize_code(code)
        data_min.data[AKA.MARKET_TYPE] = 'stock_cn'

        return data_min.data
