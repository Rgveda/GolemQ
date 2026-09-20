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

from QUANTAXIS.QAFetch.QAQuery_Advance import QA_fetch_stock_day_adv

from GolemQ.core.constants import AKA
from .symbol import normalize_code
from .fetch import GQ_fetch_stock_min_adv


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
        data_day = QA_fetch_stock_day_adv(
            short_code,
            start=str(start)[:10],
            end=str(end)[:10],
        )

        if data_day is None:
            return pd.DataFrame()

        # 前复权
        if fq:
            data_day = data_day.to_qfq()

        data_day.data[AKA.FULL_SYMBOL] = normalize_code(code)
        data_day.data[AKA.MARKET_TYPE] = 'stock_cn'

        # 修复复权后尾部数据缺失
        try:
            if data_day.data[[AKA.OPEN, AKA.CLOSE]].tail(10).isnull().values.any():
                predict_null = pd.isnull(data_day.data[AKA.CLOSE])
                data_null = data_day.data[predict_null == True]  # noqa: E712
                data_day.data.loc[data_null.index, :] = QA_fetch_stock_day_adv(
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
        if frequency in ['1min', '1m']:
            frequency = '1min'
        elif frequency in ['5min', '5m']:
            frequency = '5min'
        elif frequency in ['15min', '15m']:
            frequency = '15min'
        elif frequency in ['30min', '30m']:
            frequency = '30min'
        elif frequency in ['60min', '60m']:
            frequency = '60min'

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

        # 前复权
        if fq:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=FutureWarning)
                data_min = data_min.to_qfq()

        data_min.data[AKA.FULL_SYMBOL] = normalize_code(code)
        data_min.data[AKA.MARKET_TYPE] = 'stock_cn'

        return data_min.data
