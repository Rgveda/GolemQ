# coding:utf-8
"""缺失数据检测（kline / metadata）。

从 `services/align.py` 拆出（该文件 492 行，超 `services/` 300 行上限）。
**拆分为纯机械搬运，不改行为。**

``calc_stock_hourly_kline_align``          对齐小时级 K 线（⚠️ 见下，**已知坏损**）
``calc_stock_metadata_missing_queries``     找出缺 metadata 的 `(日, 代码)` 并生成问财查询串
``kline_missing_checkpoints``               检查 kline 缺失并写检查点日志

⚠️ 搬运时**原样保留、未修**的坏损
==================================
`calc_stock_hourly_kline_align` **一旦被调用必然 `NameError`**：

    start_date = ...   # 算了
    end_date   = ...   # 算了
    return stock_hourly_feats      # ← 这个名字**全树从未定义**

（`markets/StockCN/scribe.py:652` 附近的 `stock_hourly_feats_snapshot` 是**另一个**
局部变量，与此无关。）该函数**外部零调用者**，`start_date` / `end_date` 算完也没用。

**本次未删也未修**：删除会移除一个公开名字，修则需要猜原意（属策略侧）。
两者都该由所有者决定，故原样搬运并在 commit 里点名。
"""
from __future__ import annotations

import traceback
from datetime import timedelta

import pandas as pd

from GolemQ.core.constants import (
    AKA,
    FIELD as FLD,
    MARKET_TYPE,
)
from GolemQ.core.settings import (
    DATABASE as DATABASE_GolemQ,
)
from GolemQ.markets.StockCN.date_utils import (
    GQ_util_get_last_day,
)
from GolemQ.models.alias import (
    LTT,
)
from GolemQ.services.features import (
    GQ_save_stock_valuation,
)

from ._checkpoint import symbol_checkpoint_log

__all__ = [
    'calc_stock_hourly_kline_align',
    'calc_stock_metadata_missing_queries',
    'kline_missing_checkpoints',
]


def calc_stock_hourly_kline_align(
    code:str,
    freq: str = '60min',
    market_type = MARKET_TYPE.STOCK_CN,
    collections=DATABASE_GolemQ.stock_reality_feat_60min,
):
    # ⚠️ 本函数**必然 NameError**：`stock_hourly_feats` 全树从未定义。
    #    外部零调用者。原样搬运，未修 —— 见模块 docstring。
    start_date=pd.to_datetime(GQ_util_get_last_day()).tz_localize('Asia/Shanghai')-timedelta(days=1680)
    end_date=pd.to_datetime(GQ_util_get_last_day()).tz_localize('Asia/Shanghai')

    return stock_hourly_feats


def calc_stock_metadata_missing_queries(
    features: pd.DataFrame = None,
    verbose: bool = False,
    collections=DATABASE_GolemQ.stock_wencai_metadata_missing_queries,
):

    # 获取包含空值的完整行数据
    null_rows = features[
        (features[LTT.QUADRANT_LEVERAGE_MACD_TIMING_LAG_DUMMY] > 1e-12) &
        (features[LTT.QUADRANT_LEVERAGE_MACD_TIMING_LAG_DUMMY] < features[FLD.MACD_CROSS_SX_BEFORE_MAJOR]) &
        (features[LTT.QUADRANT_LEVERAGE_MACD_LEADIN_UB] > features[AKA.CLOSE]) &
        ((features[AKA.DDE_NUMBER_OF_INDIVIDUAL_INVESTORS].isnull()) |
        (features[AKA.HOT_RANK].isnull()) |
        (features[AKA.BUY_SIGNAL].isnull()) |
        (features[AKA.PCT_RANKING].isnull()))
    ]

    # 生成查询字符串列表，并去重
    query_strings = []
    seen_dates = set()

    for idx in null_rows.index:
        # 提取日期和股票代码
        datetime_str = idx[0]  # datetime部分
        code_symbol = idx[1]   # 股票代码部分

        # 格式化日期（假设需要转换为YYYY-MM-DD格式）
        if hasattr(datetime_str, 'strftime'):
            day_slice = datetime_str.strftime('%Y-%m-%d')
        else:
            day_slice = str(datetime_str).split(' ')[0]  # 如果是字符串，取日期部分

        # 创建唯一标识符（日期+股票代码）
        unique_key = f"{day_slice}_{code_symbol}"

        # 如果这个日期+股票代码组合还没见过，则添加
        if unique_key not in seen_dates:
            seen_dates.add(unique_key)

            # 生成查询字符串
            query_string = '{}日{}的DDE散户数量,{}日{}的个股热度排名,{}日{}的个股热度,{}日{}的换手率,{}日{}的资金流向,{}日{}的支撑压力解读'.format(
                day_slice, code_symbol,
                day_slice, code_symbol,
                day_slice, code_symbol,
                day_slice, code_symbol,
                day_slice, code_symbol,
                day_slice, code_symbol
            )

            query_strings.append({
                AKA.DATE: pd.to_datetime(day_slice),
                AKA.CODE: code_symbol,
                'query': query_string,
                # 'acquired': 0,
                # 'retried': 0,
            })

    stock_missing_metadata_pd = pd.DataFrame(
        query_strings).set_index(
        [AKA.DATE, AKA.CODE], drop=False)

    GQ_save_stock_valuation(
        stock_missing_metadata_pd,
        collections=collections,
        verbose=verbose,)

    return stock_missing_metadata_pd


def kline_missing_checkpoints(
        features: pd.DataFrame,
        ohlc_data: pd.DataFrame,
        checkpoint_remark: str = '',
        verbose=False,):
    """
    检查 kline 是否缺失，CLOSE 字段 是否为np.nan
    """
    join_columns = list(set([
        AKA.OPEN,
        AKA.CLOSE,
        AKA.LOW,
        AKA.HIGH,
        AKA.VOLUME,
        AKA.AMOUNT,]))

    if (AKA.NAME in features.columns):
        missing_name_idx = features.index.difference(
            features.dropna(
                subset=[AKA.NAME,],
                axis=0,
                how="all").index).get_level_values(level=0)
        missing_name = features.dropna(
            subset=[AKA.NAME],
            axis=0,
            how="all",)[AKA.NAME].tail(1).item()

        if (len(missing_name_idx)>0) and \
            ((GQ_util_get_last_day()-missing_name_idx[-1])<timedelta(days=21)):
            features.loc[missing_name_idx,
                        AKA.NAME]=missing_name

    if (features[AKA.CLOSE].tail(9).isnull().values.any()==True) or \
        (features[AKA.CLOSE].head(9).isnull().values.any()==True):
        missing_kline_idx=features.index.difference(features.dropna(subset=[AKA.CLOSE],
                                                                    axis=0,
                                                                    how="all",).index)
        try:
            features.loc[missing_kline_idx,
                        join_columns]=ohlc_data.loc[missing_kline_idx,
                                                    join_columns]
        except Exception as e:
            codelist=features.index.get_level_values(level=1)[0]
            log_msg='Code:{} Ohlc data missing: {}'.format(codelist, e)
            try:
                symbol_checkpoint_log(
                    logs=log_msg,
                    symbol=codelist,
                    length=len(ohlc_data) if (ohlc_data is not None) else None,
                    FrozenExpired=pd.to_datetime(GQ_util_get_last_day())+timedelta(days=9),
                    catalog=AKA.CHECKPOINT,)
            except Exception:
                print(u'\n{}'.format(log_msg))
                if (len(missing_kline_idx.difference(ohlc_data.index))>0) and \
                    (len(missing_kline_idx.difference(ohlc_data.index))<21):
                    traceback.print_exc()
                    pass
                else:
                    traceback.print_exc()
            finally:
                return features
        if (len(checkpoint_remark)>0):
            code = features.index.get_level_values(level=1)[0]
            log_msg='Code: {} Missing kline {} at checkpoint: {}'.format(code,
                                                                len(missing_kline_idx),
                                                                checkpoint_remark)
            try:
                symbol_checkpoint_log(logs=log_msg,
                                symbol=code,
                                length=len(ohlc_data) if (ohlc_data is not None) else None,
                                catalog=AKA.CHECKPOINT,)
            except Exception:
                print(u'\n{}'.format(log_msg))
                if (len(missing_kline_idx.difference(ohlc_data.index))>0) and \
                    (len(missing_kline_idx.difference(ohlc_data.index))<21):
                    pass
                else:
                    traceback.print_exc()
            finally:
                pass
        if (verbose):
            print(features[[AKA.OPEN,
                        AKA.CLOSE,
                        AKA.LOW,
                        AKA.HIGH,
                        AKA.VOLUME,
                        AKA.AMOUNT,]].tail(20))

    return features
