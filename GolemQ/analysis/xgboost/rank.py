# coding:utf-8
"""markup 的**评分 rank**（-9..9）—— 旧树 `portfolio/base.py:1923-2227`。

这是 `markup/shapelet.py` / `compact.py` 用的那套「按分型收益给 bar 打分」的
工具，不是 `mark_uprising` 的多空标签。核心三步：

1. :func:`get_wait_all_greedy_opt` / :func:`get_wait_all_greedy_neg` —— 用
   未来 `prd` 根的**全涨/全跌**判据生成贪婪买卖点（旧树 numba jit，这里纯 numpy）；
2. :func:`calc_fractal_stats`（`:mod:markup`）把买卖点变成交易收益；
3. 按收益分位数从 11 递减出 rank（9/7/5/3/1 与负向 -1..-11），
   写回 `PFL.STOCK_PORTFOLIO_RANK` 等列。

⚠️ 旧名 `calc_portfolio_returns`；新树叫 :func:`calc_portfolio_rank`（更贴切）。
⚠️ 旧函数**会修改入参**；这里先 `copy()`，语义不变但调用方更安全。
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from GolemQ.analysis.indicators import calc_macd_qa, cross_above
from GolemQ.analysis.quadrant_jit import (clearup_compact_dataframe_float32,
                                          clearup_compact_dataframe_int16)
from GolemQ.analysis.timing import Timeline_duration, calc_event_timing_lag
from GolemQ.core.constants import AKA, FIELD
from GolemQ.models.alias import PFL, TRD

from .markup import ATTENTION_LABEL, calc_fractal_stats

__all__ = [
    'ZEN_SHAPELET_TIMING_LAG', 'ZEN_SHAPELET_TIMING_LAG_MAJOR',
    'get_wait_all_greedy_opt', 'get_wait_all_greedy_neg',
    'macd_cross_features', 'calc_portfolio_metaphase', 'calc_portfolio_rank',
    'rank_from_83',
]

#: 旧树 `FLD.ZEN_SHAPELET_TIMING_LAG` / `_MAJOR` 的存储值（新树常量类未收录）。
ZEN_SHAPELET_TIMING_LAG = 'ZS_LAG'
ZEN_SHAPELET_TIMING_LAG_MAJOR = 'ZS_LAG_MAJ'


def get_wait_all_greedy_opt(ys: np.ndarray, prd: int = 5) -> np.ndarray:
    """贪婪买点：未来 `prd+1` 根**全部上涨** → 1，否则 -1（旧树 jit 的纯 numpy 版）。

    >>> get_wait_all_greedy_opt(np.arange(1.0, 9.0), prd=3).tolist()[:4]
    [1, 1, 1, 1]
    """
    qs = np.zeros(len(ys)).astype(np.int16)
    pre_q = 0
    for i in range(len(ys[:-prd])):
        sub = ys[i:i + prd + 1]
        log_gain_loss = np.log(sub[1:]) - np.log(sub[0])
        if pre_q == 1:
            qs[i] = -1 if np.all(log_gain_loss < 0) else 1
        elif pre_q == -1:
            qs[i] = 1 if np.all(log_gain_loss > 0) else -1
        else:
            qs[i] = 1 if np.all(log_gain_loss > 0) else -1
        pre_q = qs[i]
    return qs


def get_wait_all_greedy_neg(ys: np.ndarray, prd: int = 5) -> np.ndarray:
    """贪婪空点：`get_wait_all_greedy_opt` 的符号取反版（旧树原样）。"""
    qs = np.zeros(len(ys)).astype(np.int16)
    pre_q = 0
    for i in range(len(ys[:-prd])):
        sub = ys[i:i + prd + 1]
        log_gain_loss = -(np.log(sub[1:]) - np.log(sub[0]))
        if pre_q == 1:
            qs[i] = -1 if np.all(log_gain_loss < 0) else 1
        elif pre_q == -1:
            qs[i] = 1 if np.all(log_gain_loss > 0) else -1
        else:
            qs[i] = 1 if np.all(log_gain_loss > 0) else -1
        pre_q = qs[i]
    return qs


def macd_cross_features(ohlc: pd.DataFrame) -> pd.DataFrame:
    """从 OHLC 现算 MACD 死叉两列（rank 的必需输入）。

    逐字照旧树：

    * ``MACD_CROSS_SX_BEFORE = Timeline_duration(CROSS(DEA, DIF))``
      —— **DEA 上穿 DIF = DIF 下穿 DEA = 死叉**，值是「距上一次死叉的 bar 数」；
    * ``MACD_CROSS_SX_BEFORE_MAJOR = Timeline_duration(
        where(SX_BEFORE.diff(1) < 0, 1, 0))``
      —— 距**最近一次新死叉**的 bar 数（旧树 `features/base.py:3478`）。

    ⚠️ MACD 必须用 :func:`analysis.indicators.calc_macd_qa`（QUANTAXIS 口径），
    不是 `talib.MACD` —— 两者 EMA 初始化与柱值都不同。
    """
    if AKA.CLOSE not in ohlc.columns:
        raise ValueError('macd_cross_features 需要 close 列')
    dif, dea, _ = calc_macd_qa(ohlc[AKA.CLOSE])
    sx = cross_above(dea, dif).to_numpy(dtype=np.int16)
    before = Timeline_duration(sx).astype(np.int32)
    major = Timeline_duration(
        np.where(pd.Series(before).diff(1) < 0, 1, 0)).astype(np.int16)
    out = ohlc.copy()
    out[FIELD.MACD_CROSS_SX_BEFORE] = before
    out[FIELD.MACD_CROSS_SX_BEFORE_MAJOR] = major
    return out


def calc_portfolio_metaphase(stock_portfolio_rank_lag_np: np.ndarray,
                             macd_cross_sx_before: np.ndarray,
                             macd_cross_sx_bf_maj: np.ndarray) -> np.ndarray:
    """走势阶段（旧树 `portfolio/base.py:1852-1920`，逐字）。"""
    portfolio_metaphase_ofs = 0
    macd_sx_ofs = 0
    portfolio_metaphase_ofs_maj = 0
    macd_sx_ofs_maj = 0
    out = np.zeros((len(stock_portfolio_rank_lag_np), 2)).astype(np.int16)
    rank_lag = stock_portfolio_rank_lag_np

    for i in range(len(rank_lag)):
        if rank_lag[i] > 0:
            if rank_lag[i] > 0.0000168:
                if macd_cross_sx_before[i] > rank_lag[i]:
                    out[i, 0] = portfolio_metaphase_ofs
                    macd_sx_ofs = 0
                    portfolio_metaphase_ofs = 0
                elif macd_cross_sx_before[i] < rank_lag[i]:
                    if (macd_cross_sx_before[i - 1] > rank_lag[i - 1]) or \
                            (macd_cross_sx_before[i] < (rank_lag[i] - macd_sx_ofs)):
                        portfolio_metaphase_ofs += 1
                        macd_sx_ofs = rank_lag[i] + 1
                        out[i, 0] = portfolio_metaphase_ofs
                    elif macd_cross_sx_before[i] > (rank_lag[i] - macd_sx_ofs):
                        out[i, 0] = portfolio_metaphase_ofs

                if (macd_cross_sx_bf_maj[i] + 3) > rank_lag[i]:
                    out[i, 1] = portfolio_metaphase_ofs_maj
                    macd_sx_ofs_maj = 0
                    portfolio_metaphase_ofs_maj = 0
                elif (macd_cross_sx_bf_maj[i] + 3) < rank_lag[i]:
                    if (macd_cross_sx_bf_maj[i - 1] > rank_lag[i - 1]) or \
                            ((macd_cross_sx_bf_maj[i] + 3)
                             < (rank_lag[i] - macd_sx_ofs_maj)):
                        portfolio_metaphase_ofs_maj += 1
                        macd_sx_ofs_maj = rank_lag[i]
                        out[i, 1] = portfolio_metaphase_ofs_maj
                    elif (macd_cross_sx_bf_maj[i] + 3) > \
                            (rank_lag[i] - macd_sx_ofs_maj):
                        out[i, 1] = portfolio_metaphase_ofs_maj
            else:
                out[i, 0] = np.nan
                out[i, 1] = np.nan
    return out


def calc_portfolio_rank(features: pd.DataFrame,
                        ohlc_data: pd.DataFrame = None,
                        verbose: bool = False) -> pd.DataFrame:
    """给 bar 打 -9..9 的 rank（旧树 `calc_portfolio_returns` 的 DataFrame 版）。

    :param features: MultiIndex `(时间, code)` 的特征帧，需含 `close`
        （或由 `ohlc_data` 补）。
    :param ohlc_data: 可选的 OHLC 帧；`features` 没有 close 时用它补。
    :returns: 新帧（含 `PFL.STOCK_PORTFOLIO_RANK` 族），临时列已删。
    """
    features = features.copy()
    if AKA.CLOSE in features.columns:
        price = features[AKA.CLOSE].astype(np.float64).values
    elif ohlc_data is not None and AKA.CLOSE in ohlc_data.columns:
        price = ohlc_data[AKA.CLOSE].astype(np.float64).values
        features[AKA.CLOSE] = ohlc_data[AKA.CLOSE]
    else:
        raise ValueError('calc_portfolio_rank 需要 close：features/ohlc_data 都没有')
    for col in (FIELD.MACD_CROSS_SX_BEFORE, FIELD.MACD_CROSS_SX_BEFORE_MAJOR):
        if col not in features.columns:
            raise ValueError('calc_portfolio_rank 缺 MACD 死叉列：{}'.format(col))

    features = features.reindex(columns=list(set([
        *features.columns, ZEN_SHAPELET_TIMING_LAG,
        ZEN_SHAPELET_TIMING_LAG_MAJOR, 'neg_rank', 'neg_rank_maj'])))
    features[ZEN_SHAPELET_TIMING_LAG] = get_wait_all_greedy_opt(price, prd=36)
    features[ZEN_SHAPELET_TIMING_LAG_MAJOR] = get_wait_all_greedy_opt(price, prd=126)
    features['neg_rank'] = get_wait_all_greedy_neg(price, prd=36)
    features['neg_rank_maj'] = get_wait_all_greedy_neg(price, prd=126)

    code = features.index.get_level_values(level=1)[0]

    pos_stats = pd.concat([
        calc_fractal_stats(code, code, 'zenshap01',
                           features[ZEN_SHAPELET_TIMING_LAG].values,
                           mode='raw', ref_features=features, annual=1008),
        calc_fractal_stats(code, code, 'zenshap02',
                           features[ZEN_SHAPELET_TIMING_LAG_MAJOR].values,
                           mode='raw', ref_features=features, annual=1008),
    ], axis=0, sort=True).sort_index()

    neg_stats = pd.concat([
        calc_fractal_stats(code, code, 'neg01',
                           features['neg_rank'].values,
                           mode='raw', ref_features=features, annual=1008),
        calc_fractal_stats(code, code, 'neg02',
                           features['neg_rank_maj'].values,
                           mode='raw', ref_features=features, annual=1008),
    ], axis=0, sort=True).sort_index()

    # ---- 多头 rank：11 起、按收益分位递减到 9/7/5/3/1 ----
    pos_views = None
    if not np.isnan(pos_stats['ret_fractal'].quantile(0.382)):
        pos_views = pos_stats.query(
            'ret_fractal>{}'.format(pos_stats['ret_fractal'].quantile(0.382))
        ).sort_values(by=['ret_fractal'], ascending=[False])

    # ---- 空头 rank：-11 起、按收益分位递减到 -1 ----
    neg_views = None
    if not np.isnan(neg_stats['ret_fractal'].quantile(0.618)):
        neg_views = neg_stats.query(
            'ret_fractal<{}'.format(neg_stats['ret_fractal'].quantile(0.618))
        ).sort_values(by=['ret_fractal'], ascending=[True])

    features = features.reindex(columns=list(set([
        *features.columns,
        PFL.STOCK_PORTFOLIO_RANK, PFL.STOCK_PORTFOLIO_RETURNS,
        PFL.STOCK_PORTFOLIO_RANK_NEGATIVE,
        PFL.STOCK_PORTFOLIO_RANK_BEFORE,
        PFL.STOCK_PORTFOLIO_RANK_TRACING, ATTENTION_LABEL])))

    features[PFL.STOCK_PORTFOLIO_RANK] = np.nan
    features[PFL.STOCK_PORTFOLIO_RETURNS] = np.nan
    features[PFL.STOCK_PORTFOLIO_RANK_NEGATIVE] = np.nan

    if neg_views is not None and len(neg_views) > 0:
        current_rank = 11
        current_returns = -999
        for index, marker in neg_views.iterrows():
            if marker['ret_fractal'] > current_returns * 0.618:
                if ((marker['ret_fractal'] < -0.25) and current_rank == 11) or \
                        (marker['ret_fractal'] < current_returns * 0.382):
                    current_rank = current_rank - 2 if current_rank - 2 > 0 else 0
                else:
                    current_rank = current_rank - 4 if current_rank - 4 > 0 else 0
                current_returns = marker['ret_fractal']
            if index in features.index:
                features.loc[index, ATTENTION_LABEL] = current_rank
                marker_index = features.index.get_level_values(level=0).intersection(
                    pd.date_range(start=marker['trans_start'],
                                  end=marker['trans_end'], freq='30min'))
                has = features.loc[marker_index,
                                   PFL.STOCK_PORTFOLIO_RANK_NEGATIVE].isna()
                features.loc[marker_index,
                             PFL.STOCK_PORTFOLIO_RANK_NEGATIVE] = np.where(
                    has, current_rank,
                    features.loc[marker_index,
                                 PFL.STOCK_PORTFOLIO_RANK_NEGATIVE])
                has = features.loc[marker_index, PFL.STOCK_PORTFOLIO_RANK].isna()
                features.loc[marker_index, PFL.STOCK_PORTFOLIO_RANK] = np.where(
                    has, -current_rank,
                    features.loc[marker_index, PFL.STOCK_PORTFOLIO_RANK])
                has = features.loc[marker_index, PFL.STOCK_PORTFOLIO_RETURNS].isna()
                features.loc[marker_index, PFL.STOCK_PORTFOLIO_RETURNS] = np.where(
                    has, marker['ret_fractal'],
                    features.loc[marker_index, PFL.STOCK_PORTFOLIO_RETURNS])

    if pos_views is not None and len(pos_views) > 0:
        current_rank = 11
        current_returns = 999
        for index, marker in pos_views.iterrows():
            if marker['ret_fractal'] < current_returns * 0.618:
                if (marker['ret_fractal'] < 0.382) and current_rank > 8:
                    current_rank = current_rank - 6 if current_rank - 6 > 0 else 0
                elif ((marker['ret_fractal'] < 0.382) and current_rank > 6) or \
                        ((marker['ret_fractal'] < 0.75) and current_rank > 9):
                    current_rank = current_rank - 4 if current_rank - 4 > 0 else 0
                elif ((marker['ret_fractal'] > 0.618) and current_rank > 9) or \
                        (marker['ret_fractal'] > current_returns * 0.382):
                    current_rank = current_rank - 2 if current_rank - 2 > 0 else 0
                elif (marker['ret_fractal'] > 0.382) and current_rank > 8:
                    current_rank = current_rank - 4 if current_rank - 4 > 0 else 0
                else:
                    current_rank = current_rank - 6 if current_rank - 6 > 0 else 0
                current_returns = marker['ret_fractal']
            if index in features.index:
                features.loc[index, ATTENTION_LABEL] = current_rank
                marker_index = features.index.get_level_values(level=0).intersection(
                    pd.date_range(start=marker['trans_start'],
                                  end=marker['trans_end'], freq='30min'))
                has = features.loc[marker_index, PFL.STOCK_PORTFOLIO_RANK].isna()
                features.loc[marker_index, PFL.STOCK_PORTFOLIO_RANK] = np.where(
                    has, current_rank,
                    features.loc[marker_index, PFL.STOCK_PORTFOLIO_RANK])
                has = features.loc[marker_index, PFL.STOCK_PORTFOLIO_RETURNS].isna()
                features.loc[marker_index, PFL.STOCK_PORTFOLIO_RETURNS] = np.where(
                    has, marker['ret_fractal'],
                    features.loc[marker_index, PFL.STOCK_PORTFOLIO_RETURNS])

    features[PFL.STOCK_PORTFOLIO_RANK_BEFORE] = calc_event_timing_lag(np.where(
        np.isnan(features[PFL.STOCK_PORTFOLIO_RANK]), -1,
        np.where(
            (features[PFL.STOCK_PORTFOLIO_RANK] > 0.0000168)
            | (features[PFL.STOCK_PORTFOLIO_RANK] < -0.0000168)
            | (features[PFL.STOCK_PORTFOLIO_RANK]
               == features[PFL.STOCK_PORTFOLIO_RANK].shift(1)),
            np.where(features[PFL.STOCK_PORTFOLIO_RANK] > 0.0000168, 1, -1),
            np.where(features[PFL.STOCK_PORTFOLIO_RANK] > 0.0000168, -1, 1))))
    features[PFL.STOCK_PORTFOLIO_RANK_TRACING] = \
        features[PFL.STOCK_PORTFOLIO_RANK].ffill()

    if TRD.POLYNOMIAL9_CAST_BEFORE_COMPOUNDED in features.columns:
        cast = features[TRD.POLYNOMIAL9_CAST_BEFORE_COMPOUNDED]
        upgrade = features[(features[PFL.STOCK_PORTFOLIO_RANK] < 9)
                           & (features[PFL.STOCK_PORTFOLIO_RANK].shift(-1) == 9)
                           & (cast < 1) & (cast.shift(-1) < 1)]
        while len(upgrade) > 0:
            features.loc[upgrade.index, PFL.STOCK_PORTFOLIO_RANK] = 9
            cast = features[TRD.POLYNOMIAL9_CAST_BEFORE_COMPOUNDED]
            upgrade = features[(features[PFL.STOCK_PORTFOLIO_RANK] < 9)
                               & (features[PFL.STOCK_PORTFOLIO_RANK].shift(-1) == 9)
                               & (cast < 1) & (cast.shift(-1) < 1)]

    if FIELD.MAGIC_NINE_TURNS_WEEKLY_DUMMY in features.columns:
        m9t_wek = features[(features[FIELD.MAGIC_NINE_TURNS_WEEKLY_DUMMY] > 0)
                           & features[PFL.STOCK_PORTFOLIO_RANK].isna()]
        if len(m9t_wek) > 0:
            ratio = np.log(m9t_wek[AKA.CLOSE].tail(1).item()
                           / m9t_wek[AKA.CLOSE].head(1).item())
            if ratio > 0.618:
                rank = 9
            elif ratio > 0.512:
                rank = 7
            elif ratio > 0.382:
                rank = 5
            else:
                rank = None
            if rank is not None:
                features.loc[m9t_wek.index, PFL.STOCK_PORTFOLIO_RANK] = rank

    # NaN → 0 再 cast：预热段 / 当日缺行本来就会留下 NaN；直接 np.astype 会刷
    # `RuntimeWarning: invalid value encountered in cast`，而且 NaN 实际也是按 0
    # 落进 int（Numpy 的行为）。先显式 nan_to_num 既保持结果又不再刷警告。
    metaphase = calc_portfolio_metaphase(
        np.nan_to_num(features[PFL.STOCK_PORTFOLIO_RANK_BEFORE].values,
                      nan=0).astype(np.int32),
        np.nan_to_num(features[FIELD.MACD_CROSS_SX_BEFORE].values,
                      nan=0).astype(np.int16),
        np.nan_to_num(features[FIELD.MACD_CROSS_SX_BEFORE_MAJOR].values,
                      nan=0).astype(np.int16))
    features[PFL.STOCK_PORTFOLIO_RANK_METAPHASE] = metaphase[:, 0]
    features[PFL.STOCK_PORTFOLIO_RANK_METAPHASE_MAJOR] = metaphase[:, 1]

    features = clearup_compact_dataframe_float32(
        features, [PFL.STOCK_PORTFOLIO_RANK, PFL.STOCK_PORTFOLIO_RETURNS,
                   PFL.STOCK_PORTFOLIO_RANK_NEGATIVE,
                   PFL.STOCK_PORTFOLIO_RANK_TRACING, ATTENTION_LABEL,
                   PFL.STOCK_PORTFOLIO_RANK_COMPOUNDED])
    features = clearup_compact_dataframe_int16(
        features, [PFL.STOCK_PORTFOLIO_RANK_PRE,
                   PFL.STOCK_PORTFOLIO_RANK_BEFORE])

    return features.drop(columns=[ZEN_SHAPELET_TIMING_LAG,
                                  ZEN_SHAPELET_TIMING_LAG_MAJOR,
                                  'neg_rank', 'neg_rank_maj'])


def rank_from_83(code: str, start: str = None, end: str = None,
                 frequency: str = '30min', if_fq: str = 'qfq',
                 realtime: bool = False, verbose: bool = False):
    """从 8.3 读一只标的 → 现算 MACD 死叉 → :func:`calc_portfolio_rank`。

    :param frequency: 默认 **30min** —— rank 里 `pd.date_range(..., freq='30min')`
        的标记窗口就是按 30min bar 写的；换成别的频率时标记窗口会对不齐。
    :param realtime: 默认 ``False``（历史 rank 要确定性；盘中要看态再显式开）。
    :returns: 带 `PFL.STOCK_PORTFOLIO_RANK` 族的 DataFrame；无数据返回 ``None``。
    """
    from GolemQ.core.market_registry import get_active_market

    market = get_active_market()
    res, codename = market.get_kline_price_min(
        code, start=start, end=end, frequency=frequency,
        realtime=realtime, if_fq=if_fq, verbose=verbose)
    data = getattr(res, 'data', None)
    if data is None or len(data) == 0:
        return None
    frame = data
    if not isinstance(frame.index, pd.MultiIndex):
        if 'ts' in frame.columns and 'code' in frame.columns:
            frame = frame.set_index(['ts', 'code'], drop=True)
        else:
            raise ValueError(
                'rank_from_83: 读口返回的帧既不是 (ts, code) MultiIndex，'
                '也没有 ts/code 列（code={}）'.format(code))
    if verbose:
        print('[xgb rank] {} {} → {} bars'.format(code, frequency, len(frame)))
    return calc_portfolio_rank(macd_cross_features(frame), verbose=verbose)
