# coding:utf-8
"""`xgb_scan` 的 **markup 标签**（旧树 `markup/mark_uprising.py:130-325`）。

三层：

1. :func:`calc_transcation_stats_np` —— 分型触发 → 交易状态机的逐 bar 收益
   （旧树 `portfolio/utils.py:461-620`，纯 numpy）。
2. :func:`calc_fractal_stats` —— **仅 `mode='raw'`**（`markup_uprising_labels`
   唯一用到的模式）；`lrc` / `zen` / `hmapower` 在旧树里另有分支，本模块不搬，
   调用时显式 `NotImplementedError`。
3. :func:`markup_uprising_labels` —— 多空标签（1..6 多头、-1..-4 空头），
   规则逐条保留。

列名解析
========
新树契约列与旧存储名并不完全一致（例如 `ZEN_D_LAG_MIN` → `zDLagMinR`）。
:data:`LABEL_INPUTS` 为每个逻辑输入记 ``(canonical, legacy)``：
**先找 canonical，再回退 legacy**；两者都不在就由
:func:`validate_markup_inputs` 一次性列出并抛错 —— **不静默 NaN**。
"""
from __future__ import annotations

from typing import Dict, Iterable, Mapping, Tuple

import numpy as np
import pandas as pd

from GolemQ.analysis.timing import Timeline_duration, calc_event_timing_lag
from GolemQ.core.constants import AKA, FIELD, FEATURES, TREND_STATUS

__all__ = [
    'LABEL_INPUTS', 'ATTENTION_ZONE', 'ATTENTION_SIGNAL_BEFORE',
    'ATTENTION_SIGNAL_SX_BEFORE', 'ATTENTION_LABEL', 'ATTENTION_NEGATIVE_LABEL',
    'markup_required_columns', 'validate_markup_inputs',
    'calc_transcation_stats_np', 'calc_fractal_stats', 'markup_uprising_labels',
]

#: 旧树 `TREND_STATUS` / `FIELD` 里的标签列（新树常量类尚未收录）。
ATTENTION_ZONE = 'ATTEN_ZONE'
ATTENTION_SIGNAL_BEFORE = 'ATTEN_SIGNAL_BF'
ATTENTION_SIGNAL_SX_BEFORE = 'ATTEN_SIGNAL_SX_BF'
ATTENTION_LABEL = 'ATTEN_LABEL'
ATTENTION_NEGATIVE_LABEL = 'ATTEN_NEG_LABEL'

#: 逻辑输入 → (canonical, legacy)。canonical 为 ``None`` 表示新树还没有对应常量。
#:
#: 真值来源：旧树 `utils/parameter.py`（legacy）与新树 `core/constants.py`（canonical）。
LABEL_INPUTS: Mapping[str, Tuple[str, str]] = {
    'ZEN_WAVELET_TIMING_LAG': (None, 'ZenOLSLag'),
    'POLYNOMIAL9_Donchian_STAR_TIMING_LAG':
        (None, 'POLY9_Donchian_STAR_LAG'),
    'MAPOWER30_METRIC_CANDICATE_TIMING_LAG':
        (None, 'MPWR3_MTR_CAN_LAG'),
    'STOCK_SCORE_PUNCH': (None, 'Score_M15_PUNCH'),
    'STOCK_SCORE_M15_NORM': (FIELD.STOCK_SCORE_M15_NORM, 'SSM15_NORM'),
    'POLYNOMIAL9_Donchian_STAR': (None, 'POLY9_Donchian_STAR'),
    'POLYNOMIAL9_Donchian_BAND_CANDICATE':
        (None, 'POLY9_Donchian_BAND_CAN'),
    'MAPOWER30_METRIC_CANDICATE': (None, 'MPWR3_MTR_CAN'),
    'MTM_FLOW_TIMING_LAG': (None, 'MTM_FLOW_LAG'),
    'MONEYFLOW_IN_BEFORE': (None, 'MONEYFLOW_IN_BF'),
    'ZEN_DASH_TIMING_LAG': (FEATURES.ZEN_DASH_TIMING_LAG_MAJOR_REAL, 'zDLag'),
    'ZEN_PEAK_TIMING_LAG_MINOR':
        (FEATURES.ZEN_PEAK_TIMING_LAG_MINOR_REAL, 'ZEN_PEAK_LAG_MIN'),
    'ZEN_DASH_TIMING_LAG_MINOR':
        (FEATURES.ZEN_DASH_TIMING_LAG_MINOR_REAL, 'ZEN_D_LAG_MIN'),
    'POLYNOMIAL9_POWER30': (FIELD.POLYNOMIAL9_NORM, 'poly9pwr3'),
    'MTMPOWER30': (FIELD.MTM_NORM, 'MTM_NORM'),
    'ATR_Stopline_TIMING_LAG': (FIELD.ATR_Stopline_TIMING_LAG, 'ATR_Stopline_LAG'),
    'ATR_SuperTrend_TIMING_LAG': (FIELD.ATR_SuperTrend_TIMING_LAG, 'ATR_SUPTRD_LAG'),
    'DESINTY_CHAIN_TIMING_LAG': (None, 'DESINTY_CHAIN_LAG'),
    'MAPOWER30_RAIL': (None, 'MPWR3_RAIL'),
    'MACD_ZERO_TIMING_LAG': (FIELD.MACD_ZERO_TIMING_LAG, 'MACD_O_LAG'),
    'PEAK_LOW_TIMING_LAG': (None, 'PeakLowLag'),
    'MAGIC_NINE_TURNS': (FIELD.MAGIC_NINE_TURNS, 'm9t'),
    'MAGIC_NINE_TURNS_MAJOR': (FIELD.MAGIC_NINE_TURNS_MAJOR, 'm9tMaj'),
    'REGTREE_SLOPE': (FIELD.REGTREE_SLOPE, 'REGTREE_SLP'),
}

#: 旧树里**可选**的输入：缺了不报错，按旧代码的降级路径处理。
#: * `MAPOWER30_METRIC_CANDICATE_TIMING_LAG` —— 旧树 `if ... in columns` 才加
#:   第三个分型统计；
#: * `STOCK_SCORE_PUNCH` —— 旧树缺列时补 NaN；
#: * `REGTREE_SLOPE` —— 旧树缺列时先尝试 `calc_regtree_fractal_func`，
#:   仍缺则跳过 `-1` 空头规则。
OPTIONAL_LABEL_INPUTS = frozenset({
    'MAPOWER30_METRIC_CANDICATE_TIMING_LAG',
    'STOCK_SCORE_PUNCH',
    'REGTREE_SLOPE',
})

#: 算出来的中间列（不属于输入）。
_LABEL_OUTPUTS = (
    ATTENTION_ZONE, ATTENTION_SIGNAL_BEFORE, ATTENTION_SIGNAL_SX_BEFORE,
    ATTENTION_LABEL, ATTENTION_NEGATIVE_LABEL,
)


def _pick(features: pd.DataFrame, logical: str, required: bool = True):
    """先 canonical、再 legacy；可选输入两者都缺时返回 ``None``。"""
    canonical, legacy = LABEL_INPUTS[logical]
    for name in (canonical, legacy):
        if name and name in features.columns:
            return name
    if required:
        raise KeyError(logical)
    return None


def markup_required_columns() -> Dict[str, Tuple[str, str]]:
    """返回 ``{逻辑名: (canonical, legacy)}`` —— 供缺口报告/文档使用。"""
    return dict(LABEL_INPUTS)


def validate_markup_inputs(features: pd.DataFrame) -> None:
    """缺**必需**输入就一次性列出并 `ValueError`（可选输入见声明）。

    >>> validate_markup_inputs(pd.DataFrame())  # doctest: +ELLIPSIS
    Traceback (most recent call last):
    ...
    ValueError: markup 缺 21 列：...
    """
    missing = []
    for logical, (canonical, legacy) in LABEL_INPUTS.items():
        if logical in OPTIONAL_LABEL_INPUTS:
            continue
        if canonical not in features.columns and legacy not in features.columns:
            missing.append('{}（{}/{}）'.format(logical, canonical, legacy))
    if missing:
        raise ValueError('markup 缺 {} 列：{}'.format(
            len(missing), ', '.join(missing)))


def calc_transcation_stats_np(fractal_triggers: np.ndarray,
                              closep: np.ndarray,
                              fractal_forcast_position: np.ndarray,
                              fractal_timing_lag: np.ndarray,
                              taxfee: float = 0.0003,
                              long: int = 1) -> np.ndarray:
    """分型触发的交易状态机（逐字搬自旧树 `portfolio/utils.py:461-620`）。

    返回 ``(n_trades, 14)`` 的 float32 数组，列序：
    ``trans_stats, trans_start, trans_act, trans_end, start_principle,
    ret_3d, ret_fractal, price_settle, trans_3d, price_end_3d,
    price_end_fractal, ret_fractal_sim, long, duration_time``。

    ⚠️ `taxfee` 在旧实现里**没有实际使用**（保留形参以保持签名兼容）。
    """
    stats_nop = 0
    stats_onhold = 1
    stats_suspended = 2
    stats_closed = 3
    stats_teminated = 4

    idx_transcation = -1
    idx_stats, idx_start, idx_action, idx_endpoint = 0, 1, 2, 3
    idx_start_principle, idx_forcast_returns, idx_fractal_returns = 4, 5, 6
    idx_settle_price, idx_trans_3d, idx_end_3d, idx_end_fractal = 7, 8, 9, 10
    idx_fractal_sim_returns, idx_long, idx_duration = 11, 12, 13

    out = np.zeros((len(closep), 14)).astype(np.float32)
    onhold_price = 0.0
    onhold_position_3d = onhold_position_lineareg = False
    assert long in (1, -1)
    out[:, idx_long] = long

    for i in range(0, len(closep)):
        if (fractal_triggers[i] > 0) and \
                (not onhold_position_3d) and (not onhold_position_lineareg):
            onhold_position_3d = True
            onhold_position_lineareg = True
            idx_transcation += 1
            out[idx_transcation, idx_start] = i
            out[idx_transcation, idx_stats] = stats_onhold
            out[idx_transcation, idx_action] = i
            onhold_price = closep[i]
            out[idx_transcation, idx_settle_price] = onhold_price

        if onhold_position_lineareg and (fractal_timing_lag[i] <= 0):
            onhold_position_lineareg = False
            onhold_position_3d = False
            out[idx_transcation, idx_endpoint] = i
            ret = (closep[i] - onhold_price) / onhold_price * long
            out[idx_transcation, idx_fractal_sim_returns] = ret
            out[idx_transcation, idx_fractal_returns] = ret
            out[idx_transcation, idx_end_fractal] = closep[i]
            out[idx_transcation, idx_stats] = stats_closed
            onhold_price = 0.0

        if onhold_position_3d and (fractal_forcast_position[i] <= 0):
            onhold_position_3d = False
            out[idx_transcation, idx_trans_3d] = fractal_timing_lag[i]
            if onhold_position_lineareg:
                out[idx_transcation, idx_forcast_returns] = \
                    (closep[i] - onhold_price) / onhold_price * long
                out[idx_transcation, idx_end_3d] = closep[i]
            else:
                out[idx_transcation, idx_stats] = stats_teminated
                onhold_price = 0.0

        if onhold_position_lineareg and ((len(closep) - i) < 2):
            if int(out[idx_transcation, idx_stats]) == stats_onhold:
                out[idx_transcation, idx_endpoint] = i
                out[idx_transcation, idx_fractal_returns] = \
                    (closep[i] - onhold_price) / onhold_price * long
                out[idx_transcation, idx_end_fractal] = closep[i]

        if not ((out[idx_transcation, idx_endpoint]
                 - out[idx_transcation, idx_action]) < 0
                and ((len(closep) - i) < 2)):
            out[idx_transcation, idx_duration] = \
                out[idx_transcation, idx_endpoint] - out[idx_transcation, idx_action]

    return out[:idx_transcation + 1, :]


def calc_fractal_stats(symbol, display_name, fractal_uid, fractal_triggers,
                       ref_features: pd.DataFrame = None, rsk_fre: float = 0.04,
                       annual: int = 252, taxfee: float = 0.0003,
                       long: int = 1, mode: str = 'raw') -> pd.DataFrame:
    """分型触发点的绩效统计 —— **只搬 `mode='raw'`**（旧 `portfolio/utils.py:49-283`）。

    `markup_uprising_labels` 的三个 `calc_fractal_stats` 调用全部是
    ``mode='raw'``；`lrc`/`zen`/`hmapower` 分支依赖旧树其它指标，未搬。
    """
    if mode != 'raw':
        raise NotImplementedError(
            'calc_fractal_stats 只搬了 mode="raw"；{} 未搬（见旧树 '
            'portfolio/utils.py:116-255）'.format(mode))

    if (annual > 125) and (annual < 366):
        fractal_limited = 3
    elif (annual > 1680) and (annual < 2560):
        fractal_limited = 24
    elif (annual > 512) and (annual < 1280):
        fractal_limited = 12
    elif (annual > 6180) and (annual < 9600):
        fractal_limited = 72
    else:
        raise ValueError('annual={} 不在旧树的任何周期分支里'.format(annual))

    fractal_cross_before = Timeline_duration(
        np.where(fractal_triggers > 0, 1, 0))
    fractal_forcast_position = np.where(fractal_cross_before < fractal_limited, 1, 0)
    fractal_forcast_3d_lag = calc_event_timing_lag(
        np.where(fractal_forcast_position > 0, 1, -1))
    fractal_forcast_3d_lag = np.where(fractal_forcast_3d_lag <= fractal_limited,
                                      fractal_forcast_3d_lag, 0)
    closep = ref_features[AKA.CLOSE].values

    fractal_position = np.where(fractal_triggers > 0, 1, 0)
    fractal_timing_lag = calc_event_timing_lag(
        np.where(fractal_position > 0, 1, -1))

    transcation_stats = calc_transcation_stats_np(
        fractal_triggers, closep.astype(np.float32),
        fractal_forcast_3d_lag, fractal_timing_lag,
        taxfee=taxfee, long=long)

    transcation_stats_df = pd.DataFrame(
        transcation_stats,
        columns=['trans_stats', 'trans_start', 'trans_act', 'trans_end',
                 'start_principle', 'ret_3d', 'ret_fractal', 'price_settle',
                 'trans_3d', 'price_end_3d', 'price_end_fractal',
                 'ret_fractal_sim', 'long', 'duration_time'])
    transcation_stats_df[AKA.CODE] = symbol
    transcation_stats_df['fractal_uid'] = fractal_uid

    # bar 序号 → 交易时间戳（旧树按 ref_features 的 level-0 索引取）
    selected_trans_start = ref_features.iloc[transcation_stats[:, 1], :]
    transcation_stats_df['trans_start'] = pd.to_datetime(
        selected_trans_start.index.get_level_values(level=0))
    selected_trans_action = ref_features.iloc[transcation_stats[:, 2], :]
    transcation_stats_df['trans_act'] = pd.to_datetime(
        selected_trans_action.index.get_level_values(level=0))
    selected_trans_end = ref_features.iloc[transcation_stats[:, 3], :]
    transcation_stats_df['trans_end'] = pd.to_datetime(
        selected_trans_end.index.get_level_values(level=0))
    transcation_stats_df = transcation_stats_df.assign(
        datetime=pd.to_datetime(
            selected_trans_start.index.get_level_values(level=0))
    ).drop_duplicates(['datetime', 'code']).set_index(['datetime', 'code'],
                                                       drop=True)
    return transcation_stats_df


def markup_uprising_labels(features: pd.DataFrame,
                           verbose: bool = False) -> pd.DataFrame:
    """`xgb_scan` 的多空标签 —— 逐条搬自旧树 `markup/mark_uprising.py:130-325`。

    :returns: **新帧**（入参不被修改），含
        :data:`ATTENTION_LABEL`（1..6 / -1..-4 / NaN）与
        :data:`ATTENTION_NEGATIVE_LABEL`（1..6 / -1..-4 / NaN）。
    :raises ValueError: 缺 label 侧输入列（见 :func:`validate_markup_inputs`）。
    """
    validate_markup_inputs(features)
    features = features.copy()

    # 缺 REGTREE_SLOPE 时补一次（旧树同款；新树复用 analysis/regtree_jit.py）。
    # 它属于**可选**输入：补不出来就跳过 `-1` 空头规则，而不是整个标签失败。
    ohlc_cols = (AKA.OPEN, AKA.CLOSE, AKA.HIGH, AKA.LOW, AKA.VOLUME)
    regtree_col = _pick(features, 'REGTREE_SLOPE', required=False)
    if regtree_col is None and len(features) > 456 and \
            all(c in features.columns for c in ohlc_cols):
        from GolemQ.analysis.regtree_jit import calc_regtree_fractal_func
        features = calc_regtree_fractal_func(features[list(ohlc_cols)],
                                             features=features,
                                             bar_limit=450, rate=25, dur=10)
        regtree_col = _pick(features, 'REGTREE_SLOPE', required=False)

    code = features.index.get_level_values(level=1)[0]
    stock_name = (features[AKA.NAME].iloc[0]
                  if AKA.NAME in features.columns else code)
    each_day = features.index.get_level_values(level=0)

    col = {k: _pick(features, k, required=(k not in OPTIONAL_LABEL_INPUTS))
           for k in LABEL_INPUTS}

    fstats_stars = [
        calc_fractal_stats(code, stock_name, 'alpha_zw01',
                           features[col['ZEN_WAVELET_TIMING_LAG']].values,
                           mode='raw', ref_features=features, annual=1008),
        calc_fractal_stats(
            code, stock_name, 'alpha_stars01',
            features[col['POLYNOMIAL9_Donchian_STAR_TIMING_LAG']].values,
            mode='raw', ref_features=features, annual=1008),
    ]
    mpwr_mtr_col = col['MAPOWER30_METRIC_CANDICATE_TIMING_LAG']
    if mpwr_mtr_col is not None:
        fstats_stars.append(calc_fractal_stats(
            code, stock_name, 'alpha_mpwr_mtr',
            features[mpwr_mtr_col].values,
            mode='raw', ref_features=features, annual=1008))
    fstats = pd.concat(fstats_stars, axis=0, sort=True).sort_index()

    if verbose:
        brief = fstats.query('trans_stats==3')
        pos = fstats.query('trans_stats==3 & ret_fractal>0.00382')
        neg = fstats.query('trans_stats==3 & ret_fractal<0.00382')
        print('Code:{} 模型:{}\n时间:{}-{}\n交易次数:{} 胜率:{:.2%}'.format(
            code, fstats['fractal_uid'].unique(), each_day[0], each_day[-1],
            len(brief), len(pos) / len(brief) if len(brief) else 0.0))

    fstats_views = fstats.query(
        '{}>{}'.format('ret_fractal', fstats['ret_fractal'].quantile(0.382)))
    features[ATTENTION_ZONE] = np.nan
    for index, marker in fstats_views.iterrows():
        if index in features.index:
            features.loc[index, ATTENTION_ZONE] = 1
            sub = features.loc[slice(index[0], marker['trans_end'])]
            features.loc[sub.index, ATTENTION_ZONE] = 1
            if (marker['trans_end'], index[1]) in features.index:
                features.loc[(marker['trans_end'], index[1]),
                             ATTENTION_ZONE] = -1

    if col['STOCK_SCORE_PUNCH'] is None:
        # 旧树同款：缺 `STOCK_SCORE_PUNCH` 时补 NaN（条件自然为 False）。
        col['STOCK_SCORE_PUNCH'] = 'Score_M15_PUNCH'
        features[col['STOCK_SCORE_PUNCH']] = np.nan

    zone = features[ATTENTION_ZONE]
    features[ATTENTION_SIGNAL_BEFORE] = Timeline_duration(
        np.where(zone > 0.00168, 1, 0))
    features[ATTENTION_SIGNAL_SX_BEFORE] = Timeline_duration(
        np.where(zone < -0.00168, 1, 0))
    zone = np.where(features[ATTENTION_SIGNAL_BEFORE]
                    < features[ATTENTION_SIGNAL_SX_BEFORE], 1, 0)
    features[ATTENTION_ZONE] = zone

    punch = features[col['STOCK_SCORE_PUNCH']]
    band = features[col['POLYNOMIAL9_Donchian_BAND_CANDICATE']]
    mpwr_can = features[col['MAPOWER30_METRIC_CANDICATE']]
    zen_dash = features[col['ZEN_DASH_TIMING_LAG']]
    zen_p_min = features[col['ZEN_PEAK_TIMING_LAG_MINOR']]
    zen_d_min = features[col['ZEN_DASH_TIMING_LAG_MINOR']]
    poly9pwr = features[col['POLYNOMIAL9_POWER30']]
    mtm = features[col['MTMPOWER30']]
    mtm_flow = features[col['MTM_FLOW_TIMING_LAG']]
    mf_in_bf = features[col['MONEYFLOW_IN_BEFORE']]
    atr_stop = features[col['ATR_Stopline_TIMING_LAG']]
    atr_sup = features[col['ATR_SuperTrend_TIMING_LAG']]
    m9t = features[col['MAGIC_NINE_TURNS']]
    m9t_maj = features[col['MAGIC_NINE_TURNS_MAJOR']]

    label = np.where((zone > 0) & ((punch > 0.00168) | (band > 0.00168)),
                     1, np.nan)
    label = np.where((zone > 0) & (mpwr_can > 0.00168), 2, label)
    label = np.where((zone > 0)
                     & ((mtm_flow > 0.00168) | (mf_in_bf < 20))
                     & (features[ATTENTION_ZONE].shift(6) < 0.00168),
                     3, label)
    label = np.where((zone > 0.000168) & (zen_dash > 0) & (zen_p_min > 0)
                     & (zen_p_min == zen_d_min), 4, label)
    label = np.where((zone > 0.000168) & (zen_dash > 0)
                     & (mtm_flow == zen_dash)
                     & ((zen_p_min.shift(6) == zen_d_min.shift(6))
                        | (zen_p_min.shift(3) == zen_d_min.shift(3))),
                     5, label)
    label = np.where((zone > 0.000168) & (zen_dash > 0)
                     & ((poly9pwr.diff(1) > 0.00168)
                        | (mtm.diff(1) > 0.00168))
                     & (atr_stop > 0)
                     & ((zen_p_min.shift(6) == zen_d_min.shift(6))
                        | (zen_p_min.shift(3) == zen_d_min.shift(3)))
                     & (atr_stop == atr_sup), 6, label)

    if regtree_col is not None:
        # 旧树把这条规则整个包在 `if FLD.REGTREE_SLOPE in features.columns` 里。
        regtree_slope = features[regtree_col]
        label = np.where((zone < 0.000168)
                         & ((regtree_slope < -0.00168)
                            | (features[col['DESINTY_CHAIN_TIMING_LAG']] < 0))
                         & ((mtm_flow > 0.00168) | (mf_in_bf < 20)
                            | (poly9pwr.diff(1) > 0)), -1, label)
    label = np.where((zone < 0.000168)
                     & (features[col['MAPOWER30_RAIL']] < -0.00168)
                     & (features[col['MACD_ZERO_TIMING_LAG']] > 0.00168)
                     & (features[col['PEAK_LOW_TIMING_LAG']] < -0.00168)
                     & (band > 0.00168), -2, label)
    label = np.where((zone < 0.000168) & (mpwr_can < -0.00168), -3, label)
    label = np.where((zone < 0.000168) & (zen_p_min < 0)
                     & (zen_p_min == zen_d_min), -4, label)
    features[ATTENTION_LABEL] = label

    ssm15 = features[col['STOCK_SCORE_M15_NORM']]
    donchian_star = features[col['POLYNOMIAL9_Donchian_STAR']]

    neg = np.where((zone < 0.00168) & (band < -0.00168)
                   & ~((ssm15.shift(6) < 0.00168) & (ssm15 < 0.00168)),
                   1, np.nan)
    neg = np.where((features[ATTENTION_ZONE].shift(-10) < 0.00168)
                   & (ssm15 < 0.00168)
                   & (donchian_star < -0.00168)
                   & (band < -0.00168)
                   & ~((ssm15.shift(6) < 0.00168) & (ssm15 < 0.00168)),
                   1, neg)
    neg = np.where((zone < 0.00168) & (mpwr_can < -0.00168)
                   & ~((ssm15.shift(6) < 0.00168) & (ssm15 < 0.00168)),
                   2, neg)
    neg = np.where((zone < 0.00168) & (mtm_flow < 0.00168)
                   & (features[ATTENTION_ZONE].shift(6) > 0.00168)
                   & ~((ssm15.shift(6) < 0.00168) & (ssm15 < 0.00168)),
                   3, neg)
    neg = np.where((features[ATTENTION_ZONE].shift(-10) < 0.00168)
                   & ((band < -0.00168) | (ssm15.shift(6) > 0.996))
                   & (zen_p_min < -0.00168) & (zen_p_min == zen_d_min)
                   & ~((ssm15.shift(6) < 0.00168) & (ssm15 < 0.00168)),
                   4, neg)
    neg = np.where((zone < 0.00168) & (mtm_flow < 0.00168)
                   & (features[col['MAPOWER30_RAIL']] < -0.00168)
                   & ~((ssm15.shift(6) < 0.00168) & (ssm15 < 0.00168)),
                   5, neg)
    neg = np.where((zone < 0.00168) & (mtm_flow < 0.00168)
                   & (features[col['MAPOWER30_RAIL']] < -0.00168)
                   & ((m9t < -0.00168) | (m9t_maj < -0.00168))
                   & ((m9t.shift(3) < -0.00168)
                      | (m9t_maj.shift(3) < -0.00168)),
                   6, neg)
    neg = np.where((zone > 0.00168) & (band < -0.00168)
                   & ~(neg > 0.00168), -1, neg)
    neg = np.where((zone > 0.00168) & (mpwr_can < -0.00168)
                   & ~(neg > 0.00168), -3, neg)
    neg = np.where((zone > 0.00168)
                   & (features[col['MAPOWER30_RAIL']] < -0.00168)
                   & ~(neg > 0.00168), -4, neg)
    features[ATTENTION_NEGATIVE_LABEL] = neg
    return features
