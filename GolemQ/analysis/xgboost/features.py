# coding:utf-8
"""`xgb_scan` 的**特征契约**（旧树 `models/preprocessing.py:242-300` 的 58 个条目）。

为什么单独一个模块
================
旧 `feature_names(model='xgb_scan')` 返回 **58 个条目**，但其中
`MTMPOWER30` / `MTMPOWER30_MAJOR` / `MTMPOWER30_WEEKLY` 与 `MTM_NORM` 系列
指向**同一存储列**（`MTM_NORM` / `MTM_NORM_MAJ` / `MTM_NORM_WEK`）
⇒ 实际只有 **55 个唯一列**。把「列名 / 旧名 / 生产状态」集中在这里，
训练与预测都从同一份契约取列，避免再出现两套平行列表。

canonical 名优先取**新树**的常量值（`FIELD` / `FEATURES`）；新树没有对应常量
的保留旧存储名，并在 `note` 里标 `needs constant`。旧存储名放在 `legacy`，
用于从旧特征帧（4.4 快照 / 对拍数据）映射过来。

状态
====
* ``produced``          review 管线已经产出（或可 1:1 映射到契约列）；
* ``ambiguous``         新树有同名/近名列，但口径（比值 / 频率 / major）**未核对**；
* ``derivable``         用现有 OHLC + talib 可算，只是还没接线；
* ``missing_producer``  需要新函数/新指标，当前无生产者；
* ``missing_source``    数据源本身没搬（千股千评/牛股诊股、资金流、massive）；
* ``deprecated``        用户 2026-10-21 定：上游链未就绪，本轮直接废弃
                        （M9T before / STOCK_SCORE 随后改判为 derivable）。

⚠️ **`feature_names()` 默认不含 deprecated**；训练入口还会用
:func:`validate_features` 拒绝缺列 —— 不允许把缺列静默当成 NaN 喂给模型。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

from GolemQ.core.constants import FIELD, FEATURES

#: 状态常量（给报告与外部调用方用，避免裸字符串）。
STATUS_PRODUCED = 'produced'
STATUS_AMBIGUOUS = 'ambiguous'
STATUS_DERIVABLE = 'derivable'
STATUS_MISSING_PRODUCER = 'missing_producer'
STATUS_MISSING_SOURCE = 'missing_source'
STATUS_DEPRECATED = 'deprecated'

STATUSES = (
    STATUS_PRODUCED, STATUS_AMBIGUOUS, STATUS_DERIVABLE,
    STATUS_MISSING_PRODUCER, STATUS_MISSING_SOURCE, STATUS_DEPRECATED,
)


@dataclass(frozen=True)
class FeatureSpec:
    """一列 `xgb_scan` 特征的契约。

    :param name: canonical 列名（新树常量值；没有对应的保留旧存储名）
    :param legacy: 旧树存储名（`FIELD.*` 的真值）
    :param status: :data:`STATUSES` 之一
    :param producer: 生产方（模块/函数），没有则空串
    :param note: 口径说明 / 迁移注意点
    """

    name: str
    legacy: str
    status: str
    producer: str = ''
    note: str = ''

    @property
    def deprecated(self) -> bool:
        return self.status == STATUS_DEPRECATED

    @property
    def effective_status(self) -> str:
        """报告用状态：`analysis/compact.py` 已接线的列算 `produced`。

        `status` 字段保留**迁移时**的原始分类（derivable/missing_*），
        `effective_status` 再叠加 compact builder 的产出情况 —— 这样缺口报告
        不会永远停留在「37 列待接线」。
        """
        if self.name in COMPACT_DEFERRED:
            return STATUS_MISSING_PRODUCER
        if self.name in COMPACT_PRODUCED:
            return STATUS_PRODUCED
        return self.status


#: 旧 58 条目去重后的 55 列，**保持旧顺序**（首个重复项保留）。
#:
#: 去重规则：`MTMPOWER30` / `MTMPOWER30_MAJOR` / `MTMPOWER30_WEEKLY` 与
#: `MTM_NORM` / `MTM_NORM_MAJOR` / `MTM_NORM_WEEKLY` 是同一列，保留前者。
XGB_SCAN_FEATURES: Tuple[FeatureSpec, ...] = (
    FeatureSpec('CCI_NORM', 'CCI_NORM', STATUS_DERIVABLE,
                'talib.CCI（未接线）'),
    FeatureSpec('RSI', 'RSI', STATUS_DERIVABLE, 'talib.RSI（未接线）'),
    FeatureSpec(FIELD.MAXFACTOR, 'MAXFACTOR', STATUS_DERIVABLE,
                'OHLC 滚动区间最大因子',
                '2026-10-21 用户定：可重新计算'),
    FeatureSpec(FEATURES.ZEN_DASH_TIMING_LAG_MAJOR_REAL, 'zDLag',
                STATUS_PRODUCED, 'analysis/zen.py / review day',
                '2026-10-21 用户定：**使用新口径**（zDLagMajR），旧 zDLag 不追平'),
    FeatureSpec(FEATURES.ZEN_DASH_TIMING_LAG_MINOR_REAL, 'ZEN_D_LAG_MIN',
                STATUS_PRODUCED, 'review 60min/15min'),
    FeatureSpec(FEATURES.ZEN_PEAK_TIMING_LAG_MINOR_REAL, 'ZEN_PEAK_LAG_MIN',
                STATUS_PRODUCED, 'review 60min/15min'),
    FeatureSpec('ZW_COEF_SN', 'ZW_COEF_SN', STATUS_DEPRECATED,
                '', 'ZEN wavelet 序列：新树无 wavelet 生产者（用户定：废弃）'),
    FeatureSpec('ZW_COEF', 'ZW_COEF', STATUS_DEPRECATED,
                '', 'ZEN wavelet 系数：新树无 wavelet 生产者（用户定：废弃）'),
    FeatureSpec(FIELD.BOLL_LB_HMA5_TIMING_LAG_MAJOR,
                'BOLL_L_HMA5_LAG_MAJ', STATUS_DERIVABLE,
                'talib.BBANDS + TA_HMA + analysis/timing',
                '2026-10-21 用户定：可重新计算'),
    FeatureSpec(FIELD.MACD_ZERO_TIMING_LAG_MAJOR, 'MACD_O_LAG_MAJ',
                STATUS_DERIVABLE,
                'analysis/indicators.calc_macd_qa + analysis/timing',
                '2026-10-21 用户定：可重新计算'),
    FeatureSpec('MA90_CLR_LAG', 'MA90_CLR_LAG', STATUS_DERIVABLE,
                'talib.MA + analysis/timing',
                '2026-10-21 用户定：可重新计算'),
    FeatureSpec('ZEN_PEAK_RET_GAR', 'ZEN_PEAK_RET_GAR',
                STATUS_DEPRECATED, '',
                '2026-10-21 用户定：放弃引入；用 zPLag + GAR/STAGE 替代'),
    FeatureSpec('ZEN_PEAK_RET_GAR_MAJOR', 'ZEN_PEAK_RET_GAR_MAJOR',
                STATUS_DEPRECATED, '', '同上（major）'),
    FeatureSpec('MA90_CLR_RET_GAR', 'MA90_CLR_RET_GAR',
                STATUS_DEPRECATED, '',
                '2026-10-21 用户定：放弃引入；用 MA90_CLR/MA90_CLR_LAG + garP 替代'),
    FeatureSpec('MA90_CLR', 'MA90_CLR', STATUS_DERIVABLE,
                '', 'close / MA90 − 1；talib.MA 可算'),
    FeatureSpec('DRDN_RATIO', 'DRDN_RATIO', STATUS_DERIVABLE,
                '', '近 N 日回撤比例；OHLC 可算'),
    FeatureSpec(FIELD.MAPOWER30, 'MPWR3', STATUS_DERIVABLE,
                'analysis/quadrant_jit.mapower_np_vXIIs',
                '2026-10-21 用户定：可重新计算'),
    FeatureSpec(FIELD.MAPOWER30_TIMING_LAG, 'MPWR3_LAG',
                STATUS_DERIVABLE, 'analysis/timing.calc_event_timing_lag',
                '2026-10-21 用户定：可重新计算'),
    FeatureSpec(FIELD.MAPOWER30_MAJOR, 'MPWR3_MAJ', STATUS_DERIVABLE,
                'analysis/quadrant_jit.mapower_np_vXIIs',
                'major 口径（周线），2026-10-21 用户定：可重新计算'),
    FeatureSpec('MPWR3_LAG_MAJ', 'MPWR3_LAG_MAJ', STATUS_DERIVABLE,
                'analysis/timing.calc_event_timing_lag',
                '2026-10-21 用户定：可重新计算；needs constant'),
    FeatureSpec(FIELD.POLYNOMIAL9_TIMING_LAG, 'poly9_LAG',
                STATUS_DERIVABLE, 'analysis/poly9.py + analysis/timing',
                '2026-10-21 用户定：可重新计算'),
    FeatureSpec('poly9_LAG_MAJ', 'poly9_LAG_MAJ', STATUS_DERIVABLE,
                'analysis/poly9.py + analysis/timing',
                '2026-10-21 用户定：可重新计算（needs constant）'),
    FeatureSpec(FIELD.MTM_NORM, 'MTM_NORM', STATUS_DERIVABLE,
                'MTM / rolling MA',
                '旧 MTMPOWER30；与后面的 MTM_NORM 是同一列（已去重）'),
    FeatureSpec(FIELD.MTM_NORM_MAJOR, 'MTM_NORM_MAJ', STATUS_DERIVABLE,
                'MTM / rolling MA（major）', '旧 MTMPOWER30_MAJOR（已去重）'),
    FeatureSpec('MTM_NORM_WEK', 'MTM_NORM_WEK', STATUS_DERIVABLE,
                'MTM / rolling MA（weekly）',
                '旧 MTMPOWER30_WEEKLY（已去重）；needs constant'),
    FeatureSpec('MPWR3_MED', 'MPWR3_MED', STATUS_DERIVABLE,
                'MAPOWER30 滚动中位数', 'needs constant'),
    FeatureSpec('MPWR3_MAJ_MED', 'MPWR3_MAJ_MED', STATUS_DERIVABLE,
                'MAPOWER30 major 滚动中位数', 'needs constant'),
    FeatureSpec(FIELD.STOCK_SCORE_M15, 'Score_M15', STATUS_DERIVABLE,
                '千股千评/牛股诊股（metadata/moneyflow）',
                '2026-10-21 用户定：可派生，保留'),
    FeatureSpec(FIELD.STOCK_SCORE_M15_NORM, 'SSM15_NORM',
                STATUS_DERIVABLE, '由 Score_M15 归一化',
                '2026-10-21 用户定：可派生，保留'),
    FeatureSpec(FIELD.STOCK_SCORE_M15_NORM_TIMING_LAG, 'SSM15_NORM_LAG',
                STATUS_DERIVABLE, 'analysis/timing.calc_event_timing_lag',
                '2026-10-21 用户定：可派生，保留'),
    FeatureSpec('VOL_NORM', 'VOL_NORM', STATUS_DERIVABLE,
                '', '成交量 / 滚动均值；OHLCV 可算'),
    FeatureSpec('VOL_NORM_LAG', 'VOL_NORM_LAG', STATUS_DERIVABLE,
                '', 'VOL_NORM 的 timing lag'),
    FeatureSpec(FIELD.POLYNOMIAL9_NORM, 'poly9pwr3',
                STATUS_DERIVABLE, 'analysis/poly9.py',
                '新常量名 POLYNOMIAL9_NORM，值 `poly9pwr3`（与旧值相同）'),
    FeatureSpec(FIELD.POLYNOMIAL9_NORM_MAJOR, 'poly9pwr3_MAJ',
                STATUS_DERIVABLE, 'analysis/poly9.py',
                '新值 `poly9pwr3Maj`（旧值带下划线，注意映射）'),
    FeatureSpec('PeakLowLag', 'PeakLowLag', STATUS_DERIVABLE,
                'analysis/peak.py',
                '2026-10-21 用户定：可重新计算（needs constant）'),
    FeatureSpec('DESINTY_CHAIN_LAG', 'DESINTY_CHAIN_LAG', STATUS_DEPRECATED,
                '', 'density chain：旧 calc_desinty_chain_fractal 未搬（用户定：废弃）'),
    FeatureSpec(FIELD.MONEYFLOW_IN, 'MONEYFLOW_IN', STATUS_DEPRECATED,
                '', '2026-10-21 用户定：资金流数据源放弃；用 VOL_NORM/scoreM15/RSI 替代'),
    FeatureSpec('MONEYFLOW_IN_BF', 'MONEYFLOW_IN_BF', STATUS_DEPRECATED,
                '', '同上（before）；needs constant'),
    FeatureSpec(FIELD.MONEYFLOW_VOLUME, 'moneyflow_volume',
                STATUS_DEPRECATED, '', '同上'),
    FeatureSpec(FIELD.MONEYFLOW_PERCENT, 'MONEYFLOW_PCT',
                STATUS_DEPRECATED, '', '同上'),
    FeatureSpec(FIELD.MAGIC_NINE_TURNS, 'm9t', STATUS_PRODUCED,
                'review 60min coldstart / day m9tMajR',
                '2026-10-21 用户定：**使用新口径**，旧 m9t 不追平'),
    FeatureSpec(FIELD.MAGIC_NINE_TURNS_BEFORE, 'm9tBf', STATUS_DERIVABLE,
                '由 m9t 经 timing lag 派生（待接线）',
                '2026-10-21 用户定：可派生，保留'),
    FeatureSpec(FIELD.MAGIC_NINE_TURNS_SX_BEFORE, 'm9tSxBf',
                STATUS_DERIVABLE, '由 m9t 经死叉 timing lag 派生（待接线）',
                '2026-10-21 用户定：可派生，保留'),
    FeatureSpec(FIELD.MAGIC_NINE_TURNS_MAJOR, 'm9tMaj', STATUS_PRODUCED,
                'review 60min coldstart / day m9tMajR',
                '2026-10-21 用户定：**使用新口径**，旧 m9tMaj 不追平'),
    FeatureSpec('m9tBfMaj', 'm9tBfMaj', STATUS_DERIVABLE,
                '由 m9tMaj 经 timing lag 派生（待接线）',
                '2026-10-21 用户定：可派生，保留'),
    FeatureSpec('m9tSxBfMaj', 'm9tSxBfMaj', STATUS_DERIVABLE,
                '由 m9tMaj 经死叉 timing lag 派生（待接线）',
                '2026-10-21 用户定：可派生，保留'),
    FeatureSpec('MASS_TRD_JX_BF', 'MASS_TRD_JX_BF', STATUS_DEPRECATED, '',
                '2026-10-21 用户定：放弃引入；用 MPWR3/MTM_NORM/m9t 替代'),
    FeatureSpec('MASS_SEMI_TRD_JX_BF', 'MASS_SEMI_TRD_JX_BF',
                STATUS_DEPRECATED, '',
                '同上（semi）；用 mpwr3Lag/m9tBf 替代'),
    FeatureSpec('MPWR3_MTR', 'MPWR3_MTR', STATUS_DEPRECATED, '',
                'MAPOWER30 metric：旧 calc_mpwr3_poly9_massive_trend_vXI 未搬（用户定：废弃）'),
    FeatureSpec('MTM_FLOW_LAG', 'MTM_FLOW_LAG', STATUS_DEPRECATED, '',
                '2026-10-21 用户定：资金流派生，一并放弃；用 VOL_NORM_LAG/MTM_NORM 替代'),
    FeatureSpec('BBW_diff_1T', 'BBW_diff_1T', STATUS_DERIVABLE,
                '', 'BOLL 带宽一阶差分；talib.BBANDS 可算'),
    FeatureSpec('BOLL_WIDTH', 'BOLL_WIDTH', STATUS_DERIVABLE,
                '', 'BOLL 带宽；talib.BBANDS 可算'),
    FeatureSpec('MTMMA_NORM', 'MTMMA_NORM', STATUS_DERIVABLE,
                'MTM + rolling MA', '2026-10-21 用户定：可重新计算'),
    FeatureSpec('MTMMA_NORM_WEK', 'MTMMA_NORM_WEK', STATUS_DERIVABLE,
                'MTM + rolling MA（weekly）', '同上'),
    FeatureSpec('MTMMA_NORM_MAJ', 'MTMMA_NORM_MAJ', STATUS_DERIVABLE,
                'MTM + rolling MA（major）', '同上'),
)

#: 旧存储名 → canonical 名（含 3 个重复别名的隐式合并）。
LEGACY_TO_CANONICAL: Dict[str, str] = {s.legacy: s.name for s in XGB_SCAN_FEATURES}

#: 用户 2026-10-21 定：上游链未就绪 / 数据源放弃，本轮废弃。
DEPRECATED_FEATURES = frozenset(s.name for s in XGB_SCAN_FEATURES if s.deprecated)

#: 废弃列的**替代特征**（全部来自已产出集合；只做记录，不自动改列）。
REPLACEMENTS = {
    # MASSIVE 趋势：用市场力度/动量/九转替代
    'MASS_TRD_JX_BF': ('MPWR3', 'mtmNorm', 'm9t'),
    'MASS_SEMI_TRD_JX_BF': ('mpwr3Lag', 'm9tBf'),
    # ZEN peak returns × GAR：用 zen peak lag + Poly9 替代（GAR/STAGE 在 review 契约里）
    'ZEN_PEAK_RET_GAR': ('zPLagMinR', 'poly9Lag'),
    'ZEN_PEAK_RET_GAR_MAJOR': ('zPLagMinR', 'poly9Lag'),
    # MA90 clearance returns × GAR：用 clearance 本身 + lag + 回撤替代
    'MA90_CLR_RET_GAR': ('MA90_CLR', 'MA90_CLR_LAG', 'DRDN_RATIO'),
    # 资金流：用成交量归一 / 打分 / 强弱指标替代
    'moneyflowIn': ('VOL_NORM', 'scoreM15', 'RSI'),
    'MONEYFLOW_IN_BF': ('VOL_NORM', 'ssm15Norm', 'RSI'),
    'moneyflowVolume': ('VOL_NORM', 'VOL_NORM_LAG'),
    'moneyflowPct': ('VOL_NORM_LAG', 'mtmNorm'),
    'MTM_FLOW_LAG': ('VOL_NORM_LAG', 'mtmNorm'),
    # 其它废弃项
    'ZW_COEF_SN': ('poly9Lag', 'poly9pwr3'),
    'ZW_COEF': ('poly9Lag', 'poly9pwr3'),
    'DESINTY_CHAIN_LAG': ('DRDN_RATIO', 'MPWR3'),
    'MPWR3_MTR': ('MPWR3', 'MPWR3_MED'),
}

#: `analysis/compact.py` 仍**未实现**的列（返回 NaN）→ 报告为 missing_producer。
#: 2026-10-21 用户放弃 `MA90_CLR_RET_GAR` 后为空。
COMPACT_DEFERRED = frozenset()

#: `analysis/compact.py::calc_compact_features` 已产出的 derivable 列。
COMPACT_PRODUCED = frozenset(
    s.name for s in XGB_SCAN_FEATURES
    if s.status == STATUS_DERIVABLE and s.name not in COMPACT_DEFERRED)

_MODEL_ALIASES = {
    'xgb_scan': 'xgb_scan',
    'mark_uprising': 'xgb_scan',
    'scan': 'xgb_scan',
}


def _normalise_model(model: str) -> str:
    for prefix, canonical in _MODEL_ALIASES.items():
        if model == prefix or model.startswith(prefix):
            return canonical
    raise KeyError('未知的 xgboost 特征集：{!r}（当前只有 xgb_scan）'.format(model))


def feature_specs(model: str = 'xgb_scan',
                  include_deprecated: bool = False) -> Tuple[FeatureSpec, ...]:
    """返回该模型的 :class:`FeatureSpec` 元组。

    >>> len(feature_specs()), len(feature_specs(include_deprecated=True))
    (41, 55)
    """
    _normalise_model(model)
    return tuple(s for s in XGB_SCAN_FEATURES
                 if include_deprecated or not s.deprecated)


def feature_names(model: str = 'xgb_scan',
                  include_deprecated: bool = False) -> Tuple[str, ...]:
    """训练/预测要取的列（默认 41 列，不含废弃的 14 列）。

    >>> len(feature_names()), 'ZW_COEF' in feature_names(include_deprecated=True)
    (41, True)
    """
    return tuple(s.name for s in feature_specs(model, include_deprecated))


def missing_columns(features: Iterable[str], model: str = 'xgb_scan',
                    include_deprecated: bool = False) -> Tuple[str, ...]:
    """`features`（列名集合）里缺的契约列，按契约顺序返回。"""
    have = set(features)
    return tuple(c for c in feature_names(model, include_deprecated) if c not in have)


def validate_features(features, model: str = 'xgb_scan',
                      include_deprecated: bool = False) -> None:
    """训练/预测前的硬闸：缺列直接 `ValueError`，**绝不**静默补 NaN。

    >>> validate_features([])  # doctest: +ELLIPSIS
    Traceback (most recent call last):
    ...
    ValueError: xgb_scan 缺 41 列：...
    """
    columns = features.columns if hasattr(features, 'columns') else features
    missing = missing_columns(columns, model, include_deprecated)
    if missing:
        raise ValueError('{} 缺 {} 列：{}'.format(
            _normalise_model(model), len(missing), ', '.join(missing)))


def feature_status_report(model: str = 'xgb_scan') -> Dict[str, Tuple[str, ...]]:
    """按状态分组的缺口报告（含 deprecated）。

    >>> report = feature_status_report()
    >>> len(report['deprecated'])
    14
    >>> sum(len(v) for v in report.values())
    55
    """
    _normalise_model(model)
    out = {status: [] for status in STATUSES}
    for spec in XGB_SCAN_FEATURES:
        out[spec.effective_status].append(spec.name)
    return {k: tuple(v) for k, v in out.items()}


__all__ = [
    'FeatureSpec', 'XGB_SCAN_FEATURES', 'LEGACY_TO_CANONICAL',
    'DEPRECATED_FEATURES', 'REPLACEMENTS', 'STATUSES',
    'STATUS_PRODUCED', 'STATUS_AMBIGUOUS', 'STATUS_DERIVABLE',
    'STATUS_MISSING_PRODUCER', 'STATUS_MISSING_SOURCE', 'STATUS_DEPRECATED',
    'feature_specs', 'feature_names', 'missing_columns', 'validate_features',
    'feature_status_report',
]
