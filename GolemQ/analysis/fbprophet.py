# coding:utf-8
"""Prophet 价格区间（旧树 `FBPROPHET_*` 四列）—— **只搬算术，落库存比值**。

搬自 `GolemQ_old/fractal/v9.py:4394 calc_fbprophet_band` 的**核心那一段**，
以及 `GolemQ_old/features/base.py:3296-3297`（把日线那组抄进 `_MAJOR` 的两行）。

搬了什么 / 没搬什么
==================
搬：Prophet 拟合（`changepoint_range=0.96`）→ `yhat_lower` / `yhat_upper`
对齐回**原 bar 索引**；`_MAJOR` = 同一套算法在**日线**上算一遍。

**没搬**（都是打分链的输入，本模块不消费，见 D53）：
`CHANGE_POINT*` / `FBPROPHET_PUNCH_CANDIDATE` / `LRC_CLEARANCE`
（依赖 `lineareg_cross_v2_func`，新树没有）/ `POLY9_CLEARANCE`，
以及 `annual > 365` 时那 3 次「留出末 i 根」的额外拟合（只为 PUNCH_CANDIDATE）。

⚠️ **`yhat_lower` / `yhat_upper` 是抽样出来的**（2026-10-18 实测）
==============================================================
`mcmc_samples=0` 时 Prophet 用 `np.random` 从后验近似里抽 `uncertainty_samples`（默认
1000）份再取分位数 ⇒ **同一个模型、同一份数据，连跑两次结果都不同**（实测同一输入
两次 `predict`，`yhat_lower` 最大差 **0.22**，约 2%）。`yhat` 本身是确定的。

所以本模块在 `predict` 前**固定种子**（`UNCERTAINTY_SEED`），跑完还原全局 RNG。
不固定的话，「重算」每次都会给同一根 bar 写出不同的区间，"重跑幂等"就是空话。
固定后与老树（同样 seed 后调用）**逐位相同**（老树落库是 float32，我们算 float64，
比对时转 float32）——见 `test_fbprophet.TestMatchesOldTree`。

⚠️ 落库存**比值**，不存绝对价（`DECISIONS.md` **D53**）
====================================================
`yhat_lower` / `yhat_upper` 是绝对价位，一落库就跟着复权过期（与 `GAR_P` 同一课，
D49）。这里存 `yhat / close`（**同一个 bar**）—— `1.0` = 与 close 持平，
还原价 = `比值 × 当日 close`。

**比值带来的收益是"历史不用重写"**：除权后 `close` 整段按 k 缩放，
`比值 × 新 close` 自动跟着缩放，旧行**不用动**。

⚠️ **但它不等于"重算也一模一样"**：Prophet 的先验是**绝对量纲**
（`changepoint_prior_scale` 作用在 y 的增量上），所以它**不是尺度等变**的。
2026-10-18 实测（600519 日线，整窗 ×0.5 再除以 0.5 比较）：

========================  ======================
量                         最大相对差
========================  ======================
比值                        **0.82%**
绝对价                     **0.82%**（同量）
========================  ======================

⇒ 差的是「**除权后要不要重算**」这件事，**不是**存比值的缺陷：
不重算 ⇒ 完全跟随；重算 ⇒ 与不重算差 ~0.8%。别在文档里写成"数学不变"。
"""
from __future__ import annotations

import logging
from contextlib import contextmanager

import numpy as np
import pandas as pd

from GolemQ.core.constants import AKA, FIELD

__all__ = [
    'prophet_band',
    'fbp_bands_to_ratio', 'fbp_bands_from_ratio',
    'calc_fbprophet_bands',
]

#: 预测期数 —— 旧树三种 `annual` 分支里两种是 84、只有 `annual == 365` 是 21
#: （`v9.py:4446-4451`）。这里默认按最长的那种（日线/小时线都用 84）。
DEFAULT_PERIODS = 84

#: 变点范围 —— 旧树原值，别改（改了对历史带的影响是全局的）。
DEFAULT_CHANGEPOINT_RANGE = 0.96

#: 不确定区间的**抽样种子**。
#:
#: ⚠️ **`yhat_lower` / `yhat_upper` 是抽出来的，不是解析解**：`mcmc_samples=0` 时
#: Prophet 用 `np.random` 从后验近似里抽 `uncertainty_samples`（默认 1000）份，
#: 再取分位数。**同一个模型、同一份数据，连跑两次结果都不同** ——
#: 2026-10-18 实测：同一输入两次 `predict`，`yhat_lower` 最大差 **0.22**（约 2%）。
#:
#: 所以这里**固定种子**，否则每次重算都会把同一根 bar 的区间写成不同的数，
#: "重跑幂等"就成了一句空话。固定后实测**逐位相同**。
#: （`yhat` 本身是确定的，只有区间受这个影响。）
UNCERTAINTY_SEED = 20260101


@contextmanager
def _quiet_prophet_logging():
    """临时压掉 `cmdstanpy` / `prophet` 的 INFO，并避免 `Logging error`。

    为什么需要两步
    ==============
    * cmdstanpy 通过 Python `logging` 打 `Chain [1] start processing` ——
      `suppress_stdout_stderr` 只重定向 **文件描述符**，压不住 `logging`；
    * 它的 handler 可能在 fd 重定向期间被创建/关闭，`__exit__` 关掉 devnull 后
      再写日志就会打 `--- Logging error ---`。临时换成 `NullHandler` 一并解决。

    ⚠️ 只在 fit 期间生效，退出后**原样还原**（level / handlers / propagate）。
    """
    saved = []
    for name in ('cmdstanpy', 'prophet'):
        logger = logging.getLogger(name)
        saved.append((logger, logger.level, logger.handlers[:],
                      logger.propagate))
        logger.setLevel(logging.ERROR)
        logger.handlers = [logging.NullHandler()]
        logger.propagate = False
    try:
        yield
    finally:
        for logger, level, handlers, propagate in saved:
            logger.setLevel(level)
            logger.handlers = handlers
            logger.propagate = propagate


def _seeded_predict(model, future) -> pd.DataFrame:
    """在**固定种子**下 `predict`，跑完**还原**全局 RNG（不污染调用方）。"""
    state = np.random.get_state()
    try:
        np.random.seed(UNCERTAINTY_SEED)
        return model.predict(future)
    finally:
        np.random.set_state(state)


def _one_code(ohlc) -> str:
    """取（并校验）**单只**标的 —— Prophet 是逐只拟合的。"""
    codes = pd.Index(ohlc.index.get_level_values(1)).unique()
    if len(codes) != 1:
        raise ValueError('prophet_band 只接受单只标的，收到 {} 只: {}'
                         .format(len(codes), list(codes[:5])))
    return str(codes[0])


def prophet_band(ohlc, periods: int = DEFAULT_PERIODS,
                 changepoint_range: float = DEFAULT_CHANGEPOINT_RANGE) -> pd.DataFrame:
    """单只标的 → **绝对价**区间（`yhat` / `yhat_lower` / `yhat_upper`）。

    :param ohlc: 索引 `(ts, code)` 两层、**单只**标的；必须有 `close` 列。
    :returns: 索引与 `ohlc.index` **逐行相同**；只含历史那段（未来段丢弃）。

    ⚠️ `periods` 传的是"往后预测多少步"，**不影响**历史段的数值；
       保留它是为了与旧树同参（旧树 `make_future_dataframe(periods=84)`）。
    """
    from prophet import Prophet                      # 惰性：只有本模块要它

    from GolemQ.core.presentation import suppress_stdout_stderr

    _one_code(ohlc)
    ds = pd.to_datetime(ohlc.index.get_level_values(0))
    if ds.has_duplicates:
        raise ValueError('prophet_band 的时间轴有重复值（{} 个）'.format(int(ds.duplicated().sum())))
    y = pd.to_numeric(ohlc[AKA.CLOSE], errors='coerce').astype(float).to_numpy()

    with _quiet_prophet_logging(), suppress_stdout_stderr():
        model = Prophet(changepoint_range=changepoint_range)
        model.fit(pd.DataFrame({'ds': ds, 'y': y}))
        # ⚠️ 必须走 `_seeded_predict`：区间是**抽样**出来的（见 `UNCERTAINTY_SEED`）
        forecast = _seeded_predict(model, model.make_future_dataframe(periods=periods))

    out = forecast.set_index('ds').reindex(ds)[['yhat', 'yhat_lower', 'yhat_upper']]
    out.index = ohlc.index
    return out


def fbp_bands_to_ratio(bands: pd.DataFrame, close) -> pd.DataFrame:
    """**绝对价**带 → 相对**同一 bar** `close` 的**比值**带。

    >>> import pandas as pd
    >>> idx = pd.MultiIndex.from_tuples([(pd.Timestamp('2026-01-05'), '600519')],
    ...                                 names=['ts', 'code'])
    >>> b = pd.DataFrame({'yhat_lower': [9.0], 'yhat_upper': [11.0]}, index=idx)
    >>> r = fbp_bands_to_ratio(b, pd.Series([10.0], index=idx))
    >>> float(r['yhat_lower'].iloc[0]), float(r['yhat_upper'].iloc[0])
    (0.9, 1.1)
    """
    if not isinstance(close, pd.Series):
        close = pd.Series(close, index=bands.index)
    return bands.div(close.astype(float), axis=0)


def fbp_bands_from_ratio(ratios: pd.DataFrame, close) -> pd.DataFrame:
    """比值带 → 绝对价带（`比值 × 当日 close`）。与 :func:`fbp_bands_to_ratio` 互逆。

    >>> import pandas as pd
    >>> idx = pd.MultiIndex.from_tuples([(pd.Timestamp('2026-01-05'), '600519')],
    ...                                 names=['ts', 'code'])
    >>> r = pd.DataFrame({'yhat_lower': [0.9], 'yhat_upper': [1.1]}, index=idx)
    >>> b = fbp_bands_from_ratio(r, pd.Series([10.0], index=idx))
    >>> float(b['yhat_lower'].iloc[0]), float(b['yhat_upper'].iloc[0])
    (9.0, 11.0)
    """
    if not isinstance(close, pd.Series):
        close = pd.Series(close, index=ratios.index)
    return ratios.mul(close.astype(float), axis=0)


def _reindex_by_date(series: pd.Series, index) -> pd.Series:
    """把**日线**序列按「日期」对齐到任意频率的索引上（`_MAJOR` 就是这么用的）。"""
    dates = pd.Index(pd.to_datetime(index.get_level_values(0))).normalize()
    src = series.copy()
    src.index = pd.Index(pd.to_datetime(series.index.get_level_values(0))).normalize()
    src = src[~src.index.duplicated(keep='last')]
    return pd.Series(src.reindex(dates).to_numpy(), index=index)


def calc_fbprophet_bands(ohlc, features: pd.DataFrame = None, major_ohlc=None,
                         periods: int = DEFAULT_PERIODS,
                         changepoint_range: float = DEFAULT_CHANGEPOINT_RANGE
                         ) -> pd.DataFrame:
    """在 `features` 上写 **4 列比值**：当前频率两列 + 日线两列。

    ==========================  ==========================================
    列（`FIELD.`）               来源
    ==========================  ==========================================
    `FBPROPHET_LB` / `_UB`      `ohlc`（当前频率）的 `yhat_lower/upper ÷ close`
    `FBPROPHET_LB_MAJOR` / `_UB_MAJOR`  `major_ohlc`（日线）同上，按**日期**对齐
    ==========================  ==========================================

    :param major_ohlc: 日线帧。``None`` ⇒ 认为 `ohlc` 本身就是日线，
        `_MAJOR` 直接沿用 `LB`/`UB`（旧树 `features/base.py:3296` 就是这么抄的）。
    """
    bands = prophet_band(ohlc, periods=periods, changepoint_range=changepoint_range)
    close = pd.to_numeric(ohlc[AKA.CLOSE], errors='coerce').astype(float)
    ratios = fbp_bands_to_ratio(bands, close)

    feat = features if features is not None else pd.DataFrame(index=ohlc.index)
    feat[FIELD.FBPROPHET_LB] = ratios['yhat_lower'].reindex(feat.index)
    feat[FIELD.FBPROPHET_UB] = ratios['yhat_upper'].reindex(feat.index)

    if major_ohlc is None:
        feat[FIELD.FBPROPHET_LB_MAJOR] = feat[FIELD.FBPROPHET_LB]
        feat[FIELD.FBPROPHET_UB_MAJOR] = feat[FIELD.FBPROPHET_UB]
    else:
        mbands = prophet_band(major_ohlc, periods=periods,
                              changepoint_range=changepoint_range)
        mclose = pd.to_numeric(major_ohlc[AKA.CLOSE], errors='coerce').astype(float)
        mratios = fbp_bands_to_ratio(mbands, mclose)
        feat[FIELD.FBPROPHET_LB_MAJOR] = _reindex_by_date(mratios['yhat_lower'], feat.index)
        feat[FIELD.FBPROPHET_UB_MAJOR] = _reindex_by_date(mratios['yhat_upper'], feat.index)
    return feat
