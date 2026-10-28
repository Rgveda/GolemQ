# coding:utf-8
"""CompactPipeline —— 从 8.3 读 60min + day，算 `calc_compact_features`。

⚠️ **只读，不落库**（用户 2026-10-21 定：先不落库）。输出只用于统计/抽样/后续
pipeline 接入；没有任何 `save_*` 调用。
"""
from __future__ import annotations

from typing import Iterable, Optional

import pandas as pd

from GolemQ.analysis.compact import COMPACT_COLUMNS, calc_compact_features
from GolemQ.markets.StockCN.date_utils import GQ_util_get_last_day
from GolemQ.models.alias import PFL
from GolemQ.pipeline.base import BasePipeline, CodeResult

PIPELINE_NAME = 'compact'
DEFAULT_COMPACT_SCOPE = 'fast'

#: 取数暖机：日线 4000 天（与 review 一致）、60min 800 天。
DAY_LOOKBACK_DAYS = 4000
MIN_LOOKBACK_DAYS = 800
#: 单码超时：compact 含 Poly9/MAPOWER/rank，60min 800 天约数分钟。
COMPACT_CODE_TIMEOUT = 30 * 60


class CompactPipeline(BasePipeline):
    """逐码 compact 管线（**只读**）。固定 60min + day + rank，无开关。"""

    def __init__(self, **kwargs):
        kwargs.setdefault('timeout', COMPACT_CODE_TIMEOUT)
        super().__init__(name=PIPELINE_NAME, **kwargs)

    @staticmethod
    def market_type_of(code: str):
        from GolemQ.markets.StockCN.symbol import is_stock_cn
        return is_stock_cn(str(code))[1]

    @staticmethod
    def _fetch_frames(code: str, as_of: pd.Timestamp):
        """读 60min（必需）+ day（可选），一律走 Active Market。

        :returns: ``(m60, day, display_name)`` —— 无 60min 数据时 ``day`` 为 None。
        """
        from GolemQ import get_active_market
        market = get_active_market()
        end = str(as_of.date())
        min_start = str((as_of - pd.Timedelta(days=MIN_LOOKBACK_DAYS)).date())
        day_start = str((as_of - pd.Timedelta(days=DAY_LOOKBACK_DAYS)).date())

        res60, display_name = market.get_kline_price_min(
            code, start=min_start, end=end + ' 23:59:59',
            frequency='60min', realtime=False, verbose=False)
        m60 = None if res60 is None else getattr(res60, 'data', None)
        if m60 is None or len(m60) == 0:
            return m60, None, display_name

        resd, _ = market.get_kline_price_v3(
            code, start=day_start, end=end, realtime=False, verbose=False)
        day = None if resd is None else getattr(resd, 'data', None)
        if day is not None and len(day) == 0:
            day = None
        return m60, day, display_name

    @staticmethod
    def calc_features(code: str, offset: str = '', verbose: bool = False,
                      ret_meta: bool = False):
        """单码 compact 特征的**研究入口**（Jupyter / 脚本）。

        范式同老树 cookbook 的
        ``calc_stock_poolcoef_analysis(symbol, offset=..., ret_meta=True)`` ——
        取数与计算一次完成，返回整段历史，由调用方自取 ``.tail(n)``：

            features = CompactPipeline.calc_features('600519')
            ohlc, features, display_name = CompactPipeline.calc_features(
                '600519', offset='2026-10-20', ret_meta=True)

        ⚠️ **只读**：不落库、不写 watermark、不做实时合并。默认取最近交易日。

        :param code: 股票代码。
        :param offset: 截止日期 ``'YYYY-MM-DD'``；空串 ⇒ 最近交易日。
        :param verbose: 打印计算明细。
        :param ret_meta: True ⇒ 返回 ``(ohlc, features, display_name)``。
        :returns: 特征 DataFrame；``ret_meta=True`` 时为三元组。
            无 60min 数据时返回 ``None`` / ``(None, None, None)``。
        """
        as_of = (pd.Timestamp(offset) if offset
                 else pd.Timestamp(GQ_util_get_last_day()))
        m60, day, display_name = CompactPipeline._fetch_frames(code, as_of)
        if m60 is None or len(m60) == 0:
            return (None, None, None) if ret_meta else None
        features = calc_compact_features(m60, ohlc_day=day, include_rank=True,
                                         verbose=verbose)
        if ret_meta:
            return m60, features, display_name
        return features

    def run_code(self, code: str, as_of=None, **kwargs) -> CodeResult:
        market_type = self.market_type_of(code)
        if as_of is None:
            as_of = pd.Timestamp(GQ_util_get_last_day())
        else:
            as_of = pd.Timestamp(as_of)

        m60, day, _ = self._fetch_frames(code, as_of)
        if m60 is None or len(m60) == 0:
            return CodeResult(code=code, status='no_data',
                              error='no 60min data', market_type=market_type)
        try:
            out = calc_compact_features(m60, ohlc_day=day,
                                       include_rank=True,
                                       verbose=self.verbose)
        except Exception as exc:            # noqa: BLE001 交给 CodeResult 分类
            return CodeResult(code=code, status='error',
                              error='{}: {}'.format(type(exc).__name__, exc),
                              market_type=market_type)

        rank_rows = 0
        if PFL.STOCK_PORTFOLIO_RANK in out.columns:
            rank_rows = int(out[PFL.STOCK_PORTFOLIO_RANK].notna().sum())
        last = out.index.get_level_values(0)[-1] if len(out) else None
        return CodeResult(
            code=code, status='ok', market_type=market_type,
            result={'rows': len(out), 'cols': len(out.columns),
                    'rank_rows': rank_rows, 'last': last,
                    'missing': [c for c in COMPACT_COLUMNS if c not in out.columns]})


def run_compact(eval_range: str = DEFAULT_COMPACT_SCOPE,
                codes: Optional[Iterable[str]] = None,
                pct_of_cores: float = 0.32,
                offset: str = '',
                verbose: bool = False, **kwargs):
    """解析 scope → 逐码算 compact → 返回 ``(results, stats)``。**不写库。**"""
    from GolemQ.analysis.cv.scopes import perpar_symbol_range
    from GolemQ import get_active_market

    if codes:
        codelist = [str(c).strip() for c in codes if str(c).strip()]
    else:
        got = perpar_symbol_range(eval_range,
                                  dataset=get_active_market().scan_dataset(),
                                  verbose=verbose)
        if got is False:
            raise ValueError('无法解析 eval_range={!r}'.format(eval_range))
        codelist = list(got)
    cpu = int(pct_of_cores) if pct_of_cores else 0
    if cpu > 1:
        import multiprocessing
        cpu = max(1, int((cpu / 100.0) * multiprocessing.cpu_count()))
    elif 0 < cpu <= 1:
        import multiprocessing
        cpu = max(1, int(cpu * multiprocessing.cpu_count()))

    pipe = CompactPipeline(verbose=verbose)
    return pipe.run_codes(codelist, pool_size=cpu, verbose=verbose,
                          as_of=(offset or None))
