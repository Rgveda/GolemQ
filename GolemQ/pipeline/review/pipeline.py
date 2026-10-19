# coding: utf-8
"""ReviewPipeline —— 逐码、逐快照、因果的 review 批处理管线。

职责边界
========
* **取数**：一律经市场实例（`get_active_market()`），不直接碰 Mongo；
* **计算**：`FeatureStep`（`pipeline/steps.py`），runtime/daily 两档共用；
* **落库**：经 `markets/StockCN/reality_save.save_reality_feat`（唯一写口）；
* **水位**：`pipeline_watermarks` 侧表（`markets/StockCN/reality_save` 内）；
* **runtime**：`runtime.py::compute_runtime` 复用同一套 step，但**零写**。

⚠️ 逐码分类：ETF/指数/基金必须逐码 `is_stock_cn`，禁止给混合清单传单一 market_type。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

import pandas as pd

from GolemQ.core.constants import MARKET_TYPE
from GolemQ.pipeline.base import BasePipeline, CodeResult
from GolemQ.pipeline.steps import (FeatureStep, ReviewContext, StepRunner,
                                   TIER_DAILY, TIER_RUNTIME, target_times)
from GolemQ.pipeline.review.contract import FAMILY_COLUMNS, LEGACY_RENAME

#: 三族；coldstart 列已合并进 60min，不再有独立族。
DEFAULT_FAMILIES = ('day', '60min', '15min')
PIPELINE_NAME = 'review'

#: 各族的读取暖机（**只影响取数窗口，不决定写哪些行**）。
DAY_LOOKBACK_DAYS = 4000
MIN_LOOKBACK_DAYS = 900
#: 固定锚点：batch 与 runtime 必须用同一个起点，否则递归指标（EMA/BS/KMeans/
#: Prophet）的值会随窗口漂移。数据不足时自然截断。
REVIEW_ANCHOR_DAY = '2016-01-01'
REVIEW_ANCHOR_15MIN = '2017-01-01'


@dataclass
class ReviewOptions:
    families: Sequence[str] = DEFAULT_FAMILIES
    force: bool = False
    dry_run: bool = False
    recheck_days: int = 0
    from_date: str = ''
    offset: str = ''
    verbose: bool = False


class ReviewPipeline(BasePipeline):
    """逐码 review 管线。"""

    def __init__(self, steps: Optional[Sequence[FeatureStep]] = None,
                 options: ReviewOptions = None, **kwargs):
        super().__init__(name=PIPELINE_NAME, **kwargs)
        self.steps = list(steps) if steps is not None else build_default_steps()
        self.options = options or ReviewOptions()
        self._runner = StepRunner()

    # ------------------------------------------------------------------ #
    # 取数 / 分类
    # ------------------------------------------------------------------ #
    @staticmethod
    def market_type_of(code: str):
        from GolemQ.markets.StockCN.symbol import is_stock_cn
        return is_stock_cn(str(code))[1]

    @staticmethod
    def _required_freqs(families) -> set:
        """按请求的族决定要取哪些频率 —— 只跑 day 时不拉 15min（否则每次白读 3 万根）。"""
        fam = set(families)
        req = set()
        if fam & {'day'}:
            req.add('day')
        if fam & {'60min'}:
            # 60min 里的 `hour_legacy`（STAGE/BST/M9T）要从日线采样。
            req |= {'60min', '15min', 'day'}
        if fam & {'15min'}:
            req.add('15min')
        return req

    def _fetch_frames(self, code: str, as_of: pd.Timestamp,
                      freqs=None) -> Dict[str, pd.DataFrame]:
        from GolemQ import get_active_market
        market = get_active_market()

        day_start = REVIEW_ANCHOR_DAY
        min_start = str((as_of - pd.Timedelta(days=MIN_LOOKBACK_DAYS)).date())
        frames: Dict[str, pd.DataFrame] = {}
        want = set(freqs) if freqs is not None else {'day', '60min', '15min'}

        if 'day' in want:
            day_res, _ = market.get_kline_price_v3(code, start=day_start,
                                                   end=str(as_of.date()),
                                                   realtime=False, verbose=False)
            if day_res is not None and len(getattr(day_res, 'data', ())) > 0:
                frames['day'] = day_res.data.loc[
                    pd.Index(pd.to_datetime(day_res.data.index.get_level_values(0))) <= as_of]

        for freq, key in (('60min', '60min'), ('15min', '15min')):
            if key not in want:
                continue
            freq_start = REVIEW_ANCHOR_15MIN if key == '15min' else min_start
            res, _ = market.get_kline_price_min(
                code, start=freq_start, end=str(as_of.date()) + ' 23:59:59',
                frequency=freq, realtime=False, verbose=False)
            data = None if res is None else getattr(res, 'data', None)
            if data is not None and len(data) > 0:
                frames[key] = data.loc[
                    pd.Index(pd.to_datetime(data.index.get_level_values(0))) <= as_of]
        return frames

    # ------------------------------------------------------------------ #
    # 目标行 / 水位
    # ------------------------------------------------------------------ #
    def _targets(self, ctx: ReviewContext, family: str,
                 db=None) -> pd.DatetimeIndex:
        src = ctx.frames.get(family)
        if src is None or len(src) == 0:
            return pd.DatetimeIndex([])
        last_written = None
        if not (self.options.force or self.options.from_date):
            from GolemQ.markets.StockCN.reality_save import get_watermark, reality_feat_name
            table = reality_feat_name(family, ctx.market_type)
            wm = get_watermark(PIPELINE_NAME, table, ctx.code, db=db)
            if wm is not None:
                last_written = wm.get('last_written_ts')
        if last_written is not None and self.options.recheck_days:
            last_written -= int(self.options.recheck_days) * 86400
        only_after = pd.Timestamp(last_written, unit='s') if last_written else None
        if self.options.from_date:
            only_after = pd.Timestamp(self.options.from_date)
        as_of = ctx.as_of
        if self.options.offset:
            as_of = min(as_of, pd.Timestamp(self.options.offset))
        return target_times(src, as_of, only_after=only_after)

    @staticmethod
    def _restrict_to_targets(frame: pd.DataFrame, targets) -> pd.DataFrame:
        """只保留目标时间戳的行 —— 防止把 `minor_signals` 的**全量 15min 帧**整表写库。"""
        if frame is None or len(frame) == 0 or targets is None or len(targets) == 0:
            return frame
        ts = pd.DatetimeIndex(pd.to_datetime(frame.index.get_level_values(0)))
        return frame.loc[ts.isin(pd.DatetimeIndex(pd.to_datetime(targets)))]

    def _write_family(self, ctx: ReviewContext, family: str,
                      frame: pd.DataFrame, db=None) -> Optional[dict]:
        cols = [c for c in FAMILY_COLUMNS.get(family, ()) if c in frame.columns]
        if not cols or len(frame) == 0:
            return None
        # 老 `reality_features` 列名 → 新列名
        frame = frame.rename(columns={k: v for k, v in LEGACY_RENAME.items()
                                      if k in frame.columns})
        payload = frame.loc[:, [c for c in cols if c in frame.columns]]
        if self.options.dry_run:
            return {'name': family, 'rows': len(payload), 'dry_run': True}
        from GolemQ.markets.StockCN.reality_save import (save_reality_feat,
                                                         set_watermark)
        stats = save_reality_feat(payload, family, ctx.market_type,
                                  verbose=self.options.verbose,
                                  write_nulls=True, db=db)
        if len(payload):
            ts = pd.DatetimeIndex(pd.to_datetime(payload.index.get_level_values(0)))
            set_watermark(PIPELINE_NAME, stats['name'], family, ctx.code,
                          last_written_ts=int(ts.max().timestamp()),
                          last_source_ts=int(ts.max().timestamp()),
                          rows=len(payload), db=db)
        return stats

    # ------------------------------------------------------------------ #
    # 单码
    # ------------------------------------------------------------------ #
    def run_code(self, code: str, **kwargs) -> CodeResult:
        as_of = pd.Timestamp(kwargs.get('as_of') or self.options.offset or
                             pd.Timestamp.now())
        market_type = self.market_type_of(code)
        families = tuple(self.options.families)
        # 「库名不外泄」：pipeline 不拼库名，只从 **Active Market** 取实例，
        # 再作为参数传给写口/水位函数。
        from GolemQ import get_active_market
        db = get_active_market().GOLEMQ_STOCK_CN
        frames = self._fetch_frames(code, as_of,
                                    freqs=self._required_freqs(families))
        if not frames:
            return CodeResult(code=code, status='no_data', market_type=market_type)

        ctx = ReviewContext(code=str(code), market_type=market_type,
                            frames=frames, as_of=as_of,
                            verbose=self.options.verbose)
        # ⚠️ 目标要为**所有族**准备：依赖闭包里的 15min step 即使不落库也要有 targets。
        for fam in DEFAULT_FAMILIES:
            ctx.targets[fam] = self._targets(ctx, fam, db=db)

        requested_tiers = {TIER_RUNTIME, TIER_DAILY}
        outs = self._runner.run(ctx, self.steps, families=families,
                                tiers=requested_tiers)
        stats = {}
        for fam, frame in outs.items():
            if fam not in FAMILY_COLUMNS or fam not in families:
                continue
            frame = self._restrict_to_targets(frame, ctx.targets.get(fam))
            got = self._write_family(ctx, fam, frame, db=db)
            if got is not None:
                stats[fam] = got
        return CodeResult(code=code, status='ok', result=stats,
                          market_type=market_type)


# --------------------------------------------------------------------------- #
# 默认 step 注册（实际实现在 `pipeline/review/steps/`）
# --------------------------------------------------------------------------- #
def build_default_steps() -> List[FeatureStep]:
    from GolemQ.pipeline.review.steps.daily import build_daily_steps
    from GolemQ.pipeline.review.steps.minor import build_minor_steps
    from GolemQ.pipeline.review.steps.zen import build_zen_steps
    from GolemQ.pipeline.review.coldstart import build_coldstart_steps

    return (build_daily_steps() + build_minor_steps() +
            build_zen_steps() + build_coldstart_steps())


# --------------------------------------------------------------------------- #
# 顶层入口
# --------------------------------------------------------------------------- #
def run_review(eval_range: str = 'fast', codes: Optional[Iterable[str]] = None,
               pct_of_cores: float = 0.32, families=DEFAULT_FAMILIES,
               force: bool = False, dry_run: bool = False, verbose: bool = False,
               offset: str = '', from_date: str = '', recheck_days: int = 0,
               limit: int = 0, **kwargs):
    """解析 scope → 逐码跑 → 返回 ``(CodeResult 列表, 统计)``。"""
    from GolemQ.analysis.cv.scopes import perpar_symbol_range
    from GolemQ import get_active_market

    if codes:
        codelist = [str(c).strip() for c in codes if str(c).strip()]
    else:
        got = perpar_symbol_range(eval_range, dataset=get_active_market().scan_dataset(),
                                  verbose=verbose)
        if got is False:
            raise ValueError(f'无法解析 eval_range={eval_range!r}')
        codelist = list(got)
    if limit and limit > 0:
        codelist = codelist[:int(limit)]

    cpu = int(pct_of_cores) if pct_of_cores else 0
    if cpu > 1:
        import multiprocessing
        cpu = max(1, int((cpu / 100.0) * multiprocessing.cpu_count()))
    elif 0 < cpu <= 1:
        import multiprocessing
        cpu = max(1, int(cpu * multiprocessing.cpu_count()))
    # joblib 的 n_jobs 是**进程数**；旧 `--pct_of_cores 32` = 32% 核数。

    opts = ReviewOptions(families=tuple(families), force=force, dry_run=dry_run,
                         recheck_days=recheck_days, from_date=from_date,
                         offset=offset, verbose=verbose)
    pipe = ReviewPipeline(options=opts, verbose=verbose)
    return pipe.run_codes(codelist, pool_size=cpu, verbose=verbose,
                          as_of=(offset or None))
