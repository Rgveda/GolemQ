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

import hashlib
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence

import pandas as pd

from GolemQ.core.constants import MARKET_TYPE
from GolemQ.markets.StockCN.date_utils import GQ_util_last_trade_as_of
from GolemQ.pipeline.base import BasePipeline, CodeResult
from GolemQ.pipeline.steps import (FeatureStep, ReviewContext, StepRunner,
                                   TIER_DAILY, TIER_RUNTIME, target_times)
from GolemQ.pipeline.review.contract import (FAMILY_COLUMNS, LEGACY_RENAME,
                                             persist_check_columns)

#: 三族；coldstart 列已合并进 60min，不再有独立族。
DEFAULT_FAMILIES = ('day', '60min', '15min')
PIPELINE_NAME = 'review'

#: 各族的读取暖机（**只影响取数窗口，不决定写哪些行**）。
DAY_LOOKBACK_DAYS = 4000
#: 固定锚点：batch 与 runtime 必须用同一个起点，否则递归指标（EMA/BS/KMeans/
#: Prophet）的值会随窗口漂移。数据不足时自然截断。
REVIEW_ANCHOR_DAY = '2016-01-01'
REVIEW_ANCHOR_15MIN = '2017-01-01'

#: **写库目标**的回看上限（不是取数窗口）：首次回填也最多算这么多天，
#: 与老树一致（日线 `last_day-2560d`；60/15min 默认 800 天）。
#: 早于这个下界的数据只用来做暖机，不产生新行。
TARGET_LOOKBACK_DAYS = {'day': 2560, '60min': 800, '15min': 800}


def chunk_targets(targets_by_family: Dict[str, pd.DatetimeIndex],
                  max_years: float = 1.0) -> Iterator[Dict[str, pd.DatetimeIndex]]:
    """把三族目标按**日历年**切块（生成器），每块算完**立刻落库**。

    起因（D88）：一个 review 从没跑过的码要一次性回填 800 天 × 三族
    （day 1,700 / 60min 2,037 / 15min 8,145），实测约 **2.5 小时**，
    撞单码护栏就**全丢**。按年切块后每块 ~30~40 分钟，块末即写；
    中断最多丢一块，下次运行 `_targets` 只返回缺口，天然续跑。

    :param targets_by_family: ``{family: DatetimeIndex}``
    :param max_years: 每块最多跨几个日历年（``<=0`` 或 ``<=1`` 都按 1 年；
        目标本来就不跨年时只产出**一块**，行为与切块前完全一致）

    >>> import pandas as pd
    >>> t = pd.DatetimeIndex(pd.to_datetime(['2024-06-03', '2025-02-10']))
    >>> [list(c['day'].year) for c in chunk_targets({'day': t})]
    [[2024], [2025]]
    >>> [len(c['day']) for c in chunk_targets({'day': t}, max_years=0)]
    [2]
    """
    fams = list(targets_by_family)
    years = sorted({int(pd.Timestamp(t).year)
                    for v in targets_by_family.values() if v is not None
                    for t in v})
    if not years or max_years is None or max_years <= 0 or len(years) <= 1:
        yield targets_by_family
        return
    step = max(1, int(round(max_years)))
    for k in range(0, len(years), step):
        bucket = set(years[k: k + step])
        out = {}
        for fam in fams:
            v = targets_by_family.get(fam)
            if v is None or len(v) == 0:
                out[fam] = v
                continue
            keep = [t for t in v if int(pd.Timestamp(t).year) in bucket]
            out[fam] = pd.DatetimeIndex(keep)
        yield out


def as_of_bound(value) -> Optional[pd.Timestamp]:
    """把**纯日期**形式的截止时间补到当日 **23:59:59**（F7）。

    `--offset 2026-01-05` 以前会被当成 `00:00`：日族（bar 时间戳 00:00）
    含当天，60min / 15min 却停在**前一交易日 15:00** —— 同一截止日在三族间
    不一致，看起来像分钟数据缺口。纯日期一律按「含当日全部」解释；
    带时刻的输入原样保留（runtime 的 as-of 就在盘中）。

    >>> as_of_bound('2026-01-05')
    Timestamp('2026-01-05 23:59:59')
    >>> as_of_bound('2026-01-05 10:30')
    Timestamp('2026-01-05 10:30:00')
    >>> as_of_bound(None) is None
    True
    """
    if value is None or value == '':
        return None
    ts = pd.Timestamp(value)
    if ts == ts.normalize():
        return ts.normalize() + pd.Timedelta(hours=23, minutes=59, seconds=59)
    return ts


@dataclass
class ReviewOptions:
    families: Sequence[str] = DEFAULT_FAMILIES
    force: bool = False
    dry_run: bool = False
    recheck_days: int = 0
    from_date: str = ''
    offset: str = ''
    verbose: bool = False
    #: 单次算完并落库的**最大跨年数**（D88）。超过就按日历年切块：
    #: 每块算完立刻写库，撞护栏最多丢一块，下次运行从缺口续跑。
    #: ``0`` = 不切块（老行为）。
    max_years: float = 1.0


#: 单码超时 = **1.5 小时**（用户 2026-11-22 定）。
#:
#: 成本重估（同一轮优化之后）：一个 review 从没跑过的码要全量回填
#: day 1,700 + 60min 2,037 + 15min 8,145 个目标，实测约：
#:
#: * `day_core`       1,700 天 × 0.79s ≈ 22 分钟（ChipDist 定容+数组直算 0.27s/天、
#:   RegTree 接回 jit 0.04s/天、Prophet 0.66s/天）
#: * `zen_snapshots`  2,037 目标 × 0.69s ≈ 23 分钟（含 `ZenOLSLag` 逐窗口 0.075s）
#:
#: 合计 ≈ **45~48 分钟**。护栏留 ~2 倍余量 ⇒ 90 分钟。超过它基本可以判定卡住了，
#: 而不是"慢"；且 D88 ④ 的按年切块会保证最多丢一块（下次从缺口续跑）。
REVIEW_CODE_TIMEOUT = 90 * 60


class ReviewPipeline(BasePipeline):
    """逐码 review 管线。"""

    def __init__(self, steps: Optional[Sequence[FeatureStep]] = None,
                 options: ReviewOptions = None, **kwargs):
        kwargs.setdefault('timeout', REVIEW_CODE_TIMEOUT)
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
            # 日族只用日线（`se`/`ZenPivotCnt` 2026-10-30 起回归 60min，D78）。
            req |= {'day'}
        if fam & {'60min'}:
            # 60min 里的 coldstart 列（STAGE/BST/M9T）要从日线采样。
            req |= {'60min', '15min', 'day'}
        if fam & {'15min'}:
            # 15min 族由 `zen_snapshots`（60min 对齐）取样，故也要 60min。
            req |= {'60min', '15min'}
        return req

    def _fetch_frames(self, code: str, as_of: pd.Timestamp,
                      freqs=None) -> Dict[str, pd.DataFrame]:
        from GolemQ import get_active_market
        market = get_active_market()

        day_start = REVIEW_ANCHOR_DAY
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
            if key == '15min':
                freq_start = REVIEW_ANCHOR_15MIN
            else:
                # ⚠️ 60min **一律从 2016 锚点取**（D78）：`se` 的窗口是 2160 根、
                # `ZenPivotCnt` 是累计值，目标下界最早可到 `as_of-800d`，
                # 再往前还要 ≥2160 根（≈780 自然日）暖机。900 天窗口会让
                # **目标区间头部**（如 2024 年以前）少掉一半以上的基础数据，
                # SE/中枢计数全部失真。这条不许按“省 IO” reclaim。
                freq_start = REVIEW_ANCHOR_DAY
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
        """以**目标表里已有哪些 review 行**为准，返回还没算/没写的 bar。

        判据见 :mod:`pipeline.review.persistence`：一行里任一契约列非空即算已持久化；
        CV-only 行（4.4 迁移来的 `ClsMaj` 等）不算。`pipeline_watermarks` 不参与判据。
        """
        from GolemQ.pipeline.review.persistence import missing_targets

        src = ctx.frames.get(family)
        if src is None or len(src) == 0:
            return pd.DatetimeIndex([])
        as_of = ctx.as_of
        if self.options.offset:
            # 纯日期 ⇒ 当日 23:59:59（与 run_code 同一口径，F7）
            as_of = min(as_of, as_of_bound(self.options.offset))
        source_index = target_times(src, as_of)
        # 首次回填的**目标下界**：更早的数据只做暖机、不落新行（与老树 2560/800 天一致）。
        lookback = TARGET_LOOKBACK_DAYS.get(family)
        if lookback:
            floor = as_of - pd.Timedelta(days=int(lookback))
            source_index = source_index[source_index >= floor]
        return missing_targets(
            source_index, ctx.code, family, ctx.market_type, db,
            force=self.options.force, from_date=self.options.from_date,
            recheck_days=self.options.recheck_days,
            # ⚠️ 判完成时**排除外部工具写的列**（`4QPwrlnWek`，见 contract
            # 的 `PERSIST_EXCLUDE`）—— 否则「只有周线列的行」会被当成已算完。
            columns=persist_check_columns(family))

    @staticmethod
    def _restrict_to_targets(frame: pd.DataFrame, targets) -> pd.DataFrame:
        """只保留目标时间戳的行 —— 防止把 `_minute15_rows` 的**全量 15min 帧**整表写库。"""
        if frame is None or len(frame) == 0:
            return frame
        if targets is None or len(targets) == 0:
            # ⚠️ 没有目标 ⇒ 一行都不该写。以前这里原样返回 frame，会把整帧
            # （包括已经持久化、或只作依赖算出来的行）再 upsert 一遍（F4）。
            return frame.iloc[0:0]
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
        # 暖机不足的行可能整行 NaN —— 不写空行（否则持久化检查会把它当"已持久化"）。
        payload = payload.loc[payload.notna().any(axis=1)]
        if len(payload) == 0:
            return None
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
        as_of = as_of_bound(kwargs.get('as_of') or self.options.offset or
                            GQ_util_last_trade_as_of())
        market_type = self.market_type_of(code)
        families = tuple(self.options.families)
        # 「库名不外泄」：pipeline 不拼库名，只从 **Active Market** 取实例，
        # 再作为参数传给写口/水位函数。
        from GolemQ import get_active_market
        db = get_active_market().get_database()
        # 分阶段耗时只进 `CodeResult.meta`，由 `Statistics` 的「Slowest」打印
        t_fetch = time.monotonic()
        frames = self._fetch_frames(code, as_of,
                                    freqs=self._required_freqs(families))
        fetch_s = time.monotonic() - t_fetch
        if not frames:
            return CodeResult(code=code, status='no_data',
                              market_type=market_type,
                              meta={'fetch': fetch_s})

        ctx = ReviewContext(code=str(code), market_type=market_type,
                            frames=frames, as_of=as_of,
                            verbose=self.options.verbose)
        # ⚠️ 目标要为**所有族**准备：依赖闭包里的 15min step 即使不落库也要有 targets。
        full = {}
        for fam in DEFAULT_FAMILIES:
            ctx.targets[fam] = self._targets(ctx, fam, db=db)
            full[fam] = ctx.targets[fam]

        requested_tiers = {TIER_RUNTIME, TIER_DAILY}
        stats: Dict[str, Any] = {}
        steps_s = write_s = 0.0
        chunks = 0
        for chunk in chunk_targets(full, self.options.max_years):
            chunks += 1
            ctx.targets = chunk
            ctx.outputs = {}
            t_steps = time.monotonic()
            outs = self._runner.run(ctx, self.steps, families=families,
                                    tiers=requested_tiers)
            steps_s += time.monotonic() - t_steps
            t_write = time.monotonic()
            for fam, frame in outs.items():
                if fam not in FAMILY_COLUMNS or fam not in families:
                    continue
                frame = self._restrict_to_targets(frame, ctx.targets.get(fam))
                got = self._write_family(ctx, fam, frame, db=db)
                if got is None:
                    continue
                # 分块写库要**累加**行数，不能后来块覆盖前面块的统计
                if fam in stats:
                    for k, v in got.items():
                        if isinstance(v, int):
                            stats[fam][k] = stats[fam].get(k, 0) + v
                else:
                    stats[fam] = dict(got)
            write_s += time.monotonic() - t_write
        return CodeResult(code=code, status='ok', result=stats,
                          market_type=market_type,
                          meta={'fetch': fetch_s, 'steps': steps_s,
                                'write': write_s, 'chunks': chunks})


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
def shard_codes(codes, spec: str = '') -> List[str]:
    """按 ``md5(code) % N == i-1`` 取第 i 片（1-based），保持原顺序。

    与 `tools/backfill_hour_extra.py --shard` **同一口径**（两处只留这一份实现）：
    多机各跑一片、合起来恰好覆盖全集且两两不交。

    >>> shard_codes(['600519', '000001', '301308'], '1/2')
    ['600519']
    >>> shard_codes(['600519', '000001', '301308'], '2/2')
    ['000001', '301308']
    >>> shard_codes(['600519', '000001', '301308'], '')
    ['600519', '000001', '301308']
    """
    out = [str(c).strip() for c in (codes or []) if str(c).strip()]
    if not spec:
        return out
    try:
        i_s, n_s = str(spec).split('/')
        i, n = int(i_s), int(n_s)
    except Exception as exc:        # noqa: BLE001
        raise ValueError(f'shard 形如 1/3，收到 {spec!r}') from exc
    if n < 1 or not (1 <= i <= n):
        raise ValueError(f'shard 形如 1/3（1 ≤ i ≤ N），收到 {spec!r}')
    return [c for c in out
            if int(hashlib.md5(c.encode()).hexdigest(), 16) % n == i - 1]


def run_review(eval_range: str = 'fast', codes: Optional[Iterable[str]] = None,
               pct_of_cores: float = 0.32, families=DEFAULT_FAMILIES,
               force: bool = False, dry_run: bool = False, verbose: bool = False,
               offset: str = '', from_date: str = '', recheck_days: int = 0,
               limit: int = 0, shard: str = '', code_timeout: int = 0,
               **kwargs):
    """解析 scope → 逐码跑 → 返回 ``(CodeResult 列表, 统计)``。

    ``shard='i/N'`` 在**解析出码表之后**按 md5 再切一片（多机分布式用，
    内部旗子 `--review-shard`；`limit` 作用在切分之后）。
    """
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
    codelist = shard_codes(codelist, shard)
    if limit and limit > 0:
        codelist = codelist[:int(limit)]

    # ⚠️ 换算收口到 `worker_count`（`pipeline/base.py`，与主流管线同一份）：
    # 以前是 `int(pct_of_cores)`，把默认的比例 0.32 截成 0 ⇒ 整条管线**串行**，
    # 而且并行横幅不出（F1，见 worklog 2026-11-16）。
    from GolemQ.pipeline.base import worker_count
    cpu = worker_count(pct_of_cores)
    # joblib 的 n_jobs 是**进程数**；旧 `--pct_of_cores 32` = 32% 核数。

    opts = ReviewOptions(families=tuple(families), force=force, dry_run=dry_run,
                         recheck_days=recheck_days, from_date=from_date,
                         offset=offset, verbose=verbose)
    _kw = {}
    if code_timeout and int(code_timeout) > 0:
        # 内部旗子 `--review-timeout`：全市场回填时，单码从零算满 800 天
        # 在并发下可能超过默认 2h（实测 002462/300520 超时）。
        _kw['timeout'] = int(code_timeout)
    pipe = ReviewPipeline(options=opts, verbose=verbose, **_kw)
    return pipe.run_codes(codelist, pool_size=cpu, verbose=verbose,
                          as_of=(offset or None))
