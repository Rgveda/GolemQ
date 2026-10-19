# coding: utf-8
"""**Runtime API**：在内存里算 review 特征，**零 DB 写**。

用户口径（2026-10-19）：

* ``--daily`` 是**唯一持久化路径**；
* runtime 只在内存里跑轻量/高频 step，重的/低频的从最近一次落库行读取；
* as-of ``T`` 只读 ``ts <= T`` 的数据（与批处理同一套 `slice_asof`）；
* 同一 ``T`` 的 runtime 输出必须与批处理行**逐值一致**。

本模块不调用任何写口；测试会 patch `upsert_fields` / `save_reality_feat` 断言。
"""
from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence

import pandas as pd

from GolemQ.pipeline.steps import (FeatureStep, ReviewContext, StepRunner,
                                   TIER_DAILY, TIER_RUNTIME, target_times)
from GolemQ.pipeline.review.contract import FAMILY_COLUMNS
from GolemQ.pipeline.review.pipeline import (DEFAULT_FAMILIES, ReviewOptions,
                                             ReviewPipeline)


def latest_closed_bar(frequency: str = '15min', now=None) -> pd.Timestamp:
    """把墙上时间映射到最近一根**已收盘**的 15min / 60min bar 的时间戳。

    A 股时段：09:30-11:30、13:00-15:00。只做粗粒度判断（非交易时段返回上一根）。
    """
    now = pd.Timestamp(now or pd.Timestamp.now())
    freq = str(frequency).lower()
    minutes = 15 if freq.startswith('15') else 60
    ts = now.floor(f'{minutes}min')
    # 收盘后的最后一根：15:00（15min）/ 15:00（60min）
    if now.time() > pd.Timestamp('15:00').time():
        return now.normalize() + pd.Timedelta(hours=15)
    if now.time() < pd.Timestamp('09:30').time():
        prev = now.normalize() - pd.Timedelta(days=1)
        return prev + pd.Timedelta(hours=15)
    return ts


def compute_runtime(code: str,
                    as_of=None,
                    families: Sequence[str] = ('day', '60min', '15min'),
                    include_daily: bool = False,
                    include_heavy: bool = False,
                    restore_price: bool = True,
                    market_type=None,
                    steps: Optional[Sequence[FeatureStep]] = None,
                    **kwargs) -> Dict[str, pd.DataFrame]:
    """返回 ``{family: DataFrame}``；**不写任何行/水位/索引**。

    :param include_daily: 把 daily 档也**在内存里**算（默认 False，从落库行读）。
    :param include_heavy: 允许重档（zen KlineAnalyze 等）在 runtime 里现算；
        只建议 watchlist 小范围使用。``include_daily=True`` 时自动包含。
    :param restore_price: 默认 True —— 把落库的「÷close」比值列乘回 close，
        还原价格量纲（D49/D54 的双向约定）。要拿原始比值传 False。
    """
    as_of = pd.Timestamp(as_of or latest_closed_bar())
    opts = ReviewOptions(families=tuple(families), dry_run=True, verbose=kwargs.get('verbose', False))
    pipe = ReviewPipeline(steps=steps, options=opts,
                          market_type=market_type,
                          verbose=kwargs.get('verbose', False))
    if market_type is None:
        market_type = pipe.market_type_of(code)
    frames = pipe._fetch_frames(code, as_of,
                                freqs=ReviewPipeline._required_freqs(families))
    if not frames:
        return {}

    ctx = ReviewContext(code=str(code), market_type=market_type,
                        frames=frames, as_of=as_of,
                        verbose=kwargs.get('verbose', False))
    for fam in DEFAULT_FAMILIES:
        ctx.targets[fam] = target_times(frames.get(fam), as_of)

    tiers = {TIER_RUNTIME}
    if include_daily or include_heavy:
        tiers.add(TIER_DAILY)
    out = StepRunner().run(ctx, pipe.steps, families=families, tiers=tiers)

    # 未现算的 daily 档：从最近一次落库行补齐（只读）
    if not (include_daily or include_heavy):
        out = _attach_persisted(out, code, market_type, families, as_of)
    if restore_price:
        # 落库是「÷close」比值；runtime 作为读路径**还原价格量纲**。
        from GolemQ.markets.StockCN.reality_save import restore_price_columns
        for fam, frame in list(out.items()):
            src = frames.get(fam)
            if src is None or len(src) == 0 or 'close' not in src.columns:
                continue
            out[fam] = restore_price_columns(frame, src['close'])
    return out


def _attach_persisted(out: Dict[str, pd.DataFrame], code: str, market_type,
                      families: Iterable[str], as_of: pd.Timestamp) -> Dict[str, pd.DataFrame]:
    from GolemQ.markets.StockCN.reality_save import fetch_reality_feat
    from GolemQ import get_active_market

    db = get_active_market().GOLEMQ_STOCK_CN
    start = (as_of - pd.Timedelta(days=365 * 5)).strftime('%Y-%m-%d')
    end = as_of.strftime('%Y-%m-%d')
    for fam in families:
        need = [c for c in FAMILY_COLUMNS.get(fam, ())
                if c not in out.get(fam, pd.DataFrame()).columns]
        if not need:
            continue
        try:
            got = fetch_reality_feat([code], start, end, family=fam,
                                     market_type=market_type, db=db)
        except Exception:        # noqa: BLE001 读不到就当没有
            got = None
        if got is None or len(got) == 0:
            continue
        have = [c for c in need if c in got.columns]
        if not have:
            continue
        got = got.loc[:, have]
        if fam in out and len(out[fam]) > 0:
            out[fam] = out[fam].join(got, how='outer')
        else:
            out[fam] = got
    return out
