# coding: utf-8
"""`FeatureStep` 与因果切片工具 —— review 管线的**唯一**编排层。

设计要点
========
* 每个 step 只声明：输入频率、输出族、列、暖机/历史规则、依赖、成本档；
* **runtime 档**只做轻量、可增量计算的部分；**daily 档**包含逐快照重算
  （KlineAnalyze 等），只由 `--daily` 落库；
* 所有 step 共用 :func:`slice_asof`，保证「T 的值只依赖 `ts <= T` 的数据」；
* step 函数签名统一为 ``func(ctx, targets) -> DataFrame``，返回索引
  ``(ts, code)`` 的载荷列，便于按族合并/落库。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

TIER_RUNTIME = 'runtime'
TIER_DAILY = 'daily'
TIERS = (TIER_RUNTIME, TIER_DAILY)


def slice_asof(frame: Optional[pd.DataFrame], as_of, n_bars: Optional[int] = None) -> pd.DataFrame:
    """取 ``ts <= as_of`` 的行；``n_bars`` 再截最后 n 根（含 as_of）。

    ⚠️ 不用 ``get_loc(as_of)``：runtime 的 as_of 可能落在两根 bar 之间；
    一律用布尔掩码 + 尾部截取。
    """
    if frame is None or len(frame) == 0:
        return pd.DataFrame() if frame is None else frame.iloc[0:0]
    ts = pd.DatetimeIndex(pd.to_datetime(frame.index.get_level_values(0)))
    mask = ts <= pd.Timestamp(as_of)
    out = frame.loc[mask]
    if n_bars is not None and n_bars > 0:
        out = out.iloc[-n_bars:]
    return out


def target_times(frame: Optional[pd.DataFrame], as_of, only_after=None) -> pd.DatetimeIndex:
    """目标快照时间：``frame`` 里 ``<= as_of`` 的唯一时间戳（升序）。"""
    if frame is None or len(frame) == 0:
        return pd.DatetimeIndex([])
    ts = pd.DatetimeIndex(pd.to_datetime(frame.index.get_level_values(0)))
    ts = pd.DatetimeIndex(sorted(set(ts[ts <= pd.Timestamp(as_of)])))
    if only_after is not None:
        ts = ts[ts > pd.Timestamp(only_after)]
    return ts


@dataclass(frozen=True)
class FeatureStep:
    """一个特征步骤的声明。

    :param name: 唯一名（deps 用它引用）
    :param tier: ``runtime`` / ``daily``
    :param input_freq: ``day`` / ``60min`` / ``15min``
    :param output_family: ``day`` / ``60min`` / ``15min`` / ``coldstart`` / ``None``
    :param columns: 该 step 产出的列（runner 会校验）
    :param func: ``func(ctx, targets) -> DataFrame``
    :param deps: 依赖的 step 名
    :param history: ``bars``（每快照固定窗口）或 ``anchor``（固定锚点前缀）
    :param warmup: ``history='bars'`` 时的窗口根数
    """

    name: str
    tier: str
    input_freq: str
    output_family: Optional[str]
    columns: Tuple[str, ...]
    func: Callable[['ReviewContext', pd.DatetimeIndex], pd.DataFrame]
    deps: Tuple[str, ...] = ()
    history: str = 'bars'
    warmup: Optional[int] = None
    snapshot_wise: bool = True


@dataclass
class ReviewContext:
    """一个 code 的计算上下文（batch 与 runtime 共用）。"""

    code: str
    market_type: Any
    frames: Dict[str, pd.DataFrame] = field(default_factory=dict)
    as_of: Any = None
    targets: Dict[str, pd.DatetimeIndex] = field(default_factory=dict)
    persisted: Dict[str, pd.DataFrame] = field(default_factory=dict)
    outputs: Dict[str, pd.DataFrame] = field(default_factory=dict)
    verbose: bool = False

    def frame(self, freq: str) -> pd.DataFrame:
        return self.frames.get(freq, pd.DataFrame())

    def slice(self, freq: str, as_of=None, n_bars: Optional[int] = None) -> pd.DataFrame:
        return slice_asof(self.frame(freq), self.as_of if as_of is None else as_of, n_bars)


class StepRunner:
    """按依赖顺序执行 step；只合并 **已请求** 的族/档。"""

    def run(self, ctx: ReviewContext,
            steps: Sequence[FeatureStep],
            families: Optional[Iterable[str]] = None,
            tiers: Optional[Iterable[str]] = None) -> Dict[str, pd.DataFrame]:
        want_families = set(families) if families is not None else None
        want_tiers = set(tiers) if tiers is not None else set(TIERS)
        chosen = [s for s in steps
                  if s.tier in want_tiers and
                  (want_families is None or s.output_family in want_families or
                   s.output_family is None)]

        # ⚠️ 依赖闭包：请求 15min 时，`minute15_aligned` 依赖 60min 对齐的
        # `zen_snapshots` —— 后者必须一并算（15min 表只取样，不重算）。
        by_name = {s.name: s for s in steps}
        chosen_names = {s.name for s in chosen}
        stack = list(chosen_names)
        while stack:
            name = stack.pop()
            step = by_name.get(name)
            if step is None:
                raise ValueError(f'FeatureStep 依赖不存在：{name!r}')
            for dep in step.deps:
                if dep not in chosen_names:
                    chosen_names.add(dep)
                    stack.append(dep)
        chosen = [s for s in steps if s.name in chosen_names]

        # 简单拓扑排序（deps 必须在本步骤之前）
        pending = list(chosen)
        done = set()
        ordered: List[FeatureStep] = []
        while pending:
            progressed = False
            for s in list(pending):
                if all(d in done for d in s.deps):
                    ordered.append(s)
                    done.add(s.name)
                    pending.remove(s)
                    progressed = True
            if not progressed:
                names = [s.name for s in pending]
                raise ValueError(f'FeatureStep 依赖成环/缺失：{names}')

        for step in ordered:
            targets = ctx.targets.get(step.output_family, pd.DatetimeIndex([]))
            if step.output_family is not None and len(targets) == 0:
                continue
            out = step.func(ctx, targets)
            if out is None or len(out) == 0:
                continue
            missing = [c for c in step.columns if c not in out.columns]
            if missing:
                raise ValueError(f'FeatureStep {step.name!r} 缺少声明列：{missing}')
            out = out.loc[:, list(step.columns)]
            ctx.outputs[step.name] = out
            if step.output_family is not None:
                prev = ctx.outputs.get(f'__family__{step.output_family}')
                ctx.outputs[f'__family__{step.output_family}'] = (
                    out if prev is None else prev.join(out, how='outer'))

        return {k[len('__family__'):]: v
                for k, v in ctx.outputs.items() if k.startswith('__family__')}


def ensure_sorted(frame: pd.DataFrame) -> pd.DataFrame:
    """按 (ts, code) 排序 —— 落库/对齐前统一顺序。"""
    if frame is None or len(frame) == 0:
        return frame
    return frame.sort_index()
