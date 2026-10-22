# coding:utf-8
"""`--strategy compact` —— 逐码计算 compact 特征（只读，不落库）。

入口见 :func:`GolemQ.pipeline.compact.pipeline.run_compact`。
"""
from .pipeline import CompactPipeline, DEFAULT_COMPACT_SCOPE, run_compact

__all__ = ['CompactPipeline', 'DEFAULT_COMPACT_SCOPE', 'run_compact']
