"""元数据已收口到 `pyproject.toml` 的 `[project]` 表 —— 此处不再重复。

保留本文件只为兼容仍按 `setup.py` 找包的旧工具；`pip install -e .` 走的是
`pyproject.toml`（`[project]` 优先，`setup.py` 的 kwargs 本来就会被忽略）。

⚠️ 别在这里重新填 name/version/description —— 那会与 pyproject 分叉，
且分叉时**不报错**（pyproject 静默胜出），改了两边不一致没人知道。
"""
from setuptools import setup

setup()
