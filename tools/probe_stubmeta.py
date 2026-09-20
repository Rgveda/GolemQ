#!/usr/bin/env python3
"""量出 `_StubMeta` 伪造字段名的爆炸半径 —— **只读探测，不改代码**。

背景（见 `PITFALLS.md` P1）
===========================
`core/constants.py` 的 `_StubMeta.__getattr__` 对**未定义**的属性静默返回
`f'{类名小写}_{属性名小写}'`。于是：

* 拼错的常量名不报错，返回一个「看起来合理」的假名
* 该假名在数据里从不存在 → 取值得到 KeyError 或全 NaN
* 下游信号全 False → 回测「跑通了」但零成交，**无任何报错**

本脚本回答一个问题：**把 `_StubMeta` 改成抛 `AttributeError` 之后，会炸出多少处？**

方法：AST 扫全树，收集形如 `<别名>.<大写常量>` 的访问，与对应类**已定义**的属性
集合比对。未定义的即为「当前靠伪造在工作」的位置。

为什么先探测而不是直接改
========================
直接改会一次性炸出所有问题，且报错发生在运行时、分散在各处，难以定位。
先出清单 → 可按字典批量修正 → 再改 `_StubMeta`，改动可控。

用法
====
    python tools/probe_stubmeta.py            # 出清单
    python tools/probe_stubmeta.py --summary  # 只出按类聚合的计数
"""
from __future__ import annotations

import argparse
import ast
import os
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: 用了 `_StubMeta` 的常量类 → 代码里常见的别名。
#: 别名的来源见各文件顶部 import（如 `FROM ... import FIELD as FLD`）。
STUB_CLASSES = {
    'AKA': ('AKA',),
    'FIELD': ('FIELD', 'FLD'),
    'FEATURES': ('FEATURES', 'FTR'),
    'TREND_STATUS': ('TREND_STATUS', 'ST'),
    'STATE': ('STATE', 'STE'),
    'MAS': ('MAS',),
    'RSK': ('RSK',),
    'LTT': ('LTT',),
    'TRD': ('TRD',),
    'RAIL': ('RAIL',),
    'ZEN': ('ZEN',),
    'FBP': ('FBP',),
    'PFL': ('PFL',),
}

#: 别名 → 类名
ALIAS_TO_CLASS = {a: c for c, aliases in STUB_CLASSES.items() for a in aliases}


def defined_attrs() -> dict:
    """取各常量类**真正定义**的属性（不含 `_StubMeta` 伪造的）。

    用 ast 直接读源码 —— 不用 `getattr`，因为那正是会被伪造骗到的东西。
    """
    out = {}
    targets = {
        'core/constants.py': ('AKA', 'FIELD', 'FEATURES', 'TREND_STATUS', 'STATE'),
        'models/massive.py': ('MAS',),
        'models/risk.py': ('RSK',),
        'models/alias.py': ('LTT',),
        'models/poolcoef.py': ('TRD', 'RAIL', 'ZEN', 'FBP'),
        'portfolio/base.py': ('PFL',),
    }
    for rel, classes in targets.items():
        path = os.path.join(ROOT, 'GolemQ', rel)
        if not os.path.isfile(path):
            continue
        tree = ast.parse(open(path, encoding='utf-8').read())
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name in classes:
                names = set()
                for sub in node.body:
                    if isinstance(sub, ast.Assign):
                        for t in sub.targets:
                            if isinstance(t, ast.Name):
                                names.add(t.id)
                    elif isinstance(sub, ast.AnnAssign) and isinstance(sub.target, ast.Name):
                        names.add(sub.target.id)
                out.setdefault(node.name, set()).update(names)
    return out


def scan(known: dict):
    """扫全树，返回 `[(类名, 常量名, 文件, 行号)]`，只含**未定义**的。"""
    hits = []
    for dirpath, dirnames, filenames in os.walk(os.path.join(ROOT, 'GolemQ')):
        dirnames[:] = [d for d in dirnames
                       if d not in ('__pycache__', 'easyquotation', 'cookbooks')]
        for fn in filenames:
            if not fn.endswith('.py'):
                continue
            full = os.path.join(dirpath, fn)
            try:
                tree = ast.parse(open(full, encoding='utf-8').read())
            except Exception:                            # noqa: BLE001
                continue
            for node in ast.walk(tree):
                if not (isinstance(node, ast.Attribute)
                        and isinstance(node.value, ast.Name)):
                    continue
                cls = ALIAS_TO_CLASS.get(node.value.id)
                if cls is None:
                    continue
                # 只关心全大写的常量名（小写多是方法调用）
                if not node.attr.isupper():
                    continue
                if node.attr not in known.get(cls, set()):
                    hits.append((cls, node.attr,
                                 os.path.relpath(full, ROOT), node.lineno))
    return hits


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--summary', action='store_true')
    args = ap.parse_args()

    known = defined_attrs()
    print('各常量类已定义的属性数：')
    for cls in sorted(known):
        print(f'  {cls:14s} {len(known[cls]):>4}')
    print()

    hits = scan(known)
    if not hits:
        print('未发现未定义的常量访问 —— `_StubMeta` 改成抛错应当无爆炸。')
        return 0

    if args.summary:
        agg = defaultdict(set)
        for cls, attr, _, _ in hits:
            agg[cls].add(attr)
        print(f'未定义的常量访问：{len(hits)} 处，涉及 {len({h[1] for h in hits})} 个不同名字\n')
        for cls in sorted(agg):
            print(f'  {cls:14s} {len(agg[cls]):>3} 个未定义名: {sorted(agg[cls])[:6]}')
        return 1

    print(f'未定义的常量访问：{len(hits)} 处\n')
    for cls, attr, rel, line in sorted(hits):
        print(f'  {cls}.{attr:44s} {rel}:{line}')
    return 1


if __name__ == '__main__':
    sys.exit(main())
