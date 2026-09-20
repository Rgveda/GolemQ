#!/usr/bin/env python3
"""审计**实际被使用**的常量，并与老树的权威值对账 —— 只读，不改代码。

为什么不全搬
============
老树 `utils/parameter.py` 有约 2496 个常量，新树只移植了约 135 个。
但**全搬是浪费** —— 绝大多数常量没有任何代码引用。

本脚本只关心**被引用的那些**：扫描全树的 `AKA.X` / `FLD.X` / `FTR.X` 等访问，
取出名字集合，再拿老树的值对账。

为什么老树的值是权威
====================
老树的值是**数据在 MongoDB 里的实际字段名**（如 `'ZPLagMajR'`、`'m9tMaj'`）。
新树若写成 `'zen_peak_timing_lag_major_real'`，读的就是一个从不存在的列 ——
**报错也好，静默全 NaN 也好，都是读不到**。

三种结果
========
* ``OK``      新树已定义且值与老树一致 —— 不用动
* ``WRONG``   新树已定义但值不同 —— **正在静默读错列**，必须改
* ``MISSING`` 新树未定义（靠 `_StubMeta` 伪造）—— 必须补

用法
====
    python tools/audit_constants.py              # 出对账报告
    python tools/audit_constants.py --code       # 只输出 MISSING/WRONG 的赋值语句
"""
from __future__ import annotations

import argparse
import ast
import os
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: 代码里的别名 → **规范化类名**（统一基准，避免两棵树类名不同导致查不到）
ALIAS_TO_CLASS = {
    'AKA': 'AKA',
    'FIELD': 'FIELD', 'FLD': 'FIELD',
    'FEATURES': 'FEATURES', 'FTR': 'FEATURES',
    'TREND_STATUS': 'TREND_STATUS', 'ST': 'TREND_STATUS',
    'STATE': 'STATE', 'STE': 'STATE',
    'MAS': 'MAS',
    'RSK': 'RSK',
    'LTT': 'LTT',
}

#: 规范化类名 → (新树文件, 新树类名, 老树文件, 老树类名)。
#: **两棵树的类名与文件都可能不同**（新树 `FIELD` 在 `core/constants.py`，
#: 老树 `INDICATOR_FIELD` 在 `utils/parameter.py`；`RSK` 新树在
#: `models/risk.py`、老树在 `models/alias.py`）。
#:
#: 早先的版本只覆盖 `core/constants.py`，于是 **`MAS` / `RSK` 的同类漂移
#: 完全没被查到** —— 实测 `MAS.CONCEPT_XGB_ECHO_TIMING_LAG_COMBO` 新树写
#: `'concept_xgb_echo_timing_lag_combo'`、老树是 `'cmXgbLagCmb'`，
#: `RSK.CVaR_PEAK_PRICE` 新树 `'cvar_peak_price'`、老树 `'ES_PEAK_P'`。
CONST_SOURCES = {
    'AKA': ('GolemQ/core/constants.py', 'AKA',
            'GolemQ_old/utils/parameter.py', 'AKA'),
    'FIELD': ('GolemQ/core/constants.py', 'FIELD',
              'GolemQ_old/utils/parameter.py', 'INDICATOR_FIELD'),
    'FEATURES': ('GolemQ/core/constants.py', 'FEATURES',
                 'GolemQ_old/utils/parameter.py', 'FEATURES'),
    'TREND_STATUS': ('GolemQ/core/constants.py', 'TREND_STATUS',
                     'GolemQ_old/utils/parameter.py', 'TREND_STATUS'),
    'STATE': ('GolemQ/core/constants.py', 'STATE',
              'GolemQ_old/utils/parameter.py', 'STATE'),
    'MAS': ('GolemQ/models/massive.py', 'MAS',
            'GolemQ_old/models/massive.py', 'MAS'),
    'RSK': ('GolemQ/models/risk.py', 'RSK',
            'GolemQ_old/models/alias.py', 'RSK'),
    'LTT': ('GolemQ/models/alias.py', 'LTT',
            'GolemQ_old/models/alias.py', 'LTT'),
}


def _class_consts(path: str, classes) -> dict:
    """从一个文件里取指定类的**字面量赋值**。解析失败的跳过而非中断。"""
    out = {}
    if not os.path.isfile(path):
        return out
    tree = ast.parse(open(path, encoding='utf-8').read())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name in classes:
            for sub in node.body:
                if (isinstance(sub, ast.Assign) and len(sub.targets) == 1
                        and isinstance(sub.targets[0], ast.Name)):
                    try:
                        out[(node.name, sub.targets[0].id)] = ast.literal_eval(sub.value)
                    except Exception:                    # noqa: BLE001
                        pass
    return out


def used_constants() -> dict:
    """扫全树，返回 `{(新树类名, 常量名): [(文件, 行号)]}` 只含**被引用**的。"""
    used = defaultdict(list)
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
                if cls is None or node.attr.startswith('__'):
                    continue
                used[(cls, node.attr)].append(
                    (os.path.relpath(full, ROOT), node.lineno))
    return used


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--code', action='store_true',
                    help='只输出 MISSING/WRONG 的赋值语句，便于直接粘贴')
    args = ap.parse_args()

    old, new = {}, {}
    for canon, (new_rel, new_cls, old_rel, old_cls) in CONST_SOURCES.items():
        for (_, name), val in _class_consts(
                os.path.join(ROOT, new_rel), (new_cls,)).items():
            new[(canon, name)] = val
        for (_, name), val in _class_consts(
                os.path.join(ROOT, old_rel), (old_cls,)).items():
            old[(canon, name)] = val

    used = used_constants()

    ok, wrong, missing, noold = [], [], [], []
    for (cls, name), sites in sorted(used.items()):
        old_key = (cls, name)
        old_val = old.get(old_key)
        new_val = new.get(old_key)

        if old_val is None:
            noold.append((cls, name, new_val, sites))
        elif new_val is None:
            missing.append((cls, name, old_val, sites))
        elif new_val != old_val:
            wrong.append((cls, name, new_val, old_val, sites))
        else:
            ok.append((cls, name))

    print(f'被引用的常量：{len(used)} 个（分布在 {sum(len(v) for v in used.values())} 处）\n')
    print(f'  OK      {len(ok):>4}  已定义且与老树一致')
    print(f'  WRONG   {len(wrong):>4}  已定义但**值不同** —— 正在静默读错列')
    print(f'  MISSING {len(missing):>4}  未定义，靠 _StubMeta 伪造')
    print(f'  老树无  {len(noold):>4}  老树也没有同名常量（新树自造，需人工判断）')
    print()

    if args.code:
        print('# ===== 需修正（WRONG）=====')
        for cls, name, nv, ov, _ in wrong:
            print(f'  {name} = {ov!r}    # 原新树值 {nv!r}')
        print()
        print('# ===== 需补充（MISSING）=====')
        for cls, name, ov, _ in missing:
            print(f'  {name} = {ov!r}')
        return 0

    if wrong:
        print('=== WRONG：值不同，必须改成老树的值 ===')
        for cls, name, nv, ov, sites in wrong:
            print(f'  [{cls}] {name}')
            print(f'      新={nv!r}  老={ov!r}')
            for rel, line in sites[:3]:
                print(f'      {rel}:{line}')
        print()
    if missing:
        print('=== MISSING：新树未定义 ===')
        for cls, name, ov, sites in missing[:40]:
            print(f'  [{cls}] {name:44s} = {ov!r}   ({len(sites)} 处引用)')
        if len(missing) > 40:
            print(f'  …另有 {len(missing)-40} 个')
        print()
    if noold:
        print('=== 老树无同名：需人工判断是否为真常量 ===')
        for cls, name, nv, sites in noold[:20]:
            print(f'  [{cls}] {name:44s} 新树值={nv!r}  ({len(sites)} 处)')
    return 1 if (wrong or missing) else 0


if __name__ == '__main__':
    sys.exit(main())
