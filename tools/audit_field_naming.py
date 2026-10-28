#!/usr/bin/env python3
# coding: utf-8
"""审计常量里的**列名值**是否符合「伪驼峰缩写」约定（只读）。

约定
====
列名值应是 `lowerCamel` / `lowerCamel+缩写后缀`（如 `garP` / `poly9MajD` /
`MsLeadinBf` / `btmP`），**不用** `UPPER_SNAKE` 或全大写。

分类
====
* **旧列名映射源**：出现在 `pipeline/review/contract.LEGACY_RENAME` 的键里 ——
  这是老表列名，必须原样保留（否则映射断）。
* **老树真值**：在 `GolemQ_old/` 常量表里出现过 —— 可能已落库；
  改名要走 `$rename` + 读写口迁移（D41 / D65 / D70 的流程）。
* **新树自造**：老树没有 —— 若也未落库，可直接改。

用法
====
```bash
python tools/audit_field_naming.py            # 分组报告
python tools/audit_field_naming.py --fail     # 有新树自造的违例时 exit 1
```
"""
from __future__ import annotations

import argparse
import ast
import os
import re
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

_FIELD_CLASSES = ('FIELD', 'FEATURES', 'AKA', 'LTT', 'TRD', 'ZEN', 'MAS')
_UPPER_RE = re.compile(r'^[A-Z0-9]+$')

#: 用户 2026-10-28 确认**保留**的名字：业界标准指标名 / 已落库元数据 / 频率令牌。
_KEEP_EXACT = {
    'MACD', 'DEA', 'DIF', 'RSI', 'CCI', 'MTM', 'MTMMA', 'ATR_UB',
    'SED', 'SE', 'JGCYD', 'ZLCB', '1M', 'GBP',
}
_KEEP_RE = (re.compile(r'^BOLL_.*$'), re.compile(r'^MA\d+$'),
            re.compile(r'^HMA\d+$'), re.compile(r'^Q[1-4]$'))


def _keep(v: str) -> bool:
    return v in _KEEP_EXACT or any(r.match(v) for r in _KEEP_RE)


def _bad_value(v: str) -> bool:
    """违例 = 含下划线，或全大写/数字（`MsLeadinBf` / `4QPwrln` 这类伪驼峰合法）。"""
    return ('_' in v) or bool(_UPPER_RE.fullmatch(v))


def _class_assignments(path):
    out = {}
    tree = ast.parse(open(path, encoding='utf-8').read())
    for cls in [n for n in tree.body if isinstance(n, ast.ClassDef)]:
        for n in cls.body:
            if (isinstance(n, ast.Assign) and isinstance(n.value, ast.Constant)
                    and isinstance(n.value.value, str)):
                for t in n.targets:
                    if isinstance(t, ast.Name):
                        out.setdefault(cls.name, {})[t.id] = n.value.value
    return out


def _old_values():
    vals = set()
    for rel in ('GolemQ_old/utils/parameter.py', 'GolemQ_old/models/alias.py',
                'GolemQ_old/models/massive.py', 'GolemQ_old/models/risk.py'):
        p = os.path.join(_REPO, rel)
        if not os.path.exists(p):
            continue
        try:
            for m in _class_assignments(p).values():
                vals.update(m.values())
        except SyntaxError:
            continue
    return vals


def _legacy_rename_keys():
    from GolemQ.pipeline.review.contract import LEGACY_RENAME
    return set(LEGACY_RENAME)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--fail', action='store_true',
                    help='有新树自造且未映射的违例时 exit 1')
    args = ap.parse_args()

    new = {}
    for rel in ('GolemQ/core/constants.py', 'GolemQ/models/alias.py'):
        for cls, m in _class_assignments(os.path.join(_REPO, rel)).items():
            if cls in _FIELD_CLASSES:
                new.update({f'{cls}.{k}': v for k, v in m.items()})

    old = _old_values()
    rename_src = _legacy_rename_keys()

    bad = {k: v for k, v in new.items() if _bad_value(v)}
    cats = {'保留（用户确认）': [], '映射源（保留）': [],
            '老树真值（改名需迁移）': [], '新树自造（可直接改）': []}
    for k, v in sorted(bad.items(), key=lambda x: x[1]):
        if _keep(v):
            cats['保留（用户确认）'].append((v, k))
        elif v in rename_src:
            cats['映射源（保留）'].append((v, k))
        elif v in old:
            cats['老树真值（改名需迁移）'].append((v, k))
        else:
            cats['新树自造（可直接改）'].append((v, k))

    print('列名值违例（非 lowerCamel）：%d 个' % len(bad))
    for cat, rows in cats.items():
        print('\n=== %s：%d ===' % (cat, len(rows)))
        for v, k in rows:
            print('  %-24s %s' % (v, k))
    return 1 if (args.fail and cats['新树自造（可直接改）']) else 0


if __name__ == '__main__':
    raise SystemExit(main())
