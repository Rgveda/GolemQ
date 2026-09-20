#!/usr/bin/env python3
"""生成紧凑的 API 索引 —— `API_INDEX.md`。

为谁而做
========
**主要为「接手项目的下一个会话」而做**，不是为人类浏览。

上下文容量有限（本项目实测约两天一轮），新会话若靠逐个 `Read` 源文件来
建立全局观，会烧掉大量预算在**导航**上而非**工作**上。本索引把
「有哪些模块、各自提供什么」压到一份 ~300 行的 markdown 里，
读它比读 20 个源文件省一个数量级。

**它是「索引」不是「文档」** —— 只出「名称 + 首行摘要」，不出完整 docstring。
完整文档对agent是负收益：读完摘要仍要去读源码，全文只是多花一遍 token。
人类要浏览完整文档，那是 mkdocs/mkdocstrings 的活，与本脚本目标不同。

为什么不用现成工具
==================
`pyproject.toml` 目前**没有声明任何 dependencies**。为一份摘要索引装
mkdocs + mkdocstrings + material 十余个包，代价与收益不成比例。
本脚本零依赖，只做一件窄事：**用 ast 抽取签名与首行摘要**。

用法
====
    python tools/gen_api_index.py            # 写 API_INDEX.md
    python tools/gen_api_index.py --check    # 只检查是否已过期（CI/提交前用）

`--check` 返回非零表示索引与代码不同步 —— 索引会腐烂，**必须能被发现**。
"""
from __future__ import annotations

import argparse
import ast
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG = os.path.join(ROOT, 'GolemQ')
OUT = os.path.join(ROOT, 'API_INDEX.md')

#: 不收录的目录 —— 第三方 vendored 代码与缓存
SKIP_DIRS = {'__pycache__', 'easyquotation', 'cookbooks'}


def _first_line(doc: str | None) -> str:
    """取 docstring 第一段的第一行（跳过全为空的前导行）。"""
    if not doc:
        return ''
    for line in doc.strip().splitlines():
        line = line.strip()
        if line:
            # 去掉 markdown 加粗与反引号，索引里保持纯文本更省 token
            return line.replace('**', '').replace('`', '')
    return ''


def _has_doctest(doc: str | None) -> bool:
    return bool(doc) and '>>>' in doc


def _public(node) -> bool:
    return not node.name.startswith('_')


def _signature(node) -> str:
    """尽力还原一行签名。失败则退回名称 —— 索引里宁可少也不要错。"""
    try:
        args = [a.arg for a in node.args.args if a.arg not in ('self', 'cls')]
        return f"({', '.join(args)})"
    except Exception:                                    # noqa: BLE001
        return '(…)'


def scan_module(path: str, modname: str) -> list:
    """抽取一个模块的信息。语法错误时返回一条显式的错误行，不静默跳过。"""
    try:
        src = open(path, encoding='utf-8').read()
        tree = ast.parse(src)
    except Exception as exc:                             # noqa: BLE001
        return [('!', modname, f'解析失败: {exc!r}', False)]

    rows = [('M', modname, _first_line(ast.get_docstring(tree)),
             _has_doctest(ast.get_docstring(tree)))]

    for node in tree.body:
        if isinstance(node, ast.ClassDef) and _public(node):
            doc = ast.get_docstring(node)
            rows.append(('C', node.name, _first_line(doc), _has_doctest(doc)))
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)) and _public(sub):
                    d = ast.get_docstring(sub)
                    rows.append(('m', f'{node.name}.{sub.name}{_signature(sub)}',
                                 _first_line(d), _has_doctest(d)))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and _public(node):
            doc = ast.get_docstring(node)
            rows.append(('f', f'{node.name}{_signature(node)}',
                         _first_line(doc), _has_doctest(doc)))
    return rows


def render() -> str:
    lines = [
        '# GolemQ API 索引',
        '',
        '> **本文件由 `tools/gen_api_index.py` 自动生成，请勿手工编辑。**',
        '> 它是**导航索引**，不是文档 —— 只出「名称 + 首行摘要」。',
        '> 完整说明在各模块的 docstring 与 `PITFALLS.md` / `DECISIONS.md` / `GLOSSARY.md`。',
        '',
        '`[dt]` = 该 docstring 含 doctest，由 `GolemQ/test_cases/test_doctests.py` 收集执行。',
        '',
    ]

    for dirpath, dirnames, filenames in os.walk(PKG):
        dirnames[:] = sorted(d for d in dirnames
                             if d not in SKIP_DIRS and not d.startswith('.'))
        pyfiles = sorted(f for f in filenames
                         if f.endswith('.py') and f != '__init__.py')
        if not pyfiles:
            continue

        rel = os.path.relpath(dirpath, ROOT).replace(os.sep, '.')
        lines.append(f'## {rel}')
        lines.append('')

        for fname in pyfiles:
            full = os.path.join(dirpath, fname)
            modname = f'{rel}.{fname[:-3]}'
            rows = scan_module(full, modname)
            mod_row = rows[0]
            lines.append(f'### `{fname[:-3]}`')
            if mod_row[2]:
                lines.append(f'{mod_row[2]}')
            lines.append('')
            body = [r for r in rows[1:] if r[0] in ('C', 'f', 'm', '!')]
            if not body:
                lines.append('*(无公开成员)*')
                lines.append('')
                continue
            lines.append('| | 名称 | 摘要 |')
            lines.append('|:--|:--|:--|')
            for kind, name, summary, dt in body:
                mark = 'C' if kind == 'C' else ('f' if kind in ('f', 'm') else '!')
                dt_flag = ' `[dt]`' if dt else ''
                lines.append(f'| {mark} | `{name}` | {summary}{dt_flag} |')
            lines.append('')

    return '\n'.join(lines).rstrip() + '\n'


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--check', action='store_true',
                    help='只检查索引是否过期；过期返回非零')
    args = ap.parse_args()

    content = render()

    if args.check:
        current = open(OUT, encoding='utf-8').read() if os.path.exists(OUT) else ''
        if current != content:
            print('API_INDEX.md 已过期，请运行: python tools/gen_api_index.py')
            return 1
        print('API_INDEX.md 与代码同步')
        return 0

    with open(OUT, 'w', encoding='utf-8') as fh:
        fh.write(content)
    n_lines = content.count('\n')
    n_mods = content.count('\n### ')
    print(f'已写入 {OUT}（{n_lines} 行，{n_mods} 个模块）')
    return 0


if __name__ == '__main__':
    sys.exit(main())
