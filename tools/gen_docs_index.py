#!/usr/bin/env python3
"""生成 `DECISIONS.md` —— 决策**索引表**（正文在 `docs/decisions/`）。

为谁而做
========
与 `gen_api_index.py` 同一个理由：**为「接手项目的下一个会话」而做**。
但这里的病不是"找不到东西"，而是**单文件只增不减**。

`DECISIONS.md` 是 ADR（架构决策记录）日志 —— 按定义**只增不减**。
46 条、每条 30–80 行（背景 / 弃案 / 实测证据），到 2026-10-15 已 2,200 行，
而项目**才刚搭完脚手架**。单文件装 ADR 到 100 条必崩，且崩的方式很难受：
**每读一条决策都要把整份日志读进来**。

故拆成 **一份决策一个文件**（`docs/decisions/D45.md`），本文件退化成一个
~50 行的索引表 —— 它回答"有哪几条决策、各自在哪"，答案要**一眼扫完**。

⚠️ **编号（`D44`）是稳定键，不许重排**
====================================
全树代码注释写的是「见 `DECISIONS.md` D44」这种**编号引用**（grep 过几十处）。
所以：**文件名 = 编号**（`D44.md`），编号一旦发布就**永不改动**。
本表由编号排序生成，不做任何重排或归档 —— 迁移时只搬文件，不改号。

用法
====
    python tools/gen_docs_index.py            # 写 DECISIONS.md
    python tools/gen_docs_index.py --check    # 只检查是否已过期（提交前用）

`--check` 返回非零表示索引与 `docs/decisions/` 不同步 —— 索引会腐烂，
**必须能被发现**（同 `gen_api_index.py`）。
"""
from __future__ import annotations

import argparse
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEC_DIR = os.path.join(ROOT, 'docs', 'decisions')
OUT = os.path.join(ROOT, 'DECISIONS.md')

#: 生成块的头尾标记 —— 标记之外的内容（前言）由人手写、不覆盖。
BEGIN = '<!-- BEGIN GENERATED -->'
END = '<!-- END GENERATED -->'

_TITLE_RE = re.compile(r'^#\s+(D(\d+))\.\s*(.*?)\s*$')
_DATE_RE = re.compile(r'^-\s+\*\*日期\*\*：\s*(\S*)\s*$')
#: 标题尾部的日期（`—— 2026-10-10` / `--- 2026-10-10`），索引里由「日期」列承担
_TITLE_TAIL_DATE_RE = re.compile(r'\s*[—-]{2,}\s*20\d\d-\d\d-\d\d\s*$')


def _read_header(path: str) -> tuple[str, str, str]:
    """决策文件 → (编号, 标题, 日期)。日期缺失时返回 `''`，索引里显示 `—`。

    ⚠️ **日期不许瞎猜**：D1–D32 那批原始标题里**根本没有日期**（那是后来才加的
    写法）。猜一个出来会让索引看起来比实际更权威 —— 缺就是缺。
    """
    num = title = date = ''
    with open(path, encoding='utf-8') as fh:
        for line in fh:
            if not num:
                m = _TITLE_RE.match(line.rstrip('\n'))
                if m:
                    num, title = m.group(1), m.group(3)
                    continue
            if num and not date:
                m = _DATE_RE.match(line.rstrip('\n'))
                if m:
                    date = m.group(1)
                    continue
            # 标题与日期之后的行不用再扫
            if num and date:
                break
    return num, title, date


def collect() -> list[tuple[int, str, str, str, str]]:
    """扫 `docs/decisions/` → [(序号, 编号, 标题, 日期, 文件名)]，按编号排序。"""
    rows = []
    for name in os.listdir(DEC_DIR):
        if not (name.startswith('D') and name.endswith('.md')):
            continue
        num, title, date = _read_header(os.path.join(DEC_DIR, name))
        if not num:
            print(f'⚠️  {name}: 找不到 `# D<n>. 标题` 头行 —— 已跳过', file=sys.stderr)
            continue
        rows.append((int(num[1:]), num, title, date or '—', name))
    rows.sort()
    return rows


def render(rows) -> str:
    """索引表。**只管"有哪几条、在哪"** —— 摘要一律不放（放了就又会长回来）。"""
    out = [BEGIN, '',
           '| 编号 | 决策 | 日期 | 正文 |',
           '|:--|:--|:--|:--|']
    for _, num, title, date, name in rows:
        # 标题尾部的 `—— 2026-10-10` 是**日期**，表里已有独立一列 ——
        # 不剥掉就是同一件事写两遍（且两列将来会对不上）。**只改显示，不改文件**。
        shown = _TITLE_TAIL_DATE_RE.sub('', title) or title
        out.append('| `{num}` | {title} | {date} | [decisions/{name}](docs/decisions/{name}) |'
                   .format(num=num, title=shown, date=date, name=name))
    out += ['', END]
    return '\n'.join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--check', action='store_true',
                    help='只检查是否已过期，不写文件（过期返回 1）')
    args = ap.parse_args(argv)

    if not os.path.isdir(DEC_DIR):
        print(f'❌ {DEC_DIR} 不存在', file=sys.stderr)
        return 1

    rows = collect()
    block = render(rows)

    with open(OUT, encoding='utf-8') as fh:
        cur = fh.read()

    if BEGIN not in cur or END not in cur:
        print(f'❌ {OUT} 里找不到生成标记 {BEGIN} / {END}', file=sys.stderr)
        return 1

    head, rest = cur.split(BEGIN, 1)
    _, tail = rest.split(END, 1)
    new = head + block + tail

    if args.check:
        if new != cur:
            print(f'❌ {OUT} 已过期 —— 跑 `python tools/gen_docs_index.py` 更新')
            return 1
        print(f'✅ {OUT} 与 docs/decisions/ 同步（{len(rows)} 条决策）')
        return 0

    if new == cur:
        print(f'✅ {OUT} 无变化（{len(rows)} 条决策）')
        return 0
    with open(OUT, 'w', encoding='utf-8') as fh:
        fh.write(new)
    print(f'✅ {OUT} 已更新（{len(rows)} 条决策）')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
