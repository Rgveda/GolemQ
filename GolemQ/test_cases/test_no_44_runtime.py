# coding: utf-8
"""守卫：`GolemQ/` 与 `examples/` **零 4.4 依赖**（`DECISIONS.md` **D83**）。

用户 2026-11-18：**除 `tools/` 类脚本外，运行时代码必须与 4.4 旧库完全解耦。**
允许留在运行时的只有文档里对 4.4 的**历史说明**；不允许出现：

* `GolemQ.core.migrate44` 这个模块（已移到 `tools/migrate44.py`）；
* 连接 4.4 的符号（`db44` / `client44` / `mongo44_uri`）；
* 4.4 的库/集合名（`golemqq` / `stock_ta_reviews` / `golemq` 作为 DB 名）。

⚠️ 判据是 **AST**，不是 grep —— 文档字符串里写「4.4 曾……」是允许的。
"""
from __future__ import annotations

import ast
import os
import unittest

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_ROOTS = ('GolemQ', 'examples')
_FORBIDDEN_MODULES = ('GolemQ.core.migrate44', 'core.migrate44', 'migrate44')
_FORBIDDEN_NAMES = ('db44', 'client44', 'mongo44_uri')


def _py_files():
    for root in _ROOTS:
        base = os.path.join(_REPO, root)
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = [d for d in dirnames
                           if d not in ('__pycache__', 'GolemQ_old')]
            for fn in filenames:
                if fn.endswith('.py'):
                    yield os.path.join(dirpath, fn)


def _violations(path):
    with open(path, encoding='utf-8') as fh:
        src = fh.read()
    tree = ast.parse(src, filename=path)
    bad = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if any(alias.name == m or alias.name.startswith(m + '.')
                       for m in _FORBIDDEN_MODULES):
                    bad.append('import {}'.format(alias.name))
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ''
            if node.level:          # 相对导入不指向 4.4 通道
                continue
            if any(mod == m or mod.startswith(m + '.') for m in _FORBIDDEN_MODULES):
                bad.append('from {} import ...'.format(mod))
            for alias in node.names:
                if alias.name in _FORBIDDEN_NAMES:
                    bad.append('from {} import {}'.format(mod, alias.name))
        elif isinstance(node, ast.Name) and node.id in _FORBIDDEN_NAMES:
            bad.append('name {}'.format(node.id))
        elif isinstance(node, ast.Attribute) and node.attr in _FORBIDDEN_NAMES:
            bad.append('attr .{}'.format(node.attr))
    return bad


class TestNo44Runtime(unittest.TestCase):
    def test_bridge_module_gone(self):
        self.assertFalse(os.path.exists(
            os.path.join(_REPO, 'GolemQ', 'core', 'migrate44.py')))
        self.assertTrue(os.path.exists(
            os.path.join(_REPO, 'tools', 'migrate44.py')))

    def test_runtime_zero_44(self):
        fails = {}
        for path in _py_files():
            bad = _violations(path)
            if bad:
                fails[os.path.relpath(path, _REPO)] = bad
        self.assertEqual(fails, {}, '运行时代码仍依赖 4.4：{}'.format(fails))


if __name__ == '__main__':
    unittest.main()
