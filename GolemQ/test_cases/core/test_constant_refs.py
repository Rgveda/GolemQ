import os, sys
try:
    import GolemQ
except ImportError:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

import ast
import collections
import importlib
import pathlib
import unittest

_ROOT = pathlib.Path(__file__).resolve().parents[3]
_PKG = _ROOT / 'GolemQ'

#: 常量类 → 所在模块。（新树里这 8 个是列名/别名常量。）
_CLASSES = {
    'FIELD': 'GolemQ.core.constants',
    'FEATURES': 'GolemQ.core.constants',
    'AKA': 'GolemQ.core.constants',
    'STATE': 'GolemQ.core.constants',
    'TREND_STATUS': 'GolemQ.core.constants',
    'TRD': 'GolemQ.models.alias',
    'PFL': 'GolemQ.models.alias',
    'LTT': 'GolemQ.models.alias',
}


def _refs():
    refs = collections.defaultdict(set)
    for f in _PKG.rglob('*.py'):
        if '__pycache__' in f.parts:
            continue
        try:
            tree = ast.parse(f.read_text(encoding='utf-8'))
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if (isinstance(node, ast.Attribute)
                    and isinstance(node.value, ast.Name)
                    and node.value.id in _CLASSES):
                refs[node.value.id].add(node.attr)
    return refs


class TestConstantRefs(unittest.TestCase):
    """`FLD.X` / `AKA.X` 这类引用必须在常量类里真实存在。

    `_StubMeta` 不会伪造属性 —— 引用未定义的名字会**在运行时**才炸，
    而且常常在冷门分支里（例如 ChipDistribution）。这里把它提前到测试期。
    """

    def test_no_undefined_constant_attribute(self):
        refs = _refs()
        missing = {}
        for cls, modname in _CLASSES.items():
            mod = importlib.import_module(modname)
            defined = set(dir(getattr(mod, cls)))
            gone = sorted(n for n in refs.get(cls, ())
                          if n not in defined and not n.startswith('_'))
            if gone:
                missing[cls] = gone
        self.assertEqual(missing, {}, '常量类缺列名：{}'.format(missing))


if __name__ == '__main__':
    unittest.main()
