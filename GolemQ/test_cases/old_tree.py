# coding:utf-8
"""载入 `GolemQ_old/` 里任意模块的**公用 harness** —— 对拍用例共用。

为什么抽出来
============
`test_stages` / `test_magic_nine` / `test_poly9` 各写一遍
「注入替身 → 按命名空间包载入老模块 → 用完还原」—— 那就是**又一份平行实现**，
而这个项目最恨的正是它（`PITFALLS.md`：「平行实现不会报错，**只会分叉**」）。
三处踩出来的坑各不相同，但**机制是同一套**，抽在这里。

三条纪律（每条都是实测踩出来的，别省）
====================================
1. **必须还原 `sys.modules`。** 替身留着会污染同进程后续用例 ——
   `test_renko` 实测：留下的 `QUANTAXIS*` 让 `test_no_quantaxis` 当场红。
2. **真模块只能「补属性」，不能替换。** 拿替身去覆盖
   `GolemQ.analysis.timeseries` 会**打断 `StockCN` 注册**（`test_renko` 实测）。
   但**有些老模块就是要那几个新树没有的名字** —— 那种情况得显式列进 `override`，
   并接受"这段时间内它是替身"。
3. **常量表用老树自己那份。** 老模块的**模块体**里就在用 `FLD.ZEN_PEAK_TIMING_LAG`
   这类新树没有的名字，而 `_StubMeta` 早就不伪造（`MIGRATION_STATUS` #3）⇒
   喂新树常量会当场 `AttributeError`。

怎么用
======
    from .old_tree import load_old, old_tree_available
    _OLD = load_old('analysis/stages.py')        # 老树不在场返回 None
    ...
    @unittest.skipIf(_OLD is None, '老树不在场，对拍整份跳过')

⚠️ 本模块名以 `_` 开头**不是**为了私有，是为了避开 `unittest` 的 `test*.py` 发现规则
（它不是用例，不该被收集）。
"""
from __future__ import annotations

import importlib
import os
import sys
import types

from GolemQ.core import constants as _C

OLD_TREE = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..', 'GolemQ_old'))

#: 老模块常要、而新树里**也是真模块**的那几个 —— 默认不覆盖。
#: 需要覆盖的调用方显式传 `override=(...)`，见模块 docstring 第 2 条。
_SKIP_PREFIXES = ('GolemQ.', 'QUANTAXIS')

_MISSING_MSG = '老树 GolemQ_old/ 不在场，对拍整份跳过'


def old_tree_available() -> bool:
    """`GolemQ_old/` 在不在。用例拿它做 `skipIf`。"""
    return os.path.isdir(OLD_TREE)


def _auto_shim(name: str) -> types.ModuleType:
    """惰性 no-op 模块：`from X import Y` 拿到一个什么都不做的函数。

    ⚠️ **`__path__` 必须置空**：替身若只是个普通模块，`import GolemQ.indices.base`
    会报 `'GolemQ.indices' is not a package`（父模块没有 `__path__`，子模块查找
    直接失败 —— **元路径查找器根本没机会被问**）。置成 `[]` 之后它就是个
    **空命名空间包**，子模块查找会正常走到 :class:`_ShimFinder`。
    """
    m = types.ModuleType(name)
    m.__path__ = []
    return m


class _ShimFinder:
    """**任何**缺失的 `GolemQ.*` / `QUANTAXIS*` 子模块 → 自动 no-op 替身。

    逐个往 `sys.modules` 里塞太脆：老模块有 12+ 个顶层 import，模块体里还夹着
    `from GolemQ.czsc import …` 这种 —— 漏一个就 `ModuleNotFoundError`。
    一个元路径查找器把这些一次兜住。

    ⚠️ 只在 :func:`load_old` 的**那一段**装上，用完立刻摘。
    """

    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(_SKIP_PREFIXES):
            return importlib.machinery.ModuleSpec(fullname, _ShimLoader())
        return None


class _ShimLoader:
    def create_module(self, spec):
        return _auto_shim(spec.name)

    def exec_module(self, module):
        pass


class _Missing:
    """「这个属性本来就没有」的哨兵 —— 与"原来就是 `None`"区分开。"""


_MISSING = _Missing()


def load_old(relpath: str, override=(), patch_attrs=None):
    """按**命名空间包**载入 `GolemQ_old/<relpath>`，绕开老树的 `__init__.py`。

    :param relpath: 相对 `GolemQ_old/` 的路径，如 `'analysis/stages.py'`
    :param override: 要**用替身顶替**的模块名（新树里也有真模块、但名字对不上时）。
        例：老 `features/base.py` 从 `GolemQ.analysis.timeseries` 拿 7 个名字，
        而新树那个模块只有 1 个 ⇒ 只能顶替。
        ⚠️ 顶替是**下策**：新树里那些名字**有真实现**时，顶替就等于**静默换成了
        no-op** —— 老代码一调就炸（`.astype(None)` 之类），而且报错点离原因很远。
    :param patch_attrs: ``{'模块名': {'属性名': 值}}`` —— **在真模块上补属性**，
        加载完**还原**。这是模块 docstring 第 2 条那个"**补属性，不替换**"的入口，
        新树里那个名字**有真实现**时该用它（如 `timing.Timeline_duration`）。
    :returns: 老模块对象；`GolemQ_old/` 不在场时返回 ``None``
    """
    path = os.path.join(OLD_TREE, relpath)
    if not os.path.isfile(path):
        return None

    # ⚠️ **先导入真的第三方**，再记 `before` —— 否则它们会被当成「本次加载新增」
    # 在 finally 里删出 `sys.modules`；后续 `import polars` 会拿到**第二个模块实例**，
    # 与已缓存的 `pl.Expr` / `DataFrame` 混用会在 `df.select(exprs)` 处炸
    # （实测：全量跑时 `test_kline_save` 5 条 error）。
    for name in ('talib', 'polars'):
        if name not in sys.modules:
            try:
                importlib.import_module(name)
            except Exception:            # noqa: BLE001 没装就留给下面的 shim
                pass

    before = set(sys.modules)
    replaced = {}
    patched = []
    finder = _ShimFinder()
    sys.meta_path.insert(0, finder)
    try:
        # 「补属性，不替换」—— 先做，因为它 import 的是**真模块**
        for modname, attrs in (patch_attrs or {}).items():
            m = importlib.import_module(modname)
            for attr, val in attrs.items():
                prev = getattr(m, attr, _MISSING)
                patched.append((m, attr, prev))
                setattr(m, attr, val)
        # 第三方里可能有老树要、本机没装的（如 QUANTAXIS 生态）—— 一并兜住。
        # ⚠️ **装了就用真的**：无条件替身会让 `statsmodels`（旧 timeseries 导入）
        # 看到 `polars` 空模块 ⇒ `AttributeError: module 'polars' has no attribute
        # 'DataFrame'`，`test_poly9` 单独跑就挂。
        for name in ('talib', 'polars'):
            if name not in sys.modules:
                try:
                    importlib.import_module(name)
                except Exception:
                    replaced[name] = None
                    sys.modules[name] = _auto_shim(name)

        for name in override:
            replaced[name] = sys.modules.get(name)
            sys.modules[name] = _auto_shim(name)

        # ⚠️ 老树的常量表：`utils/const.py` 是 `parameter.py` / `models/alias.py`
        # 唯一依赖，**不能给替身** —— `parameter.py` 里是 `class AKA(_const)`，
        # `_const` 是个**基类**，替身会返回函数 ⇒ `TypeError: function() argument
        # 'code' must be code, not str`。
        _mount_old_constants(replaced)

        pkgname = '_old_' + relpath.replace('/', '_').replace('.py', '')
        pkg = types.ModuleType(pkgname)
        pkg.__path__ = [os.path.dirname(path)]
        sys.modules[pkgname] = pkg
        return importlib.import_module(
            pkgname + '.' + os.path.basename(path)[:-3])
    finally:
        for m, attr, prev in reversed(patched):
            if prev is _MISSING:
                try:
                    delattr(m, attr)
                except AttributeError:
                    pass
            else:
                setattr(m, attr, prev)
        try:
            sys.meta_path.remove(finder)
        except ValueError:
            pass
        for name, prev in replaced.items():
            if prev is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = prev
        for name in [n for n in list(sys.modules) if n not in before]:
            if name.startswith(_SKIP_PREFIXES) or name in ('talib', 'polars') \
                    or name.startswith('_old_'):
                del sys.modules[name]


class _FieldCompat:
    """**新树常量优先、老树常量兜底**的 `FLD` 代理 —— 喂给老模块用。

    为什么需要它（两个方向都会炸，各踩过一次）
    ========================================
    * **只用新树常量**：老模块的**模块体**里就在用新树没有的名字
      （`ZEN_PEAK_TIMING_LAG` 之类）⇒ `AttributeError`。
    * **只用老树整张表**：老函数写出的**列名**是新树的代码读不到的 ——
      实测 `GARIDENT_PRICE_COUNT` 老表是 `'GAR_P_COUNT'`、新树是 `'garPCount'`
      ⇒ 老函数写一列、用例读另一列，`KeyError`。

    所以：**先问新树，问不到再问老树**。于是
    * 两边**都有的**列 → 用新树的值 ⇒ **对拍两边同名**（值层面的比对才有意义）；
    * 只有老树有的（如已删的 `STOCK_TREND_*`）→ 拿老值，老代码照样跑得动。
    """

    def __init__(self, old_field):
        self._old = old_field

    def __getattr__(self, name):
        try:
            return getattr(_C.FIELD, name)
        except AttributeError:
            old = self._old
            if old is None:
                raise
            return getattr(old, name)


_FIELD_COMPAT = None


def field_compat():
    """取（并缓存）那个兼容代理。老树不在场时退化成"只用新树常量"。"""
    global _FIELD_COMPAT
    if _FIELD_COMPAT is None:
        old_param = load_old('utils/parameter.py')
        _FIELD_COMPAT = _FieldCompat(
            getattr(old_param, 'INDICATOR_FIELD', None) if old_param else None)
    return _FIELD_COMPAT


def _mount_old_constants(replaced: dict):
    """把老树自己的 `utils/const.py` / `utils/parameter.py` / `models/alias.py`
    挂到 `GolemQ.*` 名下 —— 见模块 docstring 第 3 条。

    ⚠️ **是无条件顶替**：`GolemQ.models.alias` 在新树里**也是真模块**（`models/alias.py`），
    所以不能写 `if name in sys.modules: continue` —— 那样会静默用到新树的 alias。
    原对象一律记进 `replaced`，由 `load_old` 的 `finally` 还原。
    """
    for pkgname, sub in (('_oldutils', 'utils'), ('_oldmodels', 'models')):
        if pkgname not in sys.modules:
            m = types.ModuleType(pkgname)
            m.__path__ = [os.path.join(OLD_TREE, sub)]
            sys.modules[pkgname] = m
    for name, src in (('GolemQ.utils.const', '_oldutils.const'),
                      ('GolemQ.utils.parameter', '_oldutils.parameter'),
                      ('GolemQ.models.alias', '_oldmodels.alias')):
        prev = sys.modules.get(name)
        try:
            sys.modules[name] = importlib.import_module(src)
        except Exception:               # noqa: BLE001 常量表载不进来就不强求
            if prev is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = prev
            continue
        replaced[name] = prev
