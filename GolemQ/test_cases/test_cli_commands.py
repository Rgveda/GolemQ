#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""`cli/commands/` 的注册表契约。

为什么值得测：拆 `cli/__main__.py` 时把「哪条命令先判」从 `elif` 的**书写顺序**
搬进了 `commands/__init__.py` 的**显式列表** —— 那是**语义**，不是排版：
`--save tdx --save-coverage` 同时给时必须选 coverage（与老链一致）。
`FLAGS` 打错一个字母则 `matches` **永远不命中**，而命令会静默变成"打了没反应"。
这两件事在真实 CLI 上都不报错，只能靠测试钉住。
"""

import os
import sys

try:
    import GolemQ  # noqa: F401
except ImportError:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import contextlib
import io
import unittest
import unittest.mock          # `unittest.mock` 不随 `import unittest` 自动可用

from GolemQ.cli.__main__ import build_parser
from GolemQ.cli.commands import COMMANDS
from GolemQ.cli.commands import _registry
from GolemQ.cli.commands._registry import Command


class TestCommandRegistry(unittest.TestCase):
    def test_build_parser_succeeds(self):
        """每个命令的 `add_arguments` 各注册一次 —— 有重复 argparse 会直接抛。

        这条同时防「两条命令共用一个 `add_arguments`」（会让同一个开关注册多次）。
        """
        parser = build_parser()
        self.assertIsNotNone(parser)

    def test_every_flag_is_a_real_argparse_dest(self):
        """`FLAGS` 里的名字必须是 parser 真会产出的属性 —— 打错一个字母
        就是**静默永不命中**（命令打了没反应，也不报错）。"""
        args = build_parser().parse_args([])
        for cmd in COMMANDS:
            for flag in cmd.flags:
                with self.subTest(command=cmd.name, flag=flag):
                    self.assertTrue(hasattr(args, flag),
                                    '{} 的 FLAGS 里 {!r} 不是 parser 的属性'.format(
                                        cmd.name, flag))

    def test_command_names_are_unique(self):
        names = [c.name for c in COMMANDS]
        self.assertEqual(len(names), len(set(names)), '命令名重复: {}'.format(names))

    def test_save_coverage_is_dispatched_before_save(self):
        """⚠️ **顺序即语义**：两个开关同时给时，老链选 coverage。"""
        names = [c.name for c in COMMANDS]
        self.assertLess(names.index('save-coverage'), names.index('save'))


class TestMatchSemantics(unittest.TestCase):
    """`matches` 的判据是**真值** —— 等价老链的 `elif args.X:`。"""

    def test_default_is_truthiness(self):
        cmd = Command('t', ('x',), lambda p: None, lambda a: None)
        self.assertFalse(cmd.matches(_ns(x=False)))
        self.assertFalse(cmd.matches(_ns(x=None)))
        self.assertFalse(cmd.matches(_ns(x='')))
        self.assertTrue(cmd.matches(_ns(x=True)))
        self.assertTrue(cmd.matches(_ns(x='tdx')))

    def test_none_does_not_match(self):
        cmd = Command('t', ('x',), lambda p: None, lambda a: None)
        self.assertFalse(cmd.matches(_ns(x=None)))


class TestSaveSourceValidation(unittest.TestCase):
    """`--save` 的取值校验**只有一处**：argparse 的 `choices=`。

    ⚠️ 曾经是手写的 `if value not in (...)` + `exit 1`。用户 2026-10-09 明确要
    argparse 那套 —— 标准 usage + `invalid choice`（退出码 2），可选值还会自动
    进 `--help`。**别在 `run_save` 里再写一遍**：两处都写 = 两个真相源。
    """

    def test_invalid_source_rejected_by_argparse(self):
        # 挡住 argparse 写到 stderr 的 usage，免得淹掉测试输出
        buf = io.StringIO()
        with contextlib.redirect_stderr(buf), self.assertRaises(SystemExit) as cm:
            build_parser().parse_args(['--save', 'tdxx'])
        self.assertEqual(cm.exception.code, 2)          # argparse 的用法错误码

    def test_empty_source_rejected_by_argparse(self):
        buf = io.StringIO()
        with contextlib.redirect_stderr(buf), self.assertRaises(SystemExit) as cm:
            build_parser().parse_args(['--save', ''])
        self.assertEqual(cm.exception.code, 2)

    def test_bare_save_means_tdx(self):
        self.assertEqual(build_parser().parse_args(['--save']).save, 'tdx')

    def test_uppercase_is_lowered_before_the_choices_check(self):
        """老代码是 `str(args.save).lower()` → `--save TDX` 能用。
        `type=str.lower` 在 choices 校验**之前**生效，故行为不变。"""
        self.assertEqual(build_parser().parse_args(['--save', 'TDX']).save, 'tdx')

    def test_absent_save_is_none(self):
        self.assertIsNone(build_parser().parse_args([]).save)


class TestExitCodeConvention(unittest.TestCase):
    """退出码口径：**用法/参数错 = 2**、**运行期失败 = 1**。

    分界线是「**调用错了**」还是「**跑起来没成**」，不是严重程度。
    实测过混用（用户 2026-10-09 指出）：`--save tdxx` 给 2 而 `--purge-l1 tdx` 给 1
    —— 同一类错误在 CLI 上有两种表现。统一之后，argparse 表达不了的那几处
    （值来自**运行期注册表**：`--sub` 的键、`--save-collections` 的集合名）
    也走 :func:`usage_error`，报成**同一个形状**。
    """

    def setUp(self):
        self._saved = _registry._PARSER
        self.addCleanup(lambda: setattr(_registry, '_PARSER', self._saved))

    def test_constants(self):
        self.assertEqual(_registry.EXIT_USAGE, 2)
        self.assertEqual(_registry.EXIT_FAILURE, 1)

    def test_usage_error_exits_2_in_argparse_shape(self):
        _registry.bind_parser(build_parser())
        buf = io.StringIO()
        with contextlib.redirect_stderr(buf), self.assertRaises(SystemExit) as cm:
            _registry.usage_error('argument --x: bad', hint='提示: 一句话')

        self.assertEqual(cm.exception.code, _registry.EXIT_USAGE)
        out = buf.getvalue()
        self.assertIn('usage:', out)              # 与 argparse 同形
        self.assertIn('error: argument --x: bad', out)
        self.assertIn('提示: 一句话', out)

    def test_usage_error_without_bound_parser_still_exits_2(self):
        """没 `bind_parser` 过也不能炸 —— 退化成不打 usage 行，码不变。"""
        _registry._PARSER = None
        with contextlib.redirect_stderr(io.StringIO()), \
                self.assertRaises(SystemExit) as cm:
            _registry.usage_error('boom')
        self.assertEqual(cm.exception.code, _registry.EXIT_USAGE)


def _ns(**kw):
    """最小 Namespace —— 只带测试要看的属性。"""
    class _N:
        pass
    n = _N()
    for k, v in kw.items():
        setattr(n, k, v)
    return n


if __name__ == '__main__':
    unittest.main()


class TestBlockingCallsLightUpTheBanner(unittest.TestCase):
    """**阻塞调用之前必须先点 `RUNNING`** —— 否则那一段屏上是个灰点。

    为什么值得测（2026-10-09 实报）：复权那四个节点原先**只在跑完时** `mark(DONE)`，
    而 `save_xdxr_tdx` / `save_adj` 是**阻塞**调用 —— 整段（几千只 × ~0.3s，分钟级）
    屏上一直停在 `·`，**看着像没在动**。K线（靠 `on_progress` 的 `'start'`）与
    参考数据都有绿点，唯独复权漏了。

    这个缺陷**不报错、不影响结果**，只让人以为卡住了 —— 所以只能靠结构钉住：
    在源码里，每个阻塞的 `save_*` 调用之前都要有一次 `mark(..., RUNNING)`。
    用 `ast` 而不是正则：按**调用行号**判先后，不受注释与换行影响。
    """

    SOURCE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          'cli', 'commands', 'save.py')
    #: 阻塞的取数函数 —— 每个都要先点亮它对应的节点。
    BLOCKING = ('save_kline_tdx', 'save_xdxr_tdx', 'save_adj')

    def _calls(self):
        import ast
        with open(self.SOURCE, encoding='utf-8') as fh:
            tree = ast.parse(fh.read())
        out = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            f = node.func
            name = f.id if isinstance(f, ast.Name) else getattr(f, 'attr', None)
            out.append((name, node.lineno, node))
        return out

    @staticmethod
    def _mentions(call, state):
        """该调用（含**嵌套**表达式）里有没有出现 `state` 这个名字。

        ⚠️ 必须**递归**找：实际写法是 ``banner.mark(nm, RUNNING if phase == 'start'
        else DONE)`` —— `RUNNING` 在 `IfExp` 里面，不在 `args` 顶层。只查顶层的
        第一版把两个回调全漏了（实测）。
        """
        import ast
        for arg in call.args:
            for sub in ast.walk(arg):
                if isinstance(sub, ast.Name) and sub.id == state:
                    return True
        return False

    def _marks(self, calls, state):
        """所有点了 `state` 的 `mark(...)` 的行号。"""
        return sorted(line for name, line, node in calls
                      if name == 'mark' and self._mentions(node, state))

    @staticmethod
    def _marks_inside_on_progress(node):
        """该调用**自己**有没有在 `on_progress=` 里点 RUNNING。

        `save_kline_tdx` 走的就是这条：它的绿点由回调 `on_progress` 的
        `'start'` 相位点亮，前面没有单独的 `mark` 行。两种写法都算数 ——
        钉的是「这个阻塞段开头有没有绿点」，不是"必须写成哪一种"。
        """
        import ast
        for kw in getattr(node, 'keywords', []):
            if kw.arg != 'on_progress':
                continue
            for sub in ast.walk(kw.value):
                if (isinstance(sub, ast.Call)
                        and getattr(sub.func, 'attr', None) == 'mark'
                        and TestBlockingCallsLightUpTheBanner._mentions(sub, 'RUNNING')):
                    return True
        return False

    def test_every_blocking_call_is_preceded_by_running(self):
        calls = self._calls()
        running = self._marks(calls, 'RUNNING')
        self.assertTrue(running, '一次 RUNNING 都没点 —— 用例失去意义')
        for name, line, node in calls:
            if name not in self.BLOCKING:
                continue
            with self.subTest(call=name, line=line):
                lit = (any(0 < line - r <= 12 for r in running)
                       or self._marks_inside_on_progress(node))
                self.assertTrue(
                    lit,
                    '{}（第 {} 行）之前 12 行内没有 mark(..., RUNNING)，'
                    '它自己的 on_progress 里也没有 —— 那段会停在灰点上'
                    .format(name, line))


class TestXdxrRefreshGate(unittest.TestCase):
    """复权段的 **TTL 闸**（用户 2026-10-09 定，推翻 D25 原先的「只记账不开闸」）。

    复权的闸是**集合级**的 —— 它**没有** K 线那样的逐 code 水位兜底，所以两条要钉：

    1. 闸开 ⇒ **整段跳过**（不建连、不取数）—— 代价是新除权事件最多滞后一个 TTL，
       那段窗口 `{tgt}_adj` 是旧基准；
    2. **只有全宇宙的跑法才记账** —— 一次 `--save-codes 600519` 的小范围跑若也记账，
       就会把闸打开 ⇒ 之后**全量跑整段跳过 xdxr** ⇒ 新除权事件**静默漏掉**。
       （K 线那边其实有逐 code 兜底，但这条规则两边**刻意一致**：别让试跑开闸。）
    """

    def _drive(self, argv, *, gate_open):
        args = build_parser().parse_args(argv)
        calls = {'xdxr': [], 'marked': [], 'out': ''}
        from GolemQ.markets.StockCN import kline_save as ks
        from GolemQ.markets.StockCN import refdata_save as rs
        from GolemQ.cli.commands import save as save_cmd

        def _fake_xdxr(*a, **k):
            calls['xdxr'].append(k.get('target'))
            return {'codes': 0, 'updated': 0, 'events_changed': [], 'errors': []}

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), \
                unittest.mock.patch.object(rs, 'save_refdata', return_value={}), \
                unittest.mock.patch.object(ks, 'save_kline_tdx',
                                           return_value={'kline': {}, 'universe': {}}), \
                unittest.mock.patch.object(ks, 'save_xdxr_tdx', side_effect=_fake_xdxr), \
                unittest.mock.patch.object(ks, 'kline_sweep_age_hours', return_value=1.5), \
                unittest.mock.patch.object(ks, 'allow_xdxr_shortcircuit',
                                           return_value=gate_open), \
                unittest.mock.patch.object(ks, 'mark_kline_sweep',
                                           side_effect=lambda n: calls['marked'].append(n)):
            save_cmd.run_save(args)
        calls['out'] = buf.getvalue()
        return calls

    def test_gate_message_only_under_verbose(self):
        """⚠️ 用户 2026-10-09：刷新闸那句话**只在 `-v` 下打**。

        与参考数据的刷新闸、K线的「整段跳过」**同一口径** —— 闸的细节是排障信息，
        不是每次运行都要看的东西（`age` 也只在 `-v` 下才算）。
        第一版漏了这个门，用户实跑看到了两行 `[xdxr] … 刷新闸：…`。
        """
        quiet = self._drive(['--save', 'pytdx'], gate_open=True)
        self.assertNotIn('刷新闸', quiet['out'])
        loud = self._drive(['--save', 'pytdx', '-v'], gate_open=True)
        self.assertIn('刷新闸', loud['out'])
        # 两种情况都确实跳过了（只是说不说而已）
        self.assertEqual(quiet['xdxr'], [])
        self.assertEqual(loud['xdxr'], [])

    def test_gate_open_skips_the_whole_pass(self):
        got = self._drive(['--save', 'pytdx'], gate_open=True)
        self.assertEqual(got['xdxr'], [], '闸开了却还是去取数了')
        self.assertEqual(got['marked'], [], '跳过的遍**不许**记账（记了闸就被无限推后）')

    def test_gate_closed_runs_both_targets(self):
        got = self._drive(['--save', 'pytdx'], gate_open=False)
        self.assertEqual(got['xdxr'], ['stock', 'etf'])

    def test_save_refresh_forces_it(self):
        """一个旗子 = 别信缓存、全查一遍（与参考数据闸、K 线短路同一口径）。"""
        got = self._drive(['--save', 'pytdx', '--save-refresh'], gate_open=True)
        self.assertEqual(got['xdxr'], ['stock', 'etf'])

    def test_limited_codes_do_not_open_the_gate(self):
        """⚠️ 这条是**静默丢数据**的防线：试跑不许记账。"""
        got = self._drive(['--save', 'pytdx', '--save-codes', '600519'],
                          gate_open=False)
        self.assertEqual(got['xdxr'], ['stock', 'etf'], '限定 code 仍要取数')
        self.assertEqual(got['marked'], [], '限定 code 的跑法**不许**记账')

    def test_full_universe_marks_the_sweep(self):
        got = self._drive(['--save', 'pytdx'], gate_open=False)
        self.assertEqual(got['marked'], ['stock_xdxr', 'etf_xdxr'])


class TestAdjNodeMarking(unittest.TestCase):
    """`stock_adj` / `etf_adj` 两个节点**什么时候点白**。

    ⚠️ 用户 2026-10-10：「**这两个 `_adj` 跑过了就更新白●更合理**」。

    原先的做法是「**没有事件变化就不点**」，理由写成「点亮等于替没做的事谎报成功」——
    **那是反的**：白点表示「这一步**已确认完成 / 数据是好的**」，而"事件比对通过 ⇒
    `_adj` 已是最新"正是这个状态。同树里**参考数据那条早就这么做了**
    （`cached ⇒ DONE`：「数据是好的、只是没重取，留灰会被读成「没取到」」），
    两处口径必须一致。灰点留给"真的没轮上"。

    但**有一条反向的**：事件**变了**却按 `--save-no-adj` 跳过重算 ⇒ `_adj` 此刻是
    **过期**的（旧因子）—— 那种情况**不许点白**，否则才是真的谎报。
    """

    def _marks(self, events_changed, *, no_adj=False):
        """跑一次 `run_save`，记下每个节点的 `mark` 序列。"""
        args = build_parser().parse_args(['--save', 'pytdx'] + (['--save-no-adj'] if no_adj else []))
        marked = []
        from GolemQ.core import presentation
        from GolemQ.markets.StockCN import kline_save as ks
        from GolemQ.markets.StockCN import refdata_save as rs
        from GolemQ.cli.commands import save as save_cmd
        real = presentation.Banner

        class _Spy(real):
            def mark(self, name, state=None):
                marked.append((name, state))
                return super().mark(name) if state is None else super().mark(name, state)

        with contextlib.redirect_stdout(io.StringIO()), \
                unittest.mock.patch.object(presentation, 'Banner', _Spy), \
                unittest.mock.patch.object(rs, 'save_refdata', return_value={}), \
                unittest.mock.patch.object(ks, 'save_kline_tdx',
                                           return_value={'kline': {}, 'universe': {}}), \
                unittest.mock.patch.object(
                    ks, 'save_xdxr_tdx',
                    side_effect=lambda *a, **k: {'codes': 0, 'updated': 0,
                                                 'events_changed': events_changed,
                                                 'errors': []}), \
                unittest.mock.patch.object(ks, 'save_adj'), \
                unittest.mock.patch.object(ks, 'kline_sweep_age_hours', return_value=1.0), \
                unittest.mock.patch.object(ks, 'allow_xdxr_shortcircuit', return_value=False), \
                unittest.mock.patch.object(ks, 'mark_kline_sweep'):
            save_cmd.run_save(args)
        return marked

    def test_adj_lights_white_when_there_is_nothing_to_do(self):
        from GolemQ.core.presentation import DONE
        marked = self._marks([])                       # 事件没变
        self.assertIn(('stock_adj', DONE), marked, '事件没变 ⇒ _adj 已是最新，该点白')
        self.assertIn(('etf_adj', DONE), marked)

    def test_adj_stays_gray_when_events_changed_but_recompute_skipped(self):
        """⚠️ **反向那条**：事件变了却 `--save-no-adj` ⇒ `_adj` 是**过期**的。

        这时点白才是真的谎报 —— 所以它**必须**留灰。
        """
        from GolemQ.core.presentation import DONE
        marked = self._marks(['600519'], no_adj=True)
        self.assertNotIn(('stock_adj', DONE), marked, '_adj 过期时不许点白')
        self.assertNotIn(('etf_adj', DONE), marked)

    def test_adj_lights_white_after_recomputing(self):
        from GolemQ.core.presentation import DONE
        marked = self._marks(['600519'])               # 事件变了、正常重算
        self.assertIn(('stock_adj', DONE), marked)
        self.assertIn(('etf_adj', DONE), marked)


class TestSaveStageStartDoneLines(unittest.TestCase):
    """`--save` 阶段的**起止两行**（用户 2026-10-10，与 bootstrap 同构）：

    起 ``[t]: saving stock_cn klines``（**在 `数据源` 那一行之前**），
    止 ``[t]: … done.``（灰色仅 TTY，**在 `banner.close()` 之后**）。

    这样日志里 `--save` 这一段**可检索**（起止都能 grep），不是只有一堆状态行。
    """

    def _run(self, argv, *, boom=False):
        args = build_parser().parse_args(argv)
        from GolemQ.core import presentation
        from GolemQ.markets.StockCN import kline_save as ks
        from GolemQ.markets.StockCN import refdata_save as rs
        from GolemQ.cli.commands import save as save_cmd

        side = RuntimeError('模拟中途炸了') if boom else None
        kw = {'side_effect': side} if boom else {'return_value': {}}
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), \
                unittest.mock.patch.object(presentation, 'ansi_enabled', return_value=False), \
                unittest.mock.patch.object(rs, 'save_refdata', **kw), \
                unittest.mock.patch.object(ks, 'save_kline_tdx',
                                           return_value={'kline': {}, 'universe': {}}), \
                unittest.mock.patch.object(ks, 'kline_sweep_age_hours', return_value=1.0):
            try:
                save_cmd.run_save(args)
            except SystemExit:
                pass
        return buf.getvalue()

    @staticmethod
    def _stamped(out, caption):
        """``caption`` 那一行的索引（**整行匹配** —— 起行是止行的前缀，子串会撞）。"""
        import re
        want = re.compile(r'\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\]: ' + caption + r'$')
        return [i for i, ln in enumerate(out.splitlines()) if want.match(ln)]

    def _row(self, out, prefix):
        return [i for i, ln in enumerate(out.splitlines()) if ln.startswith(prefix)]

    def test_start_line_is_before_the_source_row(self):
        out = self._run(['--save', 'tdx'])
        start = self._stamped(out, 'saving stock_cn klines')
        self.assertEqual(len(start), 1,
                         '起行应恰好一行（整行匹配）：{}'.format(out.splitlines()[:6]))
        self.assertLess(start[0], self._row(out, '数据源')[0],
                        '起行必须在 `数据源` **之前**（用户明确）')

    def test_done_line_comes_last(self):
        out = self._run(['--save', 'tdx'])
        done = self._stamped(out, r'saving stock_cn klines done\.')
        self.assertEqual(len(done), 1, '止行应恰好一行')
        self.assertGreater(done[0], self._row(out, '数据源')[0],
                           '止行应在 `数据源` 那一段**之后**')

    def test_caption_matches_what_actually_runs(self):
        """⚠️ `--save qmt` **不取 K 线**（只做参考数据）—— 给它打 "klines" 就是假话。"""
        out = self._run(['--save', 'qmt'])
        self.assertEqual(len(self._stamped(out, 'saving stock_cn refdata')), 1)
        self.assertEqual(self._stamped(out, 'saving stock_cn klines'), [])

    def test_no_done_line_when_it_crashes(self):
        """⚠️ 中途炸了**不许打 `done.`** —— 那一段没 done。

        且报错必须是「**意外终止**」而不是「被用户终止」（`PITFALLS.md` P28）。
        """
        out = self._run(['--save', 'tdx'], boom=True)
        self.assertNotIn('done.', out, '崩了还打 done. 就是谎报')
        self.assertIn('意外终止', out)
        self.assertNotIn('被用户终止', out)

    def test_no_escape_codes_when_not_a_tty(self):
        out = self._run(['--save', 'tdx'])
        self.assertNotIn('\033', out)
