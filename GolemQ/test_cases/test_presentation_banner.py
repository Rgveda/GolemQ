#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
`core/presentation.py` 的状态 banner —— 纯渲染、显示宽度、光标行数记账、三态。

为什么值得测：banner 靠「光标上移 N 行」原地重画，**行数算错就会把整屏写花**，
而 N 是渲染结果算出来的；另外**状态只由那个点表达**（三态：`·` 队列中 / `●` 绿 正在读取 /
`●` 白 读取完成），节点上不许冒出「已获取/未获取」那种文字 —— 这两条都得钉住。
"""

import datetime
import io
import os
import re
import sys

try:
    import GolemQ  # noqa: F401
except ImportError:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import unittest
import unittest.mock

from GolemQ.core.presentation import (
    Banner,
    DONE,
    PENDING,
    PHASE_WIDTH,
    RUNNING,
    aligned_row,
    ansi_enabled,
    display_width,
    identity,
    render_pipeline_banner,
)

#: 固定时刻 —— 身份块的戳要逐字断言，不能取"现在"
_FIXED = datetime.datetime(2026, 10, 9, 15, 12, 57)

FREQS = ('day', '1min', '5min')


def _rows():
    return [
        ('参考数据', None, ['stock_list', 'stock_info']),
        ('K线', 'stock', ['stock_day', 'stock_1min', 'stock_5min']),
        ('复权', None, ['stock_xdxr', 'stock_adj']),
    ]


class _FakeTty:
    """假终端：`isatty()` 为真，把写进去的东西攒起来。"""

    def __init__(self):
        self.chunks = []

    def isatty(self):
        return True

    def write(self, text):
        self.chunks.append(text)

    def flush(self):
        pass

    def getvalue(self):
        return ''.join(self.chunks)


class TestAnsiEnabled(unittest.TestCase):
    def test_non_tty_is_off(self):
        self.assertFalse(ansi_enabled(io.StringIO()))

    def test_stream_without_isatty_is_off(self):
        class _Bare:
            def write(self, _): pass
        self.assertFalse(ansi_enabled(_Bare()))

    def test_tty_needs_vt_support_too(self):
        """`isatty()` 为真**还不够** —— Windows conhost 默认不解释 ANSI，
        那种终端上开颜色会让「光标上移」失效 → 每次重画变成追加（实测就是这么写出
        两条 `source:` 的）。故还要求 :func:`_vt_supported`。"""
        from GolemQ.core import presentation as p
        with unittest.mock.patch.object(p, '_vt_supported', return_value=False):
            self.assertFalse(ansi_enabled(_FakeTty()))
        with unittest.mock.patch.object(p, '_vt_supported', return_value=True):
            self.assertTrue(ansi_enabled(_FakeTty()))


class TestDisplayWidth(unittest.TestCase):
    def test_cjk_counts_two(self):
        self.assertEqual(display_width('K线'), 3)          # K=1 + 线=2
        self.assertEqual(display_width('参考数据'), 8)

    def test_ascii_counts_one(self):
        self.assertEqual(display_width('stock_day'), 9)


class TestRenderPipelineBanner(unittest.TestCase):
    def test_plain_text_layout(self):
        out = render_pipeline_banner(_rows(), {}, color=False)
        self.assertEqual(out.split('\n'), [
            '参考数据  stock_list ·  stock_info ·',
            'K线       stock  day ·  1min ·  5min ·',
            '复权      stock_xdxr ·  stock_adj ·',
        ])

    def test_repeated_phase_name_is_printed_once(self):
        """连续同阶段名的行只在第一行打阶段名（K线 两行不重复打）。"""
        rows = [('K线', 'stock', ['stock_day']), ('K线', 'index', ['index_day'])]
        lines = render_pipeline_banner(rows, {}, color=False).split('\n')
        self.assertTrue(lines[0].startswith('K线'))
        self.assertTrue(lines[1].startswith(' ' * display_width('K线')))
        self.assertNotIn('K线', lines[1])

    def test_group_prefix_stripped_from_node_label(self):
        out = render_pipeline_banner(_rows(), {}, color=False)
        self.assertIn('stock  day ·', out)
        self.assertNotIn('stock_stock_day', out)

    def test_no_state_words_anywhere(self):
        """状态**只由那个点表达** —— 不许出现「已获取 / 未获取」。"""
        for color in (True, False):
            out = render_pipeline_banner(
                _rows(), {'stock_list': DONE}, color=color)
            self.assertNotIn('已获取', out)
            self.assertNotIn('未获取', out)

    def test_three_states_use_two_symbols_and_three_colors(self):
        """`·`=队列中(灰) / `●`=正在读取(绿) / `●`=读取完成(白)。"""
        out = render_pipeline_banner(
            _rows(),
            {'stock_list': DONE, 'stock_info': RUNNING, 'stock_day': PENDING},
            color=True)
        self.assertIn('\033[97m●\033[0m', out)      # 完成 = 白
        self.assertIn('\033[32m●\033[0m', out)      # 进行中 = 绿
        self.assertIn('\033[90m·\033[0m', out)      # 队列中 = 灰

    def test_only_the_dot_is_coloured(self):
        """**只给点上色** —— 节点名保持默认色（用户口径：「白●点」）。"""
        out = render_pipeline_banner(_rows(), {'stock_list': DONE}, color=True)
        self.assertIn('stock_list \033[97m●\033[0m', out)
        self.assertNotIn('\033[97mstock_list', out)

    def test_missing_key_counts_as_pending(self):
        out = render_pipeline_banner(_rows(), {}, color=False)
        self.assertNotIn('●', out)
        self.assertEqual(out.count('·'), 7)          # 2 + 3 + 2 个节点


class TestBannerNonTty(unittest.TestCase):
    """非 TTY：不上色。**表头只打一次**（= 作业单），之后每次变化补一行 `  名字 符号`。

    不整块重打：26 个节点 × 7 行表头 = 182 行噪声，而日志真正需要的是「走到哪了」。
    """

    def setUp(self):
        self.stream = io.StringIO()
        self.banner = Banner('pytdx', _rows(), stream=self.stream)

    def test_no_escape_codes_at_all(self):
        self.banner.render()
        self.banner.mark('stock_list', DONE)
        self.banner.echo('hello')
        self.assertNotIn('\033', self.stream.getvalue())

    def test_header_printed_at_render(self):
        self.banner.render()
        self.assertIn('参考数据  stock_list ·', self.stream.getvalue())

    def test_mark_appends_one_line_with_symbol(self):
        """表头只打一次，之后每次变化补**一行** —— 不整块重打（26 节点会变 182 行噪声）。"""
        self.banner.render()
        before = len(self.stream.getvalue())
        self.banner.mark('stock_list', DONE)
        out = self.stream.getvalue()[before:]

        self.assertEqual(out, '  stock_list ●\n')          # 就一行，符号即状态
        self.assertNotIn('source: pytdx', out)              # 表头不重复
        self.assertNotIn('已获取', out)

    def test_header_printed_exactly_once(self):
        self.banner.render()
        for name, state in (('stock_list', RUNNING), ('stock_list', DONE),
                            ('stock_info', DONE)):
            self.banner.mark(name, state)
        self.assertEqual(self.stream.getvalue().count('source: pytdx'), 1)

    def test_unchanged_state_is_not_reprinted(self):
        self.banner.render()
        self.banner.mark('stock_list', DONE)
        before = self.stream.getvalue()
        self.banner.mark('stock_list', DONE)                # 同值，无变化
        self.assertEqual(self.stream.getvalue(), before)

    def test_state_can_go_backwards(self):
        """`RUNNING → DONE → PENDING` 都算变化 —— 别把它们当「无变化」早退。"""
        self.banner.render()
        self.banner.mark('stock_list', RUNNING)
        before = len(self.stream.getvalue())
        self.banner.mark('stock_list', DONE)
        self.banner.mark('stock_list', PENDING)
        self.assertIn('stock_list ·', self.stream.getvalue()[before:])

    def test_unknown_name_is_ignored(self):
        self.banner.render()
        before = self.stream.getvalue()
        self.banner.mark('index_day', DONE)                 # 不在表头里
        self.assertEqual(self.stream.getvalue(), before)

    def test_unknown_state_raises(self):
        with self.assertRaises(ValueError):
            self.banner.mark('stock_list', 'finished')

    def test_echo_prints_in_order(self):
        self.banner.render()
        self.banner.echo('[kline] stock_day 完成：写 10 删 6')
        self.assertTrue(self.stream.getvalue().rstrip().endswith(
            '[kline] stock_day 完成：写 10 删 6'))


class TestBannerTty(unittest.TestCase):
    """TTY：原地重画，`source:` 只写一次，回退量 = 上一次画的总行数。"""

    def setUp(self):
        # `ansi_enabled` 现在还要求 VT 支持；假终端在测试进程里拿不到真控制台，
        # 故把那一层挡掉 —— 本类测的是**颜色/光标**那块，不是 VT 探测。
        from GolemQ.core import presentation as p
        patcher = unittest.mock.patch.object(p, '_vt_supported', return_value=True)
        patcher.start()
        self.addCleanup(patcher.stop)

        self.stream = _FakeTty()
        self.banner = Banner('pytdx', _rows(), stream=self.stream, echo_window=2)

    @staticmethod
    def _rewinds(raw):
        return re.findall(r'\033\[(\d+)A', raw)

    def test_first_render_does_not_move_cursor(self):
        self.banner.render()
        self.assertEqual(self._rewinds(self.stream.getvalue()), [])

    def test_redraw_rewinds_by_block_height(self):
        self.banner.render()                       # 区块 3 行（source 在区块之外）
        self.banner.mark('stock_list', RUNNING)
        self.assertEqual(self._rewinds(self.stream.getvalue()), ['3'])

    def test_rewind_tracks_growth_from_echo_window(self):
        """状态窗从 0 行长到 1 行 → 下一次回退量必须跟着变大，否则错位。"""
        self.banner.render()                       # 区块 3 行
        self.banner.echo('一行')                   # 3 + 1 = 4 行
        self.banner.mark('stock_list', DONE)
        self.assertEqual(self._rewinds(self.stream.getvalue()), ['3', '4'])

    def test_echo_window_limits_visible_lines(self):
        self.banner.render()
        for i in range(5):
            self.banner.echo('line-{}'.format(i))
        last = self.stream.getvalue().split('\r')[-1]
        self.assertIn('line-4', last)
        self.assertIn('line-3', last)
        self.assertNotIn('line-2', last)           # 窗口只留 2 行

    def test_close_dumps_full_log(self):
        self.banner.render()
        for i in range(5):
            self.banner.echo('line-{}'.format(i))
        self.banner.close()

        last = self.stream.getvalue().split('\r')[-1]
        for i in range(5):                         # 窗口挤掉的也要补回来
            self.assertIn('line-{}'.format(i), last)

    def test_running_is_green_and_done_is_white(self):
        self.banner.render()
        self.banner.mark('stock_list', RUNNING)
        self.assertIn('\033[32m●\033[0m', self.stream.getvalue())
        self.banner.mark('stock_list', DONE)
        self.assertIn('\033[97m●\033[0m', self.stream.getvalue())


class TestBannerSurvivesAnActiveBar(unittest.TestCase):
    """**banner 与 tqdm 进度条共存**的不变量。

    这是 `--save tdx` 4 worker 跑起来时屏上能不能看的前提。契约是两条：

    1. 进度条**绝不能 `leave=True`** —— 留在屏上，banner 的回退量就不对了
       （`kline_save` / `pytdx_source` 的每一处 `tqdm(...)` 都写了 `leave=False`）；
    2. 进度条**活着的时候**谁也不许另写 stdout —— 多一行，回退量同样错。

    第 2 条由 `test_kline_save.TestNothingPrintsWhileTheBarIsAlive` 钉；
    这里钉第 1 条：**在两次 banner 重画之间开关一个 `leave=False` 的进度条，
    回退量必须不变**。
    """

    def setUp(self):
        from GolemQ.core import presentation as p
        patcher = unittest.mock.patch.object(p, '_vt_supported', return_value=True)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.stream = _FakeTty()
        from GolemQ.core.presentation import Banner
        self.banner = Banner('pytdx', _rows(), stream=self.stream, echo_window=2)

    @staticmethod
    def _rewinds(raw):
        # 用 '\033' 显式拼 + r-string —— **别在源码里放字面 ESC 字节**（不可见，编辑器和 git 都可能弄坏）
        return re.findall('\033' + r'\[(\d+)A', raw)

    def test_rewind_is_unchanged_across_a_closed_bar(self):
        from tqdm import tqdm

        self.banner.render()
        self.banner.mark('stock_list', RUNNING)
        steady = self._rewinds(self.stream.getvalue())

        # 进度条在两次重画之间活着又关掉（leave=False → 自己擦干净）
        before = len(self.stream.getvalue())
        bar = tqdm(total=3, unit='code', disable=False, leave=False,
                   file=self.stream, ncols=40)
        bar.update(1)
        bar.close()
        self.banner.echo('进度条关掉之后吐的诊断')

        raw = self.stream.getvalue()[before:]
        self.assertTrue(raw, '这一段什么都没写，用例失去意义')
        self.assertEqual(self._rewinds(raw)[-1], steady[-1],
                         '进度条开过之后回退量变了 —— 进度条没擦干净，或者期间有人写了 stdout')


if __name__ == '__main__':
    unittest.main()


class TestIdentityBlockAndColumn(unittest.TestCase):
    """开头的**身份块** + 跨 banner 的**阶段列**（2026-10-09 改版）。

    为什么值得测：改版前是「版权行 + `[戳]: bootstrap` + 自检行 + `[戳]: source: pytdx`」
    四行 —— 用户口径「眼花」，毛病是**两个戳紧挨着**（像同一件事说两遍）与**四种行同字重**。
    改版后时刻戳**只出现一次**（在身份行上），阶段名跨两个 banner 对齐成一列。
    这两条都是**排版不变量**，破了不报错、只是变难看，所以只能靠用例钉。
    """

    def test_identity_block_shape(self):
        out = identity('GolemQ', 'Copyright (c) 2018-2026 azai/Rgveda/GolemQ(uant)',
                       when=_FIXED)
        self.assertEqual(out.split('\n'),
                         ['GolemQ  [2026-10-09 15:12:57]',
                          'Copyright (c) 2018-2026 azai/Rgveda/GolemQ(uant)',
                          '', ''])
        # 戳只出现一次 —— 这就是"两个戳紧挨着"那条毛病的解药
        self.assertEqual(out.count('[2026-10-09'), 1)

    def test_contact_line_is_dimmed_only_on_tty(self):
        """**版权行压暗**（`brew`/`npm` 的老做法）—— 但**只在真 TTY 上**。

        非 TTY（管道/日志）里掺转义码是本项目颜色规则的第一条禁忌：
        `grep` 与日志切分都会看见 `^[[90m`。所以 `identity(color=)` 由调用方
        按 `ansi_enabled()` 传，本函数保持纯的。

        ⚠️ **本用例刻意不写 `\033` 字面量**：源码里放转义序列被 heredoc/编辑器
        弄坏过两次（会变成真的 ESC 字节，肉眼看不见）。用 `chr(27)` 拼，
        任何工具链都不会碰坏它。
        """
        esc, lf = chr(27), chr(10)
        dim_esc, reset = esc + '[90m', esc + '[0m'
        plain = identity('GolemQ', 'Copyright (c) x', when=_FIXED)
        self.assertNotIn(esc, plain)                  # 不上色时一个码都没有
        colored = identity('GolemQ', 'Copyright (c) x', when=_FIXED, color=True)
        self.assertIn(esc + '[90mCopyright (c) x' + esc + '[0m', colored)
        # ⚠️ **两行都压暗**（用户 2026-10-10 订正：原来是只压版权行 ——
        # 「第一句 `GolemQ  [t]` 压暗」是后来明确要求的）
        self.assertTrue(colored.startswith(dim_esc + 'GolemQ  [2026-10-09 15:12:57]'))
        self.assertIn(dim_esc + 'Copyright (c) x' + reset, colored)
    def test_identity_without_contact_is_one_line(self):
        self.assertEqual(identity('GolemQ', when=_FIXED),
                         'GolemQ  [2026-10-09 15:12:57]\n\n')

    def test_header_false_prints_no_static_header(self):
        """`header=False` ⇒ **一个静态头行都不打** —— 时刻戳交给身份块去打。"""
        stream = io.StringIO()
        banner = Banner('pytdx', (('参考数据', None, ['stock_list']),),
                        stream=stream, header=False)
        banner.render()
        out = stream.getvalue()
        self.assertNotIn('source:', out)
        self.assertNotIn('[2026', out)
        self.assertIn('参考数据  stock_list ·', out)     # 阶段行还在

    def test_header_true_still_prints_it(self):
        """默认 `header=True` 保持旧行为 —— 直接调用方与旧用例不受影响。"""
        stream = io.StringIO()
        Banner('pytdx', (('参考数据', None, ['stock_list']),),
               stream=stream, when=_FIXED).render()
        self.assertIn('[2026-10-09 15:12:57]: source: pytdx', stream.getvalue())

    def test_phase_column_is_shared(self):
        """⚠️ **跨 banner 的那一列必须同宽** —— `环境自检`（自检块）与
        `参考数据`（取数块）落在同一列上，屏上才不参差。这条一旦破，
        两块就会各排各的。"""
        self.assertEqual(display_width('环境自检'), PHASE_WIDTH)
        self.assertEqual(display_width('参考数据'), PHASE_WIDTH)

    def test_aligned_row_matches_the_phase_column(self):
        row = aligned_row('数据源', 'pytdx')
        self.assertEqual(row, '数据源    pytdx')
        # 「数据源」只占 6 列，要多补 2 列才与 8 列宽的阶段名同列
        self.assertEqual(display_width(row.split('pytdx')[0]), PHASE_WIDTH + 2)
        # 与真实渲染的阶段行**同列** —— ⚠️ 比的是**显示宽度**不是 `str.index`：
        # 字符数上 `数据源` 是 3、`参考数据` 是 4，用 index 会比出 7 vs 6 的假失败
        # （`_ljust` 的文档里警告的就是这件事）。
        real = render_pipeline_banner((('参考数据', None, ['stock_list']),), {},
                                      color=False)
        self.assertEqual(display_width(row[:row.index('pytdx')]),
                         display_width(real[:real.index('stock_list')]))


class TestDimming(unittest.TestCase):
    """**压暗**（用户 2026-10-10 一次点了三处）：身份块两行、banner 头行、阶段起行。

    三处都走 :func:`dim` —— 散着写 `'\033[90m'` 就会在"哪几行该压暗"上分叉。
    ⚠️ **只在真 TTY 上**：非 TTY 里掺转义码是本项目颜色规则的第一条禁忌。
    """

    def test_dim_wraps_only_when_color(self):
        from GolemQ.core.presentation import dim
        self.assertEqual(dim('x'), 'x')
        self.assertEqual(dim('x', color=True), '\033[90mx\033[0m')

    def test_identity_both_lines_are_dimmed(self):
        from GolemQ.core.presentation import identity
        out = identity('GolemQ', 'Copyright', when=_FIXED, color=True)
        self.assertTrue(out.startswith('\033[90mGolemQ  [2026-10-09 15:12:57]\033[0m'))
        self.assertIn('\033[90mCopyright\033[0m', out)

    def test_banner_head_line_is_dimmed_on_tty_only(self):
        from GolemQ.core.presentation import Banner
        stream = _FakeTty()
        with unittest.mock.patch.object(
                __import__('GolemQ.core.presentation', fromlist=['x']),
                '_vt_supported', return_value=True):
            Banner('bootstrap', (('环境自检', None, ['python']),),
                   stream=stream, caption='bootstrap', when=_FIXED).render()
        self.assertIn('\033[90m[2026-10-09 15:12:57]: bootstrap\033[0m',
                      stream.getvalue())

    def test_banner_head_line_has_no_escape_on_a_pipe(self):
        """非 TTY 的头行是**日志** —— 一个转义码都不许有。"""
        stream = io.StringIO()
        from GolemQ.core.presentation import Banner
        Banner('bootstrap', (('环境自检', None, ['python']),),
               stream=stream, caption='bootstrap', when=_FIXED).render()
        self.assertIn('[2026-10-09 15:12:57]: bootstrap\n', stream.getvalue())
        self.assertNotIn('\033', stream.getvalue())
