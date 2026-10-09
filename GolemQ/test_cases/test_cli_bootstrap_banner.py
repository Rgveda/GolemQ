#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""环境自检的 banner（`cli/bootstrap.py` + `core/presentation.py` 的四态）。

为什么值得测：自检结果**只由那个点表达**（用户口径：「banner 提示检查结果，
`--verbose` 显示详细文字」），所以「点对不对」就是全部的信息量 —— 而它有四个坑：

1. **非 TTY 下四态必须靠符号区分** —— 通过/警告/失败在高亮下是三个同形的 `●`，
   一旦落到日志里完全同形，而日志正是出事后唯一能翻的东西；
2. **`未检查`（灰）不是失败** —— CUDA 探不到要灰不要红，否则每台没显卡的机器
   与每个 CI 都变红，而那条路根本不跑 GPU；
3. **banner 活跃期间一个 `print` 都不许有**（`PITFALLS.md` P22）—— 自检里唯一会
   `print` 的是「硬拦项不过」与「配置有问题」两条，它们都必须在 `close()` **之后**；
4. **时间戳只盖静态头行** —— 盖到会重画的状态行上，会让「变化规则」失真。
"""

import contextlib
import datetime
import io
import os
import sys

try:
    import GolemQ  # noqa: F401
except ImportError:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import unittest
import unittest.mock

from GolemQ.cli import bootstrap
from GolemQ.cli.commands._registry import EXIT_FAILURE
from GolemQ.core.presentation import (
    Banner,
    FAIL,
    OK,
    PENDING,
    WARN,
    render_pipeline_banner,
    stamp,
)

FIXED = datetime.datetime(2026, 10, 9, 15, 12, 57)


class TestStamp(unittest.TestCase):
    def test_exact_format(self):
        self.assertEqual(stamp('bootstrap', FIXED), '[2026-10-09 15:12:57]: bootstrap')

    def test_only_the_header_gets_stamped(self):
        """`caption` 覆盖头行全文，且**只有头行**带戳 —— 状态行不带。"""
        stream = io.StringIO()
        banner = Banner('bootstrap', bootstrap.SELF_CHECK_ROWS, stream=stream,
                        caption='bootstrap', when=FIXED)
        banner.render()
        banner.mark('python', OK)
        out = stream.getvalue()

        self.assertIn('[2026-10-09 15:12:57]: bootstrap\n', out)
        self.assertEqual(out.count('[2026-10-09'), 1, '戳只该出现一次（头行）')

    def test_save_banner_keeps_source_prefix(self):
        """取数 banner 不给 `caption` → 头行仍是 `source: {名}`，只是前面多了戳。"""
        stream = io.StringIO()
        banner = Banner('pytdx', (('参考数据', None, ['stock_list']),),
                        stream=stream, when=FIXED)
        banner.render()
        self.assertIn('[2026-10-09 15:12:57]: source: pytdx', stream.getvalue())


class TestFourStates(unittest.TestCase):
    """四态在**两种渲染模式**下都要分得出来。"""

    ROWS = (('环境自检', None, ['通过', '警告', '失败', '未查']),)

    def _render(self, color):
        states = {'通过': OK, '警告': WARN, '失败': FAIL, '未查': PENDING}
        return render_pipeline_banner(self.ROWS, states, color=color)

    def test_colored_uses_three_colours_and_one_gray(self):
        out = self._render(color=True)
        self.assertIn('\033[32m●\033[0m', out)      # 通过 = 绿
        self.assertIn('\033[33m●\033[0m', out)      # 警告 = 黄
        self.assertIn('\033[31m●\033[0m', out)      # 失败 = 红
        self.assertIn('\033[90m·\033[0m', out)      # 未检查 = 灰

    def test_plain_mode_distinguishes_by_symbol(self):
        """**非 TTY 的命门**：没有颜色时 警告/失败 必须换成不同符号。

        否则日志里 通过 / 警告 / 失败 全是 `●`，翻日志的人分不出哪个是坏的。
        """
        out = self._render(color=False)
        self.assertIn('通过 ●', out)
        self.assertIn('警告 !', out)
        self.assertIn('失败 ✗', out)
        self.assertIn('未查 ·', out)

    def test_pending_is_not_a_failure(self):
        """灰 = 未检查 / 不适用，**不是**失败 —— 别把它算进红。"""
        out = self._render(color=False)
        self.assertIn('未查 ·', out)
        self.assertNotIn('未查 ✗', out)

    def test_new_states_do_not_change_the_old_three(self):
        """扩四态不得动 `--save` 那三态的渲染（那是 2026-10-09 刚定的契约）。"""
        from GolemQ.core.presentation import DONE, RUNNING
        rows = (('K线', 'stock', ['stock_day']),)
        self.assertEqual(render_pipeline_banner(rows, {})
                         .replace('\033[90m·\033[0m', '·'), 'K线  stock  day ·')
        self.assertIn('\033[32m●\033[0m',
                      render_pipeline_banner(rows, {'stock_day': RUNNING}))
        self.assertIn('\033[97m●\033[0m',
                      render_pipeline_banner(rows, {'stock_day': DONE}))


class TestCheckResults(unittest.TestCase):
    def test_run_checks_returns_one_entry_per_node_in_order(self):
        got = bootstrap.run_checks()
        self.assertEqual([node for node, _, _ in got], list(bootstrap.SELF_CHECK_NODES))

    def test_every_node_is_in_the_banner_header(self):
        """节点名与表头键**必须是同一批** —— 不然 `mark` 会**静默忽略**（不报错）。"""
        keys = [k for _, _, keys in bootstrap.SELF_CHECK_ROWS for k in keys]
        self.assertEqual(keys, list(bootstrap.SELF_CHECK_NODES))

    def test_details_are_always_a_list_of_lines(self):
        """明细**一律是 list** —— 裸字符串会被下游逐**字**迭代（实测踩过）。"""
        for node, _, lines in bootstrap.run_checks():
            with self.subTest(node=node):
                self.assertIsInstance(lines, list)
                self.assertTrue(lines)
                self.assertTrue(all(isinstance(x, str) for x in lines))

    def test_cuda_is_never_a_failure(self):
        """新树零 GPU 依赖 → 探不到只算「未检查」。"""
        state, _ = bootstrap.check_cuda()
        self.assertIn(state, (OK, PENDING))
        self.assertNotEqual(state, FAIL)

    def test_blas_warns_on_a_non_one_value(self):
        """`OPENBLAS_NUM_THREADS=4` 与「没设」后果几乎一样，旧版却报绿。"""
        with unittest.mock.patch.dict(os.environ, {'OPENBLAS_NUM_THREADS': '4'}):
            ok, detail = bootstrap.check_blas()
        self.assertFalse(ok)
        self.assertIn('不是 1', detail)

    def test_pytdx_has_no_version_and_is_not_penalised(self):
        """`want=''` = 只查 import —— pytdx **没有** `__version__`，不能因此报红。"""
        rows = dict((d.split()[0], ok) for ok, d in bootstrap.check_packages())
        self.assertIn('pytdx', rows)
        self.assertTrue(rows['pytdx'])


class TestHardGate(unittest.TestCase):
    """硬拦项（python / 依赖包）才拦人，且**修配置那四条命令必须放行**。"""

    def _with_broken(self, **patch):
        return unittest.mock.patch.multiple(bootstrap, **patch)

    def test_failing_python_exits_when_strict(self):
        broken = ((False, 'python 3.9.0（要求 >= 3.12）'),)
        with self._with_broken(check_python=lambda: broken[0]):
            with contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(SystemExit) as ctx:
                    bootstrap.check_environment(verbose=False, strict=True)
        self.assertEqual(ctx.exception.code, EXIT_FAILURE)

    def test_same_failure_does_not_exit_when_not_strict(self):
        with self._with_broken(check_python=lambda: (False, 'python 3.9.0')):
            with contextlib.redirect_stdout(io.StringIO()):
                ok = bootstrap.check_environment(verbose=False, strict=False)
        self.assertFalse(ok)

    def test_a_warning_never_exits(self):
        """黄点（线程环境 / 时区 / CUDA）**不拦** —— 它们可能是显式选择或不适用。"""
        with unittest.mock.patch.dict(os.environ, {'OPENBLAS_NUM_THREADS': '4'}):
            with contextlib.redirect_stdout(io.StringIO()):
                ok = bootstrap.check_environment(verbose=False, strict=True)
        self.assertTrue(ok)

    def test_repair_commands_are_exactly_the_config_ones(self):
        """放行集合必须就是那四条修配置的命令 —— 多一条就绕过闸，少一条就修不回来。"""
        self.assertEqual(bootstrap.REPAIR_COMMANDS,
                         frozenset({'setup', 'mongodb-init',
                                    'dingtalk-init', 'serverchan-init'}))


class TestNothingPrintsWhileTheBannerIsAlive(unittest.TestCase):
    """`PITFALLS.md` P22：banner 活跃期间一个 `print` 都不许有。

    自检里会 `print` 的只有两处 —— 「硬拦项不过」与「配置有问题」——
    两者都**必须在 `close()` 之后**。这条用例把顺序钉住：用一个记账版的 Banner
    记录 `render`/`close` 的先后，再断言这期间的 `print` 次数为 0。

    ⚠️ 与 `test_kline_save.TestNothingPrintsWhileTheBarIsAlive` 并列：
    那条钉的是**取数**内层漏 print，这条钉的是**自检**自己。
    """

    def _run(self, broken_cfg=False):
        from GolemQ.core import presentation

        opened = {'live': False, 'prints': 0, 'was_live': False}
        real_banner = presentation.Banner

        class _Spy(real_banner):
            def render(self):
                super().render()
                opened['live'] = True
                opened['was_live'] = True

            def close(self):
                super().close()
                opened['live'] = False

        def _counting_print(*args, **kwargs):
            if opened['live']:
                opened['prints'] += 1
            return None

        patch_cfg = (unittest.mock.patch.object(bootstrap, 'check_config',
                                                lambda: (False, '没找到配置文件'))
                     if broken_cfg else contextlib.nullcontext())
        with unittest.mock.patch.object(presentation, 'Banner', _Spy), \
                unittest.mock.patch('builtins.print', _counting_print), \
                patch_cfg, \
                contextlib.redirect_stdout(io.StringIO()):
            bootstrap.check_environment(verbose=False, strict=False)
        return opened

    def test_no_print_when_everything_is_fine(self):
        got = self._run()
        # ⚠️ 先证「监视真的挂上了」—— 否则 spy 没生效时 `prints == 0` 也会绿，
        # 这条用例就变成了永远通过的空转。
        self.assertTrue(got['was_live'], 'Banner 根本没被打开，用例失去意义')
        self.assertEqual(got['prints'], 0)

    def test_no_print_even_when_there_is_a_problem(self):
        """有问题的路径也要走 「关掉 banner → 再 print」，不然屏上会花。"""
        got = self._run(broken_cfg=True)
        self.assertTrue(got['was_live'])
        self.assertEqual(got['prints'], 0)


if __name__ == '__main__':
    unittest.main()


class TestTradingCalendarCheck(unittest.TestCase):
    """交易日历节点（用户 2026-10-10 定）。

    为什么值得单独测：`TRADE_DATE_SSE` 是**手维护的静态表**，过期了**不报错** ——
    只让"今天"被静默判成非交易日，于是**短路判据 / TTL / 调度全按错的日子走**。
    四档 + 两个异常都在下面钉住（含**用户没给、我补的那一档**）。
    """

    CAL = ['2026-12-29', '2026-12-30', '2026-12-31']      # 末端 = 今年年底

    def _state(self, cal, today):
        return bootstrap.check_calendar(cal, today)[0]

    def test_expired_is_fail(self):
        """末端 < 今天 ⇒ 红（后面的日期全被判成非交易日）。"""
        self.assertEqual(self._state(['2025-12-31'], '2026-10-10'), FAIL)

    def test_fresh_before_nov10_is_ok(self):
        self.assertEqual(self._state(self.CAL, '2026-10-10'), OK)

    def test_after_nov10_should_renew_is_warn(self):
        self.assertEqual(self._state(self.CAL, '2026-11-11'), WARN)

    def test_nov10_boundary_itself_is_still_ok(self):
        """边界取**含** 11-10（用户口径「11月10日以前」）。"""
        self.assertEqual(self._state(self.CAL, '2026-11-10'), OK)

    def test_short_coverage_is_warn(self):
        """⚠️ **用户没给的第四档**：末端在未来、但没到今年年底 ⇒ 还能跑、覆盖不够长。

        （注意别拿"末端 6-30 + 今天 10-10"测 —— 那是**已过期**，先命中红。）
        """
        self.assertEqual(self._state(['2026-06-30'], '2026-05-01'), WARN)

    def test_empty_calendar_is_fail(self):
        self.assertEqual(self._state([], '2026-10-10'), FAIL)

    def test_malformed_tail_is_fail(self):
        self.assertEqual(self._state(['2026-12-3x'], '2026-10-10'), FAIL)

    def test_year_rollover_expires(self):
        """跨年：到了次年 1 月而日历没续 ⇒ 红（这是这条检查最该抓的情形）。"""
        self.assertEqual(self._state(self.CAL, '2027-01-02'), FAIL)

    def test_it_is_a_node_but_not_a_hard_gate(self):
        """在 banner 上有节点，但**不进硬拦**（红点也不拦启动）。"""
        self.assertIn('交易日历', bootstrap.SELF_CHECK_NODES)
        self.assertNotIn('交易日历', bootstrap.ENV_GATE_NODES)

    def test_real_calendar_is_current(self):
        """真日历此刻应该是**绿**（覆盖到今年年底）。

        ⚠️ 这条会在**跨年且没续日历**时变红 —— 那是对的（它就该提醒你去续），
        若那天到了而这条红了，请**更新 `TRADE_DATE_SSE`**，别改这条用例。
        """
        import datetime as _dt
        state, detail = bootstrap.check_calendar()
        year = _dt.date.today().year
        if state == WARN:
            self.skipTest('已过 11-10 且日历未续下一年 —— 这正是黄点要提示的：{}'
                          .format(detail))
        self.assertEqual(state, OK)
        self.assertIn('{}-12-31'.format(year), detail)


class TestBannerTwoColumns(unittest.TestCase):
    """自检 banner **分两栏**（用户 2026-10-10）。

    11 个节点挤一行太长（2026-10-09 那版 7 个时是一行平铺）。两栏按**语义**分：
    上栏「机器 / 解释器」，下栏「环境 / 数据源」。
    """

    def test_two_rows(self):
        self.assertEqual(len(bootstrap.SELF_CHECK_ROWS), 2,
                         '应是**两栏**（两行），不是一行平铺')

    def test_rows_flatten_to_the_node_list_in_order(self):
        """两栏的**顺序拼起来**必须等于 `SELF_CHECK_NODES` —— 顺序是语义。"""
        keys = [k for _, _, ks in bootstrap.SELF_CHECK_ROWS for k in ks]
        self.assertEqual(keys, list(bootstrap.SELF_CHECK_NODES))

    def test_column_split_is_semantic(self):
        upper = bootstrap.SELF_CHECK_ROWS[0][2]
        lower = bootstrap.SELF_CHECK_ROWS[1][2]
        # 上栏「本机」：含 **时区**（本机属性，跟着机器走 —— 用户 2026-10-10 调的换行）
        for node in ('操作系统', 'python', '依赖包', '线程环境', 'CPU 架构', 'CUDA', '时区'):
            self.assertIn(node, upper, '本机项该在上栏')
        # 下栏「外部依赖」：日历 + 数据源 + 通知渠道
        for node in ('交易日历', 'tdxidata', 'tushare', 'iwencai',
                     'serverchan', '讯投QMT'):
            self.assertIn(node, lower, '外部依赖该在下栏')

    def test_second_row_shares_the_phase_name(self):
        """两行的阶段名相同 ⇒ 第二行**留白对齐**（渲染成一块，不是两块）。"""
        self.assertEqual(bootstrap.SELF_CHECK_ROWS[0][0],
                         bootstrap.SELF_CHECK_ROWS[1][0])
        lines = render_pipeline_banner(bootstrap.SELF_CHECK_ROWS, {},
                                       color=False).split('\n')
        self.assertEqual(len(lines), 2)
        self.assertTrue(lines[0].startswith('环境自检'))
        self.assertFalse(lines[1].startswith('环境自检'),
                         '第二行不该重复阶段名（应留白对齐）')


class TestOptionalSourceChecks(unittest.TestCase):
    """`tdxidata` / `tushare` / `iwencai` 三个节点。

    ⚠️ 判据**复用数据层自己的 `available()` / `unavailable_reason()`** ——
    不在 CLI 里重写「配置了没」（那是平行实现，且配置键名会散成两处真相）。
    """

    def _check(self, name, *, available=True, reason='没配', boom=None):
        fake = unittest.mock.MagicMock()
        fake.available.return_value = available
        fake.unavailable_reason.return_value = reason
        if boom is not None:
            fake.available.side_effect = boom
        with unittest.mock.patch('GolemQ.markets.StockCN.datasource.get_source',
                                 return_value=fake):
            return bootstrap.check_source(name)

    def test_configured_is_ok(self):
        self.assertEqual(self._check('tushare')[0], OK)

    def test_not_configured_is_pending_not_warn(self):
        """**未配置 ⇒ 灰**，不是红也不是黄：可选源没配不影响任何命令跑得通。"""
        state, detail = self._check('tushare', available=False, reason='未配置 token')
        self.assertEqual(state, PENDING)
        self.assertIn('未配置 token', detail)

    def test_probe_failure_does_not_raise(self):
        """探测抛错也要给状态（灰 + 原因），**不许把启动自检搞崩**。"""
        state, detail = self._check('tushare', boom=RuntimeError('适配器炸了'))
        self.assertEqual(state, PENDING)
        self.assertIn('RuntimeError', detail)

    def test_get_source_failure_does_not_raise(self):
        with unittest.mock.patch('GolemQ.markets.StockCN.datasource.get_source',
                                 side_effect=KeyError('没这个源')):
            state, detail = bootstrap.check_source('nope')
        self.assertEqual(state, PENDING)
        self.assertIn('KeyError', detail)

    def test_iwencai_is_pending_and_says_unimplemented(self):
        """⚠️ 问财在新树**尚未实现** —— 节点如实说，且**不发明没人读的配置键**。"""
        state, detail = bootstrap.check_iwencai()
        self.assertEqual(state, PENDING)
        self.assertIn('尚未实现', detail)

    def test_three_nodes_are_present_but_not_hard_gates(self):
        for node in ('tdxidata', 'tushare', 'iwencai'):
            with self.subTest(node=node):
                self.assertIn(node, bootstrap.SELF_CHECK_NODES)
                self.assertNotIn(node, bootstrap.ENV_GATE_NODES,
                                 '可选源不许拦启动')

    def test_optional_source_map_covers_the_two_adapters(self):
        """节点名 → 适配器名的映射必须与 `datasource/` 的注册键一致。"""
        from GolemQ.markets.StockCN.datasource import get_source
        for node, adapter in bootstrap.OPTIONAL_SOURCES.items():
            with self.subTest(node=node):
                self.assertIsNotNone(get_source(adapter))


class TestServerchanAndQmtChecks(unittest.TestCase):
    """`serverchan` 与 `讯投QMT` 两个节点。"""

    def test_serverchan_configured_is_ok(self):
        with unittest.mock.patch('GolemQ.agents.messenger.check_serverchan_config',
                                 return_value=True):
            self.assertEqual(bootstrap.check_serverchan()[0], OK)

    def test_serverchan_unconfigured_is_pending(self):
        """未配 ⇒ **灰**：推送是可选渠道，不配不影响任何命令。"""
        with unittest.mock.patch('GolemQ.agents.messenger.check_serverchan_config',
                                 return_value=False):
            state, detail = bootstrap.check_serverchan()
        self.assertEqual(state, PENDING)
        self.assertIn('sendkey', detail)

    def test_serverchan_probe_failure_does_not_raise(self):
        with unittest.mock.patch('GolemQ.agents.messenger.check_serverchan_config',
                                 side_effect=RuntimeError('读配置炸了')):
            state, detail = bootstrap.check_serverchan()
        self.assertEqual(state, PENDING)
        self.assertIn('RuntimeError', detail)

    def test_qmt_checks_the_xtquant_config_section(self):
        """用户 2026-10-10 明确：「**讯投QMT 检查的是这一段** `[XTQUANT] account/min_path`」。

        ⚠️ 这是**订正**：我上一版让它走 `QmtSource.available()`（恒 False）⇒ 恒灰，
        那答的是"**这条路能用吗**"，不是用户要的"**配置齐没齐**"。
        """
        with unittest.mock.patch.object(bootstrap, '_ini_value',
                                        side_effect=lambda sec, opt: '设了'):
            state, detail = bootstrap.check_xtquant()
        self.assertEqual(state, OK)

    def test_qmt_missing_keys_is_warn(self):
        with unittest.mock.patch.object(bootstrap, '_ini_value',
                                        side_effect=lambda sec, opt:
                                        '' if opt == 'min_path' else '设了'):
            state, detail = bootstrap.check_xtquant()
        self.assertEqual(state, WARN)
        self.assertIn('min_path', detail)

    def test_qmt_unreadable_config_is_warn(self):
        with unittest.mock.patch.object(bootstrap, '_ini_value', return_value=None):
            self.assertEqual(bootstrap.check_xtquant()[0], WARN)

    def test_qmt_green_still_reports_the_shutdown(self):
        """⚠️ **绿点不表示这条路可用** —— 「已停服」必须留在 detail 里。

        判据既然按用户口径定在**配置**上，那个事实就不能从屏上消失 ——
        否则一个绿点会让人以为 QMT 能用（D13：MiniQMT 自 2026-10-01 停服）。
        """
        with unittest.mock.patch.object(bootstrap, '_ini_value',
                                        side_effect=lambda sec, opt: '设了'):
            state, detail = bootstrap.check_xtquant()
        self.assertEqual(state, OK)
        self.assertIn('停服', detail)
        self.assertIn('D13', detail)

    def test_qmt_is_not_an_adapter_source(self):
        """它查的是**配置段**，不该混进"适配器可用吗"那张表。"""
        self.assertNotIn('讯投QMT', bootstrap.OPTIONAL_SOURCES)
        self.assertEqual(bootstrap.XTQUANT_KEYS, ('account', 'min_path'))
