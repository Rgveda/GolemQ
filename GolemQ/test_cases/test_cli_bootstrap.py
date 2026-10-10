#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""`cli/bootstrap.py` —— 环境自检的口径。

为什么值得测：自检的**失败口径**是刻意的，且每一条都有代价 ——
版本不满足只警告（硬失败会让 CLI 立刻全废）、TTY 不算不通过（管道是正常用法）、
**配置类命令绝不连库**（否则配错了就永远修不回来）。这些在真实 CLI 上都不报错，
只能靠测试钉住。
"""

import contextlib
import io
import os
import sys

try:
    import GolemQ  # noqa: F401
except ImportError:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import unittest

from GolemQ.cli import bootstrap
from GolemQ.cli.commands import COMMANDS, pick


class TestVersionGate(unittest.TestCase):
    """版本比较的边界 —— doctest 覆盖了主干，这里钉几个容易错的。"""

    def test_meets_pads_shorter_found(self):
        # (2, 5) vs (2, 5, 0) 必须算「满足」—— 不能因为长度不同就判 False
        self.assertTrue(bootstrap._meets((2, 5), (2, 5, 0)))

    def test_meets_unequal_lengths(self):
        self.assertTrue(bootstrap._meets((2, 5, 1), (2, 5)))
        self.assertFalse(bootstrap._meets((2, 4, 9), (2, 5)))

    def test_empty_found_is_not_meeting(self):
        """取不到版本（空元组）算**不满足** —— 别把「不知道」当成「没问题」。"""
        self.assertFalse(bootstrap._meets((), (2, 5)))

    def test_parse_version_stops_at_non_numeric(self):
        self.assertEqual(bootstrap._parse_version('8.3.11-rc1'), (8, 3, 11))
        self.assertEqual(bootstrap._parse_version(''), ())


class TestEnvironmentChecks(unittest.TestCase):
    def test_every_required_package_gets_a_line(self):
        got = bootstrap.check_packages()
        self.assertEqual([d.split()[0] for _, d in got],
                         [name for name, _ in bootstrap.MIN_PACKAGES])

    def test_pandas_floor_is_the_last_2x_not_an_imaginary_2_5(self):
        """pandas 的门槛必须是 **2.3**（3.0 之前最后一个 2.x 是 2.3.3，2025-09-29）。

        ⚠️ 这条是**订正**：最初按口头说法写成了 `2.5` —— **那个版本不存在**，
        于是这条检查永远不可能通过、还看不出错。写成 2.3 之后它同时接纳 2.3.x
        与 3.x（元组比较 3.x > 2.3）。
        """
        self.assertEqual(dict(bootstrap.MIN_PACKAGES)['pandas'], '2.3')

    def test_pandas_floor_accepts_3x(self):
        self.assertTrue(bootstrap._meets((3, 0, 0), (2, 3)))

    def test_checks_return_ok_and_detail(self):
        for fn in (bootstrap.check_python, bootstrap.check_tty, bootstrap.check_config):
            ok, detail = fn()
            self.assertIsInstance(ok, bool, fn.__name__)
            self.assertTrue(detail, fn.__name__)

    def test_tty_does_not_affect_pass_fail(self):
        """非 TTY 是**正常用法**（管道/重定向），不该让自检失败、也不该报 ⚠️。

        实测踩过：一开始把 TTY 混进 pass/fail，于是每次 `... > out.txt`
        都白打一行 ⚠️。
        """
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            bootstrap.check_environment(verbose=False)
        self.assertNotIn('终端', buf.getvalue())

    def test_non_verbose_prints_only_problems(self):
        """非 verbose 的**明细**只有出问题才打 —— 一切正常时不该往外铺。

        口径变化（2026-10-09）：自检改挂 banner 后，「不通过」不再用 `⚠️` 文字表达，
        而是**红/黄点**。所以这里改成断言「没有明细行」（明细一律以节点名开头）。
        """
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            bootstrap.check_environment(verbose=False)
        out = buf.getvalue()
        self.assertNotIn('✓', out)                       # 老口径的勾也没了
        for node in bootstrap.SELF_CHECK_NODES:
            self.assertNotIn('{}: '.format(node), out)   # 明细行的形如「节点: 说明」

    def test_verbose_prints_passing_lines_too(self):
        """`-v` 时每个节点的**明细行**都要出来（用户口径：banner 只报结果，`-v` 报文字）。"""
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            bootstrap.check_environment(verbose=True)
        out = buf.getvalue()
        self.assertIn('python: ', out)                   # python 这条本机是过的
        self.assertIn('终端: ', out)                     # TTY 只在 -v 下作信息行


class TestNeedsDb(unittest.TestCase):
    """哪些命令要求先连上库。

    ⚠️ **配置类命令必须不要求** —— 它们的作用就是修配置，
    要求先连上库等于「配置连错了就永远修不回来」。
    """

    def _cmd(self, name):
        return next(c for c in COMMANDS if c.name == name)

    def test_config_commands_do_not_need_db(self):
        for name in ('setup', 'mongodb-init', 'dingtalk-init', 'serverchan-init',
                     'subscribe'):
            with self.subTest(cmd=name):
                self.assertFalse(self._cmd(name).needs_db)

    def test_data_commands_need_db(self):
        for name in ('save', 'save-coverage', 'save-status', 'purge',
                     'eneloop-list', 'heartbeat-watchdog'):
            with self.subTest(cmd=name):
                self.assertTrue(self._cmd(name).needs_db)


class TestPick(unittest.TestCase):
    def test_pick_returns_none_when_nothing_matches(self):
        self.assertIsNone(pick(_ns()))        # 空 Namespace → 没有任何命令命中

    def test_pick_returns_the_matching_command(self):
        self.assertEqual(pick(_ns(save='tdx')).name, 'save')
        self.assertEqual(pick(_ns(save_coverage=True)).name, 'save-coverage')


def _ns(**kw):
    class _N:
        pass
    n = _N()
    for k, v in kw.items():
        setattr(n, k, v)
    return n


if __name__ == '__main__':
    unittest.main()
