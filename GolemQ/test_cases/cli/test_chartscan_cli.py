# coding:utf-8
"""`--scan ChartScan` / `--server ChartScan` 两条 CLI 入口（`cli/commands/chartscan.py`）。

钉住三件事：

1. **参数形状**与旧树一致（`--scan ChartScan --eval <scope> --port <p>` /
   `--server ChartScan --port <p>`），且非法值按 D19 给 **2**；
2. **自动切 env**：不在 GolemAI env 里时起子进程（解释器 / 参数 / cwd / 线程数）；
   已经在里面就直接跑模块，不再套一层；
3. **围栏没有豁免**：这两条命令**不在** `bootstrap.REPAIR_COMMANDS` 里
   —— 用户 2026-10-17 定死「围栏不加豁免，依赖不齐就去升级依赖」。
   它们之所以用不着豁免，是因为切过去跑的是**模块**、不是 `GolemQ.cli`。
"""
from __future__ import annotations

import os
import sys
import unittest
from unittest import mock

from GolemQ.cli import bootstrap
from GolemQ.cli.commands import COMMANDS, pick
from GolemQ.cli.commands import chartscan


def _parse(argv):
    from GolemQ.cli.__main__ import build_parser
    args = build_parser().parse_args(argv)
    return args, pick(args)


class TestCommandShape(unittest.TestCase):
    def test_both_registered(self):
        names = {c.name for c in COMMANDS}
        self.assertIn('chartscan-scan', names)
        self.assertIn('chartscan-server', names)

    def test_scan_parses(self):
        args, cmd = _parse(['--scan', 'ChartScan', '--eval', 'fast', '--port', '6666'])
        self.assertEqual(cmd.name, 'chartscan-scan')
        self.assertEqual(args.scan, 'ChartScan')
        self.assertEqual(args.eval_scope, 'fast')
        self.assertEqual(args.port, '6666')

    def test_server_parses(self):
        args, cmd = _parse(['--server', 'ChartScan', '--port', '6666,7777'])
        self.assertEqual(cmd.name, 'chartscan-server')
        self.assertEqual(args.server, 'ChartScan')
        self.assertEqual(args.port, '6666,7777')

    def test_short_scan_alias(self):
        """旧树的 `-sc` 保留；`-s` **已归 `--setup`** ⇒ `--server` 只有长选项。"""
        _, cmd = _parse(['-sc', 'ChartScan'])
        self.assertEqual(cmd.name, 'chartscan-scan')
        _, cmd = _parse(['-s'])                 # `-s` 是 `--setup` 的短开关
        self.assertEqual(cmd.name, 'setup')

    def test_bad_value_is_usage_error(self):
        """旧链对非 `ChartScan` 的值**静默忽略**；新 CLI 按 D19 判成用法错（2）。"""
        import contextlib
        import io
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as cm:
                _parse(['--scan', 'NotChartScan'])
        self.assertEqual(cm.exception.code, 2)


class TestFenceHasNoExemption(unittest.TestCase):
    def test_chartscan_commands_are_not_repair_commands(self):
        """围栏只装「修配置」那四条 —— ChartScan **不许**混进去。"""
        self.assertNotIn('chartscan-scan', bootstrap.REPAIR_COMMANDS)
        self.assertNotIn('chartscan-server', bootstrap.REPAIR_COMMANDS)
        self.assertEqual(bootstrap.REPAIR_COMMANDS,
                         frozenset({'setup', 'mongodb-init',
                                    'dingtalk-init', 'serverchan-init'}))

    def test_gate_still_fires_for_these_commands(self):
        """把它们当成「普通命令」判严格度 —— 硬拦项不过时**照拦**。"""
        for name in ('chartscan-scan', 'chartscan-server'):
            self.assertNotIn(name, bootstrap.REPAIR_COMMANDS)


class TestEnvResolution(unittest.TestCase):
    def test_override_wins(self):
        with mock.patch.dict(os.environ, {'GOLEMQ_CV_PYTHON': '/x/GolemAI/python'}):
            self.assertEqual(chartscan.cv_python(), '/x/GolemAI/python')

    def test_derived_from_interpreter(self):
        with mock.patch.dict(os.environ, {'GOLEMQ_CV_PYTHON': ''}):
            got = chartscan.cv_python()
        exe = os.path.basename(sys.executable)
        self.assertEqual(os.path.basename(got), exe)
        self.assertEqual(os.path.basename(os.path.dirname(got)), chartscan.CV_ENV_NAME)

    def test_in_cv_env_compares_real_prefix(self):
        target = os.path.dirname(os.path.abspath(chartscan.cv_python()))
        with mock.patch.object(chartscan.sys, 'prefix', target):
            self.assertTrue(chartscan.in_cv_env())
        with mock.patch.object(chartscan.sys, 'prefix', target + '_other'):
            self.assertFalse(chartscan.in_cv_env())


class TestReExec(unittest.TestCase):
    """不在 CV env 时起子进程 —— 参数、cwd、线程数、退出码。"""

    def _run(self, returncode=0, exe_exists=True):
        fake = mock.Mock()
        fake.wait.return_value = returncode
        with mock.patch.object(chartscan, 'cv_python', return_value='/x/GolemAI/python'), \
                mock.patch.object(chartscan.os.path, 'exists', return_value=exe_exists), \
                mock.patch.object(chartscan.subprocess, 'Popen', return_value=fake) as popen:
            code = chartscan._re_exec('GolemQ.mod', ['--a', 'b'])
        return code, fake, popen

    def test_missing_interpreter_is_failure(self):
        code, fake, popen = self._run(exe_exists=False)
        self.assertEqual(code, 1)
        popen.assert_not_called()

    def test_command_and_cwd(self):
        code, fake, popen = self._run(returncode=7)
        self.assertEqual(code, 7)
        argv = popen.call_args[0][0]
        self.assertEqual(argv, ['/x/GolemAI/python', '-m', 'GolemQ.mod', '--a', 'b'])
        self.assertEqual(popen.call_args[1]['cwd'], chartscan._REPO_ROOT)

    def test_threads_and_unbuffered(self):
        """父进程的 BLAS 守卫把它们设成 1 ⇒ 子进程**必须**显式改回 4，否则 torch 单线程。"""
        _, _, popen = self._run()
        env = popen.call_args[1]['env']
        from GolemQ import BLAS_THREAD_VARS
        for name in BLAS_THREAD_VARS:
            self.assertEqual(env[name], '4', name)
        self.assertEqual(env['PYTHONUNBUFFERED'], '1')

    def test_threads_override(self):
        with mock.patch.dict(os.environ, {'GOLEMQ_CV_THREADS': '8'}):
            _, _, popen = self._run()
        self.assertEqual(popen.call_args[1]['env']['OMP_NUM_THREADS'], '8')


class TestRunDelegates(unittest.TestCase):
    def _args(self, **kw):
        ns = mock.Mock(eval_scope='fast', port='6666', verbose=False)
        for k, v in kw.items():
            setattr(ns, k, v)
        return ns

    def test_scan_in_cv_env_calls_module_in_process(self):
        with mock.patch.object(chartscan, 'in_cv_env', return_value=True), \
                mock.patch('GolemQ.analysis.cv.client.main') as main:
            self.assertEqual(chartscan.run_chartscan_scan(self._args()), 0)
        main.assert_called_once_with(['--eval', 'fast', '--port', '6666'])

    def test_scan_outside_cv_env_re_execs(self):
        with mock.patch.object(chartscan, 'in_cv_env', return_value=False), \
                mock.patch.object(chartscan, '_re_exec', return_value=3) as re_exec:
            self.assertEqual(chartscan.run_chartscan_scan(self._args()), 3)
        re_exec.assert_called_once_with(
            'GolemQ.analysis.cv.client', ['--eval', 'fast', '--port', '6666'])

    def test_server_in_cv_env_calls_module_in_process(self):
        with mock.patch.object(chartscan, 'in_cv_env', return_value=True), \
                mock.patch('GolemQ.analysis.cv.server.main') as main:
            self.assertEqual(chartscan.run_chartscan_server(self._args()), 0)
        main.assert_called_once_with(['--port', '6666'])

    def test_server_outside_cv_env_re_execs(self):
        with mock.patch.object(chartscan, 'in_cv_env', return_value=False), \
                mock.patch.object(chartscan, '_re_exec', return_value=0) as re_exec:
            self.assertEqual(chartscan.run_chartscan_server(self._args()), 0)
        re_exec.assert_called_once_with(
            'GolemQ.analysis.cv.server', ['--port', '6666'])


if __name__ == '__main__':
    unittest.main()
