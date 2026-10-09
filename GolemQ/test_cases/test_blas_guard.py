#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""BLAS / OpenMP 线程数守卫（`GolemQ/__init__.py` 顶部）。

⚠️ **这段守卫丢过一次，代价是几小时。** 老树 `GolemQ_old/__init__.py` 有它
（日志 `GolemQ_old/docs/claude_change_log.md:3182` 记着定位过程），新树重构时
**没搬过来**，于是 2026-10-09 `--save tdx` 跑到 76% 时卡死：**17.33s/code**
（正常亚秒级），迟迟不结束。

机制：numpy 的 OpenBLAS **按线程**预留提交量。本机 **18 核** 实测 ——

| 条件 | 单进程提交量 |
|:--|--:|
| 无守卫 | **6573.9 MB** |
| 设线程=1 | **19.8 MB** |

`jobs=4` 时（1 主 + 4 worker）：**32.1 GB → 0.43 GB**。撞的是 Windows 的
**commit limit**（物理内存 + 页面文件），不是物理内存。

所以这条**必须有测试钉住** —— 它丢了不会报错，只会"跑着跑着越来越慢"。
"""

import json
import os
import subprocess
import sys

try:
    import GolemQ
except ImportError:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import unittest

from GolemQ import BLAS_THREAD_VARS

#: 这些变量一旦被父进程设过，`setdefault` 就不会覆盖 —— 那条用例要跳过。
_ENV_KEYS = set(BLAS_THREAD_VARS)


class TestBlasThreadGuard(unittest.TestCase):
    def test_all_variables_are_present(self):
        for name in BLAS_THREAD_VARS:
            with self.subTest(var=name):
                self.assertIsNotNone(
                    os.environ.get(name),
                    '{} 没被设上 —— `GolemQ/__init__.py` 顶部的守卫是不是被删了？'.format(name))

    def test_guard_sets_one_in_a_clean_interpreter(self):
        """**真回归**：起一个**剥掉这些变量**的新解释器，`import GolemQ` 之后必须变成 '1'。

        在本进程里断不出「守卫被删」—— 变量可能早被父进程设过（`setdefault` 不覆盖）。
        子进程才测得到真行为。
        """
        env = {k: v for k, v in os.environ.items() if k not in _ENV_KEYS}
        out = subprocess.run(
            [sys.executable, '-c',
             'import GolemQ, os, json;'
             'print(json.dumps({k: os.environ.get(k) for k in '
             'GolemQ.BLAS_THREAD_VARS}))'],
            capture_output=True, text=True, env=env)

        self.assertEqual(out.returncode, 0, out.stderr[-800:])
        got = json.loads(out.stdout.strip())
        self.assertEqual(set(got), set(BLAS_THREAD_VARS))
        for name, value in got.items():
            self.assertEqual(value, '1', '{} 在干净解释器里没被压到 1'.format(name))

    def test_guard_does_not_override_explicit_setting(self):
        """`setdefault` 语义：调用方自己 export 的都该原样保留。

        单进程做数值密集计算时可以 `export OPENBLAS_NUM_THREADS=4` 放开 ——
        守卫不该把这种显式选择抹掉。
        """
        env = {k: v for k, v in os.environ.items() if k not in _ENV_KEYS}
        env['OPENBLAS_NUM_THREADS'] = '4'
        out = subprocess.run(
            [sys.executable, '-c',
             'import GolemQ, os; print(os.environ["OPENBLAS_NUM_THREADS"])'],
            capture_output=True, text=True, env=env)

        self.assertEqual(out.stdout.strip(), '4')


if __name__ == '__main__':
    unittest.main()
