#!/usr/bin/env python3
# coding: utf-8
"""BLAS / OpenMP 线程守卫与 **提交量（commit）** 探针 —— 只读，不连库。

为什么需要这个探针
==================
numpy 自带的 OpenBLAS **按线程预留提交量**；`joblib`/`loky` 的每个 worker
都是独立 python 进程 ⇒ 预留量按 worker 数成倍放大，撞的是 Windows 的
**commit limit（物理内存 + 页面文件）**，不是物理内存。

`GolemQ/__init__.py` 顶部已把六个环境变量 `setdefault` 到 `1`（见
[`PITFALLS.md` P21](../PITFALLS.md)）。这个脚本把「守卫有没有生效、每个
进程实际占多少提交量、joblib worker 有没有继承」一次性打出来，供**新机部署**
与排障使用。

跑法
====
    python tools/check_blas_commit.py
    python tools/check_blas_commit.py --workers 4

输出示例（Windows，18 逻辑核）::

    [当前进程] import GolemQ 前: {'OPENBLAS_NUM_THREADS': None, ...}
    [当前进程] import GolemQ 后: {'OPENBLAS_NUM_THREADS': '1', ...}
    [子进程] 剥掉变量 + import numpy : 2200.1 MiB
    [子进程] 变量=1   + import numpy :   17.5 MiB
    [子进程] import GolemQ + numpy   :   17.5 MiB  (numba threads=1)
    [worker 0] pid=... OPENBLAS=1 commit=27.1 MiB
    ...

⚠️ **只在 Windows 上读 commit**（`PROCESS_MEMORY_COUNTERS.PagefileUsage`）。
Linux 没有同名的 per-process commit 概念，脚本会打 `None`；用
`grep -E 'CommitLimit|Committed_AS' /proc/meminfo` 看系统级数据。
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: 与 `GolemQ.BLAS_THREAD_VARS` 同义。这里先写一份，是因为**读当前进程环境必须
#: 在 `import GolemQ` 之前** —— 而 import 之后会再与包里的定义核对一次。
VARS = ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'NUMBA_NUM_THREADS',
        'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')

_COMMIT_SNIPPET = r'''
def _commit_mib():
    import sys
    if sys.platform != 'win32':
        return None
    import ctypes
    import ctypes.wintypes as w

    class PMC(ctypes.Structure):
        _fields_ = [
            ('cb', w.DWORD), ('PageFaultCount', w.DWORD),
            ('PeakWorkingSetSize', ctypes.c_size_t),
            ('WorkingSetSize', ctypes.c_size_t),
            ('QuotaPeakPagedPoolUsage', ctypes.c_size_t),
            ('QuotaPagedPoolUsage', ctypes.c_size_t),
            ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t),
            ('QuotaNonPagedPoolUsage', ctypes.c_size_t),
            ('PagefileUsage', ctypes.c_size_t),
            ('PeakPagefileUsage', ctypes.c_size_t)]
    psapi = ctypes.WinDLL('psapi', use_last_error=True)
    psapi.GetProcessMemoryInfo.argtypes = [w.HANDLE, ctypes.POINTER(PMC), w.DWORD]
    psapi.GetProcessMemoryInfo.restype = w.BOOL
    k32 = ctypes.WinDLL('kernel32', use_last_error=True)
    k32.GetCurrentProcess.restype = w.HANDLE
    p = PMC()
    p.cb = ctypes.sizeof(PMC)
    psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(p), p.cb)
    return round(p.PagefileUsage / 1024 / 1024, 1)
'''

#: 子进程 1：剥掉变量，只 import numpy（复现"守卫缺失"）。
_CHILD_NAIVE = _COMMIT_SNIPPET + r'''
import json, os, sys
before = _commit_mib()
import numpy
print(json.dumps({'before': before, 'after': _commit_mib(),
                  'env': {k: os.environ.get(k) for k in sys.argv[1:]}}))
'''

#: 子进程 2：显式把变量设成 1，再 import numpy。
_CHILD_FORCED = _COMMIT_SNIPPET + r'''
import json, os, sys
before = _commit_mib()
import numpy
print(json.dumps({'before': before, 'after': _commit_mib(),
                  'env': {k: os.environ.get(k) for k in sys.argv[1:]}}))
'''

#: 子进程 3：**按真实入口** import GolemQ（守卫自动执行），再 import numpy/numba。
_CHILD_GOLEMQ = _COMMIT_SNIPPET + r'''
import json, os, sys
sys.path.insert(0, sys.argv[1])
before = _commit_mib()
import GolemQ
import numpy
import numba
print(json.dumps({'before': before, 'after': _commit_mib(),
                  'numba_threads': int(numba.get_num_threads()),
                  'env': {k: os.environ.get(k) for k in GolemQ.BLAS_THREAD_VARS}}))
'''


def _run_child(code: str, env: dict, args=()):
    out = subprocess.run([sys.executable, '-c', code, *args],
                         cwd=_REPO_ROOT, env=env,
                         capture_output=True, text=True)
    if out.returncode != 0:
        return None, out.stderr.strip()[-600:]
    return json.loads(out.stdout.strip().splitlines()[-1]), None


def _fmt_commit(value) -> str:
    return 'N/A' if value is None else '{:.1f} MiB'.format(value)


def _worker(i: int) -> dict:
    """joblib worker 里跑：确认变量继承 + 各自的提交量。"""
    import time
    import numpy  # noqa: F401  （worker 是独立进程，这行才会真正执行）

    time.sleep(0.5)         # 让 n_jobs 个 worker 真正同时在跑（否则会被复用成一个 pid）
    exec(_COMMIT_SNIPPET, globals())
    return {'worker': i, 'pid': os.getpid(),
            'env': {k: os.environ.get(k) for k in VARS},
            'commit_mib': _commit_mib()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--workers', type=int, default=0,
                    help='额外验证 joblib 多进程继承（0 = 跳过；建议 2..8）')
    args = ap.parse_args()

    print('CPU 逻辑核数: {}'.format(os.cpu_count()))
    print('[当前进程] import GolemQ 前: {}'.format(
        {k: os.environ.get(k) for k in VARS}))

    # 父进程 import 之后，env 会被守卫改写；后面每个子进程自己控制 env。
    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)
    from GolemQ import BLAS_THREAD_VARS
    if tuple(BLAS_THREAD_VARS) != VARS:
        print('⚠️ 包里的 BLAS_THREAD_VARS 与探针不一致: {}'.format(BLAS_THREAD_VARS),
              file=sys.stderr)
    print('[当前进程] import GolemQ 后: {}'.format(
        {k: os.environ.get(k) for k in BLAS_THREAD_VARS}))

    stripped = {k: v for k, v in os.environ.items() if k not in VARS}
    forced = dict(stripped, **{k: '1' for k in VARS})

    got, err = _run_child(_CHILD_NAIVE, stripped, VARS)
    if err:
        print('[子进程] 剥掉变量失败: {}'.format(err), file=sys.stderr)
    else:
        print('[子进程] 剥掉变量 + import numpy : {}（import 前 {}）'.format(
            _fmt_commit(got['after']), _fmt_commit(got['before'])))

    got, err = _run_child(_CHILD_FORCED, forced, VARS)
    if err:
        print('[子进程] 变量=1 失败: {}'.format(err), file=sys.stderr)
    else:
        print('[子进程] 变量=1   + import numpy : {}'.format(
            _fmt_commit(got['after'])))

    got, err = _run_child(_CHILD_GOLEMQ, stripped, (_REPO_ROOT,))
    if err:
        print('[子进程] import GolemQ 失败: {}'.format(err), file=sys.stderr)
    else:
        print('[子进程] import GolemQ + numpy   : {}  (numba threads={})'.format(
            _fmt_commit(got['after']), got['numba_threads']))
        print('          env: {}'.format(got['env']))

    if args.workers > 0:
        try:
            from joblib import Parallel, delayed
        except Exception as exc:                # noqa: BLE001
            print('跳过 worker 检查（joblib 不可用: {}）'.format(exc))
            return 0
        rows = Parallel(n_jobs=args.workers)(delayed(_worker)(i)
                                             for i in range(args.workers))
        for row in rows:
            print('[worker {worker}] pid={pid} commit={commit} OPENBLAS={ob} OMP={omp}'
                  .format(worker=row['worker'], pid=row['pid'],
                          commit=_fmt_commit(row['commit_mib']),
                          ob=row['env'].get('OPENBLAS_NUM_THREADS'),
                          omp=row['env'].get('OMP_NUM_THREADS')))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
