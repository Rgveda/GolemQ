# coding:utf-8
"""ChartScan（CV 图形识别）的两条 CLI 入口 —— `--scan` / `--server`。

    python -m GolemQ.cli --server ChartScan --port 5555
    python -m GolemQ.cli --scan   ChartScan --eval fast --port 5555

搬自旧树 `GolemAI/GolemQ/cli/__main__.py`（参数在 `:86-100`，实现分别委托给
`analysis/cv/{server,client}.py` —— 那两个模块也是从那里搬的）。
**本模块只做参数校验 + 委托**，业务逻辑不在这里。

⚠️ 这两条命令**自动切换到 GolemAI conda env**
==========================================
torch / ultralytics 只装在那儿；GolemQ env 是 **NoMKL**，跑不了 torch（D42）。
**一个进程没法"切 env"** ⇒ 在 GolemQ env 里调用时，本模块用 **GolemAI 的 python
重新起一个子进程**（stdio / cwd 都接上，退出码原样透传）。已经在 GolemAI env 里
就直接 import 跑，不再多套一层 —— 旧树的 `chartscan_*.ps1` 走的就是后者。

⚠️ 切过去之后跑的是**模块**，不是 CLI
=====================================
`_re_exec` 起的是 ``python -m GolemQ.analysis.cv.{server,client}``
—— **不是** ``python -m GolemQ.cli``。所以 `bootstrap` 那道「依赖包」硬闸
**根本不会在 CV env 里执行**，也就**不需要**给它开任何豁免
（用户 2026-10-17 定死：围栏不加豁免，依赖不齐就去升级依赖）。

反面也成立：**在 CV env 里直接敲 `python -m GolemQ.cli ...` 会被闸拦住** ——
那正是它该有的样子，别去改 `bootstrap.REPAIR_COMMANDS`。

环境怎么定位
------------
默认由**当前解释器**推导：``<sys.executable 的上两级>/GolemAI/python.exe``
（conda 的 ``<root>/envs/<name>/`` 布局）。可用 ``$GOLEMQ_CV_PYTHON`` 直接覆盖。

⚠️ **线程数必须由本模块显式重设**：`GolemQ/__init__.py:49-50` 在 import 时把
`BLAS_THREAD_VARS` 全部 `setdefault` 成 **1**（那是为 joblib 多进程防 commit-limit
的守卫）。父进程已经 import 过 GolemQ ⇒ 子进程会**继承 1**，torch/MKL 会被限成
单线程、推理慢一个数量级。所以 `_re_exec` 把它们显式改成
``$GOLEMQ_CV_THREADS``（默认 4）—— 旧树 ps1 在启动前设 4，是同一个理由、同一个值。

⚠️ **不检查 CV env 里装了什么**：缺包会在 `client` / `server` 的 import 处当场报出。
在 CLI 里再抄一份"装没装"的判据 = 第二处真相（那条闸是 `bootstrap` 的事，
而它**只管 CLI 自己所在的那个 env**）。
"""
from __future__ import annotations

import os
import subprocess
import sys

from ._registry import EXIT_FAILURE, Command

#: 承载 CV 的 conda env 名。
CV_ENV_NAME = 'GolemAI'

#: 仓库根（`.../GolemQ/GolemQ/cli/commands/chartscan.py` 往上**四层**）。
#: 子进程要用它当 cwd —— GolemAI env 没装本包，只能靠 cwd 找到 `GolemQ` 包
#: （旧树 ps1 的 `Set-Location` 是同一件事）。
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))


def cv_python() -> str:
    """GolemAI env 的 python 路径。

    ``$GOLEMQ_CV_PYTHON`` 优先；否则按 conda 的 ``<root>/envs/<name>/`` 布局，
    从**当前解释器**往上两层推（不硬编码盘符 —— 换机器/换 conda 根都不用改）。

    >>> import os, sys
    >>> p = cv_python()
    >>> os.path.basename(os.path.dirname(p))     # 末层目录名恒为 env 名
    'GolemAI'
    >>> os.path.basename(p) == os.path.basename(sys.executable)   # 同一种解释器名
    True
    """
    override = os.environ.get('GOLEMQ_CV_PYTHON', '').strip()
    if override:
        return override
    exe_name = os.path.basename(sys.executable)          # python.exe / python
    envs_dir = os.path.dirname(os.path.dirname(sys.executable))   # .../envs
    return os.path.join(envs_dir, CV_ENV_NAME, exe_name)


def in_cv_env() -> bool:
    """当前进程是不是就跑在 GolemAI env 里（→ 不必再起子进程）。"""
    target = os.path.dirname(os.path.abspath(cv_python()))
    return (os.path.normcase(os.path.realpath(target))
            == os.path.normcase(os.path.realpath(sys.prefix)))


def _re_exec(module: str, argv) -> int:
    """用 GolemAI 的 python 跑 ``module argv``，退出码原样返回。

    :param module: ``GolemQ.analysis.cv.server`` / ``...client``
    :param argv: 传给那个模块的参数列表
    """
    exe = cv_python()
    if not os.path.exists(exe):
        print('[chartscan] 找不到 {} env 的 python：{}'.format(CV_ENV_NAME, exe),
              file=sys.stderr)
        print('[chartscan] 用 $GOLEMQ_CV_PYTHON 指定，或确认 conda env 名是 {}'
              .format(CV_ENV_NAME), file=sys.stderr)
        return EXIT_FAILURE

    env = os.environ.copy()
    # 见模块 docstring：父进程的 BLAS 守卫把它们设成了 1，子进程会继承。
    threads = os.environ.get('GOLEMQ_CV_THREADS', '').strip() or '4'
    from GolemQ import BLAS_THREAD_VARS
    for name in BLAS_THREAD_VARS:
        env[name] = threads
    # 输出被重定向到文件/管道时 Python 用**块缓冲** ⇒ 长驻的 `--server` 被 kill 时
    # 「Server is running.」那行会整段丢掉（2026-10-17 实测）。强制无缓冲。
    env['PYTHONUNBUFFERED'] = '1'

    print('[chartscan] 切换到 {} env：{}'.format(CV_ENV_NAME, exe), flush=True)
    proc = subprocess.Popen([exe, '-m', module, *argv], cwd=_REPO_ROOT, env=env)
    try:
        return proc.wait()
    except KeyboardInterrupt:
        # 控制台信号会同时送到子进程；这里只负责确认它真的退出（`--server` 是长驻）。
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        return 130                       # 128 + SIGINT 的惯例值


def add_scan_arguments(parser) -> None:
    parser.add_argument('-sc', '--scan',
                        metavar='ChartScan',
                        choices=['ChartScan'],
                        default=None,
                        help="【ChartScan】K 线识别模式（客户端）：按 scope 解析出待扫任务，"
                             "逐条 REQ 发给 --server。会自动切换到 GolemAI env")
    # NOTE: `-e/--eval` 是 **全局选项**（`cli/__main__.py` 的 `eval_scope`），
    # 这里不重复注册 —— 它可与 `--scan` / `--review` / 将来的 `--strategy` 组合。


def add_server_arguments(parser) -> None:
    # ⚠️ 没有 `-s` 短开关 —— 它已被 `--setup` 占了（`setup.py:22`）。
    # 旧树的 `-s/--server` 到新树只剩长选项；`--scan` 仍保留 `-sc`。
    parser.add_argument('--server',
                        metavar='ChartScan',
                        choices=['ChartScan'],
                        default=None,
                        help="【ChartScan】CV 自服务模式（服务端）：载 YOLOv8 权重、"
                             "绑 ZeroMQ REP 端口。会自动切换到 GolemAI env")
    parser.add_argument('-p', '--port',
                        default='5555',
                        help="ZeroMQ 端口。--server 可给逗号分隔的多个（一端口一线程）；"
                             "--scan 只给一个")


def run_chartscan_scan(args) -> int:
    """`--scan ChartScan` → `chartscan.client.run_client`。

    ⚠️ **不传 `--quiet`**：这条命令一次要扫上千只、可能跑几小时，静默下来
    与"挂死"分不出来。进度条**默认就有**（与旧 ps1 / 旧 CLI 一致）——
    这也是本项目「成功静默」那条纪律的边界：静默的是**状态话术**，
    不是**长任务的进度**。
    """
    argv = ['--eval', str(args.eval_scope), '--port', str(args.port)]
    if not in_cv_env():
        return _re_exec('GolemQ.analysis.cv.client', argv)

    from GolemQ.analysis.cv.client import main as client_main
    client_main(argv)
    return 0


def run_chartscan_server(args) -> int:
    """`--server ChartScan` → `chartscan.server.main`（含逗号分隔的多端口）。

    ⚠️ 走 `main` 而不是 `run_server`：拆多端口、每端口一线程、各载一份模型那段
    逻辑只该有一处（`server.py` 里已经有了）。
    """
    argv = ['--port', str(args.port)]
    if not in_cv_env():
        return _re_exec('GolemQ.analysis.cv.server', argv)

    from GolemQ.analysis.cv.server import main as server_main
    server_main(argv)
    return 0


CHARTSCAN_SCAN = Command(
    name='chartscan-scan',
    flags=('scan',),
    add_arguments=add_scan_arguments,
    run=run_chartscan_scan,
)

CHARTSCAN_SERVER = Command(
    name='chartscan-server',
    flags=('server',),
    # `--port` 由 `add_server_arguments` 注册；parser 是**全局**的，两条命令都看得见它
    add_arguments=add_server_arguments,
    run=run_chartscan_server,
)
