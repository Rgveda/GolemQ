# coding:utf-8
"""CLI 环境自检 —— 版权页 + 机器 / 配置 / 服务三道闸。

为什么单独一个模块
==================
`cli/__main__.py` 只该负责「建 parser → 分发 → help」（`DECISIONS.md` D17）。
环境自检是**横切**的（每条命令都要），塞进 `__main__` 会让它又长回去。
（名字 2026-10-09 从「启动自检」改成「环境自检」：它检的是**这台机器够不够格跑**，
不是"启动流程"，而且它由 :func:`check_environment` 驱动 —— 一个名字一件事。）

⚠️ 三段的**时机不同**，这是刻意的
=================================
* **版权行**在 `parse_args()` **之前**打（`print_copyright`）—— 所以 `--help` 与
  argparse 的用法错误上也看得见（「全局输出」的意思）。
  ⚠️ **订正**（2026-10-09）：旧文本写的是「本地段（含自检）在 `parse_args()` 之前跑」，
  那是**错的** —— `--help` 会让 argparse 在 `parse_args()` **内部**直接 `exit(0)`，
  自检根本轮不到。所以自检从来不曾在 `--help` 上出现过，只有版权行在。
* **本地自检**（`check_environment`）在 `parse_args()` **之后**、**知道是哪条命令之后**跑：
  详略要由 `--verbose` 定，严格度要由「是不是 `REPAIR_COMMANDS`」定。
* **服务段**（MongoDB 连接 + 8.3 版本）在自检之后跑，且只对 `Command.needs_db` 的命令。
  两个理由：
  1. 不该为了打帮助去连一次库；
  2. **`--setup` / `--mongodb-init` 绝不能要求 DB** —— 否则配置连错了就永远修不了。
  它连不上时**硬失败**（`EXIT_FAILURE`）：那不是"某个集合没取到"，是整条命令跑不了。

失败口径（`DECISIONS.md` D19：用法错=2 / 运行期失败=1）
=====================================================
* **硬拦**：`python` 与 `依赖包` 这两项不过 → `EXIT_FAILURE`（`ENV_GATE_NODES`）。
  它们是**跑不动**而不是「跑得不好」：python 太老这棵树可能根本 import 不进来，
  依赖包缺了主链路立刻断。
  ⚠️ **但 `REPAIR_COMMANDS` 放行** —— 那四条是「修配置」的入口，
  拦掉等于「环境不对就永远修不回来」，与 `require_mongodb` 不为配置类命令连库同理。
* **只警告不拦**：其余全部（操作系统 / 线程环境 / CPU 架构 / CUDA / 时区）。
  它们可能只是「这台机器不适用」（无 N 卡）或用户显式选择（export 过线程上限），
  判红会误伤。
* 配置文件缺失/不可解析 → 只警告（**不进 banner 的节点**，见 :func:`check_environment`）；
  **只有真要连库时才因它而失败**（见 :func:`require_mongodb`）。

屏上形态
========
本地段自检**不再逐行 `print`**，而是挂在与 `--save` 同一套 `core.presentation.Banner` 上
（单阶段一行平铺 + 四态点：绿=通过 / 黄=警告 / 红=失败 / 灰·=未检查）。
所以本模块**必须遵守 banner 的记账约定**：活跃期间一个 `print` 都不许有，
一切都走 `banner.echo`（`PITFALLS.md` P22）。
"""
from __future__ import annotations

import configparser
import datetime
import os
import platform
import re
import shutil
import subprocess
import sys

from .commands._registry import EXIT_FAILURE

#: 四态常量与 banner —— 在**模块级**导入是安全的：`import GolemQ` 本身就会拉进
#: pandas / numpy / pymongo（2026-10-09 实测 `sys.modules`），所以「pandas 没装 →
#: 自检自己先崩、红点变堆栈」这条路**走不到**。版本不达（如 pandas 2.2.3 < 2.3）
#: 才是真实场景 —— 那时 pandas 在、导入照样成功，红点报得出来。
from GolemQ.core.presentation import FAIL, OK, PENDING, WARN

#: 版权行 —— `cli/__main__.py` 每次运行都打（转出到这里，别处不要再抄一份）。
copyright_infos = ("Copyright (c) 2018-2026 azai/Rgveda/GolemQ(uant)"
                   " | https://github.com/Rgveda | 知乎@阿财")

#: Python 最低版本。
MIN_PYTHON = (3, 12)

#: 依赖包最低版本。装的是 `conda env`，以**实装版本**为准判。
#:
#: **pandas 为什么是 2.3**：3.0 之前的**最后一个 2.x 是 2.3.3**（2025-09-29 发布；
#: 3.0.0 是 2026-01-21）。而且 pandas **官方升级路径**就是「先升到 2.3、把告警清干净，
#: 再上 3.0」—— 所以门槛定在 2.3 既是最高的 2.x，也**顺带接纳 3.x**
#: （版本元组比较 3.x > 2.3，不需要再写一条）。
MIN_PACKAGES = (
    ('numpy', '2.0'),
    ('pandas', '2.3'),
    ('polars', '1.24'),
    # 下面三个是 2026-10-09 补的**主链路硬依赖**（全树出现次数：pymongo 9 / tqdm 8 /
    # pytdx 9）。此前 `pyproject.toml` **不声明任何运行期依赖**，所以这里是全树
    # 唯一一处依赖声明 —— 缺了它们，`--save` 与所有要连库的命令当场就断。
    ('pymongo', '3.0'),
    ('tqdm', '4.0'),
    ('pytdx', ''),        # ⚠️ `''` = **只查能不能 import**（pytdx 没有 `__version__`，实测）
)

#: MongoDB 最低版本 —— 本项目**只用 8.3**（`CLAUDE.md`「Single database」）。
MIN_MONGODB = (8, 3)

#: 连库自检的超时（毫秒）。默认 30s 太久 —— 服务器不可达时是**一次挂死**，
#: 自检的意义就是别让人干等。
MONGO_TIMEOUT_MS = 3000

#: 自检 banner 的节点，**顺序即屏上顺序**（`DECISIONS.md` D17 的先例：顺序是语义）。
SELF_CHECK_NODES = ('操作系统', 'python', '依赖包', '线程环境', 'CPU 架构', 'CUDA',
                    '时区', '交易日历', 'tdxidata', 'tushare', 'iwencai',
                    'serverchan', '讯投QMT')

#: 交易日历「**该续下一年了**」的分界（月, 日）—— 用户 2026-10-10 定。
#: 过了这一天而日历仍只到今年年底 ⇒ 黄点提醒（次年的安排通常那时已经公布）。
CALENDAR_RENEW_AFTER = (11, 10)

#: **可选源**节点 → 适配器名（`markets/StockCN/datasource/` 的注册键）。
#: ⚠️ `iwencai` **不在**这张表里 —— 它不是 `datasource/` 的适配器（见 `check_iwencai`）；
#: `讯投QMT` 也**不在** —— 它的适配器恒不可用，用户要的是查配置段（见 :data:`XTQUANT_KEYS`）。
OPTIONAL_SOURCES = {'tdxidata': 'tdxaidata', 'tushare': 'tushare'}

#: `讯投QMT`（迅投 QMT）要检查的**配置项** —— 用户 2026-10-10 明确「检查的是这一段」。
#: ⚠️ 它**不走** `OPTIONAL_SOURCES`：那个是「适配器可用吗」，而 QMT 的适配器
#: **恒不可用**（`QMT_SOURCE_ENABLED = False`，MiniQMT 已停服，D13）。
XTQUANT_KEYS = ('account', 'min_path')

#: 自检 banner 的表头 —— **两栏**（用户 2026-10-10 定，换行位置同日调整）。
#:
#: 13 个节点挤一行太长（2026-10-09 那版是 7 个），故按**语义**折成两栏：
#: **上栏 = 「本机」**（操作系统/python/依赖包/线程环境/CPU 架构/CUDA/**时区** ——
#: 时区是**本机属性**，跟着机器走），**下栏 = 「外部依赖」**
#: （交易日历/txidata/tushare/iwencai/serverchan/讯投QMT）。
#: 阶段名只在第一行打（`render_pipeline_banner` 对连续同名阶段的行为），
#: 第二行留白对齐 —— 看起来仍是**一块**，只是折了两行。
#:
#: ⚠️ **两栏拼起来必须逐项等于 `SELF_CHECK_NODES`**（顺序即语义）—— 有用例钉着。
SELF_CHECK_ROWS = (
    ('环境自检', None, ['操作系统', 'python', '依赖包', '线程环境', 'CPU 架构', 'CUDA',
                        '时区']),
    ('环境自检', None, ['交易日历', 'tdxidata', 'tushare', 'iwencai',
                        'serverchan', '讯投QMT']),
)

#: `nvidia-smi` 两次调用的超时（秒）。实测本机 `--query-gpu` 47ms + 全量 126ms，
#: 这个上限只是为了别在一台驱动装坏的机器上挂死。
_CUDA_TIMEOUT_S = 10

#: **硬拦项** —— 不过就退出。见模块 docstring 的失败口径。
ENV_GATE_NODES = ('python', '依赖包')

#: 这四条命令**连硬拦项也放行** —— 它们是「修配置」的入口（`Command.name`）。
REPAIR_COMMANDS = frozenset({'setup', 'mongodb-init', 'dingtalk-init', 'serverchan-init'})


def _parse_version(text):
    """``'8.3.11'`` → ``(8, 3, 11)`` —— 只认打头的数字段，非数字尾巴丢掉。

    >>> _parse_version('2.2.3')
    (2, 2, 3)
    >>> _parse_version('8.3')
    (8, 3)
    >>> _parse_version('1.27.1+local')
    (1, 27, 1)
    >>> _parse_version('unknown')
    ()
    """
    parts = []
    for chunk in str(text).split('.'):
        digits = ''
        for ch in chunk:
            if not ch.isdigit():
                break
            digits += ch
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


def _meets(found, want):
    """``found`` 是否 >= ``want``（两个都是版本元组）。**空元组算不满足**。

    >>> _meets((2, 2, 3), (2, 3))      # 本机 pandas 实装版本：不满足
    False
    >>> _meets((2, 3, 3), (2, 3))      # 2.x 的最后一版：满足
    True
    >>> _meets((3, 0, 0), (2, 3))      # 3.x 也满足（元组比较，不用另写一条）
    True
    >>> _meets((), (2, 3))             # 取不到版本 = 不满足
    False
    """
    if not found:
        return False
    return tuple(found) + (0,) * (len(want) - len(found)) >= tuple(want)


def _fmt(version_tuple):
    return '.'.join(str(x) for x in version_tuple)


def check_python():
    """``(ok, 说明)``。``sys.version_info`` 前两位与本项目声明的最低版本比。"""
    found = tuple(sys.version_info[:3])
    ok = _meets(found, MIN_PYTHON)
    return ok, 'python {}（要求 >= {}）'.format(_fmt(found), _fmt(MIN_PYTHON))


def check_packages():
    """逐包检查，返回 ``[(ok, 说明), ...]``。**没装的**也算不满足并说明。

    ``want=''`` = **只查能不能 import**，不判版本。⚠️ 这条不是偷懒：`pytdx`
    **没有 `__version__`**（实测 2026-10-09），若照旧走 `_meets`，它会被判成
    「不满足」（取不到版本 = 不满足，见 :func:`_meets`）—— 一个装了却永远报红的包。
    """
    out = []
    for name, want in MIN_PACKAGES:
        try:
            module = __import__(name)
        except ImportError:
            out.append((False, '{} 未安装{}'.format(
                name, '' if not want else '（要求 >= {}）'.format(want))))
            continue
        found = _parse_version(getattr(module, '__version__', ''))
        want_t = _parse_version(want)
        if not want_t:
            out.append((True, '{} {}（已安装，不判版本）'.format(
                name, _fmt(found) or '无版本号')))
            continue
        out.append((_meets(found, want_t),
                    '{} {}（要求 >= {}）'.format(name, _fmt(found) or '未知版本', want)))
    return out


def check_tty():
    """``(是色终端吗, 说明)``。

    **复用** :func:`core.presentation.ansi_enabled` —— 它同时看了 `isatty()`
    与 Windows 的 VT 支持（后者不加会让「光标上移」失效、屏上叠字）。
    别在这里另写一套 TTY 判断。

    ⚠️ **结果只作信息，不算"不通过"**：把输出重定向到文件/管道是**正常用法**，
    每次都报一个 ⚠️ 就是噪声。所以它只在 `-v` 下以 `ℹ` 打出来
    （见 :func:`check_environment`）。
    """
    from GolemQ.core.presentation import ansi_enabled
    ok = ansi_enabled()
    return ok, ('彩色终端（ANSI 可用）' if ok
                else '非彩色终端（管道/重定向，或控制台不支持 ANSI）—— '
                     '输出走无颜色、无重绘的降级形态')


def check_blas():
    """``(ok, 说明)``：numpy 的 BLAS/OpenMP **线程数上限**有没有压到 1。

    ⚠️ 「没设」= `GolemQ/__init__.py` 顶部那段守卫**被删了**。后果实测（本机 18 核）：
    单进程提交量 **6573.9 MB → 19.8 MB**；`jobs=4` 时 **32.1 GB → 0.43 GB**。
    撞的是 Windows 的 commit limit，表现是「越跑越慢、迟迟不结束」而不是报错。

    ⚠️ **值也看，不只看"设没设"**（2026-10-09 补）：`OPENBLAS_NUM_THREADS=4` 与
    「压根没设」对提交量的后果**几乎一样**（`jobs=4` → 4 worker × 4 线程），
    而旧版只看 `is None`，于是那种情况报绿。两种都返回 `ok=False`（调用方映射成**黄**）：
    守卫被删要提醒；显式改过也要提醒，但那是你的选择，**不拦**。
    """
    from GolemQ import BLAS_THREAD_VARS
    vals = {name: os.environ.get(name) for name in BLAS_THREAD_VARS}
    missing = [name for name, value in vals.items() if value is None]
    if missing:
        return False, ('未设 {} —— `GolemQ/__init__.py` 的守卫是不是被删了？'
                       '多进程会成倍撑大提交量'.format(', '.join(missing)))
    not_one = {name: value for name, value in vals.items() if str(value).strip() != '1'}
    if not_one:
        return False, ('{} 不是 1 —— 应为 1（一个进程一个 BLAS 线程，'
                       '并行度交给 worker 数）；若是你显式 export 的，忽略这条'
                       .format('、'.join('{}={}'.format(k, v)
                                         for k, v in sorted(not_one.items()))))
    return True, '线程上限 1（numpy/OpenBLAS）'


# ---------------------------------------------------------------------------
# 下面四条是 2026-10-09 新增的「这台机器怎么样」类检查。
# 它们**都不硬拦**（见模块 docstring 的失败口径）—— 只把事实摆到屏上。
# ---------------------------------------------------------------------------
def check_os():
    """``(状态, 说明)``：操作系统 + 位数。

    Windows 那条判据不是摆设：banner 的「光标上移」要在 conhost 上生效，靠的是
    :func:`core.presentation._vt_supported` 去开 VT —— 那套 API **Windows 10 才有**。
    非 Windows 一律按平台报告（Linux/Darwin 直接通过，ANSI 本来就在）。
    """
    name = platform.system() or '未知系统'
    bits = 64 if sys.maxsize > 2 ** 32 else 32
    detail = '{} {} · {} 位'.format(name, platform.release() or '?', bits)
    if bits != 64:
        return WARN, detail + ' —— 非 64 位'
    if name == 'Windows' and not _meets(_parse_version(platform.version())[:2], (10, 0)):
        return WARN, detail + ' —— Win10 以下没有 VT，banner 的光标记账会失效（屏上叠字）'
    return OK, detail


def _cpu_topology():
    """→ ``(逻辑线程数, 物理核数|None, {EfficiencyClass: 核数})``。

    Windows 走 **`GetSystemCpuSetInformation`**（文档化 API）；Linux 走 `/proc/cpuinfo`；
    取不到的项一律 `None` —— **宁可说不知道，别猜**。

    ⚠️ **两个坑，都是 2026-10-09 在本机当场踩到的**（Ultra 5 250K Plus，18 核 18 线程）：

    1. **条目步长必须读条目头的 `Size` 字段**（本机 32 字节），别自己按
       `PROCESSOR_RELATIONSHIP` 硬算 —— 我按 `24+16*GroupCount` 算错过，
       解析出 **22 个核**（真值 18）与一串垃圾 EfficiencyClass，而它**不报错**。
    2. **别读 `AllFlags` 的 `Smt` 位** —— 本机实测它报「4 个核有超线程」，
       而这台机器 **18/18 无超线程**（CIM 的 `NumberOfLogicalProcessors` 也是 18 佐证）。
       超线程判据用**逻辑数 > 物理数**，别用那个 flag bit。
    """
    logical = os.cpu_count()
    if sys.platform == 'win32':
        try:
            import ctypes
            from ctypes import wintypes

            k32 = ctypes.windll.kernel32
            need = wintypes.ULONG(0)
            k32.GetSystemCpuSetInformation(None, 0, ctypes.byref(need), None, 0)
            if not need.value:
                return logical, None, {}
            buf = ctypes.create_string_buffer(need.value)
            if not k32.GetSystemCpuSetInformation(buf, need.value, ctypes.byref(need),
                                                  None, 0):
                return logical, None, {}
            raw, off, end = buf.raw, 0, need.value
            cores, classes = set(), {}
            while off + 8 <= end:
                size = int.from_bytes(raw[off:off + 4], 'little')
                if size <= 0:
                    break
                if int.from_bytes(raw[off + 4:off + 8], 'little') == 0:  # CpuSetInformation
                    # 组号 + 核号 —— 多组（>64 逻辑处理器）时组号必须参与去重
                    cores.add((raw[off + 12:off + 14], raw[off + 15]))
                    eff = raw[off + 18]                                  # EfficiencyClass
                    classes[eff] = classes.get(eff, 0) + 1
                off += size
            return logical, (len(cores) or None), classes
        except Exception:      # noqa: BLE001 探不到就当不知道（只报逻辑数）
            return logical, None, {}
    return logical, _physical_cores_posix(), {}


def _physical_cores_posix():
    """Linux/macOS 的物理核数：数 `/proc/cpuinfo` 里**不重复的 (physical id, core id)**。

    取不到（macOS 没有 `/proc`、或字段缺失）返回 `None` —— 调用方按「物理核未知」报。
    """
    try:
        pairs, phys, core = set(), None, None
        with open('/proc/cpuinfo', encoding='utf-8', errors='replace') as fh:
            for line in fh:
                if line.startswith('physical id'):
                    phys = line.split(':', 1)[1].strip()
                elif line.startswith('core id'):
                    core = line.split(':', 1)[1].strip()
                elif not line.strip():
                    if phys is not None and core is not None:
                        pairs.add((phys, core))
                    phys = core = None
        return len(pairs) or None
    except Exception:          # noqa: BLE001
        return None


def _cpu_vendor():
    """`Intel` / `AMD` / 其它原样 —— 只用于说明，判不了就 `'未知厂商'`。"""
    text = platform.processor() or ''
    low = text.lower()
    if 'genuineintel' in low or 'intel' in low:
        return 'Intel'
    if 'authenticamd' in low or 'amd' in low:
        return 'AMD'
    try:
        with open('/proc/cpuinfo', encoding='utf-8', errors='replace') as fh:
            for line in fh:
                if line.lower().startswith(('vendor_id', 'model name')):
                    return line.split(':', 1)[1].strip()
                if line.strip() == '':
                    break
    except Exception:          # noqa: BLE001
        pass
    return '未知厂商'


def check_cpu():
    """``(状态, 说明)``：厂商 · 物理核/逻辑线程 · 有无超线程 · 是否混合大小核。

    **为什么值得看**：本项目所有并发数都是**硬编码**的（`kline_save.DEFAULT_JOBS=4`、
    `tdx_hosts.PROBE_WORKERS=16`、`--save-jobs` 默认 4），**没有一处**读 CPU 数
    （2026-10-09 全树核实：`cpu_count` 零命中）。所以这条**不承重**，它的价值在于
    一旦将来要按核数调 `jobs`，屏上先有一份可信的拓扑 —— 尤其**混合大小核**机器，
    按「逻辑核数」开线程会开到 12 颗 E-core 上去。

    ⚠️ **快慢方向不进代码**：`EfficiencyClass` 两个类之间**只报分布**，不贴 P/E 标签。
    本机实测「值大 = 快」（class 1 的 6 颗微基准 0.0536s vs class 0 的 12 颗 0.0683s，
    对上 6P+12E 的规格），但那是**一台机器的一个观测**，不足以当 API 的语义用。
    """
    logical, physical, classes = _cpu_topology()
    parts = []
    if physical:
        parts.append('{} 物理核'.format(physical))
    else:
        parts.append('物理核未知')
    parts.append('{} 逻辑线程'.format(logical or '?'))
    if physical and logical:
        parts.append('超线程开（{} 路）'.format(logical // physical)
                     if logical > physical else '无超线程')
    if len(classes) > 1:
        parts.append('混合大小核 {}'.format(
            '+'.join(str(n) for _, n in sorted(classes.items(), reverse=True))))
    elif classes:
        parts.append('非混合架构')
    return (OK if physical else WARN), '{} · {}'.format(_cpu_vendor(), ' · '.join(parts))


def check_cuda():
    """``(状态, 说明)``：有 N 卡就报卡型 + 驱动 + 驱动支持的 CUDA 版本。

    ⚠️ **永远不判失败，探不到就是「未检查」**（灰）。理由：新树**零 GPU 依赖**
    （2026-10-09 全树核实：`torch / cupy / xgboost / numba.cuda / cuda` **0 命中**；
    `NUMBA_NUM_THREADS` 在守卫里但 numba 从未 import）。把「没有 CUDA」判成失败，
    会让每台没显卡的机器、每个 CI 都变红 —— 而那条路根本不跑 GPU。

    实测 `nvidia-smi` 很便宜（本机 `--query-gpu` 47ms + 全量 126ms），值得每次探。
    """
    exe = shutil.which('nvidia-smi')
    if not exe:
        return PENDING, '未检出 nvidia-smi（无 N 卡 / 未装驱动 —— 不适用）'
    try:
        query = subprocess.run(
            [exe, '--query-gpu=name,driver_version', '--format=csv,noheader'],
            capture_output=True, text=True, errors='replace', timeout=_CUDA_TIMEOUT_S)
        cards = [ln.strip() for ln in (query.stdout or '').splitlines() if ln.strip()]
        if query.returncode != 0 or not cards:
            return PENDING, 'nvidia-smi 无输出（不适用）'
        detail = ' · '.join(cards)
        #: CUDA 版本只在全量输出的横幅里（`--query-gpu` 有 compute_cap，但那不是 CUDA 版本）
        full = subprocess.run([exe], capture_output=True, text=True,
                              errors='replace', timeout=_CUDA_TIMEOUT_S)
        found = re.search(r'CUDA Version:\s*([\d.]+)', full.stdout or '')
        if found:
            detail += ' · CUDA {}'.format(found.group(1))
        return OK, detail
    except Exception as exc:      # noqa: BLE001 驱动坏了也只报「不适用」，不拦人
        return PENDING, 'nvidia-smi 探测失败（{}: {}）'.format(type(exc).__name__, exc)


def check_source(name):
    """``(状态, 说明)``：某个**可选数据源**能不能用（配了 token / 包在不在）。

    ⚠️ 判据**复用数据层自己的** `available()` / `unavailable_reason()` ——
    `markets/StockCN/datasource/` 的适配器**都实现了这两个**（`qmt_source.py` 的注释
    还专门点过：「别处四个源（baostock / eastmoney / tdxaidata / tushare）都实现了」）。
    **不要在 CLI 里重写一套「配置了没」** —— 那正是「平行实现不会报错，只会分叉」，
    而且配置键名会散成两处真相。

    **未配置 ⇒ 灰（`:data:`PENDING`）**，不是红也不是黄：这些都是**可选源**
    （主源是 pytdx），没配不影响任何命令跑得通。灰点在这里表达的是
    「**没去检查 / 不适用**」—— 与 CUDA 那条同一个口径。

    :param name: 适配器注册名（`OPTIONAL_SOURCES` 的值）
    """
    try:
        # 函数内导入：`markets/StockCN/__init__.py` 那一串不轻（quotes → easyquotation…），
        # 而本函数在**每条命令**的启动自检里都会跑。
        from GolemQ.markets.StockCN.datasource import get_source
        src = get_source(name)
    except Exception as exc:      # noqa: BLE001 源没注册也算「探不到」，不拦启动
        return PENDING, '取不到该源适配器（{}: {}）'.format(type(exc).__name__, exc)
    try:
        if src.available():
            return OK, '已配置（适配器 {}）'.format(getattr(src, 'name', name))
        return PENDING, src.unavailable_reason()
    except Exception as exc:      # noqa: BLE001
        return PENDING, '探测失败（{}: {}）'.format(type(exc).__name__, exc)


def _ini_value(section, option):
    """读 ``~/.GolemQ/settings/config.ini`` 的某一项。

    :returns: 去掉首尾空白的值（**键不存在返回 `''`**）；**整个文件/段读不到返回 `None`**
        （不抛 —— 由调用方按"缺"处理）。

    ⚠️ `check_xtquant` 与 `_config_uri` 都走它（读 config.ini 的"取值"只此一处）。
    **`check_config` 不走** —— 它要区分「文件不在 / 解析失败 / 没有该键」三种错并
    各报各的，只能在 `open` 那一层自己接异常。**别硬把它塞进来**：那会把三句
    不同的报错压成一句「读不到」。
    """
    from GolemQ.core.path import setting_path
    try:
        parser = configparser.ConfigParser()
        with open(os.path.join(setting_path, 'config.ini'), encoding='utf-8') as fh:
            parser.read_file(fh)
        return (parser.get(section, option, fallback='') or '').strip()
    except Exception:      # noqa: BLE001
        return None


def check_xtquant():
    """``(状态, 说明)``：`[XTQUANT]` 那一段**配置**齐不齐。

    ⚠️ 用户 2026-10-10 明确：「**讯投QMT 检查的是这一段** `[XTQUANT] account / min_path`」
    —— 所以本节点判的是「**配置到位**」，**不是**"QMT 现在能不能用"。

    ⚠️ **两者不是一回事，所以 detail 里必须都写**：MiniQMT 自 **2026-10-01 停服**
    （`DECISIONS.md` D13），取数与订阅**都已关闭**（`QMT_SOURCE_ENABLED = False`）。
    ⇒ **绿点只表示「配置齐」，不表示这条路可用。** 判据既然按用户口径定在配置上，
    就不能让那个事实从屏上消失（那才是真的会误导人）。
    """
    vals = {k: _ini_value('XTQUANT', k) for k in XTQUANT_KEYS}
    if vals is None or any(v is None for v in vals.values()):
        return WARN, '读不到 config.ini 的 [XTQUANT] 段'
    missing = [k for k, v in vals.items() if not v]
    if missing:
        return WARN, ('缺 {} —— 在 ~/.GolemQ/settings/config.ini 的 [XTQUANT] 补上'
                      .format('/'.join(missing)))
    return OK, ('{} 已配；⚠️ MiniQMT 自 2026-10-01 停服（D13）取数与订阅均已关闭，'
                '**绿点只表示配置齐、不表示这条路可用**'
                .format('/'.join(XTQUANT_KEYS)))


def check_serverchan():
    """``(状态, 说明)``：Server酱（推送告警渠道）配了没。

    **判据复用** :func:`agents.messenger.check_serverchan_config`（它读
    `[SERVERCHAN] sendkey` 并排除空值/默认值）—— 不在 CLI 里重读一遍配置：
    那是第二处真相，改了 key 名两边就会不同步。

    **未配置 ⇒ 灰**：推送是**可选**渠道（不配就是不发通知，不影响任何命令）——
    与 `tushare` / `tdxidata` 那几个可选源同一口径。
    """
    try:
        from GolemQ.agents.messenger import check_serverchan_config
        if check_serverchan_config():
            return OK, '已配置（[SERVERCHAN] sendkey）'
        return PENDING, '未配置 [SERVERCHAN] sendkey —— 配了才会推送告警'
    except Exception as exc:      # noqa: BLE001 探测失败不该拦启动
        return PENDING, '探测失败（{}: {}）'.format(type(exc).__name__, exc)


def check_iwencai():
    """``(状态, 说明)``：东方财富**问财**的配置。

    ⚠️ **新树尚未实现**（2026-10-10 核实）：`services/iwencai.py` 只有一个 `__init__`
    （造了个空 index），**零请求逻辑、零调用点**（全树只有那个文件自己提到 iwencai）。
    老树那边是**爬虫**（`GQ_SU_crawl_stock_*_from_iwencai_*`），**也没有 token 配置**。

    ⇒ **没有可判的配置项**，故恒为**灰**并在 detail 里说明现状。
    **不发明一个没人读的配置键** —— 那等于假配置（`config.ini` 里放个 token
    却没有任何代码读它，比不放更坏）。真要接入，先把抓取实现搬过来。
    """
    return PENDING, ('新树尚未实现：`services/iwencai.py` 是空壳（零请求逻辑、'
                     '零调用点），没有可判的配置项')


def check_calendar(calendar=None, today=None):
    """``(状态, 说明)``：交易日历（`TRADE_DATE_SSE`）**够不够用**。

    **为什么值得单独摆一个节点**：全树所有「今天是不是交易日 / 上一个交易日是哪天」
    都读它（`kline_doc.alive_threshold`、`trade_days_between`、`GQ_util_if_trade`…），
    而它是一张**手维护的静态表** —— 过期了**不会报错**，只会让"今天"被静默判成
    非交易日（于是短路判据、TTL、调度全按错的日子走）。

    判据（用户 2026-10-10 定，**外加一档见末行**）：

    | 情形 | 状态 |
    |:--|:--|
    | 末端 **< 今天** | **红** —— 日历已过期 |
    | 末端 = **今年年底**，且今天 **<= 11-10** | **绿** —— 正常，明年的还没到公布时候 |
    | 末端 = **今年年底**，且今天 **> 11-10** | **黄** —— 该续下一年了 |
    | 末端 **> 今天但 < 今年年底** | **黄** —— ⚠️ **这一档用户没给，是我补的**：还能跑，但覆盖不够长 |
    | 日历空 / 末端不是 `YYYY-MM-DD` | **红** |

    :param calendar: 交易日列表（``'YYYY-MM-DD'`` 字符串，字典序即时间序）；
        ``None`` = `TRADE_DATE_SSE`。**可注入**是为了能测（纯函数）
    :param today: 可注入的"今天"（`date` / `datetime` / 字符串都行），``None`` = 现在
    """
    if calendar is None:
        from GolemQ.markets.StockCN.constants import TRADE_DATE_SSE as calendar
    today_s = str(today)[:10] if today is not None else datetime.date.today().isoformat()
    last = calendar[-1] if calendar else None
    if not last:
        return FAIL, '交易日历是**空的** —— 全树所有「今天是不是交易日」都会退化'
    try:
        year = int(str(last)[:4])
        int(str(last)[5:7])
        int(str(last)[8:10])
    except ValueError:
        return FAIL, '日历末端 {!r} 不是 YYYY-MM-DD'.format(last)

    if last < today_s:
        return FAIL, ('末端 {} **早于今天 {}** —— 之后的日期全被判成非交易日，'
                      '短路判据 / TTL / 调度都会按错的日子走'.format(last, today_s))

    yearend = '{}-12-31'.format(year)
    renew_at = '{}-{:02d}-{:02d}'.format(year, *CALENDAR_RENEW_AFTER)
    if last >= yearend:
        if today_s <= renew_at:
            return OK, '覆盖到 {}（{} 个交易日）'.format(last, len(calendar))
        return WARN, ('末端 {} 只到今年年底，而今天已过 {} —— **该续下一年了**'
                      .format(last, renew_at))
    return WARN, ('末端 {} **早于今年年底 {}** —— 还能跑，但覆盖不够长'
                  .format(last, yearend))


def check_tz():
    """``(状态, 说明)``：本机时区是不是北京时间（UTC+08:00）。

    ⚠️ 这条**是真会静默错的**，不是装饰：`kline_save.py:396-398` 用**裸
    `datetime.now()`** 当北京时间判「收盘」（`(hour, minute) >= (15, 0)`），
    `today` 也取自它。机器时区一偏，那个「盘中不写日线」的闸就在**错的时间**
    开合，而且**不报错**。
    """
    now = datetime.datetime.now().astimezone()
    off = now.utcoffset() or datetime.timedelta(0)
    total = int(off.total_seconds() // 60)
    detail = '{}（UTC{}{:02d}:{:02d}）'.format(
        now.tzname() or '未知时区', '+' if total >= 0 else '-',
        abs(total) // 60, abs(total) % 60)
    if off == datetime.timedelta(hours=8):
        return OK, detail
    return WARN, detail + ' —— 不是北京时间；kline_save 的收盘闸按裸 datetime.now() 判，会静默错开'


def check_config():
    """``(ok, 说明)``。查 ``~/.GolemQ/settings/config.ini`` 在不在、能不能解析、
    有没有 ``[MONGODB] uri``。

    ⚠️ 路径是 ``~/.GolemQ/settings/config.ini``（`core.path.setting_path`），
    **不是** ``~/.GolemQ/config.ini`` —— 后者只在 `setup_mongodb_config` 的
    一处老逻辑里出现过，`CLAUDE.md` 的「Configuration File」一节把它写成了前者。
    """
    from GolemQ.core.path import setting_path
    path = os.path.join(setting_path, 'config.ini')
    if not os.path.exists(path):
        return False, '没找到配置文件 {}'.format(path)
    parser = configparser.ConfigParser()
    try:
        with open(path, encoding='utf-8') as fh:
            parser.read_file(fh)
    except Exception as exc:      # noqa: BLE001 解析失败要报人话，不要堆栈
        return False, '{} 解析失败（{}: {}）'.format(path, type(exc).__name__, exc)
    uri = (parser.get('MONGODB', 'uri', fallback='') or '').strip()
    if not uri:
        return False, '{} 里没有 [MONGODB] uri'.format(path)
    return True, '{}（uri 已配）'.format(path)


def check_mongodb(uri):
    """``(ok, 说明)``。连上并取 `server_info()` 判版本 —— 本项目**只用 8.3**。

    :param uri: 连接串；来自 :func:`check_config`（配置文件是本项目唯一的出处）。
    """
    from GolemQ.core.mongo import GQ_util_mongodb_client
    client = None
    try:
        client = GQ_util_mongodb_client(uri, serverSelectionTimeoutMS=MONGO_TIMEOUT_MS)
        version = str(client.server_info().get('version', ''))
    except Exception as exc:      # noqa: BLE001 连不上/超时都要报人话
        return False, '连不上（{}: {}）'.format(type(exc).__name__, exc)
    finally:
        if client is not None:
            try:
                client.close()
            except Exception:     # noqa: BLE001 关闭失败无所谓
                pass
    ok = _meets(_parse_version(version), MIN_MONGODB)
    return ok, 'MongoDB {}（要求 >= {}）'.format(
        version or '未知版本', _fmt(MIN_MONGODB))


def print_copyright():
    """只打版权那一行（**兼容保留**；新版用 :func:`print_identity`）。"""
    print(copyright_infos, flush=True)


def print_identity(when=None):
    """**身份块** —— 产品名 + 本次运行的时刻戳 + 版权行 + 一个空行。

    **与自检分开**：它在 `parse_args()` **之前**打（所以 `--help` 与用法错误上也看得见），
    而自检要等 `--verbose` 解析出来才能决定详略。

    ⚠️ 时刻戳从此**只在这里出现一次**（2026-10-09 改版）—— 原先是两个 banner
    各打一条 `[戳]: …`，紧挨着像同一件事说了两遍。`Banner(header=False)` 让阶段行
    不再带头行，于是 `环境自检` / `数据源` / `参考数据` / `K线` / `复权`
    **跨两个 banner 对齐成一列**。见 :func:`core.presentation.identity`。
    """
    from GolemQ.core.presentation import ansi_enabled, identity
    # 版权/联系行**压暗**（层次再进一层，`brew`/`npm` 的老做法）—— 但只在真 TTY 上：
    # 非 TTY（管道/日志）里绝不能掺转义码，那是本项目颜色规则的第一条。
    print(identity('GolemQ', copyright_infos, when=when, color=ansi_enabled()),
          end='', flush=True)


def _as_lines(result):
    """``(状态, '说明')`` → ``(状态, ['说明'])`` —— 明细行**一律是 list**。

    ⚠️ 这一层不是为了好看：`:func:`run_checks`` 里有两种写法（`('python',) + (state, [d])`
    与 `('python', state, [d])`），漏了它就会出现「一半是 list、一半是裸字符串」，
    而字符串是**可迭代的** —— 下游 `for line in lines` 会**逐字**迭代，
    屏上打出 `W | i | n | d | o | w | s`（2026-10-09 实测踩到）。
    """
    state, detail = result
    return state, detail if isinstance(detail, list) else [detail]


def run_checks(packages=None):
    """跑完七个节点，**只返回结果、不打印**。

    → ``[(节点名, 状态, [明细行, ...]), ...]``，顺序同 :data:`SELF_CHECK_NODES`。

    :param packages: `check_packages()` 的结果（可由调用方传进来复用，省一次 import）。

    明细行是**给 `-v` 用的**多行文本（`依赖包` 就是逐包一行）；非 verbose 下只有
    硬拦项不过时才把它打出来。这样「节点 = 一个点」与「明细 = 一段文字」两件事
    分得干净 —— 点回答"对不对"，文字回答"为什么"。
    """
    packages = check_packages() if packages is None else packages
    ok_py, py_detail = check_python()
    ok_blas, blas_detail = check_blas()
    bad_pkg = [d for ok, d in packages if not ok]
    return [
        ('操作系统',) + _as_lines(check_os()),
        ('python', OK if ok_py else FAIL, [py_detail]),
        #: ⚠️ `依赖包` 的节点名就是 `ENV_GATE_NODES` 里取的那个键。
        ('依赖包', FAIL if bad_pkg else OK, [d for _, d in packages]),
        #: 线程环境**归黄不归红**：守卫用的是 `setdefault`，你显式 export 也算"不全是 1"
        #: —— 那是你的选择，不是环境坏了（`PITFALLS.md` P21 的代价仍然由 `check_blas` 说明）。
        ('线程环境', OK if ok_blas else WARN, [blas_detail]),
        ('CPU 架构',) + _as_lines(check_cpu()),
        ('CUDA',) + _as_lines(check_cuda()),
        ('时区',) + _as_lines(check_tz()),
        ('交易日历',) + _as_lines(check_calendar()),
        ('tdxidata',) + _as_lines(check_source(OPTIONAL_SOURCES['tdxidata'])),
        ('tushare',) + _as_lines(check_source(OPTIONAL_SOURCES['tushare'])),
        ('iwencai',) + _as_lines(check_iwencai()),
        ('serverchan',) + _as_lines(check_serverchan()),
        ('讯投QMT',) + _as_lines(check_xtquant()),
    ]


def _notes():
    """**不进 banner 节点**的附加信息 → ``[(标签, 说明, 是否算问题), ...]``。

    这两条刻意不占节点：用户 2026-10-09 定的节点清单里没有它们，但它们各自
    有存在的理由 ——
    * `终端`：非 TTY 是**正常用法**（管道/重定向），所以 `是否算问题` **恒为 False**，
      只在 `-v` 下以信息行的形式出现；
    * `配置`：配置文件缺失/不可解析是真问题，但**不必硬拦** —— 只有真要连库时
      才由 :func:`require_mongodb` 因它而失败（配置类命令更不该被它拦）。
    """
    _, tty_detail = check_tty()
    ok_cfg, cfg_detail = check_config()
    return [('终端', tty_detail, False), ('配置', cfg_detail, not ok_cfg)]


def check_environment(verbose=False, strict=False):
    """**本地段自检**：画 banner + 报异常。返回「硬拦项是否全过」。

    :param verbose: 把每个节点的**明细行**也 `echo` 出来（用户 2026-10-09 定：
                      `--verbose` 显示详细文字警告信息，banner 只提示检查结果）。
    :param strict: 硬拦项（:data:`ENV_GATE_NODES`）不过时打明细并 `sys.exit`
                     :data:`EXIT_FAILURE`。**调用方要按 `REPAIR_COMMANDS` 决定**：
                     修配置那四条命令必须传 `strict=False`。

    ⚠️ banner 由**本函数自己开、自己关**（`close()` 在返回前）—— 关掉之后光标就在
    那块区域**下方**，`require_mongodb` 与 `cmd.run` 的打印才落得干净。
    否则两个 banner 的光标记账会打架（`PITFALLS.md` P22）。
    """
    from GolemQ.core.presentation import Banner

    packages = check_packages()
    results = run_checks(packages=packages)
    notes = _notes()

    # `header=False`：身份块（产品名 + 戳 + 版权）已由 `print_identity` 打过了，
    # 这里只出行 —— 且这些行要与 `--save` 那块的阶段名**对齐成一列**。
    banner = Banner('bootstrap', SELF_CHECK_ROWS, header=False)
    banner.render()
    for node, state, lines in results:
        banner.mark(node, state)
        if verbose:
            for line in lines:
                banner.echo('{}: {}'.format(node, line))
    for label, detail, _bad in notes:
        if verbose:
            banner.echo('{}: {}'.format(label, detail))
    banner.close()          # ← 之后才允许 print（光标已在 banner 区域之下）

    failed = [node for node, state, _ in results
              if state == FAIL and node in ENV_GATE_NODES]
    if failed:
        # 非 verbose 下屏上只有一个红点，必须把「为什么」补出来（不然等于没说）
        for node, state, lines in results:
            if node in failed:
                for line in lines:
                    print('✗ {}: {}'.format(node, line), flush=True)
        if strict:
            print('✗ 环境自检未通过（{}）—— 先修环境。'
                  '配置类命令（--setup 等）不受此限。'.format('、'.join(failed)), flush=True)
            sys.exit(EXIT_FAILURE)

    if not verbose:
        for label, detail, bad in notes:
            if bad:
                print('⚠️ {}: {}'.format(label, detail), flush=True)

    return not failed


def require_mongodb(verbose=False):
    """**服务段自检**：MongoDB 连接 + 8.3 版本。**不过就退出** :data:`EXIT_FAILURE`。

    只在 `Command.needs_db` 的命令上调用 —— 见模块 docstring 的时机说明。
    """
    ok_cfg, cfg_detail = check_config()
    if not ok_cfg:
        print('✗ 无法连库: {}'.format(cfg_detail))
        sys.exit(EXIT_FAILURE)

    uri = _config_uri()
    ok, detail = check_mongodb(uri)
    if not ok:
        print('✗ MongoDB 自检未通过: {}'.format(detail))
        sys.exit(EXIT_FAILURE)
    if verbose:
        print('✓ MongoDB: {}'.format(detail), flush=True)


def _config_uri():
    """从配置文件取 `[MONGODB] uri`（:func:`check_config` 已确认它存在）。"""
    value = _ini_value('MONGODB', 'uri')
    if value is None:
        raise configparser.NoSectionError('MONGODB')
    return value
