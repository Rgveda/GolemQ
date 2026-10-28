# 多进程提交内存守卫（numpy / OpenBLAS 线程数）

> **一句话**：numpy 的 BLAS 会**按线程**预留提交量（commit）。`joblib` 每开一个
> worker 就是一个新 python 进程，各自再乘一遍线程数 —— 撞的是 Windows 的
> **commit limit（物理内存 + 页面文件）**，不是物理内存。修法是在
> `GolemQ/__init__.py` 顶部把六个线程数变量 `setdefault` 到 `1`，并行度交给
> **worker 数**。
>
> 相关记录：[`PITFALLS.md` P21](../../PITFALLS.md)、[`DECISIONS.md` D24](../decisions/D24.md)、
> 探针脚本 [`tools/check_blas_commit.py`](../../tools/check_blas_commit.py)。

---

## 一、症状：不是「内存不够」，是「提交量撞上限」

典型表现（都不一定报错）：

| 表现 | 说明 |
|:--|:--|
| 批处理**越跑越慢**（如 `17.33s/code`，正常亚秒级） | 提交量逼近上限后系统在拿页面文件硬撑 |
| 进度条长时间停在某个百分比、不结束 | 同上 |
| `OSError [WinError 1455] 页面文件太小，无法完成操作` | 真名是 `ERROR_COMMITMENT_LIMIT` |
| 任务管理器看**物理内存还有富余** | 因为撞的不是物理内存 |

Windows 的提交上限：

```
commit limit = 物理内存 + 页面文件
```

一个进程「提交」了内存，不代表它真的占了那么多物理页 —— 但**提交额度是记账的**，
超过上限就分配失败。

### 看当前提交量

PowerShell（系统级）：

```powershell
Get-Counter '\Memory\Committed Bytes','\Memory\Commit Limit' |
  ForEach-Object { $_.CounterSamples } |
  ForEach-Object { '{0,-28} {1,8:N1} GB' -f $_.Path, ($_.CookedValue / 1GB) }
```

每个 python 进程的提交量（`PrivateMemorySize64` ≈ 提交大小）：

```powershell
Get-Process python -ErrorAction SilentlyContinue |
  Select-Object Id, @{n='CommitGB'; e={[math]::Round($_.PrivateMemorySize64 / 1GB, 2)}}
```

求和：

```powershell
(Get-Process python -ErrorAction SilentlyContinue |
  Measure-Object PrivateMemorySize64 -Sum).Sum / 1GB
```

---

## 二、根因：线程数 × 进程数

numpy 自带的 OpenBLAS 在**每个线程**上预留一份工作缓冲。实测（**同一台机器**，
`import numpy` 前后对比 `PagefileUsage`）：

| 机器 / 时间 | 不设变量 | 六变量全 = 1 |
|:--|--:|--:|
| 老树，32 逻辑核（2026-09） | **11.69 GB** | **0.02 GB** |
| 新树，18 逻辑核（2026-10-09）| **6573.9 MB** | **19.8 MB** |
| 本机，18 逻辑核（2026-10-26 复测）| **2200.6 MiB** | **17.5 MiB** |

⚠️ **每线程预留量随 OpenBLAS/numpy 版本而变**（老树约 366 MB/线程，本机当前约
122 MB/线程）—— 所以**别记死一个数**，用探针现测：

```bash
python tools/check_blas_commit.py --workers 4
```

在 `joblib`/`loky` 下，每个 worker 是**独立 python 进程**：

```
总提交量 ≈ worker 数 × 每进程预留
```

老树 32 核、24 workers 时约 **280 GB**，必然撞上 ~110 GB 的 commit limit。
这就是「物理内存明明很空却崩」的来源。

⚠️ **换 Python 版本、加物理内存都不是修复**：预留量由「核数 × 线程数」决定；
加内存只是把上限抬高，浪费仍在。

---

## 三、修复：已固化在代码里

### 3.1 守卫

`GolemQ/__init__.py` 的**最顶部、numpy 首次 import 之前**：

```python
BLAS_THREAD_VARS = ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'NUMBA_NUM_THREADS',
                    'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')

for _v in BLAS_THREAD_VARS:
    _os.environ.setdefault(_v, '1')
```

* **为什么是 `1`**：一个进程只留一个 BLAS 线程；并行度交给 **worker 进程数**
  （`joblib`）。既避免内存按线程叠加，也避免 `18 核 × 5 worker = 90 个线程`
  在 18 个核上互相抢。
* **为什么用 `setdefault`**：允许**单进程**数值密集任务自己 `export` 覆盖
  （例如 `OPENBLAS_NUM_THREADS=4`）。**多进程批处理不要覆盖** —— 会按 worker 数乘上去。
* **执行时机**：任何 `import GolemQ` / `python -m GolemQ.cli` 都会先执行包
  `__init__`，所以 CLI、管线的入口天然覆盖。**必须在 numpy 首次 import 之前**；
  numpy 已经加载后再设变量无效。

### 3.2 多进程会不会继承？

会。`joblib`/`loky` 的 worker 继承父进程环境变量；而父进程在创建 `Parallel`
之前已经 `import GolemQ`（`pipeline/base.py` 的 CLI 调用链如此）。

本机实测（`python tools/check_blas_commit.py --workers 4`）：

```
[worker 0] pid=17532 commit=27.2 MiB OPENBLAS=1 OMP=1
[worker 1] pid=19640 commit=27.1 MiB OPENBLAS=1 OMP=1
[worker 2] pid=20728 commit=27.1 MiB OPENBLAS=1 OMP=1
[worker 3] pid=24240 commit=27.1 MiB OPENBLAS=1 OMP=1
```

（worker 只 import numpy；真实管线 worker 还会 import GolemQ 各模块，基线约
**90–150 MB/worker**，仍远小于未守卫的 GB 级。）

### 3.3 自检

`python -m GolemQ.cli`（不带参数即可）会打印环境自检 banner：

```
环境自检  操作系统 ●  python ●  依赖包 ●  线程环境 ●  ...
```

`线程环境` 的判据（`cli/bootstrap.py::check_blas`）：

| 情形 | 结果 |
|:--|:--|
| 六变量都是 `1` | 绿 |
| 有变量**未设** | 黄（守卫被删/没走到 import 路径）|
| 有变量**不是 1** | 黄（显式覆盖：是你的选择，但批处理会乘上去）|

⚠️ 它**只警告、不硬拦**（`DECISIONS.md` D24）—— 显式覆盖可能是合理的。

---

## 四、新电脑部署检查清单

### 步骤 0：装环境

```bash
conda create -n GolemQ python=3.12
conda activate GolemQ
pip install -e .
```

⚠️ 本机现有的 `GolemQ` env 与老树 `GolemQ_old/` 共用，见 `CLAUDE.md`；
**不要**为了"清理无关包"卸载东西。

### 步骤 1：确认没有外部覆盖

守卫是 `setdefault` —— **外部已设的值优先**。先在新机器上查：

PowerShell（用户级 / 机器级 / 当前进程）：

```powershell
'OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','NUMBA_NUM_THREADS',
'MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS' |
  ForEach-Object {
    '{0,-28} User={1,-6} Machine={2,-6} Process={3}' -f $_,
      [Environment]::GetEnvironmentVariable($_, 'User'),
      [Environment]::GetEnvironmentVariable($_, 'Machine'),
      [Environment]::GetEnvironmentVariable($_, 'Process')
  }
```

Git Bash / Linux：

```bash
env | grep -E 'OPENBLAS_NUM_THREADS|OMP_NUM_THREADS|NUMBA_NUM_THREADS|MKL_NUM_THREADS|NUMEXPR_NUM_THREADS|VECLIB_MAXIMUM_THREADS'
```

期望：**都没有值**（或都是 `1`）。若有 `4`、`8` 之类：

* 来源可能是：系统/用户环境变量、conda env 的 `config vars`、IDE 的 Run
  Configuration、`.bashrc`/PowerShell profile、调度任务；
* 找到来源并删除，或至少确认**它只作用于单进程任务**。

⚠️ **不要给 GolemAI env 设这些变量**（它跑 torch，CPU 推理会被压成单线程）。
详见第六节。

### 步骤 2：验证守卫

**干净解释器**（把六个变量从环境里剥掉）跑：

```bash
python -c "import GolemQ, os, json; print(json.dumps({k: os.environ.get(k) for k in GolemQ.BLAS_THREAD_VARS}))"
```

期望：

```json
{"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "NUMBA_NUM_THREADS": "1",
 "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1"}
```

再确认 numba 也降下来了：

```bash
python -c "import GolemQ, numba; print(numba.get_num_threads())"
# → 1
```

### 步骤 3：验证提交量

```bash
python tools/check_blas_commit.py --workers 4
```

期望输出（本机实测，仅数值供参考）：

```
CPU 逻辑核数: 18
[当前进程] import GolemQ 前: {..., None, ...}
[当前进程] import GolemQ 后: {..., '1', ...}
[子进程] 剥掉变量 + import numpy : 2200.6 MiB（import 前 9.2 MiB）
[子进程] 变量=1   + import numpy :   17.5 MiB
[子进程] import GolemQ + numpy   :   91.3 MiB  (numba threads=1)
[worker 0] pid=... commit=27.2 MiB OPENBLAS=1 OMP=1
...
```

判据：

| 项 | 期望 |
|:--|:--|
| 「剥掉变量」那一行 | 明显大于「变量=1」—— 说明**本机确实有这份预留**（大小随 numpy 版本变）|
| 「变量=1」/「import GolemQ」| MiB 级，不是 GB 级 |
| worker 行 | `OPENBLAS=1`、`OMP=1`，commit 为几十 MB |

⚠️ 若「剥掉变量」那一行也只有几十 MB，说明本机的 OpenBLAS 构建不按线程预留 ——
守卫仍建议保留（零成本，且换 numpy 版本后可能变），但不必为此紧张。

### 步骤 4：验证 CLI 自检

```bash
python -m GolemQ.cli
```

看 `线程环境` 是否为绿。若为黄，按它给的原因处理：

* `未设 ...` → `GolemQ/__init__.py` 的守卫被删了，或入口没走 `import GolemQ`；
* `... 不是 1` → 步骤 1 找到的外部覆盖还在。

### 步骤 5：设定 worker 数

守卫之后，内存占用主要回到**数据本身**。仍要按提交上限留余量：

```
总提交 ≈ 主进程 + worker 数 × (包基线 + 单码数据峰值)
```

本机参考：包基线约 90–150 MB/worker，单码数据峰值视窗口而定。

CLI 的 `--pct-of-cores`（`review` / `compact`）：

| 值 | 含义 |
|:--|:--|
| `0` | 串行 |
| `0 < x <= 1` | 比例（`0.32` = 32% 核）|
| `> 1` | 百分数（老树 `32` = 32%）|

`--review` 默认 `0.32`（18 核 → 5 workers）。机器内存小 / 单码窗口大时，
显式降到 `0.25` 或 `0.125`；**不要**因为"物理内存还很空"就开满。

### 步骤 6：GolemAI / ChartScan（特例）

ChartScan 走 **GolemAI env**（torch/CUDA 只装在那里），且是**单进程**服务。
`cli/commands/chartscan.py::_re_exec` 会：

* `os.environ.copy()` 继承父进程；
* 再把 `BLAS_THREAD_VARS` 六个变量**统一改成 `GOLEMQ_CV_THREADS`**（默认 `4`）——
  因为 torch CPU 推理需要多线程，而父进程的守卫是 `1`；
* `PYTHONUNBUFFERED=1`（长驻服务的日志不被缓冲吞掉）。

部署新机器时：

* **不要**把六个变量设成 `1` 之后还期望 torch 快 —— ChartScan 会自己覆盖回 4；
* 若想改线程数：`export GOLEMQ_CV_THREADS=8`（或 Windows 用户环境变量）；
* 若 GolemAI env 不叫 `GolemAI` / 不在 `<conda root>/envs/` 下：
  `export GOLEMQ_CV_PYTHON=<GolemAI 的 python 绝对路径>`；
* **启动脚本必须在 GolemQ env 里跑**（`chartscan_svr.ps1` / `chartscan_cli.ps1`
  现在都是）。若在 GolemAI env 里直接敲 `GolemQ.cli`，`in_cv_env()` 为真 ⇒
  **跳过 `_re_exec`**，线程覆盖不会执行，守卫把 torch 压成 1 线程。
  2026-10-18 的一次启动脚本改写曾踩过这个坑（服务端在 GolemAI 里起），
  2026-10-26 已改回 GolemQ 启动。

### 步骤 7：不走 GolemQ 的入口

守卫只在 `import GolemQ` 时执行。以下情况**不受保护**：

* 自己写的脚本只 `import numpy` / `import joblib`，从不 import GolemQ；
* Jupyter notebook 先 import numpy，再 import GolemQ（**太晚**，numpy 已加载）；
* 第三方库内部起进程、且清空了环境变量。

处理办法（二选一）：

1. **入口第一行 `import GolemQ`**（推荐，零额外配置）；
2. 在启动脚本里显式设置（只在**这个进程/脚本**里生效）：

   ```bash
   # Git Bash / Linux
   export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
          MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
   ```

   ```powershell
   # PowerShell
   $env:OPENBLAS_NUM_THREADS='1'; $env:OMP_NUM_THREADS='1'; $env:NUMBA_NUM_THREADS='1'
   $env:MKL_NUM_THREADS='1'; $env:NUMEXPR_NUM_THREADS='1'; $env:VECLIB_MAXIMUM_THREADS='1'
   ```

### 步骤 8：定时任务 / 服务

Windows 计划任务、NSSM 服务等会继承**机器/用户环境**：

* 用步骤 1 的命令确认没有残留的非 `1` 值；
* 任务的「起始于」目录与解释器路径要对（`pip install -e .` 的 env）；
* 守卫仍会在 `import GolemQ` 时执行，但**外部显式值优先**。

### 步骤 9：ChartScan 启动脚本（`chartscan*.ps1`）

| 脚本 | 作用 | 换机器要点 |
|:--|:--|:--|
| `chartscan.ps1` | 用 Windows Terminal 开 1 个服务端 + 4 个客户端标签页 | 需要 `wt.exe`；没有就分别跑下面两个脚本 |
| `chartscan_svr.ps1` | 服务端：GolemQ env → CLI → 自动切 GolemAI | 仓库根 = `$PSScriptRoot`；conda = `$env:CONDA_EXE` → PATH |
| `chartscan_cli.ps1` | 客户端：GolemQ env → CLI → 自动切 GolemAI | 同上 |

2026-10-26 起**不再硬编码** `Y:\Projects\GolemQ` 与 `C:\ProgramData\miniconda3`：

* 仓库可以放任意路径（脚本用 `$PSScriptRoot` 定位）；
* conda 可以装在任意位置（脚本用 `$env:CONDA_EXE` / PATH 找，再 `conda info --base` 取 hook）；
* **env 名**仍是硬约束：必须有 `GolemQ` 与 `GolemAI` 两个 conda env；
* GolemAI 里必须有 `torch` / `ultralytics` / `zmq`（pyzmq）/ `talib`；
* GolemAI 不在标准 `<conda root>/envs/GolemAI/` 时，设 `GOLEMQ_CV_PYTHON` 指向它的 python；
* 三个 `.ps1` 现在是 **UTF-8 with BOM**（跨语言 Windows 不会把中文注释解成语法错）。

⚠️ `while ($n -lt 6)` 里 `$n` **从不自增** —— 这是**有意的**无限 supervisor 重启
（服务端崩了 1 秒后自动拉起），不是 bug。

CLI 侧本来就是可移植的：`cv_python()` 从 `sys.executable` 推 `<conda root>/envs/GolemAI/`，
`_REPO_ROOT` 从 `__file__` 推，`_re_exec` 用仓库根当 `cwd`（所以 GolemAI env
**不需要** `pip install` 本包）。

### Linux 补充

守卫同样有效（变量名是跨平台的）。Linux 的 overcommit 通常不会像 Windows 那样
立刻报 `1455`，但线程池预留**仍然是真实的内存压力**。看系统级提交：

```bash
grep -E 'CommitLimit|Committed_AS' /proc/meminfo
ps -eo pid,rss,vsz,comm --sort=-rss | grep python | head
```

---

## 五、排障表

| 现象 | 先查 | 常见原因 | 处理 |
|:--|:--|:--|:--|
| `线程环境` 黄：`未设 ...` | `python -c "import GolemQ, os; ..."` | 守卫被删/入口绕过 GolemQ | 恢复 `GolemQ/__init__.py` 顶部守卫；入口先 import GolemQ |
| `线程环境` 黄：`... 不是 1` | 步骤 1 的三级环境变量 | 系统/用户/IDE/conda env 覆盖 | 删除覆盖；或确认只用于单进程 |
| 仍然 `WinError 1455` | `Get-Process python` 的 CommitGB | ① 有未守卫的 python 进程；② worker 开太多；③ 页面文件太小 | 用探针确认每个进程；降 `--pct-of-cores`；扩大页面文件/换机器 |
| 越跑越慢、物理内存却空 | `Committed Bytes / Commit Limit` | 提交量逼近上限 | 同上 |
| ChartScan / torch 很慢 | `GOLEMQ_CV_THREADS` | 被压成 1 线程 | 设 `GOLEMQ_CV_THREADS=4`（或更高）；别在 GolemAI 全局设 1 |
| 单进程数值任务变慢 | `OPENBLAS_NUM_THREADS` | 守卫把它压到 1 | **只在该进程**临时 `export OPENBLAS_NUM_THREADS=4`；别让批处理继承 |
| `numba.get_num_threads() == 1` | — | 正常（守卫设了 `NUMBA_NUM_THREADS=1`） | 除非该任务用 `parallel=True` 且需要多线程，否则不用改 |

---

## 六、FAQ

**Q：为什么不直接把变量设成系统级、一劳永逸？**
A：会连带影响 **GolemAI env**（torch CPU 推理需要多线程）。老树曾评估过
「user site-packages 级注入」，正因此否决。包级守卫只影响 import GolemQ 的进程，
边界清楚。

**Q：为什么不设成 2 / 4，单进程更快？**
A：单进程可以（用 `setdefault` 留了口子）；**多进程不行** —— worker 数会再乘一遍。
本项目批处理的并行度来自 `joblib` 的 worker 数，不是 BLAS 线程。

**Q：加内存 / 换 Python 3.13 能解决吗？**
A：不能。这是 numpy/OpenBLAS 的预留行为，由「核数 × 线程数」决定。加内存只是
抬高 commit limit，浪费仍在；换 CPython 版本不改变 BLAS 行为。

**Q：守卫会影响数值结果吗？**
A：BLAS 多线程会改变浮点归约顺序，最后几位可能不同。压到 1 线程反而让
**跨机器/跨次运行更稳定**。但**不要**把「1 线程算出来的结果」与「多线程算出来的
结果」做逐位比较。

**Q：为什么 `check_blas` 不硬拦？**
A：`setdefault` 的语义就是允许显式覆盖；判红会误伤合理的单进程调优（D24）。
它只把事实摆出来。

---

## 七、改动位置与测试

| 文件 | 作用 |
|:--|:--|
| `GolemQ/__init__.py:25-51` | 守卫本体（必须在 numpy 之前）|
| `GolemQ/cli/bootstrap.py:281-305` | `check_blas`：查「设没设」+「值是不是 1」|
| `GolemQ/pipeline/base.py:226-254` | `joblib.Parallel(n_jobs=...)` 的 worker 数 |
| `GolemQ/cli/commands/chartscan.py:103-111` | GolemAI 子进程的线程数覆盖 |
| `GolemQ/test_cases/core/test_blas_guard.py` | 用**干净解释器**钉住守卫（本进程测不出）|
| `tools/check_blas_commit.py` | 部署/排障探针（只读）|

回归测试：

```bash
python -m unittest GolemQ.test_cases.core.test_blas_guard -v
```
