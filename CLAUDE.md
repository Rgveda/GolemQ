# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 接手本项目，按此顺序读

**Claude Code 每次会话自动加载本文件**，所以阅读入口放在这里 —— 其余文档挂在下面。

| 顺序 | 文档 | 读它干什么 |
|:--|:--|:--|
| 0 | [`HANDOFF.md`](HANDOFF.md) | **当前进度与下一步**（含待用户决定的事项）。跨会话的进度载体，**接手先读它** |
| 1 | [`PITFALLS.md`](PITFALLS.md) | **已知陷阱。看起来像 bug 的刻意设计，勿"修正"** |
| 2 | [`DECISIONS.md`](DECISIONS.md) | 已定的架构决定与**弃案理由**，别重新论证 |
| 3 | [`GLOSSARY.md`](GLOSSARY.md) | 行话。第四节的词**几乎全是老代码继承**，别当拼写错误改 |
| 4 | [`RESTRUCTURE_PLAN.md`](RESTRUCTURE_PLAN.md) | 两层模型、Active Market、portfolio 方案 |
| 5 | [`MIGRATION_STATUS.md`](MIGRATION_STATUS.md) | 重构遗留缺陷清单（含修复顺序建议）|
| 6 | [`API_INDEX.md`](API_INDEX.md) | **导航索引**（110 模块 / 名称 + 一行摘要）。找东西先查它，比逐个 `Read` 源文件省一个数量级 |

**改代码前先查 `PITFALLS.md` 与 `GLOSSARY.md`。**
**加函数前先走「[新增函数前的思维链](#新增函数前的思维链任何情况下都要走一遍)」。**
**每告一段落就更新 `HANDOFF.md`** —— 进度写进文件才算存下来，留在对话里会随上下文一起丢。

另：`git log --oneline -30` 的 commit message 记录了当时的决定、弃案与验证方式 ——
它们是事实上的 ADR，成本为零且已有内容。

### 为什么有这几份文档

上下文容量有限（本项目实测约两天就需要重来一轮），而**代码与 docstring 装不下**
四类东西：**决策与弃案、当前进度、行话、陷阱**。这四份文档就是补这个缺口。

写这些文档时的一条纪律：**只写已验证的事实**。没验证过的宁可留空 ——
一份「猜的」文档比没有文档更坏，因为它会被当成依据。

### doctest

纯函数（`portfolio/`、`core/market_registry`、`datasource/` 的机制部分）的 docstring
里带 doctest，由 `GolemQ/test_cases/test_doctests.py` 收集进 `unittest` 发现。

```bash
python -m unittest GolemQ.test_cases.test_doctests -v
```

**需要 DB / 网络 / QMT 客户端的函数不要加 doctest** —— 加了也跑不了，
只会让测试套件变慢变红。它们的正确性验证在 `PITFALLS.md` 的边界条件里。

---

## 新增函数前的思维链（**任何情况下**都要走一遍）

> **起因**：复权曾经是**两套平行实现** —— 股票一套、ETF 一套，机制相同、只差策略，
> 各自维护一份 `_row_dates` / `_row_codes` / 对齐逻辑，直到 `85bba0f` 才收敛进
> `fq.py`（现由 `datastruct.py` 与 `etf_fq.py` 各自 import）。
> **平行实现不会报错，只会分叉** —— 分叉之后同一个 Bug 修一次只修一半。

### 四问，按顺序答，**不允许跳步**

| # | 问 | 「是」→ 怎么办 |
|:--|:--|:--|
| 1 | **已有函数不够用吗？** | **先搜，别先写**：查 `API_INDEX.md`（110 模块 / 名称 + 一行摘要），再 grep 动词与名词，再看**同层相邻模块**。够用就调，**不新增**，也不"顺手包一层" |
| 2 | **是旧函数有 Bug 吗？** | 「差一点」先判**是 Bug 还是设计**（查 `PITFALLS.md`：看起来像 bug 的刻意设计，勿"修正"）。是 Bug 就**就地修**，不要复制一份改好的 |
| 3 | **修好 Bug 后能满足吗？** | 能 → **就地修**，并在 commit message 写明修了什么、怎么验的。**到此结束，没有新函数** |
| 4 | **旧函数真的实现不了吗？** | 才允许新增 —— 但**必须先答完下面三个定位问题**再动手 |

### 第 4 步放行后的三个定位问题

**① 放哪层？** 根层（`pipeline/` `portfolio/` `analysis/` `services/` `datasource/`
`core/`）= **各交易系统共用**；`markets/<Market>/` = **该市场专有**。
**数据库操作只能进 `services/`**（本文件「Coding Conventions」的硬规定，无例外）。
将超 `services/` 的 300 行上限时，按 `services/features/` 的先例**拆子模块**，
不是放宽行数。

**② 命名怎么跟同级对齐？** 决定放进哪个模块后，**先看那个模块里已有的函数名** ——
前缀、动词习惯、返回约定，照着写。全树实测现状（2026-09-25）：

| 形态 | 含义 | 出现在 |
|:--|:--|:--|
| `GQ_xxx_yyy` | **老树继承**的公开 API 风格，**无机制含义** —— 不是分发前缀，CLI 是按**类继承**发现市场的（`cli/tools.py` 找 `purge_historical_collections`），不按名字前缀 | `etf_fq` / `refdata` / `scribe` / `symbol` / `maintenance` / `kline83` |
| `xxx_yyy` | 新写的**纯内部**模块 | `fq.py` / `datastruct.py` |
| `_xxx` | 模块私有 | 各处 |

⚠️ **这条我自己就没对齐**：同为新建，`maintenance.py` 用了 `GQ_`、`fq.py` 没用。
所以第 ② 问**必须现场看同级**，**别照抄另一个模块的结论**。
（`GLOSSARY.md` 第四节的词几乎全是老代码继承，别当拼写错误改。）

**③ 接口能被分发吗？** 若该函数将成为 CLI / 订阅器入口，**必须能无参调用** ——
调用方不带参数（见 `PITFALLS.md` P10）。

### 判成「新增」时要留痕

commit message 或 `HANDOFF.md` 里写明**为什么不复用**。写了才算做过这个判断 ——
否则下一个接手的人只看到两个近似函数，**无从分辨是取舍还是疏忽**。

## Project Overview

GolemQ is a quantitative trading framework for the Chinese A-share stock market. It provides market data ingestion, feature engineering, backtesting, live trading via XTQuant, and monitoring/scheduling — backed by MongoDB and RabbitMQ.

Cross-platform: supports Windows x64 and major Linux distributions.

Requires Python >= 3.12. Install in editable mode:

```bash
conda create -n GolemQ python=3.12
conda activate GolemQ
pip install -e .
```

### Windows Conda Preamble

On Windows, activate the conda environment before running commands:

```powershell
C:\ProgramData\miniconda3\shell\condabin\conda-hook.ps1 ; conda activate C:\ProgramData\miniconda3 ; cd "y:/projects/GolemQ" ; conda activate GolemQ
```

### Windows Encoding (UTF-8)

This project contains Chinese identifiers, comments, and data. Windows consoles
default to the legacy ANSI code page (cp936/GBK on a zh-CN system), which
mangles Chinese output and breaks comparisons against UTF-8 source. Always
work in UTF-8.

**Already configured** — `.claude/settings.json` pins these for every session:

| Variable | Value | Purpose |
|:---|:---|:---|
| `PYTHONUTF8` | `1` | Python 3.7+ UTF-8 mode: stdin/stdout/files default to UTF-8 |
| `PYTHONIOENCODING` | `utf-8` | Belt-and-braces for stdout/stderr |
| `LANG` / `LC_ALL` | `en_US.UTF-8` | Git Bash locale |

**Bash (Git Bash)** — if you need it manually, or the vars above are absent:

```bash
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8 LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8
```

**PowerShell** — env vars cannot fix the *console* encoding; set it explicitly:

```powershell
chcp 65001 > $null
[Console]::OutputEncoding = [Text.Encoding]::UTF8
$OutputEncoding          = [Text.Encoding]::UTF8
$PSDefaultParameterValues['Out-File:Encoding'] = 'utf8'
```

PowerShell 7+ already defaults to UTF-8; this matters mainly for Windows
PowerShell 5.1.

**Python source** — always pass `encoding` explicitly when touching files:

```python
open(path, encoding="utf-8")                      # read
json.dump(data, fh, ensure_ascii=False)           # write, keep CJK readable
```

Do not rely on `open()`'s default encoding even with `PYTHONUTF8=1`; being
explicit keeps the code correct on Linux too.

**Symptoms of getting this wrong**: `事件数` renders as `��¼��`; a Chinese
path or symbol compares unequal to its own literal; `UnicodeDecodeError` on a
file that looks fine in an editor.

### Hook Maintenance (after any `ruflo init`)

`ruflo init` regenerates `.claude/settings.json` and will undo the hook fix in
commit `7e542fe`. Re-apply it with:

```bash
python tools/fix_ruflo_hooks.py            # this project + ~/.claude
python tools/fix_ruflo_hooks.py --dry-run  # report only
python tools/fix_ruflo_hooks.py --scan     # every project under Y:/Projects, Y:/代码
```

The generated hooks wrap the node call as
`cmd /c "IF EXIST "..." (...) ELSE (...)"`. The `\"` in the JS template looks
like escaping but collapses to a bare quote, so the command fragments and
pieces of it get created as 0-byte files in the working directory. Upgrading
ruflo does not help — the template is byte-identical in `@claude-flow/cli`
3.10.2, 3.41.2 and 3.42.4.

The replacement must use **bash** syntax (`${CLAUDE_PROJECT_DIR:-.}`,
`$USERPROFILE`), never `%VAR%` — the hook executor is bash, which does not
expand `%VAR%`, and node then reports
`Cannot find module '...\%USERPROFILE%\...'`. That mistake disables every
hook while appearing to cure the stray files.

**Verify with metacharacters.** `c=$(echo x); echo "brace {a,b}" 'quote'
"1e-12)" && ls -a | wc -l` — a plain `echo` reproduces nothing either way and
gives a false pass. The `Stop` hook also surfaces path errors on session exit.

## Commands

```bash
# Run all tests
python GolemQ/test_cases/run_tests.py

# Run a specific test module
python -m unittest GolemQ.test_cases.test_messenger -v

# Run a single test method
python -m unittest GolemQ.test_cases.test_messenger.TestDingtalkConfig.test_check_config_success -v

# CLI - configuration
python -m GolemQ.cli --setup              # Initialize MongoDB, DingTalk, Server酱
python -m GolemQ.cli --mongodb-init       # MongoDB config only
python -m GolemQ.cli --dingtalk-init      # DingTalk config only
python -m GolemQ.cli --xtquant-init       # XTQuant config only

# CLI - data operations
python -m GolemQ.cli --purge-l1           # Purge MongoDB historical collections
python -m GolemQ.cli --xtquant-sync       # One-shot XTQuant positions sync to MongoDB
python -m GolemQ.cli --xtquant-sync-daemon # Daemon mode: sync during trading hours

# CLI - monitoring
python -m GolemQ.cli --heartbeat-watchdog      # View heartbeat status
python -m GolemQ.cli --stop-heartbeat-monitor  # Stop all monitoring

# CLI - subscriptions & watchlists
python -m GolemQ.cli --sub l1_tencent           # Run L1 Tencent data subscription
python -m GolemQ.cli --eneloop-add --symbols "000001,000002"
python -m GolemQ.cli --eneloop-remove --symbols "000001,000002"
python -m GolemQ.cli --eneloop-list
```

## Architecture

### Infrastructure

- **MongoDB** — primary database for all persistence (market data, operational state, configuration)
- **RabbitMQ** — cache and message queue layer

### Package Structure (top-level `GolemQ/`)

| Module | Purpose |
|--------|---------|
| `markets/` | Market abstraction layer. `base_market.py` defines the `BaseMarket` ABC. `StockCN/` is the A-share implementation (singleton via `__new__`). `StockHK/` is a stub. Markets self-register into the global `GQMARKETS` dict. |
| `gateway/xtquant/` | XTQuant trading gateway. `trader.py` wraps `XtQuantTrader`; `xtquant_tools.py` provides position export, available volume queries; `config.py` manages XTQuant credentials in `~/.GolemQ/config.ini`. |
| `services/` | Data operations layer. `features.py` saves/fetches feature metadata to/from MongoDB. `align.py` provides checkpoint logging, kline alignment, and missing-data detection. `iwencai.py` interfaces with iWenCai queries. `persistence/` (package) provides general-purpose MongoDB CRUD helpers. |
| `analysis/` | Analysis modules. `timeseries.py` provides multi-frequency resampling and timeline utilities. |
| `pipeline/` | Stock processing pipeline. `base.py` has abstract pipeline classes with joblib parallelization. Sub-modules: `mainstream_benchmark.py`, `poolcoef_benchmark.py`, `compact_benchmark.py`. |
| `agents/` | External communication. `messenger.py` handles DingTalk bot (alibabacloud_dingtalk SDK) and Server酱 push notifications. |
| `supervisor/` | Operational monitoring. `heartbeat.py` provides `HeartbeatMonitor` (MongoDB-backed, per-module heartbeat with timeout detection) and `HeartbeatModule` (per-instance mutex/checkin). `scheduler.py` runs `schedule`-based `XtquantSyncScheduler` during A-share trading hours (9:30-11:30, 13:00-15:00 Beijing time). `function_checkin.py` provides rate-limiting for alert functions. `messenger.py` wraps alert dispatch. |
| `cli/` | CLI entry point (`__main__.py` → `main()`). `tools.py` auto-discovers and registers market modules. `watchdog_manager.py` manages symbol watchlists. |
| `core/` | `settings.py` — `GQ_Setting` wraps `~/.GolemQ/config.ini` with MongoDB fallback; also ties into QUANTAXIS settings. `constants.py` — `AKA` (field aliases), `FIELD`, `MARKET_TYPE`, `STATE` constants. `mongo.py` — MongoDB client helpers. `preprocessing.py` — pandas-to-JSON converters and data masking. |

### Key Design Patterns

- **Market registry**: `GQMARKETS` (dict in `GolemQ.__init__`) holds market instances. `cli/tools.py` auto-discovers and registers them. `GQSUBSCRIBER` maps subscription keys (e.g., `l1_tencent`) to subscriber functions.
- **StockCN singleton**: `StockCN.__new__` enforces a single instance. Auto-instantiated on module import and registered into `GQMARKETS`.
- **Configuration**: `GQ_Setting` reads/writes `~/.GolemQ/config.ini`. Sections `DINGTALK`, `SERVERCHAN`, `XTQUANT` are stored in the INI file; other sections fall back to MongoDB. QUANTAXIS settings are used in parallel (`QASETTING`, `DATABASE_QA`).
- **Dual database**: GolemQ stores operational data under `DATABASE.golemq.*` collections; QUANTAXIS stores market data under `DATABASE.quantaxis.*`.
- **Heartbeat/mutex**: `HeartbeatModule.mutex()` checks for conflicting running instances in MongoDB before starting, using a timeout-based lock. `HeartbeatMonitor` runs a daemon thread that detects stale modules and fires alerts.
- **Data flow**: Free data sources (Tencent, Sina, EastMoney via `easyquotation/`) → `services/` feature extraction → MongoDB → `analysis/` / `pipeline/` consumption. XTQuant gateway provides live position/order data.

## Coding Conventions

- **Database operations**: ALL database operations (MongoDB CRUD, queries, aggregations) MUST live in the `services/` layer. No other module may access the database directly — they call `services/` functions instead.
- **Naming**: Use `snake_case` for all function, variable, and file names.
- **CLI functions**: CLI command handlers in `cli/` do ONLY parameter validation and then delegate to `pipeline/` functions. No business logic in the CLI layer.
- **File size**: Each `services/` file must not exceed 300 lines. Split larger files into focused sub-modules.

### External Dependencies

- **QUANTAXIS**: Core market data framework. Provides `QA_util_*` date utilities, market type constants, database client.
- **xtquant**: Commercial trading SDK (QMT/miniQMT). Only available on Windows with the QMT client installed.
- **MongoDB**: All data persistence. Connection string defaults to `mongodb://localhost:27017`.
- **RabbitMQ**: Cache and message queue. Used for inter-service communication and data buffering.
- **alibabacloud_dingtalk**: DingTalk robot SDK for push notifications.

### Configuration File

`~/.GolemQ/config.ini` — sections: `MONGODB` (uri), `DINGTALK` (appkey, appsecret, robot_code, user_id_list), `SERVERCHAN` (sendkey), `XTQUANT` (account, min_path).
