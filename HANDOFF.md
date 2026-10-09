# 交接：当前进度与下一步

> 写于 2026-09-20，上一轮会话因上下文超限（请求 1,080,619 tokens > 上限 1,048,576）
> 在 `to_qfq()` 验证中途被 API 硬中断，无法 resume。
>
> **本文件是跨会话的进度载体，只写已验证的事实。** 每告一段落就更新它；
> 接手时先读它，再读 `PITFALLS.md`。诊断与弃案仍在 `DECISIONS.md`，
> 缺陷清单在 `MIGRATION_STATUS.md`，本文件只管「现在在哪、下一步是什么」。

---

## 一、上一轮死在哪

死点很具体：正在为 **`QA_DataStruct_*` 替身（shim）** 做前置验证，卡在一个查询上：

```
stock_day 里 code=600519 有 6006 行，但日期范围查询命中 0
```

**已闭合（本轮实测）**：不是数据问题，是**类型不匹配**。

| 字段 | 实际类型 | 说明 |
|:--|:--|:--|
| `date` | **`str`** `'2026-09-18'` | **用 datetime 查它永远命中 0，且不报错** |
| `ts` | `datetime` | 收盘 16:00（`2001-08-26 16:00` 对应 `date='2001-08-27'`）|
| `date_stamp` / `time_stamp` | `int` | 两者同值 |

实测三种查法（`code=600519`, 2024-01-01 ~ 2024-01-10）：

```
date 用 str      → 7 行 ✓
date 用 datetime → 0 行   ← 上一轮卡在这
ts   用 datetime → 7 行 ✓
```

**含义**：对 8.3 时序库做范围查询，要么传 `str` 给 `date`，要么传 `datetime` 给 `ts`。
传错类型是**静默返回空集**，不是抛异常。

---

## 二、任务看板

| 任务 | 内容 | 状态 |
|:--|:--|:--|
| **A** | 5 个参考集合灌进 MongoDB 8.3 | **4/5** —— `financial` 待你决定 |
| **D-ETF** | ETF 前复权（`etf_fq.py` 回迁） | ✅ 提交 `c83b593` |
| **D-min** | 分钟线读取器重写 | ✅ 提交 `fe2fd95` |
| **C1** | `portfolio/` 骨架（strategy/sizing/costs/rules）| ✅ 提交 `f993d31` |
| **C2** | `zen_bt.py` 撮合按契约移植进 `engine.py` | ⛔ **阻塞在你**：需先写策略实现 |
| **D** | QUANTAXIS 完全解耦 | ✅ **完成（2026-10-08）** —— **全树 import 归零**，5 个句柄或删或改绑（`DECISIONS.md` D12/D13）|
| **E** | ETF 独立成 `ETF_CN` + `etf_*` | 🔶 **代码完成**（提交 `1cc47e2`）／⛔ **数据拆分待 MongoDB** |
| **RT** | 实时落库改按日时间序列集合 + 读取器切 8.3 | ✅ **代码完成**（`DECISIONS.md` D10）／⏳ **开盘时段端到端待跑**（见「实时行情 L1/L2」一节）|
| **SAVE-TDX** | `--save tdx`：K 线直写 8.3 + 增量 + 覆盖核对 | ✅ **代码完成并端到端验过**（`DECISIONS.md` D11）／⏳ **全市场首跑待做**（见下一节）|
| **TDX-DIST** | 服务器选择改成**每 worker 线程各粘一台 + 随机起点**（分布式） | ✅ **完成、实测量过**（`DECISIONS.md` D23）—— 4 worker → 4 台不同服务器；TCP 账：连接及时关（TIME_WAIT 1:1），动态端口 16,384，全量 33,474 次连接，**跑得越快 TIME_WAIT 稳态越高**（20min→21%，5min→84%）|
| **TDX-HOSTS** | 通达信服务器池：每周探活 + 活动列表落 `~/.GolemQ`（不硬编码） | ✅ **完成、实跑过**（`DECISIONS.md` D22）—— 从 pytdx 包内池探到 **66 台可用**，最快 75.8ms，写进 `~/.GolemQ/settings/tdx_hosts.json`；`TdxSource` 已默认用它 |
| **TQDM×BANNER** | 保证 4 worker 下 tqdm 进度条正确、banner 不被破坏 | ✅ **完成**（**`PITFALLS.md` P22**）—— 查出 `pytdx_kline._retry_page` 的重连诊断是**裸 print**、且跑在 tqdm 进度条活着期间（带 `-v` 会打花进度条与表头）；已改成 `note` 回调缓冲、`bar.close()` 之后再吐。`pytdx_kline.py` 现零裸 print，4 个不变量上了测试。⚠️ 末段记了 **`rich` 是另一套显示模型**（实测无光标上移 + 自带 stdout 重定向），**不许照搬 P22 的结论** |
| **ENV-SHARED** | conda env `GolemQ` 与老树 `GolemQ_old` **共用** | ✅ 已记 （`CLAUDE.md` 安装段 + 长期记忆）—— pip 报的依赖冲突（`gm`/`protobuf`）来自老树侧，**不是本项目的问题，别清理** |
| **SAVE-HANG** | `--save tdx` 跑到 76% 卡死（17.33s/code）—— **两个真因** | ✅ **已修、实测量过**（`DECISIONS.md` D21 / `PITFALLS.md` P21）：① BLAS 守卫没从老树搬来（提交量 32.1 GB → 0.43 GB）；② 服务器表首台是死服（`new_api()` 20.040s → 0.150s，折合 2795 分钟 → 20.9 分钟）|
| **CLI-BOOT** | CLI 环境自检：版权页 + TTY / python / 包版本 / 配置文件 / MongoDB 连接与 8.3 版本 | ✅ **完成、实跑验过**（`DECISIONS.md` D20）—— 实测本机 MongoDB **8.3.11** 达标；⚠️ `pandas 2.2.3 < 2.3` 会每次打一行提醒（2.3 是 3.0 前最后一个 2.x，即 2.3.3；门槛同时接纳 3.x）|
| **CLI-EXIT** | CLI 退出码口径统一（用法错=2 / 运行期失败=1）+ 全局版权行 | ✅ **完成、矩阵复核过**（`DECISIONS.md` D18/D19）|
| **CLI-SPLIT** | `cli/__main__.py` 742 行 → 99 行 + `cli/commands/` 10 个模块 | ✅ **完成**（`DECISIONS.md` D17）—— 对拍：**拆分当时六条逐字一致**；此后两处**刻意**改动（`--help` 选项按命令分组、`--save` 校验改用 argparse `choices=` 见 D18），故现在只剩四条逐字一致 |
| **BOOT-BANNER** | 环境自检也挂 banner：**四态**（灰·未检查 / 绿●通过 / 黄●警告 / 红●失败）+ 头行**时刻戳** | ✅ **完成、实跑验过**（`DECISIONS.md` D24 / `PITFALLS.md` P23+P24）—— 七个节点：`操作系统 python 依赖包 线程环境 CPU 架构 CUDA 时区`；本机跑出来是 `Intel · 18 物理核 · 18 逻辑线程 · 无超线程 · 混合大小核 6+12` 与 `RTX 4070 Ti SUPER, 591.74 · CUDA 13.1`。⚠️ 硬拦只有 `python`/`依赖包`，且 **`--setup` 等四条修配置命令放行**（偏离「只说硬拦」的唯一一处）。⚠️ MongoDB **不进** banner |
| **TTL** | 参考数据刷新闸（**按集合**分阈值：名单类盘中 5h/盘后 24h，`stock_info` 恒定 24h） | ✅ **代码完成、测试跑过**（`DECISIONS.md` D16）—— 复用 `supervisor` 的签到表 |
| **SAVE-CLI** | `--save` 收口成 `<SOURCE>` + 参考数据 banner | ✅ **代码完成、测试跑过**（`DECISIONS.md` D14）／⏳ **真机看 banner 待做**（见下下节）|
| **KLINE-SHORT** | K 线**短路**：集合自证探针 + 逐只水位 + TTL 兜底（三条缺一不可）| ✅ **完成、端到端验过**（`DECISIONS.md` **D25** / `PITFALLS.md` **P25**）—— 实跑 `index_day` 两遍：第①遍建连 3 次/写 2 行，**第②遍建连 1 次/写 0 行/skipped_fresh=2**。每集合探针 **0.02–0.15 s**（9 次），前沿与该 code 最新一根**逐秒相同**。⚠️ 复权四节点**只记账不开闸**；`--save-refresh` 一旗两用（参考数据闸 + K 线短路一起关）|

### ✅ `--save` 收口 + 参考数据 banner（2026-10-09）

| 项 | 内容 |
|:--|:--|
| 命令面 | `--save` 只收 `tdx` / `pytdx` / `qmt`；**`x`、`all`、`--save-x`、`--save-qmt` 全部删除**（`DECISIONS.md` **D14**）|
| 参考集合 | `REFDATA_BY_SOURCE`（在 `refdata_save.py`）：pytdx = `stock_list`/`stock_info`/`stock_block`/**`etf_list`**；qmt = 前三者 |
| 等价 | `--save tdx` ≡ `--save pytdx`（适配器的真名就叫 pytdx，`tdx` 只是命令面的历史名字）|
| `--save qmt` | 只做参考数据，K线/xdxr/adj **跳过**。⚠️ `QmtSource.available()` 恒 False → 三个集合**必然**全失败并以非零码退出 —— 那是 D13「只留结构」的刻意结果，别当 Bug |
| Banner | `core/presentation.Banner` / `render_pipeline_banner` / `display_width` / `ansi_enabled`。**整条流程画成表头**（参考数据 4 → K线 3 族×6 频率 → 复权 4）。**三态**（用户 2026-10-09 定）：`·` 灰=队列中 / `●` 绿=正在读取 / `●` 白=读取完成（**含「未过期、本次不重取」**）——**只给那一个点上色**，节点名与阶段名不上色、不挂文字。TTY：原地重画 + 表头下 3 行滚动状态窗；非 TTY（含**不支持 ANSI 的控制台**）：表头只打一次，之后每次变化补一行 `  名字 符号` —— 不整块重打（26 节点 × 7 行 = 182 行噪声）。`save_refdata` / `save_kline_tdx` / `save_xdxr_tdx` / `save_adj` 都新增 `on_progress`（`save_refdata` 的形参是 `(集合, 'start'|'done', entry)`）+ `echo`。⚠️ **参数细节只在 `-v` 下打印**（刷新闸那句、逐集合状态表在「全是 ok/命中缓存」时也省掉）|
| ⚠️ **`echo` 是承重的** | **banner 活跃时，它下方的一切输出都必须走 `Banner.echo`** —— 直接 `print` 会让 banner 的行数记账少算一行，下次重画整块写花。所以生产侧那四个函数都提供 `echo=` 形参（默认 `print`），CLI 传 `banner.echo`。**加新的打印时别忘了这条。** |
| 进度条 | `fetch_stock_info` 的 tqdm `desc` 从固定的 `[pytdx:stock_info]` 改成**当前 code**（集合名 banner 上已有）。akshare 的 `[akshare:financial]` 同批改（同一口径）。两者的逐只进度条都 `leave=False`（进度条不留在屏上，banner 才能原地重画）|
| 实测 | **2026-10-09 复测**：`--save x` / `all` / `tdxx` / `--save-x` **全部 exit 2**（D18 改用 argparse `choices=` 之后，取值非法一律走用法错）；`--help` 取值只列三个。**`--save qmt` 实跑 → exit 0**：K 线跳过，参考数据**走回退源成功**（三个节点全绿）—— ⚠️ 此行原记「三节点全灰 + exit 1」，那是「不许 akshare 回退」时的结论，与下面 `etf_list` 行的「不再传 `exclude_sources`」自相矛盾；现按实测更正 |
| `etf_list` | **入口已补回**（`--save-x` 删掉后它一度没有 CLI 路径）。实测：akshare **1694 行 / 1.4 秒**；⚠️ `tdxaidata` 的 `get_trackzs_etf_info` 返回 **0 行 + `[错误码 2]`**，那条路是坏的。故 `--save` **不再传 `exclude_sources=('akshare',)`**。upsert 键 `['code']`、**无差量删除**，补写不会误删 |
| **ETF 复权已接通** | `etf_xdxr` + `etf_adj` **改由 pytdx 直供**（`DECISIONS.md` **D15**）：`save_xdxr_tdx` / `save_adj` 各加 `target='stock'|'etf'`，股票与 ETF 跑**同一条**编排。复权行现在是 `stock_xdxr → stock_adj → etf_xdxr → etf_adj`，四个都会点亮 |
| ⚠️ **修掉一个 ~100 倍的假跳空** | 存量 QMT 的 `etf_adj` **漏了份额折算**（`category=11 扩缩股`）：159110 跨 2025-09-18 原始跳变 **9891%**，本实现因子后 **-0.09%**，旧数据仍是 9891%；159398 同理（9918.8% → 0.188%）。**TDX 口径不是等价互换，是修错** |
| ⚠️ **未闭合** | 换口径**判据察觉不到**，故跑完还要 `save_adj(全部 etf_xdxr code, target='etf')` 强制重算一次（已跑：335 只 / 467,457 行 / `refused=0`）。重算后 188 个大幅事件**残余中位 1.55%**（= 正常市场波动），但**仍有 6 个 2026-07 的份额折算残余 7~14%**（588200 / 159558 / 561310 / 159538 …）—— TDX 的 `suogu` 是整数比例（如 3.0）而实际价格比是 2.62，推测是按**净值**折算vs 价格含**溢价收敛**，**未验证**。见 `DECISIONS.md` D15 |
| 两个已修的实现坑 | ① `fq.xdxr_to_adj` 原先只认 `category==1`，扩缩股（乘性：参考价 = 前收盘/suogu）被完全忽略；② `kline_doc.xdxr_adj_events_changed` 同样只比 `category==1`，而它**同时把门着写入**（判「未变」就 `return` 不写库）→ 份额折算事件**一行都进不了库**。两处都已扩到认 `category==11`，各有 doctest |
| 不用 QMT 的 `dr` | QMT 的 `dr` 与事件字段**没有稳定换算关系**（1116 行里分红公式只对得上 16 行，中位差 0.16%、尾部 99%）。故改用 `fq.xdxr_to_adj` 从**我们的**事件+收盘价自算 —— 与 `stock_adj` 同一条公式。akshare 另有 `fund_etf_hist_em(adjust='qfq')`，但它给的是**复权后价格序列**、不是事件/因子，当不了 `etf_xdxr`，只适合做交叉验证 |

**一条必须记住的取舍**：`fetch_stock_info` 的逐只 tqdm 改为 **`leave=False`** ——
进度条不留在屏上，banner 才能原地重画。代价是任何调用路径下 `stock_info` 都少一行滞留的进度条。

**实跑 `--save qmt` 时揪出的两个自己的 Bug（都已修）**：
1. 退出码判据写成了 `status == 'failed'`，但**源不可用走的是 `_pick_source` 抛异常那条路，
   `entry['status']` 停在初始的 `'skipped'`** → 三个集合全废却 `exit 0`。改成按
   「**一个集合都没取到行**」（`not any(e['rows'])`）判。
2. `QmtSource` 是**唯一没实现 `unavailable_reason()` 的源**（另外四个源都有），
   于是报错只打一句「qmt 当前不可用：**原因未知**」。已补上，现在报的是
   「QMT 源已停用（QMT_SOURCE_ENABLED = False）：MiniQMT 自 2026-10-01 停服…」。

**顺手清掉了 purge 那条链上的三个毛病**（用户指出「两处打印不是同一个对象」）：

1. **测试真删库**：`test_cli_tools` 那两个 purge 用例**原来真连库** ——
   `--purge-l1` 那条链会把 `GOLEMQ_STOCK_CN_REALTIME` 里真实的 `realtime_*` **drop 掉**。
   已改成 mock 市场方法。
2. **两处打印分属两个模块**：逐日 `print` 在 `markets/StockCN/tools.py`，
   而测试 patch 的是 `GolemQ.cli.tools.print` → 那 14 行照样漏到屏上。
   **现在只留一个打印处**：市场侧 `purge_historical_collections(client)` 改成
   **纯逻辑、返回删掉的名单**，由 `cli/tools.py::purge_mongodb_database` 独占打印。
3. **`cli/tools.py` 的汇总分支是死代码**：市场函数**没有 `return`** → 返回 None →
   `if verbose and collections:` 永远为假。现在返回真实名单，那行才活。
   顺带：`list_collection_names()` 原来**在循环里逐日问**（一轮 28 次），现改成取一次。
   异常也不再吞成「一次未命中」（那样 Mongo 不可达会**静默空转 28 轮然后报成功**），
   改为抛出、由 CLI 统一 `[warn]`。

**全量跑里 `未找到集合` 行数：28 → 0。测试数 170 → 175**（合并 2 个 purge 用例为 1，
新增 banner / on_progress / etf_list 三组）。

**测试状态**：`Ran 194 tests`，**0 失败（全绿）**。曾经那 6 个失败已清：钉钉 3 个是
**测试自身的陈旧假设**（与 token 无关），StockHK 3 个是把「磁盘上只有 StockCN 一个市场包」
当成了事实。原句：6 个失败**全部改动前就有**（已 `git stash` 对干净树复核）：
3 个 `test_messenger`（钉钉配置缺失）+ 3 个是 StockHK 被 `auto_register_markets`
注册进 `GQMARKETS` 撂倒的老断言（`test_cli_tools` ×2、`test_stockcn_singleton` ×1）。
**都不是本次引入的。**

**另一条已核实的结论**：`000001` 那六种转义里，**`sz.000001` 与 `000001sz` 取不到数据**
（平行实现分叉：`fetch_stock_info` 的 `split('.')[0][-6:]` 各自为政）。可用的是
`000001` / `000001.sz` / `sz000001` / `000001.XSHE`。实测表在
`datasource/pytdx_source.py` 模块 docstring 与 `PITFALLS.md` P2。

### ✅ `--save tdx`（2026-10-08）：K 线直写 8.3

| 项 | 内容 |
|:--|:--|
| 命令 | `python -m GolemQ.cli --save tdx`（取值 `tdx`/`pytdx`/`qmt`，见 `DECISIONS.md` D14）|
| 落库 | 8.3 `golemq_stock_cn`：`stock_day`/`stock_1min`…/`index_*`/`etf_*` + `stock_xdxr` + `stock_adj` |
| 增量 | 逐 code 取该集合 `ts` 最大的一根，**再往前 5 个交易日**作为窗口，窗口整体先删后插 |
| 新文件 | `markets/StockCN/kline_doc.py`（纯函数写侧契约，进 doctest）、`datasource/pytdx_kline.py`（分页取数）、`kline_save.py`（编排）、`kline_status.py`（只读覆盖核对）|
| 配套开关 | `--save-dry-run`（只取数 + 逐字段对拍）、`--save-coverage`（只读缺口报告）、`--save-codes/-targets/-frequencies/-jobs/-start`、`--save-no-adj` |

**已端到端验证**（600519，真写库）：写 10 行/删 6、写 1680 行/删 1439；
**再跑一遍集合行数不变（幂等）**；`--save-dry-run` 逐字段对拍
**1439 行里 1433 行一致到 1e-6**，超差的 6 行全是 **09:31** 那一根。

**三条实测口径**（详见 `PITFALLS.md` P16/P17）：分钟标签**同口径不偏移**；
`vol` 单位按市场/频率不同（股票/ETF 分钟 **÷100**、指数日线 **×100**、指数分钟**原值**）；
`amount` 哨兵归零。

**顺带查明**：库内 09-24 的 1min `vol` 之和比它自己的日线少 **183 手**，而 pytdx 的
09:31 恰好**多 183 手** → **旧数据漏了开盘集合竞价，pytdx 是对的**；且存量普遍缺
**15:00** 那根，我们的写入会补上。

**试跑（2026-10-08）后修的三件事**：

1. **输出不可读 + 闪烁**：闪烁来自 **akshare 自己的 tqdm**（`stock_financial_analysis_indicator`
   内部，实测 `0/2 … it/s`），已用项目既有的 `suppress_stdout_stderr` 压掉。
   同时补了**按批打印的进度行**（不做逐条刷屏）：
   * K 线：`[stock_day] 1800/5573 32.3% | 300333 2026-09-30→今天 | 写 N 删 M | 空 X 跳过 Y 重连 Z 错 W | 已用 1m23s 预计 3m57s`
   * 参考数据：`[pytdx:stock_info] 400/5579 7.2% | 000980`
   短循环（< 200 条）**不报** —— 3 个 code 打进度比不打还吵。开关 `--save-progress-every`。

2. ⚠️ **`--save-codes` 会删数据（已修）**：`codelist`（部分取数）× `delete_delta_key`
   （语义是"本次取到的就是全部"）= **把没取到的标的静默删掉**。实测把 `stock_info`
   从 **5,574 行删到 250 行**（我两轮试跑 + 用户一轮）。护栏放在 `save_refdata` 里
   （传了 `codelist` 就不删差量，并打印一行说明），`PITFALLS.md` **P19** 记录，
   回归测试 `test_cases/test_refdata_guard.py`。**已全量重取恢复**。

3. **xtquant / akshare 都不再进 `--save tdx` 的 import 路径**（见下 + `PITFALLS.md` P18）：
   xtquant 的导入会打 banner（原先**任何** CLI 命令都打）；akshare 是因为
   `scribe.py` 在模块级 `try: import akshare`，而 `markets/StockCN/__init__.py`
   会导入 scribe —— 实测拖进 **375** 个模块。现已实测：两者**各 0 个模块**。
   同批还修了 `core/settings.py`（QUANTAXIS 句柄改惰性）、`GolemQ/__init__.py`
   与 `core/__init__.py`（子包/re-export 改惰性）、`supervisor/function_checkin.py`
   的模块级单例、`symbol.py` 把 `DATABASE` 当默认参数这几处。

   ⚠️ **仍有遗留**：`--save tdx` 还会加载 QUANTAXIS（+tushare/statsmodels/matplotlib，
   约 670 / 3,728 个模块），因为 `scribe.py`/`fetch.py`/`supervisor/heartbeat.py` 等
   约 10 个文件仍在**模块级**（或当默认参数）取 4.4 的 `DATABASE` —— 那是 `D9` 记的
   遗留耦合，要改成函数级才能让 tdx 路径彻底不碰 QUANTAXIS。

**顺带清掉一颗雷**：原先**任何** CLI 命令都会先打一行 `xtquant文档地址：…` ——
那是 xtquant 包**自己**在 import 时打的，链条是
`cli/__main__.py` 顶层 → `supervisor/scheduler.py` 顶层 → `xtquant_tools.py` 顶层。
既然 MiniQMT 已停服、xtquant **不会再被使用**，已把全部顶层 `import xtquant` 下移到
**使用处**（含 `QmtSource.available()` 改成恒 False 的 `QMT_SOURCE_ENABLED=False`，
不再 import）。见 `PITFALLS.md` P18。

**已知缺口（跑 `--save-coverage` 复现）**：`etf_day` 抽查 1,689 只里 **739 只有缺日**；
`stock_adj` **138 只**落后于 `stock_day`；`stock_adj` 的移植保真度抽样 60 只里 **57 只逐值相同**
（差的 3 只是日线有洞的早期段）。⚠️ 无成交日在源端就没有 bar，**缺日 ≠ 都该补**。

### ✅ QUANTAXIS 已全树剔除（2026-10-08，`DECISIONS.md` D12/D13）

**验收**：`grep -rn "^ *from QUANTAXIS\|^ *import QUANTAXIS" GolemQ/` → **零命中**；
守卫 `test_cases/test_no_quantaxis.py`（源码零 import / 运行时不加载 / 5 个句柄取用抛 `AttributeError`）。

| 阶段 | 做了什么 | 证明 |
|:--|:--|:--|
| 0 取证 | 基线：import 1 处；`import GolemQ.cli.__main__` → **3,564** 个模块；4.4 `StockCN_watchdog_eneloop` 3 行 + 归档 31 行；`--eneloop-list` = **13 只** | 见下表 |
| 1 删死代码 | 删 `align.py`(602) / `crawler.py`(358) / `scribe.py`(907) / `services/align/`(562) / `services/persistence/`(1318) / `services/features/`(1211) 整包整文件；`fetch.py` 删 6 个函数（1668→272 行）；`symbol.py` 删全部 4.4 读取器（909→425 行）；`maintenance`/`timeseries` 各删 1–2 个；CLI 删 6 个参数 + 4 个分支 | 144 项测试，失败项**与基线逐项相同**（4 fail + 2 err）|
| 2 改绑 + 改名 | 新增 `GOLEMQ`（8.3 `golemq`）；心跳/签到/关注列表改绑它；`DATABASE_STOCK_CN*` → `GOLEMQ_STOCK_CN*`（10 个 .py，0 残留）| **读回非空**：checkin 写入后 8.3 `function_checkins` 0→1（4.4 不动）、心跳 8.3 0→1 |
| 3 迁移通道 + 搬运 | 新增 `core/migrate44.py`（**一次性专用**，地址常量 + `GQ_MIGRATE44_URI`，不读 `~/.QUANTAXIS`）；新增 `--migrate-eneloop`；`--migrate-financial` 换到该通道 | `--migrate-eneloop` 源 3/31 → 目标 3/31（跳过 0）；`--eneloop-list` = **13 只**，与基线一致 |
| 4 拔 QUANTAXIS | 删 `_qa_handles()`/`__getattr__`/`change()` 与三处惰性转发 | `grep` 归零；5 个符号取用抛 `AttributeError`；`sys.modules` 无 `QUANTAXIS` |
| 5 守卫 + 文档 | 新增守卫测试；改 D9（作废）/D12/D13、MIGRATION_STATUS #2/#5/§九、Project.md、CLAUDE.md、PITFALLS P18、GLOSSARY、MONGODB83、RESTRUCTURE_PLAN | `--save tdx` 实测：QUANTAXIS/xtquant/akshare/tushare/statsmodels/matplotlib **全 0**，模块总数 **3,564 → 1,709** |

**活路径逐条验过**：`--save tdx --save-dry-run`、`--save-status`（5 集合，qmt 标不可用）、
`--heartbeat-watchdog`（跑通，显示"没有模块记录"= 8.3 从空开始，**预期**）、
`--eneloop-list`（13 只）、`--purge-l1 --verbose`（跑通；当前无超期日集合可删）、
`--sub` 契约测试（3/3 ok）。

**已删功能记档（要用就跑旧树）**：`--stock-min-aligned` 整条链（换手率/估值对齐）——
它读写 4.4 `golemq` 的 `stock_a_snapshot*`/`stock_metadata*`/`stock_diagnosis`/
`stock_valuation`/`stock_ranking`/`stock_moneyflow`，**8.3 里都没有落点**。

**未搬、且是有意不搬的**：`module_heartbeats*` / `function_checkins*`（8.3 从空开始 ——
搬过期的运行锁反而危险）、`index_list`（唯一消费者 `symbol.GQ_fetch_index_name`
随那条链一并删了）。

### ✅ `--save tdx` 两处更正（2026-10-08，用户指出）

1. **窗口余量默认 5 → 0**：水位是 per-code 的，不需要靠 margin 兜"跑挂在中途"。
   实测（200 只票、日线）`margin=5` 每票写/删 **6.0** 行 → `margin=0` **1.0** 行；
   5,574 只票每天少写少删约 **2.8 万行**。新增 `--save-margin-days`（修补时显式开）。
2. **窗口粒度：天 → 那根 bar**（`floor_ts`）。已收盘 bar 冻结，按天划窗口时
   `1min` 会把水位那天已冻结的 **240 根**全重写；按根划只重写**最后一根**。
   实测每票删除量：`1min` **1.00**、`5min` **1.00**、`day` **1.00** 根（改前是 241/241/6）。
3. **删除范围：区间删 → 精确删**。原写法 `{'code': c, 'ts': {'$gte': floor}}`
   会**越过本次写入集**删东西：源端临时缺数的那些天、库里明明是好的 → 被删掉且补不回来。
   现改为 `{'code': c, 'ts': {'$in': [本次要写的 ts]}}` —— **冻结日一根都不碰**。
   实证（临时库）：库内 3 根、本次只写第 3 根 → 旧写法**丢 2 根**，新写法两根原样、第 3 根被覆盖。
3. **领域约定记档**（用户明确，`PITFALLS.md` **P20**）：盘中 **`day` 不更新**（仍是昨天那根），
   盘中行情由 **REALTIME 的 L1 tick 合成**。⚠️ 但 pytdx 的 day 接口**盘中会返回「当日累计」bar**
   → 盘中跑本命令会把未收盘的当日 bar 写进去；**收盘后再跑一次即被覆盖**，且命令会打印告警。

### A —— 集合计数（本轮实测）

库 `golemq_stock_cn`：

| 集合 | 行数 |
|:--|--:|
| `stock_list` | 5,574 |
| `stock_info` | 5,574 |
| `etf_list` | 1,674 |
| `stock_block` | 72,856 |
| `financial` | **0** |
| `stock_day` | 17,871,122 |
| `stock_adj` | 17,547,519 |
| `index_day` | 8,602,127 |
| `stock_min` | **0** |

### D —— 已完成（提交 `6d93dfe` / `6f20152` / `72f3025` / `de24c8b`）

| 阶段 | 内容 | 验证方式 |
|:--|:--|:--|
| 常量 | 149 个被引用常量全部对齐老树存储名；值错误 20 → 0，未定义 62 → 0 | 两树共有常量漂移 0/165 |
| `_StubMeta` | 4 个类改为一律抛 `AttributeError`（P1 闸门已翻）| —— |
| 日期助手 | `QA_util_*` → `GQ_util_*`，含 2 个新建 | 1600 组对比 **0 差异** |
| 股市列表 | `symbol.py` 改本地 Mongo 读 | 5591 行，与 QA `code` 集合完全一致 |
| 参考集合库 | `scribe.py` 的 `DATABASE` 改从 `core.settings` 导入 | —— |

QUANTAXIS import 计数：**32 → 11**（另有 37 处注释/文档提及，非依赖）。

---

## 三、D —— **QUANTAXIS 已解耦到只剩 1 处**

```
$ grep -rn "^ *import QUANTAXIS\|^ *from QUANTAXIS" GolemQ/ | wc -l
1        # 只有 core/settings.py:34（QA_Setting）—— D9 定的「暂留」
$ # 裸 QA.* 前缀引用（排除注释/docstring）：0
```

**这一节的内容已全部完成**：替身（`7fcb5ec`）、ETF 复权（`c83b593`）、
分钟读取器重写（`fe2fd95`）、HIGH #11 股票复权（`ed3a994`）、
最后六个模块解耦 + 未定义名清零（`ce58c0d`）。

### 解耦时查到的事实（都写进了 commit message，此处只留索引）

| 事项 | 结论 |
|:--|:--|
| `QA.MARKET_TYPE` vs `GolemQ.core.constants.MARKET_TYPE` | **12 个常量值全同** → 等价替换 |
| `symbol.GQ_fetch_index_name` 默认集合 | 原为 `etf_list`（**复制粘贴错**，来自 `GQ_fetch_etf_name`）。`etf_list` 无真指数（`000300` 命中 0），而 `index_list` 里是 `'沪深300'`。QA 自己的默认就是 `index_list`。该函数零调用者 → 修它不改变既有行为 |
| `realtime.py` 的 QA 实时兜底 | 读 `quantaxis` 库，**该库 realtime_\* 集合数为 0**（`QAREALTIME` 有 10 个）→ 只能返回 None，移除无代价 |
| `QA_data_min_resample` / `min_to_day` | 已回迁到 `analysis/timeseries.py`，与 QUANTAXIS **逐值 0 差**（960 根 1min → 5/15/30/60min 与 1D 全部形状与数值相同）|
| `fetch.py` 的 26 处未定义 `QA.` | 回归已清（`ce58c0d`）。INDEX 分支接 `GQ_fetch_stock_day_adv` / `GQ_fetch_index_min_adv`；**CRYPTOCURRENCY 分支改抛 `NotImplementedError`** —— 本树无数字货币数据源，返回空会让「未实现」与「没有数据」无从区分（两处都不在 try 内，抛得出去）|
| `services/features.py` / `_hourly_fetch.py` | 也用 `QA.MARKET_TYPE` 却**从未 import QUANTAXIS**（未定义名）。逐文件普查漏了它们，靠**全树普查**才捞出来 —— 教训：普查要按「有没有用」而不是「有没有 import」|

### ✅ `services/features` 已补齐成包（2026-09-21，提交 `58c616b`）

原先：拆分的 6 个分册早就写好了，但**从未建 `__init__.py`**，同层还留着 956 行的
旧巨石 `features.py` —— 普通模块优先于命名空间包，所以解析永远落到巨石上，
**6 个分册一个字节都没生效过**（全树零引用）。

现在：新增 `_hourly_crud.py`（原巨石里最后 3 个函数，逐字抽取）、建 `__init__.py`
（再导出全部 11 个函数）、**删除巨石**。

**一个不删不行的理由**：留着巨石是个陷阱 —— 它是死的，但任何人编辑它都"改了没反应"。

### ⚠️ 补齐时发现的两个缺陷（**均为老树继承，非重构回归**）

**① `GQ_fix_daily_metadata` 永远修不了东西。** `_daily_crud.py` 里：
```python
int64_dates['date'].apply(lambda: (x/1000) != int64_dates['date_stamp'])
```
lambda **没有参数**却用了 `x`（上面两行都正确地写了 `lambda x:`）。只要
`int64_dates` 非空就抛 `NameError` → 被 `except` 吞成一行打印 + `return None` →
**下面的 MongoDB 写回循环从不执行**。空的 `int64_dates` 不会调用 lambda，所以
"无事可修"时它看起来正常。
老树 `GolemQ_old/scribe/features.py:580` **逐字相同** → 继承缺陷。

**② `GQ_fetch_hourly_metadata_reality` 恒返回 None。** 它按 `time_stamp` 过滤，
而默认集合 `stock_diagnosis` **含该字段的文档数为 0**（日线兄弟按 `date_stamp`
过滤，那个字段有 → 能出 24 行）。它的写入目标 `stock_metadata_60min` / `_15min`
在库里**也不存在**。老树 `markets/StockCN/scribe.py:364-392` 同样 →
继承缺陷。需要先定「小时级 metadata 到底住哪个集合」。

**两者我都没擅自改**：① 修好会启用一条从未跑过的 Mongo 写路径；② 需要先决定数据归属。

### 已完成部分（全部实测）

| 件 | 落点 | 验证 |
|:--|:--|:--|
| 4 个 K 线容器 + `to_qfq`/`select_code` | **新增** `markets/StockCN/datastruct.py` | 8 只 × 全历史 **40,192 行逐值 0 差** |
| 板块容器 | 同上 | `.block_name`/`.get_block`/`.get_blocklist` |
| 日线读取器 | `kline83.GQ_fetch_stock_day_adv` | 端到端 `StockCNQuotes` **8,011 行 0 差** |
| `quotes.py` / `fetch.py` 接线 | —— | 单测 12/12；全量测试与基线**逐项相同** |

### ✅ ETF 复权已恢复（2026-09-21）

`markets/StockCN/etf_fq.py` 已从老树回迁 —— 这正是 `MIGRATION_STATUS.md`
**HIGH #9**（重构把整个模块丢了，只留一句「ETF 需要人工复权」的打印）。

调用点（**每条路径恰好一次**，因为重复调用会二次乘因子）：

| 路径 | 位置 |
|:--|:--|
| 门面 | `kline83.get_kline_price_v3` / `get_kline_price_min` |
| legacy 行情 | `quotes.py` 的 `_apply_fq`（按标的选 `to_qfq` 或 ETF 复权）|
| legacy fetch | `fetch.py` 里原打印提示的桩 |

验证：**独立重算 `qfq == raw × adj(date)` 逐根 0 差**；除权日上原始收益
−0.73%/−2.13%/−2.49% → 复权后 +1.27%/+0.29%/+0.10%，修正量与因子台阶精确吻合；
股票与真指数原样返回、一次 Mongo 都不查。

**顺带修了两件事：**

1. 回迁时发现老树 docstring **声称幂等但从未实现**（它置 `if_fq='qfq'` 却从不读），
   连调两次会乘两遍因子（实测比值 0.931873 = 其除权因子）。新模块补了 `if_fq` 早退。
   **这是接线时必须逐路径核对「只调一次」的原因。**
2. 上一轮我给 `quotes.py` 换日线读取器时**引入过一个回归**：ETF 现在返回
   `GQ_DataStruct_Index_day`（按设计无 `to_qfq`），而 `quotes.py` 无条件调
   `.to_qfq()` → `AttributeError`。`_apply_fq` 一并修掉。

### ✅ 分钟线读取器已重写（2026-09-21，提交 `fe2fd95`）

`GQ_fetch_stock_min` 不是「改个库名」——旧实现**五个缺陷叠加、从未返回过数据**：
读 `DATABASE.stock_min`（golemq 库无此集合）→ 空游标 → `res.vol` 抛 → 被光秃秃的
`except` 吞成 `None`。**实测每次调用都返回 None。**

重写后读 8.3 分频集合，并：去掉 `collections=` 参数、频率归一化收敛到
`kline83.normalize_frequency`（原表抄了三份、失败处理各不相同）、改按 `(code, ts)`
查询（`ts` 是时序 timeField，能分桶剪枝；8.3 没有 `type` 字段）、`format='numpy'`
无数据返回 `None` 而非 0 维 object 数组。`_adv` 现在**按市场选容器**（ETF/指数 →
`GQ_DataStruct_Index_min`）。

验证：分钟对照 QUANTAXIS **354 行 0 差**；日线复跑 **8,011 行 0 差**；bar 数自洽
（1min=240 / 5min=48 / 15min=16 / 30min=8 / 60min=4 = 一个交易日）。

**顺带修掉一个静默读空**：`_read_timeseries` 现在把代码截到 6 位。`quotes.py` 传的是
`normalize_code(code)`（`'600519.XSHG'`），而集合里存 `'600519'` —— 分钟路径因此
返回 0 行，日线路径也只差一个 `[:6]` 就会同样静默读空。

### 两条刻意分歧（勿当 bug「修」回去）

**① QUANTAXIS 的 `drop_duplicates()` 会吞真实 K 线。** 基类构造里的
`DataFrame.drop_duplicates()` **不带参数 → 只比列值、不看索引**，而那时
`date`/`code` 已进索引。于是**任何 OHLCV 与更早一根完全相同的 K 线会被静默删除**
（`000001` 丢 1993-06-04 与 1998-06-20，都是真实交易日）。替身**不复制**。
详见 `datastruct.py` 模块 docstring。

**② 零成交分钟的 bar 保留。** QUANTAXIS 的 `QA_fetch_stock_min` 有
`.query('volume>1')`，**丢掉所有零成交的分钟**（600519 在 2024-01-02 的 1min 因此
少 2 根：14:58/14:59，均 `volume=0`）。本树**不过滤** —— 消费方需要看见它们：
`quotes.py` 的分钟路径与 `get_kline_price_min` 都有专门的 zero_trading 处理
（删 4 根一组的午休段、修正 13:00 时间戳），**bar 不在就永远不会触发**。
这是全树「适配器只管取数，编排层决定范围」的同一条分工（`PITFALLS.md` P1）。

### ✅ 门面路径的股票前复权已修（2026-09-21）

`MIGRATION_STATUS.md` HIGH #11。老树 `get_kline_price_v3` 里 **`:906 to_qfq()`
（股票）+ `:1068 GQ_apply_etf_qfq`（ETF）两条都做**，新树此前只接了 ETF 那条。

`kline83` 新增 `_apply_adjustments(result, market, codelist, verbose)`，
两个读取器各调一次：`market=='stock'` → `datastruct.apply_qfq`（`stock_adj`）；
`market=='index'` → `GQ_apply_etf_qfq`（`etf_adj`）。**互斥由 `market` 保证**，
不存在二次复权。

验证：`600519` 2024-01-02 门面现给 **`1531.145598`**（= 老树值，修复前是
`1685.01`）；门面日线 6 只 **9,640 行 0 差**；门面分钟 36 行 0 差；真指数保持原始；
`quotes.py` 路径未被二次复权。

### ✅ 停牌伪 0 清理：工具已就绪并在跑（2026-09-21）

**决定**：**物理清理**（移动，不是删除，也不是读时过滤）。理由是所有者两条硬约束：
**读取速度优先** + 不希望每个消费方都记得过滤。工具：
`markets/StockCN/maintenance.py` → `GQ_purge_suspended(dry_run=True)`。

| 操作 | 函数 | 状态 |
|:--|:--|:--|
| 找出停牌日 | `GQ_suspension_dates` | ✅ 17,701 组 / 2,252 只 / 2.0 秒 |
| 移出（→ `stock_*_removed`）| `GQ_purge_suspended` | ✅ **已执行**：**3,908,747 行** |
| 回迁（可逆）| `GQ_restore_suspended` | ✅ 已验证幂等 + 来源护栏 |
| 搬 4.4 人工归档 | `GQ_migrate_removed_from_44` | ✅ **已执行**：326,522 行 |

**全量清理结果**（每集合 `found == moved == deleted` → **无一行丢失**）：

```
stock_1min  3,662,598    stock_15min    46,848    stock_60min  7,516
stock_5min    151,918    stock_30min    22,166    stock_day   17,701
```

`stock_day` 移走 **17,701** 行 = 停牌日总数（一天一根，一一对应）。
`GQ_suspension_dates` 现在返回 **0** —— 标记随日线移走，**重灌后须重跑**（当前已干净，重跑是 no-op）。

⚠️ **归档里 235 个 `(code,ts)` 被重标**：同一根 bar 同时出现在两批归档时，
`ReplaceOne(upsert)` 以后运行者为准，清理在搬迁之后跑 → 这些行标成 `purge`。
**没有丢数据，只是标记被覆盖**，且语义正确（它们所在日确实判为停牌）。

**三条约定（见 P8b）**：归档是**普通集合**（才能建唯一索引/upsert → 重跑幂等）；
归档 **append-only**（回迁写回热数据但不删归档，这才叫可逆）；行带
**`removed_by`** 来源标记（`purge` / `import44`），**回迁默认只回迁 `purge`**。

⚠️ **那条来源护栏是必须的**：4.4 人工归档的行**所在交易日有正常日线**，
清理规则**认不出它们** —— 一旦回迁进热数据就**再也清不掉**。

⚠️ **两个我踩过的坑**：① 回迁**不能 upsert**（主集合是时间序列），要
**先删 `(code, ts)` 再 `insert_many`**；② 搬 4.4 时**别让整数过 numpy/pandas**，
否则 int32 会被拓宽成 int64（实测源与目标**本来就是 int32**，无需转换，只需别拓宽）。

⚠️ **重灌/重新迁移后必须重跑清理** —— 选物理清理的代价，判据可随时重算。

### ✅ 实时行情 L1/L2（2026-10-08 改为「按日时间序列集合」）

| 项 | 状态 |
|:--|:--|
| `--sub l1_tencent` | 腾讯全市场快照（**含五档**），2 秒一轮、约 4,900 只 |
| `--sub l2_tencent` | 腾讯 3 秒一轮的盘口流（字段是 L1 行的**真子集**，重复**刻意保留**）；ETF 那条留给 MiniQMT —— **已停服，见下** |
| 落库 | 8.3 `golemq_stock_cn_realtime.realtime_YYYY-MM-DD`：**按日**、**时间序列** `{timeField:'ts', metafield:'code', granularity:'seconds'}`；行内**保留 `datetime`**（北京时字符串）供重采样 |
| 分流 | 同一个日集合里靠 `source` 区分：`tencent_l1` / `tencent_l2` / `qmt` —— **它同时是去重键的一部分** |
| 读取 | 三个读取器已切到 8.3 同一批集合，按 `ts` 过滤 + 排序 |

**为什么按日、且名字必须是 `realtime_YYYY-MM-DD`**：保留策略是**按集合名**删整日集合
（`markets/StockCN/tools.py` 的 purge，从 14 天前起、连续 14 次 miss 停）。名字一旦分叉
**不会报错** —— purge 只是安静退出，**保留策略静默失效**（磁盘无声地涨）。故名字生成
收敛到 `realtime_collection_name()` 一处，并有单测钉住「写出的名字 = purge 要找的名字」。

**写入是「先删后插」，但只在首次见到该 `(source, code)` 时删**（时间序列不能 upsert、
也没有唯一索引 —— MongoDB 8.3.11 实测）：`delete_many({'code': …'ts': …'source': …})`
只在**本进程还没写过这个 `(source, code)`** 时执行 —— 因为每轮无条件删是**跑不动**的。
实测（探针，4,900 个 metaField 值）：

| 操作 | 单轮耗时 |
|:--|:--|
| `insert_many` 4,900 行 | 均值 **0.95 s** / 峰值 2.1 s |
| `delete_many` 同 4,900 个 `(code)` + 本轮的 `ts` | 均值 **3.7 s** / 峰值 4.5 s |

时序删除是「解压桶 → 摘测量 → 回写桶」，而命中的正是**当前热桶**；L1 每 2 秒一轮。
收窄后语义不变：进程内的重复由 `last_ts` 挡，跨进程的重复只可能出现在**重启后的
第一批**（那时 `last_ts` 是空的，所有行都算 first-seen，照样会删）。

`source` **必须**在删除条件里：L1 与 L2 落在同一日集合、同一 `(code, ts)` 各有一行，
不带 source 会互删。`_write_ts_rows` 的 `last_ts` 也按 `(source, code)` 记，同理。

**顺带修掉一个更根本的 bug**：读取器原先拼的是 `'realtime_{}'.format(dt.today())`，
而本模块 `dt` 是 **`datetime` 类**（老树用的是 `date.today()`），实测拼出
`realtime_2026-10-08 01:00:27.222907` —— **带时分秒**，那个集合**不可能存在**。
即那条读路径**从来没读到过任何集合**（一直返回 `None`），不存在需要兼容的旧读行为。

**已验证（临时库端到端，未碰生产集合）**：名字 ↔ purge 一致；「模拟重启后再写同一批」
**仍是 5 行**（幂等成立）；L1 重写不会删掉 L2 的同 `(code, ts)` 行；读取器返回的
索引是 `(datetime, code)`、无 `_id`、默认只含 `tencent_l1`、按 `ts` 倒序；
`drop()` 对时序集合有效（保留机制成立）。

⚠️ 那次端到端**抓到过一个真回归**并已修：收窄 delete 的第一版按**行**判「是否首次见到」，
于是同一批里同一 code 的第二行起不进删除集 → 重启后库内 5 行涨到 **9 行**。
判据必须**按 key**，再删该 key 在本批里的全部 `ts`（回归钉子见
`test_cases/test_realtime_store.py::test_first_seen_covers_every_ts_of_that_code_in_the_batch`）。

**待做（只能开盘时段做）**：跑一次 `--sub l1_tencent`，确认
① 8.3 出现 `realtime_<今天>` 且文档带 UTC-aware `ts` + 北京时 `datetime`；
② `GQ_fetch_stock_realtime_adv('600519', num=8000)` 读回来非空；
③ `--purge-l1 --verbose` 能打印出 `✅ 成功删除历史集合`（打印的全是 `⏩ 未找到` 就说明
名字格式与 `realtime_collection_name` 分叉了）。

#### ⚠️ MiniQMT 自 2026-10-01 起停服（监管），目前彻底无数据

替代方案（cfquant）尚未落地，且**不是当前首要任务**。处置：

* QMT 那条路**只保留结构**：`QMT_REALTIME_ENABLED = False` 关掉取数与订阅，
  不再每轮刷错；`_l2_rows_from_qmt` / `_l2_qmt_xt_codes` / 后台订阅线程全部原样保留
* `gateway/xtquant/realtime.py`（**第三条**写入路径）当前零调用点 —— 只加注释，不删
* 复活前须知两件事：① 必须先 `subscribe_quote(period='tick')`，否则拿到的是陈旧缓存；
  ② 它走的是「按日**普通**集合 + 唯一索引 + upsert」，与现在的时间序列存储**不兼容**，
  必须先改写入方式，否则第一次写就抛 `Cannot perform a non-multi update`

#### ✅ 本机 QMT 不提供盘口深度（2026-09-21 实测，随停服一起留档）

| 调用 | 结果 |
|:--|:--|
| `get_full_tick` 的 `bidPrice/askPrice` | ETF 与股票**恒为 0**（价格是活的）|
| `get_fullspeed_orderbook` | `当前客户端未支持此功能，请更新客户端或升级投研版` |
| `get_l2_quote` / `get_l2_order` / `get_l2_transaction` | 全部返回 `[]` |
| `subscribe_quote(period='tick')` | 无改善 |

所以 QMT 那条**只记价格**，并把深度标成 `depth: 'unavailable'` —— **不写 0**，
因为「盘口是空的」与「这个源给不了」必须能区分。
另外：订阅 1,674 只 ETF 实测要 **58 秒**，已改为**后台线程**订阅，不阻塞 3 秒主循环
（价格本就不需要订阅）。

#### ✅ 已收口：读写不再指向不同存储

`GQ_fetch_stock_realtime_adv` 与 `*_realtime_adv` 两个 kline 取数**已切到 8.3**
（按日集合 + 按 `ts` 查），与写入端同一批集合 —— 原先「写 8.3、读 4.4」的断裂已消除。
4.4 `QAREALTIME` 里那批历史实时数据（`realtime_2026-09-07…09-30`）**不迁移、不再读**。
判据：那些文档带 `bid1..ask5` 全套五档、**没有 `ts` 字段** —— 新树的写入端总会补
`ts`（`bj_date(...)`），所以它们是**老树**按旧写法写进去的；形态与现在的时间序列
集合不同，要读得先把 `datetime` 换算成 `ts` 再回填。

#### ⚠️ 心跳互斥的健壮性问题 —— **2026-09-25 更正：先前记的成因不成立**

> **先前记的是**：「被 kill 的订阅器会留下 `status='running'` 而 `last_checkin=None`
> 的记录，该记录永不超时、永久占锁」。**复核后该机制不成立**，勿据此排查。

复核证据（`supervisor/heartbeat.py`）：

- 字段名是 **`last_checkin_timestamp`**（不是 `last_checkin`；见 `:52,101,148,237,467`）
- 写入侧（`start_module` `:101`、`checkin` `:148`）**永远写 int，从不写 `None`** → 前提不成立
- `mutex()` 用 `.get('last_checkin_timestamp', 0)`（`:467`）→ **键缺失 = 0 = 视为已超时**；
  `_check_timeouts()` 的 `$lt` 查询（`:237`）同样会命中 null

**真正的锁缺陷是非原子 check-then-act**（记在 `MIGRATION_STATUS.md` §六）：
唯一索引建在 `(module_name, instance_id)`，而 `instance_id` 是 **per-process sha256**
（`:442,447`）→ 两个进程各写各的 id，**索引永不冲突，永远拦不住第二个进程**。

⚠️ **若真观察到「一条记录挡住重启」，成因不在这条机制上** —— 更可能是那个竞态，
或 `last_checkin_timestamp` 为 null 时 `mutex()` 抛 `TypeError`（`:471` 的 `None + int`）。
**要确认必须连库看那条记录的实际字段**；本次无 DB，标为**待复现**。

### ✅ `--save-x` / `--save-qmt`（2026-09-21）—— ⚠️ **两个开关已于 2026-10-09 删除**

> 见 `DECISIONS.md` **D14**：收口进 `--save <SOURCE>`（QMT 参考数据改走
> `--save qmt`）。下面的实测记录是当时的，保留作历史。
> 连带后果——`etf_list` 一度失去唯一的 CLI 入口，**同日已补回** `--save tdx`
> （走 akshare；理由见 D14 与上面的 SAVE-CLI 一节）。

**4/5 通**：`stock_list`(5573) / `stock_info`(5574) / `etf_list`(1674) /
`stock_block`(72856) 全部 `ok`；`--save-qmt` 也正常（MiniQMT 在线）。
`financial` 按项目所有者决定**从 4.4 搬运**（不联网取数），新增
`--migrate-financial`：182,751 行，与源一致，按 `(code, report_date)` upsert 可反复跑。

⚠️ 搬来的 580 个指标列**名字就是位置编号** `'001'…'580'` —— 那是通达信 gpcw 财务
文件的**原生形态**（QUANTAXIS 自己的解析器就这么生成的），不是搬运造成的。
要用这些指标得先拿到通达信的指标目录。

### ✅ 复权代码已收敛到单一实现（2026-09-21，提交 `85bba0f`）

股票与 ETF 的复权原本是**两套平行实现**，各自带一份 `_row_dates` / `_row_codes`
（ETF 侧还多一个 `_index_field`），对齐逻辑也各写一遍（一边 `merge`、一边
`Series.map`）。但两者**只差策略，不差机制**，故机制收敛到新模块 **`fq.py`**：

```
align_factors(data, flat, ffill_by_code=...)      对齐 + 缺失策略
multiply_ohlc(data, factor, keep_factor_column=...)  乘法
```

两个入口只保留各自**真正独有**的部分（因子来自哪张表）。**两条策略差异作为显式
参数保留，不得「统一」**：股票按标的 ffill（表密集但可能缺行），
**ETF 绝不 ffill**（稀疏约定，ffill 会静默算错价）；`adj` 列股票留、ETF 不留。

`fq.py` **不碰数据库**，所以是纯函数、已收进 doctest（**15 → 22 条**）。
全量测试 58 → 65，失败项与基线逐项相同；**两道数值基准仍是 0 差**
（股票 40,192 行 vs QUANTAXIS；ETF 独立重算）。

**一处更正**：我先前说 `_adj_collection` 也重复 —— 不对，两者同名但读**不同集合**
（`stock_adj` vs `etf_adj`），保留是正确的。

### 📌 通达信 0 成交分钟：已写成 `PITFALLS.md` **P8b**（跨系统通用）

**同一症状（0 量 + OHLC 持平）有三个成因，处理相反**：① 收盘集合竞价
（14:57–15:00，**每天每票都有**）→ 保留；② 封死涨跌停 → 保留；③ 停牌伪 0 → 剔除。
QUANTAXIS 的 `.query('volume>1')` 把三者混为一谈，等于**每天都在删所有股票的
收盘竞价分钟**。

实测：`600519` 在 2026-09-01~18 的 **19 根** 0 量 bar **全在 14:58/14:59**。
成因 ③ 的唯一可靠判别特征（4.4 里是 **`1e-34` 级 float**）**在迁到 8.3 时被 cast
成 int64 而丢失**，现在只能靠「时间位置 + 整日形态」判别。

**正确的判别判据（已用真值集验证）**：查**当日日线的 `vol`** ——
**哨兵值（`5.877471754e-39` = 2^-127）或 0 → 当日停牌**，分钟的 0 量是伪 0（成因 ③）
→ 剔除；**正常数值 → 当日有成交** → 那些 0 量分钟是 ①② → 保留。

**验证过程（两步都被数据改了结论）**：

1. 所有者最初给的是「**停牌当天日线缺失**」—— **实测不成立**。在所有者当年人工
   挑出的真值集 `stock_min_removed`（1,062 组 `(code,date)` / 409 只）上，
   **日线缺失 0 个**：通达信给停牌日**仍然落一根日线**，只是那根是空的。
2. 改查日线 `vol` 后，同一批真值**均匀抽 200 组全部命中**；反方向的正常交易日
   （含 ①② 的 0 量分钟）日线 `vol` 都是实数，**不误判**。

⚠️ **哨兵值本身不是判据** —— 它同时出现在**必须保留**的收盘竞价分钟
（实测 600519 的 14:58/14:59 `amount` 也是它）。区分点在**哪一层带哨兵**：
**日线**带 = 全天没成交 = 停牌；**分钟**带 = 那几分钟没成交 = 正常。

⚠️ **另一个坑**：日线 `vol` **混着 int 与 float 哨兵**，`vol == 0` 会**漏掉全部
停牌日**（本项目统计时被骗过一次）。按量判断要用 `< 极小阈值`，不能等值比 0。

**开销**：每读一次分钟数据多**一次**日线索引区间查询（不是每根 bar 一次）。
按所有者意见**列为可选、不写进读取器**，需要的消费方自行 opt-in。配方见 P8b。

### 🔶 E —— ETF 独立成 `ETF_CN` / `etf_*`（2026-09-25，提交 `1cc47e2`）

**做了什么**：ETF 从 `INDEX_CN` 提成一等类型。此前它与真指数**共用 `index_*`**
（老树 `save_qa.py` 刻意「与 QUANTAXIS 一致」），代价是 ETF 拿不到 `to_qfq()`、
`GQ_is_etf` 只能嗅探描述串 `'ETF基金'`。`MARKET_TYPE.ETF_CN` 常量**早就存在却无人用**。
详细记录见 `MIGRATION_STATUS.md`；号段口径见 `PITFALLS.md` **P12**。

**联网核实的号段**（原实现深市 5 段里**错了 4 段**）：`150`分级 / `16x`LOF /
`180`REITs / `20`B股 全被误判成「深交所ETF基金」；`82`是**优先股**却写成「北证A股」；
`158` ETF 段缺失；`200`(B股) 分支被 `20` 抢先命中而**从未生效过**。

**验证方式（可复用）**：`is_stock_cn` 是纯函数 → 扫**全部 100 万六位代码**与旧实现
逐条比对，实测 9 秒。92,000 条差异 / 7 类，逐类有意，**零意外改动**。
工具 `tools/dump_is_stock_cn_baseline.py`（转储 12MB 已 gitignore）。
测试 65 → 93，失败项与基线逐项相同。

**顺带修掉两个老实现的真 bug**：`is_stock_cn('XSHE004000')` **返回 `None`**
（深市分支 `elif startswith('XSHE'): pass` 掉出函数 → 调用方解包即 `TypeError`）；
`XSHG510000` / `sh.510000` / `sz.160000` 等带标记写法去掉不标记 → 判成「未知」。

**⛔ 未做 —— 数据拆分**（MongoDB 未起）。**已定方案**：可逆拆，复用
`maintenance._move`；只记「搬了哪些 code」进一张小记账集合，**不做全量归档**
（那等于复制几百万行）。时序集合不支持 upsert → 写入走 `insert_many`。

⚠️ **拆分前必须先跑一致性核对**：`is_stock_cn()==ETF_CN` 的代码集 ⟷ `etf_list`
（1,674 只）**双向**比对，差异逐条定性后才动数据 —— 拆错方向不会报错，只会读空/读错。

⚠️ **两处连带的集合路由变更**（分类修正的必然后果，**需查库确认有无数据**）：

| 代码段 | 旧 | 新 | 路由 |
|:--|:--|:--|:--|
| `200`–`209` B股 | `index_cn` | `stock_cn` | `index_*` → **`stock_*`**（所有者定「先查库再定」）|
| `161`–`169` LOF、`184` 封基 | `None` | `fund_cn` | `stock_*` → **`index_*`** |

### 仍未做

- 「绑错库」残留：`fetch.py` 的 `GQ_fetch_stock_list_day`
  （`collections=DATABASE.stock_day` 默认值 → `golemq` 库无此集合）、`scribe.py`
  的 `QA_fetch_stock_list/index_list/stock_terminated` 默认值
  （`stock_list` 只在 8.3 有；`index_list` 在 **`quantaxis`** 库有；
  `stock_terminated` **两个库都没有**）。
  注：`GQ_fetch_stock_list_day` 全树零调用者，属潜在缺陷不是活 bug。

---

## 四、待你决定

| # | 事项 | 选项 |
|:--|:--|:--|
| 1 | **`financial` 集合（当前 0 行）** | **(a)** 暂缓（上一轮倾向此项：无消费方 + akshare 全量 46.5h + tdxaidata 字段目录未知）；**(b)** 你给 tdxaidata 字段名，走批量通道几分钟灌完；**(c)** 限定代码范围跑 akshare |
| 2 | **C2 撮合移植** | 需你先给策略实现（`XGB_ECHO_TIMING_LAG` + `QUADRANT_LEVERAGE_MACD_TIMING_LAG_DUMMY`）—— 信号公式属策略侧，不猜 |
| 3 | 未被引用但值错的常量 | 只有**全量审计**能闭合（两个工具都只看被引用的）。只读检查，成本低 |
| 4 | **`GQ_fix_daily_metadata` 的 lambda 缺陷** | 修 = 启用一条从未跑过的 Mongo 写路径，属行为变更 |
| 5 | **小时级 metadata 的归属** | 读函数按 `time_stamp` 过滤、写目标集合不存在 —— 先定它住哪 |
| 6 | **MongoDB 起一下**（当前 27017 超时、无服务、默认路径无 `mongod.exe`）| **阻塞 E 的数据拆分**与三项验证（数值基准、拆分后对照、端到端复权一次）。代码半边已完成 |
| 7 | **E 的两处路由变更要不要随之搬数据** | `200–209`（`index_*`→`stock_*`）、`161–169`/`184`（`stock_*`→`index_*`）。查库定；若无数据则纯属分类修正 |

### ✅ 两处「低垂果实」**已做掉**（2026-09-25）

全量复核 `MIGRATION_STATUS.md` 时挖出来的，**真实现已在同树、只是没接上**：

1. **`GQ_util_get_last_day` 接到真实现** —— 提交 `e798362`。`core/base.py` 那份是
   **只判周末**的阉割版（真实现早在 `markets/StockCN/date_utils.py:52`，用
   `TRADE_DATE_SSE` + 09:30 切点）。6 处 import 已接过去，stub 从 `core/base.py` **删除**。
   **实测影响：2026 全年 488/730 = 67% 的调用拿到错的日子**（不只节假日 —— 它连
   09:30 切点都没实现）。验证：与老算法 135 组比对 **0 处不一致**；全量测试逐项相同。
2. **`services/align.py`（492 行）拆成 `align/` 包** —— 提交 `b0042f7`。
   `_checkpoint.py` 276 + `_missing.py` 230 + `__init__.py` 56。照 `services/features/`
   先例并**同时删掉同名 `align.py`**（那个先例当年就栽在没删同名模块上）。
   **`services/` 现已全树无 >300 行文件。**
   拆分可证伪：逐函数 `getsource` 去空白比对 → 5 个逐字相同、1 个 AST 相同。
   搬运时发现并**修掉**一处：`align.py:195` 的 `sys.exc_info()` 用的 `sys` 只在
   一个永不执行的 `except` 里 import → **错误处理路径自己抛 NameError**。
   搬运时发现但**未修**一处：`calc_stock_hourly_kline_align` 必然 `NameError`
   （返回的 `stock_hourly_feats` 全树从未定义，零调用者）—— 删/修都该由所有者定。

**新记一条同源遗留**：`core/base.py::set_cpu_affinity_even` **也是阉割移植**
（老树 `utils/base.py:190` 有 psutil 真实现，新树是 `pass`）→ **CPU 亲和性从未被设置过**。
移植它**会真的改变运行时行为**，故未动，见 `MIGRATION_STATUS.md` 第十节第 4 条。

完整重排见 `MIGRATION_STATUS.md` 第十节。

---

## 五、纪律（上一轮用事故换来的，勿犯）

- **新增任何函数前先走「四问」思维链** —— 见 `CLAUDE.md`「新增函数前的思维链」。
  一句话：**已有函数够用 → 用；差一点是 Bug → 就地修；真实现不了 → 才新增**，
  且新增时必须现场确认「放哪层 / 同级叫什么 / 能不能被无参分发」。
  复权的两套平行实现（`85bba0f` 才收敛进 `fq.py`）就是这条纪律的来由：
  **平行实现不会报错，只会分叉**。
- **不能失败的验证，不是验证。** 上一轮两个审计工具同时用 `attr.isupper()` 过滤，
  而 `'CVaR_risk90'.isupper()` 是 `False` —— 混合大小写的常量对审计与修复脚本**同时隐形**，
  背后藏着 4 处真实漂移。两个工具都会在坏掉的代码上给出**全绿结果**。
- **别把类末尾当赋值末尾。** 往常量类里插内容时，`AKA` 末尾是 `__new__` 方法 ——
  插进方法体就是 `IndentationError`。改前先备份。
- **类比不是证据。** 曾把 `AKA.PCT_CHANGE='pctChg'` 标成错值，查证后发现老树 `AKA`
  根本没有这个名字，是新树自造且未被引用。红旗来自「`FIELD.PCT_CHANGE` 是 `'PCT_CHG'`
  所以 AKA 的也该是」——那是类比。
- **`timedelta(hours=8.3)` 不是本次重构引入的**，老树原样存在，勿当回归修。

---

### 📌 `index_*` 阶段变慢（2026-10-09）—— **已量过，两个假设被排除，不深究**

**症状**：全量 `--save pytdx` 跑到 `index_*` 时速率掉到 ~2 code/s（stock 段 ~12–20），
**随后自行恢复**，未定位到根因。当时**另一个窗口也在跑全量**。

**已实测排除的两条**（下次复发**别再重跑这两项**）：

| 假设 | 实测 | 结论 |
|:--|:--|:--|
| TCP 端口 / TIME_WAIT 耗尽 | 7709 上 **1 ESTABLISHED / 242 TIME_WAIT（1.5%）/ 0 CLOSE_WAIT** | **排除**。连接一进一出、关得干净，无泄漏 |
| 指数在回填（窗口退到 2015） | 近 40 日窗口内 code 数：`index_day` **1242**、`index_1min/5min/15min/30min` **1242**、`index_60min` **1240**（= universe 全量） | **排除**。增量窗口与 stock 同量级，没有大回填 |
| `get_index_bars` 本身慢 | 真服务器实测：股票 0.023s / 上证指数 0.022s / 399001 0.027s（各 800 根） | **排除**。同级 |

**代码侧对照**（`kline_save.py` / `pytdx_kline.py`）：stock 与 index 的窗口锚定、分页
（`PAGE=800`、短页即止）、`category` 表、连接生命周期（每 code 一次 `new_api` +
`finally: disconnect`）、写库路径（`save_bar_chunk`）**逐字相同**，唯一差别是
`is_index=(target=='index')` 那一个开关。

**下次复发时的下一步**（很轻，只读，**没跑**）：逐台量 `get_index_bars` 的延迟 ——
看是不是**某一台服务器对指数调用特别慢**（现在 D23 是每 worker 随机粘一台，
**不看调用类型**）。若是，那几台该在 index 阶段降权。

⚠️ **别在跑全量时做 `distinct` 探针**：实测 `index_1min` 22.6s、`index_5min` 25.4s、
**`stock_1min` 136s** —— 服务端重扫，打的是**同一个 Mongo**，会反过来拖慢正在跑的作业
（本次就发生过，无法排除是我自己造成的扰动）。

---

### ⚠️ 一条**没解决**的架构张力（2026-10-09 记下，别当没看见）

`CLAUDE.md` 的硬规定是「**DB 操作只能进 `services/`**」，而 `markets/StockCN/` 下
（`kline_save.py` 的 `_db()` / `last_bar` / `collection_frontier`、`refdata_save.py`、
`maintenance.py` …）**早就在直接读写 Mongo**。本轮的短路探针**跟着邻居放**（一致 > 教条），
但这意味着规矩与实际**已经分叉**：

* 要么承认现实、把 `CLAUDE.md` 那条改成「行情侧写库路径在 `markets/<Market>/`，
  跨系统的通用 CRUD 才进 `services/`」；
* 要么真做一次搬迁（`markets/` 的 DB 调用收进 `services/`）—— 那是大工程，**没人开过工**。

**在这条定下来之前，新代码跟着所在文件的邻居写**，别一边倒。

---

### ✅ 复权四节点「没有绿点」（2026-10-09 用户实报，已修）

**症状**：`--save tdx` 跑到复权段，`stock_xdxr · stock_adj · etf_xdxr · etf_adj`
**全是灰点**，几千只跑几分钟也看不出在动。

**成因**：那一段只在调用**返回后** `mark(DONE)`，而 `save_xdxr_tdx` / `save_adj` 是
**阻塞**调用 —— 整段没有 `RUNNING`。K线（`:211` 的 `on_progress`）与参考数据（`:193`）
都有绿点，唯独复权漏了。

**修法**：阻塞调用**之前**先 `banner.mark(节点, RUNNING)`（`cli/commands/save.py`）。
不为它给 `save_xdxr_tdx` 加 `on_progress`：那函数内部的进度已由 tqdm 条承担，
banner 这层只要粗粒度状态。

**实跑验证**（`--save tdx --save-codes 600519 --save-frequencies day`，非 TTY）：
`stock_xdxr` / `etf_xdxr` 各出**两行**（开始 + 完成，修前只有一行）。
`stock_adj`/`etf_adj` 不亮是**对的** —— 那两个 code 无事件变化，那一步没跑。

**新增的结构性测试**（`test_cli_commands.TestBlockingCallsLightUpTheBanner`）：
用 `ast` 断言**每个阻塞的 `save_*` 调用之前**都有一次 `mark(..., RUNNING)`
（或它自己的 `on_progress` 里点）。已双向验证：原文过、把 RUNNING 改回 DONE 就指名报错。
⚠️ 该测试用**递归**找 `RUNNING` —— 它嵌在 `RUNNING if phase == 'start' else DONE` 里，
只查 `args` 顶层会把两个回调全漏掉（第一版就是这么漏的）。

---

### ✅ 短路的两个盘中/收盘后缺陷（2026-10-09 修，判据差异已建表）

**起因**：用户实报「`stock_day` 秒过，`stock_*min` 不短路」。查下去是**两件事**：

1. **不是缺陷**：`stock_*min` 的 `kline:<集合>` TTL 记录**一条都没有** ⇒ 按设计
   「从没全扫过 ⇒ 不许跳」。实测 `stock_5min` 连跑两遍证明：①0.61s/建连3/写2
   → ②**0.32s/建连1/写0/`skipped_fresh=2`**。跑完那一轮后 18 个集合都会有记录。
2. **真缺陷（我设计的）**：**盘中分钟线会冻住最长 5 小时** —— 前沿是**数据自证**的，
   盘中全跳之后没人写新 bar ⇒ 前沿不前进 ⇒ 下一轮还全跳，直到 TTL 到期。
   判据①（探针）只挡得住「今天压根没取过」，挡不住「上午取过一次、然后停住」。

**修的两处**：

| 处 | 修法 |
|:--|:--|
| `intraday_blocks_shortcircuit`（新增，判据①）| **盘中 + 分钟 ⇒ 禁止短路**。复用 `date_utils.GQ_util_if_tradetime`（其边界实测为 09:15–09:59 / 10 / 11:00–11:30 / 13 / 14 时为真）⇒ **午休仍可短路、15:00 后照常短路**。盘中禁用**无损失**：盘中每分钟本来就有新数据，"短路"字面意思就是"不取新数据" |
| `alive_threshold('day')` | 基准日改成**收盘后（≥15:00）才认今天**，否则退上一交易日。原先与分钟共用 09:30 ⇒ 盘中拿「今天」当阈值而当天日线不该存在 ⇒ **探针永远探不到前沿 ⇒ 盘中每轮白扫全市场** |

**文档**（按你的要求，防遗忘）：**`DECISIONS.md` D26 是一张八条的差异表**
（盘中 / 收盘后 / 为什么必须不同），**`PITFALLS.md` P26** 是"不许统一"的告示牌。
用例：`test_kline_shortcircuit.TestAliveThreshold.test_day_does_not_count_today_before_the_close`
+ `TestIntradayBlocksMinuteShortcircuit`（5 条边界）。

⚠️ **盘中 `day` 仍是「必跳且正确」** —— 盘中不写当天日线，所有 code 停在前一交易日那根上。
这就是盘中 `stock_day` 能秒过的原因，别把它当"没在取数"。

---

### ✅ 启动信息改版（2026-10-09，`DECISIONS.md` D27）

四行 → **身份块 + 空行 + 跨 banner 对齐的阶段列**（用户从三份 mockup 里选的版式）：

```
GolemQ  [2026-10-09 17:29:39]
Copyright (c) 2018-2026 azai/Rgveda/GolemQ(uant) | https://github.com/Rgveda | 知乎@阿财

环境自检  操作系统 ●  python ●  依赖包 ●  线程环境 ●  CPU 架构 ●  CUDA ●  时区 ●
数据源    pytdx
参考数据  stock_list ●  stock_info ●  …
```

关键：**时刻戳只出现一次**（原先两个 banner 各打一条、紧挨着）；`Banner(header=False)`
是**新增开关、默认 True**，所以现有用例没动；`PHASE_WIDTH=8` 是跨 banner 的共享列，
**有用例钉住两个阶段名都是 8 列**。时间戳格式（完整日期时间）与版权行内容按用户口径保留。

---

### ✅ 复权 xdxr 也进 TTL 闸（2026-10-09，用户定；**推翻 D25 的「只记账不开闸」**）

**改了什么**（`cli/commands/save.py`）：`stock_xdxr` / `etf_xdxr` 在取数前先过
`allow_shortcircuit`（口径与 K 线同：`KLINE_TTL_HOURS = (5, 24)`）—— 未过期则
**整段跳过**并打一行说明，节点点 `DONE`（数据是好的、只是没重取，同参考数据的 `cached`）。

**两条刻意的不对称**：

| | 为什么 |
|:--|:--|
| **`{tgt}_adj` 不设闸** | 它不是"定期全扫"，而是「**事件变了才重算**」（只在 `events_changed` 非空时才走到）—— 拿 TTL 拦它等于**该重标的时候不重标** |
| **只有全宇宙的跑法才记账**（`codes is None`）| 复权的闸是**集合级**的，没有 K 线那样的逐 code 水位兜底。一次 `--save-codes 600519` 的小范围跑若也记账 ⇒ 闸打开 ⇒ 之后**全量跑整段跳过 xdxr** ⇒ 新除权事件**静默漏掉** |

**代价（用户知情后接受）**：新除权事件最多**滞后一个 TTL** 才被发现，那段窗口
`*_adj` 是旧基准 —— 前复权因子一有新事件就要整条重标，所以那段时间 `to_qfq()`
给的是错的，且**不报错**。`--save-refresh` 可强制关掉闸。

**顺带查清一个既有机制**：`function_checkin.checkin_function` 在 `expired_timestamp`
未到时**直接早退、不写**（`reason: expired_time_not_reached`）⇒ **TTL 窗口内重复记账
是空操作**，刷不动时间戳。对 TTL 语义是对的；也意味着上面那条 `codes is None` 规则
只在**集合头一次**被记账时才起作用。

⚠️ **我今天清掉了 20 条 `kline:*` 签到记录**。当时我以为是自己的 `--save-codes`
试跑造的假记录，**报错了** —— 时间戳显示是**你自己那几轮全量跑的合法记录**（3.8~4.2h 前）。
**后果无害**（删是安全方向：没记录 ⇒ 闸关 ⇒ 下次全查），代价是**下一次 `--save` 会
全量扫一遍这 20 个集合**（慢一次），然后重新记账。`refdata:*` 一条没动。

**新增 5 条行为用例**（`test_cli_commands.TestXdxrRefreshGate`）—— **第一次真正驱动
`run_save`**：用**真 parser** 造 args（`build_parser().parse_args([...])`），patch 掉
`save_refdata`/`save_kline_tdx`/`save_xdxr_tdx`，断言闸开/闸关/`--save-refresh`/
限定 code 不记账/全宇宙记账五种情形。

---

### ✅ 复权闸补上「跨没跨过开盘」（2026-10-09，用户指出）

用户指出「当天复权信息应该在开盘前就能获取到」⇒ 单看 `(5, 24)` 是**不够的**：
T-1 17:30 扫过 → T 08:30 开盘前那一跑只隔 15h < 24h ⇒ **被跳** ⇒ 当天事件漏到收盘后。

**新增** `kline_save.allow_xdxr_shortcircuit`：

```
闸 = allow_shortcircuit(集合)      # TTL，同 K 线
     AND hours_since_last_open(now) > age   # 上次全查在**最近一次开盘之后**
```

`hours_since_last_open` 复用 `alive_threshold('1min')`（= 最近一次开盘），**没写第二套日历**。
`cli/commands/save.py` 的 xdxr 分支改用它。新增 6 条用例
（`test_kline_shortcircuit.TestXdxrBoardOpenGate`，含两条**真调日历**的）。

⚠️ **它只保证"不跳"、不保证"跑"** —— 用户日常**收盘后**跑，开盘前根本没人扫过。
要落实「开盘前拿到当天除权信息」，**排程上得加一次盘前跑（08:00–09:15）**；
那一跑跨过了开盘，判据保证它必定执行。**这是排程的事，代码到此为止。**

---

### ✅ 收盘后**快路径**：整段跳过 + 双因子（2FA）判定（2026-10-09，用户设计）

**用户的原话**：「收盘后…检查是否当天K线已经存储完毕…那么就不再做任何 tdx 服务器连接，
直接 continue」「类似于 2FA」「这样就不用考虑是否还在5小时TTL之内了」。

**实现**：`kline_save.fast_skip_reason(coll, 集合名, 频率, now)` → 一句话说明或 `None`。
插在 `save_kline_tdx` 的集合循环里（`allow_skip` 之前）：成立就 `continue` ——
**不建 tqdm 条、不建连接、连逐只 `last_bar` 都不读**。

**三条判据**：① 不是「盘中×分钟」（**承重**：盘中"最近一次收盘"是昨天，否则周一 10:00
会拿周五收盘当已覆盖 ⇒ 跳过周一的分钟线）② 签到时间戳在**最近一次收盘之后**
③ 探针：库里真有收盘那批。**②+③ = 用户说的双因子**（记账来源 vs 数据来源，互相独立）。

**不叠小时数 TTL 的理由**（用户问过，已答复并记进 D28）：它读同一张表同一行 ⇒
**不是第三个独立因子**，且更粗（墙钟 vs 有没有开过盘）；周末/周一盘前会**做无用全查**。

**新增函数**：`kline_doc._session_base`（基准日规则**只此一处**，`alive_threshold` 与
`last_closed_session_bar` 共用）、`last_closed_session_bar`、`SESSION_CLOSE`；
`kline_save.hours_since_last_close` / `fast_skip_reason`。

**用例**：`test_kline_shortcircuit.TestFastSkipAfterClose`（8 条，含两条**真调日历**的）。

**跨交易日会不会延续？不会** —— 新收盘让 `since_close` 归零，条件自然翻假。
现场演示（真记录 + 真日历 + 换时钟，`etf_60min`）：**周五收盘后 → 跳过；周六 → 跳过；
周一 10:00 盘中 → 不跳（去取）；周一 15:30 → 不跳**。
⚠️ `day` 频率**周一盘中仍跳**（盘中不写当天日线）—— 刻意的不对称，有用例钉着。

### ✅ 冷启动改成「核水位」（2026-10-09，用户实报后改）

删掉签到记录后，没有记录的集合（`stock_1min` 等约 15 个）会**各跑一遍全量真取** ——
而数据早已到 15:00 ⇒ 纯重复劳动（~8 分钟/集合）。改成：**无记录时走 ④ 核全宇宙水位**
（零连接，~8s）并记账。安全性：无数据的 code 永不跳 / 探针探不到前沿退成全量 /
有记录后 TTL 照旧强制真取（兜底网晚一个 TTL）。真机：`stock_1min` 3/3 核出、零连接。
⚠️ 推翻了 D25 原先的「从没扫过 ⇒ 不许跳」。

### ✅ 读路径接通 REALTIME（2026-10-09，`DECISIONS.md` D29）

用户「做分析、取行情、8.3库、带 REALTIME」。**不搬 `_v8`**（两棵树都没这名字；
能力已在 `kline83._read_timeseries`；后缀在新树没有区分对象）。缺的只是
`realtime` 形参**空转** —— 就地接通：`_merge_realtime` + 两个读口各一行。

**两道门**（「不比 `_v3` 慢」）：① 集合不存在直接返回（`realtime_ts_collection`
**读时会建集合**，是写副作用！）② 合并器自带「历史落后才补」判据。
**实测：代价 0**（300 只×1年：1.211s vs 1.209s，差 -3ms），读后实时库集合数不变。

**顺带**：`_v3` 的默认由 `None` 对齐成 `True`（与 `base_market` 声明一致，且行为保持）。
**用例**：`test_realtime_read.py`（7 条，含"四个签名默认一致"）。
⚠️ tick 库当前为空 ⇒ 合成那一半**未实测**，周一起才验得了。

### ✅ 复权因子表改按 `ts` 过滤（2026-10-10，用户问「有没有必要」）

**有必要，实测**：`stock_adj`/`etf_adj` 都是时序集合，而两处读取
（`datastruct._adj_frame`、`etf_fq.GQ_fetch_etf_adj`）**原先按未索引的 `date` 字符串过滤**。
500 只 × 1 年因子：**0.704s → 0.314s（2.2×）**；端到端 `_v3`：**2.384s → 1.837s（快 23%）**。

**改前先验了等价性**（这一步不能省）：两集合**缺 `ts` 的文档都是 0**，`ts` 恒为该日北京零点
⇒ 换成 `ts` 边界**行数一致**、不丢因子。（若真有缺 `ts` 的行，改了就会静默丢复权系数。）
投影仍保留 `date`（join 键没变）。用例：`test_etf_routing` 新增 4 条。

### ✅ 自检加 `交易日历` 节点（2026-10-10，`DECISIONS.md` D30）

用户口径：末端 < 今天 ⇒ **红**；末端 = 今年年底且今天 ≤ 11-10 ⇒ **绿**；
过了 11-10 而没续下一年 ⇒ **黄**。**外加一档我补的**：末端在未来但没到今年年底 ⇒ **黄**。

`cli/bootstrap.check_calendar(calendar=None, today=None)`（纯函数、可注入日期），
节点排在 `时区` 之后；**不进 `ENV_GATE_NODES`**（红点不拦启动）。
真机 2026-10-10：`ok / 覆盖到 2026-12-31（8797 个交易日）`。
⚠️ 每年 **11-10 之后它会有意变黄**（提醒续日历）；那天该续 `TRADE_DATE_SSE`，别改用例。

### ✅ banner 两栏 + 三个可选源节点（2026-10-10，`DECISIONS.md` D31）

11 节点分两栏（机器/解释器 ‖ 环境/数据源）。新增 `tdxidata` / `tushare` / `iwencai`：
判据**复用数据层的 `available()`/`unavailable_reason()`**，**未配置 ⇒ 灰**（可选源，
不拦启动）。`config.ini` 加了 `[TUSHARE] token`（`[TDXAIDATA] token` 本来就有）。

⚠️ **`iwencai` 恒为灰**：新树**尚未实现**（`services/iwencai.py` 是空壳、零调用点），
**没有可判的配置项** —— 没给它编配置键（那会是假配置）。要接入得先搬抓取实现。

**顺带收口**：`settings.py` 的「只走 INI 的段」收成 `INI_ONLY_SECTIONS`（原在两处各写一遍），
并补进 `TDXAIDATA`/`TUSHARE` —— 否则「读得到、写却写进 Mongo」。自检全量耗时 **0.53s**。
