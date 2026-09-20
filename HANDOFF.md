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
| **D** | QUANTAXIS 完全解耦 | **进行中** —— 替身已完成，import **11 → 7**，见第三节 |

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

## 三、D —— 替身已完成，剩 7 处 import

提交 `7fcb5ec`：`QA_DataStruct_*` 替身 + 日线读取器 + block/list 本地化已完成。

### QUANTAXIS import：11 → 7

| 文件:行 | 内容 |
|:--|:--|
| `core/settings.py:34` | 根（QA_Setting）。**D9 已定：数据不搬，暂留** |
| `markets/StockCN/align.py:65` | 裸 `import QUANTAXIS as QA` |
| `markets/StockCN/crawler.py:51` | 同上 |
| `markets/StockCN/realtime.py:68` | resample 栈（`QA_data_min_resample` 等）|
| `pipeline/base.py:49` | 同上（`QA_AVAILABLE` 仍是活逻辑，**勿拆 try**）|
| `pipeline/compact_benchmark.py:35` | 同上 |
| `services/align.py:34` | 同上 |

另有 **26 处裸 `QA.` 前缀引用**分布在上述模块里。

### ⚠️ 未修的已知缺陷：`fetch.py` 的 `QA` 是未定义名

老树 `GolemQ_old/fetch/kline.py:31` 在 `try` 里写了 `import QUANTAXIS as QA`，
**新树 port 把 import 丢了、26 处 `QA.` 用法全留着**。今天**不可达**
（StockCN 绑定的是 `kline83` 的实现，`align.py` 只走 STOCK_CN 分支），
但一执行就是 `NameError`。**这是重构回归，不是老树缺陷。**

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

### 仍未做

- 同类的「绑错库」还有 `fetch.py` 的 `GQ_fetch_stock_list_day`
  （`collections=DATABASE.stock_day` 默认值）、`scribe.py` 的
  `QA_fetch_stock_list/index_list/stock_terminated` 默认值
  （`stock_list` 只在 8.3 有；`index_list`/`stock_terminated` **两个库都没有**）。

---

## 四、待你决定

| # | 事项 | 选项 |
|:--|:--|:--|
| 1 | **`financial` 集合（当前 0 行）** | **(a)** 暂缓（上一轮倾向此项：无消费方 + akshare 全量 46.5h + tdxaidata 字段目录未知）；**(b)** 你给 tdxaidata 字段名，走批量通道几分钟灌完；**(c)** 限定代码范围跑 akshare |
| 2 | **C2 撮合移植** | 需你先给策略实现（`XGB_ECHO_TIMING_LAG` + `QUADRANT_LEVERAGE_MACD_TIMING_LAG_DUMMY`）—— 信号公式属策略侧，不猜 |
| 3 | 未被引用但值错的常量 | 只有**全量审计**能闭合（两个工具都只看被引用的）。只读检查，成本低 |

---

## 五、纪律（上一轮用事故换来的，勿犯）

- **不能失败的验证，不是验证。** 上一轮两个审计工具同时用 `attr.isupper()` 过滤，
  而 `'CVaR_risk90'.isupper()` 是 `False` —— 混合大小写的常量对审计与修复脚本**同时隐形**，
  背后藏着 4 处真实漂移。两个工具都会在坏掉的代码上给出**全绿结果**。
- **别把类末尾当赋值末尾。** 往常量类里插内容时，`AKA` 末尾是 `__new__` 方法 ——
  插进方法体就是 `IndentationError`。改前先备份。
- **类比不是证据。** 曾把 `AKA.PCT_CHANGE='pctChg'` 标成错值，查证后发现老树 `AKA`
  根本没有这个名字，是新树自造且未被引用。红旗来自「`FIELD.PCT_CHANGE` 是 `'PCT_CHG'`
  所以 AKA 的也该是」——那是类比。
- **`timedelta(hours=8.3)` 不是本次重构引入的**，老树原样存在，勿当回归修。
