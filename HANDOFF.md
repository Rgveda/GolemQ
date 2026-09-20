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
| **C1** | `portfolio/` 骨架（strategy/sizing/costs/rules）| ✅ 提交 `f993d31` |
| **C2** | `zen_bt.py` 撮合按契约移植进 `engine.py` | ⛔ **阻塞在你**：需先写策略实现 |
| **D** | QUANTAXIS 完全解耦 | **进行中** —— 机械部分已完成，见下 |

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

## 三、D 的剩余 11 处 —— 唯一在飞行中的工作

```
core/settings.py:34               ← 根（QA_Setting）。D9 已定：数据不搬，暂留
markets/StockCN/fetch.py:66,73,77 ← QA_DataStruct_* / QAQuery_Advance / QAQuery
markets/StockCN/quotes.py:28      ← QA_fetch_stock_day_adv
markets/StockCN/realtime.py:68    ← resample 栈（QA_data_min_resample 等）
markets/StockCN/align.py:65       ← 裸 import QUANTAXIS as QA
markets/StockCN/crawler.py:51     ← 同上
pipeline/base.py:49               ← 同上（QA_AVAILABLE 仍是活逻辑，勿拆）
pipeline/compact_benchmark.py:35  ← 同上
services/align.py:34              ← 同上
```

**这不是改 import，是设计工作**（`RESTRUCTURE_PLAN.md` 列为步骤 4）：需要
`QA_DataStruct_*` 的替身。

### 替身的调用面（已量出，比预想小）

`QA_DataStruct_*` **只有 `markets/StockCN/fetch.py` 在用**：

- 4 个 import（`:66-70`：`Index_min` / `Index_day` / `Stock_day` / `Stock_min`）
- `isinstance` 用来**区分股票与指数**（`:1116`、`:1449`）→ **替身必须有类层级，不能是单个类**
- 构造 1 处（`:714`）
- `.to_qfq()` 6 处（`:888`、`:1283` 为活代码，`:1035`、`:1422` 已注释）
- `.select_code()` 4 处

外加 `stock_block` 的 `.block_name` / `.get_block()` / `.get_blocklist()`。

### `to_qfq()` 的替身有现成依据（本轮实测）

| 事实 | 证据 |
|:--|:--|
| `stock_adj` 是**预计算的前复权乘法因子表** | 字段 `{code, date, adj, ts}`；`600519` 最新日期 `adj=1.0`，越早越小（2001 年 `0.1312`）→ 乘上去最新价不变 |
| `stock_day` 与 `stock_adj` **1:1 对齐** | `600519` 两边各 6006 行 |
| 老树有 QUANTAXIS-free 的复权实现可照搬 | `GolemQ_old/markets/StockCN/etf_fq.py` 的 `GQ_apply_etf_qfq`，**行为契约完整**（只处理 ETF；只乘 OHLC；真指数与股票原样返回且一次 Mongo 都不查）|

**未完成**：`raw_close × adj` 与 QUANTAXIS `to_qfq()` 的逐值比对**尚未跑通**
（上一轮因上面那个日期类型问题拿到空集）。这是写 shim 前的**最后一道验证**，
必须逐值一致才算数 —— 不能只看几个样例。

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
