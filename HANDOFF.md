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
| **D** | QUANTAXIS 完全解耦 | ✅ **完成** —— 只剩 `core/settings.py`（D9 定的暂留）|

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

### ✅ 实时行情 L1/L2（2026-09-21，提交 `f0f37be`）

| 项 | 状态 |
|:--|:--|
| `--sub l1_tencent` | **原本是坏的**（CLI 无参调用，而它的 `database_realtime` 是必填 → `TypeError` 被 CLI 的 except 吞成一行"发生错误"）。已补默认值 |
| `--sub l2_tencent` | **新增**。股票走腾讯（0.6s 一轮 / 4,936 只），ETF 走 MiniQMT；`sleep_time=3.0` |
| 落库 | 8.3 的 `golemq_stock_cn_realtime`，集合 `realtime_l1` / `realtime_l2`，**时间序列** `{timeField:'ts', metafield:'code', granularity:'seconds'}` |

**两条实测约束决定了写入方式（勿改回）**：MongoDB 8.3.2 的时间序列集合
**不支持唯一索引**、因而**不能 upsert**。旧写法（按日集合 + 唯一索引 + upsert）走不通，
现为 `insert_many` 追加 + 调用方按 `ts` 判新（`_write_ts_rows`）。

**实测（开盘时段）**：首轮 4,936 行，之后每轮约 6,200（腾讯 + 1,674 只 ETF）；
`600519` 样本 `ts=2026-09-21 01:35:07Z` ↔ `datetime=09:35:07` 北京，盘口
`bid1 1254.5×100 / ask1 1255.0×100` 为真值。

#### ⚠️ 本机 QMT **不提供盘口深度**（实测，非猜测）

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

#### ⚠️ 未做：读写指向了不同存储

三个读取器（`GQ_fetch_stock_realtime_adv` 与 `*_realtime_adv` 两个 kline 取数）
**仍读 `QAREALTIME.realtime_YYYY-MM-DD`**，而 L1 现在写 8.3 的新库 ——
**写进去的读不出来**。要么把读取器一起迁到新库（按 `ts` 区间查），要么改回旧存储。

#### ⚠️ 心跳互斥的健壮性问题

被 kill 的订阅器会留下 `status='running'` 而 `last_checkin=None` 的记录，
**该记录永不超时、永久占锁**，于是重启时报「只能运行一个实例」。
`--sub l1_tencent` 现在就被这样一条记录挡着（我没擅自清，怕你另有 L1 在跑）。

### ✅ `--save-x` / `--save-qmt`（2026-09-21）

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

**另收到一条更可靠的辅助判据（所有者经验，可选不强制）**：**成因 ③ 当天日线缺失**
—— 分钟有 bar 而日线没有该日 → 停牌；①② 当天日线都在。这是两个集合之间的交叉
验证，比形态猜测直接。**开销是多一次索引区间查询**（每读一次多一次，不是每根 bar
一次），故列为消费方自行 opt-in，**不写进读取器**。配方见 P8b。

⚠️ 该判据**本项目未能实测确认**：抽 20 只（2022~2026，1143 个交易日）只找到 1 个
日线缺口日（`600030`），那天分钟 bar 数为 0 —— 样本全是大盘股、几乎不停牌，
**没有检出力，属未证实而非被否证**。

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
