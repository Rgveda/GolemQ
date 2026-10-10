# GolemQ 迁移完整性报告

> 生成日期：2026-09-20
> 对照基线：`GolemQ_old/`（1308 文件 / 66 MB，重构前的老项目）
> 被审对象：`GolemQ/`（约 123 文件）
> 方法：按模块分片做**老 ↔ 新逐文件对比**。因为本仓库的 git 历史从重构后的基线提交才开始，`git diff` 无法呈现重构差异，必须直接比对两棵树。

**重申：本报告只回答一个问题 —— 哪些逻辑在重构中丢失、走样、或接线错误。** 它不评价代码风格。

> ## ⚠️ 状态标注（2026-09-25 全量复核）
>
> 本清单生成于 2026-09-20，此后**多条已被修复却未回标**，导致「已做的事被当成待做」。
> 2026-09-25 逐条对着代码复核了一遍，每条现在都带**状态**。
>
> **复核纪律（复核时踩过，务必照做）**：
>
> 1. **以代码实际行为为准，不要相信 docstring / 注释 / 本清单的旧描述。**
>    实例：`models/alias.py` 的类 docstring 至今写着 *"stub — to be populated"*，
>    而它的 `LTT` **已注册 15 个常量**。
> 2. **行号必然漂移**（各类后来补入了常量）。按**符号名**定位，别按行号。
> 3. **无法判定的就写「无法判定」**，并说明缺什么（本次多数需 MongoDB）。
>    一份「猜的」清单比没有清单更坏 —— 它会被当成依据。
>
> 状态取值：**已修** / **未修** / **部分修**（注明哪部分）/ **已过时**（描述与代码不符）/
> **无法判定**（需 DB 或环境）。
>
> **本次复核结论**：HIGH 11 项中**已修 6**（#3 #6 #7 #8 #9 #11）、**未修 5**（#1 #2 #4 #5 #10）；
> MEDIUM 10 项中**已修 2**（300 行规则、`features` 同名陷阱）、**部分修 2**（`timeseries`、`--sub`）、
> **未修 6**（`core/base`、越层直连、CLI 业务逻辑、估值源、`resample`、测试覆盖）；
> LOW 2 项均未修（但 LOW-1 影响确为低 —— 它**两边都是死代码**）。
> 另：本文件的**第一、二、九节**（总纲、因果链、对账）建立在已修的环节上，**整体过时**，
> 已就地更正；**第八节**把 `portfolio` 列为「已砍掉」，而它**已由 C1 建回**。

---

## 一、总体结论 —— ⚠️ **本节整体过时（2026-09-25）**

> 以下两段是 **2026-09-20 的报告结论**，保留作为历史记录。**它今天已不成立**。

> 这次重构建成了**模块骨架与接口**，但**计算层与数据层是 stub**。真实实现大多仍留在老树；少数已迁移的真实代码（如 `markets/StockCN/fetch.py`）**没有被 `services/` 接上**。

关键点：问题不是"功能没写"，而是**写了一半又接了根假线** —— 大量 stub 会静默返回空值/伪造字段名，让下游失败看起来像"数据本来就没有"。

**现状更正**：这句话的两根支柱都已拆掉 ——

| 支柱 | 当时 | 现在 |
|:--|:--|:--|
| `_StubMeta` 伪造字段名（#3）| 静默返回假名 | **已修**：未定义名一律抛 `AttributeError` |
| `fetch/kline.py` 是 stub（#6）| 返回空结果 | **已修**：132 行真门面（⛔ 该门面已于 2026-10-10 随 `GolemQ/fetch/` 整包删除），调度到市场实现 |
| 字段名漂移（#8）| 11 个常量是假名 | **已修**：11 个全部等于老值（实测 0 处不符）|

**但「接了根假线」这个模式本身仍在**，只是换了地方：`GolemQ/features/` 仍返回空
DataFrame 却被 `services/persistence/` 直接接线（#5）；`core/base.py` 的
`GQ_util_get_last_day` 仍是 stub 却被 5 处调用，而真实现就在同树的
`markets/StockCN/date_utils.py:52`。**这正是本次复核最该带走的一条：修完要回标，
否则下一个人会把力气花在已经做完的事情上。**

---

## 二、主失败因果链 —— ⚠️ **链条已断（2026-09-25）**

```
core/constants.py:34-37  _StubMeta.__getattr__ 伪造字段名（静默，不抛 AttributeError）
   → 查询 'field_maxfactor_major'，而历史数据写在 'MFT_MAJ' 下
   → fetch/kline.py 是 stub，kline baseline 为空
   → services/persistence/_daily.py:128  each_day[0] → IndexError
   → 持久化检查中止，ratio = 0
   → 完整性监控对全部标的报 0% 完整度
   → 真实的数据断流被这个假的 0% 掩盖，无人察觉
```

**逐环现状**（这是本文件里最该看的一张表）：

| 环 | 当时 | 现在 |
|:--|:--|:--|
| ① `_StubMeta` 伪造字段名 | 静默假名 | ✅ **已修**（#3）—— 现在抛 `AttributeError` |
| ② 查询 `field_maxfactor_major` 而数据在 `MFT_MAJ` | 漂移 | ✅ **已修**（#8）—— `FIELD.MAXFACTOR_MAJOR == 'MFT_MAJ'`，实测 11/11 等于老值 |
| ③ `fetch/kline.py` 是 stub | 空结果 | ✅ **已修**（#6）—— 132 行真门面（⛔ 该门面已于 2026-10-10 随 `GolemQ/fetch/` 整包删除） |
| ④ `_daily.py` `each_day[0]` → `IndexError` | 报错 | ⚠️ **仍在**，但**上游不再必然为空**了 |
| ⑤ ratio = 0 → 全标的报 0% 完整度 | 假 0% | ⚠️ **仍未闭** —— 现在的主因是 #5（`GolemQ/features/` 仍是 stub）|

**含义**：链条前三环修掉之后，**「完整性恒报 0%」这个症状的成因已经换了**。
再去排这条链，会排到已经修好的地方。

---

## 三、HIGH（11 项）

| # | 问题 | 位置（旧 → 新） | 性质 |
|:--|:--|:--|:--|
| 1 | **迁移入口整块消失** ⛔ **未修**（核心缺陷仍在，但**清单给的证据已失效**）—— 老项目有完整的分钟线迁移：timeseries 建表、8.3 读路径 `get_kline_price_min_v8`、污染数据周期边界过滤、`(code, month)` 断点续传、写前去重、北京时区处理。**「全树 grep `migrate`/`timeseries`/`timeField` 零命中」这句现在不对** —— 命中很多，但**无一等价**：① `--migrate-financial`（`cli/__main__.py:192,486`，迁 4.4 财务）；② `GQ_migrate_removed_from_44`（`maintenance.py:229`，迁 4.4 的**归档集合**）；③ `timeField`/`timeseries` 命中都是 8.3 时序集合的**规格与读路径**，不是迁移 | `GolemQ_old/scribe/min_migrate83.py`(345行) + `min_migrate83_cli.py`(207行) → **无对应物**；新树无 `scripts/`、无 `--migrate-min83` | 未迁移 |
| 2 | **QUANTAXIS 未剥离** ✅ **已修（2026-10-08，`DECISIONS.md` D12）** —— 全树 `grep -rn "^ *from QUANTAXIS\|^ *import QUANTAXIS" GolemQ/` **零命中**；`core/settings.py` 的 `_qa_handles()` 与三处惰性转发已删；5 个句柄去向：`DATABASE`→`GOLEMQ`（8.3 的 `golemq`）、`DATABASE_QA`/`QAREALTIME`/`QASETTING`/`DATABASE_ASYNC` **删**；`DATABASE_STOCK_CN*` → `GOLEMQ_STOCK_CN*`（名字即库名）。4.4 仅剩一条**一次性**通道 `core/migrate44.py` —— ⛔ **该通道亦已于 2026-10-10 删除**（`DECISIONS.md` **D33**），全树零 4.4 引用 | 见 D12/D13/D33；守卫 `test_cases/test_no_quantaxis.py` | ✅ 达成 |
| 3 | ~~**`_StubMeta` 伪造字段名**~~ **✅ 已修** —— 现对任何未定义属性一律抛 `AttributeError`（`PITFALLS.md` P1 闸门已翻）。实测 `class T(metaclass=_StubMeta)` 取未定义名 → 抛错，不再伪造 | `core/constants.py:34-37` | **新引入** → 已修 |
| 4 | **三个 benchmark 子类全部失效** ⛔ **未修**（复核确认）—— 悬空 import，`calculate()` 静默返回 None，`success_count=0` 与"确实没数据"无法区分。实测目标**根本不存在**：`GolemQ/models/mainstream.py` 不存在、`GolemQ/pipeline/compact.py` 不存在、`GolemQ/models/poolcoef.py` 存在但**无** `calc_stock_poolcoef_analysis`（唯一顶层 def 是 `calc_4Quad_push_credit`）。三个 import 都包在 `try/except` 里并置 `*_AVAILABLE=False` | 新 `pipeline/mainstream_benchmark.py:34-37`、`poolcoef_benchmark.py:34-38`、`compact_benchmark.py:39-43` | 未迁移 |
| 5 | **`features/` 用 stub 顶替真实 Mongo 读写** ⛔ **未修**（复核确认）—— ⚠️ **本条的"调用方"已随 D12 作废一半**：原文列的四个调用方 `services/persistence/_daily.py` / `_stock.py` / `_concept.py` / `_review.py` **整个包已删**（零调用者），故现在**没有任何活调用方**；剩下的 `GolemQ/features/` 是零调用者 stub。属「接线问题」而非「QUANTAXIS 遗留」，另案处置 | `features/empirical.py`、`features/reviews.py` | 未迁移（但已无调用方） |
| 6 | ~~**真实 kline 实现就在同一棵树里，`services/` 却接了 stub**~~ **✅ 已修** —— `fetch/kline.py` 现为 **132 行的真门面（⛔ 该门面已于 2026-10-10 随 `GolemQ/fetch/` 整包删除，取数改经市场实例 `get_active_market()`）**（`resolve_market` + `MARKET_TYPE_TO_MARKET` 调度到市场实现），不再返回空结果。`persistence/*` 仍 import 它，那是**设计如此**（门面 → `BaseMarket` → `markets/StockCN`） | 调用方 `persistence/_daily.py:54`、`_stock.py:52`、`_review.py:50`、`_concept.py:51` | **接线错误** → 已修 |
| 7 | ~~**`models/alias.py` 是空类**~~ **✅ 已修** —— `LTT` 现注册 **15 个常量**，含 `QUADRANT_LEVERAGE_MACD_TIMING_LAG_DUMMY = 'QLevMACDLagD'`（实测可取）。⚠️ 类 docstring 仍写着 "stub — to be populated"，**已过时**，别再据此判定它没实现 | 老 `models/alias.py:732` → 新 `models/alias.py:7` | 未迁移 → 已修 |
| 8 | ~~**模型字段常量与存储 schema 不再匹配**~~ **✅ 已修**（2026-09-25 实测复核）—— 3.8 表里的 **11 个常量全部等于「老值（数据实际存储名）」，0 处不符**：`FIELD.MAXFACTOR='MAXFACTOR'`、`FIELD.MAXFACTOR_MAJOR='MFT_MAJ'`、`AKA.STAGE='STAGE'`、`MAS.STAGE_MODE='STAGE_MOD'`、`MAS.BOOTSTRAP_STAGE_MODE_BEFORE='BST_STG_MOD_BF'`、`MAS.MACD_COMPOUDED_BAND_RATIO_MEDIAN='mMacdCpdBandRtoMed'`、`FEATURES.ZEN_PEAK_TIMING_LAG_MAJOR_REAL='ZPLagMajR'`、`RSK.CVaR_PEAK_PRICE/LOW/LOW_PRICE/LOW_BEFORE='ES_PEAK_P/LO/P/BF'`。**注：`_StubMeta` 已改为抛 `AttributeError`，所以「仍缺的常量」会是显式报错而非静默假名** | `core/constants.py`、`models/massive.py`、`models/risk.py` | **新引入** → 已修 |
| 9 | ~~**ETF 前复权模块整删**~~ **✅ 已于 2026-09-21 修复** —— 拉 ETF 日线（如 510300）跨除权日时，老代码返回**连续的前复权 OHLC**，新代码只打印"ETF 需要人工复权"就返回**不复权数据**，产生约 10% 假跳空，**污染 ETF 回测** | 老 `markets/StockCN/etf_fq.py`（260 行 / 3 函数，`GQ_is_etf` / `GQ_fetch_etf_adj` / `GQ_apply_etf_qfq`）→ 新 `markets/StockCN/fetch.py:1416-1423` 仅打印提示 | **逻辑丢失** → 已回迁 |
| 10 | **K线新鲜度告警消失** —— 数据源静默断流（pytdx/QMT 返回空包、QUANTAXIS 吞掉）不再触发任何告警 | 老 `supervisor/data_freshness.py` + `cli/__main__.py:1465-1469` 调 `notify_if_stale()` → 新 `cli/__main__.py:198-227` 无检查 | **逻辑丢失** |
| 11 | ~~**门面路径的股票前复权消失**~~ **✅ 已于 2026-09-21 修复**（见下方修复记录）—— 老树 `get_kline_price_v3` 在**股票分支**里调 `to_qfq()`（`:906`）**且**在指数分支调 `GQ_apply_etf_qfq`（`:1068`）；新树 `kline83` 只做了后者，**股票返回不复权价**。实测 `600519` 2024-01-02 新路径给 `1685.01`（= 库中原始值），老树给复权价 `1531.145598`。`services/persistence/*` 全树 grep `qfq`/`复权` **零命中**，即下游不会自行补 —— 所有 persistence 消费方拿到的都是原始价 | 老 `GolemQ_old/fetch/kline.py:906` → 新 `markets/StockCN/kline83.py` 的 `get_kline_price_v3`（只接了 ETF 那条）| **逻辑丢失** → 已修 |

### #11 的修复记录（2026-09-21）

`kline83` 新增 `_apply_adjustments(result, market, codelist, verbose)`，在
`get_kline_price_v3` 与 `get_kline_price_min` 两处各调一次：

- `market == 'stock'` → `datastruct.apply_qfq`（因子表 `stock_adj`）
- `market == 'index'` → `GQ_apply_etf_qfq`（因子表 `etf_adj`；真指数 no-op）

**互斥由 `market` 保证**：ETF 经 `market_prefix` 归类为 `'index'`，所以股票分支
碰不到 ETF 数据、反之亦然，**不存在二次复权**。老树两个函数同构
（日线 `:906`+`:1068`，分钟 `:1484`+`:1666`）。

验证（2026-09-21 实测）：

- **靶子是已知正确值**：`600519` 2024-01-02 门面现给 `1531.145598`，与老树一致
  （修复前是 `1685.01` = 库中原始值）。
- **门面日线 6 只 × 2020 至今日线 vs QUANTAXIS**：**9,640 行，最大差 0**。
- **门面分钟 2 只 vs QUANTAXIS 的分钟 `to_qfq()`**：36 行，最大差 0
  （顺带证实 QA 分钟那套兜底算法 `QA_data_stock_to_fq` 与 `stock_adj` 乘法数值一致）。
- **真指数**（`000300`）保持原始价、`if_fq` 不被设置。
- **`quotes.py` 路径未被二次复权**（它走 `GQ_fetch_stock_day_adv` + `to_qfq`，
  不过 `_apply_adjustments`）：与 QUANTAXIS 最大差 0。

### #9 的修复记录（2026-09-21，与 3.8 并列，不进编号）

`markets/StockCN/etf_fq.py` 已回迁（`GQ_is_etf` / `GQ_fetch_etf_adj` /
`GQ_apply_etf_qfq`），并在**每条产出 K 线的路径上各调用一次**：`kline83` 的
`get_kline_price_v3` / `get_kline_price_min`（门面路径）、`quotes.py` 的
`_apply_fq`（legacy 路径）、`fetch.py` 里那段打印提示的桩。

验证（2026-09-21，全部实测）：

- **独立重算**：`qfq == raw × adj(date)` 逐根 0 差（510300 日线 659 行、
  159119/159118 分钟线共 400 行）。
- **除权日连续性**：510300 三个除权日上，原始收益 −0.73% / −2.13% / −2.49%
  → 复权后 +1.27% / +0.29% / +0.10%；修正量与因子台阶精确吻合
  （`1.0127/0.9927 = 1.0201 = 0.950618/0.931873`）。
- **非 ETF 不受影响**：股票与真指数原样返回，`if_fq` 不被改写，一次 Mongo 都不查。

两条**移植时发现并已修**的事：

1. 老树 docstring 声称 `GQ_apply_etf_qfq` **幂等**，但**从未实现** —— 它置
   `if_fq='qfq'` 却从不读它，连调两次会把因子乘两遍（实测 510300 第二次/第一次
   = 0.931873，正是其除权因子）。新模块补上 `if_fq` 早退，使契约成真。
   因此**每条路径只能调用一次**，接线时已逐路径核对。
2. 该模块的 `DATABASE_QA.etf_adj` 与 8.3 的 `golemq_stock_cn.etf_adj`
   **逐行相同**（397,499 行 / 294 只），故取 8.3。

### ETF 独立成 `ETF_CN` / `etf_*`（2026-09-25，提交 `1cc47e2`）

**性质**：**既有设计缺陷的修正**，不是重构回归 —— 但它把一条**老约定**推翻了，
所以记为独立条目。

**老约定**：ETF 归 `MARKET_TYPE.INDEX_CN`，行情与真指数**共用** `index_*`。
依据是老树 `GolemQ_old/gateway/xtquant/save_qa.py:12,279,319` 的「与 QUANTAXIS 一致，
ETF 也进 `index_day`/`index_min`」；`symbol.py` 两处 ETF 分支至今留着
`# QA 把ETF归类为INDX_CN` 的注释。`MARKET_TYPE.ETF_CN` 常量**早已存在
（`core/constants.py:79`）却全树无人使用**。

**代价**：ETF 拿不到 `to_qfq()`（消费方得绕开类型系统单独调 `etf_fq`）；
`GQ_is_etf` 只能**嗅探描述串** `'ETF基金'` —— 措辞一改，ETF 复权全线静默停摆。

**新约定**：ETF 是一等类型 —— `is_stock_cn()` 返回 `ETF_CN`，行情走 `etf_*`，
容器 `GQ_DataStruct_ETF_day/_min` 带 `to_qfq()`（走 `etf_adj`），接口与股票一致。
`FUND_CN`（50x 封基/LOF/分级）**有意不动**，仍走 `index_*`。

**同批修正的号段误判**（联网核实交易所规则，见 `PITFALLS.md` P12）：
深市 `150`/`16x`/`180`/`20` 曾被一并判成「深交所ETF基金」（5 段里错 4 段）；
`82`/`820` 是**优先股**却被写成「北证A股」；`158` ETF 段缺失；
`200`（B股）分支被 `20` 抢先命中而**从未生效**。

**验证**：`is_stock_cn` 是纯函数，故扫**全部 100 万六位代码**与旧实现逐条比对
（`tools/dump_is_stock_cn_baseline.py`，9 秒）→ 92,000 条差异 / 7 类，逐类有意，
**零意外改动**。测试 65 → 93，失败项与基线逐项相同。

**⛔ 数据侧未做**（MongoDB 未起）：`index_*` → `etf_*` 的实际拆分。
**已定方案**：可逆拆分，复用 `maintenance._move`（它已封装「先写目标后删源 +
`(code,ts)` 唯一索引），只记「搬了哪些 code」而不做全量归档。
**⚠️ 拆分前必须先跑一致性核对**：`is_stock_cn()==ETF_CN` 的代码集 ⟷ `etf_list`
（1,674 只）双向比对，差异逐条定性后才动数据。

**两处连带的**路由**变更**（分类修正的必然后果，需查库确认有无数据）：
- `200`–`209`：`index_*` → `stock_*`（所有者定为「先查库再定」）
- `161`–`169` LOF、`184` 封基：`stock_*` → `index_*`（原本 `market_type=None`）

### 3.8 字段名漂移明细

| 常量 | 老值（数据实际存储名）| 新解析结果 | 位置 |
|:--|:--|:--|:--|
| `FIELD.MAXFACTOR` | `MAXFACTOR` | `field_maxfactor` | `core/constants.py:356` |
| `FIELD.MAXFACTOR_MAJOR` | `MFT_MAJ` | `field_maxfactor_major` | 同上 |
| `AKA.STAGE` | `STAGE` | `aka_stage` | `core/constants.py:242` |
| `MAS.STAGE_MODE` | `STAGE_MOD` | `stage_mode` | `models/massive.py:36` |
| `MAS.BOOTSTRAP_STAGE_MODE_BEFORE` | `BST_STG_MOD_BF` | `bootstrap_stage_mode_before` | `models/massive.py:37` |
| `MAS.MACD_COMPOUDED_BAND_RATIO_MEDIAN` | `mMacdCpdBandRtoMed` | `macd_compouded_band_ratio_median` | `models/massive.py:35` |
| `FEATURES.ZEN_PEAK_TIMING_LAG_MAJOR_REAL` | `ZPLagMajR` | `zen_peak_timing_lag_major_real` | `core/constants.py:412` |
| `RSK.CVaR_PEAK_PRICE` / `LOW` / `LOW_PRICE` / `LOW_BEFORE` | `ES_PEAK_P` / `ES_PEAK_LO` / `ES_PEAK_LO_P` / `ES_PEAK_LO_BF` | `cvar_peak_price` / … | `models/risk.py:18-21` |

**因为这些名字被当作 `peek_column` / `dropna` 子集 / 索引键使用，查询会命中从未写入过的字段。** 即便把 stub 全部补齐，只要字段名不改回来，历史数据依然读不到。

---

## 四、MEDIUM

| 问题 | 位置 | 后果 |
|:--|:--|:--|
| `analysis/timeseries.py` 被 stub（1399 → 18 行）🔶 **部分修** —— 文件现 112 行；`GQ_data_min_resample`（`:57-93`，含 A 股 9:30–11:30 / 13:00–15:00 分段重采样）与 `GQ_data_min_to_day`（`:96-111`）**已是真实现**；`Timeline_duration` **已于 2026-10-10 由 `analysis/timing.py` 提供真实现**（从旧树逐字搬、6 函数 × 500 样本对拍逐值一致）；`align_kline_timeline` **仍不存在**（它的调用方随 `services/persistence/` 一起没了） | `analysis/timeseries.py`；被 `persistence/_concept.py:55` 调用 | 两个重采样器已恢复；对齐两个仍是空操作 |
| `core/base.py` 被 stub（243 → 20 行）✅ **已修**（2026-09-25，提交 `e798362`）—— `core/base.py` 里的 `GQ_util_get_last_day` **只判周末**，真实现早在 `markets/StockCN/date_utils.py:52`。**6 处 import 已接过去**（`services/align.py`×2、`services/iwencai.py`、`persistence/_daily.py`、`_stock.py`），stub **已从 `core/base.py` 删除**（不转发 —— 它需要 A 股交易日历，属市场知识，放根层会让 `core/` 反向依赖 `markets/StockCN/`）。**实测影响：2026 全年逐日 488/730 = 67% 的调用拿到错的日子**（不只是节假日，它连 09:30 切点都没实现）。验证：与老算法 135 组比对 **0 处不一致** | `core/base.py` | checkpoint `FrozenExpired` 已拿到正确交易日 |
| ⚠️ 同源遗留：`core/base.py::set_cpu_affinity_even` **仍是 `pass`** —— 老树 `utils/base.py:190` 有真实现（psutil 判物理/逻辑核 + 超线程，按平台设亲和性）→ **CPU 亲和性从未被设置过**。**未修**：移植会真的改变运行时行为（多进程管线的 CPU 绑定），属行为变更，需所有者点头。唯一调用方 `pipeline/base.py:50` | `core/base.py` | 潜伏；已记入该文件 docstring |
| 越层直连 MongoDB ⛔ **未修**（复核确认，三处全在）| `core/settings.py:107`(`find_one`)、`:147`(`update_one`)；`cli/watchdog_manager.py` 整个文件仍直连（`find_one`:93,155、`insert_one`:109,169、`delete_one`:173,180、`create_index`:120,189、`find`:241,275）| 违反约定「所有 DB 操作必须在 `services/`」 |
| CLI 层业务逻辑 🔶 **部分修**（2026-10-09，`DECISIONS.md` D17）| 三块已各自落到命令模块：超时判定 + `_send_timeout_alert` → `cli/commands/heartbeat.py`；daemon 循环随 `--sub` 走 `cli/commands/subscribe.py`。⚠️ **但"打印/编排留在 CLI"这条仍在** —— D14 明确定过 banner 由 CLI 构造后注入，所以它不算纯粹的"只做校验" | 违反约定「CLI 只做参数校验再委托 `pipeline/`」；无一处委托 `pipeline/` |
| 估值源回退 xtquant → baostock ⛔ **未修，且比清单所述更彻底** —— 现在 `crawler.py` 里**完全没有 xtquant 估值路径**，直接只走 baostock（`:163`）；`adjustflag="3"`（**不复权**）在 `:230`。老树 `GolemQ_old/markets/StockCN/crawler.py:221` 用的是 `GQ_featch_stock_valuation_from_xtquant`（`dividend_type='front_ratio'` **前复权**）→ **xtquant 那条源被整个删了**，估值只剩不复权一个源 | `markets/StockCN/crawler.py:163,230` | 不复权 vs 前复权的语义差仍在 |
| `resample_features_frequency` 被 stub ⛔ **未修**（复核确认）| `markets/StockCN/base.py:8-13` 仍是占位（docstring 自述 *placeholder — currently returns `features` unchanged*），老 `StockCN/base.py:39-94` 的多重采样未回迁 | 目前无人调用（潜伏）；一旦启用，15/30min 对齐会静默拿到未重采样数据 |
| 测试覆盖损失 ⛔ **未修，两个子点均成立**（复核确认）| 老 `test_monitor.py`、`test_frequency_control.py` 在新树不存在；新 `test_heartbeat_fix.py` **既非 `unittest.TestCase`**（全文件只有一个普通函数 + `__main__`，discovery 发现不了）**又本身坏**（`:66` 调 `monitor._hash_instance_id`，而该方法只定义在 `HeartbeatModule`（`supervisor/heartbeat.py:447`），`HeartbeatMonitor` 没有 → `AttributeError`）| heartbeat / mutex / 限流 / watchdog 实质无测试 |
| `--sub` 命令行面收窄 🔶 **部分修** —— 清单的「只有 `l1_tencent` 一个键」**已过时**：现有 `l1_tencent`（`markets/StockCN/__init__.py:124`）+ `l2_tencent`（`:127`）**两个键**。但整体收窄结论仍成立：老树 8 种模式（`sina_l1`/`tencent`/`xtquant`/`huobi_realtime`/`huobi`/`okex`/`binance`/`tencent_1min`）**其余 6 种全无注册点** | 另：`--save-qmt` **原先存在**，已于 2026-10-09 **收口**进 `--save qmt`（`DECISIONS.md` D14；`--save-x` 同批删除，其 `etf_list` 能力**同日补回 `--save tdx`**）；`--migrate-min83` **不存在**（全树 grep 零命中，现有的是 `--migrate-financial`）| 6 种订阅模式 CLI 不可达 |
| 300 行文件规则被破 ✅ **已修，全树合规**（2026-09-25，提交 `b0042f7`）—— `services/features.py`（956 行巨石）已删；`services/align.py`（492 行）已拆成 `align/` 包（`_checkpoint.py` 276 + `_missing.py` 230 + `__init__.py` 56）。**全量复核：`services/` 下已无任何 > 300 行的文件**（其余最大 `persistence/_concept.py` 295）。拆分照 `services/features/` 先例并**同时删掉同名 `align.py`**（该先例当年就栽在没删同名模块上）| `services/align/` | 合规 |
| `features.py` / `features/` 同名陷阱 ✅ **已修** —— `services/features/__init__.py` 已建，导出**全部 11 个**函数，`__all__` 含清单担心的三个（`GQ_update_hourly_metadata`/`GQ_remove_hourly_metadata`/`GQ_move_hourly_metadata`，来自 `_hourly_crud.py`）；巨石 `features.py` 已删，无残留 | `services/features/`（8 个子模块）| **死包陷阱已解除** |

---

## 五、LOW

- `is_furture_cn` → `is_future_cn` 改名未留兼容别名 ⛔ **未修，但影响确为低**（复核确认）。
  现在定义在 `markets/StockCN/symbol.py:453`（清单写的 `:280` 已过时）。实测：
  新树 `is_furture_cn` **零出现**，且 **`is_future_cn` 本身也零调用点**（全树只有那个
  `def`）—— 也就是说它**两边都是死代码**，要不要留别名其实无所谓。
  老树 `symbol.py:269` 用旧名，被 `cli/review.py`、`imitation/jqdata.py` 等引用，
  但老树内部自洽、新树不 import 老树。
- `supervisor.messenger.send_alert`（Server酱 webhook 派发）无测试 ⛔ **未修**（复核确认）。
  函数存在（`supervisor/messenger.py:70` 的 `Messenger.send_alert`、`:225` 的模块级包装；
  Server酱派发在 `:142-158` 的 `_send_serverchan`）。
  ⚠️ **别被文件名骗了**：`test_cases/test_messenger.py` 测的是
  **`GolemQ.agents.messenger`（钉钉）**，与这里无关；唯一触及它的是
  `test_xtquant_sync_simple.py:81-88`，一个非 `unittest.TestCase` 的普通函数，**无断言**。

---

## 六、既有问题（**非本次重构引入**，勿误记为回归）

**本节 4 条复核后全部成立**（行号有漂移，已更正；另修正一处引用错误）。它们**不是重构回归**，勿计入缺陷总数。

| 问题 | 位置 | 说明 |
|:--|:--|:--|
| `timedelta(hours=8.3)` ⛔ **未修** | `services/persistence/_stock.py:123`、`_daily.py:126`（清单写 `:125`/`:128`，漂移 2 行）| 老树 `scribe/persistence.py:681,940` **逐字相同**。A股应为 UTC+8，`8.3` 可疑；老树自身在同角色上还用了 `hours=8.5`（`analysis/ChipDistribution.py:347`、`omnipath.py:2693`）→ **既有不一致**。⚠️ **清单里 `stack.py:259` 这条引用是错的** —— 实际那里是 `timedelta(hours=9.5)`，不是 8.5；核心结论（8.3 与 8.5 并存）不受影响 |
| `HeartbeatModule.mutex()` 非原子 ⛔ **未修**（复核确认）| `supervisor/heartbeat.py:459-493` | check-then-act，且唯一索引建在 `(module_name, instance_id)`（`:45-48`），而 `instance_id` 是 **per-process sha256**（`:442,447-457`；如 `sub_l1_from_tencent_{YYYYmmdd_HHMMSS}`）→ 两个进程各写各的 id，**索引永不冲突，永远拦不住第二个进程**。`heartbeat.py` 新旧逐字相同 —— CLAUDE.md 声称的 "timeout-based lock" **从未成立过**。**这才是真正的锁缺陷** |
| `min_interval_minutes=15` / `max_interval_minutes=1` 命名反了 ⛔ **未修** | `supervisor/scheduler.py:61-62` | 实际每 1 分钟跑一次（`:155` `schedule.every(self.max_interval_minutes).minutes`）。新旧一致 |
| `core/preprocessing.py` `normalize()` 中 `skp.StandardScaler` **NameError** ⛔ **未修** | `core/preprocessing.py:166`（`normalize` 定义在 `:160`）| import 区 `:26-29` 只有 `json`/`numpy`/`pandas`/`warnings`，全文无 `sklearn`/`skp` 导入 → **运行即 `NameError`**。新旧都有 |

### 附：复核时**被推翻**的一条既有记载

`HANDOFF.md` 曾记「被 kill 的订阅器留下 `status='running'` 且 `last_checkin=None` 的记录，
该记录**永不超时、永久占锁**」。**2026-09-25 复核：该机制不成立。**

- 字段名是 **`last_checkin_timestamp`**（不是 `last_checkin`；见 `heartbeat.py:52,101,148,237,467`）
- 写入侧（`start_module` `:101`、`checkin` `:148`）**永远写 int**，**从不写 `None`** → 前提不成立
- `mutex()` 用 `.get('last_checkin_timestamp', 0)`（`:467`）→ **键缺失 = 0 = 视为已超时**，
  不是永不超时；`_check_timeouts()` 的 `$lt` 查询（`:237`）同样会命中 null

**推论**：若真观察到「一条记录挡住重启」，成因**不在**这条机制上 ——
更可能是上面那条**非原子 check-then-act** 的竞态，或 `last_checkin_timestamp`
为 null 时 `mutex()` 抛 `TypeError`（`:471` 的 `None + int`）。
**要确认必须连库看那条记录的实际字段** —— 本次无 DB，标为**待复现**。

---

## 七、明确无问题（agent 逐字节比对确认）

| 范围 | 结论 |
|:--|:--|
| `pipeline/base.py` | 忠实迁移：停牌阈值 84 天、`days=960`、realtime 窗口 18h、pool 回退、统计口径**全部一致** |
| `gateway/xtquant/*` | 迁移且有扩展（持仓/委托同步、心跳、告警），未发现逻辑丢失 |
| `supervisor/scheduler.py` | 与老逐字节相同（仅 settings import 路径变）。午休 9:30-11:30 / 13:00-15:00、周末判断、Asia/Shanghai 时区、1 分钟轮询均未变 |
| `services/persistence/` 拆分本身 | 老 `scribe/persistence.py` 的 13 个函数在新 `persistence/__init__.py` 全部导出 —— **拆分没丢函数**（丢的逻辑来自它们调用的 stub）|
| `constants.py`、`tools.py`、`utils.py`、`base_market.py`、`StockHK/`、全部 `easyquotation/*` | ⚠️ **本行部分过时** —— `tools.py`/`utils.py`/`base_market.py`/`StockHK/`/`easyquotation/*` 仍成立；但 **`constants.py` 已不再「仅 CRLF 差异」**（实测新树独有 188 行 / 老树独有 31 行）。那是**解耦工作的有意改造**（`_StubMeta` 翻闸、常量值补真），不是漂移 |
| `date_utils.py` | ⚠️ **本行表述不足** —— 实测新树独有 74 行 / 老树独有 12 行，**不只是 import 路径改名**：`QA_util_*` 全部改名为 `GQ_util_*`。但那是**逐函数对齐过行为的等价替换**（解耦时 1600 组对比 **0 差异**），**无逻辑漂移** —— 结论对，理由需补 |
| `core/path.py`、`presentation.py`、`mongo.py`、`symbol.py` | 实质实现，非 stub |

---

## 八、被删除的模块：**新树无依赖，属有意砍掉**

`strategy` `portfolio` `fractal` `signal` `indices` `label` `markup` `imitation` `train` `czsc` `huobi`，以及老 `analysis/` 下的建模模块、`gateway/xtquant/` 的 `data_source.py`/`save_cli.py`/`save_qa.py`。

逐个核查：**新树没有任何地方依赖它们**（`models/poolcoef.py:83` 有一处 `from GolemQ.portfolio.base import PFL`，但包在 `try/except ImportError` 里，是死代码）。

**结论：这些是有意移除，不是漏迁。** 无需列为缺陷。

> ⚠️ **一条更正（2026-09-25 复核）**：名单里的 **`portfolio` 已不适用** ——
> `GolemQ/portfolio/` **已由 C1 重新建回**（`strategy.py` / `sizing.py` / `costs.py` /
> `rules.py` / `engine.py`，见 `HANDOFF.md` 的 C1/C2）。其余 10 个目录复核确认仍不存在。

---

## 九、与 `Project.md` 既定目标的对账

| Project.md 目标 | 现状 |
|:--|:--|
| 完全独立于 QUANTAXIS，所有新增代码重新实现 | ✅ **达成（2026-10-08，D12）** —— 全树 import 归零 + 运行时 `sys.modules` 无 `QUANTAXIS`；守卫 `test_no_quantaxis.py`。~~原评：基本达成（2026-09-25 复核更正）~~ —— 清单原写的引用数（`fetch.py` 48、`realtime.py` 25、`symbol.py` 10…）早已过时。实测：全树 `import QUANTAXIS` **0 处**、裸 `QA.` 活引用 0 |
| 所有 A 股接口在 `markets/StockCN/` 重写实现 | 🔶 **大部分达成**。真实实现已迁移到位；但 `services/` 仍有**两处接到 stub 而非同树的真实现**：`GolemQ/features/`（#5）与 `core/base.py::GQ_util_get_last_day`（MEDIUM）。这是同一类「接线错误」，且**成本都极低** |
| 只连 MongoDB 8.3+，4.4 仅一次性迁移源 | ❌ **未达成**。`DATABASE`/`DATABASE_QA`/`DATABASE_ASYNC` 三个符号**仍全挂 `QASETTING`**（`core/settings.py:274-279`）；`GQ_Setting.change()` 有重绑定能力但**零调用点**。迁移入口整块缺失（#1）|
| 旧库仅通过迁移脚本一次性读取 | ❌ **未达成**（但**部分已发生**）。分钟线**已经**由老树的脚本迁完（数据在 8.3 里），然而**新树没有那份脚本的能力** —— 要重迁（如 ETF 拆分后重灌分钟线）就得回老树跑，或补迁（#1）|
| Python 3.12+ / Pandas 3.0+ / PyMongo 4.18+ | 未审（本次范围外）|

---

## 十、建议修复顺序 —— **2026-09-25 重排**

原顺序的前两步**已经做完**（`kline` 接线、`_StubMeta` 抛错），故重排。判据仍是
**成本 / 收益**，且优先挑「真实现已在同树、只是没接上」这一类 —— 改动小、可立刻验证、
且修完能让被掩盖的问题显形。

| 顺序 | 事项 | 为什么排这里 |
|:--|:--|:--|
| ~~1~~ | ~~修 kline 接线错误（#6）~~ | ✅ **已完成** |
| ~~2~~ | ~~`_StubMeta` 改抛 `AttributeError`（#3）~~ | ✅ **已完成** |
| ~~3~~ | ~~字段名回滚或补迁移（#8）~~ | ✅ **已完成**（11 个常量实测全部等于老值）|
| ~~**1**~~ | ~~`core/base.py::GQ_util_get_last_day` 接到真实现~~ | ✅ **已完成**（`e798362`）|
| ~~**5**~~ | ~~`services/align.py` 拆到 300 行以下~~ | ✅ **已完成**（`b0042f7`）—— `services/` 现已全树合规 |
| **1** | **`GolemQ/features/` 的 stub 接真实现**（#5）| 与上面那条同一类接线错误，但**真实现要回迁**（老 `empirical.py` 4614 行 / `reviews.py` 3006 行），成本高一些。它现在是「完整性监控恒报缺失」的**主因** |
| **2** | **K线新鲜度告警**（#10）| 老树 `supervisor/data_freshness.py` 147 行整体可回迁；**数据源静默断流是无人察觉的**，收益高 |
| **3** | **benchmark 三个子类**（#4）| 目标模块根本不存在，要连 `models/mainstream.py` 等一起补 —— 成本更高 |
| **4** | **`core/base.py::set_cpu_affinity_even`**（新发现）| 同一类阉割移植，老树真实现在 `utils/base.py:190`。**属行为变更**（多进程管线的 CPU 绑定），需先确认要不要 |
| **5** | **迁移入口**（#1）与 **QUANTAXIS 运行时绑定**（#2）| 架构级决定，牵涉 CLAUDE.md 与 Project.md 的冲突，**需要先定方向**（#2 的 import 已剥到 1 处，只差把 `DATABASE` 改由 `GQ_Setting` 绑 —— 但那是行为变更）|

**不排在修复序列里的（本节复核确认它们不是重构回归，各有专门记载）**：
`timedelta(hours=8.3)`、`HeartbeatModule.mutex()` 非原子、`min/max_interval_minutes` 命名反、
`preprocessing.normalize` 的 `skp` NameError（§六）；~~`realtime` 读写指向不一致~~（**2026-10-08 已修**：
读写都切到 8.3 的按日时间序列集合，见 `DECISIONS.md` D10 与 `PITFALLS.md` P13/P14）、
`GQ_fix_daily_metadata` 的 lambda、`_review.py` 的 format 占位符（§六附，见 `PITFALLS.md` 与 `HANDOFF.md`）。

---

## 附：本报告未覆盖的范围

- 未做运行验证（无 MongoDB / RabbitMQ 实例，且 stub 会让多数路径提前失败）
- 未审 `cookbooks/`、`.claude/`、构建配置
- 未逐个核对 Python 3.12 / Pandas 3.0 API 兼容性
