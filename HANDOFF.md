# 交接：当前进度与下一步

> **本文件只管「现在在哪、下一步是什么」。**
> 「每轮做了什么、为什么」在 [`docs/worklog/`](docs/worklog/2026-10.md)；
> 架构决策的索引是 [`DECISIONS.md`](DECISIONS.md)，正文在 [`docs/decisions/`](docs/decisions/)。
>
> 唯一的纪律：**只写已验证的事实**。没验证过的宁可留空 ——
> 一份「猜的」文档比没有文档更坏，因为它会被当成依据。

---

## 一、接手先读

**阅读顺序的唯一出处是 [`CLAUDE.md`](CLAUDE.md) 的「接手本项目，按此顺序读」** ——
不在这里复制一份（两份必然分叉）。本文件排第 0 位：先看下面三节。

⚠️ **2026-10-15 文档结构变了**（单文件只增不减，到 2,200 行就不好用了）：

| 原来 | 现在 |
|:--|:--|
| `DECISIONS.md` 2,200 行 / 46 条 | `DECISIONS.md` **81 行索引** + `docs/decisions/D1..D46.md`（一份一文件） |
| `HANDOFF.md` 2,400 行 | 本文件（活文档）+ `docs/worklog/2026-10.md`（历史） |
| — | `tools/gen_docs_index.py` 生成索引（`--check` 可在提交前验漏跑） |

**编号 `D<n>` 不变** —— 代码注释里那句「见 `DECISIONS.md` D44」照旧可查（索引里有链接）。

---

## 二、未完成

### Review 管线搬迁（`DECISIONS.md` **D54**）—— 🚧 进行中

老树 `--review --daily` + `zen_minor` + coldstart 子集 → 一套逐 Bar 因果管线。
（新树入口：`--strategy review` 或别名 `--review`；**`--daily` 已删除**，见 D58。）
已落地：`pipeline/base.py`（BasePipeline/BaseBenchmark）、`pipeline/steps.py`、
`pipeline/review/{pipeline,contract,runtime,coldstart}.py`、`cli/commands/review.py`、
`analysis/zen.py`（**czsc 0.7.10**，KlineAnalyze 0.53 已剔除）、
`tools/{dump,compare}_review_golden.py`、9 MB 4.4 金标准 fixtures。
实测：日族 Poly9/RegTree/M9T 逐值一致、zen ±1；**真实写库已验证**（stock day/60/15、
ETF day 路由、水位幂等、目标行过滤）；coldstart 最小集 + Prophet **已并入 60min 表**
（无独立 `*_coldstart_reviews`）；Review 精简到 5 个 step；
3 个死 benchmark 壳与 `BaseBenchmark` 已删 —— `pipeline/` 只剩 `BasePipeline` + `review/`；
四量图的 `QuadSt`/`ChoBits`/`QuadCodex` 状态机已删（**D66**：零消费方，`--review`
每码省约 28 秒）；
落库/水位已改为 **Active Market 取 DB 实例 + 参数注入**；持久化检查见
`pipeline/review/persistence.py`（按目标表已有 review 行选缺失 bar，不看侧表水位；
目标下界 day 2560 / min 800 天）。878 条测试全绿。

**下一步**：全量对拍分类（`compare_review_golden.py`）、coldstart `m9tBsl`
（需 compact 特征链）、`--eval fast/etf/index/all` 回填；internal 入口的 `--review-runtime` 端到端冒烟。
⚠️ `--eval` 是**全局唯一**的 scope 选项（`eval_scope`），与 `--review` / `--scan` /
`--strategy` 组合；**不要再造** `--review-eval` 这类前缀变体（D58）。
⚠️ `--daily` **已删除**（语义不明）；`--review` == `--strategy review`。

#### Review 列覆盖度（2026-10-27 实测，见 D71/D72）

| 列 | 覆盖 | 结论 / 下一步 |
|:--|--:|:--|
| `stage` / `garPD` | 99.7% | 不用重跑；缺的多为短历史 / 定带失败 |
| `poly9MajD` / `poly9WekD` | 100% | — |
| `rTrePrcMajD` / `rTreSlopMajD` | 93% | 缺的是 `<450` 根 / 拟合失败，**历史不足补不了** |
| `prophetLbD` 等 | **42%** | 需专门「只补 Prophet」的重型工具，或接受现状（**待你决定**） |
| `se`（**day 表日频**） | **99.995%** | D74：2,935,066 / 2,935,202；缺 136 行 = 27 只标的上市初期暖机 |
| `ZenPivotCnt`（**day 表 15:00 采样**） | **99.94%** | D74：2,933,407 / 2,935,202；缺 1,795 行 = 2 只北交所 920xxx（60min 数据 2025-09 才起）。⚠️ 全历史重算使计数变大（000001 8→47） |
| `renkoSpriceD` 等 14 列 | **100%** | `tools/backfill_renko.py`（D72）已补齐 2,935,202 行 / 1,883 码，14 列缺失 0 |

### 指标搬迁（`DECISIONS.md` D49 / **D50**）—— ✅ **四块全部收口**（③ 的编排按你的决定挂账）

| 步 | 内容 | 状态 |
|:--|:--|:--|
| 1 | 常量补齐 11 个 + 修活雷 | ✅ D49 第一节 |
| 2a | `analysis/stages.py`（Garident STAGE）| ✅ D49 第二节 |
| 2b | `analysis/timeseries.rolling_poly9_v2` | ✅ D49 第六节 |
| 2e | `analysis/magic_nine_jit.py`（**只搬核**；原 `magic_nine.py` 已并入删除）| ✅ D49 第五节 |
| **2c** | `mapower_np_vXIIs` / `inv_nums` → `quadrant_jit.py`（原 `quadrant.py` 已并入删除） | ✅ **D49 第九节**（12 条对拍+jit 一致）|
| **2d** | `TA_HMA` → `analysis/indicators.py`（自实现，QUANTAXIS 已剔除）。⚠️ **`CROSS`/MACD 没搬** —— 无消费方（M9T 挂账），照「只搬用到的」| ✅ **D49 第九节**（7 条）|
| **2f** | `analysis/poly9.py`：`calc_polynomial9_vXIs` + `calc_polynomial9_norm_vXIs`。⚠️ **端到端对拍同时是 `TA_HMA` 的真验证**（它没法单独与 QUANTAXIS 对拍）| ✅ **D49 第十节**（9 条对拍）|
| **2g** | ~~`POLY9_REGTREE_DIVERGENCE`~~ → **整列已删**（连同相邻同型的 `ATR_LB_REGTREE_DIVERGENCE`）。唯一消费方是老树 `calc_Empirical_AMIRA_fractal`，**该函数已废弃、不搬** | ✅ **D49 第八节** |
| **第 4 步** | 两表合并：`stock_reality_feat_day` ← `stock_reality_feat_day`（**17,133,065 行**）| ✅ **搬完并核对通过**（D50）：读/写 **17,133,065 / 17,133,065**、丢弃 0；抽样 20,000 行 **缺 0 / 不符 0**；写口全切。**旧表已 drop**（删前又抽 500 条核过）|
| **第 5 步** | ~~解掉 `features/` 两个 stub~~ → **整个 `GolemQ/features/` 包已删**（5 个函数零调用方；2 个新树已有等价物、3 个数据源在 8.3 不存在）| ✅ **D50** |

~~另有一条待办：`BOTTOM_PRICE` / `ROOF_PRICE` 同病未治~~ —— **2026-10-28 已更正**：
它们是**绝对价**，链路 `garPD`（比值）→ 还原 `garP`（绝对）→ 取
`btmP`/`roofP`（绝对），**不落库、不归一化**。常量已补、attach 已接线
（**D76**）。

### CV 存量搬运（`docs/decisions/D51.md`）—— ✅ **搬完并核对通过**

| 项 | 状态 |
|:--|:--|
| 核查 4.4 六表 | ✅ 都有 CV 列（⚠️ 我先用 `find_one` 单条误判成"没有"，用户提示后 `$sample` 测对）；`Cancelckpt*` 实测**从未写过**；`time_stamp` 两边同为真实 UTC |
| `etf_` 前缀分支 | ✅ `reality_feat_name` 补上（K 线侧 2026-09 就拆了，CV 侧没跟上 ⇒ 早已分叉） |
| 迁移函数 | ✅ `migrate_reality_feat_from44` —— 逐码路由、只搬 CV 列、**复用 `upsert_fields`**（用户明确定：「迁移用 Upsert，写入函数共用」） |
| 执行 | ✅ **16,287,825 / 16,287,825，丢弃 0**（首轮，约 44 分钟）<br>🔁 **2026-10-18 二次补搬**（上卷列 + `ChartScaned` 标记）：**19,816,524 行，exit 0** |
| ⚠️ 首轮口径缺陷（已修，D51 §五之三） | ① 日族只搬 `ClsMaj` 那套，丢了 `ClsMin`(511万)/`Class`(487万)；② 只有 `ChartScaned` 的标记行(312万)被过滤；③ 60min 族没带上卷来的 `ClsMin`(4.4 抽样 916/3000)。**后果：`all_nat_min` 恒真 ⇒ 每只全扫 1008 天**。实测 mode=1：4.4 **0/40** → 修复前 8.3 **36/40** → 修复后 **0/40** |
| 二次核对 | ✅ internal 入口的 `--migrate-reality-verify` 抽 **120,000 行 → 缺 0 / 不符 0**；`mode=1` 抽样 **0/40** |
| 核对 | ✅ 抽样 120,000 行 → **缺 0 / 不符 0**；**两侧独立全扫**对账 16,287,825 vs 16,287,827（**+2 = D42 的两行冒烟数据**），九个集合逐集合精确计数**全 0 差** |
| **ChartScan 写库路由** | ✅ **揪出 4 处分叉并修**（见下） |
| **CLI 入口** | ✅ `--scan ChartScan` / `--server ChartScan` 搬进本仓库（`cli/commands/chartscan.py`）。**自动切到 GolemAI env**（子进程 re-exec），**围栏不加豁免**（D51 五之二）。18 条用例 |
| ✅ **GolemAI env 现状**（2026-10-18 **已收口**）| conda 的 `opencv 4.10.0` **已移除**（它自身坏了：`import cv2` → `DLL load failed`，`Library/bin` 里 `videoio/highgui/gapi/ccalib/superres/videostab` 6 颗加载不了；哈希与 conda 缓存一致、PE 依赖全可解析，**原因未查明**）。现在装的是 **pip `opencv-python 5.0.0.93`，在 env 内**（不是用户目录），遗留 `cv2.cp312-win_amd64.pyd` 与用户目录那份都已清掉。实测 `PYTHONNOUSERSITE=1` 下 `import cv2` 正常 ⇒ **单一来源**。`pip check` 对 ultralytics / opencv / joblib **全满足**。|
| **端到端** | ✅ 真跑通：GolemQ env 敲 `--server ChartScan` → 自动切 GolemAI → YOLO 推理 → 写 8.3 三族。日族六列 CV 全齐，`datetime` 为字符串，`date_stamp`/`time_stamp` 值正确 |
| **渲染后端** | ✅ **2026-10-18 真机踩到并修**：多线程服务端 + matplotlib **TkAgg** ⇒ `RuntimeError: main thread is not in main loop`，且被 `predict` 的 `except` 吞掉 ⇒ **每个标的都静默失败、一行都写不进库**。`render.py` 已钉 `MPLBACKEND=Agg`。见 **P36**。多端口（2 线程）实测：报错 0 次，三族正常落库 |
| **演示 app** | ✅ `examples/app_chartscan.py`（Streamlit，参考 `ChartScanAI/app.py`，取数换 8.3）。**必须跑在 GolemAI env**（torch/ultralytics + streamlit 都在那儿）。`AppTest` 实测：生成图 → YOLO `3 Buys, 3 Sells` → 像素↔bar 映射 → 库内检出对照，**零异常** |
| **pandas 3 / numpy 2 适配** | ✅ **D52**：4 处（µs 精度、`Series[int]`、`float(1×1 matrix)`、日族 `datetime` 类型分裂）。全量 **774 条绿** |
| **EPOCH-DIVISOR** | ✅ datetime → 秒的系数收敛到 `constants.epoch_divisor(dtype)`。⚠️ **不能用一个全局标量** —— pandas 3 的精度**按输入而定**（`date` 对象 ⇒ `datetime64[s]`，字符串/Series ⇒ `[us]`），实测单标量把 `date_stamp` 算成 `1767`。守卫扫全树 token，已验会红。见 **D52** / **P35** |

**⚠️ 写库路由 4 处分叉（D51 第四节，本次最大发现）**：D42 承诺「行为逐字保留」，
但它的对拍**只覆盖 `freq='day'`**。15min/60min 两条支路在合并时被改写 ——
最重的一条：旧树三频率**都写日族**（`:342`/`:363`/`:406`），新树改成「按 freq 选族」
⇒ 日族永远收不到 `CoordinatesMin`/`Coordinates` ⇒ `all_nat_min` 恒真 ⇒
**每天全量重扫 618 交易日 × 3 频率**。已修 + 补 `test_chartscan_predict.py`（14 条）；
**回退修复会 6 失败 + 2 错误**，用例确实咬得住。

### ChartScan 盘中扫描 no-op（2026-10-21 查明并修复，**P41**）

**现象**：服务端刷
`without_processing code: 920183 True True True [Timestamp('2026-10-21 00:00:00')]`，
盘中 60min / 15min 的 `*_reality_feat_*` 当天一行都不更新。

**根因**（已用 600519 / 000001 / 920183 实测）：
1. 盘中客户端在 `calc_ChartScaned_persistence` 里把 `recent_list` **强制成
   `[今天]`** ⇒ 已扫完的码拿到 `mode=2`、`start=end=今天` 的单日任务。
2. 服务端 `_fetch('day', 今天, 今天, ...)` 走 `get_kline_price_v3(realtime=False)`：
   `stock_day` 里今天那根要等**收盘后 `--save`** 才有 ⇒ 历史查询为空 ⇒
   `get_kline_price_v3` **在实时合并之前**就返回 `None` ⇒ `kline_baseline is None`
   ⇒ 三个频率的循环全部跳过 ⇒ 打印 `without_processing`。
3. 旧树 `calc_ChartScanAI_reality` 有**独立的 `GQ_fetch_stock_day_realtime`
   补数**（`ohlc_day=None` 也能仅凭实时库造出当日日线）；新树按
   「`realtime=True` 已内建」整块删掉了它 —— **这个等价不成立**，只在
   「窗口里至少有一根历史日线」时才成立。实测：
   `_fetch(600519, 'day', 10-21, 10-21)` → `None`；
   `_fetch(600519, 'day', 10-19, 10-21)` → 3 根，含 `10-21`（实时合并生效）。

**影响**：
* SH/SZ 盘中 **60min / 15min 不更新**；只有等收盘 `--save` 写入当日日线后，
  下一次扫描才会补上当天。分钟行的时间戳也只会停在最后一次跑批的时刻。
* **北交所 `920xxx` 即使修好日线回退也不会盘中更新**：实测
  `realtime_2026-10-21` 里 `920` 开头 **0 行**（腾讯 L1 快照不供北交所），
  只能等 `--save`。
* `chartscan_cli.ps1` 的 `while($n -lt 6)` **没有 `$n++`** ⇒ 实际是死循环，
  即“一直扫”。如果本意是 5 轮，这是个笔误。

**已修（2026-10-21）**：`_fetch` 在「日线为空且窗口末端就是最近交易日」时，
往前回看 **512 天** 重读一次（`realtime=True`），再把结果裁回原请求窗口；
回看日只负责触发实时合并，不参与扫描。不动 `kline83` 的 `None` 语义。
实测 `_fetch('600519','day',今天,今天)` 修复后返回今天 1 行；`920183` 仍为
`None`（北交所无实时，符合边界）。新增 6 条回归用例（896 条全绿）。

**已修（2026-10-21 续）**：分钟扫描挑「最近框」的锚点由 `end_date`（00:00）
改为交易日 **15:00** ⇒ 选当天**最晚**的检出；日频与非交易日保持原行为
（`_closest_anchor`，新增 4 条用例）。全量 **900 条测试全绿**。

**仍未做**：同一根未走完的 bar 每个 pass 会重复扫（无「bar 未前进就跳过」的判断）。

### XGBoost 迁移（**D61 / D62**）—— ✅ 框架 + compact 生产者，⏳ 训练/落库未做

`analysis/xgboost/` 已落地：`features.py`（契约）、`markup.py`（标签 +
分型统计）、`rank.py`（旧 `calc_portfolio_returns` 的 -9..9 评分 rank，
含 `rank_from_83`：8.3 分钟线 → MACD 死叉 → rank）、`train.py`
（蓝本 `benchmark/train_xgboost.py`）、`predict.py`、`model.py`。
**纯 DataFrame 进/出**。

**compact 生产者（D62）**：`analysis/compact.py::calc_compact_features`
已接 **37 个 derivable 列**，并内置 rank 7 列；支持 60min-only 与
60min+day（日线 `_MAJ`/weekly 按**因果**映射：盘中用上一交易日、15:00 用当日）。
另补搬 `ATR_RSI_Stops_v2` / `ATR_SuperTrend_cross_v2`（`analysis/indicators.py`
的 `atr_rsi_stops` / `atr_supertrend`），产出 ATR Stopline/SuperTrend 及
`_MAJOR`/`_WEEKLY` 列；以及 `calc_peak_point_vX`（`analysis/peak.py` +
`lineareg_band`），产出 `PEAK_LOW_TIMING_LAG` 等。
`--strategy compact`（`pipeline/compact/`）已可跑，**只读、不落库**；
固定 60min+day+rank，**没有命令专属开关**；`--eval` 缺省 fast。
仍未搬的 2 列：`MA90_CLR_RET_GAR`、`PeakLowLag`（返回 NaN）。

特征契约现状（`feature_status_report()`）：
旧 58 条目 → **55 唯一列**（3 组 `MTMPOWER30*` 与 `MTM_NORM*` 同列）；
用户定废弃 **4 列**（ZEN wavelet / DESINTY_CHAIN / MAPOWER30_METRIC），
用户定**废弃 14 列**（旧 4 列 + MASSIVE 2 + `ZEN_PEAK_RET_GAR` 2 +
`MA90_CLR_RET_GAR` + 资金流 4 + `MTM_FLOW_LAG`），默认 **41 列**。
`feature_status_report()`：`produced` **41** / `missing_*` **0** /
`deprecated` **14**。按实际非空列：**compact 单独 37/41，compact+review
41/41（无缺口）**；替代特征见 `features.REPLACEMENTS`。

常量侧：AST 审计 8 个常量类，**0 个未定义**（补了 AKA 14 + TRD 3）；
守卫 `test_constant_refs` 钉住。

**下一步**：
1. markup 的 21 个必需输入（`ZEN_WAVELET_TIMING_LAG`、Donchian、ATR 等）
   补齐后才可能真正训练 —— 当前缺列**明确拒绝**，不静默 NaN；
2. 横截面统计（按日聚合 rank/特征）→ 再决定落库形态（用户：详细数据暂不落库）；
3. `--train` / `--strategy xgb` 与预测落点。

### Mainstream 移植（`models/mainstream.py`）—— ⏳ 待启动

老树 2,103 行 + workload 848 行，输出 **83 列**；其中 **64 列**新树无生产者、
也不在 review 契约里。`calc_compact_features` 的**新树精简版**已在 D62 落地
（37 列 + rank），但 mainstream 的 83 列还缺 mswing/ChartScan/4quad/czsc/
mainstream-credits 等模型函数；牛股诊股数据也未搬。
架构与阶段拆分见 worklog 2026-10-20「再续六」：
`pipeline/mainstream/` + `MainstreamPipeline(BasePipeline)` + `--strategy mainstream`，
先搬 compact builder 并对拍，不能一次搬完。

### 卡在**上游特征链**上（不是卡在谁身上，是前置没到）

| 任务 | 内容 | 卡在哪 |
|:--|:--|:--|
| **M9T 收尾** | `calc_magic_nine_turns_dummy` / `_baseline` / `_baselines`（`analysis/magic_nine_jit.py` 只搬了**核** `np_vXs`，见 `D49`）| 它们的输入列（`MACD_ZERO_TIMING_LAG` / `DEA_NORM_MAJOR` / `DIF_NORM_MAJOR` / `MTM_NORM_MAJOR` / `MAPOWER30_MAJOR` / `MAGIC_NINE_TURNS_MAJOR` / `_WEEKLY`）**新树一个都没有生产者**；另需 `calc_zen_chopstickss`（走 CZSC）与 `calc_renko_atr_vX` —— **两个函数整个不存在**。⚠️ 现在硬搬会算出**静默的错值**（`NaN` 参与比较）。前置就绪后**只是两行编排** |

### 卡在你身上

| 任务 | 内容 | 状态 |
|:--|:--|:--|
| **C2** | `zen_bt.py` 撮合按契约移植进 `engine.py` | ⛔ **需你先写策略实现**（`XGB_ECHO_TIMING_LAG` + `QUADRANT_LEVERAGE_MACD_TIMING_LAG_DUMMY`）—— 信号公式属策略侧，不猜（D7）|
| **A** | 5 个参考集合灌进 8.3 | **4/5** —— `financial` 已有 **182,769 行**（4.4 早搬完，见 worklog 的计数表注），但**停在 2020-03-31**；续数据须走 `tdxaidata`（要 field_list）|

### 代码完成、等一个时间窗口验证

| 任务 | 待验什么 | 决策 |
|:--|:--|:--|
| **RT** | 开盘时段跑一次 `--sub l1_tencent`：8.3 出 `realtime_<今天>`、带 UTC-aware `ts`、读回来非空、`--purge-l1` 打得出来 | D10 |
| **L1-AUDIT 收尾** | 两处浪费已改并单测过（单遍与两遍**逐值相同**）。开盘跑一次确认实时集合照常写入 | D46 |

### 待你决定（不是 Bug，是取舍）

| # | 事项 | 选项 |
|:--|:--|:--|
| 1 | **未被引用但值错的常量** | 只有**全量审计**能闭合（现有两个工具都只看被引用的）。只读检查，成本低 |
| 2 | **`880xxx` 那 689 只指数** | 腾讯**不供**（通达信自编指数），已从码表剔除。要它们得换源（通达信本地 / pytdx）—— 但**它们本来就不是交易标的**，多半不必 |
| 3 | **LEADOUT 是否接入 review** | 库函数已就绪（D73）。选项：① 只留库代码、由策略自行调用；② 进 review 契约并落库（需定阈值校准 + backfill，参照 D69/D72）。快测 edge 弱、高价股 0 触发 |

> **E 已关闭（2026-10-20）**：查库确认 `200–209` 与 `161–169`/`184` 在 `index_*`/`stock_*`
> **都没有数据** ⇒ 纯分类修正，无需搬数据。ETF 拆分本身已完成（`index_*` 零 ETF）。
> `150xxx`（34,888,580 行，在 `etf_*`）与 `16xxxx`（124,750,356 行，在 `fund_*`）按
> **已知残留**处理、不搬 —— 见 **D56**。

---

## 三、已完成（细节在 worklog + 对应决策文件）

| 任务 | 一句话 | 决策 |
|:--|:--|:--|
| D-ETF | ETF 前复权回迁（`c83b593`） | — |
| D-min | 分钟线读取器重写（`fe2fd95`） | — |
| C1 | `portfolio/` 骨架（`f993d31`） | D7 |
| D | QUANTAXIS 全树剔除，import 归零 | D12 / D13 |
| TDX-DIST | 每 worker 各粘一台 + 随机起点；TCP 账实量过 | D23 |
| TDX-HOSTS | 探活 66 台可用，活动列表落 `~/.GolemQ` | D22 |
| TQDM×BANNER | `pytdx_kline` 零裸 print，4 个不变量上测试 | P22 |
| ENV-SHARED | conda env 与老树共用，pip 冲突**别清理** | — |
| SAVE-HANG | 卡死两个真因：BLAS 守卫没搬 + 首台死服 | D21 / P21 |
| CLI-BOOT | 环境自检；本机 MongoDB 8.3.11 达标 | D20 |
| CLI-EXIT | 用法错 = 2 / 运行期失败 = 1 | D18 / D19 |
| CLI-SPLIT | `__main__.py` 742 → 99 行 + 10 个命令模块 | D17 |
| BOOT-BANNER | 自检四态 + 时刻戳；硬拦只有 `python`/`依赖包` | D24 |
| TTL | 参考数据刷新闸，按集合分阈值 | D16 |
| SAVE-CLI | `--save` 收口成 `<SOURCE>` + 参考数据 banner | D14 |
| KLINE-SHORT | 探针 + 逐只水位 + TTL 兜底（三条缺一不可） | D25 / P25 |
| QUADRANT | 四量图 + 6 指标，与老树逐值对拍 | D39 |
| RSRS | `pre_rsrs_data_func`，与老树逐值对拍 | D40 |
| RT-FIX | REALTIME 合成读路径三处**静默** bug | D29 / P33 |
| CHARTSCAN | CV 并入，数据走 8.3，env/ZMQ 边界保留 | D42 / D43 |
| SCOPE-MOVE | `scopes.py` 归位 `analysis/cv/`；筹码不再自补换手率 | D44 |
| CODE-TABLE | 码表缓存挂进 `--save tdx`（此前**全树无人写**）| D45 |
| L1-AUDIT | L1 走拉取全市场，不改「全推」；两处浪费已修（174→7 请求/轮）| D46 |
| **L2-DEL** | **腾讯 L2 整条删除** + QMT 五档占位一并删（子集流、还比 L1 慢）。`GQSUBSCRIBER` 回到 2 键 | D47 |
| **ETF/INDEX-L1** | **ETF 与指数进 L1 tick 流**：码表扩成 `stock ∪ etf ∪ index` = 7,878 只，一律带交易所前缀 | D48 |
| **DOC-SPLIT** | 文档拆成「索引 + 一份决策一文件 + worklog」，`gen_docs_index.py` 守索引 | — |
| **IND-PORT** | 三组指标搬迁（`BOOTSTRAP_STAGE_BEFORE` / Poly9 / Magic Nine Turns）+ 两表合并 | 🔶 **算法侧只剩 `2f`（Poly9 编排）**（`DECISIONS.md` **D49**）：常量补齐 ✅ / Garident STAGE ✅ / `rolling_poly9_v2` ✅ / M9T 核 ✅ / `mapower`+JIT ✅ / `TA_HMA` ✅ / 两列「背离」与废弃函数一并清除 ✅。⚠️ **第 4 步两表合并、第 5 步解 stub 还没动 —— 见下** |
| **CV-PORT** | 4.4 的 ChartScan CV 检出结果搬进 8.3 六族（**1628.8 万行，丢弃 0**） | ✅ D51 |
| **CV-ROUTE-FIX** | ChartScan 写库路由 4 处分叉（日族漏写 / 60min 索引错 / 60min 少一笔 / 15min 标记少一张） | ✅ D51 第四节，14 条用例钉住 |
| **CV-CLI** | `--scan ChartScan` / `--server ChartScan` 进 CLI，**自动切 GolemAI env**（re-exec），围栏无豁免 | ✅ D51 五之二，18 条用例 |
| **PANDAS3** | pandas 3.0 / numpy 2 兼容（µs 精度 / `Series[int]` / `float(1×1 matrix)` / 日族 datetime 类型） | ✅ D52 / P35 |
| **PROPHET** | `FBPROPHET_{LB,UB}{,_MAJOR}` 四列搬入 `analysis/fbprophet.py`：**只搬算术**（拟合→区间对齐），落库存 **`yhat/close` 比值**；4.4 存量不搬。⚠️ 比值**不是**数学不变（先验是绝对量纲，重算差 0.82%）—— 收益是"历史不用重写"。⚠️ **区间是抽样出来的**（同数据两次预测差 0.22）⇒ 已固定 `UNCERTAINTY_SEED`，否则重跑不幂等 | ✅ D53，**10 条**用例（含**与老树逐位对拍**）；prophet 1.1.7→**1.5.0**（1.1.7 在 pandas 3 上 fit 就崩） |
| **EPOCH-DIVISOR** | datetime → 秒的系数 = `constants.epoch_divisor(dtype)`（**按每条序列的 dtype 查**，不是全局标量） | ✅ D52 / P35，9 条守卫用例 |
| **QUPRICOEF** | 上升系数 `QUprICoefD` 实际消费链 = **`CAND × log10(有效 MACD lag)`**（老树现跑 1904/1904 逐位对拍）。夹取链 + 第二 OR 分支已按**最小表达式**搬完（`4QuadZenDCnt` 0/2075、`QLevMACDLagD` 21/2075、`QLevPwrlnUprLag` 26/2075）；剩余差全是**老树输入侧 3 个怪癖**（分钟 qfq 提前一天 / 2026-09-30 老树分钟脏点 / 由此少一次星线），**不应复制**。⚠️ 老树这条路用 **QUANTAXIS 口径 MACD**（`calc_macd_qa`），不是 talib | worklog 2026-10-19，12 条用例 |
| **RT-SHORTWIN** | `--save tdx` turnover 阶段全市场刷屏（短窗口被误判成"新股"）：提示改为受 `verbose` 控制；`<30` 不再跳过实时合并（`realtime=True` 恢复生效） | worklog 2026-10-19，2 条用例 |
| **E-ETF-DATA** | ETF 拆分核对：`index_*` 五频率 **ETF 分类代码 = 0**，拆分已完成；`index_*_removed` 的 9 只 ETF 在 `etf_*` 全有数据（无丢失）；两组路由变更查库**均无数据** | ✅ **D56**（残留不搬） |
| **SAVE-TDX / SAVE-CLI** | 全市场 `--save tdx` 实跑正常（用户 2026-10-20 确认） | D11 / D14 |
| **CODE-TABLE** | 码表闸命中：`stock_list` 等签到 **2026-10-19 11:27–11:32** 全成功，`stock_codes.conf` 于 **11:32:48** 被重写 | ✅ D45 |
| **MONEYFLOW-FIX** | 落库前帧列去重（`keep` 重复 `code/date` ⇒ Length mismatch）；同花顺源端分页重叠按 `(date,code)` 去重；`[metadata_day]` → `[reality_feat_day]` | ✅ worklog 2026-10-20，4 条用例 |
| **MSG-LAYER** | `agents.messenger` → `agents.channels`；`supervisor.messenger` → `supervisor.alerts`（`Messenger` → `AlertDispatcher`）；告警复用 `sc_send`，删零调用 API | ✅ **D57**，6 条用例 |
| **TS/TURNOVER-FIX** | pandas 3 `resample(offset=)` 对 `1D` 告警（分钟级保留 offset，日频不传）；`fill_turnover` 用不存在的 `ts` 列取日期 ⇒ `DateParseError: None`，改取索引第 0 层 | ✅ P37/P38，4 条新用例 + 夹具改实物形状 |
| **REVIEW-CLI** | `--review` == `--strategy review`；`--eval` 全局唯一（删 `--review-eval` / `--review-codes`）；`--daily` 删除；`--force` / `--from` 全局且可组合；scope 必填、pct 默认明示；Ctrl+C → 130 | ✅ **D58** / P39 |
| **15MIN-ZEN** | 15min minor 补齐老树 `calc_zen_wavelet_reality_minor` 的 **19 列**（新增 `zDLagMinR/zPLagMinR`，用 `calc_zen_timing_lag`）；60min 的这两列仍由 `zen_snapshots` 产出 | ✅ worklog 2026-10-20 |
| **REGTREE-RS** | 删除零读取的 `REGTREE_TREND_RS` 常量 + 4 处赋值；`TREND`/`RETURNS` 保留（内部依赖） | ✅ worklog 2026-10-20 |
| **GAR-EP** | coldstart 输出 `EP_STG_BF` + `garP/garPCount/garPClr/garPAnchor/garSLag` 到 60min；`garP` 存比值带；历史未回填 | ✅ worklog 2026-10-20 |
| **CN-COMMENT-AUDIT** | 千股千评只搬了 `TurnoverRate`；牛股诊股（`stock_diagnosis*`）与旧 `stock_moneyflow` **完全未搬**；新采集仅当日/近 7 日 | ⚠️ 见 worklog 2026-10-20 |
| **M9T-AUDIT** | raw+lag 四列齐全（`m9tMajR/m9tLagMajR/m9tWekR/m9tLagWekR`）；`m9tBsl/m9tBslLag/m9tBf/m9tSxBf/m9tDmy` 缺失，卡上游特征链 | ⚠️ worklog 2026-10-20 |
| **POOLCOEF** | `--strategy poolcoef` 暂不能恢复：新 `models/poolcoef.py` 缺 `calc_stock_poolcoef_analysis`；需单独移植 `benchmark/poolcoef.py::workload` | ⏳ 待排期 |
| **SE-PIVOT** | 60min 新增 `SE`（15:00 日频样本熵，≤2160 根）+ `ZenPivotCnt`（因果中枢计数）；60min 契约 36 列 | ✅ worklog 2026-10-20，882 条用例 |
| **REVIEW-ONEPASS** | day 832 / 周线 2160；60min 一趟合并 zen60+minor17+zen15+SE+pivot；day 一趟合并 daily_snapshot+coldstart（M9T/Poly9 共享，周线按周缓存）；15min 表 1:4 取样 | ✅ worklog 2026-10-20 |

| **JIT-MERGE** | `analysis/{magic_nine,quadrant,renko,regtree}.py` 并入对应 `*_jit.py`，原文件删除；核心核显式 numba 签名 | ✅ **D59**，881 条测试 |

| **PARALLEL-FIX** | `tqdm` 未传 iterable ⇒ `tqdm_joblib` 回调抛错 ⇒ joblib 零 worker 挂死；已修并打核心数横幅 | ✅ **P40**，885 条测试 |
| **CV-MOVE** | `markets/StockCN/chartscan/` 五个模块整体迁至 `analysis/cv/`；库实例改由 `StockCN.get_database()` 中性注入，市场包外不再出现库名/句柄；新增「库名不外泄」树扫描守卫 | ✅ **D60**，890 条测试 |
| **CV-INTRADAY** | 单日窗口日线为空的回退（回看 512 天再裁回），修复盘中 60min/15min 不更新；分钟 `closest` 锚点改为交易日 15:00 | ✅ **P41**，900 条测试 |
| **XGB-FRAME** | `analysis/xgboost/`：55 列特征契约（51 默认 + 4 废弃）、markup/rank/训练/预测/模型路径；`rank_from_83` 已接 8.3 读取 | ✅ **D61 / P42**，944 条测试 |
| **XGB-COMPACT** | `analysis/compact.py::calc_compact_features`（37 derivable + rank 7 列 + ATR + peak，60min/day 因果映射）；`pipeline/compact/` + `--strategy compact` 只读 | ✅ **D62**，958 条测试 |
| **D59-EXAMPLES** | `examples/` 里 `analysis.renko` / `analysis.quadrant` 旧 import 同步到 `*_jit`（D59 合并后漏改） | ✅ 2026-10-21 |
| **NOISE-FIX** | Prophet 1.5 的 cmdstanpy INFO / `Logging error` 与 Chebyshev `RankWarning` 刷屏；`old_tree` 不再误删真实 polars | ✅ **P44**，960 条测试 |
| **COMPACT-STUDY** | `CompactPipeline.calc_features(code, offset, ret_meta)` 研究入口（范式同老树 cookbook 的 `calc_stock_poolcoef_analysis`）；新增 `cookbooks/calc_compact_features.ipynb`（只产特征 + matplotlib） | ✅ worklog 2026-10-22，962 条测试 |
| **META-4SRC** | `--save tdx` 元数据段扩成 `turnover · moneyflow · eastmoney · diagnosis`：千股千评复用 `save_moneyflow`；牛股诊股新增 `markets/StockCN/crawler/diagnosis.py`（页面已从 stockpage 搬到 doctor，五维评分 + 综合 + 平均成本）；逐只 ~20–40 分钟，失败降级；进度走 tqdm + 每 100 只 `say` 锚点；eastmoney/diagnosis 有当日覆盖 >=90% 短路闸 | ✅ **D63**，980 条测试 |
| **BOOTSTRAP-DETAIL** | 环境自检「依赖包」明细**逐包**打勾：新增 `presentation.status_line`，不再因 pandas 一项不过就把 numpy/polars 等全标 `✗` | ✅ worklog 2026-10-23；pandas 3.0.6 全量 982 条全绿；epoch 测试已改按实际 dtype 断言（s/ms/us/ns 四单位模拟均过） |
| **CLI-INTERNAL** | 用户入口（29 个 dest）与内部入口（`python -m GolemQ.cli.internal`）分离：所有 `--save*` 调参/试跑旗子、review/migrate 的 limit+dry-run、fill-turnover 范围，以及 `--save-coverage` / `--save-status` / `--metadata-align` / `--review-runtime` / 全部 `--migrate-*` 整条命令，用户 `--help` 不再出现 | ✅ **D64**，985 条测试 |
| **REVIEW-CAMEL** | review/coldstart 列名统一伪驼峰：`stage` / `bstStgBf` / `epStgBf` / `prophet*` / `poly9Maj` / `poly9pwr3Maj` / `se` / `zPLagMajR`；存量 1,217,020 行用 `$rename` 原地改（不重算） | ✅ **D65**，986 条测试 |
| **M9T-LAG-DEL** | review 不再生产/落库/attach `m9tLag`（库里全 None）；`day_core_step` 盘中补算上一可用交易日的 coldstart 行 | ✅ **D67** |
| **NOTEBOOK-READONLY** | notebook 只读 review 已落库的 `ZenPivotCnt`/`se`，不再本地重算；attach 增加 `HOUR_EXTRA_COLUMNS` 精确 join | ✅ **D68** |
| **HOUR-EXTRA-BACKFILL** | `tools/backfill_hour_extra.py` 只补 `se`/`ZenPivotCnt`；已 apply **23 码 / 47,191 行** | ✅ **D69** |
| **PRICE-DUMMY** | 归一化价格统一 `D`（落库）/ `_DUMMY`（常量）；DB `$rename` 6,650,307 行 + 60min `poly9Maj` `$unset` 3,797,971 行；attach `*close` 还原无后缀价格 | ✅ **D70** |
| **RENKO-DAY** | review 日族加 Renko S/L 时序；`tools/backfill_renko.py --partial --apply` 两轮，2,875,464 / 2,878,469 日行（99.9%） | ✅ **D71** |
| **RENKO-FAMILY** | review 日族保存 Renko 全族 14 列，6 个价格列落 `*D` 比值；backfill 扩到 14 列并修掉 `$ifNull` 缺 `$` 的静默 bug | ✅ **D72**：存量 **2,935,202 / 2,935,202 行**（1,883 码），14 列缺失均 0；997 条测试 |
| **LEADOUT** | 主力吸筹 6 列的镜像（HIGH/HHV/上涨偏离 + KDJ/MACD 反向）：`analysis/quadrant_jit.py::calc_mainstream_leadout_feats` → `MsLeadout*` 6 列；与吸筹共用内核，老树对拍仍过 | ✅ **D73**（库代码；**未接 review/落库**，快测 edge 弱、阈值需校准）；1002 条测试 |
| **DAY-EXTRA** | `se` / `ZenPivotCnt` 从 60min 契约移到日族：`se` 改日收盘样本熵（540 日窗口），`ZenPivotCnt` 用 **2016 起全历史 60min** 取 15:00 值；60min 侧 `$unset`，attach 走日族映射 | ✅ **D74**：day `se` 99.996%、`ZenPivotCnt` **99.94%**（缺 2 只北交所）；1002 条测试 |
| **DATA-INVENTORY** | 数据暗线审计：8.3 频率阶梯齐全、1min 母数据 21.5 亿行（2017-11 起）、60min 1999 起、1day 1990 起；4.4 多出的 60min 行是重复/错标频率，**标准时刻缺失 0** | ✅ 新文档 [`docs/data_inventory.md`](docs/data_inventory.md)（已挂 `CLAUDE.md` 第 7 条） |
| **OLD-REVIEW-NOMIGRATE** | 4.4 `golemq.stock_reality_feat_*` 旧 review 特征**明确不搬**：未归一化、复权锚冻结在过去，无参考意义；8.3 历史缺口是刻意的（CV 列仍保留） | ✅ **D75**，无代码改动 |
| **BOTTOM-ROOF** | `btmP`/`roofP` 是**绝对价**：`garPD`(比值)→还原`garP`(绝对)→取 band 首底/末顶；补老树常量、`stages.garident_bottom_roof`、attach 接线；**不落库** | ✅ **D76**，由用户更正 D49 的「同病」误记 |
| **NAMING-AUDIT** | 列名伪驼峰：39 个老树真值全改（`mpwr3*`/`hmpwr12`/`lrgPrice`/`cciNorm`/`ma90Clr*`/`volNorm*`/`peak*`/`fin*`/`lastClose` 等），28 个标准名/已落库元数据保留；8.3 存量扫描 0 命中（4.4 只读不动） | ✅ **D77**，全量 1003 条测试 |

---

## 四、纪律（上一轮用事故换来的，勿犯）

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
