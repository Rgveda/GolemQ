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
目标下界 day 2560 / min 800 天）。2026-11-16 全量 **1,125 OK（skipped=1）**。

**下一步（2026-11-17 复核）**：全量对拍分类（`compare_review_golden.py`，**仍无运行记录**）、
coldstart `m9tBsl`（仍卡 compact 特征链）、`--eval` 回填（股票只剩 **228** 码、ETF 仍
**2/1,737**，见下「全市场覆盖缺口」）；internal 入口的 `--review-runtime` 端到端冒烟
**仍无记录**（只有 mock 单测 `TestRuntimeNoWrites`）。
⚠️ `--eval` 是**全局唯一**的 scope 选项（`eval_scope`），与 `--review` / `--scan` /
`--strategy` 组合；**不要再造** `--review-eval` 这类前缀变体（D58）。
⚠️ `--daily` **已删除**（语义不明）；`--review` == `--strategy review`。

#### `--review` 复审 F1–F7 —— ✅ **已修**（2026-11-16）

清单、机制与验证见 [`docs/worklog/2026-11.md`](docs/worklog/2026-11.md) 同日「代码复审」与「再续」两条。
要点：

1. `worker_count()` 挪进 `pipeline/base.py`，review/compact 改调它 → 默认 0.32 不再退化成串行；
2. ChipDist 盘中改用**前一交易日**的分子（旧值有未来函数）；
3. `StepRunner` 按依赖闭包决定是否执行 + `zen_snapshots` 为 15min 补所在整点的 60min bar
   → 60min 已全落库时 15min 也能补；空目标不再整帧 upsert；
4. F3 NameError、F5 死代码、F6 重复计算、F7 `--offset` 三族边界（纯日期 ⇒ 23:59:59）一并处理。

#### 比值列 as-of 口径（**D82** / P51）—— 代码已改，**存量待重建**

60min 行的比值列（`*D` / `*_DUMMY`）分母统一为**分子快照日的 as-of 收盘**
（盘中 = 上一交易日；15:00 = 当日）。实测只有 **10 列**口径不对 ——
ChipDist 4 + Renko 价格 6；其余 28 个比值列本来就是 as-of。

✅ **存量已重建（2026-11-16）**：`tools/fix_asof_ratios.py --apply --workers 16`
扫码 7,406 个 / 更新 **7,833,787 行** / 3,337s。复核：600519 / 000711 / 603366
各 21 个交易日 + 随机 25 码 —— 盘中 3 根**全一致**；三码 21 条比值列 ×
fetch/attach 的还原值与重算分子逐值一致。做法是**复制上一交易日 15:00 行**，
不重算（幂等，重跑 `changed=0`）。

还原侧（2026-11-16 收尾）也已对齐：`PRICE_RATIO_COLUMNS` 补齐 `rTrePrcD/rTreSlopD`、
删死名 `regtreePriceD`；帧首日盘中用日线兜底；close 为空时不写 NaN。

#### Review 列覆盖度（**2026-11-17 全库实测**，见 D71/D72）

口径：`stock_reality_feat_60min` 的 review 行（`zDLagR` 非空，**10,658,363** 行 / 5,356 码）。

| 列 | 覆盖 | 结论 / 下一步 |
|:--|--:|:--|
| `stage` / `garPD` | 99.8% | 不用重跑；缺的多为短历史 / 定带失败 |
| `poly9MajD` / `poly9WekD` | 99.7% | — |
| `rTrePrcMajD` / `rTreSlopMajD` | 95.3% | 缺的是 `<450` 根 / 拟合失败，**历史不足补不了** |
| `prophetLbD` | **30.7%**（2,654/5,356 码） | `prophetLbMajD` 26.0%。**全码要么全有要么全无**（2024/25/26 各约 30%），不是按年丢失。需专门「只补 Prophet」的重型工具，或接受现状（**待你决定**） |
| `se`（**60min 表**，D78） | ✅ 99.98% | 10,656,494 行 / 5,356 码；D74 已撤销，列只落 60min |
| `ZenPivotCnt`（**60min 表**，D78） | ✅ 99.98% | 10,656,949 行 / 5,354 码；口径 = 截至当前 60min bar 的因果中枢数 |
| `renkoSpriceD` 等 14 列 | 100% | `tools/backfill_renko.py`（D72）已补齐 2,935,202 行 / 1,883 码，14 列缺失 0 |

> 2026-10-27 旧表里的 99.7 / 100 / 93 / 42 已过时；`se` / `ZenPivotCnt` 的
> 「回填中」也已收口（2026-11-17 复核，见 worklog 同日「文档核对」）。
> `stage` 自 **2026-11-18** 起由 review 按 **4.4 算法**重算（D81 更正），
> 表中覆盖率仍是 11-17 存量口径；新 bar 要跑一轮 `--review` 才有。

#### ~~`stage` 源断供~~ —— ✅ **2026-11-18 已修：改回「移植 4.4 算法」**

**D81 的「从 4.4 搬落库行」是对用户意图的误读**（用户 2026-11-18：
「我不是要 4.4 搬运落库数据，而是要搬运 4.4 旧树的算法」）。当时的症状：
`stage` 非空行数 11-11 15,694 → 11-12/13 6,543 → 11-16 241 → 11-17 **0**，
`STAGE_MODE` 从 11-16 起 NaN。

**修复**（D81 顶部更正 + D83）：
* `analysis/stages.py` 新增 `calc_cluster_checkpoints`（4.4
  `calc_blh_ground_zero_anchor` 的 checkpoint 段，**不用 `iloc[::21]` 近似**）
  与 `calc_stage_frame`（`CLT_GRP_CHK` → `calc_garident_stages`）；
* `compact._base_frame` 补 `deaOLag` / `difOLag`，`_derive_major` 补
  `POLYNOMIAL9_MAJOR`；
* review 的 `build_hour_extra_frame(..., day=ctx.frame('day'))` 已产出 `stage`
  并进 `contract.HOUR_STAGE_COLUMNS` / `ZEN_COLUMNS`；
* 历史存量（1,337 万行）保留作对拍基准；`tools/migrate_stage_from44.py`
  **降级为校验工具**，不再是数据源。

**仍需做**：① 跑一轮 `--review` 把 11-14 之后的 `stage` 补上（当前库内最新只到
11-13）；② 重跑 `--strategy mainstream --eval all` 刷新 `STAGE_MODE`。
反弹统计见 worklog 2026-11-17「STAGE_MODE=-0.5 区间」。
**2026-11-19 再次触发**（`STAGE_MODE=-0.5` + `MasPosOnhold=0` + `PosBottom=1`，
`StageModeStreak=1`），为第 7 段（1 天，尚无后续数据）；18 天 / 7 段的完整统计见
worklog 2026-11-21「STAGE_MODE=-0.5 事件区间复核」。

#### `ZenPivotLag` 三条（D84，2026-11-20）—— ✅ 代码已接，历史待回填

| 列 | 表 | 状态 |
|:--|:--|:--|
| `ZenPivotLag` | 60min | ✅ 原有 |
| `ZenPivotLagMaj` | day | ✅ 新增（`pivot_stats(freq=D)`，逐目标滚动） |
| `ZenPivotLagMin` | 15min | ✅ 新增（`pivot_lag_minor_window`，尾部 4 根） |

* attach 三条都在（修了 `_attach_15min` 丢原样列的 bug）；
* 因果性由 `test_pivot_lag_causal.py` 锁定（全量 vs 截断逐值相同）；
* ⚠️ **checkpoint 000711 2026-01-22~23**：`ZenPivotLag` / `ZenPivotLagMaj`
  > 63 ✅；`ZenPivotLagMin` **不转负**（CZSC 中枢仍 open）；真正突然转负的是已有
  列 `zDLagMinR`（71→16→-1）。60min `ZenPivotLag` 实际首次转负 =
  **2026-04-01**（-138），day = 2026-04-02（-44）。
* **待做**：`ZenPivotLagMaj` / `ZenPivotLagMin` 的历史日/15min 行还没全市场回填；
  全市场形态扫描（用 `zDLagMinR`）4,830 事件 / 67 个 60 日翻倍，详见 D84。

#### `COST5_UPRISING_BEFORE`（D85）—— 纯函数已落，**未接 massive**

* `analysis/cost5_uprising.py::calc_cost5_uprising_before`（L2 化简：只留
  `RETURN_COST5` + `bstStgBf`/MACD 零轴交叉前两个分支）；doctest + 3 条单测；
* `chip_cost_by_day(..., with_return_cost5=True)` 可带出 `RetCost5`（不乘 adj）；
  `compact._derive_major` 补 `macdMaj`；
* ⚠️ 原型（000711 2025-04-07）**不触发**：stage=2 ⇒ `bstStgBf` ~1,900，丢掉的
  XGB/1K/Prophet 分支才是原式里能拉回条件的那部分；日线 `RetCost5` 也进不了
  `(0.00618, 1.68)`；
* 待你定：① 补 60min `RETURN_COST5` 生产者；② 放宽成 L1（只留 RetCost5 区间）；
  ③ 补齐 XGB echo / 1K stage / Prophet（工作量大）。

#### 全市场覆盖缺口（2026-11-01 查明，进行中）

`--review` 历史上只跑 `--eval fast`，而 `fast`/`block` 都解析成 **`focus_block`**
（`analysis/cv/scopes.py:175-176`）⇒ 只覆盖关注板块：

| 口径 | 数值 |
|:--|--:|
| `stock_list` 在册 / 有 review 行（`zDLagR`） | 5,580 / **2,433** |
| 缺口合计 | **3,147** = `etc` 3,019 + `fast` 128 |
| ETF（`etf_list` 1,716） | 有 60min 行，**0** 有 review 列 |
| 未覆盖码的 60min 行 | 只有 ChartScan 列；全表 `ZEN_DASH_LAG` 文档数 **0**，不是旧契约 |

处理（计划已批）：`--review --eval etc` + 补一轮 `--eval fast` + `--eval etf`，
新增 **internal** `--review-shard i/N`（`shard_codes()` 在
`pipeline/review/pipeline.py`，`tools/backfill_hour_extra.py` 改用同一函数）。

进度（2026-11-03）：缺口实测 **3,140 只股票**（`etc` 3,013 + `fast` 127；ETF 1,716 本轮不含）。
初版按 md5 均分 1,043/1,072/1,025，实测 Babel(18 核)/Gilgamesh(32 核)/Nightwatcher(160 核)
能力差太大，Babel 会成 8–9 天长尾 ⇒ 按 **9:16:61** 权重重分片：
`_review_rebal1/2/3.txt` = **329 / 584 / 2,227**（Babel / Gilgamesh / Nightwatcher），
三台预计 ~70h 同时收口。冒烟（002380 / 002400，`--from 2026-10-20`）真写正常：
40 行/码，`ZenPivotCnt`/`renkoSbar`/`zDLagR` 均落库。

⚠️ 远端启动方式（2026-11-03 实测）：WinRM 当前账户可登录，但**双跳**访问不了
`\\rgveda` 共享；已把代码（commit `a3dbc1c`）以 zip 复制到两台本地
`C:\Users\azai\GQ_review_20261103`，用 WMI `Win32_Process.Create` 脱离会话启动。
日志在**各机本地** `_review_rebal{1,2,3}.log`（不是 Y:）。Nightwatcher 上另有一个
既有的 `--strategy compact`（56 workers）+ MongoDB，review 只跑 **61 workers 一份**，
不再加并发。实测单码约 **1.9h**（800 天窗口 / 2,100 行，比早先注释的 36–40 分钟慢）。
跑完按「码表 − 有 `zDLagR` 的码」生成重跑表；`audit_review_nan.py` 抽检。

进度（2026-11-07）：其余两机停止；三片剩余码合并为 `_review_resume3.txt`
（**1,605 码**，含 Gilgamesh 未跑完的 161 码），在 Nightwatcher 本机重启
`tools/run_review_shard.py ... --pct-of-cores 50 --review-timeout 14400`
（Windows joblib 上限 **61 workers**），日志 `_review_resume3.log`。
已完成的行由 `missing_targets` 自动跳过；本轮只跑 review，不再启动另外两机。

进度（2026-11-12）：实测 `stock_list` 5,582 码中已有 `zDLagR` 的 **4,363**，
**缺 1,220**（清单 `_review_missing_20261112.txt`；ETF 仍 0/1,737，未在本轮）。
在 Nightwatcher（50.210）用**当前工作树**的代码重启：整包 zip
`_review_sync_20261112.zip`（335 项，仅 `GolemQ/`+`tools/`，无 `__pycache__`），
六份 review 关键文件 SHA256 与本地逐一对齐；本地全量测试 **1,075 OK（skipped=2）**
先跑过。命令写进 `cmd.exe` 后由 WMI 脱离会话启动（PID 29868，30 workers）：

```
tools/run_review_shard.py _review_missing_20261112.txt --pct-of-cores 19 --review-timeout 14400
```

日志 `_review_missing_20261112.log`（各机本地 `C:\Users\azai\GQ_review_20261103`）。
⚠️ 该目录**没有 .git**，同步一律靠整包覆盖；远端 mtime 不能当版本依据。

进度（**2026-11-17 复核**）：`stock_list` 5,583 码，有 review 行 **5,355**，缺 **228**
（225 只日线 ≥250 根 —— 是真缺口、不是新股；2 只无日线：688845 / 920067；
1 只短历史）。另有 1 个码在 review 表但不在码表（000016）。
ETF 仍只有 **2/1,737** 码有 review（4,045 行）。
下一步：这 228 只重跑 `--eval etc`；ETF 单独排 `--eval etf`。

#### ChipDist 四列（2026-11-06 修复）

* 根因：`hour_extra.chip_cost_by_day` 遇到 960 日窗口内**任一天**缺换手率就
  整段 `continue` ⇒ 该码 `cost*PrzD` 四列全空（000711 的 2023-12-15 即如此）。
  已改为**剔除缺日**，不再整段跳过。
* `tools/backfill_hour_extra.py` 的断点探针漏了 `HOUR_CHIP_COLUMNS`，已补；
  存量 **60min 缺 898 码 / day 缺 2,605 码**（review 3,934 码）。
* 补 60min：`python tools/backfill_hour_extra.py --apply --workers 8`
  （或 `--shard i/3` 三机）；该工具只写 60min，day 列需另跑。
* 进度（2026-11-07）：两机 `--apply --chip-only` 分片 —
  **50.204 Gilgamesh** 主分片完成 `ok=364 / skipped=1598 / error=0 /
  rows=688,947`（3h39m）；**50.38 Babel** 卡在慢码，已停止，剩余 60 个
  转到 Gilgamesh 补跑，结果 `ok=58 / error=0 / rows=81,089`。
  全库仍缺 **23** 个码：均为日线 <30 根或换手率 <30 天（如 001381 日线 5 根、
  920799 换手率仅 20 天），需先补 turnover 才能算 chip，不是回填进程问题。
  日志 `_chip_only_shard{1,2}.log`、`_chip_rescue61.log`。
* 存量复核（**2026-11-17**）：60min 表仍缺 chip 的码 **385**（5,356 个 review 码中
  4,971 有值）。样本全为 2026-11 新增 review 的新上市 / 短历史码（001232 / 301569 /
  301583 等，日线 0–48 根）—— 不是回填进程问题，需先有足够日线才能算。
  日表 chip 缺 **4,722,037 行**（见下 Gate 5）。
* ~~**正在跑的 review 进程加载的是旧代码**，收尾后仍需 backfill 一遍。~~
  2026-11-11 已停止旧进程（PID 34036），同步 `pivot.py` 修复后重启。
  存量 `ZenPivotLag` **9,041,118 行全负（正值 0）**，涉及 **4,364 码**
  （stock 4,362 + ETF 2；index 0）。
  不再走整表 review：`tools/backfill_hour_extra.py --zen-pivot-only` 只算
  `pivot_stats` 并只 `$set` `ZenPivotLag`，单码 ≈23s（整表 review 约 1.9h）。
  **已完成**：`ok=4364 / error=0 / rows=9,041,118`，耗时 6,367s；
  4,357 码出现正值，其余 7 码窗口内始终在中枢外。
  原因见 `PITFALLS.md` P46 与 `docs/worklog/2026-11.md`（未完成中枢 `end_dt`）。
  反弹策略回测（2024-08~2026-11）也在工作日志：**无限仓 avg 215 持仓不可行**；
  限仓 30~50 时 CAGR 约 **29.5%**（截断规则尚未定，见日志表）。

#### ~~消除补数与 `--review` 的编排平行实现（2026-10-31 挂账）~~ —— ✅ **已收口**

**2026-11-07 `c7b53f5` 已完成**：`hour_extra.py` 提供
`build_hour_extra_frame` / `build_chip_frame` / `chip_ratios_from_abs` /
`sample_day_frame`，review 与 `tools/backfill_hour_extra.py` **全部改调同一份**，
两套 `searchsorted` 映射已删；ChipDist 取数差异保留、但同经 `chip_cost_by_day`。
回归测试在 `test_review_chip.py` / `test_review_pipeline.py`。
（2026-11-17 文档核对时发现本节仍挂在「未完成」，已更正。）

#### 与上面同批待办（Gate 5，2026-10-31 挂账）

1. **日表 ChipDist 4 列回填**：新建 `tools/backfill_day_chip.py`，复用
   `hour_extra.chip_cost_by_day`（960 日线窗口 + `adj` 缩放），逐码对已存在的
   review 日行算绝对成本 → ÷当日 close 写 `cost5/15/85/95PrzD`；
   **2026-11-17 实测**：stock day 缺 **4,722,037 行**（review 7,734,337 /
   有 chip 3,012,300），ETF day 2,492 行全缺；
2. **日表清理**：跑 `backfill_hour_extra --unset-day --unset-day-renko`
   （`$unset` `se` / `ZenPivotCnt` / Renko 14 列）—— **2026-11-17 实测**仍有
   `se` **3,806,255** 行 / `ZenPivotCnt` **3,804,589** 行 / Renko **3,806,384** 行
   （stock day），D78 要求只在 60min 表；
3. 顺序：60min 全量回填收尾 → 抽公共映射（**✅ 2026-11-07 `c7b53f5` 已完成**）
   → chip 回填 → `$unset` → 提交。
   （映射已共享，`backfill_day_chip.py` 可直接复用同一份日采样/chip 映射。）

### 指标搬迁（`DECISIONS.md` D49 / **D50**）—— ✅ **四块全部收口**（原「③ 的编排」挂账已于 2026-11-07 `c7b53f5` 收口）

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

### ChartScan 日线 Maj 过期被 `ChartScaned` 掩盖（2026-11-06 查明并修复）

**现象**：000711 的日线 `ClsMaj` 在 **2026-09-22** 后不再更新；
`Class`/`ClsMin` 仍在更新，脚本每天跑也没断。

**根因**：`calc_ChartScaned_persistence` 在 `ChartScaned` 存在时，把
`all_nat_maj` 覆盖成 `recent['ChartScaned'].isna().all()`。分钟扫描写进日族的
`ChartScaned=2` 让日线 Maj 被永久判成“已扫”，**`CoordinatesMaj` 过期被忽略**；
又没有 0/1 行 ⇒ `recent_list=[]` ⇒ `mode=0`，客户端根本不发任务。

**修复（2026-11-06）**：
* 三套坐标各看各的：`Coordinates` / `CoordinatesMaj` / `CoordinatesMin`；
* `ChartScaned` 只用于找 0/1 的盘中末端，不再覆盖 `all_nat_maj`；
* 只有 Maj/Min 过期、基础 60min 坐标仍新时 ⇒ **mode 2 尾巴扫描**（最近交易日），
  不整段重扫 1,008 天。

**验证**：000711 → `all_nat_maj=True`、`recent=[2026-11-06]`、`mode=2`；
600519 / 000001 → 仍 `mode=0`（Maj 新鲜）。新增 2 条回归用例：
`ChartScaned=2 + Maj 过期 → mode 2`、`Maj 新鲜 → mode 0`；全量 **1,062 条 OK**。

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

### Mainstream 移植（`models/mainstream.py`）—— 🚧 第一段已落地（D79）

老树 2,103 行 + workload 848 行，输出 **83 列**；其中 **64 列**新树无生产者、
也不在 review 契约里。`calc_compact_features` 的**新树精简版**已在 D62 落地
（37 列 + rank），但 mainstream 的 83 列还缺 mswing/ChartScan/4quad/czsc/
mainstream-credits 等模型函数；牛股诊股数据也未搬。
架构与阶段拆分见 worklog 2026-10-20「再续六」：
`pipeline/mainstream/` + `MainstreamPipeline(BasePipeline)` + `--strategy mainstream`，
先搬 compact builder 并对拍，不能一次搬完。

**已完成的一段（2026-11-01，D79）**：「筛选计算全局智能仓位」
（Dempster–Shafer 策略仓位控制）——

* `pipeline/mainstream/` + `--strategy mainstream --eval <scope>`；
* `analysis/{mainstream_timing,position}.py`、`quadrant_jit.calc_quadrant_downrisk_major`；
* 输出 **`stock_massive_reality_60min_models`**（旧表名；全系统共用的 massive
  模型表，仓位只是其中一批字段 —— 不许另起 `*_position_*` 新表）：
  `PosBsln/PosPad/PosNegBsln/MasPosOnhold/DS_*`；
* **口径**：新树自洽、**不逐值复刻 4.4**（缠论实现差异已证实：600495
  2026-03-02 旧 `zDLagD=+21` vs 新 `-4`；数据/复权/窗口/参数均已排除）；
* 验证：2025-04-02→07 做空比 0.69→0.82、仓位 0（旧窗口版本复现）；
  无后视镜版本改为「三段脉冲式清仓」（04-02~04-07 仓位 0.21~0.24）；
  2026 年 7 月内有 31 根零仓位（07-01 起），见 D79。
* 事件列（D79 增补）：`PosFull` / `PosFullOnset` / `PosReClear`（清仓暴击，
  ≤4 根内二次清仓）；基准指数默认用**科创50 + 创业板指**，沪深300 仅历史对照。

### QuadrantGear / Mainstream 特征 DAG（D80）—— ✅ 已接入 mainstream

* `analysis/feature_flow.py`：Apache Hamilton（发行包 `apache-hamilton` 1.90，
  导入名 `hamilton`）列级 DAG。节点：
  `q60/q15/qd → pwrln(_minor/_major) → qlev_frame → qlev_macd_lag`、
  `q60/qd → quad_downrisk(_major)`、`zd_lag → prelude_frame`、
  `ms60/ms_day → ms_leadout(_maj)`、`bs_day → 4 个 BS*BfMaj`，
  终点 `QuadGear0` / `QuadGear0Bf`（`@extract_columns`）。
* `MainstreamPipeline.code_features` 已切到 `driver.execute`；新增纯计算方法
  `_features_from_frames`（三帧进、不取数）。`zDLagD` 只在 prelude 被请求时执行。
* `calc_quadrant_downrisk_major(..., feat_day=...)` 复用 `qd` ⇒ 日线四象限只算一次；
  一次 `execute` 内 `calc_four_quadrant_boost` 的 `annual={1008,4032,252}` 各一次。
* 2026-11-06 第五行改为 `Bu>Sel | QLevCharmOBf<QLev`（**不含 Charm_I**）；
  新增 `calc_quadrant_leverage_charm` 与 `charm_frame` 节点，Charm_I 只保留
  为独立特征。000711 01-23 全部 4 根触发；01-22 仍卡在第 3 行
  （`Bu < QLev`）。
* 验证：`test_feature_flow.py` 8 条（含与旧手写链逐值对拍）；真实 600519 /
  as_of=2026-11-04 冒烟 **3.4s / 4,248 行 / 15 列**，`code_masks` 正常；
  全量 **1,059 条 OK（skipped=1）**。
* examples 三图已加 `QuadGear0Bf == 0` 的**橙色半透明**触发带
  （`examples/_gear_bands.py`，默认开；60min 口径）。
* **下一步**：跨运行缓存（`.with_cache()`）未开启；`--review` / `--compact`
  不接入 Hamilton（review 已有 `FeatureStep` DAG；compact 是固定 37 列单函数）。
* 依赖声明在 `cli/bootstrap.py::MIN_PACKAGES`（`hamilton>=1.90`）。

### OmniPathUpR1Coef* 链（2026-11-02 盘点，待移植）

**生产者**：旧树 `analysis/base.py:936 calc_omnipath_coef_scores`
（引擎 `analysis/pool_coef.py::calc_wavlet_coef_of_fractal_vXI`；新树已有
`analysis/quadrant_coef.py::calc_wavlet_coef_of_fractal`）。

**5 个输入的精确公式与状态**：

| 输入 | 公式（旧树） | 新树状态 |
|:--|:--|:--|
| `ATR_CHANNEL_MAJOR` | `lineareg_band(daily)` 的 `(lrc_u-lrc_l)/lrc`（**不是 ATR**，名字误导） | ✅ 原语齐（`indicators.lineareg_band`、`peak.py`）→ 易 |
| `BOLL_TREND_TIMING_LAG_MAJOR` | `event_lag(close > BOLL_MAJOR)`（日线） | ✅ `analysis/compact.py` 已算 BOLL → 易 |
| `MAGIC_NINE_TURNS_DUMMY`（`m9tDmy`） | `M9T_BASELINE = f(m9t_lag + m9t_lag_maj)`，dummy 即 baseline | ⚠️ m9t/m9tLag 有；缺 `m9tLagMaj`（日线）→ 中 |
| `LINEAREG_WAVELET_UPRISING_TIMING_LAG_CANDIDATE` | `event_lag(CHOSEN_TRIGGER_EXPECTED_LINEAREG_LB < close)`；`CHOSEN_TRIGGER_*` 依赖 `MASSIVE_XGB_ECHO_UPRISING_BEFORE` + `ATR_LB/UB_MAJOR` | ❌ **卡 XGB echo**（新树无 producer） |
| `FBPROPHET_LB_UPRISING_TIMING_LAG_DUMMY`（`fbpLUpRsgLagDmy`） | `event_lag(FBPROPHET_LB_UPRISING_CANDIDATE>0 \| (FBPROPHET_LB_UPRISING_TIMING_LAG + CVaR_PEAK_PRICE_TIMING_LAG)>0)` | ❌ **卡 CVaR peak candidate**（只有常量） |

**顺序**：3 个易项 → XGB echo / CVaR peak 两条上游 → `calc_omnipath_coef_scores`
→ `QLevLagD` → `QLevLagDMed` / `QuadUpr1CoefLagMed`。

### 卡在**上游特征链**上（不是卡在谁身上，是前置没到）

| 任务 | 内容 | 卡在哪 |
|:--|:--|:--|
| **M9T 收尾** | `calc_magic_nine_turns_dummy` / `_baseline` / `_baselines`（`analysis/magic_nine_jit.py` 只搬了**核** `np_vXs`，见 `D49`）| 它们的输入列（`MACD_ZERO_TIMING_LAG` / `DEA_NORM_MAJOR` / `DIF_NORM_MAJOR` / `MTM_NORM_MAJOR` / `MAPOWER30_MAJOR` / `MAGIC_NINE_TURNS_MAJOR` / `_WEEKLY`）**新树一个都没有生产者**；另需 `calc_zen_chopstickss`（走 CZSC）与 `calc_renko_atr_vX` —— **两个函数整个不存在**。⚠️ 现在硬搬会算出**静默的错值**（`NaN` 参与比较）。前置就绪后**只是两行编排** |

### 卡在你身上

| 任务 | 内容 | 状态 |
|:--|:--|:--|
| **C2** | `zen_bt.py` 撮合按契约移植进 `engine.py` | ⛔ **需你先写策略实现**（`XGB_ECHO_TIMING_LAG` + `QUADRANT_LEVERAGE_MACD_TIMING_LAG_DUMMY`）—— 信号公式属策略侧，不猜（D7）。**2026-11-17 核对**：`portfolio/engine.py` 仍是骨架（`run()` → `NotImplementedError`），`strategy.py` 只有 docstring 示例 |
| **A** | 5 个参考集合灌进 8.3 | **4/5** —— `financial` 已有 **182,769 行**（4.4 早搬完，见 worklog 的计数表注），但**停在 2020-03-31**；续数据须走 `tdxaidata`（要 field_list）|

### ~~代码完成、等一个时间窗口验证~~ —— ✅ **2026-11-17 已由实时库数据证实**

| 任务 | 待验什么 | 结论 / 决策 |
|:--|:--|:--|
| **RT** | 开盘时段跑一次 `--sub l1_tencent`：8.3 出 `realtime_<今天>`、带 UTC-aware `ts`、读回来非空、`--purge-l1` 打得出来 | ✅ 实时库有 **24** 个 `realtime_*`，最新 `realtime_2026-11-17` **3,921,225** 行；样本 `ts` 为 UTC-aware（01:00Z = 北京 09:00）。`--purge-l1` **未跑**（会删真数据，按既有纪律不跑）。D10 |
| **L1-AUDIT 收尾** | 两处浪费已改并单测过（单遍与两遍**逐值相同**）。开盘跑一次确认实时集合照常写入 | ✅ 同上：当日集合在写，两处改动后写入未断。D46 |

### 待你决定（不是 Bug，是取舍）

| # | 事项 | 选项 |
|:--|:--|:--|
| 1 | **未被引用但值错的常量** | 只有**全量审计**能闭合（现有两个工具都只看被引用的）。只读检查，成本低 |
| 2 | **`880xxx` 那 689 只指数** | 腾讯**不供**（通达信自编指数），已从码表剔除。要它们得换源（通达信本地 / pytdx）—— 但**它们本来就不是交易标的**，多半不必 |
| 3 | **LEADOUT 是否接入 review** | **已退回未校准**（用户 2026-10-29）：`calc_mainstream_leadout_feats` = D73 逐字镜像，无 `min_price`/事件校准。**2026-10-29 新增折叠法**：`fold_high_price`（股价 >10 → `log10`，≤10 原值），两族都接受 `fold=True`（**默认 False**）。折叠后七点验收 ±7 天 **7/7**、严格「目标日或之后」**5/7**。⚠️ 降级 60min 单独无效（量纲缺陷在任何频率都成立）。选项：① 只留库代码；② 把 fold 设为默认并进 review 契约（参照 D69/D72）。**已知冗余买点**（用户 2026-10-29 决定维持现状、后期再过滤）：000711 2026-09-21（原公式固有）、301308 2026-07-22 / 601869 2026-10-20（折叠法引入，均在 MA20 下方）。区间收益工具：`tools/mainstream_segments.py` |

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
| **IND-PORT** | 三组指标搬迁（`BOOTSTRAP_STAGE_BEFORE` / Poly9 / Magic Nine Turns）+ 两表合并 | ✅ **全部收口**（`DECISIONS.md` **D49 / D50**）：常量 / Garident STAGE / `rolling_poly9_v2` / M9T 核 / `mapower`+JIT / `TA_HMA` / Poly9 编排（`2f`）/ 第 4 步两表合并 / 第 5 步解 stub 均已完成。⚠️ 本行原写「🔶 只剩 `2f`、第 4/5 步还没动」是过期状态，2026-11-17 核对时更正 |
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
| **CN-COMMENT-AUDIT** | 千股千评只搬了 `TurnoverRate`；牛股诊股（`stock_diagnosis*`）与旧 `stock_moneyflow` 完全未搬；新采集仅当日/近 7 日 | ⚠️ 采集已由 **D63** 并入 `--save tdx`（`crawler/diagnosis.py` + moneyflow）；**历史存量仍未搬**，新采集仍只有当日/近 7 日（2026-11-17 核对） |
| **M9T-AUDIT** | raw+lag 四列齐全（`m9tMajR/m9tLagMajR/m9tWekR/m9tLagWekR`）；`m9tBsl/m9tBslLag/m9tBf/m9tSxBf/m9tDmy` 缺失，卡上游特征链 | ⚠️ worklog 2026-10-20 |
| **POOLCOEF** | `--strategy poolcoef` 暂不能恢复：新 `models/poolcoef.py` 缺 `calc_stock_poolcoef_analysis`；需单独移植 `benchmark/poolcoef.py::workload` | ⏳ 待排期 |
| **REVIEW-MODEL** | **模型说明** [`docs/review_model.md`](docs/review_model.md)：日线/60min/15min 三族、落库关系、互相映射 | 📌 接手 review 先读 |
| **60MIN-SPEC** | **权威规格** [`docs/review_model_60min_feature_spec.md`](docs/review_model_60min_feature_spec.md)：79 列、as-of、命名单一真相 | 📌 改契约先改这份 |
| **SE-PIVOT / 一体化** | **60min 行 69 列**（D78）：zen4 + minor17 + coldstart16 + `se`/`ZenPivotCnt`2 + **Renko14（60min 原生 2160 as-of）** + **日族14（日表原生 + 60min 采样）** + **M9T2（60min 原生）**；日表 = 14 列。Poly9/RegTree/Prophet 按「算过就免算」复用日频结果 | ✅ **D78**（撤销 D74）；审计 [`docs/review_model_asof_audit.md`](docs/review_model_asof_audit.md)；回填 `tools/backfill_hour_extra.py` |
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
| **LEADOUT** | 主力吸筹 6 列的**逐字镜像**（未校准）：`calc_mainstream_leadout_feats(ohlc, features, fold=False)` → `MsLeadout*` 6 列；与吸筹共用内核；`fold=True` 时股价 >10 折成 `log10` | ✅ **D73**（库代码；**未接 review/落库**）；校准尝试已撤回；折叠法默认 `False`（用户 2026-10-29 确认现版本） |
| **DAY-EXTRA** | （已撤销）D74 曾把 `se` / `ZenPivotCnt` 移到日族 | ⛔ **D74 → D78 撤销**（2026-10-30）：两列回到 60min 逐 bar；日表两列 `$unset` |
| **DATA-INVENTORY** | 数据暗线审计：8.3 频率阶梯齐全、1min 母数据 21.5 亿行（2017-11 起）、60min 1999 起、1day 1990 起；4.4 多出的 60min 行是重复/错标频率，**标准时刻缺失 0** | ✅ 新文档 [`docs/data_inventory.md`](docs/data_inventory.md)（已挂 `CLAUDE.md` 第 7 条） |
| **OLD-REVIEW-NOMIGRATE** | 4.4 `golemq.stock_reality_feat_*` 旧 review 特征**明确不搬**：未归一化、复权锚冻结在过去，无参考意义；8.3 历史缺口是刻意的（CV 列仍保留） | ✅ **D75**，无代码改动 |
| **BOTTOM-ROOF** | `btmP`/`roofP` 是**绝对价**：`garPD`(比值)→还原`garP`(绝对)→取 band 首底/末顶；补老树常量、`stages.garident_bottom_roof`、attach 接线；**不落库** | ✅ **D76**，由用户更正 D49 的「同病」误记 |
| **NAMING-AUDIT** | 列名伪驼峰：39 个老树真值全改（`mpwr3*`/`hmpwr12`/`lrgPrice`/`cciNorm`/`ma90Clr*`/`volNorm*`/`peak*`/`fin*`/`lastClose` 等），28 个标准名/已落库元数据保留；8.3 存量扫描 0 命中（4.4 只读不动） | ✅ **D77**，全量 1003 条测试 |
| **EXAMPLES-RENKO-ASOF** | examples 的 Renko 图改读落库 `*_reality_feat_60min`（as-of）：`app_renko` 砖序 = 快照变化点；`app_pivot_renko{,_coef}` 非 60min 禁用 Renko；顺带修 `_attach_day` 的 ChipDist 列重叠（510300/000711） | ✅ 2026-11-03，全量 1029 条测试；详见 [`docs/worklog/2026-11.md`](docs/worklog/2026-11.md) |
| **DECOUPLE-44** | **`GolemQ/` + `examples/` 与 4.4 完全解耦**：通道移到 `tools/migrate44.py`；删除运行时的换手率 / CV 存量 / `index_list` 搬运函数与 `--migrate-*`（只留 8.3→8.3 的 metadata merge）；守卫 `test_no_44_runtime.py`（AST） | ✅ **D83**，2026-11-18，全量 **1,127 OK（skipped=1）** |
| **STAGE-ALGO** | **D81 更正**：`stage` 不搬 4.4 的行，改为移植 4.4 算法 —— `CLT_GRP_CHK` checkpoint + `calc_garident_stages`；review 重算并落库 | ✅ 2026-11-18（待补 11-14 之后的 bar + 重跑 mainstream） |
| **PIVOT-LAG-3** | 60min `ZenPivotLag` + day `ZenPivotLagMaj` + 15min `ZenPivotLagMin` 三表落库，attach 全部映射到 60min features；因果性测试锁定 | ✅ **D84**，2026-11-20，1,131 测试；历史 Maj/Min 待回填 |
| **COST5-UPRISING** | `COST5_UPRISING_BEFORE` 的 **L2 化简版**：`analysis/cost5_uprising.py`（纯函数 + 截断因果对拍）；`chip_cost_by_day` 可带出 `RetCost5`，`compact` 补 `macdMaj` | ✅ **D85**，2026-11-21；**未接 massive**，原型在 000711 2025-04-07 不触发 |
| **WEK-BACKFILL-2018** | `tools/backfill_quad_pwrln_wek.py` 支持长区间回填：`--day-start`（默认 start−3000 自然日 ≈2050 交易日，**MA233 周线需要 ≈1165 根日线**，原来写死 2016 起会让 2018 年段系统性偏低 1~2 档）、日期轴改取日线表交易日历（massive 15:00 行只有 2022-06-16 起）、`--out` CSV；尾部截断周用**延伸日历**判「该周最后交易日」；`--min-weeks 233` 只算成熟码（原来 `len(day)<520` 闸让**面板随 `end` 变**，2022-10 之后 83 天 Med 不一致） | ✅ **D86**；用例 `test_cases/tools/test_quad_pwrln_wek.py` 9 条；60min `4QPwrlnWek` 写 **2,753,648 行** + `--synthetic-60min` 补 **3,752,516 行**（**D87**）；CSV `output/mas_quad_pwrln_wek_2018_2026.csv`（2018-01-02~2026-11-20，2,012 交易日，面板 2,226→4,615）；**massive 未写**（2018~2022-06 无行，按用户口径不插稀疏行） |
| **MASSIVE-TIMING** | **模型说明** [`docs/massive_timing.md`](docs/massive_timing.md)：`MasPosOnhold`（`ds_conf×(1-uncertainty)`，`m_bear>0.618` 强制清仓）/ `STAGE_MODE` / `PosFull·PosReClear·PosBottom` / 周线 `4QPwrln` 共振；复跑 `tools/massive_timing_report.py` + `tools/masquad_bull_episodes.py`（用例 `test_cases/tools/test_masquad_bull_episodes.py`） | ✅ 2026-11-22；**四基准评估（沪深300/科创200/创业板/中证1000，中证2000 库里无数据用中证1000 替代）**：底带首日 20d 67~83%、`PosFullOnset` 60d 67~78%；周线「跨上 2.2」顶部警戒 20d 11~33%；**沪深300 上几乎无效、中小盘才有效**；`STAGE_MODE=-0.5` 7 段/18 天，20d 后失效 |
| **QUAD-MAINSTREAM** | 多周期买点：基础信号 `QuadMsUprBfMin` / `QuadMsUprBf` / `QuadMsUprBfMaj`（全称 `QUADRANT_MAINSTREAM_UPRISING_BEFORE` + `_MINOR`/`_MAJOR`）+ 权重 `QuadMsCredCpd`（全称 `QUADRANT_MAINSTREAM_CREDIT_COMPOUNDED`）；三周期各算买点 6 条件，15min=1 / 60min=2 / day=3，`leadin<sx` 再 +1，**按自然日相加 ≥4 为共振**。`analysis/mtf_buy_weight.py`；mainstream 提供 `code_mtf_buy_weight()` 与 `code_features(with_mtf_weight=True)`（默认不附加，约 0.7s/码） | ✅ 2026-11-24；000711 实测 01-20 W2 / 01-22 W4 / 01-23 W6、11-18 W6 / 11-19·20·23 W4；用例 `test_mtf_buy_weight.py` 8 条 |
| **GAR-WIRE** | `garPD/garPAnchor/garPCount/garPClr/garSLag` 由 review 的 60min stage 步**现算落库**（`analysis/stages.py::stage_garident_columns`：绝对带 ÷ 锚日 15:00 close，D49；`garSLag=event_lag(stage)`）；`HOUR_STAGE_COLUMNS` 6 列、`zen_snapshots_step` 按列取值。恢复 `bstStgBf/epStgBf` 与两个 GAR 类 xgb 特征的输入。⚠️ 4.4 的 `garSLag` 与其 `stage` 列不自洽（stage 恒定仍翻转），新值按文档公式；历史 11-17 之后需 `--review --force --from 2026-11-16` 补 | ✅ 2026-11-24；600519 对拍 stage 一致、garPD 差 ≤0.5%（K-Means）；用例 `TestStageGaridentColumns` 5 条 | 
| **MASQUAD-MAJ-WEK** | `MasQuadPwrlnMajMed/Mode` **落库**；`tools/plot_mas_quad_pwrln.py` 成为 Wek/Maj **统一入口**（同一 CSV `mas_quad_pwrln_wek_2018_2026.csv`（含 `MasPosOnhold`）+ `mas_quad_pwrln.png`，`--apply` 四列一起写 massive）；周线日期轴改 `stock_day.date`（原 `stock_reality_feat_day` 在 2022-06~10 只有 10 天）；窄口径 mainstream 不覆盖市场级 Maj。**同期修主流口径**：输出/落库只留最近 800 天（Maj 预热）、Wek 233 周 gate 进逐码列、review 的 week-end 15:00 与周线工具对齐 | ✅ 2026-11-24；CSV **2,156** 行（2018-01-02~2026-11-23）；massive **15:00 行 1,077** 四列全有值（日内行另由主流按 bar 写）；用例 `test_mas_quad_pwrln.py` 4 + 主流窗口/scope 6 + `_mature_mask` 3 |
| **ZEN-OLS-LAG** | `ZenOLSLag` **重新实现**（D89）：分段 = 60min `ZenPivotLag` 符号，段内 LIS/LDS 找波浪趋势，`in_pivot_marks=True`；峰点用 CZSC 分型（带确认时间 ⇒ 因果），**逐 target 2160 窗口**算（整帧 CZSC 有未来函数）；走 review 落库，不经 compact/mainstream | ✅ `analysis/zen_ols.py` + 10 条用例；`HOUR_ZEN_COLUMNS` 已含该列；**历史未回填**（review 跑到即算） |
| **REVIEW-PERF** | ChipDist 与 RegTree 的 JIT 核查：ChipDist 的 JIT 核在跑但被**762MB/次的超额数组**+**3.5M 元素 dict 往返**拖慢（2.14→0.27s/天）；RegTree 的 jit 建树器**写了没接**，review 走的是 np.matrix 纯 Python 版（0.81→0.007s/次）。实测 `day_core` 59.6→11.8s、`zen_snapshots` 69.7→41.1s | ✅ **D88 ⑤⑥**；用例 `TestArrayFastPath` 3 条 + `test_regtree_builder.py` 3 条；全量回填外推 2.5h→45min；旧 Python 实现加**不可过滤** stderr 提醒（`presentation.legacy_python_warning`，防误用不删代码） |
| **REVIEW-TIMEOUT** | `Slowest 1%` 里那些 7200s 的码：**不是数据不足**，是「review 从没跑过」的全量回填（day 缺 1,700 / 15min 缺 8,100+）。实测 `day_core` 3.97s/天（ChipDist 2.3 + Prophet 0.66 + _frame_features 1.05）、`zen_snapshots` 1.16s/目标 ⇒ 全新码约 **2.5 小时**。修了三处：①完成判据剔除外部列（`4QPwrlnWek`，真 bug，否则回填 2024 以前会静默漏算）②`raw_bars()` 向量化（`minor_features` -37%）③单码护栏 2h→3h ④**按年切块、块末落库**（`chunk_targets` 生成器 + `ReviewOptions.max_years=1`，切块 vs 一次性逐列验证一致）| ✅ **D88**；`test_persist_check_columns.py` 4 条；**未做**：ChipDist/Prophet/RegTree 逐日算法（需数值对拍，另立项） |
| **STATS-SLOWEST** | `Statistics` 增「Slowest 1%」：按 **sum(各码耗时)** 算占比（并行下 sum≫墙钟）、慢码带状态标记与 `CodeResult.meta` 分阶段耗时（review 写 `fetch/steps/write`）；「深挖单个码」用 `python -m GolemQ.cli.internal --review-runtime --runtime-code` | ✅ 2026-11-21；用例 `test_cases/pipeline/test_pipeline_slowest.py` 4 条 |
| **REVIEW-RT** | `--review` 盘中带 realtime 合成（**D90**）：`_fetch_frames` 三频率 `realtime=True`；快照型 tick 走 `_snapshot_ticks_to_bars` —— 60min/15min 按**区间增量**、日线取**最后累计**（股→手）+ 快照**日级** open/high/low；`recheck_days=1` 收盘后复核半根 bar | ✅ 2026-11-23；`test_snapshot_ticks.py` 8 条；⚠️ 除权日 qfq/原始价对齐**未处理**，订阅起点前的分钟 bar 合成不出来 |

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
