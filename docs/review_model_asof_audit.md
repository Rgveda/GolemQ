# Review 指标 as-of 审计（2026-10-30）

> 触发：用户发现 `ZenPivotCnt` 用「全历史跑一次再按 `known_dt` 过滤」与
> 「只用 ≤T 的数据重跑」**结果不同**（600519 2024-08-16：44 vs 47），
> 要求逐个核验 review 里还有哪些指标没有按日期 as-of、哪些暴露未来函数，
> 并核验 attach 的 NaN 形态、区分稀疏/稠密字段。

## 一、判定方法

对 T 时刻产出的值，看两件事：

1. **输入**是否只含 ≤T 的 bar（取数/切片层面）；
2. **算法**是否会在 T 之后**改写** T 时刻的结论（状态机层面：CZSC 会改笔、
   Renko 会重排砖块、全样本拟合会改参数）。

判据：`prefix(T)`（只用 ≤T 重跑）与 `full + 因果过滤`（全序列跑一次再过滤）
**逐值比较**；不同 ⇒ 存在后视镜。

## 二、逐步清单

| # | 步骤 / 指标 | 计算窗口 | as-of | 结论 |
|:--|:--|:--|:--|:--|
| 1 | `day_core`：zen lag / poly9 / M9T / RegTree（`_frame_features`） | `slice_asof(day, T, 832)`，逐 T | ✅ | 窗口内计算，无未来 |
| 2 | `day_core`：周线特征 | `slice_asof(day, 周末, 2160)`，逐完成周 | ✅ | 只用已完成周 |
| 3 | **Renko 14 列** | 原：日线**全帧一次跑**再取 T；现：**60min 2160 根 as-of** | ❌ → 已修（D78） | 600519 2024-08-16：`renkosBarLag` 全量 7 / as-of 8；`renkoLbarLag` -18 / -50。Renko 已归 60min 族，按目标日 15:00 快照、盘中沿用上一交易日 |
| 4 | `day_core`：coldstart 行（STAGE/BST/garP/m9t/Prophet） | `slice_asof(day, T, 832)`，逐 T | ✅ | Prophet 逐 T 在 ≤T 窗口拟合 |
| 5 | `coldstart_step`：日→60min 映射 | 盘中用上一交易日、15:00 用当日 | ✅ | 明确防未来函数 |
| 6 | `zen_snapshots`：`zDLagR/zPLagR` | `slice_asof(60min, T, 2160)` | ✅ | — |
| 7 | `zen_snapshots`：minor 17 + zen15 | `slice_asof(15min, T, 2160)` | ✅ | — |
| 8 | `zen_snapshots`：`se` | 只 15:00 算，窗口 2160，盘中沿用上一交易日 | ✅ | — |
| 9 | `zen_snapshots`：**`ZenPivotCnt`** | 原：**全序列一次跑**再按 `known_dt` 过滤 | ❌ → 已修 | 44 vs 47（2024-08-16）、47 vs 50（2025-06-30）。现改为**2160 根 as-of 窗口**（用户 2026-10-30 指定，与 `se` 同窗口） |
| 10 | `minute15_aligned` | 只从 #7 取样 | ✅ | 不重算 |
| 11 | `minor_signals_step`（**兼容入口，不在默认管线**） | 整条 15min 帧一次算 | ⚠️ | 若外部调用方拿它算历史行 ⇒ 有后视镜；默认 `build_minor_steps` 不注册它 |
| 12 | `runtime.compute_runtime` | 复用同一套 step | ✅ | 与 review 同步修复 |

## 三、attach 核验

| 路径 | 口径 | 结论 |
|:--|:--|:--|
| `_attach_60min` coldstart | 按日最后值 + 上一可用交易日（盘中） | ✅ 盘中不会拿到当日收盘 |
| `_attach_60min` ChartScan | 精确时间戳 join（稀疏事件，不 ffill） | ✅ NaN 是正常 |
| `_attach_60min` `se`/`ZenPivotCnt`（D78） | 精确时间戳 join（60min 行原生值） | ✅ 前提：review 把**当天 4 根都写**（见 §五 验证） |
| `_attach_day` | `_map_causal`：盘中用上一可用交易日，15:00 用当日 | ✅ |
| `_attach_15min` | `merge_asof` 取当前 bar 之前最后一个已完成值 | ✅ |
| `_attach_dea_norm_weekly` | 上一已完成周 | ✅ |
| `_ffill_prophet` | 只在 15:00 真值之后 ffill 连续量 | ✅ |

**重点排查的 Bug 形态**：`15:00 有值、同一天盘中 bar 为 NaN`。
`se` / `ZenPivotCnt` 的写入是**逐目标**（当天 4 根都写，盘中取上一交易日
15:00 的值），因此不应出现该形态；若出现，说明写入端漏了目标或 attach 用了
错误的对齐方式。

## 四、稀疏 / 稠密

**稠密**（每个 bar 都应有值；NaN 必须能解释）：

* 60min：`zDLagR/zPLagR/zDLagMinR/zPLagMinR`、17 个 minor、19 个 coldstart、
  **`se` / `ZenPivotCnt`**；
* day：`DAY_COLUMNS` 28 列（含 Renko 14 列）；
* 15min：19 个 minor。

**稀疏**（事件型，NaN 是正常）：

* ChartScan：`Class/Confidence/Coordinates`（三族各自）；
* `prophet*`：拟合失败时为 NaN（源数据不足/常数序列）。

**稠密字段 NaN 的已知根因**（按出现顺序）：

1. **暖机不足**：`<30` 根不出一行；`<832` 根日线算不出 poly9/RegTree；
   `se` 需要 ≥2 个 60min 收盘；`ZenPivotCnt` 需要窗口内有 ≥30 根。
2. **源数据缺失**：北交所 920xxx 的 60min 数据 2025-09 才起；
   上市初期 / 长期停牌没有 bar。
3. **算法失败**：Prophet 拟合异常、CZSC 异常（已 try/except → NaN 而非崩）。
4. **review 未跑到该 bar**：新上市/回填范围外，需跑 `--review` 补齐。

## 五、验证结果（2026-10-30 抽样）

工具：`python tools/audit_review_nan.py --codes ... --days 60 --family 60min`

抽样 8 码（600519 / 000001 / 000002 / 300750 / 601318 / 688981 / 002594 / 600000）：

| 形态 | 结果 | 根因 |
|:--|:--|:--|
| `zDLagR` 等 zen/minor：15:00 有值、盘中缺 | 仅 600519 命中 1 天 | 该日存在一根 **`00:00` 的脏 bar**（NaN），4 根正常 bar 都在 —— 不是写入漏目标 |
| **coldstart（`stage` 等 11 列）**：15:00 有值、盘中缺 | 修前：000001 3 天（10-22/23/26）、000002 4 天、600519 1 天、002594 1 天。**修复后抽样 8 码只剩 600519 的 1 天（那是 `00:00` 脏 bar）** | 增量运行的 cache 缺口（见下）；另需排除「上一交易日本身就是 NaN」的假阳性 |
| `prophetLbD/prophetUbD` 等 | 0% ~ 100% 缺失 | 已知 Prophet 覆盖 42%（HANDOFF）；拟合失败/未跑到的行留空，属稠密列里可解释的缺失 |
| `se` / `ZenPivotCnt` | 0 处「15:00 有值盘中缺」 | 写入端逐目标写 4 根，盘中取上一交易日 15:00 —— 符合预期 |
| zen/minor 15 列 | 8 码抽样中仅 600519 的 `00:00` 脏 bar 命中 | 脏数据，见下 |

### coldstart 盘中 NaN 的根因（已修）

`coldstart_step` 给盘中行映射的是**上一交易日**的 coldstart 行。原实现只在
「本次有盘中目标」时才把上一交易日放进 `need_dates` 去算 cache：

* 某次增量运行若只写了当天 **15:00** 行（盘中行已存在 → `missing_targets` 跳过），
  上一交易日的 coldstart 就不会被计算；
* 下一次运行写盘中行时，cache 里找不到上一交易日 ⇒ 盘中 NaN、15:00 有值。

**修复**（`steps/daily.py`）：`prev_dates` 对**所有** 60min 目标都加进去，
不再只对盘中目标加。已写入的坏行需要用修复模式重写（见下）。

### 数据修复

`tools/audit_review_nan.py --repair`：对命中的 `(code, 日)` 用 ≤(D-1) 的日线
重算 coldstart 行，只把 `COLDSTART_COLUMNS` 写回当天盘中 3 根（不动其它列）。
全量修复需要覆盖全部有 60min review 行的码（工具逐码跑）。

⚠️ **假阳性**：冷启动列盘中取的是**上一交易日**的值；若上一交易日该列本来就是
NaN（例如 `m9t` 那天算不出），盘中 NaN 是**正确**的。工具已按「上一交易日
非空」过滤（000002 2026-09-04 即此类）。

### 另一处数据卫生问题

600519 2026-10-16 有一根 **`00:00:00` 的 60min 行**，review 列全空。
60min 合法时刻只有 10:30/11:30/14:00/15:00；该行来源待查（疑似实时合并/旧写入），
属脏数据，不应参与任何计算。

## 六、结论

* review 里**曾经**没有 as-of 的两处：**Renko S/L 全族**、**`ZenPivotCnt`**，
  均已修复（D78：Renko 归 60min、与 se 同 2160 as-of 窗口）；其余指标要么
  逐 T 在窗口内计算，要么只是对已算结果的映射/取样。
* 60min 一体化行 = zen 4 + minor 17 + coldstart 16 + se/计数 2 + Renko 14
  + 日族 14（采样）+ M9T 2（60min 原生）= **69 列**；Poly9/RegTree/Prophet
  按用户口径「算过就免算」复用日频结果。
* `minor_signals_step` 是兼容入口，默认管线不注册；外部若用它算历史，需自行
  改为逐 T 调用。
