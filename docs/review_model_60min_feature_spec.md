# 60min 特征行规格（`--review` 模型的一部分，权威版）

- **状态**：生效（2026-10-30）
- **决策依据**：[D78](decisions/D78.md)（撤销 D74）
- **审计**：[review_model_asof_audit.md](review_model_asof_audit.md)
- **改本文件 = 改契约**；实现、回填、attach 一律以这里为准，不再口头补充。

---

## 一、总原则

1. **`*_reality_feat_60min` 每行是一体化特征行**（用户 2026-10-30）：
   zen / minor / se / ZenPivotCnt / Renko / M9T / Poly9 / RegTree / Prophet / STAGE
   都要在 60min 口径下产出。
2. **日频特征（带 `Maj` / `Wek` 标识）两表都留**：
   日表原生计算 → `*_reality_feat_day`；同时按日采样进 `*_reality_feat_60min`
   （两表同值，不会分叉）。
3. **每类指标另有 60min 原生版**（同算法、输入换 60min K 线），**只落 60min 表**。
4. **所有指标逐日 as-of**：
   - T 时刻只能用 **≤T 的 bar**；
   - **禁止**「整条序列跑一次再按时间过滤」（CZSC/Renko 会改写历史结论）；
   - 窗口：`se` / `ZenPivotCnt` / Renko / M9T / Poly9 / RegTree / Prophet / STAGE
     统一 **2160 根 60min**；zen/minor 2160；日族 832 根日线；
   - 60min 取数一律从 **2016 锚点**，保证目标区间头部有 ≥2160 根暖机。
5. **交易日尾部（15:00）**：日线已收盘，日频数据在这根成立；
   **盘中 3 根（10:30/11:30/14:00）**：60min 原生列各自 as-of；日频列**沿用
   上一交易日**的值（用户 2026-10-30 确认）。
6. **回填频次**：历史按本规格全补（一次性）；之后每日 review 增量产出，
   不再整段重算。

---

## 二、列清单（`HOUR_COLUMNS` = 79）

### 2.1 60min 原生（只落 60min 表）

| 组 | 列 | 说明 |
|:--|:--|:--|
| zen | `zDLagR` `zPLagR` `zDLagMinR` `zPLagMinR` `ZenPivotLag` | 60min / 15min zen lag；中枢 lag |
| minor | 17 列（`AroUprsgLagMin` … `MsLeadinx5BfMin`） | 15min 窗口、60min 对齐 |
| se / 中枢 | `se` `ZenPivotCnt` | 15:00 样本熵、2160 窗口中枢计数 |
| Renko | 14 列（`renkoSpriceD` … `renkoLboostLag`） | 60min K 线、2160 as-of |
| M9T | `m9t` `m9tLag` | 60min 原生 |
| Poly9 | `poly9D` `poly9NorR` | `poly9D` = Poly9 ÷ close；`poly9NorR` = 归一化 0..1 |
| RegTree | `rTrePrcD` `rTreLagR` `rTreSlopD` | 价格/斜率 ÷ close |
| Prophet | `prophetLbD` `prophetUbD` `prophetClrNorm` | 60min 拟合；LB/UB 是 ÷close 比值 |
| STAGE | `stage` | 60min 原生阶段机 |

### 2.1b ChipDist（**日频**，在**日线循环**内计算；日表 + 60min 都落）

`cost5PrzD` `cost15PrzD` `cost85PrzD` `cost95PrzD` —— 用户 2026-10-30 纠正：
**ChipDist 是日频计算**，不是 60min 原生。

* 输入：**960 根日线窗口**（用户 2026-10-30 指定），每个需要的交易日单独算一次；
* ⚠️ **必须补 `adj` 价格缩放**：`ChipDistribution.get_data()` 只在**自己重新取数**
  时按区间 ÷100/50/10/5；直接传窗口不缩放会让分箱数爆炸（38s）且数值错
  （1177.81 vs 正确 1222）。调用层（`hour_extra.chip_cost_by_day`）按同样阈值
  缩放，算完再乘回；
* ⚠️ **传入帧不得超过 1920 根**：`get_data` 见 `len>1920` 会**丢弃传入数据、
  自己重新取数**（`realtime=True`，含未来）——as-of 场景绝对不能用；
* 成本：960 窗口 + adj 修正后 **≈0.9s/天/码**（实测 600519）；
* 取样规则：15:00 = 当日日线行，盘中 = **上一交易日**日线行；
* 落库：绝对成本 ÷ **该 60min 行 close** 存 `*D`（restore 乘回）。

### 2.2 日频采样（日表原生 + 60min 采样，两表同值）

`DAY_COLUMNS` = 14 列：

| 组 | 列 |
|:--|:--|
| zen-daily | `zDLagMajR` `zPLagMajR` |
| zen-weekly | `zDLagWekR` |
| M9T | `m9tMajR` `m9tLagMajR`（日）、`m9tWekR` `m9tLagWekR`（周） |
| Poly9 | `poly9MajD` `poly9NorMajR`（日）、`poly9WekD` `poly9NorWekR`（周） |
| RegTree | `rTrePrcMajD` `rTreLagMajR` `rTreSlopMajD` |
| ChipDist | `cost5PrzD` `cost15PrzD` `cost85PrzD` `cost95PrzD`（日频 960 窗口） |

### 2.3 coldstart（日频采样，13 列中的 12 列）

`bstStgBf` `epStgBf` `garPD` `garPCount` `garPClr` `garPAnchor` `garSLag`
`m9tMaj` `poly9pwr3Maj` `prophetLbMajD` `prophetUbMajD` `prophetClrNormMaj`
（`stage` 已改为 60min 原生，不在 coldstart。）

**合计**：zen 5 + minor 17 + se/中枢 2 + Renko 14 + 原生 11
+ 日采样 18（含 ChipDist 4）+ coldstart 12 = **79**。

---

## 三、命名

| 形态 | 含义 |
|:--|:--|
| 无后缀（`m9t` / `poly9D` / `rTrePrcD` / `prophetLbD` / `stage`） | **60min 原生**（本频，60min 表） |
| `Maj` / `MajR` | **日频**计算（日表原生 + 60min 采样） |
| `Wek` / `WekR` | **周频**计算（日表原生 + 60min 采样） |
| `Min` / `D`（末位） | `Min` = 15min 口径；`D` = ÷该行 close 的比值 |
| `Nor` / `NorR` / `ClrNorm` | 归一化到 0..1 |
| `R`（`zDLagR` / `m9tMajR` / `rTreLagR` / `poly9NorR`） | **Reality**（早期命名）：最初先做的是**不带 R、带后视镜**的列；之后为区分才给**无后视镜**的列加 `R`。⚠️ **该阶段之后新增的契约列（`se` / `ZenPivotCnt` / `ZenPivotLag` / Renko 14 / `cost*PrzD` / 60min 原生组）没有刻意区分**（加不加 `R` 不代表语义），因此**不能用「有没有 R」判断是否存在后视镜效应** —— review 全表一律按 as-of 产出 |

> 名字是新树自造；`Maj` 不是拼写错误。改名前先查 `GLOSSARY.md` 与 D77。

---

## 四、归一化与填充

1. **价格量纲一律存比值**（D49）：
   `*D` = 原值 ÷ **同一行 close**；attach 时 `restore_price_columns` 乘回。
2. **Garident Price**：
   - `garPD` = 绝对价格带 ÷ **锚 bar** 的 close（`garPAnchor` 记录锚日）；
   - **稀疏**：只在定带事件上产生，NaN 是正常；
   - **向后填充（ffill）**：`garPD` 要沿时间轴向后填充到下一事件；
     填充时 **`garPAnchor` 同步 ffill**，保证还原用的是同一锚 close
     （用户 2026-10-30：「Garident_Price_DUMMY 归一化，向后填充，同时注意
     对应数据归一化」）。
3. `*Nor*` / `prophetClrNorm` / `poly9NorR` 是 0..1 归一化强度，**不是**价格，
   不参与 `*D` 还原。

---

## 五、稀疏 vs 稠密

| 类别 | 列 | NaN 是否正常 |
|:--|:--|:--|
| 稠密 | zen/minor/se/中枢/Renko/M9T/Poly9/RegTree/Prophet/STAGE、日采样 14、coldstart（除 garP） | 否，NaN 必须能落到「暖机不足 / 源数据缺失 / 算法失败」 |
| 稀疏 | `garPD` 及 `garPAnchor`（定带事件）、ChartScan 四列 | **是**；garP 需 ffill |

---

## 六、回填与验收

1. 工具：`tools/backfill_hour_extra.py --apply --redo --workers N`
   —— 一个码一次算完：se/中枢、Renko、60min 原生 11 列、日采样 14 列；
2. 日表清理：`--unset-day --unset-day-renko --apply`；
3. 验收：
   - 任一 T：各 60min 原生列 = 「≤T 最近 2160 根重算」的值；
   - 日采样列 = 日表同日值（盘中 = 上一交易日）；
   - 稠密列除暖机/源缺失外无 NaN；`garPD` ffill 后无空洞；
   - `audit_review_nan.py` 抽检 0 处异常；
   - 全量测试通过。

## 七、成本（实测，90% 负载下）

| 组 | 每交易日 15:00 一次 |
|:--|--:|
| `_frame_features`（Poly9+RegTree+zen+M9T） | 1.2s |
| `_core_daily`（STAGE+garP） | 1.6s |
| Prophet（60min 拟合） | 1.0s |
| ChipDist（960 根日线窗口，逐日，adj 缩放） | ≈0.9s / 天 / 码 |
| **合计** | **≈3.8s / 天 / 码** + 2s/码 |

历史 510 天 × 2393 码 ≈ 1290 CPU 小时；12 进程约 **110 小时**；
增量每天约 **13 分钟**（12 进程）。

## 八、与 4.4 老表的关系（收录准则）

**4.4 `stock_reality_feat_60min` 是历史宽表（22,265,420 行 / 抽样 64 个字段），
不作为 8.3 契约依据。**

**收录准则（用户 2026-10-30）**：

> 8.3 `--review` 只落库**存在后视镜效应**的指标 —— 算法会随后续 bar 改写
> 历史结论的那一类，目的是**防止未来数据污染**。

按此准则，需要 review/as-of 的是：

* 状态机可被改写：**CZSC**（笔/中枢）、**Renko**（砖块/趋势）；
* 全样本/滚动拟合：**Prophet**、**Poly9**、**RegTree**；
* 事件/阶段累计：**STAGE**、**garP**、**M9T**、**se / ZenPivotCnt**。

纯因果、只看过去的滚动量**不因为“老表有”就搬进来**。

因此，4.4 老表里未进新契约的 51 列**不补**：

| 族 | 列数 | 例子 |
|:--|--:|:--|
| 主流 60min | 12 | `MsLeadinBf` + Maj、`MsLeadinVar5/Entry/Sko/x3Bf/x5Bf` |
| BS 星线 60min | 8 | `BSBuBf/BSSelBf/BSStarUpBf/BSStarDwnBf` + Maj |
| Mswing | 5 | `Mswing/MswingEMA/MswingMaj/MswingUprLag/MSEMAMaj` |
| OmniPath | 5 | `OmniPathUpR0Coef*` / `R1Coef*` |
| QLev | 9 | `QLevMACDLagD`、`QLevBst*`、`QLevEng*` |
| Quad | 9 | `QuadPldTick*`、`QuadMsPldBf`、`QuadBstPldLag*` |
| 其他 | 3 | `rStpls0Bf`、`rStplsIBf`、`cost5BstBf` |

已覆盖/改名的老列：`SED` → `se`（实测 0.259 vs 0.26，同一口径）、
`BST_STG_BF` → `bstStgBf`；`Class/Confidence/Coordinates/ChartScaned` 由
ChartScan 单独写（attach 时改名 `Cls60/Conf60/Coord60/ChartScaned60`），
`ClsMin…` 在 15min 表，`position_level` 属组合层、不在 review。

### 8.1 计算策略：只有后视镜指标才走逐日 as-of

用户 2026-10-30 补充：

> 其他指标没有后视镜效应，向量化计算快一百多倍。

因此分两条链：

| 链 | 适用 | 算法 |
|:--|:--|:--|
| **review 逐日 as-of** | CZSC / Renko / Poly9 / RegTree / Prophet / STAGE / garP / M9T / se / ZenPivotCnt —— 这些**会随后续 bar 改写历史结论** | 每个交易日 15:00 只用 ≤T 的 2160 根重算（贵，但必须） |
| **向量化一次算** | 纯因果滚动量（EMA/滚动窗口/lag 计数等，例如老表的 `MsLeadin*`、`BS*`、`QLev*`、`Mswing*`、`OmniPath*`） | 整条序列一次向量化，**快 100 倍以上**；它们在 T 的值与「只用 ≤T 重算」逐值相同（前缀稳定），**不需要**进 review 的逐日重算 |

> ⚠️ 判断标准不是「老表有没有」，而是**算法是否会在 T 之后改写 T 时刻的值**。
> 会 → 必须逐日 as-of；不会 → 向量化一次算，别用 review 的慢路径。

---

## 九、算法 → 列 映射（防漏清单）

> 用户 2026-10-30：「每个算法涵盖了哪些 columns，要写清楚，否则还会缺漏。」
> **新增/修改任何算法，必须同步改这张表。**

| 算法 | 覆盖列 | 状态 |
|:--|:--|:--|
| **CZSC（60/15min zen）** | `zDLagR` `zPLagR` `zDLagMinR` `zPLagMinR` | ✅ 契约 |
| **CZSC 中枢计数** | `ZenPivotCnt` | ✅ 契约 |
| **CZSC 中枢 lag**（`calc_pivot_timing_lag`） | `ZenPivotLag` | ✅ 契约 |
| CZSC 日/周 | `zDLagMajR` `zPLagMajR` `zDLagWekR` | ✅ 日表 |
| **Renko** | `renkoSpriceD` `renkoLpriceD` `renkosBarLbD` `renkosBarUbD` `renkolBarLbD` `renkolBarUbD` `renkoSbar` `renkoLbar` `renkoBar` `renkoOpt` `renkosBarLag` `renkoLbarLag` `renkosBstLag` `renkoLboostLag` | ✅ 契约 |
| **Poly9** | 原生 `poly9D` `poly9NorR`；日/周 `poly9MajD` `poly9NorMajR` `poly9WekD` `poly9NorWekR`；`poly9pwr3Maj`（日 norm） | ✅ 契约 |
| **RegTree** | 原生 `rTrePrcD` `rTreLagR` `rTreSlopD`；日 `rTrePrcMajD` `rTreLagMajR` `rTreSlopMajD` | ✅ 契约 |
| **Prophet** | 原生 `prophetLbD` `prophetUbD` `prophetClrNorm`；日 `prophetLbMajD` `prophetUbMajD` `prophetClrNormMaj` | ✅ 契约 |
| **STAGE** | `stage`；伴随 `bstStgBf` `epStgBf` `garPD` `garPCount` `garPClr` `garPAnchor` `garSLag` | ✅ 契约 |
| **M9T** | 原生 `m9t` `m9tLag`；日 `m9tMaj` `m9tMajR` `m9tLagMajR`；周 `m9tWekR` `m9tLagWekR` | ✅ 契约 |
| **se** | `se` | ✅ 契约 |
| **minor（15min）** | `AroUprsgLagMin` `1QUprLagMin`…`4QUprLagMin` `4QPwrlnMin` `4QPwrlnLedInBfMin` `4QPwrlnLedOutBfMin` `BSBuBfMin` `BSSelBfMin` `BSStarUpBfMin` `BSStarDwnBfMin` `MsLeadinVarMin` `MsLeadinEntryMin` `MsLeadinBfMin` `MsLeadinx3BfMin` `MsLeadinx5BfMin` | ✅ 契约 |
| **ChipDist（日频）** | `cost5PrzD` `cost15PrzD` `cost85PrzD` `cost95PrzD`（960 根日线窗口，adj 缩放） | ✅ 契约（**日线循环内**计算；日表 ÷当日 close，60min ÷本 bar close） |

### 已知缺口（待补）

1. ~~`ZenPivotLag`~~ —— **已补**：与 `ZenPivotCnt` 同一批 CZSC 中枢，
   `causal_pivot_series` 展开 ZD 后做事件 lag；
2. ~~ChipDist `cost5/15/85/95`~~ —— **已补**：`cost*PrzD`（÷close），
   前缀稳定，**每码整条序列算一次**（≈2s/码），不进逐日重算。
