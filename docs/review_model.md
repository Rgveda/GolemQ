# `--review` 模型说明（日线 / 60min / 15min）

- **状态**：生效（2026-10-30）
- **权威列契约**：[`review_model_60min_feature_spec.md`](review_model_60min_feature_spec.md)（60min 79 列的口径以它为准）
- **决策**：[D78](decisions/D78.md)；审计：[review_model_asof_audit.md](review_model_asof_audit.md)

`--review` 是一个**逐码、逐快照、因果（as-of）** 的特征管线：每个目标时刻 T
只用 ≤T 的数据计算，**禁止**「整条序列跑完再按时间过滤」。
产出三张表：`*_reality_feat_day`、`*_reality_feat_60min`、`*_reality_feat_15min`。

```
                 ┌──────────── day_core (日线 832 根窗口) ────────────┐
 日线 K 线 ─────▶ │ zen/poly9/m9t/regtree、周线、coldstart、day_sample │ ──▶ _feat_day (18 列)
                 └───────────────┬───────────────────────────────────┘
                                 │ 18 列 + coldstart 12 列按日采样
                                 ▼
         ┌──────────── zen_snapshots (60min 2160 根窗口) ─────────────┐
60/15min │ zen 5 / minor 17 / se+中枢 2 / Renko 14 / 原生 11 / chip 4 │ ──▶ _feat_60min (79 列)
K 线 ───▶│ + day_sample 18 + coldstart 12（来自日线，按日映射）        │
         └───────────────┬───────────────────────────────────────────┘
                         │ 19 列按「每小时第 4 根」取样
                         ▼
                     _feat_15min (19 列)
```

---

## 一、通用口径（三族共用）

| 项 | 规则 |
|:--|:--|
| **as-of** | T 时刻只用 ≤T 的 bar；状态机类指标必须**逐 T 重算**，不能整序列跑完再过滤 |
| **窗口** | 60min 族 2160 根；日线 832 根；15min 2160 根 |
| **取数锚点** | 日线/60min 从 **2016-01-01**；15min 从 **2017-01-01**（保证头部 ≥2160 根暖机） |
| **交易日尾部** | 15:00 那根 = 日线已收盘，日频数据在它上面成立 |
| **盘中 3 根** | 10:30 / 11:30 / 14:00：日频列沿用**上一交易日**的值；60min 原生列各自 as-of |
| **命名** | 无后缀 = 本频原生；`Maj`/`Wek` = 日/周频；`Min` = 15min；`D` = ÷close 比值；`Nor`/`ClrNorm` = 0..1；`R` = **Reality**（早期命名：先有无 R 的带后视镜列，后为区分给无后视镜列加 R；**该阶段之后新增的列没有刻意区分，R 不是完备判据**） |
| **价格** | 价格量纲一律存 `*D` 比值（D49），attach 时乘回 |

---

## 二、日线模型（`_feat_day`，18 列）

**入口**：`pipeline/review/steps/daily.py::day_core_step` → `daily_snapshot_step`

| 内容 | 说明 |
|:--|:--|
| 输入 | 日线，`slice_asof(day, T, 832)`，逐 T |
| 计算 | `_frame_features(day)`：zen lag（CZSC）、Poly9、M9T、RegTree；周线特征用「已完成周」；coldstart 逐 T（STAGE/garP）、Prophet 逐 T |
| 输出列 | `zDLagMajR` `zPLagMajR` `zDLagWekR` `poly9MajD` `poly9NorMajR` `poly9WekD` `poly9NorWekR` `m9tMajR` `m9tLagMajR` `m9tWekR` `m9tLagWekR` `rTrePrcMajD` `rTreLagMajR` `rTreSlopMajD` **`cost5PrzD` `cost15PrzD` `cost85PrzD` `cost95PrzD`** |
| 采样缓存 | `_day_sample_rows`（同样 18 列，供 60min 采样）、`_chip_abs_rows`（chip 绝对成本）、`_coldstart_rows`（12 列） |
| 落库 | `*_reality_feat_day`，唯一键 `(code, date_stamp)` |

> Renko、`se`、`ZenPivotCnt` **不在日表**（D78 已移出）。

---

## 三、60min 模型（`_feat_60min`，79 列）

**入口**：`pipeline/review/steps/zen.py::zen_snapshots_step`（外加 `coldstart_step`、`day_sample`）

| 组 | 列数 | 来源 | 口径 |
|:--|--:|:--|:--|
| zen | 5 | `zen_snapshots` | `zDLagR` `zPLagR` `zDLagMinR` `zPLagMinR` **`ZenPivotLag`**；2160 根 as-of |
| minor | 17 | `zen_snapshots` | 15min 窗口（2160 根）算完按 60min 时刻对齐 |
| se / 中枢 | 2 | `zen_snapshots` | `se`（15:00 样本熵）、`ZenPivotCnt`（2160 根 as-of 窗口的因果中枢数） |
| Renko | 14 | `zen_snapshots` | 60min K 线、2160 根 as-of；价格列 `*D` |
| 60min 原生组 | 11 | `zen_snapshots` | `stage`、`m9t`/`m9tLag`、`poly9D`/`poly9NorR`、`rTrePrcD`/`rTreLagR`/`rTreSlopD`、`prophetLbD`/`prophetUbD`/`prophetClrNorm` |
| ChipDist | 4 | `chip_cost_by_day` | **960 根日线窗口**、逐日 as-of、**补 adj 缩放**；`cost5PrzD` `cost15PrzD` `cost85PrzD` `cost95PrzD` |
| 日族采样 | 18 | `day_sample` | 日表 18 列；chip 4 列按「绝对成本 ÷ 本 bar close」重算比值 |
| coldstart | 12 | `coldstart_step` | `bstStgBf` `epStgBf` `garPD` `garPCount` `garPClr` `garPAnchor` `garSLag` `m9tMaj` `poly9pwr3Maj` `prophetLbMajD` `prophetUbMajD` `prophetClrNormMaj`；garP 稀疏、`garPD`/`garPAnchor` **ffill** |

**落库**：`*_reality_feat_60min`，唯一键 `(code, time_stamp)`。

### 60min 内部关键规则

* **CZSC/Renko/M9T/Poly9/RegTree/Prophet/STAGE** 都按「每个交易日 15:00 用 ≤T 的
  2160 根重算」；盘中 3 根用上一交易日快照（因果）；
* **ChipDist 例外**：日频计算、960 根日线窗口、调用层补 `adj` 缩放、传帧 ≤1920；
* 实测证据见 `review_model_asof_audit.md`：CZSC 44/47/12、Renko 7/8 与 -18/-50。

---

## 四、15min 模型（`_feat_15min`，19 列）

**入口**：`pipeline/review/steps/minor.py::minute15_aligned_step`

* **不重算**：60min 的 `zen_snapshots` 在算每个 60min 目标时，已经把该小时 4 根
  15min 的 minor/zen 结果缓存在 `_minute15_rows`；15min 族只是**取样**；
* 只取「每小时第 4 根」（与 60min 时刻对齐的那根）；
* 列：`zDLagMinR` `zPLagMinR` + 17 个 minor（`AroUprsgLagMin` … `MsLeadinx5BfMin`）；
* 落库：`*_reality_feat_15min`，唯一键 `(code, time_stamp)`。

---

## 五、落库关系

### 表名

`reality_feat_name(family, market_type)`：

| market_type | 前缀 | 例 |
|:--|:--|:--|
| `STOCK_CN` | `stock_` | `stock_reality_feat_60min` |
| `ETF_CN` | `etf_` | `etf_reality_feat_60min` |
| 其余（INDEX/FUND） | `index_` | `index_reality_feat_day` |

### 写口与唯一键

| 族 | 唯一键 | 每行代表 |
|:--|:--|:--|
| day | `(code, date_stamp)` | 一个交易日的一行 |
| 60min | `(code, time_stamp)` | 一根 60min bar |
| 15min | `(code, time_stamp)` | 一根 15min bar |

* 唯一写口：`markets/StockCN/reality_save.save_reality_feat` → `datasource.writer.upsert_fields`；
* review 写库时 `write_nulls=True`（重算要能清掉旧值）；回填工具 `write_nulls=False`；
* 水位：`pipeline_watermarks`（`set_watermark`），只是运维信息，**不参与**「已持久化」判断；
* **已持久化判据**：`pipeline/review/persistence.py::missing_targets` —— 一行里任一契约列非空即算已写；
* 4.4 旧表（`stock_reality_features` 等）**不写不搬**（D75）。

### 列数一览

| 表 | 列数 |
|:--|--:|
| `_feat_day` | 18 |
| `_feat_60min` | 79 |
| `_feat_15min` | 19 |

---

## 六、互相映射关系

### 1) 日线 → 60min（按日采样）

| 源 | 目标 | 规则 |
|:--|:--|:--|
| 日表 18 列（`DAY_SAMPLE_COLUMNS`） | 60min 行 | 15:00 = **当日**日行；盘中 = **上一交易日**日行；chip 4 列用绝对成本 ÷ 本 bar close |
| coldstart 12 列 | 60min 行 | 同上（`coldstart_step` 映射） |
| ChipDist 4 列 | 日表 + 60min 行 | **日线循环内**用 960 根日线窗口算；日表存 ÷当日 close；60min 存 ÷本 bar close |

### 2) 60min → 15min（取样，不重算）

`zen_snapshots` 对每个 60min 目标 T 顺带算出「T 所在小时」的 4 根 15min 结果，
`minute15_aligned` 直接取样 → 15min 表。

### 3) attach（读侧：把落库结果贴回 60min 特征帧）

`analysis/reality_attach.attach_reality_feats(features_60min)`：

| 来源 | 方式 | `loaded` 键 |
|:--|:--|:--|
| 60min 表 coldstart | 按日取最后值 + 上一可用交易日 | `60min_cold` |
| 60min 表 `se`/`ZenPivotCnt` | 精确时间戳 join | `60min_extra` |
| 60min 表 Renko 14 | 精确时间戳 join | `60min_renko` |
| 60min 表 原生组 11 | 精确时间戳 join | `60min_native` |
| 60min 表 zen 5 | 精确时间戳 join | `60min_zen` |
| 60min 表 chip 4 | 精确时间戳 join（`*D` 后由 restore 乘回） | `60min_chip` |
| 60min 表 ChartScan | 精确时间戳 join（稀疏事件，不 ffill） | `60min_chart` |
| day 表 18 列 | `_map_causal`：盘中上一交易日、15:00 当日 | `day` |
| 15min 表 19 列 | `merge_asof` 取当前 bar 之前最后一个已完成值 | `15min` |
| 日线 → 周线 DEA | 上一已完成周 | `weekly` |
| Prophet 连续量 | 15:00 真值之后 ffill | `ffill_prophet` |
| 价格 `*D` | `restore_price_columns`（乘回 close） | `garident` 等 |

---

## 七、不可违反的约束（踩过的坑）

1. **禁止整序列跑完再按 known_dt 过滤**（CZSC 会改写历史结论）；
2. **60min 取数必须从 2016 锚点**，保证目标区间头部 ≥2160 根；
3. **ChipDist 必须补 `adj` 缩放，且传入帧 ≤1920 根**（否则 38s + 数值错误 / 偷偷重取含未来）；
4. **`garPD` 稀疏，必须 ffill 且 `garPAnchor` 同步**；
5. **Renko/se/ZenPivotCnt 只在 60min 表**，日表不得再写（旧列已 `$unset`）；
6. 新增列先改 `contract.py` + `review_model_60min_feature_spec.md` §九「算法 → 列」表，再实现。
