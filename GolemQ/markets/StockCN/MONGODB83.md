# StockCN 接入 MongoDB 8.3 时序库 —— 设计说明

> 建立日期：2026-09-20
> 相关提交：`eabcc8a`
> 相关代码：`GolemQ/markets/StockCN/__init__.py`、`GolemQ/markets/StockCN/kline83.py`、`GolemQ/services/persistence/*`

---

## 一、背景

分钟线已从旧 GolemQ 系统的 MongoDB 4.4 迁移到 8.3 的时序集合，但代码里**没有任何地方指向新库**，且 `services/persistence/*` 仍从 stub 模块 `GolemQ.fetch.kline` 导入（返回空结果）。后果是一条静默失败链：

```
空数据 → each_day[0] IndexError → 持久化检查中止 → 完整性监控对全部标的报 0%
```

本文记录修复该链的设计，以及为什么若干关键决策**不能照搬**参照实现。

---

## 二、MongoDB 拓扑（实测，非推测）

两台服务器，同一主机 `192.168.50.39`：

| 端口 | 版本 | 库 | 内容 |
|:--|:--|:--|:--|
| **27017** | MongoDB **4.4.30** | `golemq` | **83 集合**（`stock_day`、`StockCN_watchdog_*`、`concept_*`、`etf_*` …）。`stock_min` 已清空 |
| | | `quantaxis` | 26 集合（QUANTAXIS 行情源）|
| **57017** | MongoDB **8.3.2** | `golemq` | 仅 4 个运维集合（`function_checkins*`、`module_heartbeats*`），**无行情** |
| | | `golemq_stock_cn` | **迁移后的分钟线**，26 个条目（含 `system.buckets.*`）|

**两个关键错配（本次修复的对象）**：

1. `core/settings.py:276` 的 `DATABASE = QASETTING.client.golemq` 由 QUANTAXIS 配置驱动 → 指向 **4.4**。整个 `services/` 层用的就是它。
2. `markets/StockCN/__init__.py` 的 `self.DATABASE` 曾指向 `DATABASE.GolemQ_StockCN` —— 该库在 57017 上**实测 0 集合**。真数据在 `golemq_stock_cn`。

### 集合布局

```
golemq_stock_cn                       ← 历史行情（读多写少、全量保留）
  ├─ stock_1min | stock_5min | stock_15min | stock_30min | stock_60min    ← 时序集合
  └─ index_1min | index_5min | index_15min | index_30min | index_60min    ← 正在转时序（另一会话）

golemq_stock_cn_realtime              ← 实时行情（追加写、按日退役）
  └─ realtime_YYYY-MM-DD              ← 一天一个时序集合，保留约 14 天
```

时序规格：`{timeField: 'ts', metaField: 'code', granularity: 'minutes'|'hours', bucketMaxSpanSeconds: 86400|2592000}`

集合名推导统一为 `f'{market}_{frequency}'`，`market ∈ {'stock','index'}`。

### 实时库的集合布局（2026-10-08 定，取代 2026-09-21 的单集合写法）

* **按日**：`realtime_YYYY-MM-DD`，`{timeField:'ts', metaField:'code', granularity:'seconds'}`。
  名字格式是**硬契约** —— 保留策略（`markets/StockCN/tools.py` 的
  `purge_historical_collections`）正是按这个名字删整日集合的。名字生成收敛在
  `realtime.realtime_collection_name()` 一处，并有单测钉住「写入端产出的名字 =
  purge 要找的名字」。
* **一个日集合里混三条流**，靠行内 `source` 区分；它**同时是去重键的一部分**：

| `source` | 内容 | 写入方 |
|:--|:--|:--|
| `tencent_l1` | 腾讯全市场快照（**含五档**），2 秒一轮 | `realtime.sub_l1_from_tencent` |
| `tencent_l2` | 腾讯盘口（字段是 L1 的**真子集**），3 秒一轮 | `realtime.sub_l2_from_tencent` |
| `qmt` | MiniQMT 五档 —— **2026-10-01 起停服，当前无数据** | 同上（`QMT_REALTIME_ENABLED=False` 已关） |

* 写入**先删后插，但只在「本进程首次见到该 `(source, code)`」时删**：
  `delete_many({'code':…, 'ts':…, 'source':…})` + `insert_many`
  —— 时序集合不能 upsert、也不拦重复（见 `PITFALLS.md` P14）。
  ⚠️ 每轮无条件删是跑不动的：实测同规模 `delete_many` 要 **3.7 s** 而
  `insert_many` 只要 **0.95 s**，L1 却是 2 秒一轮（数字见 `HANDOFF.md`）。
* 读取按 `ts`（timeField）过滤 + 排序；`datetime`（北京时字符串）只作可读列与
  重采样（`GQ_data_tick_resample_1min`）用。
* 每条行内**同时保留** `ts`（UTC-aware）与 `datetime`（北京时），时区换算仍只经
  `kline83.bj_date` 一处。

---

## 三、常量设计

**决策：库名硬编码在 `markets/StockCN`，连接 URI 仍取自配置。**

```python
# markets/StockCN/__init__.py
mongo_uri = GQSETTING.get_config('MONGODB', 'uri')   # ~/.GolemQ/settings/config.ini
DATABASE = GQ_util_mongodb_client(mongo_uri)          # → 57017

GOLEMQ_STOCK_CN_NAME = 'golemq_stock_cn'
GOLEMQ_STOCK_CN = DATABASE[GOLEMQ_STOCK_CN_NAME]
```

**理由**：StockCN **就是** A 股市场，其存储库名是市场定义的一部分，不随部署环境变化，故归市场所有、就地硬编码。而"连到哪台服务器"是部署信息，归配置所有。

> ⚠️ 注意 `[MONGODB]` 目前**只有 `uri` 键，没有库名键** —— 这是刻意的，不要"顺手"把库名挪进配置。

**实时库句柄**（2026-10-08 更新：先前的"待定"已定，此处先前记的是旧状态）：

```python
self.DATABASE = GOLEMQ_STOCK_CN             # golemq_stock_cn（历史行情）
self.GQREALTIME = GOLEMQ_STOCK_CN_REALTIME  # golemq_stock_cn_realtime（实时）
```

`DATABASE.GolemQ_StockCN_REALTIME` 在 8.3 上**实测 0 集合**，故不用它；库名同样是
硬编码的市场定义（`__init__.py:81-82`）。

---

## 四、读取器设计（`markets/StockCN/kline83.py`）

### 4.1 按 `ts` 查，不按 `time_stamp`

时序集合的加速**只对 `timeField`(`ts`) 与 `metaField`(`code`) 生效** —— 规划器靠它们做**分桶剪枝**。

`time_stamp`(int32) 虽是 unix 秒、与时区无关，但规划器**不知道它与时间单调**，用它过滤只能走普通二级索引，拿不到时序加速。故一律按 `(code, ts)` 查、按 `ts` 排序。

### 4.2 时区收敛到一处

裸时间**绝不能**直接查 `ts` —— pymongo 会把 naive 当作 **UTC**，区间静默偏移 8 小时（查北京 10:00 那根会得到 0 条）。

```python
def _bj_date(x):
    t = pd.Timestamp(x)
    if t.tzinfo is None:
        t = t.tz_localize('Asia/Shanghai')   # 裸时间一律按北京时间理解
    return t.tz_convert('UTC').to_pydatetime()
```

`start` / `end` 一律按**北京时间**解释；返回索引是 **naive 北京时间**，与下游 `services/persistence/` 的口径一致。

`end` 只给到日（10 字符）时补到当天 23:59:59 —— 否则 `start=end='2017-03-14'` 会塌成零长区间而查空。

### 4.3 必须重命名 `vol` → `volume`

迁移把成交量存为 **`vol`**，但 **QUANTAXIS 口径与全部下游消费方用的是 `volume`**：

- `markets/StockCN/fetch.py:829` —— `data_min.data['volume']`
- `GolemQ_old/analysis/ChipDistribution.py:65` —— `VOLUME = 'volume'`，`:463` 用 `amount/volume`
- `ChipDistribution_jit.py:959`、`chip_dist.py:496` 同样

**不重命名，真实消费方会静默拿不到该列。**

### 4.4 返回形状必须是 `(ts, code)` 两层 MultiIndex

`services/persistence/_stock.py:119` 与 `_daily.py:122` 都做：

```python
each_day = sorted(kline.index.get_level_values(level=0).unique())
```

`_review.py:113` 还用 `level=1` 取 code。

**⚠️ 参照实现不能直接照搬**：`GolemQ_old/scribe/min_migrate83.py:200-264` 的 `get_kline_price_min_v8` 返回**带 `ts` 列的扁平表**，不是 MultiIndex。本模块补了一层形状适配。

### 4.5 空值契约**刻意不对称**（勿"修正"）

| 函数 | 空时返回 | 理由 |
|:--|:--|:--|
| `get_kline_price_v3`（日线）| **`None`** | `_daily.py:105` 有 `if data_baseline is None` 分支；`_stock.py:87` 预先初始化了 `kline_daily_baseline = None`。返回 `None` 命中两条既有保护 |
| `get_kline_price_min`（分钟）| **空 DataFrame** | `_stock.py:138` 是 `kline_hour_baseline = hour_baseline.data` —— **既无 `None` 判断、也未预先初始化该变量**。返回 `None` 会抛 `AttributeError` 逃出 `try`，最终在 `:288` 炸成 `UnboundLocalError` |

**空表也保留两层 MultiIndex**。若返回裸 `pd.DataFrame()`（单层空 Index），下游 `get_level_values(level=1)` 会抛 `Too many levels` —— 那是**形状错误**，会被误读成 bug；而形状正确的空表让调用方既有的 `len(...) < 1` 判空逻辑正常工作，如实表达"这只标的没有数据"。

### 4.6 市场路由与已知局限

沿用既有 `is_stock_cn()` 分类器（`00xxxx` 这类代码光看数字有歧义），由 `kline83.market_prefix()` 映射：

| `market_type` | 集合族 |
|:--|:--|
| `ETF_CN` | **`etf_*`**（2026-09 起独立） |
| `INDEX_CN`、`FUND_CN` | `index_*` （`FUND_CN` = 50x 封基/LOF/分级，**有意**与指数同族） |
| 其余 | `stock_*` |

**⚠️ 2026-09 变更**：ETF 此前与真指数**共用** `index_*`（老树 `save_qa.py` 刻意「与 QUANTAXIS 一致」），现已拆成独立的 `etf_*` 与 `MARKET_TYPE.ETF_CN`。同批修正的还有**深市号段误判** —— `150`(分级子份额) / `16x`(LOF) / `180`(REITs) / `20`(B股) 曾被一并判成「深交所ETF基金」。核实来源与完整对照见 `MIGRATION_STATUS.md`。

**⚠️ 已知局限**：指数集合里的 `code` 是 `'000001'`（上证指数），而 `is_stock_cn('000001')` 判为 **stock**（平安银行）。按代码路由会**返回平安银行的行情冒充上证指数** —— 比返回空更危险。调用方若已知市场类型，**应显式传 `market_type=`**。

---

## 五、数据契约

单条文档（`stock_1min` 实测样本）：

| 字段 | 类型 | 说明 |
|:--|:--|:--|
| `ts` | BSON Date (UTC) | **时序 timeField**，桶剪枝依赖它 |
| `code` | str | **metaField**，裸 6 位代码 |
| `datetime` | str | `'2017-11-24 09:31:00'` 北京时间字符串 |
| `date` | str | `'2017-11-24'` |
| `date_stamp` | int32 | 当日 00:00 的 unix 秒 |
| `time_stamp` | int32 | bar 结束的 unix 秒 |
| `open/high/low/close` | float | |
| `vol` | **int32** | 读取时重命名为 `volume` |
| `amount` | float | |

`index_*` 另含 `up_count` / `down_count`(int32)。

### ⚠️ `vol` 必须是 int32，**绝不能是 int16**

实测最大值：

| 集合 | `vol` 单日最大 | int16 上限 32767 |
|:--|--:|:--|
| `stock_1min` | **35,455,100** | ❌ 溢出约 1000 倍 |
| `index_1min` | **131,132,928** | ❌ 溢出约 4000 倍 |

`date_stamp` / `time_stamp` 也只在 int32 范围内。（`int16` 一说系笔误，已确认应为 `int32`。）

---

## 六、遗留缺口

| 项 | 状态 |
|:--|:--|
| ~~**日线未迁移**~~ **已作废**（2026-10-08） | 先前那条是**错的**：`stock_day` **17,893,343** 行、`index_day` 4,564,152 行、`etf_day` 3,220,050 行 —— 日线**迁全了**。当时据以判断的计数来自时序集合上不可靠的 `$collStats count`，见 `PITFALLS.md` P15。⚠️ **实测真有洞的是 ETF**：`etf_day` 抽查 1,689 只里 **739 只有缺日**（`--save-coverage` 可复现）|
| **指数集合数据不全** | ~~先前的说法~~ 实测 `index_day` 有 8,733 个 code、`index_1min` 亦然（2026-10-08）。⚠️ 但**指数分钟与 pytdx 不同源**：`vol` 比值 14–18 非常数、`close` 精度也不同 → `--save tdx` 里指数分钟**按 pytdx 原值写**（`INDEX_MIN_VOL_SCALE`），边界处会与存量跳变 |
| **概念 K 线仍是 stub** | `GolemQ.fetch.concept` 无真实实现；真实版本只在 `GolemQ_old/fetch/concept.py:865`（读 4.4）。`_concept.py:56` 已加 TODO |
| **下游 stub 未解** | `load_massive_reviews` / `attach_reality_features` / `align_kline_timeline` 仍是 stub，端到端检查会停在这些点 —— 与 kline 通路无关 |

### 附：本次发现的既有缺陷（非本次引入）

- `_review.py:210` format 字符串占位符比参数多：`'{:.2%},{:.2%},{:.2%} Total:{}'` → `IndexError: Replacement index 3 out of range`
- `core/settings.py:249` 的 `setup_mongodb_config()` 写入 `~/.GolemQ/config.ini`，而读取路径（`:46`）是 `~/.GolemQ/settings/config.ini` —— 写读不一致

---

## 七、验证方法

**禁止全表扫描**（`stock_1min` 逾 14 亿行）。一律「单 code + 窄时间窗」：

```python
from GolemQ.markets.StockCN import GOLEMQ_STOCK_CN
assert GOLEMQ_STOCK_CN.name == 'golemq_stock_cn'

from GolemQ.markets.StockCN.kline83 import get_kline_price_min, get_kline_price_v3

# 分钟线：活跃标的，默认窗口
r, _ = get_kline_price_min('600496')
assert r.data.index.nlevels == 2 and len(r.data) > 0
assert 'volume' in r.data.columns and 'vol' not in r.data.columns

# 空值契约
assert get_kline_price_v3('600496')[0] is None          # 日线未就绪
assert get_kline_price_min('002070')[0].data.index.nlevels == 2   # 退市股，形状仍正确
```

已实测：`600496` 取到 **2130 行**，覆盖 `2024-07-12 .. 2026-09-18`；退市股 `002070` 返回形状正确的空表；`get_kline_price_v3` 打印明确告警并返回 `None`。
