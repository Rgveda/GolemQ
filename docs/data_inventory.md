# 数据暗线：这套系统里埋着哪些数据源

> **目的**：把「数据在哪、有多长、谁是谁的母数据、哪里会骗人」写清楚。
> 2026-10-28 实测，只写**验证过**的事实；没验证的标「待核」。
>
> 为什么要有这份：`ZenPivotCnt` 覆盖率一度只有 33%，第一反应是「4.4 的
> 60min 没搬全」。**实测结论是数据早就搬全了，是我们自己的 review 只算了
> 900 天窗口** —— 数据层比代码层深得多，不写下来下次还会重演。

---

## 一、三个库

| 库 | 句柄 / 地址 | 装什么 |
|:--|:--|:--|
| `golemq_stock_cn` | `GOLEMQ_STOCK_CN`（`markets/StockCN/__init__.py`） | **行情主库**：day / 1/5/15/30/60min + reality_feat + 参考数据 |
| `golemq_stock_cn_realtime` | `GOLEMQ_STOCK_CN_REALTIME` | **滚动实时库**：`realtime_YYYY-MM-DD`，只保留最近若干天（实测 2026-10-27 时剩 10-16 ~ 10-27） |
| `golemq` | `core.settings.GOLEMQ` | 框架运维：心跳 / 签到 / 关注列表 / 订单 |
| `quantaxis`（4.4，`:27017`） | `core.migrate44.db44('quantaxis')` | **4.4 原始行情源**：`stock_min`（单集合按 `type` 存所有频率）、`stock_day`、`stock_adj`、`stock_xdxr` |
| `golemq`（4.4） | `core.migrate44.db44('golemq')` | **4.4 旧 review 特征**：`stock_reality_feat_{day,60min,15min}` 等（**未整体迁到 8.3**） |

⚠️ 4.4 通道**只读、只服务一次性搬运**（D33 修正 / D34），运行时代码禁止 import。

---

## 二、频率阶梯（8.3 实测行数，`estimated_document_count`）

| 族 | 1min | 5min | 15min | 30min | 60min | day |
|:--|--:|--:|--:|--:|--:|--:|
| stock | **2,152,208,584** | 485,423,914 | 164,035,029 | 82,009,021 | 66,215,536 | 17,988,060 |
| index | 529,064,414 | 121,219,712 | 41,060,075 | 20,507,863 | 10,232,085 | 4,587,923 |
| etf | 540,402,918 | 124,574,057 | 41,615,168 | 20,518,972 | 9,973,221 | 3,249,299 |
| fund | 93,579,970 | 22,744,201 | 7,633,450 | 3,768,278 | 1,835,729 | 826,926 |
| future | 9,405,137 | 5,088,761 | 2,721,117 | 1,702,223 | 867,233 | 282,220 |

**这些不是独立数据源，是同一条时间线的不同分辨率**：

- **1min 是母数据**（2017-11 起，逐码不同；000001/000711 是 2018-11 起）；
- **5/15/30/60min、day 都可由 1min 重采样**，工具是
  `analysis/timeseries.GQ_data_min_resample`（A 股分段：上午 9:30–11:30、
  下午 13:00–15:00 分别 resample）与 `GQ_data_min_to_day`；
- **60min / day 另有更长历史**（不必从 1min 推）：
  - `stock_60min` 最早 **1999-07-26**（000001 / 000711）；
  - `stock_day` 最早 **1990-12-19**（600602）；
  - 1min 反而只有 2017-11 起。

> **所以 review 的 800 天 / 2560 天是「写库目标窗口」，不是「数据上限」。**
> `TARGET_LOOKBACK_DAYS` 只限制**写哪些行**；取数窗口由 `REVIEW_ANCHOR_DAY`
> 决定。D74 起日族的 60min 取数已改为 **2016 锚点**。

---

## 三、读取入口（别自己拼集合名）

| 需求 | 入口 |
|:--|:--|
| 日线 | `get_active_market().get_kline_price_v3(code, start, end, if_fq=...)` |
| 分钟线（任意频率） | `get_active_market().get_kline_price_min(code, start, end, frequency='1min'/'60min', if_fq=...)` |
| 内部读库 | `markets/StockCN/kline83._read_timeseries`（按 `(code, ts)` 查时序集合，**强制带 code**） |
| 复权 | 读时由 `_apply_adjustments` 按 `if_fq`（默认 `qfq`）套 `stock_adj`；**库里存的是 bfq 原始价** |
| 实时合并 | `_merge_realtime`：读 `realtime_YYYY-MM-DD`，只补当日；⚠️ 北交所 `920xxx` 无 L1（腾讯不供） |

时间约定：

- `ts` = **UTC naive**（时序集合的 `timeField`，读取时转北京）；
- `datetime` = 北京时间字符串；
- `time_stamp` = 真实 UTC epoch；`date_stamp` = 北京零点 epoch；
- **审计抽样**（000001 2026-10-09）：`datetime` / `time_stamp` 一致
  （10:30 / 1791513000），未发现错位。

---

## 四、暗线清单（容易踩的）

1. **`*_removed` 集合不是垃圾**：`stock_1min_removed` 391 万行、
   `stock_15min_removed` 6.3 万行等，是 maintenance 把**停牌伪 0 bar**
   从主集合移出去的（`GQ_purge_suspended`）。对比「缺多少」时要先排除它们。
2. **时序集合是 view**：真正的存储是 `system.buckets.*`（`timeField=ts`,
   `metaField=code`）；`system.views` 里是视图定义。直接往 `system.buckets.*`
   写是错的，插 `stock_60min` 即可。
3. **4.4 `quantaxis.stock_min` 是单集合多频率**：靠 `type` 字段区分
   `1min/5min/15min/30min/60min`。**它有重复行，也有错标频率**——
   实测 `type='60min'` 里混着 09:45/10:00/13:15 等 15min 时刻的行。
   拿它跟 8.3 比「谁少了」时，必须先按 `(code, time_stamp)` 去重、
   且只认标准 60min 时刻（10:30 / 11:30 / 14:00 / 15:00）。
4. **4.4 旧 review 特征：明确不搬（D75）**：
   `golemq.stock_reality_feat_60min`（4.4）000001 从 **2021-10-13** 起、
   4,891 行；8.3 同码 review 行从 **2024-08-13** 起、2,128 行。
   这是**特征层**差距，不是行情层。用户 2026-10-28 定：**不搬** ——
   旧表不是归一化数据，复权锚冻结在过去，无参考意义；8.3 的历史缺口是刻意的。
   已搬的只有 CV 列（D51，事件型稀疏数据，不涉及价格归一化）。
5. **`ZenPivotCnt` 的 33% 不是数据问题**：它是 60min CZSC 中枢数，
   旧实现只取 900 天（review 60min 目标 800 天）；数据本身有 1999 年起的
   60min。D74 改为**全历史**计算后覆盖率 **99.94%**（缺 1,795 行 =
   2 只北交所 920xxx，其 60min 数据 2025-09 才开始）。计数会**变大**
   （000001 2026-10-27：8 → 47；000711 2026-10-23：5 → 28），
   这是口径修正，不是回归。
6. **复权是读时动作**：DB 里的 OHLC 是 bfq；同一根 bar 用 `if_fq='qfq'`
   读出来数值不同。比较「库 vs 接口」前先对齐复权口径。

---

## 五、2026-10-28 审计结论（抽样）

| 项 | 000001 | 000711 | 600519 |
|:--|--:|--:|--:|
| 8.3 `stock_60min` 唯一 bar | 25,502 | 22,804 | 22,849 |
| 4.4 `stock_min(type=60min)` 唯一 | 26,256 | 25,581 | 24,674 |
| **4.4 多出的「标准时刻」60min** | **0** | **0** | **0** |
| 4.4 多出的非标准时刻（15min 混入等） | 755 | 2,778 | 1,827 |
| 8.3 `stock_1min` 唯一 bar | 463,195 | 462,932 | — |
| 4.4 `stock_min(type=1min)` 唯一 | 463,209 | 462,958 | — |
| 1min 差异 | 14 | 26 | — |

**30 码随机抽样**（2026-10-28）：4.4-only 共 18,290 行，其中「标准时刻」129 行；
这 129 行里约 120 行是**当天（10-28）8.3 还没 `--save` 的 bar**，其余是
2026-10-09 / 10-20 的**特殊交易日口径差**：

- 8.3：`10:30 / 13:00 / 14:00 / 15:00`（下午 13:00 开盘）；
- 4.4：`10:30 / 11:30 / 14:00 / 15:00`（把 13:00 标成 11:30）。

⇒ **8.3 的 60min / 1min 行情没有成块缺失**；4.4 多出来的行是重复行、
错标频率与特殊日口径差。**不需要再搬行情**（2001–2015 的非标准时刻行
也不该混进 60min；2026-10-09 这类日不要按 4.4 把 13:00 改成 11:30）。

---

## 六、怎么查（只读片段）

```python
from GolemQ import get_active_market
db = get_active_market().get_database()
db.list_collection_names()
db['stock_60min'].find_one({'code': '000001'})
db['stock_60min'].find({'code': '000001'}).sort('ts', 1).limit(1)   # 最早
db['stock_60min'].find({'code': '000001'}).sort('ts', -1).limit(1)  # 最新
```

4.4（只在一次性搬运脚本里用）：

```python
from GolemQ.core.migrate44 import db44
q = db44('quantaxis', serverSelectionTimeoutMS=3000)
q['stock_min'].find({'code': '000001', 'type': '60min'}).sort('time_stamp', 1).limit(1)
```
