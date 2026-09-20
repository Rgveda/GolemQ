# GolemQ 迁移完整性报告

> 生成日期：2026-09-20
> 对照基线：`GolemQ_old/`（1308 文件 / 66 MB，重构前的老项目）
> 被审对象：`GolemQ/`（约 123 文件）
> 方法：按模块分片做**老 ↔ 新逐文件对比**。因为本仓库的 git 历史从重构后的基线提交才开始，`git diff` 无法呈现重构差异，必须直接比对两棵树。

**重申：本报告只回答一个问题 —— 哪些逻辑在重构中丢失、走样、或接线错误。** 它不评价代码风格。

---

## 一、总体结论

> 这次重构建成了**模块骨架与接口**，但**计算层与数据层是 stub**。真实实现大多仍留在老树；少数已迁移的真实代码（如 `markets/StockCN/fetch.py`）**没有被 `services/` 接上**。

关键点：问题不是"功能没写"，而是**写了一半又接了根假线** —— 大量 stub 会静默返回空值/伪造字段名，让下游失败看起来像"数据本来就没有"。

---

## 二、主失败因果链

```
core/constants.py:34-37  _StubMeta.__getattr__ 伪造字段名（静默，不抛 AttributeError）
   → 查询 'field_maxfactor_major'，而历史数据写在 'MFT_MAJ' 下
   → fetch/kline.py 是 stub，kline baseline 为空
   → services/persistence/_daily.py:128  each_day[0] → IndexError
   → 持久化检查中止，ratio = 0
   → 完整性监控对全部标的报 0% 完整度
   → 真实的数据断流被这个假的 0% 掩盖，无人察觉
```

---

## 三、HIGH（10 项）

| # | 问题 | 位置（旧 → 新） | 性质 |
|:--|:--|:--|:--|
| 1 | **迁移入口整块消失** —— 老项目有完整的分钟线迁移：timeseries 建表、8.3 读路径 `get_kline_price_min_v8`、污染数据周期边界过滤、`(code, month)` 断点续传、写前去重、北京时区处理。新树连 `scripts/` 目录都没有，全树 grep `migrate` / `timeseries` / `timeField` 零命中 | `GolemQ_old/scribe/min_migrate83.py`(345行) + `min_migrate83_cli.py`(207行) → **无对应物** | 未迁移 |
| 2 | **QUANTAXIS 未剥离，运行时数据库仍接在它上面** —— `DATABASE = QASETTING.client.golemq` 正是所有 `services/` 模块使用的对象（如 `persistence/_daily.py:46`），由 QUANTAXIS 的 `config.json` 驱动。新写的 `GQ_Setting`（读 `~/.GolemQ/config.ini`）只被 CLI 用来写配置，**从未用于绑定 `DATABASE`** | `core/settings.py:34,274-279`（与老 `utils/settings.py:273-279` 逐字节相同） | 未迁移 |
| 3 | **`_StubMeta` 伪造字段名** —— 对任何未定义属性返回 `f'{类名小写}_{属性名小写}'`，拼错或缺失都不报错 | `core/constants.py:34-37` | **新引入** |
| 4 | **三个 benchmark 子类全部失效**（悬空 import，`calculate()` 静默返回 None，`success_count=0` 与"确实没数据"无法区分）| 老 `models/mainstream.py:858`、`models/poolcoef.py:3252`、`benchmark/compact.py:171` → 新 `pipeline/mainstream_benchmark.py:34`、`poolcoef_benchmark.py:34`、`compact_benchmark.py:42` 全部 import 失败 | 未迁移 |
| 5 | **`features/` 用 stub 顶替真实 Mongo 读写**，且被 `services/persistence/` 调用 —— 每次完整性检查都加载空 DataFrame，**即使集合里数据齐全也报"全部缺失"** | 老 `features/empirical.py:956/344/3269/2966`、`reviews.py:729` → 新 `features/empirical.py:10/15/20/25`、`reviews.py:10`（`return features_dummy.copy()`）| 未迁移 |
| 6 | **真实 kline 实现就在同一棵树里，`services/` 却接了 stub** —— 真实代码在 `markets/StockCN/fetch.py:721`(`get_kline_price_min`)、`:1176`(`get_kline_price_v3`)，而 `fetch/kline.py` 退化成 23 行空 stub | 调用方 `persistence/_daily.py:56`、`_stock.py:54`、`_review.py:52`、`_concept.py:53` | **接线错误** |
| 7 | **`models/alias.py` 是空类** → `services/align.py:336-338` 使用 `LTT.QUADRANT_LEVERAGE_MACD_TIMING_LAG_DUMMY` 时 **AttributeError** | 老 `models/alias.py:732`（完整常量注册表）→ 新 `models/alias.py:7` `class LTT: pass` | 未迁移 |
| 8 | **模型字段常量与存储 schema 不再匹配** —— 见下表 | `models/massive.py`、`models/risk.py` | **新引入** |
| 9 | **ETF 前复权模块整删** —— 拉 ETF 日线（如 510300）跨除权日时，老代码返回**连续的前复权 OHLC**，新代码只打印"ETF 需要人工复权"就返回**不复权数据**，产生约 10% 假跳空，**污染 ETF 回测** | 老 `markets/StockCN/etf_fq.py`（260 行 / 3 函数，`GQ_is_etf` / `GQ_fetch_etf_adj` / `GQ_apply_etf_qfq`）→ 新 `markets/StockCN/fetch.py:1416-1423` 仅打印提示 | **逻辑丢失** |
| 10 | **K线新鲜度告警消失** —— 数据源静默断流（pytdx/QMT 返回空包、QUANTAXIS 吞掉）不再触发任何告警 | 老 `supervisor/data_freshness.py` + `cli/__main__.py:1465-1469` 调 `notify_if_stale()` → 新 `cli/__main__.py:198-227` 无检查 | **逻辑丢失** |

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
| `analysis/timeseries.py` 被 stub（1399 → 18 行）| `analysis/timeseries.py:11-18`，被 `persistence/_concept.py:56-57`（用于 `:220,243,291,293`）调用 | 对齐成空操作，`MAS.BOOTSTRAP_STAGE_MODE_BEFORE` 恒为 0 |
| `core/base.py` 被 stub（243 → 20 行）| `core/base.py:10-15`，被 `persistence/_daily.py:113-114`、`_stock.py:110-111` 用于 checkpoint `FrozenExpired` | 老代码用 `QA.trade_date_sse` 取**交易日历**，新的只判周末 → **法定节假日算错** |
| 越层直连 MongoDB | `core/settings.py:107`(`find_one`)、`:147`(`update_one`)；`cli/watchdog_manager.py:85-127/146-197/211-280` | 违反约定「所有 DB 操作必须在 `services/`」 |
| CLI 层业务逻辑 | `cli/__main__.py:339-423`（约 85 行超时判定+表格构建）、`:457-522`（`_send_timeout_alert`）、`:229-250`（daemon 循环）| 违反约定「CLI 只做参数校验再委托 `pipeline/`」 |
| 估值源回退 xtquant → baostock | `markets/StockCN/crawler.py:168,212,235` | 老代码注释称改用 xtquant 是为「解决网络接收错误」；且 baostock `adjustflag="3"`（不复权）与 xtquant `dividend_type='front_ratio'` 语义不同，换手率口径也不同 |
| `resample_features_frequency` 被 stub | `markets/StockCN/base.py:8-13`（老 `StockCN/base.py:39-94` 有多重采样）| 目前无人调用（潜伏）；一旦启用，15/30min 对齐会静默拿到未重采样数据 |
| 测试覆盖损失 | 老 `test_monitor.py`、`test_frequency_control.py` 无对应物；新 `test_heartbeat_fix.py` **既非 `unittest.TestCase`（discovery 发现不了）又本身坏**（`:66` 调 `HeartbeatMonitor._hash_instance_id`，该方法只存在于 `HeartbeatModule`）；`__pycache__/test_watchdog_manager.cpython-312.pyc` 存在但 `.py` 已不在 | heartbeat / mutex / 限流 / watchdog 实质无测试 |
| `--sub` 命令行面收窄 | 老支持 8 种模式（`sina_l1`/`tencent`/`xtquant`/`huobi_realtime`/`huobi`/`okex`/`binance`/`tencent_1min`）→ 新 `GQSUBSCRIBER` 只有 `l1_tencent` 一个键 | 其余模式 CLI 不可达；`--save-qmt`、`--migrate-min83` 入口也消失 |
| 300 行文件规则被破 | `services/features.py` 956 行、`services/align.py` 490 行 | 违反 CLAUDE.md 规定 |
| `features.py` / `features/` 同名陷阱 | `services/features/` **无 `__init__.py`**，因此是 namespace package，实际被导入的是 `features.py` 模块（完整）| 包是死代码，但**包里只有 11 个函数中的 8 个**，缺 `GQ_update_hourly_metadata`、`GQ_remove_hourly_metadata`、`GQ_move_hourly_metadata`。**若有人"顺手"补个 `__init__.py`，会静默丢掉这 3 个函数** |

---

## 五、LOW

- `is_furture_cn` → `is_future_cn` 改名未留兼容别名（`markets/StockCN/symbol.py:280`）。新树内**零调用点**，老调用点在新树已不存在，影响低。
- `supervisor.messenger.send_alert`（Server酱 webhook 派发）无测试。

---

## 六、既有问题（**非本次重构引入**，勿误记为回归）

| 问题 | 位置 | 说明 |
|:--|:--|:--|
| `timedelta(hours=8.3)` | `services/persistence/_stock.py:125`、`_daily.py:128` | 老树 `scribe/persistence.py:681,940` **逐字相同**。A股应为 UTC+8，`8.3` 可疑；老树自身在同角色上还用了 `hours=8.5`（`analysis/ChipDistribution.py:347`、`omnipath.py:2693`、`stack.py:259`）→ **既有不一致**。仍是 bug，但不是重构造成的 |
| `HeartbeatModule.mutex()` 非原子 | `supervisor/heartbeat.py:459-462` | check-then-act，且唯一索引建在 `(module_name, instance_id)`，而 `instance_id` 每进程唯一 → **永远拦不住第二个进程**。两个调度器在 60 秒窗口内会双双通过并重复同步。`heartbeat.py` 新旧逐字相同 —— CLAUDE.md 声称的 "timeout-based lock" **从未成立过** |
| `min_interval_minutes=15` / `max_interval_minutes=1` 命名反了 | `supervisor/scheduler.py` | 实际每 1 分钟跑一次。新旧一致 |
| `core/preprocessing.py` `normalize()` 中 `skp.StandardScaler` NameError | `core/preprocessing.py` | 新旧都有 |

---

## 七、明确无问题（agent 逐字节比对确认）

| 范围 | 结论 |
|:--|:--|
| `pipeline/base.py` | 忠实迁移：停牌阈值 84 天、`days=960`、realtime 窗口 18h、pool 回退、统计口径**全部一致** |
| `gateway/xtquant/*` | 迁移且有扩展（持仓/委托同步、心跳、告警），未发现逻辑丢失 |
| `supervisor/scheduler.py` | 与老逐字节相同（仅 settings import 路径变）。午休 9:30-11:30 / 13:00-15:00、周末判断、Asia/Shanghai 时区、1 分钟轮询均未变 |
| `services/persistence/` 拆分本身 | 老 `scribe/persistence.py` 的 13 个函数在新 `persistence/__init__.py` 全部导出 —— **拆分没丢函数**（丢的逻辑来自它们调用的 stub）|
| `constants.py`、`tools.py`、`utils.py`、`base_market.py`、`StockHK/`、全部 `easyquotation/*` | 与老树一致（仅 CRLF 差异）|
| `date_utils.py` | 仅 import 路径改名（`utils.constants` → `core.constants`），零逻辑漂移 |
| `core/path.py`、`presentation.py`、`mongo.py`、`symbol.py` | 实质实现，非 stub |

---

## 八、被删除的模块：**新树无依赖，属有意砍掉**

`strategy` `portfolio` `fractal` `signal` `indices` `label` `markup` `imitation` `train` `czsc` `huobi`，以及老 `analysis/` 下的建模模块、`gateway/xtquant/` 的 `data_source.py`/`save_cli.py`/`save_qa.py`。

逐个核查：**新树没有任何地方依赖它们**（`models/poolcoef.py:83` 有一处 `from GolemQ.portfolio.base import PFL`，但包在 `try/except ImportError` 里，是死代码）。

**结论：这些是有意移除，不是漏迁。** 无需列为缺陷。

---

## 九、与 `Project.md` 既定目标的对账

| Project.md 目标 | 现状 |
|:--|:--|
| 完全独立于 QUANTAXIS，所有新增代码重新实现 | ❌ **未达成**。`services/` 与 `markets/StockCN/` 仍大量 import；`markets/StockCN/` 引用数：`fetch.py` 48、`realtime.py` 25、`symbol.py` 10、`scribe.py` 9、`quotes.py` 3、`align.py` 3、`crawler.py` 3 |
| 所有 A 股接口在 `markets/StockCN/` 重写实现 | ⚠️ **部分**。真实实现已迁移到位，但 `services/` 接的是 stub |
| 只连 MongoDB 8.3+，4.4 仅一次性迁移源 | ❌ **未达成**。`DATABASE` 仍是 `QASETTING.client.golemq`；迁移代码整块缺失 |
| 旧库仅通过迁移脚本一次性读取 | ❌ **未达成**。脚本不存在，老实现已随 `scribe/` 删除 |
| Python 3.12+ / Pandas 3.0+ / PyMongo 4.18+ | 未审（本次范围外）|

---

## 十、建议修复顺序

1. **修 kline 接线错误**（问题 6）—— 真实实现已在同树，只需把 `services/persistence/*` 的 `from GolemQ.fetch.kline import ...` 改为 `markets/StockCN/fetch.py` 的对应位置。**成本最低、收益最大**，且修完能让下游被掩盖的问题显形。
2. **`_StubMeta` 改为抛 `AttributeError`**（问题 3、8）—— 把静默失败变成显式失败。在此之前，任何"跑通了"的结论都不可信。
3. **字段名回滚或补迁移**（3.8 表）—— 决定是改回老名字，还是写数据迁移脚本。二选一，不能两头都不做。
4. **按业务优先级修其余 HIGH**：ETF 前复权（问题 9）、K线新鲜度告警（问题 10）、benchmark（问题 4）、`features/` stub（问题 5）、`alias.py`（问题 7）。
5. **迁移入口**（问题 1）与 **QUANTAXIS 剥离**（问题 2）—— 这两项是架构级决定，牵涉 CLAUDE.md 与 Project.md 的冲突，需要先定方向。

---

## 附：本报告未覆盖的范围

- 未做运行验证（无 MongoDB / RabbitMQ 实例，且 stub 会让多数路径提前失败）
- 未审 `cookbooks/`、`.claude/`、构建配置
- 未逐个核对 Python 3.12 / Pandas 3.0 API 兼容性
