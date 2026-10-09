# 术语表

> 建立日期：2026-09-20
> 本项目的**行话**。新会话容易把其中一部分当成拼写错误「修掉」——
> 第四节的词**几乎全是老代码的原样继承**，改动会静默打断数据读取。

---

## 一、架构术语

| 术语 | 含义 | 别误解成 |
|:--|:--|:--|
| **两层模型** | 本项目的总纲：第一层 = 一切交易系统的共性（根目录）；第二层 = 某市场怎么做（`markets/<Market>/`）| 目录审美偏好，实际是判据 |
| **门面（facade）** | `fetch/` —— 只定义契约、调度到当前市场，**不含任何市场知识** | 「还没实现的占位层」 |
| **Active Market** | 系统的**默认市场**指针。`gqm` 里叫 `_active_market_name`，默认 `StockCN` | 全局变量，可省 |
| **第一层 / 第二层** | 见「两层模型」。判据：描述「做什么」→ 一层；描述「某市场怎么做」→ 二层 | 重要程度排序 |
| **机制层 / 实现层** | `datasource/` 的拆分：`GolemQ/datasource/` 是机制（无市场假设），`markets/<M>/datasource/` 是实现 | 重复代码 |
| **落盘（export）** | 回测产物写入 `datastore/export/` | 日志 |

**判据**：当前只实现了 A 股，是**实现进度**，不是归属依据。（曾据此错误地把
`fetch`/`features`/`models`/`pipeline`/`portfolio` 判为 A 股特有 —— 见 `DECISIONS.md` D1。）

---

## 二、量化领域术语

| 术语 | 含义 |
|:--|:--|
| **槽位（slot）** | 一个持仓位。参照口径：**每 10 万元权益 1 槽**，下限 33。权益增长时**加槽位数而非单仓规模**（33 槽时单仓占 3%，300 槽降到 0.33%，压低冲击成本）|
| **`util`** | 仓位利用率（资金使用率）。`_util.csv` 里的列名 |
| **交割单（trades）** | 成交明细：`datetime, code, side, qty, price, amount, fee, pnl`。**期末清算也落成 `side='S'` 行**，否则胜率无法由产物复算 |
| **对账（reconcile）** | 独立于回测器的第二套实现：重放交割单 → 复算现金/持仓 → 与 `_util.csv` 对比。参照 `OneWaveQuant/extras/_reconcile.py` |
| **预热（warmup）** | 显式区间回测时起点前需要的 bar 数。特征计算需历史窗口，预热不足会让区间开头的特征全为 NaN —— **那不是「策略没信号」，是「数据不够」** |
| **T+N** | 买入后第 N 日才可卖。A 股 T+1。本实现按**自然持有期**判，不按 bar 数 |
| **一字板** | 涨跌停且无成交。`limit_check` 开启时不成交 —— 不检查会凭空得到买在涨停、卖在跌停的收益 |

---

## 三、数据结构

| 术语 | 形态 | 说明 |
|:--|:--|:--|
| **`features_dummy`** | `DataFrame`，两层 MultiIndex **`(date, code)`**，列 = 特征名 | **本项目特征矩阵的规范名**。各 `features/` `analysis/` 模块往里**追加列**；`portfolio/` 也消费它 |
| **`KlineResult`** | 只有 `.data` 的最小接口 | 与 QUANTAXIS `QA_DataStruct_*` 对齐 —— 调用方只取 `.data`。定义在 `markets/StockCN/kline83.py` |
| **`.data`** | `DataFrame`，两层 MultiIndex `(date\|ts, code)` | 下游用 `index.get_level_values(level=0/1)` 取时间轴与代码 |
| **`MARKET_TYPE`** | 常量类 | ⚠️ **混了两个层级**：`STOCK_CN`/`STOCK_HK`/`STOCK_US` 是**市场归属**；`INDEX_CN`/`ETF_CN`/`FUND_CN` 等是**品种类型**（同属一个市场）。A 股指数仍属 `StockCN` |

---

## 四、项目专有词汇（**几乎全是老代码继承，勿当拼写错误改**）

| 词 | 实际含义 | 易被误认为 |
|:--|:--|:--|
| `reality` | 领域词汇（老代码 **588 处**），指「实际/实盘」数据，与 `baseline` 相对 | `realtime` 的笔误 |
| `shaplet` | 领域词汇，**MongoDB 集合名** `concept_shaplet_reviews`（老代码也有）| `shapelet` 的笔误 |
| `metaphase` | 领域词汇（老代码 `calc_metaphase_stack`）| 生物学词，或笔误 |
| `moneyflw_vol_min` | **持久化字段名**，从老项目逐字继承 | `moneyflow` 的笔误 |
| `frequence` | **QUANTAXIS 公开 API 参数名**，且出现在逐字复制的错误串里 | `frequency` 的笔误 |
| `ckpo` | cookbook 名的一部分（`ckpo_align_stock_turnover_rate`）| 缩写错误 |

**判断方法**：拿不准时去 `GolemQ_old/` 搜同一个词。**老树也有 = 领域词汇；只在拼写上可疑 = 真笔误。**

已确认的真实笔误（**已修**，见 `MIGRATION_STATUS.md`）：`BACKETEST`→`BACKTEST`、
`quandrant`→`quadrant`、`qmt_trder`→`qmt_trader`、`is_furture_cn`→`is_future_cn`、
`perpar_symbol_range`→`prepare_symbol_range`、`ret_banedlist`→`ret_bannedlist`、
`ecach_code`→`each_code`、`each_symbo`→`each_symbol`、
`calc_stock_matadata_missing_queris`→`calc_stock_metadata_missing_queries`、
`projetc.md`→`Project.md`。

---

## 五、老代码里保留的两个「不许简化」的名字

| 名字 | 位置 | 为什么不能删 |
|:--|:--|:--|
| `prefer_index` | `markets/StockCN/datasource/qmt_source.py` | 指数路径**不能走** `is_stock_cn` —— 否则 `000905.SZ`（厦门港务）会覆盖 `000905.SH`（中证500）的历史。见 `PITFALLS.md` P2 |
| `SECTOR_SKIP` | 同上 | QMT 把北证指数 `899050/899601/810011` 混进「京市A股」、上证基金 `910000`-`910005` 混进「沪深京A股」；不排除会让下游把指数当股票 |
