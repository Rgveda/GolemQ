# 数据接口分层方案（修正版）

> 建立日期：2026-09-20
> 方案来源：**项目所有者提出**，取代本文件早先的「A 股特有就搬进 `markets/StockCN/`」版本
> 状态：方案，未执行

---

## 零、总纲：两层模型

> 由项目所有者提出，是本文档所有判断的**唯一依据**。

### 第一层 —— 一切交易系统的共性（**根目录**）

| 模块 | 共性职责 |
|:--|:--|
| `analysis/` | 技术分析**算子**（重采样、时间轴、能量）|
| `features/` | 由算子产出的**技术分析特征** |
| `models/` | **特征集合**（供模型使用的特征组）|
| `portfolio/` | **策略组合**（组合收益、比例、优化）|
| `fetch/` | **数据获取门面**（抽象接口，调度到当前市场）|
| `pipeline/` | **批处理管线**（多股、多核计算能力的封装）|
| `core/` `services/` `cli/` `supervisor/` `agents/` | 基础设施 / 持久化 / 入口 / 运维 / 通信 |

### 第二层 —— 某市场怎么做（`markets/<Market>/`）

| 模块 | 市场职责 |
|:--|:--|
| `markets/StockCN/kline83.py` | A 股的 8.3 时序读法 |
| `markets/StockCN/refdata*.py` | A 股参考集合读写 |
| `markets/StockCN/datasource/` | A 股数据源适配 |
| `markets/StockCN/MONGODB83.md` | A 股的库布局 |

### 判据

> **描述「任何交易系统都要做的事」→ 第一层（根目录）。**
> **描述「某个市场怎么做」→ 第二层（`markets/<Market>/`）。**
>
> **当前只实现了 A 股，那是实现进度，不是归属依据。**

---

## 一、修正：早先判据错在哪

早先版本的判据是「只对 A 股有意义 → 迁进 `markets/StockCN/`」。**这条判据有个根本错误：**

> **它把「当前只有一种实现」当成了「概念上属于 A 股」。**

`fetch/` 是**门面（facade）**。门面即使当前只有一个实现，它仍然是门面 ——
把它搬进 `markets/StockCN/` 等于**把市场写死**，那是反解耦的，不是解耦。
同理 `features/`（技术分析特征）、`models/`（特征集合）、`pipeline/`（批处理管线）
都是**抽象层**，只是恰好当前只有一个市场的实现。

按旧判据搬完，将来加港股要做的不是「新增一个市场」，而是**把抽象层从 A 股目录里再拆出来** —— 白做一遍。

**结论：`fetch/` `features/` `models/` `pipeline/` `cookbooks/` 全部不迁移。**

---

## 二、各模块的正确定位

| 模块 | 定位 | 当前状态 |
|:--|:--|:--|
| `fetch/` | **抽象、隐含的 datasource 接口** —— 门面 | stub，待按门面实现 |
| `features/` | **技术分析特征** —— 由 `analysis/` 技术分析产出的指标特征，**多市场通用** | stub |
| `models/` | **features 的技术分析特征集合** | 多为 stub（`poolcoef.py` 371 行是真实代码）|
| `pipeline/` | **默认市场的批处理管线** —— 多股、多核计算的抽象封装，结果提交给 `portfolio` | 真实代码，但 import 悬空 |
| `cookbooks/` | 研究笔记本 | 真实内容 |
| `analysis/` | 技术分析算子（重采样/时间轴）—— `features` 的上游 | stub |
| `markets/StockCN/` | **A 股市场的具体实现** | 已就位（`kline83.py` `refdata*.py` `datasource/`）|

### 与已落地代码的关系

已完成的 `markets/StockCN/kline83.py`（8.3 时序读取）**位置正确** ——
它是**A 股实现**，门面 `fetch/kline.py` 将来调度到它。同理 `refdata*.py`
与 `datasource/` 都是 A 股实现，留在 `markets/StockCN/` 下没错。

---

## 三、核心机制：`Active Market`

### 设计

`GQMARKETS` 已是现成的注册表（`GolemQ/__init__.py:38`，由
`markets/StockCN/__init__.py:112-113` 注册、`cli/tools.py:33-75` 自动发现）。
在此之上再加一个**「当前激活市场」指针**，系统即获得一个**默认证券市场属性**。

```python
# GolemQ/__init__.py
GQMARKETS = {}                     # 已存在：所有已注册市场
GQSUBSCRIBER = {}                  # 已存在

_active_market_name = 'StockCN'    # 新增：默认市场（系统的隐含属性）

def register_market(name, instance):
    """注册市场。由市场模块自注册或 cli/tools.py 自动发现调用。"""
    GQMARKETS[name] = instance
    if name not in GQSUBSCRIBER:
        pass

def set_active_market(name: str):
    """切换激活市场。未注册则明确报错，不静默回落。"""
    if name not in GQMARKETS:
        raise KeyError(f'市场 {name!r} 未注册；已注册: {sorted(GQMARKETS)}')
    global _active_market_name
    _active_market_name = name

def get_active_market():
    """返回当前激活的市场实例。

    **惰性触发注册**：若注册表为空，先走一次自动发现（`cli/tools.py`
    的 `auto_register_markets`），避免 import 顺序决定成败。
    """
    if not GQMARKETS:
        from GolemQ.cli.tools import auto_register_markets
        auto_register_markets()
    if _active_market_name not in GQMARKETS:
        raise KeyError(f'默认市场 {_active_market_name!r} 未注册；'
                       f'已注册: {sorted(GQMARKETS)}')
    return GQMARKETS[_active_market_name]
```

### 为什么这样比搬文件更好

* **加新市场 = 新增一个市场包**，不改任何抽象层。
* **切换市场 = 一次 `set_active_market()` 调用**，而非改 import。
* `fetch/` `features/` `pipeline/` 里的代码**完全不含市场判断**，市场差异全部收敛到「当前激活的是谁」。

---

## 四、`fetch/` 门面的调度设计

### 契约归属

门面定义**调用方需要的契约**，`BaseMarket` 声明对应的抽象方法，
`markets/StockCN/` 提供实现。三者关系：

```
services/persistence/*          调用方
        │
        ▼
GolemQ/fetch/kline.py           门面：定义契约 + 调度（无市场逻辑）
        │
        ▼
BaseMarket.get_kline_price_min() 抽象声明
        │
        ▼
markets/StockCN/__init__.py       A 股实现 → 委托给 kline83.py
```

### `fetch/kline.py` 形态

```python
def get_kline_price_min(symbol, start=None, end=None, verbose=False, realtime=True):
    """分钟线。**本函数不含任何市场知识** —— 一律调度给当前激活市场。"""
    return get_active_market().get_kline_price_min(
        symbol, start=start, end=end, verbose=verbose, realtime=realtime)
```

### `BaseMarket` 需要新增的抽象方法

现有 5 个抽象成员（`name` / `get_stock_codes` / `get_kline_quotes` /
`get_kline_quotes_min` / `purge_historical_collections`）**不足以覆盖门面契约**：

| 门面函数 | 需要的市场方法 | 现状 |
|:--|:--|:--|
| `get_kline_price_min` | 同名方法，返回 `(result, codename)` | **缺**（现有 `get_kline_quotes_min` 返回裸 DataFrame，签名不符）|
| `get_kline_price_v3` | 同名方法 | **缺** |
| `get_stock_concept_kline` | 同名方法 | **缺** |

即：`BaseMarket` 要按门面契约补三个抽象方法（或把一个统一的 `fetch` 子对象挂到市场上）。
**这是本方案的主要新增工作量。**

### 空值契约（沿用既有约定，勿改）

* `get_kline_price_v3`（日线）空 → 返回 `None`，命中 `_daily.py:105` 的既有分支
* `get_kline_price_min`（分钟）空 → 返回**空对象**，因 `_stock.py:138` 无 `None` 保护且未预初始化

详见 `markets/StockCN/MONGODB83.md`。

---

## 五、`pipeline/` 与 `portfolio/`

### 已定（项目所有者，2026-09-20）

**`portfolio/` 恢复；它不是 A 股特有；先做单一市场，跨市场后续再议。**

恢复位置：**顶层** —— 策略组合是跨市场的抽象，不属任何单一市场。

### 恢复的真实体量（实测，勿低估）

| 文件 | 行数 |
|:--|--:|
| `base.py` | **2399** |
| `utils.py` | **2087** |
| `fof_v3/v2/v1.py` | 778 / 655 / 523 |
| `by_trend_indices.py` | 208 |
| `__main__.py` | 138 |
| **合计** | **6789** |

**QUANTAXIS 引用 50 处**（集中在 `base.py` 的 `QA_util_timestamp_to_str` /
`QA_util_str_to_datetime` 与 `DATABASE`）。体量与整个解耦工作相当。

### ⚠️ 不能原样搬：`portfolio/base.py` 混了三种职责

```
class PFL(_const)                     常量类（PFL.STOCK_PORTFOLIO_RANK 等）
calc_massive_trend_vXI/XII/XIII       趋势特征计算   ← 属 features/
calc_flash_in_out_tau_optimizer      优化器
calc_portfolio_returns / _ratio       组合收益与比例 ← 这才是 portfolio
save_stock_portfolio_stats(...)       直接写 Mongo   ← 属 services/（CLAUDE.md 约定）
```

原样恢复 = 把这三种职责的混乱一并搬进来。**恢复应同时拆分。**

### 依赖方向（既有事实）

`portfolio/` **反向依赖特征层**：

```
base.py:39   → analysis.timeseries
base.py:1908 → models.rail
base.py:1919 → features.base
utils.py:31  → from GolemQ.analysis.timeseries import *   ← 通配符，搬迁时一并清理
```

故「features → portfolio」的方向不能靠目录移动实现，而是：

```
features/ ──▶ pipeline/ ──▶ portfolio/ ──▶ services/（落库）
（特征）      （批处理）      （策略组合）     （DB 操作）
```

### 当前 `pipeline/` → `portfolio/` 的连接实为一个字符串参数

老代码里两者**并无数据提交关系**：

```
benchmark/base.py:261,276,316-334   portfolio_batch: str = ''   传入并被透传
```

即「提交给 portfolio」是**目标形态**，不是现状。恢复时**要新建这个契约**，
而不是把老代码的字符串参数当成契约。

### 恢复方案（分阶段，不一次性搬 6789 行）

**阶段 1 —— 骨架与契约（小，可独立验证）**
* 建 `portfolio/__init__.py`，**给出真实导出**（老的是空文件，等于无契约）
* 定义 `Portfolio` 上下文与 `pipeline → portfolio` 的**数据契约**
* 此时不含策略逻辑，但接口固定

**阶段 2 —— 按依赖序移植，边移边拆**

| 老位置 | 去向 | 理由 |
|:--|:--|:--|
| `calc_massive_trend_vXI/XII/XIII` | `features/` | 是特征计算，非组合逻辑 |
| `save_stock_portfolio_stats` | `services/` | DB 写入，CLAUDE.md 约定 |
| `calc_portfolio_returns/_ratio`、优化器、`PFL` | `portfolio/` | 真正的组合逻辑 |
| `fof_v1/v2/v3` | `portfolio/` | 组合构建 |

**阶段 3 —— 剥 QUANTAXIS（50 处）** ✅ **已完成（2026-10-08，`DECISIONS.md` D12）：全树 import 归零**
`QA_util_timestamp_to_str` / `QA_util_str_to_datetime` 在
`markets/StockCN/date_utils.py` 已有等价物；`DATABASE` 改指
`core.settings`。**与整体解耦合并做，不要单独一轮。**

### 在此之前

`pipeline/` **不要改动** —— 它是真实代码，且下游契约（阶段 1）尚未定义。

---

## 六、顺带查出的既有缺陷

| # | 缺陷 | 位置 |
|:--|:--|:--|
| 1 | `# from . import models` **被注释掉**，`__all__` 里也没有 `models`，但 `services/` 在引用它 —— `models` 不在包注册内 | `GolemQ/__init__.py:29,43` |
| 2 | `mainstream_benchmark.py` / `poolcoef_benchmark.py` 引用的 `models/mainstream`、`models/compact`、`pipeline/compact` **在新树不存在**，是悬空 import（基准跑不起来，静默返回 None）| `pipeline/*.py:34,38,42` |
| 3 | `models` 的 `_StubMeta.__getattr__` **静默返回假字段名**，拼错不报错 | `core/constants.py:34-37` |
| 4 | `fetch/kline.py` 已被 `markets/StockCN/kline83.py` 取代，当前无人引用（门面化时可删）| `GolemQ/fetch/kline.py` |

---

## 七、实施步骤

**前提：本方案不搬任何目录，只做「门面化 + 激活市场」。**

### 步骤 1 —— 引入 `Active Market`（可独立验证）

* 在 `GolemQ/__init__.py` 增加 `_active_market_name` / `get_active_market` / `set_active_market`
* 把 `markets/StockCN/__init__.py:112-113` 的自注册改为调用 `register_market()`
* `cli/tools.py` 的自动发现同步改走 `register_market()`

验证：
```python
python -c "import GolemQ; from GolemQ import get_active_market; print(get_active_market().name)"
```
应输出 `中国A股市场`。

### 步骤 2 —— 给 `BaseMarket` 补门面契约

* 新增 `get_kline_price_min` / `get_kline_price_v3` / `get_stock_concept_kline` 三个抽象方法
* `StockCN` 实现之，委托给 `markets/StockCN/kline83.py` 与后续的概念实现
* `StockHK`（stub）实现为抛 `NotImplementedError`，**不返回空** —— 空与未实现必须区分

验证：
```python
python -c "from GolemQ import get_active_market as g; m=g(); print(m.get_kline_price_min('600496')[0].data.shape)"
```

### 步骤 3 —— 把 `fetch/` 改成门面 ⛔ **已作废（2026-10-10）：走向了反面 —— 整包删除**

> 本节当时的计划是「把 `fetch/` 改造成调度门面」。实际执行时发现那层只是
> `return get_active_market().xxx(...)` 的一次改名，于是**直接删掉整包**（`DECISIONS.md` 与
> `HANDOFF.md` 有完整论证：零生产调用者、加一个形参要改四处、docstring 引用已删除的调用方）。
> 下面保留原文，仅作历史记录 —— **`GolemQ/fetch/` 已不存在**，别照它做。

* ~~`fetch/kline.py` 重写为调度层（`get_active_market().xxx(...)`）~~
* ~~`fetch/concept.py` 同~~
* ~~4 个 `services/persistence/*` 的 import 不用改~~ —— 那 4 个调用方**已随 `services/persistence/` 整包删除**

验证（**现在是反面**）：
```bash
grep -rn "from GolemQ.fetch\|GolemQ\.fetch" GolemQ/ --include=*.py   # 应为**零命中**
python -m GolemQ.cli --save-status                                   # 回归
```

### 步骤 4 —— 修既有缺陷 1/2/4

* 删除 `GolemQ/__init__.py:29` 的死注释，明确 `models` 的注册策略
* 修 `pipeline/*` 的悬空 import（属 `MIGRATION_STATUS.md` 问题 4）
* 删除已被取代的 `fetch/kline.py` 旧 stub 内容（步骤 3 会覆盖）

### 步骤 5（待定，见第五节）

`portfolio/` 恢复后再接 `pipeline/` 的下游。

---

## 八、判据的重新表述

旧判据（作废）：
> ~~只对 A 股有意义 → 进 `markets/StockCN/`~~

新判据：
> **如果它在描述「做什么」（接口、特征、加工、组合），它属于抽象层，留顶层；
> 如果它在描述「某市场怎么做」，它属于 `markets/<Market>/`。**
>
> 当前只有一种实现**不构成**抽象层该下沉的理由。

按新判据复核已落地的代码：

| 已落地 | 判定 |
|:--|:--|
| `markets/StockCN/kline83.py` | ✅ 正确 —— A 股的 8.3 时序读法，是「怎么做」 |
| `markets/StockCN/refdata*.py` | ✅ 正确 —— A 股参考集合的读写 |
| `markets/StockCN/datasource/` | ✅ 正确 —— 但见下 |
| `markets/StockCN/MONGODB83.md` | ✅ 正确 —— A 股的库布局 |

> **一处待议**：`datasource/` 里的适配器（pytdx/akshare/baostock/tushare…）
> 多数**不只服务 A 股**（tushare/akshare 也有港股美股）。按新判据它们倾向顶层。
> 但它们当前的产出 schema 与集合名是 A 股口径。**建议暂留**，待出现第二个市场时再提取公共部分。
