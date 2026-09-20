# 数据接口分层方案（修正版）

> 建立日期：2026-09-20
> 方案来源：**项目所有者提出**，取代本文件早先的「A 股特有就搬进 `markets/StockCN/`」版本
> 状态：方案，未执行

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

## 五、`pipeline/` 与 `portfolio`

你描述 `pipeline/` 的职责是「多股、多核计算的抽象封装，**最后计算结果提交给 `portfolio` 进行交易策略组合**」。

⚠️ **`portfolio/` 在新树不存在** —— 重构时被删除。老树有：

```
GolemQ_old/portfolio/   __init__.py  __main__.py  base.py  ...
```

**故 `pipeline/` 当前的「提交给 portfolio」这一步在新树里是断的。**
需要你定：

1. `portfolio/` 是否恢复？按什么形态？
2. 若恢复，它属于抽象层（顶层）还是市场实现（`markets/StockCN/`）？
   —— 从「交易策略组合」的语义看，策略是**跨市场**的，应在顶层。
3. 若不恢复，`pipeline/` 的结果提交到哪里？

**在这一步定下来之前，`pipeline/` 不宜改动** —— 它是真实代码，且下游去向未定。

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

### 步骤 3 —— 把 `fetch/` 改成门面

* `fetch/kline.py` 重写为调度层（`get_active_market().xxx(...)`）
* `fetch/concept.py` 同
* 4 个 `services/persistence/*` 的 import **不用改** —— 它们从 `GolemQ.fetch.kline` 导入是**正确**的（门面就该从那里调）

验证：
```bash
grep -rn "from GolemQ.fetch" GolemQ/ --include=*.py   # 应仍指向 fetch/
python -m GolemQ.cli --save-status                     # 回归
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
