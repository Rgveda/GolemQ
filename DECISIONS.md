# 架构决定记录（ADR）

> 建立日期：2026-09-20
> **接手项目时请先读本文档**，不要重新论证已定的事。
>
> 每条记录四件事：**决定 / 拒绝的方案 / 弃案理由 / 约束**。
> 其中「拒绝的方案」与「弃案理由」是 docstring 与代码都装不下的 ——
> 也正是新会话最容易重犯的地方。

---

## D1. 两层模型：什么留根目录，什么进 `markets/`

**决定**：`analysis/` `features/` `models/` `portfolio/` `fetch/` `pipeline/` 全部**留根目录**；
只有「某市场怎么做」才进 `markets/<Market>/`。

**拒绝的方案**：按「只对 A 股有意义就迁进 `markets/StockCN/`」搬迁。

**弃案理由**：那条判据**把「当前只有一种实现」当成了「概念上属于 A 股」**。
按它执行会把 **6 个第一层模块中的 5 个**搬进 A 股目录 —— 那不是解耦，
是**把框架降级成单一市场的脚本**。而且将来加港股要做的不是「新增一个市场」，
而是**把抽象层从 A 股目录里再拆出来** —— 白做一遍。

**约束**：判据是「描述**做什么** → 第一层；描述**某市场怎么做** → 第二层」。
**当前只实现 A 股是实现进度，不是归属依据。**

---

## D2. Active Market：把「默认市场」显式化

**决定**：在 `GQMARKETS` 之上加一个「当前激活市场」指针（`core/market_registry.py`）。
`fetch/` 调度到它；`set_active_market()` 可切换。

**拒绝的方案**：不加指针，由「哪个市场包先被 import」隐式决定。

**弃案理由**：调用方说「取 600519 的日线」时并未指定市场，这个信息由系统状态提供。
隐式化会让**导入顺序决定成败** —— 先 import core 再 import 市场就静默失败。

**约束**：`set_active_market` 对未注册市场**抛错、不回落** —— 回落会让你
「以为在取美股，实际拿到 A 股」，比直接失败危险得多。
`get_active_market` 与 `get_market` **都必须触发惰性发现**（曾不一致，实测踩过）。

---

## D3. `fetch/` 是门面，带可选 `market` 参数

**决定**：

```python
get_kline_price_min(symbol, ...)                              # 隐含默认市场
get_kline_price_min(symbol, ..., market=MARKET_TYPE.STOCK_CN) # 单次覆盖
```

门面**不含任何市场知识**，一律经 `core.market_registry` 调度。

**拒绝的方案**：让门面自己判断市场，或只支持全局激活市场。

**弃案理由**：门面自带市场判断 = 把市场写死。只支持全局激活则无法**单次覆盖** ——
查一次别的市场就得改动全局状态。

**约束**：`MARKET_TYPE` 混了两个层级（市场归属 vs 品种类型），
分发按**市场归属**，品种类型由市场内部判定（`kline83` 用 `is_stock_cn`）。

---

## D4. `datasource/` 拆成机制层与实现层

**决定**：`GolemQ/datasource/`（机制：基类/注册表/限频/代理/落库）
+ `markets/<M>/datasource/`（实现：适配器 + **该市场的**源优先级表）。

**拒绝的方案（我的初版）**：全部放 `markets/StockCN/datasource/`，理由是
「等第二个市场出现再拆，现在拆无法验证边界」。

**弃案理由**：**那是拖延的包装。**「机制 vs 实现」是标准分层，
**不需要第二个市场来验证**。而且抓数据任何市场都要做 ——
美股要 IBKR/Alpaca、港股要富途/港交所。基类、限频、代理、落库**没有一处含市场假设**，
留在某个市场目录下是错的。（这条是被项目所有者当场纠正的，记此存证。）

**约束**：机制层**不认识任何市场**；注册由**导入实现层**触发。
机制层的 `get_source` 在注册表为空时须给出**可操作的报错**（指明怎么修），**不返回 None**。

---

## D5. 5 个参考集合落 8.3 的 `golemq_stock_cn`

**决定**：`stock_list` / `stock_info` / `etf_list` / `stock_block` / `financial`
统一落 `golemq_stock_cn`（与分钟线同库）。

**拒绝的方案**：落 8.3 的 `golemq`（按数据形态分库）。

**弃案理由**：项目所有者定的是「StockCN 即 A 股市场，库名归市场所有」——
参考数据与行情同属该市场，分库会让「A 股的库在哪」变成两个答案。

**约束**：`stock_block` 原先读写库不一致（读 `golemq`、旧 writer 写 `quantaxis`），
**落地时必须择一并让读写一致**。

---

## D6. 成本与交易规则**不提供默认值**

**决定**：`portfolio/costs.py` 的 `CostModel` 与 `rules.py` 的 `TradeRules`
是中立的**数据类**；A 股口径以 `ASHARE_COST` / `ASHARE_RULES` / `ASHARE_SIZER`
**命名预设**提供，**须显式传入**。

**拒绝的方案**：把万0.85、T+1 写成模块常量或默认值。

**弃案理由**：成本与规则是**通用概念、数值因市场而异**。给默认值会让跨市场使用时
**静默按 A 股算** —— 而**费率错一个数量级在回测里看不出来**（结果依然"合理"）。

**约束**：`BacktestEngine.__init__` 收到 `costs=None` 或 `rules=None` 时抛错。
引擎做成**具体类而非 ABC** —— 抽象基类无法实例化，构造函数里的这条校验就永远走不到。

---

## D7. `portfolio/` 恢复，但**不原样搬**

**决定**：`portfolio/` 恢复在**顶层**（策略组合是跨市场抽象），
采用**策略 / 引擎分离**：策略只给 `hold_signal` + `priority`，其余全归引擎。
**分阶段**：先定接口与契约，再照契约选择性移植。

**拒绝的方案**：把老树 `portfolio/`（6789 行、50 处 QUANTAXIS）整体搬进来。

**弃案理由**：三点 ——
① 体量与整个解耦工作相当；
② `base.py` **混了三种职责**（特征计算 `calc_massive_trend_*`、组合逻辑、
直接写 Mongo），原样搬等于把混乱一起搬回来；
③ 参照实现 `zen_bt.py` 长在 QUANTAXIS 树上，直接搬会把耦合带进来。

**约束**：`portfolio/` **反向依赖** `analysis/` `models/` `features/`
（`base.py:39,1908,1919`），所以「features → portfolio」的方向**靠搬目录得不到**，
是要新建的：`features → pipeline → portfolio → services`。
另：`pipeline → portfolio` 的数据契约**当前不存在**（老代码只有一个
`portfolio_batch` 字符串参数），恢复时要**新建契约**。

---

## D8. 空值契约**刻意不对称**

**决定**：日线 `get_kline_price_v3` 无数据返回 `None`；
分钟 `get_kline_price_min` 无数据返回**空对象**。

**拒绝的方案**：「统一一下，都返回 None / 都返回空对象」。

**弃案理由**：两个调用点的保护方式不同 ——
`_daily.py:105` 有 `if data_baseline is None` 分支；
而 `_stock.py:138` **直接取 `.data` 且未预初始化目标变量**，返回 None 会一路
变成 `UnboundLocalError`。统一会打断其中一个。

**约束**：空表**必须保留两层 MultiIndex**（裸 DataFrame 只有单层索引，
下游 `get_level_values(level=1)` 会抛 `Too many levels` —— 那是**形状错误**，
会被误读成 bug 而不是「没数据」）。详见 `PITFALLS.md` P3。

---

## D9. 只解耦，不搬数据

**决定**：`DATABASE` / `DATABASE_QA` **维持指向 4.4 不动**；
只为参考集合复用 `DATABASE_STOCK_CN`（8.3）。

**拒绝的方案（我的初版）**：把 `DATABASE` 等整体改由 `GQ_Setting` 构造，
理由写成「且不改变数据去向」。

**弃案理由**：**那个前提是错的，实测证伪。** `QASETTING` 读 **QUANTAXIS 自己的配置**
→ `192.168.50.39:27017`（**4.4**）；`GQSETTING` 读 `~/.GolemQ/settings/config.ini`
→ `57017`（**8.3**）。**两者指向不同服务器。** 整体替换会让
`stock_day`、`stock_a_snapshot*`、`stock_metadata*` 等**全部读空** ——
不是解耦，是拔掉半个系统。（4.4 的 `quantaxis.stock_min` 有 **30 亿行**，源库仍在服役。）

**约束**：**解耦（去掉配置依赖）与搬迁（换库）是两件事，不能混成一步。**
