# `examples/Deprecated/xtquant/` —— **已停用**：xtquant / MiniQMT 示例

> ## ⚠️ 为什么在 `Deprecated/` 下
>
> **因监管要求，基于 MiniQMT 实现的 xtquant 于 2026-10-01 起停止量化交易服务。**
>
> 也就是说这些脚本**已经跑不通了** —— 它们不是「暂时没配好」，而是**上游服务已经终止**。
> 本项目据此在 `DECISIONS.md` **D13** 里定：**只保留代码，入口已删**
> （`QMT_SOURCE_ENABLED = False` / `QMT_REALTIME_ENABLED = False`，
> 那两条路不再取数、不再订阅，但代码不删 —— 将来若换到别的通道，结构还能用）。

保留这 4 个文件只为「当初怎么用 xtquant 的 `XtQuantTrader` / `xtdata`」这个参考价值。
**别照着跑**，也**别**把其中的写法当成现行约定。

| 文件 | 行数 | 是什么 |
|:--|--:|:--|
| `xtquant_01_hello_quant.py` | 108 | 最小连通示例：`StockAccount` + `xttrader`，从 `GolemQ.gateway.xtquant.config.get_xtquant_config()` 取 `min_path` |
| `xtquant_02_自动逆回购.py` | 214 | 自动逆回购（GC001 之类）的挂单/撤单循环 |
| `xtquant_03_趋势网格策略.py` | 372 | 趋势网格策略（真策略逻辑：网格价位、回调、下单）|
| `xtquant_03_趋势网格策略_test.py` | 150 | ⚠️ **不是测试**，见下 |

## ⚠️ 关于 `xtquant_03_趋势网格策略_test.py`

名字带 `_test`，但**它不是测试**，而且**从来没被跑过**：

* `unittest` 只收 **`test_*.py`（前缀）**，它是 **`*_test.py`（后缀）** ⇒ 永远不被发现；
* 它**零 import**（实测），自己在文件里定义了一个 `MockTrendGridStrategy`
  —— 也就是**只测它自己写的 mock**，不碰 `xtquant_03_趋势网格策略.py` 里的真策略。

它是**开发那个网格策略时的草稿**。留着无妨（读代码时能看出当时的想法），
但**别把它当回归用例**，也别据此以为「网格策略有测试覆盖」。
真要给它上回归，得让它 `import` 真策略并把 xtquant 的调用打桩 —— 那是另一件事。

## 若哪天真要复活这条路

`DECISIONS.md` D13 记了**两条必须先处理**的事：

1. 必须先 `subscribe_quote(period='tick')`，否则拿到的是**陈旧缓存**；
2. 它走的是「按日**普通**集合 + 唯一索引 + upsert」，与 8.3 现行的**时间序列**存储
   **不兼容** —— 不改写入方式，第一次写就会抛 `Cannot perform a non-multi update`。
