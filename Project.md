# GolemQ 量化交易系统项目说明

## 项目概述

GolemQ 是一个模块化的量化交易系统，支持多市场、多策略的股票分析与自动交易。
**长期规划：完全独立于 QUANTAXIS** —— ✅ **已达成（2026-10-08，`DECISIONS.md` D12）**：
QUANTAXIS 已从全树剔除（`grep` 零 import、运行时不加载），
并由 `test_cases/test_no_quantaxis.py` 三条断言守着。

> **⚠️ 迁移状态见 [`MIGRATION_STATUS.md`](MIGRATION_STATUS.md)**（2026-09-20）
>
> 对照老项目 `GolemQ_old/` 做了逐模块比对。**结论：本次重构建成了模块骨架与接口，但计算层与数据层是 stub —— 真实实现大多仍在老树；少数已迁移的真实代码（如 `markets/StockCN/fetch.py`）没有被 `services/` 接上。**
>
> ~~本文档下述目标中，「完全独立于 QUANTAXIS」与「只连 MongoDB 8.3+、4.4 仅作一次性迁移源」两项均未达成~~
> —— **这条已过时**：两项**均已达成**（2026-10-08，D12）。剩余缺口见 `MIGRATION_STATUS.md`（分钟线迁移入口 #1 等）。开工前仍需先读它。
>
> 另注：本文档正文在「### 数据迁移策略（一次性过渡）」处**尚未写完**。

> **📐 MongoDB 8.3 接入设计见 [`GolemQ/markets/StockCN/MONGODB83.md`](GolemQ/markets/StockCN/MONGODB83.md)**
>
> 记录分钟线迁移后的库拓扑（4.4 与 8.3 两台服务器的实测内容）、`StockCN` 的 `DATABASE_STOCK_CN` 常量设计、读取器 `kline83.py` 的各项关键决策（按 `ts` 查询做桶剪枝、时区单点收敛、`vol`→`volume` 重命名、空值契约为何刻意不对称），以及数据契约与遗留缺口。

---

## 技术栈与版本要求

| 组件 | 版本要求 | 说明 |
|:---|:---|:---|
| **Python** | **3.12+** | 使用最新语法特性，不再兼容 3.11 及以下 |
| **Pandas** | **3.0+** | 新版 DataFrame 操作，性能优化，无旧版兼容负担 |
| **MongoDB** | **8.3+** | 仅支持 8.3 及以上版本，不再兼容 4.4 |
| **RabbitMQ** | 3.12+ | 消息队列服务 |
| **PyMongo** | **4.18+** | 官方驱动最新版，支持 MongoDB 8.3+ / 9.0，覆盖 Python 3.12 |

> **兼容性说明**：
> - 代码仅需支持 **Python 3.12+**，可自由使用 3.12 新特性（如 `override` 装饰器、改进的 f-string 等）。
> - 代码仅需支持 **Pandas 3.0+**，可直接使用新 API，无需兼容 2.x 的已弃用写法。
> - 数据库仅支持 **MongoDB 8.3+**，4.4 仅作为一次性迁移源，迁移完成后彻底下线。
> - 驱动固定为 **PyMongo 4.18+**，其新增对 MongoDB Server 9.0 的支持，为未来数据库升级预留空间。

---

## 技术架构

- **消息队列**: RabbitMQ（异步任务缓存和队列管理）
- **主数据库**: **MongoDB 8.3+**（全新部署）
- **数据库驱动**: **PyMongo 4.18+**
- **数据处理**: **Pandas 3.0+**（向量化计算）
- **部署平台**: Windows x64 / Linux（Ubuntu 22.04+、CentOS 9+）
- **开发语言**: **Python 3.12+**

---

## 数据库配置与版本管理

| 数据库实例 | 配置目录 | 版本 | 用途 | 状态 |
|:---|:---|:---|:---|:---|
| **GolemQ (目标)** | **`~/.GolemQ/config/`** | **MongoDB 8.3+** | 新数据存储、新策略开发、实盘交易 | ✅ **唯一目标** |
| **QUANTAXIS (迁移源)** | ~~不再读配置（地址是常量）~~ | MongoDB 4.4 | ~~一次性迁移源~~ | ⛔ **已退休（2026-10-10，`DECISIONS.md` D33）** —— 通道已删，全树零 4.4 引用 |

> **重要原则**：
> 1. GolemQ 运行时**只连接 MongoDB 8.3+**，严禁连接 4.4
> 2. ~~旧数据库仅通过 **`core/migrate44.py`** 一次性读取~~ → ⛔ **已退休（2026-10-10，D33）**：搬运完成，`core/migrate44.py` / 整条 `--migrate-*` / `GQ_MIGRATE44_URI` **均已删除**，全树零 4.4 引用。~~⚠️ 原文提到的 `scripts/migrate_mongodb.py` **从来不存在**~~
> 3. 所有 A 股数据接口在 `markets/StockCN/` 中**重写实现**，严禁 import QUANTAXIS ✅ **已达成**（`test_no_quantaxis.py` 守）
> 4. 迁移完成后，`~/.QUANTAXIS/` 配置与 4.4 实例一并移除

### 数据迁移策略（一次性过渡）
