# 决策记录（ADR）

> **本文件是索引。正文在 [`docs/decisions/`](docs/decisions/)** —— 一份决策一个文件。
> 索引表由 `python tools/gen_docs_index.py` 生成，**别手工编辑标记区**。

为什么拆
========
这里是 ADR 日志，**按定义只增不减**。原先单文件到 2026-10-15 已 2,200 行、
46 条决策，而项目刚搭完脚手架 —— 再涨下去，「查一条决策」要先把整份日志读进来。

⚠️ **`D<n>` 是稳定键**：全树代码注释写的是「见 `DECISIONS.md` D44」这种**编号引用**。
文件名就等于编号（`D44.md`），**编号一旦发布永不改动**，索引按编号排序、不做归档重排。

加一条新决策
============
1. 写 `docs/decisions/D47.md`，头两行必须是：

   ```
   # D47. 一句话标题

   - **日期**：YYYY-MM-DD
   ```

2. 跑 `python tools/gen_docs_index.py`（`--check` 可在提交前验是否漏跑）。

选读顺序仍见 [`CLAUDE.md`](CLAUDE.md) 的「接手本项目，按此顺序读」。

---

<!-- BEGIN GENERATED -->

| 编号 | 决策 | 日期 | 正文 |
|:--|:--|:--|:--|
| `D1` | 两层模型：什么留根目录，什么进 `markets/` | — | [decisions/D1.md](docs/decisions/D1.md) |
| `D2` | Active Market：把「默认市场」显式化 | — | [decisions/D2.md](docs/decisions/D2.md) |
| `D3` | `fetch/` 是门面，带可选 `market` 参数 | — | [decisions/D3.md](docs/decisions/D3.md) |
| `D4` | `datasource/` 拆成机制层与实现层 | — | [decisions/D4.md](docs/decisions/D4.md) |
| `D5` | 5 个参考集合落 8.3 的 `golemq_stock_cn` | — | [decisions/D5.md](docs/decisions/D5.md) |
| `D6` | 成本与交易规则**不提供默认值** | — | [decisions/D6.md](docs/decisions/D6.md) |
| `D7` | `portfolio/` 恢复，但**不原样搬** | — | [decisions/D7.md](docs/decisions/D7.md) |
| `D8` | 空值契约**刻意不对称** | — | [decisions/D8.md](docs/decisions/D8.md) |
| `D9` | 只解耦，不搬数据 —— ⛔ **已作废（2026-10-08，见 D12）** | 2026-10-08 | [decisions/D9.md](docs/decisions/D9.md) |
| `D10` | 实时行情落**按日时间序列集合**（`realtime_YYYY-MM-DD`） | — | [decisions/D10.md](docs/decisions/D10.md) |
| `D11` | `--save tdx`：K 线直写 8.3 + **增量按水位**，参考数据**不强制 pytdx** | — | [decisions/D11.md](docs/decisions/D11.md) |
| `D12` | **把 QUANTAXIS 从全树剔除**（D9 的反面） | — | [decisions/D12.md](docs/decisions/D12.md) |
| `D13` | QMT / xtquant：**只删入口，代码保留** | — | [decisions/D13.md](docs/decisions/D13.md) |
| `D14` | `--save` 收口成 **`<SOURCE>` 一个开关**（`--save-x` / `--save-qmt` 删除） | — | [decisions/D14.md](docs/decisions/D14.md) |
| `D15` | ETF 除权与复权**改由 pytdx 直供**（`etf_xdxr` / `etf_adj` 不再依赖 QMT） | — | [decisions/D15.md](docs/decisions/D15.md) |
| `D16` | 参考数据的**刷新闸**：**按集合**分阈值，复用 supervisor 的签到表 | — | [decisions/D16.md](docs/decisions/D16.md) |
| `D17` | `cli/__main__.py` 拆成「命令模块 + 有序注册表」 | — | [decisions/D17.md](docs/decisions/D17.md) |
| `D18` | `--save` 的取值校验交给 **argparse `choices=`**（一处真相源） | — | [decisions/D18.md](docs/decisions/D18.md) |
| `D19` | CLI 退出码口径：**用法/参数错 = 2，运行期失败 = 1** | — | [decisions/D19.md](docs/decisions/D19.md) |
| `D20` | CLI 环境自检（`cli/bootstrap.py`）：版权页 + 环境 / 配置 / 服务三道闸 | — | [decisions/D20.md](docs/decisions/D20.md) |
| `D21` | `--save tdx` 跑到 76% 卡死 —— **两个真因**，都不是业务逻辑 | — | [decisions/D21.md](docs/decisions/D21.md) |
| `D22` | 通达信服务器池：**每周探活一次**，活动列表落 `~/.GolemQ`，**不硬编码** | — | [decisions/D22.md](docs/decisions/D22.md) |
| `D23` | 服务器选择：**每 worker 线程各粘一台 + 随机起点**（分布式） | — | [decisions/D23.md](docs/decisions/D23.md) |
| `D24` | 环境自检也挂 banner（四态）+ **硬拦口径**与**放行集合** | — | [decisions/D24.md](docs/decisions/D24.md) |
| `D25` | K 线短路：**集合自证探针 + 逐只水位 + TTL 兜底**（三条缺一不可） | — | [decisions/D25.md](docs/decisions/D25.md) |
| `D26` | **盘中 vs 收盘后**：判据差异汇总表（**改这一带之前先读它**） | — | [decisions/D26.md](docs/decisions/D26.md) |
| `D27` | 启动信息改版：**身份块 + 跨 banner 对齐的阶段列** | — | [decisions/D27.md](docs/decisions/D27.md) |
| `D28` | 收盘后的**快路径**：整段跳过（连逐只读都省）—— 双因子（2FA）判定 | — | [decisions/D28.md](docs/decisions/D28.md) |
| `D29` | 读路径接通 **REALTIME**：`_v3` / `get_kline_price_min` 的 `realtime` **不再是空转** | — | [decisions/D29.md](docs/decisions/D29.md) |
| `D30` | 自检加 **`交易日历`** 节点：那张手维护的静态表也得有人看着 | — | [decisions/D30.md](docs/decisions/D30.md) |
| `D31` | 自检 banner **分两栏** + 三个**可选数据源**节点（`tdxidata` / `tushare` / `iwencai`） | — | [decisions/D31.md](docs/decisions/D31.md) |
| `D32` | 阶段**起止各一行**：`[t]: bootstrap` … `[t]: bootstrap done.` | — | [decisions/D32.md](docs/decisions/D32.md) |
| `D33` | **4.4 通道整条退休**（D12 的收尾） | 2026-10-10 | [decisions/D33.md](docs/decisions/D33.md) |
| `D34` | 换手率落 `stock_metadata_day`：**两列并列，不拼序列** | 2026-10-10 | [decisions/D34.md](docs/decisions/D34.md) |
| `D35` | RENKO 搬运：**三条平行实现只搬一条**，落点 `analysis/` | 2026-10-10 | [decisions/D35.md](docs/decisions/D35.md) |
| `D36` | RENKO 加速：**另写一份 jit 对照实现**，`renko.py` 一个字不动 | 2026-10-10 | [decisions/D36.md](docs/decisions/D36.md) |
| `D37` | `--moneyflow` 的源被**实测**改掉：东财资金流已死，交易流走同花顺 | 2026-10-11 | [decisions/D37.md](docs/decisions/D37.md) |
| `D38` | `stock_min_aligned` → **「元数据对齐到 K 线」**（不是照搬） | 2026-10-11 | [decisions/D38.md](docs/decisions/D38.md) |
| `D39` | 四量图 + 6 个指标搬入 `analysis/quadrant.py`：**只搬算术，不搬那一段破坏性写** | 2026-10-11 | [decisions/D39.md](docs/decisions/D39.md) |
| `D40` | RSRS **只搬指标计算**：`pre_rsrs_data_func` 一条，那个坏孪生不搬 | 2026-10-12 | [decisions/D40.md](docs/decisions/D40.md) |
| `D41` | 字段常量的**值**用驼峰（≤16 字符，少数 18 容忍）—— 且**只改未入库的** | 2026-10-12 | [decisions/D41.md](docs/decisions/D41.md) |
| `D42` | ChartScan（CV 图形识别）并入 GolemQ：**代码进库、env 与 ZMQ 边界保留** | 2026-10-13 | [decisions/D42.md](docs/decisions/D42.md) |
| `D43` | 读口加 `if_fq` 三模式：**不复权 / 前复权 / 后复权** | 2026-10-13 | [decisions/D43.md](docs/decisions/D43.md) |
| `D44` | scan scope 归位根层 `analysis/cv/` + 筹码分布**不再自补换手率** | 2026-10-13 | [decisions/D44.md](docs/decisions/D44.md) |
| `D45` | 码表缓存 `stock_codes.conf` **由 `--save tdx` 从 8.3 `stock_list` 导出** | 2026-10-15 | [decisions/D45.md](docs/decisions/D45.md) |
| `D46` | `--sub tencent` 的 L1 走**拉取全市场**，**不改"全推"** | 2026-10-15 | [decisions/D46.md](docs/decisions/D46.md) |
| `D47` | 删除 L2 流：腾讯 L2 **整条撤掉**，QMT 五档占位一并删 | 2026-10-15 | [decisions/D47.md](docs/decisions/D47.md) |
| `D48` | ETF 与指数**进 L1 tick 流**：码表扩容成 `stock ∪ etf ∪ index` | 2026-10-15 | [decisions/D48.md](docs/decisions/D48.md) |
| `D49` | 指标搬迁（一）：常量补齐 / Garident STAGE / **价位改存比值** / Poly9 核 / M9T 核 | 2026-10-16 | [decisions/D49.md](docs/decisions/D49.md) |
| `D50` | 两表合并（`stock_reality_feat_day` ← `stock_metadata_day`）+ 删掉 `GolemQ/features/` 整包 | 2026-10-16 | [decisions/D50.md](docs/decisions/D50.md) |
| `D51` | ChartScan CV 检出结果从 4.4 搬进 8.3（+ `etf_` 前缀分支 + 写库路由的 4 处分叉） | 2026-10-17 | [decisions/D51.md](docs/decisions/D51.md) |
| `D52` | pandas 3.0 / numpy 2.x 兼容性修复（4 处，全部**静默**） | 2026-10-17 | [decisions/D52.md](docs/decisions/D52.md) |
| `D53` | Prophet 价格区间（`FBPROPHET_*` 四列）搬进新树 —— **只搬算术，落库存比值** | 2026-10-18 | [decisions/D53.md](docs/decisions/D53.md) |
| `D54` | 统一因果 Review 管线（daily + zen_minor + coldstart 子集 + runtime） | 2026-10-19 | [decisions/D54.md](docs/decisions/D54.md) |
| `D55` | `reality_feat_*` 的持久化检查与写入口收口 | 2026-10-19 | [decisions/D55.md](docs/decisions/D55.md) |
| `D56` | ETF 数据拆分收口与 fund 历史残留（不搬） | 2026-10-20 | [decisions/D56.md](docs/decisions/D56.md) |
| `D57` | 消息推送分层：`agents.channels`（渠道）与 `supervisor.alerts`（告警） | 2026-10-20 | [decisions/D57.md](docs/decisions/D57.md) |
| `D58` | Review CLI 参数语义：`--strategy review`（别名 `--review`）、全局 `--eval`、删除 `--daily` | 2026-10-20 | [decisions/D58.md](docs/decisions/D58.md) |
| `D59` | `analysis/*_jit.py` 合并为唯一实现，删除非 jit 原版 | 2026-10-21 | [decisions/D59.md](docs/decisions/D59.md) |
| `D60` | CV 链迁出市场包；行情库句柄改由 `get_database()` 中性暴露 | 2026-10-21 | [decisions/D60.md](docs/decisions/D60.md) |
| `D61` | XGBoost 迁入 `analysis/xgboost/`：先落框架与特征缺口，不做特征生产/落库 | 2026-10-21 | [decisions/D61.md](docs/decisions/D61.md) |
| `D62` | `calc_compact_features`（8.3 版）+ `--strategy compact`：先只读，不落库 | 2026-10-21 | [decisions/D62.md](docs/decisions/D62.md) |
| `D63` | `--save tdx` 元数据段加入千股千评与牛股诊股 | 2026-10-22 | [decisions/D63.md](docs/decisions/D63.md) |
| `D64` | 用户 CLI 与内部/调试入口分离 | 2026-10-24 | [decisions/D64.md](docs/decisions/D64.md) |
| `D65` | review/coldstart 输出列统一为伪驼峰缩写（含存量原地改名） | 2026-10-24 | [decisions/D65.md](docs/decisions/D65.md) |
| `D66` | 删除四量图的 `QuadSt` / `ChoBits` / `QuadCodex` 状态机（review 不再计算） | 2026-10-25 | [decisions/D66.md](docs/decisions/D66.md) |
| `D67` | 删除 review 的 M9T lag 列（`m9tLag` / `m9tLagD` 都不再落库） | 2026-10-27 | [decisions/D67.md](docs/decisions/D67.md) |
| `D68` | notebook 只读 review 已落库的 `ZenPivotCnt` / `se`，不得重算 | 2026-10-27 | [decisions/D68.md](docs/decisions/D68.md) |
| `D69` | review 新增契约列不会自动回填旧行；用 `tools/backfill_hour_extra.py` 补 | 2026-10-27 | [decisions/D69.md](docs/decisions/D69.md) |
| `D70` | 归一化价格列统一 `D` / `_DUMMY` 后缀；attach `* close` 还原无后缀价格 | 2026-10-27 | [decisions/D70.md](docs/decisions/D70.md) |
| `D71` | review 日族加入 Renko S/L 时序（按日频率）；以及本次覆盖度检查 | 2026-10-27 | [decisions/D71.md](docs/decisions/D71.md) |
| `D72` | review 日族保存 Renko 全族 14 列；价格量纲列落 `D` 比值 | 2026-10-27 | [decisions/D72.md](docs/decisions/D72.md) |
| `D73` | 主力派发（LEADOUT）指标：主力吸筹 6 列的镜像，用于下跌指示 | 2026-10-27 | [decisions/D73.md](docs/decisions/D73.md) |
| `D74` | `se` / `ZenPivotCnt` 从 60min 契约移到日族（日频） | 2026-10-28 | [decisions/D74.md](docs/decisions/D74.md) |
| `D75` | 4.4 旧 review 特征表不搬（未归一化 / 复权锚已冻结，无参考意义） | 2026-10-28 | [decisions/D75.md](docs/decisions/D75.md) |
| `D76` | `BOTTOM_PRICE` / `ROOF_PRICE` 是**绝对价**，不归一化、不落库 | 2026-10-28 | [decisions/D76.md](docs/decisions/D76.md) |
| `D77` | 列名统一为伪驼峰：39 个老值改名 + 28 个保留 | 2026-10-28 | [decisions/D77.md](docs/decisions/D77.md) |

<!-- END GENERATED -->
