# 已知陷阱

> 建立日期：2026-09-20
> **接手项目时请先读本文档。** 这里记的都是「看起来像 bug 的刻意设计」与
> 「已经踩过、但代码里不留痕迹」的坑 —— 它们靠读源码发现不了。

判据：**凡是被"顺手修正"会造成损失的，都在这里。**

---

## 一、最危险：静默失败类

这一类不报错、不崩溃，只是**什么都没发生**。比崩溃难查得多。

### P1. `_StubMeta` 伪造字段名 —— 会让回测「跑通了但零成交」

**位置**：`GolemQ/core/constants.py:34-37`

```python
class _StubMeta(type):
    def __getattr__(cls, name):
        return f'{cls.__name__.lower()}_{name.lower()}'   # 未定义也不报错
```

`AKA` / `FIELD` / `FEATURES` / `TREND_STATUS` / `STATE` 都用它。任何**未定义的常量**
会被静默合成一个「看起来合理」的 snake_case 名字。

**后果链**（真实案例，portfolio 策略信号就在这条链上）：

```
常量未定义 → 得到 'features_xgb_echo_timing_lag'（数据里从不存在此列）
          → 信号全 False → 不产生任何交易
          → 回测"成功"返回，交割单为空，无任何报错
```

**已被证实的字段名漂移**（新树常量值 ≠ 数据实际存储名）：

| 常量 | 老树（**数据实际存储名**）| 新树 |
|:--|:--|:--|
| `XGB_ECHO_TIMING_LAG` | **`'XGB_ECHO_LAG'`** | 未定义 → 伪造 |
| `ZEN_PEAK_TIMING_LAG_MAJOR_REAL` | **`'ZPLagMajR'`** | `'zen_peak_timing_lag_major_real'` |
| `MAGIC_NINE_TURNS_MAJOR` | **`'m9tMaj'`** | 同类伪造 |
| `MAXFACTOR_MAJOR` | `'MFT_MAJ'` | `'field_maxfactor_major'` |
| `AKA.STAGE` | `'STAGE'` | `'aka_stage'` |

**该怎么办**：把 `_StubMeta` 改成**抛 `AttributeError`**（修复顺序见
`MIGRATION_STATUS.md` 建议 #2）。**在改完之前，任何「跑通了」的结论都不足信。**

---

### P2. `000001` 同码 —— 指数与股票共用 6 位代码

**位置**：所有按裸 6 位代码推断市场的地方

`000001` 既是**上证指数**（沪市）又是**平安银行**（深市）。实测指数池与股票池
**交集 216 个** `000xxx`。

**已被证实的两次踩坑**：

1. **pytdx 适配器**（`markets/StockCN/datasource/pytdx_source.py`）：早先用一张
   统一前缀表过滤所有市场 → 沪市列表里的 `00xxxx` 全是指数 → 股票记录被填进
   上证指数的名字（`name='上证指数'` 却 `sse='sz'`）。
   **正解：按市场各自的前缀**（深 00/30、沪 60/68、北 82/92）。

2. **QMT 适配器**（`qmt_source.py` 的 `prefer_index` 参数，不要删）：按
   `is_stock_cn` 路由会把 `000905.SZ`（厦门港务）写进指数路径，
   **覆盖 `000905.SH`（中证500）的历史**。指数路径必须按号段显式定交易所。

**该怎么办**：**永远不要用裸 6 位代码推断市场。**

---

### P3. 空值契约**刻意不对称** —— 勿"统一"

**位置**：`markets/StockCN/kline83.py`、`markets/base_market.py`

| 函数 | 无数据时返回 | 原因 |
|:--|:--|:--|
| `get_kline_price_v3`（日线）| **`None`** | `_daily.py:105` 有 `if data_baseline is None` 分支依赖它 |
| `get_kline_price_min`（分钟）| **空对象**（非 None）| `_stock.py:138` 直接取 `.data` 且**未预初始化**目标变量；返回 None 会一路变成 `UnboundLocalError` |

两者看起来应该一致，**但不能**。改成一致会打断其中一个调用点。

**附带**：空表**必须保留两层 MultiIndex**。返回裸 `pd.DataFrame()`（单层空 Index）
会让下游 `get_level_values(level=1)` 抛 `Too many levels` —— 那是**形状错误**，
会被误读成 bug，而不是「这只标的没数据」。

---

### P3b. pytdx：**一次失败调用会毒死整条连接**

**位置**：`markets/StockCN/datasource/pytdx_source.py`

`get_security_list(2, 0)`（北交所）返回 `None` —— 而**那一次 None 之后，同一连接上
所有调用都失效**：`get_finance_info` 返回 None、`get_security_list` 返回空，
**且不抛任何异常**。

pytdx 是请求/响应式 socket，一个畸形响应让**字节流错位**，此后每次调用都读错位置。

**真实受害案例**：`fetch_stock_info` 曾静默返回 0 行并报 `skipped`，
看起来像「没有财务数据」，**实际是 `fetch_stock_list()` 内部那句「试北交所」
把连接打死了**。而那句试调用是为了「哪天上游修了能自动接上」才留的。

**该怎么办**：

- `market_enum` **默认不含 market 2**（已修）
- 若日后确要重试北交所，**必须用独立连接，用完即弃**
- **更一般的教训**：pytdx 的调用返回 `None`/空时，先怀疑连接已错位，
  而不是「上游没这个数据」

### P4. 静默覆盖落盘产物

**位置**：回测落盘（参照实现 `OneWaveQuant/GolemQ/benchmark/zen_bt.py`）

2026-09-16 前用固定名 `csi300_*`，**两次回测产物被静默覆盖**。现改为：

- 文件名含运行时间戳
- 目标已存在则自动换名（`-2` / `-3`…）
- **`util` 与 `trades` 的后缀成对决定一次** —— 否则「util 已存在但 trades 不存在」时
  会给这一对配上不同前缀，按前缀找同批产物的对账脚本就会错位

**改动落盘命名时，先看 `OneWaveQuant/extras/_reconcile.py`** —— 它按前缀定位产物。

---

## 二、数据与契约类

### P5. `vol` 必须重命名成 `volume`

**位置**：`markets/StockCN/kline83.py`

迁移把成交量存为 `vol`，但 **QUANTAXIS 口径与全部下游消费方用的是 `volume`**
（`markets/StockCN/fetch.py:829`、`GolemQ_old/analysis/ChipDistribution.py:65`）。
不重命名，真实消费方**静默拿不到该列**。

### P6. `vol` 必须是 int32，**绝不能是 int16**

实测最大值：`stock_1min` **35,455,100**、`index_1min` **131,132,928**；
int16 上限仅 **32,767**（溢出 1000~4000 倍）。

### P7. `stock_block` 读写库不一致

新树 `fetch.py` 读 `golemq.stock_block`，而旧 QMT writer 写的是
`quantaxis.stock_block`。**必须择一并让读写一致**，否则新 writer 会继续写错库。

### P8. `timedelta(hours=8.3)` —— 既有问题，**非重构引入**

`services/persistence/_stock.py:125`、`_daily.py:128`。老树
`scribe/persistence.py:681,940` **逐字相同**。

A 股应为 UTC+8，`8.3` 可疑；老树自身在同角色上还用过 `hours=8.5`
（`analysis/ChipDistribution.py:347` 等）→ **既有不一致**。仍是 bug，但**别当成回归**。

---

## 三、工具与配置类

### P9. ruflo 生成的 hook 会掉垃圾文件，且 `ruflo init` 会覆盖修复

**症状**：仓库根目录出现 0 字节的怪文件名（`cmd`、`'`、`$c`、`{corr!r}`、
`each_symbol`…）。

**根因**：hook 命令是 `cmd /c "IF EXIST "..." (...) ELSE (...)"`，JS 模板里的
`\"` 折叠成裸引号 → 嵌套未转义引号 → 命令被切碎，碎片被当成文件名创建。

**关键细节**：

- 上游 `@claude-flow/cli` **3.10.2 / 3.41.2 / 3.42.4 三个版本逐字节相同**，
  **升级修不了**。
- **执行 hook 的是 bash 不是 cmd**，所以修复须用 bash 语法
  （`${CLAUDE_PROJECT_DIR:-.}` / `$USERPROFILE`），**不能用 `%VAR%`** ——
  用 `%VAR%` 会得到「hook 全部静默失效」而垃圾文件消失，看起来像修好了。
- **验证必须带 shell 元字符**（`$()`、引号、花括号）。纯 `echo` 给**假阴性** ——
  这个坑我踩过。

**该怎么办**：修复脚本在 `tools/fix_ruflo_hooks.py`，跑 `--dry-run` 先看。
`ruflo init` 之后必须重跑。

### P10. `--sub` 分发以零参数调用订阅函数

**位置**：`cli/__main__.py`。而 `sub_l1_from_tencent(database_realtime)` 需要一个
位置参数 → `--sub l1_tencent` **必然 `TypeError`**。新订阅函数须适配该约定，
或先修分发器。

### P11. 笔记本无 markdown 头 = 论证丢失

`cookbooks/` 是**人用的**技术论证笔记本。当前 `ckpo_align_stock_turnover_rate.ipynb`
是 6 个 cell **全代码、零 markdown** —— 记录了「怎么跑」，没记录「为什么跑、得出什么」。

**每个笔记本开头应有三行**：问题 / 方法 / 结论（结论必须由人填，不可代写）。

---

## 附：本项目**刻意不做**的事（别当缺失补上）

| 事项 | 为什么不做 |
|:--|:--|
| 代理池 | 需持续维护与可用性验证，仓促造会引入随机失败被误当成上游限频。**只留注入点** |
| A 股默认成本/规则 | 成本与交易规则是通用概念、数值因市场而异；给默认值会让跨市场误用**静默算错一个数量级** |
| `datasource/` 机制层的市场知识 | 机制层不认识任何市场，注册由**导入实现层**触发 |
