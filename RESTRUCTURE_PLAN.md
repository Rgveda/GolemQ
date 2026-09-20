# A 股特有代码的归属调整方案

> 建立日期：2026-09-20
> 判据来源：项目所有者提出 ——「若某个模块**只对 A 股有意义**，它就该在 `markets/StockCN/` 下；若它对多市场通用（哪怕当前只实现了 A 股），留顶层。」
> 状态：**方案，未执行**

---

## 一、为什么现在做

**因为大多数待迁模块还是 stub。**

| 模块 | 行数 | 性质 |
|:--|--:|:--|
| `models/alias.py` | 9 | stub |
| `fetch/concept.py` | 12 | stub |
| `features/reviews.py` | 12 | stub |
| `analysis/timeseries.py` | 18 | stub |
| `fetch/kline.py` | 22 | stub（已被 `kline83.py` 取代）|
| `models/risk.py` | 26 | stub |
| `features/empirical.py` | 27 | stub |
| `models/massive.py` | 42 | stub |
| `models/poolcoef.py` | **371** | **真实代码** |
| `pipeline/*.py` | 111–451 | 真实代码 |

**在 stub 阶段搬家，成本几乎为零；等实现完再搬，就要改一大片 import。**
这是本次调整最主要的时机理由 —— 越晚做越贵。

### 这不是重构引入的问题

老树的结构与新树**一样**：顶层同样有 `fetch/` `features/` `models/` `analysis/`，
而且老树的 `StockCN/` **同时存在于顶层和 `markets/` 下**（`StockCN/base.py kline.py realtime.py`）。
属历史遗留，重构只是原样继承。

---

## 二、逐模块判定

判据：**只对 A 股有意义 → 迁；对多市场通用 → 留**。

| 模块 | 判定 | 依据 |
|:--|:--|:--|
| `fetch/` | **迁** | 两个文件全是 A 股：`kline.py`（A股K线，且已被 `markets/StockCN/kline83.py` 取代）、`concept.py`（A股概念）|
| `features/` | **迁** | `empirical.py`/`reviews.py` 处理 ZEN_*_TIMING_LAG、MAGIC_NINE_TURNS 等 A 股特征 |
| `models/` | **迁** | `alias.py`(LTT/TRD)、`massive.py`(MAS)、`poolcoef.py`(筹码系数)、`risk.py`(RSK) —— 全是 A 股特征常量与模型 |
| `pipeline/` | **迁** | `mainstream_benchmark`(主力)、`poolcoef_benchmark`(筹码)、`compact_benchmark` —— 都是 A 股概念 |
| `cookbooks/` | **迁** | `ckpo_align_stock_turnover_rate.ipynb` —— A 股换手率对齐 |
| `analysis/` | **拆** | `timeseries.py` 是通用工具（重采样/时间轴）→ **留**；但里面的 stub 内容若只服务 A 股需具体判断 |
| `services/` | **留** | DB 操作层，按 CLAUDE.md 约定；其中 `iwencai`(问财) 是 CN 特有的，见「待定」 |
| `core/` `cli/` `supervisor/` `agents/` | **留** | 通用设施 |
| `gateway/xtquant/` | **留**（待议）| MiniQMT 是 A 股券商通道，但它是**交易网关**而非市场数据，且 `markets/StockCN/` 装不下交易职责。见「待定」 |

### 待定项（需所有者判断）

1. **`gateway/xtquant/`** —— MiniQMT 只服务 A 股，按判据该迁；但它是交易/下单通道，`markets/StockCN/` 目前只装市场数据。**建议保留在顶层**，因为「网关」是与「市场」并列的一类职责，不是它的子类。
2. **`services/iwencai.py`** —— 问财是 CN 特有服务，但它在 services 层（DB 操作层）的定位优先。**建议保留**。
3. **`pipeline/`** —— 若将来要支持港股同类 benchmark，留在顶层更合理。**当前三个 benchmark 全是 A 股概念，故判为迁。**

---

## 三、引用面（迁动的真实成本）

| 目标模块 | 外部引用 | 引用位置 |
|:--|--:|:--|
| `fetch/` | 1 | `services/persistence/_concept.py:56` |
| `features/` | 6 | `_concept.py:52,206`、`_daily.py:52`、`_review.py:51`、`_stock.py:52,53` |
| `models/` | 8 | `services/align.py:79`、`_concept.py:51`、`_schema.py:43,44`、`pipeline/mainstream_benchmark.py:34,38`、`pipeline/poolcoef_benchmark.py:34,38` |
| `analysis/` | 2 | `_concept.py:57`（`timeseries`）—— **留则不用改** |
| `pipeline/` | **0（真实）** | 仅 `GolemQ/__init__.py:31,43` 的包注册；**注意**其内部 benchmark 引用 `models/mainstream`、`models/poolcoef`、`pipeline/compact` —— 而这两个 models 模块**在新树根本不存在**（悬空 import）|
| `cookbooks/` | **0** | 无 |

**总计需改约 15 处 import。**

### 顺带发现的既有缺陷（搬迁时一并处理）

- `GolemQ/__init__.py:29` 的 `# from . import models` 是**被注释掉的**，`__all__` 里也没有 `models` —— 但 `services/` 在引用它。即 `models` 不在包注册内。
- `pipeline/*_benchmark.py` 的 import 目标 `models/mainstream.py`、`pipeline/compact.py` **在新树不存在**，是悬空 import（见 `MIGRATION_STATUS.md` 问题 4）。

---

## 四、搬迁方案

### 目标结构

```
GolemQ/
  markets/
    StockCN/
      __init__.py  base.py  kline83.py  refdata.py  refdata_save.py
      MONGODB83.md
      datasource/        ← 已有
      fetch/             ← 迁入
      features/          ← 迁入
      models/            ← 迁入
      pipeline/          ← 迁入
      cookbooks/         ← 迁入
  core/ analysis/ cli/ services/ agents/ gateway/ supervisor/   ← 不变
```

### 分批顺序（按成本从低到高）

**批次 1 —— 零引用，先搬（成本最低，用来验证搬迁流程）**

- `pipeline/` → `markets/StockCN/pipeline/`
  - 改 `GolemQ/__init__.py:31,43`（去掉注册；A 股特有的 benchmark 不该在包根注册）
  - 同时修 `mainstream_benchmark.py:34/38`、`poolcoef_benchmark.py:34/38` 的悬空 import
- `cookbooks/` → `markets/StockCN/cookbooks/`（0 引用，纯移动）

**批次 2 —— 单引用 + stub，搬完立刻可验证**

- `fetch/` → `markets/StockCN/fetch/`
  - 改 `services/persistence/_concept.py:56`
  - ⚠️ `fetch/kline.py` 已被 `kline83.py` 取代，**搬迁时一并删除**（当前无人引用它）

**批次 3 —— 多引用但多为 stub**

- `features/` → `markets/StockCN/features/`
  - 改 6 处：`_concept.py:52,206`、`_daily.py:52`、`_review.py:51`、`_stock.py:52,53`

**批次 4 —— 多引用且有真实代码（收尾）**

- `models/` → `markets/StockCN/models/`
  - 改 8 处（含 pipeline 内部 4 处）
  - 顺带修正 `GolemQ/__init__.py:29` 的注册状态（搬走后本就不该在包根注册，注释掉的 `import models` 应直接删除）

### 每批的验证

```bash
# 1. 无悬空引用
grep -rn "from GolemQ\.\(fetch\|features\|models\|pipeline\)" GolemQ/ --include=*.py

# 2. 四个 persistence 模块仍可导入（它们是最密集的引用方）
python -c "import GolemQ.services.persistence._concept, GolemQ.services.persistence._daily, GolemQ.services.persistence._review, GolemQ.services.persistence._stock"

# 3. 包仍可导入、市场仍注册
python -c "import GolemQ; from GolemQ import GQMARKETS; print(sorted(GQMARKETS))"

# 4. CLI 仍工作
python -m GolemQ.cli --save-status
```

---

## 五、风险

1. **`services/` 是最大引用方（12 处）** —— 而它是 DB 操作层，改 import 不影响逻辑，但要逐处确认符号名没变（只改模块路径，不改函数名）。
2. **`models/` 的 `_StubMeta` 会在拼错时静默返回假名** —— 搬迁期间若路径写错，`from ... import AKA` 会抛 `ImportError`（模块级），比字段名错误好定位。但仍建议先修 `_StubMeta`（见 `MIGRATION_STATUS.md` 问题 3），否则搬迁掩盖的字段问题仍无法暴露。
3. **`pipeline/` 从包根注册中移除后**，若有外部代码靠 `GolemQ.pipeline` 访问会断 —— 当前树内 0 引用，但**外部使用者未核实**。
4. **`cookbooks/` 若含硬编码相对路径**（notebook 常见），移动后会断。搬迁前需检查 notebook 内的路径。
5. **建议与 QUANTAXIS 解耦合并做** —— `services/` 的 12 处 import 与解耦要改的 `QA_util_*` 在同一批文件里，分两轮改等于改两遍同一个文件。

---

## 六、不在本方案范围

- QUANTAXIS 解耦（115 处，见 `MIGRATION_STATUS.md`）
- `_StubMeta` 改为抛 `AttributeError`
- `gateway/xtquant/` 与 `services/iwencai.py` 的归属（见「待定项」，建议不动）
- `StockHK/` 的实现

---

## 附：判据本身的边界

「只对 A 股有意义」在某些模块上不是非黑即白：

- `analysis/timeseries.py` 的重采样是**通用工具**，但 `services/persistence/_concept.py` 用它处理 A 股概念数据 —— **工具留顶层，用法在 A 股侧**，这是正确的分层。
- `pipeline/base.py`（抽象基类，含 joblib 并行）是通用的，却是三个 A 股 benchmark 的父类。**建议 `base.py` 跟随 `pipeline/` 一起迁**，因为它的抽象是为这三个子类设计的；若将来港股要复用，再提取到顶层不迟。
