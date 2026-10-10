# GolemQ

**A 股量化框架**：行情取数 → 复权 → 落库 → 回测 → 实时订阅，全部落在**一个 MongoDB 8.3** 上。

> ⚠️ **本项目正在重构中**：新树 `GolemQ/` 逐模块替换老树 `GolemQ_old/`（后者不入库）。
> 进度与下一步见 [`HANDOFF.md`](HANDOFF.md)，遗留缺陷见 [`MIGRATION_STATUS.md`](MIGRATION_STATUS.md)。

## 它是什么 / 不是什么

- ✅ A 股（沪深北）**日线与分钟线**的取数 → 复权 → 落库 → 消费，一条链
- ✅ **实时 L1/L2 快照订阅**（腾讯源）→ 按日时间序列集合 `realtime_YYYY-MM-DD`
- ✅ 回测与组合（`portfolio/`，策略与引擎分离）
- ❌ **不是** “quantum-inspired algorithms” —— 那是 `pyproject.toml` 里一句**过时的模板残留**

## 环境要求

| | |
|:--|:--|
| Python | **≥ 3.12** |
| MongoDB | **8.3，且只用 8.3** —— 时间序列集合是数据模型的基础 |
| 平台 | Windows x64 / 主流 Linux |

## 安装

```bash
conda create -n GolemQ python=3.12
conda activate GolemQ
pip install -e .
```

⚠️ `pyproject.toml` **不声明运行期依赖** —— 真实依赖表在 `GolemQ/cli/bootstrap.py` 的
`MIN_PACKAGES`（全树唯一的声明处），CLI 启动自检会逐条核。核心几条：

| 包 | 为什么 |
|:--|:--|
| `numpy >= 2.0` / `pandas >= 2.3` / `polars >= 1.24` | 计算与列式构造 |
| `pymongo >= 3.0` | 8.3 的唯一入口 |
| `pytdx` | **K 线的唯一数据源**（通达信协议） |
| `tqdm` | 进度条 |
| `numba >= 0.61` | **可选加速器**：`analysis/peak.py` 与 `analysis/regtree_jit.py` 用它（缺了会降级/报错，但 `--save` 主链路不依赖它）|

可选源（缺 token 就自动不可用，不影响主链路）：`akshare`、`tushare`、`tdxaidata`。

## 配置：全在 `~/.GolemQ/`（**凭证不入库**）

```
~/.GolemQ/settings/config.ini      # MongoDB uri / 钉钉 / Server酱 / tdxaidata token …
~/.GolemQ/settings/tdx_hosts.json  # 通达信服务器池（每周探活自动写）
```

## 用法

```bash
python -m GolemQ.cli --save tdx        # 参考数据 + K线(stock/index/etf × day,1,5,15,30,60min) + 复权
python -m GolemQ.cli --save-coverage   # 只读：K 线覆盖缺口报告
python -m GolemQ.cli --sub l1_tencent  # 实时 L1 快照订阅（腾讯）
python -m GolemQ.cli --help
```

```python
from GolemQ import get_active_market

market = get_active_market()                                                    # 当前激活市场
res, code = market.get_kline_price_v3(['600519'], start='2024-01-01')           # 日线
res, code = market.get_kline_price_min(['600519'], start='2026-09-01',
                                       frequency='5min')                        # 分钟线
df = res.data        # MultiIndex (ts, code)，已前复权
```

`frequency` 收 `1min / 5min / 15min / 30min / 60min`（默认 `60min`）；它是**分钟线独有**
的形参，日线走 `get_kline_price_v3`。两者**默认 `realtime=True`**：读到的历史会与当天的
`realtime_<日期>` tick 合成。

⚠️ **没有「市场无关的门面层」** —— `GolemQ/fetch/` 已于 2026-10-10 整包删除（那层只是
`return get_active_market().xxx(...)` 的改名）。取数一律**经市场实例**；不想指明市场就
`get_active_market()`（尊重切换）或 `get_default_market()`（始终系统默认的那个）。

## 数据规模（2026-10 实测，供参考）

| 集合 | 行数 |
|:--|--:|
| `stock_1min` | **21.4 亿** |
| `stock_day` | 1792 万 |
| `index_day` / `etf_day` | 457 万 / 322 万 |

整个 `golemq_stock_cn`：**逻辑 213.6 GB → 磁盘 61.5 GB**（时间序列列式压缩 ≈3.5×），
另有索引 7.1 GB。

## 文档是资产，按顺序读

| 顺序 | 文档 | 读它干什么 |
|:--|:--|:--|
| 0 | `HANDOFF.md` | **当前进度与下一步** |
| 1 | `PITFALLS.md` | 已知陷阱（看起来像 bug 的刻意设计，**勿“修正”**） |
| 2 | `DECISIONS.md` | 架构决定与**弃案理由** |
| 3 | `GLOSSARY.md` | 行话（第四节几乎全是老代码继承） |
| 4 | `API_INDEX.md` | 模块导航索引（名称 + 一行摘要，由 `tools/gen_api_index.py` 生成）|
| 5 | `RESTRUCTURE_PLAN.md` / `MIGRATION_STATUS.md` | 重构方案与遗留缺陷 |

## 测试

```bash
python GolemQ/test_cases/run_tests.py     # 全量回归（跑完会打印条数与通过/失败）
```

## 已知状态（诚实交代）

- **MiniQMT 自 2026-10-01 停服** ⇒ QMT 那条路**只留结构**（`QMT_SOURCE_ENABLED = False`）
- `tushare` 适配器是**骨架**，缺 token 未启用；`iwencai` 尚未实现
- 重构进行中 —— `MIGRATION_STATUS.md` 有未闭合清单

## 许可证

MIT，见 [`LICENSE`](LICENSE)。

## 联系

`4910163@qq.com` ｜ GitHub [@Rgveda](https://github.com/Rgveda) ｜ 知乎 @阿财
