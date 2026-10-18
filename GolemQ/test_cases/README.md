# 测试

## 怎么跑

```bash
python GolemQ/test_cases/run_tests.py                                     # 全部（递归发现子目录）
python -m unittest GolemQ.test_cases.markets.test_kline83 -v              # 单模块
python -m unittest GolemQ.test_cases.markets.test_kline83.<类>.<方法> -v  # 单方法
python -m unittest GolemQ.test_cases.test_doctests -v                     # 只跑 docstring 里的 doctest
```

⚠️ 模块路径是 **`GolemQ.test_cases.<层>.<模块>`**（2026-10-18 起按层分了子目录），
不是 `GolemQ.tests.*` —— 本 README 的**更早一版**把示例全写成 `GolemQ.tests.…`
（那目录**不存在**），照着敲必然 `ModuleNotFoundError`。

## 目录结构（按**被测层**分，不按功能随机分）

| 目录 | 放什么 | 被测代码 |
|:--|:--|:--|
| `markets/` | 行情 / 参考数据 / 实时 / 元数据 / 路由 / ChartScan 写库 | `GolemQ/markets/**` |
| `analysis/` | 指标算法、与老树逐值对拍的模块 | `GolemQ/analysis/**` |
| `cli/` | 环境自检、banner、命令注册表 | `GolemQ/cli/**` |
| `core/` | 常量 / 表现层 / 全树守卫 + **基础设施**（`agents/` `supervisor/` `gateway/`） | `GolemQ/core/**` 等 |
| 顶层 | `old_tree.py`（老树对拍夹具）、`run_tests.py`、`test_doctests.py` | — |

⚠️ **`old_tree.py` / `run_tests.py` / `test_doctests.py` 必须留在顶层** ——
前两个被多处用**绝对导入** `from GolemQ.test_cases.old_tree import …` 引用，
最后一个靠 `discover` 从顶层收集。

### ⚠️ 搬动测试文件时，`sys.path` 的**深度**要跟着改（会静默跳过）

48 个用例用 `os.path.join(os.path.dirname(__file__), '..', '..')` 推仓库根，
另有几处 `Path(__file__).parents[N]` 与 `'..', '..', 'GolemQ_old'`。
**少改一层不会报 `ImportError`** —— 而是让依赖老树的用例 `skipUnless(路径存在)` **静默跳过**。

2026-10-18 拆分实测：漏改 3 处 ⇒ **18 个用例从"跑"变成"跳过"**，而套件照样报 **OK**。
⇒ 拆完必须核对 **skipped 数**是否回到基线（当时 1，跳完变 2 才露馅）。

## 这一层放什么

| 放 | 不放 |
|:--|:--|
| `test_*.py` —— **能被发现、能失败**的用例 | **一次性 / 排查脚本** → [`tools/diagnostics/`](../../tools/diagnostics/) |
| `run_tests.py`（本层入口）、本 README | **示例 / demo** → [`examples/`](../../examples/) |

`run_tests.py` 用的是 `unittest discover(pattern='test_*.py')` —— 所以
`check_*` / `cleanup_*` / `debug_*` 这类脚本**本来就不参与测试**，混在这里只会让
「这一层有多少真测试」看不出来。2026-10-10 已把它们分出去。

## 约定

1. **命名**：文件 `test_*.py`（不以此开头就**不会被发现**）、类 `TestXxx`、方法 `test_*`。
2. **外部依赖用 mock 隔离**：DB / 网络 / QMT 客户端。
   ⚠️ **真库用例必须能跳过**（`unittest.skipUnless(能连上 8.3, …)`），且探活要**用短超时**
   （`serverSelectionTimeoutMS=1500`）—— `GQ_util_mongodb_client` 默认 30 秒，
   库没起时会把整个套件拖住半分钟。
3. **纯函数的 doctest 走 `test_doctests.py` 的 `DOCTEST_MODULES` 清单**，别新开收集器。
   想加就先读那个文件顶部的规则：**需要 DB / 网络 / QMT 的模块不该进清单**，
   判据是**逐函数**而不是逐模块。
4. **每个用例都要能失败**。占位式用例（只打印、无断言）没有价值 ——
   2026-10-10 就把 `test_chip_distribution` 从一个「只打印耗时的脚本」补成了带不变量的真用例。

## ⚠️ 别在这里枚举测试文件

**这个 README 的上一版**是一张「现有测试文件」清单（6 条），而其中
`test_market_align.py` / `test_market_crawler.py` **早已不存在**，真正的 30 多个用例一个没提。
清单会腐烂，而且腐烂时**不报错**。

要看有哪些用例，跑 `run_tests.py` 或问 `unittest`：

```bash
python -m unittest discover -s GolemQ/test_cases -p 'test_*.py' -v --locals 2>&1 | grep -c ' ... ok'
```
