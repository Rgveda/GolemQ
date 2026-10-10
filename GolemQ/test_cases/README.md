# 测试

## 怎么跑

```bash
python GolemQ/test_cases/run_tests.py                                        # 全部
python -m unittest GolemQ.test_cases.test_messenger -v                       # 单模块
python -m unittest GolemQ.test_cases.test_messenger.TestDingtalkConfig.test_check_config_success -v   # 单方法
python -m unittest GolemQ.test_cases.test_doctests -v                        # 只跑 docstring 里的 doctest
```

⚠️ 模块路径是 **`GolemQ.test_cases.*`**，不是 `GolemQ.tests.*` —— 本 README 的**上一版**
把三条示例命令全写成了 `GolemQ.tests.…`（那目录**不存在**），照着敲必然 `ModuleNotFoundError`。

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
