# examples

可独立运行的**演示**。与 `GolemQ/` 下的库代码的区别：这里的文件**不被任何模块 import**，
只为「把这条链跑给人看」而存在。

## `app_pivot.py` —— 缠论盘整中枢（箱体）演示

```bash
streamlit run examples/app_pivot.py
```

侧边栏可选：股票代码 / **K 线频率（1、5、15、30、60min）** / 是否显示箱体 / 是否显示
GG-DD 延伸边界 / 是否加长历史。

### 这条链上有什么

| 环节 | 落点 | 说明 |
|:--|:--|:--|
| 取数 | `GolemQ.fetch.kline.get_kline_price_min` | 门面；`frequency` 由此传入 |
| ↓ | `markets/StockCN/kline83.py` | MongoDB **8.3** 时序集合，**已前复权** |
| 分笔 | `czsc.CZSC(bars).bi_list` | pip `czsc`（本环境 0.7.10） |
| 中枢 | `GolemQ.analysis.pivot` | 中枢识别 / 走势分类 / 绘图 |
| 算法核心 | `GolemQ.analysis._zs.find_zs` | 自旧树单独抽出的那一个函数 |

### 三条刻意设计，别当 bug「修」

1. **箱体不铺满整段。** `ZG/ZD` 自**中枢成立 bar** 起才画，`GG/DD` 是逐根刷新极值的
   台阶线。一上来就画到整段极值 = **未来函数**（用了当时还不知道的价格）。
   口径的实现与证明在 `GolemQ/analysis/pivot.py` 的 `causal_pivot_series`。
2. **总是截到最后 4096 根。** 读取器的默认窗口是按**小时**算的（60min 约 2000 根），
   切到 5min/1min 会变成两万根以上 —— 画不动，也不是演示要看的粒度。
3. **无数据走警告，不走异常。** 取不到 K 线时给 `st.warning` 而不是报错：
   `czsc.CZSC([])` 会抛 `IndexError`（`czsc/analyze.py:228` 直接取 `bars[0].symbol`），
   那是把「这只票没有数据」报成了程序错 —— 空结果与错误必须能分开。

### 与旧树的差别

数据源从 QUANTAXIS 4.4 的 `stock_min`（`type='60min'`）换成 8.3 的 `stock_*min`；
多出频率选择（旧树门面拿不到 `frequency`）；中枢层是同一套口径的搬运
（逐字段对拍过：5 只标的 × 全部字段 0 差异）。

旧树还有一份 `app_pivot_pl.py`（polars 重特征流水线版），**未搬** ——
它依赖约 1.5 万行的上游模块与 8.3 里不存在的集合。详见 `HANDOFF.md`。
