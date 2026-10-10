# examples

可独立运行的**演示**。与 `GolemQ/` 下的库代码的区别：这里的文件**不被任何模块 import**，
只为「把这条链跑给人看」而存在。

> ⚠️ **子目录 [`deprecated/`](deprecated/) 里的示例已经跑不通** —— 那是**上游服务终止**
> （不是没配好），逐个目录的 README 里写了原因。别把它与上面这些能跑的混为一谈。

## `app_pivot.py` —— 缠论盘整中枢（箱体）演示

```bash
streamlit run examples/app_pivot.py
```

侧边栏可选：股票代码 / **K 线频率（1、5、15、30、60min）** / 是否显示箱体 / 是否显示
GG-DD 延伸边界 / 是否加长历史。

### 这条链上有什么

| 环节 | 落点 | 说明 |
|:--|:--|:--|
| 取数 | `GolemQ.get_active_market().get_kline_price_min` | 市场实例；`frequency` 由此传入（无中转门面）|
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

## `app_renko.py` —— 砖块图（RENKO）演示

```bash
streamlit run examples/app_renko.py
```

侧边栏可选：股票代码 / **K 线频率（日线 + 1、5、15、30、60min）** / 砖高（自动 ATR 中位数
或手动）/ 最多用多少根 K 线。

### 这条链上有什么

| 环节 | 落点 | 说明 |
|:--|:--|:--|
| 取数 | `get_active_market()` | 日线 `get_kline_price_v3`、分钟 `get_kline_price_min`（**两者签名不同**，日线无数据返回 `None`）|
| ↓ | `markets/StockCN/kline83.py` | MongoDB **8.3** 时序集合，**已前复权** |
| 砖块序列 | `analysis/renko.renko` | `set_brick_size(auto=True)` → `talib.ATR(14)` 中位数定砖高 |
| 特征列 | `analysis/renko.renko_trend_cross_func` | S 族（numba 压缩砖）/ L 族（1200 bar 分窗搜砖高）/ `RENKO_BAR` 合成 |

### 两个页签看的是同一份数据的两个面

1. **砖块图（按砖序）** —— 横轴是**第几块砖**，与时间无关。一根 K 线可能产生多块砖，
   也可能连续多根都不产生（横盘）。这才是 Renko 剔噪的方式。
   砖身按 `renko_prices` + `renko_directions` 摆（方向 +1 → `[p-砖高, p]`）。
2. **特征叠加（按时间轴）** —— K 线 + S/L 两族的上下界台阶线 + `RENKO_BAR` 信号标记。

### 三条刻意设计，别当 bug「修」

1. **砖块图的「砖」与特征列的「砖」不是同一批数。** 砖块图用 `class renko` 的**真实砖序**；
   特征列的 S 族用同一套砖高，但 **L 族另起一套**（按 1200 bar 分窗、每窗自己搜最优砖高）。
   这正是 small / large 的由来 —— 把两者画在一条轴上会以为实现错了。
2. **S 族是 `float16`。** `renko_trend_cross_func` 末尾对 `RENKO_TREND_S` / `_LB` / `_UB`
   做 `astype(np.float16)`（旧树如此）。看图无碍，但拿这几列做算术要知情。
3. **`RENKO_PRICE_S/L` 是砖位价，不是收盘价**；`RENKO_OPTIMAL` 只在每个 1200 bar 窗口
   首行有值、其余为 **0**；这三列旧树也**写后无人读**（`RENKO_PRICE_S` 被下游统一 drop）。

### ⚠️ `source_aligned` 的两列在下跌砖上是**反序**的

`class renko` 的 `source_aligned[idx]` 在下跌砖上返回 `[上一砖位, 上一砖位 - 砖高]`
（即 **lb > ub**）。本 demo 走的是 `renko_chart`，它的 lb/ub **有序** —— 实测两族
都满足 `lb <= ub`。要走 `source_aligned` 的那条线（另案的 `calc_renko_atr_vX`）
必须自己 `min/max`。详见 `analysis/renko.py` 的模块 docstring。
