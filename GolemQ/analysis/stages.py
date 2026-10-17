# coding:utf-8
"""Garident STAGE —— 用 K-Means 把收盘价聚成「支撑/压力带」，再判断当前价落在哪一带。

它是什么
========
`AKA.STAGE` 是**「当前收盘价所处的 Garident 聚簇带序号」**：

* 先拿最近 502 根收盘价做 K-Means（`K=1..14`），用 **KneeLocator** 在惯量曲线上找肘点
  定出真实簇数 —— 那几簇的价格区间就是「支撑/压力带」；
* 再看当前收盘价落在哪一带里，得 `STAGE ∈ {-0.5, 0, 0.5, 1, 1.5, …}`：
  `idx` = 带内、`idx+0.5` = 在带上方、`-0.5` = **跌破最底那一带**。

`BOOTSTRAP_STAGE_BEFORE` 就是它的下游：`Timeline_duration(STAGE < -0.00168)`
—— 阈值 `-0.00168` 实际**只命中 `STAGE == -0.5`**，也就是「跌破最底支撑带」这个
bootstrap（起步）状态。

来源
====
⚠️ **老树有这一族的「平行实现」两份**（`PITFALLS.md` 那句「平行实现不会报错，只会分叉」）：

| 老树位置 | 被调用？ | 名字 |
|:--|:--|:--|
| `GolemQ_old/analysis/stages.py:69-223` | ❌ **全树没有任何 `import stages`** —— 死文件 | `calc_garident_stages` |
| `GolemQ_old/fractal/vXII.py:124-267` | ✅ 真的被调（`vXII.py:649`）| `calc_garident_prices` |

**已机械化 diff 核对（2026-10-16）**：

* 两个**内层**函数 `calc_garident_price` / `calc_garident_stage` —— 归一化后**逐字相同**；
* 外层那个编排函数 —— **只差错误分支的写法**：死文件是
  `except Exception: traceback.print_exc(); continue`（赋值在 `try` **外**）；
  被调用那份多一个 `except ValueError`（只打印、不 `continue`，但赋值在 `try` **内**、照样被跳过）。
  **两侧的净效果相同**：某 checkpoint 失败 ⇒ 跳过它的赋值、继续下一个。

⇒ 本模块以**被调用那份的行为**为准，函数名取 `calc_garident_stages`
（死文件里的名字 —— 更达意；`vXII` 那个叫 `calc_garident_prices` 但算的其实是 stage）。
**不要再搬第三份。**

算法一字未改。

按 `DECISIONS.md` **D39 / D40** 的搬运惯例做了四处裁剪：

1. **删依赖**：`from GolemQ.utils.settings import DATABASE as DATABASE_GolemQ`
   —— 那个名字在原文件里**只出现在 import 行、正文一次都没用**（D39 同款处置）。
   连带删掉 `pd.set_option('expand_frame_repr')`（改全局显示选项的副作用）、
   `import copy`、`from kneed import DataGenerator`（未用）、以及那段
   `try/except` 的 `sys.path` 自适应 hack（新树 `analysis/` 已在包内）。
2. **常量解耦**：`GolemQ.utils.parameter` → `GolemQ.core.constants`。
3. **诊断打印改成 `echo` 回调**（⚠️ **这是本模块唯一一处刻意偏离逐字照搬**）：
   老树有 5 处 `print(...)` 与一处 `traceback.print_exc()`，都跑在**数据异常**的路径上。
   * 直接照搬会**打花 banner**（`PITFALLS.md` **P22**：banner 活跃时子层不许裸 `print`）；
   * 直接删掉又会让「这个 checkpoint 被跳过了」变成**静默** —— 而这正是本项目反复
     吃亏的那类失效。

   故改成：`calc_garident_stages(..., verbose=False, echo=None)`，异常**按 checkpoint
   计数**，`verbose` 时才经 `echo` 吐出。**计算值一字未变**（打印不影响返回值），
   所以 D39/D40 的 `array_equal` 逐值对拍照常成立。

⚠️ **`calc_garident_price` / `calc_garident_stage` 保持纯函数**（无 `verbose`/`echo`）——
它们是逐 bar 调用的内层，把回调穿进去会让签名与老树偏离，且它们**本来就没返回值之外的效果**。
"""
from __future__ import annotations

import warnings
from functools import cmp_to_key

import numpy as np
import pandas as pd

from GolemQ.core.constants import AKA, FIELD as FLD

__all__ = [
    'calc_garident_price',
    'calc_garident_stage',
    'calc_garident_stages',
    'garident_bands_to_ratio',
    'garident_bands_from_ratio',
]

#: 定带用的滚动窗口：取 checkpoint 之前这么多根收盘价。
#: 老树写死在 `calc_garident_stages` 里（`endpint_index - 502`），提到模块级只为好读。
GARIDENT_WINDOW = 502

#: K-Means 的簇数搜索范围 —— `KneeLocator` 在这条曲线上找肘点。
K_RANGE = range(1, 15)


def calc_garident_price(closep) -> list:
    """收盘价序列 → 排好序的「支撑/压力带」列表 ``[[上沿, 下沿], ...]``。

    做法照 <https://towardsdatascience.com/using-k-means-clustering-to-create-support-and-resistance-b13fdeeba12>：
    用 K-Means 把价格聚成若干簇，簇的 `[max, min]` 就是一条带。

    ⚠️ **肘点由 `KneeLocator` 自己定**（`curve='convex'` / `direction='decreasing'`），
    所以簇数不是常数 —— 不同标的、不同窗口会得到不同条数的带。
    这决定了 `STAGE` 的上界（`(knee-1)+0.5`），下游**不要假设它是固定宽度**。

    ⚠️ **K-Means 有随机性**：老树没设 `random_state`，本函数**也照旧不设** ——
    设了会改变结果、破坏与老树的对拍。代价是**同一个输入两次调用可能得到不同的带**
    （`n_init` 默认值下的局部最优）。这是老树就有的性质，**不是本次引入的**。

    :param closep: 收盘价序列（`pd.Series` / `np.ndarray` / `list` 皆可，走 `np.array()`）
    :returns: `[[上沿, 下沿], ...]`，按上沿排序（老树用的是 `cmp_to_key(x[0]-y[1])`）

    ⚠️ **`sklearn` / `kneed` 是惰性 import**（本函数内），不是模块级 ——
    照 `renko_jit` 的「被间接导入时不该为此付代价」。它们**都不在
    `cli/bootstrap.py` 的 `MIN_PACKAGES`** 里（那两个是重型/小众依赖），
    所以 `import GolemQ` 不该因为缺它们而断。
    真要跑本函数时缺了，这里会给一句人话；若将这条链接进 `--save`，
    应把两者**提升进 `MIN_PACKAGES`**（那时缺了就是硬断，该由自检拦）。
    """
    try:
        from kneed import KneeLocator
        from sklearn.cluster import KMeans
        from sklearn.exceptions import ConvergenceWarning
    except ImportError as exc:      # pragma: no cover
        raise ImportError(
            'calc_garident_price 需要 scikit-learn 与 kneed（{}）。'
            '它们不在 MIN_PACKAGES 里，装一下：'
            'pip install scikit-learn kneed'.format(exc)) from exc

    X = np.array(closep)
    sum_of_squared_distances = []
    for k in K_RANGE:
        km = KMeans(n_clusters=k)
        with warnings.catch_warnings():
            # 老树显式忽略收敛告警 —— 簇数偏大时 KMeans 必然报，而这里是**故意**试大 K
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            km = km.fit(X.reshape(-1, 1))
        sum_of_squared_distances.append(km.inertia_)
    kn = KneeLocator(K_RANGE, sum_of_squared_distances, S=1.0,
                     curve="convex", direction="decreasing")
    kmeans = KMeans(n_clusters=kn.knee).fit(X.reshape(-1, 1))

    c = kmeans.predict(X.reshape(-1, 1))
    minmax = [[-np.inf, np.inf] for _ in range(kn.knee)]
    for i in range(len(X)):
        cluster = c[i]
        if X[i] > minmax[cluster][0]:
            minmax[cluster][0] = X[i]
        if X[i] < minmax[cluster][1]:
            minmax[cluster][1] = X[i]

    minmax.sort(key=cmp_to_key(lambda x, y: x[0] - y[1]))
    return minmax


def _is_missing(bands) -> bool:
    """`None` / `NaN` / 空 —— 「这一行没有带」。"""
    if bands is None:
        return True
    if isinstance(bands, float) and np.isnan(bands):
        return True
    return len(bands) == 0


def garident_bands_to_ratio(bands, anchor_close):
    """带（**绝对价位**）→ 相对**锚 bar** close 的**比值**带。

    为什么要做这件事（`docs/decisions/D49.md`）
    ========================================
    绝对价位一旦落库就**跟着复权过期** —— 前复权因子每次除权都会把整条历史重标，
    而库里那串绝对价不会动。**比值不会过期**：`calc_garident_price` 对整窗缩放是
    **尺度等变的**（实测 k=0.37/2.5/100 与 `k*基准` 的相对误差 **0.000e+00**），
    而未来除权对锚 bar 及整窗是**均匀**缩放 ⇒ 比值**数学上不变**。

    ⚠️ **"不变"是数学意义，不是逐位相等**：浮点下 `(b*k)/(p*k)` 与 `b/p` 会差
    ~1e-16（float64 机器精度，实测 1.1e-16）。**没有系统性漂移**才是这里要的性质 ——
    写测试时别用 `array_equal`（会假红）。

    ⚠️ **还原时乘的是锚 bar 的 close，不是当前 bar 的**
    ================================================
    `GAR_P` 是「最近一个 checkpoint 的**绝对价位水平**」，被 `ffill` 往后带；
    `STAGE` 问的是「**今天的**收盘价落在这些**当时的**价位水平的哪里」。
    若用**当前 bar** 的 close 缩放回去，等于让带跟着今天的价走 ⇒ 今天的价永远
    落在同一相对位置 ⇒ **`STAGE` 会塌成常数**（实测：第 -1/-60/-300 根全变 `0`，
    而原本是 `0/1/1`）。所以必须配 `FLD.GARIDENT_PRICE_ANCHOR` 一起落库。

    >>> garident_bands_to_ratio([(110.0, 100.0), (125.0, 111.0)], 110.0)
    [(1.0, 0.9090909090909091), (1.1363636363636365, 1.009090909090909)]
    >>> garident_bands_to_ratio(None, 110.0) is None
    True
    >>> garident_bands_to_ratio(float('nan'), 110.0) is None
    True

    :param bands: `calc_garident_price` 的产物（`[[上沿, 下沿], ...]`），或 `None`/`NaN`
    :param anchor_close: **锚 bar**（算这组带那一根）的收盘价 —— 复权后的
    :returns: 同样形状的比值带；没有带时返回 `None`
    """
    if _is_missing(bands):
        return None
    k = float(anchor_close)
    return [(float(a) / k, float(b) / k) for a, b in bands]


def garident_bands_from_ratio(ratio_bands, anchor_close):
    """比值带 → **绝对价位**带（乘**锚 bar** 的 close）。

    ⚠️ 传进来的 `anchor_close` 必须是**锚 bar 那个日期**的（已复权的）close ——
    取自 K 线，**不是**当前行的 close。理由见 :func:`garident_bands_to_ratio`。

    >>> garident_bands_from_ratio([(1.0, 0.5), (1.2, 1.05)], 100.0)
    [(100.0, 50.0), (120.0, 105.0)]
    >>> garident_bands_from_ratio(None, 100.0) is None
    True
    """
    if _is_missing(ratio_bands):
        return None
    k = float(anchor_close)
    return [(float(a) * k, float(b) * k) for a, b in ratio_bands]


def calc_garident_stage(curr_closep: float, curr_garident_prices) -> float:
    """当前收盘价落在第几条带里 → `STAGE`。

    * 落在带 `idx` 的 `[下沿, 上沿]` 之内 → `idx`
    * 高于带 `idx` 的上沿 → `idx + 0.5`
    * 低于**最底那一带**的下沿 → `-0.5`
    * `curr_closep` 是 `NaN` → `NaN`
    * 都判不出来（理论上的兜底）→ `-1`

    ⚠️ 传进来的 `curr_garident_prices` 是**逐 bar 值**，可能不是 list：
    `float`（老树把某一根的 `GAR_P` 存成了标量）返回 `NaN`；`tuple` 转 list；
    **其它类型**（老树那里是 `print` 一句然后继续）在这里**原样继续**，
    于是 `for` 循环走 0 次、落到 `ret_stage == -1` 分支 —— **返回值与老树相同**。

    :param curr_closep: 当前收盘价
    :param curr_garident_prices: `calc_garident_price` 的产物（或 `NaN`/`tuple`）
    """
    ret_stage = -1
    if isinstance(curr_garident_prices, list):
        pass
    elif isinstance(curr_garident_prices, float):
        return np.nan
    elif isinstance(curr_garident_prices, tuple):
        curr_garident_prices = list(curr_garident_prices)

    for idx, garident_price in enumerate(curr_garident_prices):
        if (curr_closep <= garident_price[0]) and (curr_closep >= garident_price[1]):
            ret_stage = idx
            break
        elif curr_closep >= garident_price[0]:
            ret_stage = idx + 0.5
        elif (idx == 0) and (curr_closep < garident_price[1]):
            ret_stage = idx - 0.5

    if ret_stage == -1 and np.isnan(curr_closep):
        return np.nan
    return ret_stage


def calc_garident_stages(checkpoints: pd.DataFrame, features: pd.DataFrame,
                         verbose: bool = False, echo=None) -> pd.DataFrame:
    """在 `checkpoints` 逐行定带，把 `AKA.STAGE` 及其伴随列写进 `features`。

    写了 **5 列**：

    ==========================  ================================================
    列                           含义
    ==========================  ================================================
    `AKA.STAGE`                 当前价所处的带序号（本模块的主角）
    `FLD.GARIDENT_PRICE`        那根 checkpoint 上的带（**`tuple`**，ffill）
    `FLD.GARIDENT_PRICE_COUNT`  带的总条数
    `FLD.GARIDENT_PRICE_CLEARANCE`  条数 − STAGE（离最顶还差几条）
    `FLD.GARIDENT_PRICE_ANCHOR`  **这组带是哪个 checkpoint 算的**（日期串；⚠️ **老树没有这一列** —— 8.3 落比值后还原要用它，见 D49）
    ==========================  ================================================

    ⚠️ **老树那条 `STOCK_TREND_*` 链整条删掉了**（2026-10-16，用户定）
    ============================================================
    老树还多写 4 列 —— `STOCK_TREND_CHECKOUT_CLOSE` → `STOCK_TREND_RETURNS`
    → `STOCK_TREND_RETURNS_GARIDENT`（外加独立的 `STOCK_TREND_BEFORE`）。
    **连计算一起删**，理由是三条实测：

    * **老树从来没落过库** —— 五个 `*_columns_of_persistence` 清单**全都不含**
      `STOCK_TREND_*`（`GolemQ_old/scribe/persistence.py`）；
    * **新树 8.3 三库里这些列名一个都没出现过** ⇒ 删了不影响任何已存数据与读路径；
    * **没有真消费** —— 老树那几处全是 benchmark 的「选列 / 豁免」
      （`compact.py:4007` 的 `column not in [...]`），**不是拿它做判断**；
      唯一像消费的 `benchmark/scan_old.py:1174` 是**给它重新赋值**
      （那是另一条平行链，不是读本函数的产物）。

    ⚠️ 所以：**别照着老树把它加回来**。真需要时先找出消费方。

    ⚠️ **`AKA.STAGE` / `GARIDENT_PRICE_COUNT` 只在原本是 `NaN` 时才写**
    （`np.where(np.isnan(...), 新值, 旧值)`）—— 所以**重复调用是幂等的**，
    且 `features` 里预先存在的 `STAGE` 会被保留。别改成无条件覆盖。

    ⚠️ **入参 `features` 会被就地修改**（老树如此）。要留原样请调用方自己 `copy()`。

    :param checkpoints: 逐行一个「检查点」，索引需能 `features.index.get_loc(...)`
    :param features: 承载列的帧，需有多层索引（后面按 `level=[0]` groupby）
    :param verbose: 是否报「哪几个 checkpoint 定不出 stage」（老树的 5 处诊断 `print`）
    :param echo: `verbose=True` 时的输出汇（默认 `print`）—— 见模块 docstring 第 3 条
    """
    say = echo or print

    if FLD.GARIDENT_PRICE in features.columns:
        ret_garident_prices = features[FLD.GARIDENT_PRICE].values
    else:
        ret_garident_prices = [np.nan for _ in range(len(features))]

    features = features.reindex(columns=list(set([
        *features.columns,
        *[AKA.STAGE,
          FLD.GARIDENT_PRICE_COUNT,
          FLD.GARIDENT_PRICE,
          FLD.GARIDENT_PRICE_CLEARANCE,
          # ⚠️ **老树没有这一列** —— 8.3 起 `GAR_P` 落**比值**，还原要知道乘哪个
          # 日期的 close。见 `docs/decisions/D49.md`。
          FLD.GARIDENT_PRICE_ANCHOR]])))

    #: 每一行的带是**哪个 checkpoint** 算出来的（日期字符串）。与 `GAR_P` 同步 ffill。
    ret_anchor_dates = [np.nan] * len(features)

    # 老树在这里直接 `except Exception: traceback.print_exc(); continue` ——
    # 「定带失败就跳过这个 checkpoint」。**跳过这件事本身保留**（它决定结果），
    # 只把「跳过」变得**可数、可按 verbose 报出来**。
    skipped = []

    for checkpoint_timestamp, checkpoint_features in checkpoints.iterrows():
        endpint_index = features.index.get_loc(checkpoint_timestamp)
        startpoint_index = (endpint_index - GARIDENT_WINDOW) if (endpint_index > GARIDENT_WINDOW) else 0
        try:
            ret_garident_price = calc_garident_price(
                features[AKA.CLOSE].values[startpoint_index + 1:endpint_index + 1])
            stage = calc_garident_stage(
                checkpoint_features[AKA.CLOSE],
                ret_garident_price)
        except Exception as exc:            # noqa: BLE001 一个 checkpoint 失败不该毁掉其余
            skipped.append((checkpoint_timestamp, f'{type(exc).__name__}: {exc}'))
            continue
        features.loc[checkpoint_timestamp, AKA.STAGE] = stage
        ret_garident_prices[endpint_index] = tuple(ret_garident_price)
        ret_anchor_dates[endpint_index] = pd.Timestamp(
            features.index.get_level_values(level=0)[endpint_index]).strftime('%Y-%m-%d')

    if verbose and skipped:
        say('[stages] {} 个 checkpoint 定带失败（该行 STAGE 留原值）：'.format(len(skipped)))
        for ts, why in skipped[:10]:
            say('    {} {}'.format(ts, why))
        if len(skipped) > 10:
            say('    …另有 {} 个'.format(len(skipped) - 10))

    features[FLD.GARIDENT_PRICE] = ret_garident_prices
    features[FLD.GARIDENT_PRICE] = features[FLD.GARIDENT_PRICE].ffill()
    # 锚日期与 `GAR_P` **同步 ffill** —— 两者必须同行前进，否则读侧会拿错锚的 close
    features[FLD.GARIDENT_PRICE_ANCHOR] = pd.Series(
        ret_anchor_dates, index=features.index).ffill()

    # ⚠️ 老树这里还有 `STOCK_TREND_CHECKOUT_CLOSE` / `RETURNS` / `BEFORE` /
    # `RETURNS_GARIDENT` 四列的计算与落列 —— **已整条删除**，理由见 docstring。

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        ret_garident_stages = features.groupby(
            level=[0],
            group_keys=True).apply(
                lambda x: calc_garident_stage(
                    x[AKA.CLOSE].item(),
                    x[FLD.GARIDENT_PRICE].item()))
        features[FLD.GARIDENT_PRICE] = features[FLD.GARIDENT_PRICE].ffill()
        ret_garident_stages_count = features.groupby(
            level=[0],
            group_keys=True).apply(
                lambda x: len(x[FLD.GARIDENT_PRICE].item())
                if ((isinstance(x[FLD.GARIDENT_PRICE].item(), list)) or
                    (isinstance(x[FLD.GARIDENT_PRICE].item(), tuple))) else 1)

    features[AKA.STAGE] = np.where(
        np.isnan(features[AKA.STAGE]),
        ret_garident_stages,
        features[AKA.STAGE]).astype(np.float16)
    features[FLD.GARIDENT_PRICE_COUNT] = np.where(
        np.isnan(features[FLD.GARIDENT_PRICE_COUNT]),
        ret_garident_stages_count,
        features[FLD.GARIDENT_PRICE_COUNT]).astype(np.float16)
    features[FLD.GARIDENT_PRICE_CLEARANCE] = \
        features[FLD.GARIDENT_PRICE_COUNT] - features[AKA.STAGE].astype(np.float16)
    return features
