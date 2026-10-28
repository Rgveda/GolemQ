# coding:utf-8
"""`*_reality_feat_*` —— **ChartScan（CV）结果的落点**。

从哪来
======
搬自 `Y:\\Projects\\GolemAI\\GolemQ\\collections\\features.py` 的
``GQ_save_metadata_reality`` / ``GQ_save_daily_metadata_reality`` /
``GQ_fetch_daily_metadata_reality``（那棵树把 CV 结果写在 4.4 的 ``golemq``）。
落点改为 **8.3 的 ``golemq_stock_cn``**（用户 2026-10-13 定）。

六个集合
========
``stock_reality_feat_{day,60min,15min}`` / ``index_reality_feat_{day,60min,15min}``。
命名照搬旧树，**不「现代化」**（`DECISIONS.md` D41「在库的不许改」）。

⚠️ **必须是普通集合，不能是时间序列**（`PITFALLS.md` P14）：本表要的是
「同一 ``(code, 键)`` 文档被反复 `$set` 不同列」的幂等 —— 时序集合不支持单文档
upsert、也不让建唯一索引，且 `ensure_indexes` 会把那个失败**吞掉** ⇒ 静默写重复行。
理由与 `metadata_save.py` 的「为什么是普通集合」同源。

⚠️ **只 `$set` 自己那几列**（`upsert_fields`，不是 `save_collection`）——
同一个 ``(code, date_stamp)`` 文档由**三个生产者**分别写不同字段：

======================  ==============================================
生产者                   写的列
======================  ==============================================
日线                     ``ClsMaj`` / ``ConfidenceMaj`` / ``CoordinatesMaj``
15min                   ``ClsMin`` / ``ConfidenceMin`` / ``CoordinatesMin`` / ``ChartScaned``
60min                   ``Class`` / ``Confidence`` / ``Coordinates`` / ``ChartScaned``
「没探到」的标记路径     只有 ``ChartScaned``
======================  ==============================================

`ReplaceOne` 会把别人的列整片抹掉 —— 这正是 `upsert_fields` 存在的理由
（与旧树的 ``UpdateOne(..., {'$set':…}, upsert=True)`` 语义**逐条等价**）。

两处**刻意的小偏离**（都无害，记在这里免得下一个人以为是漏搬）
========================================================
1. ``created_at`` 旧树写 ``int(time.mktime(dt.now().utctimetuple()))`` —— 那是
   「本机当前时间**当 UTC** 再转 epoch」，**带时区偏差**。8.3 里本表**一行都还没有**，
   且**没有任何消费方读它** ⇒ 改用规规矩矩的 ``int(time.time())``。
2. ``time_stamp`` **不加** `ts` 字段。它是本表的**唯一键**，口径是
   **真实 UTC epoch**（北京 15:00 → 07:00Z）。
   ⚠️ **2026-10-17 更正**：本处原写「北京墙上时间当 UTC」（照抄旧树 `features.py:122`
   的说法），**那是错的** —— 实测两边都是真实 UTC：
   `2026-01-05 15:00` 北京 → `1767596400`（= 07:00Z），而「墙上当 UTC」会给
   `1767625200`；4.4 存量同口径（`2021-10-13 10:30` → `1634092200`）。
   本函数的 doctest 值也一直是真实 UTC —— **注释与它自己的 doctest 相矛盾，以 doctest 为准**。
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from .kline83 import bj_date, epoch_seconds

#: 三个族。集合名 = ``{stock|index}_reality_feat_{family}``。
REALITY_FAMILIES = ('day', '60min', '15min')

#: **唯一键**（照搬旧树）：分钟族按 bar 的 ``time_stamp``，日族按 ``date_stamp``。
UNIQUE_KEYS = {
    'day': ('code', 'date_stamp'),
    '60min': ('code', 'time_stamp'),
    '15min': ('code', 'time_stamp'),
}

#: 除键之外的元字段（总是写）。其余列一律当**载荷**（`ClsMaj` 等）原样透传。
_META_FIELDS = ('code', 'date', 'datetime', 'date_stamp', 'time_stamp', 'created_at')


def reality_feat_name(family: str, market_type=None) -> str:
    """``(族, 市场类型)`` → 集合名。

    ⚠️ **ETF 2026-10-17 起独立成 ``etf_`` 前缀**（原与指数共用 ``index_``）
    ================================================================
    这一条**与 K 线侧对齐**：`kline83.market_prefix` 早在 **2026-09** 就把 ETF 拆到
    ``etf_*`` 了，而本函数当时没跟上 —— 于是**同一个 ETF 的 K 线在 ``etf_*``、
    它的 CV 结果却在 ``index_*``**。这里补上，修掉那个分叉。

    ⚠️ **``FUND_CN`` 仍走 ``index_``** —— 与 `market_prefix` **同一条纪律**
    （``50x`` 封基/LOF/分级，本就不在 ``etf_*`` 里）。**别顺手"统一"进来。**

    >>> reality_feat_name('day', 'stock_cn')
    'stock_reality_feat_day'
    >>> reality_feat_name('60min', 'index_cn')
    'index_reality_feat_60min'
    >>> reality_feat_name('15min', 'etf_cn')     # ⚠️ 2026-10-17 起是 etf_，不再是 index_
    'etf_reality_feat_15min'
    >>> reality_feat_name('day', 'fund_cn')      # fund 仍与指数同前缀
    'index_reality_feat_day'
    """
    if family not in REALITY_FAMILIES:
        raise ValueError(f'未知的 reality 族 {family!r}；可用: {REALITY_FAMILIES}')
    from GolemQ.core.constants import MARKET_TYPE
    if market_type == MARKET_TYPE.STOCK_CN:
        prefix = 'stock'
    elif market_type == MARKET_TYPE.ETF_CN:
        prefix = 'etf'
    else:                       # INDEX_CN / FUND_CN / None …
        prefix = 'index'
    return f'{prefix}_reality_feat_{family}'


def _db():
    """8.3 的库句柄。**函数据导入是刻意的** —— `markets/StockCN/__init__.py` 里的
    `GOLEMQ_STOCK_CN` 定义在文件中部，模块级导入会撞上半初始化的 `ImportError`。
    同层邻居（`metadata_save._db` / `kline_save._db` / `refdata._stock_cn_db`）都这么写。
    """
    from . import GOLEMQ_STOCK_CN
    return GOLEMQ_STOCK_CN


def reality_feat_rows(features: pd.DataFrame, family: str, created_at=None,
                      write_nulls: bool = False) -> list:
    """CV 结果帧 → ``*_reality_feat_*`` 文档列表。**纯函数**（不碰 DB）。

    :param features: 索引为 ``(时间, code)`` 两层（第 0 层 datetime64、第 1 层 6 位代码），
        其余列即载荷（``ClsMaj`` / ``ConfidenceMaj`` / ``CoordinatesMaj`` …）。
    :param family: :data:`REALITY_FAMILIES` 之一。日族**不写** `datetime`/`time_stamp`。
    :param created_at: 统一写入的 `created_at`（秒）；``None`` = 现在。

    >>> import pandas as pd
    >>> idx = pd.MultiIndex.from_tuples(
    ...     [(pd.Timestamp('2026-01-05 15:00'), '600519')], names=['ts', 'code'])
    >>> df = pd.DataFrame({'Class': ['Buy'], 'Confidence': [0.51],
    ...                    'Coordinates': [1767600000000.0]}, index=idx)
    >>> d = reality_feat_rows(df, '60min', created_at=0)[0]
    >>> d['code'], d['date'], d['datetime'], d['Class']
    ('600519', '2026-01-05', '2026-01-05 15:00:00', 'Buy')

    `time_stamp` / `date_stamp` 都是**真实 UTC** epoch（北京 15:00 → 07:00Z；
    ⚠️ 别信旧注释说的"墙上时间当 UTC"，见模块 docstring 的更正条）；

    >>> d['time_stamp'], d['date_stamp']
    (1767596400, 1767542400)

    日族**不新增** `datetime` / `time_stamp`（与旧树 `GQ_save_daily_metadata_reality` 一致）：

    >>> d2 = reality_feat_rows(df, 'day', created_at=0)[0]
    >>> 'datetime' in d2, 'time_stamp' in d2, d2['date_stamp'] == d['date_stamp']
    (False, False, True)

    但调用方**自带** `datetime` 列时要**格式化成字符串**再透传 —— 15min / 60min 的检出
    会同时往日族写一笔，那个 `datetime` 是 `Timestamp`；不转就会以 **BSON date** 落库，
    与 4.4 存量（字符串）成为同一列的两种类型：

    >>> idx2 = pd.MultiIndex.from_tuples(
    ...     [(pd.Timestamp('2026-01-05 15:00'), '600519')], names=['datetime', 'code'])
    >>> df3 = df.assign(datetime=pd.Timestamp('2026-01-05 15:00')).set_index(idx2)
    >>> d3 = reality_feat_rows(df3, 'day', created_at=0)[0]
    >>> d3['datetime']
    '2026-01-05 15:00:00'
    """
    if family not in REALITY_FAMILIES:
        raise ValueError(f'未知的 reality 族 {family!r}；可用: {REALITY_FAMILIES}')
    if features is None or len(features) == 0:
        return []
    if created_at is None:
        created_at = int(time.time())

    out = features.copy()
    ts0 = pd.to_datetime(out.index.get_level_values(level=0))
    out['code'] = out.index.get_level_values(level=1).astype(str).str[:6]
    # `date` 取**当日**（北京），`datetime` 取完整时刻；两者都按旧树口径
    # `tz_localize('Asia/Shanghai')` 后再取整数 epoch。
    day_dt = pd.to_datetime(ts0.date).tz_localize('Asia/Shanghai')
    out['date'] = day_dt.strftime('%Y-%m-%d').to_numpy()
    # ⚠️ 用 `kline83.epoch_seconds`，**别写 `astype(np.int64) // 10**9`** ——
    # pandas 3.0 起默认精度是 µs，那样写会静默少 1000 倍（见该函数的 docstring）。
    out['date_stamp'] = epoch_seconds(day_dt).to_numpy()
    if family != 'day':
        full_dt = pd.Series(ts0, index=out.index).dt.tz_localize('Asia/Shanghai')
        out['datetime'] = full_dt.dt.strftime('%Y-%m-%d %H:%M:%S')
        out['time_stamp'] = epoch_seconds(full_dt)
    elif 'datetime' in out.columns:
        # ⚠️ 日族**不新增** `datetime`，但调用方自带了这个列要**统一成字符串**：
        # 15min / 60min 的检出会**同时往日族写一笔**（见 `predict.py`），它们带的
        # `datetime` 是 `Timestamp`；不转换就会以 **BSON date** 落库，
        # 而 4.4 那边日族写口是 `dt.strftime(...)` 成**字符串**的
        # （`GQ_save_daily_metadata_reality`），分钟族这边也是字符串
        # ⇒ 同一列两种类型。2026-10-17 端到端实跑时暴露（迁移来的 400 行是 str，
        # 新写入的 2 行是 datetime），此前只看代码看不出来。
        out['datetime'] = pd.to_datetime(out['datetime']).dt.strftime('%Y-%m-%d %H:%M:%S')
    out['created_at'] = int(created_at)

    # 载荷列照原样。默认 `NaN` → 不写该列（与 `metadata_save` 同口径）；
    # `write_nulls=True` 时写 `None` —— review 重算修正后需要**清掉旧值**。
    rows = []
    for _, r in out.iterrows():
        doc = {}
        for col in out.columns:
            v = r[col]
            if pd.isna(v) if not isinstance(v, (list, dict)) else False:
                if write_nulls:
                    doc[col] = None
                continue
            doc[col] = v.item() if hasattr(v, 'item') else v
        rows.append(doc)
    return rows


def save_reality_feat(features: pd.DataFrame, family: str, market_type,
                      verbose: bool = True, echo=None, created_at=None,
                      write_nulls: bool = False, db=None) -> dict:
    """取数（其实只组行）→ `upsert_fields` 落 ``golemq_stock_cn``。

    :param market_type: **该 code 的**市场类型（一只 code 一次调用，同旧 `charts.py`）；
        决定落 ``stock_`` 还是 ``index_`` 前缀，见 :func:`reality_feat_name`。
    :param db: 行情库的**数据库实例**（调用方从市场实例取，
        如 ``get_active_market().get_database()``）。``None`` ⇒ 本模块自己取
        （仅供**市场层内部**调用；pipeline/CLI 必须显式传，见 CLAUDE.md 的「库名不外泄」）。
    :returns: ``{'name','rows','written','upserted','modified'}``
    """
    from GolemQ.datasource.writer import upsert_fields

    name = reality_feat_name(family, market_type)
    rows = reality_feat_rows(features, family, created_at=created_at,
                             write_nulls=write_nulls)
    stats = {'name': name, 'rows': len(rows), 'written': 0, 'upserted': 0, 'modified': 0}
    if not rows:
        if verbose:
            (echo or print)(f'[reality] {name}: 0 行，跳过')
        return stats
    target_db = _db() if db is None else db
    out = upsert_fields(target_db[name], rows, list(UNIQUE_KEYS[family]))
    stats.update(written=len(rows), upserted=out['upserted'], modified=out['modified'])
    if verbose:
        (echo or print)('[reality] {}: 写 {} 行 / upserted {} / modified {}'.format(
            name, stats['written'], stats['upserted'], stats['modified']))
    return stats


#: **存的是「÷close」比值、读回要乘回 close** 的列（D49/D54 的双向约定）。
#: `*Nor*` 是 0..1 的归一化强度，不是价格量纲，**不还原**。
PRICE_RATIO_COLUMNS = (
    'poly9MajD', 'poly9WekD',
    'rTrePrcMajD', 'rTreSlopMajD',
    'prophetLbD', 'prophetUbD',
    'prophetLbMajD', 'prophetUbMajD',
    'poly9D', 'regtreePriceD',
    'cost5PrzD', 'cost15PrzD', 'cost85PrzD', 'cost95PrzD',
    'renkoSpriceD', 'renkoLpriceD',
    'renkosBarLbD', 'renkosBarUbD', 'renkolBarLbD', 'renkolBarUbD',
)


def restore_price_columns(df: pd.DataFrame, close, columns=PRICE_RATIO_COLUMNS):
    """把存成**比值**的价格列乘回 close，还原价格量纲。**纯函数**。

    :param df: 读回的帧（索引 `(ts, code)` 或 `(date, code)`）
    :param close: 与 `df` 可对齐的 close 序列/帧（索引同形；或含 `close` 列）
    :returns: `df` 的副本；只改 `columns` 中实际存在的列。
    """
    if df is None or len(df) == 0:
        return df
    out = df.copy()
    if isinstance(close, pd.DataFrame):
        close = close['close']
    close = pd.Series(close).reindex(out.index)
    for col in columns:
        if col not in out.columns:
            continue
        # `_DUMMY` / `D` 是归一化落库名；乘回 close 后写成**无后缀 base 价格列**
        base = col[:-6] if col.endswith('_DUMMY') else col[:-1]
        out[base] = pd.to_numeric(out[col], errors='coerce') * close
    return out


def fetch_reality_feat(codes, start, end, family='day', market_type=None,
                       restore_price=None, close=None, db=None, columns=None):
    """读回 ``*_reality_feat_*``；**空结果返回 `None`**（不是空帧）。

    ⚠️ 这里刻意**偏离** `refdata.py` 那边「空返回空帧」的惯例：唯一的消费方
    （搬过来的 `calc_ChartScaned_persistence`）本来就是按 `is not None` 写的，
    而空帧还要照顾两层索引，容易踩 `Too many levels`。

    :param codes: 6 位代码或列表。
    :param start, end: ``'YYYY-MM-DD'``（**含**当日）。
    :param restore_price: ``None``（默认）⇒ 帧里有 `PRICE_RATIO_COLUMNS` 才自动
        乘回 close；``True`` 强制还原；``False`` 返回落库的原始比值。
    """
    key = UNIQUE_KEYS[family][1]          # date_stamp 或 time_stamp
    from .metadata_save import day_stamp

    if isinstance(codes, str):
        codes = [codes]
    codes = [str(c)[:6] for c in codes]
    # `date_stamp` 与 `day_stamp` **同值**（都是「北京零点的真实 UTC 时刻」）——
    # 旧树两边各写一遍公式，这里收敛到 `metadata_save.day_stamp`（带 doctest）。
    lo, hi = day_stamp(str(start)[:10]), day_stamp(str(end)[:10]) + 86400 * 1
    q = {'code': {'$in': codes}, key: {'$gte': lo, '$lte': hi}}
    target_db = _db() if db is None else db
    proj = {'_id': 0}
    if columns:
        # 键 + 索引层（`date`/`datetime`）必须一起投影，否则下面的 `df[lvl]` 会 KeyError。
        proj.update({key: 1, 'code': 1, 'date': 1, 'datetime': 1,
                     **{c: 1 for c in columns}})
    cur = target_db[reality_feat_name(family, market_type)].find(q, proj).sort(key, 1)
    df = pd.DataFrame(list(cur))
    if df.empty:
        return None
    # ⚠️ **必须把索引设成 `(date|datetime, code)`** —— 旧树的读口就是这么做的
    # （`features.py:338-390`：`assign(date=pd.to_datetime(res.date))` →
    # `drop_duplicates(['date','code'])` → `set_index(['date','code'], drop=False)`）。
    # 下游 `calc_ChartScaned_persistence` 要 `index.get_level_values('date')` 与
    # `reindex([(last_day, ticker)])`，**扁平帧会 `KeyError: Requested level (date)…`**。
    lvl = 'date' if family == 'day' else 'datetime'
    df[lvl] = pd.to_datetime(df[lvl])
    out = df.drop_duplicates([lvl, 'code']).set_index([lvl, 'code'], drop=False)
    if restore_price is None:
        # 自动：帧里**有价格比值列**才还原（CV 读口没有这些列 ⇒ 不多取一次 K 线）。
        restore_price = any(c in out.columns for c in PRICE_RATIO_COLUMNS)
    if restore_price:
        if close is None:
            close = _close_for_restore(codes, start, end, family)
        out = restore_price_columns(out, close)
    return out


def _close_for_restore(codes, start, end, family):
    """取与 `fetch_reality_feat` 同形的 close（供比值列还原）。"""
    from GolemQ import get_active_market
    market = get_active_market()
    if family == 'day':
        res, _ = market.get_kline_price_v3(codes, start=start, end=end,
                                           realtime=False, verbose=False)
    else:
        res, _ = market.get_kline_price_min(codes, start=start, end=end,
                                            frequency=family, realtime=False,
                                            verbose=False)
    data = None if res is None else getattr(res, 'data', None)
    if data is None or len(data) == 0:
        return pd.Series(dtype=float)
    out = data['close'].copy()
    ts = pd.DatetimeIndex(pd.to_datetime(out.index.get_level_values(0)))
    if family == 'day':
        ts = ts.normalize()
        lvl = 'date'
    else:
        lvl = 'datetime'
    out.index = pd.MultiIndex.from_arrays(
        [ts, out.index.get_level_values(1)], names=[lvl, 'code'])
    return out[~out.index.duplicated(keep='last')]


#: ⚠️ **刻意不保留旧树那三个名字**（`GQ_save_metadata_reality` /
#: `GQ_save_daily_metadata_reality` / `GQ_fetch_daily_metadata_reality`）。
#: 它们的签名是 `(features, collections=<一个 Collection 句柄>)`，而本模块收的是
#: `(features, family, market_type)` —— **同名不同义**。留着当别名只会诱出错误调用
#: （`collections=` 一传就 TypeError），而那样的错在移植时最容易被顺手"修好"成
#: 传错的东西。搬过来的 `analysis/cv/predict.py` 一律用本模块的**新名**。
__all__ = [
    'REALITY_FAMILIES', 'UNIQUE_KEYS',
    'reality_feat_name', 'reality_feat_rows', 'save_reality_feat', 'fetch_reality_feat',
    'PRICE_RATIO_COLUMNS', 'restore_price_columns',
]


#: 每张 `*_reality_feat_*` 表里**真正的 CV 载荷列**（= 新树 writer 写的那套，见
#: `analysis/cv/predict.py` 的写口）。⚠️ **4.4 的同名表里混着三套命名**
#: （`ClsMaj` / `ClsMin` / `Class`），但实测**同一行上多套并存**
#: （60min 表：带 CV 的 1,340 行里 1,242 行两套都有，日期范围完全相同）
#: ⇒ 按本表挑一套**不丢数据**。
CV_COLUMNS = {
    'day':   ('ClsMaj', 'ConfidenceMaj', 'CoordinatesMaj'),
    '60min': ('Class', 'Confidence', 'Coordinates'),
    '15min': ('ClsMin', 'ConfidenceMin', 'CoordinatesMin'),
}

#: 除键之外的元字段 —— 原样搬。
_CV_META = ('code', 'date', 'datetime', 'date_stamp', 'time_stamp',
            'created_at', 'updated_at', 'ChartScaned')

#: **每个族实际要搬的列 = 该族 + 所有**上卷**到它的族**（2026-10-18 实测修正）
#: =====================================================================
#: 旧树是**逐级上卷**写的（`predict.py`）：
#:
#: ==========  ==========  ==========  ==========
#: 扫描频率     `_day`      `_60min`    `_15min`
#: ==========  ==========  ==========  ==========
#: day          ✔           —           —
#: 60min        ✔           ✔           —
#: 15min        ✔           ✔（收盘）    ✔
#: ==========  ==========  ==========  ==========
#:
#: 所以同一份文档里会**并存不同族的列**。实测 4.4：
#:
#: * `stock_reality_feat_day`：`ClsMaj` 231 万 / **`ClsMin` 511 万** / **`Class` 487 万**
#: * `stock_reality_feat_60min`：抽样 3000 里 **916 行带的是 `ClsMin`**（15min 的列）、
#:   49 行带 `Class`；时刻**全部落在** `CLOSE_TIMES`（10:30/11:30/14:00/15:00）
#:
#: ⇒ 只按自己那套列搬，会把上卷过来的**整批丢掉**。后果不是"少几列"：
#: `calc_ChartScaned_persistence` 读日族的 `CoordinatesMin` / `Coordinates`
#: 判「要不要重扫」—— 一空 ⇒ `all_nat_min` 恒真 ⇒ **几乎每只都判 mode=1，
#: 全扫 1008 天**（用户 2026-10-18 跑批量扫描时发现"范围太大"的直接原因）。
_CV_COLUMNS_BY_FAMILY = {
    'day': tuple(dict.fromkeys(CV_COLUMNS['day'] + CV_COLUMNS['60min'] + CV_COLUMNS['15min'])),
    '60min': tuple(dict.fromkeys(CV_COLUMNS['60min'] + CV_COLUMNS['15min'])),
    '15min': CV_COLUMNS['15min'],
}


def migrate_reality_feat_from44(families=('day', '60min', '15min'),
                                sources=('stock', 'index'), limit=None,
                                dry_run=False, verbose=True, echo=None) -> dict:
    """把 4.4 的 CV 检出结果搬进 8.3（**一次性**，`DECISIONS.md` D51）。

    为什么必须搬而不是重算
    ====================
    那些列是**买卖点 CV 识别事件**（GolemAI 环境下 CUDA + YOLOv8 跑出来的），
    重算要显卡、且是对全历史逐日扫 —— 用户明确「很难重头重新计算」。

    它搬什么 / 不搬什么
    ==================
    * **只搬 CV 列**（`CV_COLUMNS`）+ 元字段；`ChartScaned` 标记也搬。
      ⚠️ **「只有 `ChartScaned`、没有任何 CV 列」的行也要搬** —— 旧树「没探到」的
      标记路径就只写这一列。漏了它 ⇒ `calc_ChartScaned_persistence` 判不出
      「已扫过」⇒ 整段 1008 天全量重扫（2026-10-18 实测踩到，见下方查询处的说明）。
    * **不搬** 4.4 那张表里的**指标列**（`stock_reality_feat_day` 有 136 个：
      `MA30` / `poly9MajD` / `ATR_LB` …）—— 与 D49「**指标全部重算**」一致，
      且避免把 10.4M × 136 的指标矩阵灌进来。

    逐码路由
    =======
    4.4 的 `index_reality_feat_*` 里 **94–98% 是 ETF**（用新树的 `is_stock_cn` 实测）。
    所以**不按源表名照搬**，而是逐码分类后落对应前缀（`reality_feat_name`）：
    `etf_cn` → `etf_reality_feat_*`、`index_cn`/`fund_cn` → `index_*`、
    `stock_cn` → `stock_*`。

    ⚠️ **必须显式带 `market_type`**：裸码有歧义（`000001` = 上证指数 or 平安银行），
    一律经 `is_stock_cn` 判 —— 与 K 线侧 `market_prefix` 同一条纪律。

    幂等
    ====
    只 `$set`、键是各族自己的 `UNIQUE_KEYS`（日族 `(code, date_stamp)`、
    分钟族 `(code, time_stamp)`）⇒ **重跑不会插重复行**，中途挂了直接再跑。

    :param families: 要搬的族
    :param sources: 要读的 4.4 源表前缀（`'stock'` / `'index'`）
    :param limit: 每个源表最多读多少行（试跑用）
    :param dry_run: 只读只统计，**不写 8.3**
    :returns: ``{目标集合: {'read','written','upserted','modified','skipped'}}``
    """
    from GolemQ.core.migrate44 import db44
    import pandas as pd

    from GolemQ.core.constants import MARKET_TYPE
    from .symbol import is_stock_cn
    # ⚠️ 写库**复用** `save_reality_feat`（D55 的唯一入口），不为本次搬运另写 upsert 帮手
    # —— 平行实现不会报错、只会分叉（`helpers.save_stock_codes` 那一课）。
    # 本函数只负责「取数 → 组行 → 路由 → 分批」。

    say = echo or print
    g44 = db44('golemq')
    out: dict = {}

    for family in families:
        # ⚠️ 日族用**三族之并**（见 `_CV_COLUMNS_BY_FAMILY` 的说明）——
        # 迁移与核对**必须同一个判据**，否则「核对通过」证明不了标记行也在。
        cv_cols = _CV_COLUMNS_BY_FAMILY[family]
        keys = list(UNIQUE_KEYS[family])
        for src_prefix in sources:
            src_name = f'{src_prefix}_reality_feat_{family}'
            src = g44[src_name]
            # 只取**带 CV 的行** —— 源表里绝大多数是不带 CV 的指标行
            # ⚠️ **`ChartScaned` 单独成一项**（2026-10-18 补）：旧树「没探到」的那条路径
            # (`GQ_markup_*ChartScaned`) 只写 `ChartScaned`、**不写任何 CV 列**。
            # 只按 CV 列过滤会把这类行整批漏掉 —— 实测 4.4 的 600519 有 966 行
            # `ChartScaned=2`，首轮只搬过去 436 行。漏掉的后果不是"少几条数据"：
            # `calc_ChartScaned_persistence` 靠这个旗标判「要不要重扫」，旗标一缺
            # ⇒ `all_nat` 恒真 ⇒ **整段 1008 天全量重扫**（扫描范围暴涨）。
            query = {'$or': [{c: {'$exists': True}} for c in cv_cols]
                             + [{'ChartScaned': {'$exists': True}}]}
            proj = {'_id': 0, **{c: 1 for c in cv_cols}, **{c: 1 for c in _CV_META}}
            cursor = src.find(query, proj, no_cursor_timeout=True)
            if limit:
                cursor = cursor.limit(int(limit))

            # 按**目标集合**分批攒 —— 一批里混多个目标时不能一次 upsert
            buckets: dict = {}
            try:
                for doc in cursor:
                    if not all(k in doc for k in keys):
                        continue
                    try:
                        mt = is_stock_cn(doc['code'])[1]
                    except Exception:            # noqa: BLE001 认不出的码跳过并计数
                        mt = MARKET_TYPE.STOCK_CN
                    dst_name = reality_feat_name(family, mt)
                    st = out.setdefault(dst_name,
                                        {'read': 0, 'written': 0, 'upserted': 0,
                                         'modified': 0, 'skipped': 0})
                    st['read'] += 1
                    buckets.setdefault(mt, []).append(doc)

                    if sum(len(v) for v in buckets.values()) >= 4000:
                        lvl = 'date' if family == 'day' else 'datetime'
                        if not dry_run:
                            for mt, rows in buckets.items():
                                name = reality_feat_name(family, mt)
                                r = save_reality_feat(
                                    pd.DataFrame(rows).set_index([lvl, 'code']),
                                    family, mt, verbose=False, db=_db())
                                out[name]['upserted'] += r['upserted']
                                out[name]['modified'] += r['modified']
                        for mt, rows in buckets.items():
                            out[reality_feat_name(family, mt)]['written'] += len(rows)
                        buckets = {}
                if buckets and not dry_run:
                    lvl = 'date' if family == 'day' else 'datetime'
                    for mt, rows in buckets.items():
                        name = reality_feat_name(family, mt)
                        r = save_reality_feat(
                            pd.DataFrame(rows).set_index([lvl, 'code']),
                            family, mt, verbose=False, db=_db())
                        out[name]['upserted'] += r['upserted']
                        out[name]['modified'] += r['modified']
                for mt, rows in buckets.items():
                    out[reality_feat_name(family, mt)]['written'] += len(rows)
            finally:
                cursor.close()
            if verbose:
                say(f'[reality44] {src_name} 读完'
                    f'（{sum(v["read"] for v in out.values()):,} 行累计）')

    if verbose:
        say('[reality44] 汇总{}：'.format('（**干跑，没写库**）' if dry_run else ''))
        for name in sorted(out):
            s = out[name]
            say('    {:30s} 读 {:>10,} / 写 {:>10,} / 新增 {:>9,} / 更新 {:>9,}'.format(
                name, s['read'], s['written'], s['upserted'], s['modified']))
    return out


def verify_reality_feat_migration(families=('day', '60min', '15min'),
                                  sources=('stock', 'index'), sample=20000,
                                  verbose=True, echo=None) -> dict:
    """**只读**核对：4.4 的 CV 行是否都在 8.3 的对应表里、且**逐值相同**。

    抽样从**每个源表尾部**取（`$natural:-1`）—— 尾部是最可能被漏搬的那一批。
    逐字段比，浮点**按位**比（搬运不该动任何一个 bit）。

    :returns: ``{'checked','missing','mismatch','examples'}``
    """
    from GolemQ.core.migrate44 import db44
    from .symbol import is_stock_cn

    say = echo or print
    g44 = db44('golemq')
    out = {'checked': 0, 'missing': 0, 'mismatch': 0, 'examples': []}

    for family in families:
        # ⚠️ 日族用**三族之并**（见 `_CV_COLUMNS_BY_FAMILY` 的说明）——
        # 迁移与核对**必须同一个判据**，否则「核对通过」证明不了标记行也在。
        cv_cols = _CV_COLUMNS_BY_FAMILY[family]
        keys = list(UNIQUE_KEYS[family])
        for src_prefix in sources:
            src = g44[f'{src_prefix}_reality_feat_{family}']
            # 与迁移**同一个判据**（含只有 `ChartScaned` 的标记行）—— 否则
            # 「核对通过」只证明了带 CV 的那些，标记行会悄悄漏在核对之外。
            query = {'$or': [{c: {'$exists': True}} for c in cv_cols]
                             + [{'ChartScaned': {'$exists': True}}]}
            for doc in src.find(query, {'_id': 0}).sort('$natural', -1).limit(int(sample)):
                if not all(k in doc for k in keys):
                    continue
                out['checked'] += 1
                dst_name = reality_feat_name(family, is_stock_cn(doc['code'])[1])
                got = _db()[dst_name].find_one({k: doc[k] for k in keys}, {'_id': 0})
                if got is None:
                    out['missing'] += 1
                    if len(out['examples']) < 5:
                        out['examples'].append(('missing', dst_name, keys))
                    continue
                bad = None
                for c in cv_cols:
                    if c not in doc:
                        continue
                    if c not in got:
                        bad = (dst_name, keys, c, 'absent')
                    else:
                        a, b = doc[c], got[c]
                        same = (a == b) or (a != a and b != b)
                        if not same:
                            bad = (dst_name, keys, c, a, b)
                    if bad:
                        out['mismatch'] += 1
                        if len(out['examples']) < 5:
                            out['examples'].append(bad)
                        break

    if verbose:
        say('[verify44] 抽 {:,} 行：缺 {:,} / 不符 {:,}'.format(
            out['checked'], out['missing'], out['mismatch']))
        for ex in out['examples']:
            say('    {}'.format(ex))
    return out


# --------------------------------------------------------------------------- #
# review 管线水位（侧表）
# --------------------------------------------------------------------------- #
#: 为什么用侧表：reality 表是 `$set`-only 且**跳过 NaN 列** ⇒「算过但全 NaN」与
#: 「从没算过」在表里不可区分；侧表还能记错误状态而不往数据契约里加死列。
WATERMARK_COLL = 'pipeline_watermarks'
WATERMARK_KEYS = ('pipeline', 'table', 'code')


def get_watermark(pipeline: str, table: str, code: str, db=None):
    """读某条水位；不存在返回 ``None``。"""
    target_db = _db() if db is None else db
    return target_db[WATERMARK_COLL].find_one(
        {'pipeline': pipeline, 'table': table, 'code': str(code)[:6]})


def set_watermark(pipeline: str, table: str, family: str, code: str,
                  last_written_ts=None, last_source_ts=None, rows: int = 0,
                  status: str = 'ok', error=None, db=None) -> dict:
    """写/更新水位（`upsert_fields`，键 `(pipeline, table, code)`）。"""
    import time as _time
    from GolemQ.datasource.writer import upsert_fields
    doc = {
        'pipeline': pipeline,
        'table': table,
        'family': family,
        'code': str(code)[:6],
        'last_written_ts': None if last_written_ts is None else int(last_written_ts),
        'last_source_ts': None if last_source_ts is None else int(last_source_ts),
        'rows': int(rows),
        'status': status,
        'error': error,
        'updated_at': int(_time.time()),
    }
    target_db = _db() if db is None else db
    return upsert_fields(target_db[WATERMARK_COLL], [doc], list(WATERMARK_KEYS))


def clear_watermark(pipeline: str, table: str = None, code: str = None,
                    db=None) -> int:
    """删除水位（`--force`/测试用）。返回删除数。"""
    q = {'pipeline': pipeline}
    if table is not None:
        q['table'] = table
    if code is not None:
        q['code'] = str(code)[:6]
    target_db = _db() if db is None else db
    return target_db[WATERMARK_COLL].delete_many(q).deleted_count
