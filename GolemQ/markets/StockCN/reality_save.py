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
from datetime import time as dtime
from typing import Optional

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


def _row_differs(cur: Optional[dict], row: dict) -> bool:
    """库里的 ``cur`` 与要写的 ``row`` 在**载荷列**上有没有差别。

    ``created_at`` 是每次运行都变的写入时间戳，**不参与比较** —— 否则每行都算
    「变了」，差量写会退化成全量写。
    """
    if cur is None:
        return True
    for k, v in row.items():
        if k == 'created_at':
            continue
        old = cur.get(k, _MISSING)
        if old is _MISSING:
            return True
        if isinstance(v, float) and isinstance(old, float):
            if v != v and old != old:       # 两边都是 NaN
                continue
            if v != old:
                return True
            continue
        if old != v:
            return True
    return False


#: 库文档里「没有这个字段」的哨兵（与「值为 ``None``」区分开）。
_MISSING = object()


def save_position_series(df: pd.DataFrame, market_type=None, db=None,
                         verbose: bool = False, echo=None,
                         delta: bool = True) -> dict:
    """「全局智能仓位」→ ``stock_/index_massive_reality_60min_models``（**旧名**）。

    这张表是**全系统共用的 massive 模型表**（每根 60min bar 一行、唯一键
    ``time_stamp``），仓位只是其中一批字段（`PosBsln` / `PosPad` /
    `PosNegBsln` / `MasPosOnhold` / `DS_*` / `mberCnt`）。**不要另起
    `*_position_*` 这类按用途命名的表** —— 那会把全系统表切碎。

    ⚠️ 与旧树的区别只在数据来源（8.3 现算 vs 4.4 历史），**表名与列名照旧**
    （D79）。

    :param delta: **差量写**（用户 2026-11-25）。``df`` 是最近 800 天窗口的
        **全量**（≈2,100 根 bar），但一次运行真正变的只有最后几根 —— 默认先读回
        这窗口里已有的文档，只 upsert「库里没有 / 载荷列有变化」的行。
        ``False`` 时退回旧的「整窗重写」。
        ⚠️ 差量**不比 `created_at`**，也不影响回填：库里没有的行照样插。
    :returns: ``{'name', 'rows'(提交候选), 'written'(实写), 'unchanged',
        'upserted', 'modified'}``

    用户 2026-11-25 的疑问「为什么写 2113 行而不是 4 行的差量」：因为
    `run_market` 返回的是**整窗**，旧实现把这 2,113 行全部 `$set` 一遍
    （值没变的由 Mongo 记成 `modified=0`，但网络与写锁照样走满）。现在默认只写
    真正变化的那几行，并把两个数都报出来。
    """
    from GolemQ.core.constants import MARKET_TYPE
    from GolemQ.datasource.writer import upsert_fields

    mt = MARKET_TYPE.STOCK_CN if market_type is None else market_type
    name = ('stock_massive_reality_60min_models' if mt == MARKET_TYPE.STOCK_CN
            else 'index_massive_reality_60min_models')
    stats = {'name': name, 'rows': 0, 'written': 0, 'unchanged': 0,
             'upserted': 0, 'modified': 0}
    if df is None or len(df) == 0:
        return stats
    out = df.copy()
    dt = pd.to_datetime(out['datetime'])
    out['datetime'] = dt.dt.strftime('%Y-%m-%d %H:%M:%S')
    out['time_stamp'] = [int(t.timestamp()) for t in dt]
    out['date_stamp'] = [int(t.normalize().timestamp()) for t in dt]
    out['created_at'] = int(pd.Timestamp.now().timestamp())
    rows = []
    for _, r in out.iterrows():
        doc = {}
        for col in out.columns:
            v = r[col]
            if isinstance(v, float) and pd.isna(v):
                continue
            doc[col] = v.item() if hasattr(v, 'item') else v
        rows.append(doc)
    target_db = _db() if db is None else db
    stats['rows'] = len(rows)
    if delta:
        coll = target_db[name]
        stamps = [r['time_stamp'] for r in rows]
        proj = {'time_stamp': 1, '_id': 0}
        for r in rows[:1]:
            proj.update({k: 1 for k in r if k not in ('created_at',)})
        cur = {d['time_stamp']: d
               for d in coll.find({'time_stamp': {'$in': stamps}}, proj)}
        kept = [r for r in rows if _row_differs(cur.get(r['time_stamp']), r)]
        stats['unchanged'] = len(rows) - len(kept)
        rows = kept
    stats['written'] = len(rows)
    if not rows:
        if verbose:
            (echo or print)(f'[position] {name}: {stats["rows"]} 行全部未变，跳过写库')
        return stats
    res = upsert_fields(target_db[name], rows, ['time_stamp'])
    stats.update(upserted=res['upserted'], modified=res['modified'])
    if verbose:
        (echo or print)(f'[position] {name}: 提交 {stats["rows"]} 行 / 实写 '
                        f'{stats["written"]} 行（unchanged {stats["unchanged"]}）/ '
                        f'upserted {res["upserted"]} / modified {res["modified"]}')
    return stats


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
#: review 契约里**所有**价格量纲比值列（`D` / `_DUMMY`）。
#: ⚠️ 加契约列时必须同步这里 —— 否则该列永远不还原（2026-11-16 复审发现
#: `rTrePrcD` / `rTreSlopD` 漏了，而 `regtreePriceD` 是个库里根本不存在的旧名）。
#: `test_cases/markets/test_reality_save.py` 有覆盖度守卫。
PRICE_RATIO_COLUMNS = (
    'poly9MajD', 'poly9WekD',
    'rTrePrcMajD', 'rTreSlopMajD',
    'prophetLbD', 'prophetUbD',
    'prophetLbMajD', 'prophetUbMajD',
    'poly9D', 'rTrePrcD', 'rTreSlopD',      # 60min 原生（D78）
    'cost5PrzD', 'cost15PrzD', 'cost85PrzD', 'cost95PrzD',
    'renkoSpriceD', 'renkoLpriceD',
    'renkosBarLbD', 'renkosBarUbD', 'renkolBarLbD', 'renkolBarUbD',
)


def _daily_maps(daily):
    """`daily` close → ``{code: (交易日 DatetimeIndex, close array)}``（私有）。"""
    if daily is None or len(daily) == 0:
        return {}
    d = pd.Series(daily)
    if isinstance(d.index, pd.MultiIndex):
        dts = pd.DatetimeIndex(pd.to_datetime(d.index.get_level_values(0)))
        keys = d.index.get_level_values(1)
    else:
        dts = pd.DatetimeIndex(pd.to_datetime(d.index))
        keys = np.zeros(len(d), dtype=int)
    out = {}
    for key in pd.unique(keys):
        sel = np.flatnonzero(np.asarray(keys) == key)
        order = np.argsort(dts[sel])
        out[key] = (dts[sel][order].normalize(),
                    d.to_numpy(dtype=float)[sel][order])
    return out


def asof_close(close, daily=None):
    """把逐 bar 的 close 映射成**该行的 as-of 收盘**（F8，2026-11-16）。

    比值列（`*D` / `*_DUMMY`）的分母必须是**分子所在快照**的 close，否则
    同一份「上一交易日的特征」被当天的 4 根 60min bar 各除一次，比值会随
    盘中价格漂移（D82 / `PITFALLS.md` P51）。

    ==============  ============================================
    行的时间戳        as-of close
    ==============  ============================================
    00:00（日频行）   自身（日线 bar 的 close 就是当日收盘）
    15:00             自身（当日已收盘）
    盘中（其他时刻）   最近一个 **≤ ts 的 15:00 bar** 的 close
                      （= 上一交易日收盘）
    ==============  ============================================

    ⚠️ **帧首日**的盘中行在 `close` 里找不到更早的 15:00（`fetch_reality_feat`
    只取请求区间、attach 的帧从首日开始）。此时：
    传了 `daily`（日线 close，索引 `(date, code)` 或 `DatetimeIndex`）⇒ 用
    **上一交易日**的日线收盘；没传 ⇒ 回落到自身（= 旧的 bar close 行为）。
    生产路径（attach / runtime / fetch）都会传 `daily`。

    100% 纯函数：不取数、不看库。`index` 可以是 `(ts, code)` 多层（逐码处理），
    也可以是单码的 `DatetimeIndex`。

    >>> idx = pd.DatetimeIndex(['2024-01-02 15:00', '2024-01-03 10:30',
    ...                         '2024-01-03 15:00'])
    >>> asof_close(pd.Series([10.0, 11.0, 12.0], index=idx)).tolist()
    [10.0, 10.0, 12.0]
    >>> # 帧首日盘中：close 里没有前一日的 15:00 ⇒ 用 daily 兜底
    >>> asof_close(pd.Series([11.0], index=pd.DatetimeIndex(['2024-01-03 10:30'])),
    ...            daily=pd.Series([10.0], index=pd.DatetimeIndex(['2024-01-02']))
    ...            ).tolist()
    [10.0]
    """
    idx = close.index
    vals = pd.to_numeric(pd.Series(close), errors='coerce').to_numpy(dtype=float)
    out = vals.copy()
    if isinstance(idx, pd.MultiIndex):
        ts = pd.DatetimeIndex(pd.to_datetime(idx.get_level_values(0)))
        keys = idx.get_level_values(1)
    else:
        ts = pd.DatetimeIndex(pd.to_datetime(idx))
        keys = np.zeros(len(vals), dtype=int)
    dmap = _daily_maps(daily)
    times = np.asarray(ts.time)
    is_15 = times == dtime(15, 0)
    is_00 = times == dtime(0, 0)
    for _key in pd.unique(keys):
        sel = np.flatnonzero(np.asarray(keys) == _key)
        t = ts[sel]
        resolved = np.zeros(len(sel), dtype=bool)     # 已经定下来（15:00/日频）
        resolved |= is_15[sel] | is_00[sel]
        # 1) 帧内的 15:00 轴
        r15 = np.flatnonzero(is_15[sel])
        if len(r15):
            order = np.argsort(t[r15])
            t15 = t[r15][order]
            v15 = vals[sel][r15][order]
            cand = np.flatnonzero(~resolved)
            pos = t15.searchsorted(t[cand], side='right') - 1
            ok = pos >= 0
            out[sel[cand[ok]]] = v15[pos[ok]]
            resolved[cand[ok]] = True
        # 2) 还没有定下来的（帧首日盘中 / 整帧没有 15:00）→ 日线兜底：
        #    盘中行取**严格早于当日**的最后一个交易日收盘。
        if dmap and _key in dmap:
            dts, dvals = dmap[_key]
            for i in np.flatnonzero(~resolved):
                p = int(dts.searchsorted(t[i].normalize(), side='left')) - 1
                if p >= 0:
                    out[sel[i]] = dvals[p]
                    resolved[i] = True
    return pd.Series(out, index=idx, name=getattr(close, 'name', None))


def restore_price_columns(df: pd.DataFrame, close, columns=PRICE_RATIO_COLUMNS):
    """把存成**比值**的价格列乘回 close，还原价格量纲。**纯函数**。

    :param df: 读回的帧（索引 `(ts, code)` 或 `(date, code)`）
    :param close: 与 `df` 可对齐的 close 序列/帧（索引同形；或含 `close` 列）。
        ⚠️ 必须是**as-of 收盘**（`:func:`asof_close` 的结果），不是逐 bar close ——
        比值是按分子所在快照日归一的（D82）。
    :returns: `df` 的副本；只改 `columns` 中实际存在的列。

    ⚠️ `close` 为空（比如该码没有 K 线，取数返回空）时**原样返回**，不把价格列
    写成 NaN —— 宁可保留比值，也不静默丢掉整列。
    """
    if df is None or len(df) == 0:
        return df
    out = df.copy()
    if isinstance(close, pd.DataFrame):
        close = close['close']
    close = pd.Series(close)
    if len(close) == 0:
        return out
    close = close.reindex(out.index)
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
            # 比值按 **as-of 收盘**归一化（F8）⇒ 还原也必须乘同一个锚点。
            close = asof_close(_close_for_restore(codes, start, end, family))
        out = restore_price_columns(out, close)
    return out


def _close_for_restore(codes, start, end, family):
    """取与 `fetch_reality_feat` 同形的 close（供比值列还原）。

    ⚠️ **往前多取 30 天**：as-of 还原需要「帧首日之前的 15:00 bar」（D82）。
    只取请求区间会让首日盘中行找不到上一交易日收盘 ⇒ 静默退化成 bar close。
    多出来的行由 `restore_price_columns` 的 reindex 丢掉。
    """
    from GolemQ import get_active_market
    market = get_active_market()
    start = str((pd.Timestamp(start) - pd.Timedelta(days=30)).date())
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
    'PRICE_RATIO_COLUMNS', 'restore_price_columns', 'asof_close',
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
