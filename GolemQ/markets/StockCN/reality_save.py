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
2. ``time_stamp`` **不加** `ts` 字段。它是本表的**唯一键**，口径是「北京墙上时间
   当 UTC 再取 epoch」（旧树 `features.py:122` 的原公式），**必须逐字复现**；
   与 8.3 别处 `ts` 的「UTC-aware」口径**不是一回事**。要 8.3 那条轴的话，
   另**加**一列，别改名。
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from .kline83 import bj_date

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

    ⚠️ **除股票外一律落 ``index_`` 前缀** —— 与旧树 `charts.py` 的
    ``if market_type == QA.MARKET_TYPE.STOCK_CN`` 选择**逐字一致**。

    >>> reality_feat_name('day', 'stock_cn')
    'stock_reality_feat_day'
    >>> reality_feat_name('60min', 'index_cn')
    'index_reality_feat_60min'
    >>> reality_feat_name('15min', 'etf_cn')     # ETF 与指数同前缀（旧树如此）
    'index_reality_feat_15min'
    """
    if family not in REALITY_FAMILIES:
        raise ValueError(f'未知的 reality 族 {family!r}；可用: {REALITY_FAMILIES}')
    from GolemQ.core.constants import MARKET_TYPE
    prefix = 'stock' if market_type == MARKET_TYPE.STOCK_CN else 'index'
    return f'{prefix}_reality_feat_{family}'


def _db():
    """8.3 的库句柄。**函数据导入是刻意的** —— `markets/StockCN/__init__.py` 里的
    `GOLEMQ_STOCK_CN` 定义在文件中部，模块级导入会撞上半初始化的 `ImportError`。
    同层邻居（`metadata_save._db` / `kline_save._db` / `refdata._stock_cn_db`）都这么写。
    """
    from . import GOLEMQ_STOCK_CN
    return GOLEMQ_STOCK_CN


def reality_feat_rows(features: pd.DataFrame, family: str, created_at=None) -> list:
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

    `time_stamp` 是**北京墙上时间当 UTC** 的 epoch（15:00 → 15:00Z），不是 8.3 的 `ts` 口径；

    >>> d['time_stamp'], d['date_stamp']
    (1767596400, 1767542400)

    日族**不带** `datetime` / `time_stamp`（与旧树 `GQ_save_daily_metadata_reality` 一致）：

    >>> d2 = reality_feat_rows(df, 'day', created_at=0)[0]
    >>> 'datetime' in d2, 'time_stamp' in d2, d2['date_stamp'] == d['date_stamp']
    (False, False, True)
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
    out['date_stamp'] = (day_dt.astype(np.int64) // 10 ** 9).to_numpy()
    if family != 'day':
        full_dt = pd.Series(ts0, index=out.index).dt.tz_localize('Asia/Shanghai')
        out['datetime'] = full_dt.dt.strftime('%Y-%m-%d %H:%M:%S')
        out['time_stamp'] = full_dt.astype(np.int64) // 10 ** 9
    out['created_at'] = int(created_at)

    # 载荷列照原样（`NaN` → 不写该列，而不是写 `None` —— 与 `metadata_save` 同口径）
    rows = []
    for _, r in out.iterrows():
        doc = {}
        for col in out.columns:
            v = r[col]
            if pd.isna(v) if not isinstance(v, (list, dict)) else False:
                continue
            doc[col] = v.item() if hasattr(v, 'item') else v
        rows.append(doc)
    return rows


def save_reality_feat(features: pd.DataFrame, family: str, market_type,
                      verbose: bool = True, echo=None, created_at=None) -> dict:
    """取数（其实只组行）→ `upsert_fields` 落 ``golemq_stock_cn``。

    :param market_type: **该 code 的**市场类型（一只 code 一次调用，同旧 `charts.py`）；
        决定落 ``stock_`` 还是 ``index_`` 前缀，见 :func:`reality_feat_name`。
    :returns: ``{'name','rows','written','upserted','modified'}``
    """
    from GolemQ.datasource.writer import upsert_fields

    name = reality_feat_name(family, market_type)
    rows = reality_feat_rows(features, family, created_at=created_at)
    stats = {'name': name, 'rows': len(rows), 'written': 0, 'upserted': 0, 'modified': 0}
    if not rows:
        if verbose:
            (echo or print)(f'[reality] {name}: 0 行，跳过')
        return stats
    out = upsert_fields(_db()[name], rows, list(UNIQUE_KEYS[family]))
    stats.update(written=len(rows), upserted=out['upserted'], modified=out['modified'])
    if verbose:
        (echo or print)('[reality] {}: 写 {} 行 / upserted {} / modified {}'.format(
            name, stats['written'], stats['upserted'], stats['modified']))
    return stats


def fetch_reality_feat(codes, start, end, family='day', market_type=None):
    """读回 ``*_reality_feat_*``；**空结果返回 `None`**（不是空帧）。

    ⚠️ 这里刻意**偏离** `refdata.py` 那边「空返回空帧」的惯例：唯一的消费方
    （搬过来的 `calc_ChartScaned_persistence`）本来就是按 `is not None` 写的，
    而空帧还要照顾两层索引，容易踩 `Too many levels`。

    :param codes: 6 位代码或列表。
    :param start, end: ``'YYYY-MM-DD'``（**含**当日）。
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
    cur = _db()[reality_feat_name(family, market_type)].find(q, {'_id': 0}).sort(key, 1)
    df = pd.DataFrame(list(cur))
    return None if df.empty else df


#: ⚠️ **刻意不保留旧树那三个名字**（`GQ_save_metadata_reality` /
#: `GQ_save_daily_metadata_reality` / `GQ_fetch_daily_metadata_reality`）。
#: 它们的签名是 `(features, collections=<一个 Collection 句柄>)`，而本模块收的是
#: `(features, family, market_type)` —— **同名不同义**。留着当别名只会诱出错误调用
#: （`collections=` 一传就 TypeError），而那样的错在移植时最容易被顺手"修好"成
#: 传错的东西。搬过来的 `chartscan/predict.py` 一律用本模块的**新名**。
__all__ = [
    'REALITY_FAMILIES', 'UNIQUE_KEYS',
    'reality_feat_name', 'reality_feat_rows', 'save_reality_feat', 'fetch_reality_feat',
]
