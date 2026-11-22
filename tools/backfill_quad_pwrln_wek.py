# coding: utf-8
"""全市场周线 `4QPwrln` → `MasQuadPwrlnWekMed/Mode`（落 massive）+ 可选 60min 列。

口径：
* 每码日线 → `weekly_ohlc` → `calc_four_quadrant_boost(annual=52)` → `4QPwrln`；
* **日线回溯**：`--day-start` 之前至少 `--lookback-bars`（默认 3000 自然日 ≈
  2050 交易日）——`4QPwrln` 里最长的是周线 MA233，即 233 周 ≈ 1100 根日线，
  回看不足会让 MA233 全 NaN ⇒ powerline 系统性低 1。`--day-start` 为空时
  由 `start - lookback` 自动推；
* 日期轴取**交易日历**（`stock_reality_feat_day` 里 ref 码的 distinct 日期），
  不再依赖 massive 表已有的行 —— massive 15:00 行只有 2022-06-16 起，
  用它当轴会把 2018~2022 整段丢掉；
* `weekly_ohlc` 的周标签是**该周最后一个实际交易日**，所以对某日 `D`
  直接 `ffill` 拿到的就是「D 之前已走完的那一周」（D 本身是周末交易日时
  就是当周）—— **无未来函数**，不需要再按 15:00 做分支；
* 横截面：中位数 / 众数 → `MasQuadPwrlnWekMed` / `MasQuadPwrlnWekMode`；
* 可选把 `4QPwrlnWek` 写到 `stock_reality_feat_60min`（默认只**更新已存在的行**）。

用法::

    python tools/backfill_quad_pwrln_wek.py --start 2018-01-01 --end 2024-07-31
    python tools/backfill_quad_pwrln_wek.py ... --apply            # 写 massive
    python tools/backfill_quad_pwrln_wek.py ... --write-60min      # 写 60min 已有行
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from GolemQ.models.alias import LTT  # noqa: E402


def _day_start(start: str, lookback_days: int) -> str:
    """首批输出的日线回溯起点（自然日）。"""
    return (pd.Timestamp(start) - pd.Timedelta(days=int(lookback_days))
            ).strftime('%Y-%m-%d')


def _weekly_series(code: str, day_start: str, end: str) -> pd.Series:
    from GolemQ import get_active_market
    from GolemQ.analysis.quadrant_jit import calc_four_quadrant_boost
    from GolemQ.analysis.timeseries import weekly_ohlc
    m = get_active_market()
    r, _ = m.get_kline_price_v3(code, start=day_start, end=end,
                                realtime=False, verbose=False)
    day = None if r is None else getattr(r, 'data', None)
    if day is None or len(day) < 520:
        return pd.Series(dtype=float)
    wk = weekly_ohlc(day)
    if wk is None or len(wk) < 30:
        return pd.Series(dtype=float)
    try:
        f = calc_four_quadrant_boost(
            code, features=pd.DataFrame(index=wk.index), ohlc_data=wk,
            annual=52)
    except Exception:               # noqa: BLE001
        return pd.Series(dtype=float)
    if LTT.QUADRANT_POWERLINE not in f.columns:
        return pd.Series(dtype=float)
    s = pd.Series(
        np.asarray(f[LTT.QUADRANT_POWERLINE], dtype=float),
        index=pd.DatetimeIndex(pd.to_datetime(
            wk.index.get_level_values(0))).normalize())
    s = s[~s.index.duplicated(keep='last')].sort_index()
    return s


def _week_end_mask(dates: pd.DatetimeIndex,
                   cal_ext: pd.DatetimeIndex) -> np.ndarray:
    """`dates[i]` 是否是该 ISO 周的**最后一个交易日**（用延伸日历判，含节假日）。

    区间被截断时（如 end=2024-07-31 是周三），`weekly_ohlc` 会把那一周
    打成「只含 3 天」的标签=07-31，直接 ffill 就会在这一天吃到不完整周。
    延伸日历能认出该周真正的末日是 08-02 ⇒ 判成「未走完」，回退上一完整周。
    """
    if len(dates) == 0:
        return np.zeros(0, dtype=bool)
    per = pd.PeriodIndex(cal_ext, freq='W')
    last = pd.Series(cal_ext, index=per).groupby(level=0).max()
    want = last.reindex(pd.PeriodIndex(dates, freq='W')).to_numpy()
    return pd.DatetimeIndex(want) == pd.DatetimeIndex(dates)


def _on_dates(code: str, dates: pd.DatetimeIndex, day_start: str, end: str,
              week_end: np.ndarray, min_weeks: int = 0):
    s = _weekly_series(code, day_start, end)
    if len(s) == 0:
        return None
    cur = s.reindex(dates, method='ffill').to_numpy(dtype=float)
    prev = s.reindex(dates - pd.Timedelta(days=1),
                     method='ffill').to_numpy(dtype=float)
    vals = np.where(week_end, cur, prev)
    if min_weeks:
        # `4QPwrln` 最长窗口是周线 MA233 —— 该码在 D 之前攒够 233 根周线之前，
        # 后几项恒为 0，powerline 结构性偏低。攒够之前的日期填 NaN（不进截面）。
        seen = pd.Series(np.arange(1, len(s) + 1), index=s.index).reindex(
            dates, method='ffill').to_numpy(dtype=float)
        vals = np.where(np.nan_to_num(seen) >= min_weeks, vals, np.nan)
    return vals


def _trading_days(start: str, end: str, ref: str = '600519') -> pd.DatetimeIndex:
    """交易日历：取 ref 码（长期不停牌）在日线表里的 distinct 日期。"""
    from GolemQ import get_active_market
    db = get_active_market().get_database()
    raw = db.stock_reality_feat_day.distinct(
        'datetime', {'code': ref,
                     'datetime': {'$gte': start, '$lte': end + ' 23:59:59'}})
    s = pd.to_datetime(pd.Series(raw), errors='coerce').dropna()
    return pd.DatetimeIndex(s).normalize().unique().sort_values()


def _write_60min(code: str, start: str, end: str, day_start: str,
                 week_end_dates) -> int:
    from GolemQ import get_active_market
    from GolemQ.markets.StockCN.reality_save import save_reality_feat
    from GolemQ.markets.StockCN.symbol import is_stock_cn
    m = get_active_market()
    s = _weekly_series(code, day_start, end)
    if len(s) == 0:
        return 0
    db = m.get_database()
    rows = list(db.stock_reality_feat_60min.find(
        {'code': code, 'datetime': {'$gte': start,
                                    '$lte': end + ' 23:59:59'}},
        {'datetime': 1, '_id': 0}).sort('datetime', 1))
    if not rows:
        return 0
    ts = pd.DatetimeIndex([pd.Timestamp(r['datetime']) for r in rows])
    # 当周（标签=当周最后交易日）只在「该周最后交易日的 15:00」才允许用；
    # 其余一律取上一已完成周 —— 区间尾部被截断的周也不会漏进不完整周
    cur = s.reindex(ts, method='ffill')
    prev = s.reindex(ts - pd.Timedelta(days=1), method='ffill')
    is_we = pd.DatetimeIndex(ts).normalize().isin(week_end_dates)
    vals = np.where((ts.strftime('%H:%M') == '15:00') & is_we, cur, prev)
    frame = pd.DataFrame(
        {LTT.QUADRANT_POWERLINE_WEEKLY: vals},
        index=pd.MultiIndex.from_arrays([ts, [str(code)] * len(ts)],
                                        names=['ts', 'code']))
    save_reality_feat(frame, '60min', is_stock_cn(code)[1], verbose=False,
                      db=db)
    return len(frame)


def _write_60min_synthetic(code: str, start: str, end: str,
                           day_start: str, week_end_dates,
                           dry_run: bool = False) -> int:
    """给该码**有日线但没有 60min 行**的交易日补一根 15:00 行，只带 `4QPwrlnWek`。

    `stock_reality_feat_60min` 是**特征表**（不是纯 OHLCV 表）：库里 2021-11-23
    的 15:00 行只有 16 个字段、没有 close/open，所以稀疏行本来就是合法形态。

    只补「该码当天真的在交易」的日期 —— 用该码自己的日线日期，不用全市场
    交易日历，否则会把上市前的幽灵行插进去。

    :returns: 写过（或 dry_run 下将要写）的行数
    """
    from GolemQ import get_active_market
    from GolemQ.markets.StockCN.reality_save import save_reality_feat
    from GolemQ.markets.StockCN.symbol import is_stock_cn
    m = get_active_market()
    s = _weekly_series(code, day_start, end)
    if len(s) == 0:
        return 0
    r, _ = m.get_kline_price_v3(code, start=start, end=end,
                                realtime=False, verbose=False)
    day = None if r is None else getattr(r, 'data', None)
    if day is None or len(day) == 0:
        return 0
    traded = pd.DatetimeIndex(
        pd.to_datetime(day.index.get_level_values(0))).normalize().unique()
    traded = traded.sort_values()
    db = m.get_database()
    have = {pd.Timestamp(r['datetime']).normalize() for r in
            db.stock_reality_feat_60min.find(
                {'code': code, 'datetime': {'$gte': start,
                                            '$lte': end + ' 23:59:59'}},
                {'datetime': 1, '_id': 0})}
    todo = pd.DatetimeIndex([d for d in traded if d not in have])
    if len(todo) == 0:
        return 0
    if dry_run:
        return len(todo)
    ts = todo + pd.Timedelta(hours=15)
    is_we = todo.normalize().isin(week_end_dates)
    cur = s.reindex(ts, method='ffill')
    prev = s.reindex(ts - pd.Timedelta(days=1), method='ffill')
    vals = np.where(is_we, cur, prev)
    frame = pd.DataFrame(
        {LTT.QUADRANT_POWERLINE_WEEKLY: vals},
        index=pd.MultiIndex.from_arrays([ts, [str(code)] * len(ts)],
                                        names=['ts', 'code']))
    save_reality_feat(frame, '60min', is_stock_cn(code)[1], verbose=False,
                      db=db, write_nulls=False)
    return len(frame)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--start', default='2018-01-01')
    p.add_argument('--end', default='2024-07-31')
    p.add_argument('--lookback-days', type=int, default=3000,
                   help='首批输出的日线回溯自然日（默认 3000 ≈ 2050 交易日）')
    p.add_argument('--day-start', default='',
                   help='显式指定日线起点（默认 start - lookback-days）')
    p.add_argument('--workers', type=int, default=16)
    p.add_argument('--out', default='',
                   help='逐日 CSV 路径（默认 output/mas_quad_pwrln_wek.csv）')
    p.add_argument('--apply', action='store_true', help='写 massive 表')
    p.add_argument('--write-60min', action='store_true',
                   help='给**已有** 60min 行写 4QPwrlnWek')
    p.add_argument('--synthetic-60min', action='store_true',
                   help='无 60min 行的交易日补建 15:00 行（只带周线列）')
    p.add_argument('--dry-run', action='store_true',
                   help='配合 --synthetic-60min / --write-60min 只统计不写库')
    p.add_argument('--min-weeks', type=int, default=0,
                   help='截面只算「周线序列已有 N 根」的码（N=233 才是完整 powerline）')
    p.add_argument('--skip-cross', action='store_true',
                   help='跳过全市场矩阵/CSV，只做 60min 侧')
    a = p.parse_args(argv)
    day_start = a.day_start or _day_start(a.start, a.lookback_days)
    out = a.out or 'output/mas_quad_pwrln_wek.csv'

    from GolemQ import get_active_market
    m = get_active_market()
    db = m.get_database()
    codes = sorted({str(c) for c in m.scan_dataset().stock_list()['code']})

    dates = _trading_days(a.start, a.end)
    # 延伸日历判「该周最后一个交易日」，尾部截断的周不会被当成完整周
    cal_ext = _trading_days(
        a.start, (pd.Timestamp(a.end) + pd.Timedelta(days=10)).strftime('%Y-%m-%d'))
    week_end = _week_end_mask(dates, cal_ext)
    week_end_dates = set(pd.DatetimeIndex(cal_ext[_week_end_mask(cal_ext, cal_ext)]))
    print('[wek] day_start', day_start, '| dates', len(dates),
          dates.min(), dates.max(), '| week_end days', int(week_end.sum()))

    from joblib import Parallel, delayed
    if not a.skip_cross:
        arrays = Parallel(n_jobs=a.workers, backend='loky')(
            delayed(_on_dates)(c, dates, day_start, a.end, week_end,
                               a.min_weeks)
            for c in codes)
        ok = [(c, x) for c, x in zip(codes, arrays) if x is not None]
        mat = np.vstack([x for _, x in ok])
        print('[wek] codes', len(ok), 'mat', mat.shape)

    if not a.skip_cross:
        with np.errstate(invalid='ignore'):
            import warnings as _w
            with _w.catch_warnings():
                _w.simplefilter('ignore', category=RuntimeWarning)
                med = np.nanmedian(mat, axis=0)
        dfm = pd.DataFrame(mat)
        mode_tab = dfm.mode(axis=0, dropna=True)
        mode = np.full(len(dates), np.nan)
        if len(mode_tab):
            mode[np.asarray(mode_tab.columns)] = mode_tab.iloc[0].to_numpy(
                dtype=float)
        n_valid = np.isfinite(mat).sum(axis=0)

        out_df = pd.DataFrame({'date': dates, 'MasQuadPwrlnWekMed': med,
                               'MasQuadPwrlnWekMode': mode, 'n': n_valid})
        print(out_df.head(3).to_string(index=False))
        print(out_df.tail(3).to_string(index=False))
        os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
        out_df.to_csv(out, index=False, encoding='utf-8')
        print('[wek] wrote', out, 'rows', len(out_df))

    if a.apply and not a.skip_cross:
        from GolemQ.datasource.writer import upsert_fields
        # 只对 massive 已有的 15:00 日期写，避免插出稀疏行
        rows = list(db.stock_massive_reality_60min_models.find(
            {'datetime': {'$regex': ' 15:00:00$',
                          '$gte': a.start + ' 00:00:00',
                          '$lte': a.end + ' 23:59:59'}},
            {'_id': 0, 'datetime': 1, 'time_stamp': 1}).sort('datetime', 1))
        stamp_by_dt = {pd.Timestamp(r['datetime']): int(r['time_stamp'])
                       for r in rows if r.get('time_stamp') is not None}
        docs = []
        for d, med_v, mode_v in zip(dates, med, mode):
            dt = pd.Timestamp(d) + pd.Timedelta(hours=15)
            if dt not in stamp_by_dt:
                continue
            docs.append({'datetime': dt.strftime('%Y-%m-%d %H:%M:%S'),
                         'time_stamp': stamp_by_dt[dt],
                         'MasQuadPwrlnWekMed': (None if not np.isfinite(med_v)
                                                else float(med_v)),
                         'MasQuadPwrlnWekMode': (None if not np.isfinite(mode_v)
                                                 else float(mode_v))})
        res = upsert_fields(db.stock_massive_reality_60min_models, docs,
                            ['time_stamp'])
        print('[wek] massive upserted', res['upserted'], 'modified',
              res['modified'], 'docs', len(docs))

    if a.write_60min:
        n = Parallel(n_jobs=a.workers, backend='loky')(
            delayed(_write_60min)(c, a.start, a.end, day_start,
                                  week_end_dates) for c in codes)
        print('[wek] 60min rows written', int(sum(n)))

    if a.synthetic_60min:
        n = Parallel(n_jobs=a.workers, backend='loky')(
            delayed(_write_60min_synthetic)(c, a.start, a.end, day_start,
                                            week_end_dates, a.dry_run)
            for c in codes)
        print('[wek] 60min synthetic rows', int(sum(n)),
              '(dry-run)' if a.dry_run else '')
    return 0


if __name__ == '__main__':
    sys.exit(main())
