"""给单个码补 ZenOLSLag（单列 $set，最小花销）。"""
import sys, time
import numpy as np, pandas as pd
from GolemQ import get_active_market
from GolemQ.analysis.zen_ols import calc_zen_ols_timing_lag
from GolemQ.models.alias import ZEN
from GolemQ.markets.StockCN.reality_save import save_reality_feat, reality_feat_name
from GolemQ.pipeline.review.steps.hour_extra import SE_WINDOW

code = sys.argv[1]
apply = '--apply' in sys.argv
m = get_active_market(); db = m.get_database()
name = reality_feat_name('60min', 'stock_cn')
rows = list(db[name].find({'code': code}, {'_id': 0, 'datetime': 1, 'time_stamp': 1}).sort('datetime', 1))
if not rows:
    print('无行'); sys.exit(0)
targets = pd.DatetimeIndex(pd.to_datetime([r['datetime'] for r in rows]))
print(f'{code}: 待算 {len(targets)} 行  {targets.min()} ~ {targets.max()}', flush=True)

r, _ = m.get_kline_price_min(code, start='2016-01-01', end=str(targets.max().date()) + ' 23:59:59',
                             frequency='60min', realtime=False, verbose=False)
ohlc = r.data
ohlc = ohlc[~ohlc.index.duplicated(keep='last')].sort_index()
ts = pd.DatetimeIndex(pd.to_datetime(ohlc.index.get_level_values(0)))
print(f'  60min 源 {len(ohlc)} 根', flush=True)

vals = np.full(len(targets), np.nan)
t0 = time.time()
for i, T in enumerate(targets):
    w = ohlc.loc[ts <= T].tail(SE_WINDOW)
    if len(w) < 30:
        continue
    try:
        vals[i] = float(calc_zen_ols_timing_lag(w).iloc[-1])
    except Exception:
        continue
    if (i + 1) % 500 == 0:
        print(f'  {i+1}/{len(targets)}  {time.time()-t0:.0f}s', flush=True)

ok = int(np.isfinite(vals).sum())
print(f'完成 {time.time()-t0:.0f}s  有值 {ok}/{len(vals)}  '
      f'正 {int((vals>0).sum())} 负 {int((vals<0).sum())}  极值 [{np.nanmin(vals):.0f},{np.nanmax(vals):.0f}]')
frame = pd.DataFrame({ZEN.OLS_TIMING_LAG: vals},
                     index=pd.MultiIndex.from_arrays([targets, [code]*len(targets)], names=['ts','code']))
if apply:
    out = save_reality_feat(frame, '60min', 'stock_cn', verbose=False, write_nulls=False, db=db)
    print('写入:', out)
else:
    print('dry-run（加 --apply 才写库）')
    print(frame.tail(5).to_string())
