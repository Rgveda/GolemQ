# coding:utf-8
"""ChartScan 的**推理与编排** —— 搬自旧树 `GolemAI/GolemQ/CV/charts.py:124-743`。

搬了什么
========
:func:`GQ_predict_ChartScanAI`（单标的 × 单频率）、
:func:`calc_ChartScaned_persistence`（读回已扫结果、决定 mode 0/1/2）、
:func:`calc_ChartScanAI_reality`（编排：日线 baseline → 逐日 × 三频率）、
以及两个「没探到」时的标记写入 :func:`GQ_markup_ChartScaned` / :func:`GQ_markup_daily_ChartScaned`。

**两处整块删除**（新读口已内建，见下），其余**行为逐字保留**。

取数解耦（QUANTAXIS → 8.3）
=========================
===============  ==========================================================
旧               新
===============  ==========================================================
`QA_fetch_stock_day_adv`      `get_active_market().get_kline_price_v3(..., realtime=False, if_fq=...)`
`QA_fetch_index_day_adv`      同上（市场实例按 code 自行分 stock/index/etf，**不传 market_type**）
`QA_fetch_stock_min_adv`      `.get_kline_price_min(..., frequency=freq, ...)`
`QA_fetch_index_min_adv`      同上
`QA_fetch_stock_name(code)`   **读口的第二个返回值 `codename`**（真名，查不到回落代码）
=========================  ==========================================================

⚠️ **删掉的两块**（删掉是**必须**，不是可选）：
1. **`.to_qfq()`**（旧 `charts.py:220-224`）—— 新读口**已经复权一次**
   （`kline83._apply_adjustments`）⇒ 再调会二次缩放；而且 `KlineResult` **没有**
   这个方法，留着直接 `AttributeError`。
2. **实时补数**（`GQ_fetch_stock_day_realtime` / `GQ_fetch_stock_min_realtime`）——
   已折进读口的 `realtime=True`（`kline83._merge_realtime`）。

⚠️ **复权口径是刻意变更**（用户 2026-10-13 定）：旧代码对指数/ETF **不调 `to_qfq`**
（即不复权），新读口对非股票一律前复权。故本模块**显式传 `if_fq='qfq'`** ——
「模型该看复权后的图」是修正，不是回退。

⚠️ **两个必须显式传参的日期助手**（不传会静默出错）：
* `GQ_util_get_last_day(ts=dt.today())` —— 它带 `@lru_cache`，24/7 服务里不传 `ts`
  会把「今天」缓存到**永远**。
* `GQ_util_if_tradetime(dt.now())` —— 它的默认值是**导入期**求值的，不传就是启动那一刻。

⚠️ **别「优化」每请求的重取**：`calc_ChartScanAI_reality` 先取日线 baseline，再在
**每个交易日 × 每个频率**上重新取一次完整历史（最多 ~618×3 次 Mongo 往返）。
那是 45 秒客户端超时的标定基准 —— 改了它，超时就跟着错位。
"""
from __future__ import annotations

import time
import traceback
from datetime import datetime as dt
from datetime import timedelta

import pandas as pd

from GolemQ.core.constants import AKA, MARKET_TYPE

from .render import (CHUNK_SIZE, CLOSE_TIMES, GQ_generate_chart_stock,
                     calculate_i)
from .model import load_model

__all__ = ['GQ_predict_ChartScanAI', 'calc_ChartScaned_persistence',
           'calc_ChartScanAI_reality', 'GQ_markup_ChartScaned',
           'GQ_markup_daily_ChartScaned']

#: 检出置信度阈值 —— 照旧树 `charts.py:262`。
CONFIDENCE = 0.26


def _last_day():
    """`GQ_util_get_last_day` 的**显式传参**包装（见模块 docstring 的告警）。"""
    from ..date_utils import GQ_util_get_last_day
    return GQ_util_get_last_day(ts=dt.today())


def _if_tradetime(now=None):
    """`GQ_util_if_tradetime` 的**显式传参**包装（见模块 docstring 的告警）。"""
    from ..date_utils import GQ_util_if_tradetime
    return GQ_util_if_tradetime(now or dt.now())


def _chart_scaned_flag(kline_data):
    """`ChartScaned`：**0** = 盘中且最后一根就是最近交易日（即在扫「今日末端」）；否则 **2**。

    逐字照搬旧树那三个重复表达式（`:134` / `:157` / `:310` / `:320`）。
    """
    last_date = pd.to_datetime(kline_data.tail(1).index.get_level_values(level=0)[-1].date())
    if _if_tradetime() and (timedelta(days=-0.618) < (_last_day() - last_date) < timedelta(days=0.618)):
        return 0
    return 2


def _fetch(code, freq, start_date, end_date, if_fq, verbose=False):
    """按频率取数 → `(DataFrame | None, codename)`。**旧树那两段 if/elif 的唯一替代点。**

    ⚠️ **必须「先不带实时取、再按判据决定合不合并」** —— 顺序不能反：旧树的三条判据里
    有一条要用**取回来的 `last_date`**，而实时合并会把「今天那根」并进来，先合成就再也
    判不出来了（实测：对 `end_date=2026-01-05` 的历史取数，无条件 `realtime=True`
    把末根改成了 `2026-10-12`）。

    :returns: ``(帧, 展示名)``；无数据时帧为 `None`（`get_kline_price_min` 空返回空对象，
        也一并折成 `None`，与旧代码「`len < chunk_size` 就 return」的分支对齐）。
    """
    # ⚠️ `_merge_realtime` 是 `kline83` 的**私有**名 —— 这里刻意复用而不是另写一份：
    # 合并逻辑（两道门 + 组装）只能有一处，否则又是一次「平行实现」。
    from ..kline83 import _merge_realtime

    # ⚠️ **取 K 线一律走市场实例**（2026-10-18 用户定：「必须走 get_kline_price_v3/_min」、
    # 「不要直接操作 MongoDB」）。此前从 `..kline83` 直接 import —— 那是**实现**，
    # 绕过 `get_active_market()` 就绕过了「取数唯一入口」（`CLAUDE.md`）。
    # `market_type` 不再显式传：市场侧的包装函数没有这个形参，由 kline83 按 code 自行分类
    # （与 `client.market_type_of(code)` 同一条规则，结果一致）。
    from GolemQ.core.market_registry import get_active_market
    market = get_active_market()

    _end = pd.to_datetime(end_date)
    if freq == 'day':
        res, codename = market.get_kline_price_v3(
            code, start=start_date.strftime('%Y-%m-%d'),
            end=_end.strftime('%Y-%m-%d'), realtime=False, if_fq=if_fq, verbose=False)
    else:
        res, codename = market.get_kline_price_min(
            code, start=start_date.strftime('%Y-%m-%d %H:%M:%S'),
            end=(_end + timedelta(hours=17)).strftime('%Y-%m-%d %H:%M:%S'),
            frequency=freq, realtime=False, if_fq=if_fq, verbose=False)
    if res is None or len(getattr(res, 'data', ())) == 0:
        return None, codename
    last_date = pd.to_datetime(res.data.tail(1).index.get_level_values(level=0)[-1])
    # 旧树 `charts.py:229-232` 那三条判据，逐字（第三条要用刚取回的 `last_date`）
    if _if_tradetime() or ((_last_day() - _end) < timedelta(hours=18)) or \
            (freq == 'day' and (_end - last_date) != timedelta(days=0)):
        _merge_realtime(res, [code], 'day' if freq == 'day' else freq, verbose=verbose)
    return res.data, codename


def GQ_markup_daily_ChartScaned(code, kline_data, market_type, verbose=False):
    """「没探到」时在**日族**留一行**只有 `ChartScaned`** 的标记（照旧 `:124-143`）。"""
    from ..reality_save import save_reality_feat
    try:
        frame = pd.DataFrame([{
            'code': code,
            'date': pd.to_datetime(kline_data.tail(1).index.get_level_values(level=0)[-1].date()),
            'ChartScaned': _chart_scaned_flag(kline_data),
        }]).set_index(['date', 'code'], drop=False)
        save_reality_feat(frame, 'day', market_type, verbose=verbose)
    except Exception:                     # noqa: BLE001 照旧：吞掉但打栈
        traceback.print_exc()


def GQ_markup_ChartScaned(code, kline_data, market_type, verbose=False,
                          family='60min'):
    """「没探到」时在 **`family` 族**留一行只有 `ChartScaned` 的标记（照旧 `:146-166`）。

    :param family: 落哪个分钟族（`'60min'` / `'15min'`）。⚠️ 旧树的同名函数收的是
        ``collections=`` —— 两个调用点分别传 60min 与 15min **两张表**（`:380`/`:385`）。
        搬过来时把参数换成 `family`（收 `market_type` 由 `reality_feat_name` 出前缀），
        默认 `'60min'` **保持旧默认行为**。
    """
    from ..reality_save import save_reality_feat
    try:
        frame = pd.DataFrame([{
            'code': code,
            'date': pd.to_datetime(kline_data.tail(1).index.get_level_values(level=0)[-1].date()),
            'datetime': pd.to_datetime(kline_data.tail(1).index.get_level_values(level=0)[-1]),
            'ChartScaned': _chart_scaned_flag(kline_data),
        }]).set_index(['datetime', 'code'], drop=False)
        save_reality_feat(frame, family, market_type, verbose=verbose)
    except Exception:                     # noqa: BLE001
        traceback.print_exc()


def GQ_predict_ChartScanAI(code: str = '300085', market_type=MARKET_TYPE.STOCK_CN,
                          freq: str = 'day', end_date=pd.to_datetime('2024-07-03'),
                          chunk_size: int = CHUNK_SIZE, model=None,
                          if_fq: str = 'qfq', verbose: bool = False):
    """单标的 × 单频率：取数 → 出图 → YOLO → 挑最近 `end_date` 的框 → 落库。

    逐字搬自旧树 `charts.py:169-445`；上表那几处取数/复权/实时已换（见模块 docstring）。
    """
    from ..reality_save import save_reality_feat

    if verbose:
        print(f'Code: {code}', end_date, freq)

    # 历史窗口：日 512 天 / 15min 63 天 / 60min 126 天（旧树原值，别改）
    if freq == 'day':
        start_date = end_date - timedelta(days=512)
    elif freq == '15min':
        start_date = end_date - timedelta(days=63)
    else:
        start_date = end_date - timedelta(days=126)

    kline_data, codename = _fetch(code, freq, start_date, end_date, if_fq)
    if kline_data is None or len(kline_data) < chunk_size:
        return
    last_date = kline_data.tail(1).index.get_level_values(level=0)[-1]
    if (freq == 'day') and ((end_date - last_date) != timedelta(days=0)) and verbose:
        print(end_date, last_date, (end_date - last_date))
    if (kline_data[AKA.CLOSE].isna().all()):
        print(kline_data[[AKA.CLOSE, AKA.OPEN, AKA.HIGH, AKA.LOW, AKA.VOLUME]].tail(60))
        time.sleep(10)

    try:
        chart_buffer = GQ_generate_chart_stock(kline=kline_data,
                                               ticker=f'{codename}({code}) ',
                                               chunk_size=chunk_size, freq=freq)
    except Exception:                     # noqa: BLE001 照旧：打栈后继续（下面会因 buffer 为 None 炸）
        traceback.print_exc()
        return

    model = load_model(model)
    from PIL import Image                  # noqa: PLC0415 惰性：GolemAI env 才有
    uploaded_image = Image.open(chart_buffer)
    res = model.predict(uploaded_image, conf=CONFIDENCE)
    boxes = res[0].boxes

    try:
        results = []
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            class_id = int(box.cls[0])
            conf = box.conf[0].item()
            # ⚠️ 取 **level 0**：新读口的索引是 `(ts, code)` **两层 MultiIndex**，
            # 直接 `index[i]` 会拿到 `(ts, '600519')` 元组（旧树 QUANTAXIS 的 `.data`
            # 是单层索引，所以老代码那样写能得到 Timestamp）。实测踩过：
            # `TypeError: unsupported operand type(s) for -: 'tuple' and 'Timestamp'`。
            date_x2 = kline_data.tail(chunk_size).index.get_level_values(level=0)[calculate_i(x2)]
            base = {
                'code': code,
                'date': pd.to_datetime(kline_data.tail(1).index.get_level_values(level=0)[-1].date()),
            }
            if freq == 'day':
                results.append({**base, 'ClsMaj': res[0].names[class_id],
                                'ConfidenceMaj': conf, 'CoordinatesMaj': date_x2})
            elif freq == '15min':
                results.append({**base,
                                'datetime': pd.to_datetime(kline_data.tail(1).index.get_level_values(level=0)[-1]),
                                'ClsMin': res[0].names[class_id], 'ConfidenceMin': conf,
                                'CoordinatesMin': date_x2,
                                'ChartScaned': _chart_scaned_flag(kline_data)})
            else:
                results.append({**base,
                                'datetime': pd.to_datetime(kline_data.tail(1).index.get_level_values(level=0)[-1]),
                                'Class': res[0].names[class_id], 'Confidence': conf,
                                'Coordinates': date_x2,
                                'ChartScaned': _chart_scaned_flag(kline_data)})

        if not results:
            # 一个框都没有：非日频的按「收盘那几根」补标记（照旧 `:431-443`）
            if freq != 'day':
                if kline_data.tail(1).index.get_level_values(level=0)[-1].strftime('%H:%M') in CLOSE_TIMES:
                    GQ_markup_ChartScaned(code, kline_data, market_type, verbose)
                GQ_markup_daily_ChartScaned(code, kline_data, market_type, verbose)
            return

        key = {'day': 'CoordinatesMaj', '15min': 'CoordinatesMin'}.get(freq, 'Coordinates')
        closest = min(results, key=lambda x: abs((x[key] - end_date).total_seconds()))
        # 只写「近 31 天」的框；远的放弃（照旧 `:333` / `:356` / `:396`）
        anchor = closest['date'] if freq == 'day' else closest['datetime']
        if (anchor - closest[key]) < timedelta(days=31):
            closest[key] = closest[key].tz_localize('Asia/Shanghai').timestamp() * 1000
            # ⚠️ **第一笔永远是日族** —— 旧树三条支路（`:342` / `:363` / `:406`）都调
            # `GQ_save_daily_metadata_reality`。**分钟频率也往日族写**，因为
            # `calc_ChartScaned_persistence` 读的正是日族里的 `CoordinatesMin` /
            # `Coordinates` 两列来判「要不要重扫」。索引传 date 还是 datetime 无所谓：
            # 该写口按 level0 的 **`.date`** 取 `date_stamp`（见 `reality_feat_rows`）。
            # ⚠️ 曾把这里写成「按 freq 选族」，后果是日族永远收不到分钟列的检出 ⇒
            # `all_nat_min` 恒真 ⇒ **每天全量重扫 618 交易日 × 3 频率**。
            save_reality_feat(pd.DataFrame([closest]).set_index(['date', 'code'], drop=False),
                              'day', market_type, verbose=verbose)
            if freq == '15min':
                # 15min 的检出**同时**写 60min 与 15min 两个族（照旧 `:367-377`），
                # 但 60min 只认「收盘那几根」。
                ok60 = closest['datetime'].strftime('%H:%M') in CLOSE_TIMES
                for family in (('60min', '15min') if ok60 else ('15min',)):
                    save_reality_feat(pd.DataFrame([closest]).set_index(['datetime', 'code'], drop=False),
                                      family, market_type, verbose=verbose)
            elif freq != 'day':
                # 60min：**无条件**写 60min 族（照旧 `:411-414`，没有收盘时刻闸）
                save_reality_feat(pd.DataFrame([closest]).set_index(['datetime', 'code'], drop=False),
                                  '60min', market_type, verbose=verbose)
        else:
            if verbose:
                print(f'远端买卖点，自动放弃：{closest[key]}')
            if freq == 'day':
                return
            if closest['datetime'].strftime('%H:%M') in CLOSE_TIMES:
                GQ_markup_ChartScaned(code, kline_data, market_type, verbose)
            if freq == '15min':
                # ⚠️ 旧树 15min 这条**无条件**再写一张 15min 表（`:385`）——
                # 漏掉它 ⇒ 15min 族在「探到但太远」时永远没有 `ChartScaned` 记录。
                GQ_markup_ChartScaned(code, kline_data, market_type, verbose,
                                      family='15min')
            GQ_markup_daily_ChartScaned(code, kline_data, market_type, verbose)
    except Exception:                     # noqa: BLE001 照旧：整段吞掉打栈
        traceback.print_exc()


def calc_ChartScaned_persistence(ticker, market_type=MARKET_TYPE.STOCK_CN,
                                 start_date=None, end_date=None, ret_metadata: bool = False):
    """读回已扫结果，决定该标的要**重扫什么** —— 搬自旧树 `charts.py:448-565`。

    :returns: ``ret_metadata=True`` → `(全部NaT?, 日NaT?, 分钟NaT?, 待扫日期列表)`；
        否则 → `(ticker, market_type, mode, start, end)`，`mode` 0=不用扫 / 1=全扫 / 2=只扫末端。
    """
    from ..reality_save import fetch_reality_feat

    if start_date is None:
        start_date = _last_day() - timedelta(days=1008)
    if end_date is None:
        end_date = _last_day()

    stock_daily_feats = fetch_reality_feat(ticker, start_date.strftime('%Y-%m-%d'),
                                           end_date.strftime('%Y-%m-%d'),
                                           'day', market_type)
    if stock_daily_feats is not None:
        stock_daily_feats = _coords_to_datetime(stock_daily_feats)
        last_day = _last_day()
        stock_daily_feats = stock_daily_feats.reindex(
            stock_daily_feats.index.union([(last_day, ticker)]))
        current_date = pd.to_datetime(dt.now())
        one_month_ago = current_date - pd.DateOffset(months=1)
        idx_dates = stock_daily_feats.index.get_level_values('date')
        recent = stock_daily_feats.loc[(idx_dates >= one_month_ago) & (idx_dates <= current_date)]
        all_nat = recent['Coordinates'].isna().all()
        all_nat_maj = recent['CoordinatesMaj'].isna().all()
        all_nat_min = recent['CoordinatesMin'].isna().all()
        recent_list = []
        if 'ChartScaned' in stock_daily_feats.columns:
            all_nat = recent['ChartScaned'].isna().all()
            all_nat_maj = recent['ChartScaned'].isna().all()
            non = recent[(recent['ChartScaned'] != 1) & (recent['ChartScaned'] != 2)]
            recent_list = sorted(non.index.get_level_values(level=0)) if not non.empty else False
    else:
        all_nat = all_nat_maj = all_nat_min = True
        recent_list = []

    if ((timedelta(hours=-1) < (end_date - start_date) < timedelta(hours=9)) and
            (timedelta(hours=-1) < (end_date - _last_day()) < timedelta(hours=9))) | \
            (_if_tradetime() and (timedelta(days=-0.618) < (_last_day() - pd.to_datetime(end_date)) < timedelta(days=0.618))):
        recent_list = sorted([pd.to_datetime(end_date)])

    if ret_metadata:
        return all_nat, all_nat_maj, all_nat_min, recent_list
    if all_nat or all_nat_maj or all_nat_min:
        mode = 1
    elif recent_list:
        mode = 2
    else:
        mode = 0
    return (ticker, market_type, mode,
            recent_list[0] if mode == 2 else pd.to_datetime(start_date),
            recent_list[-1] if mode == 2 else pd.to_datetime(end_date))


def _coords_to_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """`Coordinates*`（**毫秒**）→ datetime 列 + `*Timestamp` 列（照旧 `:463-495`）。"""
    out = df.copy()
    for col in ('Coordinates', 'CoordinatesMaj', 'CoordinatesMin'):
        if col in out.columns:
            out[col] = out[col].replace({pd.NaT: None}) / 1000
            out = out.rename(columns={col: col + 'Timestamp'})
            out[col] = pd.to_datetime(out[col + 'Timestamp'].dropna(), unit='s')
        else:
            out[col] = None
    return out


def calc_ChartScanAI_reality(ticker: str = '300085', market_type=MARKET_TYPE.STOCK_CN,
                             start_date=pd.to_datetime('2020-01-01'),
                             end_date=pd.to_datetime('2024-07-03'),
                             enforce: bool = False, model=None, if_fq: str = 'qfq'):
    """编排：读回已扫状态 → 取日线 baseline → **逐日 × 三频率**跑预测。搬自 `:568-742`。

    :returns: `without_processing`（True = 这一趟什么都没跑）。
    """
    if not enforce:
        all_nat, all_nat_maj, all_nat_min, recent = calc_ChartScaned_persistence(
            ticker=ticker, market_type=market_type,
            start_date=start_date, end_date=end_date, ret_metadata=True)
    else:
        all_nat = all_nat_maj = all_nat_min = True
        recent = [start_date, end_date]

    if not (all_nat or all_nat_maj or all_nat_min or recent):
        return True

    model = load_model(model)
    if all_nat or all_nat_maj or all_nat_min:
        lo, hi = start_date, end_date
    else:
        lo, hi = recent[0], recent[-1]
    # 日线 baseline：**新读口的 `realtime=True` 已把「当日实时那根」并进来**，
    # 故旧树那两段 `GQ_fetch_stock_day_realtime` 补数整块删掉。
    kline_baseline, _ = _fetch(ticker, 'day', lo, hi, if_fq)
    if kline_baseline is not None and len(kline_baseline) > 0:
        kline_baseline = kline_baseline.set_index(['ts', 'code'], drop=True).sort_index() \
            if 'ts' in kline_baseline.columns else kline_baseline

    end_date = pd.to_datetime(end_date)
    without_processing = True
    for cond, freq in ((all_nat_maj or recent is not False, 'day'),
                       (all_nat_min or recent is not False, '15min'),
                       (all_nat or recent is not False, '60min')):
        if not cond or kline_baseline is None:
            continue
        kline_baseline.tail(618).groupby(level=[0], group_keys=True).apply(
            lambda x, _f=freq: GQ_predict_ChartScanAI(
                code=ticker, market_type=market_type, freq=_f,
                end_date=pd.to_datetime(x.index.get_level_values(level=0)[0]),
                model=model, if_fq=if_fq, verbose=bool(enforce)))
        if len(kline_baseline) > 0:
            without_processing = False

    if without_processing:
        print(f'without_processing code: {ticker}', all_nat_min, all_nat,
              all_nat_maj, recent, kline_baseline.index if kline_baseline is not None else '')
    return without_processing
