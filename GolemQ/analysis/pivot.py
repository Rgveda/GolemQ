# coding: utf-8
"""
缠中说禅 走势中枢（pivot，盘整箱体）识别与绘制
==============================================

中枢（pivot）是缠论的核心概念：由至少三个连续次级别走势类型（笔 BI）
的重叠部分构成。重叠区间 [ZD, ZG] 即为中枢区间，也就是盘整（震荡）期间
价格反复波动的“箱体”：

    ZG = min(各笔高点)   —— 中枢上沿（上轨 / 箱顶）
    ZD = max(各笔低点)   —— 中枢下沿（下轨 / 箱底）
    GG = max(各笔高点)   —— 中枢最高点（上轨延伸上界）
    DD = min(各笔低点)   —— 中枢最低点（下轨延伸下界）

盘整本质上是价格在某个中枢箱体内反复波动；趋势则是中枢沿某一方向逐级
抬升/下降。本模块从 czsc 的笔序列（bi_list）出发识别所有中枢，判断走势
类型（盘整 / 上涨趋势 / 下跌趋势），并可在 K 线图上绘制箱体价位。

来源
====
自旧树 ``GolemQ_old/czsc/pivot.py`` 搬运（两处依赖改造见下）。算法核心
``find_zs`` 在同包 ``GolemQ.analysis._zs``（从旧树 ``czsc/analyze.py`` 单独抽出，
只此一个函数 —— 那个 vendored 包里的 `KlineAnalyze`/`signals`/`cobra`/`data`/`utils`
**一律未搬**）。

搬运时只动了两个 import：

1. ``from .analyze import find_zs`` → ``from ._zs import find_zs``（见上述）；
2. ``from GolemQ.models.alias import ZEN`` —— 新树原本没有 ``ZEN``，
   现已在同源位置 ``GolemQ/models/alias.py`` 补上（值与旧树逐字相同）。

``from czsc import CZSC`` / ``from czsc.objects import RawBar, Freq`` **未动** ——
装着的 czsc 0.7.10 上这两个路径实测成立（`CZSC`/`RawBar`/`Freq` 均可用，
``bi.fx_a.mark`` 为 ``Mark.G``/``Mark.D``）。
"""
from typing import List, Dict, Callable, Optional

import numpy as np
import pandas as pd

from ._zs import find_zs


def _mark_to_str(mark) -> str:
    """把 czsc 的 Mark 枚举归一化为 'd'（底分型）/ 'g'（顶分型）。"""
    name = getattr(mark, 'name', None)
    if name is None:
        name = str(mark)
    if name in ('D', '底分型', 'd'):
        return 'd'
    return 'g'


def bi_confirm_map(bars, max_bi_count: int = 512, verbose: bool = False) -> Dict:
    """逐 bar 回放 CZSC，得到每根笔端点（``bi.fx_b.dt``）**首次成为末笔**的时间。

    czsc 只有在后续 bar 上才能确认一笔，故 ``fx_b`` 的价格发生在 ``dt``，但要到
    回放中它成为末笔的那根 bar 才真正"可被知道"。返回的映射即为该确认时间，
    供 ``strict`` 因果口径使用（对齐后分型点不会在被确认前进入极值/箱体计算）。

    :param bars: RawBar 序列（须与建 ``bi_list`` 时同一批 bar）。
    :return: ``{fx_b.dt: 确认 bar 的 dt}``。
    """
    from czsc import CZSC

    conf: Dict = {}
    bars = list(bars)
    if not bars:
        return conf
    czsc = CZSC(bars[:1], max_bi_count=max_bi_count, verbose=verbose)
    prev = None
    for bar in bars[1:]:
        czsc.update(bar)
        if czsc.bi_list:
            key = czsc.bi_list[-1].fx_b.dt
            if key != prev:
                conf[key] = bar.dt
                prev = key
    return conf


def bi_list_to_points(bi_list: List, conf_map: Optional[Dict] = None) -> List[Dict]:
    """把 czsc 的笔列表（BI 对象）转成 ``find_zs`` 所需的标记点序列。

    每个点含 ``dt / fx_mark / bi / fx`` 字段，按时间升序、底顶交替排列。
    笔的端点（``fx_a`` / ``fx_b``）即为分型点：相邻笔首尾相接，所以只需
    取首笔的 ``fx_a`` 加上每笔的 ``fx_b`` 即可得到完整的分型序列。

    传入 ``conf_map``（``bi_confirm_map`` 的结果）时，每个点额外带 ``known_dt``
    = 该笔被确认的时间；首笔的起点用首笔自身的确认时间（保守）。

    :param bi_list: czsc 的笔列表，元素为 ``BI`` 对象（含 ``fx_a``/``fx_b``）。
    :param conf_map: ``{fx_b.dt: 确认时间}``，可选。
    :return: 标记点列表，每个点为 dict。
    """
    points = []
    for i, bi in enumerate(bi_list):
        conf = conf_map.get(bi.fx_b.dt) if conf_map else None
        if i == 0:
            points.append({
                'dt': bi.fx_a.dt,
                'fx_mark': _mark_to_str(bi.fx_a.mark),
                'bi': bi.fx_a.fx,
                'fx': bi.fx_a.fx,
                'known_dt': conf if conf is not None else bi.fx_a.dt,
            })
        points.append({
            'dt': bi.fx_b.dt,
            'fx_mark': _mark_to_str(bi.fx_b.mark),
            'bi': bi.fx_b.fx,
            'fx': bi.fx_b.fx,
            'known_dt': conf if conf is not None else bi.fx_b.dt,
        })
    return points


def calc_pivots(
    bi_list: Optional[List] = None,
    points: Optional[List] = None,
    conf_map: Optional[Dict] = None,
    strict: bool = True,
) -> List[Dict]:
    """计算缠论走势中枢（盘整中枢 / pivot）。

    输入既可以是 czsc 的笔列表（``bi_list``），也可以是已经归一化好的
    标记点序列（``points``），二选一。

    :param bi_list: czsc 的笔列表（``BI`` 对象）；为 ``None`` 时使用 ``points``。
    :param points: 标记点序列（与 ``find_zs`` 输入一致）；为 ``None`` 时由
        ``bi_list`` 生成。
    :return: 中枢列表。每个中枢为 dict，含以下字段：

        - ``ZD`` / ``ZG``：中枢下沿 / 上沿（箱底 / 箱顶）
        - ``DD`` / ``GG``：中枢最低点 / 最高点（箱体延伸边界）
        - ``D`` / ``G``：低点序列最大值 / 高点序列最小值（等价于 ZD/ZG）
        - ``start_dt`` / ``end_dt``：中枢起止时间（``end_dt`` 为 ``None``
          表示中枢尚未结束）
        - ``direction``：中枢方向，``up``（上涨中枢）/ ``down``（下跌中枢）
        - ``points``：构成中枢的标记点
        - ``zn``：中枢内的 Z 走势段（含 ``high``/``low``/``mid``）
        - ``third_buy`` / ``third_sell``：若存在三买 / 三卖，则为对应标记点

    注意：``find_zs`` 的 ``GG/DD/G/D`` 是对**整个闭合窗口**取极值，而窗口末点
    ``zs_xd[-1]`` 在 ``end_point=zs_xd[-2]`` 之外——它可能落在 ``end_dt`` 之后
    （例如三买前的冲高），直接使用会把窗口外的价位回填进中枢（未来函数）。
    这里在 ``calc_pivots`` 层做**因果修正**：``GG/DD/G/D`` 只取 ``[start_dt, end_dt]``
    区间内的分型点，保证中枢记录不含区间外（未走到过）的价位。

    ``conf_map``（``bi_confirm_map`` 结果）传入时，每个分型点带 ``known_dt`` =
    该笔被确认的时间，并把中枢成立时刻记为前 4 个分型点 ``known_dt`` 的最大值
    （``pivot['known_dt']``），供 ``causal_pivot_series``/绘图做逐 bar 因果。
    注意：记录中的 ``GG/DD/G/D`` **只按 dt 切窗口**，不叠加确认时间——它们表示
    该中枢区间内真实出现过的极值；确认滞后仅由逐 bar 特征处理。

    :param bi_list: czsc 笔列表（``bi_list`` 与 ``points`` 二选一）。
    :param points: 已归一化的标记点序列。
    :param conf_map: ``{fx_b.dt: 确认时间}``，配合 ``bi_list`` 使用。
    :param strict: 保留参数（记录口径不再使用；逐 bar 因果见 ``causal_pivot_series``）。
    """
    if points is None:
        if bi_list is None:
            raise ValueError('bi_list 与 points 至少提供一个')
        points = bi_list_to_points(bi_list, conf_map=conf_map)

    raw = find_zs(points)
    pivots = []
    for pv in raw:
        sp = pv['start_point']
        ep = pv['end_point']
        pts = pv.get('points', [])
        start_dt = sp['dt']
        if ep is None and pts:
            # 未完成的中枢，用最后一个标记点作为终点便于绘图
            end_dt = pts[-1]['dt']
        else:
            end_dt = ep['dt'] if ep else None

        # 因果修正：GG/DD/G/D 仅取窗口 [start_dt, end_dt] 内的分型点，
        # 丢弃 end_dt 之后（或 start_dt 之前）的窗口点，避免未来函数。
        # 注：这里只按“价格发生时间”切窗口——记录里的 GG/DD 就是该中枢区间内
        # 真实出现过（dt 落界内）的极值；分型点的“确认滞后”只在**逐 bar 特征**
        # （causal_pivot_series）里用 known_dt 处理，不在此处叠加，否则会把中枢
        # 自己落在 end_dt 上的末点排除，导致 DD/GG 被削掉真实极值。
        def _in_win(x):
            return start_dt <= x['dt'] <= end_dt

        in_win = [x for x in pts if _in_win(x)] if (end_dt is not None) else list(pts)
        g_vals = [x['xd'] for x in in_win if x['fx_mark'] == 'g']
        d_vals = [x['xd'] for x in in_win if x['fx_mark'] == 'd']

        # 中枢成立时刻 = 前 4 个分型点确认时间的最大值（strict 时）
        pivot_known = None
        if pts:
            ks = [x.get('known_dt', x['dt']) for x in pts[:4]]
            pivot_known = max(ks) if ks else None

        pivots.append({
            'ZD': pv['ZD'],
            'ZG': pv['ZG'],
            'GG': max(g_vals) if g_vals else pv['GG'],
            'DD': min(d_vals) if d_vals else pv['DD'],
            'G': min(g_vals) if g_vals else pv.get('G', pv['ZG']),
            'D': max(d_vals) if d_vals else pv.get('D', pv['ZD']),
            'start_dt': start_dt,
            'end_dt': end_dt,
            'known_dt': pivot_known,
            'direction': 'up' if sp['fx_mark'] == 'd' else 'down',
            'points': pts,
            'zn': pv.get('zn', []),
            'third_buy': pv.get('third_buy'),
            'third_sell': pv.get('third_sell'),
        })
    return pivots


def _classify_seq(pivots: List[Dict]) -> Dict:
    """在给定中枢序列（前缀）上判断走势类型（``classify_pivots`` 的内核）。"""
    n = len(pivots)
    if n == 0:
        return {'kind': '无中枢', 'n': 0, 'n_up': 0, 'n_down': 0}
    if n == 1:
        return {'kind': '盘整', 'n': 1, 'n_up': int(pivots[0]['direction'] == 'up'),
                'n_down': int(pivots[0]['direction'] == 'down')}

    up = [p for p in pivots if p['direction'] == 'up']
    down = [p for p in pivots if p['direction'] == 'down']

    for i in range(len(up) - 1):
        if up[i]['GG'] < up[i + 1]['DD']:
            return {'kind': '上涨趋势', 'n': n, 'n_up': len(up), 'n_down': len(down)}
    for i in range(len(down) - 1):
        if down[i]['DD'] > down[i + 1]['GG']:
            return {'kind': '下跌趋势', 'n': n, 'n_up': len(up), 'n_down': len(down)}
    return {'kind': '盘整', 'n': n, 'n_up': len(up), 'n_down': len(down)}


def classify_pivots_per(pivots: List[Dict]) -> List[str]:
    """**逐中枢**判断走势类型——走到每个中枢结束时重新判一次（因果）。

    对第 i 个中枢，只用 ``pivots[:i+1]``（截至该中枢、且它已走完）套用同一套
    缠论规则，得到该时刻的走势类型，而不是拿整段序列的结论回填给每一行。
    已成立的同向不重叠中枢对不会被后续中枢"取消"，故标签只会在
    ``盘整 → （上涨/下跌）趋势`` 之间单向切换，不会来回翻转。

    :param pivots: ``calc_pivots`` 返回的中枢列表（按时间升序）。
    :return: 与 ``pivots`` 等长的标签列表（'盘整'/'上涨趋势'/'下跌趋势'）。
    """
    return [_classify_seq(pivots[:i + 1])['kind'] for i in range(len(pivots))]


def classify_pivots(pivots: List[Dict]) -> Dict:
    """判断中枢序列构成的走势类型：盘整 / 上涨趋势 / 下跌趋势。

    缠论定义：

    - **盘整**：仅含一个中枢，或同向中枢之间价格区间存在重叠。
    - **趋势**：至少两个同向中枢，且彼此价格区间不重叠——上涨趋势满足
      前一个中枢的 ``GG < 后一个中枢的 DD``；下跌趋势满足
      前一个中枢的 ``DD > 后一个中枢的 GG``。

    注意：``kind`` 是**整段序列**的结论，别把它当作每个中枢各自的标签；
    逐中枢（截至每个中枢重新判断）的标签见返回的 ``kinds``
    （等价于 ``classify_pivots_per``）。

    :param pivots: ``calc_pivots`` 返回的中枢列表。
    :return: dict，含 ``kind``（整段：'无中枢'/'盘整'/'上涨趋势'/'下跌趋势'）、
        ``kinds``（逐中枢标签列表）、``n``（中枢数量）、
        ``n_up``/``n_down``（上涨/下跌中枢数量）。

    手工构造中枢序列（只给分类需要的字段）：

    >>> def _pv(direction, ZD, ZG, GG, DD):
    ...     return {'direction': direction, 'ZD': ZD, 'ZG': ZG, 'GG': GG, 'DD': DD}
    >>> classify_pivots([])['kind']
    '无中枢'
    >>> classify_pivots([_pv('up', 10, 11, 11.5, 9.5)])['kind']        # 单中枢 ⇒ 盘整
    '盘整'
    >>> classify_pivots([_pv('up', 10, 11, 11.5, 9.5),                 # 前 GG < 后 DD ⇒ 逐级抬升
    ...                  _pv('up', 14, 15, 15.5, 13.5)])['kind']
    '上涨趋势'
    >>> classify_pivots([_pv('down', 14, 15, 15.5, 13.5),              # 前 DD > 后 GG ⇒ 逐级下降
    ...                  _pv('down', 10, 11, 11.5, 9.5)])['kind']
    '下跌趋势'
    >>> classify_pivots([_pv('up', 10, 12, 12.5, 9.5),                 # 同向但区间重叠 ⇒ 盘整
    ...                  _pv('up', 11, 13, 13.5, 10.5)])['kind']
    '盘整'
    """
    out = _classify_seq(pivots)
    out['kinds'] = classify_pivots_per(pivots)
    return out


def pivots_to_df(pivots: List[Dict], kind=None) -> pd.DataFrame:
    """把中枢列表转成 DataFrame，便于检查与导出。

    :param pivots: ``calc_pivots`` 返回的中枢列表。
    :param kind: 走势类型标签。传 **字符串** 时所有行写同一标签（兼容旧行为）；
        传 **列表/序列**（如 ``classify_pivots(pivots)['kinds']``）时逐行写各自
        的标签；``None`` 则该列为空。
    :return: 含 start_dt/end_dt/ZG/ZD/GG/DD/direction 等列的 DataFrame。

    逐中枢标签（列表）只写到对应行，长度不足的行留空：

    >>> pv = [{'direction': 'up', 'start_dt': 't0', 'end_dt': 't1',
    ...        'ZG': 11.0, 'ZD': 10.0, 'GG': 11.5, 'DD': 9.5},
    ...       {'direction': 'down', 'start_dt': 't2', 'end_dt': None,
    ...        'ZG': 13.0, 'ZD': 12.0, 'GG': 13.5, 'DD': 11.5}]
    >>> df = pivots_to_df(pv, kind=['盘整', '下跌趋势'])
    >>> list(df['kind'])
    ['盘整', '下跌趋势']
    >>> list(df['amplitude'])          # ZG - ZD
    [1.0, 1.0]
    >>> pivots_to_df(pv, kind='盘整')['kind'].tolist()      # 字符串 ⇒ 全行同标签
    ['盘整', '盘整']
    """
    per_row = kind is not None and not isinstance(kind, str)
    kinds = list(kind) if per_row else None
    rows = []
    for i, p in enumerate(pivots):
        if kinds is not None:
            k = kinds[i] if i < len(kinds) else None
        else:
            k = kind
        rows.append({
            'index': i,
            'kind': k,
            'direction': p['direction'],
            'start_dt': p['start_dt'],
            'end_dt': p['end_dt'],
            'ZG': p['ZG'],
            'ZD': p['ZD'],
            'GG': p['GG'],
            'DD': p['DD'],
            'amplitude': p['ZG'] - p['ZD'],
        })
    return pd.DataFrame(rows)


def causal_pivot_series(pivots: List[Dict], dts, strict: bool = True) -> tuple:
    """把中枢字段展开为与 ``dts`` 对齐的**因果（无未来函数）**特征列。

    与 ``attach_pivot_features`` 直接铺满区间的口径不同，这里只使用截至每根 K 线
    已经发生的信息：

    - 五个字段**同起止**：都从中枢成立 bar 起填充到 ``end_dt``，之前一律 NaN
      （未完成中枢右端取最后一根 K 线）；
    - ``ZG`` / ``ZD`` / ``direction``：成立时为常量；
    - ``GG`` / ``DD``：按"该中枢截至当前已看到的极值"逐 bar 滚动，杜绝回填终值/未来价。

    可见时刻取"分型点 ``dt``"还是"该笔的确认时间 ``known_dt``"由 ``strict`` 决定：

    - ``strict=True``（默认，配合 ``bi_confirm_map`` 生成的 ``known_dt``）：用
      ``known_dt``——一个分型点要到其所属笔被 czsc 确认的那一刻才可用于计算，
      中枢成立 bar 也顺延为前 4 点 ``known_dt`` 的最大值；
    - ``strict=False``：退回按分型点 ``dt`` 计（无 ``known_dt`` 时自动等价于此）。

    :param pivots: ``calc_pivots`` 返回的中枢列表（需含 ``points`` 字段）。
    :param dts: 与特征行对齐、升序的时间序列（DatetimeIndex / Series / ndarray）。
    :param strict: 是否按分型点确认时间（``known_dt``）计算（默认 True）。
    :return: ``(zd, zg, gg, dd, direction)`` 五条 numpy 数组（direction 为 object）。

    因果性：成立 bar 之前一律 NaN，而不是把整段终值铺满区间：

    >>> import pandas as pd, numpy as np
    >>> dts = pd.date_range('2026-01-01', periods=6, freq='h')
    >>> pv = [{'ZD': 10.0, 'ZG': 11.0, 'direction': 'up',
    ...        'start_dt': dts[1], 'end_dt': dts[4],
    ...        'points': [{'dt': dts[1], 'xd': 11.0, 'fx_mark': 'g'},
    ...                   {'dt': dts[1], 'xd': 10.0, 'fx_mark': 'd'},
    ...                   {'dt': dts[2], 'xd': 11.0, 'fx_mark': 'g'},
    ...                   {'dt': dts[2], 'xd': 10.0, 'fx_mark': 'd'}]}]
    >>> zd, zg, gg, dd, dr = causal_pivot_series(pv, dts)
    >>> np.isnan(zd[:2]).tolist()          # 中枢成立之前是 NaN
    [True, True]
    >>> zd[2:5].tolist()                   # 成立后铺到 end_dt（含端点）为止
    [10.0, 10.0, 10.0]
    >>> dr[2:5].tolist()
    ['up', 'up', 'up']
    >>> bool(np.isnan(zd[5]))              # end_dt 之后不再铺 —— 不到未来去
    True
    """
    dti = dts if isinstance(dts, pd.DatetimeIndex) \
        else pd.DatetimeIndex(np.asarray(dts).reshape(-1))
    if dti.tz is not None:
        dti = dti.tz_convert(None)
    dt_ns = dti.astype('datetime64[ns]').astype(np.int64)

    n = len(dt_ns)
    zd = np.full(n, np.nan)
    zg = np.full(n, np.nan)
    gg = np.full(n, np.nan)
    dd = np.full(n, np.nan)
    direction = np.full(n, None, dtype=object)
    if n == 0:
        return zd, zg, gg, dd, direction
    last_ns = dt_ns[-1]

    def _kt(x):
        """分型点的可见时刻：strict 取 known_dt，否则取 dt（价格发生时刻）。"""
        v = x.get('known_dt') if strict else None
        return v if v is not None else x['dt']

    for p in pivots:
        pts = p.get('points') or []
        if not pts:
            continue
        start_ns = pd.Timestamp(p['start_dt']).value
        end_ns = pd.Timestamp(p['end_dt']).value if p['end_dt'] is not None else last_ns
        lo = int(np.searchsorted(dt_ns, start_ns, side='left'))
        hi = int(np.searchsorted(dt_ns, end_ns, side='right'))
        if hi <= lo:
            continue

        # 中枢成立 bar：前 4 个分型点的可见时刻取最大
        head = pts[:4] if len(pts) >= 4 else pts
        conf_ns = max(pd.Timestamp(_kt(x)).value for x in head)
        conf_ns = max(conf_ns, start_ns)
        clo = min(max(int(np.searchsorted(dt_ns, conf_ns, side='left')), lo), hi)
        zd[clo:hi] = p['ZD']
        zg[clo:hi] = p['ZG']
        direction[clo:hi] = p['direction']

        # GG/DD：与 ZG/ZD 同起止，逐 bar 滚动极值——只取可见时刻 <= t 的分型点
        for mark, out, is_max in (('g', gg, True), ('d', dd, False)):
            ps = sorted(
                ((pd.Timestamp(_kt(x)).value, float(x['xd']))
                 for x in pts if x['fx_mark'] == mark),
                key=lambda t: t[0])
            if not ps:
                continue
            p_ns = np.array([t[0] for t in ps], dtype=np.int64)
            p_xd = np.array([t[1] for t in ps], dtype=np.float64)
            cum = np.maximum.accumulate(p_xd) if is_max else np.minimum.accumulate(p_xd)
            pos = np.searchsorted(p_ns, dt_ns[clo:hi], side='right')
            vals = np.where(pos > 0, cum[np.maximum(pos - 1, 0)], np.nan)
            out[clo:hi] = vals
    return zd, zg, gg, dd, direction


def _causal_step_segments(pts, mark, start_dt, end_dt, is_max, strict: bool = True):
    """构造某中枢 GG（``mark='g'``）/ DD（``mark='d'``）的因果台阶折线断点。

    只用可见时刻不晚于当前的分型点，故折线只在新分型点刷新极值时抬升/下降；
    ``start_dt`` 处的起始水平取"截至 ``start_dt`` 已可见分型点"的极值。
    ``strict=True`` 时可见时刻取 ``known_dt``（笔确认时间），否则取 ``dt``。
    :return: ``[(dt, value), ...]``，按时间升序，落在 ``[start_dt, end_dt]``。
    """
    def _kt(x):
        v = x.get('known_dt') if strict else None
        return v if v is not None else x['dt']

    ps = sorted([(_kt(x), float(x['xd'])) for x in pts if x['fx_mark'] == mark],
                key=lambda t: t[0])
    if not ps:
        return []

    breakpoints = []
    cur = None
    for dt, xd in ps:
        new = xd if cur is None else (max(cur, xd) if is_max else min(cur, xd))
        if cur is None or new != cur:
            cur = new
            breakpoints.append((dt, cur))

    init_val = None
    for dt, v in breakpoints:
        if dt <= start_dt:
            init_val = v
        else:
            break

    segs = []
    if init_val is not None:
        segs.append((start_dt, init_val))
    segs.extend([(dt, v) for dt, v in breakpoints if start_dt < dt <= end_dt])
    return segs


def plot_pivots(
    ax,
    pivots: List[Dict],
    x_mapper: Callable,
    x_last: Optional[float] = None,
    box_color: str = 'cyan',
    box_alpha: float = 0.118,
    edge_color: str = 'steelblue',
    line_width: float = 1.2,
    label: bool = True,
    fontsize: int = 8,
    strict: bool = True,
):
    """在 K 线图上绘制中枢箱体（**因果口径，无未来函数**）。

    与 ``causal_pivot_series`` 的特征口径一致：

    - ``ZG/ZD`` 箱体自中枢成立 bar 起才绘制，之前不画；
    - ``GG/DD`` 画成因果**台阶线**——只在新的分型点（可见后）刷新极值时抬升/下降，
      不会一上来就画到整段的极值（更不会画到 ``end_dt`` 之后的价位）。

    ``strict=True``（默认）时按分型点的**确认时间**（``known_dt``，需 ``calc_pivots``
    传入 ``conf_map``）绘制；否则按分型点 ``dt``。

    :param ax: matplotlib 坐标轴（通常是 K 线主图轴 ``ax1``）。
    :param pivots: ``calc_pivots`` 返回的中枢列表。
    :param x_mapper: 将 datetime 映射为 x 坐标的可调用对象，例如 notebook 里的
        ``dt_to_xcoord``。
    :param x_last: 最后一个 K 线对应的 x 坐标；用于给未完成中枢（``end_dt``
        为 ``None``）补齐右边界。为 ``None`` 时回退到 ``start_dt`` 对应的坐标。
    :param box_color / box_alpha / edge_color / line_width: 箱体样式。
    :param label: 是否在箱体右侧标注 ZG/ZD 价位。
    :param fontsize: 标注字号。
    :param strict: 是否按分型点确认时间（``known_dt``）绘制（默认 True）。
    """
    import matplotlib.patches as mpatches

    def _kt(x):
        v = x.get('known_dt') if strict else None
        return v if v is not None else x['dt']

    for p in pivots:
        pts = p.get('points') or []
        start = p['start_dt']
        end = p['end_dt'] if p['end_dt'] is not None else None
        end_eff = end if end is not None else start
        # 中枢成立 bar：前 4 个分型点可见时刻的最大值（strict=确认时间）
        if len(pts) >= 4:
            conf = max(_kt(x) for x in pts[:4])
        else:
            conf = start
        if conf > end_eff:
            # 该中枢在其自身区间走完之前都不可知（strict），与特征口径一致：不绘制
            continue
        x0_conf = x_mapper(conf)

        if end is not None:
            x1 = x_mapper(end_eff)
        else:
            x1 = x_last if x_last is not None else x0_conf
        if x1 <= x0_conf:
            x1 = x0_conf + 1

        # 中枢箱体 [ZD, ZG]（自确认 bar 起）
        rect = mpatches.Rectangle(
            (x0_conf, p['ZD']), x1 - x0_conf, p['ZG'] - p['ZD'],
            facecolor=box_color, edgecolor=edge_color,
            linewidth=line_width, alpha=box_alpha, zorder=2,
        )
        ax.add_patch(rect)

        # GG / DD：因果台阶线（只在新分型点刷新极值时抬升/下降）
        for mark, value, is_max in (('g', p['GG'], True), ('d', p['DD'], False)):
            segs = _causal_step_segments(pts, mark, conf, end_eff, is_max, strict=strict)
            if not segs:
                # 退化：无分型点信息，回退为水平虚线
                ax.hlines(value, x0_conf, x1, color=edge_color,
                          linestyle='--', linewidth=line_width, alpha=0.618, zorder=2)
                continue
            xs = [x_mapper(dt) for dt, _ in segs]
            ys = [v for _, v in segs]
            if x1 > xs[-1]:                      # 末端延伸至区间右界
                xs.append(x1)
                ys.append(ys[-1])
            ux, uy = [xs[0]], [ys[0]]            # x 去重（严格递增）
            for xi, yi in zip(xs[1:], ys[1:]):
                if xi <= ux[-1]:
                    uy[-1] = yi
                else:
                    ux.append(xi)
                    uy.append(yi)
            ax.plot(ux, uy, drawstyle='steps-post', color=edge_color,
                    linestyle='--', linewidth=line_width, alpha=0.618, zorder=2)

        if label:
            ax.text(
                x1, p['ZG'], f' ZG {p["ZG"]:.2f}',
                color=edge_color, fontsize=fontsize, va='bottom', ha='left',
            )
            ax.text(
                x1, p['ZD'], f' ZD {p["ZD"]:.2f}',
                color=edge_color, fontsize=fontsize, va='top', ha='left',
            )


def plot_pivots_plotly(
    fig,
    pivots: List[Dict],
    x_last=None,
    row: Optional[int] = None,
    col: Optional[int] = None,
    box_color: str = 'rgba(0,180,255,0.13)',
    edge_color: str = 'steelblue',
    line_width: float = 1.4,
    label: bool = True,
    show_ggdd: bool = True,
    strict: bool = True,
):
    """在 Plotly K 线图上绘制中枢箱体（矩形 + GG/DD 因果台阶 + ZG/ZD 标注）。

    与 ``plot_pivots`` / ``causal_pivot_series`` 同口径（无未来函数）：

    - ``ZG/ZD`` 箱体自中枢成立 bar 起才绘制；
    - ``GG/DD`` 画成分段水平线 + 竖连接符组成的**因果台阶**——只在新分型点（可见后）
      刷新极值时抬升/下降，不会一上来就画到整段（更不会画到 ``end_dt`` 之后）的价位。

    ``strict=True``（默认）时按分型点**确认时间**（``known_dt``）绘制。

    :param fig: plotly.graph_objects.Figure（通常是含 candlestick 的 figure）。
    :param pivots: ``calc_pivots`` 返回的中枢列表。
    :param x_last: 最后一个 K 线的时间（datetime），用于给未完成中枢（``end_dt``
        为 ``None``）补齐右边界。
    :param row / col: 目标子图位置。为 ``None`` 时作用于单图（非 subplot），
        传入整数时配合 ``plotly.subplots.make_subplots`` 使用。
    :param box_color / edge_color / line_width / label / show_ggdd: 样式开关。
    :param strict: 是否按分型点确认时间（``known_dt``）绘制（默认 True）。
    """
    def _rc(kwargs):
        if row is not None:
            kwargs['row'] = row
        if col is not None:
            kwargs['col'] = col
        return kwargs

    def _kt(x):
        v = x.get('known_dt') if strict else None
        return v if v is not None else x['dt']

    dash_line = dict(color=edge_color, width=1, dash='dash')

    for p in pivots:
        pts = p.get('points') or []
        start = p['start_dt']
        end = p['end_dt'] if p['end_dt'] is not None else x_last
        if end is None:
            end = start
        if end < start:
            end = start
        # 中枢成立 bar：前 4 个分型点可见时刻的最大值（strict=确认时间）
        if len(pts) >= 4:
            conf = max(_kt(x) for x in pts[:4])
        else:
            conf = start
        if conf > end:
            # 该中枢在其自身区间走完之前都不可知（strict），与特征口径一致：不绘制
            continue
        x0 = conf

        # 箱体 [ZD, ZG]（自确认 bar 起）
        fig.add_shape(**_rc(dict(
            type='rect', x0=x0, x1=end, y0=p['ZD'], y1=p['ZG'],
            fillcolor=box_color, line=dict(color=edge_color, width=line_width),
            layer='below',
        )))
        if show_ggdd:
            for mark, value, is_max in (('g', p['GG'], True), ('d', p['DD'], False)):
                segs = _causal_step_segments(pts, mark, conf, end, is_max, strict=strict)
                if not segs:
                    # 退化：无分型点信息，回退为水平虚线
                    fig.add_shape(**_rc(dict(
                        type='line', x0=x0, x1=end, y0=value, y1=value,
                        line=dash_line, layer='below',
                    )))
                    continue
                # 分段水平线 + 竖连接符（模拟 steps-post 台阶）
                for i, (dt, v) in enumerate(segs):
                    xb = segs[i + 1][0] if i + 1 < len(segs) else end
                    if xb > dt:
                        fig.add_shape(**_rc(dict(
                            type='line', x0=dt, x1=xb, y0=v, y1=v,
                            line=dash_line, layer='below',
                        )))
                    if i > 0:                       # 台阶跳变处的竖线
                        v_prev = segs[i - 1][1]
                        if v_prev != v:
                            fig.add_shape(**_rc(dict(
                                type='line', x0=dt, x1=dt, y0=v_prev, y1=v,
                                line=dash_line, layer='below',
                            )))
        if label:
            fig.add_annotation(**_rc(dict(
                x=end, y=p['ZG'], text=f' ZG {p["ZG"]:.2f}',
                showarrow=False, font=dict(color=edge_color, size=10),
                xanchor='left', yanchor='bottom',
            )))
            fig.add_annotation(**_rc(dict(
                x=end, y=p['ZD'], text=f' ZD {p["ZD"]:.2f}',
                showarrow=False, font=dict(color=edge_color, size=10),
                xanchor='left', yanchor='top',
            )))


def attach_pivot_features(
    features: pd.DataFrame,
    symbol: str = 'stock',
    max_bi_count: int = 512,
    strict: bool = True,
) -> pd.DataFrame:
    """把中枢字段（ZD/ZG/GG/DD/direction）作为列附加到特征 DataFrame 上。

    供完整特征流水线调用，把盘整中枢箱体信息沉淀为特征列，便于后续策略/模型
    直接使用。

    **因果口径（无未来函数）**：填充严格只用截至每根 K 线已发生的信息——
    ``ZG/ZD/direction`` 自中枢成立 bar 起才填，之前保持 ``NaN``；
    ``GG/DD`` 为逐 bar 滚动极值，不会把中枢后半段、甚至 ``end_dt`` 之后的终值
    回填到前面的 bar。实现见 ``causal_pivot_series``；若需要后视绘图用的整段箱体，
    请直接用 ``calc_pivots`` 的结果（``plot_pivots``）。

    :param features: 含 OHLC 列（open/high/low/close/volume/amount）的
        DataFrame，索引为 ``(datetime[, code])``。
    :param symbol: 股票代码（写入 RawBar.symbol）。
    :param max_bi_count: czsc 最大笔数。
    :param strict: 是否按分型点**确认时间**（逐 bar 回放 czsc 得到）计算，默认 True；
        False 时退回按分型点 ``dt``（价格发生时刻）。
    :return: 附加了中枢字段的新 DataFrame。
    """
    from czsc import CZSC
    from czsc.objects import RawBar, Freq
    from GolemQ.models.alias import ZEN

    bars = []
    for i, (idx, row) in enumerate(features.iterrows()):
        dt = idx[0] if isinstance(idx, tuple) else idx
        bars.append(RawBar(
            symbol=symbol, dt=dt, id=i, freq=Freq.F60,
            open=row['open'], close=row['close'], high=row['high'],
            low=row['low'], vol=row.get('volume', 0),
            amount=row.get('amount', 0),
        ))

    czsc = CZSC(bars, max_bi_count=max_bi_count, verbose=False)
    conf_map = bi_confirm_map(bars, max_bi_count=max_bi_count) if strict else None
    pivots = calc_pivots(bi_list=czsc.bi_list, conf_map=conf_map, strict=strict)

    cols = (ZEN.PIVOT_ZD, ZEN.PIVOT_ZG, ZEN.PIVOT_GG,
            ZEN.PIVOT_DD, ZEN.PIVOT_DIRECTION)
    missing_cols = [col for col in cols if col not in features.columns]
    if missing_cols:
        # 一次性 reindex 批量补列，避免逐列 insert 造成 DataFrame 高度碎片化（PerformanceWarning）
        features = features.reindex(columns=[*features.columns, *missing_cols])

    dt_index = features.index.get_level_values(level=0)
    zd, zg, gg, dd, direction = causal_pivot_series(pivots, dt_index, strict=strict)

    m = ~np.isnan(zd)
    features.loc[m, ZEN.PIVOT_ZD] = zd[m]
    m = ~np.isnan(zg)
    features.loc[m, ZEN.PIVOT_ZG] = zg[m]
    m = ~np.isnan(gg)
    features.loc[m, ZEN.PIVOT_GG] = gg[m]
    m = ~np.isnan(dd)
    features.loc[m, ZEN.PIVOT_DD] = dd[m]
    m = np.array([v is not None for v in direction])
    features.loc[m, ZEN.PIVOT_DIRECTION] = direction[m]
    return features
