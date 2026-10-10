# coding: utf-8
"""缠论中枢识别的**算法核心**（`find_zs`）。

出处
====

取自旧树 ``GolemQ_old/czsc/analyze.py``（原算法作者 zengbin93，MIT）。
按「只搬 czsc 核心代码」的口径，**只取这一个函数** ——
同文件的 `KlineAnalyze`、`signals.py`、`cobra/`、`data/`、`utils/` 一律不带。

单独成文件是为了把**算法**与**我们的编排层**分开：

* 本文件的算法改一个字就改变了中枢识别结果 —— 应当尽量不动；
* ``GolemQ/analysis/pivot.py`` 是编排、因果修正与绘图，会持续演进。

⚠️ 不要去 ``from czsc.analyze import find_zs``：装着的 czsc 0.7.10 里
**没有这个名字**（实测 `ImportError`），且 0.7.x 全包扫描也没有任何中枢(ZS)
实现（`CZSC` 实例只有 ``bi_list``/``finished_bis``，无 ``zs_list``）——
中枢这一层在 czsc 0.7.x 里是空缺的，只能由本模块提供。

⚠️ 本函数**就地修改入参** ``points``（给每个元素补 ``xd`` 键）。
``pivot.bi_list_to_points`` 每次返回新 dict，故现有调用路径无碍；
若要复用同一份 points 列表，请先自己复制。
"""

__all__ = ['find_zs']


def find_zs(points):
    """输入笔或线段标记点，输出中枢识别结果。

    算法（缠中说禅走势中枢）：窗口累积到 5 个标记点后，用**前 4 个**算重叠区间
    ``ZG = min(高点)`` / ``ZD = max(低点)``；``ZG > ZD`` 即有重叠，构成中枢。
    此后：

    * 若新点是**底分型且价格高于 ZG** ⇒ 线段在中枢上方结束，**三买**，中枢闭合；
    * 若新点是**顶分型且价格低于 ZD** ⇒ 线段在中枢下方结束，**三卖**，中枢闭合；
    * 否则并入窗口，中枢继续延伸（GG/DD 随之扩张）。

    窗口重叠不成立（``ZG <= ZD``）时，把最老的点挤出、继续滑动寻找 ——
    这就是「从每个位置尝试成枢」。

    :param points: 标记点序列，每个点为 dict，须含 ``dt`` / ``fx_mark``
        （``'d'`` 底分型 / ``'g'`` 顶分型）与 ``bi``（该点的价格）。
        ⚠️ **会被就地修改** —— 每个元素补一个 ``xd`` 键（= ``bi`` 的值）。
    :return: 中枢列表。每个中枢为 dict，含 ``ZD``/``ZG``/``G``/``GG``/``D``/``DD``、
        ``start_point`` / ``end_point``（``end_point`` 为 ``None`` 表示未完成）、
        ``zn``（中枢内的 Z 走势段，含 ``high``/``low``/``mid``）、
        ``points``（构成窗口的全部标记点），以及可选的 ``third_buy`` / ``third_sell``。
        点数不足 5 时返回**空列表** —— 不抛异常。

    合成点序列（价格即 ``bi``；期望值由本实现实测得出，不是推算的）：

    >>> pts = [{'dt': i, 'fx_mark': m, 'bi': p} for i, (m, p) in enumerate([
    ...     ('d', 10.0), ('g', 12.0), ('d', 10.5), ('g', 11.8), ('d', 10.2),
    ...     ('g', 11.5), ('d', 10.1), ('g', 11.2), ('d', 10.0),
    ... ])]
    >>> zs = find_zs(pts)
    >>> len(zs)
    1
    >>> round(zs[0]['ZD'], 2), round(zs[0]['ZG'], 2)      # 前 4 点：max(10.0,10.5), min(12.0,11.8)
    (10.5, 11.8)
    >>> round(zs[0]['GG'], 2), round(zs[0]['DD'], 2)      # 全窗口极值
    (12.0, 10.0)
    >>> zs[0]['end_point'] is None                        # 未出现三买/三卖 ⇒ 仍在延伸
    True

    点数不足 5 —— 返回空列表而不是报错：

    >>> find_zs([{'dt': 0, 'fx_mark': 'd', 'bi': 1.0}])
    []
    """
    if len(points) < 5:
        return []

    # 当输入为笔的标记点时，新增 xd 值
    for j, x in enumerate(points):
        if x.get("bi", 0):
            points[j]['xd'] = x["bi"]

    def __get_zn(zn_points_):
        """把与中枢方向一致的次级别走势类型称为Z走势段，按中枢中的时间顺序，
        分别记为Zn等，而相应的高、低点分别记为gn、dn"""
        if len(zn_points_) % 2 != 0:
            zn_points_ = zn_points_[:-1]

        if zn_points_[0]['fx_mark'] == "d":
            z_direction = "up"
        else:
            z_direction = "down"

        zn = []
        for i in range(0, len(zn_points_), 2):
            zn_ = {
                "start_dt": zn_points_[i]['dt'],
                "end_dt": zn_points_[i + 1]['dt'],
                "high": max(zn_points_[i]['xd'], zn_points_[i + 1]['xd']),
                "low": min(zn_points_[i]['xd'], zn_points_[i + 1]['xd']),
                "direction": z_direction
            }
            zn_['mid'] = zn_['low'] + (zn_['high'] - zn_['low']) / 2
            zn.append(zn_)
        return zn

    k_xd = points
    k_zs = []
    zs_xd = []

    for i in range(len(k_xd)):
        if len(zs_xd) < 5:
            zs_xd.append(k_xd[i])
            continue
        xd_p = k_xd[i]
        zs_d = max([x['xd'] for x in zs_xd[:4] if x['fx_mark'] == 'd'])
        zs_g = min([x['xd'] for x in zs_xd[:4] if x['fx_mark'] == 'g'])
        if zs_g <= zs_d:
            zs_xd.append(k_xd[i])
            zs_xd.pop(0)
            continue

        # 定义四个指标,GG=max(gn),G=min(gn),D=max(dn),DD=min(dn)，n遍历中枢中所有Zn。
        # 定义ZG=min(g1、g2), ZD=max(d1、d2)，显然，[ZD，ZG]就是缠中说禅走势中枢的区间
        if xd_p['fx_mark'] == "d" and xd_p['xd'] > zs_g:
            zn_points = zs_xd[3:]
            # 线段在中枢上方结束，形成三买
            k_zs.append({
                'ZD': zs_d,
                "ZG": zs_g,
                'G': min([x['xd'] for x in zs_xd if x['fx_mark'] == 'g']),
                'GG': max([x['xd'] for x in zs_xd if x['fx_mark'] == 'g']),
                'D': max([x['xd'] for x in zs_xd if x['fx_mark'] == 'd']),
                'DD': min([x['xd'] for x in zs_xd if x['fx_mark'] == 'd']),
                'start_point': zs_xd[1],
                'end_point': zs_xd[-2],
                "zn": __get_zn(zn_points),
                "points": zs_xd,
                "third_buy": xd_p
            })
            zs_xd = []
        elif xd_p['fx_mark'] == "g" and xd_p['xd'] < zs_d:
            zn_points = zs_xd[3:]
            # 线段在中枢下方结束，形成三卖
            k_zs.append({
                'ZD': zs_d,
                "ZG": zs_g,
                'G': min([x['xd'] for x in zs_xd if x['fx_mark'] == 'g']),
                'GG': max([x['xd'] for x in zs_xd if x['fx_mark'] == 'g']),
                'D': max([x['xd'] for x in zs_xd if x['fx_mark'] == 'd']),
                'DD': min([x['xd'] for x in zs_xd if x['fx_mark'] == 'd']),
                'start_point': zs_xd[1],
                'end_point': zs_xd[-2],
                "points": zs_xd,
                "zn": __get_zn(zn_points),
                "third_sell": xd_p
            })
            zs_xd = []
        else:
            zs_xd.append(xd_p)

    if len(zs_xd) >= 5:
        zs_d = max([x['xd'] for x in zs_xd[:4] if x['fx_mark'] == 'd'])
        zs_g = min([x['xd'] for x in zs_xd[:4] if x['fx_mark'] == 'g'])
        if zs_g > zs_d:
            zn_points = zs_xd[3:]
            k_zs.append({
                'ZD': zs_d,
                "ZG": zs_g,
                'G': min([x['xd'] for x in zs_xd if x['fx_mark'] == 'g']),
                'GG': max([x['xd'] for x in zs_xd if x['fx_mark'] == 'g']),
                'D': max([x['xd'] for x in zs_xd if x['fx_mark'] == 'd']),
                'DD': min([x['xd'] for x in zs_xd if x['fx_mark'] == 'd']),
                'start_point': zs_xd[1],
                'end_point': None,
                "zn": __get_zn(zn_points),
                "points": zs_xd,
            })
    return k_zs
