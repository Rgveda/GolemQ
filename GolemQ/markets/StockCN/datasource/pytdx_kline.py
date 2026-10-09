# coding:utf-8
#
# The MIT License (MIT)
#
# Copyright (c) 2020-2025 azai/GolemQ(uant)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#

"""pytdx 的 **K 线取数**层：只管取 bar / xdxr，不碰数据库、不管文档形态。

与 :mod:`pytdx_source` 的分工（**不要**把连接逻辑复制到这里）
==========================================================
* `TdxSource` 管**连接**：换服、探活、`new_api()`（每 task 一条新连接）。
* 本模块管**分页取数**：所有函数的第一个参数都是 `api`（鸭子对象），
  调用方从 `TdxSource.new_api()` 拿。**正因为连接是入参**，
  单测能塞一个 `FakeApi` 真跑「分页 / 短页即止 / 连接坏掉后重连」三条路径。

PITFALLS P3b：`None` 与 `[]` 必须分开
====================================
pytdx **一次失败调用会毒死整条连接** —— 之后同一条连接上所有调用都**静默返回
`None`**（不抛异常）。所以本模块把语义定死：

* 返回 **`list`**（含空列表、短页）：**调用成功**；空 = 已经翻到头了。
* 返回 **`None`**：**连接已废** → 调用方必须换一条新连接重试，绝不能用旧连接接着翻。

把空列表当成失败、或把 `None` 当成"没有数据"，是这条路上最容易犯的两个反向错误。
"""
from __future__ import annotations

from ..kline_doc import PAGE

__all__ = ['MAX_RETRY', 'bars_page', 'bars_paged', 'probe_market_bars', 'xdxr_rows']

#: 连接废掉后的最大重连次数（每次都是**新连接**）。
MAX_RETRY = 3


def bars_page(api, market, code, category, offset, count=PAGE, is_index=False):
    """取**一页** bar。

    :param offset: pytdx 的 `start` —— **从最新往旧的偏移**，`0` = 最新一根
    :param is_index: 用 `get_index_bars` 还是 `get_security_bars`。
        指数要走前者（实测 `get_index_bars(9, 1, '000001', 0, 5)` 可取上证指数日线）；
        股票/ETF 走后者 —— 存量 `etf_day` 里**没有** `up_count/down_count`，
        所以 ETF 不要用 `get_index_bars`（那会多出两个字段，与存量不一致）。
    :returns: `list`（可空）= 调用成功；`None` = **连接已废**（P3b）
    """
    fn = api.get_index_bars if is_index else api.get_security_bars
    return fn(category, market, code, int(offset), int(count))


def bars_paged(api, market, code, category, offsets, *, is_index=False, page=PAGE,
               retry=MAX_RETRY, reconnect=None, verbose=False, note=None):
    """按 `offsets`（**升序**）翻页取 bar，**短页即止**。

    ⚠️ 不要照抄 QUANTAXIS 的 `(int(lens/800)-i)*800` 降序写法：它要先估准总根数，
    估小了会**静默丢掉最老的一页**。这里升序翻、翻到短页/空页自然停，
    `offsets` 只是「最多翻到哪」的上限 —— 估多估少都不丢数据。

    :param offsets: `kline_doc.page_offsets(total)` 的产物；`[]` = 不发任何请求
    :param reconnect: **零参可调用**，返回一条新连接（`TdxSource.new_api`）。
        给了它，页级失败才会重试 —— P3b 的处置只有「换连接」，没有第二种。
    :returns: ``(bars, status, api)``；`status ∈ {'ok', 'empty', 'aborted'}`；
        `api` 可能是重连后的**新**连接（调用方要用它、并负责 `disconnect`）
    """
    bars_out: list = []
    if not offsets:
        return bars_out, 'empty', api

    for off in offsets:
        bars = bars_page(api, market, code, category, off, page, is_index=is_index)

        if bars is None:
            api, bars = _retry_page(api, market, code, category, off, page, is_index,
                                    retry, reconnect, verbose, note=note)
            if bars is None:
                # 重试耗尽：**整只放弃**，绝不拿半截数据去写库
                # （写了就等于「删了窗口只补半截」，比不写更糟）
                return bars_out, 'aborted', api

        if not bars:                       # 空页 = 已经到头
            break
        bars_out.extend(bars)
        if len(bars) < page:               # 短页 = 已经到头
            break

    return bars_out, ('ok' if bars_out else 'empty'), api


def _retry_page(api, market, code, category, off, page, is_index, retry, reconnect,
                verbose, note=None):
    """连接废掉后就换新连接重试**同一页**。返回 `(api, bars|None)`。

    ⚠️ **这里绝不 `print`。** 本函数跑在**每只 code 的内层** —— 此刻调用方的
    tqdm 进度条**正活着**，而 banner 的表头压在它上面。任何直接写 stdout 都会
    同时打花这两样：banner 靠「光标上移 N 行」重画，多出来的行会让 N 算错。
    所以诊断交给 `note` 回调 —— 调用方把它**缓冲起来、等条关了再吐**。
    """
    if reconnect is None:
        return api, None
    for attempt in range(1, int(retry) + 1):
        try:
            new_api = reconnect()
        except Exception as exc:          # noqa: BLE001 连不上就再试/放弃
            if verbose and note is not None:
                note('[pytdx] {} 重连失败（第 {} 次）：{!r}'.format(code, attempt, exc))
            continue
        try:
            api.disconnect()
        except Exception:                 # noqa: BLE001
            pass
        api = new_api
        bars = bars_page(api, market, code, category, off, page, is_index=is_index)
        if verbose and note is not None:
            note('[pytdx] {} 连接已废，重连后第 {} 次重试本页：{}'.format(
                code, attempt, '成功' if bars is not None else '仍失败'))
        if bars is not None:
            return api, bars
    return api, None


def xdxr_rows(api, market, code):
    """`get_xdxr_info`：该 code 的**全部**除权除息事件。

    :returns: `list`（**空列表是正常的** —— 该票从未除权）；`None` = 连接已废（P3b）
    """
    return api.get_xdxr_info(market, code)


def probe_market_bars(api, code, market, category=9):
    """探「这个 market 号能不能取到 bar」（北交所要探，见 kline_save 的说明）。

    ⚠️ **必须用一条独立连接、用完即弃** —— P3b 的原始毒源就是
    `get_security_list(2, 0)` 返回 `None` 之后整条连接静默失效。

    :returns: True = 取到了 bar；False = 空/None（都不代表"这只票没数据"，
        只代表**这个 market 号在这台服务器上不可用**）
    """
    try:
        bars = bars_page(api, market, code, category, 0, 1)
    except Exception:                     # noqa: BLE001
        return False
    return bool(bars)
