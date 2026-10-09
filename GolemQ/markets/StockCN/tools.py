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


from datetime import (
    datetime as dt,
    timedelta
)


def purge_historical_collections(client) -> list:
    """清理过期的实时集合 —— **纯逻辑，不打印**。

    ⚠️ **唯一的打印处是 `cli/tools.py::purge_mongodb_database`。**
    这里曾经逐日 `print('⏩ 未找到集合: …')`，于是同一条命令**有两处打印、分属两个
    模块**，测试只 patch 得到其中一个（patch 了 `GolemQ.cli.tools.print`，
    `markets/StockCN/tools.py` 这 14 行照样漏到屏上）。现在统一：本函数产出结果，
    CLI 负责说。

    规则：``realtime_YYYY-MM-DD`` 按名字**从 14 天前往回**逐日找，命中就 drop
    并**继续往回**；连续 14 次没命中即停（即「昨天到今天一个都没剩」的自然终止）。

    异常**不再吞**：集合清单拿不到（Mongo 不可达）或 drop 失败都直接抛出，
    由 `cli/tools.py` 的 `[warn] 清理 X 数据时出错: …` 统一报 —— 原来把它记成
    「一次未命中」等于**静默空转 28 轮然后报成功**。

    :param client: 实时库句柄（`StockCN.GOLEMQ_STOCK_CN_REALTIME`）
    :returns: 已 drop 的集合名，**按 drop 顺序**；没删到就是空列表。
    """
    existing = set(client.list_collection_names())   # 一次取全，别在循环里反复问
    dropped = []
    consecutive_misses = 0
    day_offset = 14

    while consecutive_misses < 14:
        target_date = dt.now() - timedelta(days=day_offset)
        collection_name = f"realtime_{target_date.strftime('%Y-%m-%d')}"

        if collection_name in existing:
            client[collection_name].drop()
            dropped.append(collection_name)
            consecutive_misses = 0          # 命中就重置，继续往回走
        else:
            consecutive_misses += 1

        day_offset += 1                     # 始终递增（原先靠 finally 保证）

    return dropped
