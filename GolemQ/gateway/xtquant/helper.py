# coding:utf-8
#
# The MIT License (MIT)
#
# Copyright (c) 2020-2025 azai/GolemQ
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#

from typing import List
from GolemQ.core.settings import DATABASE as DATABASE_GolemQ
import time


def get_recent_xtquant_order_symbols(days: int = 30, collection=DATABASE_GolemQ.StockCN_xtquant_orders) -> List[str]:
    """
    获取最近一段时间内（默认30天）XTQuant挂单的股票代码列表
    
    Args:
        days: 回溯天数，默认30天
        collection: 数据库集合
        
    Returns:
        股票代码列表（去重）
    """
    try:
        # 创建索引以提高查询性能
        collection.create_index([('source', 1), ('time_stamp', -1)])
        collection.create_index('stock_code')
        
        # 计算days天前的时间戳（秒）
        current_timestamp = int(time.time())
        days_ago = current_timestamp - (days * 86400)  # days * 24 * 60 * 60
        
        # 查询最近days天内的所有订单
        recent_orders = collection.find({
            'source': 'xtquant',
            'time_stamp': {'$gte': days_ago}
        }, {'stock_code': 1})
        
        # 提取唯一的股票代码
        symbols = set()
        for order in recent_orders:
            if 'stock_code' in order:
                symbols.add(order['stock_code'])
        
        symbol_list = list(symbols)
        print(f"从数据库加载了 {len(symbol_list)} 个最近{days}天内挂单的股票代码")
        return symbol_list
        
    except Exception as e:
        print(f"获取最近挂单股票代码时出错: {e}")
        return []
