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


def purge_historical_collections(client):
    """清理历史数据集合"""
    consecutive_misses = 0  # 连续失败计数器
    day_offset = 14         # 起始时间偏移
    
    while consecutive_misses < 14:
        try:
            # 计算目标日期
            target_date = dt.now() - timedelta(days=day_offset)
            collection_name = f"realtime_{target_date.strftime('%Y-%m-%d')}"
            
            # 执行删除操作
            if collection_name in client.list_collection_names():
                client[collection_name].drop()
                print(f"✅ 成功删除历史集合: {collection_name}")
                consecutive_misses = 0  # 重置计数器
            else:
                print(f"⏩ 未找到集合: {collection_name}")
                consecutive_misses += 1
                
        except Exception as e:
            print(f"❌ 处理 {collection_name} 时发生异常: {str(e)}")
            consecutive_misses += 1  # 异常视为失败
        finally:
            day_offset += 1  # 确保偏移量始终递增
            