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

from xtquant.xttype import StockAccount
from xtquant import xttrader
import random
from GolemQ.gateway.xtquant.config import get_xtquant_config

# 订阅账户
# 获取 XTQuant 配置
xtquant_config = get_xtquant_config()
min_path = xtquant_config['min_path']
account = xtquant_config['account']

# 设置 QMT 交易端的数据路径和会话ID
session_id = int(random.randint(100000, 999999))

# 创建 XtQuantTrader 实例并启动
xt_trader = xttrader.XtQuantTrader(min_path, session_id)
xt_trader.start()

# 连接 QMT 交易端
connect_result = xt_trader.connect()
if connect_result == 0:
    print('连接成功')
else:
    print('连接失败')
    xt_trader.stop()
    exit()

# 设置账户信息
acc = StockAccount(account)

# 订阅账户
res = xt_trader.subscribe(acc)
if res == 0:
    print('订阅成功')
else:
    print('订阅失败', res)

# 持仓查询
# xt_trader为XtQuant API实例对象
positions = xt_trader.query_stock_positions(acc)

print("=" * 100)
print("持仓明细")
print("=" * 100)

# 方式一：简洁表格格式
print(f"{'证券代码':<10} {'持仓数量':<10} {'可用数量':<10} {'成本价':<10} {'市值':<15} {'冻结数量':<10}")
print("-" * 70)

for position in positions:
    print(f"{position.stock_code:<12} {position.volume:<12} {position.can_use_volume:<12} "
          f"{position.avg_price:<10.3f} {position.market_value:<15.2f} {position.frozen_volume:<10}")

print("\n" + "=" * 100)
print("详细持仓信息")
print("=" * 100)

# 方式二：详细输出每个持仓
for i, position in enumerate(positions, 1):
    print(f"\n持仓 #{i}:")
    print(f"  证券代码: {position.stock_code}")
    print(f"  持仓数量: {position.volume}")
    print(f"  可用数量: {position.can_use_volume}")
    print(f"  成本均价: {position.avg_price:.3f}")
    print(f"  开仓价格: {position.open_price:.3f}")
    print(f"  市值: {position.market_value:.2f}")
    print(f"  冻结数量: {position.frozen_volume}")
    print(f"  在途股份: {position.on_road_volume}")
    print(f"  昨夜拥股: {position.yesterday_volume}")
    print(f"  资金账号: {position.account_id}")
    print(f"  账号类型: {position.account_type}")
    print(f"  多空方向: {position.direction}")

# 方式三：汇总统计
total_market_value = sum(pos.market_value for pos in positions)
total_volume = sum(pos.volume for pos in positions)

print("\n" + "=" * 100)
print("持仓汇总")
print("=" * 100)
print(f"总持仓数量: {len(positions)} 只股票")
print(f"总持仓股数: {total_volume} 股")
print(f"总市值: {total_market_value:.2f} 元")