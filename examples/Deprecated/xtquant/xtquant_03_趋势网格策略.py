
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

import time
import datetime
import numpy as np
from typing import List, Optional
from xtquant import xtdata
from xtquant.xttrader import XtQuantTrader, XtQuantTraderCallback
from xtquant.xttype import StockAccount
from xtquant import xtconstant
from GolemQ.gateway.xtquant.config import get_xtquant_config


class TrendGridStrategy:
    """趋势网格做多策略"""
    
    def __init__(self, xt_trader: XtQuantTrader, acc: StockAccount):
        self.xt_trader = xt_trader
        self.acc = acc
        self.grid_levels = {}  # 存储每个股票的网格层级
        self.base_prices = {}  # 存储每个股票的基准价格
        
    def is_trend_up(self, stock_code: str) -> bool:
        """
        判断股票是否处于多头趋势
        条件：日线MA5>MA10>MA20>MA30>MA60 或 MA10>MA5>MA20>MA30>MA60
        """
        try:
            # 获取日线数据
            daily_data = xtdata.get_market_data_ex(
                ['close'], [stock_code], period='1d', 
                count=60, dividend_type='front_ratio'
            )
            
            if daily_data is None or len(daily_data) < 60:
                return False
                
            closes = daily_data['close'].values.flatten()
            
            # 计算移动平均线
            ma5 = np.mean(closes[-5:])
            ma10 = np.mean(closes[-10:])
            ma20 = np.mean(closes[-20:])
            ma30 = np.mean(closes[-30:])
            ma60 = np.mean(closes[-60:])
            
            # 判断多头排列条件
            condition1 = (ma5 > ma10 > ma20 > ma30 > ma60)
            condition2 = (ma10 > ma5 > ma20 > ma30 > ma60)
            
            return condition1 or condition2
            
        except Exception as e:
            print(f"判断趋势时出错 {stock_code}: {e}")
            return False
    
    def get_15min_ma30(self, stock_code: str) -> Optional[float]:
        """
        获取15分钟线的MA30
        """
        try:
            # 获取15分钟数据
            min15_data = xtdata.get_market_data_ex(
                ['close'], [stock_code], period='15m', 
                count=30, dividend_type='front_ratio'
            )
            
            if min15_data is None or len(min15_data) < 30:
                return None
                
            closes = min15_data['close'].values.flatten()
            return np.mean(closes[-30:])
            
        except Exception as e:
            print(f"获取15分钟MA30时出错 {stock_code}: {e}")
            return None
    
    def get_current_price(self, stock_code: str) -> Optional[float]:
        """
        获取当前价格
        """
        try:
            tick_data = xtdata.get_full_tick([stock_code])
            if stock_code in tick_data:
                return tick_data[stock_code]['lastPrice']
            return None
        except Exception as e:
            print(f"获取当前价格时出错 {stock_code}: {e}")
            return None
    
    def initialize_grid(self, stock_code: str, current_price: float):
        """
        初始化网格
        """
        if stock_code not in self.grid_levels:
            self.grid_levels[stock_code] = 0
            self.base_prices[stock_code] = current_price
            print(f"初始化 {stock_code} 网格，基准价格: {current_price}")
    
    def calculate_grid_level(self, stock_code: str, current_price: float) -> int:
        """
        计算当前网格层级
        """
        if stock_code not in self.base_prices:
            return 0
            
        base_price = self.base_prices[stock_code]
        price_change_percent = (current_price - base_price) / base_price * 100
        
        # 每2%为一个网格
        grid_size = 2.0
        grid_level = int(price_change_percent / grid_size)
        
        return grid_level
    
    def should_buy(self, stock_code: str) -> bool:
        """
        判断是否应该加仓
        条件：跌破一定网格并且低于15分钟MA30
        """
        current_price = self.get_current_price(stock_code)
        ma30_15min = self.get_15min_ma30(stock_code)
        
        if current_price is None or ma30_15min is None:
            return False
            
        # 初始化网格
        self.initialize_grid(stock_code, current_price)
        
        # 计算当前网格层级
        current_grid = self.calculate_grid_level(stock_code, current_price)
        
        # 加仓条件：当前价格低于15分钟MA30，且网格层级为负（跌破基准价）
        if current_price < ma30_15min and current_grid < -1:
            print(f"{stock_code} 满足加仓条件: 价格{current_price} < MA30{ma30_15min}, 网格层级{current_grid}")
            return True
            
        return False
    
    def should_sell(self, stock_code: str) -> bool:
        """
        判断是否应该减仓
        条件：高于一定网格数并且高于15分钟MA30
        """
        current_price = self.get_current_price(stock_code)
        ma30_15min = self.get_15min_ma30(stock_code)
        
        if current_price is None or ma30_15min is None:
            return False
            
        # 计算当前网格层级
        current_grid = self.calculate_grid_level(stock_code, current_price)
        
        # 减仓条件：当前价格高于15分钟MA30，且网格层级为正（高于基准价）
        if current_price > ma30_15min and current_grid > 2:
            print(f"{stock_code} 满足减仓条件: 价格{current_price} > MA30{ma30_15min}, 网格层级{current_grid}")
            return True
            
        return False
    
    def get_position_quantity(self, stock_code: str) -> int:
        """
        获取持仓数量
        """
        try:
            positions = self.xt_trader.query_stock_positions(self.acc)
            for pos in positions:
                if pos.stock_code == stock_code:
                    return pos.volume
            return 0
        except Exception as e:
            print(f"获取持仓数量时出错 {stock_code}: {e}")
            return 0
    
    def execute_buy(self, stock_code: str):
        """
        执行买入操作
        """
        try:
            current_price = self.get_current_price(stock_code)
            if current_price is None:
                return
                
            # 获取可用资金
            asset = self.xt_trader.query_stock_asset(self.acc)
            if asset.cash < 1000:  # 最小交易金额
                print(f"资金不足，无法买入 {stock_code}")
                return
                
            # 计算买入数量（使用可用资金的20%）
            buy_amount = asset.cash * 0.2
            buy_volume = int(buy_amount // (current_price * 100)) * 100  # 按手数取整
            
            if buy_volume <= 0:
                return
                
            # 执行买入
            seq = self.xt_trader.order_stock_async(
                self.acc, stock_code, xtconstant.STOCK_BUY, buy_volume,
                xtconstant.LATEST_PRICE, 0, f'趋势网格加仓 {stock_code}', '买入'
            )
            print(f"{datetime.datetime.now()} 买入 {stock_code} {buy_volume}股，订单号: {seq}")
            
        except Exception as e:
            print(f"执行买入操作时出错 {stock_code}: {e}")
    
    def execute_sell(self, stock_code: str):
        """
        执行卖出操作（减仓但不清仓）
        """
        try:
            current_quantity = self.get_position_quantity(stock_code)
            if current_quantity <= 0:
                return
                
            # 减仓数量为当前持仓的25%
            sell_volume = int(current_quantity * 0.25)
            if sell_volume <= 0:
                return
                
            # 执行卖出
            seq = self.xt_trader.order_stock_async(
                self.acc, stock_code, xtconstant.STOCK_SELL, sell_volume,
                xtconstant.LATEST_PRICE, 0, f'趋势网格减仓 {stock_code}', '卖出'
            )
            print(f"{datetime.datetime.now()} 卖出 {stock_code} {sell_volume}股，订单号: {seq}")
            
        except Exception as e:
            print(f"执行卖出操作时出错 {stock_code}: {e}")
    
    def run_strategy(self, stock_codes: List[str], check_interval: int = 60):
        """
        运行策略主循环
        """
        print(f"开始运行趋势网格策略，监控股票: {stock_codes}")
        
        while True:
            try:
                for stock_code in stock_codes:
                    # 检查趋势条件
                    if not self.is_trend_up(stock_code):
                        print(f"{stock_code} 不满足多头趋势条件，跳过")
                        continue
                    
                    # 检查加仓条件
                    if self.should_buy(stock_code):
                        self.execute_buy(stock_code)
                    
                    # 检查减仓条件
                    if self.should_sell(stock_code):
                        self.execute_sell(stock_code)
                
                # 等待下一次检查
                time.sleep(check_interval)
                
            except KeyboardInterrupt:
                print("用户中断策略运行")
                break
            except Exception as e:
                print(f"策略运行出错: {e}")
                time.sleep(check_interval)


class MyXtQuantTraderCallback(XtQuantTraderCallback):
    """交易回调类"""
    
    def on_disconnected(self):
        """连接断开回调"""
        print(datetime.datetime.now(), '连接断开回调')

    def on_stock_order(self, order):
        """委托回报推送"""
        print(datetime.datetime.now(), '委托回调', order.order_remark)

    def on_stock_trade(self, trade):
        """成交变动推送"""
        print(datetime.datetime.now(), '成交回调', trade.order_remark)

    def on_order_error(self, order_error):
        """委托失败推送"""
        print(f"委托报错回调 {order_error.order_remark} {order_error.error_msg}")

    def on_cancel_error(self, cancel_error):
        """撤单失败推送"""
        print(datetime.datetime.now(), '撤单失败回调')

    def on_order_stock_async_response(self, response):
        """异步下单回报推送"""
        print(f"异步委托回调 {response.order_remark}")

    def on_cancel_order_stock_async_response(self, response):
        """异步撤单回报推送"""
        print(datetime.datetime.now(), '异步撤单回调')

    def on_account_status(self, status):
        """账户状态回调"""
        print(datetime.datetime.now(), '账户状态回调')


if __name__ == '__main__':
    print("开始运行趋势网格做多策略")
    
    # 获取 XTQuant 配置
    xtquant_config = get_xtquant_config()
    min_path = xtquant_config['min_path']
    account = xtquant_config['account']
    
    # 生成session id
    session_id = int(time.time())
    xt_trader = XtQuantTrader(min_path, session_id)
    
    # 创建证券账号对象
    acc = StockAccount(account, 'STOCK')
    
    # 创建交易回调类对象
    callback = MyXtQuantTraderCallback()
    xt_trader.register_callback(callback)
    
    # 启动交易线程
    xt_trader.start()
    
    # 建立交易连接
    connect_result = xt_trader.connect()
    print('交易连接结果:', connect_result)
    
    # 订阅交易回调
    subscribe_result = xt_trader.subscribe(acc)
    print('订阅结果:', subscribe_result)
    
    # 创建策略实例
    strategy = TrendGridStrategy(xt_trader, acc)
    
    # 要监控的股票代码列表（示例）
    stock_list = [
        "000001.SZ",  # 平安银行
        "600036.SH",  # 招商银行
        "601318.SH",  # 中国平安
        # 添加更多符合条件的股票
    ]
    
    # 运行策略
    try:
        strategy.run_strategy(stock_list, check_interval=300)  # 每5分钟检查一次
    except KeyboardInterrupt:
        print("策略已停止")
    finally:
        xt_trader.stop()
