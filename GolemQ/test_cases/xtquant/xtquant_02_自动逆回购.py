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

# 逆回购交易自动交易源码

import time
import datetime
import sys
from xtquant import xtdata
from xtquant.xttrader import XtQuantTrader, XtQuantTraderCallback
from xtquant.xttype import StockAccount
from xtquant import xtconstant
from GolemQ.gateway.xtquant.config import get_xtquant_config


# 创建一个空对象
class A:
    pass


a_instance = A()


# 比较两个股票，得到价格高的，用于卖出。
def price_compare(list2):
    """
    参数为两个股票代码，如'204001.SH'，
    比较后返回最新价格高的品种代码，
    """
    a, b = list2[0], list2[1]
    res = xtdata.get_full_tick(list2)
    print(res)
    if res[a]['lastPrice'] >= res[b]['lastPrice']:
        return a
        print(res[a]['lastPrice'])
    else:
        return b


# 写一个市场1天的逆回购函数
# 获取股票最新价格
def get_last_price(stock_code):
    tick_dict = xtdata.get_full_tick([stock_code])
    return tick_dict[stock_code]['lastPrice']


# 逆回购交易
def reverse_repos(xt_trader, acc, symbol):
    # 查询账号内的资金
    a_instance.asset = xt_trader.query_stock_asset(acc)
    # 查看资金，如果资金大于10000
    if a_instance.asset.cash > 10000:
        # 拿出一半的资金用于逆回购
        a_instance.asset.cash = a_instance.asset.cash / 2
        # 购买数量
        vol = int((a_instance.asset.cash // 1000) * 10)
        print('购买数量:', {vol})
        # 用异步订单卖出，注意逆回购订单类型选择卖出
        seq = xt_trader.order_stock_async(
            acc, symbol, xtconstant.STOCK_SELL, vol,
            xtconstant.LATEST_PRICE, 0, '逆回购', '卖出'
        )
        print(time.strftime("%Y-%m-%d, %H:%M:%S"), f'逆回购订单号：{seq}。卖出数量{vol}')
    else:
        vol = a_instance.asset.cash
        print(time.strftime("%Y-%m-%d, %H:%M:%S"), f'卖出数量不足{vol}')


class MyXtQuantTraderCallback(XtQuantTraderCallback):

    def on_disconnected(self):
        """
        连接断开
        :return:
        """
        print(datetime.datetime.now(), '连接断开回调')

    def on_stock_order(self, order):
        """
        委托回报推送
        :param order: XtOrder对象
        :return:
        """
        print(datetime.datetime.now(), '委托回调', order.order_remark)

    def on_stock_trade(self, trade):
        """
        成交变动推送
        :param trade: XtTrade对象
        :return:
        """
        print(datetime.datetime.now(), '成交回调', trade.order_remark)

    def on_order_error(self, order_error):
        """
        委托失败推送
        :param order_error: XtOrderError 对象
        :return:
        """
        print(f"委托报错回调 {order_error.order_remark} {order_error.error_msg}")

    def on_cancel_error(self, cancel_error):
        """
        撤单失败推送
        :param cancel_error: XtCancelError 对象
        :return:
        """
        print(datetime.datetime.now(), sys._getframe().f_code.co_name)

    def on_order_stock_async_response(self, response):
        """
        异步下单回报推送
        :param response: XtOrderResponse 对象
        :return:
        """
        print(f"异步委托回调 {response.order_remark}")

    def on_cancel_order_stock_async_response(self, response):
        """
        :param response: XtCancelOrderResponse 对象
        :return:
        """
        print(datetime.datetime.now(), sys._getframe().f_code.co_name)

    def on_account_status(self, status):
        """
        :param response: XtAccountStatus 对象
        :return:
        """
        print(datetime.datetime.now(), sys._getframe().f_code.co_name)


if __name__ == '__main__':
    print("开始使用miniQMT交易逆回购")
    
    # 获取 XTQuant 配置
    xtquant_config = get_xtquant_config()
    min_path = xtquant_config['min_path']
    account = xtquant_config['account']
    
    # 生成session id 整数类型 同时运行的策略不能重复
    session_id = int(time.time())
    xt_trader = XtQuantTrader(min_path, session_id)
    # 开启主动请求接口的专用线程 开启后在on_stock_xxx回调函数里调用XtQuantTrader.query_xxx函数不会卡住回调线程，但是查询和推送的数据在时序上会变得不确定
    # 详见: http://docs.thinktrader.net/vip/pages/ee0e9b/#开启主动请求接口的专用线程
    # xt_trader.set_relaxed_response_order_enabled(True)

    # 创建证券账号对象
    acc = StockAccount(account, 'STOCK')
    # 创建交易回调类对象，并声明接收回调
    callback = MyXtQuantTraderCallback()
    xt_trader.register_callback(callback)
    # 启动交易线程
    xt_trader.start()
    # 建立交易连接，返回0表示连接成功
    connect_result = xt_trader.connect()
    print('建立交易连接，返回0表示连接成功', connect_result)
    # 对交易回调进行订阅，订阅后可以收到交易主推，返回0表示订阅成功
    subscribe_result = xt_trader.subscribe(acc)
    print('对交易回调进行订阅，订阅后可以收到交易主推，返回0表示订阅成功', subscribe_result)
    
    # 查询用户可用资金
    a_instance.asset = xt_trader.query_stock_asset(acc)
    print('\n持仓市值：', a_instance.asset.market_value, '\n总资产',
          a_instance.asset.total_asset, '\n可用金额', a_instance.asset.cash)
    # 深圳市场和上海市场的1天逆回购代码
    symbol_list = ["204001.SH", "131810.SZ"]
    # 比较价格
    symbol = price_compare(symbol_list)
    print(symbol)
    # 使用while循环进行交易
    while True:
        # 在2点55到3点半进行交易
        if '15:00:00' <= time.strftime('%H:%M:%S') <= '15:30:00':
            print(symbol)
            reverse_repos(xt_trader, acc, symbol)
            time.sleep(60)
        # 超过3点30就终止程序
        elif time.strftime('%H:%M:%S') > '15:30:00':
            break
        # 其他时间就等待
        else:
            print(time.strftime('%H:%M:%S'), '未到时间，请耐心等候！')
            time.sleep(60)

    # 这一行是注册全推回调函数 包括下单判断 安全起见处于注释状态 确认理解效果后再放开
    # xtdata.subscribe_whole_quote(["SH", "SZ"], callback=f)
    # 阻塞主线程退出
    # xt_trader.run_forever()
    # 如果使用vscode pycharm等本地编辑器 可以进入交互模式 方便调试 （把上一行的run_forever注释掉 否则不会执行到这里）
    # interact()
