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

import pandas as pd
from xtquant.xttrader import XtQuantTrader
from xtquant.xttype import StockAccount
from xtquant import xtconstant
from .config import get_xtquant_config
import random
from datetime import (
    datetime as dt,
)


class xtQmtTrader:

    def __init__(
            self,
            path='',
            account='xxxxxx',
            account_type='STOCK',
            is_slippage=True, slippage=0.01) -> None:
        '''
        简化版的qmt_trader方便大家做策略的开发类的继承
        '''
        self.xt_trader = ''
        self.acc = ''
        xtquant_config = get_xtquant_config()
        self.session_id = int(self.random_session_id())

        # 获取XTQuant配置
        self.min_path = xtquant_config['min_path']
        self.account = xtquant_config['account']
        self.account_type = account_type
        if is_slippage is True:
            self.slippage = slippage
        else:
            self.slippage = 0

    def random_session_id(self):
        '''
        随机id
        '''
        session_id = ''
        for i in range(0, 9):
            session_id += str(random.randint(1, 9))
        return session_id

    def connect(self):
        '''
        连接
        path qmt userdata_min是路径
        session_id 账户的标志,随便
        account账户,
        account_type账户内类型
        '''
        print(f'[{dt.now().strftime("%Y-%m-%d %H:%M:%S")}]: 开始链接QMT...')
        # path为mini qmt客户端安装目录下userdata_mini路径
        path = self.min_path

        # session_id为会话编号，策略使用方对于不同的Python策略需要使用不同的会话编号
        session_id = self.session_id
        xt_trader = XtQuantTrader(path, session_id)
        # 创建资金账号为1000000365的证券账号对象
        account = self.account
        account_type = self.account_type
        acc = StockAccount(
            account_id=account,
            account_type=account_type)

        # 启动交易线程
        xt_trader.start()
        # 建立交易连接，返回0表示连接成功
        connect_result = xt_trader.connect()
        if connect_result == 0:
            self.xt_trader = xt_trader
            self.acc = acc
            print(f'[{dt.now().strftime("%Y-%m-%d %H:%M:%S")}]: QMT连接成功!')
        else:
            print(f'[{dt.now().strftime("%Y-%m-%d %H:%M:%S")}]: QMT连接失败!')

    def get_position(self):
        '''
        查询账户所有的持仓
        '''
        positions = self.xt_trader.query_stock_positions(self.acc)
        print("持仓数量:", len(positions))
        data = pd.DataFrame()
        if len(positions) != 0:
            for i in range(len(positions)):
                df = pd.DataFrame()
                df['账号类型'] = [positions[i].account_type]
                df['资金账号'] = [positions[i].account_id]
                df['证券代码'] = [positions[i].stock_code]
                df['证券代码'] = df['证券代码'].apply(lambda x: str(x)[:6])
                df['持仓数量'] = [positions[i].volume]
                df['可用数量'] = [positions[i].can_use_volume]
                df['平均建仓成本'] = [positions[i].open_price]
                df['市值'] = [positions[i].market_value]
                data = pd.concat([data, df], ignore_index=True)
            return data
        else:
            print('没有持股')
            df = pd.DataFrame()
            df['账号类型'] = [None]
            df['资金账号'] = [None]
            df['证券代码'] = [None]
            df['持仓数量'] = [None]
            df['可用数量'] = [None]
            df['平均建仓成本'] = [None]
            df['市值'] = [None]
            return df

    def get_balance(self):
        '''
        返回当前证券账号的资产数据
        '''
        asset = self.xt_trader.query_stock_asset(account=self.acc)
        data_dict = {}
        if asset:
            data_dict['账号类型'] = asset.account_type
            data_dict['资金账户'] = asset.account_id
            data_dict['可用金额'] = asset.cash
            data_dict['冻结金额'] = asset.frozen_cash
            data_dict['持仓市值'] = asset.market_value
            data_dict['总资产'] = asset.total_asset
            return data_dict
        else:
            print('获取失败资金')
            data_dict['账号类型'] = [None]
            data_dict['资金账户'] = [None]
            data_dict['可用金额'] = [None]
            data_dict['冻结金额'] = [None]
            data_dict['持仓市值'] = [None]
            data_dict['总资产'] = [None]
            return data_dict

    def today_trades(self):
        '''
        当日成交
        '''
        trades = self.xt_trader.query_stock_trades(self.acc)
        print("成交数量:", len(trades))
        data = pd.DataFrame()
        if len(trades) != 0:
            for i in range(len(trades)):
                df = pd.DataFrame()
                df['账号类型'] = [trades[i].account_type]
                df['资金账号'] = [trades[i].account_id]
                df['证券代码'] = [trades[i].stock_code]
                df['证券代码'] = df['证券代码'].apply(lambda x: str(x)[:6])
                df['委托类型'] = [trades[i].order_type]
                df['成交编号'] = [trades[i].traded_id]
                df['成交时间'] = [trades[i].traded_time]
                df['成交均价'] = [trades[i].traded_price]
                df['成交数量'] = [trades[i].traded_volume]
                df['成交金额'] = [trades[i].traded_amount]
                df['订单编号'] = [trades[i].order_id]
                df['柜台合同编号'] = [trades[i].order_sysid]
                df['策略名称'] = [trades[i].strategy_name]
                df['委托备注'] = [trades[i].order_remark]
                data = pd.concat([data, df], ignore_index=True)
            data['成交时间'] = pd.to_datetime(data['成交时间'], unit='s')
            return data

    def today_entrusts(self):
        '''
        当日委托
        :param account: 证券账号
        :param cancelable_only: 仅查询可撤委托
        :return: 返回当日所有委托的委托对象组成的list
        '''
        orders = self.xt_trader.query_stock_orders(self.acc)
        print("委托数量", len(orders))
        data = pd.DataFrame()
        if len(orders) != 0:
            for i in range(len(orders)):
                df = pd.DataFrame()
                df['账号类型'] = [orders[i].account_type]
                df['资金账号'] = [orders[i].account_id]
                df['证券代码'] = [orders[i].stock_code]
                df['证券代码'] = df['证券代码'].apply(lambda x: str(x)[:6])
                df['订单编号'] = [orders[i].order_id]
                df['柜台合同编号'] = [orders[i].order_sysid]
                df['报单时间'] = [orders[i].order_time]
                df['委托类型'] = [orders[i].order_type]
                df['委托数量'] = [orders[i].order_volume]
                df['报价类型'] = [orders[i].price_type]
                df['委托价格'] = [orders[i].price]
                df['成交数量'] = [orders[i].traded_volume]
                df['成交均价'] = [orders[i].traded_price]
                df['委托状态'] = [orders[i].order_status]
                df['委托状态描述'] = [orders[i].status_msg]
                df['策略名称'] = [orders[i].strategy_name]
                df['委托备注'] = [orders[i].order_remark]
                data = pd.concat([data, df], ignore_index=True)
            data['报单时间'] = pd.to_datetime(data['报单时间'], unit='s')
            return data
        else:
            print('目前没有委托')
            return data

    def check_stock_is_av_buy(self, stock='600031', price=17.70, amount=10,
                              hold_limit=100000):
        '''
        检查是否可以买入
        '''
        hold_stock = self.get_position()
        try:
            del hold_stock['Unnamed: 0']
        except KeyError:
            pass
        account = self.get_balance()
        try:
            del account['Unnamed: 0']
        except KeyError:
            pass
        # 买入是价值
        value = price * amount
        cash = account['可用金额']
        # 移除未使用的变量
        # frozen_cash = account['冻结金额']
        # market_value = account['持仓市值']
        # total_asset = account['总资产']
        if cash >= value:
            print('允许买入{} 可用现金{}大于买入金额{} 价格{} 数量{}'.format(
                stock, cash, value, price, amount))
            return True
        else:
            print('不允许买入{} 可用现金{}小于买入金额{} 价格{} 数量{}'.format(
                stock, cash, value, price, amount))
            return False

    def check_stock_is_av_sell(self, stock='600031',  amount=10):
        '''
        检查是否可以卖出
        '''
        hold_data = self.get_position()
        try:
            del hold_data['Unnamed: 0']
        except KeyError:
            pass
        account = self.get_balance()
        try:
            del account['Unnamed: 0']
        except KeyError:
            pass

        # 移除未使用的变量
        # cash = account['可用金额']
        # frozen_cash = account['冻结金额']
        # market_value = account['持仓市值']
        # total_asset = account['总资产']
        stock_list = hold_data['证券代码'].tolist()

        if stock in stock_list:
            hold_num = hold_data[hold_data['证券代码'] == stock]['可用余额']
            if hold_num >= amount:
                print('允许卖出：{} 持股{} 卖出{}'.format(stock, hold_num, amount))
                return True
            else:
                print('不允许卖出持股不足：{} 持股{} 卖出{}'.format(stock, hold_num, amount))
                return False
        else:
            print('不允许卖出没有持股：{} 持股{} 卖出{}'.format(stock, 0, amount))
            return False

    def make_buy(self, security='600031.SH', amount=100, price=20,
                 strategy_name='', order_remark=''):
        '''
        单独独立股票买入函数
        '''
        order_type = xtconstant.STOCK_BUY

        if price == 0:
            price_type = xtconstant.LATEST_PRICE
        else:
            price_type = xtconstant.FIX_PRICE

        order_volume = amount
        # 使用指定价下单，接口返回订单编号，后续可以用于撤单操作以及查询委托状态
        if order_volume > 0:
            fix_result_order_id = self.xt_trader.order_stock(
                account=self.acc, stock_code=security, order_type=order_type,
                order_volume=order_volume, price_type=price_type,
                price=price, strategy_name=strategy_name,
                order_remark=order_remark)

            print('交易类型{} 代码{} 价格{} 数量{} 订单编号{}'.format(
                order_type, security, price, order_volume,
                fix_result_order_id))
            return fix_result_order_id
        else:
            print('买入 标的{} 价格{} 委托数量{}小于0有问题'.format(
                security, price, order_volume))

    def make_sell(self, security='600031.SH', amount=100, price=20,
                  strategy_name='', order_remark=''):
        '''
        单独独立股票卖出函数
        '''
        order_type = xtconstant.STOCK_SELL

        if price == 0:
            price_type = xtconstant.LATEST_PRICE
        else:
            price_type = xtconstant.FIX_PRICE

        order_volume = amount
        # 使用指定价下单，接口返回订单编号，后续可以用于撤单操作以及查询委托状态
        if order_volume > 0:
            fix_result_order_id = self.xt_trader.order_stock(
                account=self.acc, stock_code=security, order_type=order_type,
                order_volume=order_volume, price_type=price_type,
                price=price, strategy_name=strategy_name,
                order_remark=order_remark)
            print('交易类型{} 代码{} 价格{} 数量{} 订单编号{}'.format(
                order_type, security, price, order_volume,
                fix_result_order_id))
            return fix_result_order_id
        else:
            print('卖出 标的{} 价格{} 委托数量{}小于0有问题'.format(
                security, price, order_volume))
