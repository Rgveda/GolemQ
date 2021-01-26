# coding:utf-8
#
# The MIT License (MIT)
#
# Copyright (c) 2018-2020 azai/Rgveda mods with yutiansut/QUANTAXIS
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
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as sco

try:
    import talib
except:
    print('PLEASE run "pip install TALIB" to call these modules')
    pass
try:
    import QUANTAXIS as QA
    from QUANTAXIS.QAUtil.QAParameter import ORDER_DIRECTION
    from QUANTAXIS.QAData.QADataStruct import (
        QA_DataStruct_Index_min, 
        QA_DataStruct_Index_day, 
        QA_DataStruct_Stock_day, 
        QA_DataStruct_Stock_min,
        QA_DataStruct_CryptoCurrency_day,
        QA_DataStruct_CryptoCurrency_min,
        )
    from QUANTAXIS.QAIndicator.talib_numpy import *
    from QUANTAXIS.QAUtil.QADate_Adv import (
        QA_util_timestamp_to_str,
        QA_util_datetime_to_Unix_timestamp,
        QA_util_print_timestamp
    )
    from QUANTAXIS.QAUtil.QALogs import (
        QA_util_log_info, 
        QA_util_log_debug, 
        QA_util_log_expection
    )
except:
    print('PLEASE run "pip install QUANTAXIS" before call GolemQ.utils.portfolio_optimization_demo modules')
    pass

from GolemQ.indices.indices import *
from GolemQ.cli.show_number import (
    position,
)
if __name__ == '__main__':
    # 暗色主题
    plt.style.use('Solarize_Light2')

    # 正常显示中文字体
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']

    # 股票代码，如果选股以后：我们假设有这些代码
    codelist = ['002337','603636','600848','002736','002604', 
                '600004','600015','600023','600033','600183',
                '000157']

    # 股票代码，我直接用我的选股程序获取选股列表。
    position_signals = position(portfolio='sharpe_scale_patterns_day',
                        frequency='day',
                        market_type=QA.MARKET_TYPE.STOCK_CN,
                        verbose=False)
    codelist = position_signals.index.get_level_values(level=1).to_list()

    # 获取股票中文名称，只是为了看得方便，交易策略并不需要股票中文名称
    stock_names = QA.QA_fetch_stock_name(codelist)
    codename = [stock_names.at[code, 'name'] for code in codelist]

    noa = len(codelist)
    data_day = QA.QA_fetch_stock_day_adv(codelist,
        start='2014-01-01',
        end='{}'.format(datetime.date.today())).to_qfq()

    # 收益率序列
    rets_jotion = data_day.add_func(kline_returns_func)
    returns = pd.DataFrame(columns=codelist, index=data_day.data.index.get_level_values(level=0).unique())
    for code in codelist:
        returns[code] = rets_jotion.loc[(slice(None), code), :].reset_index(level=[1], drop=True)

    #returns.plot(figsize=(8, 5))

    # 蒙特卡洛模拟权重
    port_returns = []
    port_variance = []
    for p in range(4000):
        weights = np.random.random(noa)
        weights /= np.sum(weights)
        port_returns.append(np.sum(returns.mean() * weights) * 252)
        port_variance.append(np.sqrt(np.dot(weights.T,np.dot(returns.cov() * 252,weights))))

    #无风险利率设定为4%
    risk_free = 0.04

    port_returns = np.array(port_returns)
    port_variance = np.array(port_variance)

    plt.figure(figsize=(8,4))
    plt.scatter(port_variance,port_returns,c=port_returns / port_variance,marker = "o")
    plt.grid(True)
    plt.xlabel("expecter volatility")
    plt.ylabel('expected return')
    plt.colorbar(label="Sharpe ratio")

    # 传入权重，输出收益率、风险、夏普的一个函数
    def statistics(weights):
        weights = np.array(weights)
        pret = np.sum(returns.mean() * weights) * 252
        pvol = np.sqrt(np.dot(weights.T,np.dot(returns.cov() * 252,weights)))
        return np.array([pret,pvol,pret / pvol])
    
    # 最大化夏普指数
    def min_func_sharpe(weights):
        return -statistics(weights)[2]
    
    # 最小化投资组合的方差
    def min_func_variance(weights):
        return statistics(weights)[1] ** 2
    
    cons = ({"type":'eq',"fun":lambda x :np.sum(x) - 1})
    bnds = tuple((0,1) for x in range(noa))

    opts = sco.minimize(min_func_sharpe, noa * [1. / noa,], 
                        method="SLSQP", bounds=bnds, constraints=cons)
    optv = sco.minimize(min_func_variance, noa * [1. / noa,], 
                        method="SLSQP", bounds=bnds, constraints=cons)

    print(u'Markowitz投资组合优化，\n第一次迭代：夏普率收益最大化(高风险模型适合牛市)\n', 
          np.c_[codelist, 
                codename, 
                np.round(opts['x'],2)])
    print(np.c_[['预期收益率', '波动率', '夏普系数'],
                 statistics(opts['x']).round(3)])

    print(u'Markowitz投资组合优化，\n第一次迭代：方差最小优化(低风险模型，适合大盘下降趋势)。\n', 
          np.c_[codelist, 
                codename, 
                np.round(optv['x'],2)])
    print(np.c_[['预期收益率', '波动率', '夏普系数'],
                 statistics(optv['x']).round(3)])
    # 关于约束条件的一些说明,其他的一些约束条件根据这个修改即可
    #cons = ({"type":'eq',"fun":lambda x :np.sum(x) - 1}, #这个约束条件表明权重加起来为1
    #        { "type":'eq',"fun":lambda x :x[0] - 0.05}, # 这个表明第一个权重要大于0.05
    #        x[0] >= 0.05
    #        {"type":'eq',"fun":lambda x :-x[2] + 0.4}, # 这个表明第三个权重要小于等于0.4
    #        x[2] <= 0.4
    #        )
    def min_func_port(weights):
        return statistics(weights)[1]

    # 在不同目标收益率水平（target_returns）循环时，最小化的一个约束条件会变化。 np.linspace(0.0, 0.25, 50)
    # 这就是每次循环中更新条件字典对象的原因：

    target_returns = np.linspace(0.0,0.5,50)
    target_variance = []
    for tar in target_returns:

        cons = ({'type':'eq','fun':lambda x:statistics(x)[0] - tar},
                {'type':'eq','fun':lambda x:np.sum(x) - 1})

        res = sco.minimize(min_func_port, noa * [1. / noa,], 
                           method='SLSQP', bounds=bnds, constraints=cons)
        target_variance.append(res['fun'])

    target_variance = np.array(target_variance)

    print(u'Markowitz投资组合优化，\n第二次迭代：计算平衡风险和利润组合。\n', 
          np.c_[codelist, 
                codename, 
                np.round(res['x'],2)])
    print(np.c_[['预期收益率', '波动率', '夏普系数'],
                 statistics(res['x']).round(3)])

    plt.figure(figsize=(8,4))
    # random portfolio composition
    plt.scatter(port_variance,
                port_returns,
                c=port_returns / port_variance, 
                marker='o')
    # efficient frontier
    plt.scatter(target_variance, 
                target_returns,
                c=target_returns / target_variance, 
                marker='x')
    # portfolio with highest Sharpe ratio
    plt.plot(statistics(opts['x'])[1],statistics(opts['x'])[0],'b*',markersize=15.0)
    # minimum variance portfolio
    plt.plot(statistics(optv['x'])[1],statistics(optv['x'])[0],'y*',markersize=15.0)
    # balance risk/returns portfolio
    plt.plot(statistics(res['x'])[1],statistics(res['x'])[0],'r*',markersize=15.0)
    plt.grid(True)
    plt.xlabel('expected volatility')
    plt.ylabel('expected return')
    plt.colorbar(label='Sharpe ratio')

    import scipy.interpolate as sci

    ind = np.argmin(target_variance)
    evols = target_variance[ind:]
    erets = target_returns[ind:]

    tck = sci.splrep(evols, erets)

    # 通过这条数值化路径，最终可以为有效边界定义一个连续可微函数
    # 和对应的一阶导数函数df(x):

    def f(x):
        """
        Efficient frontier function (splines approximation)
        :param x:
        :return:
        """
        return sci.splev(x, tck, der=0)

    def df(x):
        """
        First derivative of efficient frontier function.
        :param x:
        :return:
        """
        return sci.splev(x, tck, der=1)

    # 定义一个函数，返回给定参数集p=(a,b,x)
    def equations(p, rf=0.04):
        eq1 = rf - p[0]
        eq2 = rf + p[1] * p[2] - f(p[2])
        eq3 = p[1] - df(p[2])
        return eq1,eq2,eq3

    # 数值优化得到如下的值
    opt = sco.fsolve(equations,[0.01,0.5,0.15])

    plt.figure(figsize=(8, 4))
    # random portfolio composition
    plt.scatter(port_variance, 
                port_returns, 
                c=(port_returns - 0.01) / port_variance, 
                marker="o")
    # efficient frontier
    plt.plot(evols, erets, 'g', lw=4.0)
    cx = np.linspace(0.0, 0.3)
    plt.plot(cx, opt[0] + opt[1] * cx, lw=1.5)
    # capital market line
    plt.plot(opt[2], f(opt[2]), 'r*', markersize=15.0)
    plt.grid(True)
    plt.axhline(0, color='k', ls='-', lw=2.0)
    plt.axvline(0, color='k', ls='-', lw=2.0)
    plt.xlabel('expected volatility')
    plt.ylabel('expected return')
    plt.colorbar(label='Sharp ratio')

    cons = ({'type': 'eq', 'fun': lambda x: statistics(x)[0] - f(opt[2])},
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    res = sco.minimize(min_func_port, noa * [1. / noa,],
                       method='SLSQP', bounds=bnds, constraints=cons)

    print(u'Markowitz投资组合优化，\n第三次迭代：计算考虑现金组合。\n',
          np.c_[codelist,
                codename,
                res['x'].round(3)])
    print(np.c_[['预期收益率', '波动率', '夏普系数'],
                 statistics(res['x']).round(3)])

    print(u'Markowitz投资组合优化，\n第四次迭代：这个叫阿财的家伙很懒，挖了个天坑，还没有填上。\n',)
    plt.show()