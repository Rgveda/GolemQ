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

import QUANTAXIS as QA
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
    print('PLEASE run "pip install QUANTAXIS" before call GolemQ.cli.portfolio modules')
    pass

from GolemQ.indices.indices import *
from GolemQ.cli.show_number import (
    position,
)

def show_verbose(opt_res, obj, rm, no_print=False):
    """
    解读资产配置优化结果
    """
    if (obj == 'Sharpe'):
        obj_name = '夏普率'
    elif (obj == 'Sortino'):
        obj_name = 'Sortino Ratio'
    elif (obj == 'MinRisk'):
        obj_name = '最小风险'

    if (rm == 'MV'):
        rm_name = '均衡收益'
    elif (rm == 'WR'):
        rm_name = '最坏可能性'
    else:
        rm_name = 'CVaR风险最低'

    res_weights = pd.DataFrame(opt_res, columns=['code', 'name', 'weights'])
    ret_verbose = '按{}{}化计算有推荐仓位的股票：\n{}'.format(obj_name, rm_name, res_weights[res_weights['weights'].gt(0.001)])
    ret_verbose = '{}\n{}'.format(ret_verbose, 
                                  '剩下都是没有推荐仓位(牌面)的：\n{}'.format(res_weights.loc[res_weights['weights'].lt(0.001), ['code', 'name']].values))
    if (no_print == False):
        print(ret_verbose)
    return ret_verbose


def portfolio_optimizer(rm='CVaR', 
                        alpha=0.1, 
                        risk_free=0.02,
                        strategy='sharpe_scale_patterns_day',):
    pd.options.display.float_format = '{:.1%}'.format

    # 股票代码，我直接用我的选股程序获取选股列表。
    position_signals = position(portfolio=strategy,
                        frequency='day',
                        market_type=QA.MARKET_TYPE.STOCK_CN,
                        verbose=False)
    if (position_signals is not None) and \
        (len(position_signals) > 0):
        datestamp = position_signals.index[0][0]
        position_signals_best = position_signals.loc[position_signals[FLD.LEVERAGE_ONHOLD].gt(0.99), :]
        if (len(position_signals_best) > 20):
            position_signals = position_signals_best
        else:
            pass

        codelist = position_signals.index.get_level_values(level=1).to_list()

        # 获取股票中文名称，只是为了看得方便，交易策略并不需要股票中文名称
        stock_names = QA.QA_fetch_stock_name(codelist)
        codename = [stock_names.at[code, 'name'] for code in codelist]
        codename_T = {codename[i]:codelist[i] for i in range(len(codelist))}

        data_day = QA.QA_fetch_stock_day_adv(codelist,
            start='2014-01-01',
            end='{}'.format(datetime.date.today())).to_qfq()

        # 收益率序列
        rets_jotion = data_day.add_func(kline_returns_func)
        returns = pd.DataFrame(columns=codelist, 
                               index=sorted(data_day.data.index.get_level_values(level=0).unique()))
        for code in codelist:
            returns[code] = rets_jotion.loc[(slice(None), code), :].reset_index(level=[1], drop=True)

        returns = returns.fillna(0)
        returns = returns.rename(columns={codelist[i]:codename[i] for i in range(len(codelist))})

        import riskfolio.Portfolio as pf

        # Building the portfolio object
        port = pf.Portfolio(returns=returns)
        # Calculating optimum portfolio

        # Select method and estimate input parameters:

        method_mu = 'hist' # Method to estimate expected returns based on historical data.
        method_cov = 'hist' # Method to estimate covariance matrix based on historical data.

        port.assets_stats(method_mu=method_mu, method_cov=method_cov, d=0.94)

        ## Estimate optimal portfolio:
        model = 'Classic' # Could be Classic (historical), BL (Black Litterman) or FM (Factor Model)
        obj = 'Sharpe' # Objective function, could be MinRisk, MaxRet, Utility or Sharpe
        hist = True # Use historical scenarios for risk measures that depend on scenarios
        rf = risk_free / 365 # Risk free rate
        l = 0 # Risk aversion factor, only useful when obj is 'Utility'
        port.alpha = alpha

        # 暗色主题
        plt.style.use('Solarize_Light2')

        # 正常显示中文字体
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']

        import riskfolio.PlotFunctions as plf

        # Plotting the composition of the portfolio

        w = port.optimization(model=model, rm=rm, obj=obj, rf=rf, l=l, hist=hist)

        opt_weights = w.copy()
        opt_weights['code'] = opt_weights.apply(lambda x: codename_T[x.name], axis=1)
        opt_weights['name'] = opt_weights.apply(lambda x: x.name, axis=1)
        opt_weights = opt_weights.set_index(['code'], drop=False)
        print(u'交易日', datestamp)
        show_verbose(opt_weights, obj, rm)

        if (rm == 'CVaR'):
            # Risk measure CVaR
            title = 'Sharpe Mean CVaR'
        elif (rm == 'MV'):
            # Risk measure used, this time will be variance
            title = 'Sharpe Mean Variance'
        elif (rm == 'WR'):
            title = 'Sharpe Mean WR'
        elif (rm == 'Sortino'):
            title = 'Sortino Mean WR'
        else:
            rm = 'CVaR'
            title = 'Sharpe Mean CVaR'

        ax = plf.plot_pie(w=w, title=title, others=0.05, nrow=25, cmap = "tab20",
                          height=6, width=10, ax=None)

        plt.show()

        ## Plotting efficient frontier composition

        #points = 10 # Number of points of the frontier

        #frontier = port.efficient_frontier(model=model, rm=rm, points=points,
        #rf=rf, hist=hist)

        ##print(frontier.T.head())

        #ax = plf.plot_frontier_area(w_frontier=frontier, cmap="tab20",
        #height=6,
        #width=10, ax=None)

        #plt.show()
    else:
        print(u'没有可用的选股数据。')


if __name__ == '__main__':
    pass