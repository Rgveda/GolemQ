# coding:utf-8
#
# The MIT License (MIT)
#
# Copyright (c) 2018-2020 azai/Rgveda/GolemQuant
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
from datetime import datetime, timezone, timedelta
import time
import threading
import os
import sys

# 命令行分析库
import argparse


try:
    import QUANTAXIS as QA
except:
    print('PLEASE run "pip install QUANTAXIS" before call GolemQ.cli modules')
    pass

# 东方财富爬虫
from QUANTAXIS.QASU.main import (QA_SU_crawl_eastmoney)

from GolemQ.cli.sub import (
    sub_l1_from_sina,
    sub_1min_from_tencent_lru,
)
from GolemQ.GQUtil.symbol import (
    get_codelist,
)


if __name__ == '__main__':

    # 使用 argparse 库，定义脚本版本
    parser = argparse.ArgumentParser(description=u'而今证我绝学，你也算是死得其所！ QUANTAXIS 阿财/GolemQ mod 自动量化交易决策系统 v0.8')

    # 处理命令行参数
    parser.add_argument("-v", "--verbose", help=u"increase output verbosity 更多显示信息", action="store_true", default=False)
    parser.add_argument('-t', '--strategy', help=u"strategy will be evaluated 执行单一策略", type=str, default='',)
    parser.add_argument('-e', '--eval', help=u"[all,etf,index,block,fast] will be evaluated 执行评估模式[all=全体A股，etf=主要etf，index=中证系列指数，block=重要指标股(默认)，fast=重要指标股，无仓位和止损控制]", type=str, default='block')
    parser.add_argument("-f", "--frequency", help=u"frequency 交易周期频率", type=str, default='60min',)
    parser.add_argument('-p', '--portfolio', help=u"portfolio will be evaluated 策略组合方案", type=str, default='sharpe_onewavelet_day',)
    parser.add_argument('-c', '--pct_of_cores', help=u"percentage of CPU cores will be used. 并行计算时最大CPU占用率", type=int, default=0,)
    parser.add_argument('-s', '--show', help=u"Show Me a Numer. 显示指定策略的选股结果", type=str, default='',)
    parser.add_argument('-S', '--sub', help=u"Subscibe stock. with ws server ip 接收实盘行情", type=str, default='',)
    parser.add_argument('-r', '--risk', help=u"risk portfolio optimizer 策略仓位优化", type=str, default='CVaR',)
    parser.add_argument('-m', '--moneyflow', help=u"moneyflow 资金流向", type=str, default='CVaR',)
    parser.add_argument('-k', '--check', help=u"Check code", type=str, default='',)
    parser.add_argument('-l', '--labelimg', help=u"labelimg", type=str, default='',)

    cmdline_args = parser.parse_args()
    done = False

    if (cmdline_args.show == 'me_number') or \
        (cmdline_args.show == 'me_a_number') or \
        (cmdline_args.show == 'show_me_a_number'):
        from GolemQ.cli.show_number import show_me_number
        # 比我一只Number啦
        show_me_number(cmdline_args.strategy)
        done = True
    elif (cmdline_args.moneyflow == 'eastmoney'):
        # 比我一只Number啦
        QA_SU_crawl_eastmoney(action='zjlx', stockCode='all')
        done = True
    elif (cmdline_args.strategy == 'onewavelet'):
        from GolemQ.GQBenchmark.onewavelet import onewavelet
        # 怼渠全市场
        onewavelet(frequency=cmdline_args.frequency, 
                   cpu_usage=cmdline_args.pct_of_cores,
                   verbose=cmdline_args.verbose,
                   eval_range=cmdline_args.eval)
        done = True
    elif (cmdline_args.strategy == 'uprising'):
        from GolemQ.GQBenchmark.find_uprising import find_uprising
        # 怼渠全市场
        find_uprising(cpu_usage=cmdline_args.pct_of_cores,
                   verbose=cmdline_args.verbose,
                   eval_range=cmdline_args.eval)
        done = True
    elif (cmdline_args.strategy == 'sharpe'):
        from GolemQ.GQBenchmark.sharpe import calc_sharpe
        # 怼渠全市场
        calc_sharpe(cpu_usage=cmdline_args.pct_of_cores,
                   verbose=cmdline_args.verbose,
                   eval_range=cmdline_args.eval)
        done = True
    elif (cmdline_args.strategy == 'onewavelet_day'):
        from GolemQ.GQBenchmark.onewavelet import onewavelet
        # 怼渠全市场日线
        onewavelet(frequency = 'day', 
                   cpu_usage=cmdline_args.pct_of_cores,
                   verbose=cmdline_args.verbose,
                   eval_range=cmdline_args.eval)
        done = True
    elif (cmdline_args.strategy == 'onewavelet_all'):
        from GolemQ.GQBenchmark.onewavelet import onewavelet
        # 怼渠全市场日线
        onewavelet(frequency = 'day', 
                   cpu_usage=cmdline_args.pct_of_cores,
                   verbose=cmdline_args.verbose,
                   eval_range='all')
        done = True
    elif (cmdline_args.strategy == 'onewavelet_neo'):
        from GolemQ.GQBenchmark.onewavelet import onewavelet
        # 怼渠全市场
        onewavelet_neo(frequency = 'day', 
                   cpu_usage=cmdline_args.pct_of_cores,
                   verbose=cmdline_args.verbose,
                   eval_range=cmdline_args.eval)
        done = True
    elif (cmdline_args.strategy == 'scale_patterns'):
        from GolemQ.GQBenchmark.scale_patterns import scale_patterns
        # 怼渠
        scale_patterns(frequency=cmdline_args.frequency, 
                       cpu_usage=cmdline_args.pct_of_cores,
                       verbose=cmdline_args.verbose,
                       eval_range=cmdline_args.eval)
        done = True
    elif (cmdline_args.strategy == 'scale_patterns_day'):
        from GolemQ.GQBenchmark.scale_patterns import scale_patterns
        scale_patterns(frequency = 'day', 
                       cpu_usage=cmdline_args.pct_of_cores,
                       verbose=cmdline_args.verbose,
                       eval_range=cmdline_args.eval)
        done = True
    elif (cmdline_args.strategy == 'gostrike'):
        from GolemQ.GQBenchmark.gostrike import gostrike
        # 怼渠
        gostrike(frequency=cmdline_args.frequency, 
                       cpu_usage=cmdline_args.pct_of_cores,
                       verbose=cmdline_args.verbose,
                       eval_range=cmdline_args.eval)
        done = True
    elif (cmdline_args.strategy == 'gostrike_day'):
        from GolemQ.GQBenchmark.gostrike import gostrike
        gostrike(frequency = 'day', 
                       cpu_usage=cmdline_args.pct_of_cores,
                       verbose=cmdline_args.verbose,
                       eval_range=cmdline_args.eval)
        done = True
    elif ((len(cmdline_args.sub) > 0) and \
        (cmdline_args.sub == 'sina_l1')):
        # 收线，收线
        sub_l1_from_sina()
        done = True
    elif ((len(cmdline_args.sub) > 0) and \
        (cmdline_args.sub == 'huobi_realtime')):
        # 接收火币实时数据
        from QUANTAXIS.QASU.save_huobi import (
            QA_SU_save_huobi_realtime,
        )
        QA_SU_save_huobi_realtime()
        done = True
    elif ((len(cmdline_args.sub) > 0) and \
        (cmdline_args.sub == 'huobi')):
        # 接收火币行情历史数据
        from QUANTAXIS.QASU.save_huobi import (
            QA_SU_save_huobi_symbol,
            QA_SU_save_huobi_1day,
            QA_SU_save_huobi_1hour,
            QA_SU_save_huobi,
            QA_SU_save_huobi_1min,
        )
        QA_SU_save_huobi_symbol()
        QA_SU_save_huobi_1day()
        QA_SU_save_huobi_1hour()
        QA_SU_save_huobi_1min()
        done = True
    elif ((len(cmdline_args.sub) > 0) and \
        (cmdline_args.sub == 'okex')):
        # 接收OKEx行情历史数据
        from QUANTAXIS.QASU.save_okex import (
            QA_SU_save_okex_symbol,
            QA_SU_save_okex,
            QA_SU_save_okex_1day,
            QA_SU_save_okex_1hour,
            QA_SU_save_okex_1min,
        )
        QA_SU_save_okex_symbol()
        QA_SU_save_okex_1day()
        QA_SU_save_okex_1hour()
        QA_SU_save_okex('1800')
        QA_SU_save_okex('900')
        QA_SU_save_okex('300')
        QA_SU_save_okex_1min()
        done = True
    elif ((len(cmdline_args.sub) > 0) and \
        (cmdline_args.sub == 'binance')):
        # 接收币安行情历史数据
        from QUANTAXIS.QASU.save_binance import (
            QA_SU_save_binance_symbol,
            QA_SU_save_binance,
            QA_SU_save_binance_1day,
            QA_SU_save_binance_1hour,
            QA_SU_save_binance_1min,
        )
        QA_SU_save_binance_symbol()
        QA_SU_save_binance_1day()
        QA_SU_save_binance_1hour()
        QA_SU_save_binance_1min()
        done = True
    elif ((len(cmdline_args.sub) > 0) and \
        (cmdline_args.sub == 'tencent_1min')):
        # 收线，收线
        sub_1min_from_tencent_lru()
        done = True
    elif (len(cmdline_args.check) > 0):
        blockname = ['MSCI中国',
                 'MSCI成份', 
                 'MSCI概念',
                 '三网融合',
                 '上证180', 
                 '上证380',
                 '沪深300', 
                 '上证380', 
                 '深证300',
                 '上证50',
                 '上证电信',
                 '电信等权',
                 '上证100',
                 '上证150',
                 '沪深300',
                 '中证100',
                 '中证500',
                 '全指消费',
                 '中小板指',
                 '创业板指',
                 '综企指数',
                 '1000可选',
                 '国证食品',
                 '深证可选',
                 '深证消费',
                 '深成消费',
                 '中证酒', 
                 '中证白酒',
                 '行业龙头',
                 '白酒',
                 '证券', 
                 '消费100', 
                 '消费电子', 
                 '消费金融',
                 '富时A50',
                 '银行', 
                 '中小银行', 
                 '证券', 
                 '军工', 
                 '白酒', 
                 '啤酒', 
                 '医疗器械', 
                 '医疗器械服务', 
                 '医疗改革', 
                 '医药商业', 
                 '医药电商', 
                 '中药', 
                 '消费100', 
                 '消费电子', 
                 '消费金融',
                 '黄金', 
                 '黄金概念',
                '4G5G', 
                '5G概念', 
                '生态农业', 
                '生物医药', 
                '生物疫苗',
                '机场航运',
                '数字货币', 
                '文化传媒'
                ]
        blockname = list(set(blockname))
        codelist_candidate = QA.QA_fetch_stock_block_adv().get_block(blockname).code
        codelist_candidate = [code for code in codelist_candidate if not code.startswith('300')]

        print('批量评估板块成分股：{} Total:{}'.format(blockname, 
                                                   len(codelist_candidate)))
        
        codelist = get_codelist(cmdline_args.check)
        miss_codelist = [item for item in codelist if item not in codelist_candidate]
        print('\n=====> 以下{}个个股没有进入候选策略计算清单：'.format(len(miss_codelist)), miss_codelist)

        done = True
    elif (len(cmdline_args.risk) > 0):
        from GolemQ.cli.portfolio import portfolio_optimizer
        # 收线，收线
        if (len(cmdline_args.portfolio) > 0):
            print(u'读取策略 {} 进行 {} 模型优化'.format(cmdline_args.portfolio, cmdline_args.risk))
            portfolio_optimizer(cmdline_args.risk, strategy=cmdline_args.portfolio)
        else:
            portfolio_optimizer(cmdline_args.risk)
        done = True

    if (done == False):
        print('U need Help, Try type "Help"')
