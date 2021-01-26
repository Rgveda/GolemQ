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
from tqdm import tqdm

try:
    from joblib import Parallel, delayed
    import joblib
    import multiprocessing
except:
    print('Some command line args "-c" & "--pct_of_cores" need to run "pip install joblib" before use it.')
    print('Some command line args "-c" & "--pct_of_cores" need to run "pip install multiprocessing" before use it.')
    pass

# 命令行分析库
import argparse


try:
    import QUANTAXIS as QA
except:
    print('PLEASE run "pip install QUANTAXIS" before call GolemQ.cli modules')
    pass

from GolemQ.cli.sub import (
    sub_l1_from_sina,
    sub_codelist_l1_from_sina,
    sub_1min_from_tencent_lru,
)
from GolemQ.utils.symbol import (
    get_codelist,
)


if __name__ == '__main__':

    # 使用 argparse 库，定义脚本版本
    parser = argparse.ArgumentParser(description=u'而今证我绝学，你也算是死得其所！ QUANTAXIS 阿财/GolemQ mod 自动量化交易决策系统 v0.8')

    # 处理命令行参数
    parser.add_argument("-v", "--verbose", help=u"increase output verbosity 更多显示信息", action="store_true", default=False)
    parser.add_argument('-t', '--strategy', help=u"strategy will be evaluated 执行单一策略", type=str, default=False,)
    parser.add_argument('-x', '--train', help=u"训练识别典型主升浪概率的xgboost模型", action="store_true", default=False,)
    parser.add_argument('-e', '--eval', help=u"[all,etf,index,block,fast] will be evaluated 执行评估模式[all=全体A股，etf=主要etf，index=中证系列指数，block=重要指标股(默认)，fast=重要指标股，无仓位和止损控制]", type=str, default='fast')
    parser.add_argument("-f", "--frequency", help=u"frequency 交易周期频率", type=str, default='60min',)
    parser.add_argument('-p', '--portfolio', help=u"portfolio will be evaluated 策略组合方案", type=str, default='uprising',)
    parser.add_argument('-c', '--pct_of_cores', help=u"percentage of CPU cores will be used. 并行计算时最大CPU占用率", type=int, default=0,)
    parser.add_argument('-C', '--codelist', help=u"Codelist 指定 Code 列表", type=str, default=False,)
    parser.add_argument('-s', '--show', help=u"Show Me a Numer. 显示指定策略的选股结果", type=str, default=False,)
    parser.add_argument('-X', '--save', help=u"save stock data", type=str, default=False,)
    parser.add_argument('-i', '--shift', help=u"时移信号计算. 显示指定策略的时移选股结果", type=int, default=0,)
    parser.add_argument('-I', '--csindex', help=u"A股指数 + 时移信号计算. 显示A股指数的时移大盘趋势判断结果", type=int, default=0,)
    parser.add_argument('-S', '--sub', help=u"Subscibe stock. with ws server ip 接收实盘行情", type=str, default=False,)
    parser.add_argument('-r', '--risk', help=u"risk portfolio optimizer 策略仓位优化", type=str, default=False,)
    parser.add_argument('-m', '--mark', help=u"Mark uprising", action="store_true", default=False,)
    parser.add_argument('-k', '--check', help=u"Check code", type=str, default=False,)
    parser.add_argument('-l', '--log', help=u"截图 log", type=str, default=False,)
    parser.add_argument('-n', '--snapshot', help=u"Snapshot 保存图形图像", type=str, default=False,)
    parser.add_argument('-b', '--briefing', help=u"Fractal briefing", type=str, default=False,)

    cmdline_args = parser.parse_args()
    done = False

    if (cmdline_args.show == 'me_number') or \
        (cmdline_args.show == 'me_a_number') or \
        (cmdline_args.show == 'show_me_a_number'):
        from GolemQ.cli.show_number import show_me_number
        # 比我一只Number啦
        show_me_number(cmdline_args.strategy)
        done = True
    elif (cmdline_args.csindex > 0):
        from GolemQ.benchmark.csindex import csindex
        print(u'执行指令动作：在\'{}\'范围内寻找并使用csindex寻找主升浪'.format(cmdline_args.eval))
        csindex(cpu_usage=cmdline_args.pct_of_cores,
                verbose=cmdline_args.verbose,
                shift=cmdline_args.csindex,
                eval_range=cmdline_args.eval)
        done = True
    elif (cmdline_args.snapshot == 'snapshot'):
        # 评估
        print(u'没实现')
        done = True
    elif (cmdline_args.mark == True) or \
        (cmdline_args.strategy == 'markup'):
        # 评估
        from GolemQ.benchmark.mark_uprising import mark_uprising
        # 怼渠全市场
        print(u'执行指令动作：在\'{}\'范围内寻找并标记典型主升浪'.format(cmdline_args.eval))
        mark_uprising(cpu_usage=cmdline_args.pct_of_cores,
                   verbose=cmdline_args.verbose,
                   eval_range=cmdline_args.eval)
        done = True
    elif (cmdline_args.train == True) or \
        (cmdline_args.strategy == 'train_xgboost'):
        # 评估
        from GolemQ.benchmark.train_xgboost import train_xgboost
        # 怼渠全市场
        print(u'执行指令动作：在\'{}\'范围内训练识别典型主升浪概率的xgboost模型'.format(cmdline_args.eval))
        train_xgboost(cpu_usage=cmdline_args.pct_of_cores,
                      verbose=cmdline_args.verbose,
                      eval_range=cmdline_args.eval)
        done = True
    elif (cmdline_args.strategy == 'onewavelet'):
        from GolemQ.benchmark.onewavelet import onewavelet
        # 怼渠全市场
        onewavelet(frequency=cmdline_args.frequency, 
                   cpu_usage=cmdline_args.pct_of_cores,
                   verbose=cmdline_args.verbose,
                   eval_range=cmdline_args.eval)
        done = True
    elif (cmdline_args.strategy == 'block'):
        from GolemQ.benchmark.find_block import find_block
        # 怼渠全市场
        find_block(cpu_usage=cmdline_args.pct_of_cores,
                   verbose=cmdline_args.verbose,
                   eval_range=cmdline_args.eval)
        done = True
    elif (cmdline_args.strategy == 'regtree'):
        from GolemQ.benchmark.regtree import regtree
        # 怼渠全市场
        regtree(cpu_usage=cmdline_args.pct_of_cores,
                verbose=cmdline_args.verbose,
                eval_range=cmdline_args.eval)
        done = True
    elif (cmdline_args.strategy == 'uprising'):
        from GolemQ.benchmark.find_uprising import find_uprising
        # 怼渠全市场
        if (cmdline_args.eval in ['csindex', 'etf', 'test', 'fast', 'hs300', 'zz500', 'sz150']):
            find_uprising(cpu_usage=cmdline_args.pct_of_cores,
                          verbose=cmdline_args.verbose,
                          eval_range=cmdline_args.eval)
        elif (cmdline_args.eval in ['600031', '603444', '603713', '002557', '002352', '000963', '600436']):
            find_uprising(cpu_usage=cmdline_args.pct_of_cores,
                          verbose=cmdline_args.verbose,
                          eval_range=cmdline_args.eval)
        else:
            print(u'invalid eval range')
        done = True
    elif (cmdline_args.shift > 0.5):
        from GolemQ.benchmark.shift import shifting
        # 怼渠全市场
        shifting(cpu_usage=cmdline_args.pct_of_cores,
                 verbose=cmdline_args.verbose,
                 shift=cmdline_args.shift,
                 eval_range=cmdline_args.eval)
        done = True
    elif (cmdline_args.strategy == 'sharpe'):
        from GolemQ.benchmark.sharpe import calc_sharpe
        # 怼渠全市场
        calc_sharpe(cpu_usage=cmdline_args.pct_of_cores,
                   verbose=cmdline_args.verbose,
                   eval_range=cmdline_args.eval)
        done = True
    elif (cmdline_args.strategy == 'onewavelet_day'):
        from GolemQ.benchmark.onewavelet import onewavelet
        # 怼渠全市场日线
        onewavelet(frequency = 'day', 
                   cpu_usage=cmdline_args.pct_of_cores,
                   verbose=cmdline_args.verbose,
                   eval_range=cmdline_args.eval)
        done = True
    elif (cmdline_args.strategy == 'onewavelet_all'):
        from GolemQ.benchmark.onewavelet import onewavelet
        # 怼渠全市场日线
        onewavelet(frequency = 'day', 
                   cpu_usage=cmdline_args.pct_of_cores,
                   verbose=cmdline_args.verbose,
                   eval_range='all')
        done = True
    elif (cmdline_args.strategy == 'onewavelet_neo'):
        from GolemQ.benchmark.onewavelet import onewavelet
        # 怼渠全市场
        onewavelet_neo(frequency = 'day', 
                   cpu_usage=cmdline_args.pct_of_cores,
                   verbose=cmdline_args.verbose,
                   eval_range=cmdline_args.eval)
        done = True
    elif (cmdline_args.strategy == 'scale_patterns'):
        from GolemQ.benchmark.scale_patterns import scale_patterns
        # 怼渠
        scale_patterns(frequency=cmdline_args.frequency, 
                       cpu_usage=cmdline_args.pct_of_cores,
                       verbose=cmdline_args.verbose,
                       eval_range=cmdline_args.eval)
        done = True
    elif (cmdline_args.strategy == 'scale_patterns_day'):
        from GolemQ.benchmark.scale_patterns import scale_patterns
        scale_patterns(frequency = 'day', 
                       cpu_usage=cmdline_args.pct_of_cores,
                       verbose=cmdline_args.verbose,
                       eval_range=cmdline_args.eval)
        done = True
    elif (cmdline_args.strategy == 'gostrike'):
        from GolemQ.benchmark.gostrike import gostrike
        # 怼渠
        gostrike(frequency=cmdline_args.frequency, 
                       cpu_usage=cmdline_args.pct_of_cores,
                       verbose=cmdline_args.verbose,
                       eval_range=cmdline_args.eval)
        done = True
    elif (cmdline_args.strategy == 'gostrike_day'):
        from GolemQ.benchmark.gostrike import gostrike
        gostrike(frequency = 'day', 
                       cpu_usage=cmdline_args.pct_of_cores,
                       verbose=cmdline_args.verbose,
                       eval_range=cmdline_args.eval)
        done = True
    elif (cmdline_args.strategy == 'void'):
        from GolemQ.benchmark.dodge_void import dodge_void
        dodge_void(cpu_usage=cmdline_args.pct_of_cores,
                   verbose=cmdline_args.verbose,
                   eval_range=cmdline_args.eval)
        done = True
    elif (cmdline_args.sub != False) and \
        (cmdline_args.sub == 'sina_l1'):
        if (cmdline_args.codelist != False):
            # 收线，收线
            sub_codelist_l1_from_sina(get_codelist(cmdline_args.codelist))
        else:
            # 收线，收线
            sub_l1_from_sina()
        done = True
    elif (cmdline_args.sub != False) and \
        (cmdline_args.sub == 'huobi_realtime'):
        # 接收火币实时数据
        from QUANTAXIS.QASU.save_huobi import (
            QA_SU_save_huobi_realtime,
        )
        QA_SU_save_huobi_realtime()
        done = True
    elif (cmdline_args.sub != False) and \
        (cmdline_args.sub == 'huobi'):
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
    elif (cmdline_args.sub != False) and \
        (cmdline_args.sub == 'okex'):
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
    elif (cmdline_args.sub != False) and \
        (cmdline_args.sub == 'binance'):
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
    elif (cmdline_args.save == 'stock_block'):
        from QUANTAXIS.QACmd import QA_SU_save_stock_block
        from QUANTAXIS.QAFetch import QA_fetch_get_stock_block
        from QUANTAXIS.QAUtil import (
            DATABASE,
            QASETTING,
            QA_util_log_info, 
            QA_util_log_debug, 
            QA_util_log_expection,
            QA_util_to_json_from_pandas
        )    

        client=DATABASE
        client.drop_collection('stock_block')
        coll = client.stock_block
        coll.create_index('code')

        stock_block = QA_fetch_get_stock_block('tdx')
        print(stock_block)
        coll.insert_many(
            QA_util_to_json_from_pandas(stock_block)
        )

        from QUANTAXIS.QAFetch.QATushare import QA_fetch_get_stock_block as QATushare_fetch_get_stock_block
        codelist = QATushare_fetch_get_stock_block()
        print(codelist)
        coll.insert_many(
            QA_util_to_json_from_pandas(codelist)
        )
        
        from GolemQ.utils.symbol import get_sz150s
        codelist = get_sz150s()
        print(codelist)
        coll.insert_many(
            QA_util_to_json_from_pandas(codelist)
        )
        done = True
    elif (cmdline_args.sub != False) and \
        (cmdline_args.sub == 'tencent_1min'):
        # 收线，收线
        sub_1min_from_tencent_lru()
        done = True
    elif (cmdline_args.check != False):
        blockname = ['MSCI中国', 'MSCI成份', 'MSCI概念', '三网融合',
                 '上证180', '上证380', '沪深300', '上证380', 
                 '深证300', '上证50', '上证电信', '电信等权', 
                 '上证100', '上证150', '沪深300', '中证100',
                 '中证500', '全指消费', '中小板指', '创业板指',
                 '综企指数', '1000可选', '国证食品', '深证可选',
                 '深证消费', '深成消费', '中证酒', '中证白酒',
                 '行业龙头', '白酒', '证券', '消费100', 
                 '消费电子', '消费金融', '富时A50', '银行', 
                 '中小银行', '证券', '军工', '白酒', '啤酒', 
                 '医疗器械', '医疗器械服务', '医疗改革', '医药商业', 
                 '医药电商', '中药', '消费100', '消费电子', 
                 '消费金融', '黄金', '黄金概念', '4G5G', 
                 '5G概念', '生态农业', '生物医药', '生物疫苗',
                '机场航运', '数字货币', '文化传媒']
        blockname = list(set(blockname))
        codelist_candidate = QA.QA_fetch_stock_block_adv().get_block(blockname).code
        #codelist_candidate = [code for code in codelist_candidate if not
        #code.startswith('300')]
    
        print('批量评估板块成分股：{}'.format(blockname))

        codelist = get_codelist(cmdline_args.check)
        miss_codelist = [item for item in codelist if item not in codelist_candidate]
        print('=====> 以下{}个个股没有进入候选策略计算清单：'.format(len(miss_codelist)), miss_codelist)

        print(u'\n Total:{} 正在处理... '.format(len(codelist_candidate)))
        if (cmdline_args.pct_of_cores == 0):
            # *** DEMO *** 这里完全是为了测试进度条效果是否正常，并不是真的需要读条
            try:
                pbar = tqdm(codelist_candidate, ascii=True)
                for code in pbar:
                    pbar.set_description(u"A股({})".format(code))
                    pbar.update(1)

                    time.sleep(0.5)
                    pass
            except KeyboardInterrupt:
                pbar.close()
                raise
            pbar.close()
        else:
            cpu_usage = cmdline_args.pct_of_cores

            # 多线程并行处理 DEMO
            if (cpu_usage > 1):
                # 如果 cpu_usage 参数大于1，则认为是百分比
                cpu_usage = cpu_usage / 100

            # 按百分比例分配使用当前主机CPU核心数量，每CPU核心一次性批次读取并处理个股数量
            max_usage_of_cores, step = int(cpu_usage * multiprocessing.cpu_count()), 5

            print(u'*** Standalone 单机多线程模式 ***')
            print(u'即将使用 {:.0%} —> {} 个CPU核心进行指定策略的全市场数据并行处理。'.format(cpu_usage, 
                                                max_usage_of_cores,))

            # *** DEMO *** 这里完全是为了测试进度条效果是否正常，并不是真的需要读条
            verbose = False
            portfolio_batch = 'Fooo'
            codelist_grouped = [(verbose,
                                 portfolio_batch,
                                 codelist_candidate[i:i + step]) for i in range(0,
                                                                          len(codelist_candidate),
                                                                          step)]
        
            def calc_batch_with_dodge_void(codelist,
                                           portfolio_batch,
                                           verbose=False):
                """
                假负载函数
                """
                time.sleep(1.25)
                pass

            import contextlib
            
            @contextlib.contextmanager
            def tqdm_joblib(tqdm_object):
                """Context manager to patch joblib to report into tqdm progress bar given as argument"""
                class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
                    def __init__(self, *args, **kwargs):
                        super().__init__(*args, **kwargs)

                    def __call__(self, *args, **kwargs):
                        tqdm_object.set_description(u"A股({})".format(tqdm_object.iterable[tqdm_object.n]))
                        if ((tqdm_object.n + step) < len(tqdm_object.iterable)):
                            tqdm_object.update(n=step)  # self.batch_size
                        else:
                            tqdm_object.update(n=(len(tqdm_object.iterable) - tqdm_object.n))  # self.batch_size
                        return super().__call__(*args, **kwargs)

                old_batch_callback = joblib.parallel.BatchCompletionCallBack
                joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
                try:
                    yield tqdm_object
                finally:
                    joblib.parallel.BatchCompletionCallBack = old_batch_callback
                    tqdm_object.close()    

            overall_progress = tqdm(codelist_candidate, unit='stock')
            with tqdm_joblib(overall_progress) as progress_bar:
                # calc_my_func 是我的策略封装执行回测的函数
                Parallel(n_jobs=max_usage_of_cores)(delayed(calc_batch_with_dodge_void)(codelist,
                                                                      portfolio_batch,
                                                                      verbose) for verbose,
                             portfolio_batch,
                             codelist in codelist_grouped)
            pass
        done = True
    elif (cmdline_args.risk != False):
        from GolemQ.cli.portfolio import portfolio_optimizer
        # 收线，收线
        if (len(cmdline_args.portfolio) > 0):
            print(u'读取策略 {} 进行 {} 模型优化'.format(cmdline_args.portfolio, cmdline_args.risk))
            portfolio_optimizer(cmdline_args.risk, strategy=cmdline_args.portfolio)
        else:
            portfolio_optimizer(cmdline_args.risk)
        done = True
    elif (cmdline_args.briefing != False):

        done = True

    if (done == False):
        print('I think U need Help, Try type "python -m GolemQ.cli --help"')
