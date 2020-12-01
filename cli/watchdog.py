# coding:utf-8
#
# The MIT License (MIT)
#
# Copyright (c) 2018-2020 azai/Rgveda/GolemQuant base on QUANTAXIS/yutiansut
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
import json
import sys
import websocket
import easyquotation
import datetime
import time as timer
import numba as nb

import collections
import requests

from GolemQ.utils.symbol import (
    get_codelist,
)


"""
看门狗程序，监控选股池股票，如有异动，微信通知
"""

Contact_Sheet = collections.namedtuple('_Contact', 'name gender type contact token')

def send_servercha(msg, sckey, desp=''):
    """
    发送Server酱
    """
    url = 'https://sc.ftqq.com/{}.send'.format(sckey)
    if (desp == ''):
        body = {"text": u"GolemQ关注买点和股价异动通知" , "desp": msg}
    else:
        body = {"text": msg , "desp": desp}

    response = requests.post(url, data = body)


def rolling_watchdog(eval):
    """
    滚动查询和发送系统消息到关联的微信号上面
    """
    codepool = ['600332', '601888', '002230', '603444', '300715', '002007', 
                '300750', '600887', '300122', '002271', '000333', '601318', 
                '000661', '600036', '000651', '603520', '002705', '600519', 
                '600009', '600900', '002714', '000538', '603486', '600309', 
                '603259', '002475', '300059', '600276', '600600', '002050', 
                '600612', '000858', '603288', '600031', '300015', '300146', 
                '300760', '600563', '603713']

    codelist_candidate = get_codelist(codepool)
    codelist = codelist_candidate
    print(len(codelist), codelist)

    max_usage_of_cores = 4

    # 少打印信息，多进程运行时打印调试信息会会刷疯狂屏
    verbose = False

    alive_conter = 0
    while(True):
        # 单线程读取测试
        for code in codelist:
            analysis_stock(code, verbose)
        
        time.sleep(1)
        if ((QA_util_datetime_to_Unix_timestamp(datetime.now(timezone(timedelta(hours=8)))) - alive_conter) >= 60):
            alive_conter = QA_util_datetime_to_Unix_timestamp(datetime.now(timezone(timedelta(hours=8))))
            print(u'%s WeChat/微信/Server酱信息推送提醒 Alive...' % (QA_util_timestamp_to_str(datetime.now(timezone(timedelta(hours=8))))))


if __name__ == '__main__':

    print('Type of Contact_Sheet:', type(Contact_Sheet))

    azai = Contact_Sheet(name=u'鑫鑫家的阿财', gender=u'男', 
                         type=u'方糖', 
                         contact='SCU31414Td59ab2c6b23f5edaee3fb917778f005c5b8579e0043aa', 
                         token='SCU31414Td59ab2c6b23f5edaee3fb917778f005c5b8579e0043aa')
    
    rolling_watchdog(eval = 'mao30')
