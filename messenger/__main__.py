# coding:utf-8
# Author: 阿财（11652964@qq.com）
# Created date : 2018-08-14
#
# The MIT License (MIT)
#
# Copyright (c) 2016-2018 yutiansut/QUANTAXIS
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


# 原理是先将需要发送的文本放到剪贴板中，然后将剪贴板内容发送到qq窗口
# 之后模拟按键发送enter键发送消息

#import win32gui, win32api, win32con
import time
#import win32clipboard as w
import numpy as np
import requests

from datetime import datetime, timezone, timedelta
from QUANTAXIS.QAUtil.QADate import QA_util_time_stamp
from QUANTAXIS.QAUtil.QADate_Adv import (QA_util_timestamp_to_str, QA_util_str_to_Unix_timestamp, QA_util_datetime_to_Unix_timestamp)
from QUANTAXIS.QAUtil.QAConst import (THRESHOLD, TREND_STATUS, AKA, INDICATOR_FIELD as FLD, WORKER_MODE)
from QUANTAXIS.QAUtil.QAParameter import (FREQUENCE, ORDER_DIRECTION, MARKET_TYPE, AMOUNT_MODEL, ORDER_MODEL)
from QUANTAXIS.QAUtil.QASetting import (QA_Setting, DATABASE)
from QUANTAXIS.QAFetch.QAQuery_Advance import (QA_fetch_crypto_min_adv, QA_fetch_crypto_day_adv)
from QUANTAXIS.QAMarket.QAOrder_Adv import (check_realtime)
from QUANTAXIS.QAHuobi.QAHuobiSettings import (stock_rts)

def click_position(hwd, x_position, y_position, sleep):
    """
    鼠标左键点击指定坐标
    :param hwd: 
    :param x_position: 
    :param y_position: 
    :param sleep: 
    :return: 
    """
    # 将两个16位的值连接成一个32位的地址坐标
    long_position = win32api.MAKELONG(x_position, y_position)
    # win32api.SendMessage(hwnd, win32con.MOUSEEVENTF_LEFTDOWN,
    # win32con.MOUSEEVENTF_LEFTUP, long_position)
    # 点击左键
    win32api.SendMessage(hwd, win32con.WM_LBUTTONDOWN, win32con.MK_LBUTTON, long_position)
    win32api.SendMessage(hwd, win32con.WM_LBUTTONUP, win32con.MK_LBUTTON, long_position)
    time.sleep(int(sleep))


def getText():
    """获取剪贴板文本"""
    w.OpenClipboard()
    d = w.GetClipboardData(win32con.CF_UNICODETEXT)
    w.CloseClipboard()
    return d

def setText(aString):
    """设置剪贴板文本"""
    w.OpenClipboard()
    w.EmptyClipboard()
    w.SetClipboardData(win32con.CF_UNICODETEXT, aString)
    w.CloseClipboard()

def input_content(hwd, content, sleep, is_enter):
    """
    从站贴板中查找输入的内容
    :param hwd: 
    :param content: 
    :param sleep: 
    :param is_enter 是否要在最后输入enter键,内容与enter之间间隔一秒
    :return: 
    """
    setText(content)
    time.sleep(0.3)
    click_keys(hwd, win32con.VK_CONTROL, 86)
    if is_enter:
        time.sleep(1)
        click_keys(hwd, win32con.VK_RETURN)
    time.sleep(sleep)


def click_keys(hwd, *args):
    """
    定义组合按键
    :param hwd: 
    :param args: 
    :return: 
    """
    for arg in args:
        win32api.keybd_event(arg,0,0,0)  #ctrl键位码是17
        # win32api.SendMessage(hwd, win32con.WM_KEYDOWN, arg, 0)
    for arg in args:
        win32api.keybd_event(arg, 0, win32con.KEYEVENTF_KEYUP, 0) #释放按键
        # win32api.SendMessage(hwd, win32con.WM_KEYUP, arg, 0)

def send_servercha(msg, desp=''):
    url = 'https://sc.ftqq.com/SCU31414Td59ab2c6b23f5edaee3fb917778f005c5b8579e0043aa.send'
    if (desp == ''):
        body = {"text": u"QUANTAXIS (moded by 阿财/Azai)" , "desp": msg}
    else:
        body = {"text": msg , "desp": desp}

    response = requests.post(url, data = body)


def send_wechat(msg):
    """发送qq消息
    to_who：qq消息接收人
    msg：需要发送的消息
    """

    # 获取qq窗口句柄
    hwnd = win32gui.FindWindow('WeChatMainWndForPC', u'微信')
    print(hwnd)
    while hwnd == 0:
        hwnd = win32gui.FindWindow('WeChatMainWndForPC', u'微信')
        print(wechat)

    if int(hwnd) <= 0:
        print(u"没有找到微信PC版，退出进程................")
        exit(0)

    print(u"查询到微信PC版: %s " % hwnd)

    #由于keybd_event需要激活才能成功发送快捷键
    win32gui.ShowWindow(hwnd,1)  
    win32gui.SetForegroundWindow(hwnd)

    # 发送消息一
    input_content(hwnd, msg, 1, False)
    click_keys(hwnd, win32con.VK_RETURN)
    time.sleep(1)


def check_fetch_huobi(stock_conf, time_window=2):
    """
    检查实时行情数据抓取是否中断，如果交易中断则微信发送提醒
    """
    stock_conf['from'] = (datetime.strptime(stock_conf['to'] + ' +0800', '%Y-%m-%d %H:%M:%S %z') - timedelta(hours = time_window)).astimezone(timezone(timedelta(hours=8))).strftime('%Y-%m-%d %H:%M:%S')
    data1min = QA_fetch_crypto_min_adv([stock_conf['stock_code']], stock_conf['from'], stock_conf['to'], frequence=FREQUENCE.ONE_MIN,)

    Period_Time = 60

    if (len(data1min.data) != 0):
        utc_timeline = int(data1min.data.iloc[-1].timestamp)
        time1 = datetime(1970,1,1, tzinfo=timezone.utc) + timedelta(seconds = utc_timeline)
    else:
        time1 = datetime(1970,1,1, tzinfo=timezone.utc)
    time2 = datetime.now(timezone(timedelta(hours=8)))

    if (len(data1min.data) == 0) or \
        (Period_Time * 3 < (time2 - time1).total_seconds()):
        # * 代表非实时数据，已经超出当前时间区间
        if ((time2 - time1).total_seconds() > 14400):
            msg = u"实时行情 MariaDB 数据库已经走丢了，已经超过 %d 小时没有更新，请主人检查 DS1517+ 情况。—— %s" % (time_window, QA_util_timestamp_to_str(datetime.now(timezone(timedelta(hours=8)))),)
        else:
            msg = u"huobi.pro 实时行情数据已经超过 %d 秒没有更新，请主人检查那台宇宙超级无敌的主机和海外网络连接情况。—— %s" % ((time2 - time1).total_seconds(), QA_util_timestamp_to_str(datetime.now(timezone(timedelta(hours=8)))),)
        if ((QA_util_datetime_to_Unix_timestamp(datetime.now(timezone(timedelta(hours=8)))) - stock_conf['alert_counter']) >= 300):
            stock_conf['alert_counter'] = QA_util_datetime_to_Unix_timestamp(datetime.now(timezone(timedelta(hours=8))))
            send_servercha(msg)
            #send_wechat(msg)
            print(msg)

    return stock_conf


def order_messenger(stock_conf, time_window=2):
    """
    新生成的订单操作，推送微信消息
    """
    order_queue = []
    stock_conf['from'] = (datetime.strptime(stock_conf['to'] + ' +0800', '%Y-%m-%d %H:%M:%S %z') - timedelta(hours = time_window)).astimezone(timezone(timedelta(hours=8))).strftime('%Y-%m-%d %H:%M:%S')
    order_res = DATABASE.order.find({ 'code': [stock_conf['stock_code'], '='], 'market_name': [stock_conf['market_name'], '='], 'timeline':["%d and %d" % (QA_util_time_stamp(stock_conf['from']), QA_util_time_stamp(stock_conf['to'])), 'BETWEEN']})

    order_idx = 0
    timeline = 1
    stock_id = 2
    market_name = 3
    market_type = 4
    code = 5
    frequence = 6
    price = 7
    towards = 8
    profit = 9
    status_code = 10
    remark = 11
    amount = 12
    money = 13
    count = 14
    date = 15
    fld_datetime = 16

    onhold = { 'buy_price': 0.0, 'buy_time': None, 'phase': None, 'sell_price': 0.0, 'sell_time': None, 'order_price': 0.0, 'fast_escape': False, 'count':0, 'total':0 }

    for row in order_res:
        order_need_process = False
        if (check_realtime(row, 900) and (row[count] > 0)) or \
           (check_realtime(row, 900) and ((row[remark].find('STOPLOSS') != -1) or (row[remark].find('DRAWDOWN4') != -1))):
            towards_str = ''
            if (row[towards] == ORDER_DIRECTION.BUY) and (row[count] > 0) and ((int(row[status_code]) == 100) or (int(row[status_code]) == 200) or (int(row[status_code]) == 300)):
                towards_str = u'买入'
                remark_str = '仓位：%.2f%%' % (stock_conf['weight'] * 100) 
                order_need_process = True
            elif (row[towards] == ORDER_DIRECTION.SELL) and (row[count] > 0) and ((int(row[status_code]) == 100) or (int(row[status_code]) == 200) or (int(row[status_code]) == 300)):
                towards_str = u'卖出'
                remark_str = '收益率：%.2f%%' % (row[profit] * 100)
                order_need_process = True
            else:
                remark_str = '已推送或未满条件：count = %d' % (row[count]) 

            msg = u'交易所:%s，币种:%s，操作:%s，价格:%.4f %s —— %s ' % (row[market_name], row[code], towards_str, row[price], remark_str, row[fld_datetime],)
            if (order_need_process):
                send_servercha(msg, desp = u"备注信息：%s" % row[remark])
                print(msg)
                #send_wechat(msg)
                #send_wechat(u"备注信息：%s" % row[remark])
                order_data = { 'order_idx':row[order_idx], 'status_code': int(row[status_code]) + 1}
                DATABASE.order.update(order_data)
            #else:
            #    print(msg)

def rolling_messenger(stock_params):
    """
    滚动查询和发送系统消息到关联的微信号上面
    """
    
    alive_conter = 0
    while(True):
        stock_params['btcusdt']['to'] = QA_util_timestamp_to_str(datetime.now(timezone(timedelta(hours=8))))
        #stock_params['btcusdt']['from'] =
        #QA_util_timestamp_to_str(datetime.now(timezone(timedelta(hours=8))))
        order_messenger(stock_params['btcusdt'])
        time.sleep(1)
        stock_params['htusdt']['to'] = QA_util_timestamp_to_str(datetime.now(timezone(timedelta(hours=8))))
        order_messenger(stock_params['htusdt'])
        time.sleep(1)
        stock_params['ethusdt']['to'] = QA_util_timestamp_to_str(datetime.now(timezone(timedelta(hours=8))))
        order_messenger(stock_params['ethusdt'])
        time.sleep(1)
        stock_params["eosusdt"]['to'] = QA_util_timestamp_to_str(datetime.now(timezone(timedelta(hours=8))))
        order_messenger(stock_params["eosusdt"])
        time.sleep(1)
        stock_params["etcusdt"]['to'] = QA_util_timestamp_to_str(datetime.now(timezone(timedelta(hours=8))))
        order_messenger(stock_params["etcusdt"])
        time.sleep(1)
        stock_params["bchusdt"]['to'] = QA_util_timestamp_to_str(datetime.now(timezone(timedelta(hours=8))))
        order_messenger(stock_params["bchusdt"])
        time.sleep(1)
        stock_params["dashusdt"]['to'] = QA_util_timestamp_to_str(datetime.now(timezone(timedelta(hours=8))))
        order_messenger(stock_params["dashusdt"])
        time.sleep(1)
        stock_params["ltcusdt"]['to'] = QA_util_timestamp_to_str(datetime.now(timezone(timedelta(hours=8))))
        order_messenger(stock_params["ltcusdt"])
        time.sleep(1)
        stock_params["xmrusdt"]['to'] = QA_util_timestamp_to_str(datetime.now(timezone(timedelta(hours=8))))
        order_messenger(stock_params["xmrusdt"])

        stock_params['btcusdt']['to'] = QA_util_timestamp_to_str(datetime.now(timezone(timedelta(hours=8))))
        stock_params['btcusdt']['from'] = QA_util_timestamp_to_str(datetime.now(timezone(timedelta(hours=8))) - timedelta(minutes = 5))
        stock_params['btcusdt'] = check_fetch_huobi(stock_params['btcusdt'])
        time.sleep(1)
        if ((QA_util_datetime_to_Unix_timestamp(datetime.now(timezone(timedelta(hours=8)))) - alive_conter) >= 60):
            alive_conter = QA_util_datetime_to_Unix_timestamp(datetime.now(timezone(timedelta(hours=8))))
            print(u'%s WeChat/微信/Server酱信息推送提醒 Alive...' % (QA_util_timestamp_to_str(datetime.now(timezone(timedelta(hours=8))))))


if __name__ == '__main__':

    rolling_messenger(stock_rts)