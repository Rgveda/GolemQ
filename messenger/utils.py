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

import collections
import requests

Contact_Sheet = collections.namedtuple('_Contact', 'name gender type contact token')

azai = Contact_Sheet(name=u'鑫鑫家的阿财', gender=u'男', 
                        type=u'方糖', 
                        contact='SCU31414Td59ab2c6b23f5edaee3fb917778f005c5b8579e0043aa', 
                        token='SCU31414Td59ab2c6b23f5edaee3fb917778f005c5b8579e0043aa')
   
def send_servercha(user, msg, desp=''):
    url = 'https://sc.ftqq.com/{}.send'.format(user.token)
    if (desp == ''):
        body = {"text": u"QUANTAXIS (moded by 阿财/Azai)" , "desp": msg}
    else:
        body = {"text": msg , "desp": desp}

    response = requests.post(url, data = body)