# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from random import randint

USER_AGENTS = ["Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1; SV1; AcooBrowser; .NET CLR 1.1.4322; .NET CLR 2.0.50727)",
             "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 6.0; Acoo Browser; SLCC1; .NET CLR 2.0.50727; Media Center PC 5.0; .NET CLR 3.0.04506)",
             "Mozilla/4.0 (compatible; MSIE 7.0; AOL 9.5; AOLBuild 4337.35; Windows NT 5.1; .NET CLR 1.1.4322; .NET CLR 2.0.50727)",
             "Mozilla/5.0 (Windows; U; MSIE 9.0; Windows NT 9.0; en-US)",
             "Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; Win64; x64; Trident/5.0; .NET CLR 3.5.30729; .NET CLR 3.0.30729; .NET CLR 2.0.50727; Media Center PC 6.0)",
             "Mozilla/5.0 (compatible; MSIE 8.0; Windows NT 6.0; Trident/4.0; WOW64; Trident/4.0; SLCC2; .NET CLR 2.0.50727; .NET CLR 3.5.30729; .NET CLR 3.0.30729; .NET CLR 1.0.3705; .NET CLR 1.1.4322)",
             "Mozilla/4.0 (compatible; MSIE 7.0b; Windows NT 5.2; .NET CLR 1.1.4322; .NET CLR 2.0.50727; InfoPath.2; .NET CLR 3.0.04506.30)",
             "Mozilla/5.0 (Windows; U; Windows NT 5.1; zh-CN) AppleWebKit/523.15 (KHTML, like Gecko, Safari/419.3) Arora/0.3 (Change: 287 c9dfb30)",
             "Mozilla/5.0 (X11; U; Linux; en-US) AppleWebKit/527+ (KHTML, like Gecko, Safari/419.3) Arora/0.6",
             "Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.8.1.2pre) Gecko/20070215 K-Ninja/2.1.1",
             "Mozilla/5.0 (Windows; U; Windows NT 5.1; zh-CN; rv:1.9) Gecko/20080705 Firefox/3.0 Kapiko/3.0",
             "Mozilla/5.0 (X11; Linux i686; U;) Gecko/20070322 Kazehakase/0.4.5",
             "Mozilla/5.0 (X11; U; Linux i686; en-US; rv:1.9.0.8) Gecko Fedora/1.9.0.8-1.fc10 Kazehakase/0.5.6",
             "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/535.11 (KHTML, like Gecko) Chrome/17.0.963.56 Safari/535.11",
             "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_7_3) AppleWebKit/535.20 (KHTML, like Gecko) Chrome/19.0.1036.7 Safari/535.20",
             "Opera/9.80 (Macintosh; Intel Mac OS X 10.6.8; U; fr) Presto/2.9.168 Version/11.52",
             'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/536.6 (KHTML, like Gecko) Chrome/20.0.1092.0 Safari/536.6',]

random_agent = USER_AGENTS[randint(0, len(USER_AGENTS) - 1)]
headers = {
    'User-Agent':random_agent,
    }

import requests
import time
from requests.exceptions import RequestException
from bs4 import BeautifulSoup
from collections import namedtuple
import pandas as pd

def get_csindex_stocklist_from_sina(url_home = 'http://vip.stock.finance.sina.com.cn/corp/go.php/vII_NewestComponent/indexid/000133.phtml',
                                    headers=headers, encoding='gbk', verbose=True):
    """
    从新浪财经抓取股票成分列表，默认抓取上证150
    """
    resp = requests.get(url_home, headers=headers)
    ret_stocklist = []
    if resp.status_code == 200:
        resp.encoding = encoding #读取网页内容，用utf-8解码成字节
        soup = BeautifulSoup(resp.text,'lxml')#将网页的内容规范化
        soup_stocklist_tables = soup.find('table',
                                          {'id':"NewStockTable"}).find_all('tr')
        for i in range(0,len(soup_stocklist_tables)):
            cell = soup_stocklist_tables[i].find_all('td')
            # print(len(cell))
            if (len(cell)>2):
                if (len(cell[0].text)==6) and (cell[0].text.isdigit()==True):
                    #print(cell[0].text, cell[1].text, cell[2].text)
                    ret_stocklist.append({'code':cell[0].text, 
                                          'name':cell[1].text, 
                                          'date':cell[2].text, 
                                          'source':'sina', 
                                          'type':'csindex'})
    try:
        csindex500 = pd.DataFrame(ret_stocklist)
        csindex500 = csindex500.drop(['date', 'name'], axis=1)
        return csindex500.set_index('code', drop=False)
    except:
        return None
    return ret_stocklist