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
# Context 类
# 你知道的，大部分量化策略系统里面都有一个叫Context的玩意，QA使用QAStrategy，所以拿别家策略
# 抄过去都不知道如何下嘴。我最擅长就是搞Dummy接口，那就创造一个Portfolio_Context来代替，糅合
# 各家的数据，对了，这个类设计成可以保存到 Mongodb 进行持久化，这样你就不会每天都要重新回测
# 重新计算，它也包容了其他系统的回测数据。
#
import numpy as np
import pandas as pd
import datetime as dt

try:
    from QUANTAXIS.QAUtil.QADate_Adv import (
        QA_util_timestamp_to_str, 
        QA_util_str_to_Unix_timestamp, 
        QA_util_datetime_to_Unix_timestamp, 
        QA_util_str_to_datetime
    )
except:
    print(u'PLEASE run "pip install QUANTAXIS" before call Portfolio_Context modules')
    pass


def GQ_Portfolio_Context():
    """
    针对不同资产交易对象的Portfolio仓位组合实例化，支持多标的，跨种类，
    """

    __indices : pd.DataFrame = None
    __ohlc_bars : pd.DataFrame = None

    __period_time = dt.timedelta(hours = 1)
    __xtick_each_hour = 0
    __bar_index: int = 0
    __bar_time: dt.datetime = dt.date.today()

    __candidate_assets :dict = {}

    __current_dt: dt.datetime = dt.date.today()

    def __init__(self, *args, **kwargs):
        """
        初始化数据
        """
        pass


    @property 
    def barIndex(self) -> int:
        """
        返回当前主时间的游标索引（0~len(self._ohlc_bars)-1）
        """
        return self.__bar_index


    @barIndex.setter
    def barIndex(self, value) -> int:
        """
        设置当前时间游标索引（0~len(self._ohlc_bars)-1）
        """
        self.__bar_index = value


    @property 
    def barTime(self) -> str:
        """
        返回当前时间游标（时间格式 YYYY-mm-dd HH:MM:SS）
        """
        return self.__bar_time


    @barTime.setter
    def barTime(self, value):
        """
        设置当前时间游标（时间格式 YYYY-mm-dd HH:MM:SS）,并且计算当前标量
        """
        self.__bar_time = value


    @property
    def nextBarTime(self,) -> str:
        """
        当前时间和下一个Bar时间时间间隔大于蜡烛图数据周期，那么代表回测数据而非实盘数据，
        否则代表实盘模式数据运行中
        """
        time1 = QA_util_str_to_datetime(self.tickTime)

        # 当前实时时间：北京时间
        cur_realtime = dt.now(dt.timezone(dt.timedelta(hours=8)))

        if (self.__period_time < (cur_realtime - time1)):
            # * 代表回测数据而非实盘数据
            return QA_util_timestamp_to_str(time1 + self.__period_time)
        else:
            return QA_util_timestamp_to_str(cur_realtime)


    @property
    def periodTime(self,) -> dt.timedelta:
        """
        返回 序列时间间隔（dt.timedelta）
        """
        return self.__period_time


    @property
    def Period(self) -> str:
        """
        返回 序列时间间隔（Str）
        """
        return '%dmin' % (60 / self.__xtick_each_hour,)


    @property
    def xtickEachHour(self) -> float:
        """
        返回 序列时间间隔(每小时)
        """
        return self.__xtick_each_hour

