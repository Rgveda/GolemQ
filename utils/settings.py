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
import os
import pandas as pd

from GolemQ.utils.path import (
    mkdirs_user,
)

#def load_settings(filename = 'password.hdf5'):
#    settings = pd.read_hdf(os.path.join(mkdirs_user('settings'), filename),
#                           key='df', mode='r')
#    return settings

#def save_settings(filename = 'password.hdf5', metadata = None):
#    setting = metadata.to_hdf(os.path.join(mkdirs_user('settings'), filename),
#                              key='df', mode='w')
#    return True
def exists_settings(filename='password.pickle'):
    """
    将用户密码token等信息保存到系统的当前用户目录 %userprofile%/mkdirs_user() 目录下，
    这样保证在GitHub共享代码也不会泄露重要账户信息风险
    检查指定的配置文件是否存在。
    """
    settings_file = os.path.join(mkdirs_user('settings'), filename)
    if (os.path.exists(settings_file) and os.path.isfile(settings_file)):
        return True
    else:
        return False


def input_settings(filename='password.pickle', pattern=pd.DataFrame(),):
    """
    将用户密码token等信息保存到系统的当前用户目录 %userprofile%/mkdirs_user() 目录下，
    这样保证在GitHub共享代码也不会泄露重要账户信息风险
    按照输入模板，要求用户从键盘输入指定的登录信息
    """
    settings_filename = os.path.join(mkdirs_user('settings'), filename)
    for index, settings in pattern.iterrows():
        for column in pattern.columns:
            if (pattern.loc[index, column] is not None):
                hostname = pattern.loc[index, column]
            else:
                prompt = u'请输入登录信息 Please enter {} "{}":'.format(hostname,
                                                                     column)
                pattern.loc[index, column] = input(prompt)
    return pattern


def load_settings(filename='password.pickle'):
    """
    将用户密码token等信息保存到系统的当前用户目录 %userprofile%/mkdirs_user() 目录下，
    这样保证在GitHub共享代码也不会泄露重要账户信息风险
    保存用户登录信息到 pickle 文件
    """
    settings = pd.read_pickle(os.path.join(mkdirs_user('settings'), filename))
    return settings


def save_settings(filename='password.pickle', metadata=None):
    """
    将用户密码token等信息保存到系统的当前用户目录 %userprofile%/mkdirs_user() 目录下，
    这样保证在GitHub共享代码也不会泄露重要账户信息风险
    从 pickle 文件读取用户登录信息
    """
    setting = metadata.to_pickle(os.path.join(mkdirs_user('settings'), 
                                              filename))
    return True