#
# The MIT License (MIT)
#
# Copyright (c) 2020-2025 azai/Rgveda/GolemQuant
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
"""
这里定义的是一些本地目录
"""

import sys
import os
import pandas as pd

"""创建本地文件夹


1. setting_path ==> 用于存放配置文件 setting.cfg
2. cache_path ==> 用于存放临时文件
3. log_path ==> 用于存放储存的log
4. download_path ==> 下载的数据/财务文件
5. strategy_path ==> 存放策略模板
6. bin_path ==> 存放一些交易的sdk/bin文件等
"""

basepath = os.getcwd()
path = os.path.expanduser('~')
user_path = os.path.join(path, '.GolemQ')


def get_python_version_suffix():
    """获取Python版本后缀，如 'py38'"""
    major = sys.version_info.major
    minor = sys.version_info.minor
    return f"py{major}{minor}"


def get_pickle_filename(base_name, suffix=None):
    """
    生成带版本后缀的pickle文件名

    参数:
        base_name: 基础文件名，如 'codelist_firstDayTrading.pickle'
        suffix: 自定义后缀，如 'py38'，如果不提供则自动生成
    """
    if suffix is None:
        suffix = get_python_version_suffix()

    # 分离文件名和扩展名
    name, ext = os.path.splitext(base_name)
    # 生成新文件名：原文件名_后缀.扩展名
    return f"{name}_{suffix}{ext}"


def cache_path(
    dirname,
    portable=False,
    prefix=''
):
    """
    返回本地用户目录下的'.GolemQ'为根目录的缓存临时文件目录，如果 portable 参数等于 True，
    则返回程序代码启动目录为根目录的缓存目录。
    """
    if (portable):
        if (len(prefix) > 0):
            ret_cache_path = os.path.abspath(
                os.path.join(
                    basepath,
                    prefix,
                    'datastore',
                    'cache',
                    dirname
                )
            )
        else:
            ret_cache_path = os.path.join(basepath, 'datastore', 'cache', dirname)
    else:
        ret_cache_path = os.path.join(user_path, 'datastore', 'cache', dirname)
    if not (os.path.exists(ret_cache_path) and os.path.isdir(ret_cache_path)):
        try:
            os.makedirs(ret_cache_path)
        except Exception:
            # 如果目录已经存在，那么可能是并发冲突，当做什么事情都没发生
            if not (os.path.exists(ret_cache_path)):
                # 否则继续触发异常
                os.makedirs(os.path.join(ret_cache_path))
    return ret_cache_path


def mkdirs_user(dirname):
    if not (os.path.exists(os.path.join(user_path, dirname)) and os.path.isdir(os.path.join(user_path, dirname))):
        try:
            os.makedirs(os.path.join(user_path, dirname))
        except Exception:
            # 如果目录已经存在，那么可能是并发冲突，当做什么事情都没发生
            if not (os.path.join(user_path, dirname)):
                # 否则继续触发异常
                os.makedirs(os.path.join(user_path, dirname))
    return os.path.join(user_path, dirname)


def mkdirs(dirname):
    if not (os.path.exists(os.path.join(basepath, dirname)) and os.path.isdir(os.path.join(basepath, dirname))):
        try:
            os.makedirs(os.path.join(basepath, dirname))
        except Exception:
            # 如果目录已经存在，那么可能是并发冲突，当做什么事情都没发生
            if not (os.path.exists(os.path.join(basepath, dirname))):
                # 否则继续触发异常
                os.makedirs(os.path.join(basepath, dirname))
    return os.path.join(basepath, dirname)


def load_snapshot_cache(dirpath, filename='cache.pickle'):
    filename = filename.replace(' ', '_').replace(':', '_')
    metadata = pd.read_pickle(os.path.join(mkdirs(dirpath), filename))
    return metadata


def save_snapshot_cache(dirpath, filename='cache.pickle', metadata=None):
    filename = filename.replace(' ', '_').replace(':', '_')
    metadata = metadata.to_pickle(os.path.join(mkdirs(dirpath), filename))
    return os.path.join(mkdirs(dirpath), filename)


setting_path = os.path.join(user_path, 'settings')
# datastore_path = os.join(cache_path('cache')
# log_path = GolemQ_Path('log')
# download_path = GolemQ_Path('downloads')
# strategy_path = GolemQ_Path('strategy')
# bin_path = GolemQ_Path('bin')     #   给一些dll文件存储用

mkdirs(setting_path,)
