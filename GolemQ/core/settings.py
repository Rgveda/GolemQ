# coding:utf-8
#
# The MIT License (MIT)
#
# Copyright (c) 2020-2026 acai/GolemQ
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import configparser
import json
import os
from multiprocessing import Lock
from GolemQ.core.path import setting_path
from GolemQ.core.mongo import (
    GQ_util_mongodb_client_async,
    GQ_util_mongodb_client,
)


# quantaxis有一个配置目录存放在 ~/.quantaxis
# 如果配置目录不存在就创建，主要配置都保存在config.json里面
# 貌似yutian已经进行了，文件的创建步骤，他还会创建一个setting的dir
# 需要与yutian讨论具体配置文件的放置位置 author:Will 2018.5.19

DEFAULT_MONGO = os.getenv('MONGODB', 'localhost')
DEFAULT_DB_URI = 'mongodb://{}:27017'.format(DEFAULT_MONGO)
CONFIGFILE_PATH = os.path.join(setting_path, 'config.ini')

#: **只从 INI 读、不回落到 MongoDB 的段**（凭证类）。
#: ⚠️ 原先这串在 `get_config` / `set_config` 里**各写了一遍**，两处一旦不同步
#: 就会出现「读得到、写却写进 Mongo」这种半吊子状态 —— 收成一处。
#: `TDXAIDATA` / `TUSHARE` 是 2026-10-10 补进来的（`tushare_source` 的模块
#: docstring 里早就写着这条待办）。
INI_ONLY_SECTIONS = ('DINGTALK', 'SERVERCHAN', 'XTQUANT', 'TDXAIDATA', 'TUSHARE')



class GQ_Setting():

    def __init__(self, uri=None):
        self.lock = Lock()

        self.mongo_uri = uri or self.get_mongo()
        self.username = None
        self.password = None

        # 加入配置文件地址

    def get_mongo(self):
        config = configparser.ConfigParser()
        if os.path.exists(CONFIGFILE_PATH):
            config.read(CONFIGFILE_PATH)

            try:
                res = config.get('MONGODB', 'uri')
            except Exception:
                res = DEFAULT_DB_URI

        else:
            config = configparser.ConfigParser()
            config.add_section('MONGODB')
            config.set('MONGODB', 'uri', DEFAULT_DB_URI)
            f = open(os.path.join(setting_path, 'config.ini'), 'w')
            config.write(f)
            res = DEFAULT_DB_URI

        return res

    def get_config(
            self,
            section='MONGODB',
            option='uri',
            default_value=DEFAULT_DB_URI
    ):
        """[summary]

        Keyword Arguments:
            section {str} -- [description] (default: {'MONGODB'})
            option {str} -- [description] (default: {'uri'})
            default_value {[type]} -- [description] (default: {DEFAULT_DB_URI})

        Returns:
            [type] -- [description]
        """

        try:
            config = configparser.ConfigParser()
            config.read(CONFIGFILE_PATH)
            return config.get(section, option)
        except Exception:
            # For DingTalk, Server酱, and XTQuant configuration, don't fall back to MongoDB
            if section in INI_ONLY_SECTIONS:
                return default_value
            else:
                # For other sections, use MongoDB as fallback
                res = self.client.GolemQ.settings.find_one(
                    {'section': section})
                if res:
                    return res.get(option, default_value)
                else:
                    self.set_config(section, option, default_value)
                    return default_value

    def set_config(
            self,
            section='MONGODB',
            option='uri',
            default_value=DEFAULT_DB_URI
    ):
        """[summary]

        Keyword Arguments:
            section {str} -- [description] (default: {'MONGODB'})
            option {str} -- [description] (default: {'uri'})
            default_value {[type]} -- [description] (default: {DEFAULT_DB_URI})

        Returns:
            [type] -- [description]
        """
        # For DingTalk, Server酱, and XTQuant configuration, write to INI file instead of MongoDB
        if section in INI_ONLY_SECTIONS:
            config = configparser.ConfigParser()
            if os.path.exists(CONFIGFILE_PATH):
                config.read(CONFIGFILE_PATH)

            if not config.has_section(section):
                config.add_section(section)

            config.set(section, option, str(default_value))

            with open(CONFIGFILE_PATH, 'w') as f:
                config.write(f)
        else:
            # For other sections, use MongoDB as before
            t = {'section': section, option: default_value}
            self.client.GolemQ.settings.update_one(
                {'section': section}, {'$set': t}, upsert=True)

    def get_or_set_section(
            self,
            config,
            section,
            option,
            DEFAULT_VALUE,
            method='get'
    ):
        """[summary]

        Arguments:
            config {[type]} -- [description]
            section {[type]} -- [description]
            option {[type]} -- [description]
            DEFAULT_VALUE {[type]} -- [description]

        Keyword Arguments:
            method {str} -- [description] (default: {'get'})

        Returns:
            [type] -- [description]
        """

        try:
            if isinstance(DEFAULT_VALUE, str):
                val = DEFAULT_VALUE
            else:
                val = json.dumps(DEFAULT_VALUE)
            if method == 'get':
                return self.get_config(section, option)
            else:
                self.set_config(section, option, val)
                return val

        except Exception:
            self.set_config(section, option, val)
            return val

    def env_config(self):
        return os.environ.get("MONGOURI", None)

    @property
    def client(self):
        return GQ_util_mongodb_client(self.mongo_uri)

    @property
    def client_async(self):
        return GQ_util_mongodb_client_async(self.mongo_uri)



def test_mongodb_connection(uri: str) -> bool:
    """测试MongoDB连接"""
    try:
        from GolemQ.core.mongo import GQ_util_mongodb_client
        client = GQ_util_mongodb_client(uri)
        # 尝试获取服务器信息来测试连接
        server_info = client.server_info()
        version = server_info.get('version', '未知版本')
        print(f"✓ MongoDB连接测试成功! 版本: {version}")
        client.close()
        return True
    except Exception as e:
        print(f"✗ MongoDB连接测试失败: {e}")
        return False


def setup_mongodb_config() -> None:
    """交互式设置MongoDB配置"""
    print("MongoDB配置设置:")
    current_uri = GQSETTING.get_mongo()
    print("当前MongoDB URI:", current_uri)

    # 测试当前配置的连接
    if current_uri != 'mongodb://localhost:27017':
        print("测试当前配置的连接...")
        if not test_mongodb_connection(current_uri):
            print("当前配置无法连接，建议重新配置")

    choice = input("是否修改MongoDB配置? (y/N): ").strip().lower()
    if choice == 'y':
        while True:
            mongodb_uri = input("请输入MongoDB连接URI (格式: mongodb://host:port/database): ").strip()
            if not mongodb_uri:
                print("URI不能为空，请重新输入")
                continue

            # 测试新配置的连接
            print("测试新配置的连接...")
            if test_mongodb_connection(mongodb_uri):
                # 更新配置文件
                import configparser
                config = configparser.ConfigParser()
                config_path = os.path.join(os.path.expanduser('~/.GolemQ'), 'config.ini')

                if os.path.exists(config_path):
                    config.read(config_path)
                else:
                    os.makedirs(os.path.dirname(config_path), exist_ok=True)
                    config.add_section('MONGODB')

                config.set('MONGODB', 'uri', mongodb_uri)

                with open(config_path, 'w') as f:
                    config.write(f)

                print(f"✓ MongoDB配置已更新: {mongodb_uri}")
                break
            else:
                retry = input("连接测试失败，是否重新输入配置? (y/N): ").strip().lower()
                if retry != 'y':
                    print("配置未更改")
                    break
    else:
        print("MongoDB配置未更改")


GQSETTING = GQ_Setting()

#: 8.3 的**框架运维库**（心跳、函数签到、关注列表）。**名字即库名**（用户 2026-10-08 定的命名规则）。
#:
#: 为什么定义在这里而不是 `markets/StockCN/`：里面装的是**与市场无关的运维数据**，
#: 三个消费者（`supervisor/heartbeat`、`supervisor/function_checkin`、`cli/watchdog_manager`）
#: 都在根层/CLI 层 —— 让它们反向 import 某个市场包会违反 D1/D4 的分层。
GOLEMQ_NAME = 'golemq'
GOLEMQ = GQ_util_mongodb_client(
    GQSETTING.get_config('MONGODB', 'uri'))[GOLEMQ_NAME]
