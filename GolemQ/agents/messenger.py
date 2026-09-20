# coding:utf-8
# Author: 阿财（11652964@qq.com）
# Created date : 2025-08-11
#
# The MIT License (MIT)
#
# Copyright (c) 2020-2025 azai/GolemQ
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


from typing import List, Optional, Dict, Union
from datetime import datetime
from GolemQ.core.settings import GQSETTING
from GolemQ.core.settings import DEFAULT_DB_URI
from GolemQ.core.preprocessing import mask_sensitive_info
from alibabacloud_dingtalk.oauth2_1_0.client import Client as dingtalkoauth2_1_0Client
from alibabacloud_tea_openapi import models as open_api_models
from alibabacloud_dingtalk.oauth2_1_0 import models as dingtalkoauth_2__1__0_models
from alibabacloud_tea_util.client import Client as UtilClient
from alibabacloud_tea_util import models as util_models
from alibabacloud_dingtalk.robot_1_0.client import Client as dingtalkrobot_1_0Client
from alibabacloud_dingtalk.robot_1_0 import models as dingtalkrobot__1__0_models
import requests
import re


def check_dingtalk_config() -> bool:
    """检查钉钉配置是否存在"""
    appkey = GQSETTING.get_config('DINGTALK', 'appkey')
    appsecret = GQSETTING.get_config('DINGTALK', 'appsecret')
    robot_code = GQSETTING.get_config('DINGTALK', 'robot_code')
    user_id_list = GQSETTING.get_config('DINGTALK', 'user_id_list')
    
    # Check if all required configurations are present and not empty/default values
    return (all([appkey, appsecret, robot_code, user_id_list]) and
            appkey != DEFAULT_DB_URI and
            appsecret != DEFAULT_DB_URI and
            robot_code != DEFAULT_DB_URI and
            user_id_list != DEFAULT_DB_URI)


def setup_dingtalk_config() -> None:
    """交互式设置钉钉配置"""
    print("钉钉机器人配置缺失，请按提示输入配置信息:")
    appkey = input("请输入钉钉AppKey: ")
    appsecret = input("请输入钉钉AppSecret: ")
    robot_code = input("请输入钉钉机器人Code: ")
    user_id_list = input("请输入接收用户ID列表(多个用逗号分隔): ")
    
    GQSETTING.set_config('DINGTALK', 'appkey', appkey)
    GQSETTING.set_config('DINGTALK', 'appsecret', appsecret)
    GQSETTING.set_config('DINGTALK', 'robot_code', robot_code)
    GQSETTING.set_config('DINGTALK', 'user_id_list', user_id_list)
    print("钉钉配置已保存!")


def get_dingtalk_config() -> Dict[str, Union[str, List[str]]]:
    """获取钉钉配置"""
    if not check_dingtalk_config():
        setup_dingtalk_config()

    return {
        'appkey': GQSETTING.get_config('DINGTALK', 'appkey'),
        'appsecret': GQSETTING.get_config('DINGTALK', 'appsecret'),
        'robot_code': GQSETTING.get_config('DINGTALK', 'robot_code'),
        'user_id_list': GQSETTING.get_config('DINGTALK', 'user_id_list').split(',')
    }


class DingtalkAccessToken:
    """钉钉访问令牌管理类"""
    def __init__(self):
        pass

    @staticmethod
    def create_client() -> dingtalkoauth2_1_0Client:
        """
        使用 Token 初始化账号Client
        @return: Client
        @throws Exception
        """
        config = open_api_models.Config()
        config.protocol = 'https'
        config.region_id = 'central'
        return dingtalkoauth2_1_0Client(config)

    @staticmethod
    def main(
        appkey: Optional[str] = 'appkey_dinggtkolxz1u****eqd',
        appsecret: Optional[str] = 'appsecret_dinggtkolxz1u****eqd',
    ) -> Optional[str]:
        config = get_dingtalk_config()
        appkey = appkey or config['appkey']
        appsecret = appsecret or config['appsecret']
        
        client = DingtalkAccessToken.create_client()
        get_access_token_request = dingtalkoauth_2__1__0_models.GetAccessTokenRequest(
            app_key=appkey,
            app_secret=appsecret,
        )
        try:
            response = client.get_access_token(get_access_token_request)
            return response.body.access_token
        except Exception as err:
            if not UtilClient.empty(err.code) and not UtilClient.empty(err.message):
                print(f"获取AccessToken失败: {err.code} - {err.message}")
            return None


class DingReminder:
    """钉钉消息提醒类"""

    def __init__(self):
        pass

    @staticmethod
    def create_client() -> dingtalkrobot_1_0Client:
        """
        使用 Token 初始化账号Client
        @return: Client
        @throws Exception
        """
        config = open_api_models.Config()
        config.protocol = 'https'
        config.region_id = 'central'
        return dingtalkrobot_1_0Client(config)

    @staticmethod
    def send_message(
        robot_code: str = 'dinggtkolxz1u****eqd',
        user_id_list: List[str] = ['root'],
        content: str = 'GolemQ推送提醒',
        appkey: Optional[str] = None,
        appsecret: Optional[str] = None,
    ) -> None:
        dingtalk_assess_token = DingtalkAccessToken()
        access_token = dingtalk_assess_token.main(
            appkey=appkey,
            appsecret=appsecret,
        )

        client = DingReminder.create_client()
        robot_send_ding_headers = dingtalkrobot__1__0_models.RobotSendDingHeaders()
        robot_send_ding_headers.x_acs_dingtalk_access_token = access_token
        robot_send_ding_request = dingtalkrobot__1__0_models.RobotSendDingRequest(
            robot_code=robot_code,
            remind_type=1,
            receiver_user_id_list=user_id_list,
            content=content
        )
        try:
            res = client.robot_send_ding_with_options(
                robot_send_ding_request, 
                robot_send_ding_headers, 
                util_models.RuntimeOptions()
            )
            return res
        except Exception as err:
            if not UtilClient.empty(err.code) and not UtilClient.empty(err.message):
                # err 中含有 code 和 message 属性，可帮助开发定位问题
                print(err)
                pass


def setup_dingtalk_config_interactive() -> None:
    """交互式设置钉钉配置"""
    try:
        if check_dingtalk_config():
            print("钉钉配置已存在:")
            try:
                config = {
                    'appkey': GQSETTING.get_config('DINGTALK', 'appkey'),
                    'appsecret': GQSETTING.get_config('DINGTALK', 'appsecret'),
                    'robot_code': GQSETTING.get_config('DINGTALK', 'robot_code'),
                    'user_id_list': GQSETTING.get_config('DINGTALK', 'user_id_list')
                }
                for key, value in config.items():
                    # 对所有配置值进行脱敏处理
                    masked_value = mask_sensitive_info(value)
                    print(f"  {key}: {masked_value}")
                
                choice = input("是否重新配置钉钉? (y/N): ").strip().lower()
                if choice != 'y':
                    # 发送测试消息
                    send_dingtalk_test_message()
                    return
            except Exception as e:
                print(f"读取现有配置时出错: {e}")
                print("将重新配置钉钉...")
    
    except Exception as e:
        print(f"检查钉钉配置时出错: {e}")
        print("将重新配置钉钉...")
    
    setup_dingtalk_config()
    # 发送测试消息
    send_dingtalk_test_message()


def send_dingtalk_test_message() -> None:
    """发送钉钉测试消息"""
    try:
        config = get_dingtalk_config()
        print("发送钉钉测试消息...")
        ret_result = DingReminder.send_message(
            robot_code=config['robot_code'],
            user_id_list=config['user_id_list'],
            content="✅ GolemQ 钉钉配置测试成功！\n这是一条测试消息，表示钉钉机器人配置正确。"
        )
        if (ret_result.status_code == 200):
            print("✓ 钉钉测试消息发送成功！")
        else:
            # 枚举响应对象属性
            print("响应对象属性:")
            for attr in dir(ret_result):
                if not attr.startswith('_'):
                    try:
                        value = getattr(ret_result, attr)
                        print(f"  {attr}: {type(value).__name__}")
                    except Exception:
                        print(f"  {attr}: <无法获取值>")
    except Exception as e:
        print(f"✗ 钉钉测试消息发送失败: {e}")


def check_serverchan_config() -> bool:
    """检查Server酱配置是否存在"""
    sendkey = GQSETTING.get_config('SERVERCHAN', 'sendkey')
    
    # Check if required configuration is present and not empty/default values
    return (sendkey and sendkey != DEFAULT_DB_URI)


def setup_serverchan_config() -> None:
    """交互式设置Server酱配置"""
    print("Server酱配置缺失，请按提示输入配置信息:")
    sendkey = input("请输入Server酱SendKey: ")
    
    GQSETTING.set_config('SERVERCHAN', 'sendkey', sendkey)
    print("Server酱配置已保存!")


def get_serverchan_config() -> Dict[str, str]:
    """获取Server酱配置"""
    if not check_serverchan_config():
        setup_serverchan_config()

    return {
        'sendkey': GQSETTING.get_config('SERVERCHAN', 'sendkey')
    }


def sc_send(sendkey: str, title: str, desp: str = '', options: Optional[Dict] = None) -> Dict:
    """发送Server酱消息
    
    Args:
        sendkey: Server酱SendKey
        title: 消息标题
        desp: 消息内容
        options: 额外选项
        
    Returns:
        dict: 发送结果
    """
    if options is None:
        options = {}
    
    # 判断 sendkey 是否以 'sctp' 开头，并提取数字构造 URL
    if sendkey.startswith('sctp'):
        match = re.match(r'sctp(\d+)t', sendkey)
        if match:
            num = match.group(1)
            url = f'https://{num}.push.ft07.com/send/{sendkey}.send'
        else:
            raise ValueError('Invalid sendkey format for sctp')
    else:
        url = f'https://sctapi.ftqq.com/{sendkey}.send'
    
    params = {
        'title': title,
        'desp': desp,
        **options
    }
    
    headers = {
        'Content-Type': 'application/json;charset=utf-8'
    }
    
    try:
        response = requests.post(url, json=params, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Server酱消息发送失败: {e}")
        return {'error': str(e)}


def send_serverchan_message(title: str, content: str = '', options: Optional[Dict] = None) -> Dict:
    """发送Server酱消息
    
    Args:
        title: 消息标题
        content: 消息内容
        options: 额外选项
        
    Returns:
        dict: 发送结果
    """
    config = get_serverchan_config()
    return sc_send(config['sendkey'], title, content, options)


def setup_serverchan_config_interactive() -> None:
    """交互式设置Server酱配置"""
    try:
        if check_serverchan_config():
            print("Server酱配置已存在:")
            try:
                config = {
                    'sendkey': GQSETTING.get_config('SERVERCHAN', 'sendkey')
                }
                for key, value in config.items():
                    # 对所有配置值进行脱敏处理
                    masked_value = mask_sensitive_info(value)
                    print(f"  {key}: {masked_value}")
                
                choice = input("是否重新配置Server酱? (y/N): ").strip().lower()
                if choice != 'y':
                    # 发送测试消息
                    send_serverchan_test_message()
                    return
            except Exception as e:
                print(f"读取现有配置时出错: {e}")
                print("将重新配置Server酱...")
    
    except Exception as e:
        print(f"检查Server酱配置时出错: {e}")
        print("将重新配置Server酱...")
    
    setup_serverchan_config()
    # 发送测试消息
    send_serverchan_test_message()


def send_serverchan_test_message() -> None:
    """发送Server酱测试消息"""
    try:
        print("发送Server酱测试消息...")
        result = send_serverchan_message(
            title="✅ GolemQ Server酱配置测试",
            content="这是一条测试消息，表示Server酱配置正确。\n\n测试时间: " +
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
        
        if 'error' not in result:
            print("✓ Server酱测试消息发送成功！")
            if 'data' in result and 'pushid' in result['data']:
                print(f"  消息ID: {result['data']['pushid']}")
        else:
            print(f"✗ Server酱测试消息发送失败: {result['error']}")
            
    except Exception as e:
        print(f"✗ Server酱测试消息发送失败: {e}")


