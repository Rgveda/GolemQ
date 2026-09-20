# coding:utf-8
#
# GolemQ 消息推送器
# 支持钉钉、Server酱等多种消息推送方式
#

import requests
import json
from typing import Optional, Dict
from enum import Enum
from GolemQ.core.settings import GQSETTING
from GolemQ.core.settings import DEFAULT_DB_URI
from GolemQ.agents import (
    DingReminder,
)


class MessageLevel(Enum):
    """消息级别"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class Messenger:
    """消息推送器"""
    
    def __init__(self):
        """检查钉钉配置是否存在"""
        appkey = GQSETTING.get_config('DINGTALK', 'appkey')
        appsecret = GQSETTING.get_config('DINGTALK', 'appsecret')
        robot_code = GQSETTING.get_config('DINGTALK', 'robot_code')
        user_id_list = GQSETTING.get_config('DINGTALK', 'user_id_list')
        user_id_list = user_id_list if (isinstance(user_id_list, list)) else [user_id_list]
        
        if (all([appkey, appsecret, robot_code, user_id_list]) and
                appkey != DEFAULT_DB_URI and
                appsecret != DEFAULT_DB_URI and
                robot_code != DEFAULT_DB_URI and
                user_id_list != DEFAULT_DB_URI):
            self.dingtalk_webhook = {
                'appkey': appkey,
                'appsecret': appsecret,
                'robot_code': robot_code,
                'user_id_list': user_id_list,
            }
        else:
            self.dingtalk_webhook = None

        sendkey = GQSETTING.get_config('SERVERCHAN', 'sendkey')
        if (sendkey and sendkey != DEFAULT_DB_URI):
            self.serverchan_key = sendkey
        else:
            self.serverchan_key = None
        self.custom_webhooks = {}
        
    def configure_dingtalk(self, webhook_url: str):
        """配置钉钉机器人webhook"""
        self.dingtalk_webhook = webhook_url
    
    def configure_serverchan(self, sendkey: str):
        """配置Server酱sendkey"""
        self.serverchan_key = sendkey
    
    def configure_custom_webhook(self, name: str, webhook_url: str):
        """配置自定义webhook"""
        self.custom_webhooks[name] = webhook_url
    
    def send_alert(self, title: str, message: str, level: str = "info") -> bool:
        """
        发送告警消息
        
        Args:
            title: 消息标题
            message: 消息内容
            level: 消息级别 (info, warning, error, critical)
            
        Returns:
            bool: 是否发送成功
        """
        success = False
        
        # # 尝试钉钉推送
        # if self.dingtalk_webhook:
        #     try:
        #         self._send_dingtalk(title, message, level)
        #         success = True
        #     except Exception as e:
        #         print(f"钉钉消息发送失败: {e}")
        
        # 尝试Server酱推送
        if self.serverchan_key:
            try:
                self._send_serverchan(title, message, level)
                success = True
            except Exception as e:
                print(f"Server酱消息发送失败: {e}")

        if (not self.dingtalk_webhook) and (not self.serverchan_key):
            print(u'系统未配置默认 Messenger ! 请配置默认 Messenger 以确保消息送达！')
        
        # 尝试自定义webhook推送
        for name, webhook_url in self.custom_webhooks.items():
            try:
                self._send_custom_webhook(name, webhook_url, title, message, level)
                success = True
            except Exception as e:
                print(f"自定义webhook {name} 消息发送失败: {e}")
        
        return success
    
    def _send_dingtalk(self, title: str, message: str, level: str):
        """发送钉钉消息"""
        if not self.dingtalk_webhook:
            return

        try:
            ret_result = DingReminder.send_message(
                appkey=self.dingtalk_webhook['appkey'],
                appsecret=self.dingtalk_webhook['appsecret'],
                robot_code=self.dingtalk_webhook['robot_code'],
                user_id_list=self.dingtalk_webhook['user_id_list'],
                content=f"# {title}\n\n**级别**: {level.upper()}\n\n{message}\n\n",
            )
            if (ret_result.status_code == 200):
                pass
                # print("✓ 钉钉测试消息发送成功！")
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
    
    def _send_serverchan(self, title: str, message: str, level: str):
        """发送Server酱消息"""
        if not self.serverchan_key:
            return
        
        # Server酱消息格式
        text = f"{title} - {level.upper()}"
        desp = f"**消息级别**: {level.upper()}\n\n{message}"
        
        url = f"https://sctapi.ftqq.com/{self.serverchan_key}.send"
        payload = {
            "title": text,
            "desp": desp
        }
        
        response = requests.post(url, data=payload, timeout=10)
        response.raise_for_status()
    
    def _send_custom_webhook(self, name: str, webhook_url: str, title: str, message: str, level: str):
        """发送自定义webhook消息"""
        payload = {
            "title": title,
            "message": message,
            "level": level,
            "source": "GolemQ Monitor"
        }
        
        headers = {'Content-Type': 'application/json'}
        response = requests.post(
            webhook_url,
            data=json.dumps(payload),
            headers=headers,
            timeout=10
        )
        response.raise_for_status()
    
    def send_test_message(self) -> Dict[str, bool]:
        """发送测试消息到所有配置的渠道"""
        results = {}
        
        test_title = "GolemQ 监控系统测试消息"
        test_message = "这是一条测试消息，用于验证消息推送配置是否正确。"
        
        if self.dingtalk_webhook:
            try:
                self._send_dingtalk(test_title, test_message, "info")
                results["dingtalk"] = True
            except Exception as e:
                results["dingtalk"] = False
                print(f"钉钉测试消息发送失败: {e}")
        
        if self.serverchan_key:
            try:
                self._send_serverchan(test_title, test_message, "info")
                results["serverchan"] = True
            except Exception as e:
                results["serverchan"] = False
                print(f"Server酱测试消息发送失败: {e}")
        
        for name in self.custom_webhooks:
            try:
                self._send_custom_webhook(name, self.custom_webhooks[name], test_title, test_message, "info")
                results[name] = True
            except Exception as e:
                results[name] = False
                print(f"自定义webhook {name} 测试消息发送失败: {e}")
        
        return results


# 全局消息推送器实例
global_messenger = Messenger()


def configure_global_messenger(dingtalk_webhook: Optional[str] = None,
                               serverchan_key: Optional[str] = None):
    """配置全局消息推送器"""
    if dingtalk_webhook:
        global_messenger.configure_dingtalk(dingtalk_webhook)
    if serverchan_key:
        global_messenger.configure_serverchan(serverchan_key)


def send_alert(title: str, message: str, level: str = "info") -> bool:
    """使用全局消息推送器发送告警"""
    return global_messenger.send_alert(title, message, level)