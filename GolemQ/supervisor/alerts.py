# coding:utf-8
#
# GolemQ 告警派发器
# 只负责「告警级别 + 渠道扇出」；协议细节（鉴权 / URL / 编码）在 agents.channels。
#

from GolemQ.core.settings import GQSETTING
from GolemQ.agents.channels import (
    check_serverchan_config,
    sc_send,
)


class AlertDispatcher:
    """告警派发器（当前渠道：Server酱）。

    ⚠️ **不要在这里再写一份 Server酱/钉钉发送** —— 传输一律走
    :mod:`GolemQ.agents.channels`。此前这里私写过 `_send_serverchan`
    （form 编码、不支持 `sctp` key），与 agents 的 JSON 实现**已经分叉**。
    """

    def __init__(self):
        """读取渠道配置。未配置时**静默禁用**（不交互式追问）。"""
        self.serverchan_key = (
            GQSETTING.get_config('SERVERCHAN', 'sendkey')
            if check_serverchan_config() else None
        )

    def send_alert(self, title: str, message: str, level: str = "info") -> bool:
        """发送告警消息。

        :param level: 消息级别（info / warning / error / critical），只影响正文标注。
        :return: 是否至少有一个渠道发送成功。
        """
        if not self.serverchan_key:
            print(u'系统未配置 Server酱告警渠道 ! 请配置 SERVERCHAN 以确保消息送达！')
            return False

        text = f"{title} - {level.upper()}"
        desp = f"**消息级别**: {level.upper()}\n\n{message}"
        try:
            result = sc_send(self.serverchan_key, text, desp)
            if 'error' in result:
                raise RuntimeError(result['error'])
            return True
        except Exception as e:
            print(f"Server酱消息发送失败: {e}")
            return False


# 全局告警派发器实例
_global_dispatcher = AlertDispatcher()


def send_alert(title: str, message: str, level: str = "info") -> bool:
    """使用全局告警派发器发送告警"""
    return _global_dispatcher.send_alert(title, message, level)
