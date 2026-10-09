# coding:utf-8
"""钉钉 / Server酱 推送的测试 —— **全部是 mock 的**。

⚠️ **钉钉这边目前没有可用的 Access Token**（`DingtalkAccessToken.main` 里那两个
默认 `appkey`/`appsecret` 是占位串）。但**这不影响本模块**：这里的用例把
`get_dingtalk_config` / `dingtalk*Client` / `DingtalkAccessToken.main` 全 patch 掉了，
**不碰网络、不读真 token**。所以「token 失效」**不会**让本模块变红 ——
**别把这里的失败归因于 token**（2026-10-09 就是这么误判过一次：三个用例红了，
查下来全是**测试自身的陈旧假设**，与 token 无关，已修）：

* `test_check_config_failure` 断言了 `check_dingtalk_config` 里不存在的 try/except；
* `test_get_access_token_failure` 扔的是普通 `Exception`，而代码读 `err.code`（SDK 错误对象）；
* `test_send_message_*` 把正文写成了**位置参数**，实际落在 `robot_code` 上。

**约定**：要加用例就照现在的样子继续 mock。**不要**加联网 / 真 token 的用例 ——
那种用例只会永远红，并把真失败埋进噪声里。
"""
import os, sys
try:
    import GolemQ
except ImportError:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import unittest
from unittest.mock import patch, MagicMock
from GolemQ.agents.messenger import (
    check_dingtalk_config,
    setup_dingtalk_config,
    get_dingtalk_config,
    DingtalkAccessToken,
    DingReminder
)
from GolemQ.core.settings import GQSETTING


class TestDingtalkConfig(unittest.TestCase):
    @patch.object(GQSETTING, 'get_config')
    def test_check_config_success(self, mock_get):
        mock_get.side_effect = ['appkey', 'appsecret', 'robot_code', 'user1,user2']
        self.assertTrue(check_dingtalk_config())

    @patch.object(GQSETTING, 'get_config')
    def test_check_config_failure(self, mock_get):
        """配置**缺失/为空** → False。

        ⚠️ 原写成 `side_effect = Exception('Config error')` 并断言 False —— 但
        `check_dingtalk_config` **没有 try/except**，异常会直接抛，测试必红。
        那是测试写错了（配置读取失败**响亮地炸**是对的，不该吞）。
        """
        mock_get.return_value = ''
        self.assertFalse(check_dingtalk_config())

    @patch.object(GQSETTING, 'set_config')
    @patch('builtins.input', side_effect=['test_key', 'test_secret', 'test_robot', 'user1,user2'])
    def test_setup_config(self, mock_input, mock_set):
        setup_dingtalk_config()
        self.assertEqual(mock_set.call_count, 4)

    @patch('GolemQ.agents.messenger.check_dingtalk_config')
    @patch('GolemQ.agents.messenger.setup_dingtalk_config')
    @patch.object(GQSETTING, 'get_config')
    def test_get_config(self, mock_get, mock_setup, mock_check):
        mock_check.return_value = False
        mock_get.side_effect = ['appkey', 'appsecret', 'robot_code', 'user1,user2']
        config = get_dingtalk_config()
        self.assertEqual(config['appkey'], 'appkey')
        self.assertEqual(config['user_id_list'], ['user1', 'user2'])
        mock_setup.assert_called_once()


class TestDingtalkAccessToken(unittest.TestCase):
    @patch('GolemQ.agents.messenger.dingtalkoauth2_1_0Client')
    @patch('GolemQ.agents.messenger.get_dingtalk_config')
    def test_get_access_token(self, mock_config, mock_client):
        mock_config.return_value = {
            'appkey': 'test_key',
            'appsecret': 'test_secret'
        }
        mock_response = MagicMock()
        mock_response.body = MagicMock(access_token='test_token')
        mock_client.return_value.get_access_token.return_value = mock_response
        
        token = DingtalkAccessToken.main()
        self.assertEqual(token, 'test_token')

    @patch('GolemQ.agents.messenger.dingtalkoauth2_1_0Client')
    @patch('GolemQ.agents.messenger.get_dingtalk_config')
    def test_get_access_token_failure(self, mock_config, mock_client):
        mock_config.return_value = {
            'appkey': 'test_key',
            'appsecret': 'test_secret'
        }
        # ⚠️ 必须是**带 .code/.message 的 SDK 错误对象** —— `main` 的 except 里读
        # `err.code`，扔普通 `Exception` 会 `AttributeError`（原来就是这么红的）。
        class _SdkError(Exception):
            code = 'InvalidAppKey'
            message = 'invalid appkey'

        mock_client.return_value.get_access_token.side_effect = _SdkError()
        
        token = DingtalkAccessToken.main()
        self.assertIsNone(token)


class TestDingReminder(unittest.TestCase):
    @patch('GolemQ.agents.messenger.DingtalkAccessToken.main')
    @patch('GolemQ.agents.messenger.dingtalkrobot_1_0Client')
    @patch('GolemQ.agents.messenger.get_dingtalk_config')
    def test_send_message(self, mock_config, mock_client, mock_token):
        mock_config.return_value = {
            'robot_code': 'test_robot',
            'user_id_list': ['user1']
        }
        mock_token.return_value = 'test_token'
        mock_client.return_value.robot_send_ding_with_options.return_value = 'success'
        
        DingReminder.send_message(content="Test message")     # 同上，别用位置参数
        mock_client.return_value.robot_send_ding_with_options.assert_called_once()

    @patch('GolemQ.agents.messenger.DingtalkAccessToken.main')
    @patch('GolemQ.agents.messenger.get_dingtalk_config')
    def test_send_message_markdown_content(self, mock_config, mock_token):
        mock_config.return_value = {
            'robot_code': 'test_robot',
            'user_id_list': ['user1']
        }
        mock_token.return_value = 'test_token'
        
        with patch('GolemQ.agents.messenger.dingtalkrobot_1_0Client') as mock_client:
            # ⚠️ 必须**关键字**传 content：`send_message(robot_code, user_id_list,
            # content, ...)` 的第一个位置参数是 robot_code —— 写成位置参数时
            # content 仍是默认值，断言必然失败（原来就是这么红的）。
            DingReminder.send_message(content="### Heading\nContent")
            args = mock_client.return_value.robot_send_ding_with_options.call_args[0][0]
            self.assertIn("### Heading", args.content)

    @patch('GolemQ.agents.messenger.DingtalkAccessToken.main')
    @patch('GolemQ.agents.messenger.get_dingtalk_config')
    def test_send_message_no_token(self, mock_config, mock_token):
        mock_config.return_value = {
            'robot_code': 'test_robot',
            'user_id_list': ['user1']
        }
        mock_token.return_value = None
        
        # 这里应该测试没有token时的行为，但当前实现不会抛出异常
        # 只是会跳过发送，所以这个测试需要调整
        DingReminder.send_message(content="Test message")     # 同上，别用位置参数


if __name__ == '__main__':
    unittest.main()
