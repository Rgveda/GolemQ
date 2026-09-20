# coding:utf-8
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
        mock_get.side_effect = Exception("Config error")
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
        mock_client.return_value.get_access_token.side_effect = Exception("API error")
        
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
        
        DingReminder.send_message("Test message")
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
            DingReminder.send_message("### Heading\nContent")
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
        DingReminder.send_message("Test message")


if __name__ == '__main__':
    unittest.main()
