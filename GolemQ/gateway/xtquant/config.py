# coding:utf-8
# Author: 阿财（11652964@qq.com）
# Created date : 2025-08-29
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

from typing import Dict
from GolemQ.core.settings import GQSETTING, DEFAULT_DB_URI


def check_xtquant_config() -> bool:
    """检查XTQuant配置是否存在"""
    account = GQSETTING.get_config('XTQUANT', 'account')
    min_path = GQSETTING.get_config('XTQUANT', 'min_path')
    
    # Check if all required configurations are present and not empty/default values
    return (all([account, min_path]) and
            account != DEFAULT_DB_URI and
            min_path != DEFAULT_DB_URI)


def setup_xtquant_config() -> None:
    """交互式设置XTQuant配置"""
    print("XTQuant配置缺失，请按提示输入配置信息:")
    account = input("请输入XTQuant账户号: ")
    min_path = input("请输入XTQuant Mini路径: ")
    
    GQSETTING.set_config('XTQUANT', 'account', account)
    GQSETTING.set_config('XTQUANT', 'min_path', min_path)
    print("XTQuant配置已保存!")


def get_xtquant_config() -> Dict[str, str]:
    """获取XTQuant配置"""
    if not check_xtquant_config():
        setup_xtquant_config()

    return {
        'account': GQSETTING.get_config('XTQUANT', 'account'),
        'min_path': GQSETTING.get_config('XTQUANT', 'min_path')
    }


def setup_xtquant_config_interactive() -> None:
    """交互式设置XTQuant配置"""
    try:
        from GolemQ.core.preprocessing import mask_sensitive_info
        
        if check_xtquant_config():
            print("XTQuant配置已存在:")
            try:
                config = {
                    'account': GQSETTING.get_config('XTQUANT', 'account'),
                    'min_path': GQSETTING.get_config('XTQUANT', 'min_path')
                }
                for key, value in config.items():
                    # 对所有配置值进行脱敏处理
                    masked_value = mask_sensitive_info(value)
                    print(f"  {key}: {masked_value}")
                
                choice = input("是否重新配置XTQuant? (y/N): ").strip().lower()
                if choice != 'y':
                    return
            except Exception as e:
                print(f"读取现有配置时出错: {e}")
                print("将重新配置XTQuant...")
    
    except Exception as e:
        print(f"检查XTQuant配置时出错: {e}")
        print("将重新配置XTQuant...")
    
    setup_xtquant_config()
    print("XTQuant配置已保存!")