#!/usr/bin/env python3
# coding:utf-8
"""
测试时间戳格式
"""

import datetime
import pandas as pd

def test_timestamp_format():
    """测试时间戳格式转换"""
    print("=== 测试时间戳格式 ===")
    
    # 模拟当前时间
    current_time = datetime.datetime.now()
    unix_timestamp = int(current_time.timestamp())
    
    print(f"当前时间: {current_time}")
    print(f"Unix时间戳 (int32): {unix_timestamp}")
    print(f"时间戳类型: {type(unix_timestamp)}")
    
    # 创建测试DataFrame
    test_data = {
        'symbol': ['000001.SZ'],
        'updated_at': [unix_timestamp],
        'ckop_at': [unix_timestamp]
    }
    
    df = pd.DataFrame(test_data)
    print(f"\nDataFrame内容:")
    print(df)
    print(f"\nupdated_at 数据类型: {df['updated_at'].dtype}")
    print(f"ckop_at 数据类型: {df['ckop_at'].dtype}")
    
    # 验证是否为int32
    if df['updated_at'].dtype in ['int32', 'int64']:
        print("✓ 时间戳格式正确 (int32/int64)")
        return True
    else:
        print("✗ 时间戳格式不正确")
        return False

if __name__ == '__main__':
    success = test_timestamp_format()
    if success:
        print("\n✓ 时间戳格式测试通过!")
    else:
        print("\n✗ 时间戳格式测试失败!")