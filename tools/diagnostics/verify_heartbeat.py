#!/usr/bin/env python3
# coding:utf-8
#
# 验证心跳功能
#

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from GolemQ.supervisor.heartbeat import HeartbeatMonitor

def test_heartbeat():
    print("测试心跳功能...")
    
    monitor = HeartbeatMonitor()
    
    # 测试创建心跳记录
    instance_id = f"test_{int(time.time())}"
    record_id = monitor.start_module(
        module_name="test_heartbeat",
        instance_id=instance_id,
        initial_message="测试心跳功能"
    )
    print(f"✓ 创建心跳记录: {record_id}")
    
    # 测试签到
    success = monitor.checkin(
        module_name="test_heartbeat",
        instance_id=instance_id,
        message="测试签到"
    )
    print(f"✓ 签到成功: {success}")
    
    # 测试完成
    success = monitor.complete_module(
        module_name="test_heartbeat",
        instance_id=instance_id,
        completion_message="测试完成"
    )
    print(f"✓ 完成模块: {success}")
    
    print("心跳功能测试完成!")

if __name__ == "__main__":
    import time
    test_heartbeat()