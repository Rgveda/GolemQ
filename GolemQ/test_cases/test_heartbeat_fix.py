#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试HeartbeatMonitor超时实例清理修复
"""

import os, sys
try:
    import GolemQ
except ImportError:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import time
from GolemQ.supervisor.heartbeat import HeartbeatMonitor


def test_timeout_instance_cleanup():
    """测试超时实例清理功能"""
    print("=== 测试HeartbeatMonitor超时实例清理 ===")
    
    # 创建监控实例
    monitor = HeartbeatMonitor()
    
    # 创建一个测试模块记录
    module_name = "test_module"
    instance_id = f"test_instance_{int(time.time())}"
    
    print(f"创建测试模块: {module_name}, 实例ID: {instance_id}")
    
    # 开始模块执行
    monitor.start_module(
        module_name=module_name,
        instance_id=instance_id,
        timeout_seconds=60,  # 1分钟超时
        initial_message="测试模块启动"
    )
    
    # 模拟一次签到
    monitor.checkin(
        module_name=module_name,
        instance_id=instance_id,
        message="测试签到"
    )
    
    # 检查模块是否在运行中
    running_modules = monitor.get_running_modules()
    print(f"运行中的模块数量: {len(running_modules)}")
    
    for module in running_modules:
        print(f"模块: {module['module_name']}, 状态: {module['status']}")
    
    # 尝试归档运行中的模块（模拟超时清理场景）
    print("\n尝试归档运行中的模块...")
    try:
        monitor._archive_completed_record(module_name, instance_id)
        print("✓ 归档成功！")
        
        # 检查是否已归档
        running_modules_after = monitor.get_running_modules()
        print(f"归档后运行中的模块数量: {len(running_modules_after)}")
        
        # 检查归档记录
        archived = list(monitor.archive_collection.find({
            "module_name": module_name,
            "instance_id": monitor._hash_instance_id(instance_id)
        }))
        print(f"找到归档记录: {len(archived)} 条")
        
        if len(archived) > 0:
            print("✓ 测试通过：超时实例清理功能正常工作")
            return True
        else:
            print("✗ 测试失败：未找到归档记录")
            return False
            
    except Exception as e:
        print(f"✗ 测试失败：{e}")
        return False


if __name__ == "__main__":
    success = test_timeout_instance_cleanup()
    if success:
        print("\n🎉 所有测试通过！")
    else:
        print("\n❌ 测试失败")
        exit(1)