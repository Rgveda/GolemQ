#!/usr/bin/env python3
# coding:utf-8
#
# 简单测试 XTQuant 同步调度器
#

import sys
import os

try:
    import GolemQ
except ImportError:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from GolemQ.supervisor.scheduler import XtquantSyncScheduler, TradingTimeChecker
from GolemQ.supervisor.heartbeat import HeartbeatMonitor

def test_trading_time_checker():
    """测试交易时间检查器"""
    print("测试交易时间检查器...")
    checker = TradingTimeChecker()
    
    # 简单测试当前时间
    result = checker.is_trading_time()
    print(f"当前时间是否为交易时间: {result}")
    return True

def test_scheduler_initialization():
    """测试调度器初始化"""
    print("\n测试调度器初始化...")
    try:
        scheduler = XtquantSyncScheduler()
        print("✓ 调度器初始化成功")
        print(f"  最小间隔: {scheduler.min_interval_minutes} 分钟")
        print(f"  最大间隔: {scheduler.max_interval_minutes} 分钟")
        return True
    except Exception as e:
        print(f"✗ 调度器初始化失败: {e}")
        return False

def test_heartbeat_integration():
    """测试心跳监控集成"""
    print("\n测试心跳监控集成...")
    try:
        monitor = HeartbeatMonitor()
        print("✓ 心跳监控器初始化成功")
        
        # 测试开始模块
        record_id = monitor.start_module(
            module_name="test_module",
            instance_id="test_instance",
            initial_message="测试消息"
        )
        print(f"✓ 模块启动成功，记录ID: {record_id}")
        
        # 测试签到
        success = monitor.checkin(
            module_name="test_module",
            instance_id="test_instance",
            message="测试签到"
        )
        print(f"✓ 签到成功: {success}")
        
        # 测试完成模块
        success = monitor.complete_module(
            module_name="test_module",
            instance_id="test_instance",
            completion_message="测试完成"
        )
        print(f"✓ 模块完成: {success}")
        
        return True
    except Exception as e:
        print(f"✗ 心跳监控测试失败: {e}")
        return False

def test_messenger_integration():
    """测试消息通知集成"""
    print("\n测试消息通知集成...")
    try:
        from GolemQ.supervisor.messenger import send_alert
        # 测试发送一个信息级别的消息
        success = send_alert(
            title="测试消息",
            message="这是一条测试消息",
            level="info"
        )
        print(f"✓ 消息发送测试完成: {success}")
        return True
    except Exception as e:
        print(f"✗ 消息通知测试失败: {e}")
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("XTQuant 同步调度器简单测试")
    print("=" * 50)
    
    # 运行测试
    all_tests_passed = True
    
    all_tests_passed &= test_trading_time_checker()
    all_tests_passed &= test_scheduler_initialization()
    all_tests_passed &= test_heartbeat_integration()
    all_tests_passed &= test_messenger_integration()
    
    print("\n" + "=" * 50)
    if all_tests_passed:
        print("✓ 所有测试通过！")
    else:
        print("✗ 部分测试失败")
    
    print("可以使用以下命令启动守护进程:")
    print("python -m GolemQ.cli --xtquant-sync-daemon")
    print("=" * 50)