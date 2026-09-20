#!/usr/bin/env python3
# coding:utf-8
#
# 测试 XTQuant 同步调度器
#

import sys
import os

try:
    import GolemQ
except ImportError:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from GolemQ.supervisor.scheduler import XtquantSyncScheduler, TradingTimeChecker
from datetime import datetime, time as dt_time
import pytz

def test_trading_time_checker():
    """测试交易时间检查器"""
    print("测试交易时间检查器...")
    checker = TradingTimeChecker()
    
    # 测试几个时间点
    test_times = [
        (dt_time(9, 0), False),    # 9:00 - 非交易时间
        (dt_time(9, 30), True),    # 9:30 - 交易时间开始
        (dt_time(10, 0), True),    # 10:00 - 交易时间
        (dt_time(11, 30), True),   # 11:30 - 交易时间结束
        (dt_time(12, 0), False),   # 12:00 - 午休时间
        (dt_time(13, 0), True),    # 13:00 - 下午交易时间开始
        (dt_time(14, 30), True),   # 14:30 - 交易时间
        (dt_time(15, 0), True),    # 15:00 - 交易时间结束
        (dt_time(16, 0), False),   # 16:00 - 非交易时间
    ]
    
    for test_time, expected in test_times:
        # 创建一个测试日期时间对象
        test_datetime = datetime.now().replace(
            hour=test_time.hour,
            minute=test_time.minute,
            second=0,
            microsecond=0
        )
        
        # 模拟当前时间
        import GolemQ.supervisor.scheduler as scheduler_module
        original_now = scheduler_module.datetime.now
        
        def mock_now(tz=None):
            if tz:
                return test_datetime.astimezone(tz)
            return test_datetime
            
        scheduler_module.datetime.now = mock_now
        
        try:
            result = checker.is_trading_time()
            status = "✓" if result == expected else "✗"
            print(f"{status} {test_time.strftime('%H:%M')}: 预期={expected}, 实际={result}")
        finally:
            # 恢复原始函数
            scheduler_module.datetime.now = original_now

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
        from GolemQ.supervisor.heartbeat import HeartbeatMonitor
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

if __name__ == "__main__":
    print("=" * 50)
    print("XTQuant 同步调度器测试")
    print("=" * 50)
    
    # 运行测试
    test_trading_time_checker()
    test_scheduler_initialization()
    test_heartbeat_integration()
    
    print("\n" + "=" * 50)
    print("测试完成！")
    print("可以使用以下命令启动守护进程:")
    print("python -m GolemQ.cli --xtquant-sync-daemon")
    print("=" * 50)