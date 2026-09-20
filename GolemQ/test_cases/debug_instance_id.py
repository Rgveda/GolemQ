#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
调试实例ID问题
"""

from GolemQ.supervisor.heartbeat import HeartbeatMonitor


def debug_instance_id():
    """调试实例ID问题"""
    monitor = HeartbeatMonitor()
    
    print("=== 调试实例ID问题 ===")
    
    # 获取所有运行中的模块
    running_modules = monitor.get_running_modules()
    print(f"运行中的模块数量: {len(running_modules)}")
    
    for module in running_modules:
        print(f"模块: {module['module_name']}")
        print(f"  实例ID: {module['instance_id']}")
        print(f"  状态: {module['status']}")
        print(f"  最后签到: {module.get('last_checkin_datetime', 'N/A')}")
        print()


if __name__ == "__main__":
    debug_instance_id()