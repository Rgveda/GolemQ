#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
检查当前运行中的模块
"""

from GolemQ.supervisor.heartbeat import HeartbeatMonitor


def check_running_modules():
    """检查当前运行中的模块"""
    monitor = HeartbeatMonitor()
    
    print("=== 当前运行中的模块 ===")
    running_modules = monitor.get_running_modules()
    print(f"运行中的模块数量: {len(running_modules)}")
    
    for module in running_modules:
        print(f"模块: {module['module_name']}")
        print(f"  状态: {module['status']}")
        print(f"  实例ID: {module['instance_id']}")
        print(f"  最后签到: {module.get('last_checkin_datetime', 'N/A')}")
        print()


if __name__ == "__main__":
    check_running_modules()