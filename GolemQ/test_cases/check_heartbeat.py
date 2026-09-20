#!/usr/bin/env python3
# coding:utf-8
#
# 检查心跳记录
#

import sys
import os

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from GolemQ.supervisor.heartbeat import HeartbeatMonitor

def check_heartbeat_records():
    """检查心跳记录"""
    print("检查心跳记录...")
    
    monitor = HeartbeatMonitor()
    
    # 查看运行中的模块
    running_modules = monitor.get_running_modules()
    print(f"运行中的模块数量: {len(running_modules)}")
    for module in running_modules:
        print(f"  模块: {module['module_name']}, 实例: {module['instance_id']}, 状态: {module['status']}")
    
    # 查看已完成的心跳记录
    completed_modules = list(monitor.module_collection.find({'status': 'completed'}))
    print(f"\n已完成的心跳记录数量: {len(completed_modules)}")
    
    # 显示最近的5条记录
    recent_completed = list(monitor.module_collection.find({'status': 'completed'}).sort('start_timestamp', -1).limit(5))
    for record in recent_completed:
        print(f"  模块: {record['module_name']}")
        print(f"    实例: {record['instance_id']}")
        print(f"    开始时间: {record['start_datetime']}")
        print(f"    完成消息: {record.get('completion_message', 'N/A')}")
        print(f"    签到次数: {record.get('checkin_count', 0)}")
        print()

if __name__ == "__main__":
    check_heartbeat_records()