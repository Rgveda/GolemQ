#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
直接清理超时的xtquant_sync_loop实例
"""

import hashlib
from GolemQ.supervisor.heartbeat import HeartbeatMonitor


def direct_cleanup():
    """直接清理超时的xtquant_sync_loop实例"""
    monitor = HeartbeatMonitor()
    
    # 原始实例ID
    original_instance_id = "7770c4b948cb3a3d3a9f16f71a4b97189b6b5edbb02ca4f779705453e3b505f0"
    module_name = "xtquant_sync_loop"
    
    print(f"=== 直接清理超时实例: {original_instance_id} ===")
    
    # 计算哈希后的实例ID
    hashed_instance_id = hashlib.sha256(original_instance_id.encode()).hexdigest()
    print(f"哈希后的实例ID: {hashed_instance_id}")
    
    # 直接查找并删除记录
    record = monitor.module_collection.find_one({
        "module_name": module_name,
        "instance_id": hashed_instance_id
    })
    
    if record:
        print(f"找到记录: {record['_id']}")
        print(f"状态: {record['status']}")
        print(f"最后签到: {record.get('last_checkin_datetime', 'N/A')}")
        
        # 插入到归档表
        monitor.archive_collection.insert_one(record)
        print("✓ 记录已插入归档表")
        
        # 从活动表删除
        result = monitor.module_collection.delete_one({"_id": record["_id"]})
        if result.deleted_count > 0:
            print("✓ 记录已从活动表删除")
            return True
        else:
            print("✗ 从活动表删除失败")
            return False
    else:
        print("✗ 未找到匹配的记录")
        return False


if __name__ == "__main__":
    success = direct_cleanup()
    if success:
        print("\n🎉 直接清理成功！")
    else:
        print("\n❌ 直接清理失败")
        exit(1)