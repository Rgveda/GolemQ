#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
最终清理超时的xtquant_sync_loop实例
"""

from GolemQ.supervisor.heartbeat import HeartbeatMonitor


def final_cleanup():
    """最终清理超时的xtquant_sync_loop实例"""
    monitor = HeartbeatMonitor()
    
    print("=== 最终清理超时实例 ===")
    
    # 获取所有运行中的xtquant_sync_loop实例
    running_modules = monitor.module_collection.find({
        "module_name": "xtquant_sync_loop",
        "status": "running"
    })
    
    cleaned_count = 0
    for module in running_modules:
        print(f"找到运行中的实例: {module['instance_id']}")
        print(f"最后签到: {module.get('last_checkin_datetime', 'N/A')}")
        
        # 插入到归档表
        monitor.archive_collection.insert_one(module)
        print("✓ 记录已插入归档表")
        
        # 从活动表删除
        result = monitor.module_collection.delete_one({"_id": module["_id"]})
        if result.deleted_count > 0:
            print("✓ 记录已从活动表删除")
            cleaned_count += 1
        else:
            print("✗ 从活动表删除失败")
    
    print(f"\n总共清理了 {cleaned_count} 个超时实例")
    return cleaned_count > 0


if __name__ == "__main__":
    success = final_cleanup()
    if success:
        print("\n🎉 最终清理成功！")
        
        # 验证清理结果
        monitor = HeartbeatMonitor()
        running_modules = monitor.get_running_modules()
        xtquant_modules = [m for m in running_modules if m['module_name'] == 'xtquant_sync_loop']
        
        if len(xtquant_modules) == 0:
            print("✓ 所有xtquant_sync_loop实例已清理完成")
        else:
            print(f"✗ 仍有 {len(xtquant_modules)} 个xtquant_sync_loop实例在运行")
            
    else:
        print("\n❌ 最终清理失败")
        exit(1)