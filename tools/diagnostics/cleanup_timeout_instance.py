#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
清理超时的xtquant_sync_loop实例
"""

from GolemQ.supervisor.heartbeat import HeartbeatMonitor


def cleanup_timeout_instance():
    """清理超时的xtquant_sync_loop实例"""
    monitor = HeartbeatMonitor()
    
    # 要清理的实例ID（原始值）
    original_instance_id = "7770c4b948cb3a3d3a9f16f71a4b97189b6b5edbb02ca4f779705453e3b505f0"
    module_name = "xtquant_sync_loop"
    
    print(f"=== 清理超时实例: {original_instance_id} ===")
    
    # 首先更新状态为timeout
    success = monitor.update_module_status(
        module_name=module_name,
        instance_id=original_instance_id,
        status="timeout",
        message="手动清理超时实例"
    )
    
    if success:
        print("✓ 成功更新实例状态为timeout")
    else:
        print("✗ 更新实例状态失败")
        return False
    
    # 然后归档该实例
    try:
        monitor._archive_completed_record(module_name, original_instance_id)
        print("✓ 成功归档超时实例")
        
        # 验证是否已清理
        running_modules = monitor.get_running_modules()
        xtquant_modules = [m for m in running_modules if m['module_name'] == module_name]
        
        if len(xtquant_modules) == 0:
            print("✓ 所有xtquant_sync_loop实例已清理完成")
            return True
        else:
            print(f"✗ 仍有 {len(xtquant_modules)} 个xtquant_sync_loop实例在运行")
            return False
            
    except Exception as e:
        print(f"✗ 归档失败: {e}")
        return False


if __name__ == "__main__":
    success = cleanup_timeout_instance()
    if success:
        print("\n🎉 超时实例清理成功！")
    else:
        print("\n❌ 清理失败")
        exit(1)