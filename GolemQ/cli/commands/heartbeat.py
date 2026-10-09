# coding:utf-8
"""心跳监控：`--heartbeat-watchdog`（查看）与 `--stop-heartbeat-monitor`（停止）。

`_send_timeout_alert` 从 `cli/__main__.py` **搬进来**的 —— 它是本命令私有的
报表面板构建（Server酱 Markdown 表格），放在命令模块里才自洽。
"""
from __future__ import annotations

import sys
import time
from datetime import timedelta

from GolemQ.agents.messenger import send_serverchan_message
from GolemQ.supervisor.function_checkin import checkin_function
from GolemQ.supervisor.heartbeat import HeartbeatMonitor

from ._registry import Command


def add_watchdog_arguments(parser) -> None:
    parser.add_argument('--heartbeat-watchdog',
                        help="查看心跳监控状态",
                        action="store_true",
                        default=False)


def add_stop_arguments(parser) -> None:
    parser.add_argument('--stop-heartbeat-monitor',
                        help="停止所有心跳监控并清理资源",
                        action="store_true",
                        default=False)


def _send_timeout_alert(timeout_modules: list) -> None:
    """发送超时模块提醒到Server酱

    Args:
        timeout_modules: 超时模块列表
    """
    if not timeout_modules:
        return

    # 导入频率控制模块
    try:
        # 检查调用频率（15分钟限制）
        checkin_result = checkin_function(
            function_name="_send_timeout_alert",
            expired_time=timedelta(minutes=15)
        )

        if not checkin_result["allowed"]:
            remaining_time = checkin_result['expired_timestamp'] - checkin_result['current_time']
            print(f"Server酱消息发送频率限制：距离下次可发送还有 {remaining_time} 秒")
            return
    except ImportError:
        # 如果导入失败，直接发送消息（向后兼容）
        print("警告: FunctionCheckinManager不可用，跳过频率限制")

    # 构建Markdown格式的消息
    title = f"⚠️ GolemQ 模块超时告警 ({len(timeout_modules)}个模块)"

    # 构建表格内容
    table_header = "| 模块名称 | 实例ID | 最后签到 | 超时阈值 | 实际超时 |\n"
    table_separator = "|----------|--------|----------|----------|----------|\n"
    table_rows = []

    for module in timeout_modules:
        # 截断过长的实例ID
        instance_id = module['instance_id']
        if len(instance_id) > 12:
            instance_id = instance_id[:12] + "..."

        table_rows.append(
            f"| {module['module_name']} | `{instance_id}` | {module['last_checkin']} | "
            f"{module['timeout_threshold']} | **{module['actual_timeout']}** |"
        )

    # 组合完整的Markdown内容
    markdown_content = (
        f"## 🚨 模块执行超时告警\n\n"
        f"检测到 **{len(timeout_modules)}** 个模块执行超时，请及时处理！\n\n"
        f"{table_header}{table_separator}{chr(10).join(table_rows)}\n\n"
        f"**处理建议:**\n"
        f"- 检查模块是否正常运行\n"
        f"- 确认网络连接是否正常\n"
        f"- 查看日志文件排查问题\n"
        f"- 必要时重启相关服务\n\n"
        f"⏰ 告警时间: {time.strftime('%Y-%m-%d %H:%M:%S')}"
    )

    try:
        # 发送Server酱消息
        result = send_serverchan_message(title, markdown_content)
        if 'error' not in result:
            print("✓ Server酱超时告警发送成功!")
        else:
            print(f"✗ Server酱消息发送失败: {result.get('error', '未知错误')}")
    except Exception as e:      # noqa: BLE001 告警发不出去不该炸掉查看命令
        print(f"发送Server酱消息时出错: {e}")


def run_watchdog(args) -> None:
    # 查看心跳监控状态
    monitor = HeartbeatMonitor()
    timeout_modules = []  # 存储超时模块信息

    print("=== 当前运行中的模块 ===")
    running_modules = monitor.get_running_modules()
    if running_modules:
        for module in running_modules:
            print(f"模块: {module['module_name']}")
            print(f"  实例ID: {module['instance_id']}")
            print(f"  状态: {module['status']}")
            print(f"  最后签到: {module['last_checkin_datetime']}")
            print(f"  签到次数: {module['checkin_count']}")
            print(f"  超时阈值: {module['timeout_seconds']}秒")
            print()
    else:
        print("  没有运行中的模块")
        print()

    print("=== 所有模块状态 ===")
    all_modules = monitor.get_all_modules()
    if all_modules:
        current_time = int(time.time())

        for module in all_modules:
            # 检查是否超时（状态为running但当前时间大于最后签到时间+超时时间）
            actual_status = module['status']
            if module['status'] == 'running':
                last_checkin = module.get('last_checkin_timestamp', 0)
                timeout_seconds = module.get('timeout_seconds', 300)  # 默认5分钟
                if current_time > last_checkin + timeout_seconds:
                    actual_status = 'timeout (实际)'
                    status_icon = "🔴"
                else:
                    status_icon = "🟢"
            else:
                status_icon = "🟡" if module['status'] == 'completed' else \
                             "🔴" if module['status'] == 'timeout' else "⚫"

            print(f"{status_icon} {module['module_name']} - {actual_status}")
            print(f"   最后签到: {module.get('last_checkin_datetime', 'N/A')}")
            if module['status'] != 'running':
                print(f"   结束时间: {module.get('end_datetime', 'N/A')}")
            print(f"   超时阈值: {module.get('timeout_seconds', 300)}秒")

            # 显示超时信息（如果实际超时）
            if actual_status == 'timeout (实际)':
                # 计算实际超时时间（当前时间 - 最后签到时间 - 超时阈值）
                actual_timeout_seconds = current_time - last_checkin - timeout_seconds
                timeout_minutes = actual_timeout_seconds // 60
                print(f"   ⚠️ 已超时: {timeout_minutes}分钟 (超过阈值{timeout_seconds//60}分钟)")

                # 收集超时模块信息
                timeout_modules.append({
                    'module_name': module['module_name'],
                    'instance_id': module['instance_id'],
                    'last_checkin': module.get('last_checkin_datetime', 'N/A'),
                    'timeout_threshold': f"{timeout_seconds//60}分钟",
                    'actual_timeout': f"{timeout_minutes}分钟"
                })

            print()
    else:
        print("  没有模块记录")
        print()

    print("=== 最近模块执行历史 ===")
    recent_history = monitor.get_module_history(limit=10)
    if recent_history:
        for history in recent_history:
            status_icon = "✅" if history['status'] == 'completed' else \
                         "⏰" if history['status'] == 'timeout' else "❌"
            print(f"{status_icon} {history['module_name']} - {history['status']}")
            print(f"   开始: {history['start_datetime']}")
            print(f"   结束: {history['end_datetime']}")
            if 'completion_message' in history:
                print(f"   消息: {history['completion_message']}")
            print()
    else:
        print("  没有历史记录")

    # 如果有超时模块，发送Server酱提醒
    if timeout_modules:
        _send_timeout_alert(timeout_modules)


def run_stop(args) -> None:
    # 停止所有心跳监控并清理资源
    monitor = HeartbeatMonitor()

    if not args.verbose:
        confirm = input("警告: 这将停止所有心跳监控并标记运行中的模块为停止状态! "
                        "确认操作? (y/N): ").strip().lower()
        if confirm != 'y':
            print("操作已取消")
            return

    print("正在停止心跳监控并清理资源...")
    monitor.stop_all_monitoring()
    print("心跳监控已停止，所有运行中的模块已标记为停止状态并转移到归档库")


HEARTBEAT_WATCHDOG = Command('heartbeat-watchdog', ('heartbeat_watchdog',),
                             add_watchdog_arguments, run_watchdog)
STOP_HEARTBEAT = Command('stop-heartbeat-monitor', ('stop_heartbeat_monitor',),
                         add_stop_arguments, run_stop)
